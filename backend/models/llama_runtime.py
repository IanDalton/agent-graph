"""Manage the local llama.cpp server container (GPU detect + model load), tolerantly.

The agent's LLM is served by an **external** ``llama-server``. When that server runs as a Docker
container on the host whose daemon the backend can reach (the bundled ``llamacpp`` compose service —
the backend already mounts ``/var/run/docker.sock`` and ships the ``docker`` CLI for the sandbox),
this module lets the Model Manager do two things the backend otherwise can't:

1. **Auto-detect the GPU** even though the backend image has no ``nvidia-smi``: run it *inside* the
   GPU-bearing llama-server container via ``docker exec`` (``detect_nvidia_via_docker``).
2. **Load a model**: recreate the llama-server container so it serves a chosen GGUF with the
   recommended flags (``load_model`` → ``inspect_container`` → ``recreate_with_command`` →
   ``wait_until_serving``).

Everything here is **best-effort/tolerant** (modeled on :mod:`backend.sandbox.runner`): a missing
``docker`` binary, an absent/remote container, or a non-zero exit degrades to ``[]`` / a structured
``LoadResult(ok=False, …)`` — it never raises into the API. When the server is *not* a manageable
local container (no socket, container not found, or ``LLAMACPP_BASE_URL`` points at a remote host),
``load_model`` returns ``unmanaged=True`` so the UI can disable the Load button with an explanation.

Faithfulness on recreate: we ``docker inspect`` the existing container and replicate ALL of its
config (image, network + aliases, mounts, env, GPU device request, restart policy, compose labels),
changing only the command. The **critical** invariant is that the ``llamacpp`` network alias (the
host ``LLAMACPP_BASE_URL`` resolves) survives — we force-include it regardless of what inspect
reports, or the backend would lose connectivity to its own model after a load.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field

import httpx

from backend.models.hardware import GpuInfo

logger = logging.getLogger("agent_graph.llama_runtime")

_DOCKER_TIMEOUT = 20.0  # for inspect/exec/create/start/rm (each)
_DEFAULT_LOAD_TIMEOUT = 180.0  # readiness poll budget (model load into VRAM)
_POLL_INTERVAL = 2.0
_DEFAULT_CONTAINER = "agent_graph_llamacpp"


def container_name() -> str:
    """The resolvable llama-server container name (``LLAMACPP_BASE_URL`` only gives the network
    alias ``llamacpp``, not the container name)."""
    return os.getenv("LLAMACPP_CONTAINER", _DEFAULT_CONTAINER)


def load_timeout() -> float:
    try:
        return float(os.getenv("LLAMACPP_LOAD_TIMEOUT", str(_DEFAULT_LOAD_TIMEOUT)))
    except ValueError:
        return _DEFAULT_LOAD_TIMEOUT


# --------------------------------------------------------------------------- subprocess helper


async def _run_docker(*args: str, input_bytes: bytes | None = None, timeout: float = _DOCKER_TIMEOUT):
    """Run a ``docker`` CLI command, returning ``(returncode, stdout, stderr)`` or ``None`` when the
    binary is missing / it times out. Never raises (mirrors the sandbox's tolerant subprocess use)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        logger.debug("docker CLI unavailable for: docker %s", " ".join(args))
        return None
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(input=input_bytes), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        logger.warning("docker command timed out: docker %s", " ".join(args))
        return None
    return proc.returncode, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


# --------------------------------------------------------------------------- GPU detection


async def detect_nvidia_via_docker(container: str | None = None) -> list[GpuInfo]:
    """List NVIDIA GPUs by running ``nvidia-smi`` inside the GPU-bearing llama-server container.

    The fallback for hosts where the backend container itself has no ``nvidia-smi`` but the
    llama-server container does. Parses the same ``name, memory.total`` CSV as
    :func:`backend.models.hardware.detect_nvidia`. ``[]`` on any failure (no docker, container
    absent/stopped, non-zero exit) — best-effort.
    """
    container = container or container_name()
    res = await _run_docker(
        "exec", container, "nvidia-smi",
        "--query-gpu=name,memory.total", "--format=csv,noheader,nounits",
    )
    if res is None or res[0] != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in res[1].splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, mem = line.rpartition(",")
        try:
            gpus.append(GpuInfo(name=name.strip(), vram_mb=int(float(mem.strip()))))
        except ValueError:
            continue
    return gpus


# --------------------------------------------------------------------------- inspect / recreate


@dataclass
class InspectInfo:
    """The subset of a container's config we replicate when recreating it with a new command."""

    found: bool = False
    image: str = ""
    network: str = ""
    aliases: list[str] = field(default_factory=list)
    binds: list[str] = field(default_factory=list)   # ``-v`` args (``src:dst[:ro]``)
    env: list[str] = field(default_factory=list)     # ``KEY=VALUE``
    gpu_args: list[str] = field(default_factory=list)  # e.g. ``["--device", "nvidia.com/gpu=all"]``
    restart: str = ""
    labels: dict[str, str] = field(default_factory=dict)


def _gpu_args_from_host_config(host_config: dict) -> list[str]:
    """Reconstruct the GPU CLI flags from an inspected ``HostConfig`` — CDI (``--device …``) or the
    legacy nvidia runtime (``--runtime nvidia --gpus …``). Derived from inspect, never hard-coded, so
    a non-CDI host is reproduced faithfully."""
    args: list[str] = []
    runtime = (host_config.get("Runtime") or "").lower()
    if runtime == "nvidia":
        args += ["--runtime", "nvidia"]
    for dr in host_config.get("DeviceRequests") or []:
        driver = (dr.get("Driver") or "").lower()
        ids = [d for d in (dr.get("DeviceIDs") or []) if d]
        if driver == "cdi":
            for did in ids:
                args += ["--device", did]
        else:  # legacy nvidia GPU request
            if not ids or ids == ["all"]:
                args += ["--gpus", "all"]
            else:
                args += ["--gpus", "device=" + ",".join(ids)]
    return args


async def inspect_container(container: str | None = None) -> InspectInfo:
    """Capture the config we need to recreate ``container`` with a new command. ``found=False`` when
    the container can't be inspected (no socket / absent / remote) — the cue for the ``unmanaged``
    path."""
    container = container or container_name()
    res = await _run_docker("inspect", container)
    if res is None or res[0] != 0:
        return InspectInfo(found=False)
    try:
        data = json.loads(res[1])
        c = data[0] if isinstance(data, list) else data
        config = c.get("Config") or {}
        host_config = c.get("HostConfig") or {}
        networks = (c.get("NetworkSettings") or {}).get("Networks") or {}

        network = ""
        aliases: list[str] = []
        if networks:
            network = next(iter(networks))
            aliases = [a for a in (networks[network].get("Aliases") or []) if a]

        binds: list[str] = []
        for m in c.get("Mounts") or []:
            dst = m.get("Destination")
            src = m.get("Name") or m.get("Source")
            if not (src and dst):
                continue
            suffix = "" if m.get("RW", True) else ":ro"
            binds.append(f"{src}:{dst}{suffix}")

        restart = ((host_config.get("RestartPolicy") or {}).get("Name") or "").strip()
        if restart in ("", "no"):
            restart = ""

        labels = {
            k: v for k, v in (config.get("Labels") or {}).items()
            if k.startswith("com.docker.compose")
        }
        return InspectInfo(
            found=True,
            image=config.get("Image") or "",
            network=network,
            aliases=aliases,
            binds=binds,
            env=list(config.get("Env") or []),
            gpu_args=_gpu_args_from_host_config(host_config),
            restart=restart,
            labels=labels,
        )
    except (ValueError, KeyError, IndexError, TypeError):
        logger.warning("could not parse docker inspect for %s", container, exc_info=True)
        return InspectInfo(found=False)


def _create_args(info: InspectInfo, command: list[str], *, container: str) -> list[str]:
    """The ``docker create`` argv replicating ``info`` but running ``command``.

    The ``llamacpp`` alias + the container name are force-included so the backend keeps reaching the
    server at ``http://llamacpp:8080`` after a recreate (the single highest-risk invariant)."""
    args = ["create", "--name", container]
    if info.network:
        args += ["--network", info.network]
        aliases = list(dict.fromkeys([*info.aliases, "llamacpp", container]))
        for alias in aliases:
            args += ["--network-alias", alias]
    args += info.gpu_args
    for env in info.env:
        args += ["-e", env]
    for bind in info.binds:
        args += ["-v", bind]
    if info.restart:
        args += ["--restart", info.restart]
    for k, v in info.labels.items():
        args += ["--label", f"{k}={v}"]
    args += [info.image, *command]
    return args


async def recreate_with_command(
    info: InspectInfo, command: list[str], *, container: str | None = None
) -> tuple[bool, str | None]:
    """Remove the existing container and recreate it from ``info`` with ``command``. Returns
    ``(ok, error)``. The old container is removed first; on a create/start failure the model server is
    gone until restored (the error tells the caller how)."""
    container = container or container_name()
    if not info.found or not info.image:
        return False, "no container config to recreate from"
    await _run_docker("rm", "-f", container)  # best-effort; ignore "no such container"
    created = await _run_docker(*_create_args(info, command, container=container))
    if created is None or created[0] != 0:
        detail = created[2].strip() if created else "docker create unavailable"
        return False, f"could not create the llama-server container: {detail}"
    started = await _run_docker("start", container)
    if started is None or started[0] != 0:
        detail = started[2].strip() if started else "docker start unavailable"
        return False, f"could not start the llama-server container: {detail}"
    return True, None


# --------------------------------------------------------------------------- readiness poll


async def _served_models(base_url: str, api_key: str) -> list[str] | None:
    """The model ids the server currently serves (``GET {base_url}/models``), or ``None`` if
    unreachable. Shared by the readiness poll and the status endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"})
            resp.raise_for_status()
            return [m.get("id") for m in (resp.json().get("data") or []) if m.get("id")]
    except Exception:  # noqa: BLE001 — unreachable is expected while the server restarts.
        return None


async def wait_until_serving(
    base_url: str, expected_alias: str, *, timeout: float, api_key: str
) -> str | None:
    """Poll until the server serves a model, returning its id (preferring ``expected_alias``), or
    ``None`` on timeout."""
    base_url = base_url.rstrip("/")
    waited = 0.0
    while waited < timeout:
        ids = await _served_models(base_url, api_key)
        if ids:
            return expected_alias if expected_alias in ids else ids[0]
        await asyncio.sleep(_POLL_INTERVAL)
        waited += _POLL_INTERVAL
    return None


# --------------------------------------------------------------------------- orchestration


@dataclass
class LoadResult:
    ok: bool = False
    unmanaged: bool = False
    served_model: str | None = None
    error: str | None = None
    notes: list[str] = field(default_factory=list)


async def load_model(
    *,
    command: list[str],
    expected_alias: str,
    base_url: str,
    api_key: str,
    container: str | None = None,
    timeout: float | None = None,
) -> LoadResult:
    """Recreate the llama-server container to run ``command`` (flags only, no ``llama-server`` token)
    and wait until it serves a model. Tolerant — returns a structured result, never raises."""
    container = container or container_name()
    timeout = timeout or load_timeout()
    info = await inspect_container(container)
    if not info.found:
        return LoadResult(
            ok=False,
            unmanaged=True,
            error=(
                "The llama-server isn't a local Docker container this app can manage "
                f"(couldn't inspect {container!r}). Run the launch command on the GPU host yourself."
            ),
        )
    ok, err = await recreate_with_command(info, command, container=container)
    if not ok:
        return LoadResult(
            ok=False,
            error=err,
            notes=["If the server is now down, run `docker compose up -d llamacpp` to restore it."],
        )
    served = await wait_until_serving(base_url, expected_alias, timeout=timeout, api_key=api_key)
    if served is None:
        return LoadResult(
            ok=False,
            error=(
                f"Recreated the container but it didn't start serving within {timeout:.0f}s. "
                "Check the model file and the container logs (`docker logs %s`)." % container
            ),
        )
    return LoadResult(ok=True, served_model=served)


__all__ = [
    "InspectInfo",
    "LoadResult",
    "container_name",
    "detect_nvidia_via_docker",
    "inspect_container",
    "recreate_with_command",
    "wait_until_serving",
    "load_model",
]
