"""Tests for the llama-server container manager (docker CLI monkeypatched; no real Docker)."""

from __future__ import annotations

import asyncio
import json

from backend.models import llama_runtime as lr

# A minimal `docker inspect` payload mirroring the real bundled llamacpp container.
INSPECT_JSON = json.dumps(
    [
        {
            "Config": {
                "Image": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                "Env": ["LLAMA_CACHE=/models", "PATH=/usr/bin"],
                "Labels": {
                    "com.docker.compose.project": "agent-graph",
                    "com.docker.compose.service": "llamacpp",
                    "maintainer": "NVIDIA",
                },
            },
            "HostConfig": {
                "RestartPolicy": {"Name": "unless-stopped"},
                "DeviceRequests": [{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=all"]}],
                "Runtime": "runc",
            },
            "Mounts": [
                {"Type": "volume", "Name": "agent-graph_llamacpp_models", "Destination": "/models", "RW": True}
            ],
            "NetworkSettings": {
                "Networks": {"agent-graph_default": {"Aliases": ["agent_graph_llamacpp", "llamacpp"]}}
            },
        }
    ]
)


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._out = stdout
        self._err = stderr

    async def communicate(self, input=None):  # noqa: A002 — matches asyncio API
        return self._out, self._err

    def kill(self) -> None:  # pragma: no cover - only used on timeout paths
        pass


def _fake_docker(handlers):
    """Build a fake create_subprocess_exec dispatching on the docker subcommand (args[1])."""

    async def fake_exec(*args, **_kwargs):
        sub = args[1] if len(args) > 1 else ""
        return handlers.get(sub, _FakeProc(0))

    return fake_exec


def test_detect_nvidia_via_docker_parses(monkeypatch) -> None:
    handlers = {"exec": _FakeProc(0, b"NVIDIA GeForce RTX 3090, 24576\n")}
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_docker(handlers))
    gpus = asyncio.run(lr.detect_nvidia_via_docker("c"))
    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 3090"
    assert gpus[0].vram_mb == 24576


def test_detect_via_docker_absent(monkeypatch) -> None:
    async def boom(*_a, **_k):
        raise FileNotFoundError()

    monkeypatch.setattr("asyncio.create_subprocess_exec", boom)
    assert asyncio.run(lr.detect_nvidia_via_docker("c")) == []


def test_detect_via_docker_nonzero_exit(monkeypatch) -> None:
    handlers = {"exec": _FakeProc(1, b"", b"no such container")}
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_docker(handlers))
    assert asyncio.run(lr.detect_nvidia_via_docker("c")) == []


def test_inspect_container_parses(monkeypatch) -> None:
    handlers = {"inspect": _FakeProc(0, INSPECT_JSON.encode())}
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_docker(handlers))
    info = asyncio.run(lr.inspect_container("agent_graph_llamacpp"))
    assert info.found is True
    assert info.image == "ghcr.io/ggml-org/llama.cpp:server-cuda"
    assert info.network == "agent-graph_default"
    assert "llamacpp" in info.aliases
    assert "agent-graph_llamacpp_models:/models" in info.binds
    assert info.gpu_args == ["--device", "nvidia.com/gpu=all"]
    assert info.restart == "unless-stopped"
    assert info.labels.get("com.docker.compose.service") == "llamacpp"
    # Image-baked label is not a compose label and should be dropped.
    assert "maintainer" not in info.labels


def test_inspect_absent_when_command_fails(monkeypatch) -> None:
    handlers = {"inspect": _FakeProc(1, b"", b"No such object")}
    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_docker(handlers))
    info = asyncio.run(lr.inspect_container("missing"))
    assert info.found is False


def test_create_args_force_includes_llamacpp_alias() -> None:
    # An inspected container that (for whatever reason) lacks the critical alias must still get it.
    info = lr.InspectInfo(
        found=True,
        image="img",
        network="net",
        aliases=["agent_graph_llamacpp"],  # note: no "llamacpp"
        binds=["vol:/models"],
        env=["LLAMA_CACHE=/models"],
        gpu_args=["--device", "nvidia.com/gpu=all"],
        restart="unless-stopped",
        labels={"com.docker.compose.service": "llamacpp"},
    )
    args = lr._create_args(info, ["-m", "/models/x.gguf"], container="agent_graph_llamacpp")
    # The llamacpp alias the backend reaches via LLAMACPP_BASE_URL must be present.
    alias_values = [args[i + 1] for i, a in enumerate(args) if a == "--network-alias"]
    assert "llamacpp" in alias_values
    assert "--device" in args and "nvidia.com/gpu=all" in args
    assert args[-2:] == ["-m", "/models/x.gguf"]
    assert "--restart" in args


def test_gpu_args_legacy_runtime() -> None:
    host_config = {
        "Runtime": "nvidia",
        "DeviceRequests": [{"Driver": "", "DeviceIDs": ["all"], "Capabilities": [["gpu"]]}],
    }
    args = lr._gpu_args_from_host_config(host_config)
    assert args[:2] == ["--runtime", "nvidia"]
    assert "--gpus" in args and "all" in args


def test_load_model_unmanaged_when_inspect_fails(monkeypatch) -> None:
    async def absent(_container=None):
        return lr.InspectInfo(found=False)

    monkeypatch.setattr(lr, "inspect_container", absent)
    res = asyncio.run(
        lr.load_model(command=["-m", "/models/x.gguf"], expected_alias="local/x", base_url="http://x/v1", api_key="k")
    )
    assert res.ok is False
    assert res.unmanaged is True
    assert res.error
