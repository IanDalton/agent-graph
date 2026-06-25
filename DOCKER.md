# Running agent-graph in containers

The whole stack is containerized: **arcadedb** (graph memory), **searxng** (web search),
**backend** (FastAPI/SSE app), **frontend** (the React UI), and a **caddy** reverse proxy that is
the single published entry point. Secrets/model config come from `.env` in the repo root.

## Single-port access

A **Caddy** reverse proxy (`docker/caddy/Caddyfile`) is the *only* service with a host port: it
publishes **`${APP_PORT:-8080}`** and serves the whole app on one origin —

- `http://localhost:8080/`      → the frontend SPA
- `http://localhost:8080/api/*` → the FastAPI backend (SSE streamed, never buffered)

`arcadedb`, `searxng`, `backend`, and `frontend` have **no host ports**; they talk to each other
over the compose network and are reached only through Caddy. Each profile gets its own proxy
(`caddy` for prod, `caddy-dev` for dev) pointing at that profile's backend + frontend. Override the
published port with `APP_PORT` in `.env`.

## Prerequisites

- Docker Desktop (or Docker Engine) running.
- A `.env` file (already present) with `LLAMACPP_BASE_URL` pointing at your **llama.cpp
  `llama-server`** (`/v1` endpoint) — a LAN box, a host process, or the optional `llamacpp` compose
  service (`http://llamacpp:8080/v1`). Add `LLAMACPP_API_KEY` only if the server was launched with
  `--api-key`, and `HF_TOKEN` for gated/rate-limited HuggingFace downloads. `LLAMACPP_NUM_PREDICT`
  (max answer tokens) widens the output budget so long chains-of-thought aren't cut off; the context
  window is the server's `-c` flag (the Model Manager generates the command). `AGENT_MODEL` +
  provider key (e.g. `OPENAI_API_KEY`) is a hosted escape hatch.
- **Optional local GPU server:** `docker compose --profile llamacpp up llamacpp` runs a CUDA
  `llama-server` (needs the NVIDIA Container Toolkit). Edit its `command` to the one the Model
  Manager generates; it shares the `llamacpp_models` volume (`/models`) with the backend so
  downloaded GGUFs are visible. Set `MODELS_HOST_DIR` to a host path to share the models with a
  host-run server instead.

Everything lives in `docker-compose.yml`, selected by a **`dev` or `prod` profile**. The infra
services (`arcadedb`, `searxng`) have no profile, so they always come up under either profile (and
a bare `docker compose up` brings up just the infra for host-run development).

## Production stack (`prod` profile)

Built images, nginx-served UI:

```bash
docker compose build sandbox                 # one-time: the agent-sandbox image run_python launches
docker compose --profile prod up -d --build  # arcadedb + searxng + backend + frontend + caddy
```

- App (UI + API):  http://localhost:8080  (Caddy → SPA at `/`, FastAPI at `/api`)

The other services are internal-only (no host ports). To inspect one directly during debugging,
temporarily add a `ports:` mapping back to it in `docker-compose.yml`, or
`docker compose exec <service> …`.

## Dev stack (`dev` profile, auto-reloading)

Source is bind-mounted, so **saving a file updates the running app**:

```bash
docker compose --profile dev up --build
```

- Reach the dev app the same way: **http://localhost:8080** (the `caddy-dev` proxy fronts the
  Vite dev server and `backend-dev`, and forwards the Vite HMR WebSocket so hot reload works).
- `backend-dev` runs `uvicorn --reload` (restarts on `backend/**.py` changes).
- `frontend-dev` runs the **Vite dev server with HMR**; saving a file updates the running app.

Stop with `Ctrl-C`, or `docker compose --profile dev down`.

## The `run_python` sandbox (Docker-out-of-Docker)

`run_python` launches a fresh, locked-down container per call against the **host** Docker daemon.
For that to work from inside the backend container:

1. The host Docker socket is mounted into `backend` (`/var/run/docker.sock`).
2. A shared scratch dir is mounted at the **same path** on host and in the container
   (`${SANDBOX_SHARED_DIR:-/sandbox-tmp}`, also the backend's `TMPDIR`) so the nested
   `docker run -v {tmp}:/out` resolves to the same directory the host daemon sees — that is how
   `/out` artifacts (PDFs, images) round-trip back as documents.
3. The `agent-sandbox` image must exist on the host daemon: `docker compose build sandbox`.

If `agent-sandbox` isn't built, `run_python` still runs Python on the stdlib-only
`python:3.12-slim` fallback (no fpdf2 / PDF support) and says so in its notes.

> Override the shared scratch path with `SANDBOX_SHARED_DIR` in `.env` if `/sandbox-tmp` is
> unsuitable on your host.

## Automatic model loading (same Docker socket)

The Model Manager (Settings → Models) reuses the mounted Docker socket for two things, but **only**
when llama-server is the bundled local container (`LLAMACPP_CONTAINER`, default
`agent_graph_llamacpp`):

- **GPU auto-detect** — the backend image has no `nvidia-smi`, so `POST /api/hardware/detect` falls
  back to `docker exec <container> nvidia-smi …` to read the GPU the container *can* see.
- **Load a model** — `POST /api/llamacpp/load` `docker inspect`s the llama-server container, then
  `docker rm -f` + recreates it serving the chosen GGUF (`-m /models/<file>` + the recommended
  flags). It replicates the inspected config — image, the **`llamacpp` network alias** (force-kept so
  the backend keeps reaching it), the models volume, the **GPU device** (CDI `--device
  nvidia.com/gpu=all`, or legacy `--runtime nvidia --gpus all` — derived from inspect, never
  hard-coded), and the restart policy — changing only the command.

A recreate detaches the container from compose until the next `docker compose --profile llamacpp up
-d llamacpp` reconciles it (the compose labels are replicated so reconciliation is clean). For a
hand-run or **remote** llama-server the endpoint returns `unmanaged` and the UI disables Load (copy
the Configure launch command instead). **Security:** like the sandbox, the socket gives the backend
host-root-equivalent power; the load endpoint contains this to a path-traversal-guarded library
filename + a recommender-generated command on the single `LLAMACPP_CONTAINER`, but you should still
auth-gate `/api` (at Caddy) before exposing the app beyond localhost.

## Service endpoints inside the compose network

The backend reaches the other services by name (set in `docker-compose.yml`):
`ARCADE_URL=http://arcadedb:2480`, `SEARXNG_URL=http://searxng:8080`. None of these are published
to the host — Caddy is the only host-facing port.

> The prod and dev backends (`backend` / `backend-dev`) share one config via a YAML anchor and
> both `expose` `:8000` on the network (no host port); only one runs at a time because each belongs
> to a single profile, and Caddy proxies `/api` to whichever is up.
