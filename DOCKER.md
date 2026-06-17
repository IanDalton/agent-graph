# Running agent-graph in containers

The whole stack is containerized: **arcadedb** (graph memory), **searxng** (web search),
**backend** (FastAPI/SSE app), and **frontend** (the React UI). Secrets/model config come from
`.env` in the repo root.

## Prerequisites

- Docker Desktop (or Docker Engine) running.
- A `.env` file (already present) with at least `OLLAMA_BASE_URL`/`OLLAMA_MODEL`, or
  `AGENT_MODEL` + the matching provider key (e.g. `OPENAI_API_KEY`). For local reasoning models,
  set `OLLAMA_NUM_PREDICT` (max answer tokens) and raise the Ollama server's `OLLAMA_CONTEXT_LENGTH`
  so long chains-of-thought aren't cut off mid-reasoning.

Everything lives in `docker-compose.yml`, selected by a **`dev` or `prod` profile**. The infra
services (`arcadedb`, `searxng`) have no profile, so they always come up under either profile (and
a bare `docker compose up` brings up just the infra for host-run development).

## Production stack (`prod` profile)

Built images, nginx-served UI:

```bash
docker compose build sandbox                 # one-time: the agent-sandbox image run_python launches
docker compose --profile prod up -d --build  # arcadedb + searxng + backend + frontend
```

- UI:        http://localhost:8080  (nginx serves the SPA and proxies `/api` → backend)
- API:       http://localhost:8000
- ArcadeDB:  http://localhost:2480
- SearXNG:   http://localhost:8085

## Dev stack (`dev` profile, auto-reloading)

Source is bind-mounted, so **saving a file updates the running app**:

```bash
docker compose --profile dev up --build
```

- `backend-dev` runs `uvicorn --reload` (restarts on `backend/**.py` changes).
- `frontend-dev` runs the **Vite dev server with HMR** at http://localhost:5173
  (proxies `/api` → the `backend-dev` service via `API_PROXY_TARGET`).

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

## Service endpoints inside the compose network

The backend reaches the other services by name (set in `docker-compose.yml`):
`ARCADE_URL=http://arcadedb:2480`, `SEARXNG_URL=http://searxng:8080`. The host-published ports
above are only for your browser/tools.

> The prod and dev backends (`backend` / `backend-dev`) share one config via a YAML anchor and
> both publish `:8000`; only one runs at a time because each belongs to a single profile.
