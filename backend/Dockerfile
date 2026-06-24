# Backend image: FastAPI/SSE app (uvicorn) + the Docker CLI.
#
# Why the Docker CLI? The `run_python` sandbox (backend/sandbox/runner.py) shells out to
# `docker run` against the HOST daemon. Running the backend in a container makes this
# Docker-out-of-Docker: we install only the CLI here and mount the host's docker.sock at
# runtime (see docker-compose.yml). The daemon itself stays on the host.
FROM python:3.12-slim AS base

# Static Docker CLI binary (client only — no daemon). amd64/arm64 selected from the build arch.
ARG DOCKER_CLI_VERSION=27.3.1
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) dockerarch=x86_64 ;; \
        arm64) dockerarch=aarch64 ;; \
        *) echo "unsupported arch: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${dockerarch}/docker-${DOCKER_CLI_VERSION}.tgz" \
        | tar -xz -C /usr/local/bin --strip-components=1 docker/docker; \
    docker --version; \
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code (the dev compose overlays this with a bind mount for hot reload).
COPY backend ./backend

# Shared temp dir for sandbox artifacts. It is mounted at the SAME host path so the
# nested `docker run -v {tmp}:/out` the runner emits resolves to the same directory the
# host daemon sees. See docker-compose.yml (SANDBOX_SHARED_DIR / TMPDIR).
ENV TMPDIR=/sandbox-tmp
RUN mkdir -p /sandbox-tmp

# Downloaded GGUF models live here (a shared volume in compose). No nvidia-smi in this image —
# the backend can't see a remote GPU anyway, so hardware auto-detect returns nothing and the
# manual hardware profile (the source of truth) is used. See backend/models/.
ENV LLAMACPP_MODELS_DIR=/models
RUN mkdir -p /models

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
