# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`agent-graph` is a **Pydantic AI agent with persistent graph memory** backed by ArcadeDB.
ArcadeDB is both the agent's memory (the LLM queries/stores conversation context) and the
business datastore (every turn, tool call, and error is persisted). A SearXNG instance is
included for web search. `testing.ipynb` holds the original prototype; the real implementation
now lives in the `backend/` package.

## Architecture

Data flows through one repository layer so tools and hooks never duplicate SQL. Module map:

- **`backend/db/arcade_db.py`** — `ArcadeClient`: async `httpx` wrapper over ArcadeDB's HTTP API.
  `command()` → `POST /api/v1/command/{db}` (writes/DDL); `query()` → `POST /api/v1/query/{db}`
  (idempotent reads). `ensure_schema()` creates the vertex/edge types + indexes idempotently.
  Connection comes from env (`ARCADE_URL`/`ARCADE_DATABASE`/`ARCADE_USER`/`ARCADE_PASSWORD`)
  with docker-compose defaults.
- **`backend/db/dependencies.py`** — `GraphDependencies(db, user_id, conversation_id)`, injected via
  `deps_type`. `user_id` isolates each user's memory; `conversation_id` scopes the current thread.
- **`backend/db/repository.py`** — the only place that runs SQL: `create_conversation`,
  `append_message`, `get_recent_messages`, `search_messages`, `store_fact`, `search_facts`,
  `write_log`. Graph model: `User -HAS_CONVERSATION-> Conversation -HAS_MESSAGE-> Message`,
  `User -KNOWS-> Fact`, `Conversation -LOGGED-> LogEntry`.
- **`backend/schemas/graph_schemas.py`** — Pydantic tool I/O: `RawQuery`, `StoreFactArgs`,
  `MemorySearchResult`/`MemoryHit`.
- **`backend/skills/graph_capability.py`** — the bundle, exposed via `build_memory()`:
  - a `Capability` with tools `search_memory`, `get_conversation_history`, `store_fact`, and a
    **read-only** `run_query` escape hatch (guarded by `is_read_only()` + the idempotent endpoint).
    Its instructions keep the CRITICAL RULE: *check existing data/schema before creating nodes*.
  - a `Hooks` object that auto-persists: `before_run` creates the conversation, `after_run` writes
    the user + assistant turn, `after_tool_execute` logs each tool call, `run_error` logs failures.
- **`backend/main.py`** — `build_agent()` (model from `AGENT_MODEL`, else local Ollama via
  `OLLAMA_MODEL`) and an async `run(prompt, user_id, conversation_id)` that streams events using the
  `async with agent.run_stream_events(...) as stream:` form (the bare `async for` form is deprecated).

## Infrastructure (docker-compose.yml)

- **arcadedb** (`agent_memory_db`) — graph DB. HTTP API on `:2480`, binary on `:2424`. Database
  `AgentMemory`. Server superuser `root` / password `playwithdata`; the per-database `admin` user
  (from `defaultDatabases`) **cannot alter the schema**, so `ArcadeClient` defaults to `root`.
  Data persisted in the `arcadedb_data` volume.
- **searxng** — web search at `http://localhost:8085`. Expects config in `./docker/searxng` (this
  directory does not exist yet and must be created before the service is useful).

## Commands

```bash
docker compose up -d arcadedb     # start ArcadeDB (DB AgentMemory auto-created)
pip install -r requirements.txt   # pydantic-ai-slim[openai], httpx, python-dotenv
python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
python -m pytest backend/tests/   # unit tests run without a DB; the integration test skips if :2480 is down
```

Local-model runs also need a reachable Ollama (`OLLAMA_MODEL`); set `AGENT_MODEL` (e.g.
`openai:gpt-5.2`) to use a hosted provider instead. Secrets load from `.env` via `python-dotenv`.

## Conventions / gotchas

- Built on **Pydantic AI** — use the Pydantic AI skill for `Agent`, `Capability`, `Hooks`,
  `RunContext`, `deps_type`, and streaming APIs.
- **ArcadeDB SQL DDL quirk:** `IF NOT EXISTS` is a *suffix* for types/properties
  (`CREATE VERTEX TYPE X IF NOT EXISTS`) but a *prefix* for indexes
  (`CREATE INDEX IF NOT EXISTS ON ...`). See `ArcadeClient.ensure_schema`.
- All DB access goes through `backend/db/repository.py` — add new persistence there, not inline in
  tools or hooks, so the two paths stay consistent.
- Keep the read-only guard on `run_query` (`is_read_only` + the idempotent `query()` endpoint) and
  the "check existing data/schema first" instruction when editing the capability.
