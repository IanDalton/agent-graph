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
  (idempotent reads). `ensure_schema()` creates the vertex/edge types + indexes idempotently;
  `ensure_database()` creates the database itself if missing (via `POST /api/v1/server`
  `create database`, root only). **Per-user isolation:** `database_name_for_user(user_id)` maps
  each user to their own database (`{ARCADE_DATABASE}_{sanitized}_{hash}`, e.g.
  `AgentMemory_u1_3f2a1b9c`) so one user's data can never appear in another user's queries.
  `ARCADE_DATABASE` is the *base/prefix*, not a literal database. Connection comes from env
  (`ARCADE_URL`/`ARCADE_DATABASE`/`ARCADE_USER`/`ARCADE_PASSWORD`) with docker-compose defaults.
- **`backend/db/dependencies.py`** — `GraphDependencies(db, user_id, conversation_id, proposed_schemas)`,
  injected via `deps_type`. The per-user database (carried by `db`) is the real isolation boundary;
  `user_id` still keys the `User`/`Fact` vertices and survives in `WHERE` filters as
  defense-in-depth. `conversation_id` scopes the current thread. `proposed_schemas` is run-scoped
  state (a fresh dict per run) backing the ontology pipeline's ordering guard.
- **`backend/db/repository.py`** — the only place that runs SQL: `create_conversation`,
  `append_message`, `get_recent_messages`, `search_messages`, `store_fact`, `search_facts`,
  `write_log`, `update_fact`/`delete_fact` (revise/remove a fact by `fact_id`, user-scoped), plus the
  ontology DDL/DML: `vertex_type_exists`, `list_vertex_types`,
  `create_vertex_type` (usage stored as type-level `CUSTOM description`), `create_node`,
  `create_edge_type`, `node_type`/`node_exists` (tolerant: a bad rid → `None`/`False`, never raises),
  `type_category` (`'vertex'`/`'edge'`/`None` from `schema:types`), `create_edge`,
  `update_node`/`delete_node` (revise/remove an instance by rid, user-scoped; delete cleans edges via
  `DELETE VERTEX FROM (SELECT …)`), and `drop_vertex_type`/`drop_edge_type` (retire a whole type:
  delete all instances/edges then `DROP TYPE … IF EXISTS` — `DROP TYPE` refuses a non-empty type, so
  the records go first; vertex types clear via `DELETE VERTEX FROM <T>`, edge types via
  `DELETE FROM <T> UNSAFE`, which is required to delete edge records and does clean endpoint adjacency).
  `search_facts` returns `fact_id`.
  **Faithful replay:** `append_run_messages`/`get_run_history` store each run's serialized Pydantic
  AI messages (via `new_messages_json()`) as `RunMessages` vertices — tool calls AND their returns
  included — *separately* from the human-readable `Message` vertices. `Message` rows keep role/content
  text for `search_messages`/`get_recent_messages`; `RunMessages` blobs are what `main.run()` reloads
  into `message_history` so the agent sees the tool work it actually did (not just its text claims)
  and stops re-doubting/redoing completed work.
  Graph model: `User -HAS_CONVERSATION-> Conversation -HAS_MESSAGE-> Message`,
  `Conversation -HAS_RUN_MESSAGES-> RunMessages`,
  `User -KNOWS-> Fact`, `Conversation -LOGGED-> LogEntry`, `User -HAS_NODE-> <agent-created type>`,
  plus agent-created `<instance> -<EDGE_TYPE>-> <instance>` relationships.
- **`backend/schemas/graph_schemas.py`** — Pydantic tool I/O: `RawQuery`, `StoreFactArgs`,
  `MemorySearchResult`/`MemoryHit`, and the ontology models `VertexProperty`, `ProposeSchemaArgs`,
  `SchemaProposal`, `VertexTypeInfo`, `CreateNodeArgs` (identifier/type validators here are the
  DDL-injection boundary, since ArcadeDB can't bind type/property *names* as parameters).
- **`backend/skills/graph_capability.py`** — the bundle, exposed via `build_memory()`:
  - a `Capability` with tools `search_memory` (returns each fact's `fact_id`),
    `get_conversation_history`, `store_fact`, `update_fact`/`delete_fact` (revise/remove a fact in
    place to avoid duplicates), and a **read-only** `run_query` escape hatch (guarded by
    `is_read_only()` + the idempotent endpoint). `run_query` is *tolerant* and can never abort the
    run: a query-level error — notably a `SELECT FROM <Type>` (vertex **or** edge) where the type
    doesn't exist yet, which ArcadeDB answers with **500** + a `SchemaException` instead of an empty
    result — is returned as an ordinary `no_records` result row (with the DB's `detail` + a `hint`),
    **not** a `ModelRetry` (the tool's `max_retries` is 1, so retrying a second missing-type query
    would raise `UnexpectedModelBehavior` and crash the run). Querying a not-yet-created type is the
    normal "check before create" path and its truthful answer is "there are none"; the model reads
    the note and proceeds to `list_vertex_types`/create. A genuine transient **503** still propagates.
    Its instructions keep the CRITICAL RULE: *check
    existing data/schema before creating nodes*, and *update an existing fact rather than duplicating*.
  - a `Hooks` object that auto-persists: `before_run` creates the conversation, `after_run` writes
    both the serialized run messages (`RunMessages`, for faithful replay) and the human-readable
    user + assistant turn (`Message`, for search), `after_tool_execute` logs each tool call,
    `run_error` logs failures.
    **All persistence here is best-effort** (`_best_effort`): a DB failure (e.g. a 503 that outlasts
    the client's retries) is logged via the `agent_graph.*` loggers and swallowed, never crashing the
    agent loop. `ArcadeClient._request_with_retry` retries 503s + transport errors with capped
    backoff and logs each retry; `main._configure_logging` sends `agent_graph.*` logs to stderr
    (level via `LOG_LEVEL`).
- **`backend/skills/ontology_capability.py`** — the `OntologyManager` bundle, exposed via
  `build_ontology()`: lets the agent grow its own ontology through a guarded pipeline.
  - a `Capability` with vertex tools `list_vertex_types` (read the current ontology + usage notes),
    `propose_schema_change` (cognitive layer — validates + records, no DB write),
    `create_vertex_type` (creates the **type**), `create_node` (creates an **instance** of an
    existing type, linked to the user via `HAS_NODE`); and the parallel edge tools
    `propose_edge_type` → `create_edge_type` (UPPER_SNAKE_CASE relationship type) → `create_edge`
    (connects two existing instances by record id); plus `update_node`/`delete_node` to revise/remove
    an existing instance by rid (avoids duplicates), and `delete_vertex_type`/`delete_edge_type` to
    **retire a whole type** the agent created (drops the type AND all its instances/edges — full
    schema control). Flow: list → propose → create_*_type → create_node/create_edge.
    Instances/edges can only be created for types that already exist. The destructive tools require
    no proposal but are guarded: they validate the identifier, confirm the type exists *and* matches
    the requested category (`type_category` — so you can't `delete_edge_type` a vertex type, whose
    `UNSAFE` delete would strip records), and refuse the internal types. `_PROTECTED_VERTEX_TYPES`
    (User, Conversation, Message, Fact, LogEntry, **RunMessages**) and `_PROTECTED_EDGE_TYPES`
    (HAS_CONVERSATION, HAS_MESSAGE, HAS_RUN_MESSAGES, KNOWS, LOGGED, HAS_NODE) can never be
    edited/dropped here; `update_node`/`delete_node` reject the protected *vertex* types too.
  - a `Hooks` object whose `before_tool_execute` guard rejects `create_vertex_type` /
    `create_edge_type` unless a matching `propose_schema_change` / `propose_edge_type` ran earlier in
    the same run (uses `proposed_schemas` / `proposed_edges` on the deps).
- **`backend/web/client.py`** — `WebClient`: async `httpx` wrapper for the live internet, modeled on
  `ArcadeClient` (env-driven, context manager, capped-backoff retry on transport/5xx). `search()`
  hits SearXNG's JSON API (`GET {SEARXNG_URL}/search?format=json`, default
  `http://localhost:8085`) and returns the trimmed `results` list; `fetch()` downloads a page
  (byte-capped → `truncated`) and runs it through the stdlib-only `html_to_text` extractor.
- **`backend/schemas/search_schemas.py`** — web tool I/O: `WebSearchArgs`/`WebSearchHit`/
  `WebSearchResult`, `FetchUrlArgs` (its `http`/`https`-only validator is the safety boundary) and
  `FetchPageResult`.
- **`backend/skills/search_capability.py`** — the `WebSearch` bundle, exposed via `build_search()`:
  a `Capability` with `web_search` (SearXNG search → ranked title/url/snippet) and `fetch_url`
  (download + read a page's text). Both take the `WebClient` from `ctx.deps.web` when present, else
  build a short-lived one from env. **Tolerant** like `run_query`: any failure (SearXNG down, HTTP
  error, bad page) is caught and returned as a structured `error` result, never raised — so a web
  hiccup can't abort the run. Tool calls are logged automatically by the memory capability's
  `after_tool_execute` hook (no extra persistence here).
- **`backend/main.py`** — `build_agent()` (model from `AGENT_MODEL`, else local Ollama via
  `OLLAMA_MODEL`) and an async `run(prompt, user_id, conversation_id)` that points `ArcadeClient` at
  the user's own database (`database_name_for_user`), calls `ensure_database()` then `ensure_schema()`,
  loads prior turns into `message_history` via `_to_message_history(repo.get_run_history(...))` (the
  serialized `RunMessages` blobs deserialized with `ModelMessagesTypeAdapter` — faithful, tool calls
  included; a corrupt blob is skipped, not fatal), and streams events using the
  `async with agent.run_stream_events(...) as stream:` form (the bare `async for` form is deprecated).
  `build_agent()` adds `build_search()` to the capability list, and `run()` opens a `WebClient`
  alongside the `ArcadeClient` and injects it via `GraphDependencies(web=...)`. The streaming itself
  lives in `stream_run(prompt, user_id, conversation_id)`, an async generator that maps Pydantic AI
  events to a **stable event vocabulary** — `thinking`/`text`/`tool_call`/`tool_result`/`final` dicts
  — so callers never depend on the library's event classes; `run()` just consumes it for the CLI
  (thinking in blue, text plain). This is the single streaming source of truth, shared with the API.
- **`backend/api.py`** — `app`: a **thin** FastAPI/SSE wrapper over the existing machinery (adds no
  DB/agent logic; every handler calls `repo.*` / `main.stream_run`, opening a short-lived
  `ArcadeClient` on the caller's per-user DB via `_client_for(user_id)`). Endpoints: `GET /api/config`
  (read-only model/DB/search/log surface, **no secrets**), `GET|POST /api/conversations` (list /
  create — create mints a uuid `conversation_id`), `GET /api/conversations/{id}/messages`,
  `GET /api/conversations/{id}/summary` (one-shot LLM digest; tolerant — errors return an empty
  summary), and `POST /api/chat/stream` (a `StreamingResponse` of `text/event-stream` that writes each
  `stream_run` event as one `data:` frame; a failure becomes a final `{"type":"error"}` frame rather
  than a dropped connection). CORS allows the Vite dev origin (`:5173`). This backs the `frontend/` UI.

## Frontend (`frontend/`)

React + Vite + TypeScript SPA using **shadcn/ui** components (Tailwind + Radix). A three-pane
"Mission Control" shell built so future modes (Research/Swarm/Council) slot in as components:
left `Sidebar` (conversation list + New Chat, with a per-row mode icon — only 💬 Regular today),
middle `Canvas` (streaming chat bubbles + collapsible tool-call chips, the seed of the future
chain-of-thought timeline), right `ContextPane` (read-only config card + live LLM summary). State is
plain React Context (`AppContext`: `userId`, conversations, active id) + a `useChat` reducer for
per-conversation message/streaming state — no React Query, to keep the dependency surface small.
Streaming uses `fetch` + a `ReadableStream` reader (in `api/stream.ts`) since `EventSource` is
GET-only; the frame vocabulary mirrors `stream_run`'s. Dev: `npm run dev` (proxies `/api` → `:8000`).

## Infrastructure (docker-compose.yml)

- **arcadedb** (`agent_memory_db`) — graph DB. HTTP API on `:2480`, binary on `:2424`. The compose
  `defaultDatabases=AgentMemory` only seeds the base/template database; the real per-user databases
  (`AgentMemory_<user>_<hash>`) are created on demand by `ensure_database()`. Server superuser `root`
  / password `playwithdata`; the per-database `admin` user (from `defaultDatabases`) **cannot alter
  the schema or create databases**, so `ArcadeClient` defaults to `root`. Data persisted in the
  `arcadedb_data` volume.
- **searxng** — web search at `http://localhost:8085`, backing the `WebSearch` capability. Config in
  `./docker/searxng/settings.yml`, which enables the **JSON output format** (`search.formats: [html,
  json]`) the `WebClient` calls and turns the bot **`limiter` off** — both are disabled by default,
  so the JSON API is unusable without it. The placeholder `secret_key`/`limiter: false` are dev-only.
  Override the base URL with `SEARXNG_URL` if not on the compose default.

## Commands

```bash
docker compose up -d arcadedb searxng  # ArcadeDB (DB AgentMemory auto-created) + SearXNG (web search)
pip install -r requirements.txt   # pydantic-ai-slim[openai], httpx, python-dotenv, fastapi, uvicorn
python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
python -m pytest backend/tests/   # unit tests run without a DB/network; the integration test skips if :2480 is down
uvicorn backend.api:app --reload --port 8000   # HTTP/SSE API backing the web UI
cd frontend && npm install && npm run dev       # web UI on http://localhost:5173 (proxies /api -> :8000)
# verify the SearXNG JSON API the WebClient depends on:
curl "http://localhost:8085/search?q=arcadedb&format=json"
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
- The web tools (`web_search`/`fetch_url`) must stay **tolerant**: catch failures and return an
  `error` result rather than raising, so a SearXNG/network hiccup never aborts the agent run (same
  contract as `run_query`). Keep the `http`/`https`-only validator on `FetchUrlArgs`.
