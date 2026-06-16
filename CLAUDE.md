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
  state (a fresh dict per run) backing the ontology pipeline's ordering guard. `model` carries the
  UI-selected model label so delegated sub-agents (swarm/deep research) run on the conversation's
  model.
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
  `search_facts` returns `fact_id`. **Documents:** `create_document`/`update_document`/
  `get_document`/`list_documents`/`delete_document` persist agent-authored artifacts as `Document`
  vertices (user-scoped, `updated_at`-ordered; `list_documents` returns metadata only, no bodies);
  `update_document` is shared by agent revisions AND user edits from the web UI. A document's
  `encoding` is `"text"` (literal content) or `"base64"` (binary artifacts — PDFs/images the
  sandbox produced).
  **Modes:** `create_conversation(..., mode=)` stamps the conversation's agent profile at creation
  (`'regular'`/`'research'`/`'swarm'`; the idempotent hook call can't overwrite it because an
  existing conversation returns early), `set_conversation_mode` switches it later (the user can
  change a conversation's mode mid-thread — `stream_run` re-reads the stored mode each turn, so the
  change persists), `get_conversation_mode` reads it back (default `'regular'` for unknown/pre-mode
  conversations), and `list_conversations` reports it with the same fallback.
  **Swarm roster:** `create_agent_spec`/`get_agent_spec` (by id OR name)/`list_agent_specs`/
  `update_agent_spec`/`delete_agent_spec` persist sub-agent definitions (name, role, system prompt,
  tool grants) as `AgentSpec` vertices, user-scoped, linked `User -HAS_AGENT-> AgentSpec`.
  **Faithful replay:** `append_run_messages`/`get_run_history` store each run's serialized Pydantic
  AI messages (via `new_messages_json()`) as `RunMessages` vertices — tool calls AND their returns
  included — *separately* from the human-readable `Message` vertices. `Message` rows keep role/content
  text for `search_messages`/`get_recent_messages`; `RunMessages` blobs are what `main.run()` reloads
  into `message_history` so the agent sees the tool work it actually did (not just its text claims)
  and stops re-doubting/redoing completed work.
  Graph model: `User -HAS_CONVERSATION-> Conversation -HAS_MESSAGE-> Message`,
  `Conversation -HAS_RUN_MESSAGES-> RunMessages`, `Conversation -HAS_DOCUMENT-> Document`,
  `User -KNOWS-> Fact`, `Conversation -LOGGED-> LogEntry`, `User -HAS_NODE-> <agent-created type>`,
  `User -HAS_AGENT-> AgentSpec`,
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
    (User, Conversation, Message, Fact, LogEntry, **RunMessages**, **Document**, **AgentSpec**) and
    `_PROTECTED_EDGE_TYPES` (HAS_CONVERSATION, HAS_MESSAGE, HAS_RUN_MESSAGES, KNOWS, LOGGED,
    HAS_NODE, HAS_DOCUMENT, HAS_AGENT) can never be
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
- **`backend/schemas/document_schemas.py`** — document tool I/O: `CreateDocumentArgs` (mime-type
  validator, default `text/markdown`), `UpdateDocumentArgs`, `DocumentInfo` (metadata, no body),
  `DocumentContent` (full document).
- **`backend/skills/document_capability.py`** — the `Documents` bundle, exposed via
  `build_documents()`: lets the agent author durable artifacts (reports, plans, notes, code) the
  user sees — and can EDIT, when text-based — in the web UI's Documents tab. Tools:
  `create_document` (markdown by default; returns `document_id`), `update_document` (revise in
  place — replaces the FULL content), `read_document`, `list_documents` (current conversation,
  metadata only), `delete_document`. Its instructions enforce the house style: check
  `list_documents` before creating (no near-duplicates), and **always `read_document` before
  revising** — the user may have edited the body since the agent last wrote it. Bad ids raise
  `ModelRetry` (mirroring `update_fact`/`delete_fact`).
- **`backend/sandbox/runner.py`** — `PythonSandbox`: containerized Python execution via ephemeral
  `docker run --rm` against the host's Docker daemon (no extra Python dependency). Each call is a
  fresh, locked-down container: `--network none`, `--memory`/`--cpus`/`--pids-limit` caps,
  `--cap-drop ALL`, `--read-only` root FS with a small tmpfs `/tmp`, non-root (`--user nobody`),
  hard wall-clock timeout (default 30s, max 120s), and per-stream output caps. The code is piped
  over **stdin** (`python -I -`) so nothing is shell-interpolated and there is no argv length
  limit. On timeout the container is `docker rm -f`'d by name (killing the docker CLI alone would
  leave it running). **File artifacts:** a host temp dir is mounted at `/out` (also `$OUTPUT_DIR`);
  files the program writes there come back as `SandboxFile` blobs on the result (max 8 files,
  5MB each — drops noted in `notes`). Default image is the project's **`agent-sandbox`**
  (`docker/sandbox/Dockerfile`: python:3.12-slim + **fpdf2** for PDFs; `docker compose build
  sandbox`); if it isn't built, the run transparently retries on plain `python:3.12-slim`
  (stdlib only) and says so in `notes`. Env: `SANDBOX_IMAGE`, `SANDBOX_TIMEOUT_SECONDS`,
  `SANDBOX_MEMORY`, `SANDBOX_CPUS`, `SANDBOX_NETWORK`. **Failure contract:** `run()` never raises
  for expected problems (Docker missing, timeout) — it returns a `SandboxResult` describing what
  happened; a non-zero exit + traceback in `stderr` is a normal result the model reads and fixes.
- **`backend/schemas/sandbox_schemas.py`** — sandbox tool I/O: `RunPythonArgs` (code +
  optional 1–120s timeout override) and `PythonRunResult` (stdout/stderr/exit_code/timed_out/
  truncated/`documents` — the /out files already persisted as documents — /notes/error).
- **`backend/skills/sandbox_capability.py`** — the `PythonSandbox` bundle, exposed via
  `build_sandbox()`: one tool, `run_python`. **Stateless** (fresh container per call — the
  instructions tell the model to send complete, self-contained programs), stdlib + fpdf2, no
  network. Files the program wrote to `/out` are persisted here as `Document` vertices
  (text mimes as literal text, binary as **base64**; deterministic mime map in `_mime_for`) and
  returned in `PythonRunResult.documents` so the model references them instead of re-creating
  them. The capability instructions carry the **PDF skill** (the fpdf2 recipe: write to
  `/out/report.pdf`, Helvetica/multi_cell, latin-1-safe text). Takes the sandbox from
  `ctx.deps.sandbox` when present (test seam), else builds one from env. **Tolerant** like
  `run_query`/`web_search`: every failure becomes a structured `error` result, never an
  exception, so Docker being offline can't abort the run (a per-file persistence failure becomes
  a `note`, keeping the code's stdout).
- **`backend/skills/subagent.py`** — the shared **delegated-run machinery** + the agency
  communication primitive: `run_subagent(deps, instructions=, tool_groups=, prompt=, recipients=,
  request_limit=, model=)` builds a fresh single-purpose Pydantic AI agent per dispatch with a
  granted subset of the existing capability bundles (`capabilities_for`:
  `web`/`documents`/`sandbox`/`memory` — tool capabilities ONLY, never `persistence_hooks`, since
  the parent run's `after_run` already records the orchestrating turn) and runs it on the parent's
  deps (same per-user DB/conversation/web/sandbox; fresh `proposed_*` dicts) with a
  `UsageLimits(request_limit=)` runaway backstop (`SUBAGENT_REQUEST_LIMIT`, default 25). A
  `_document_collector` Hooks records every document the delegate persists (create_document results
  + run_python artifacts) onto `SubagentOutcome.documents`, which is how delegated artifacts reach
  the UI (see `main`'s document frames). **Agency multi-hop:** when the dispatched specialist has
  its own chart edges (`recipients`) and its depth is under `SWARM_MAX_DEPTH` (default 3), it is
  granted its own `send_message` tool via `build_communication_capability()`, and its deps are
  stamped with `agency_recipients`/`agency_depth` so its own dispatches are chart-enforced and
  depth-bounded. `dispatch_message(deps, recipient, message, context)` is the shared send_message
  mechanism (used by the orchestrator's tools AND every specialist's granted send_message): it
  enforces the caller's chart (`deps.agency_recipients` — `None` = entry point, may message anyone),
  resolves the recipient spec, and runs it one hop deeper. **Tolerant**: `run_subagent` /
  `dispatch_message` never raise — failures (incl. exhausted limits, bad/out-of-chart recipients)
  come back as `SubagentOutcome.error` / an `error` report, keeping any documents persisted before
  the failure. `model` is a test seam; production resolves `deps.model` (the UI-selected label) so
  delegates run on the conversation's model.
- **`backend/schemas/swarm_schemas.py`** — swarm tool I/O: `TOOL_GROUPS` (the grantable bundles),
  `CreateAgentArgs` (kebab-case name validator, tools-subset validator, default grant
  `["web", "documents"]`, `recipients` = the agent's outgoing chart edges, validated kebab-case),
  `UpdateAgentArgs`, `AgentSpecInfo` (carries `recipients`), `SendMessageArgs` (recipient by id or
  name + self-contained message/context), `SendMessagesArgs` (1–8 independent messages),
  `AgentRunReport`/`SwarmRunResult` (tolerant — failures land in `error`),
  `DeepResearchArgs`/`DeepResearchResult`.
- **`backend/skills/swarm_capability.py`** — the `SwarmOrchestrator` bundle, exposed via
  `build_swarm()` and added **only in swarm-mode conversations**: the main agent becomes the
  **entry point of an agency** — a **pure router** that designs its own specialists AND the
  communication chart between them (each `AgentSpec.recipients` are the teammates that agent may
  `send_message`). **The orchestrator has NO "doing" tools** — `build_agent` gives swarm mode only
  memory + this bundle, so it can't browse, run code, grow the ontology, or write documents; it
  *must* delegate. Roster tools `list_agents`/`create_agent`/`update_agent`/`delete_agent`
  (persisted `AgentSpec`s — durable across turns/conversations; duplicates/unknown names raise
  `ModelRetry`); communication tools `send_message` (one recipient) and `send_messages` (a batch of
  INDEPENDENT messages delivered **concurrently** via `asyncio.gather` under a `SWARM_MAX_PARALLEL`
  semaphore, default 4; reports return in message order and one failure never affects the rest) —
  both delegate to `subagent.dispatch_message`; plus a built-in `deep_research` tool (web+documents
  delegate on `DEEP_RESEARCH_INSTRUCTIONS`, request limit `DEEP_RESEARCH_REQUEST_LIMIT`, default
  40). The entry point may message any roster agent; a dispatched specialist may only message its
  own `recipients`, and those messages flow multi-hop along the chart (bounded by
  `SWARM_MAX_DEPTH`). **Seeded roster:** `swarm_seed_hooks` (a `before_run` hook in `build_swarm`)
  creates `DEFAULT_SWARM_AGENTS` — `web-researcher`, `report-writer`, `website-builder` (live HTML
  doc), `pdf-author` (fpdf2 PDF), `presentation-designer` (slide-deck PDF) — on a user's **first**
  swarm turn (seed-when-empty: skipped if the user already has any agent, so later edits/deletes
  aren't fought); best-effort, never blocks the turn. Deliverables are the documents the specialists
  produce; the orchestrator references (never recreates) them. Communication is tolerant end-to-end:
  an unknown/out-of-chart recipient or broken delegate becomes an `error` report, never an
  exception. Specialists don't see the conversation — instructions make every message
  self-contained.
- **`backend/skills/research_capability.py`** — `DeepResearch`: the shared research method
  (plan sub-questions → fan out searches → fetch/cross-check sources → synthesize a cited
  markdown report via create_document). `build_research()` returns an **instructions-only**
  capability overlaid in research-mode conversations (the work runs on the existing
  web/document tools, so every step streams as normal tool chips);
  `DEEP_RESEARCH_INSTRUCTIONS` is the same method framed as a delegate's system prompt, used by
  the swarm's `deep_research` tool.
- **`backend/skills/system_prompt.py`** — the agent-level system prompt (distinct from the
  tool-scoped capability `instructions`). `BASE_SYSTEM_PROMPT` is a fixed best-practices identity +
  behaviour prompt (agency, memory use, honesty/no-fabrication, style) attached to the **main agent
  only** via `Agent(instructions=...)`. `register_system_prompt(agent)` adds two **dynamic**
  `@agent.instructions` callables: the current date, and — the key piece — `relevant_facts_block`,
  which embeds the run's latest user prompt (`_latest_user_prompt(ctx.messages)`) and injects the
  top-`_MAX_FACTS` most relevant stored facts via the existing `repo.search_facts(..., embedding=)`
  (semantic ranking with LIKE fallback). So every turn starts grounded in what we know about the
  user without waiting for the model to call `search_memory`. **Best-effort** like the persistence
  hooks: any DB/embedder failure logs (`agent_graph.system_prompt`) and degrades to no fact block,
  never aborting a turn. Sub-agent delegates keep their own task-specific prompts (not wired here).
  **Custom per-conversation prompt:** a conversation can carry its own extra instructions (set from
  the web UI's Configuration card, stored on the `Conversation` vertex). `main.compose_instructions`
  appends them under an `ADDITIONAL INSTRUCTIONS (from the user)` header on top of
  `BASE_SYSTEM_PROMPT`; `stream_run` reads them each turn via `repo.get_conversation_system_prompt`
  (tolerant) and passes them to `build_agent(..., system_prompt=)`. Main agent only.
- **`backend/main.py`** — `build_agent()` (model from `AGENT_MODEL`, else local Ollama via
  `OLLAMA_MODEL`) and an async `run(prompt, user_id, conversation_id)` that points `ArcadeClient` at
  the user's own database (`database_name_for_user`), calls `ensure_database()` then `ensure_schema()`,
  loads prior turns into `message_history` via `_to_message_history(repo.get_run_history(...))` (the
  serialized `RunMessages` blobs deserialized with `ModelMessagesTypeAdapter` — faithful, tool calls
  included; a corrupt blob is skipped, not fatal), and streams events using the
  `async with agent.run_stream_events(...) as stream:` form (the bare `async for` form is deprecated).
  `build_agent()` adds `build_search()` to the capability list, attaches the best-practices
  `BASE_SYSTEM_PROMPT` and calls `register_system_prompt(agent)` (see
  `backend/skills/system_prompt.py` — base prompt + auto-loaded relevant user facts), and `run()`
  opens a `WebClient` alongside the `ArcadeClient` and injects it via `GraphDependencies(web=...)`.
  **Modes:** `build_agent(model, effort, mode)` keeps the full base capability set for `regular`
  and `research` (the latter overlays `build_research()`); `swarm` instead builds a **lean
  pure-router** set — only `Thinking` + `build_memory()` (tools + persistence hooks) +
  `build_swarm()`, NO web/sandbox/ontology/documents — so the orchestrator can't do work itself and
  must delegate. `MODES`/`DEFAULT_MODE` are the source of truth, unknown values fall back to
  regular. `stream_run`
  resolves the conversation's stored mode via `repo.get_conversation_mode` and its custom prompt via
  `repo.get_conversation_system_prompt` (both tolerantly — a lookup failure means regular / no custom
  prompt) *before* building the agent, and passes the UI model label into
  `GraphDependencies(model=...)` so swarm/deep-research delegates run on the same model. The
  streaming itself
  lives in `stream_run(prompt, user_id, conversation_id)`, an async generator that maps Pydantic AI
  events to a **stable event vocabulary** — `thinking`/`text`/`tool_call`/`tool_result`/`document`/
  `final` dicts — so callers never depend on the library's event classes; `run()` just consumes it
  for the CLI (thinking in blue, text plain). `document` frames
  (`{action, document_id, title, mime_type}`) are emitted right after the tool_result of
  `create_document`/`update_document`/`run_python`/`send_message`/`deep_research`/`send_messages`
  (see `_document_events`; update's id comes from
  the call args tracked per tool_call_id since its return is a plain string; the others yield one
  frame per document on the result — run_python's /out artifacts, a delegate's
  `documents`, or each report's documents inside `send_messages.reports`) — the UI uses them to drop
  artifact cards into the chat and
  spotlight the document in the side panel. **`_jsonable` recurses** into dicts/lists and dumps
  Pydantic models (`model_dump(mode="json")`) — tool results can be containers OF models (e.g.
  `list_documents` → `list[DocumentInfo]`), and a bare `json.dumps` on those used to kill the SSE
  stream; `api._sse` also passes `default=str` as a net. This is the single streaming source of
  truth, shared with the API.
- **`backend/api.py`** — `app`: a **thin** FastAPI/SSE wrapper over the existing machinery (adds no
  DB/agent logic; every handler calls `repo.*` / `main.stream_run`, opening a short-lived
  `ArcadeClient` on the caller's per-user DB via `_client_for(user_id)`). Endpoints: `GET /api/config`
  (read-only model/DB/search/log surface incl. the selectable `modes` and the `base_system_prompt`,
  **no secrets**),
  `GET|POST /api/conversations` (list /
  create — create mints a uuid `conversation_id` and stamps the requested `mode`, validated by a
  `Literal`), `PATCH /api/conversations/{id}` (partial update of `mode` and/or the custom
  `system_prompt`; only the fields in `model_fields_set` are applied, so `system_prompt:""` clears
  it), `GET /api/conversations/{id}/messages`,
  `GET /api/conversations/{id}/summary` (one-shot LLM digest; tolerant — errors return an empty
  summary), the document surface — `GET /api/conversations/{id}/documents` (metadata list,
  tolerant), `GET|PUT|DELETE /api/documents/{id}` (PUT applies a **user edit** of title/content via
  `repo.update_document` and returns the updated record; 404 when not the caller's) — and
  `POST /api/chat/stream` (a `StreamingResponse` of `text/event-stream` that writes each
  `stream_run` event as one `data:` frame; a failure becomes a final `{"type":"error"}` frame rather
  than a dropped connection). CORS allows the Vite dev origin (`:5173`). This backs the `frontend/` UI.

## Frontend (`frontend/`)

React + Vite + TypeScript SPA using **shadcn/ui** components (Tailwind + Radix). A three-pane
"Mission Control" shell built so modes slot in as components (Council still reserved):
left `Sidebar` (conversation list with a per-row mode icon — 💬 Regular, 🔬 Deep Research,
🕸 Swarm — and a **split New Chat button**: the main button creates a regular chat, the chevron
opens a downward mode menu — local to the Sidebar, since the shared `ui/popover.tsx` anchors
upward for the composer. The chosen mode rides `POST /api/conversations`; it can be changed later
mid-conversation via the **mode chip** in the composer (`Composer`'s `ModeChip` →
`AppContext.setConversationMode` → `PATCH /api/conversations/{id}`), which optimistically updates
the local row so the sidebar icon and the `Canvas` renderer switch at once),
middle `Canvas` (streaming chat bubbles + collapsible tool-call chips, the seed of the future
chain-of-thought timeline), right `ContextPane` (440px) — **tabbed** (hand-rolled `ui/tabs.tsx`,
no Radix dep, like the popover; supports controlled `value`/`onValueChange`): a *Context* tab
(config + summary + memory graph; the config card's `SystemPromptRow` is a per-conversation custom
system-prompt textarea, saved on blur via `AppContext.setConversationSystemPrompt`) and a
*Documents* tab (`panes/DocumentsPane.tsx`) listing the
active conversation's agent-authored documents. Opening one renders by media type
(`DocumentBody`): **`text/html` runs as a live interactive app** in a sandboxed iframe
(`sandbox="allow-scripts allow-forms allow-popups"` — deliberately NO `allow-same-origin`, so
embedded apps can't reach our cookies/API; a toolbar button toggles preview ⇄ source),
**`application/pdf`** (base64) shows in an iframe via a data URL, **`image/*`** (base64) as an
`<img>`, markdown via the shared `Markdown` component, everything else as a monospace `<pre>`;
every document gets a download button (base64 decoded back to real bytes). Text-encoded documents
flip into a textarea editor whose Save PUTs `/api/documents/{id}`; base64 artifacts are read-only.
**Document spotlight:** a `document` stream frame appends an artifact-card step to the assistant
turn (`ChatBubble`'s `DocumentCard` — a big button, like Claude's artifacts) and, for `created`,
auto-features the document via `AppContext.featureDocument` → `featuredDoc {id, ts}`; the
`ContextPane` flips to the Documents tab and `DocumentsCard` opens that document (clicking a card
re-features it; `featuredDoc` is cleared on conversation switch, and the post-turn refresh no
longer closes an open document). The list also re-fetches on the same `refreshKey` bump the
summary uses (after each completed turn). State is
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
- **Python sandbox** — `run_python` launches an ephemeral, locked-down container per call via the
  host's `docker` CLI (see `backend/sandbox/runner.py`), *not* a long-running service. The compose
  `sandbox` entry is **build-only** (`profiles: [build-only]`, never started by `up`): it builds
  the `agent-sandbox` image (`docker/sandbox/Dockerfile` = python:3.12-slim + fpdf2 for PDFs) via
  `docker compose build sandbox`. Unbuilt ⇒ runs fall back to plain `python:3.12-slim`.

## Commands

The **whole stack is containerized** (arcadedb + searxng + backend + frontend); see `DOCKER.md`.

One `docker-compose.yml` with `dev`/`prod` profiles (infra is profile-less, so it always starts):

```bash
docker compose build sandbox                  # one-time: agent-sandbox image for run_python (python:3.12-slim + fpdf2)
docker compose --profile prod up -d --build   # prod stack (built images, nginx UI) — UI :8080, API :8000, DB :2480, SearXNG :8085
# hot-reloading dev stack (uvicorn --reload + Vite HMR; source bind-mounted, saving a file updates it):
docker compose --profile dev up --build       # UI :5173, API :8000
```

Run pieces directly (infra in Docker, app on the host) instead:

```bash
docker compose up -d arcadedb searxng  # just the infra (ArcadeDB DB AgentMemory auto-created + SearXNG)
pip install -r requirements.txt   # pydantic-ai-slim[openai], httpx, python-dotenv, fastapi, uvicorn
python -m backend.main "remember I like Recoleta apartments" --user u1 --conversation c1
python -m pytest backend/tests/   # unit tests run without a DB/network/Docker; integration tests skip when :2480 / the sandbox image is unavailable
uvicorn backend.api:app --reload --port 8000   # HTTP/SSE API backing the web UI
cd frontend && npm install && npm run dev       # web UI on http://localhost:5173 (proxies /api -> :8000)
# verify the SearXNG JSON API the WebClient depends on:
curl "http://localhost:8085/search?q=arcadedb&format=json"
```

Local-model runs also need a reachable Ollama (`OLLAMA_MODEL`); set `AGENT_MODEL` (e.g.
`openai:gpt-5.2`) to use a hosted provider instead. Secrets load from `.env` via `python-dotenv`.

The **backend image** bundles the Docker CLI and mounts the host `docker.sock` so `run_python`'s
sandbox containers launch on the host daemon (Docker-out-of-Docker); a same-path
`SANDBOX_SHARED_DIR`/`TMPDIR` bind lets `/out` artifacts round-trip. The `frontend` prod image is
nginx serving the built SPA and proxying `/api` → `backend`. See `DOCKER.md`.

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
- `run_python` shares that tolerance contract, and the sandbox hardening in
  `PythonSandbox._docker_args` (no network, cap-drop, read-only FS, non-root, pids/memory/time
  caps, code via **stdin**) is the safety boundary for model-authored code — don't relax flags or
  move the code into a shell string when editing it.
- Documents are user-editable: agent code must `read_document` before revising, and any new
  "agent writes / user edits" surface should reuse `repo.update_document` so both paths stay
  consistent.
- Delegated runs (`send_message`/`send_messages`/`deep_research`) share the tolerance contract too:
  `run_subagent`/`dispatch_message` must never raise — failures (incl. out-of-chart recipients)
  become `error` on the outcome/report. Specialists get tool capabilities ONLY, never
  `persistence_hooks` (the parent run already persists the turn; hooks on a delegate would
  double-write messages), and always a `UsageLimits` request cap. The agency communication chart
  lives on `AgentSpec.recipients`; multi-hop delegation is bounded by `SWARM_MAX_DEPTH` and enforced
  by `dispatch_message` against `deps.agency_recipients` — keep that enforcement when editing.
  A conversation's `mode` is stamped at creation but user-changeable later via
  `PATCH /api/conversations/{id}` → `repo.set_conversation_mode`; the same endpoint also sets a
  conversation's custom `system_prompt` (→ `repo.set_conversation_system_prompt`, appended to the
  base prompt by `main.compose_instructions`). Agent-profile *composition* still
  lives in `main.build_agent` (which `stream_run` rebuilds each turn from the stored mode + prompt).
  The `research` overlay also guarantees its report: `build_research` adds an `after_run` safeguard
  that persists the answer as a `Document` if the model skipped `create_document`, and the swarm's
  `deep_research` tool persists its digest as a fallback when the delegate created none.
