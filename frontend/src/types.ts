/** Conversation mode (the agent profile, fixed at creation). "regular", "research"
 *  (deep research) and "swarm" are implemented; "council" is reserved. */
export type Mode = "regular" | "research" | "swarm" | "council";

export interface Conversation {
  conversation_id: string;
  title: string | null;
  started_at?: string;
  mode: Mode;
  /** Custom system prompt appended to the base prompt for this conversation ("" when unset). */
  system_prompt?: string;
}

export interface StoredMessage {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
}

export interface ToolEvent {
  toolName: string | null;
  toolCallId: string | null;
  args?: unknown;
  result?: unknown;
  done: boolean;
}

/** Identity of the sub-agent that produced a step/frame in swarm mode. `undefined` on a step
 *  means the orchestrator (the main agent) produced it. `instanceId` is per-dispatch, so the
 *  same spec dispatched twice concurrently stays in two separate bubbles. */
export interface AgentRef {
  agentId: string;
  name: string;
  instanceId: string;
}

/** One node of the agent's chronological execution chain. Thinking runs and tool
 *  calls are kept in arrival order so the UI can render `thinking → tool → thinking`
 *  exactly as it streamed, rather than collapsing reasoning into a single block.
 *  In swarm mode each step carries the `agent` that produced it (undefined = orchestrator);
 *  a sub-agent's streamed report text becomes an `agent_text` step inside its bubble. */
export type Step =
  | { id: string; kind: "thinking"; text: string; agent?: AgentRef }
  | { id: string; kind: "agent_text"; text: string; agent?: AgentRef }
  | { id: string; kind: "tool"; tool: ToolEvent; agent?: AgentRef }
  | {
      id: string;
      kind: "document";
      documentId: string;
      title: string;
      mimeType: string;
      action: "created" | "updated";
      agent?: AgentRef;
    };

/** A rendered chat turn. The assistant turn additionally carries its ordered
 *  reasoning/tool `steps` chain plus the streamed `content` (the final answer,
 *  rendered as markdown below the chain). */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Ordered reasoning/tool chain (assistant turns only). */
  steps?: Step[];
  /** Swarm-mode per-instance run state, keyed by instanceId — drives each agent bubble's
   *  running spinner (set on agent_start, cleared on agent_end). */
  agents?: Record<string, { name: string; running: boolean }>;
  streaming?: boolean;
  error?: string;
}

/** The stable streaming vocabulary emitted by the backend `stream_run` generator
 *  (see backend/main.py). Mirrored here so the UI never couples to Pydantic AI.
 *  In swarm mode, sub-agent frames carry `agent_id`/`name`/`instance_id` (orchestrator frames
 *  omit them), plus `agent_start`/`agent_end` lifecycle frames bracket each delegate's run. */
export type AgentTag = { agent_id?: string; name?: string; instance_id?: string };

export type StreamEvent =
  | ({ type: "thinking"; delta: string } & AgentTag)
  | ({ type: "text"; delta: string } & AgentTag)
  | ({ type: "tool_call"; tool_name: string; tool_call_id: string; args: unknown } & AgentTag)
  | ({
      type: "tool_result";
      tool_name: string | null;
      tool_call_id: string | null;
      content: unknown;
    } & AgentTag)
  | {
      type: "document";
      action: "created" | "updated";
      document_id: string;
      title: string;
      mime_type: string;
    }
  | { type: "agent_start"; agent_id: string; name: string; instance_id: string }
  | { type: "agent_end"; agent_id: string; name: string; instance_id: string }
  | { type: "final"; text: string }
  | { type: "error"; message: string };

/** Live swarm activity for the orchestrator→agents flow diagram (ContextPane). Built from the
 *  stream's agent_start/agent_end/tool_* frames and shared via AppContext, so the right pane can
 *  render the fan-out while the chat shows the per-agent traces. Ephemeral (reset each turn). */
export interface SwarmAgentNode {
  agentId: string;
  name: string;
  instanceId: string;
  status: "running" | "done";
  /** Number of tool calls this agent has made so far (a rough "how busy" signal). */
  toolCount: number;
}

export interface SwarmFlowState {
  /** True while the orchestrator turn is still streaming. */
  active: boolean;
  /** Dispatched sub-agents, keyed by instanceId (so concurrent same-spec dispatches stay separate). */
  agents: Record<string, SwarmAgentNode>;
}

export interface AppConfig {
  model: string;
  /** Selectable model labels for the dropdown (backend's AGENT_MODELS or a default list). */
  models?: string[];
  model_source: string;
  /** Default thinking-effort level, and the selectable set for the dropdown. */
  effort?: string;
  efforts?: string[];
  /** Conversation modes (agent profiles) selectable at conversation creation. */
  modes?: string[];
  /** The fixed base system prompt; a conversation's custom prompt is appended to it. */
  base_system_prompt?: string;
  arcade_url: string;
  searxng_url: string;
  log_level: string;
  embeddings?: boolean;
  embed_model?: string | null;
}

/** One node of the agent-built knowledge graph (see backend repo.get_user_graph).
 *  `id` is a DOM-safe sanitized record id (e.g. "38_0"). */
export interface GraphNode {
  id: string;
  type: string;
  label: string;
  /** Memory kind of the node's type: "semantic" (durable state) or "episodic" (time-ordered event).
   *  null for legacy/internal types with no marker — rendered as semantic. */
  kind?: "semantic" | "episodic" | null;
  properties: Record<string, unknown>;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface MemoryGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** Document metadata as returned by the list endpoint (no body — fetch it separately). */
export interface DocumentMeta {
  document_id: string;
  conversation_id?: string;
  title: string;
  mime_type: string;
  /** "text" (content is literal text) or "base64" (binary artifact, e.g. a sandbox-made PDF). */
  encoding?: "text" | "base64";
  created_at?: string;
  updated_at?: string;
}

/** A full document, body included (GET /api/documents/{id}). */
export interface DocumentFull extends DocumentMeta {
  content: string;
}
