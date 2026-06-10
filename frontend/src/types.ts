/** Conversation mode. Only "regular" is implemented today; the others are
 *  reserved so the sidebar's mode-icon logic and future canvases have a stable field. */
export type Mode = "regular" | "research" | "swarm" | "council";

export interface Conversation {
  conversation_id: string;
  title: string | null;
  started_at?: string;
  mode: Mode;
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

/** A rendered chat turn. The assistant turn additionally carries live thinking
 *  and the tool-call timeline as it streams. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  tools?: ToolEvent[];
  streaming?: boolean;
  error?: string;
}

/** The stable streaming vocabulary emitted by the backend `stream_run` generator
 *  (see backend/main.py). Mirrored here so the UI never couples to Pydantic AI. */
export type StreamEvent =
  | { type: "thinking"; delta: string }
  | { type: "text"; delta: string }
  | { type: "tool_call"; tool_name: string; tool_call_id: string; args: unknown }
  | { type: "tool_result"; tool_name: string | null; tool_call_id: string | null; content: unknown }
  | { type: "final"; text: string }
  | { type: "error"; message: string };

export interface AppConfig {
  model: string;
  model_source: string;
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
