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

/** One node of the agent's chronological execution chain. Thinking runs and tool
 *  calls are kept in arrival order so the UI can render `thinking → tool → thinking`
 *  exactly as it streamed, rather than collapsing reasoning into a single block. */
export type Step =
  | { id: string; kind: "thinking"; text: string }
  | { id: string; kind: "tool"; tool: ToolEvent }
  | {
      id: string;
      kind: "document";
      documentId: string;
      title: string;
      mimeType: string;
      action: "created" | "updated";
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
  | {
      type: "document";
      action: "created" | "updated";
      document_id: string;
      title: string;
      mime_type: string;
    }
  | { type: "final"; text: string }
  | { type: "error"; message: string };

export interface AppConfig {
  model: string;
  /** Selectable model labels for the dropdown (backend's AGENT_MODELS or a default list). */
  models?: string[];
  model_source: string;
  /** Default thinking-effort level, and the selectable set for the dropdown. */
  effort?: string;
  efforts?: string[];
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
