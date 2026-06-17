import type {
  AppConfig,
  CatalogSkill,
  ContextUsage,
  Conversation,
  DocumentFull,
  DocumentMeta,
  Fact,
  MemoryGraph,
  Mode,
  SkillInfo,
  SkillSyncResult,
  StoredMessage,
} from "@/types";

const json = async <T>(res: Response): Promise<T> => {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
};

export const api = {
  getConfig: () => fetch("/api/config").then(json<AppConfig>),

  listConversations: (userId: string) =>
    fetch(`/api/conversations?user_id=${encodeURIComponent(userId)}`).then(
      json<Conversation[]>
    ),

  createConversation: (userId: string, title?: string, mode: Mode = "regular") =>
    fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, title, mode }),
    }).then(json<Conversation>),

  // Partial update: only the keys present in `patch` are applied server-side (mode, the custom
  // system prompt, swarm bounds, and/or enabled skills). system_prompt: "" clears the prompt;
  // enabled_skills: [] clears the selection.
  updateConversation: (
    conversationId: string,
    userId: string,
    patch: {
      mode?: Mode;
      system_prompt?: string;
      swarm_max_parallel?: number;
      swarm_max_depth?: number;
      enabled_skills?: string[];
    }
  ) =>
    fetch(`/api/conversations/${conversationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...patch }),
    }).then(
      json<{ conversation_id: string; mode?: Mode; system_prompt?: string }>
    ),

  // The marketplace skills this user has synced (metadata only — backs the skill picker).
  listSkills: (userId: string) =>
    fetch(`/api/skills?user_id=${encodeURIComponent(userId)}`).then(
      json<SkillInfo[]>
    ),

  // The live Anthropic marketplace catalog (name + description + installed flag) for the dialog.
  getSkillCatalog: (userId: string) =>
    fetch(`/api/skills/catalog?user_id=${encodeURIComponent(userId)}`).then(
      json<CatalogSkill[]>
    ),

  // Sync skills from the Anthropic marketplace into the user's database. Omit `names` to sync all.
  syncSkills: (userId: string, names?: string[]) =>
    fetch("/api/skills/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, names }),
    }).then(json<SkillSyncResult>),

  // Remove a skill from the user's library (does not touch any conversation's selection).
  deleteSkill: (userId: string, name: string) =>
    fetch(
      `/api/skills/${encodeURIComponent(name)}?user_id=${encodeURIComponent(userId)}`,
      { method: "DELETE" }
    ).then(json<{ deleted: string }>),

  getMessages: (conversationId: string, userId: string) =>
    fetch(
      `/api/conversations/${conversationId}/messages?user_id=${encodeURIComponent(
        userId
      )}`
    ).then(json<StoredMessage[]>),

  getSummary: (conversationId: string, userId: string) =>
    fetch(
      `/api/conversations/${conversationId}/summary?user_id=${encodeURIComponent(
        userId
      )}`
    ).then(json<{ summary: string }>),

  refreshSummary: (conversationId: string, userId: string) =>
    fetch(
      `/api/conversations/${conversationId}/summary?user_id=${encodeURIComponent(
        userId
      )}`,
      { method: "POST" }
    ).then(json<{ summary: string }>),

  // Estimated context-window usage for a conversation, broken into system/tools/messages. `model`
  // and `mode` size the window and tool set for the currently-selected profile.
  getContextUsage: (
    conversationId: string,
    userId: string,
    model: string,
    mode: string
  ) =>
    fetch(
      `/api/conversations/${conversationId}/context?user_id=${encodeURIComponent(
        userId
      )}&model=${encodeURIComponent(model)}&mode=${encodeURIComponent(mode)}`
    ).then(json<ContextUsage>),

  getGraph: (userId: string, limit = 100) =>
    fetch(
      `/api/graph?user_id=${encodeURIComponent(userId)}&limit=${limit}`
    ).then(json<MemoryGraph>),

  // The durable, curator-maintained user profile (cross-conversation context). A fast DB read —
  // rewritten at write time by the background memory curator, so no LLM runs here.
  getUserProfile: (userId: string) =>
    fetch(`/api/user/profile?user_id=${encodeURIComponent(userId)}`).then(
      json<{ profile: string; profile_updated_at: string | null }>
    ),

  listFacts: (userId: string, limit = 200) =>
    fetch(
      `/api/facts?user_id=${encodeURIComponent(userId)}&limit=${limit}`
    ).then(json<Fact[]>),

  // Toggle whether a fact is included in the agent's context. All facts default to important.
  setFactImportance: (factId: string, userId: string, important: boolean) =>
    fetch(`/api/facts/${factId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, important }),
    }).then(json<{ fact_id: string; important: boolean }>),

  listDocuments: (conversationId: string, userId: string) =>
    fetch(
      `/api/conversations/${conversationId}/documents?user_id=${encodeURIComponent(
        userId
      )}`
    ).then(json<DocumentMeta[]>),

  getDocument: (documentId: string, userId: string) =>
    fetch(
      `/api/documents/${documentId}?user_id=${encodeURIComponent(userId)}`
    ).then(json<DocumentFull>),

  updateDocument: (
    documentId: string,
    userId: string,
    patch: { title?: string; content?: string }
  ) =>
    fetch(`/api/documents/${documentId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...patch }),
    }).then(json<DocumentFull>),

  deleteDocument: (documentId: string, userId: string) =>
    fetch(
      `/api/documents/${documentId}?user_id=${encodeURIComponent(userId)}`,
      { method: "DELETE" }
    ).then(json<{ deleted: string }>),
};
