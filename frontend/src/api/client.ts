import type {
  AgentSpec,
  AppConfig,
  CatalogSkill,
  ContextUsage,
  Conversation,
  DocumentFull,
  DocumentMeta,
  Fact,
  MemoryGraph,
  Mode,
  Project,
  ProjectKb,
  SkillContent,
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

  listConversations: (userId: string, includeArchived = false) =>
    fetch(
      `/api/conversations?user_id=${encodeURIComponent(userId)}&include_archived=${includeArchived}`
    ).then(json<Conversation[]>),

  createConversation: (
    userId: string,
    title?: string,
    mode: Mode = "regular",
    projectId?: string | null
  ) =>
    fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, title, mode, project_id: projectId ?? null }),
    }).then(json<Conversation>),

  // Partial update: only the keys present in `patch` are applied server-side (mode, the custom
  // system prompt, swarm bounds, enabled skills, project membership, pin/archive flags).
  // system_prompt: "" clears the prompt; enabled_skills: [] clears the selection;
  // project_id: null moves the chat to Ungrouped.
  updateConversation: (
    conversationId: string,
    userId: string,
    patch: {
      mode?: Mode;
      system_prompt?: string;
      swarm_max_parallel?: number;
      swarm_max_depth?: number;
      enabled_skills?: string[];
      project_id?: string | null;
      pinned?: boolean;
      archived?: boolean;
    }
  ) =>
    fetch(`/api/conversations/${conversationId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...patch }),
    }).then(
      json<{ conversation_id: string; mode?: Mode; system_prompt?: string }>
    ),

  deleteConversation: (conversationId: string, userId: string) =>
    fetch(
      `/api/conversations/${conversationId}?user_id=${encodeURIComponent(userId)}`,
      { method: "DELETE" }
    ).then(json<{ deleted: string }>),

  // --- Projects ---------------------------------------------------------------------------
  listProjects: (userId: string) =>
    fetch(`/api/projects?user_id=${encodeURIComponent(userId)}`).then(
      json<Project[]>
    ),

  createProject: (userId: string, title?: string, systemPrompt = "") =>
    fetch("/api/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, title, system_prompt: systemPrompt }),
    }).then(json<Project>),

  updateProject: (
    projectId: string,
    userId: string,
    patch: { title?: string; system_prompt?: string }
  ) =>
    fetch(`/api/projects/${projectId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...patch }),
    }).then(json<Project>),

  // Cascade-delete: removes the project, its conversations, and its non-global documents.
  deleteProject: (projectId: string, userId: string) =>
    fetch(`/api/projects/${projectId}?user_id=${encodeURIComponent(userId)}`, {
      method: "DELETE",
    }).then(json<{ deleted: string; conversations: number; documents: number }>),

  // A project's reference documents (plus the user's global ones), metadata only.
  listProjectDocuments: (projectId: string, userId: string) =>
    fetch(
      `/api/projects/${projectId}/documents?user_id=${encodeURIComponent(userId)}`
    ).then(json<DocumentMeta[]>),

  // Upload one reference file (base64 bytes) into a project's document set.
  uploadProjectDocument: (
    projectId: string,
    userId: string,
    file: { filename: string; mime_type: string; data: string }
  ) =>
    fetch(`/api/projects/${projectId}/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...file }),
    }).then(json<DocumentMeta>),

  // A project's compiled knowledge base (status + the generated wiki pages).
  getProjectKb: (projectId: string, userId: string) =>
    fetch(
      `/api/projects/${projectId}/kb?user_id=${encodeURIComponent(userId)}`
    ).then(json<ProjectKb>),

  // Kick a full knowledge-base rebuild for a project (runs in the background).
  rebuildProjectKb: (projectId: string, userId: string) =>
    fetch(`/api/projects/${projectId}/kb/rebuild?user_id=${encodeURIComponent(userId)}`, {
      method: "POST",
    }).then(json<{ status: string }>),

  // Mark a document global (survives project cascade-delete, queryable everywhere) or not.
  setDocumentGlobal: (documentId: string, userId: string, isGlobal: boolean) =>
    fetch(`/api/documents/${documentId}/global`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, is_global: isGlobal }),
    }).then(json<{ document_id: string; is_global: boolean }>),

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

  // Remove a skill from the user's library (synced uninstall or authored delete).
  deleteSkill: (userId: string, name: string) =>
    fetch(
      `/api/skills/${encodeURIComponent(name)}?user_id=${encodeURIComponent(userId)}`,
      { method: "DELETE" }
    ).then(json<{ deleted: string }>),

  // Create or edit (by name) a user-authored skill. source = "user".
  createSkill: (
    userId: string,
    draft: { name: string; description: string; body: string }
  ) =>
    fetch("/api/skills", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...draft }),
    }).then(json<{ name: string; description: string; source: string }>),

  // A skill's full content (body + files) for the editor.
  getSkillContent: (userId: string, name: string) =>
    fetch(
      `/api/skills/${encodeURIComponent(name)}/content?user_id=${encodeURIComponent(userId)}`
    ).then(json<SkillContent>),

  // --- Swarm roster (AgentSpecs) ----------------------------------------------------------
  listAgents: (userId: string) =>
    fetch(`/api/agents?user_id=${encodeURIComponent(userId)}`).then(json<AgentSpec[]>),

  createAgent: (userId: string, agent: Omit<AgentSpec, "agent_id">) =>
    fetch("/api/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...agent }),
    }).then(json<AgentSpec>),

  updateAgent: (
    agentId: string,
    userId: string,
    patch: Partial<Omit<AgentSpec, "agent_id" | "name">>
  ) =>
    fetch(`/api/agents/${agentId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, ...patch }),
    }).then(json<AgentSpec>),

  deleteAgent: (agentId: string, userId: string) =>
    fetch(
      `/api/agents/${agentId}?user_id=${encodeURIComponent(userId)}`,
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
