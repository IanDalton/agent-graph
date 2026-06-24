import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { api } from "@/api/client";
import { streamDownload } from "@/api/stream";
import { conversationLabel } from "@/panes/sidebar";
import type {
  AgentSpec,
  AppConfig,
  CatalogSkill,
  Conversation,
  DownloadEvent,
  HardwareProfile,
  LlamacppStatus,
  LocalModel,
  Mode,
  Project,
  SkillContent,
  SkillInfo,
  SwarmFlowState,
} from "@/types";

// Single source of the current user. Hardcoded for now; swapping in real auth later
// is a one-line change here, and every call already threads the id through.
const USER_ID = "default";

// Where the chosen model / thinking effort are persisted. Both selections live in the browser and
// are sent per-request (see useChat); the server keeps no such state. Empty string means "use the
// server default".
const MODEL_KEY = "agent-graph:model";
const EFFORT_KEY = "agent-graph:effort";

// The URL reflects the open chat (`/c/<conversation_id>`) so a link reopens that exact chat.
// We drive it with the History API — no router dependency, since "the open chat" is a single
// piece of state. nginx prod (try_files -> index.html) and Vite dev both serve index.html for
// these paths.
const CHAT_PATH = /^\/c\/([^/?#]+)/;

function chatIdFromUrl(): string | null {
  const m = window.location.pathname.match(CHAT_PATH);
  return m ? decodeURIComponent(m[1]) : null;
}

function chatPath(id: string): string {
  return `/c/${encodeURIComponent(id)}`;
}

/** A document the UI should bring into focus (side panel → Documents tab → open it).
 *  `ts` makes re-featuring the same document retrigger the effect. */
export interface FeaturedDoc {
  id: string;
  ts: number;
}

interface AppState {
  userId: string;
  conversations: Conversation[];
  activeId: string | null;
  /** The project a new chat will be created into (set by selecting a project or a chat in one).
   *  null ⇒ new chats are ungrouped. */
  activeProjectId: string | null;
  /** Select a project (so "New Chat" lands in it); pass null to clear the selection. */
  selectProject: (id: string | null) => void;
  loading: boolean;
  /** True when the "new chat" mode picker should be shown in the canvas. */
  pendingNewChat: boolean;
  /** Runtime config (model/effort options, URLs). Fetched once; null until loaded. */
  config: AppConfig | null;
  /** The document to spotlight in the side panel (set when the agent creates one, or when the
   *  user clicks a document card in the chat). */
  featuredDoc: FeaturedDoc | null;
  featureDocument: (id: string) => void;
  /** Live orchestrator→agents activity for the swarm flow diagram (ContextPane). Updated by
   *  useChat as frames stream; null when no swarm turn is in flight. */
  swarmFlow: SwarmFlowState | null;
  setSwarmFlow: (flow: SwarmFlowState | null) => void;
  /** Selected model label, or "" to use the backend default. Sent per chat request. */
  model: string;
  setModel: (model: string) => void;
  /** Selected thinking-effort level, or "" to use the backend default. Sent per chat request. */
  effort: string;
  setEffort: (effort: string) => void;
  selectConversation: (id: string) => void;
  /** Show the mode picker in the canvas (clears active conversation). */
  openNewChatPicker: () => void;
  /** Create and select a conversation; `mode` picks its agent profile. `projectId` its project —
   *  omit it to use the currently-selected project (`activeProjectId`). */
  newConversation: (mode?: Mode, projectId?: string | null) => Promise<void>;
  /** Switch a conversation's agent mode mid-thread; persists and takes effect next turn. */
  setConversationMode: (id: string, mode: Mode) => Promise<void>;
  /** Set a conversation's custom system prompt; persists and takes effect next turn. */
  setConversationSystemPrompt: (id: string, prompt: string) => Promise<void>;
  setConversationSwarmSettings: (
    id: string,
    patch: { swarm_max_parallel?: number; swarm_max_depth?: number }
  ) => Promise<void>;
  /** Set the marketplace skills enabled for a conversation; persists, takes effect next turn. */
  setConversationSkills: (id: string, names: string[]) => Promise<void>;
  /** Move a conversation into a project (id) or out of one (null = ungrouped). */
  setConversationProject: (id: string, projectId: string | null) => Promise<void>;
  /** Pin/unpin a conversation (pinned float to the top of their group). */
  setConversationPinned: (id: string, pinned: boolean) => Promise<void>;
  /** Archive/unarchive a conversation (archived are hidden unless "Show archived" is on). */
  setConversationArchived: (id: string, archived: boolean) => Promise<void>;
  /** Permanently delete a conversation (and its messages/documents). */
  deleteConversation: (id: string) => Promise<void>;
  refreshConversations: () => Promise<Conversation[]>;
  /** Whether archived conversations are shown in the sidebar. */
  showArchived: boolean;
  setShowArchived: (show: boolean) => void;
  // --- Projects --------------------------------------------------------------------------
  projects: Project[];
  refreshProjects: () => Promise<Project[]>;
  /** Create a project and return it (does not select anything). */
  newProject: (title?: string) => Promise<Project | null>;
  /** Set a project's system prompt; persists and takes effect next turn. */
  setProjectSystemPrompt: (id: string, prompt: string) => Promise<void>;
  /** Rename a project. */
  setProjectTitle: (id: string, title: string) => Promise<void>;
  /** Cascade-delete a project (its conversations + non-global documents). */
  deleteProject: (id: string) => Promise<void>;
  /** The user's account-wide skill library (synced + authored). Active in every regular/research chat. */
  skills: SkillInfo[];
  /** True while a marketplace sync is in flight (drives the Sync button's spinner). */
  syncingSkills: boolean;
  /** Re-fetch the library. */
  refreshSkills: () => Promise<void>;
  /** Sync ALL marketplace skills into the library, then refresh. */
  syncSkills: () => Promise<void>;
  /** Whether the Skill Marketplace dialog is open. */
  skillMarketplaceOpen: boolean;
  openSkillMarketplace: () => void;
  closeSkillMarketplace: () => void;
  /** The live marketplace catalog (name + description + installed) shown in the dialog. */
  catalog: CatalogSkill[];
  /** True while the catalog is being fetched from GitHub. */
  catalogLoading: boolean;
  /** Re-fetch the live marketplace catalog. */
  refreshCatalog: () => Promise<void>;
  /** Install a marketplace skill into the library (active everywhere). */
  installSkill: (name: string) => Promise<void>;
  /** Remove a skill from the library (uninstall / delete authored). */
  removeSkill: (name: string) => Promise<void>;
  /** Create or edit (by name) a user-authored skill, then refresh. */
  saveSkill: (draft: { name: string; description: string; body: string }) => Promise<void>;
  /** Fetch a skill's full content (body + files) for the editor. */
  getSkillContent: (name: string) => Promise<SkillContent>;
  /** The user's swarm roster (AgentSpecs). Loaded for swarm conversations. */
  agents: AgentSpec[];
  refreshAgents: () => Promise<void>;
  createAgent: (agent: Omit<AgentSpec, "agent_id">) => Promise<void>;
  updateAgent: (agentId: string, patch: Partial<Omit<AgentSpec, "agent_id" | "name">>) => Promise<void>;
  deleteAgent: (agentId: string) => Promise<void>;
  // --- Local model manager (llama.cpp) ---------------------------------------------------
  /** Whether the Model Manager dialog is open. */
  modelsPageOpen: boolean;
  openModelsPage: () => void;
  closeModelsPage: () => void;
  /** The downloaded local GGUF models (Library tab + composer picker). */
  localModels: LocalModel[];
  refreshModels: () => Promise<void>;
  /** The editable hardware profile (the recommendation source of truth); null until loaded. */
  hardware: HardwareProfile | null;
  refreshHardware: () => Promise<void>;
  /** Persist the user-entered hardware profile (source = "manual"). */
  saveHardware: (profile: {
    gpus: { name: string; vram_mb: number }[];
    system_ram_mb: number;
    cpu_threads: number;
  }) => Promise<void>;
  /** Best-effort NVIDIA/system auto-detect (a suggestion; not saved until saveHardware). */
  detectHardware: () => Promise<HardwareProfile | null>;
  /** llama-server connectivity + served model; null until first checked. */
  llamacppStatus: LlamacppStatus | null;
  refreshLlamacppStatus: () => Promise<void>;
  /** Download a GGUF (SSE), forwarding progress frames; refreshes the library on completion. */
  downloadModel: (
    body: { repo_id: string; file_path: string; revision?: string; quant?: string },
    onEvent: (event: DownloadEvent) => void,
    signal?: AbortSignal
  ) => Promise<void>;
  /** Delete a downloaded GGUF, then refresh the library. */
  deleteModel: (filename: string) => Promise<void>;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [pendingNewChat, setPendingNewChat] = useState(false);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [featuredDoc, setFeaturedDoc] = useState<FeaturedDoc | null>(null);
  const [swarmFlow, setSwarmFlow] = useState<SwarmFlowState | null>(null);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [syncingSkills, setSyncingSkills] = useState(false);
  const [skillMarketplaceOpen, setSkillMarketplaceOpen] = useState(false);
  const [catalog, setCatalog] = useState<CatalogSkill[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [agents, setAgents] = useState<AgentSpec[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [showArchived, setShowArchived] = useState(false);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [modelsPageOpen, setModelsPageOpen] = useState(false);
  const [localModels, setLocalModels] = useState<LocalModel[]>([]);
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [llamacppStatus, setLlamacppStatus] = useState<LlamacppStatus | null>(null);

  const featureDocument = useCallback((id: string) => {
    setFeaturedDoc({ id, ts: Date.now() });
  }, []);

  const readStored = (key: string) => {
    try {
      return localStorage.getItem(key) ?? "";
    } catch {
      return "";
    }
  };
  const writeStored = (key: string, value: string) => {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch {
      // Private-mode / disabled storage: keep the in-memory selection, just don't persist.
    }
  };

  const [model, setModelState] = useState<string>(() => readStored(MODEL_KEY));
  const setModel = useCallback((next: string) => {
    setModelState(next);
    writeStored(MODEL_KEY, next);
  }, []);

  const [effort, setEffortState] = useState<string>(() => readStored(EFFORT_KEY));
  const setEffort = useCallback((next: string) => {
    setEffortState(next);
    writeStored(EFFORT_KEY, next);
  }, []);

  // Fetch runtime config once so the composer and the config card share one source (and one
  // network call). Best-effort: a failure just leaves config null and the selectors fall back.
  useEffect(() => {
    api.getConfig().then(setConfig).catch((e) => console.error("config", e));
  }, []);

  const refreshConversations = useCallback(async () => {
    const rows = await api.listConversations(USER_ID, showArchived);
    setConversations(rows);
    return rows;
  }, [showArchived]);

  const refreshProjects = useCallback(async () => {
    const rows = await api.listProjects(USER_ID);
    setProjects(rows);
    return rows;
  }, []);

  const openNewChatPicker = useCallback(() => {
    setActiveId(null);
    setFeaturedDoc(null);
    setSwarmFlow(null);
    setPendingNewChat(true);
  }, []);

  const newConversation = useCallback(
    async (mode: Mode = "regular", projectId?: string | null) => {
      // Default a new chat into the currently-selected project (so picking a project then
      // "New Chat" lands there); an explicit argument (incl. null for ungrouped) wins.
      const target = projectId === undefined ? activeProjectId : projectId;
      const convo = await api.createConversation(USER_ID, undefined, mode, target);
      // The server stamps project_id but may echo it back null in the create response shape;
      // ensure the local row carries it so the sidebar groups the new chat correctly.
      setConversations((prev) => [{ ...convo, project_id: target }, ...prev]);
      setActiveId(convo.conversation_id);
      setActiveProjectId(target);
      setFeaturedDoc(null);
      setSwarmFlow(null);
      setPendingNewChat(false);
    },
    [activeProjectId]
  );

  const selectConversation = useCallback(
    (id: string) => {
      setActiveId(id);
      // Follow the conversation's project so a subsequent "New Chat" continues in the same place.
      const conv = conversations.find((c) => c.conversation_id === id);
      setActiveProjectId(conv?.project_id ?? null);
      setFeaturedDoc(null);
      setSwarmFlow(null);
      setPendingNewChat(false);
    },
    [conversations]
  );

  const selectProject = useCallback((id: string | null) => {
    setActiveProjectId(id);
  }, []);

  const setConversationMode = useCallback(async (id: string, mode: Mode) => {
    // Optimistically flip the local row so the sidebar icon and the Canvas renderer switch
    // immediately; the next turn re-reads the persisted mode from the server.
    setConversations((prev) =>
      prev.map((c) => (c.conversation_id === id ? { ...c, mode } : c))
    );
    try {
      await api.updateConversation(id, USER_ID, { mode });
    } catch (err) {
      console.error("failed to update conversation mode", err);
      // Re-sync from the server so a failed switch doesn't leave a stale local mode.
      refreshConversations().catch(() => {});
    }
  }, [refreshConversations]);

  const setConversationSystemPrompt = useCallback(async (id: string, prompt: string) => {
    // Optimistically store the prompt on the local row; it takes effect on the next turn.
    setConversations((prev) =>
      prev.map((c) => (c.conversation_id === id ? { ...c, system_prompt: prompt } : c))
    );
    try {
      await api.updateConversation(id, USER_ID, { system_prompt: prompt });
    } catch (err) {
      console.error("failed to update conversation system prompt", err);
      refreshConversations().catch(() => {});
    }
  }, [refreshConversations]);

  const setConversationSwarmSettings = useCallback(
    async (id: string, patch: { swarm_max_parallel?: number; swarm_max_depth?: number }) => {
      // Optimistically merge the bounds onto the local row; they take effect on the next turn.
      setConversations((prev) =>
        prev.map((c) => (c.conversation_id === id ? { ...c, ...patch } : c))
      );
      try {
        await api.updateConversation(id, USER_ID, patch);
      } catch (err) {
        console.error("failed to update conversation swarm settings", err);
        refreshConversations().catch(() => {});
      }
    },
    [refreshConversations]
  );

  const setConversationSkills = useCallback(
    async (id: string, names: string[]) => {
      // Optimistically store the selection on the local row; it takes effect on the next turn.
      setConversations((prev) =>
        prev.map((c) => (c.conversation_id === id ? { ...c, enabled_skills: names } : c))
      );
      try {
        await api.updateConversation(id, USER_ID, { enabled_skills: names });
      } catch (err) {
        console.error("failed to update conversation skills", err);
        refreshConversations().catch(() => {});
      }
    },
    [refreshConversations]
  );

  const setConversationProject = useCallback(
    async (id: string, projectId: string | null) => {
      setConversations((prev) =>
        prev.map((c) => (c.conversation_id === id ? { ...c, project_id: projectId } : c))
      );
      // If the moved chat is the active one, follow it so "New Chat" lands in its new project.
      setActiveProjectId((prev) => (id === activeId ? projectId : prev));
      try {
        await api.updateConversation(id, USER_ID, { project_id: projectId });
      } catch (err) {
        console.error("failed to move conversation", err);
        refreshConversations().catch(() => {});
      }
    },
    [activeId, refreshConversations]
  );

  const setConversationPinned = useCallback(
    async (id: string, pinned: boolean) => {
      setConversations((prev) => {
        const next = prev.map((c) =>
          c.conversation_id === id ? { ...c, pinned } : c
        );
        // Keep pinned-first ordering locally so the row jumps to the top immediately.
        return [...next].sort((a, b) => Number(!!b.pinned) - Number(!!a.pinned));
      });
      try {
        await api.updateConversation(id, USER_ID, { pinned });
      } catch (err) {
        console.error("failed to pin conversation", err);
        refreshConversations().catch(() => {});
      }
    },
    [refreshConversations]
  );

  const setConversationArchived = useCallback(
    async (id: string, archived: boolean) => {
      // Drop it from the list immediately when archiving (unless archived are shown).
      setConversations((prev) =>
        archived && !showArchived
          ? prev.filter((c) => c.conversation_id !== id)
          : prev.map((c) => (c.conversation_id === id ? { ...c, archived } : c))
      );
      if (archived && activeId === id) setActiveId(null);
      try {
        await api.updateConversation(id, USER_ID, { archived });
      } catch (err) {
        console.error("failed to archive conversation", err);
        refreshConversations().catch(() => {});
      }
    },
    [activeId, showArchived, refreshConversations]
  );

  const deleteConversation = useCallback(
    async (id: string) => {
      setConversations((prev) => prev.filter((c) => c.conversation_id !== id));
      if (activeId === id) setActiveId(null);
      try {
        await api.deleteConversation(id, USER_ID);
      } catch (err) {
        console.error("failed to delete conversation", err);
        refreshConversations().catch(() => {});
      }
    },
    [activeId, refreshConversations]
  );

  const newProject = useCallback(async (title?: string) => {
    try {
      const project = await api.createProject(USER_ID, title);
      setProjects((prev) => [project, ...prev]);
      return project;
    } catch (err) {
      console.error("failed to create project", err);
      return null;
    }
  }, []);

  const setProjectSystemPrompt = useCallback(
    async (id: string, prompt: string) => {
      setProjects((prev) =>
        prev.map((p) => (p.project_id === id ? { ...p, system_prompt: prompt } : p))
      );
      try {
        await api.updateProject(id, USER_ID, { system_prompt: prompt });
      } catch (err) {
        console.error("failed to update project prompt", err);
        refreshProjects().catch(() => {});
      }
    },
    [refreshProjects]
  );

  const setProjectTitle = useCallback(
    async (id: string, title: string) => {
      setProjects((prev) =>
        prev.map((p) => (p.project_id === id ? { ...p, title } : p))
      );
      try {
        await api.updateProject(id, USER_ID, { title });
      } catch (err) {
        console.error("failed to rename project", err);
        refreshProjects().catch(() => {});
      }
    },
    [refreshProjects]
  );

  const deleteProject = useCallback(
    async (id: string) => {
      if (!id) return;
      // Optimistically drop the project and ONLY its member conversations. Normalize project_id to
      // null so an ungrouped chat (project_id null/undefined) is never matched and removed.
      setActiveProjectId((prev) => (prev === id ? null : prev));
      setProjects((prev) => prev.filter((p) => p.project_id !== id));
      setConversations((prev) => {
        const removed = prev
          .filter((c) => (c.project_id ?? null) === id)
          .map((c) => c.conversation_id);
        if (activeId && removed.includes(activeId)) setActiveId(null);
        return prev.filter((c) => (c.project_id ?? null) !== id);
      });
      try {
        await api.deleteProject(id, USER_ID);
      } catch (err) {
        console.error("failed to delete project", err);
      } finally {
        // Reconcile with the server's actual cascade result (source of truth) so the local
        // optimistic guess can never strand or drop the wrong conversations.
        refreshProjects().catch(() => {});
        refreshConversations().catch(() => {});
      }
    },
    [activeId, refreshProjects, refreshConversations]
  );

  const refreshSkills = useCallback(async () => {
    try {
      setSkills(await api.listSkills(USER_ID));
    } catch (err) {
      console.error("failed to load skills", err);
    }
  }, []);

  const syncSkills = useCallback(async () => {
    setSyncingSkills(true);
    try {
      await api.syncSkills(USER_ID);
      setSkills(await api.listSkills(USER_ID));
    } catch (err) {
      console.error("failed to sync skills", err);
    } finally {
      setSyncingSkills(false);
    }
  }, []);

  const refreshCatalog = useCallback(async () => {
    setCatalogLoading(true);
    try {
      setCatalog(await api.getSkillCatalog(USER_ID));
    } catch (err) {
      console.error("failed to load skill catalog", err);
    } finally {
      setCatalogLoading(false);
    }
  }, []);

  const openSkillMarketplace = useCallback(() => {
    setSkillMarketplaceOpen(true);
    refreshCatalog().catch(() => {});
  }, [refreshCatalog]);

  const closeSkillMarketplace = useCallback(() => setSkillMarketplaceOpen(false), []);

  // Install a marketplace skill into the account library (active in every regular/research chat).
  const installSkill = useCallback(
    async (name: string) => {
      try {
        await api.syncSkills(USER_ID, [name]);
      } catch (err) {
        console.error("failed to install skill", err);
      }
      refreshSkills().catch(() => {});
      refreshCatalog().catch(() => {});
    },
    [refreshSkills, refreshCatalog]
  );

  // Remove a skill from the library (uninstall a synced one or delete an authored one).
  const removeSkill = useCallback(
    async (name: string) => {
      try {
        await api.deleteSkill(USER_ID, name);
      } catch (err) {
        console.error("failed to remove skill", err);
      }
      refreshSkills().catch(() => {});
      refreshCatalog().catch(() => {});
    },
    [refreshSkills, refreshCatalog]
  );

  const saveSkill = useCallback(
    async (draft: { name: string; description: string; body: string }) => {
      await api.createSkill(USER_ID, draft);
      refreshSkills().catch(() => {});
      refreshCatalog().catch(() => {});
    },
    [refreshSkills, refreshCatalog]
  );

  const getSkillContent = useCallback((name: string) => api.getSkillContent(USER_ID, name), []);

  const refreshAgents = useCallback(async () => {
    try {
      setAgents(await api.listAgents(USER_ID));
    } catch (err) {
      console.error("failed to load agents", err);
    }
  }, []);

  const createAgent = useCallback(
    async (agent: Omit<AgentSpec, "agent_id">) => {
      try {
        await api.createAgent(USER_ID, agent);
        await refreshAgents();
      } catch (err) {
        console.error("failed to create agent", err);
        throw err;
      }
    },
    [refreshAgents]
  );

  const updateAgent = useCallback(
    async (agentId: string, patch: Partial<Omit<AgentSpec, "agent_id" | "name">>) => {
      setAgents((prev) =>
        prev.map((a) => (a.agent_id === agentId ? { ...a, ...patch } : a))
      );
      try {
        await api.updateAgent(agentId, USER_ID, patch);
      } catch (err) {
        console.error("failed to update agent", err);
        refreshAgents().catch(() => {});
      }
    },
    [refreshAgents]
  );

  const deleteAgent = useCallback(
    async (agentId: string) => {
      setAgents((prev) => prev.filter((a) => a.agent_id !== agentId));
      try {
        await api.deleteAgent(agentId, USER_ID);
      } catch (err) {
        console.error("failed to delete agent", err);
        refreshAgents().catch(() => {});
      }
    },
    [refreshAgents]
  );

  // --- Local model manager ----------------------------------------------------------------
  const refreshModels = useCallback(async () => {
    try {
      setLocalModels(await api.getModels());
    } catch (err) {
      console.error("failed to load local models", err);
    }
  }, []);

  const refreshHardware = useCallback(async () => {
    try {
      setHardware(await api.getHardware());
    } catch (err) {
      console.error("failed to load hardware profile", err);
    }
  }, []);

  const refreshLlamacppStatus = useCallback(async () => {
    try {
      setLlamacppStatus(await api.llamacppStatus());
    } catch (err) {
      console.error("failed to check llama-server status", err);
    }
  }, []);

  const openModelsPage = useCallback(() => {
    setModelsPageOpen(true);
    refreshModels().catch(() => {});
    refreshHardware().catch(() => {});
    refreshLlamacppStatus().catch(() => {});
  }, [refreshModels, refreshHardware, refreshLlamacppStatus]);

  const closeModelsPage = useCallback(() => setModelsPageOpen(false), []);

  const saveHardware = useCallback(
    async (profile: {
      gpus: { name: string; vram_mb: number }[];
      system_ram_mb: number;
      cpu_threads: number;
    }) => {
      try {
        setHardware(await api.updateHardware(profile));
      } catch (err) {
        console.error("failed to save hardware profile", err);
        refreshHardware().catch(() => {});
      }
    },
    [refreshHardware]
  );

  const detectHardware = useCallback(async () => {
    try {
      return await api.detectHardware();
    } catch (err) {
      console.error("failed to auto-detect hardware", err);
      return null;
    }
  }, []);

  const downloadModel = useCallback(
    async (
      body: { repo_id: string; file_path: string; revision?: string; quant?: string },
      onEvent: (event: DownloadEvent) => void,
      signal?: AbortSignal
    ) => {
      await streamDownload(
        body,
        (event) => {
          onEvent(event);
          if (event.type === "done") refreshModels().catch(() => {});
        },
        signal
      );
    },
    [refreshModels]
  );

  const deleteModel = useCallback(
    async (filename: string) => {
      setLocalModels((prev) => prev.filter((m) => m.filename !== filename));
      try {
        await api.deleteModel(filename);
      } catch (err) {
        console.error("failed to delete model", err);
      }
      refreshModels().catch(() => {});
    },
    [refreshModels]
  );

  // Load the skill library once so the Configuration card has it.
  useEffect(() => {
    refreshSkills();
  }, [refreshSkills]);

  // Load the projects list once for the sidebar groups (best-effort).
  useEffect(() => {
    refreshProjects().catch(() => {});
  }, [refreshProjects]);

  // Initial load: fetch conversations and select the most recent (or create one).
  useEffect(() => {
    (async () => {
      try {
        const rows = await refreshConversations();
        // Deep-link: prefer the chat named in the URL when it's one of ours; otherwise fall back
        // to the most recent (an unknown/foreign/archived id degrades gracefully, no 404).
        const urlId = chatIdFromUrl();
        const target =
          urlId && rows.some((r) => r.conversation_id === urlId)
            ? urlId
            : rows.length > 0
              ? rows[0].conversation_id
              : null;
        if (target) {
          setActiveId(target);
          // Follow the loaded conversation's project so its group is expanded and a New Chat
          // continues in the same place.
          setActiveProjectId(
            rows.find((r) => r.conversation_id === target)?.project_id ?? null
          );
        } else {
          setPendingNewChat(true);
        }
      } catch (err) {
        console.error("failed to load conversations", err);
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Mirror the active chat into the URL so it's shareable. First write replaces (no bogus history
  // entry on load); later writes push so Back/Forward walk chat history. The already-matches guard
  // makes this a no-op when the URL is already correct — including right after a popstate-driven
  // select, so navigating Back doesn't re-push.
  const didInitUrl = useRef(false);
  useEffect(() => {
    // Don't touch the URL until the initial load has resolved — otherwise this fires on mount
    // while activeId is still null and would replace the deep-link (/c/<id>) with "/" before the
    // async load gets to read it.
    if (loading) return;
    const want = activeId ? chatPath(activeId) : "/";
    if (window.location.pathname === want) {
      didInitUrl.current = true;
      return;
    }
    if (!didInitUrl.current) {
      window.history.replaceState({}, "", want);
      didInitUrl.current = true;
    } else {
      window.history.pushState({}, "", want);
    }
  }, [activeId, loading]);

  // Browser Back/Forward: follow the URL to the chat it names (when it's one we have loaded).
  useEffect(() => {
    const onPop = () => {
      const id = chatIdFromUrl();
      if (id && conversations.some((c) => c.conversation_id === id)) {
        selectConversation(id);
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [conversations, selectConversation]);

  // Tab title follows the open chat (and updates when its title later fills in after a turn).
  useEffect(() => {
    const conv = conversations.find((c) => c.conversation_id === activeId);
    document.title = conv ? `${conversationLabel(conv)} · agent-graph` : "agent-graph";
  }, [activeId, conversations]);

  // One-time cleanup: prune leftover **empty, untitled** projects (junk from before projects
  // required a name). Safe — they hold no conversations, so nothing is lost — and it runs once,
  // only after conversations have loaded (so "empty" is accurate). New untitled projects can no
  // longer be created, so this never fights the user.
  const prunedRef = useRef(false);
  useEffect(() => {
    if (loading || prunedRef.current || projects.length === 0) return;
    prunedRef.current = true;
    const usedProjectIds = new Set(
      conversations.map((c) => c.project_id ?? null).filter(Boolean)
    );
    const junk = projects.filter(
      (p) => !(p.title && p.title.trim()) && !usedProjectIds.has(p.project_id)
    );
    junk.forEach((p) => void deleteProject(p.project_id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, projects]);

  // Re-fetch when the "Show archived" toggle flips so archived rows appear/disappear.
  useEffect(() => {
    if (!loading) refreshConversations().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showArchived]);

  return (
    <AppContext.Provider
      value={{
        userId: USER_ID,
        conversations,
        activeId,
        activeProjectId,
        selectProject,
        loading,
        pendingNewChat,
        config,
        featuredDoc,
        featureDocument,
        swarmFlow,
        setSwarmFlow,
        model,
        setModel,
        effort,
        setEffort,
        selectConversation,
        openNewChatPicker,
        newConversation,
        setConversationMode,
        setConversationSystemPrompt,
        setConversationSwarmSettings,
        setConversationSkills,
        setConversationProject,
        setConversationPinned,
        setConversationArchived,
        deleteConversation,
        refreshConversations,
        showArchived,
        setShowArchived,
        projects,
        refreshProjects,
        newProject,
        setProjectSystemPrompt,
        setProjectTitle,
        deleteProject,
        skills,
        syncingSkills,
        refreshSkills,
        syncSkills,
        skillMarketplaceOpen,
        openSkillMarketplace,
        closeSkillMarketplace,
        catalog,
        catalogLoading,
        refreshCatalog,
        installSkill,
        removeSkill,
        saveSkill,
        getSkillContent,
        agents,
        refreshAgents,
        createAgent,
        updateAgent,
        deleteAgent,
        modelsPageOpen,
        openModelsPage,
        closeModelsPage,
        localModels,
        refreshModels,
        hardware,
        refreshHardware,
        saveHardware,
        detectHardware,
        llamacppStatus,
        refreshLlamacppStatus,
        downloadModel,
        deleteModel,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useApp(): AppState {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used within AppProvider");
  return ctx;
}
