import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { api } from "@/api/client";
import type { AppConfig, Conversation } from "@/types";

// Single source of the current user. Hardcoded for now; swapping in real auth later
// is a one-line change here, and every call already threads the id through.
const USER_ID = "default";

// Where the chosen model / thinking effort are persisted. Both selections live in the browser and
// are sent per-request (see useChat); the server keeps no such state. Empty string means "use the
// server default".
const MODEL_KEY = "agent-graph:model";
const EFFORT_KEY = "agent-graph:effort";

interface AppState {
  userId: string;
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  /** Runtime config (model/effort options, URLs). Fetched once; null until loaded. */
  config: AppConfig | null;
  /** Selected model label, or "" to use the backend default. Sent per chat request. */
  model: string;
  setModel: (model: string) => void;
  /** Selected thinking-effort level, or "" to use the backend default. Sent per chat request. */
  effort: string;
  setEffort: (effort: string) => void;
  selectConversation: (id: string) => void;
  newConversation: () => Promise<void>;
  refreshConversations: () => Promise<Conversation[]>;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [config, setConfig] = useState<AppConfig | null>(null);

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
    const rows = await api.listConversations(USER_ID);
    setConversations(rows);
    return rows;
  }, []);

  const newConversation = useCallback(async () => {
    const convo = await api.createConversation(USER_ID);
    setConversations((prev) => [convo, ...prev]);
    setActiveId(convo.conversation_id);
  }, []);

  const selectConversation = useCallback((id: string) => setActiveId(id), []);

  // Initial load: fetch conversations and select the most recent (or create one).
  useEffect(() => {
    (async () => {
      try {
        const rows = await refreshConversations();
        if (rows.length > 0) {
          setActiveId(rows[0].conversation_id);
        } else {
          await newConversation();
        }
      } catch (err) {
        console.error("failed to load conversations", err);
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <AppContext.Provider
      value={{
        userId: USER_ID,
        conversations,
        activeId,
        loading,
        config,
        model,
        setModel,
        effort,
        setEffort,
        selectConversation,
        newConversation,
        refreshConversations,
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
