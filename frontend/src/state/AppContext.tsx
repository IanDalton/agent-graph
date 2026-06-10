import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import { api } from "@/api/client";
import type { Conversation } from "@/types";

// Single source of the current user. Hardcoded for now; swapping in real auth later
// is a one-line change here, and every call already threads the id through.
const USER_ID = "default";

interface AppState {
  userId: string;
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  selectConversation: (id: string) => void;
  newConversation: () => Promise<void>;
  refreshConversations: () => Promise<Conversation[]>;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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
