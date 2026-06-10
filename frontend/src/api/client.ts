import type { AppConfig, Conversation, StoredMessage } from "@/types";

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

  createConversation: (userId: string, title?: string) =>
    fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId, title }),
    }).then(json<Conversation>),

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
};
