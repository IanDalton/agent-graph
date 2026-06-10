import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { api } from "@/api/client";
import { streamChat } from "@/api/stream";
import type { ChatMessage, StreamEvent } from "@/types";

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

type Action =
  | { kind: "load"; messages: ChatMessage[] }
  | { kind: "user"; id: string; content: string; assistantId: string }
  | { kind: "event"; event: StreamEvent };

/** Apply an update to the in-flight (last) assistant message. */
function patchLast(
  messages: ChatMessage[],
  patch: (m: ChatMessage) => ChatMessage
): ChatMessage[] {
  if (messages.length === 0) return messages;
  const idx = messages.length - 1;
  const copy = messages.slice();
  copy[idx] = patch(copy[idx]);
  return copy;
}

function reducer(state: ChatMessage[], action: Action): ChatMessage[] {
  switch (action.kind) {
    case "load":
      return action.messages;
    case "user":
      return [
        ...state,
        { id: action.id, role: "user", content: action.content },
        { id: action.assistantId, role: "assistant", content: "", streaming: true, tools: [] },
      ];
    case "event": {
      const ev = action.event;
      switch (ev.type) {
        case "thinking":
          return patchLast(state, (m) => ({
            ...m,
            thinking: (m.thinking ?? "") + ev.delta,
          }));
        case "text":
          return patchLast(state, (m) => ({ ...m, content: m.content + ev.delta }));
        case "tool_call":
          return patchLast(state, (m) => ({
            ...m,
            tools: [
              ...(m.tools ?? []),
              {
                toolName: ev.tool_name,
                toolCallId: ev.tool_call_id,
                args: ev.args,
                done: false,
              },
            ],
          }));
        case "tool_result":
          return patchLast(state, (m) => ({
            ...m,
            tools: (m.tools ?? []).map((t) =>
              t.toolCallId === ev.tool_call_id && !t.done
                ? { ...t, result: ev.content, done: true }
                : t
            ),
          }));
        case "final":
          return patchLast(state, (m) => ({
            ...m,
            content: ev.text || m.content,
            streaming: false,
          }));
        case "error":
          return patchLast(state, (m) => ({
            ...m,
            streaming: false,
            error: ev.message,
          }));
        default:
          return state;
      }
    }
    default:
      return state;
  }
}

export function useChat(
  conversationId: string | null,
  userId: string,
  onTurnComplete?: () => void
) {
  const [messages, dispatch] = useReducer(reducer, []);
  const [sending, setSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Load persisted history whenever the active conversation changes.
  useEffect(() => {
    abortRef.current?.abort();
    dispatch({ kind: "load", messages: [] });
    if (!conversationId) return;
    let cancelled = false;
    (async () => {
      try {
        const stored = await api.getMessages(conversationId, userId);
        if (cancelled) return;
        dispatch({
          kind: "load",
          messages: stored.map((m) => ({
            id: newId(),
            role: m.role,
            content: m.content,
          })),
        });
      } catch (err) {
        console.error("failed to load messages", err);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [conversationId, userId]);

  const send = useCallback(
    async (prompt: string) => {
      if (!conversationId || !prompt.trim() || sending) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setSending(true);
      dispatch({
        kind: "user",
        id: newId(),
        content: prompt,
        assistantId: newId(),
      });
      try {
        await streamChat(
          { user_id: userId, conversation_id: conversationId, prompt },
          (event) => dispatch({ kind: "event", event }),
          controller.signal
        );
      } catch (err) {
        if (!controller.signal.aborted) {
          dispatch({
            kind: "event",
            event: { type: "error", message: String(err) },
          });
        }
      } finally {
        setSending(false);
        onTurnComplete?.();
      }
    },
    [conversationId, userId, sending, onTurnComplete]
  );

  return { messages, sending, send };
}
