import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { api } from "@/api/client";
import { streamChat } from "@/api/stream";
import { useApp } from "@/state/AppContext";
import type { ChatMessage, StreamEvent } from "@/types";

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

type Action =
  | { kind: "load"; messages: ChatMessage[] }
  | { kind: "user"; id: string; content: string; assistantId: string }
  | { kind: "regenerate"; assistantId: string }
  | { kind: "stopped" }
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
        { id: action.assistantId, role: "assistant", content: "", streaming: true, steps: [] },
      ];
    case "regenerate": {
      // Drop a trailing assistant turn (the one being regenerated) and add a fresh
      // streaming placeholder, leaving the preceding user message intact.
      const trimmed =
        state.length && state[state.length - 1].role === "assistant"
          ? state.slice(0, -1)
          : state.slice();
      return [
        ...trimmed,
        { id: action.assistantId, role: "assistant", content: "", streaming: true, steps: [] },
      ];
    }
    case "stopped":
      // Manual interrupt: clear the typing indicator on the in-flight turn (idempotent).
      return patchLast(state, (m) => (m.streaming ? { ...m, streaming: false } : m));
    case "event": {
      const ev = action.event;
      switch (ev.type) {
        case "thinking":
          // Coalesce into the trailing thinking step; if the last step is a tool
          // (or there are no steps), open a new thinking step — this is what renders
          // the chronological `thinking → tool → thinking` interleave.
          return patchLast(state, (m) => {
            const steps = m.steps ?? [];
            const last = steps[steps.length - 1];
            if (last && last.kind === "thinking") {
              const next = steps.slice();
              next[next.length - 1] = { ...last, text: last.text + ev.delta };
              return { ...m, steps: next };
            }
            return {
              ...m,
              steps: [...steps, { id: newId(), kind: "thinking", text: ev.delta }],
            };
          });
        case "text":
          // The final answer accumulates separately and renders below the chain.
          return patchLast(state, (m) => ({ ...m, content: m.content + ev.delta }));
        case "tool_call":
          return patchLast(state, (m) => ({
            ...m,
            steps: [
              ...(m.steps ?? []),
              {
                id: ev.tool_call_id ?? newId(),
                kind: "tool",
                tool: {
                  toolName: ev.tool_name,
                  toolCallId: ev.tool_call_id,
                  args: ev.args,
                  done: false,
                },
              },
            ],
          }));
        case "tool_result":
          return patchLast(state, (m) => ({
            ...m,
            steps: (m.steps ?? []).map((s) =>
              s.kind === "tool" && s.tool.toolCallId === ev.tool_call_id && !s.tool.done
                ? { ...s, tool: { ...s.tool, result: ev.content, done: true } }
                : s
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
  const { model, effort } = useApp();
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

  // Shared streaming core for both fresh sends and regenerations. Dispatches each
  // event into the reducer; a user-initiated abort is swallowed (no error frame).
  const runStream = useCallback(
    async (prompt: string, signal: AbortSignal) => {
      try {
        await streamChat(
          {
            user_id: userId,
            conversation_id: conversationId!,
            prompt,
            // Omit when unset so the backend falls back to its configured defaults.
            model: model || undefined,
            effort: effort || undefined,
          },
          (event) => dispatch({ kind: "event", event }),
          signal
        );
      } catch (err) {
        if (!signal.aborted) {
          dispatch({ kind: "event", event: { type: "error", message: String(err) } });
        }
      } finally {
        setSending(false);
        onTurnComplete?.();
      }
    },
    [conversationId, userId, model, effort, onTurnComplete]
  );

  const send = useCallback(
    async (prompt: string) => {
      if (!conversationId || !prompt.trim() || sending) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setSending(true);
      dispatch({ kind: "user", id: newId(), content: prompt, assistantId: newId() });
      await runStream(prompt, controller.signal);
    },
    [conversationId, sending, runStream]
  );

  // Abort the in-flight stream and clear the turn's typing indicator.
  const stop = useCallback(() => {
    abortRef.current?.abort();
    dispatch({ kind: "stopped" });
    setSending(false);
  }, []);

  // Re-run the most recent user prompt, replacing the trailing assistant turn
  // (no duplicate user bubble).
  const regenerate = useCallback(async () => {
    if (!conversationId || sending) return;
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setSending(true);
    dispatch({ kind: "regenerate", assistantId: newId() });
    await runStream(lastUser.content, controller.signal);
  }, [conversationId, sending, messages, runStream]);

  return { messages, sending, send, stop, regenerate };
}
