import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { api } from "@/api/client";
import { streamChat } from "@/api/stream";
import { useApp } from "@/state/appcontext";
import type { AgentRef, Attachment, ChatMessage, StreamEvent, SwarmFlowState } from "@/types";

const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);

/** Resolve the producing agent of a tagged stream frame (undefined = the orchestrator). */
function tagOf(ev: {
  agent_id?: string;
  name?: string;
  instance_id?: string;
}): AgentRef | undefined {
  return ev.agent_id
    ? { agentId: ev.agent_id, name: ev.name ?? "", instanceId: ev.instance_id ?? "" }
    : undefined;
}

/** Two steps belong to the same producer when their agent instance matches (both undefined =
 *  orchestrator). Used to coalesce streamed thinking/text without merging concurrent agents. */
const sameInstance = (a?: AgentRef, b?: AgentRef) => a?.instanceId === b?.instanceId;

/** Fold a stream frame into the live swarm flow snapshot (orchestrator→agents diagram). Returns
 *  the same reference when nothing relevant changed, so non-swarm turns stay null. */
function reduceFlow(prev: SwarmFlowState | null, ev: StreamEvent): SwarmFlowState | null {
  switch (ev.type) {
    case "agent_start": {
      const agents = { ...(prev?.agents ?? {}) };
      agents[ev.instance_id] = {
        agentId: ev.agent_id,
        name: ev.name,
        instanceId: ev.instance_id,
        status: "running",
        toolCount: agents[ev.instance_id]?.toolCount ?? 0,
      };
      return { active: true, agents };
    }
    case "tool_call": {
      if (!ev.instance_id || !prev) return prev;
      const node = prev.agents[ev.instance_id];
      if (!node) return prev;
      return {
        ...prev,
        agents: { ...prev.agents, [ev.instance_id]: { ...node, toolCount: node.toolCount + 1 } },
      };
    }
    case "agent_end": {
      const node = prev?.agents[ev.instance_id];
      if (!prev || !node) return prev;
      return {
        ...prev,
        agents: { ...prev.agents, [ev.instance_id]: { ...node, status: "done" } },
      };
    }
    case "final":
    case "error":
      return prev ? { ...prev, active: false } : prev;
    default:
      return prev;
  }
}

type Action =
  | { kind: "load"; messages: ChatMessage[] }
  | { kind: "user"; id: string; content: string; assistantId: string; attachments?: Attachment[] }
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
        { id: action.id, role: "user", content: action.content, attachments: action.attachments },
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
        case "thinking": {
          // Coalesce into the trailing thinking step OF THE SAME AGENT; if the last step is a
          // tool, belongs to a different agent, or there are none, open a new thinking step —
          // this renders the chronological `thinking → tool → thinking` interleave and keeps
          // concurrent swarm agents' reasoning in separate (coloured) blocks.
          const agent = tagOf(ev);
          return patchLast(state, (m) => {
            const steps = m.steps ?? [];
            const last = steps[steps.length - 1];
            if (last && last.kind === "thinking" && sameInstance(last.agent, agent)) {
              const next = steps.slice();
              next[next.length - 1] = { ...last, text: last.text + ev.delta };
              return { ...m, steps: next };
            }
            return {
              ...m,
              steps: [...steps, { id: newId(), kind: "thinking", text: ev.delta, agent }],
            };
          });
        }
        case "text": {
          // Orchestrator text (untagged) is the agent's own answer — interleaved into the chain as
          // a `text` step (coalesced like thinking, so `text → tool → text` renders chronologically)
          // AND mirrored into `content` (the canonical answer string for Copy/regenerate/reload).
          // A sub-agent's text (tagged) is its report — an `agent_text` step inside that agent's
          // bubble, never the final answer.
          const agent = tagOf(ev);
          if (!agent) {
            return patchLast(state, (m) => {
              const steps = m.steps ?? [];
              const last = steps[steps.length - 1];
              const content = m.content + ev.delta;
              if (last && last.kind === "text" && sameInstance(last.agent, agent)) {
                const next = steps.slice();
                next[next.length - 1] = { ...last, text: last.text + ev.delta };
                return { ...m, content, steps: next };
              }
              return {
                ...m,
                content,
                steps: [...steps, { id: newId(), kind: "text", text: ev.delta, agent }],
              };
            });
          }
          return patchLast(state, (m) => {
            const steps = m.steps ?? [];
            const last = steps[steps.length - 1];
            if (last && last.kind === "agent_text" && sameInstance(last.agent, agent)) {
              const next = steps.slice();
              next[next.length - 1] = { ...last, text: last.text + ev.delta };
              return { ...m, steps: next };
            }
            return {
              ...m,
              steps: [...steps, { id: newId(), kind: "agent_text", text: ev.delta, agent }],
            };
          });
        }
        case "tool_call":
          return patchLast(state, (m) => ({
            ...m,
            steps: [
              ...(m.steps ?? []),
              {
                id: ev.tool_call_id ?? newId(),
                kind: "tool",
                agent: tagOf(ev),
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
          // tool_call_ids are instance-namespaced by the backend, so this match stays correct
          // across concurrent same-spec dispatches.
          return patchLast(state, (m) => ({
            ...m,
            steps: (m.steps ?? []).map((s) =>
              s.kind === "tool" && s.tool.toolCallId === ev.tool_call_id && !s.tool.done
                ? { ...s, tool: { ...s.tool, result: ev.content, done: true } }
                : s
            ),
          }));
        case "agent_start":
          return patchLast(state, (m) => ({
            ...m,
            agents: {
              ...(m.agents ?? {}),
              [ev.instance_id]: { name: ev.name, running: true },
            },
          }));
        case "agent_end":
          return patchLast(state, (m) => {
            const agents = m.agents ?? {};
            const prev = agents[ev.instance_id];
            return {
              ...m,
              agents: {
                ...agents,
                [ev.instance_id]: { name: prev?.name ?? ev.name, running: false },
              },
            };
          });
        case "document":
          // An artifact card in the chain — a big tap target that opens the document
          // in the side panel (the auto-focus side effect lives in runStream, not here).
          return patchLast(state, (m) => ({
            ...m,
            steps: [
              ...(m.steps ?? []),
              {
                id: newId(),
                kind: "document",
                documentId: ev.document_id,
                title: ev.title,
                mimeType: ev.mime_type,
                action: ev.action,
              },
            ],
          }));
        case "skill":
          // A "using/saved skill X" chip in the chain (parallel to the document card). Tagged with
          // the producing agent so it lands in the right bubble in swarm mode.
          return patchLast(state, (m) => ({
            ...m,
            steps: [
              ...(m.steps ?? []),
              {
                id: newId(),
                kind: "skill",
                skillName: ev.skill_name,
                action: ev.action,
                agent: tagOf(ev),
              },
            ],
          }));
        case "final":
          // The streamed `text` deltas already built both `content` and the inline `text` step(s)
          // faithfully, so don't rewrite the steps here (ev.text is the WHOLE answer — overwriting
          // a single step with it would duplicate any pre-tool text fragment). Just finalize content
          // (falling back to ev.text for the no-stream case, where there are no text steps and the
          // bottom bubble renders it).
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
  const { model, effort, featureDocument, refreshConversations, setSwarmFlow } = useApp();
  const [messages, dispatch] = useReducer(reducer, []);
  const [sending, setSending] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Latest swarm flow snapshot (kept in a ref so the streaming callback never reads a stale
  // closure); mirrored into AppContext for the ContextPane flow diagram.
  const flowRef = useRef<SwarmFlowState | null>(null);
  // True once a turn has started in this conversation. The component is keyed by
  // conversation id (Canvas), so a real conversation switch remounts useChat and this
  // ref starts false again — we deliberately never reset it within the effect below.
  const turnStartedRef = useRef(false);

  // Load persisted history whenever the active conversation changes.
  useEffect(() => {
    // A turn has already started here — the first message of a brand-new chat is
    // auto-sent on mount, and React StrictMode invokes this effect a second time.
    // Nothing is persisted server-side yet, so re-loading would fetch [] and wipe the
    // live turn (and the abort below would kill its stream). Leave local state intact.
    if (turnStartedRef.current) return;
    abortRef.current?.abort();
    dispatch({ kind: "load", messages: [] });
    if (!conversationId) return;
    let cancelled = false;
    (async () => {
      try {
        const stored = await api.getMessages(conversationId, userId);
        if (cancelled || turnStartedRef.current) return;
        dispatch({
          kind: "load",
          messages: stored.map((m) => ({
            id: newId(),
            role: m.role,
            content: m.content,
            attachments: m.attachments,
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
    async (prompt: string, signal: AbortSignal, attachments: Attachment[] = []) => {
      try {
        await streamChat(
          {
            user_id: userId,
            conversation_id: conversationId!,
            prompt,
            // Omit when unset so the backend falls back to its configured defaults.
            model: model || undefined,
            effort: effort || undefined,
            // Only the live base64 bytes are uploaded; reload-only refs (document_id) are skipped.
            attachments: attachments
              .filter((a) => a.data)
              .map((a) => ({ filename: a.filename, mime_type: a.mime_type, data: a.data! })),
          },
          (event) => {
            dispatch({ kind: "event", event });
            // Keep the swarm flow diagram (ContextPane) in step with the live trace.
            const nextFlow = reduceFlow(flowRef.current, event);
            if (nextFlow !== flowRef.current) {
              flowRef.current = nextFlow;
              setSwarmFlow(nextFlow);
            }
            // Spotlight a freshly created document: the side panel flips to the Documents
            // tab and opens it while the turn is still streaming.
            if (event.type === "document" && event.action === "created") {
              featureDocument(event.document_id);
            }
          },
          signal
        );
      } catch (err) {
        if (!signal.aborted) {
          dispatch({ kind: "event", event: { type: "error", message: String(err) } });
        }
      } finally {
        setSending(false);
        await refreshConversations().catch((err) => {
          console.error("failed to refresh conversations", err);
        });
        onTurnComplete?.();
      }
    },
    [
      conversationId,
      userId,
      model,
      effort,
      onTurnComplete,
      featureDocument,
      refreshConversations,
      setSwarmFlow,
    ]
  );

  const send = useCallback(
    async (prompt: string, attachments: Attachment[] = []) => {
      if (!conversationId || (!prompt.trim() && attachments.length === 0) || sending) return;
      const controller = new AbortController();
      abortRef.current = controller;
      turnStartedRef.current = true;
      setSending(true);
      flowRef.current = null;
      setSwarmFlow(null);
      dispatch({ kind: "user", id: newId(), content: prompt, assistantId: newId(), attachments });
      await runStream(prompt, controller.signal, attachments);
    },
    [conversationId, sending, runStream, setSwarmFlow]
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
    turnStartedRef.current = true;
    setSending(true);
    flowRef.current = null;
    setSwarmFlow(null);
    dispatch({ kind: "regenerate", assistantId: newId() });
    // Re-send the same uploads — but only those still carrying live bytes. After a reload the
    // attachments are reload-only refs (document_id, no data), so a regenerated turn omits them.
    await runStream(lastUser.content, controller.signal, lastUser.attachments ?? []);
  }, [conversationId, sending, messages, runStream, setSwarmFlow]);

  return { messages, sending, send, stop, regenerate };
}
