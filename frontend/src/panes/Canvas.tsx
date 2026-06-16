import { useEffect, useRef } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useApp } from "@/state/AppContext";
import { useChat } from "@/hooks/useChat";
import { useStickyScroll } from "@/hooks/useStickyScroll";
import { ChatBubble } from "@/components/ChatBubble";
import { Composer } from "@/components/Composer";
import { ModeIcon } from "@/components/ModeIcon";
import { SwarmCanvas } from "@/panes/SwarmCanvas";
import type { Attachment, Conversation, Mode } from "@/types";

/** A message composed before a conversation exists (new-chat flow), replayed once on mount. */
type PendingMessage = { text: string; attachments: Attachment[] };

const MODE_OPTIONS: { mode: Mode; label: string; desc: string }[] = [
  { mode: "regular", label: "Regular chat", desc: "Memory-backed assistant" },
  { mode: "research", label: "Deep research", desc: "Multi-source, cited reports" },
  { mode: "swarm", label: "Agent swarm", desc: "Orchestrates parallel sub-agents" },
];

/** Shown when pendingNewChat: mode picker cards above + composer below. Sending a message
 *  without picking a mode defaults to regular chat. */
function PendingCanvas({
  onPick,
  onSend,
}: {
  onPick: (mode: Mode) => void;
  onSend: (text: string, attachments: Attachment[]) => void;
}) {
  return (
    <div className="flex h-full flex-col bg-slate-950">
      <div className="flex flex-1 flex-col items-center justify-center px-4">
        <div className="w-full max-w-lg">
          <h2 className="mb-1 text-center text-xl font-semibold text-white">
            Start a new conversation
          </h2>
          <p className="mb-8 text-center text-sm text-muted-foreground">
            Choose a mode, or just type below for a regular chat
          </p>
          <div className="flex flex-col gap-3">
            {MODE_OPTIONS.map((opt) => (
              <button
                key={opt.mode}
                type="button"
                onClick={() => onPick(opt.mode)}
                className="flex items-center gap-4 rounded-xl border border-white/10 bg-white/5 px-5 py-4 text-left transition-colors hover:border-white/20 hover:bg-white/10"
              >
                <ModeIcon mode={opt.mode} className="size-6 shrink-0 text-muted-foreground" />
                <span>
                  <span className="block font-medium text-white">{opt.label}</span>
                  <span className="block text-sm text-muted-foreground">{opt.desc}</span>
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="mx-auto w-full max-w-3xl">
        <Composer disabled={false} sending={false} onSend={onSend} onStop={() => { }} />
      </div>
    </div>
  );
}

/**
 * The middle "Dynamic Canvas". Today only the Regular renderer exists; future modes
 * (Research timeline, Swarm task board, Council debate) will branch on
 * `conversation.mode` here, each its own component, without touching the shell.
 */
function RegularCanvas({
  conversation,
  userId,
  onTurnComplete,
  autoSend,
  onAutoSent,
}: {
  conversation: Conversation;
  userId: string;
  onTurnComplete: () => void;
  autoSend?: PendingMessage;
  onAutoSent?: () => void;
}) {
  const { messages, sending, send, stop, regenerate } = useChat(
    conversation.conversation_id,
    userId,
    onTurnComplete
  );
  const viewportRef = useStickyScroll(messages);
  const autoSentRef = useRef(false);

  // Fire the deferred message once on mount (from composing before picking a mode).
  // The ref guard keeps it to a single send under React StrictMode's double mount.
  useEffect(() => {
    if (autoSend && !autoSentRef.current) {
      autoSentRef.current = true;
      send(autoSend.text, autoSend.attachments);
      onAutoSent?.();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex h-full flex-col bg-slate-950">
      <header className="flex items-center gap-2 border-b border-white/10 px-4 py-3">
        <ModeIcon mode={conversation.mode} className="size-4 text-muted-foreground" />
        <span className="text-sm font-medium">
          {conversation.title ?? "Regular chat"}
        </span>
      </header>

      <ScrollArea className="flex-1" viewportRef={viewportRef}>
        <div className="mx-auto flex max-w-3xl flex-col gap-4 p-4">
          {messages.length === 0 && (
            <div className="py-16 text-center text-sm text-muted-foreground">
              Start the conversation below.
            </div>
          )}
          {messages.map((m, i) => {
            const isLastAssistant =
              m.role === "assistant" && i === messages.length - 1;
            return (
              <ChatBubble
                key={m.id}
                message={m}
                onRegenerate={isLastAssistant && !sending ? regenerate : undefined}
              />
            );
          })}
        </div>
      </ScrollArea>

      <div className="mx-auto w-full max-w-3xl">
        <Composer disabled={sending} sending={sending} onSend={send} onStop={stop} />
      </div>
    </div>
  );
}

export function Canvas({ onTurnComplete }: { onTurnComplete: () => void }) {
  const { conversations, activeId, userId, pendingNewChat, newConversation } = useApp();
  const pendingMessageRef = useRef<PendingMessage | null>(null);
  const conversation = conversations.find((c) => c.conversation_id === activeId);

  if (!conversation) {
    if (pendingNewChat) {
      return (
        <PendingCanvas
          onPick={(mode) => void newConversation(mode)}
          onSend={(text, attachments) => {
            pendingMessageRef.current = { text, attachments };
            void newConversation("regular");
          }}
        />
      );
    }
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Select or create a conversation.
      </div>
    );
  }

  // Keying by id resets per-conversation chat state cleanly when switching threads.
  switch (conversation.mode) {
    case "swarm":
      return (
        <SwarmCanvas
          key={conversation.conversation_id}
          conversation={conversation}
          userId={userId}
          onTurnComplete={onTurnComplete}
          autoSend={pendingMessageRef.current ?? undefined}
          onAutoSent={() => { pendingMessageRef.current = null; }}
        />
      );
    case "regular":
    default:
      return (
        <RegularCanvas
          key={conversation.conversation_id}
          conversation={conversation}
          userId={userId}
          onTurnComplete={onTurnComplete}
          autoSend={pendingMessageRef.current ?? undefined}
          onAutoSent={() => { pendingMessageRef.current = null; }}
        />
      );
  }
}
