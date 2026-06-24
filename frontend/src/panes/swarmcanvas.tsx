import { useEffect, useRef } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat } from "@/hooks/usechat";
import { useStickyScroll } from "@/hooks/usestickyscroll";
import { ChatBubble } from "@/components/chatbubble";
import { Composer } from "@/components/composer";
import { ModeIcon } from "@/components/modeicon";
import type { Attachment, Conversation } from "@/types";
import { SwarmSteps } from "@/swarm/swarmsteps";

export function SwarmCanvas({
  conversation,
  userId,
  onTurnComplete,
  autoSend,
  onAutoSent,
}: {
  conversation: Conversation;
  userId: string;
  onTurnComplete: () => void;
  autoSend?: { text: string; attachments: Attachment[] };
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
          {conversation.title ?? "Agent swarm"}
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
                renderSteps={(steps) => <SwarmSteps steps={steps} agents={m.agents} />}
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
