import { useEffect, useRef } from "react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useApp } from "@/state/AppContext";
import { useChat } from "@/hooks/useChat";
import { ChatBubble } from "@/components/ChatBubble";
import { Composer } from "@/components/Composer";
import { ModeIcon } from "@/components/ModeIcon";
import type { Conversation } from "@/types";

/**
 * The middle "Dynamic Canvas". Today only the Regular renderer exists; future modes
 * (Research timeline, Swarm task board, Council debate) will branch on
 * `conversation.mode` here, each its own component, without touching the shell.
 */
function RegularCanvas({
  conversation,
  userId,
  onTurnComplete,
}: {
  conversation: Conversation;
  userId: string;
  onTurnComplete: () => void;
}) {
  const { messages, sending, send } = useChat(
    conversation.conversation_id,
    userId,
    onTurnComplete
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-2 border-b border-border px-4 py-3">
        <ModeIcon mode={conversation.mode} className="size-4 text-muted-foreground" />
        <span className="text-sm font-medium">
          {conversation.title ?? "Regular chat"}
        </span>
      </header>

      <ScrollArea className="flex-1">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 p-4">
          {messages.length === 0 && (
            <div className="py-16 text-center text-sm text-muted-foreground">
              Start the conversation below.
            </div>
          )}
          {messages.map((m) => (
            <ChatBubble key={m.id} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="mx-auto w-full max-w-3xl">
        <Composer disabled={sending} onSend={send} />
      </div>
    </div>
  );
}

export function Canvas({ onTurnComplete }: { onTurnComplete: () => void }) {
  const { conversations, activeId, userId } = useApp();
  const conversation = conversations.find((c) => c.conversation_id === activeId);

  if (!conversation) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        Select or create a conversation.
      </div>
    );
  }

  // Keying by id resets per-conversation chat state cleanly when switching threads.
  switch (conversation.mode) {
    case "regular":
    default:
      return (
        <RegularCanvas
          key={conversation.conversation_id}
          conversation={conversation}
          userId={userId}
          onTurnComplete={onTurnComplete}
        />
      );
  }
}
