import { useState } from "react";
import { Brain, ChevronRight, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types";
import { ToolChip } from "@/components/ToolChip";

function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mb-2 rounded-md border border-border bg-muted/30 text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-muted/50"
      >
        <ChevronRight
          className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")}
        />
        <Brain className="size-3 shrink-0" />
        <span>Reasoning</span>
      </button>
      {open && (
        <pre className="whitespace-pre-wrap break-words border-t border-border px-2 py-2 font-sans">
          {text}
        </pre>
      )}
    </div>
  );
}

export function ChatBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const empty =
    !message.content && !message.thinking && (message.tools?.length ?? 0) === 0;

  return (
    <div className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] space-y-2",
          isUser ? "items-end" : "items-start"
        )}
      >
        {!isUser && message.thinking && <ThinkingBlock text={message.thinking} />}

        {!isUser && (message.tools?.length ?? 0) > 0 && (
          <div className="space-y-1.5">
            {message.tools!.map((t, i) => (
              <ToolChip key={t.toolCallId ?? i} tool={t} />
            ))}
          </div>
        )}

        {(message.content || (!isUser && empty && message.streaming)) && (
          <div
            className={cn(
              "whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-foreground"
            )}
          >
            {message.content}
            {!isUser && message.streaming && empty && (
              <Loader2 className="inline size-4 animate-spin align-middle text-muted-foreground" />
            )}
          </div>
        )}

        {message.error && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {message.error}
          </div>
        )}
      </div>
    </div>
  );
}
