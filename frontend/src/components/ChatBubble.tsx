import { useState } from "react";
import { Brain, Check, ChevronRight, Copy, Loader2, RefreshCw } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ChatMessage, Step } from "@/types";
import { ToolChip } from "@/components/ToolChip";
import { Markdown } from "@/components/Markdown";

/** A collapsible reasoning run — one segment of the agent's thinking between tool calls. */
function ThinkingBlock({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/40 text-xs text-muted-foreground">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-xl px-2.5 py-1.5 text-left transition-colors hover:bg-white/5"
      >
        <ChevronRight
          className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")}
        />
        <Brain className="size-3 shrink-0" />
        <span className="font-medium">Reasoning</span>
      </button>
      {open && (
        <pre className="whitespace-pre-wrap break-words border-t border-white/10 px-2.5 py-2 font-sans">
          {text}
        </pre>
      )}
    </div>
  );
}

/** Renders one node of the chronological chain in arrival order. */
function StepItem({ step }: { step: Step }) {
  if (step.kind === "thinking") {
    if (!step.text.trim()) return null;
    return <ThinkingBlock text={step.text} />;
  }
  return <ToolChip tool={step.tool} />;
}

/** Low-profile, hover-revealed action row on a completed assistant turn. */
function MessageActions({
  content,
  onRegenerate,
}: {
  content: string;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard may be unavailable (insecure context); silently ignore.
    }
  };

  const btn =
    "rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-white/5 hover:text-foreground";

  return (
    <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
      <button type="button" onClick={copy} aria-label="Copy" title="Copy" className={btn}>
        {copied ? (
          <Check className="size-3.5 text-emerald-500" />
        ) : (
          <Copy className="size-3.5" />
        )}
      </button>
      {onRegenerate && (
        <button
          type="button"
          onClick={onRegenerate}
          aria-label="Regenerate"
          title="Regenerate"
          className={btn}
        >
          <RefreshCw className="size-3.5" />
        </button>
      )}
    </div>
  );
}

export function ChatBubble({
  message,
  onRegenerate,
}: {
  message: ChatMessage;
  /** Provided only for the latest assistant turn (re-runs the last prompt). */
  onRegenerate?: () => void;
}) {
  const isUser = message.role === "user";
  const steps = message.steps ?? [];
  const empty = !message.content && steps.length === 0;
  const completed = !isUser && !message.streaming && !message.error && !!message.content;

  return (
    <div className={cn("group flex w-full", isUser ? "justify-end" : "justify-start")}>
      <div className={cn("max-w-[80%] space-y-2", isUser ? "items-end" : "items-start")}>
        {!isUser && steps.length > 0 && (
          <div className="space-y-1.5">
            {steps.map((s) => (
              <StepItem key={s.id} step={s} />
            ))}
          </div>
        )}

        {(message.content || (!isUser && empty && message.streaming)) && (
          <div
            className={cn(
              "break-words rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
              isUser
                ? "whitespace-pre-wrap bg-primary text-primary-foreground"
                : "border border-white/10 bg-slate-900/40 text-foreground"
            )}
          >
            {isUser ? message.content : <Markdown>{message.content}</Markdown>}
            {!isUser && message.streaming && empty && (
              <Loader2 className="inline size-4 animate-spin align-middle text-muted-foreground" />
            )}
          </div>
        )}

        {message.error && (
          <div className="rounded-xl border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {message.error}
          </div>
        )}

        {completed && (
          <MessageActions content={message.content} onRegenerate={onRegenerate} />
        )}
      </div>
    </div>
  );
}
