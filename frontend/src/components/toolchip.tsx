import { useState } from "react";
import { Check, ChevronRight, Loader2, Wrench } from "lucide-react";

import { cn } from "@/lib/utils";
import type { ToolEvent } from "@/types";

const stringify = (value: unknown): string => {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

/** A collapsible record of one tool call + its result. This is the seed of the
 *  future "chain-of-thought" timeline — the Research mode will expand on it. */
export function ToolChip({ tool }: { tool: ToolEvent }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/40 text-xs">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-xl px-2.5 py-1.5 text-left transition-colors hover:bg-white/5"
      >
        <ChevronRight
          className={cn("size-3 shrink-0 transition-transform", open && "rotate-90")}
        />
        <Wrench className="size-3 shrink-0 text-sky-400" />
        <span className="font-mono font-medium text-foreground">{tool.toolName ?? "tool"}</span>
        <span className="ml-auto shrink-0">
          {tool.done ? (
            <Check className="size-3 text-emerald-500" />
          ) : (
            <Loader2 className="size-3 animate-spin text-muted-foreground" />
          )}
        </span>
      </button>
      {open && (
        <div className="space-y-2 border-t border-white/10 px-2.5 py-2">
          {tool.args != null && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground/70">
                args
              </div>
              <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950/60 p-2 font-mono text-[11px] text-zinc-200">
                {stringify(tool.args)}
              </pre>
            </div>
          )}
          {tool.done && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground/70">
                result
              </div>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950/60 p-2 font-mono text-[11px] text-zinc-200">
                {stringify(tool.result)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
