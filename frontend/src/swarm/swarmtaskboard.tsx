import { CheckCircle2, Loader2, Network } from "lucide-react";

import { ToolChip } from "@/components/toolchip";
import type { ToolEvent } from "@/types";
import { AgentTaskCard } from "./agenttaskcard";
import { parseSendMessagesArgs, parseSwarmResult } from "./parseswarm";

export function SwarmTaskBoard({ tool }: { tool: ToolEvent }) {
  const args = parseSendMessagesArgs(tool.args);
  if (!args) {
    return <ToolChip tool={tool} />;
  }

  const result = parseSwarmResult(tool.result);

  const allDone = result && result.reports.length === args.messages.length;
  const hasError = result?.reports.some((r) => r.error);

  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/30 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex items-center gap-2">
          <Network className="size-4 text-sky-400" />
          <span className="text-sm font-medium">Agency Messages</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex rounded-full bg-white/10 px-2 py-0.5 text-xs font-mono text-muted-foreground">
            {args.messages.length} messages
          </span>
          <div>
            {!allDone ? (
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
            ) : hasError ? (
              <span className="text-xs text-destructive font-mono">error</span>
            ) : (
              <CheckCircle2 className="size-4 text-emerald-500" />
            )}
          </div>
        </div>
      </div>

      {/* Message grid */}
      <div
        className={`grid gap-2 p-3 ${args.messages.length === 1 ? "grid-cols-1" : "grid-cols-2"
          }`}
      >
        {args.messages.map((message, i) => (
          <AgentTaskCard
            key={i}
            message={message}
            report={result?.reports[i] ?? null}
            index={i}
          />
        ))}
      </div>
    </div>
  );
}
