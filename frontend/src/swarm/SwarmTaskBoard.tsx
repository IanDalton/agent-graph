import { CheckCircle2, Loader2, Network } from "lucide-react";

import { ToolChip } from "@/components/ToolChip";
import type { ToolEvent } from "@/types";
import { AgentTaskCard } from "./AgentTaskCard";
import { parseSwarmArgs, parseSwarmResult } from "./parseSwarm";

export function SwarmTaskBoard({ tool }: { tool: ToolEvent }) {
  const args = parseSwarmArgs(tool.args);
  if (!args) {
    return <ToolChip tool={tool} />;
  }

  const result = parseSwarmResult(tool.result);

  const allDone = result && result.reports.length === args.tasks.length;
  const hasError = result?.reports.some((r) => r.error);

  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/30 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex items-center gap-2">
          <Network className="size-4 text-sky-400" />
          <span className="text-sm font-medium">Agent Swarm</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-flex rounded-full bg-white/10 px-2 py-0.5 text-xs font-mono text-muted-foreground">
            {args.tasks.length} tasks
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

      {/* Task grid */}
      <div
        className={`grid gap-2 p-3 ${args.tasks.length === 1 ? "grid-cols-1" : "grid-cols-2"
          }`}
      >
        {args.tasks.map((task, i) => (
          <AgentTaskCard
            key={i}
            task={task}
            report={result?.reports[i] ?? null}
            index={i}
          />
        ))}
      </div>
    </div>
  );
}
