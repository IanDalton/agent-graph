import { ToolChip } from "@/components/ToolChip";
import type { ToolEvent } from "@/types";
import { AgentTaskCard } from "./AgentTaskCard";
import { parseRunAgentArgs, parseRunAgentResult } from "./parseSwarm";

export function RunAgentCard({ tool }: { tool: ToolEvent }) {
  const args = parseRunAgentArgs(tool.args);
  if (!args) {
    return <ToolChip tool={tool} />;
  }

  const report = parseRunAgentResult(tool.result);
  const task = { agent: args.agent, task: args.task, context: args.context };

  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/30 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Single Agent</span>
        </div>
      </div>

      {/* Single task card */}
      <div className="p-3">
        <AgentTaskCard task={task} report={report} index={0} />
      </div>
    </div>
  );
}
