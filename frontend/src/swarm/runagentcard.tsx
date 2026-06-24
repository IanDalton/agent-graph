import { ToolChip } from "@/components/toolchip";
import type { ToolEvent } from "@/types";
import { AgentTaskCard } from "./agenttaskcard";
import { parseSendMessageArgs, parseSendMessageResult } from "./parseswarm";

export function RunAgentCard({ tool }: { tool: ToolEvent }) {
  const args = parseSendMessageArgs(tool.args);
  if (!args) {
    return <ToolChip tool={tool} />;
  }

  const report = parseSendMessageResult(tool.result);

  return (
    <div className="rounded-xl border border-white/10 bg-slate-900/30 overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between gap-2 border-b border-white/10 px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Message</span>
        </div>
      </div>

      {/* Single message card */}
      <div className="p-3">
        <AgentTaskCard message={args} report={report} index={0} />
      </div>
    </div>
  );
}
