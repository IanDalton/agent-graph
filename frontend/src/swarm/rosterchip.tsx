import { Check, Loader2, UserCog } from "lucide-react";

import type { ToolEvent } from "@/types";

export function RosterChip({ tool }: { tool: ToolEvent }) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full bg-amber-500/10 border border-amber-500/20 px-3 py-1 text-xs">
      <UserCog className="size-3.5 text-amber-400" />
      <span className="font-mono font-medium">{tool.toolName}</span>
      <span className="size-1.5 rounded-full bg-amber-400/60"></span>
      {tool.done ? (
        <Check className="size-3 text-emerald-500" />
      ) : (
        <Loader2 className="size-3 animate-spin text-muted-foreground" />
      )}
    </div>
  );
}
