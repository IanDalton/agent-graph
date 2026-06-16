import type { ReactNode } from "react";
import { Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import type { AgentRef } from "@/types";
import { colorForAgent } from "./agentColors";

/** Wraps one sub-agent's contiguous run of steps (thinking / tool calls / report) in a coloured
 *  bubble — a left-border accent + tint keyed to the agent, a header with its name, and a spinner
 *  while it's still running. This is what makes concurrent swarm agents legible: each agent's live
 *  work reads as its own colour, interleaved in arrival order with the others. */
export function AgentBubble({
  agent,
  running,
  children,
  className,
}: {
  agent: AgentRef;
  running: boolean;
  children: ReactNode;
  /** Extra classes for the wrapper (e.g. min-w-0 inside a grid column). */
  className?: string;
}) {
  const color = colorForAgent(agent.agentId);
  return (
    <div
      className={cn(
        "rounded-lg border border-white/5 border-l-2 py-1.5 pl-2.5 pr-1.5",
        color.border,
        color.bg,
        className
      )}
    >
      <div className="mb-1 flex items-center gap-1.5 px-0.5">
        <span className={cn("inline-block size-2 shrink-0 rounded-full", color.dot)} />
        <span className={cn("truncate text-xs font-medium font-mono", color.text)}>
          {agent.name || agent.agentId}
        </span>
        {running && (
          <Loader2 className="size-3 shrink-0 animate-spin text-muted-foreground" />
        )}
      </div>
      <div className="space-y-1.5">{children}</div>
    </div>
  );
}
