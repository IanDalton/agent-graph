import { useState } from "react";
import { Check, Loader2 } from "lucide-react";

import { cn } from "@/lib/utils";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AgentRef, ChatMessage, Step } from "@/types";
import { AgentBubble } from "./AgentBubble";
import { SwarmStepItem } from "./SwarmStepItem";
import { colorForAgent } from "./agentColors";

/** One concurrent sub-agent's merged trace within a parallel phase. */
export interface AgentLane {
  agent: AgentRef;
  steps: Step[];
}

// Up to this many concurrent agents render side-by-side as columns; beyond it, a tabbed switcher
// (so a wide fan-out doesn't overwhelm the narrow chat column).
const COLUMNS_MAX = 2;

function toolCount(steps: Step[]): number {
  return steps.reduce((n, s) => (s.kind === "tool" ? n + 1 : n), 0);
}

/** Merge adjacent thinking / agent_text steps in a lane into one block.
 *
 *  The reducer only coalesces *consecutive* same-instance deltas, but concurrent agents' frames
 *  arrive interleaved (A's token, B's token, A's token…), so within a flat list each token lands as
 *  its own step. Once we've bucketed a single agent's steps into a lane they're adjacent again, so
 *  re-merging here turns the per-token "Reasoning" spam back into proper thinking/report blocks. */
function coalesce(steps: Step[]): Step[] {
  const out: Step[] = [];
  for (const s of steps) {
    const last = out[out.length - 1];
    if (
      last &&
      (last.kind === "thinking" || last.kind === "agent_text") &&
      last.kind === s.kind
    ) {
      out[out.length - 1] = { ...last, text: last.text + (s as typeof last).text };
    } else {
      out.push(s);
    }
  }
  return out;
}

function laneSteps(lane: AgentLane) {
  return coalesce(lane.steps).map((s) => <SwarmStepItem key={s.id} step={s} />);
}

/** Renders one parallel phase (the concurrent sub-agents between two orchestrator steps) in a
 *  SHARED space instead of a wall of interleaved bubbles: a single bubble for one agent, columns
 *  for a few, and a tabbed switcher for many. Per-agent running state comes from the turn's
 *  `agents` map; each lane keeps its own colour (shared with the flow diagram). */
export function ParallelAgents({
  lanes,
  agents,
}: {
  lanes: AgentLane[];
  agents?: ChatMessage["agents"];
}) {
  const isRunning = (id: string) => agents?.[id]?.running ?? false;

  // One agent → a plain bubble, no extra chrome.
  if (lanes.length === 1) {
    const lane = lanes[0];
    return (
      <AgentBubble agent={lane.agent} running={isRunning(lane.agent.instanceId)}>
        {laneSteps(lane)}
      </AgentBubble>
    );
  }

  // A few agents → side-by-side columns; all visible at once.
  if (lanes.length <= COLUMNS_MAX) {
    return (
      <div className={cn("grid gap-2", lanes.length === 2 ? "grid-cols-2" : "grid-cols-1")}>
        {lanes.map((lane) => (
          <AgentBubble
            key={lane.agent.instanceId}
            agent={lane.agent}
            running={isRunning(lane.agent.instanceId)}
            className="min-w-0"
          >
            {laneSteps(lane)}
          </AgentBubble>
        ))}
      </div>
    );
  }

  // Many agents → a tabbed switcher: chips with status + tool count, one trace shown at a time.
  return (
    <TabbedLanes lanes={lanes} isRunning={isRunning} />
  );
}

function TabbedLanes({
  lanes,
  isRunning,
}: {
  lanes: AgentLane[];
  isRunning: (id: string) => boolean;
}) {
  // Controlled so each chip knows whether it's the active one and can highlight itself.
  const [active, setActive] = useState(lanes[0].agent.instanceId);
  return (
    <Tabs
      defaultValue={lanes[0].agent.instanceId}
      value={active}
      onValueChange={setActive}
      className="space-y-1.5"
    >
      <TabsList className="flex h-auto w-full flex-wrap justify-start gap-1 bg-transparent p-0">
        {lanes.map((lane) => (
          <LaneChip
            key={lane.agent.instanceId}
            lane={lane}
            running={isRunning(lane.agent.instanceId)}
            active={active === lane.agent.instanceId}
          />
        ))}
      </TabsList>
      {lanes.map((lane) => {
        const color = colorForAgent(lane.agent.agentId);
        return (
          <TabsContent key={lane.agent.instanceId} value={lane.agent.instanceId}>
            <div
              className={cn(
                "space-y-1.5 rounded-lg border border-white/5 border-l-2 py-1.5 pl-2.5 pr-1.5",
                color.border,
                color.bg
              )}
            >
              {laneSteps(lane)}
            </div>
          </TabsContent>
        );
      })}
    </Tabs>
  );
}

/** A tab trigger for one agent: colour dot, name, running/done indicator, and tool-call count.
 *  The active chip is highlighted (brighter background + ring); the others read as dimmed. */
function LaneChip({
  lane,
  running,
  active,
}: {
  lane: AgentLane;
  running: boolean;
  active: boolean;
}) {
  const color = colorForAgent(lane.agent.agentId);
  const tools = toolCount(lane.steps);
  return (
    <TabsTrigger
      value={lane.agent.instanceId}
      className={cn(
        "flex-none gap-1.5 border font-mono transition-colors",
        active
          ? "border-white/20 bg-white/10 text-foreground shadow-sm"
          : "border-transparent opacity-60 hover:bg-white/5 hover:opacity-100"
      )}
    >
      <span className={cn("inline-block size-2 shrink-0 rounded-full", color.dot)} />
      <span className={cn("max-w-[10rem] truncate", color.text)}>
        {lane.agent.name || lane.agent.agentId}
      </span>
      {running ? (
        <Loader2 className="size-3 shrink-0 animate-spin text-muted-foreground" />
      ) : (
        <Check className="size-3 shrink-0 text-emerald-500" />
      )}
      {tools > 0 && (
        <span className="rounded bg-white/10 px-1 text-[10px] text-muted-foreground">{tools}</span>
      )}
    </TabsTrigger>
  );
}
