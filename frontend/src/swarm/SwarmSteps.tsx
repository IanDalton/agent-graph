import React, { useMemo } from "react";

import type { ChatMessage, Step } from "@/types";
import { SwarmStepItem } from "./SwarmStepItem";
import { ParallelAgents, type AgentLane } from "./ParallelAgents";

/** A run of orchestrator steps (no agent), rendered flat. */
interface OrchestratorBlock {
  kind: "orchestrator";
  key: string;
  steps: Step[];
}
/** A run of concurrent sub-agent steps, bucketed into one lane per instance. */
interface ParallelBlock {
  kind: "parallel";
  key: string;
  lanes: AgentLane[];
}
type Block = OrchestratorBlock | ParallelBlock;

/** Segment the flat step chain into orchestrator runs (no agent) and parallel phases (contiguous
 *  agent-tagged steps). Within a phase, steps are bucketed by `instanceId` into lanes ordered by
 *  first appearance — so each agent's interleaved fragments merge into ONE lane instead of the old
 *  wall of alternating bubbles. */
function segment(steps: Step[]): Block[] {
  const blocks: Block[] = [];
  for (const s of steps) {
    const last = blocks[blocks.length - 1];
    if (!s.agent) {
      if (last && last.kind === "orchestrator") last.steps.push(s);
      else blocks.push({ kind: "orchestrator", key: s.id, steps: [s] });
      continue;
    }
    // Agent-tagged: extend the current parallel phase, or open a new one.
    let phase = last && last.kind === "parallel" ? last : undefined;
    if (!phase) {
      phase = { kind: "parallel", key: s.id, lanes: [] };
      blocks.push(phase);
    }
    const lane = phase.lanes.find((l) => l.agent.instanceId === s.agent!.instanceId);
    if (lane) lane.steps.push(s);
    else phase.lanes.push({ agent: s.agent, steps: [s] });
  }
  return blocks;
}

/** Renders a swarm turn's whole step chain: orchestrator steps flat, and each parallel phase in a
 *  SHARED space (one bubble / columns / tabbed switcher) via {@link ParallelAgents}, so concurrent
 *  sub-agents no longer compete as a stack of interleaved blocks. */
export function SwarmSteps({ steps, agents }: { steps: Step[]; agents?: ChatMessage["agents"] }) {
  const blocks = useMemo(() => segment(steps), [steps]);
  return (
    <div className="space-y-1.5">
      {blocks.map((b) =>
        b.kind === "parallel" ? (
          <ParallelAgents key={b.key} lanes={b.lanes} agents={agents} />
        ) : (
          b.steps.map((s) => (
            <React.Fragment key={s.id}>
              <SwarmStepItem step={s} />
            </React.Fragment>
          ))
        )
      )}
    </div>
  );
}
