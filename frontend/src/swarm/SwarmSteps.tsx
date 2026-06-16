import React, { useMemo } from "react";

import type { AgentRef, ChatMessage, Step } from "@/types";
import { SwarmStepItem } from "./SwarmStepItem";
import { AgentBubble } from "./AgentBubble";

interface StepGroup {
  instanceId?: string;
  agent?: AgentRef;
  steps: Step[];
}

/** Group steps into contiguous runs by producing agent instance. Orchestrator steps (no agent)
 *  stay ungrouped and render flat; each contiguous sub-agent run becomes one coloured bubble.
 *  Because parallel agents' frames arrive interleaved, the result is a stream of alternating
 *  coloured bubbles — exactly how the concurrency happened. */
function groupSteps(steps: Step[]): StepGroup[] {
  const groups: StepGroup[] = [];
  for (const s of steps) {
    const instanceId = s.agent?.instanceId;
    const last = groups[groups.length - 1];
    if (last && last.instanceId === instanceId) {
      last.steps.push(s);
    } else {
      groups.push({ instanceId, agent: s.agent, steps: [s] });
    }
  }
  return groups;
}

/** Renders a swarm turn's whole steps chain: orchestrator steps flat, each sub-agent's contiguous
 *  run inside its own coloured {@link AgentBubble} (with a running spinner from the turn's agent map). */
export function SwarmSteps({ steps, agents }: { steps: Step[]; agents?: ChatMessage["agents"] }) {
  const groups = useMemo(() => groupSteps(steps), [steps]);
  return (
    <div className="space-y-1.5">
      {groups.map((g) =>
        g.agent ? (
          <AgentBubble
            key={g.steps[0].id}
            agent={g.agent}
            running={agents?.[g.agent.instanceId]?.running ?? false}
          >
            {g.steps.map((s) => (
              <SwarmStepItem key={s.id} step={s} />
            ))}
          </AgentBubble>
        ) : (
          g.steps.map((s) => (
            <React.Fragment key={s.id}>
              <SwarmStepItem step={s} />
            </React.Fragment>
          ))
        )
      )}
    </div>
  );
}
