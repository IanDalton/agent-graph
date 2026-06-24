import { useMemo } from "react";
import ReactFlow, { Background, type Edge, type Node } from "reactflow";
import "reactflow/dist/style.css";
import { Network } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useApp } from "@/state/appcontext";
import { colorForAgent } from "@/swarm/agentcolors";

const ORCHESTRATOR_ID = "__orchestrator__";
const ORCHESTRATOR_COLOR = "hsl(215 16% 60%)";

/** Right-pane card: a live hub-and-spoke diagram of the orchestrator and the sub-agents it has
 *  dispatched this turn. Each spoke is colour-matched to its chat bubble; the edge animates while
 *  the agent is running and the node shows its tool-call count. Built from the shared `swarmFlow`
 *  snapshot (AppContext), which useChat updates as the stream arrives. Ephemeral — it resets on a
 *  new turn / conversation switch. Reuses ReactFlow (same dep as the Memory Graph). */
export function SwarmFlowCard() {
  const { swarmFlow } = useApp();

  const { nodes, edges, count } = useMemo(() => {
    const agents = swarmFlow ? Object.values(swarmFlow.agents) : [];
    const nodes: Node[] = [
      {
        id: ORCHESTRATOR_ID,
        position: { x: 0, y: 0 },
        data: {
          label: (
            <div className="text-center leading-tight">
              <div className="font-medium">Orchestrator</div>
              <div className="text-[9px] opacity-70">
                {swarmFlow?.active ? "coordinating…" : "done"}
              </div>
            </div>
          ),
        },
        style: nodeStyle(ORCHESTRATOR_COLOR, true),
      },
    ];
    const edges: Edge[] = [];

    const n = agents.length;
    const radius = Math.max(130, n * 26);
    agents.forEach((a, i) => {
      // Lay spokes on a ring, starting at the top and going clockwise.
      const angle = (2 * Math.PI * i) / Math.max(n, 1) - Math.PI / 2;
      const color = colorForAgent(a.agentId);
      const running = a.status === "running";
      nodes.push({
        id: a.instanceId,
        position: { x: radius * Math.cos(angle), y: radius * Math.sin(angle) },
        data: {
          label: (
            <div className="text-center leading-tight">
              <div className="font-medium">{a.name || a.agentId}</div>
              <div className="text-[9px] opacity-70">
                {running ? "working…" : "done"}
                {a.toolCount ? ` · ${a.toolCount} tools` : ""}
              </div>
            </div>
          ),
        },
        style: nodeStyle(color.hsl, running),
      });
      edges.push({
        id: `e-${a.instanceId}`,
        source: ORCHESTRATOR_ID,
        target: a.instanceId,
        animated: running,
        style: { stroke: color.hsl, strokeWidth: 1.5, opacity: running ? 1 : 0.5 },
      });
    });

    return { nodes, edges, count: n };
  }, [swarmFlow]);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-1.5 text-sm">
          <Network className="size-3.5 text-muted-foreground" />
          Swarm Activity
        </CardTitle>
      </CardHeader>
      <CardContent>
        {count === 0 ? (
          <div className="text-xs text-muted-foreground">
            No agents dispatched yet — sub-agents appear here as the orchestrator delegates work.
          </div>
        ) : (
          <div className="h-[260px] w-full overflow-hidden rounded-md border border-border">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              fitView
              fitViewOptions={{ padding: 0.25 }}
              minZoom={0.1}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              zoomOnScroll={false}
              panOnDrag={false}
              zoomOnDoubleClick={false}
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={16} />
            </ReactFlow>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function nodeStyle(color: string, active: boolean): React.CSSProperties {
  return {
    fontSize: 11,
    borderRadius: 8,
    padding: "6px 10px",
    minWidth: 40,
    border: `2px solid ${color}`,
    background: "hsl(var(--card))",
    color: "hsl(var(--foreground))",
    opacity: active ? 1 : 0.7,
    boxShadow: active ? `0 0 0 2px ${color}33` : undefined,
  };
}
