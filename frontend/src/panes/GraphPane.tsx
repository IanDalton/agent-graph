import { useEffect, useMemo, useState } from "react";
import ReactFlow, { Background, Controls, type Edge, type Node } from "reactflow";
import "reactflow/dist/style.css";
import { Network, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import type { MemoryGraph } from "@/types";

/** Deterministic circular layout: graph DBs don't store coordinates, so we place nodes evenly on a
 *  circle (radius grows with count). ReactFlow's fitView then frames them. A real layout engine
 *  (dagre/ELK) can replace this later without touching the data flow. */
function layout(graph: MemoryGraph): { nodes: Node[]; edges: Edge[] } {
  const n = graph.nodes.length;
  const radius = Math.max(120, n * 26);
  const nodes: Node[] = graph.nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(n, 1);
    return {
      id: node.id,
      position: { x: radius * Math.cos(angle), y: radius * Math.sin(angle) },
      data: { label: `${node.label} · ${node.type}` },
      style: {
        fontSize: 11,
        borderRadius: 8,
        padding: 6,
        border: "1px solid hsl(var(--border))",
        background: "hsl(var(--card))",
      },
    };
  });
  const edges: Edge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    label: e.label,
    labelStyle: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
    style: { stroke: "hsl(var(--muted-foreground))" },
  }));
  return { nodes, edges };
}

/** Right-pane card rendering the user's agent-built knowledge graph. Read-only; refreshes when the
 *  active conversation changes or a turn completes (refreshKey), mirroring SummaryCard. */
export function MemoryGraphCard({ refreshKey }: { refreshKey: number }) {
  const { userId } = useApp();
  const [graph, setGraph] = useState<MemoryGraph | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .getGraph(userId)
      .then((g) => !cancelled && setGraph(g))
      .catch(() => !cancelled && setGraph({ nodes: [], edges: [] }))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  const { nodes, edges } = useMemo(
    () => layout(graph ?? { nodes: [], edges: [] }),
    [graph]
  );

  function refresh() {
    setLoading(true);
    api
      .getGraph(userId)
      .then(setGraph)
      .catch(() => undefined)
      .finally(() => setLoading(false));
  }

  const hasNodes = (graph?.nodes.length ?? 0) > 0;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="flex items-center gap-1.5 text-sm">
          <Network className="size-3.5 text-muted-foreground" />
          Memory Graph
        </CardTitle>
        <Button
          variant="ghost"
          size="icon"
          className="size-6 text-muted-foreground"
          onClick={refresh}
          disabled={loading}
          title="Refresh graph"
        >
          <RefreshCw className={`size-3.5 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </CardHeader>
      <CardContent>
        {loading && !graph ? (
          <Skeleton className="h-[260px] w-full" />
        ) : hasNodes ? (
          <div className="h-[260px] w-full overflow-hidden rounded-md border border-border">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              nodesConnectable={false}
              proOptions={{ hideAttribution: true }}
            >
              <Background gap={16} />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No memory graph yet — it fills in as the agent creates entities and relationships.
          </div>
        )}
      </CardContent>
    </Card>
  );
}
