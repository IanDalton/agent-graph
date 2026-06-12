import { useEffect, useState } from "react";
import { Maximize2, Network, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/api/client";
import { useApp } from "@/state/AppContext";
import type { MemoryGraph } from "@/types";
import { MemoryGraphView } from "@/panes/graph/MemoryGraphView";

/** Right-pane card rendering the user's agent-built knowledge graph. Read-only; refreshes when the
 *  active conversation changes or a turn completes (refreshKey), mirroring SummaryCard. The card
 *  shows a compact preview; the expand button opens a full-screen interactive explorer. */
export function MemoryGraphCard({ refreshKey }: { refreshKey: number }) {
  const { userId } = useApp();
  const [graph, setGraph] = useState<MemoryGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

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
        <div className="flex items-center gap-0.5">
          <Button
            variant="ghost"
            size="icon"
            className="size-6 text-muted-foreground"
            onClick={() => setExpanded(true)}
            disabled={!hasNodes}
            title="Expand graph"
          >
            <Maximize2 className="size-3.5" />
          </Button>
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
        </div>
      </CardHeader>
      <CardContent>
        {loading && !graph ? (
          <Skeleton className="h-[260px] w-full" />
        ) : hasNodes ? (
          <button
            type="button"
            onClick={() => setExpanded(true)}
            title="Expand graph"
            className="block h-[260px] w-full overflow-hidden rounded-md border border-border"
          >
            {/* pointer-events-none: the preview is a single click target that opens the explorer. */}
            <div className="pointer-events-none size-full">
              <MemoryGraphView graph={graph!} compact />
            </div>
          </button>
        ) : (
          <div className="text-xs text-muted-foreground">
            No memory graph yet — it fills in as the agent creates entities and relationships.
          </div>
        )}
      </CardContent>

      <Dialog
        open={expanded && hasNodes}
        onClose={() => setExpanded(false)}
        title="Memory Graph"
        className="h-[85vh] w-[90vw] max-w-[1400px]"
      >
        <div className="flex items-center gap-1.5 border-b border-border px-4 py-2 pr-12 text-sm font-medium">
          <Network className="size-4 text-muted-foreground" />
          Memory Graph
        </div>
        <div className="min-h-0 flex-1">
          {graph && <MemoryGraphView graph={graph} />}
        </div>
      </Dialog>
    </Card>
  );
}
