import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
  type NodeChange,
} from "reactflow";
import "reactflow/dist/style.css";
import { Search } from "lucide-react";

import { Select } from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { GraphNode, MemoryGraph } from "@/types";
import { buildAdjacency, colorForType, computeLayout, type Point } from "./layout";

type KindFilter = "all" | "semantic" | "episodic";

/** null kind is rendered as semantic (matches the backend's legacy/untyped convention). */
const kindOf = (node: GraphNode): "semantic" | "episodic" =>
  node.kind === "episodic" ? "episodic" : "semantic";

/**
 * The shared interactive knowledge-graph canvas, used both as a compact right-pane thumbnail
 * (`compact`) and as the full-screen explorer. Force-directed layout (see ./layout), drag to
 * reposition, color-by-type, search + type/kind filters, and click-to-inspect with neighbor
 * highlighting. Read-only with respect to the data — it never mutates the graph.
 */
export function MemoryGraphView({
  graph,
  compact = false,
}: {
  graph: MemoryGraph;
  compact?: boolean;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<KindFilter>("all");
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set());
  // User drags override the computed layout for the session.
  const [overrides, setOverrides] = useState<Map<string, Point>>(new Map());

  // A new graph invalidates positions, drags, and selection.
  useEffect(() => {
    setOverrides(new Map());
    setSelectedId(null);
  }, [graph]);

  const positions = useMemo(() => computeLayout(graph), [graph]);
  const adjacency = useMemo(() => buildAdjacency(graph.edges), [graph.edges]);
  const types = useMemo(
    () => Array.from(new Set(graph.nodes.map((n) => n.type))).sort(),
    [graph.nodes]
  );

  const isVisible = useCallback(
    (node: GraphNode) =>
      !hiddenTypes.has(node.type) &&
      (kindFilter === "all" || kindOf(node) === kindFilter),
    [hiddenTypes, kindFilter]
  );

  const matchesQuery = useCallback(
    (node: GraphNode) => {
      const q = query.trim().toLowerCase();
      if (!q) return true;
      return node.label.toLowerCase().includes(q) || node.type.toLowerCase().includes(q);
    },
    [query]
  );

  const isHighlighted = useCallback(
    (id: string, node: GraphNode) => {
      if (selectedId) return id === selectedId || !!adjacency.get(selectedId)?.has(id);
      return matchesQuery(node);
    },
    [selectedId, adjacency, matchesQuery]
  );

  const rfNodes: Node[] = useMemo(
    () =>
      graph.nodes
        .filter(isVisible)
        .map((node) => {
          const pos = overrides.get(node.id) ?? positions.get(node.id) ?? { x: 0, y: 0 };
          const color = colorForType(node.type);
          const episodic = kindOf(node) === "episodic";
          const selected = node.id === selectedId;
          const dimmed = !isHighlighted(node.id, node);
          return {
            id: node.id,
            position: pos,
            data: {
              color,
              label: (
                <div className="leading-tight">
                  <div className="font-medium">{node.label}</div>
                  {!compact && <div className="text-[9px] opacity-70">{node.type}</div>}
                </div>
              ),
            },
            style: {
              fontSize: 11,
              borderRadius: 8,
              padding: "6px 10px",
              minWidth: 40,
              border: `${episodic ? "1.5px dashed" : "2px solid"} ${color}`,
              background: "hsl(var(--card))",
              color: "hsl(var(--foreground))",
              opacity: dimmed ? 0.2 : 1,
              boxShadow: selected ? `0 0 0 2px ${color}` : undefined,
              transition: "opacity 150ms ease",
            },
          } satisfies Node;
        }),
    [graph.nodes, isVisible, overrides, positions, selectedId, isHighlighted, compact]
  );

  const rfEdges: Edge[] = useMemo(() => {
    const visibleIds = new Set(rfNodes.map((n) => n.id));
    return graph.edges
      .filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target))
      .map((e) => {
        const active = selectedId
          ? e.source === selectedId || e.target === selectedId
          : !query.trim() ||
            (visibleIds.has(e.source) && visibleIds.has(e.target));
        const dimmed = selectedId ? !active : false;
        return {
          id: e.id,
          source: e.source,
          target: e.target,
          label: e.label,
          animated: !compact && active && !!selectedId,
          labelStyle: { fontSize: 10, fill: "hsl(var(--foreground))" },
          labelBgStyle: { fill: "hsl(var(--card))", fillOpacity: 0.85 },
          labelBgPadding: [3, 2] as [number, number],
          labelBgBorderRadius: 4,
          style: {
            stroke: active ? "hsl(var(--primary))" : "hsl(var(--muted-foreground))",
            strokeWidth: active && selectedId ? 2 : 1,
            opacity: dimmed ? 0.15 : 1,
          },
        } satisfies Edge;
      });
  }, [graph.edges, rfNodes, selectedId, query, compact]);

  const onNodesChange = useCallback((changes: NodeChange[]) => {
    setOverrides((prev) => {
      let next = prev;
      for (const c of changes) {
        if (c.type === "position" && c.position) {
          if (next === prev) next = new Map(prev);
          next.set(c.id, c.position);
        }
      }
      return next;
    });
  }, []);

  const selectedNode = useMemo(
    () => graph.nodes.find((n) => n.id === selectedId) ?? null,
    [graph.nodes, selectedId]
  );

  const toggleType = (type: string) =>
    setHiddenTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });

  const flow = (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      onNodesChange={compact ? undefined : onNodesChange}
      onNodeClick={compact ? undefined : (_, n) => setSelectedId(n.id)}
      onPaneClick={compact ? undefined : () => setSelectedId(null)}
      fitView
      fitViewOptions={{ padding: 0.2 }}
      minZoom={0.1}
      nodesDraggable={!compact}
      nodesConnectable={false}
      elementsSelectable={!compact}
      zoomOnScroll={!compact}
      panOnDrag={!compact}
      zoomOnDoubleClick={!compact}
      proOptions={{ hideAttribution: true }}
    >
      <Background gap={16} />
      {!compact && (
        <>
          <Controls showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            nodeColor={(n) => (n.data?.color as string) ?? "hsl(var(--muted-foreground))"}
            maskColor="hsl(var(--background) / 0.6)"
            className="!bg-card"
          />
        </>
      )}
    </ReactFlow>
  );

  if (compact) {
    return <div className="size-full">{flow}</div>;
  }

  return (
    <div className="flex h-full flex-col">
      {/* Toolbar: search + kind filter */}
      <div className="flex flex-wrap items-center gap-2 border-b border-border p-2">
        <div className="relative flex-1 min-w-[140px]">
          <Search className="pointer-events-none absolute left-2 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              // A selected node would otherwise win the highlight and hide the search result.
              if (e.target.value) setSelectedId(null);
            }}
            placeholder="Search nodes…"
            className="h-8 w-full rounded-md border border-input bg-background pl-7 pr-2 text-xs shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          />
        </div>
        <Select
          value={kindFilter}
          onChange={(e) => setKindFilter(e.target.value as KindFilter)}
          className="h-8 w-auto"
          title="Filter by memory kind"
        >
          <option value="all">All kinds</option>
          <option value="semantic">Semantic</option>
          <option value="episodic">Episodic</option>
        </Select>
      </div>

      {/* Type legend / filter chips */}
      {types.length > 0 && (
        <div className="flex flex-wrap gap-1.5 border-b border-border px-2 py-1.5">
          {types.map((t) => {
            const off = hiddenTypes.has(t);
            const color = colorForType(t);
            return (
              <button
                key={t}
                type="button"
                onClick={() => toggleType(t)}
                title={off ? `Show ${t}` : `Hide ${t}`}
                className={cn(
                  "flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] transition-opacity",
                  off ? "opacity-40" : "opacity-100"
                )}
                style={{ borderColor: color }}
              >
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ background: color }}
                />
                {t}
              </button>
            );
          })}
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <div className="relative min-w-0 flex-1">{flow}</div>
        {selectedNode && (
          <div className="w-72 shrink-0 border-l border-border">
            <NodeDetails node={selectedNode} onClose={() => setSelectedId(null)} />
          </div>
        )}
      </div>
    </div>
  );
}

/** Right-side inspector for the selected node: type, kind, and every scalar property. */
function NodeDetails({ node, onClose }: { node: GraphNode; onClose: () => void }) {
  const color = colorForType(node.type);
  const entries = Object.entries(node.properties);
  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="break-words text-sm font-medium">{node.label}</div>
            <div className="mt-1 flex items-center gap-1.5">
              <span className="inline-block size-2 rounded-full" style={{ background: color }} />
              <span className="font-mono text-[11px] text-muted-foreground">{node.type}</span>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 text-xs text-muted-foreground hover:text-foreground"
          >
            clear
          </button>
        </div>

        <div className="text-[11px] text-muted-foreground">
          {kindOf(node) === "episodic" ? "Episodic (event)" : "Semantic (state)"}
        </div>

        <div className="space-y-1.5">
          {entries.length === 0 ? (
            <div className="text-xs text-muted-foreground">No properties.</div>
          ) : (
            entries.map(([k, v]) => (
              <div key={k} className="text-xs">
                <div className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                  {k}
                </div>
                <div className="break-words">{formatValue(v)}</div>
              </div>
            ))
          )}
        </div>
      </div>
    </ScrollArea>
  );
}

function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
