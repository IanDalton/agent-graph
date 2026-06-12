import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";

import type { MemoryGraph } from "@/types";

export interface Point {
  x: number;
  y: number;
}

type SimNode = SimulationNodeDatum & { id: string };
type SimLink = SimulationLinkDatum<SimNode>;

/**
 * Run a force-directed layout to completion and return final coordinates keyed by node id.
 *
 * Graph DBs store no coordinates, so we let a physics simulation pull connected nodes together and
 * push unrelated ones apart — clusters emerge, replacing the old crossing-heavy circle. The sim is
 * ticked synchronously (`stop()` + a tick loop) so positions are ready in one pass with no render
 * loop fighting ReactFlow. Initial positions are seeded deterministically by index (no Math.random)
 * so a refresh of the same graph lands in the same place.
 */
export function computeLayout(graph: MemoryGraph): Map<string, Point> {
  const n = graph.nodes.length;
  // Seed on a ring so the simulation starts from a stable, deterministic arrangement.
  const nodes: SimNode[] = graph.nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / Math.max(n, 1);
    const r = Math.max(80, n * 12);
    return { id: node.id, x: r * Math.cos(angle), y: r * Math.sin(angle) };
  });
  const links: SimLink[] = graph.edges.map((e) => ({ source: e.source, target: e.target }));

  const sim = forceSimulation(nodes)
    .force("link", forceLink<SimNode, SimLink>(links).id((d) => d.id).distance(90).strength(0.6))
    .force("charge", forceManyBody().strength(-260))
    .force("center", forceCenter(0, 0))
    .force("collide", forceCollide(46))
    .stop();

  // ~300 ticks is plenty for graphs of this size (a few hundred nodes at most) to settle.
  for (let i = 0; i < 300; i++) sim.tick();

  const positions = new Map<string, Point>();
  for (const node of nodes) {
    positions.set(node.id, { x: node.x ?? 0, y: node.y ?? 0 });
  }
  return positions;
}

/**
 * Map a vertex type name to a stable, well-spread color. Same type → same hue across renders and
 * across the legend. A deterministic string hash picks a hue; saturation/lightness are fixed so the
 * palette stays readable on the dark card background.
 */
export function colorForType(type: string | null | undefined): string {
  if (!type) return "hsl(215 16% 55%)"; // muted fallback for legacy/untyped nodes
  let hash = 0;
  for (let i = 0; i < type.length; i++) {
    hash = (hash * 31 + type.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue} 65% 60%)`;
}

/** Undirected adjacency map (id -> set of neighbor ids), used to highlight a selected node's links. */
export function buildAdjacency(edges: MemoryGraph["edges"]): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>();
  const add = (a: string, b: string) => {
    if (!adj.has(a)) adj.set(a, new Set());
    adj.get(a)!.add(b);
  };
  for (const e of edges) {
    add(e.source, e.target);
    add(e.target, e.source);
  }
  return adj;
}
