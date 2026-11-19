import type { DagEdge, DagNode } from "./types"

export interface DagPosition { x: number; y: number; h: number }

/** Arrange the recorded tasks by dependency depth; independent branches share a column. */
export function layoutDagNodes(nodes: DagNode[], edges: DagEdge[], heights: Record<string, number>) {
  const children = new Map(nodes.map(node => [node.id, new Set<string>()]))
  const incoming = new Map(nodes.map(node => [node.id, 0]))
  const depth = new Map(nodes.map(node => [node.id, 0]))
  for (const edge of edges) {
    if (edge.source === edge.target || !children.has(edge.source) || !incoming.has(edge.target)) continue
    const next = children.get(edge.source)!
    if (!next.has(edge.target)) {
      next.add(edge.target)
      incoming.set(edge.target, incoming.get(edge.target)! + 1)
    }
  }
  const ready = nodes.filter(node => incoming.get(node.id) === 0).map(node => node.id)
  for (let index = 0; index < ready.length; index++) {
    const id = ready[index]
    for (const child of children.get(id)!) {
      depth.set(child, Math.max(depth.get(child)!, depth.get(id)! + 1))
      incoming.set(child, incoming.get(child)! - 1)
      if (incoming.get(child) === 0) ready.push(child)
    }
  }
  // Old saved records may be malformed; keep them visible without blocking rendering.
  const unresolvedColumn = Math.max(0, ...depth.values()) + 1
  const positions: Record<string, DagPosition> = {}
  const nextY = new Map<number, number>()
  for (const node of nodes) {
    const column = incoming.get(node.id)! > 0 ? unresolvedColumn : depth.get(node.id)!
    const h = heights[node.id] ?? 126
    const y = nextY.get(column) ?? 20
    positions[node.id] = { x: 20 + column * 260, y, h }
    nextY.set(column, y + h + 32)
  }
  return positions
}
