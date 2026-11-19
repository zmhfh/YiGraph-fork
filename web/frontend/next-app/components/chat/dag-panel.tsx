"use client"

import * as React from "react"
import {
  Workflow,
  Info,
  CheckCircle2,
  Loader2,
  Circle,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  XCircle,
} from "lucide-react"
import type { DagEdge, DagNode, WorkflowPlan } from "@/core/types"
import { useI18n } from "@/core/i18n/i18n-provider"
import { useAlgorithms } from "@/core/algorithms-store"
import { submitFeedback } from "@/core/api/feedback"
import { Card } from "@/components/ui/card"
import { cn } from "@/core/utils"
import { AlgorithmInfoModal } from "./algorithm-info-modal"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { layoutDagNodes } from "@/core/dag-layout"
export function DagPanel({
  plan,
  userQuestion,
  onLayoutChange,
}: {
  plan: WorkflowPlan
  userQuestion: string
  onLayoutChange: (layout: Record<string, { x: number; y: number }>) => void
}) {
  const { t } = useI18n()
  const [infoId, setInfoId] = React.useState<string | null>(null)
  const [feedback, setFeedback] = React.useState<"like" | "dislike" | null>(null)
  const [detailId, setDetailId] = React.useState<string | null>(null)
  const detailNode = plan.nodes.find(node => node.id === detailId)
  const layout = plan.layout ?? {}
  const hasCustomLayout = Object.keys(layout).length > 0

  const handleFeedback = (type: "like" | "dislike") => {
    if (feedback) return
    setFeedback(type)
    const dagSummary = JSON.stringify({
      nodes: plan.nodes.map((n) => ({ id: n.id, label: n.label, algorithmId: n.algorithmId })),
      edges: plan.edges.map((e) => ({ source: e.source, target: e.target })),
    })
    submitFeedback({
      userQuestion,
      analysisOutput: dagSummary,
      type: "dag",
      feedback: type,
    }).catch(() => {
      setFeedback(null)
    })
  }

  return (
    <Card role="region" aria-label={t("react.recordGraph", "执行记录图")} className="overflow-hidden p-0">
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-2">
          <Workflow className="size-4 text-primary" />
          <span className="text-sm font-semibold">{t("react.recordGraph", "执行记录图")}</span>
          <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
            {plan.nodes.length} {t("dag.nodes", "节点")} · {plan.edges.length} {t("dag.edges", "边")}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Info className="size-3" />
          {plan.executionMode === "react" ? t("react.taskGraphHint", "节点代表分析任务，可查看执行详情") : t("dag.inspectHint", "点击算法节点查看详情")}
          {hasCustomLayout && (
            <>
              <span className="mx-1 text-border">|</span>
              <button
                onClick={() => onLayoutChange({})}
                className="flex items-center gap-1 rounded px-1 py-0.5 transition-colors hover:bg-muted hover:text-foreground"
              >
                <RotateCcw className="size-3" />
                {t("dag.resetLayout", "重置布局")}
              </button>
            </>
          )}
          <span className="mx-1 text-border">|</span>
          <button
            onClick={() => handleFeedback("like")}
            disabled={feedback !== null}
            className={cn(
              "rounded p-0.5 transition-colors",
              feedback === "like"
                ? "text-primary"
                : "text-muted-foreground hover:text-primary",
              feedback !== null && feedback !== "like" && "opacity-40",
            )}
          >
            <ThumbsUp className={cn("size-3", feedback === "like" && "fill-primary")} />
          </button>
          <button
            onClick={() => handleFeedback("dislike")}
            disabled={feedback !== null}
            className={cn(
              "rounded p-0.5 transition-colors",
              feedback === "dislike"
                ? "text-destructive"
                : "text-muted-foreground hover:text-destructive",
              feedback !== null && feedback !== "dislike" && "opacity-40",
            )}
          >
            <ThumbsDown className={cn("size-3", feedback === "dislike" && "fill-destructive")} />
          </button>
        </div>
      </div>

      <div className="relative">
        <DagCanvas
          plan={plan}
          onInspect={setInfoId}
          onInspectTask={setDetailId}
          onMoveNode={(nodeId, pos) =>
            onLayoutChange({ ...layout, [nodeId]: pos })
          }
        />
      </div>
      {plan.executionMode === "react" && <p className="border-t border-border px-5 py-2 text-[11px] text-muted-foreground">
        {t("react.taskEdgeLegend", "实线：结果引用或汇总 · 虚线：执行顺序")}
      </p>}

      <AlgorithmInfoModal
        algorithmId={infoId}
        onClose={() => setInfoId(null)}
      />
      <Dialog open={!!detailNode} onOpenChange={open => { if (!open) setDetailId(null) }}>
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle className="pr-6 text-base leading-6">{detailNode?.label}</DialogTitle>
            <DialogDescription>{detailNode?.taskType || detailNode?.toolName} · {t(`react.taskStatus.${detailNode?.status}`, detailNode?.status ?? "")}</DialogDescription>
          </DialogHeader>
          {detailNode?.result?.summary && <p className="whitespace-pre-wrap break-words text-sm leading-6">{detailNode.result.summary}</p>}
          {detailNode?.error && <p className="break-words text-sm text-destructive">{detailNode.error}</p>}
          <details className="rounded-lg border border-border p-3 text-sm">
            <summary className="cursor-pointer font-medium">{t("react.taskRecord", "查看输入、调用与观测记录")}</summary>
            <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(detailNode?.executionRecord, null, 2)}</pre>
          </details>
        </DialogContent>
      </Dialog>

    </Card>
  )
}

/* -------------------------------- Canvas --------------------------------- */

interface Pos {
  x: number
  y: number
  h: number
}

const NODE_W = 200
const NODE_MIN_H = 78
const CANVAS_MIN_W = 400
const CANVAS_PAD = 20
/** 超过该位移才判定为拖动，避免和"点击算法详情"冲突 */
const DRAG_THRESHOLD_PX = 4
interface DragSession {
  id: string
  pointerId: number
  startX: number
  startY: number
  originX: number
  originY: number
  moved: boolean
}

function DagCanvas({
  plan,
  onInspect,
  onInspectTask,
  onMoveNode,
}: {
  plan: WorkflowPlan
  onInspect: (algoId: string) => void
  onInspectTask: (nodeId: string) => void
  onMoveNode: (nodeId: string, pos: { x: number; y: number }) => void
}) {
  const { getById } = useAlgorithms()
  const [measuredHeights, setMeasuredHeights] = React.useState<
    Record<string, number>
  >({})

  const handleHeightChange = React.useCallback((nodeId: string, height: number) => {
    setMeasuredHeights((current) =>
      current[nodeId] === height
        ? current
        : { ...current, [nodeId]: height },
    )
  }, [])

  // 自动布局（按拓扑分列）；首次用估算值，挂载后改用节点的真实高度。
  const basePositions = React.useMemo(
    () => layoutDagNodes(plan.nodes, plan.edges, measuredHeights),
    [plan.nodes, plan.edges, measuredHeights],
  )

  // 拖动预览：拖动过程中只更新本地坐标，松手后才写回 plan.layout
  const [drag, setDrag] = React.useState<{
    id: string
    x: number
    y: number
  } | null>(null)
  const dragRef = React.useRef<DragSession | null>(null)
  const dragPosRef = React.useRef<{ id: string; x: number; y: number } | null>(null)
  const rafRef = React.useRef<number | null>(null)
  const suppressClickRef = React.useRef(false)

  // plan.layout（已保存的手动位置）覆盖自动布局，拖动预览再覆盖两者
  const positions = React.useMemo(() => {
    const next: Record<string, Pos> = {}
    for (const [id, p] of Object.entries(basePositions)) {
      const saved = plan.layout?.[id]
      next[id] = saved ? { ...p, x: saved.x, y: saved.y } : p
    }
    if (drag && next[drag.id]) {
      next[drag.id] = { ...next[drag.id], x: drag.x, y: drag.y }
    }
    return next
  }, [basePositions, plan.layout, drag])

  // 卸载时取消挂起的帧回调
  React.useEffect(
    () => () => {
      if (rafRef.current !== null) window.cancelAnimationFrame(rafRef.current)
    },
    [],
  )

  const handlePointerDown =
    (nodeId: string) => (e: React.PointerEvent<HTMLDivElement>) => {
      if (e.pointerType === "mouse" && e.button !== 0) return
      // 节点内的链接/按钮保持原生点击行为，不作为拖动把手
      const target = e.target as HTMLElement | null
      if (target?.closest("button, a, input, textarea, select")) return
      const pos = positions[nodeId]
      if (!pos) return
      e.currentTarget.setPointerCapture(e.pointerId)
      dragRef.current = {
        id: nodeId,
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        originX: pos.x,
        originY: pos.y,
        moved: false,
      }
      suppressClickRef.current = false
    }

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const session = dragRef.current
    if (!session || session.pointerId !== e.pointerId) return
    const dx = e.clientX - session.startX
    const dy = e.clientY - session.startY
    if (!session.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) return
    session.moved = true
    dragPosRef.current = {
      id: session.id,
      x: Math.max(0, session.originX + dx),
      y: Math.max(0, session.originY + dy),
    }
    // 每帧最多提交一次，避免 pointermove 高频重渲染
    if (rafRef.current === null) {
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null
        setDrag(dragPosRef.current)
      })
    }
  }

  const handlePointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    const session = dragRef.current
    if (!session || session.pointerId !== e.pointerId) return
    dragRef.current = null
    if (rafRef.current !== null) {
      window.cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    const finalPos = dragPosRef.current
    dragPosRef.current = null
    setDrag(null)
    if (!session.moved) {
      // 未超过阈值：保持普通点击行为
      suppressClickRef.current = false
      return
    }
    // 拖动结束后抑制随之而来的 click，避免误开算法详情弹窗
    suppressClickRef.current = true
    if (finalPos) onMoveNode(session.id, { x: finalPos.x, y: finalPos.y })
  }

  const handleClickCapture = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!suppressClickRef.current) return
    suppressClickRef.current = false
    e.preventDefault()
    e.stopPropagation()
  }

  // 根据实际布局（含拖动预览）计算画布尺寸，拖出右下角时画布随之扩展
  const allPos = Object.values(positions)
  const width =
    Math.max(...allPos.map((p) => p.x + NODE_W), CANVAS_MIN_W) + CANVAS_PAD
  const height =
    Math.max(...allPos.map((p) => p.y + p.h), 0) + CANVAS_PAD

  return (
    <div className="bg-dots w-full overflow-auto">
      <div
        className="relative"
        style={{ width: `${width}px`, height: `${height}px` }}
      >
        {/* Edges */}
        <svg
          className="absolute inset-0"
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          aria-hidden="true"
        >
          <defs>
            <marker
              id="arrow"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 z" className="fill-border" />
            </marker>
            <marker
              id="arrow-active"
              markerWidth="10"
              markerHeight="10"
              refX="9"
              refY="5"
              orient="auto"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--primary)" />
            </marker>
          </defs>
          {plan.edges.map((edge) => (
            <EdgePath
              key={edge.id}
              edge={edge}
              nodes={plan.nodes}
              positions={positions}
            />
          ))}
        </svg>

        {/* Nodes */}
        {plan.nodes.map((node) => {
          const pos = positions[node.id]
          if (!pos) return null
          const algo = getById(node.algorithmId)
          const noAlgo = !node.algorithmId
          return (
            <DagNodeBox
              key={node.id}
              node={node}
              algoName={algo?.name ?? (noAlgo ? "" : node.algorithmId)}
              algoDisplayName={
                algo?.displayName ??
                (noAlgo ? (node.taskType ?? "") : node.algorithmId)
              }
              noAlgorithm={noAlgo}
              x={pos.x}
              y={pos.y}
              dragging={drag?.id === node.id}
              onPointerDown={handlePointerDown(node.id)}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              onClickCapture={handleClickCapture}
              onHeightChange={handleHeightChange}
              onInspect={() => onInspect(node.algorithmId)}
              onInspectTask={() => onInspectTask(node.id)}
            />
          )
        })}
      </div>
    </div>
  )
}

function EdgePath({
  edge,
  nodes,
  positions,
}: {
  edge: DagEdge
  nodes: DagNode[]
  positions: Record<string, Pos>
}) {
  const from = positions[edge.source]
  const to = positions[edge.target]
  if (!from || !to) return null
  const sourceNode = nodes.find((n) => n.id === edge.source)
  const targetNode = nodes.find((n) => n.id === edge.target)
  const x1 = from.x + NODE_W
  const y1 = from.y + from.h / 2
  const x2 = to.x
  const y2 = to.y + to.h / 2
  const midX = (x1 + x2) / 2
  const d = `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`

  const isFlowing =
    sourceNode?.status === "done" && targetNode?.status === "running"
  const isActive =
    sourceNode?.status === "done" || sourceNode?.status === "running"

  return (
    <path
      d={d}
      fill="none"
      className={cn(
        isFlowing
          ? "stroke-primary edge-flow"
          : isActive
            ? "stroke-primary/60"
            : "stroke-border",
      )}
      strokeWidth={1.8}
      strokeDasharray={edge.kind === "sequence" ? "5 4" : undefined}
      markerEnd={isActive ? "url(#arrow-active)" : "url(#arrow)"}
    ><title>{edge.kind === "sequence" ? "执行顺序" : edge.kind === "summary" ? "汇总结果" : "结果引用"}</title></path>
  )
}

function DagNodeBox({
  node,
  algoName,
  algoDisplayName,
  noAlgorithm,
  x,
  y,
  dragging,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onPointerCancel,
  onClickCapture,
  onHeightChange,
  onInspect,
  onInspectTask,
}: {
  node: DagNode
  algoName: string
  algoDisplayName: string
  noAlgorithm?: boolean
  x: number
  y: number
  dragging?: boolean
  onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void
  onPointerMove: (e: React.PointerEvent<HTMLDivElement>) => void
  onPointerUp: (e: React.PointerEvent<HTMLDivElement>) => void
  onPointerCancel: (e: React.PointerEvent<HTMLDivElement>) => void
  onClickCapture: (e: React.MouseEvent<HTMLDivElement>) => void
  onHeightChange: (nodeId: string, height: number) => void
  onInspect: () => void
  onInspectTask: () => void
}) {
  const { t } = useI18n()
  const nodeRef = React.useRef<HTMLDivElement>(null)

  React.useLayoutEffect(() => {
    const element = nodeRef.current
    if (!element) return

    const reportHeight = () => onHeightChange(node.id, element.offsetHeight)
    reportHeight()

    const observer = new ResizeObserver(reportHeight)
    observer.observe(element)
    return () => observer.disconnect()
  }, [node.id, onHeightChange])

  const roleLabel = t("dag.subtask", "子任务")
  const statusBg =
    node.status === "done"
      ? "border-accent/40 bg-accent/10"
      : node.status === "running"
        ? "border-primary/50 bg-primary/5"
        : node.status === "error"
          ? "border-destructive/50 bg-destructive/10"
          : "border-border bg-card"
  const Icon =
    node.status === "done"
      ? CheckCircle2
      : node.status === "running"
        ? Loader2
        : node.status === "error" ? XCircle : Circle
  const iconColor =
    node.status === "done"
      ? "text-accent"
      : node.status === "running"
        ? "text-primary"
        : node.status === "error" ? "text-destructive" : "text-muted-foreground"

  return (
    <div
      ref={nodeRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerCancel}
      onClickCapture={onClickCapture}
      className={cn(
        "absolute flex touch-none select-none flex-col rounded-lg border px-3 py-2 shadow-sm transition-colors",
        statusBg,
        dragging
          ? "z-20 cursor-grabbing shadow-md ring-2 ring-primary/30"
          : "cursor-grab",
      )}
      style={{
        left: `${x}px`,
        top: `${y}px`,
        width: `${NODE_W}px`,
        minHeight: `${NODE_MIN_H}px`,
      }}
      title={t("dag.dragNode", "拖动节点调整布局")}
    >
      <div className="flex items-center gap-2">
        <Icon
          className={cn(
            "size-3.5 shrink-0",
            iconColor,
            node.status === "running" && "animate-spin",
          )}
        />
        <span className="truncate text-[11px] font-medium text-muted-foreground">
          {roleLabel}
        </span>
        <span className="ml-auto rounded bg-muted px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
          {node.id}
        </span>
      </div>
      <div title={node.label} className="mt-1 line-clamp-3 break-words text-sm font-semibold leading-snug text-foreground">
        {node.label}
      </div>
      {noAlgorithm ? (
        <span className="mt-auto text-[11px] text-muted-foreground leading-tight">
          {node.toolName ? `${algoDisplayName} · ${node.toolName}` : algoDisplayName || t("dag.noAlgoTask", "—")}
        </span>
      ) : (
        <button
          onClick={onInspect}
          className="mt-auto flex items-start gap-1 self-start rounded text-[11px] text-primary underline-offset-2 hover:underline text-left"
        >
          <span className="font-mono break-all leading-tight">{algoName}</span>
          <span className="text-muted-foreground break-words leading-tight">· {algoDisplayName}</span>
        </button>
      )}
      {node.executionRecord && <button onClick={onInspectTask} className="mt-2 self-start text-xs text-primary underline-offset-2 hover:underline">
        {t("react.taskDetails", "查看节点详情")}
      </button>}
    </div>
  )
}
