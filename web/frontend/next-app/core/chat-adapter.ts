// ============================================================
// 旧版 → 新版 DAG / 聊天响应格式适配
//
// 将旧版 YiGraph 后端的 chat_response 事件映射到新版前端的
// WorkflowPlan / DagNode / DagEdge / SubProblem 等类型。
// ============================================================

import type {
  Algorithm,
  DagEdge,
  DagNode,
  SubProblem,
  WorkflowPlan,
  ReactRun,
} from "@/core/types"
import type {

  AlgorithmSelectedContent,
  ChatResponse,
  NodeExecutionContent,
  OldDag,
  OldDagEdge,
  OldDagNode,
  OldSubquery,
} from "@/core/api/chat"
import { matchAlgorithm, parseAlgorithmName } from "@/core/api/chat"
import { createReactRun, reduceReactEvent } from "./react-state"

/**
 * 将后端 tasktype 转为人可读的中文标签。
 */
export function taskTypeLabel(tasktype: string): string {
  if (!tasktype) return "Unknown Task / 未知任务"
  if (tasktype.includes("Numeric")) return "Numeric Analysis / 数值分析"
  if (tasktype.includes("Graph Query")) return "Graph Query / 图查询"
  if (tasktype.includes("Analysis Report")) return "Analysis Report / 分析报告"
  if (tasktype.includes("Graph Algorithm") && tasktype.includes("("))
    return tasktype
  if (tasktype.includes("Graph Algorithm")) return "Graph Algorithm / 图算法"
  if (tasktype.includes("Unknown")) return "Unknown Task / 未知任务"
  return tasktype
}

/**
 * 判断一个 tasktype 是否包含具体的算法名。
 */
export function hasAlgorithm(tasktype: string): boolean {
  return /\(([^)]+)\)/.test(tasktype)
}

// --- 适配结果 ---

/** 单个子问题选定的算法（key 为子问题原文） */
export interface StepAlgorithm {
  algorithm: string | null
  taskType: string
}

export interface AdapterState {
  executionMode?: "react" | "dag"
  react: ReactRun
  /** 累积的 thinking 文本 */
  thinkingText: string
  /** 当前 thinking 文本是否仅为占位提示（真实理解文本到达后将被替换） */
  thinkingIsPlaceholder: boolean
  /** 当前收集到的 DAG（可能为 null） */
  dag: OldDag | null
  /** 拆解出的子问题列表（subproblems 事件） */
  subqueries: OldSubquery[]
  /** 每个子问题选定的算法（algorithm_selected 事件，key 为子问题原文） */
  stepAlgorithms: Map<string, StepAlgorithm>
  /** 当前收集到的结果文本段落 */
  resultParagraphs: string[]
  /** 本轮是否收到过真实节点执行事件 */
  nodeEventsSeen: boolean
  /** 是否已经完成（收到 stream_end） */
  finished: boolean
  /** 错误信息 */
  error: string | null
}

export function createAdapterState(): AdapterState {
  return {
    react: createReactRun(),
    thinkingText: "",
    thinkingIsPlaceholder: false,
    dag: null,
    subqueries: [],
    stepAlgorithms: new Map(),
    resultParagraphs: [],
    nodeEventsSeen: false,
    finished: false,
    error: null,
  }
}

/**
 * 处理一条 chat_response 事件，更新适配状态。
 * 返回一个描述"发生了什么变化"的信号，供上层决定如何驱动 UI 阶段。
 */
export function processChatEvent(
  event: ChatResponse,
  state: AdapterState,
): AdapterEvent {
  if (event.error) {
    state.error = event.error
    state.finished = true
    return { kind: "error", message: event.error }
  }

  if (event.type === "stream_end") {
    state.finished = true
    return { kind: "stream_end" }
  }

  if (event.contentType?.startsWith("react_") ||
      (event.type === "thinking" && event.step !== undefined)) {
    state.executionMode = "react"
    state.react = reduceReactEvent(state.react, event)
    return { kind: "react", react: state.react }
  }

  if (event.type === "thinking" && event.contentType === "text") {
    const text = String(event.content ?? "")
    if (event.placeholder) {
      // 占位提示：暂存，等真实理解文本到达后替换
      state.thinkingText = text
      state.thinkingIsPlaceholder = true
    } else if (state.thinkingIsPlaceholder) {
      // 真实理解文本到达：替换占位提示
      state.thinkingText = text
      state.thinkingIsPlaceholder = false
    } else {
      state.thinkingText += text
    }
    return { kind: "thinking", text }
  }

  if (event.type === "result") {
    if (event.contentType === "dag") {
      state.dag = event.content as OldDag
      if (state.dag.execution_driver === "react") state.executionMode = "react"
      return { kind: "dag", dag: state.dag }
    }
    if (event.contentType === "subproblems") {
      const content = event.content as { subqueries?: OldSubquery[] } | undefined
      state.subqueries = content?.subqueries ?? []
      return { kind: "subproblems", subqueries: state.subqueries }
    }
    if (event.contentType === "algorithm_selected") {
      const selection = event.content as AlgorithmSelectedContent | undefined
      if (selection?.question) {
        state.stepAlgorithms.set(selection.question, {
          algorithm: selection.algorithm ?? null,
          taskType: selection.task_type ?? "",
        })
      }
      return {
        kind: "algorithm_selected",
        selection:
          selection ??
          ({ step_id: -1, question: "", task_type: "" } as AlgorithmSelectedContent),
      }
    }
    if (event.contentType === "node_status" || event.contentType === "node_result") {
      state.nodeEventsSeen = true
      return {
        kind: "node_update",
        content: (event.content ?? {
          step_id: -1,
          status: "failed",
        }) as NodeExecutionContent,
      }
    }
    if (event.contentType === "text") {
      const text = String(event.content ?? "")
      state.resultParagraphs.push(text)
      return { kind: "result_text", text }
    }
  }

  return { kind: "ignored" }
}

export type AdapterEvent =
  | { kind: "react"; react: ReactRun }
  | { kind: "thinking"; text: string }
  | { kind: "dag"; dag: OldDag }
  | { kind: "subproblems"; subqueries: OldSubquery[] }
  | { kind: "algorithm_selected"; selection: AlgorithmSelectedContent }
  | { kind: "node_update"; content: NodeExecutionContent }
  | { kind: "result_text"; text: string }
  | { kind: "stream_end" }
  | { kind: "error"; message: string }
  | { kind: "ignored" }

// --- DAG 格式转换 ---

/**
 * 将旧版 DAG 转换为新版 WorkflowPlan 所需的核心数据。
 * 只保留后端返回的子任务节点，不添加预处理/汇总节点。
 * 优先使用 algorithm_selected 事件累积的真实算法 id，其次才从 tasktype 解析。
 */
export function convertOldDagToPlan(
  dag: OldDag,
  understanding: string,
  algorithms: Algorithm[],
  stepAlgorithms?: Map<string, StepAlgorithm>,
): {
  understanding: string
  subProblems: SubProblem[]
  nodes: DagNode[]
  edges: DagEdge[]
  /** 后端 step_id → 前端节点 id（n-{i}），用于把执行事件映射到节点 */
  idMap: Map<string, string>
} {
  const subProblems: SubProblem[] = []
  const nodes: DagNode[] = []
  const edges: DagEdge[] = []

  const dalgNodes = dag.nodes ?? []
  const dalgEdges = dag.edges ?? []

  // 跳过可能是"用户问题描述"的首节点
  const skipFirst =
    dag.execution_driver !== "react" &&
    dalgNodes.length > 1 &&
    (dalgNodes[0].label.includes("用户") ||
      dalgNodes[0].label.includes("问题") ||
      dalgNodes[0].label.includes("分析文档"))
  const rootId = skipFirst ? dalgNodes[0].id : null
  const startIdx = skipFirst ? 1 : 0
  const taskNodes = dalgNodes.slice(startIdx)

  // 旧 ID → 新 ID 映射
  const idMap = new Map<string, string>()

  // 只创建子任务节点
  taskNodes.forEach((on: OldDagNode, i: number) => {
    const nodeId = `n-${i + 1}`
    idMap.set(on.id, nodeId)

    // 优先用 algorithm_selected 事件里的真实算法 id
    const selected = stepAlgorithms?.get(on.label)
    const algoName = on.algorithm ?? selected?.algorithm ?? parseAlgorithmName(on.tasktype)
    const algo = algoName ? matchAlgorithm(algoName, algorithms) : undefined
    const taskLabel =
      selected?.taskType === "graph_query"
        ? "Graph Query / 图查询"
        : taskTypeLabel(on.tasktype)

    const sp: SubProblem = {
      id: `sp-${i + 1}`,
      title: on.label,
      rationale: algo
        ? `${algo.displayName}（${algo.name}）：${algo.summary}`
        : taskLabel,
      algorithmId: algo?.id ?? "",
    }

    subProblems.push(sp)

    nodes.push({
      id: nodeId,
      label: on.label,
      algorithmId: algo?.id ?? "",
      subProblemId: sp.id,
      status: on.status === "success" ? "done" : on.status === "failed" ? "error"
        : on.status === "running" ? "running" : "pending",
      taskType: algo ? undefined : taskLabel,
      toolName: on.algorithm || undefined,
      executionRecord: on.record,
      result: on.summary ? { summary: on.summary } : undefined,
      error: on.error || undefined,
    })
  })

  // 保留后端返回的边结构，只映射节点 ID
  dalgEdges.forEach((e, i) => {
    const sourceId = idMap.get(e.from)
    const targetId = idMap.get(e.to)
    // 只保留两端都存在的边（排除涉及根节点或缺失节点的边）
    if (sourceId && targetId) {
      edges.push({ id: `e-${i}`, source: sourceId, target: targetId, kind: e.kind })
    } else if (!sourceId && targetId && e.from === rootId) {
      // 从根节点出发到子任务节点的边：忽略（根节点不展示）
    }
  })

  // 如果后端没给边，按节点顺序补一条链
  if (dag.execution_driver !== "react" && edges.length === 0 && taskNodes.length > 1) {
    for (let i = 0; i < taskNodes.length - 1; i++) {
      edges.push({
        id: `e-chain-${i}`,
        source: `n-${i + 1}`,
        target: `n-${i + 2}`,
      })
    }
  }

  return {
    understanding:
      understanding ||
      `用户发起了图分析请求。后端已解析问题并构建了 ${subProblems.length} 个子任务的工作流。`,
    subProblems,
    nodes,
    edges,
    idMap,
  }
}
