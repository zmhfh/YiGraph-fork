// ============================================================
// 智能分析聊天客户端（HTTP 流式传输）
//
// POST /api/chat/stream，响应为 NDJSON 流（每行一个事件 JSON），
// 事件格式与旧版后端的 chat_response 协议一致。
// 断线后自动带 resume 重新请求，后端重放缓冲事件、续接新事件。
// 后端不可用时请求失败，前端提示用户。
// ============================================================

import type { Algorithm } from "@/core/types"
import { getBaseUrl, isBackendLive } from "./base"

// --- 后端 chat_response 事件格式 ---

export interface ChatResponse {
  type: "thinking" | "result" | "stream_end"
  contentType?:
    | "text"
    | "dag"
    | "code"
    | "subproblems"
    | "algorithm_selected"
    | "node_status"
    | "node_result"
    | "react_status"
    | "react_thought"
    | "react_action"
    | "react_observation"
    | "react_artifact"
  content?: unknown
  step?: number
  event_id?: string
  thought?: string
  checklist?: string[]
  error?: string
  /** 标记为占位提示（如开头的 "Analyzing your question..."），真实文本到达后应被替换 */
  placeholder?: boolean
  /** 同一请求内单调递增的事件序号（重放去重用） */
  seq?: number
}

export interface OldDagNode {
  id: string
  label: string
  tasktype: string
  status?: "pending" | "running" | "success" | "failed"
  algorithm?: string | null
  record_type?: string
  record?: Record<string, unknown>
  summary?: string
  error?: string | null
}

export interface OldDagEdge {
  from: string
  to: string
  kind?: "data" | "sequence" | "summary"
}

export interface OldDag {
  nodes: OldDagNode[]
  edges: OldDagEdge[]
  execution_driver?: "react" | "dag"
}

/** subproblems 事件：拆解出的子问题列表（与后端 subquery_plan 一致） */
export interface OldSubquery {
  id: string
  query: string
  depends_on: string[]
}

export interface SubproblemsContent {
  subqueries: OldSubquery[]
}

/** algorithm_selected 事件：单个子问题选定的算法 */
export interface AlgorithmSelectedContent {
  step_id: number
  question: string
  task_type: string
  algorithm?: string | null
}

/** node_status / node_result 事件：单节点执行状态与结果（真实后端执行时下发） */
export interface NodeExecutionContent {
  /** 后端 step_id，经 DAG 事件的 idMap 映射到前端节点 id */
  step_id: number
  status: "running" | "success" | "failed"
  question?: string
  task_type?: string
  algorithm?: string | null
  /** 失败原因（status === "failed" 时） */
  error?: string
  /** 仅 node_result 携带：截断后的结构化输出 */
  outputs?: NodeOutputPayload[]
}

/** 后端单个 StepOutputItem 的原始（已截断）视图 */
export interface NodeOutputPayload {
  output_id: number
  source: string
  task_type?: string
  description?: string
  fields?: { key: string; type: string; desc?: string; truncated?: boolean; total?: number }[]
  value: unknown
  truncated?: boolean
  total?: number
  path?: string | null
}

// --- 发送参数 ---

export interface ChatRequest {
  message?: string
  model: string
  dataset: string
  dataset_type?: string
  mode?: "normal" | "react"
  request_id?: string
}

// --- 回调 ---

export type ChatStreamCallback = (event: ChatResponse) => void

// --- 流管理 ---

const pendingCallbacks = new Map<string, ChatStreamCallback>()
/** 每请求已接收的最大 seq（resume 重放去重） */
const lastSeqs = new Map<string, number>()
const activeControllers = new Map<string, AbortController>()

const RESUME_BASE_DELAY_MS = 1000
const RESUME_MAX_DELAY_MS = 10000

/** 后端是否在线（决定真实流 vs 客户端模拟回退） */
export function isChatConnected(): boolean {
  return isBackendLive()
}

/** Rebuild a saved display from its trace; this endpoint never runs the analysis. */
export async function fetchRunDag(runId: string): Promise<OldDag> {
  const baseUrl = await getBaseUrl()
  const response = await fetch(`${baseUrl}/api/chat/runs/${encodeURIComponent(runId)}/dag`)
  const result = await response.json()
  if (!response.ok || !result.success) throw new Error(result.error || "Cannot load execution graph")
  return result.data as OldDag
}

export function sendChatRequest(
  params: ChatRequest,
  callback: ChatStreamCallback,
): void {
  const requestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`

  const wrapped: ChatStreamCallback = (event) => {
    if (event.type === "stream_end" || event.error) {
      pendingCallbacks.delete(requestId)
      lastSeqs.delete(requestId)
      activeControllers.delete(requestId)
    }
    callback(event)
  }
  pendingCallbacks.set(requestId, wrapped)

  void openStream(params, requestId, wrapped, false, 1)
}

/** 打开一条流；断线时按退避自动带 resume 重连，直到收到 stream_end 或后端确认缓冲已丢 */
async function openStream(
  params: ChatRequest,
  requestId: string,
  callback: ChatStreamCallback,
  resume: boolean,
  attempt: number,
): Promise<void> {
  if (!pendingCallbacks.has(requestId)) return // 流已结束

  const controller = new AbortController()
  activeControllers.set(requestId, controller)

  try {
    const url = await getBaseUrl()
    const resp = await fetch(`${url}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params, request_id: requestId, resume }),
      signal: controller.signal,
    })

    if (!resp.ok) {
      // 400 = 参数错误，或 resume 时后端已无缓冲（后端重启过）→ 结束这条流
      if (resume && resp.status === 400) {
        failStream(requestId, callback, "连接中断：分析状态已丢失，请重新发起分析。")
      } else {
        failStream(requestId, callback, "Request failed. Please try again later.")
      }
      return
    }
    if (!resp.body) {
      failStream(requestId, callback, "Request failed. Please try again later.")
      return
    }

    await readEventStream(resp.body, requestId, callback)
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") return
    if (!pendingCallbacks.has(requestId)) return
    // 网络中断：退避后带 resume 重连
    const delay = Math.min(
      RESUME_BASE_DELAY_MS * 2 ** (attempt - 1),
      RESUME_MAX_DELAY_MS,
    )
    await sleep(delay)
    await openStream(params, requestId, callback, true, attempt + 1)
  }
}

/** 逐行解析 NDJSON 事件流，带 seq 去重（resume 重放不会重复投递） */
async function readEventStream(
  body: ReadableStream<Uint8Array>,
  requestId: string,
  callback: ChatStreamCallback,
): Promise<void> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) return
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, idx).trim()
        buffer = buffer.slice(idx + 1)
        if (!line || line.startsWith(":")) continue // 心跳注释行
        let data: ChatResponse
        try {
          data = JSON.parse(line)
        } catch {
          continue
        }
        if (typeof data.seq === "number") {
          const last = lastSeqs.get(requestId) ?? -1
          if (data.seq <= last) continue
          lastSeqs.set(requestId, data.seq)
        }
        callback(data)
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function failStream(
  requestId: string,
  callback: ChatStreamCallback,
  message: string,
): void {
  callback({ type: "stream_end", error: message })
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms))
}

// --- 解析 tasktype 提取算法名 ---

/**
 * 从 tasktype 字符串中提取算法名。
 * 格式示例:
 *   "Graph Algorithm (pagerank)"    → "pagerank"
 *   "Graph Algorithm"               → null
 *   "Numeric Analysis"              → null
 */
export function parseAlgorithmName(tasktype: string): string | null {
  const m = tasktype.match(/\(([^)]+)\)/)
  return m ? m[1].trim() : null
}


export function matchAlgorithm(
  name: string,
  algorithms: Algorithm[],
): Algorithm | undefined {
  // 精确匹配 name
  let a = algorithms.find((x) => x.name === name)
  if (a) return a
  // id 匹配（可能带 run_ 前缀）
  a = algorithms.find((x) => x.id === name || x.id === `run_${name}`)
  if (a) return a
  // 常见别名映射
  const aliases: Record<string, string> = {
    betweenness: "betweenness_centrality",
    closeness: "closeness_centrality",
    pagerank: "pagerank",
    hits: "hits",
    louvain: "louvain_communities",
    "label-propagation": "label_propagation_communities",
    label_propagation: "label_propagation_communities",
    dijkstra: "dijkstra_path",
    "floyd-warshall": "floyd_warshall",
    bfs: "bfs_tree",
    dfs: "dfs_tree",
    "connected-components": "connected_components",
    "max-flow": "maximum_flow",
    "min-cut": "minimum_cut",
    clustering: "clustering",
    jaccard: "jaccard_coefficient",
    "is-bipartite": "is_bipartite",
    mst: "minimum_spanning_tree",
    "topological-sort": "topological_sort",
    "greedy-color": "greedy_color",
    "is-isomorphic": "is_isomorphic",
    "find-cycle": "find_cycle",
    "max-weight-matching": "max_weight_matching",
    assortativity: "degree_assortativity_coefficient",
    diameter: "diameter",
    "global-efficiency": "global_efficiency",
    "cartesian-product": "cartesian_product",
  }
  const resolved = aliases[name] ?? aliases[name.replace(/-/g, "_")]
  if (resolved) {
    return (
      algorithms.find((x) => x.name === resolved) ??
      algorithms.find((x) => x.id === `run_${resolved}`)
    )
  }
  return undefined
}
