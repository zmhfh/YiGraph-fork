export type Lang = "zh-CN" | "en-US"

export type AlgorithmCategory = string

export interface Algorithm {
  id: string
  name: string
  displayName: string
  category: AlgorithmCategory
  summary: string
  description: string
  complexity: string
  inputs: string[]
  outputs: string[]
  useCases: string[]
  chartPreference: ChartType
  tags: string[]
}

export type ChartType = "bar" | "line" | "pie" | "table" | "scatter"

export type ChatRole = "user" | "assistant"

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  createdAt: string
  /** When role === "assistant", holds the plan/execution state */
  plan?: WorkflowPlan
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  messages: ChatMessage[]
}

export interface SubProblem {
  id: string
  title: string
  rationale: string
  algorithmId: string
}

export interface DagNode {
  id: string
  label: string
  algorithmId: string
  subProblemId: string
  status: NodeStatus
  /** 无算法时的任务类型标签（如 "数值分析"、"图算法"） */
  taskType?: string
  toolName?: string
  executionRecord?: Record<string, unknown>
  /** Simulated execution output */
  result?: NodeResult
  /** 执行失败时的错误信息（真实后端执行时下发） */
  error?: string
  /** 节点开始/结束执行的时间戳（毫秒），用于展示耗时 */
  startedAtMs?: number
  finishedAtMs?: number
}

export type NodeStatus = "pending" | "running" | "done" | "error"

export interface DagEdge {
  id: string
  source: string
  target: string
  kind?: "data" | "sequence" | "summary"
}

export interface NodeResult {
  summary: string
  code?: string
  chartType?: ChartType
  chartData?: ChartPoint[]
  /** Sub-graph for relationship/path visualization */
  subGraph?: GraphData
  /** 真实后端下发的结构化输出（node_result 事件），逐项截断后的值 */
  outputs?: NodeOutput[]
}

/** 真实后端单个 StepOutputItem 的前端视图（已截断、JSON 安全） */
export interface NodeOutput {
  /** 同一步骤内的输出序号（1 起） */
  id: number
  /** 来源：算法名 / "python code" / "graph_query" */
  source: string
  /** 输出类别（graph_algorithm / post_processing / numeric_computation / subgraph_extraction） */
  kind?: string
  description?: string
  /** output_schema 的字段描述（截断标记由后端写入） */
  fields?: { key: string; type: string; desc?: string; truncated?: boolean; total?: number }[]
  /** 截断后的值 */
  value: unknown
  /** 值是否被截断 */
  truncated?: boolean
  /** 截断前条目总数 */
  total?: number
  path?: string | null
}

export interface ChartPoint {
  label: string
  value: number
  /** Optional secondary value for line charts */
  value2?: number
}

export type ReactStage =
  | "deciding"
  | "binding"
  | "executing"
  | "observing"
  | "reporting"
  | "complete"
  | "incomplete"
  | "error"

export interface ReactAction {
  type: "algorithm" | "graph_query" | "numeric_analysis" | "finish"
  name: string
  parameters: Record<string, unknown>
  consumed: { artifact_id: string; field?: string; use_as?: string }[]
}

export interface ReactArtifact {
  artifact_id: string
  source?: string
  summary?: string
  fields?: Record<string, unknown>
  value?: unknown
}

export interface ReactObservation {
  summary: string
  success?: boolean
  empty?: boolean
  error?: string
  evidence?: string[]
  artifact_id?: string
}

export interface ReactRound {
  step: number
  stage: ReactStage
  thought?: string
  action?: ReactAction
  observation?: ReactObservation
  artifacts: ReactArtifact[]
}

export interface ReactRun {
  stage: ReactStage
  step: number
  maxSteps?: number
  checklist: string[]
  rounds: ReactRound[]
  outcome?: "running" | "completed" | "incomplete" | "failed"
  runId?: string
  terminationReason?: string
  missingGoals?: string[]
}

export interface WorkflowPlan {
  /** ReAct replies render a live timeline; legacy plans retain their DAG UI. */
  executionMode?: "react" | "dag"
  react?: ReactRun
  /** High level understanding of the user question */
  understanding: string
  subProblems: SubProblem[]
  nodes: DagNode[]
  edges: DagEdge[]
  /** 用户手动拖动后的节点坐标覆盖（仅影响视觉布局，不改变依赖与执行顺序） */
  layout?: Record<string, { x: number; y: number }>
  /** Phase of the assistant reply */
  phase:
    | "understanding"
    | "decomposing"
    | "selecting"
    | "dag-ready"
    | "executing"
    | "react-running"
    | "reporting"
    | "executed"
  /** streaming progress for each phase 0-1 */
  progress: {
    understanding: number
    decomposing: number
    selecting: number
  }
  /** 发起该消息时是否处于交互/专家模式（决定 dag-ready 的确认按钮与真实后端执行） */
  interactive?: boolean
  /** Accumulated markdown from result (text) events */
  markdown?: string
  /** Error message from backend, if any */
  error?: string
  /** Final analysis report, available after execution */
  report?: AnalysisReport
}

export interface AnalysisReport {
  title: string
  executiveSummary: string
  keyFindings: string[]
  recommendations: string[]
  metrics: { label: string; value: string; trend?: "up" | "down" | "flat" }[]
}

// === Dataset / File management types ===

export type FileType = "raw-data" | "graph-data"

export interface Dataset {
  id: string
  name: string
  fileType: FileType
  fileCount: number
  createdAt: string
}

export type ParseStatus = "pending" | "parsing" | "completed" | "error"

export interface FileItem {
  id: string
  name: string
  size: number
  type: string
  uploadTime: string
  parseStatus: ParseStatus
  progress: number
  datasetId: string
}

export interface GraphNode {
  id: string
  label: string
  group?: string
  color?: string
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  label?: string
  tooltip?: string
}

export interface GraphData {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface UploadConfig {
  vertexLabelField?: string
  edgeSourceField?: string
  edgeTargetField?: string
  graphName?: string
}

export type FileType2 = "csv" | "txt" | "pdf" | "docx" | "md"

export interface PreviewData {
  type: FileType2
  content: string
  headers?: string[]
  rows?: string[][]
  totalPages?: number
}

// === Model management types ===

export interface Model {
  id: string
  name: string
  baseUrl: string
  createdAt: string
}

export interface PaginatedResponse<T> {
  data: T[]
  total: number
  page: number
  pageSize: number
}

// === Feedback types ===

export interface FeedbackEntry {
  timestamp: string
  userQuestion: string
  analysisOutput: string
  type: "dag" | "report"
  feedback: "like" | "dislike"
}
