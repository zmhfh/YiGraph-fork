"use client"

import * as React from "react"
import { useChat } from "@/core/chat-store"
import { useAlgorithms } from "@/core/algorithms-store"
import { useDatasets } from "@/core/datasets-store"
import { useModels } from "@/core/models-store"
import { createAdapterState, convertOldDagToPlan, processChatEvent } from "@/core/chat-adapter"
import { fetchRunDag, isChatConnected, sendChatRequest } from "@/core/api/chat"
import { createReactRun, finalizeReactRun } from "@/core/react-state"
import type { ChatMessage, WorkflowPlan } from "@/core/types"
import { ChatHistory } from "./chat-history"
import { ChatMain } from "./chat-main"
import type { ChatSettings } from "./chat-input"

export function ChatWorkspace() {
  const {
    sessions, activeSession, activeId, createSession, selectSession,
    deleteSession, appendMessage, renameSession, updatePlan,
  } = useChat()
  const { algorithms } = useAlgorithms()
  const { datasets } = useDatasets()
  const { models } = useModels()
  const [settings, setSettings] = React.useState<ChatSettings>({ model: "", dataset: "" })
  const refreshedRecords = React.useRef(new Set<string>())

  React.useEffect(() => {
    if (!activeSession) return
    for (const message of activeSession.messages) {
      const plan = message.plan
      const runId = plan?.react?.runId
      if (!runId || plan?.phase !== "executed" || !plan.nodes.length ||
          plan.nodes.some(node => node.executionRecord) || refreshedRecords.current.has(runId)) continue
      refreshedRecords.current.add(runId)
      fetchRunDag(runId).then(dag => {
        const record = convertOldDagToPlan(dag, plan.understanding, algorithms)
        updatePlan(activeSession.id, message.id, previous => ({
          ...previous, nodes: record.nodes, edges: record.edges, layout: {},
        }))
      }).catch(() => { /* Keep the saved display if its original trace is unavailable. */ })
    }
  }, [activeSession, algorithms, updatePlan])

  React.useEffect(() => {
    if (!settings.model && models.length > 0) {
      setSettings((previous) => ({ ...previous, model: models[0].id }))
    }
  }, [models, settings.model])

  React.useEffect(() => {
    if (!settings.dataset && datasets.length > 0) {
      const initial = datasets.find((dataset) => dataset.fileType === "graph-data") ?? datasets[0]
      setSettings((previous) => ({ ...previous, dataset: initial.id }))
    }
  }, [datasets, settings.dataset])

  React.useEffect(() => {
    if (!activeId && sessions[0]) selectSession(sessions[0].id)
  }, [activeId, sessions, selectSession])

  const handleSubmit = React.useCallback((question: string) => {
    const sessionId = activeId ?? createSession(question.slice(0, 28))
    const current = sessions.find((session) => session.id === sessionId)
    if (current && current.messages.length === 0 && !current.title) {
      renameSession(sessionId, question.slice(0, 28))
    }
    appendMessage(sessionId, {
      id: `m-${crypto.randomUUID()}-u`, role: "user", content: question,
      createdAt: new Date().toISOString(),
    })
    const plan: WorkflowPlan = {
      executionMode: "react", react: createReactRun(),
      understanding: "", subProblems: [], nodes: [], edges: [],
      phase: "react-running", progress: { understanding: 0, decomposing: 0, selecting: 0 },
    }
    const messageId = `m-${crypto.randomUUID()}-a`
    const message: ChatMessage = {
      id: messageId, role: "assistant", content: "", createdAt: new Date().toISOString(), plan,
    }
    appendMessage(sessionId, message)
    const fail = (error: string) => updatePlan(sessionId, messageId, (previous) => ({
      ...previous, phase: "executed", error,
      react: { ...(previous.react ?? createReactRun()), stage: "error" },
    }))
    if (!isChatConnected()) {
      fail("后端服务未连接，无法开始分析。请确认后端已启动后重试。")
      return
    }

    const model = models.find((item) => item.id === settings.model)
    const dataset = datasets.find((item) => item.id === settings.dataset)
    const adapter = createAdapterState()
    sendChatRequest({
      message: question, model: model?.name ?? settings.model,
      dataset: dataset?.name ?? settings.dataset,
      dataset_type: dataset?.fileType === "graph-data" ? "graph" : "text",
      mode: "normal",
    }, (event) => {
      const result = processChatEvent(event, adapter)
      switch (result.kind) {
        case "react":
          updatePlan(sessionId, messageId, (previous) => ({
            ...previous, executionMode: "react", react: result.react,
            phase: result.react.stage === "reporting" || result.react.stage === "complete"
              ? "reporting" : "react-running",
          }))
          break
        case "thinking":
          updatePlan(sessionId, messageId, (previous) => ({
            ...previous, understanding: adapter.thinkingText,
          }))
          break
        case "dag": {
          const record = convertOldDagToPlan(result.dag, adapter.thinkingText, algorithms)
          // Receiving a record graph must never introduce a confirmation or
          // a second execution stage.
          updatePlan(sessionId, messageId, (previous) => ({
            ...previous, nodes: record.nodes, edges: record.edges,
          }))
          break
        }
        case "result_text":
          updatePlan(sessionId, messageId, (previous) => ({
            ...previous, phase: "reporting",
            markdown: (previous.markdown ?? "") + result.text + "\n\n",
          }))
          break
        case "stream_end":
          updatePlan(sessionId, messageId, (previous) => ({
            ...previous, phase: "executed",
            react: finalizeReactRun(previous.react ?? createReactRun(), !!previous.error),
          }))
          break
        case "error": fail(result.message); break
      }
    })
  }, [activeId, sessions, createSession, renameSession, appendMessage, updatePlan,
    algorithms, datasets, models, settings])

  const handleLayoutChange = React.useCallback((messageId: string,
    layout: Record<string, { x: number; y: number }>) => {
    if (activeId) updatePlan(activeId, messageId, (plan) => ({ ...plan, layout }))
  }, [activeId, updatePlan])

  return (
    <div className="flex min-h-0 flex-1 overflow-hidden">
      <ChatHistory sessions={sessions} activeId={activeId} onSelect={selectSession}
        onCreate={() => createSession()} onDelete={deleteSession} />
      <ChatMain session={activeSession} onSubmit={handleSubmit}
        onLayoutChange={handleLayoutChange} settings={settings} onSettingsChange={setSettings}
        datasets={datasets} models={models} />
    </div>
  )
}
