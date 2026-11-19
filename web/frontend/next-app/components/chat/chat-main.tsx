"use client"

import * as React from "react"
import { Sparkles, Workflow, MessageSquare } from "lucide-react"
import type { ChatSession, Dataset, Model } from "@/core/types"
import { useI18n } from "@/core/i18n/i18n-provider"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatInput, type ChatSettings } from "./chat-input"
import { MessageUser } from "./message-user"
import { MessageAssistant } from "./message-assistant"

export function ChatMain({
  session,
  onSubmit,
  onLayoutChange,
  settings,
  onSettingsChange,
  datasets,
  models,
}: {
  session: ChatSession | null
  onSubmit: (q: string) => void
  onLayoutChange: (
    messageId: string,
    layout: Record<string, { x: number; y: number }>,
  ) => void
  settings: ChatSettings
  onSettingsChange: (s: ChatSettings) => void
  datasets: Dataset[]
  models: Model[]
}) {
  const { t } = useI18n()
  const scrollRef = React.useRef<HTMLDivElement>(null)
  const endRef = React.useRef<HTMLDivElement>(null)

  // 内容签名：只反映"回复内容"的变化。
  // 手动拖动 DAG 节点只写 plan.layout，不应触发自动滚到底部。
  const scrollSignal = buildScrollSignal(session)
  React.useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" })
  }, [scrollSignal])

  const hasMessages = !!session?.messages.length

  return (
    <section className="flex min-w-0 flex-1 flex-col bg-background">
      {/* Header */}
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-background/80 px-6 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
            <Workflow className="size-4" />
          </div>
          <div>
            <h1 className="text-base font-semibold leading-tight">
              {session?.title || t("chat.heading", "新的分析会话")}
            </h1>
            <p className="text-xs text-muted-foreground">
              {t("chat.subheading", "自然语言提问 · ReAct 逐轮分析 · 生成报告")}
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground md:flex">
          <span className="size-1.5 rounded-full bg-accent" />
          {t("react.autoMode", "ReAct 自动分析")}
        </div>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="mx-auto flex max-w-4xl flex-col gap-6 px-6 py-8">
            {!hasMessages && <EmptyState onPick={onSubmit} />}
            {session?.messages.map((m, i) =>
              m.role === "user" ? (
                <MessageUser key={m.id} message={m} />
              ) : (
                <MessageAssistant
                  key={m.id}
                  message={m}
                  userQuestion={
                    session.messages
                      .slice(0, i)
                      .findLast((p) => p.role === "user")?.content ?? ""
                  }
                  onLayoutChange={(layout) => onLayoutChange(m.id, layout)}
                />
              ),
            )}
            <div ref={endRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Input */}
      <ChatInput
        onSubmit={onSubmit}
        disabled={isBusy(session)}
        settings={settings}
        onSettingsChange={onSettingsChange}
        datasets={datasets}
        models={models}
      />
    </section>
  )
}

function isBusy(session: ChatSession | null) {
  if (!session) return false
  const last = session.messages[session.messages.length - 1]
  if (!last || last.role !== "assistant") return false
  const p = last.plan?.phase
  return !last.plan?.error && (p === "understanding" || p === "decomposing" || p === "selecting" ||
    p === "executing" || p === "react-running" || p === "reporting")
}

/**
 * 自动滚动判定用的内容签名。
 * 刻意不包含 plan.layout —— 用户拖动 DAG 节点时只有该字段变化，
 * 若把它算进来，每次落位都会把会话拉到底部。
 */
function buildScrollSignal(session: ChatSession | null): string {
  const messages = session?.messages ?? []
  const last = messages[messages.length - 1]
  const plan = last?.role === "assistant" ? last.plan : undefined
  if (!plan) return `${messages.length}|${last?.content.length ?? 0}`
  return [
    messages.length,
    last?.content.length ?? 0,
    plan.phase,
    plan.react?.stage,
    ...((plan.react?.rounds ?? []).map((round) =>
      `${round.step}:${round.stage}:${round.thought?.length}:${round.observation?.summary.length}:${round.artifacts.length}`)),
    plan.progress.understanding,
    plan.progress.decomposing,
    plan.progress.selecting,
    plan.understanding.length,
    plan.subProblems.length,
    ...plan.nodes.map(
      (n) =>
        `${n.id}:${n.status}:${n.finishedAtMs ?? ""}:${n.label.length}:${n.result?.outputs?.length ?? 0}`,
    ),
    plan.edges.length,
    plan.markdown?.length ?? 0,
    plan.error ?? "",
    plan.report ? "report" : "",
  ].join("|")
}

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  const { t } = useI18n()
  const suggestions = Array.from({ length: 4 }, (_, i) =>
    t(`chat.suggestions.${i}`),
  ).filter(Boolean)

  return (
    <div className="flex flex-col items-center justify-center py-10 text-center">
      <div className="mb-5 flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <Sparkles className="size-7" />
      </div>
      <h2 className="text-balance text-2xl font-semibold">
        {t("chat.emptyTitle", "描述你的图数据分析目标")}
      </h2>
      <p className="mt-2 max-w-xl text-pretty text-sm text-muted-foreground">
        {t("chat.emptyDesc")}
      </p>

      <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
        {suggestions.map((s, i) => (
          <button
            key={i}
            onClick={() => onPick(s)}
            className="group flex items-start gap-3 rounded-lg border border-border bg-card p-4 text-left transition-colors hover:border-primary/40 hover:bg-primary/5"
          >
            <MessageSquare className="mt-0.5 size-4 text-muted-foreground group-hover:text-primary" />
            <span className="text-pretty text-sm text-foreground/90">{s}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
