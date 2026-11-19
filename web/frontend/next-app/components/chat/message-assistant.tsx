"use client"

import * as React from "react"
import { Sparkles } from "lucide-react"
import { useI18n } from "@/core/i18n/i18n-provider"
import type { ChatMessage } from "@/core/types"
import { DagPanel } from "./dag-panel"
import { AnalysisReport } from "./analysis-report"
import { ReActTimeline } from "./react-timeline"

export function MessageAssistant({
  message,
  userQuestion,
  onLayoutChange,
}: {
  message: ChatMessage
  userQuestion: string
  onLayoutChange: (layout: Record<string, { x: number; y: number }>) => void
}) {
  const { t } = useI18n()
  const plan = message.plan
  if (!plan) return null

  const phase = plan.phase

  return (
    <div className="flex items-start gap-3">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Sparkles className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2 text-xs font-medium text-muted-foreground">
          <span className="text-foreground">{t("chat.engine", "易图 ReAct 智能体")}</span>
          <span>·</span>
          <span>{plan.react ? t(`react.stage.${plan.react.stage}`, plan.react.stage) : t("react.stage.complete", "分析已结束")}</span>
        </div>

        <div className="flex flex-col gap-4">
          {plan.error && (
            <div className="rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {plan.error}
            </div>
          )}
          {plan.react && <ReActTimeline run={plan.react} />}

          {/* Markdown from backend, or structured report from mock */}
          {(plan.markdown || plan.report) && (
            <AnalysisReport report={plan.report} markdown={plan.markdown} userQuestion={userQuestion} />
          )}
          {phase === "executed" && plan.nodes.length > 0 && (
            <details className="rounded-lg border border-border bg-muted/10">
              <summary className="cursor-pointer px-4 py-3 text-xs font-medium text-muted-foreground hover:text-foreground">
                {t("react.viewRecord", "查看执行记录图")}
              </summary>
              <DagPanel plan={plan} userQuestion={userQuestion} onLayoutChange={onLayoutChange} />
            </details>
          )}
        </div>
      </div>
    </div>
  )
}
