"use client"

import { ArrowRight, CheckCircle2, Circle, Database, Loader2, RotateCw, XCircle } from "lucide-react"
import { Card } from "@/components/ui/card"
import { useI18n } from "@/core/i18n/i18n-provider"
import { cn } from "@/core/utils"
import type { ReactRound, ReactRun } from "@/core/types"

export function ReActTimeline({ run }: { run: ReactRun }) {
  const { t } = useI18n()
  const ended = ["complete", "incomplete", "error"].includes(run.stage)
  return (
    <Card className="gap-0 overflow-hidden p-0" aria-label={t("react.timeline", "ReAct 分析过程")}>
      <div className="flex items-center justify-between gap-3 border-b border-border bg-primary/5 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
            <RotateCw className={cn("size-4", !ended && "animate-spin [animation-duration:3s]")} />
          </div>
          <div>
            <h3 className="text-sm font-semibold">{t("react.timeline", "ReAct 分析过程")}</h3>
            <p className="mt-0.5 text-xs text-muted-foreground">{t("react.loopHint", "逐轮决策、执行、观测，再选择下一步")}</p>
          </div>
        </div>
        {run.step > 0 && (
          <span className="shrink-0 rounded-full border border-primary/15 bg-background px-2.5 py-1 text-xs tabular-nums text-primary">
            {t("react.round", "第 {n} 轮").replace("{n}", String(run.step))}
          </span>
        )}
      </div>

      <ol className="px-5 pt-4">
        {run.rounds.map((round) => <Round key={round.step} round={round} />)}
      </ol>

      {run.checklist.length > 0 && !ended && (
        <div className="mx-5 mb-4 rounded-lg border border-dashed border-border bg-muted/20 p-3">
          <div className="mb-2 text-xs font-medium text-muted-foreground">{t("react.checklist", "待完成目标")}</div>
          {run.checklist.map((item, index) => (
            <div key={index} className="mt-1.5 flex items-start gap-2 text-xs">
              <Circle className="mt-0.5 size-3 shrink-0 text-muted-foreground" />{item}
            </div>
          ))}
        </div>
      )}
      {run.stage === "incomplete" || run.stage === "error" ? (
        <div className="border-t border-amber-200 bg-amber-50 px-5 py-3 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200">
          <p>{t(`react.stopReasons.${run.terminationReason}`, t("react.notCompleted", "本次分析未完成，请查看具体错误及未完成项。"))}</p>
          {(run.missingGoals ?? []).length > 0 && <ul className="mt-2 list-inside list-disc space-y-1">
            {run.missingGoals!.map((goal, index) => <li key={index}>{goal}</li>)}
          </ul>}
        </div>
      ) : null}
      <div role="status" className={cn("flex items-center gap-2 border-t border-border px-5 py-3 text-xs",
        run.stage === "error" ? "bg-destructive/5 text-destructive" : run.stage === "incomplete"
          ? "bg-amber-50 text-amber-800 dark:bg-amber-950/30 dark:text-amber-200"
          : ended ? "bg-muted/30 text-muted-foreground" : "bg-primary/5 text-primary")}>
        {run.stage === "error" || run.stage === "incomplete" ? <XCircle className="size-3.5" /> : ended ? <CheckCircle2 className="size-3.5" /> : <Loader2 className="size-3.5 animate-spin" />}
        {t(`react.stage.${run.stage}`, run.stage)}
        {run.runId && <span className="ml-auto font-mono text-[10px]" title={run.runId}>
          {t("react.runRecord", "记录")} {run.runId.slice(0, 8)}
        </span>}
      </div>
    </Card>
  )
}

function Round({ round }: { round: ReactRound }) {
  const { t } = useI18n()
  const error = round.stage === "error"
  const observed = !!round.observation
  const finishing = round.action?.type === "finish"
  const done = observed || round.stage === "complete"
  const tool = round.action?.name || (round.action ? t(`react.action.${round.action.type}`, round.action.type) : "")
  return (
    <li className="relative border-l border-border pb-5 pl-6 last:border-transparent">
      <div className={cn("absolute -left-2.5 top-0 flex size-5 items-center justify-center rounded-full bg-card",
        error ? "text-destructive" : done ? "text-primary" : "text-muted-foreground")}>
        {error ? <XCircle className="size-4" /> : done ? <CheckCircle2 className="size-4" /> : <Loader2 className="size-4 animate-spin" />}
      </div>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-semibold">{t("react.round", "第 {n} 轮").replace("{n}", String(round.step))}</span>
        {tool && <span className="rounded-md bg-muted px-2 py-0.5 font-medium">{tool}</span>}
        {error && <span className="text-destructive">{t("react.toolFailed", "本轮执行失败")}</span>}
      </div>
      {round.thought && <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-foreground/85">{round.thought}</p>}
      {round.action && !finishing && (
        <div className="mt-3 rounded-lg border border-border bg-muted/20 px-3 py-2.5">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">{t("react.actionLabel", "执行动作")}</span>
            <ArrowRight className="size-3 text-muted-foreground" /><span className="font-medium">{tool}</span>
          </div>
          {(Object.keys(round.action.parameters ?? {}).length > 0 || (round.action.consumed ?? []).length > 0) && (
            <details className="mt-2 text-xs">
              <summary className="cursor-pointer text-muted-foreground hover:text-foreground">{t("react.inputs", "查看输入与引用")}</summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded-md bg-background p-2 text-[11px]">
                {JSON.stringify({ parameters: round.action.parameters, consumed: round.action.consumed }, null, 2)}
              </pre>
            </details>
          )}
        </div>
      )}
      {round.observation && (
        <div className={cn("mt-2 rounded-lg border px-3 py-2.5", error ? "border-destructive/20 bg-destructive/5" : "border-primary/15 bg-primary/[0.035]")}>
          <div className="mb-1 text-xs font-medium text-muted-foreground">{t("react.observation", "结果与观测")}</div>
          <p className="whitespace-pre-wrap break-words text-sm leading-6">{round.observation.summary}</p>
          {round.observation.error && !round.observation.summary.includes(round.observation.error) && (
            <p className="mt-1 text-xs text-destructive">{round.observation.error}</p>
          )}
          {(round.observation.evidence ?? []).length > 0 && (
            <ul className="mt-2 list-inside list-disc space-y-1 text-xs text-muted-foreground">
              {round.observation.evidence!.map((item, index) => <li key={index}>{String(item)}</li>)}
            </ul>
          )}
        </div>
      )}
      {round.artifacts.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {round.artifacts.map((artifact) => (
            <details key={artifact.artifact_id} className="max-w-full rounded-lg border border-border bg-background p-2 text-xs">
              <summary className="cursor-pointer text-muted-foreground">
                <Database className="mr-1 inline size-3" />[{artifact.artifact_id}] {artifact.source || t("react.artifact", "结果证据")}
              </summary>
              <pre className="mt-2 max-h-52 overflow-auto whitespace-pre-wrap break-all text-[11px]">
                {typeof artifact.value !== "undefined" ? JSON.stringify(artifact.value, null, 2) : artifact.summary}
              </pre>
            </details>
          ))}
        </div>
      )}
    </li>
  )
}
