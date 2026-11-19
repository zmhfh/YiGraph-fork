import type { ChatResponse } from "./api/chat"
import type { ReactAction, ReactArtifact, ReactObservation, ReactRound, ReactRun, ReactStage } from "./types"

export function createReactRun(): ReactRun {
  return { stage: "deciding", step: 0, checklist: [], rounds: [] }
}

export function finalizeReactRun(run: ReactRun, failed = false): ReactRun {
  if (failed) return { ...run, stage: "error", outcome: "failed" }
  if (["complete", "incomplete", "error"].includes(run.stage)) return run
  if (run.outcome === "failed") return { ...run, stage: "error" }
  if (run.outcome === "incomplete") return { ...run, stage: "incomplete" }
  if (run.rounds.length > 0) {
    return { ...run, stage: "incomplete", terminationReason: run.terminationReason || "stream_interrupted" }
  }
  return { ...run, stage: "complete" }
}

/** Upsert by round/artifact id so stream replay never duplicates the timeline. */
export function reduceReactEvent(previous: ReactRun, event: ChatResponse): ReactRun {
  const content = (event.content && typeof event.content === "object"
    ? event.content : {}) as Record<string, unknown>
  const step = event.step ?? (typeof content.step === "number" ? content.step : previous.step)
  const run: ReactRun = {
    ...previous,
    step,
    checklist: [...previous.checklist],
    rounds: previous.rounds.map((round) => ({ ...round, artifacts: [...round.artifacts] })),
  }
  let round = run.rounds.find((item) => item.step === step)
  if (!round && step > 0) {
    round = { step, stage: "deciding", artifacts: [] }
    run.rounds.push(round)
    run.rounds.sort((a, b) => a.step - b.step)
  }
  if (event.contentType === "react_status") {
    run.stage = (content.stage as ReactStage) ?? run.stage
    if (typeof content.max_steps === "number") run.maxSteps = content.max_steps
    if (round && !round.observation) round.stage = run.stage
    if (typeof content.run_id === "string") run.runId = content.run_id
    if (content.outcome) run.outcome = content.outcome as ReactRun["outcome"]
    if (typeof content.termination_reason === "string") run.terminationReason = content.termination_reason
    if (Array.isArray(content.missing_goals)) run.missingGoals = content.missing_goals.map(String)
  } else if (event.contentType === "react_thought" || event.type === "thinking") {
    if (round) round.thought = String(event.content ?? "")
  } else if (event.contentType === "react_action") {
    const action = content as unknown as ReactAction
    if (round) {
      round.action = action
      round.thought = event.thought ?? round.thought
      round.stage = action.type === "finish" ? "complete" : "binding"
    }
    run.stage = action.type === "finish" ? "reporting" : "binding"
    if (event.checklist) run.checklist = event.checklist
  } else if (event.contentType === "react_observation") {
    const observation = { ...content, summary: String(content.summary ?? "") } as ReactObservation
    if (round) {
      round.observation = observation
      round.stage = observation.success === false || observation.error ? "error" : "complete"
      const artifact = content.artifact as ReactArtifact | undefined
      if (artifact?.artifact_id) {
        const index = round.artifacts.findIndex((item) => item.artifact_id === artifact.artifact_id)
        if (index < 0) round.artifacts.push(artifact)
        else round.artifacts[index] = { ...round.artifacts[index], ...artifact }
      }
    }
    run.stage = "deciding"
  } else if (event.contentType === "react_artifact" && round) {
    const artifact = content as unknown as ReactArtifact
    const index = round.artifacts.findIndex((item) => item.artifact_id === artifact.artifact_id)
    if (index < 0) round.artifacts.push(artifact)
    else round.artifacts[index] = { ...round.artifacts[index], ...artifact }
  }
  return run
}
