const test = require("node:test")
const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")
const ts = require("typescript")
const root = path.resolve(__dirname, "..")
const cache = new Map()

// Load the actual TypeScript adapter without a separate test build or browser.
function load(file) {
  const filename = path.resolve(root, file.endsWith(".ts") ? file : `${file}.ts`)
  if (cache.has(filename)) return cache.get(filename).exports
  const module = { exports: {} }
  cache.set(filename, module)
  const source = ts.transpileModule(fs.readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText
  const localRequire = (name) => {
    if (name.startsWith("@/")) return load(name.slice(2))
    if (name.startsWith(".")) return load(path.resolve(path.dirname(filename), name))
    return require(name)
  }
  vm.runInNewContext(source, { module, exports: module.exports, require: localRequire,
    Map, Set, console, setTimeout, clearTimeout, AbortController, TextDecoder }, { filename })
  return module.exports
}

const { createAdapterState, processChatEvent, convertOldDagToPlan } = load("core/chat-adapter")
const { createReactRun, finalizeReactRun } = load("core/react-state")
const action = { type: "algorithm", name: "pagerank", parameters: { alpha: 0.85 }, consumed: [] }
const status = (stage, step = 1) => ({ type: "result", contentType: "react_status", step,
  content: { stage, max_steps: 8 } })

test("streamed rounds expose tool, observation and evidence before a report", () => {
  const state = createAdapterState()
  processChatEvent(status("deciding"), state)
  processChatEvent({ type: "thinking", contentType: "react_thought", step: 1, content: "Measure importance." }, state)
  processChatEvent({ type: "result", contentType: "react_action", step: 1,
    thought: "Measure importance.", checklist: ["Rank accounts"], content: action }, state)
  assert.equal(state.react.rounds[0].action.name, "pagerank")
  assert.equal(state.react.checklist[0], "Rank accounts")
  processChatEvent(status("executing"), state)
  assert.equal(state.react.stage, "executing")
  processChatEvent({ type: "result", contentType: "react_observation", step: 1,
    content: { summary: "A has score 0.7", success: true, artifact_id: "a1",
      artifact: { artifact_id: "a1", source: "pagerank" } } }, state)
  assert.equal(state.react.rounds[0].observation.summary, "A has score 0.7")
  assert.equal(state.react.rounds[0].stage, "complete")
  assert.equal(state.react.rounds[0].artifacts[0].artifact_id, "a1")
})

test("a failed round stays visible while the next round recovers", () => {
  const state = createAdapterState()
  processChatEvent(status("executing"), state)
  processChatEvent({ type: "result", contentType: "react_observation", step: 1,
    content: { summary: "Unavailable algorithm", success: false, error: "unavailable" } }, state)
  processChatEvent(status("deciding", 2), state)
  processChatEvent({ type: "result", contentType: "react_action", step: 2, content: action }, state)
  assert.equal(state.react.rounds.length, 2)
  assert.equal(state.react.rounds[0].stage, "error")
  assert.equal(state.react.rounds[0].observation.error, "unavailable")
  assert.equal(state.react.step, 2)
})

test("replayed artifact updates do not duplicate rounds or evidence", () => {
  const state = createAdapterState()
  const event = { type: "result", contentType: "react_artifact", step: 1,
    content: { artifact_id: "a1", source: "pagerank", value: { A: 0.7 } } }
  processChatEvent(event, state)
  processChatEvent(event, state)
  assert.equal(state.react.rounds.length, 1)
  assert.equal(state.react.rounds[0].artifacts.length, 1)
})

test("an end-of-run graph preserves status and cannot become a planning event", () => {
  const state = createAdapterState()
  processChatEvent(status("reporting", 2), state)
  const dag = { execution_driver: "react", nodes: [
    { id: "1", label: "Decide how to analyze the question", tasktype: "ReAct decision", status: "success" },
    { id: "2", label: "Failed tool observation", tasktype: "ReAct observation", status: "failed" },
  ], edges: [{ from: "1", to: "2" }] }
  const result = processChatEvent({ type: "result", contentType: "dag", content: dag }, state)
  assert.equal(result.kind, "dag")
  assert.equal(state.executionMode, "react")
  assert.equal(state.react.stage, "reporting")
  const record = convertOldDagToPlan(dag, "", [])
  assert.equal(record.nodes.length, 2)
  assert.equal(record.nodes[0].status, "done")
  assert.equal(record.nodes[1].status, "error")
  processChatEvent(status("complete", 2), state)
  processChatEvent({ type: "stream_end" }, state)
  assert.equal(state.finished, true)
})

test("stream end preserves an incomplete outcome instead of displaying completion", () => {
  const state = createAdapterState()
  processChatEvent({ type: "result", contentType: "react_status", step: 4,
    content: { stage: "incomplete", outcome: "incomplete", termination_reason: "timeout",
      run_id: "abc", missing_goals: ["Compute amount"] } }, state)
  const finalized = finalizeReactRun(state.react)
  assert.equal(finalized.stage, "incomplete")
  assert.equal(finalized.terminationReason, "timeout")
  assert.equal(finalized.missingGoals[0], "Compute amount")
})

test("ending a live round without terminal confirmation is not success", () => {
  const run = { ...createReactRun(), stage: "executing", step: 1,
    rounds: [{ step: 1, stage: "executing", artifacts: [] }] }
  assert.equal(finalizeReactRun(run).stage, "incomplete")
  assert.equal(finalizeReactRun(run).terminationReason, "stream_interrupted")
})

test("task graphs preserve tool details, evidence, errors and edge meaning", () => {
  const dag = { execution_driver: "react", nodes: [
    { id: "1", label: "Compute importance", algorithm: "pagerank", tasktype: "Graph Algorithm (pagerank)",
      status: "success", record: { artifact_ids: ["a1"], attempts: [{ round: 1 }] }, summary: "A has the largest score" },
    { id: "2", label: "Calculate a measure", tasktype: "Numeric Analysis", status: "failed", error: "Unknown property", record: {} },
  ], edges: [{ from: "1", to: "2", kind: "data" }] }
  const graph = convertOldDagToPlan(dag, "", [])
  assert.equal(graph.nodes[0].toolName, "pagerank")
  assert.equal(graph.nodes[0].executionRecord.artifact_ids[0], "a1")
  assert.equal(graph.nodes[0].result.summary, "A has the largest score")
  assert.equal(graph.nodes[1].error, "Unknown property")
  assert.equal(graph.edges[0].kind, "data")
  assert.equal(convertOldDagToPlan({ ...dag, edges: [] }, "", []).edges.length, 0)
})

test("record layout gives long chains distinct columns and keeps branches apart", () => {
  const { layoutDagNodes } = load("core/dag-layout")
  const nodes = ["a", "b", "c", "d", "e"].map(id => ({ id, label: id }))
  const chain = nodes.slice(1).map((node, index) => ({ source: nodes[index].id, target: node.id }))
  const positions = layoutDagNodes(nodes, chain, {})
  for (const edge of chain) assert.ok(positions[edge.source].x < positions[edge.target].x)
  const branched = layoutDagNodes(nodes.slice(0, 3), [{ source: "a", target: "b" }, { source: "a", target: "c" }], { b: 200 })
  assert.equal(branched.b.x, branched.c.x)
  assert.ok(branched.c.y >= branched.b.y + branched.b.h)
  assert.equal(layoutDagNodes(nodes.slice(0, 1), [], {}).a.x, 20)
})
