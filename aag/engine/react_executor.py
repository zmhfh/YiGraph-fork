"""ReAct execution controller for end-to-end graph analysis.

The controller is intentionally small and deterministic.  The model chooses
one action at a time; Python owns the run state, tool calls, retry budget and
termination guard.  A trace is returned to the scheduler and can be projected
to the legacy DAG view after execution.  The trace/DAG is observational data:
it never determines which action runs next.
"""

from __future__ import annotations

import json
import inspect
import logging
import time
import uuid
import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from aag.expert_search_engine.database.datatype import GraphData
from aag.models.graph_workflow_dag import GraphWorkflowDAG
from aag.utils.data_utils import take_sample
from aag.utils.graph_conversion import reconstruct_graph
from aag.engine.react_trace import ReactTraceStore, preview, is_sampled, collection_metadata
from aag.computing_engine.schema_roles import temporal_path_schema, bind_temporal_path_parameters

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    """Mutable state visible to the decision model for one user request."""

    goal: str
    dataset: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    checklist: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    termination_reason: str = ""
    missing_goals: List[str] = field(default_factory=list)
    budget: Dict[str, Any] = field(default_factory=dict)
    started_at_wall: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.monotonic)

    def prompt_view(self, observation_chars: int = 6000) -> Dict[str, Any]:
        """Return a bounded, JSON-safe view for a small decision model."""
        history = []
        for item in self.history[-12:]:
            copied = dict(item)
            if "summary" in copied:
                copied["summary"] = str(copied["summary"])[:observation_chars]
            history.append(copied)

        artifact_view = {}
        for artifact_id, artifact in list(self.artifacts.items())[-12:]:
            artifact_view[artifact_id] = {
                "source": artifact.get("source"),
                "fields": artifact.get("fields", {}),
                "summary": str(artifact.get("summary", ""))[:observation_chars],
                "evidence_scope": artifact.get("evidence_scope", {}),
                "sample": preview(artifact.get("value")),
                **collection_metadata(artifact.get("value")),
                "collection_sizes": {key: len(value) for key, value in artifact.get("value", {}).items()
                                     if isinstance(value, (dict, list))} if isinstance(artifact.get("value"), dict) else {},
            }
        return {
            "step": self.step,
            "dataset": self.dataset,
            "constraints": self.constraints,
            "checklist": self.checklist,
            "history": history,
            "artifacts": artifact_view,
            "budget": self.budget,
            "missing_goals": self.missing_goals,
        }


class ReActExecutor:
    """Run a graph query through a model/tool/observation loop."""

    ACTION_TYPES = {"algorithm", "graph_query", "numeric_analysis", "finish"}

    def __init__(
        self,
        *,
        reasoner: Any,
        computing_engine: Any,
        dependency_resolver: Any = None,
        global_graph: Optional[GraphData] = None,
        current_graph_dataset: Any = None,
        expert_search_engine: Any = None,
        config: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        trace_store: Optional[ReactTraceStore] = None,
    ):
        self.reasoner = reasoner
        self.computing_engine = computing_engine
        self.dependency_resolver = dependency_resolver
        self.global_graph = global_graph
        self.current_graph_dataset = current_graph_dataset
        self.expert_search_engine = expert_search_engine
        self.config = config or {}
        self.callback = callback
        self.trace_store = trace_store
        self._state: Optional[RunState] = None

        self.max_steps = max(1, int(self.config.get("max_steps", 8)))
        self.max_retries_per_action = max(0, int(self.config.get("max_retries_per_action", 2)))
        self.observation_chars = max(500, int(self.config.get("observation_chars", 6000)))
        self.timeout_seconds = max(1, float(self.config.get("timeout_seconds", 300)))
        configured_limit = self.config.get('available_tools_limit')
        self.available_tools_limit = max(1, int(configured_limit)) if configured_limit is not None else None

        self.events: List[Dict[str, Any]] = []
        self._event_counter = 0
        self._artifact_counter = 0
        self._action_attempts: Dict[str, int] = {}
        self._seen_actions: Set[str] = set()

    async def run(
        self,
        goal: str,
        *,
        dataset: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        graph_schema: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = RunState(goal=goal, dataset=dataset, constraints=list(constraints or []))
        state.run_id = uuid.uuid4().hex
        state.status = "running"
        state.termination_reason = ""
        self._state = state
        self.events.clear()
        self._action_attempts.clear()
        self._seen_actions.clear()
        self._event_counter = self._artifact_counter = 0
        self._record("start", {"goal": goal, "dataset": dataset, "run_id": state.run_id})
        stop_reason = "step_limit"
        for step in range(1, self.max_steps + 1):
            state.step = step
            state.budget = {
                "remaining_steps": self.max_steps - step,
                "remaining_seconds": max(0, round(self.timeout_seconds - (time.monotonic() - state.started_at), 1)),
            }
            if self._timed_out(state):
                stop_reason = "timeout"
                break
            self._emit_status("deciding", step)
            try:
                decision_call = self.reasoner.react_decide(
                    goal=goal, state=state.prompt_view(self.observation_chars),
                    available_tools=self.available_tools(), graph_schema=graph_schema,
                )
                decision = await decision_call if inspect.isawaitable(decision_call) else decision_call
                if not isinstance(decision, dict):
                    raise ValueError("Decision must be a JSON object")
                action = self._normalize_action(decision.get("action"), decision)
            except Exception as exc:
                self._record_observation(state, {}, {
                    "summary": f"下一步决策失败，需要修正后重试：{exc}", "success": False,
                    "error": str(exc), "evidence": [],
                }, step)
                stop_reason = "decision_failed"
                continue

            thought = str(decision.get("thought") or "").strip()
            state.checklist = self._normalize_checklist(decision.get("checklist"), state.checklist)
            state.history.append({"kind": "decision", "step": step, "action": action, "summary": thought})
            state.history = state.history[-20:]
            decision_event = self._record("decision", {
                "step": step, "thought": thought, "action": action,
                "checklist": list(state.checklist), "adapter": self._adapter_for("decision"),
            })
            for kind, event_type, content in [("react_thought", "thinking", thought or "决定下一步工具动作"),
                                              ("react_action", "result", action)]:
                self._emit({"type": event_type, "contentType": kind, "content": content,
                            "step": step, "event_id": decision_event["event_id"],
                            "thought": thought, "checklist": list(state.checklist)})

            if action["type"] == "finish":
                review = await self._review_completion(state)
                self._record("completion_review", {"step": step, **review})
                if review.get("complete") is True:
                    state.status = "completed"
                    state.missing_goals = []
                    state.checklist = []
                    stop_reason = "completed"
                    break
                state.missing_goals = list(review.get("missing_goals") or state.checklist or [goal])
                state.checklist = list(state.missing_goals)
                self._record_observation(state, action, {
                    "summary": "尚不能结束，需继续解决：" + "；".join(state.missing_goals),
                    "success": False, "error": "completion_rejected",
                    "repair_hint": review.get("repair_hint", "Use the actual schema and choose a different tool or corrected parameters."),
                    "evidence": [],
                }, step)
                stop_reason = "missing_evidence"
                continue

            signature = self._action_signature(action)
            self._action_attempts[signature] = self._action_attempts.get(signature, 0) + 1
            if self._action_attempts[signature] > self.max_retries_per_action + 1:
                self._record_observation(state, action, {
                    "summary": "相同输入已达到重试上限；请修正参数或选择其他工具。",
                    "success": False, "empty": True, "error": "duplicate_action", "evidence": [],
                }, step)
                stop_reason = "unresolved_errors"
                continue
            self._seen_actions.add(signature)
            try:
                tool_result, metadata = await self._execute_action(action, state)
                actual_error = self._execution_error(tool_result)
                self._emit_status("observing", step)
                try:
                    observation = await self._observe(goal, action, tool_result, state)
                except Exception:
                    observation = {"summary": str(preview(tool_result)), "evidence": []}
                observation["success"] = actual_error is None
                if actual_error:
                    observation["error"] = actual_error
                    observation["repair_hint"] = "Repair the arguments using the actual database schema; do not treat missing/null fields as evidence that no transactions exist."
                if isinstance(metadata, dict):
                    observation["tool"] = metadata.get("name") or metadata.get("algorithm")
                retained_raw = isinstance(tool_result, dict) and tool_result.get("algorithm_success") is True
                if (actual_error is None and self._has_result(tool_result)) or retained_raw:
                    observation["artifact_id"] = self._add_artifact(state, action, tool_result, metadata, step)
            except Exception as exc:
                observation = {"summary": f"工具执行失败，需修正后继续：{exc}", "success": False,
                               "empty": True, "error": str(exc), "evidence": []}
            self._record_observation(state, action, observation, step)
            if observation.get("artifact_id"):
                artifact = state.artifacts[observation["artifact_id"]]
                self._emit({"type": "result", "contentType": "react_artifact", "step": step,
                            "content": {**artifact, "value": preview(artifact.get("value"))}})
            stop_reason = "step_limit"

        else:
            stop_reason = "step_limit"

        if state.status != "completed":
            successes = [event for event in self.events if event["kind"] == "observation"
                         and event["data"].get("success") is True]
            state.status = "incomplete" if successes or state.artifacts else "failed"
            if not state.missing_goals:
                state.missing_goals = list(state.checklist) or [goal]
        state.termination_reason = stop_reason
        if state.status != "completed":
            self._record("stop", {"step": state.step, "status": state.status,
                                  "termination_reason": stop_reason, "missing_goals": state.missing_goals})
        report = await self._write_report(state, goal)
        stop_reason = state.termination_reason
        if state.status != "completed":
            reasons = {"timeout": "已达到本次运行时限", "step_limit": "已达到本次执行步数上限",
                       "decision_failed": "下一步决策未能恢复", "missing_evidence": "目标缺少计算证据",
                       "unresolved_errors": "工具错误尚未解决"}
            heading = "分析未完成" if state.status == "incomplete" else "分析失败"
            report = (f"# {heading}\n\n停止原因：{reasons.get(stop_reason, stop_reason)}。\n\n"
                      + "未完成目标：" + "；".join(state.missing_goals) + "\n\n" + report)
        self._record("finish", {"step": state.step, "status": state.status,
                                "termination_reason": stop_reason, "report": report})
        self._emit_status({"completed": "complete", "incomplete": "incomplete", "failed": "error"}[state.status], state.step)
        dag = GraphWorkflowDAG.from_react_trace(self.events)
        return {"analysis_result": report, "analysis_status": state.status,
                "termination_reason": stop_reason, "missing_goals": state.missing_goals,
                "run_id": state.run_id, "dag_info": dag.get_dag_info(),
                "react_trace": list(self.events), "state": state.prompt_view(self.observation_chars)}

    async def _review_completion(self, state: RunState) -> Dict[str, Any]:
        observations = [event["data"] for event in self.events if event["kind"] == "observation"]
        if not any(item.get("success") is True for item in observations):
            return {"complete": False, "missing_goals": list(state.checklist) or [state.goal],
                    "repair_hint": "No successful tool evidence exists. Correct the failing query or use numeric_analysis with global_graph."}
        reviewer = getattr(self.reasoner, "react_review_completion", None)
        if callable(reviewer):
            try:
                capability_reader = getattr(self.computing_engine, "graph_query_capabilities", None)
                schema = capability_reader().get("schema", {}) if callable(capability_reader) else {}
                reviewed = reviewer(goal=state.goal, observations=preview(observations, items=30),
                                    artifacts=state.prompt_view(self.observation_chars)["artifacts"], graph_schema=schema)
                reviewed = await reviewed if inspect.isawaitable(reviewed) else reviewed
                if isinstance(reviewed, dict):
                    return reviewed
            except Exception as exc:
                return {"complete": False, "missing_goals": ["无法核验目标是否完成"], "repair_hint": str(exc)}
        return {"complete": not state.checklist, "missing_goals": list(state.checklist)}

    @staticmethod
    def _execution_error(result: Any) -> Optional[str]:
        if not isinstance(result, dict):
            return None
        if result.get("success") is False or result.get("error"):
            return str(result.get("error") or "Tool reported failure")
        nested = result.get("result")
        if isinstance(nested, dict) and nested.get("error"):
            return str(nested["error"])
        if isinstance(nested, str) and nested.startswith(("Error:", "Error occurred", "错误：")):
            return nested
        rows = result.get("results")
        if isinstance(rows, list) and rows and all(isinstance(row, dict) and row.get("error") for row in rows):
            return "; ".join(str(row["error"]) for row in rows)
        return None

    def available_tools(self) -> List[Dict[str, Any]]:
        capability_reader = getattr(self.computing_engine, "graph_query_capabilities", None)
        capabilities = capability_reader() if callable(capability_reader) else {}
        tools = list(capabilities.get("tools", []))
        if not tools and not callable(capability_reader) and callable(getattr(self.computing_engine, "execute_graph_query", None)):
            tools.append({"id": "graph_query", "type": "graph_query", "description": "Execute a graph query"})
        from aag.computing_engine.algorithm_catalog import algorithm_tool_catalog
        runtime_reader = getattr(self.computing_engine, "algorithm_capabilities", None)
        runtime_tools = runtime_reader() if callable(runtime_reader) else []
        index = getattr(self.expert_search_engine, "algo_index", {}) or {}
        tools.extend(algorithm_tool_catalog(runtime_tools, index, self.available_tools_limit))
        tools.append({"id": "numeric_analysis", "type": "numeric_analysis",
                      "description": "Arithmetic, filtering, sorting and formatting on complete data or existing algorithm results. For topology, reachability, structural paths, communities or centrality, select a suitable registered algorithm. Do not reimplement an available graph algorithm in generated code. global_graph.vertices/edges provide vid, src/dst and properties. Use consumed aliases for input names and objective for the computation goal."})
        path_schema = temporal_path_schema(self.current_graph_dataset)
        path_required = ["center"] + [name for name in ("amount_field", "timestamp_field")
                                      if name not in path_schema["defaults"]]
        tools.append({"id": "temporal_paths", "type": "algorithm", "engine": "builtin",
            "description": "Implemented temporal flow-path algorithm on GraphData: connected directed edges, explicit time order and original-edge deduplication. Returns bounded path examples, counts unless capped, center outflows and scoped account flow totals/rankings. Use for chronological flow and accounting requests; this does not compute centrality, community partitions, generic shortest paths or proven business risk labels.",
            "required_parameters": path_required,
            "dataset_bindings": path_schema,
            "optional_parameters": [name for name in ["via", "incoming_hops", "outgoing_hops", "strict_time", "sample_size", "max_paths", "amount_field", "timestamp_field", "timestamp_unit", "node_id_field", "require_path_flag", "edge_flag_field", "node_flag_field", "flagged_nodes"] if name not in path_required],
            "parameter_schema": {"center": "canonical vertex ID or unique source identifier from node_id_field; resolved to canonical ID",
                "via": "ordered array of successive vertex IDs after center; configured source identifiers are also accepted; empty means any intermediary",
                "incoming_hops": "exact integer hops 0..2 before center, default 1", "outgoing_hops": "exact integer hops 1..3 after center, default 2; total hops<=4; these are exact lengths, not maxima",
                "strict_time": "boolean, default true", "sample_size": "integer 1..100, default 10", "max_paths": "integer 1..500000, default 100000",
                "amount_field": "actual relationship amount property", "timestamp_field": "actual relationship timestamp property",
                "timestamp_unit": "iso8601 (default), unix_s, unix_ms, unix_us or unix_ns; explicitly select the encoding for numeric timestamps",
                "node_id_field": "optional actual vertex source-identifier property; canonical vertex ID is always preserved",
                "require_path_flag": "boolean; true requires at least one flagged edge or node per path",
                "edge_flag_field": "optional actual edge screening property; positive numbers or true/yes/1 are flagged",
                "node_flag_field": "optional actual vertex screening property; positive numbers or true/yes/1 are flagged; a screening proxy only",
                "flagged_nodes": "optional array of explicitly designated canonical node IDs"}})
        return tools

    def _normalize_action(self, raw: Any, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("Decision.action must be an object")
        action_type = str(raw.get("type") or raw.get("action") or "").strip().lower()
        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"Unknown action type '{action_type}'; available: {sorted(self.ACTION_TYPES)}")
        params = raw.get("parameters")
        if not isinstance(params, dict):
            params = {}
        consumed = raw.get("consumed")
        if not isinstance(consumed, list):
            consumed = []
        return {
            "type": action_type,
            "name": str(raw.get("name") or raw.get("algorithm") or "").strip(),
            "parameters": params,
            "consumed": consumed,
            "post_processing_code": raw.get("post_processing_code"),
            "objective": str(raw.get("objective") or raw.get("question") or ""),
        }

    @staticmethod
    def _normalize_checklist(value: Any, previous: List[str]) -> List[str]:
        if not isinstance(value, list):
            return previous
        return [str(item)[:300] for item in value[:20] if str(item).strip()]

    @staticmethod
    def _action_signature(action: Dict[str, Any]) -> str:
        return json.dumps(action, ensure_ascii=False, sort_keys=True, default=str)

    def _timed_out(self, state: RunState) -> bool:
        return time.monotonic() - state.started_at >= self.timeout_seconds

    def _adapter_for(self, role: str) -> Optional[str]:
        configured = self.config.get("adapters", {}) or {}
        if role in configured:
            value = configured[role]
            if isinstance(value, dict):
                return value.get("name") or value.get("adapter") or value.get("lora_name")
            if value:
                return str(value)
        resolver = getattr(self.reasoner, "adapter_for", None)
        if callable(resolver):
            try:
                return resolver(role)
            except Exception:
                return None
        return None

    def _resolve_algorithm_name(self, requested: Any) -> str:
        """Resolve a model spelling to the registered algorithm id."""
        value = str(requested or "").strip()
        if not value:
            return value
        index = getattr(self.expert_search_engine, "algo_index", {}) or {}
        if not isinstance(index, dict) or value in index:
            return value
        normalize = lambda item: "".join(ch.lower() for ch in str(item) if ch.isalnum())
        wanted = normalize(value)
        for algorithm_id, info in index.items():
            names = [algorithm_id]
            if isinstance(info, dict):
                names.extend([info.get("algorithm"), info.get("name")])
            if any(normalize(name) == wanted for name in names if name):
                return str(algorithm_id)
        return value

    async def _execute_action(self, action: Dict[str, Any], state: RunState) -> Tuple[Any, Dict[str, Any]]:
        self._emit_status("binding", state.step)
        action_type = action["type"]
        if action_type == "algorithm" and action.get("name") == "temporal_paths":
            return await self._execute_temporal_paths(action, state)
        if action_type == "graph_query":
            parameters = dict(action["parameters"])
            query_type = parameters.pop("query_type", None) or action.get("name")
            if query_type == "subgraph_by_nodes" and state.artifacts and not action.get("consumed"):
                literal_values = parameters.get("values") or []
                if not literal_values or not all(re.search(r"(?<![A-Za-z0-9_])" + re.escape(str(value)) + r"(?![A-Za-z0-9_])", state.goal)
                                                 for value in literal_values):
                    raise ValueError("Bind the complete upstream node selection using consumed with alias=values and use_as=parameter. Do not copy preview IDs into a new group. Explicit IDs supplied in the user goal may be passed directly.")
            if action.get("consumed"):
                parameters = await self._bind_query_inputs(action, state, query_type, parameters)
            self._emit_status("executing", state.step)
            if query_type and query_type != "graph_query":
                result = self.computing_engine.execute_graph_query(query_type=query_type, parameters=parameters)
            else:
                query = parameters.get("query") or action.get("objective") or state.goal
                if str(query).lstrip().upper().startswith(("MATCH ", "WITH ", "CALL ", "RETURN ")):
                    raise ValueError("Use a structured graph_query name (node_lookup, neighbor_query, relationship_filter, subgraph) with actual schema properties; do not send raw Cypher for another model to reinterpret.")
                result = self.computing_engine.execute_graph_query(str(query))
            result = await result if inspect.isawaitable(result) else result
            self._record("tool_call", {"step": state.step, "action": action,
                                      "normalized_parameters": result.get("params", parameters)})
            if isinstance(result, dict) and "results" in result and "result" not in result:
                result = {**result, "result": {"results": result["results"], "count": result.get("count", len(result["results"]))}}
            if isinstance(result, dict):
                result["query_scope"] = {"query_type": query_type, "parameters": result.get("params", parameters)}
            return result, {"name": query_type or "graph_query", "description": "graph query"}
        if action_type == "numeric_analysis":
            return await self._execute_numeric_action(action, state)
        if action_type != "algorithm":
            return {"success": True, "result": {}}, {"name": action.get("name", "")}

        algorithm = self._resolve_algorithm_name(action.get("name"))
        if not algorithm:
            raise ValueError("algorithm action requires name")
        description_call = self.computing_engine.get_algorithm_description(algorithm)
        description_value = await description_call if inspect.isawaitable(description_call) else description_call
        tool_description, metadata = description_value
        if metadata is None:
            raise ValueError(f"algorithm '{algorithm}' is unavailable")

        base_parameters = dict(action.get("parameters") or {})
        post_code = action.get("post_processing_code")
        post_schema = {}
        # The decision normally supplies parameters. If it leaves them empty,
        # the binder adapter fills the tool
        # schema; this still happens inside the ReAct action boundary.
        if not base_parameters and not action.get("consumed"):
            extractor = getattr(self.reasoner, "extract_parameters_with_postprocess_new", None)
            if callable(extractor):
                vertex_schema, edge_schema = self._graph_schemas()
                extracted_call = extractor(
                    question=action.get("objective") or state.goal,
                    tool_description=tool_description,
                    vertex_schema=vertex_schema,
                    edge_schema=edge_schema,
                    error_history=[item for item in state.history if item.get("success") is False][-3:],
                )
                extracted = await extracted_call if inspect.isawaitable(extracted_call) else extracted_call
                if isinstance(extracted, dict):
                    base_parameters = dict(extracted.get("parameters") or {})
                    generated_post = extracted.get("post_processing_code")
                    if isinstance(generated_post, dict):
                        post_schema = generated_post.get("output_schema") or {}
                        generated_post = generated_post.get("code")
                    # Reuse a task extractor already produced with the parameters.
                    # Calls with prepared parameters generate one after execution.
                    post_code = post_code or generated_post

        bound = {"parameters": base_parameters, "graph_adapter_result": None, "errors": []}
        if self.dependency_resolver and action.get("consumed"):
            bound = await self.dependency_resolver.prepare_react_inputs(
                current_question=action.get("objective") or state.goal,
                alg_des_info=metadata,
                consumed=action.get("consumed"),
                artifacts=state.artifacts,
                base_parameters=base_parameters,
            )
        if bound.get("errors"):
            raise ValueError("Input binding failed; correct consumed references: " + "; ".join(bound["errors"]))
        await self._initialize_working_graph(bound.get("graph_adapter_result"), algorithm)
        self._emit_status("executing", state.step)
        self._record("tool_call", {"step": state.step, "action": action, "parameters": bound.get("parameters"),
                                  "post_processing_code": post_code})
        run_call = self.computing_engine.run_algorithm(
            algorithm,
            bound.get("parameters") or {},
            None,
            global_graph=self.global_graph,
        )
        result = await run_call if inspect.isawaitable(run_call) else run_call
        if self._execution_error(result) is None and isinstance(result, dict) and "result" in result:
            result, metadata = await self._postprocess_algorithm_result(
                action, state, result, metadata, tool_description, bound,
                post_code=post_code, post_schema=post_schema,
            )
        if bound.get("errors"):
            result = dict(result or {})
            result.setdefault("react_binding_warnings", bound["errors"])
        return result, metadata or {"name": algorithm, "description": tool_description}

    async def _bind_query_inputs(self, action, state, query_type, parameters):
        """Bind query arguments to complete artifacts through the shared binder."""
        if not self.dependency_resolver:
            raise ValueError("Graph-query artifact binding requires DependencyResolver")
        aliases = []
        for reference in action["consumed"]:
            if not isinstance(reference, dict) or reference.get("use_as", "parameter") != "parameter":
                raise ValueError("Graph-query consumed inputs use use_as=parameter and alias=<query parameter>")
            alias = reference.get("alias")
            if alias not in parameters:
                raise ValueError("Declare the target query parameter in parameters and use its name as the consumed alias")
            if alias in aliases:
                raise ValueError(f"Duplicate query input alias '{alias}'")
            aliases.append(alias)
        fixed = {key: value for key, value in parameters.items() if key not in aliases}
        input_schema = {"parameters": {key: {
            "type": type(parameters[key]).__name__ if parameters[key] is not None else "any",
            "description": f"Actual {query_type} argument {key}; adapt the complete source. Fixed query arguments: {json.dumps(fixed, ensure_ascii=False)}",
        } for key in aliases}}
        bound = await self.dependency_resolver.prepare_react_inputs(
            current_question=action.get("objective") or state.goal,
            alg_des_info={"name": query_type, "engine": "neo4j", "input_params": input_schema},
            consumed=action["consumed"], artifacts=state.artifacts, base_parameters=parameters,
        )
        if bound.get("errors"):
            raise ValueError("Query input binding failed: " + "; ".join(bound["errors"]))
        return {**parameters, **{key: bound["parameters"][key] for key in aliases}}

    async def _postprocess_algorithm_result(
        self, action: Dict[str, Any], state: RunState, result: Dict[str, Any], metadata: Dict[str, Any],
        tool_description: str, bound: Dict[str, Any], *, post_code: Any = None, post_schema: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract task fields from the complete result while retaining raw evidence."""
        generator = getattr(self.reasoner, "generate_algorithm_postprocessing", None)
        raw = result["result"]
        if isinstance(post_code, dict):
            post_schema = post_code.get("output_schema") or post_schema
            post_code = post_code.get("code")
        if not post_code and (not callable(generator) or isinstance(raw, (bool, int, float)) or raw is None):
            return result, metadata
        errors = []
        scope = self._derived_evidence_scope(action, state)
        if isinstance(raw, dict) and raw.get("truncated") is True:
            scope = {**scope, "source_truncated": True,
                     "restriction": "The original algorithm enumeration is truncated. Derived fields cover only retained items unless separately supported by exact source statistics."}
        vertex_schema, edge_schema = self._graph_schemas()
        attempts = self.max_retries_per_action + 1 if callable(generator) else 1
        for attempt in range(attempts):
            code = post_code if attempt == 0 else None
            schema = post_schema or {}
            try:
                if not code:
                    self._emit_status("binding", state.step)
                    request = generator(
                        goal=state.goal, objective=action.get("objective") or state.goal,
                        tool_description=tool_description, parameters=bound.get("parameters") or {},
                        result_view=self._bounded_result(result), vertex_schema=vertex_schema, edge_schema=edge_schema,
                        execution_scope={"uses_consumed_graph": bool(bound.get("graph_adapter_result")),
                                         "consumed": action.get("consumed") or [], "evidence_scope": scope},
                        error_history=errors,
                    )
                    spec = await request if inspect.isawaitable(request) else request
                    if not isinstance(spec, dict):
                        raise ValueError("Post-processing specification must be an object")
                    code = spec.get("code")
                    schema = spec.get("output_schema") or {}
                    if spec.get("required") is False and not code:
                        self._record("post_processing", {"step": state.step, "status": "not_required",
                                                          "description": spec.get("description", "")})
                        return result, metadata
                if not isinstance(code, str) or not code.strip():
                    raise ValueError("Post-processing requires executable process(data) code")
                if not isinstance(schema, dict) or not isinstance(schema.get("fields", {}), dict):
                    raise ValueError("Post-processing output_schema.fields must be an object")
                self._emit_status("executing", state.step)
                computed = self.computing_engine.execute_code(
                    code=code, data=copy.deepcopy(raw), global_graph=self.global_graph,
                    is_numeric_analysis=False, fallback_to_direct_exec=False,
                )
                computed = await computed if inspect.isawaitable(computed) else computed
                error = self._execution_error(computed)
                if error:
                    raise ValueError(error)
                value = computed.get("result") if isinstance(computed, dict) else None
                if not isinstance(value, dict):
                    raise ValueError("process(data) must return a dictionary of extracted fields")
                derived = {key: item for key, item in value.items() if key != "original_result"}
                json.dumps(derived, allow_nan=False)
                missing = set(schema.get("fields", {})) - {"original_result", *derived}
                if missing:
                    raise ValueError(f"Post-processing did not return declared fields: {sorted(missing)}")
                fields = dict(schema.get("fields", {}))
                fields["original_result"] = {"type": type(raw).__name__,
                                              "field_description": "Unmodified full algorithm output; preserve its original execution scope and any truncation flags."}
                for key, item in derived.items():
                    fields.setdefault(key, {"type": type(item).__name__, "field_description": "Field derived from the actual algorithm output."})
                output_schema = {**schema, "type": "dict", "fields": fields}
                self._record("post_processing", {"step": state.step, "status": "success", "attempt": attempt + 1,
                                                  "code": code, "output_schema": output_schema,
                                                  "derived_fields": list(derived)})
                return ({**result, "result": {"original_result": raw, **derived}, "evidence_scope": scope,
                         "output_schema": output_schema,
                         "postprocessing": {"status": "success", "derived_fields": list(derived)}},
                        {**metadata, "output_params": output_schema})
            except Exception as exc:
                detail = {"error": str(exc), "previous_code": code, "attempt": attempt + 1}
                errors.append(detail)
                self._record("post_processing", {"step": state.step, "status": "failed", **detail})
        error = "Algorithm succeeded but result extraction failed: " + errors[-1]["error"]
        return ({**result, "success": False, "algorithm_success": True, "error": error,
                 "result": {"original_result": raw}, "evidence_scope": scope,
                 "postprocessing": {"status": "failed", "error": error, "raw_result_retained": True}},
                {**metadata, "output_params": {"type": "dict", "fields": {
                    "original_result": {"type": type(raw).__name__, "field_description": "Retained algorithm output; requested derived fields were not computed."}}}})

    async def _execute_numeric_action(self, action: Dict[str, Any], state: RunState) -> Tuple[Any, Dict[str, Any]]:
        if action.get("name") == "temporal_paths":
            # Old decisions remain executable through the same implementation.
            return await self._execute_temporal_paths(action, state)
        consumed_items = []
        execution_data: Dict[str, Any] = {key: value for key, value in action.get("parameters", {}).items()
                                          if key not in {"code", "question"}}
        for key, value in execution_data.items():
            consumed_items.append({"field_key": key, "field_type": type(value).__name__,
                                   "value": preview(value), "field_desc": "Action argument"})
        for reference in action.get("consumed") or []:
            if isinstance(reference, str):
                reference = {"artifact_id": reference}
            if not isinstance(reference, dict):
                continue
            artifact = state.artifacts.get(str(reference.get("artifact_id") or ""))
            if not artifact:
                raise ValueError(f"Unknown artifact '{reference.get('artifact_id')}'")
            field_name = reference.get("field")
            value = copy.deepcopy(artifact.get("value"))
            if field_name and field_name != "value" and isinstance(value, dict):
                if field_name not in value:
                    raise ValueError(f"Artifact {reference.get('artifact_id')} has no field '{field_name}'; fields={list(value)[:30]}")
                value = value[field_name]
            key = str(reference.get("alias") or field_name or reference.get("artifact_id"))
            if not key.isidentifier():
                raise ValueError(f"Input alias '{key}' must be a valid Python parameter name")
            if key in execution_data:
                raise ValueError(f"Duplicate input '{key}'; give each consumed reference a unique alias")
            execution_data[key] = value
            consumed_items.append({"field_key": key, "value": preview(value), "field_type": type(value).__name__,
                                   "field_desc": f"Full result from {reference.get('artifact_id')}; displayed value is a preview",
                                   "evidence_scope": self._input_evidence_scope(artifact, field_name)})

        code = action.get("parameters", {}).get("code")
        output_schema: Dict[str, Any] = {}
        if not code:
            generator = getattr(self.reasoner, "generate_numeric_analysis_code", None)
            if not callable(generator):
                raise ValueError("numeric_analysis requires parameters.code")
            vertex_schema, edge_schema = self._graph_schemas()
            generated = generator(
                question=action.get("objective") or action.get("parameters", {}).get("question") or state.goal,
                dependency_items=consumed_items,
                vertex_schema=vertex_schema,
                edge_schema=edge_schema,
                error_history=self._numeric_error_history(state),
            )
            numeric = generated.get("numeric_analysis_code", {}) if isinstance(generated, dict) else {}
            code = numeric.get("code")
            output_schema = numeric.get("output_schema", {}) or {}
        if not code:
            raise ValueError("numeric_analysis generated empty code")
        self._emit_status("executing", state.step)
        self._record("tool_call", {"step": state.step, "action": action, "code": code,
                                  "input_names": list(execution_data)})
        result = self.computing_engine.execute_code(
            code=code,
            data=execution_data,
            global_graph=self.global_graph,
            is_numeric_analysis=True,
            fallback_to_direct_exec=False,
        )
        if output_schema:
            result = dict(result or {})
            result["output_schema"] = output_schema
        evidence_scope = self._derived_evidence_scope(action, state)
        if isinstance(result, dict) and isinstance(result.get("evidence_scope"), dict):
            evidence_scope.update(result["evidence_scope"])
        if isinstance(result, dict) and evidence_scope:
            result = {**result, "evidence_scope": evidence_scope}
        return result, {"name": "numeric_analysis", "description": "numeric analysis"}

    async def _execute_temporal_paths(self, action: Dict[str, Any], state: RunState) -> Tuple[Any, Dict[str, Any]]:
        from aag.computing_engine.temporal_paths import analyze_temporal_paths
        if self.global_graph is None:
            raise ValueError("Load a graph before temporal path analysis")
        parameters = bind_temporal_path_parameters(self.current_graph_dataset, action.get("parameters", {}))
        self._emit_status("executing", state.step)
        self._record("tool_call", {"step": state.step, "action": action, "source": "temporal_paths",
                                  "parameters": parameters})
        result = analyze_temporal_paths(self.global_graph, **parameters)
        return {"success": True, "result": result}, {"name": "temporal_paths"}

    @staticmethod
    def _input_evidence_scope(artifact: Dict[str, Any], field_name: Optional[str]) -> Dict[str, Any]:
        scope = artifact.get("evidence_scope", {})
        # Selecting a known complete field does not inherit a sibling sample's
        # limitation. Arbitrary generated fields keep conservative lineage.
        if field_name and field_name != "value" and scope.get("authoritative_fields"):
            if field_name in scope["authoritative_fields"]:
                return {}
        return scope

    def _derived_evidence_scope(self, action: Dict[str, Any], state: RunState) -> Dict[str, Any]:
        sources = {}
        query_inputs = {}
        for reference in action.get("consumed") or []:
            reference = {"artifact_id": reference} if isinstance(reference, str) else reference
            if not isinstance(reference, dict):
                continue
            artifact = state.artifacts.get(str(reference.get("artifact_id")), {})
            scope = self._input_evidence_scope(artifact, reference.get("field"))
            query_inputs.update(scope.get("query_inputs", {}))
            if scope.get("query_scope"):
                query_inputs[str(reference.get("artifact_id"))] = {
                    "scope": scope["query_scope"], "stored_collections": scope.get("stored_collections", {}),
                    "source_truncated": scope.get("source_truncated", False),
                }
            for source in scope.get("example_sources", []):
                sources[(source["artifact_id"], source["field"])] = source
        if not sources:
            return {"query_inputs": query_inputs} if query_inputs else {}
        return {**({"query_inputs": query_inputs} if query_inputs else {}), "example_sources": list(sources.values()),
                "restriction": "These inputs contain example collections. Formatting, grouping, deduplicating or renaming examples does not establish population counts, complete path-pattern counts or whole-population value ranges. Use the original source artifact for full-scope statistics. Derived exact statistics copied from that source retain its original scope."}

    def _numeric_error_history(self, state: RunState) -> List[Dict[str, Any]]:
        failures = [dict(item) for item in state.history if item.get("success") is False][-3:]
        previous_code = next((event["data"].get("code") for event in reversed(self.events)
                              if event["kind"] == "tool_call" and event["data"].get("code")), None)
        if failures and previous_code:
            failures[-1]["previous_code"] = previous_code
        return failures

    async def _initialize_working_graph(self, adapter_result: Optional[Dict[str, Any]], algorithm: Optional[str] = None) -> None:
        if self.global_graph is None:
            return
        vertices = self.global_graph.vertices
        edges = self.global_graph.edges
        if adapter_result:
            nodes = list(dict.fromkeys(str(n) for n in (adapter_result.get("nodes") or [])))
            pairs = []
            seen = set()
            for edge in adapter_result.get("edges") or []:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    pair = (str(edge[0]), str(edge[1]))
                    if pair not in seen:
                        seen.add(pair)
                        pairs.append(pair)
            if nodes or pairs:
                vertices, edges = reconstruct_graph(nodes, pairs, vertices, edges)
        runner = getattr(self.computing_engine, "run_algorithm", None)
        if not callable(runner):
            return
        payload: Dict[str, Any] = {
            "vertices": [v.to_dict() for v in vertices or []],
            "edges": [e.to_dict() for e in edges or []],
        }
        if self.current_graph_dataset and hasattr(self.current_graph_dataset, "to_dict"):
            payload["dataset_config"] = self.current_graph_dataset.to_dict()
        initializer = "initialize_graph"
        mappings = getattr(self.computing_engine, "algorithm_tool_mapping", {}) or {}
        if algorithm and isinstance(mappings, dict):
            for mapping in mappings.values():
                if algorithm in mapping:
                    initializer = next((name for name, tool in mapping.items() if tool == "run_initialize_graph"), None)
                    if not initializer:
                        raise ValueError(f"No graph initializer is registered for the engine running '{algorithm}'")
                    break
        run_call = runner(initializer, payload)
        result = await run_call if inspect.isawaitable(run_call) else run_call
        if isinstance(result, dict) and not result.get("success", False):
            raise RuntimeError(result.get("error") or "failed to initialize working graph")

    def _graph_schemas(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.global_graph is None:
            return {}, {}
        try:
            return (
                self.global_graph.get_vertex_properties_schema(),
                self.global_graph.get_edge_properties_schema(),
            )
        except (IndexError, AttributeError, TypeError):
            return {}, {}

    async def _observe(self, goal: str, action: Dict[str, Any], result: Any, state: RunState) -> Dict[str, Any]:
        observer = getattr(self.reasoner, "react_observe", None)
        if callable(observer):
            try:
                observation_call = observer(
                    goal=goal, action=action, tool_result=self._bounded_result(result),
                    artifacts={key: {"summary": value.get("summary", ""), "evidence_scope": value.get("evidence_scope", {})}
                               for key, value in state.artifacts.items()},
                )
                observation = await observation_call if inspect.isawaitable(observation_call) else observation_call
                if isinstance(observation, dict):
                    return observation
            except Exception:
                logger.warning("Observer failed; preserving the actual tool result", exc_info=True)
        return {
            "summary": str(self._bounded_result(result))[: self.observation_chars],
            "evidence": [],
            "usable_fields": list(result.keys()) if isinstance(result, dict) else [],
            "empty": not self._has_result(result),
        }

    async def _write_report(self, state: RunState, goal: str) -> str:
        self._emit_status("reporting", state.step)
        writer = getattr(self.reasoner, "react_write_report", None)
        if callable(writer):
            try:
                report_arguments = dict(
                    goal=goal,
                    observations=[
                        {"observation_id": event.get("event_id"),
                         **{key: value for key, value in event.get("data", {}).items()
                            if key in {"summary", "success", "error", "evidence", "artifact_id"}}}
                        for event in self.events
                        if event.get("kind") == "observation"
                    ],
                    artifacts={
                        key: {
                            "source": value.get("source"),
                            "fields": value.get("fields", {}),
                            "sample": preview(value.get("value")),
                            "evidence_scope": value.get("evidence_scope", {}),
                            "preview_is_truncated": is_sampled(value.get("value")),
                            **collection_metadata(value.get("value")),
                            "collection_sizes": {name: len(items) for name, items in value.get("value", {}).items()
                                                 if isinstance(items, (list, dict))} if isinstance(value.get("value"), dict) else {},
                        }
                        for key, value in state.artifacts.items()
                    },
                    execution_outcome={"status": state.status, "termination_reason": state.termination_reason,
                                       "missing_goals": state.missing_goals},
                )
                reviewer = getattr(self.reasoner, "react_review_report", None)
                for attempt in range(3):
                    report_call = writer(**report_arguments)
                    report = await report_call if inspect.isawaitable(report_call) else report_call
                    if not isinstance(report, str) or not report.strip():
                        raise ValueError("The report writer returned no report")
                    if not callable(reviewer):
                        break
                    review_call = reviewer(goal=goal, report=report, artifacts=report_arguments["artifacts"])
                    review = await review_call if inspect.isawaitable(review_call) else review_call
                    if not isinstance(review, dict) or not isinstance(review.get("valid"), bool):
                        raise ValueError("Report review must return a boolean valid field")
                    self._record("report_review", {"step": state.step, "attempt": attempt + 1, **review})
                    if review["valid"]:
                        break
                    if attempt == 2:
                        raise ValueError("Report evidence review failed: " + str(review.get("issues") or review))
                    report_arguments["execution_outcome"] = {
                        **report_arguments["execution_outcome"],
                        "report_correction": {"previous_report": report, "review": review},
                    }
                report_text = report
                evidence_ids = list(state.artifacts.keys()) + [
                    str(event.get("event_id"))
                    for event in self.events
                    if event.get("kind") == "observation"
                ]
                if evidence_ids and not any(item in report_text for item in evidence_ids):
                    report_text += "\n\n证据索引：" + ", ".join(f"[{item}]" for item in evidence_ids)
                return report_text
            except Exception as exc:
                logger.warning("ReAct report writer failed: %s", exc)
                state.status = "incomplete"
                state.termination_reason = "report_failed"
                state.missing_goals = list(state.missing_goals) + ["生成最终分析报告"]
                self._record("error", {"step": state.step, "stage": "reporting", "error": str(exc)})
        return "未能生成分析报告；请检查执行记录中的观测和错误。"

    def _add_artifact(self, state: RunState, action: Dict[str, Any], result: Any, metadata: Any, step: int) -> str:
        self._artifact_counter += 1
        artifact_id = f"a{self._artifact_counter}"
        raw_result = result.get("result") if isinstance(result, dict) and "result" in result else result
        fields = {}
        if isinstance(metadata, dict):
            output_schema = metadata.get("output_params") or metadata.get("output_schema") or {}
            fields = output_schema.get("fields", {}) if isinstance(output_schema, dict) else {}
        if not fields and isinstance(result, dict) and isinstance(result.get("output_schema"), dict):
            fields = result["output_schema"].get("fields", {})
        if not fields and isinstance(raw_result, dict):
            fields = {key: {"type": type(value).__name__} for key, value in raw_result.items()}
        evidence_scope = self._derived_evidence_scope(action, state)
        if isinstance(result, dict) and isinstance(result.get("evidence_scope"), dict):
            evidence_scope.update(result["evidence_scope"])
        if isinstance(result, dict) and result.get("query_scope"):
            evidence_scope.update({"query_scope": result["query_scope"],
                                   "stored_collections": collection_metadata(raw_result)["nested_collection_sizes"],
                                   "membership_rule": "Use complete stored collections for membership. Preview IDs alone do not establish an exhaustive node set or community."})
            rows = raw_result.get("results", []) if isinstance(raw_result, dict) else []
            if any(isinstance(row, dict) and (row.get("scope_is_complete") is False or row.get("path_limit_reached")) for row in rows):
                evidence_scope["source_truncated"] = True
        if isinstance(raw_result, dict) and raw_result.get("sample_paths_are_examples_only") is True:
            evidence_scope = {
                "example_sources": [{"artifact_id": artifact_id, "field": "sample_paths",
                                     "example_count": len(raw_result.get("sample_paths", [])),
                                     "full_path_count": raw_result.get("path_count")}],
                "authoritative_fields": [key for key in raw_result if key != "sample_paths"],
                "restriction": "sample_paths are examples, not a full population. Derived example counts/groupings/ranges remain sample-only. Exact path-pattern counts were not computed. Use this source's declared scope for counts, values and rankings.",
            }
        state.artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "source": action.get("name") or action.get("type"),
            "source_step_id": step,
            "question": action.get("objective") or state.goal,
            "value": raw_result,
            "fields": fields,
            "summary": str(preview(raw_result))[: self.observation_chars],
            "evidence_scope": evidence_scope,
        }
        return artifact_id

    def _record_observation(self, state: RunState, action: Dict[str, Any], observation: Dict[str, Any], step: int) -> None:
        observation = dict(observation or {})
        observation["step"] = step
        observation["action"] = action
        observation["adapter"] = self._adapter_for("observer")
        artifact_id = observation.get("artifact_id")
        if artifact_id and artifact_id in state.artifacts:
            artifact = state.artifacts[artifact_id]
            observation["artifact"] = {
                "artifact_id": artifact_id,
                "source": artifact.get("source"),
                "fields": artifact.get("fields", {}),
                "summary": artifact.get("summary", ""),
                "evidence_scope": artifact.get("evidence_scope", {}),
            }
        self._record("observation", observation)
        state.history.append(observation)
        state.history = state.history[-20:]
        self._emit({
            "type": "result",
            "contentType": "react_observation",
            "content": observation,
            "step": step,
        })

    def _record(self, kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        self._event_counter += 1
        event = {
            "event_id": f"e{self._event_counter}",
            "kind": kind,
            "data": data,
            "timestamp": time.time(),
            "run_id": self._state.run_id if self._state else None,
        }
        self.events.append(event)
        if self.trace_store and self._state:
            try:
                self.trace_store.append(self._state.run_id, event)
                if kind in {"start", "observation", "finish"}:
                    self.trace_store.save(self._state.run_id, {
                        "run_id": self._state.run_id, "goal": self._state.goal, "dataset": self._state.dataset,
                        "status": self._state.status, "termination_reason": self._state.termination_reason,
                        "missing_goals": self._state.missing_goals, "checklist": self._state.checklist,
                        "started_at": self._state.started_at_wall,
                        "finished_at": time.time() if kind == "finish" else None,
                        "report": data.get("report") if kind == "finish" else None,
                        "events": self.events, "artifacts": self._state.artifacts,
                    })
            except OSError:
                logger.warning("Could not persist ReAct record", exc_info=True)
        logger.info("ReAct run=%s event=%s step=%s status=%s", event["run_id"], kind,
                    data.get("step"), data.get("status", data.get("success", "")))
        return event

    def _emit_status(self, stage: str, step: int) -> None:
        self._emit({
            "type": "result", "contentType": "react_status", "step": step,
            "content": {"stage": stage, "max_steps": self.max_steps,
                        "run_id": self._state.run_id if self._state else None,
                        "outcome": self._state.status if self._state else "running",
                        "termination_reason": self._state.termination_reason if self._state else "",
                        "missing_goals": self._state.missing_goals if self._state else []},
        })

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.callback:
            try:
                self.callback(payload)
            except Exception:
                logger.debug("ReAct callback failed", exc_info=True)

    @staticmethod
    def _has_result(result: Any) -> bool:
        if result is None:
            return False
        if isinstance(result, dict):
            if result.get("success") is False:
                return False
            value = result.get("result", result)
            return value is not None and value != {} and value != []
        return result != {} and result != []

    @staticmethod
    def _bounded_result(result: Any) -> Any:
        """Keep status/result/error fields when sampling tool output."""
        if not isinstance(result, dict):
            return take_sample(result)
        output: Dict[str, Any] = {}
        for key in ("success", "result", "results", "error", "summary", "algorithm", "query_type", "count", "params", "evidence_scope", "postprocessing", "algorithm_success", "output_schema", "query_scope"):
            if key in result:
                value = result[key]
                output[key] = preview(value) if key in {"result", "results", "params", "output_schema"} else value
                if key == "result" and isinstance(value, (dict, list, tuple)):
                    output["result_is_sample"] = is_sampled(value)
                    if isinstance(value, (list, tuple)):
                        output["result_total_items"] = len(value)
                    elif isinstance(value.get("results"), list):
                        output["result_total_items"] = len(value["results"])
                    elif isinstance(value.get("count"), int):
                        output["result_total_items"] = value["count"]
                    if isinstance(value, dict):
                        output["result_field_count"] = len(value)
                        output["collection_sizes"] = {name: len(items) for name, items in value.items()
                                                      if isinstance(items, (list, dict))}
                    output.update(collection_metadata(value))
        if not output:
            return take_sample(result)
        return output
