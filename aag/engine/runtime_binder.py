"""Select complete graph artifacts using the trained input_refs adapter."""
import asyncio
import copy
import json
import os
import urllib.request

from aag.utils.graph_payload import KINDS, executable_graph, kind_from_dataset, dataset_flags


class BindingError(ValueError):
    pass


def _graph(value, global_graph, graph_kind):
    if not isinstance(value, dict) or not {"nodes", "edges"} <= value.keys():
        return None
    if not isinstance(value["nodes"], (list, tuple)) or not isinstance(value["edges"], (list, tuple)):
        return None
    if value.get("complete") is False or value.get("truncated") is True:
        return None  # Preview stays visible, but is never a graph candidate.
    try:
        return executable_graph(value, global_graph, graph_kind)
    except (ValueError, TypeError, KeyError) as exc:
        raise BindingError(str(exc)) from exc


def _post(url, request):
    body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/bind", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    # Local inference must not be redirected through HTTP_PROXY.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=float(os.environ.get("YIGRAPH_BINDER_TIMEOUT", "90"))) as response:
        return json.loads(response.read(1024 * 1024))


async def bind_prepared_inputs(executor, action, state, bound):
    url = os.environ.get("YIGRAPH_BINDER_URL")
    command = os.environ.get("YIGRAPH_BINDER_COMMAND")
    if not url and not command:
        return bound
    if executor.global_graph is None:
        raise BindingError("Runtime Binder requires a loaded G0")
    graph_kind = kind_from_dataset(executor.current_graph_dataset, os.environ.get("YIGRAPH_BINDER_GRAPH_KIND"))
    if not isinstance(bound.get("parameters", {}), dict):
        raise BindingError("Resolved parameters must be an object")
    cache = getattr(state, "_runtime_binder_graphs", None)
    if cache is None:
        state._runtime_binder_graphs = cache = {}
    available = [{"ref": "G0", "type": graph_kind, "description": "当前输入的原始图"}]
    objects, artifact_refs = {"G0": None}, {}
    for aid, artifact in state.artifacts.items():
        sid = str(artifact.get("source_step_id") or aid)
        ref = "v" + sid if sid.isdigit() else str(aid)
        if ref in objects:
            ref = str(aid)
        if ref in objects:
            raise BindingError("Duplicate artifact reference")
        artifact_refs[aid] = ref
        value = artifact.get("value")
        graph = _graph(value, executor.global_graph, graph_kind)
        value_type = graph["graph_type"] if graph is not None else ("mapping_preview" if isinstance(value, dict) and len(value) > 24 else "mapping" if isinstance(value, dict) else type(value).__name__)
        available.append({"ref": ref, "type": value_type,
                          "producer": artifact.get("source") or "transform",
                          "derived_from": artifact.get("binding_input_refs") or ["G0"]})
        objects[ref] = graph if graph is not None else value
        if graph is None and isinstance(value, dict):
            for field, field_value in value.items():
                nested = _graph(field_value, executor.global_graph, graph_kind)
                if nested is not None:
                    fref = ref + "__" + str(field)
                    if fref in objects:
                        raise BindingError("Duplicate candidate graph reference")
                    available.append({"ref": fref, "type": nested["graph_type"], "producer": "transform", "derived_from": [ref]})
                    objects[fref] = nested
    sources, parents = [], []
    for reference in action.get("consumed") or []:
        reference = {"artifact_id": reference} if isinstance(reference, str) else reference
        aid = str(reference.get("artifact_id") or reference.get("id") or "")
        if aid not in artifact_refs:
            raise BindingError("Consumed artifact is unavailable: " + aid)
        sources.append(artifact_refs[aid])
        sid = state.artifacts[aid].get("source_step_id")
        if sid is not None:
            parents.append("v" + str(sid))
    if bound.get("graph_adapter_result") is not None:
        graph = _graph(bound["graph_adapter_result"], executor.global_graph, graph_kind)
        if graph is None:
            raise BindingError("Graph adapter must return a complete graph")
        ref = "working_graph" if not cache else "working_graph_" + str(len(cache) + 1)
        cache[ref] = {"value": graph, "derived_from": sources or ["G0"]}
    for ref, entry in cache.items():
        if ref in objects:
            raise BindingError("Cached graph reference collision")
        available.append({"ref": ref, "type": entry["value"]["graph_type"], "producer": "transform", "derived_from": entry["derived_from"]})
        objects[ref] = entry["value"]
    graph = executor.global_graph
    flags = dataset_flags(executor.current_graph_dataset)
    request = {"question": state.goal,
               "actual_graph_attributes": {"num_nodes": len(graph.vertices), "num_edges": len(graph.edges),
                   "directed": "DiGraph" in graph_kind, "weighted": flags.get("weighted", any("weight" in (e.properties or {}) for e in graph.edges)),
                   "multigraph": graph_kind.startswith("Multi"), "heterogeneous": flags.get("heterogeneous", False)},
               "step": {"id": "v" + str(state.step), "description": action.get("objective") or state.goal,
                        "algorithm": action.get("name")},
               "required_steps": list(dict.fromkeys(parents)), "available_inputs": available}
    if command:
        from aag.engine.runtime_binder_transport import request_local
        result = await asyncio.to_thread(request_local, request)
    else:
        result = await asyncio.to_thread(_post, url, request)
    if not isinstance(result, dict) or set(result) != {"input_refs"}:
        raise BindingError("Binder must return exactly input_refs")
    refs = result["input_refs"]
    if not isinstance(refs, list) or len(refs) != 1 or not isinstance(refs[0], str):
        raise BindingError("This bridge supports one graph input per algorithm")
    selected = next((i for i in available if i["ref"] == refs[0]), None)
    if selected is None or selected["type"] not in KINDS:
        raise BindingError("Binder selected an unavailable or non-graph input")
    selected_graph = objects[refs[0]]
    new_bound = dict(bound, graph_adapter_result=copy.deepcopy(selected_graph))
    action["_runtime_binder_input_refs"] = refs
    size = {"nodes": len(graph.vertices), "edges": len(graph.edges)} if selected_graph is None else {
        "nodes": len(selected_graph["nodes"]), "edges": len(selected_graph["edges"])}
    executor._record("runtime_binding", {"step": state.step, "input_refs": refs,
        "selected_graph": {**size, "graph_type": selected["type"]}, "candidates": available,
        "request_step": request["step"], "endpoint": "local-subprocess" if command else url})
    return new_bound
