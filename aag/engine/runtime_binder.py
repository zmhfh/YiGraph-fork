"""Graph-only runtime binding bridge for the trained input_refs adapter.

Graph and parameter conversions still belong to DataDependencyResolver.
This bridge chooses between G0 and graph objects that already exist.
"""
import asyncio
import copy
import json
import os
import urllib.request


class BindingError(ValueError):
    pass


def _graph(value, global_graph, graph_kind):
    if not isinstance(value, dict) or not {"nodes", "edges"} <= value.keys():
        return None
    if not isinstance(value["nodes"], (list, tuple)) or not isinstance(value["edges"], (list, tuple)):
        # Some ordinary result dictionaries contain scalar node/edge counts.
        return None
    nodes = [str(n) for n in value["nodes"]]
    node_set = set(nodes)
    if len(nodes) != len(node_set):
        raise BindingError("Graph contains duplicate nodes")
    original = {str(v.vid) for v in global_graph.vertices}
    if not node_set <= original:
        raise BindingError("Bound graph contains vertices outside G0")
    canonical = (lambda a, b: tuple(sorted((a, b)))) if graph_kind == "Graph" else (lambda a, b: (a, b))
    source = {}
    for edge in global_graph.edges:
        pair = (str(edge.src), str(edge.dst))
        key = canonical(*pair)
        if key in source:
            raise BindingError("Parallel edges require a separate MultiGraph binding contract")
        source[key] = pair
    pairs, seen = [], set()
    for edge in value["edges"]:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            raise BindingError("A graph edge must contain two endpoints")
        a, b = str(edge[0]), str(edge[1])
        key = canonical(a, b)
        if a not in node_set or b not in node_set or key not in source:
            raise BindingError("Bound graph contains unavailable edges or endpoints")
        if key not in seen:
            pairs.append(list(source[key]))
            seen.add(key)
    return {"nodes": nodes, "edges": pairs, "_runtime_binder_bound": True}


def _post(url, request):
    body = json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    req = urllib.request.Request(url.rstrip("/") + "/bind", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=float(os.environ.get("YIGRAPH_BINDER_TIMEOUT", "90"))) as response:
        return json.loads(response.read(1024 * 1024))


async def bind_prepared_inputs(executor, action, state, bound):
    url = os.environ.get("YIGRAPH_BINDER_URL")
    local_command = os.environ.get("YIGRAPH_BINDER_COMMAND")
    if not url and not local_command:
        return bound
    if executor.global_graph is None:
        raise BindingError("Runtime Binder requires a loaded G0")
    graph_kind = os.environ.get("YIGRAPH_BINDER_GRAPH_KIND")
    if graph_kind not in ("Graph", "DiGraph"):
        raise BindingError("Set YIGRAPH_BINDER_GRAPH_KIND=Graph or DiGraph for the actual dataset")
    _graph({"nodes": [], "edges": []}, executor.global_graph, graph_kind)
    if not isinstance(bound.get("parameters", {}), dict):
        raise BindingError("Resolved parameters must be an object")
    cache = getattr(state, "_runtime_binder_graphs", None)
    if cache is None:
        cache = {}
        state._runtime_binder_graphs = cache
    available = [{"ref": "G0", "type": graph_kind, "description": "当前输入的原始图"}]
    objects = {"G0": None}
    artifact_refs = {}
    for aid, artifact in state.artifacts.items():
        sid = str(artifact.get("source_step_id") or aid)
        ref = "v" + sid if sid.isdigit() else str(aid)
        if ref in objects:
            ref = str(aid)
        artifact_refs[aid] = ref
        value = artifact.get("value")
        graph = _graph(value, executor.global_graph, graph_kind)
        value_type = graph_kind if graph is not None else ("mapping_preview" if isinstance(value, dict) and len(value) > 24 else "mapping" if isinstance(value, dict) else type(value).__name__)
        available.append({"ref": ref, "type": value_type,
                          "producer": artifact.get("source") or "transform",
                          "derived_from": artifact.get("binding_input_refs") or ["G0"]})
        objects[ref] = graph if graph is not None else value
        # Graph fields are addressable too, without exposing their full data to the model.
        if graph is None and isinstance(value, dict):
            for field, field_value in value.items():
                nested = _graph(field_value, executor.global_graph, graph_kind)
                if nested is not None:
                    fref = ref + "__" + str(field)
                    if fref in objects:
                        raise BindingError("Duplicate candidate graph reference")
                    available.append({"ref": fref, "type": graph_kind, "producer": "transform", "derived_from": [ref]})
                    objects[fref] = nested
    consumed = action.get("consumed") or []
    sources = []
    parents = []
    for item in consumed:
        item = {"artifact_id": item} if isinstance(item, str) else item
        aid = str(item.get("artifact_id") or item.get("id") or "")
        if aid not in artifact_refs:
            raise BindingError("Consumed artifact is unavailable: " + aid)
        sources.append(artifact_refs[aid])
        sid = state.artifacts[aid].get("source_step_id")
        if sid is not None:
            parents.append("v" + str(sid))
    prepared = bound.get("graph_adapter_result")
    if prepared is not None:
        graph = _graph(prepared, executor.global_graph, graph_kind)
        if graph is None:
            raise BindingError("Graph adapter must return complete nodes and edges")
        # Cache only completed transformations, not graph predictions or previews.
        key = "working_graph" if not cache else "working_graph_" + str(len(cache) + 1)
        cache[key] = {"value": graph, "derived_from": list(dict.fromkeys(["G0"] + sources))}
    for ref, entry in cache.items():
        if ref in objects:
            raise BindingError("Cached graph reference collision")
        available.append({"ref": ref, "type": graph_kind, "producer": "transform", "derived_from": entry["derived_from"]})
        objects[ref] = entry["value"]
    graph = executor.global_graph
    request = {"question": state.goal,
               "actual_graph_attributes": {"num_nodes": len(graph.vertices), "num_edges": len(graph.edges),
                   "directed": graph_kind == "DiGraph", "weighted": any("weight" in (e.properties or {}) for e in graph.edges),
                   "multigraph": False, "heterogeneous": False},
               "step": {"id": "v" + str(state.step), "description": action.get("objective") or state.goal,
                        "algorithm": action.get("name")},
               "required_steps": list(dict.fromkeys(parents)), "available_inputs": available}
    if len({i["ref"] for i in available}) != len(available):
        raise BindingError("Duplicate input references")
    if local_command:
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
    if selected is None or selected["type"] != graph_kind:
        raise BindingError("Binder selected an unavailable or non-graph input")
    new_bound = dict(bound, graph_adapter_result=copy.deepcopy(objects[refs[0]]))
    action["_runtime_binder_input_refs"] = refs
    selected_graph = objects[refs[0]]
    size = {"nodes": len(graph.vertices), "edges": len(graph.edges)} if selected_graph is None else {
        "nodes": len(selected_graph["nodes"]), "edges": len(selected_graph["edges"])}
    executor._record("runtime_binding", {"step": state.step, "input_refs": refs,
        "selected_graph": size, "candidates": available, "request_step": request["step"],
        "endpoint": "local-subprocess" if local_command else url})
    return new_bound
