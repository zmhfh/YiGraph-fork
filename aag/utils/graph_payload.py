"""Lossless graph envelopes shared by tools, dependency adapters and execution."""
import copy
import json
import math

KINDS = {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}


def dataset_flags(dataset):
    value = dataset.to_dict() if hasattr(dataset, "to_dict") else dataset
    schema = value.get("schema", {}) if isinstance(value, dict) else {}
    raw = schema.get("graph_structure", schema.get("graph", {}))
    flags = {}
    for key in ("directed", "multigraph", "weighted", "heterogeneous"):
        if key not in raw:
            continue
        flag = raw[key]
        if isinstance(flag, str) and flag.lower() in {"true", "false"}:
            flag = flag.lower() == "true"
        if not isinstance(flag, bool):
            raise ValueError(f"Dataset flag {key} must be a boolean")
        flags[key] = flag
    return flags


def _valid_key(key):
    if not isinstance(key, (str, int, float, bool)) or isinstance(key, float) and not math.isfinite(key):
        raise ValueError("Transaction key must be a finite JSON scalar")
    return key


def kind_from_dataset(dataset, fallback=None):
    flags = dataset_flags(dataset)
    if "directed" in flags and "multigraph" in flags:
        return ("Multi" if flags["multigraph"] else "") + ("DiGraph" if flags["directed"] else "Graph")
    if fallback not in KINDS:
        raise ValueError("Graph type is required; supply dataset graph metadata")
    return fallback


def edge_records(edges, kind):
    """Keep explicit keys; allocate deterministic, collision-free keys if missing."""
    multigraph = kind.startswith("Multi")
    directed = "DiGraph" in kind
    def pair(u, v):
        return (u, v) if directed else tuple(sorted((u, v)))
    reserved = set()
    for e in edges:
        key = e.rank
        if multigraph and key is not None:
            identity = (pair(str(e.src), str(e.dst)), _valid_key(key))
            if identity in reserved:
                raise ValueError("Duplicate explicit edge key; transaction identity would be lost")
            reserved.add(identity)
    result = []
    seen = set()
    for index, e in enumerate(edges):
        u, v = str(e.src), str(e.dst)
        key = e.rank
        if multigraph and key is None:
            key = "__edge_" + str(index)
            while (pair(u, v), key) in reserved:
                key += "_"
        identity = (pair(u, v), key) if multigraph else pair(u, v)
        if identity in seen:
            raise ValueError("Parallel edges cannot be loaded into a simple graph without an explicit projection")
        seen.add(identity)
        item = {"src": u, "dst": v, "properties": copy.deepcopy(e.properties or {})}
        if multigraph:
            item["key"] = key
        item["edge_id"] = json.dumps([u, v, key], ensure_ascii=False, separators=(",", ":"))
        result.append(item)
    return result


def from_graph_data(graph, kind):
    return {"type": "graph", "graph_type": kind, "nodes": [str(v.vid) for v in graph.vertices],
            "node_attributes": {str(v.vid): copy.deepcopy(v.properties or {}) for v in graph.vertices},
            "edges": edge_records(graph.edges, kind), "complete": True}


def normalize_graph(value, default_kind=None):
    if not isinstance(value, dict) or not {"nodes", "edges"} <= value.keys():
        return None
    if not isinstance(value["nodes"], (list, tuple)) or not isinstance(value["edges"], (list, tuple)):
        return None  # Ordinary statistics may contain numeric nodes/edges fields.
    if value.get("complete") is False or value.get("truncated") is True:
        raise ValueError("A truncated graph preview is not an executable graph")
    kind = value.get("graph_type") or default_kind
    if kind not in KINDS:
        raise ValueError("Unknown graph_type")
    nodes = [str(n) for n in value["nodes"]]
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate graph vertices")
    node_set, edges, identities = set(nodes), [], set()
    multi, directed = kind.startswith("Multi"), "DiGraph" in kind
    for raw in value["edges"]:
        if isinstance(raw, dict):
            u, v = raw.get("src", raw.get("source")), raw.get("dst", raw.get("target"))
            if u is None or v is None:
                raise ValueError("An edge needs src and dst")
            properties = raw.get("properties")
            if properties is None:
                properties = {k: copy.deepcopy(v) for k, v in raw.items() if k not in {"src", "dst", "source", "target", "key", "rank", "edge_id"}}
            item = {"src": str(u), "dst": str(v), "properties": copy.deepcopy(properties)}
            if "key" in raw or "rank" in raw:
                item["key"] = raw.get("key", raw.get("rank"))
            if "edge_id" in raw:
                item["edge_id"] = raw["edge_id"]
        elif isinstance(raw, (list, tuple)) and len(raw) in {2, 3, 4}:
            item = {"src": str(raw[0]), "dst": str(raw[1]), "properties": {}}
            if multi and len(raw) >= 3:
                item["key"] = raw[2]
                if len(raw) == 4:
                    item["properties"] = copy.deepcopy(raw[3])
            elif not multi and len(raw) == 3 and isinstance(raw[2], dict):
                item["properties"] = copy.deepcopy(raw[2])
            elif len(raw) != 2:
                raise ValueError("Invalid edge tuple; use (src,dst,key,properties) for a multigraph")
        else:
            raise ValueError("Unsupported edge record")
        if item["src"] not in node_set or item["dst"] not in node_set:
            raise ValueError("Additional graph edge refers to an undeclared node")
        if not isinstance(item["properties"], dict):
            raise ValueError("Edge properties must be an object")
        if multi and ("key" not in item or item["key"] is None):
            raise ValueError("MultiGraph edges must retain their transaction key")
        uv = (item["src"], item["dst"]) if directed else tuple(sorted((item["src"], item["dst"])))
        identity = (uv, _valid_key(item["key"])) if multi else uv
        if identity in identities:
            raise ValueError("Duplicate edge identity")
        identities.add(identity)
        edges.append(item)
    return {**copy.deepcopy(value), "type": "graph", "graph_type": kind, "nodes": nodes,
            "node_attributes": {str(k): copy.deepcopy(v) for k, v in value.get("node_attributes", {}).items()},
            "edges": edges, "complete": True}


def executable_graph(value, source_graph, source_kind):
    """Legacy pair-only subgraphs reference G0; typed derived graphs own their data."""
    legacy = isinstance(value, dict) and not value.get("graph_type") and value.get("type") != "graph"
    if legacy:
        original = from_graph_data(source_graph, source_kind)
        index = {}
        directed = "DiGraph" in source_kind
        def pair(u, v):
            return (u, v) if directed else tuple(sorted((u, v)))
        for edge in original["edges"]:
            index.setdefault(pair(edge["src"], edge["dst"]), []).append(edge)
        raw_edges = []
        for raw in value.get("edges", []):
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                matches = index.get(pair(str(raw[0]), str(raw[1])), [])
                if not matches:
                    raise ValueError("Referenced source edge does not exist")
                if len(matches) > 1 and value.get("edge_selection") != "all_parallel":
                    raise ValueError("Ambiguous transaction pair; supply keys or explicit all_parallel selection")
                raw_edges.extend(copy.deepcopy(matches))
            else:
                if isinstance(raw, dict):
                    u, v = raw.get("src", raw.get("source")), raw.get("dst", raw.get("target"))
                    key = raw.get("key", raw.get("rank"))
                elif isinstance(raw, (tuple, list)) and len(raw) == 3:
                    u, v, key = raw
                else:
                    raise ValueError("Invalid source edge reference")
                matches = index.get(pair(str(u), str(v)), [])
                matches = [e for e in matches if
                           (raw.get("edge_id") == e["edge_id"] if isinstance(raw, dict) and "edge_id" in raw
                            else type(e.get("key")) == type(key) and e.get("key") == key)]
                if len(matches) != 1:
                    raise ValueError("Source edge reference must identify one exact transaction")
                raw_edges.extend(copy.deepcopy(matches))
        if any(str(n) not in original["node_attributes"] for n in value.get("nodes", [])):
            raise ValueError("Referenced node is outside the source graph")
        value = {**value, "edges": raw_edges,
                 "node_attributes": {str(n): original["node_attributes"][str(n)] for n in value.get("nodes", [])}}
    graph = normalize_graph(value, source_kind)
    if graph is None:
        raise ValueError("Expected a complete graph payload")
    return graph


def to_graph_data(payload):
    from aag.expert_search_engine.database.datatype import VertexData, EdgeData
    graph = normalize_graph(payload)
    return ([VertexData(vid=n, properties=copy.deepcopy(graph["node_attributes"].get(n, {}))) for n in graph["nodes"]],
            [EdgeData(src=e["src"], dst=e["dst"], rank=e.get("key"), properties=copy.deepcopy(e["properties"])) for e in graph["edges"]])


def dataset_for_kind(dataset, kind):
    config = copy.deepcopy(dataset.to_dict() if hasattr(dataset, "to_dict") else dataset or {})
    schema = config.setdefault("schema", {})
    name = "graph_structure" if "graph_structure" in schema else "graph"
    schema.setdefault(name, {}).update(directed="DiGraph" in kind, multigraph=kind.startswith("Multi"))
    return config


def to_networkx(payload):
    """Restore only complete envelopes; preserve isolated nodes, keys and attributes."""
    import networkx as nx
    value = normalize_graph(payload)
    if value is None:
        raise ValueError("Expected a complete graph payload")
    graph = getattr(nx, value["graph_type"])()
    graph.graph.update(copy.deepcopy(value.get("graph_attributes", {})))
    graph.add_nodes_from((n, value["node_attributes"].get(n, {})) for n in value["nodes"])
    for edge in value["edges"]:
        if graph.is_multigraph():
            graph.add_edge(edge["src"], edge["dst"], key=edge["key"])
            graph[edge["src"]][edge["dst"]][edge["key"]].update(edge["properties"])
        else:
            graph.add_edge(edge["src"], edge["dst"], **edge["properties"])
    return graph
