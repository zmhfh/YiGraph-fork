"""Typed traversal results and explicit tree/induced-subgraph conversions."""
import copy
from aag.utils.graph_payload import normalize_graph, from_graph_data

TRAVERSALS = {"bfs_tree", "bfs_edges", "bfs_successors", "bfs_predecessors", "bfs_layers", "edge_bfs"}


def normalize_result(algorithm, value, parameters=None):
    algorithm = algorithm.removeprefix("run_")
    if algorithm not in TRAVERSALS:
        return value
    parameters = parameters or {}
    if algorithm == "bfs_tree":
        graph = normalize_graph(value)
        if graph is None:
            raise ValueError("bfs_tree did not return a graph")
        return {**graph, "algorithm": algorithm, "result_kind": "traversal_tree", "source": parameters.get("source"), "reverse": bool(parameters.get("reverse", False))}
    truncated = isinstance(value, dict) and value.get("truncated") is True
    values = value["items"] if truncated else value
    kind = "node_layers" if algorithm == "bfs_layers" else "parent_mapping" if algorithm in {"bfs_successors", "bfs_predecessors"} else "edge_set"
    return {"type": kind, "algorithm": algorithm, "items": copy.deepcopy(values),
            "source": parameters.get("source"), "reverse": bool(parameters.get("reverse", False)),
            "complete": not truncated, "truncated": truncated}


def traversal_graph(value, mode, source_graph, source_kind):
    if not isinstance(value, dict) or value.get("algorithm") not in TRAVERSALS:
        raise ValueError("Expected a typed traversal result")
    if value.get("complete") is not True or value.get("truncated"):
        raise ValueError("A partial traversal is not an executable complete graph")
    if mode not in {"bfs_tree", "induced_subgraph"}:
        raise ValueError("Traversal graph input requires graph_mode=bfs_tree or induced_subgraph")
    algorithm, items = value["algorithm"], value.get("items", [])
    nodes, pairs = set(), []
    if value.get("source") is not None:
        roots = value["source"] if isinstance(value["source"], (list, tuple)) else [value["source"]]
        nodes.update(str(n) for n in roots)
    if algorithm == "bfs_tree":
        graph = normalize_graph(value)
        if mode == "bfs_tree":
            return graph
        nodes.update(graph["nodes"])
    elif algorithm == "bfs_layers":
        if mode == "bfs_tree":
            raise ValueError("Layers do not identify tree parents; choose induced_subgraph")
        nodes.update(str(n) for layer in items for n in layer)
    elif algorithm == "bfs_successors":
        for parent, children in items:
            nodes.add(str(parent))
            for child in children:
                nodes.add(str(child)); pairs.append([str(parent), str(child)])
    elif algorithm == "bfs_predecessors":
        for child, parent in items:
            nodes.update([str(parent), str(child)]); pairs.append([str(parent), str(child)])
    else:
        for edge in items:
            nodes.update([str(edge[0]), str(edge[1])]); pairs.append([str(edge[0]), str(edge[1])])
    if mode == "induced_subgraph":
        original = from_graph_data(source_graph, source_kind)
        if not nodes <= set(original["nodes"]):
            raise ValueError("Traversal nodes are outside the selected source graph")
        return {**original, "nodes": sorted(nodes), "node_attributes": {n: original["node_attributes"][n] for n in nodes},
                "edges": [e for e in original["edges"] if e["src"] in nodes and e["dst"] in nodes]}
    if algorithm == "edge_bfs":
        raise ValueError("edge_bfs includes non-tree edges; it is not a BFS tree")
    # These are traversal arcs, not selected financial transactions.
    return normalize_graph({"type": "graph", "graph_type": "DiGraph", "nodes": sorted(nodes),
                            "edges": pairs, "complete": True, "result_kind": "traversal_tree"})
