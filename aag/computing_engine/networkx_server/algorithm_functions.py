"""JSON argument adapters for algorithms with additional graph inputs.

All calculations delegate to NetworkX. The controller injects the working graph
as G; an additional graph is supplied explicitly in the same JSON format as an
algorithm graph result (nodes/edges), or GraphData (vertices/edges).
"""

import inspect
from functools import lru_cache

import networkx as nx
from networkx.algorithms.approximation import treewidth_min_degree, treewidth_min_fill_in


def graph_argument(payload):
    if isinstance(payload, nx.Graph):
        return payload
    if not isinstance(payload, dict):
        raise ValueError('An additional graph must be an explicit nodes/edges or vertices/edges JSON object')
    from aag.utils.graph_payload import from_graph_data, to_networkx
    kind = payload.get('graph_type') or (("Multi" if payload.get('multigraph') else "") + ("DiGraph" if payload.get('directed') else "Graph"))
    if 'vertices' in payload:
        from aag.expert_search_engine.database.datatype import GraphData
        payload = from_graph_data(GraphData.from_dict(payload), kind)
    else:
        payload = {**payload, 'graph_type': kind}
    return to_networkx(payload)


def k_hop_neighbors(G, source, k=2):
    """Return nodes reachable within k hops, excluding the source.

    Parameters
    ----------
    G : graph
    source : node
        Canonical starting node ID. Directed graphs follow outgoing edges.
    k : int
        Nonnegative maximum hop count, default 2.

    Returns
    -------
    list
        All reachable node IDs within the declared hop bound, excluding source.
    """
    if not isinstance(k, int) or k < 0:
        raise ValueError('k must be a nonnegative integer')
    return [node for node in nx.single_source_shortest_path_length(G, source, cutoff=k) if node != source]


def extract_subgraph(G, nodes):
    """Return the induced subgraph on explicitly supplied canonical node IDs.

    Parameters
    ----------
    G : graph
    nodes : list
        Complete node selection; unknown IDs are rejected.

    Returns
    -------
    graph
        Induced subgraph retaining direction, attributes and parallel edges.
    """
    missing = set(nodes) - set(G)
    if missing:
        raise ValueError(f'Unknown subgraph nodes: {sorted(missing)}')
    return G.subgraph(nodes).copy()


def get_paths_between_two_nodes(G, source, target, cutoff=4):
    """Enumerate simple paths between endpoints with an explicit hop bound.

    Parameters
    ----------
    G : graph
    source : node
        Canonical starting node ID.
    target : node
        Canonical ending node ID.
    cutoff : int
        Maximum hops, default 4; must be between 1 and 16.

    Returns
    -------
    iterator
        Bounded simple node paths. Serialization marks incomplete enumeration.
    """
    if not isinstance(cutoff, int) or not 1 <= cutoff <= 16:
        raise ValueError('cutoff must be an integer from 1 to 16')
    return nx.all_simple_paths(G, source, target, cutoff=cutoff)


def is_isomorphic(G, H):
    """Test structural isomorphism between the working graph and an explicit JSON graph.

    Parameters
    ----------
    G : graph
    H : dict
        JSON graph with nodes/edges or vertices/edges and graph_type or directed/multigraph flags.

    Returns
    -------
    bool
        Structural isomorphism, ignoring node and edge attributes.
    """
    return nx.is_isomorphic(G, graph_argument(H))


def graph_edit_distance(G, H, timeout=5):
    """Compute a unit-cost graph edit distance within an explicit time budget.

    Parameters
    ----------
    G : graph
    H : dict
        Explicit JSON graph to compare against the working graph.
    timeout : float
        Seconds, greater than 0 and at most 60; default 5.

    Returns
    -------
    dict
        Distance or upper bound and an explicit statement that time-bounded search does not certify optimality.
    """
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise ValueError('timeout must be greater than 0 and at most 60 seconds')
    value = nx.graph_edit_distance(G, graph_argument(H), timeout=timeout)
    return {'distance_upper_bound': value, 'optimality_certified': False, 'timeout_seconds': timeout}


def subgraph_is_monomorphic(G, H):
    """Test whether the working host graph contains a non-induced copy of explicit JSON graph H.

    Parameters
    ----------
    G : graph
    H : dict
        Pattern graph as explicit JSON; direction and multigraph flags must match the host.

    Returns
    -------
    bool
        Whether H is structurally monomorphic to a subgraph of G.
    """
    other = graph_argument(H)
    if G.is_directed() != other.is_directed() or G.is_multigraph() != other.is_multigraph():
        raise ValueError('Host and pattern graph types must agree')
    matcher = (nx.algorithms.isomorphism.MultiDiGraphMatcher if G.is_multigraph() else nx.algorithms.isomorphism.DiGraphMatcher) if G.is_directed() else (nx.algorithms.isomorphism.MultiGraphMatcher if G.is_multigraph() else nx.algorithms.isomorphism.GraphMatcher)
    return matcher(G, other).subgraph_is_monomorphic()


def tree_isomorphism(G, H):
    """Find a tree isomorphism from the working tree to an explicit JSON tree.

    Parameters
    ----------
    G : graph
    H : dict
        Explicit JSON undirected tree.

    Returns
    -------
    list
        Corresponding node pairs, or an empty list if the trees are not isomorphic.
    """
    return nx.algorithms.isomorphism.tree_isomorphism(G, graph_argument(H))


def maximum_flow(G, s, t, capacity='capacity'):
    """Return maximum flow value and per-edge flow using capacity attributes.

    Parameters
    ----------
    G : graph
    s : node
        Source canonical node ID.
    t : node
        Sink canonical node ID.
    capacity : str
        Nonnegative edge capacity field, default capacity.

    Returns
    -------
    tuple
        Maximum flow value and complete flow dictionary.
    """
    return nx.maximum_flow(G, s, t, capacity=capacity)


def random_reference(G, niter=1, connectivity=True, seed=None):
    """Generate a degree-preserving random reference graph by edge rewiring.

    Parameters
    ----------
    G : graph
        Simple undirected graph with at least two vertex-disjoint edges.
    niter : int
        Expected rewiring attempts per edge, default 1.
    connectivity : bool
        Preserve connectedness, default true.
    seed : int
        Optional random seed.

    Returns
    -------
    graph
        Rewired graph retaining the original nodes and degree sequence.
    """
    if G.is_directed() or G.is_multigraph():
        raise ValueError('random_reference requires a simple undirected graph')
    if len(nx.max_weight_matching(G, maxcardinality=True)) < 2:
        raise ValueError('random_reference requires two vertex-disjoint edges; rewiring this graph is impossible')
    return nx.random_reference(G, niter=niter, connectivity=connectivity, seed=seed)


def _json_set(value):
    return set(value) if isinstance(value, list) else value


def is_d_separator(G, x, y, z):
    """Test d-separation of two node sets in a DAG given a conditioning set.

    Parameters
    ----------
    G : graph
    x : list
        First node set, represented as JSON node IDs.
    y : list
        Second node set.
    z : list
        Conditioning node set, which may be empty.

    Returns
    -------
    bool
        True if x and y are d-separated by z.
    """
    return nx.is_d_separator(G, _json_set(x), _json_set(y), _json_set(z))


OVERRIDES = {function.__name__: function for function in [
    k_hop_neighbors, extract_subgraph, get_paths_between_two_nodes, is_isomorphic,
    graph_edit_distance, subgraph_is_monomorphic, tree_isomorphism, maximum_flow,
    is_d_separator, random_reference, treewidth_min_fill_in, treewidth_min_degree,
]}


@lru_cache(maxsize=1)
def _scanned_functions():
    from aag.computing_engine.networkx_server.dynamic_tool_registry import MODULES_TO_SCAN
    functions={}
    for module in MODULES_TO_SCAN:
        for name,function in inspect.getmembers(module,inspect.isfunction):
            functions.setdefault(name,function)
    return functions


def resolve_function(name, scanned=None):
    """Prefer the canonical public function over colliding bipartite exports."""
    function = OVERRIDES.get(name) or getattr(nx, name, None) or (scanned if scanned is not None else _scanned_functions()).get(name)
    if not callable(function) or 'G' not in inspect.signature(function).parameters:
        return None
    backends = getattr(function, 'backends', None)
    if backends is not None and 'networkx' not in backends:
        return None
    return function
