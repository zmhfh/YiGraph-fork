"""Small deterministic example using YiGraph's existing graph processor."""


def run_demo():
    from yigraph import EdgeData, GraphComputationProcessor, VertexData

    processor = GraphComputationProcessor()
    processor.create_graph_from_edges(
        [VertexData(name) for name in ("A", "B", "C", "D")],
        [EdgeData("A", "B"), EdgeData("B", "C"), EdgeData("A", "C"), EdgeData("C", "D")],
        directed=True,
    )
    return {
        "shortest_path_A_to_D": processor.run_shortest_path("A", "D"),
        "pagerank": processor.run_pagerank(),
    }
