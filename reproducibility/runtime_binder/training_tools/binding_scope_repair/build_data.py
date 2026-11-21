"""Build new graph groups and scope pairs, without reading regression answers."""
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import networkx as nx

from binding_examples import examples
from workflows import TYPES, digest, graph_edges, rank, rename, stable_value, workflow

HERE = Path(__file__).resolve().parent


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compute(algorithm, graph):
    if algorithm == "dijkstra_path_length":
        result = dict(nx.all_pairs_dijkstra_path_length(graph, weight=None))
        assert result == dict(nx.all_pairs_shortest_path_length(graph))
        return result
    if algorithm == "pagerank":
        return nx.pagerank(graph, max_iter=1000, tol=1e-9)
    return getattr(nx, algorithm)(graph)


def scope_pair(samples, w, k, variant):
    # Select a real recorded snapshot with at least one existing derived graph.
    sample = next(s for s in samples if sum(i["type"] == "Graph" for i in
                  json.loads(s["input"])["available_inputs"]) >= 2)
    req = json.loads(sample["input"])
    graph_item = next(i for i in req["available_inputs"] if i["type"] == "Graph" and i["ref"] != "G0")
    target = graph_item["ref"]
    known = {step["id"] for step in w.steps}
    parents = sorted(w.provenance[target] & known)
    scope = (f"the induced subgraph selected using the completed ranking {parents[0]}"
             if parents else f"the induced subgraph on vertex IDs 0 through {k - 1}")
    algorithm = req["step"]["algorithm"]
    question = (f"The listed artifacts were produced by earlier completed analyses. Independently compute "
                f"{algorithm} on the complete original input graph and also on {scope}. "
                "Each requested measurement uses its own stated scope.")
    phrases = [
        ("the complete original input graph", scope),
        ("the entire network supplied before any vertex filtering", scope + ", retaining its original internal edges"),
        ("未经节点筛选的完整输入网络", "根据已完成的排名 " + parents[0] + " 选出的节点诱导子图" if parents else f"节点编号为 0 到 {k-1} 的诱导子图"),
        ("the original network, regardless of any smaller graph already available", scope),
    ]
    full, selected = phrases[variant % len(phrases)]
    out = []
    for label, graph_ref, text in (("full", "G0", full), ("selected", target, selected)):
        row = deepcopy(sample)
        current = deepcopy(req)
        current["question"] = question
        current["step"]["description"] = f"Compute {algorithm} on {text}."
        current["required_steps"] = [] if graph_ref == "G0" else parents
        row["input"] = json.dumps(current, ensure_ascii=False, separators=(",", ":"))
        row["output"] = json.dumps({"input_refs": [graph_ref]}, separators=(",", ":"))
        result = compute(algorithm, w.objects[graph_ref])
        out.append((row, {"scope": label, "source_ref": graph_ref,
                          "result_sha256": digest(stable_value(result))}))
    assert json.loads(out[0][0]["input"])["available_inputs"] == json.loads(out[1][0]["input"])["available_inputs"]
    return out


def main():
    target = HERE / "new_data"
    target.mkdir(exist_ok=False)
    used_hashes = set()
    all_questions = []
    for split, n_per_type, seed_offset in (("train", 16, 930000), ("val", 2, 940000), ("test", 2, 950000)):
        rows, metadata = [], []
        for kind_idx, kind in enumerate(TYPES):
            for n in range(n_per_type):
                seed = seed_offset + kind_idx * 100 + n
                rng = random.Random(seed)
                variant = rng.randrange(6)
                nodes, edges = 1001 + rng.randrange(300), 3000 + rng.randrange(1200)
                graph = nx.gnm_random_graph(nodes, edges, seed=seed)
                graph_sha = digest({"nodes": list(graph), "edges": graph_edges(graph)})
                assert graph_sha not in used_hashes
                used_hashes.add(graph_sha)
                w, question = workflow(kind, variant, graph)
                qid = f"scope_{split}_{kind}_{n+1:03d}"
                upstream = {"rephrased_question": question, "task_1_workflow_dag": {"subtasks": w.steps, "edges": w.edges}}
                trace = {"runtime_execution_inputs": {"input_graph": {"num_nodes": nodes, "num_edges": edges,
                         "directed": False, "weighted": False, "multigraph": False, "heterogeneous": False},
                         "step_results": w.events}}
                baseline, step_ids = examples(upstream, trace)
                originals = [(s, {"case_kind": "workflow", "case_id": sid, "step_id": sid})
                             for s, sid in zip(baseline, step_ids, strict=True)]
                for s, meta in scope_pair(baseline, w, 52 + 7 * variant, n + kind_idx):
                    originals.append((s, dict(meta, case_kind="scope_pair", case_id="probe_" + meta["scope"],
                                              step_id=json.loads(s["input"])["step"]["id"])))
                aliases = {e["step_or_transform"]: f"artifact_{token}" for e, token in
                           zip(w.events, rng.sample(range(100, 9999), len(w.events)), strict=True)
                           if e["algorithm"] is None}
                # Every scope pair shares candidate identity and ordering within a variant.
                # Naming/order variants remain in the same graph/question split.
                for naming in ("original", "aliased"):
                    for order in (0, 1):
                        for sample, meta in originals:
                            row = rename(sample, aliases) if naming == "aliased" else deepcopy(sample)
                            req = json.loads(row["input"])
                            perm = list(range(len(req["available_inputs"])))
                            # The permutation seed does not depend on the correct scope.
                            random.Random(seed + order * 17003 + len(perm) * 31).shuffle(perm)
                            req["available_inputs"] = [req["available_inputs"][p] for p in perm]
                            row["input"] = json.dumps(req, ensure_ascii=False, separators=(",", ":"))
                            refs = json.loads(row["output"])["input_refs"]
                            assert all(r in {i["ref"] for i in req["available_inputs"]} for r in refs)
                            rows.append(row)
                            metadata.append(dict(meta, question_id=qid, dag_type=kind, graph_sha256=graph_sha,
                                                 seed=seed, naming=naming, permutation=order))
                all_questions.append({"question_id": qid, "split": split, "dag_type": kind, "seed": seed,
                                      "nodes": nodes, "edges": edges, "graph_sha256": graph_sha,
                                      "algorithm_steps": w.steps, "algorithm_edges": w.edges,
                                      "trace": w.events, "result_sha256": {
                                          name: digest(stable_value(obj)) for name, obj in w.objects.items() if name != "G0"}})
        write(target / f"{split}.json", {"samples": rows, "metadata": metadata})
        print(split, "independent graphs", n_per_type * 5, "samples", len(rows),
              "scope_pairs", sum(m["case_kind"] == "scope_pair" for m in metadata), flush=True)
    write(target / "manifest.json", all_questions)
    assert Counter(q["split"] for q in all_questions) == {"train": 80, "val": 10, "test": 10}


if __name__ == "__main__":
    main()
