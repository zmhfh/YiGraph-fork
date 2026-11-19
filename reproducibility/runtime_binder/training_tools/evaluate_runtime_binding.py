"""Score held-out input bindings and replay q0750 with model-selected inputs."""

import argparse
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


REFERENCE_SHA = "691b74a24db72df84dca6267178188400f681c1c2a74366149bc00ceaab75414"
REFERENCE = {"working_nodes": 99, "working_edges": 125, "pagerank_first": 2, "maximum_finite_hops": 18}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def predict(model, parse, sample, request):
    result = {"expected": json.loads(sample["output"]), "prediction": None, "strict": False}
    query = sample["instruction"] + "\n" + json.dumps(request, ensure_ascii=False, separators=(",", ":"))
    try:
        raw = model.chat([{"role": "user", "content": query}], system=sample["system"],
                         max_new_tokens=128, do_sample=False)[0].response_text
    except Exception as error:
        return dict(result, issue="inference_error", detail=str(error))
    result["raw_response"] = raw
    try:
        obj = parse(raw)
    except (ValueError, TypeError) as error:
        return dict(result, issue="json_parse", detail=str(error))
    result["prediction"] = obj
    if not isinstance(obj, dict) or obj.keys() != {"input_refs"}:
        return dict(result, issue="wrong_fields")
    refs = obj["input_refs"]
    if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs) or len(refs) != len(set(refs)):
        return dict(result, issue="invalid_refs")
    if not set(refs) <= {item["ref"] for item in request["available_inputs"]}:
        return dict(result, issue="unavailable_ref")
    strict = obj == result["expected"]
    return dict(result, strict=strict, issue=None if strict else "wrong_binding")


def offline(model, parse, samples, metadata, output):
    if not samples or len(samples) != len(metadata):
        raise ValueError("测试样本与清单不一致")
    groups = defaultdict(list)
    results = []
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for sample, meta in zip(samples, metadata, strict=True):
            req = json.loads(sample["input"])
            if req["step"]["id"] != meta["step_id"]:
                raise ValueError("测试步骤与审核清单不一致")
            row = dict(predict(model, parse, sample, req), question_id=meta["question_id"],
                       step_id=meta["step_id"], dag_type=meta["dag_type"],
                       n_available=len(req["available_inputs"]))
            results.append(row)
            groups[meta["question_id"]].append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    ambiguous = [row for row in results if row["n_available"] > 1]
    summary = {
        "total": len(results), "passed": sum(row["strict"] for row in results),
        "multiple_candidates": {"total": len(ambiguous), "passed": sum(row["strict"] for row in ambiguous)},
        "questions": len(groups), "questions_all_passed": sum(all(row["strict"] for row in group) for group in groups.values()),
    }
    write(output / "strict_failures.json", [row for row in results if not row["strict"]])
    return summary


def execute_q0750(model, parse, samples, metadata, output):
    import networkx as nx
    import runtime
    from common import c, induced, mean, rank

    cases = {meta["step_id"]: (row, meta) for row, meta in zip(samples, metadata, strict=True)
             if meta["question_id"] == "q0750"}
    if set(cases) != {"v1", "v2", "v3"}:
        raise ValueError("测试集缺少 q0750 的三个步骤")
    source = Path(cases["v1"][1]["source_run"])
    raw_graph = (source / "input_graph.json").read_bytes()
    if hashlib.sha256(raw_graph).hexdigest() != REFERENCE_SHA or any(meta["graph_sha256"] != REFERENCE_SHA for _, meta in cases.values()):
        raise ValueError("q0750 的图与审核参考答案的哈希不同")
    graph_data = json.loads(raw_graph.decode("utf-8-sig"))
    graph = runtime.load_graph(graph_data)
    task = read(source / "input_context.json")
    dest = output / "q0750"
    dest.mkdir()
    recorder = runtime.Recorder(task, dest)
    artifacts = {"G0": graph}
    candidates = [{"ref": "G0", "type": type(graph).__name__, "description": "当前输入的原始图"}]
    decisions = []

    def bind(step_id):
        sample, _ = cases[step_id]
        request = dict(json.loads(sample["input"]), available_inputs=candidates)
        result = predict(model, parse, sample, request)
        decisions.append(dict(result, step_id=step_id))
        if result["issue"] not in (None, "wrong_binding"):
            raise ValueError(f"{step_id} 绑定无效：{result['issue']}")
        refs = result["prediction"]["input_refs"]
        if len(refs) != 1 or not isinstance(artifacts[refs[0]], nx.Graph):
            raise ValueError(f"{step_id} 需要一份图输入：{refs}")
        return artifacts[refs[0]], refs

    report = {"verified": False, "input_graph_sha256": REFERENCE_SHA, "reference": REFERENCE,
              "binding_decisions": decisions}
    try:
        bound, refs = bind("v1")
        scores = c(recorder, "v1", "clustering", bound, refs, weight=None)
        artifacts["v1"] = scores
        candidates.append({"ref": "v1", "type": "mapping", "producer": "clustering", "derived_from": refs})
        threshold = mean(scores.values())
        working = induced(recorder, "working_graph", graph, [node for node in graph if scores[node] >= threshold], ["G0", "v1"])
        artifacts["working_graph"] = working
        candidates.append({"ref": "working_graph", "type": type(working).__name__, "producer": "transform", "derived_from": ["G0", "v1"]})

        bound, refs = bind("v2")
        importance = c(recorder, "v2", "pagerank", bound, refs, alpha=0.85, weight=None, max_iter=1000, tol=1e-6)
        artifacts["v2"] = importance
        candidates.append({"ref": "v2", "type": "mapping", "producer": "pagerank", "derived_from": refs})

        bound, refs = bind("v3")
        distances = recorder.call("v3", "dijkstra_path_length", lambda: {
            u: nx.single_source_dijkstra_path_length(bound, u, weight=None) for u in bound
        }, parameters={"weight": None, "sources": "every working-graph node"}, input_refs=refs)
        actual = {"working_nodes": len(working), "working_edges": working.number_of_edges(),
                  "pagerank_first": rank(importance)[0], "maximum_finite_hops": max(max(row.values()) for row in distances.values())}
        report.update(actual=actual, verified=actual == REFERENCE and all(row["strict"] for row in decisions))
    except Exception as error:
        report["error"] = str(error)
    write(dest / "verification.json", report)
    return {"verified": report["verified"], "actual": report.get("actual"), "error": report.get("error"),
            "report": str(dest / "verification.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--executor", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--base", action="store_true", help="测试同一底座的对照结果")
    args = parser.parse_args()
    package = args.package.resolve()
    run = args.run_dir
    if run is None:
        configs = list(package.parent.glob("runtime-binding.*/binding_lora/adapter_config.json"))
        if not configs:
            raise SystemExit("找不到已完成的输入绑定 LoRA，请指定 --run-dir")
        run = max(configs, key=lambda path: path.stat().st_mtime).parents[1]
    run = run.resolve()
    adapter = run / "binding_lora"
    if not args.base and not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"找不到输入绑定 LoRA：{adapter}")

    sys.path[:0] = [str(package), str(args.executor.resolve())]
    from evaluate import parse_response
    from llamafactory.chat import ChatModel

    paths = read(package / "configs" / "paths.json")
    cfg = {"model_name_or_path": paths["model_path"], "template": paths["template"],
           "enable_thinking": paths["enable_thinking"], "trust_remote_code": True, "infer_backend": "huggingface"}
    if not args.base:
        cfg.update(adapter_name_or_path=str(adapter), finetuning_type="lora")
    samples = read(run / "data" / "schedule_binding_test.json")
    metadata = read(run / "data" / "binding_manifest.json")["test"]
    output = Path(tempfile.mkdtemp(prefix="eval-base-" if args.base else "eval-binding-", dir=run))
    print(f"测试模型：{cfg.get('adapter_name_or_path', cfg['model_name_or_path'])}", flush=True)
    print(f"测试目录：{output}", flush=True)
    model = ChatModel(cfg)
    summary = offline(model, parse_response, samples, metadata, output)
    summary["q0750"] = execute_q0750(model, parse_response, samples, metadata, output)
    summary["inference_config"] = cfg
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"测试报告：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
