"""Replay the five reviewed workflows with an input-binding check at every call."""

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from evaluate_runtime_binding import read, write, predict
from prepare_runtime_binding import value_type


EXPECTED = {"q0036": "sequential", "q0242": "parallel", "q0493": "convergence",
            "q0750": "divergence", "q0916": "diamond"}


def unpack(value):
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if isinstance(value, dict) and value.get("type") == "mapping":
        return {(tuple(key) if isinstance(key, list) else key): unpack(item) for key, item in value["items"]}
    if isinstance(value, dict) and value.get("type") in {"Graph", "DiGraph", "MultiGraph", "MultiDiGraph"}:
        directed = "DiGraph" in value["type"]
        edges = {}
        for u, v, *fields in value["edges"]:
            pair = (u, v) if directed else tuple(sorted((u, v)))
            key = pair + (fields[0],) if len(fields) == 2 else pair
            edges[key] = unpack(fields[-1])
        return {"type": value["type"], "nodes": dict(unpack(value["nodes"])), "edges": edges}
    if isinstance(value, dict):
        return {key: unpack(item) for key, item in value.items()}
    return value


def first_difference(actual, expected, name="results"):
    if isinstance(actual, float) or isinstance(expected, float):
        if type(actual) in (int, float) and type(expected) in (int, float) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
            return None
        return name
    if type(actual) is not type(expected):
        return name
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            return name
        return next((error for key in actual if (error := first_difference(actual[key], expected[key], f"{name}.{key}"))), None)
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return name
        return next((error for idx, (a, b) in enumerate(zip(actual, expected))
                     if (error := first_difference(a, b, f"{name}[{idx}]"))), None)
    return None if actual == expected else name


def replay(qid, cases, out, model, parse):
    import runtime
    from final_answer_training import evidence_preview, verified_output
    from run_selected import execute_one

    source = Path(cases[0][1]["source_run"])
    graph = source / "input_graph.json"
    reference = source / "final_result.json"
    status = read(source / "status.json")
    graph_sha = runtime.digest(graph)
    if graph_sha != status["graph_sha256"] or any(
        meta["graph_sha256"] != graph_sha or Path(meta["source_run"]) != source for _, meta in cases
    ):
        raise ValueError("测试图哈希与已审核参考记录不同")
    if runtime.digest(reference) != status["result_sha256"]:
        raise ValueError("参考输出的哈希已变化")
    expected = unpack(read(reference))["results"]
    samples = {meta["step_id"]: sample for sample, meta in cases}
    bindings = []
    original_recorder = runtime.Recorder

    class BindingRecorder(original_recorder):
        def call(self, step_id, algorithm, fn, parameters, input_refs):
            sample = samples[step_id]
            req = json.loads(sample["input"])
            if req["step"]["algorithm"] != algorithm or json.loads(sample["output"]) != {"input_refs": input_refs}:
                raise ValueError(f"{step_id} 的执行程序与已审核绑定不同")
            available = [item for item in req["available_inputs"] if item["ref"] == "G0"]
            for event in self.events:
                artifact = verified_output(self.output, event)
                available.append({"ref": event["name"], "type": value_type(evidence_preview(artifact)),
                                  "producer": event["algorithm"] or "transform", "derived_from": event["input_refs"]})
            req["available_inputs"] = available
            decision = dict(predict(model, parse, sample, req), step_id=step_id, algorithm=algorithm)
            bindings.append(decision)
            if not decision["strict"]:
                raise ValueError(f"{step_id} 输入绑定失败：{decision['issue']}")
            return super().call(step_id, algorithm, fn, parameters, decision["prediction"]["input_refs"])

    try:
        runtime.Recorder = BindingRecorder
        code = execute_one(int(qid[1:]), graph, out)
    finally:
        runtime.Recorder = original_recorder
    write(out / "model_bindings.json", bindings)
    report = {"question_id": qid, "dag_type": EXPECTED[qid], "graph_sha256": graph_sha,
              "matched_bindings": sum(row["strict"] for row in bindings), "total_bindings": len(samples)}
    if code != 0:
        report.update(passed=False, error=read(out / "status.json").get("error"))
    else:
        actual = unpack(read(out / "final_result.json"))["results"]
        difference = first_difference(actual, expected)
        report.update(passed=difference is None and len(bindings) == len(samples), first_difference=difference)
    write(out / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("package", "executor", "toolkit", "run-dir"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    package, run = args.package.resolve(), args.run_dir.resolve()
    sys.path[:0] = [str(package), str(args.executor.resolve()), str(args.toolkit.resolve())]
    from evaluate import parse_response
    from llamafactory.chat import ChatModel

    adapter = run / "binding_lora"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"找不到输入绑定 LoRA：{adapter}")
    paths = read(package / "configs" / "paths.json")
    config = {"model_name_or_path": paths["model_path"], "template": paths["template"],
              "enable_thinking": paths["enable_thinking"], "trust_remote_code": True,
              "infer_backend": "huggingface", "finetuning_type": "lora", "adapter_name_or_path": str(adapter)}
    rows = read(run / "data" / "schedule_binding_test.json")
    metadata = read(run / "data" / "binding_manifest.json")["test"]
    grouped = defaultdict(list)
    for sample, meta in zip(rows, metadata, strict=True):
        grouped[meta["question_id"]].append((sample, meta))
    if set(grouped) != EXPECTED.keys():
        raise SystemExit(f"需要五道原验收题，当前为 {list(grouped)}")

    output = Path(tempfile.mkdtemp(prefix="eval-five-workflows-", dir=run))
    print(f"验收目录：{output}", flush=True)
    model = ChatModel(config)
    results = []
    for qid, cases in grouped.items():
        try:
            report = replay(qid, cases, output / qid, model, parse_response)
        except (KeyError, OSError, TypeError, ValueError) as error:
            report = {"question_id": qid, "dag_type": EXPECTED[qid], "passed": False, "error": str(error)}
        results.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    summary = {"adapter": str(adapter), "total": len(results), "passed": sum(row["passed"] for row in results),
               "questions": results}
    write(output / "summary.json", summary)
    print(f"真实执行通过 {summary['passed']}/{summary['total']}，报告：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
