"""Evaluate input bindings on independent graphs and paired artifact renamings."""

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_runtime_binding import predict, read, write


HERE = Path(__file__).resolve().parent
GROUPS = ("original_names", "renamed_artifacts")


def load_cases():
    data = {name: read(HERE / f"{name}.json") for name in GROUPS}
    baseline, renamed = (data[name] for name in GROUPS)
    if baseline["metadata"] != renamed["metadata"]:
        raise ValueError("两组测试的步骤清单不同")
    meta = baseline["metadata"]
    if len(meta) != 84 or len({(row["question_id"], row["step_id"]) for row in meta}) != 84:
        raise ValueError("应有 84 条不同的算法步骤")
    if len({row["question_id"] for row in meta}) != 30:
        raise ValueError("应有 30 道独立测试题")
    for group in data.values():
        for sample, item in zip(group["samples"], meta, strict=True):
            request, expected = json.loads(sample["input"]), json.loads(sample["output"])
            refs = [row["ref"] for row in request["available_inputs"]]
            if len(set(refs)) != len(refs) or not expected["input_refs"] or not set(expected["input_refs"]) <= set(refs):
                raise ValueError(f"无效参考绑定：{item['question_id']}/{item['step_id']}")
            if request["step"]["id"] != item["step_id"]:
                raise ValueError(f"步骤 ID 与清单不同：{item['question_id']}")
    return data


def counts(rows):
    return {"passed": sum(row["strict"] for row in rows), "total": len(rows)}


def evaluate_group(model, parse, group, directory):
    directory.mkdir()
    rows = {}
    by_question, by_dag = defaultdict(list), defaultdict(list)
    with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for idx, (sample, metadata) in enumerate(zip(group["samples"], group["metadata"], strict=True), 1):
            request = json.loads(sample["input"])
            score = dict(predict(model, parse, sample, request), **metadata,
                         candidate_count=len(request["available_inputs"]))
            key = metadata["question_id"], metadata["step_id"]
            rows[key] = score
            by_question[metadata["question_id"]].append(score)
            by_dag[metadata["dag_type"]].append(score)
            stream.write(json.dumps(score, ensure_ascii=False) + "\n")
            stream.flush()
            if idx % 14 == 0:
                print(f"{directory.name}: {idx}/{len(group['samples'])}", flush=True)
    failures = [row for row in rows.values() if not row["strict"]]
    write(directory / "strict_failures.json", failures)
    summary = {
        "steps": counts(rows.values()),
        "multiple_candidates": counts([row for row in rows.values() if row["candidate_count"] > 1]),
        "whole_questions": {"passed": sum(all(row["strict"] for row in q) for q in by_question.values()),
                            "total": len(by_question)},
        "by_dag_type": {key: counts(q) for key, q in by_dag.items()},
        "failure_issues": dict(Counter(row["issue"] for row in failures)),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--base", action="store_true", help="使用同一底座做无 LoRA 对照")
    args = parser.parse_args()
    data = load_cases()
    print("30 道独立新题，每种命名版本各 84 次绑定。", flush=True)
    if args.check_data:
        return
    if args.package is None or args.run_dir is None:
        parser.error("运行推理需要 --package 和 --run-dir")
    package, run = args.package.resolve(), args.run_dir.resolve()
    sys.path.insert(0, str(package))
    from evaluate import parse_response
    from llamafactory.chat import ChatModel

    paths = read(package / "configs" / "paths.json")
    config = {"model_name_or_path": paths["model_path"], "template": paths["template"],
              "enable_thinking": paths["enable_thinking"], "trust_remote_code": True, "infer_backend": "huggingface"}
    if not args.base:
        adapter = run / "binding_lora"
        if not (adapter / "adapter_config.json").is_file():
            raise SystemExit(f"找不到绑定 LoRA：{adapter}")
        config.update(adapter_name_or_path=str(adapter), finetuning_type="lora")
    prefix = "eval-generalization-base-" if args.base else "eval-generalization-binding-"
    dest = Path(tempfile.mkdtemp(prefix=prefix, dir=run))
    print(f"测试目录：{dest}", flush=True)
    model = ChatModel(config)
    summaries, predictions = {}, {}
    for name in GROUPS:
        summaries[name], predictions[name] = evaluate_group(model, parse_response, data[name], dest / name)
    before, after = (predictions[name] for name in GROUPS)
    renamed_keys = [key for key in before if before[key]["expected"] != after[key]["expected"]]
    summaries["paired_target_renamed"] = {name: counts([predictions[name][key] for key in renamed_keys]) for name in GROUPS}
    lost = [key for key in before if before[key]["strict"] and not after[key]["strict"]]
    summaries["lost_after_renaming"] = [{"question_id": key[0], "step_id": key[1], "issue": after[key]["issue"]} for key in lost]
    summaries["inference_config"] = config
    write(dest / "summary.json", summaries)
    for name in GROUPS:
        scores = summaries[name]
        print(f"{name}: 步骤 {scores['steps']}，整题 {scores['whole_questions']}", flush=True)
    print(f"正确引用确实改名的步骤：{summaries['paired_target_renamed']}", flush=True)
    print(f"测试报告：{dest / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
