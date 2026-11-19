"""Prepare continued LoRA training, score bindings, and run the five real workflows."""
import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from evaluate_runtime_binding import predict, read, write

HERE = Path(__file__).resolve().parent
HINT = (
    "\n先根据当前 step.description 确定算法的作用范围。"
    "要求使用原始图、完整输入图或全图时，选择 G0，即使已有子图。"
    "在汇聚或并行任务中，每个步骤按自己的描述选择输入。"
    "要求使用筛选或合并后的子图时，依据 derived_from 核对产物来源。"
)


def prompted(sample):
    row = deepcopy(sample)
    if HINT not in row["system"]:
        row["system"] += HINT
    return row


def signature(row):
    req = json.loads(row["input"])
    return json.dumps({"instruction": row["instruction"], "input": req}, ensure_ascii=False, sort_keys=True)


def check_data():
    all_questions, all_graphs, signatures = {}, {}, {}
    request_answers = {}
    output = {}
    for split in ("train", "val", "test"):
        data = read(HERE / "new_data" / f"{split}.json")
        if len(data["samples"]) != len(data["metadata"]):
            raise ValueError("样本与来源清单长度不同")
        pairs = defaultdict(dict)
        for row, meta in zip(data["samples"], data["metadata"], strict=True):
            qid, sha = meta["question_id"], meta["graph_sha256"]
            for key, table in ((qid, all_questions), (sha, all_graphs)):
                if table.setdefault(key, split) != split:
                    raise ValueError("题目或图在 train/val/test 之间重叠")
            req, expected = json.loads(row["input"]), json.loads(row["output"])
            refs = [item["ref"] for item in req["available_inputs"]]
            if len(refs) != len(set(refs)) or expected.keys() != {"input_refs"} or len(expected["input_refs"]) != 1:
                raise ValueError("绑定结构无效")
            target = expected["input_refs"][0]
            if target not in refs or next(i for i in req["available_inputs"] if i["ref"] == target)["type"] != "Graph":
                raise ValueError("绑定未指向可用图")
            if signatures.setdefault(signature(row), split) != split:
                raise ValueError("相同请求在不同分组中出现")
            label = tuple(expected["input_refs"])
            if request_answers.setdefault(signature(row), label) != label:
                raise ValueError("相同请求存在互相冲突的绑定标签")
            if meta["case_kind"] == "scope_pair":
                pairs[(qid, meta["naming"], meta["permutation"])][meta["scope"]] = (req, expected)
        for pair in pairs.values():
            if set(pair) != {"full", "selected"}:
                raise ValueError("作用范围配对缺少一侧")
            a, b = pair["full"], pair["selected"]
            if a[0]["available_inputs"] != b[0]["available_inputs"] or a[0]["step"]["algorithm"] != b[0]["step"]["algorithm"]:
                raise ValueError("配对候选、顺序或算法不同")
            if a[1]["input_refs"] != ["G0"] or b[1]["input_refs"] == ["G0"]:
                raise ValueError("作用范围配对标签无效")
        output[split] = {"samples": len(data["samples"]), "questions": len({m["question_id"] for m in data["metadata"]})}
    # Existing 30 regression questions have never been used to build these labels.
    regression = read(HERE / "original_names.json")
    if set(m["graph_sha256"] for m in regression["metadata"]) & set(all_graphs):
        raise ValueError("新数据与旧回归图重叠")
    return output


def prepare(args):
    import yaml
    report = check_data()
    package, old = args.package.resolve(), args.previous_run.resolve()
    adapter = old / "binding_lora"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"找不到原绑定 LoRA：{adapter}")
    cfg = yaml.safe_load((old / "train_binding.yaml").read_text(encoding="utf-8"))
    paths = read(package / "configs" / "paths.json")
    old_manifest = read(old / "data" / "binding_manifest.json")
    replay = read(old / "data" / "schedule_binding_train.json")
    replay_meta = old_manifest["train"]
    if len(replay) != len(replay_meta) or not replay:
        raise ValueError("原训练集与来源清单不对应")
    original_heldout = []
    for split in ("val", "test"):
        samples = read(old / "data" / f"schedule_binding_{split}.json")
        if len(samples) != len(old_manifest[split]):
            raise ValueError("原验证/测试清单不对应")
        original_heldout.extend(samples)
    forbidden = {signature(s) for s in original_heldout}
    forbidden.update(signature(s) for s in read(HERE / "original_names.json")["samples"])
    forbidden.update(signature(s) for s in read(HERE / "renamed_artifacts.json")["samples"])
    val_q = {m["question_id"] for split in ("val", "test") for m in old_manifest[split]}
    val_g = {m["graph_sha256"] for split in ("val", "test") for m in old_manifest[split]}
    if any(m["question_id"] in val_q or m["graph_sha256"] in val_g for m in replay_meta):
        raise ValueError("原 train 与留出集发生重叠")
    train = read(HERE / "new_data" / "train.json")
    if any(signature(s) in forbidden for s in train["samples"] + replay):
        raise ValueError("训练请求与测试集重叠")
    run = args.run_dir.resolve()
    if run.exists():
        raise SystemExit(f"输出目录已经存在，请指定新的 --run-dir：{run}")
    run.mkdir(parents=True)
    data = run / "data"
    data.mkdir()
    # Uniform replay has at least 25% weight, cycling only the original training split.
    n_replay = max(len(replay), (len(train["samples"]) + 2) // 3)
    replay_rows = [prompted(replay[i % len(replay)]) for i in range(n_replay)]
    combined = [prompted(s) for s in train["samples"]] + replay_rows
    random.Random(351119).shuffle(combined)
    write(data / "schedule_scope_train.json", combined)
    val = [prompted(s) for s in read(HERE / "new_data" / "val.json")["samples"]]
    val.extend(prompted(s) for s in read(old / "data" / "schedule_binding_val.json"))
    write(data / "schedule_scope_val.json", val)
    columns = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    write(data / "dataset_info.json", {f"scope_{s}": {"file_name": f"schedule_scope_{s}.json",
          "formatting": "alpaca", "columns": columns} for s in ("train", "val")})
    # Keep the five original held-out records for guarded real-execution regression.
    for name in ("schedule_binding_test.json", "binding_manifest.json"):
        shutil.copy2(old / "data" / name, data / name)
    model_path = Path(paths["model_path"])
    if not model_path.is_absolute():
        model_path = package / model_path
    for key in ("max_samples", "max_steps", "tokenized_path", "val_size", "warmup_ratio",
                "deepspeed", "load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
        cfg.pop(key, None)
    cfg.update(model_name_or_path=str(model_path.resolve()), template=paths["template"],
               enable_thinking=paths["enable_thinking"], stage="sft", do_train=True, do_eval=True,
               finetuning_type="lora", adapter_name_or_path=str(adapter), create_new_adapter=False,
               resume_from_checkpoint=None, dataset_dir=str(data), dataset="scope_train", eval_dataset="scope_val",
               output_dir=str(run / "binding_lora"), overwrite_output_dir=False,
               num_train_epochs=1.0, gradient_accumulation_steps=4, per_device_train_batch_size=1,
               per_device_eval_batch_size=1, save_strategy="steps", save_steps=50,
               eval_strategy="steps", eval_steps=50, save_total_limit=3, save_only_model=False,
               overwrite_cache=True, preprocessing_num_workers=2, dataloader_num_workers=0,
               report_to="none", logging_steps=10)
    # Reuse the actual template/tokenizer to reject any sample that would be cut off.
    sys.path.insert(0, str(package))
    import prepare as package_prepare
    package_prepare.HERE = run
    tok, template = package_prepare.model_checks(model_path.resolve())
    package_prepare.check_tokens(tok, template, int(cfg["cutoff_len"]))
    with (run / "train_scope.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, allow_unicode=True, sort_keys=False)
    provenance = {"previous_run": str(old), "previous_adapter": str(adapter), "scope_hint": HINT,
                  "new_data_counts": report, "new_training_rows": len(train["samples"]),
                  "replay_original_rows": len(replay), "replay_training_rows": n_replay,
                  "total_training_rows": len(combined), "learning_rate": cfg["learning_rate"],
                  "epochs": 1, "data_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((HERE / "new_data").glob("*.json"))}}
    write(run / "repair_manifest.json", provenance)
    print(json.dumps(provenance, ensure_ascii=False, indent=2), flush=True)
    print("配置：", run / "train_scope.yaml", flush=True)


def counts(rows):
    rows = list(rows)
    return {"passed": sum(r["strict"] for r in rows), "total": len(rows)}


def evaluate(args):
    check_data()
    package = args.package.resolve()
    adapter_run = args.adapter_run.resolve()
    adapter = adapter_run / "binding_lora"
    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"找不到 LoRA：{adapter}")
    sys.path.insert(0, str(package))
    from evaluate import parse_response
    from llamafactory.chat import ChatModel
    paths = read(package / "configs" / "paths.json")
    model_path = Path(paths["model_path"])
    if not model_path.is_absolute():
        model_path = package / model_path
    cfg = {"model_name_or_path": str(model_path.resolve()), "template": paths["template"],
           "enable_thinking": paths["enable_thinking"], "trust_remote_code": True,
           "infer_backend": "huggingface", "finetuning_type": "lora", "adapter_name_or_path": str(adapter)}
    if args.split == "regression":
        groups = {name: read(HERE / f"{name}.json") for name in ("original_names", "renamed_artifacts")}
    else:
        groups = {args.split: read(HERE / "new_data" / f"{args.split}.json")}
    dest = Path(tempfile.mkdtemp(prefix=f"eval-scope-{args.split}-", dir=adapter_run))
    print("测试目录：", dest, flush=True)
    model = ChatModel(cfg)
    summaries, predictions = {}, {}
    for name, group in groups.items():
        all_rows = []
        question_rows, by_kind = defaultdict(list), defaultdict(list)
        path = dest / name
        path.mkdir()
        with (path / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for idx, (sample, meta) in enumerate(zip(group["samples"], group["metadata"], strict=True), 1):
                req = json.loads(sample["input"])
                candidates = req["available_inputs"]
                row = dict(predict(model, parse_response, prompted(sample), req), **meta)
                row["candidate_count"] = len(candidates)
                row["graph_candidate_count"] = sum(i["type"] in ("Graph", "DiGraph", "MultiGraph", "MultiDiGraph") for i in candidates)
                all_rows.append(row)
                question_rows[meta["question_id"]].append(row)
                by_kind[meta["dag_type"]].append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if idx % 28 == 0:
                    print(name, idx, "/", len(group["samples"]), flush=True)
        multi_g0 = [r for r in all_rows if r["candidate_count"] > 1 and r["expected"] == {"input_refs": ["G0"]}]
        scope_pairs = defaultdict(list)
        for r in all_rows:
            if r.get("case_kind") == "scope_pair":
                scope_pairs[(r["question_id"], r["naming"], r["permutation"])].append(r)
        summaries[name] = {"steps": counts(all_rows),
            "whole_questions": {"passed": sum(all(r["strict"] for r in rs) for rs in question_rows.values()), "total": len(question_rows)},
            "multi_candidate_G0": counts(multi_g0),
            "non_G0": counts(r for r in all_rows if r["expected"] != {"input_refs": ["G0"]}),
            "scope_pairs_both_correct": {"passed": sum(len(rs) == 2 and all(r["strict"] for r in rs) for rs in scope_pairs.values()), "total": len(scope_pairs)},
            "by_dag_type": {k: counts(rs) for k, rs in by_kind.items()},
            "failure_issues": dict(Counter(r["issue"] for r in all_rows if not r["strict"]))}
        predictions[name] = all_rows
        write(path / "strict_failures.json", [r for r in all_rows if not r["strict"]])
        print(name, json.dumps(summaries[name], ensure_ascii=False), flush=True)
    if args.split == "regression":
        before, after = predictions["original_names"], predictions["renamed_artifacts"]
        changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a["expected"] != b["expected"]]
        summaries["paired_target_renamed"] = {name: counts(predictions[name][i] for i in changed) for name in predictions}
        print("正确引用确实改名的步骤：", summaries["paired_target_renamed"], flush=True)
    summaries["inference_config"] = cfg
    summaries["scope_hint"] = HINT
    summaries["scope"] = "Input binding from provided status snapshots. No model-generated transformations, parameters, or final report."
    write(dest / "summary.json", summaries)
    print("测试报告：", dest / "summary.json", flush=True)


def verify_five(args):
    import verify_binding_workflows as verify
    old_predict = verify.predict

    def with_hint(model, parse, sample, request):
        return old_predict(model, parse, prompted(sample), request)

    verify.predict = with_hint
    sys.argv = ["verify_binding_workflows.py", "--package", str(args.package), "--executor", str(args.executor),
                "--toolkit", str(args.toolkit), "--run-dir", str(args.adapter_run)]
    verify.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-data")
    p = sub.add_parser("prepare")
    for name in ("package", "previous-run", "run-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p = sub.add_parser("evaluate")
    for name in ("package", "adapter-run"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test", "regression"), required=True)
    p = sub.add_parser("verify-five")
    for name in ("package", "executor", "toolkit", "adapter-run"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check-data":
        print(json.dumps(check_data(), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        verify_five(args)


if __name__ == "__main__":
    main()
