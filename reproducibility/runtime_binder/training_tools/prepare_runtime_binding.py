"""Export runtime input-binding examples from reviewed YiGraph executions."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


SPLITS = ("train", "val", "test")
SYSTEM = (
    "你负责为已经选好的图算法绑定输入。只能引用 available_inputs 中已经存在的 ref。"
    "结合当前步骤描述和依赖，选择当前这次算法调用需要读取的输入，按调用顺序列出。"
    "只返回含 input_refs 字段的 JSON 对象。"
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def value_type(output):
    if isinstance(output, dict):
        return output.get("type", "object")
    return type(output).__name__


def examples(upstream, reviewed):
    dag = upstream["task_1_workflow_dag"]
    steps = {step["id"]: step for step in dag["subtasks"]}
    parents = {sid: [] for sid in steps}
    for edge in dag.get("edges") or []:
        parents[edge["to"]].append(edge["from"])
    inputs = reviewed["runtime_execution_inputs"]
    actual = inputs["input_graph"]
    graph_type = ("Multi" if actual["multigraph"] else "") + ("DiGraph" if actual["directed"] else "Graph")
    available = [{"ref": "G0", "type": graph_type, "description": "当前输入的原始图"}]
    rows, step_names = [], []
    for output in inputs["step_results"]:
        name, algorithm = output["step_or_transform"], output["algorithm"]
        refs = output["input_refs"]
        if not refs or not set(refs) <= {item["ref"] for item in available}:
            raise ValueError(f"步骤 {name} 引用了不可用的输入")
        if algorithm is not None:
            step = steps[name]
            if algorithm != step["algorithm"]:
                raise ValueError(f"步骤 {name} 的上游算法和实际调用不同")
            request = {
                "question": upstream["rephrased_question"],
                "actual_graph_attributes": actual,
                "step": step,
                "required_steps": parents[name],
                "available_inputs": available,
            }
            rows.append({"system": SYSTEM, "instruction": "为当前算法调用绑定输入引用。",
                         "input": json.dumps(request, ensure_ascii=False, separators=(",", ":")),
                         "output": json.dumps({"input_refs": refs}, ensure_ascii=False, separators=(",", ":"))})
            step_names.append(name)
        item = {"ref": name, "type": value_type(output["output"]),
                "producer": algorithm or "transform", "derived_from": refs}
        available.append(item)
    if set(step_names) != steps.keys() or len(step_names) != len(steps):
        raise ValueError("已选步骤没有完整的真实调用标签")
    return rows, step_names


def check_disjoint(approved):
    question_splits = {}
    graph_splits = {}
    if not any(row["question_id"] == "q0750" for row in approved["test"]):
        raise ValueError("审核数据的 test 分组缺少 q0750")
    for split in SPLITS:
        for row in approved[split]:
            for field, locations in (("question_id", question_splits), ("graph_sha256", graph_splits)):
                if locations.setdefault(row[field], split) != split:
                    raise ValueError(f"留出集与其他分组重叠：{field} {row[field]}")


def export(approved, upstream, specs, candidate, destination):
    check_disjoint(approved)
    provenance = {"skipped": []}
    for split in SPLITS:
        rows, metadata = [], []
        for entry in approved[split]:
            qid = entry["question_id"]
            source = Path(entry["source_run"])
            task_spec = None if (source / "input_context.json").is_file() else specs / f"{qid}.json"
            try:
                reviewed = candidate(source, task_spec)
                if reviewed["question_id"] != qid or reviewed["graph_sha256"] != entry["graph_sha256"]:
                    raise ValueError("题号或实际图 SHA-256 与审核记录不同")
                cases, step_ids = examples(upstream[qid], reviewed)
            except (OSError, ValueError, KeyError, TypeError) as error:
                if qid == "q0750":
                    raise SystemExit(f"验收题 q0750 的同图记录无法核验：{error}") from error
                provenance["skipped"].append({"split": split, "question_id": qid, "reason": str(error)})
                continue
            rows.extend(cases)
            metadata.extend(dict(entry, step_id=sid) for sid in step_ids)
        if not rows:
            raise SystemExit(f"{split} 没有可用审核样本")
        write(destination / f"schedule_binding_{split}.json", rows)
        provenance[split] = metadata
        print(f"{split}：{len(rows)} 条，按 DAG 类型 {dict(Counter(row['dag_type'] for row in metadata))}", flush=True)
    write(destination / "binding_manifest.json", provenance)
    print(f"无法复核的审核记录：{len(provenance['skipped'])} 条", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--toolkit", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--task-specs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    package = args.package.resolve()
    sys.path[:0] = [str(package), str(args.toolkit.resolve())]
    import prepare
    import yaml
    from final_answer_training import candidate

    approved = read(args.reviewed_manifest)
    upstream = {f"q{row['question_index']:04d}": row for row in read(args.upstream)["records"]}
    run = args.run_dir.resolve()
    data = run / "data"
    data.mkdir(parents=True)
    export(approved, upstream, args.task_specs, candidate, data)

    adapter_columns = {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    write(data / "dataset_info.json", {
        f"binding_{split}": {"file_name": f"schedule_binding_{split}.json", "formatting": "alpaca",
                             "columns": adapter_columns}
        for split in SPLITS
    })

    config = yaml.safe_load((package / "configs" / "sft.yaml").read_text(encoding="utf-8"))
    paths = read(package / "configs" / "paths.json")
    for key in ("adapter_name_or_path", "max_samples", "max_steps", "tokenized_path", "val_size", "eval_steps", "warmup_ratio"):
        config.pop(key, None)
    config.update(model_name_or_path=paths["model_path"], template=paths["template"],
                  enable_thinking=paths["enable_thinking"], stage="sft", finetuning_type="lora", do_train=True,
                  dataset_dir=str(data), dataset="binding_train", eval_dataset="binding_val", do_eval=True,
                  overwrite_output_dir=False, output_dir=str(run / "binding_lora"),
                  create_new_adapter=False, resume_from_checkpoint=None, num_train_epochs=2.0,
                  per_device_train_batch_size=1, gradient_accumulation_steps=4,
                  warmup_steps=5, save_strategy="epoch", eval_strategy="epoch")
    prepare.HERE = run
    model = (package / paths["model_path"]).resolve()
    config["model_name_or_path"] = str(model)
    tokenizer, template = prepare.model_checks(model)
    prepare.check_tokens(tokenizer, template, int(config["cutoff_len"]))
    with (run / "train_binding.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    print(f"训练配置：{run / 'train_binding.yaml'}，学习率：{config['learning_rate']}", flush=True)


if __name__ == "__main__":
    main()
