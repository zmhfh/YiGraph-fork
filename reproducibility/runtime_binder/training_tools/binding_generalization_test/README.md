# 输入绑定泛化测试

30 道新构造的测试题共覆盖五种 DAG，每种 6 道。每题使用单独生成的 1001 节点、3000 边图。参考绑定来自实际运行的 NetworkX 调用记录，生成时核对了诱导子图的边和依赖来源。`reference_manifest.json` 保存图种子、图 SHA-256、选择的算法、执行输入引用和结果摘要哈希。

每道题有两个版本。`original_names.json` 使用 `working_graph` 这一类常见产物名。`renamed_artifacts.json` 给同一个转换产物分配 `artifact_数字` 的名字，同时更新输入引用和引用来源。两个版本共享图、题目描述、步骤编号、候选顺序、依赖和执行结果。每个版本各 84 条调用样本，独立题数仍是 30。本测试评估模型在给定运行状态下生成的 `input_refs`。

## 在 node2 运行

```bash
unzip -n /home/fuzq/binding_generalization_test.zip -d /home/fuzq

GPU_INDEX=1
BASE=/home/hdd3/fuzq/node1-copy-20261114/hdd2/qwen35-dag.h5GkGf
PY=/home/fuzq/finetune312/bin/python
GPU=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)

CUDA_VISIBLE_DEVICES="$GPU" "$PY" /home/fuzq/binding_generalization_test/run_test.py \
  --package "$BASE/package" \
  --run-dir "$BASE/runtime-binding.eLxc9y"
```

把 GPU 编号改为实际获分配的编号。运行结束会打印 `summary.json` 路径。`steps` 汇总单个步骤，`whole_questions` 汇总所有步骤都通过的题，`multiple_candidates` 单独报告可选输入超过一个的步骤。`paired_target_renamed` 统计正确引用本身发生改名的步骤，`lost_after_renaming` 列出配对测试中改名前通过、改名后失败的步骤及首个问题。每组都有原始预测和失败明细。可用 `--check-data` 单独检查数据，或给评测命令追加 `--base` 运行同一底座的对照。
