# 原图与子图作用范围补充训练

目标为已经选好的算法绑定输入。输出仍为 `{"input_refs":["引用名称"]}`。本轮不训练算法选择、plan、图转换代码、参数或最终报告。

## 数据

- train：80 张新图，五类各 16 张，1536 条样本。
- val：10 张新图，五类各 2 张，192 条样本。
- test：10 张新图，五类各 2 张，192 条样本。
- 每张图 1001～1300 个节点、3000～4199 条边。种子和实际算法调用记录在 `new_data/manifest.json`。
- 每张图包含执行调用样本，以及一对作用范围样本：相同候选和算法，分别要求原图和由先前排名生成的子图。名称和候选顺序各有两个版本。
- 相关变体跟随整图划分。条数包含变体，不是独立题数。新 test 留出了图，但任务结构和部分描述模板与 train 相同，不能证明对全新工作流结构的泛化。
- 标签来自受控的合成工作流和实际 NetworkX 调用；图转换由程序定义。
- 旧 30 题文件仅作回归，不进入训练。它们已用于分析错误，不能再称为全新盲测。
- 训练时回放旧 `schedule_binding_train.json`，权重至少占 25%，需要时循环重复。旧 val/test 不加入训练。
- 包内已有数据，不必运行 `build_data.py`。它是供复现的生成器，依赖 networkx、numpy、scipy。

## 1. Windows 上传

```powershell
scp -J fuzq@node1.sanzo.top "C:\Users\HP\Documents\Codex\2026-10-31\ban\outputs\binding_scope_repair.zip" fuzq@node2.sanzo.top:/home/fuzq/
```

## 2. node2 准备

这里使用之前指定的 1 号卡。先用 `nvidia-smi` 确认该卡仍分配给你。

便捷方式（与后面的手动步骤二选一，不要重复启动）：

```bash
unzip -n /home/fuzq/binding_scope_repair.zip -d /home/fuzq
bash /home/fuzq/binding_scope_repair/start_train.sh 1
```

脚本依次校验数据、生成配置、检查模型/token/GPU，全部通过后才提交后台训练。返回 PID 仅表示已提交，应查看日志确认已开始。

训练完成后可用 `bash /home/fuzq/binding_scope_repair/compare.sh 1 val` 对比验证成绩；把末尾改为 `regression` 对比旧 30 题，或最终改为 `test` 对比新留出图。

以下为等价手动方式。

```bash
unzip -n /home/fuzq/binding_scope_repair.zip -d /home/fuzq

BASE=/home/hdd3/fuzq/node1-copy-20261114/hdd2/qwen35-dag.h5GkGf
OLD="$BASE/runtime-binding.eLxc9y"
PKG="$BASE/package"
PY=/home/fuzq/finetune312/bin/python
LF=/home/fuzq/finetune312/bin/llamafactory-cli
TOOL=/home/fuzq/binding_scope_repair/scope_training.py
ROUND="$OLD/scope-repair-$(date +%Y%m%d-%H%M%S)"
GPU=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)

"$PY" "$TOOL" check-data
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TOOL" prepare \
  --package "$PKG" --previous-run "$OLD" --run-dir "$ROUND"
```

最后一条成功打印配置路径后继续。prepare 校验 CUDA/BF16、模型文件和真实模板 token 长度。超长就停止，不截断训练。需保留旧 `train_binding.yaml`。

新训练加载旧 `binding_lora`，保持学习率与 LoRA 结构，`create_new_adapter: false`。优化器和学习率调度重新启动，训练 1 epoch，batch 1、累积 4，每 50 步保存与验证，最多保留 3 个检查点。结果另存 `ROUND/binding_lora`，旧模型保留。

## 3. 启动训练

```bash
if [ -f "$ROUND/train_scope.yaml" ]; then
  printf '%s\n' "$ROUND" > "$OLD/latest_scope_repair.txt"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" "$LF" train "$ROUND/train_scope.yaml" \
    > "$ROUND/train.log" 2>&1 < /dev/null &
  printf '训练 PID：%s\n目录：%s\n' "$!" "$ROUND"
fi
tail -n 50 "$ROUND/train.log"
```

不要重复启动。若中断，可从新目录最新完整 checkpoint 恢复；本配置不自动恢复。应保留训练日志、trainer_state.json、检查点。

## 4. 检测

重连后恢复变量：

```bash
BASE=/home/hdd3/fuzq/node1-copy-20261114/hdd2/qwen35-dag.h5GkGf
OLD="$BASE/runtime-binding.eLxc9y"
PKG="$BASE/package"
PY=/home/fuzq/finetune312/bin/python
TOOL=/home/fuzq/binding_scope_repair/scope_training.py
ROUND=$(cat "$OLD/latest_scope_repair.txt")
GPU=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
```

先比较新旧模型 val。两者使用相同的 scope 提示词、底座和贪心解码。

```bash
for MODEL_RUN in "$OLD" "$ROUND"; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TOOL" evaluate \
    --package "$PKG" --adapter-run "$MODEL_RUN" --split val
done
```

随后检查旧 30 题回归：

```bash
for MODEL_RUN in "$OLD" "$ROUND"; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TOOL" evaluate \
    --package "$PKG" --adapter-run "$MODEL_RUN" --split regression
done
```

模型确定后，最后运行新 test。不要依据此集反复调参；如果再利用其错误训练，需要另留新测试题。

```bash
for MODEL_RUN in "$OLD" "$ROUND"; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TOOL" evaluate \
    --package "$PKG" --adapter-run "$MODEL_RUN" --split test
done
```

报告另存各模型目录 `eval-scope-*/summary.json`；原始预测在组内 `predictions.jsonl`，失败在 `strict_failures.json`。

重点看 `multi_candidate_G0`、`non_G0`、`scope_pairs_both_correct`、`whole_questions`。旧回归原名版 G0 多候选原先仅 5/12，改名目标已 48/48，应同时保持后者，避免一律输出 G0。

## 5. 五题真实执行回归

```bash
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TOOL" verify-five \
  --package "$PKG" \
  --executor /home/hdd3/fuzq/node1-copy-20261114/hdd/all_dag_execution/selected_dag_training \
  --toolkit "$BASE/final-answer.RXJxUs/round3" \
  --adapter-run "$ROUND"
```

继续读取原五题同图审核参考。绑定先匹配，再运行固定参考程序，逐项比对最终结果。图转换和参数仍由程序提供。参考源路径若失效，应恢复原执行文件，不用其他图替换。

## 验证范围

本地已完成 NetworkX 数据生成、配对/引用校验、分组无重叠校验和 Python 语法检查。尚未在本机运行 Qwen GPU 训练与推理，准确率以 node2 新报告为准。本包需上传，不代表已启动训练或完成双服务器备份。

LoRA 参数依据：https://llamafactory.readthedocs.io/en/latest/advanced/arguments.html
