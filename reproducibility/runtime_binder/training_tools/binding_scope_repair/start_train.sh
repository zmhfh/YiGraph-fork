#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=/home/hdd3/fuzq/node1-copy-20261114/hdd2/qwen35-dag.h5GkGf
OLD="$BASE/runtime-binding.eLxc9y"
PY=/home/fuzq/finetune312/bin/python
LF=/home/fuzq/finetune312/bin/llamafactory-cli
GPU_INDEX=${1:-1}
GPU=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
if [[ "$GPU" != GPU-* || "$GPU" == *$'\n'* ]]; then
  printf 'GPU UUID is invalid: %s\n' "$GPU" >&2
  exit 1
fi
test -x "$PY"
test -x "$LF"
ROUND="$OLD/scope-repair-$(date +%Y%m%d-%H%M%S)"
"$PY" "$HERE/scope_training.py" check-data
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/scope_training.py" prepare \
  --package "$BASE/package" --previous-run "$OLD" --run-dir "$ROUND"
test -f "$ROUND/train_scope.yaml"
printf '%s\n' "$ROUND" > "$OLD/latest_scope_repair.txt"
nohup env CUDA_VISIBLE_DEVICES="$GPU" "$LF" train "$ROUND/train_scope.yaml" \
  > "$ROUND/train.log" 2>&1 < /dev/null &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$ROUND/train.pid"
printf 'Submitted PID: %s\nRun: %s\nLog: %s\n' "$TRAIN_PID" "$ROUND" "$ROUND/train.log"
printf 'Check the log to confirm training has started. Do not submit twice.\n'
