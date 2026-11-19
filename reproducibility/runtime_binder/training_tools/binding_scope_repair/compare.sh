#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE=/home/hdd3/fuzq/node1-copy-20261114/hdd2/qwen35-dag.h5GkGf
OLD="$BASE/runtime-binding.eLxc9y"
PY=/home/fuzq/finetune312/bin/python
GPU_INDEX=${1:-1}
SPLIT=${2:-val}
case "$SPLIT" in val|regression|test) ;; *) echo 'Expected val, regression, or test' >&2; exit 2 ;; esac
ROUND=$(cat "$OLD/latest_scope_repair.txt")
GPU=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=uuid --format=csv,noheader)
if [[ "$GPU" != GPU-* || "$GPU" == *$'\n'* ]]; then
  printf 'GPU UUID is invalid: %s\n' "$GPU" >&2
  exit 1
fi
if [[ ! -f "$ROUND/binding_lora/adapter_config.json" ]]; then
  echo "No completed adapter found: $ROUND/binding_lora" >&2
  exit 1
fi
for MODEL_RUN in "$OLD" "$ROUND"; do
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$HERE/scope_training.py" evaluate \
    --package "$BASE/package" --adapter-run "$MODEL_RUN" --split "$SPLIT"
done
