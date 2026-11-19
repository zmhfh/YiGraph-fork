---
base_model: Qwen/Qwen3.5-2B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - lora
  - graph
  - input-binding
  - yigraph
---

# YiGraph runtime Binder LoRA

A LoRA adapter for selecting the existing graph input of an already-selected graph algorithm. This is not a standalone base model or a general-purpose final-answer model.

## Files and base model

- Base: `Qwen/Qwen3.5-2B`, unchanged base weights/tokenizer required.
- PEFT: LoRA, rank 16, alpha 32, dropout 0.05.
- Targets: in_proj_b, q_proj, in_proj_a, out_proj, down_proj, up_proj, gate_proj, v_proj, in_proj_qkv, k_proj, o_proj, in_proj_z.
- Final trained file: `adapter_model.safetensors`, 67,332,688 bytes.
- SHA-256: `c328e69abcf0be00f6756c6888ca5515ca6867cdd345ef0b98ea2c8c4260ccd69`.
- The original base model's Apache-2.0 license is included as `LICENSE`.

The public adapter configuration replaces the original machine-specific base-model path with the official model identifier. The trained tensors are unchanged. `SHA256SUMS.json` records the distributed files.

## Input and output

Input fields are question, actual_graph_attributes, step (id/description/algorithm), required_steps and available_inputs (ref/type/producer/derived_from).

Example output:

```json
{"input_refs":["working_graph"]}
```

The model returns a reference, not the graph itself. YiGraph validates that it identifies a currently available graph, resolves its real object, then invokes the algorithm. Exact training prompts and runtime loading code are included in this repository.

## Training and evaluation

The released model continues the previous Binder LoRA for one epoch with 1,536 new examples and 512 replay rows from 97 original training examples. Replay is 25% of the 2,048 training rows. New graph groups: 80 training, 10 validation and 10 test; related naming/order/scope variants are grouped together.

Reported results: 192/192 binding snapshots on 10 fresh held-out graphs; 84/84 on original names and 84/84 on renamed artifacts for the same 30 regression graphs; 14/14 bindings and 5/5 reviewed real-workflow replays. These are scoped binding/guarded replay results, not proof of arbitrary DAG execution or report accuracy. Regression cases were previously inspected. Some task templates are shared with training. See the raw evaluation reports and [integration documentation](../../docs/runtime-binder.md).

## Usage

Use `tools/runtime_binder/configure.py` to specify a local base model, an allocated GPU and a separate compatible LLaMA-Factory Python environment. Run `tools/runtime_binder/smoke_test.py`, then load the generated `local.env` into the backend terminal. Do not set this input_refs-only adapter as the project's entire Reasoner binder role, which also handles conversion code and parameter mapping.
