# Trained runtime input Binder

This release connects a Qwen3.5-2B LoRA to YiGraph's existing ReAct execution boundary. The adapter selects the graph input for an already-selected algorithm and returns exactly `{"input_refs":["reference"]}`. Python validates the reference, resolves the existing graph object, and initializes the algorithm's working graph.

The hook runs after `DataDependencyResolver.prepare_react_inputs` has prepared graph/parameter adaptations and before `_initialize_working_graph`. The original Reasoner binder role continues handling conversion code, parameter mapping and other tasks. This adapter does not replace that entire role.

## Included material

- `aag/engine/runtime_binder.py`: candidates, lineage, validation and actual graph selection.
- `aag/engine/runtime_binder_transport.py`: persistent JSON-line local inference subprocess.
- `tools/runtime_binder/stdio_worker.py`: original base model plus trained LoRA, greedy decoding and schema checks.
- `models/runtime_binder`: final LoRA configuration, safetensors weights, hashes and model card.
- `reproducibility/runtime_binder`: training configuration/data and selected evaluation reports.
- GitHub Release attachment `binder-training-and-evaluations.tar.gz`: complete Binder training run history, checkpoints, evaluation outputs and original training/evaluation tooling. See its archive manifest for contents and checksum.

The archive preserves historical server paths. Treat it as a snapshot: update paths for a new machine before executing the original scripts. Machine login credentials, private keys and local environment files are not distributed. The original Qwen base-model weights and Python installation binaries are separate dependencies rather than trained outputs.

## Install and configure

Use a dedicated model environment with a LLaMA-Factory version that supports Qwen3.5 and the `qwen3_5` template. The verified training/inference setup used Python 3.12, torch 2.11.0+cu128, transformers 5.8.0, peft 0.18.1, datasets 4.0.0 and LLaMA-Factory 0.9.6.dev0. The web/engine environment can remain separate: the worker is executed using the explicitly configured Python. Do not install the legacy engine's pinned Transformers version over this model environment.

Obtain the unchanged [Qwen/Qwen3.5-2B base model](https://huggingface.co/Qwen/Qwen3.5-2B) and retain its original tokenizer. This LoRA is not a standalone language model. The public adapter configuration uses the base-model identifier instead of the training server's absolute path; the safetensors are unchanged.

With the model environment activated, set `BASE_MODEL_DIR` to that model's actual local directory, then run from this repository:

```bash
python tools/runtime_binder/configure.py \
  --model "$BASE_MODEL_DIR" \
  --python "$(command -v python)" \
  --gpu 0
source tools/runtime_binder/local.env
python tools/runtime_binder/smoke_test.py
```

Select a GPU allocated to you. The two smoke probes should return `G0` and `working_graph`, respectively. This tests real inference, not graph execution. The probe closes its worker on exit. The backend starts its own persistent worker on the first algorithm request.

In the project's existing backend environment and terminal:

```bash
source tools/runtime_binder/local.env
npm run serve
```

Environment variables are terminal/process-local. The generated file must be loaded into the process that starts the backend. Local mode uses no TCP port or SSH tunnel. `YIGRAPH_BINDER_URL` remains available as the existing optional HTTP transport; local `YIGRAPH_BINDER_COMMAND` takes precedence when both are set.

## Execution evidence and limits

The `runtime_binding` trace event contains the input references, candidate metadata, selected graph node/edge counts and `endpoint: local-subprocess`. Verify the actual graph and final algorithm results against a trusted reference for the same graph.

Reported model evaluations for the released weights:

| Evaluation | Result | Scope |
|---|---:|---|
| Fresh held-out graph snapshots | 192/192 bindings; 10/10 graph groups | Includes correlated naming, ordering and scope variants; shared task templates |
| Existing regression, original names | 84/84 bindings; 30/30 questions | Previously inspected regression set, not a fresh blind test |
| Same regression, renamed artifacts | 84/84 bindings; 30/30 questions | Paired versions of the same 30 graphs |
| Reviewed five-workflow replay | 14/14 bindings; 5/5 workflows | Fixed reference conversion/parameter programs; final outputs matched same-graph references |
| Local model transport probes | 2/2 | Actual model inference; no graph computation |

The adapter was further trained for one epoch using 1,536 new binding examples plus 512 replay rows cycled from 97 original training examples. The new examples came from 80 graph groups; validation and test each used 10 separate graph groups. All graph/naming/order variants stayed within their group split. Each generated graph had at least 1,001 vertices and 3,000 edges.

This does not establish arbitrary-workflow or production end-to-end correctness. Live ReAct production metadata can differ from the training contracts. Existing ordinary results without lineage metadata use the bridge's legacy G0 lineage assumption; new calls record actual input references. Do not treat missing provenance as independent verification.

Current bridge support is one already-materialized simple Graph/DiGraph input per algorithm. The released training primarily covers undirected simple graphs. MultiGraphs, multiple graph inputs, new graph construction, parameter inference, and final-report generation need their existing components or separate training. Numeric analysis, graph queries and temporal_paths-specific execution retain their existing paths.

Invalid/non-graph references, unavailable services and incompatible graphs stop the current algorithm call. There is no silent fallback to G0. Explicit empty selected graphs remain empty, and original vertex/edge properties are retained during reconstruction.

## Tests and disabling

Contract tests use the actual executor and graph objects, while substituting model responses and external Reasoner imports:

```bash
python tools/runtime_binder/test_transport.py -v
python tools/runtime_binder/test_bridge.py --repo . -v
```

Four transport tests and twelve bridge tests passed in the preparation environment. They are not substitutes for the GPU probes or full project execution.

Unset `YIGRAPH_BINDER_COMMAND` and `YIGRAPH_BINDER_URL`, then restart the backend to disable the model hook. The model subprocess is terminated on normal backend exit.
