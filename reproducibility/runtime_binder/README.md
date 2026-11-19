# Runtime Binder reproducibility snapshot

This directory holds the released model's final training configuration/data and selected evaluation reports. The GitHub Release contains the complete larger training history, checkpoints, original helper scripts and reference execution records as compressed attachments.

The final active training run was `scope-gpu3-20261119-193920-6cd432`. This is a historical run identifier, not a required GPU setting. Pick an allocated local GPU when deploying.

Training/inference versions verified during this work: Python 3.12, torch 2.11.0+cu128, transformers 5.8.0, datasets 4.0.0, peft 0.18.1 and LLaMA-Factory 0.9.6.dev0. Machine-specific paths in archived configurations need remapping; do not blindly execute original paths or generated candidate programs.

The complete training archive includes `runtime-binding.eLxc9y`, the original LLaMA-Factory configuration package, `binding_scope_repair`, `binding_generalization_test`, and the original Binder prepare/evaluate/workflow-verification scripts. Reference execution data is distributed in a separate attachment. Archive manifests include SHA-256 and byte counts.

Secrets, private keys, environment files and Python bytecode are excluded. The snapshot is training material and evidence, not an automatically trusted production runtime.
