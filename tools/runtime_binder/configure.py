"""Configure a local input_refs LoRA worker without modifying the web environment."""
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--model", type=Path, required=True, help="Local directory containing the original base model")
    parser.add_argument("--python", type=Path, required=True, help="Python executable in the LLaMA-Factory model environment")
    parser.add_argument("--adapter", type=Path, help="Defaults to models/runtime_binder in this repository")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--graph-kind", choices=("Graph", "DiGraph"), default="Graph")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    repo = args.repo.resolve()
    model = args.model.resolve()
    adapter = args.adapter.resolve() if args.adapter else repo / "models/runtime_binder"
    if not (model / "config.json").is_file():
        parser.error("The base model directory must contain config.json and its weights")
    if not (adapter / "adapter_config.json").is_file() or not any(adapter.glob("adapter_model*.safetensors")) and not (adapter / "adapter_model.bin").is_file():
        parser.error("The adapter directory must contain its configuration and trained weights")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        parser.error("The model Python executable is missing or not executable")
    if args.timeout <= 0:
        parser.error("Timeout must be positive")
    gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    if not gpu.startswith("GPU-") or "\n" in gpu:
        parser.error("Invalid GPU UUID")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    subprocess.run([str(args.python), "-c",
        "import torch; from llamafactory.chat import ChatModel; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('Model environment:', torch.__version__, torch.cuda.get_device_name(0))"], env=env, check=True)
    tools = repo / "tools/runtime_binder"
    logfile = repo / "runtime/binder/model.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.touch(exist_ok=True)
    command = [str(args.python), str(tools / "stdio_worker.py"), "--model", str(model), "--adapter", str(adapter)]
    settings = {"YIGRAPH_BINDER_COMMAND": json.dumps(command, ensure_ascii=False),
        "YIGRAPH_BINDER_CUDA_VISIBLE_DEVICES": gpu, "YIGRAPH_BINDER_LOG": str(logfile),
        "YIGRAPH_BINDER_GRAPH_KIND": args.graph_kind, "YIGRAPH_BINDER_TIMEOUT": str(args.timeout)}
    target = tools / "local.env"
    if target.exists():
        backup = target.with_name("local.env.before-local-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        backup.write_bytes(target.read_bytes())
    target.write_text("unset YIGRAPH_BINDER_URL\n" + "".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()), encoding="utf-8")
    print("Configuration:", target)
    print("Load this file in the backend terminal before starting YiGraph.")
    print("The model is loaded lazily on the first request; no inference was performed by this setup command.")


if __name__ == "__main__":
    main()
