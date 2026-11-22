"""Load the existing LoRA locally and serve JSON-line requests on stdin/stdout."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

SYSTEM = (
    "你负责为已经选好的图算法绑定输入。只能引用 available_inputs 中已经存在的 ref。"
    "结合当前步骤描述和依赖，选择当前这次算法调用需要读取的输入，按调用顺序列出。"
    "只返回含 input_refs 字段的 JSON 对象。"
    "\n先根据当前 step.description 确定算法的作用范围。"
    "要求使用原始图、完整输入图或全图时，选择 G0，即使已有子图。"
    "在汇聚或并行任务中，每个步骤按自己的描述选择输入。"
    "要求使用筛选或合并后的子图时，依据 derived_from 核对产物来源。"
)


def parse(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj
    return json.loads(text, object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--template", default="qwen3_5")
    args = parser.parse_args()
    if not (args.model / "config.json").is_file() or not (args.adapter / "adapter_config.json").is_file():
        parser.error("Base model or adapter config is missing")
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from llamafactory.chat import ChatModel
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The selected GPU is unavailable")
        config = dict(model_name_or_path=str(args.model.resolve()), adapter_name_or_path=str(args.adapter.resolve()),
                      template=args.template, enable_thinking=False, trust_remote_code=True,
                      infer_backend="huggingface", finetuning_type="lora")
        model = ChatModel(config)
        print("LOCAL_BINDER_READY", json.dumps(config), file=sys.stderr, flush=True)
    for line in sys.stdin:
        token = None
        try:
            envelope = json.loads(line)
            token, request = envelope["request_id"], envelope["request"]
            fields = {"question", "actual_graph_attributes", "step", "required_steps", "available_inputs"}
            if not isinstance(request, dict) or set(request) != fields:
                raise ValueError("Request schema differs from the training input")
            refs = [item["ref"] for item in request["available_inputs"]]
            if not refs or len(refs) != len(set(refs)):
                raise ValueError("Invalid candidate references")
            query = "为当前算法调用绑定输入引用。\n" + json.dumps(request, ensure_ascii=False, separators=(",", ":"))
            with contextlib.redirect_stdout(sys.stderr):
                engine = model.engine
                prompt, answer = engine.template.encode_oneturn(engine.tokenizer,
                    [{"role": "user", "content": query}, {"role": "assistant", "content": ""}], system=SYSTEM)
                if len(prompt) + len(answer) + 132 > 3072:
                    raise ValueError("Input exceeds validated token budget")
                raw = model.chat([{"role": "user", "content": query}], system=SYSTEM,
                                 do_sample=False, max_new_tokens=128)[0].response_text
            result = parse(raw)
            selected = result.get("input_refs") if isinstance(result, dict) else None
            if (not isinstance(result, dict) or set(result) != {"input_refs"}
                or not isinstance(selected, list) or len(selected) != 1
                or not isinstance(selected[0], str) or selected[0] not in refs):
                raise ValueError("Invalid model binding")
            response = {"request_id": token, "result": result}
        except Exception as error:
            response = {"request_id": token, "error": str(error)}
        protocol.write(json.dumps(response, ensure_ascii=False) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
