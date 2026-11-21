"""Two real-model binding probes; no graph execution or reference-driven fallback."""
import copy
import importlib.util
import json
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[2]
    file = repo / "aag/engine/runtime_binder_transport.py"
    spec = importlib.util.spec_from_file_location("local_binder_transport", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = {"question": "先按v1的排名筛选子图，再分别在原图和该子图上计算PageRank。",
        "actual_graph_attributes": {"num_nodes": 1001, "num_edges": 3000, "directed": False,
            "weighted": False, "multigraph": False, "heterogeneous": False},
        "step": {"id": "v2", "description": "在完整原始图上计算PageRank。", "algorithm": "pagerank"},
        "required_steps": [], "available_inputs": [
            {"ref": "G0", "type": "Graph", "description": "当前输入的原始图"},
            {"ref": "v1", "type": "mapping_preview", "producer": "degree_centrality", "derived_from": ["G0"]},
            {"ref": "working_graph", "type": "Graph", "producer": "transform", "derived_from": ["G0", "v1"]}]}
    try:
        first = module.request_local(request)
        print("原图请求：", json.dumps(first, ensure_ascii=False), flush=True)
        other = copy.deepcopy(request)
        other["step"]["description"] = "在根据v1排名筛选得到的诱导子图上计算PageRank。"
        other["required_steps"] = ["v1"]
        second = module.request_local(other)
        print("子图请求：", json.dumps(second, ensure_ascii=False), flush=True)
        if first != {"input_refs": ["G0"]} or second != {"input_refs": ["working_graph"]}:
            raise SystemExit("本地推理检查未通过，请查看model.log；尚未启动项目调用。")
        print("本地模型推理检查通过2/2；此检查不执行图算法。", flush=True)
    finally:
        module.close()


if __name__ == "__main__":
    main()
