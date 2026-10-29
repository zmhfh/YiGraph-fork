"""Exercise installed behavior in subprocesses so path constants cannot leak."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from yigraph import init_workspace, load_config
from yigraph.demo import run_demo


def test_offline_graph_analysis():
    result = run_demo()
    assert result["shortest_path_A_to_D"] == ["A", "C", "D"]
    assert sum(result["pagerank"].values()) == pytest.approx(1.0)
    assert max(result["pagerank"], key=result["pagerank"].get) == "D"


def test_workspace_is_non_destructive_and_config_is_loadable(tmp_path):
    root = init_workspace(tmp_path / "workspace")
    config_path = root / "config" / "engine_config.yaml"
    config = load_config(config_path)
    assert config.retrieval.embedding["device"] == "cpu"
    assert config.reasoner.llm.openai["api_key"] == ""
    config_path.write_text("custom settings", encoding="utf-8")
    init_workspace(root)
    assert config_path.read_text(encoding="utf-8") == "custom settings"
    assert (root / "datasets/data/AMLSim1K/graph/accounts.csv").is_file()


def test_workspace_paths_and_knowledge_base_from_other_directory(tmp_path):
    root = init_workspace(tmp_path / "workspace")
    environment = {**os.environ, "YIGRAPH_HOME": str(root)}
    # This PYTHONPATH is only for running source tests; wheel smoke tests omit it.
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    code = """
import json
from aag.utils.path_utils import DEFAULT_CONFIG_PATH, DATASETS_INDEX_PATH
from aag.knowledge_base.knowledge_base_management.basic import get_algorithm_list, ALGORITHMS_PATH
print(json.dumps([str(DEFAULT_CONFIG_PATH), str(DATASETS_INDEX_PATH), str(ALGORITHMS_PATH), get_algorithm_list()['total']]))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment, text=True, capture_output=True, check=True)
    config, datasets, algorithms, count = json.loads(result.stdout)
    assert Path(config) == root / "config/engine_config.yaml"
    assert Path(datasets) == root / "datasets/dataset_schemas/datasets.yaml"
    assert Path(algorithms) == root / "knowledge_base/algorithms.yaml"
    assert count > 100


def test_cli_version_and_home(tmp_path):
    result = subprocess.run([sys.executable, "-m", "yigraph", "--version"], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "2.0.0"
    root = tmp_path / "workspace"
    subprocess.run([sys.executable, "-m", "yigraph", "init", str(root)], check=True, capture_output=True)
    # No model dependency or external service is needed to execute --help.
    subprocess.run([sys.executable, "-m", "yigraph", "--home", str(root), "chat", "--help"], check=True, capture_output=True)
