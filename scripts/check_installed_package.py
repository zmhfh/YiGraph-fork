"""Smoke test an installed wheel from outside the source checkout.

Run with the installation environment's Python using -I (isolated mode).
"""

import asyncio
import argparse
from contextlib import chdir
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


async def check_mcp():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as error_log:
        async with stdio_client(StdioServerParameters(
            command=sys.executable,
            args=["-I", "-m", "aag.computing_engine.networkx_server.mcp_server"],
            env={**os.environ},
        ), errlog=error_log) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert any(tool.name == "run_pagerank" for tool in tools.tools)
                response = await session.call_tool("run_initialize_graph", {
                    "vertices": [{"vid": name, "properties": {}} for name in ("A", "B", "C")],
                    "edges": [{"src": "A", "dst": "B"}, {"src": "B", "dst": "C"}],
                    "dataset_config": {
                        "name": "smoke", "type": "graph",
                        "schema": {
                            "vertex": [], "edge": [],
                            "graph_structure": {"directed": True}, "graph_store_info": {},
                        },
                    },
                })
                payload = json.loads(response.content[0].text)
                assert payload["success"], payload
                response = await session.call_tool("run_pagerank", {})
                payload = json.loads(response.content[0].text)
                assert payload["success"], payload
                scores = payload["result"]
                assert max(scores, key=scores.get) == "C"
                print(f"MCP graph initialization and PageRank passed ({len(tools.tools)} tools).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", action="store_true", help="Also check the optional AAG engine import")
    args = parser.parse_args()
    import yigraph

    source_root = Path(__file__).resolve().parents[1]
    assert Path(yigraph.__file__).resolve().parent != source_root / "yigraph", yigraph.__file__
    assert "site-packages" in Path(yigraph.__file__).parts, yigraph.__file__
    with tempfile.TemporaryDirectory(prefix="yigraph wheel ") as temporary, chdir(temporary):
        root = Path(temporary)
        os.environ["YIGRAPH_HOME"] = str(root / "workspace")
        yigraph.init_workspace(os.environ["YIGRAPH_HOME"])
        from aag.utils.path_utils import ALGORITHMS_PATH, DATASETS_INDEX_PATH
        from aag.knowledge_base.knowledge_base_management.basic import get_algorithm_list
        from yigraph.demo import run_demo

        assert ALGORITHMS_PATH.is_file() and DATASETS_INDEX_PATH.is_file()
        assert get_algorithm_list()["total"] > 100
        assert yigraph.load_config().retrieval.embedding["device"] == "cpu"
        assert run_demo()["shortest_path_A_to_D"] == ["A", "C", "D"]
        subprocess.run([sys.executable, "-I", "-m", "yigraph", "demo"], check=True, capture_output=True)
        command = Path(sys.executable).with_name("yigraph.exe" if os.name == "nt" else "yigraph")
        result = subprocess.run([str(command), "--version"], capture_output=True, text=True, check=True)
        assert result.stdout.strip() == "2.0.0"
        asyncio.run(asyncio.wait_for(check_mcp(), timeout=45))
        if args.engine:
            from yigraph import AAGEngine

            assert AAGEngine.__module__ == "aag.engine.aag_engine"
            print("Full AAG engine import passed.")
    print("Installed package, CLI, configuration, knowledge base and local demo passed.")


if __name__ == "__main__":
    main()
