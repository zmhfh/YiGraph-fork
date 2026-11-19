"""Contract tests: real fork executor + HTTP binder double + actual graph objects."""
import asyncio
import copy
import importlib
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(REPO).resolve()
        # Only the external LLM/config import is stubbed; executor/graphs are real source.
        reasoner = types.ModuleType("aag.reasoner.model_deployment")
        reasoner.Reasoner = object
        sys.modules[reasoner.__name__] = reasoner
        paths = types.ModuleType("aag.utils.path_utils")
        paths.RUNTIME_ROOT = Path(tempfile.gettempdir()) / "runtime-binder-test"
        sys.modules[paths.__name__] = paths
        sys.path.insert(0, str(cls.repo))
        cls.reactor = importlib.import_module("aag.engine.react_executor")
        cls.types = importlib.import_module("aag.expert_search_engine.database.datatype")
        cls.binder = importlib.import_module("aag.engine.runtime_binder")

    def setUp(self):
        t = self.types
        self.graph = t.GraphData([t.VertexData(str(i), {"label": i}) for i in range(4)],
             [t.EdgeData("0", "1", properties={"weight": 2}),
              t.EdgeData("1", "2", properties={"weight": 3}),
              t.EdgeData("2", "3", properties={"weight": 4})])
        self.calls = []
        self.requests = []
        self.choice = "working_graph"
        case = self

        class HTTP(BaseHTTPRequestHandler):
            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                case.requests.append(req)
                data = json.dumps({"input_refs": [case.choice]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), HTTP)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.env = patch.dict(os.environ, YIGRAPH_BINDER_URL=f"http://127.0.0.1:{self.server.server_port}",
                              YIGRAPH_BINDER_GRAPH_KIND="Graph")
        self.env.start()
        self.prepared = {"nodes": ["0", "1"], "edges": [["1", "0"]]}
        self.resolutions = []

        class Engine:
            async def get_algorithm_description(self, name):
                return "node count", {"name": name}

            async def run_algorithm(self, name, parameters, *args, **kwargs):
                case.calls.append((name, copy.deepcopy(parameters)))
                if name == "initialize_graph":
                    self.current = parameters
                    return {"success": True}
                import networkx as nx
                graph = nx.Graph()
                graph.add_nodes_from(v["vid"] for v in self.current["vertices"])
                graph.add_edges_from((e["src"], e["dst"]) for e in self.current["edges"])
                return {"success": True, "result": {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}}

        class Resolver:
            async def prepare_react_inputs(self, **kwargs):
                case.resolutions.append(kwargs)
                return {"parameters": {"sentinel": 17}, "graph_adapter_result": case.prepared, "errors": []}

        self.executor = self.reactor.ReActExecutor(reasoner=types.SimpleNamespace(), computing_engine=Engine(),
                      dependency_resolver=Resolver(), global_graph=self.graph)
        self.state = self.reactor.RunState(goal="Use the ranked induced subgraph", step=2)
        self.state.artifacts["a1"] = {"value": {"0": 0.7, "1": 0.3}, "source": "pagerank", "source_step_id": 1,
                                       "binding_input_refs": ["G0"]}
        self.action = {"type": "algorithm", "name": "number_of_nodes", "objective": "Count nodes in the subgraph selected using v1",
                       "parameters": {"keep": True}, "consumed": [{"artifact_id": "a1", "use_as": "graph"}]}

    def tearDown(self):
        self.env.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def execute(self):
        return asyncio.run(self.executor._execute_action(self.action, self.state))[0]

    def test_selected_graph_drives_real_engine_and_preserves_edge_properties(self):
        result = self.execute()
        self.assertEqual(result["result"], {"nodes": 2, "edges": 1})
        self.assertEqual(self.calls[0][1]["edges"][0]["properties"], {"weight": 2})
        self.assertEqual(self.calls[1][1], {"sentinel": 17})
        self.assertEqual(self.requests[0]["required_steps"], ["v1"])
        self.assertEqual(self.requests[0]["actual_graph_attributes"]["num_nodes"], 4)
        self.assertEqual(self.action["_runtime_binder_input_refs"], ["working_graph"])
        self.assertTrue(any(e["kind"] == "runtime_binding" for e in self.executor.events))

    def test_G0_replaces_prepared_subgraph(self):
        self.choice = "G0"
        self.assertEqual(self.execute()["result"], {"nodes": 4, "edges": 3})

    def test_empty_graph_never_falls_back_to_G0(self):
        self.prepared = {"nodes": [], "edges": []}
        self.assertEqual(self.execute()["result"], {"nodes": 0, "edges": 0})

    def test_unknown_ref_blocks_all_engine_calls(self):
        self.choice = "missing_graph"
        with self.assertRaises(self.binder.BindingError):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_ranking_output_is_not_accepted_as_a_graph(self):
        self.choice = "v1"
        with self.assertRaises(self.binder.BindingError):
            self.execute()
        self.assertEqual(self.calls, [])

    def test_invalid_graph_blocks_http_and_execution(self):
        self.prepared = {"nodes": ["unknown"], "edges": []}
        with self.assertRaises(self.binder.BindingError):
            self.execute()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.calls, [])

    def test_disconnected_service_does_not_execute_with_fallback(self):
        import urllib.error
        with patch.object(self.binder, "_post", side_effect=urllib.error.URLError("unavailable")):
            with self.assertRaises(urllib.error.URLError):
                self.execute()
        self.assertEqual(self.calls, [])

    def test_disabled_hook_retains_original_execution(self):
        os.environ.pop("YIGRAPH_BINDER_URL", None)
        # Original reconstruct_graph requires the original orientation.
        self.prepared["edges"] = [["0", "1"]]
        self.assertEqual(self.execute()["result"], {"nodes": 2, "edges": 1})
        self.assertEqual(self.requests, [])

    def test_cached_graph_available_on_later_branch(self):
        self.execute()
        self.state.step = 3
        self.prepared = None
        self.execute()
        self.assertIn("working_graph", [x["ref"] for x in self.requests[-1]["available_inputs"]])

    def test_count_result_is_not_misclassified_as_graph(self):
        self.state.artifacts["a2"] = {"source_step_id": 4, "source": "summary", "value": {"nodes": 4, "edges": 3}}
        self.assertEqual(self.execute()["result"], {"nodes": 2, "edges": 1})
        self.assertEqual(next(x["type"] for x in self.requests[0]["available_inputs"] if x["ref"] == "v4"), "mapping")

    def run_local_worker(self, selected):
        import tempfile
        local = importlib.import_module("aag.engine.runtime_binder_transport")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            worker = root / "worker.py"
            worker.write_text("import json,sys\nfor line in sys.stdin:\n e=json.loads(line)\n print(json.dumps({'request_id':e['request_id'],'result':{'input_refs':" + repr([selected]) + "}}),flush=True)\n", encoding="utf-8")
            with patch.dict(os.environ, YIGRAPH_BINDER_COMMAND=json.dumps([sys.executable, str(worker)]),
                            YIGRAPH_BINDER_LOG=str(root / "model.log"), YIGRAPH_BINDER_CUDA_VISIBLE_DEVICES="GPU-test"):
                try:
                    return self.execute()
                finally:
                    local.close()

    def test_local_process_drives_actual_graph_without_http(self):
        self.assertEqual(self.run_local_worker("working_graph")["result"], {"nodes": 2, "edges": 1})
        self.assertEqual(self.requests, [])
        event = next(e for e in self.executor.events if e["kind"] == "runtime_binding")
        self.assertEqual(event["data"]["endpoint"], "local-subprocess")

    def test_local_process_wrong_type_blocks_algorithm(self):
        with self.assertRaises(self.binder.BindingError):
            self.run_local_worker("v1")
        self.assertEqual(self.calls, [])
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    # unittest parses argv before setUpClass, so strip the custom flag first.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args, remaining = parser.parse_known_args()
    REPO = args.repo
    unittest.main(argv=[sys.argv[0]] + remaining)
