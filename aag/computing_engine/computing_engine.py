import importlib
import yaml
import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from aag.utils.path_utils import DEFAULT_CONFIG_SERVER_PATH

from aag.computing_engine.mcp_client import GraphMCPClient
from aag.computing_engine.code_executor import DynamicCodeExecutor
from aag.expert_search_engine.database.datatype import GraphData
from aag.computing_engine.graph_query.nl_query_engine import NaturalLanguageQueryEngine, LLMInterface
from aag.computing_engine.graph_query.graph_query import Neo4jGraphClient, Neo4jConfig


logger = logging.getLogger(__name__)


class ComputingEngine:
    """
    Unified computing scheduler layer.
    - Manages multiple engines (NetworkX / PyG / Nebula).
    - Exposes a single execution interface to the scheduler.
    - Wraps MCP client communication.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_SERVER_PATH):
        self.config_path = Path(config_path)
        self.clients: Dict[str, GraphMCPClient] = {}
        self.engine_supported_algorithms = {}   # algorithm_type -> engine_name
        self.algorithm_tool_mapping = {}        # algorithm_name -> tool_name
        self.parameter_modules = {}
        self._initialized = False
        self.code_executor = DynamicCodeExecutor(timeout=120, auto_install=True)
        self.nl_query_engine: Optional[NaturalLanguageQueryEngine] = None
        self.neo4j_config: Optional[Dict[str, Any]] = None
        self.reasoner = None  # set in initialize_graph_query_engine

    async def initialize(self):
        """Load config and connect to all MCP servers."""
        config = self._load_config()
        self.engine_supported_algorithms = config.get("engine_supported_algorithms", {})
        self._parse_algorithm_tool_mapping(config.get("engine_supported_algorithms", {}))
        
        for engine_name, server in config.get("servers", {}).items():
            client = GraphMCPClient(
                server_command=(sys.executable if server.get("command", "python") == "python" else server["command"]),
                server_args=server.get("args", [])
            )
            ok = await client.connect()
            if ok:
                self.clients[engine_name] = client
                logger.info(f"✅ Engine '{engine_name}' connected")
            else:
                logger.warning(f"⚠️ Engine '{engine_name}' failed to connect")

            try:
                module_path = f"aag.computing_engine.{engine_name}_server.parameter_utils"
                module = importlib.import_module(module_path)
                self.parameter_modules[engine_name] = module
                logger.info(f"🧩 Loaded parameter module: {module_path}")
            except ImportError as e:
                logger.warning(f"⚠️ No parameter_utils.py found for engine '{engine_name}' ({e})")
        self._initialized = True
        logger.info(f"✅ Loaded {len(self.clients)} computing engines")
        # logger.info(f"🧠 Annotation modules loaded: {list(self.parameter_modules.keys())}")


    def _load_config(self) -> dict:
        """Load engine definitions from config file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_algorithm_tool_mapping(self, engine_algorithms: dict):
        """
        engine_supported_algorithms:
          networkx:
            bfs:
              - tool: bfs_edges
        """
        
        for engine_name, algorithms in engine_algorithms.items():
            engine_mapping = {}
            
            for algo_name, tool_configs in algorithms.items():
                if isinstance(tool_configs, list) and len(tool_configs) > 0:
                    tool_config = tool_configs[0]
                    tool_base_name = tool_config.get("tool")
                    if tool_base_name and isinstance(tool_base_name, str):
                        engine_mapping[algo_name] = f"run_{tool_base_name}"
                    else:
                        logger.warning(f"⚠️ Invalid tool name for algorithm '{algo_name}' in engine '{engine_name}': {tool_base_name}")
            
            self.algorithm_tool_mapping[engine_name] = engine_mapping
        
        logger.info(f"✅ Parsed {sum(len(m) for m in self.algorithm_tool_mapping.values())} algorithm-tool mappings")
        logger.debug(f"Algorithm-tool mapping: {self.algorithm_tool_mapping}")

    def _resolve_engine(self, algo_name: str) -> str:
        """Resolve which engine owns the given algorithm."""
        algo_name = algo_name.lower()
        for engine, algorithms in self.engine_supported_algorithms.items():
            if isinstance(algorithms, dict):
                if algo_name in algorithms:
                    return engine
        logger.warning(f"⚠️ Algorithm '{algo_name}' not found in config; fallback to 'networkx'")
        return "networkx"

    def _resolve_tool_name(self, algo_name: str, engine_name: str) -> str:
        """
        Resolve the actual tool name from algorithm and engine.

        Args:
            algo_name: Algorithm name (e.g. "bfs").
            engine_name: Engine name (e.g. "networkx").

        Returns:
            Tool name (e.g. "run_bfs_edges").

        Raises:
            ValueError: If tool name cannot be resolved.
        """
        engine_mapping = self.algorithm_tool_mapping.get(engine_name, {})
        if algo_name in engine_mapping:
            return engine_mapping[algo_name]
        
        logger.error(f"❌ Cannot resolve tool name for algorithm '{algo_name}' in engine '{engine_name}'")
        raise ValueError(f"Algorithm '{algo_name}' has no tool mapping in engine '{engine_name}'. "
                         f"Please check config_servers.yaml")

    async def run_algorithm(self, algo_name: str, parameters: Dict[str, Any], post_processing_code: Optional[str] = None, global_graph: Optional[GraphData] = None) -> Dict[str, Any]:
        """Run the specified algorithm."""
        try:
            engine_name = self._resolve_engine(algo_name)
            if engine_name not in self.clients:
                return {"success": False, "error": f"Engine '{engine_name}' not connected"}

            client = self.clients[engine_name]
            tool_name = self._resolve_tool_name(algo_name, engine_name)
            
            prepared_params = parameters or {}
            if self._should_normalize(engine_name, tool_name):
                prepared_params = self._normalize_parameters(engine_name, tool_name, prepared_params)
            
            logger.info(f"🚀 Running '{algo_name}' (tool: {tool_name}) on engine [{engine_name}]")
            result = await client.call_tool(tool_name, prepared_params, post_processing_code, global_graph)
            return result

        except Exception as e:
            logger.error(f"❌ Algorithm '{algo_name}' failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_algorithm_description(self, algo_name: str) -> tuple[str, Optional[dict]]:
        """Get tool description (input/output schema) for the given algorithm."""
        engine = self._resolve_engine(algo_name)
        client = self.clients.get(engine)
        if not client:
            return f"⚠️ Engine '{engine}' not connected.\n", None

        tool_name = self._resolve_tool_name(algo_name, engine)
        tool_info = client.available_tools.get(tool_name)
        if not tool_info:
            return f"⚠️ Tool '{tool_name}' not found in engine '{engine}'.\n", None

        input_schema = tool_info.get("input_schema", {})
        output_schema = tool_info.get("output_schema", {})
        annotated_input = self._annotate_schema(input_schema, engine)
        doc = (
            f"### `{tool_name}` (Engine: {engine})\n"
            f"**Description:** {tool_info.get('description', 'No description')}\n\n"
            f"**Input Parameters** *(with system-injected annotations)*:\n"
            f"```json\n{json.dumps(annotated_input, indent=2, ensure_ascii=False)}\n```\n\n"
            f"**Output Structure** *(data received by your `process(data)` in the `result` field)*:\n"
            f"```json\n{json.dumps(output_schema, indent=2, ensure_ascii=False)}\n```\n"
            f"{'-'*50}\n"
        )

        tool_metadata = {
            "name": tool_name,
            "engine": engine,
            "description": tool_info.get("description", "No description"),
            "input_params": annotated_input,
            "output_params": output_schema,
        }
        logger.debug(f"🧩 Generated description for '{tool_name}':\n{doc}")
        return doc, tool_metadata

    def _annotate_schema(self, input_schema: Dict, engine_name: str) -> Dict:
        """
        Dispatch to each engine's schema annotation; engines define annotate_schema() in their server module.
        """
        module = self.parameter_modules.get(engine_name)
        if not module:
            logger.debug(f"⚠️ No annotation module for engine '{engine_name}'")
            return input_schema

        annotate_func = getattr(module, "annotate_schema", None)

        if callable(annotate_func):
            try:
                return annotate_func(input_schema)
            except Exception as e:
                logger.error(f"❌ Error running annotate_schema() for '{engine_name}': {e}")
        return input_schema
    

    def _should_normalize(self, engine_name: str, tool_name: str) -> bool:
        module = self.parameter_modules.get(engine_name)
        if not module:
            return False
        guard = getattr(module, "should_normalize", None)
        if callable(guard):
            return guard(tool_name)
        return hasattr(module, "normalize_parameters")

    def _normalize_parameters(self, engine_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        module = self.parameter_modules.get(engine_name)
        if not module:
            return parameters
        normalize_fn = getattr(module, "normalize_parameters", None)
        if not callable(normalize_fn):
            return parameters

        client = self.clients.get(engine_name)
        tool_info = client.available_tools.get(tool_name) if client else None
        try:
            return normalize_fn(tool_name, dict(parameters or {}), tool_info, logger=logger)
        except Exception as exc:
            logger.warning("⚠️ Parameter normalization failed (%s): %s", tool_name, exc)
            return parameters

    def execute_code(self, code: str, data: Any, global_graph: Optional[GraphData] = None, fallback_to_direct_exec: bool = True, is_numeric_analysis: bool = False) -> Any:
        """
        Execute dynamically generated code (numeric analysis, post-processing, etc.).

        Args:
            code: Python code string to execute.
            data: Data passed to the code (dict or other).
            fallback_to_direct_exec: If code is not a process() function, run it directly.
            is_numeric_analysis: If True, unpack as **data; else pass data as-is.

        Returns:
            Execution result.

        Raises:
            RuntimeError: If execution fails.
        """
        try:
            value = self.code_executor.execute(code, data, global_graph=global_graph, is_numeric_analysis=is_numeric_analysis)
            return {
                "algorithm": "numeric_analysis_code",
                "success": True,
                "result": value,
                "error": None,
                "summary": "Numeric analysis code executed successfully.",
        }
        except (ValueError, AttributeError) as e:
            if not fallback_to_direct_exec:
                return {"success": False, "result": None, "error": str(e), "summary": "Numeric analysis code execution failed."}
            try:
                if isinstance(data, dict):
                    namespace = {**data, "__builtins__": __builtins__}
                else:
                    namespace = {"data": data, "__builtins__": __builtins__}
                exec(code, namespace)
                return {
                        "algorithm": "numeric_analysis_code",
                        "success": True,
                        "result": namespace.get("result", data),
                        "error": None,
                        "summary": "Numeric analysis code executed successfully via direct exec fallback.",
                }
            except Exception as fallback_error:
                return {"success": False, "result": None, "error": str(fallback_error), "summary": "Numeric analysis code execution failed."}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e), "summary": "Numeric analysis code execution failed."}

    def initialize_graph_query_engine(self, neo4j_config: Dict[str, Any], reasoner):
        """
        Initialize the graph query engine.

        Args:
            neo4j_config: Neo4j config dict (enabled, uri, user, password).
            reasoner: Reasoner instance for LLM calls.
        """
        try:
            self.neo4j_config = neo4j_config
            self.reasoner = reasoner
            logger.info(
                "📝 Initializing graph query engine; uri=%s user=%s",
                neo4j_config.get("uri"), neo4j_config.get("user"),
            )

            if not neo4j_config.get("enabled", False):
                logger.info("ℹ NaturalLanguageQueryEngine disabled in config")
                self.nl_query_engine = None
                return

            logger.info("📝 Neo4j enabled; creating connection...")
            config = Neo4jConfig(
                uri=neo4j_config.get("uri", "bolt://localhost:7687"),
                user=neo4j_config.get("user", "neo4j"),
                password=neo4j_config.get("password", "")
            )
            logger.info(f"📝 Creating Neo4jGraphClient, uri={config.uri}")
            db_client = Neo4jGraphClient(config)
            logger.info("📝 Creating NaturalLanguageQueryEngine")
            # Wrap reasoner in LLMInterface
            llm_interface = LLMInterface(reasoner)
            self.nl_query_engine = NaturalLanguageQueryEngine(db_client, llm_interface)
            self.nl_query_engine.initialize()
            logger.info("✓ NaturalLanguageQueryEngine initialized in ComputingEngine")
        except Exception as e:
            logger.error(f"✗ NaturalLanguageQueryEngine initialization failed: {e}", exc_info=True)
            self.nl_query_engine = None

    def algorithm_capabilities(self) -> List[Dict[str, Any]]:
        """List algorithms that this executor can actually dispatch right now."""
        from aag.computing_engine.networkx_server.dynamic_tool_registry import UNDIRECTED_ONLY_ALGORITHMS, MAX_ITERATOR_ITEMS

        tools = []
        for engine_name, mapping in self.algorithm_tool_mapping.items():
            client = self.clients.get(engine_name)
            if client is None:
                continue
            for algorithm_id, tool_name in mapping.items():
                # Graph setup is performed by the controller, not a decision action.
                if tool_name in {"run_initialize_graph", "run_initialize_graph_from_file", "run_get_graph_info"}:
                    continue
                actual = client.available_tools.get(tool_name)
                if actual is None:
                    continue
                behavior = {}
                supplied = []
                if engine_name == "networkx":
                    supplied = ["G"]
                    behavior["graph"] = "Uses the MCP working graph initialized from the selected dataset or an explicitly consumed graph. G is injected by the server."
                    behavior["result_serialization"] = f"Sets/tuples become JSON lists, mapping keys become strings, and iterators are materialized. Iterator results exceeding {MAX_ITERATOR_ITEMS} items return an items/truncated/count_is_exact wrapper; a truncated result is not a complete enumeration."
                    behavior["iterator_result_limit"] = MAX_ITERATOR_ITEMS
                    if algorithm_id in UNDIRECTED_ONLY_ALGORITHMS:
                        alternative = UNDIRECTED_ONLY_ALGORITHMS[algorithm_id]
                        behavior["directed_input"] = (
                            f"The existing wrapper dispatches to {alternative}." if alternative else
                            "The existing wrapper converts the working graph to an undirected view for this call."
                        )
                tools.append({"id": algorithm_id, "engine": engine_name, "tool_name": tool_name,
                              "description": actual.get("description", ""),
                              "input_schema": self._annotate_schema(actual.get("input_schema") or {}, engine_name),
                              "output_schema": actual.get("output_schema") or {},
                              "system_supplied_parameters": supplied, "implementation_behavior": behavior})
        return tools

    def graph_query_capabilities(self) -> Dict[str, Any]:
        """Return actual schema and structured graph tools for ReAct."""
        if not self.nl_query_engine:
            return {"schema": {}, "tools": []}
        from aag.computing_engine.graph_query.templates import QUERY_TEMPLATES
        schema = self.nl_query_engine.schema_analyzer.get_schema()
        # Samples may contain personal attributes irrelevant to tool selection.
        public_schema = {
            "node_labels": {name: {"properties": info.get("properties", []),
                                    "identity_key": "node_key" if "node_key" in info.get("properties", []) else None,
                                    "identity_note": "Match the user's full entity identifier literally with identity_key first. Source identity components may use another representation; do not split the identifier or substitute another entity."}
                            for name, info in schema.get("node_labels", {}).items()},
            "relationship_types": {name: {"properties": info.get("properties", [])}
                                   for name, info in schema.get("relationship_types", {}).items()},
            "patterns": schema.get("patterns", []),
        }
        tools = [{"id": name, "type": "graph_query", "description": info.get("description", ""),
                  "required_parameters": info.get("required_params", []),
                  "optional_parameters": info.get("optional_params", []) + ["order_by", "order_direction", "limit", "return_fields"]
                    + (["where"] if name in {"neighbor_query", "path_query", "subgraph", "common_neighbor"} else []),
                  "note": info.get("note", "")}
                 for name, info in QUERY_TEMPLATES.items()]
        notes = {
            "node_lookup": "Prefer key=identity_key from the schema and value=the full user-supplied identifier. Do not guess how source identity components are ordered.",
            "neighbor_query": "Use label/key/value to anchor the requested entity. One-hop property projections use start.node_property, nbr.node_property, rel.relationship_property. Money and timestamps belong to rel when the schema says so.",
            "relationship_filter": "Filters relationship properties only. rel_conditions is an object property->value or property->[operator,value], not a list. Endpoint projections use from.node_property/to.node_property; relationship projections use rel.property. To filter by a named endpoint use neighbor_query instead, not unsupported start_conditions/end_conditions.",
            "path_query": "Use label/key/v1/v2; equal endpoints find closed paths. Return fields may be nodes, relationships, hops, totalAmount, maxAmount, minAmount, avgAmount. Computed amount projections require amount_field set to an actual relationship property. Each relationship preserves id/source/target/properties. Bound max_hops and limit; chronological constraints may be given in where.",
        }
        for tool in tools:
            tool["note"] += " " + notes.get(tool["id"], "")
            tool["parameter_enums"] = {"order_direction": ["ASC", "DESC"]}
            if "direction" in tool["optional_parameters"]:
                tool["parameter_enums"]["direction"] = ["out", "in", "both"]
        return {"schema": public_schema, "tools": tools}

    def execute_graph_query(self, query: str = "", query_type: Optional[str] = None,
                            parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a graph query.

        Args:
            query: User query.

        Returns:
            Result dict with success, results, count, query_type, etc.
        """
        if not self.nl_query_engine:
            return {
                "success": False,
                "error": "Graph query engine not initialized; check Neo4j config."
            }
        try:
            if query_type:
                # ReAct already selected a tool and its arguments. Do not ask
                # several more models to translate the same intent again.
                return self.nl_query_engine.executor.execute(query_type, dict(parameters or {}))
            result = self.nl_query_engine.ask(query)
            return result
        except Exception as e:
            logger.error(f"Graph query execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Graph query execution failed: {str(e)}"
            }

    async def shutdown(self):
        """Disconnect all engines."""
        for name, client in self.clients.items():
            await client.disconnect()
        logger.info("👋 All computing engines disconnected")
