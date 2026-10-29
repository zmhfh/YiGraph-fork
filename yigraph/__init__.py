"""Public Python API. Heavy model dependencies load only when requested."""

__version__ = "2.0.0"

__all__ = ["AAGEngine", "GraphComputationProcessor", "VertexData", "EdgeData", "GraphData", "load_config", "init_workspace"]


def load_config(path=None):
    """Load an engine YAML configuration without connecting to external services."""
    from aag.config.engine_config import load_config_from_yaml
    from aag.utils.path_utils import DEFAULT_CONFIG_PATH

    return load_config_from_yaml(path if path is not None else DEFAULT_CONFIG_PATH)


def init_workspace(path):
    """Copy bundled templates and demo data to a writable workspace."""
    from .workspace import init_workspace as initialize

    return initialize(path)


def __getattr__(name):
    if name == "AAGEngine":
        try:
            from aag.engine.aag_engine import AAGEngine
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "AAGEngine requires the engine extra: install 'yigraph[engine]' "
                "from your wheel or source checkout. Missing module: " + str(exc.name)
            ) from exc
        return AAGEngine
    if name == "GraphComputationProcessor":
        from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor

        return GraphComputationProcessor
    if name in {"VertexData", "EdgeData", "GraphData"}:
        from aag.expert_search_engine.database import datatype

        return getattr(datatype, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
