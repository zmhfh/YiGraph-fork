import os
from pathlib import Path

def get_project_root(project_name="aag"):
    path = Path(__file__).resolve()
    for parent in path.parents:
        if parent.name == project_name:
            return parent
    raise RuntimeError(f"Project root '{project_name}' not found.")

PROJECT_ROOT = get_project_root()
# Keep source-checkout behavior; installed packages use a writable workspace.
_source_checkout = (PROJECT_ROOT.parent / "config" / "engine_config.yaml").is_file()
_configured_home = os.environ.get("YIGRAPH_HOME")
RUNTIME_ROOT = (
    Path(_configured_home).expanduser().resolve()
    if _configured_home
    else PROJECT_ROOT.parent if _source_checkout else Path.home() / ".yigraph"
)
_source_resources = _source_checkout and not _configured_home
KNOWLEDGE_BASE_DIR = (PROJECT_ROOT if _source_resources else RUNTIME_ROOT) / "knowledge_base"
TASK_TYPES_PATH = KNOWLEDGE_BASE_DIR / "task_types.yaml"
ALGORITHMS_PATH = KNOWLEDGE_BASE_DIR / "algorithms.yaml"
KNOWLEDGE_PATH = KNOWLEDGE_BASE_DIR / "knowledge.yaml"

DEFAULT_CONFIG_PATH = RUNTIME_ROOT / "config" / "engine_config.yaml"
DEFAULT_DATA_UPLOAD_CONFIG_PATH = RUNTIME_ROOT / "config" / "data_upload_config.yaml"

DATA_SCHEMA_PATH2 = PROJECT_ROOT / "data_pipeline" / "data_transformer" / "dataset_schemas"
GRAPH_SCHEMA_PATH = DATA_SCHEMA_PATH2 / "graph_schemas.yaml"
TABLE_SCHEMA_PATH = DATA_SCHEMA_PATH2 / "table_schemas.yaml"
TEXT_SCHEMA_PATH = DATA_SCHEMA_PATH2 / "text_schemas.yaml"

DEFAULT_CONFIG_SERVER_PATH = (
    PROJECT_ROOT / "computing_engine" / "config_servers.yaml"
    if _source_resources else RUNTIME_ROOT / "config" / "config_servers.yaml"
)

DATASETS_DIR = (PROJECT_ROOT if _source_resources else RUNTIME_ROOT) / "datasets"
DATASETS_DATA_DIR = DATASETS_DIR / "data"
DATASETS_SCHEMA_DIR = DATASETS_DIR / "dataset_schemas"
DATASETS_INDEX_PATH = DATASETS_SCHEMA_DIR / "datasets.yaml"
