"""Check YiGraph's configured Neo4j connection without requiring an LLM."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aag.config.engine_config import load_config_from_yaml
from aag.computing_engine.graph_query.graph_query import Neo4jConfig, Neo4jGraphClient
from aag.utils.path_utils import DEFAULT_CONFIG_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    settings = load_config_from_yaml(args.config).retrieval.database.neo4j
    if not settings.get("enabled", False):
        parser.exit(1, "Neo4j is disabled in the engine configuration.\n")
    with Neo4jGraphClient(Neo4jConfig(
        uri=settings["uri"],
        user=settings.get("user", "neo4j"),
        password=settings.get("password", ""),
    )) as client:
        version = client.run(
            "CALL dbms.components() YIELD name, versions RETURN name, versions",
            show_query=False,
        )[0]
        nodes = client.run("MATCH (n) RETURN count(n) AS count", show_query=False)[0]["count"]
        edges = client.run("MATCH ()-[r]->() RETURN count(r) AS count", show_query=False)[0]["count"]
        print(f"Connected to {settings['uri']} as {settings.get('user', 'neo4j')}")
        print(f"{version['name']}: {', '.join(version['versions'])}")
        print(f"Nodes: {nodes}; relationships: {edges}")


if __name__ == "__main__":
    main()
