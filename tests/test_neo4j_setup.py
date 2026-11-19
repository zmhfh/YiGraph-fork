"""Check external credentials and the supported Neo4j 3.5 schema syntax."""

from unittest.mock import MagicMock

import pytest

from aag.config.engine_config import load_config_from_yaml, resolve_neo4j_config


def test_password_file_is_relative_to_engine_config(tmp_path, monkeypatch):
    monkeypatch.delenv("YIGRAPH_NEO4J_PASSWORD", raising=False)
    (tmp_path / "password.txt").write_text("local-password\n", encoding="utf-8")
    config = tmp_path / "engine_config.yaml"
    config.write_text(
        "retrieval:\n  database:\n    neo4j:\n"
        "      enabled: true\n      password_file: password.txt\n",
        encoding="utf-8",
    )
    assert load_config_from_yaml(config).retrieval.database.neo4j["password"] == "local-password"
    assert "local-password" not in config.read_text(encoding="utf-8")


def test_password_environment_override_does_not_require_file(tmp_path, monkeypatch):
    monkeypatch.setenv("YIGRAPH_NEO4J_PASSWORD", "environment-password")
    original = {"password_file": str(tmp_path / "missing")}
    resolved = resolve_neo4j_config(original)
    assert resolved["password"] == "environment-password"
    assert "password" not in original


def test_neo4j35_indexes_can_be_created_again():
    pytest.importorskip("pandas")
    pytest.importorskip("neo4j")
    from neo4j.exceptions import ClientError
    from aag.computing_engine.graph_query.load_data_into_neo4j import Neo4jGraphLoader

    loader = Neo4jGraphLoader.__new__(Neo4jGraphLoader)
    loader.driver = MagicMock()
    session = loader.driver.session.return_value.__enter__.return_value
    existing = ClientError.hydrate(
        message="Already exists",
        code="Neo.ClientError.Schema.IndexAlreadyExists",
    )
    version = MagicMock()
    version.single.return_value = {"version": "3.5.25"}
    session.run.side_effect = [version, MagicMock(), existing]
    assert loader.create_constraints_and_indexes([{"type": "account", "id_field": "acct_id"}])
    statements = [call.args[0] for call in session.run.call_args_list[1:]]
    assert statements == [
        "CREATE CONSTRAINT ON (n:`Account`) ASSERT n.node_key IS UNIQUE",
        "CREATE INDEX ON :`Account`(`acct_id`)",
    ]


def test_schema_permission_errors_are_reported():
    pytest.importorskip("pandas")
    pytest.importorskip("neo4j")
    from neo4j.exceptions import ClientError
    from aag.computing_engine.graph_query.load_data_into_neo4j import Neo4jGraphLoader

    loader = Neo4jGraphLoader.__new__(Neo4jGraphLoader)
    loader.driver = MagicMock()
    session = loader.driver.session.return_value.__enter__.return_value
    forbidden = ClientError.hydrate(
        message="Permission denied", code="Neo.ClientError.Security.Forbidden",
    )
    version = MagicMock()
    version.single.return_value = {"version": "3.5.25"}
    session.run.side_effect = [version, forbidden]
    assert not loader.create_constraints_and_indexes([{"type": "account", "id_field": "acct_id"}])
