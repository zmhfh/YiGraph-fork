"""Checker unit tests use saved-record fixtures, not real web/model acceptance."""
import copy
import csv
import importlib.util
from pathlib import Path

import networkx as nx
import pytest

from aag.computing_engine.networkx_server.dynamic_tool_registry import _serialize_result
from aag.engine.result_adapter import normalize_result

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('web_verifier', ROOT/'tools/runtime_binder/verify_web_run.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.fixture(scope='module')
def saved_record():
    folder = ROOT/'aag/datasets/data/AMLSim1K/process_graph'
    with (folder/'transactions.csv_vertices.csv').open(encoding='utf-8-sig', newline='') as f:
        nodes = list(csv.DictReader(f))
    with (folder/'transactions.csv_edges.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    graph = nx.MultiDiGraph()
    graph.add_nodes_from((r['acct_id'], dict(r)) for r in nodes)
    for i, row in enumerate(rows):
        graph.add_edge(row['orig_acct'], row['bene_acct'], key=str(i), **row)
    tree = nx.bfs_tree(graph, '1111')
    return {'status': 'completed', 'run_id': 'test-fixture-only', 'dataset': 'AMLSim1K',
        'artifacts': {
            'a1': {'source': 'bfs_tree', 'source_step_id': 1, 'graph_input': _serialize_result(graph),
                   'value': {'original_result': normalize_result('bfs_tree', _serialize_result(tree), {'source': '1111'})}},
            'a2': {'source': 'pagerank', 'source_step_id': 2,
                   'value': {'original_result': nx.pagerank(tree, alpha=0.85, weight=None)},
                   'evidence_scope': {'working_graph': {'metadata': {'vertex_count': len(tree),
                       'edge_count': tree.number_of_edges(), 'directed': True, 'multigraph': False}}}},
        },
        'events': [{'kind': 'runtime_binding', 'data': {'step': step, 'input_refs': [ref]}}
                   for step, ref in [(1, 'G0'), (2, 'v1')]]}


def test_checker_accepts_consistent_saved_execution(saved_record):
    result = VERIFIER.verify(saved_record)
    assert result['passed'], result
    assert result['reference_tree'] == {'nodes': 778, 'edges': 777}


@pytest.mark.parametrize('change', ['score', 'bindings', 'graph_type', 'incomplete'])
def test_checker_rejects_bad_saved_execution(saved_record, change):
    value = copy.deepcopy(saved_record)
    if change == 'score':
        value['artifacts']['a2']['value']['original_result']['1111'] += 0.1
    elif change == 'bindings':
        value['events'] = []
    elif change == 'graph_type':
        value['artifacts']['a2']['evidence_scope']['working_graph']['metadata']['multigraph'] = True
    else:
        value['status'] = 'running'
    assert not VERIFIER.verify(value)['passed']
