"""Verify one saved YiGraph web run against its actual input graph.

The test question must contain WEB-BFS-1111. This checks binding/execution,
not the factual accuracy of every sentence in the generated report.
"""
import argparse
import json
import math
import sys
import tempfile
from pathlib import Path


def unwrap(value):
    while isinstance(value, dict) and 'original_result' in value:
        value = value['original_result']
    return value


def verify(record):
    import networkx as nx
    from aag.utils.graph_payload import to_networkx

    artifacts = list(record.get('artifacts', {}).values())
    def find(name):
        candidates = [a for a in artifacts if str(a.get('source', '')).removeprefix('run_') == name]
        if not candidates:
            raise ValueError(f'No saved {name} artifact; inspect the run errors')
        return max(candidates, key=lambda a: int(a.get('source_step_id', 0)))

    bfs, rank = find('bfs_tree'), find('pagerank')
    snapshot = bfs.get('graph_input')
    if not snapshot:
        raise ValueError('Missing graph_input: this run may come from the old backend')
    original = to_networkx(snapshot)
    actual_tree_payload = unwrap(bfs['value'])
    actual_tree = to_networkx(actual_tree_payload)
    source = actual_tree_payload.get('source')
    if source is None:
        raise ValueError('The BFS artifact does not preserve its source')
    # The webpage test requests unrestricted forward BFS and unweighted PageRank.
    reference_tree = nx.bfs_tree(original, source)
    expected = nx.pagerank(reference_tree, alpha=0.85, weight=None, max_iter=100, tol=1e-6)
    actual = unwrap(rank['value'])
    numeric_match = isinstance(actual, dict) and set(actual) == set(expected)
    if numeric_match:
        numeric_match = all(isinstance(actual[n], (int, float)) and not isinstance(actual[n], bool)
                            and math.isclose(actual[n], score, rel_tol=1e-8, abs_tol=1e-10)
                            for n, score in expected.items())

    bindings = [e['data'] for e in record.get('events', []) if e.get('kind') == 'runtime_binding']
    steps = {int(e.get('step', -1)) for e in bindings}
    metadata = rank.get('evidence_scope', {}).get('working_graph', {}).get('metadata', {})
    source_account = str(source) == '1111' or str(original.nodes[source].get('acct_id', '')) == '1111'
    checks = {
        'run_completed': record.get('status') == 'completed',
        'binder_called_for_both_algorithms': all(int(a['source_step_id']) in steps for a in [bfs, rank]),
        'original_transaction_graph': original.__class__.__name__ == 'MultiDiGraph' and len(original) == 1446 and original.number_of_edges() == 17512,
        'source_account_is_1111': source_account,
        'bfs_tree_matches_actual_input': actual_tree.__class__.__name__ == 'DiGraph' and set(actual_tree) == set(reference_tree)
            and set(actual_tree.edges()) == set(reference_tree.edges()),
        'pagerank_received_tree': metadata.get('vertex_count') == len(reference_tree)
            and metadata.get('edge_count') == reference_tree.number_of_edges()
            and metadata.get('directed') is True and metadata.get('multigraph') is False,
        'all_pagerank_scores_match': bool(numeric_match),
    }
    return {'passed': all(checks.values()), 'run_id': record.get('run_id'), 'dataset': record.get('dataset'),
            'scope': 'Saved web execution, Binder events and numerical results; report prose requires separate review.',
            'checks': checks, 'original_graph': {'nodes': len(original), 'edges': original.number_of_edges()},
            'reference_tree': {'nodes': len(reference_tree), 'edges': reference_tree.number_of_edges()},
            'pagerank_top5': sorted(expected.items(), key=lambda item: (-item[1], str(item[0])))[:5],
            'bindings': [{'step': e.get('step'), 'input_refs': e.get('input_refs'),
                          'selected_graph': e.get('selected_graph')} for e in bindings]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--record', type=Path, help='Exact saved run JSON; otherwise locate the latest marked web run')
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from aag.utils.path_utils import RUNTIME_ROOT
    path = args.record
    if path is None:
        for candidate in sorted((RUNTIME_ROOT/'output/react-runs').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            record = json.loads(candidate.read_text(encoding='utf-8'))
            if 'WEB-BFS-1111' in record.get('goal', ''):
                path = candidate
                break
        if path is None:
            raise SystemExit('No WEB-BFS-1111 web run found; submit the marked question in the modified backend first')
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        report = verify(record)
    except Exception as exc:
        report = {'passed': False, 'error': repr(exc)}
    destination = repo/'runtime/binder'
    destination.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='web-verification-', dir=destination))/'verification.json'
    report['record'] = str(path)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Verification:', output)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
