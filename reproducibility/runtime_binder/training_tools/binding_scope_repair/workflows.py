import hashlib
import json
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import networkx as nx

from binding_examples import examples


TYPES = ('sequential', 'parallel', 'convergence', 'divergence', 'diamond')
DOMAINS = ['A city transit planning team studies intersections and routes.',
           'An ecology team studies contacts between habitat patches.',
           'A library team indexes collaboration across branches.',
           'A logistics group tracks connections among delivery depots.',
           'A research group studies coauthorship links among researchers.',
           'A campus team models movement between buildings.']
METRICS = ('degree_centrality', 'clustering', 'average_neighbor_degree',
           'core_number', 'pagerank', 'degree_centrality')
SECOND = ('average_clustering', 'core_number', 'pagerank',
          'average_neighbor_degree', 'degree_centrality', 'average_clustering')
THIRD = ('dijkstra_path_length', 'average_clustering', 'core_number',
         'pagerank', 'average_neighbor_degree', 'clustering')


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def graph_edges(g):
    return sorted((min(a, b), max(a, b)) for a, b in g.edges)


def output_preview(value):
    if isinstance(value, nx.Graph):
        return {'type': type(value).__name__}
    if isinstance(value, dict):
        return {'type': 'mapping_preview' if len(value) > 24 else 'mapping'}
    return value


def rank(scores, k):
    return sorted(scores, key=lambda node: (-scores[node], node))[:k]


class Workflow:
    def __init__(self, graph):
        self.objects = {'G0': graph}
        self.events = []
        self.steps = []
        self.edges = []
        self.provenance = {'G0': set()}

    def record(self, ref, algorithm, result, source_refs):
        assert ref not in self.objects
        self.objects[ref] = result
        self.events.append({'step_or_transform': ref, 'algorithm': algorithm,
                            'input_refs': list(source_refs), 'output': output_preview(result)})
        self.provenance[ref] = {ref}.union(*(self.provenance[src] for src in source_refs))
        return result

    def induced(self, ref, parent, nodes, *other_refs):
        source = self.objects[parent]
        selected = set(nodes)
        assert selected and selected <= set(source)
        subgraph = source.subgraph(selected).copy()
        edge_set = {(min(u,v),max(u,v)) for u,v in source.edges if u in selected and v in selected}
        assert set(subgraph) == selected and set(graph_edges(subgraph)) == edge_set
        return self.record(ref, None, subgraph, (parent, *other_refs))

    def call(self, name, algorithm, graph_ref, desc, parents=()):
        graph = self.objects[graph_ref]
        assert isinstance(graph, nx.Graph) and set(parents) <= self.provenance[graph_ref]
        if algorithm == 'dijkstra_path_length':
            value = dict(nx.all_pairs_dijkstra_path_length(graph, weight=None))
            assert value == dict(nx.all_pairs_shortest_path_length(graph))
        elif algorithm == 'pagerank':
            value = nx.pagerank(graph, max_iter=1000, tol=1e-9)
        else:
            value = getattr(nx, algorithm)(graph)
        self.steps.append({'id':name,'algorithm':algorithm,'description':desc})
        self.edges.extend({'id':f'e{len(self.edges)+i+1}', 'from':p,'to':name} for i,p in enumerate(parents))
        return self.record(name, algorithm, value, (graph_ref,))


def workflow(kind, i, graph):
    w = Workflow(graph)
    k = 52 + 7*i
    a, b, c = METRICS[i], SECOND[i], THIRD[i]
    domain = DOMAINS[i]
    if kind == 'sequential':
        q = (f'{domain} Rank the full network by {a}, then retain its top {k} vertices and all of their mutual edges. '
             f'Compute {b} on the resulting induced subgraph.')
        scores = w.call('v1', a, 'G0', f'Compute {a} for each vertex in the full input graph.')
        w.induced('working_graph', 'G0', rank(scores, k), 'v1')
        w.induced('sample_graph', 'G0', range(k))
        w.call('v2', b, 'working_graph', f'Compute {b} on the induced subgraph of vertices selected using the v1 ranking.', ('v1',))
    elif kind == 'parallel':
        w.induced('range_graph', 'G0', range(k))
        scope = f'the induced subgraph containing only vertices with IDs 0 through {k-1}'
        full = 'the full original graph'
        first, second = ('G0','range_graph') if i % 2 else ('range_graph','G0')
        text = {'G0':full,'range_graph':scope}
        q = f'{domain} Independently compute {a} on {text[first]} and {b} on {text[second]}. Report both measurements.'
        w.call('v1', a, first, f'Compute {a} on {text[first]}.')
        w.call('v2', b, second, f'Compute {b} on {text[second]}. This is an independent analysis.')
    elif kind == 'convergence':
        other = 'average_neighbor_degree' if a != 'average_neighbor_degree' else 'degree_centrality'
        q = (f'{domain} Make two full-graph rankings using {a} and {other}. Take the union of their top {k} '
             f'vertices and retain the original edges within it. Evaluate {b} on this combined subgraph.')
        scores = w.call('v1', a, 'G0', f'Compute {a} on the complete input graph for the first ranking.')
        w.induced('first_ranking_graph', 'G0', rank(scores, k), 'v1')
        other_scores = w.call('v2', other, 'G0', f'Compute {other} on the complete input graph for a separate ranking.')
        w.induced('joined_graph', 'G0', set(rank(scores,k)) | set(rank(other_scores,k)), 'v1','v2')
        w.call('v3', b, 'joined_graph', f'Use the induced subgraph formed by combining both top-{k} rankings from v1 and v2, then compute {b}.', ('v1','v2'))
    elif kind == 'divergence':
        q = (f'{domain} Use {a} on the full graph to select {k} vertices. Keep all original edges within that '
             f'selection and independently measure {b} and {c} on the same selected subgraph.')
        scores = w.call('v1',a,'G0',f'Rank vertices of the full input graph by {a}.')
        w.induced('working_graph','G0',rank(scores,k),'v1')
        w.induced('sample_graph','G0',range(k))
        w.call('v2',b,'working_graph',f'Run {b} on the induced subgraph derived from the v1 rankings.',('v1',))
        w.call('v3',c,'working_graph',f'Run {c} on the same induced subgraph from v1.',('v1',))
    else:
        q = (f'{domain} Rank the full graph by {a}, then preserve the induced subgraph of its top {k} vertices. '
             'On this initial scope rank by average neighbor degree and local clustering independently. '
             f'Combine the top {k//2} vertices from each of those two rankings, retain their mutual edges, and compute {b} in this refined scope.')
        scores = w.call('v1',a,'G0',f'Compute {a} on the whole original graph to determine the initial scope.')
        w.induced('first_core','G0',rank(scores,k),'v1')
        neighbors = w.call('v2','average_neighbor_degree','first_core',
                           'Compute average neighbor degree within the initial scope built using v1.',('v1',))
        clustered = w.call('v3','clustering','first_core',
                           'Compute local clustering within the initial scope built using v1.',('v1',))
        combined = set(rank(neighbors,k//2)) | set(rank(clustered,k//2))
        w.induced('second_core','first_core',combined,'v2','v3')
        w.call('v4',b,'second_core',f'Compute {b} on the refined graph built by combining the rankings from v2 and v3.',('v2','v3'))
    return w, q


def rename(sample, aliases):
    changed = deepcopy(sample)
    for field in ('input','output'):
        obj = json.loads(sample[field])
        def ref(name):
            return aliases.get(name,name)
        if field == 'input':
            for item in obj['available_inputs']:
                item['ref'] = ref(item['ref'])
                if 'derived_from' in item:
                    item['derived_from'] = [ref(n) for n in item['derived_from']]
        else:
            obj['input_refs'] = [ref(n) for n in obj['input_refs']]
        changed[field] = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    return changed


def stable_value(value):
    if isinstance(value,nx.Graph):
        return {'nodes':sorted(value),'edges':graph_edges(value)}
    return value


