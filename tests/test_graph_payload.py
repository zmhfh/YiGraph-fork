import csv
from pathlib import Path
import networkx as nx
import pytest
from aag.expert_search_engine.database.datatype import VertexData, EdgeData, GraphData
from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
from aag.computing_engine.networkx_server.dynamic_tool_registry import _serialize_result
from aag.utils.graph_payload import from_graph_data, normalize_graph, executable_graph, to_graph_data, dataset_for_kind
from aag.engine.result_adapter import normalize_result, traversal_graph


def source():
    return GraphData([VertexData('a'), VertexData('b')], [EdgeData('a','b',properties={'base_amt':10}), EdgeData('a','b',properties={'base_amt':20})])


def test_missing_ranks_keep_both_transactions_and_explicit_duplicates_fail():
    data = source(); processor = GraphComputationProcessor()
    graph = processor.create_graph_from_edges(data.vertices, data.edges, True, True)
    assert graph.number_of_edges() == 2
    assert sum(d['base_amt'] for *_, d in graph.edges(data=True)) == 30
    data.edges[0].rank = data.edges[1].rank = 'same'
    with pytest.raises(ValueError, match='Duplicate explicit'):
        processor.create_graph_from_edges(data.vertices, data.edges, True, True)


def test_actual_amlsim_transactions_are_not_collapsed():
    file = Path(__file__).resolve().parents[1] / 'aag/datasets/data/AMLSim1K/process_graph/transactions.csv_edges.csv'
    with file.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    ids = {r[k] for r in rows for k in ['orig_acct','bene_acct']}
    data = GraphData([VertexData(n) for n in ids], [EdgeData(r['orig_acct'],r['bene_acct'],properties=dict(r)) for r in rows])
    graph = GraphComputationProcessor().create_graph_from_edges(data.vertices, data.edges, True, True)
    assert graph.number_of_edges() == len(rows) == 17512
    assert graph.number_of_edges('1111','17') == 12
    restored = to_graph_data(from_graph_data(data, 'MultiDiGraph'))
    assert len(restored[1]) == len(rows)
    assert [e.properties for e in restored[1]] == [e.properties for e in data.edges]


@pytest.mark.parametrize('kind', ['Graph','DiGraph','MultiGraph','MultiDiGraph'])
def test_graph_roundtrip_keeps_type_keys_and_properties(kind):
    graph = getattr(nx,kind)()
    graph.add_node('a', balance=90); graph.add_node('b', balance=10)
    graph.add_edge('a','b', **({'key':'tx1'} if graph.is_multigraph() else {}), base_amt=20, tran_timestamp='t1')
    if graph.is_multigraph():graph.add_edge('a','b',key='tx2',base_amt=30,tran_timestamp='t2')
    payload = normalize_graph(_serialize_result(graph))
    vertices, edges = to_graph_data(payload)
    result = GraphComputationProcessor().create_graph_from_edges(vertices, edges, graph.is_directed(), graph.is_multigraph())
    assert result.__class__.__name__ == kind
    assert result.number_of_edges() == graph.number_of_edges()
    assert result.nodes['a']['balance'] == 90
    assert sum(d['base_amt'] for *_,d in result.edges(data=True)) == sum(d['base_amt'] for *_,d in graph.edges(data=True))


def test_choose_one_transaction_and_reject_ambiguous_pair():
    data=source(); full=from_graph_data(data,'MultiDiGraph')
    one={**full,'edges':full['edges'][:1]}
    assert len(to_graph_data(one)[1]) == 1
    with pytest.raises(ValueError,match='Ambiguous'):
        executable_graph({'nodes':['a','b'],'edges':[['a','b']]},data,'MultiDiGraph')
    assert len(executable_graph({'nodes':['a','b'],'edges':[['a','b']],'edge_selection':'all_parallel'},data,'MultiDiGraph')['edges']) == 2


def test_bfs_tree_is_a_derived_digraph_even_for_multigraph_source():
    graph=nx.MultiDiGraph();graph.add_edge('a','b',key='t1');graph.add_edge('a','b',key='t2')
    payload=normalize_result('bfs_tree',_serialize_result(nx.bfs_tree(graph,'a')),{'source':'a'})
    assert payload['graph_type']=='DiGraph' and len(payload['edges'])==1
    assert payload['edges'][0]['properties']=={}  # No invented transaction amount.
    data=source()
    assert len(executable_graph(payload,data,'MultiDiGraph')['edges'])==1


def test_tree_and_induced_subgraph_are_distinct_and_empty_bfs_keeps_root():
    graph=nx.DiGraph([('a','b'),('a','c'),('b','c')])
    data=GraphData([VertexData(n) for n in graph], [EdgeData(u,v) for u,v in graph.edges])
    result=normalize_result('bfs_edges',_serialize_result(nx.bfs_edges(graph,'a')),{'source':'a'})
    assert len(traversal_graph(result,'bfs_tree',data,'DiGraph')['edges'])==2
    assert len(traversal_graph(result,'induced_subgraph',data,'DiGraph')['edges'])==3
    isolated=normalize_result('bfs_edges',[],{'source':'a'})
    assert traversal_graph(isolated,'bfs_tree',data,'DiGraph')['nodes']==['a']


def test_reverse_bfs_derived_edges_do_not_have_to_exist_in_G0():
    graph=nx.DiGraph([('a','b')]); tree=nx.bfs_tree(graph,'b',reverse=True)
    payload=normalize_result('bfs_tree',_serialize_result(tree),{'source':'b','reverse':True})
    data=GraphData([VertexData('a'),VertexData('b')],[EdgeData('a','b')])
    assert executable_graph(payload,data,'DiGraph')['edges'][0]['src']=='b'


def test_truncated_traversal_and_unknown_parent_layers_fail_closed():
    value=normalize_result('bfs_edges',{'items':[['a','b']],'truncated':True},{'source':'a'})
    with pytest.raises(ValueError,match='partial'):traversal_graph(value,'bfs_tree',source(),'MultiDiGraph')
    value=normalize_result('bfs_layers',[['a'],['b']],{})
    with pytest.raises(ValueError,match='parents'):traversal_graph(value,'bfs_tree',source(),'MultiDiGraph')


def test_dataset_overlay_is_not_mutated_and_changes_graph_kind():
    config={'schema':{'graph':{'directed':True,'multigraph':True,'weighted':False}}}
    result=dataset_for_kind(config,'DiGraph')
    assert result['schema']['graph']['multigraph'] is False
    assert config['schema']['graph']['multigraph'] is True


def test_keys_that_networkx_considers_equal_cannot_overwrite_a_transaction():
    data=source();data.edges[0].rank=1;data.edges[1].rank=True
    with pytest.raises(ValueError,match='Duplicate explicit'):
        from_graph_data(data,'MultiDiGraph')


def test_business_attribute_named_key_is_not_transaction_identity():
    from aag.utils.graph_payload import to_networkx
    data=source();data.edges[0].properties['key']='business-field'
    graph=to_networkx(from_graph_data(data,'MultiDiGraph'))
    assert graph['a']['b']['__edge_0']['key']=='business-field'
    processor=GraphComputationProcessor()
    assert processor.create_graph_from_edges(data.vertices,data.edges,True,True)['a']['b']['__edge_0']['key']=='business-field'


def test_exact_legacy_transaction_reference_restores_original_properties():
    data=source();data.edges[0].rank='tx-1';data.edges[1].rank='tx-2'
    graph=executable_graph({'nodes':['a','b'],'edges':[['a','b','tx-1']]},data,'MultiDiGraph')
    assert graph['edges'][0]['properties']['base_amt']==10 and len(graph['edges'])==1


def test_string_dataset_flags_are_parsed_as_booleans():
    from aag.utils.graph_payload import kind_from_dataset
    assert kind_from_dataset({'schema':{'graph':{'directed':'false','multigraph':'false'}}})=='Graph'
