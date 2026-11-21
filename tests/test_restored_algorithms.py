"""Exercise restored graph contracts and prevent unsupported tools from resurfacing."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import yaml

from aag.computing_engine.algorithm_catalog import algorithm_tool_catalog
from aag.computing_engine.networkx_server import algorithm_functions as functions
from aag.computing_engine.networkx_server import dynamic_tool_registry as registry


def invoke(name,graph,**parameters):
    processor=SimpleNamespace(graph=graph)
    function=functions.resolve_function(name)
    tool=registry._create_tool_function('run_'+name,function,lambda:processor,lambda f:f)
    return json.loads(tool(**parameters))


def test_backend_placeholders_have_no_callable_implementation():
    assert functions.resolve_function('leiden_communities') is None
    assert functions.resolve_function('leiden_partitions') is None


def test_stale_server_metadata_cannot_reintroduce_backend_placeholders():
    from aag.computing_engine.computing_engine import ComputingEngine
    engine=ComputingEngine.__new__(ComputingEngine)
    engine.parameter_modules={}
    engine.algorithm_tool_mapping={'networkx':{'leiden_communities':'run_leiden_communities','pagerank':'run_pagerank'}}
    engine.clients={'networkx':SimpleNamespace(available_tools={
        'run_leiden_communities':{'input_schema':{}},'run_pagerank':{'input_schema':{}}})}
    assert [tool['id'] for tool in engine.algorithm_capabilities()]==['pagerank']


def test_canonical_edge_cover_works_on_a_non_bipartite_graph():
    graph=nx.cycle_graph(['a','b','c'])
    result=invoke('min_edge_cover',graph)
    assert result['success'] is True
    assert nx.is_edge_cover(graph,[tuple(edge) for edge in result['result']])
    assert len(result['result'])==2


def test_node_component_directed_conversion_matches_advertised_weak_scope():
    graph=nx.DiGraph([('a','b'),('c','b')])
    result=invoke('node_connected_component',graph,n='a')
    assert result['success'] is True and set(result['result'])==set(graph)


def test_required_undocumented_path_arguments_are_in_the_schema():
    schema=registry.generate_input_schema(nx.has_path)
    assert set(schema['required'])=={'source','target'}
    assert 'backend' not in schema['parameters']


def test_restored_algorithms_are_not_silently_cut_off_at_120():
    runtime=[{'id':f'algorithm_{i}','input_schema':{}} for i in range(226)]
    assert len(algorithm_tool_catalog(runtime,{}))==226
    from aag.engine.react_executor import ReActExecutor
    assert ReActExecutor(reasoner=None,computing_engine=None).available_tools_limit is None


def test_comparison_graph_survives_real_json_transport():
    graph=nx.path_graph(['a','b','c'])
    other={'nodes':['x','y','z'],'edges':[['x','y'],['y','z']]}
    assert invoke('is_isomorphic',graph,H=other)['result'] is True
    assert invoke('subgraph_is_monomorphic',graph,H=other)['result'] is True
    other['edges'].append(['x','z'])
    assert invoke('is_isomorphic',graph,H=other)['result'] is False


def test_additional_graph_rejects_undeclared_nodes():
    with pytest.raises(ValueError,match='undeclared'):
        functions.graph_argument({'nodes':['a'],'edges':[['a','b']]})


def test_json_matching_and_empty_conditioning_set_keep_their_meaning():
    assert invoke('is_maximal_matching',nx.path_graph(['a','b','c']),matching=[['a','b']])['result'] is True
    graph=nx.DiGraph([('a','b'),('b','c')])
    assert invoke('is_d_separator',graph,x=['a'],y=['c'],z=[])['result'] is False
    assert invoke('is_d_separator',graph,x=['a'],y=['c'],z=['b'])['result'] is True


def test_tree_decomposition_retains_machine_readable_bags():
    graph=nx.cycle_graph(['a','b','c','d'])
    result=invoke('treewidth_min_fill_in',graph)
    assert result['success']
    width,tree=result['result']
    assert width==2
    assert set().union(*map(set,tree['node_members'].values()))==set(graph)


def test_polynomial_and_infinity_are_explicit_structured_results():
    graph=nx.path_graph(['a','b','c'])
    polynomial=invoke('chromatic_polynomial',graph)['result']
    assert polynomial['type']=='polynomial' and polynomial['variables']==['x']
    result=registry._serialize_result(float('inf'))
    assert result=={'type':'nonfinite','value':'inf'}


def test_random_reference_rejects_impossible_rewiring_instead_of_spinning():
    result=invoke('random_reference',nx.star_graph(5),niter=1,seed=42)
    assert result['success'] is False and 'impossible' in result['error']


def test_configured_networkx_tools_have_local_graph_functions():
    root=Path(__file__).resolve().parents[1]
    configured=yaml.safe_load((root/'aag/computing_engine/config_servers.yaml').read_text())
    for name in configured['engine_supported_algorithms']['networkx']:
        if name in {'initialize_graph','get_graph_info'}:
            continue
        function=functions.resolve_function(name)
        assert function is not None,name
        assert 'G' in inspect.signature(function).parameters,name


def test_active_knowledge_and_category_references_match_executable_configuration():
    root=Path(__file__).resolve().parents[1]
    configured=yaml.safe_load((root/'aag/computing_engine/config_servers.yaml').read_text())
    expected=set(configured['engine_supported_algorithms']['networkx'])|set(configured['engine_supported_algorithms']['pyg'])
    expected-= {'initialize_graph','initialize_graph_pyg','get_graph_info'}
    expected.add('temporal_paths')
    knowledge=yaml.safe_load((root/'aag/knowledge_base/algorithms.yaml').read_text())
    identifiers=[row['id'] for row in knowledge]
    assert len(identifiers)==len(set(identifiers))
    assert set(identifiers)==expected
    tasks=yaml.safe_load((root/'aag/knowledge_base/task_types.yaml').read_text())
    assert {name for task in tasks for name in task.get('algorithm',[])}==expected
    assert not {'leiden_communities','spectral_modularity_bipartition','gcnconv'} & expected


def test_packaged_schemas_match_the_canonical_registered_contracts():
    root=Path(__file__).resolve().parents[1]
    schemas=json.loads((root/'aag/computing_engine/networkx_server/generated_input_schemas.json').read_text())
    assert 'run_leiden_communities' not in schemas
    assert schemas['run_min_edge_cover']==registry.generate_input_schema(nx.min_edge_cover)
    assert schemas['run_has_path']['required']==['source','target']
