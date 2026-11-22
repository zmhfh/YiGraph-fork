"""Real graph execution with a deterministic inference double, never a LoRA score."""
import asyncio
import copy
import json
import csv
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from aag.engine import runtime_binder as binder
from aag.engine.dependency_resolver import DataDependencyResolver
from aag.engine.react_executor import ReActExecutor, RunState
from aag.computing_engine.code_executor import DynamicCodeExecutor
from aag.computing_engine.networkx_server import dynamic_tool_registry as registry
from aag.computing_engine.networkx_server.algorithm_functions import resolve_function, graph_argument
from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
from aag.expert_search_engine.database.datatype import GraphData, VertexData, EdgeData
from aag.utils.graph_payload import to_networkx, from_graph_data, dataset_for_kind, executable_graph, kind_from_dataset, to_graph_data


CONFIG = {'type':'graph','name':'transactions','description':'fixture', 'schema':{
    'vertex':[],'edge':[], 'graph_store_info':{},
    'graph_structure':{'directed':True,'multigraph':True,'weighted':False,'heterogeneous':False}}}


def transactions():
    return GraphData([VertexData(n, {'balance':5}) for n in ['a','b','c','isolated']], [
        EdgeData('a','b','t1',{'base_amt':10,'tran_timestamp':'2026-01-01'}),
        EdgeData('a','b','t2',{'base_amt':20,'tran_timestamp':'2026-01-02'}),
        EdgeData('a','c','t3',{'base_amt':30}), EdgeData('b','c','t4',{'base_amt':40})])


def invoke(name, graph, **params):
    tool = registry._create_tool_function('run_'+name, resolve_function(name),
                                         lambda:SimpleNamespace(graph=graph), lambda f:f)
    result = json.loads(tool(**params))
    assert result['success'], result
    return result


class Engine:
    def __init__(self):
        self.processor = GraphComputationProcessor()
        self.calls = []

    async def get_algorithm_description(self, name):
        return name, {'name':name}

    async def run_algorithm(self, name, params, *args, **kwargs):
        self.calls.append((name,copy.deepcopy(params)))
        if name == 'initialize_graph':
            data = GraphData.from_dict(params)
            flags = params['dataset_config']['schema']['graph_structure']
            graph = self.processor.create_graph_from_edges(data.vertices,data.edges,flags['directed'],flags['multigraph'])
            return {'success':True,'result':{'nodes':len(graph),'edges':graph.number_of_edges()}}
        return invoke(name,self.processor.graph,**params)


@pytest.mark.parametrize('name', ['bfs_tree','bfs_edges','bfs_successors','bfs_predecessors','bfs_layers','edge_bfs'])
def test_every_registered_bfs_tool_returns_a_typed_result(name):
    graph = to_networkx(from_graph_data(transactions(),'MultiDiGraph'))
    params = {'sources':['a']} if name=='bfs_layers' else {'source':'a'}
    value = invoke(name,graph,**params)['result']
    assert value['complete'] is True
    if name=='bfs_tree':
        assert value['graph_type']=='DiGraph' and len(value['edges'])==2
    else:
        assert value['algorithm']==name and value['type'] in {'edge_set','parent_mapping','node_layers'}
        if name=='edge_bfs':
            assert len(value['items'])==4
            assert {e[2] for e in value['items']}=={'t1','t2','t3','t4'}


@pytest.mark.parametrize('kind',['Graph','DiGraph','MultiGraph','MultiDiGraph'])
def test_runtime_binder_can_select_G0_for_every_graph_kind(monkeypatch,kind):
    data=transactions()
    if not kind.startswith('Multi'):data.edges=data.edges[1:]
    config=dataset_for_kind(CONFIG,kind)
    executor=ReActExecutor(reasoner=None,computing_engine=Engine(),global_graph=data,current_graph_dataset=config)
    seen=[]
    def reply(url,request):
        seen.append(request);return {'input_refs':['G0']}
    monkeypatch.setenv('YIGRAPH_BINDER_URL','http://test')
    monkeypatch.delenv('YIGRAPH_BINDER_COMMAND',raising=False)
    monkeypatch.setenv('YIGRAPH_BINDER_GRAPH_KIND','Graph')  # Loaded schema wins.
    monkeypatch.setattr(binder,'_post',reply)
    action={'name':'pagerank','objective':'Use the complete original graph'}
    bound=asyncio.run(binder.bind_prepared_inputs(executor,action,RunState(goal='Rank',step=1),{'parameters':{}}))
    asyncio.run(executor._initialize_working_graph(bound['graph_adapter_result']))
    graph=executor.computing_engine.processor.graph
    assert graph.__class__.__name__==kind and graph.number_of_edges()==len(data.edges)
    assert seen[0]['actual_graph_attributes']['multigraph']==kind.startswith('Multi')


def test_real_react_bfs_then_pagerank_uses_tree_not_original_transactions(monkeypatch):
    monkeypatch.delenv('YIGRAPH_BINDER_URL',raising=False)
    monkeypatch.delenv('YIGRAPH_BINDER_COMMAND',raising=False)
    engine=Engine(); data=transactions(); config=copy.deepcopy(CONFIG)
    resolver=DataDependencyResolver(reasoner=None)
    executor=ReActExecutor(reasoner=None,computing_engine=engine,dependency_resolver=resolver,
                           global_graph=data,current_graph_dataset=config)
    state=RunState(goal='Rank the BFS tree',step=1)
    bfs={'type':'algorithm','name':'bfs_tree','objective':'BFS from a','parameters':{'source':'a'}}
    result,meta=asyncio.run(executor._execute_action(bfs,state))
    aid=executor._add_artifact(state,bfs,result,meta,1)
    state.step=2
    rank={'type':'algorithm','name':'pagerank','objective':'Rank this BFS tree',
          'parameters':{'weight':None},'consumed':[{'artifact_id':aid,'use_as':'graph'}]}
    result,_=asyncio.run(executor._execute_action(rank,state))
    assert isinstance(engine.processor.graph,nx.DiGraph) and not engine.processor.graph.is_multigraph()
    assert engine.processor.graph.number_of_edges()==2
    expected=nx.pagerank(nx.bfs_tree(to_networkx(from_graph_data(data,'MultiDiGraph')),'a'),weight=None)
    assert result['result']==pytest.approx(expected)
    assert config==CONFIG


def test_binder_accepts_derived_tree_type_and_preserves_transaction_subset(monkeypatch):
    data=transactions();engine=Engine()
    executor=ReActExecutor(reasoner=None,computing_engine=engine,global_graph=data,current_graph_dataset=CONFIG)
    tree=invoke('bfs_tree',to_networkx(from_graph_data(data,'MultiDiGraph')),source='a')['result']
    state=RunState(goal='Rank BFS tree',step=2,artifacts={'a1':{'value':tree,'source':'bfs_tree','source_step_id':1}})
    monkeypatch.setenv('YIGRAPH_BINDER_URL','http://test');monkeypatch.delenv('YIGRAPH_BINDER_COMMAND',raising=False)
    monkeypatch.setattr(binder,'_post',lambda *args:{'input_refs':['v1']})
    result=asyncio.run(binder.bind_prepared_inputs(executor,{'name':'pagerank'},state,{'parameters':{}}))
    asyncio.run(executor._initialize_working_graph(result['graph_adapter_result']))
    assert engine.processor.graph.__class__.__name__=='DiGraph' and engine.processor.graph.number_of_edges()==2
    one=from_graph_data(data,'MultiDiGraph');one['edges']=one['edges'][:1]
    asyncio.run(executor._initialize_working_graph(one))
    assert engine.processor.graph.number_of_edges()==1
    assert engine.processor.graph['a']['b']['t1']['base_amt']==10


def test_traversal_conversion_uses_actual_input_snapshot_and_requires_mode():
    data=transactions();resolver=DataDependencyResolver(None);resolver.set_graph_context(data,CONFIG)
    snapshot=from_graph_data(data,'MultiDiGraph');snapshot['edges']=snapshot['edges'][:1];snapshot['nodes']=['a','b']
    value=invoke('bfs_edges',to_networkx(snapshot),source='a')['result']
    artifacts={'a1':{'value':value,'graph_input':snapshot}}
    ref={'artifact_id':'a1','use_as':'graph','graph_mode':'induced_subgraph'}
    args=dict(current_question='Use visited nodes',alg_des_info={},artifacts=artifacts,consumed=[ref])
    bound=asyncio.run(resolver.prepare_react_inputs(**args))
    assert not bound['errors'] and len(bound['graph_adapter_result']['edges'])==1
    ref.pop('graph_mode')
    assert asyncio.run(resolver.prepare_react_inputs(**args))['errors']


def test_bfs_tree_can_explicitly_request_the_induced_transaction_graph():
    data=transactions();snapshot=from_graph_data(data,'MultiDiGraph')
    resolver=DataDependencyResolver(None);resolver.set_graph_context(data,CONFIG)
    value=invoke('bfs_tree',to_networkx(snapshot),source='a')['result']
    ref={'artifact_id':'a1','use_as':'graph','graph_mode':'induced_subgraph'}
    bound=asyncio.run(resolver.prepare_react_inputs(current_question='Induce visited accounts',alg_des_info={},
        artifacts={'a1':{'value':value,'graph_input':snapshot}},consumed=[ref]))
    assert not bound['errors'] and len(bound['graph_adapter_result']['edges'])==4
    assert bound['graph_adapter_result']['graph_type']=='MultiDiGraph'


def test_empty_derived_graph_never_falls_back_to_G0():
    engine=Engine();executor=ReActExecutor(reasoner=None,computing_engine=engine,global_graph=transactions(),current_graph_dataset=CONFIG)
    asyncio.run(executor._initialize_working_graph({'type':'graph','graph_type':'DiGraph','nodes':[],'edges':[]}))
    assert len(engine.processor.graph)==0 and engine.processor.graph.number_of_edges()==0


def test_postprocessor_and_comparison_graph_keep_keys_attributes_and_zero_node():
    graph=nx.MultiDiGraph();graph.add_node(0,balance=10)
    graph.add_edge(0,1,key='tx1',base_amt=30,tran_timestamp='t1');graph.add_edge(0,1,key='tx2',base_amt=50)
    value=registry._serialize_result(graph)
    code="def process(data):\n return {'edges':data.number_of_edges(),'amount':sum(d['base_amt'] for *_,d in data.edges(data=True)), 'balance':data.nodes['0']['balance']}"
    result=DynamicCodeExecutor(auto_install=False).execute(code,value)
    assert result=={'edges':2,'amount':80,'balance':10}
    assert set(graph_argument(value)['0']['1'])=={'tx1','tx2'}
    with pytest.raises(ValueError,match='truncated'):
        DynamicCodeExecutor(auto_install=False).execute(code,{**value,'complete':False})


def test_partial_traversal_and_unavailable_graph_are_not_executed(monkeypatch):
    monkeypatch.setattr(registry,'MAX_ITERATOR_ITEMS',1)
    data=transactions();resolver=DataDependencyResolver(None);resolver.set_graph_context(data,CONFIG)
    value=invoke('bfs_edges',to_networkx(from_graph_data(data,'MultiDiGraph')),source='a')['result']
    bound=asyncio.run(resolver.prepare_react_inputs(current_question='Tree',alg_des_info={},
        artifacts={'a1':{'value':value,'graph_input':from_graph_data(data,'MultiDiGraph')}},
        consumed=[{'artifact_id':'a1','use_as':'graph','graph_mode':'bfs_tree'}]))
    assert bound['errors'] and bound['graph_adapter_result'] is None


def test_multigraph_induced_helper_preserves_all_transactions():
    resolver=DataDependencyResolver(None);resolver.set_graph_context(transactions(),CONFIG)
    value=resolver._graph_conversion_environment()['induced_graph'](['a','b'])
    assert len(value['edges'])==2 and {e['key'] for e in value['edges']}=={'t1','t2'}


def test_actual_amlsim_bfs_tree_pagerank_matches_same_graph_reference(monkeypatch):
    monkeypatch.delenv('YIGRAPH_BINDER_URL',raising=False);monkeypatch.delenv('YIGRAPH_BINDER_COMMAND',raising=False)
    file=Path(__file__).resolve().parents[1]/'aag/datasets/data/AMLSim1K/process_graph/transactions.csv_edges.csv'
    with file.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    nodes={r[k] for r in rows for k in ['orig_acct','bene_acct']}
    data=GraphData([VertexData(n) for n in nodes],[EdgeData(r['orig_acct'],r['bene_acct'],str(i),dict(r)) for i,r in enumerate(rows)])
    engine=Engine();resolver=DataDependencyResolver(None)
    runner=ReActExecutor(reasoner=None,computing_engine=engine,dependency_resolver=resolver,global_graph=data,current_graph_dataset=CONFIG)
    state=RunState(goal='Compute PageRank on the BFS tree rooted at 1111',step=1)
    bfs={'type':'algorithm','name':'bfs_tree','parameters':{'source':'1111'}}
    result,meta=asyncio.run(runner._execute_action(bfs,state))
    assert engine.calls[0][0]=='initialize_graph' and engine.processor.graph.number_of_edges()==17512
    aid=runner._add_artifact(state,bfs,result,meta,1);state.step=2
    rank={'type':'algorithm','name':'pagerank','parameters':{'weight':None},'consumed':[{'artifact_id':aid,'use_as':'graph'}]}
    result,_=asyncio.run(runner._execute_action(rank,state))
    reference=nx.bfs_tree(to_networkx(from_graph_data(data,'MultiDiGraph')),'1111')
    assert result['result']==pytest.approx(nx.pagerank(reference,weight=None))
    assert engine.processor.graph.number_of_edges()==len(reference)-1


def test_legacy_scheduler_uses_same_lossless_initializer_without_database_imports():
    # Test the actual method independently of optional Neo4j/Nebula/RAG imports.
    file=Path(__file__).resolve().parents[1]/'aag/engine/scheduler.py'
    tree=ast.parse(file.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
    method=next(n for n in cls.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='_prepare_graph_for_execution')
    env={'executable_graph':executable_graph,'kind_from_dataset':kind_from_dataset,'to_graph_data':to_graph_data,
         'dataset_for_kind':dataset_for_kind,'logger':logging.getLogger(__name__)}
    exec(compile('from __future__ import annotations\n'+ast.unparse(method),str(file),'exec'),env)
    engine=Engine();runner=SimpleNamespace(global_graph=transactions(),current_graph_dataset=CONFIG,computing_engine=engine)
    value=from_graph_data(transactions(),'MultiDiGraph');value['edges']=value['edges'][:1]
    asyncio.run(env[method.name](runner,graph_dependencies=[object()],graph_adapter_result=value))
    assert engine.processor.graph.number_of_edges()==1 and engine.processor.graph['a']['b']['t1']['base_amt']==10
