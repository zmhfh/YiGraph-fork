"""Exercise actual stdio MCP registration and JSON transport, without an LLM."""
import asyncio
from pathlib import Path

import networkx as nx
import pytest

pytest.importorskip('mcp')
from aag.computing_engine.mcp_client import GraphMCPClient
from aag.utils.graph_payload import to_graph_data, dataset_for_kind, to_networkx


def test_real_mcp_multigraph_bfs_and_empty_graph(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    # UTF-8 must apply to the child server on Windows too.
    monkeypatch.setenv('PYTHONUTF8','1')
    config={'type':'graph','name':'fixture','description':'Synthetic fixture','schema':{
        'vertex':[],'edge':[],'graph_store_info':{},'graph_structure':{
            'directed':True,'multigraph':True,'weighted':False,'heterogeneous':False}}}
    async def run():
        client=GraphMCPClient()
        try:
            assert await client.connect()
            for name in ['bfs_tree','bfs_edges','bfs_successors','bfs_predecessors','bfs_layers','edge_bfs']:
                assert 'run_'+name in client.available_tools
            params={'vertices':[{'vid':'a'},{'vid':'b'},{'vid':'isolated'}],
                    'edges':[{'src':'a','dst':'b','rank':'tx1','properties':{'base_amt':10}},
                             {'src':'a','dst':'b','rank':'tx2','properties':{'base_amt':20}}],
                    'dataset_config':config}
            result=await client.call_tool('run_initialize_graph',params)
            assert result['success'],result
            result=await client.call_tool('run_edge_bfs',{'source':'a'})
            assert result['success'],result
            assert {edge[2] for edge in result['result']['items']}=={'tx1','tx2'}
            result=await client.call_tool('run_bfs_tree',{'source':'a'})
            assert result['success'],result
            payload=result['result']
            assert payload['graph_type']=='DiGraph' and len(payload['edges'])==1
            vertices,edges=to_graph_data(payload)
            init=await client.call_tool('run_initialize_graph',{'vertices':[v.to_dict() for v in vertices],
                'edges':[e.to_dict() for e in edges],'dataset_config':dataset_for_kind(config,'DiGraph')})
            assert init['success'],init
            result=await client.call_tool('run_pagerank',{'weight':None})
            assert result['success'],result
            assert result['result']==pytest.approx(nx.pagerank(to_networkx(payload),weight=None))
            # Empty graph inputs stay empty through the actual protocol.
            result=await client.call_tool('run_initialize_graph',{'vertices':[],'edges':[],
                'dataset_config':dataset_for_kind(config,'DiGraph')})
            assert result['success'],result
            assert result['result']['vertex_count']==result['result']['edge_count']==0
        finally:
            await client.disconnect()
    asyncio.run(asyncio.wait_for(run(),timeout=45))
