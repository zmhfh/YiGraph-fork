"""Check real AMLSim BFS -> PageRank through ReAct, optionally using the LoRA Binder.

This isolates execution from the web UI, Neo4j and MCP transport. It calls the
same NetworkX tool wrappers and processor used by the MCP server.
"""
import argparse
import asyncio
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def check(args):
    import networkx as nx
    from aag.engine.dependency_resolver import DataDependencyResolver
    from aag.engine.react_executor import ReActExecutor, RunState
    from aag.computing_engine.networkx_server import dynamic_tool_registry as registry
    from aag.computing_engine.networkx_server.algorithm_functions import resolve_function
    from aag.computing_engine.networkx_server.graph_computation_processor import GraphComputationProcessor
    from aag.config.data_upload_config import DatasetConfig
    from aag.expert_search_engine.database.datatype import GraphData, VertexData, EdgeData

    folder = ROOT/'aag/datasets/data/AMLSim1K/process_graph'
    with (folder/'transactions.csv_vertices.csv').open(encoding='utf-8-sig',newline='') as f:
        accounts=list(csv.DictReader(f))
    with (folder/'transactions.csv_edges.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    data=GraphData([VertexData(r['acct_id'],dict(r)) for r in accounts],
                   [EdgeData(r['orig_acct'],r['bene_acct'],str(i),dict(r)) for i,r in enumerate(rows)])
    # Independent direct NetworkX construction; do not use result restoration helpers.
    original=nx.MultiDiGraph()
    original.add_nodes_from((v.vid,v.properties) for v in data.vertices)
    for i,row in enumerate(rows):
        original.add_edge(row['orig_acct'],row['bene_acct'],key=str(i),**row)
    tree=nx.bfs_tree(original,args.source)
    reference=nx.pagerank(tree,weight=None)
    config=DatasetConfig.from_dict({'type':'graph','name':'AMLSim1K','description':'Public synthetic transactions',
        'schema':{'vertex':[],'edge':[],'graph_store_info':{},'graph_structure':{
            'directed':True,'multigraph':True,'weighted':False,'heterogeneous':False}}})

    class LocalTools:
        def __init__(self):
            self.processor=GraphComputationProcessor()
            self.initialized=[]

        async def get_algorithm_description(self,name):
            return name,{'name':name}

        async def run_algorithm(self,name,params,*unused,**context):
            if name=='initialize_graph':
                value=GraphData.from_dict(params)
                flags=DatasetConfig.from_dict(params['dataset_config']).schema.graph_structure
                self.processor.create_graph_from_edges(value.vertices,value.edges,flags.directed,flags.multigraph)
                self.initialized.append({'type':self.processor.graph.__class__.__name__,
                    'nodes':len(self.processor.graph),'edges':self.processor.graph.number_of_edges()})
                return {'success':True,'result':dict(self.initialized[-1])}
            tool=registry._create_tool_function('run_'+name,resolve_function(name),lambda:self.processor,lambda f:f)
            return json.loads(tool(**params))

    engine=LocalTools()
    runner=ReActExecutor(reasoner=None,computing_engine=engine,dependency_resolver=DataDependencyResolver(None),
                         global_graph=data,current_graph_dataset=config)
    state=RunState(goal='先在完整原图上生成 BFS 树，再只在该 BFS 树上计算不加权 PageRank。',step=1)
    action={'type':'algorithm','name':'bfs_tree','objective':'在完整原始图上生成 BFS 树。','parameters':{'source':args.source}}
    result,metadata=await runner._execute_action(action,state)
    if not result.get('success'):
        raise ValueError(result)
    aid=runner._add_artifact(state,action,result,metadata,1)
    state.step=2
    action={'type':'algorithm','name':'pagerank','objective':'只在刚才 bfs_tree 生成的 BFS 树上计算 PageRank，不使用原始交易图。',
            'parameters':{'weight':None},'consumed':[{'artifact_id':aid,'use_as':'graph'}]}
    result,_=await runner._execute_action(action,state)
    actual=result.get('result',{})
    passed=(result.get('success') is True and set(actual)==set(reference) and
            all(math.isclose(actual[n],v,rel_tol=1e-10,abs_tol=1e-12) for n,v in reference.items()) and
            engine.initialized[0]['edges']==len(rows) and engine.initialized[-1]['type']=='DiGraph' and
            engine.initialized[-1]['edges']==tree.number_of_edges())
    return {'passed':passed,'model_used':args.mode=='model','source':args.source,
            'scope':'Local ReAct and NetworkX wrappers; no web UI, database or MCP transport.',
            'input':{'nodes':len(original),'edges':original.number_of_edges()},
            'reference_tree':{'nodes':len(tree),'edges':tree.number_of_edges()},
            'initialized':engine.initialized,'bindings':[e['data'] for e in runner.events if e['kind']=='runtime_binding'],
            'prediction':actual,'reference':reference,'events':runner.events}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['framework','model'],default='framework')
    parser.add_argument('--source',default='1111')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.mode=='framework':
        os.environ.pop('YIGRAPH_BINDER_COMMAND',None);os.environ.pop('YIGRAPH_BINDER_URL',None)
    elif not (os.environ.get('YIGRAPH_BINDER_COMMAND') or os.environ.get('YIGRAPH_BINDER_URL')):
        parser.error('Load your tools/runtime_binder/local.env first; no Binder is configured')
    directory=ROOT/'runtime/binder'
    directory.mkdir(parents=True,exist_ok=True)
    output=args.out or Path(tempfile.mkdtemp(prefix='compat-'+args.mode+'-',dir=directory))/'report.json'
    try:
        report=asyncio.run(check(args))
    except Exception as exc:
        report={'passed':False,'model_used':args.mode=='model','error':repr(exc)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'events','prediction','reference','bindings'}},ensure_ascii=False,indent=2))
    print('Report:',output)
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
