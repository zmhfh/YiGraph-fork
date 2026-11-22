# AMLSim 与 BFS 输入适配修复

这份代码基于 YiGraph-fork main 的 `563a78ed6f89045b9a08bce82e2ab60c925ff63a`，包含 Binder 接入和本轮图兼容修复。当前 LoRA 权重保持原样。框架测试通过不代表该 LoRA 在新图类型上已经通过测试。

## 修改内容

1. 从已加载数据集获取 Graph、DiGraph、MultiGraph 或 MultiDiGraph 类型；环境变量仅作缺少元数据时的备选。
2. 多重边保留交易 key、金额、时间及其他属性。缺少 key 的源数据按本次输入顺序分配确定性 key，避免所有交易覆盖到 key=0。
3. 图序列化、比较图参数及后处理恢复共同使用完整图结构；保留节点属性、边属性、孤立节点及图属性。
4. BFS 树保持其实际 DiGraph 类型，直接作为下游输入；不再通过原图端点重建树边，反向 BFS 的树边也无需存在于原图中。
5. BFS 边集、父子映射和分层结果有独立的类型与完整性标记；被截断的预览不能作为完整工作图。
6. 新增 bfs_edges、bfs_successors、bfs_predecessors、bfs_layers、edge_bfs 的注册、知识库条目和输出说明。
7. 空子图保持为空，避免回退到原图；空图连通性标记为未定义，不再导致初始化失败。
8. 实际执行图的类型覆盖本次初始化参数，不修改源数据集配置；BFS 记录保存其实际输入图，后续诱导子图从这一输入恢复。

## 两种 BFS 消费方式

完整 `bfs_tree` 输出本身已经是一张图，可以直接消费：

```json
{"artifact_id":"a1","use_as":"graph"}
```

如果后处理把原始结果放进了 `original_result`，增加 `"field":"original_result"`。

`bfs_edges` 等输出需要明确转换方式：

```json
{"artifact_id":"a1","use_as":"graph","graph_mode":"bfs_tree"}
```

或者：

```json
{"artifact_id":"a1","use_as":"graph","graph_mode":"induced_subgraph"}
```

树模式只取遍历树的边，不把交易金额或时间凭空附到树边上；诱导子图模式从该遍历实际输入图保留访问节点之间的全部边及属性，包括多笔交易。完整 BFS 树也可以显式选择诱导子图模式。

`bfs_layers` 没有父节点信息，不能仅凭层次恢复唯一 BFS 树。`edge_bfs` 包含非树边，也不能当成 BFS 树。

原来的无类型边引用 `{nodes, edges:[[u,v]]}` 在多重图中可能对应多笔交易。现在应提供具体 `[u,v,key]`，或在确实需要全部交易时明确设置 `"edge_selection":"all_parallel"`。节点诱导转换可使用转换环境中的 `induced_graph(nodes)`，保留完整交易数据。

## 在 node1 的独立目录验证

先把代码包放到 `/home/hdd/fuzq/`，解压后目录为 `YiGraph-binder-compat-20261122`。这不会修改原来的 `/home/hdd/fuzq/YiGraph-fork`。模型和用户本地配置未复制到公开代码包。

```bash
cd /home/hdd/fuzq
unzip -n yigraph_amlsim_bfs_fix_20261122.zip
cd YiGraph-binder-compat-20261122
source /home/fuzq/finetune312/bin/activate
python tools/runtime_binder/check_workflow.py --mode framework
```

此检查使用仓库中的 AMLSim 账户表和交易表，以账户 `1111` 为源，运行“原始交易图 → BFS 树 → PageRank”，与同图直接 NetworkX 计算核对。它不调用 LoRA，也不调用网页、数据库或 MCP 传输。预期输入为 1446 个节点、17512 条边，树为 778 个节点、777 条边，`passed: true`、`model_used: false`。

然后调用你已有的真实模型：

```bash
source /home/hdd/fuzq/YiGraph-fork/tools/runtime_binder/local.env
python tools/runtime_binder/check_workflow.py --mode model
```

这里必须看到 `model_used: true` 和 `passed: true`。若 LoRA 选错输入或推理服务不可用，会返回失败，不会自动使用参考选择替代模型。`report.json` 中包含实际绑定、工作图类型、预测值和参考值。已有 local.env 的 worker 路径可以继续使用，协议字段不变；也可通过本目录的 configure.py 配置新的 worker。不要同时重复启动多个模型进程。

图与执行路径回归检查：

```bash
python -m pytest tests/test_graph_payload.py tests/test_binder_graph_compat.py tests/test_binder_mcp_compat.py tests/test_algorithm_results.py -q
```

`test_binder_mcp_compat.py` 启动真正的 stdio MCP 子进程，检查工具注册、交易 key 传输、BFS 树到 PageRank、空图初始化；它不使用 LoRA。需要当前项目的基础依赖和 pytest。模型进程仍使用自己的 LLaMA-Factory 环境；不要为此在训练环境中整体降级 transformers。

通过以上检查后，再把原项目的本地服务配置迁移到新目录并从新目录启动项目，进行网页请求验证。SSH 连接未成功前，不能视为已经部署到了 node1。旧目录和本地模型保持可用于回退。

## 验证边界

本轮本地验证包含原简单图回归、四类图结构、真实 AMLSim 数据、实际 ReAct 执行方法、实际 MCP 传输及后处理。Binder 相关自动测试使用可控的推理替身，不能作为真实模型正确率。

完整测试集在当前 Windows 环境中仍受无关的发布依赖 keyring、NLTK 资源和 POSIX 文件权限断言影响。相关执行模块的专项回归另行运行；完整 pytest 不能被宣称为全部通过。

保留多重交易不意味着所有 NetworkX 算法都支持 MultiDiGraph。算法不支持时仍会返回错误。若业务要求投影成简单图，必须另行明确方向处理、平行交易聚合方式、金额权重和时间筛选规则，不应为了让算法运行而隐式合并交易。真实 LoRA 的新类型准确率及完整网页流程需要在服务器上验收。
