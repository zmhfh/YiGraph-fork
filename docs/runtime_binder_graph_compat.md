# AMLSim 与 BFS 输入适配修复

这份代码基于 YiGraph-fork main 的 `563a78ed6f89045b9a08bce82e2ab60c925ff63a`，包含 Binder 接入和本轮图兼容修复。当前 LoRA 权重保持原样。框架测试通过不代表该 LoRA 在新图类型上已经通过测试。

已有 LoRA、复现材料与校验值见 [runtime-binder-v1.0.0 发布页](https://github.com/zmhfh/YiGraph-fork/releases/tag/runtime-binder-v1.0.0)。本次修复没有重新训练或替换这些权重。

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

## 网页部署与实际验收

前端需要 Node.js >=20.9；已在 node1 用 Node.js 22 完成静态页面构建。后端建议使用独立 Python 3.12 环境：项目的 engine 依赖固定 transformers==4.53.1，Binder 的 Qwen3.5 环境使用其原有版本。

```bash
# 在独立网页环境中安装；不要在训练环境中执行。
python -m pip install -e '.[engine]' \
  'Flask==3.0.0' 'Flask-CORS==4.0.0' 'Flask-SocketIO==5.3.6'

npm ci --prefix web/frontend/next-app
npm run build --prefix web/frontend/next-app
```

主对话模型负责生成工具调用和报告，Binder LoRA 负责输入引用选择。LLaMA-Factory 可单独用基础 Qwen 提供主对话 API；不要把只返回 input_refs 的 Binder 服务当成聊天接口。主模型服务须先通过 `/v1/models` 和 `/v1/chat/completions` 检查，再填写实际接口地址与模型名。代码包中的远程 API 配置是示例，不包含可用凭据。

LlamaIndex 的嵌入模型缓存由 LLAMA_INDEX_CACHE_DIR 控制；仅设置 HF_HOME 不会覆盖它显式传入的缓存位置。在网页后端启动终端设置自己可写的路径：

```bash
export LLAMA_INDEX_CACHE_DIR=/home/hdd/fuzq/yigraph-web-cache/llama-index
export HF_HOME=/home/hdd/fuzq/yigraph-web-cache/huggingface
mkdir -p "$LLAMA_INDEX_CACHE_DIR" "$HF_HOME"
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
```

原图查询和 GraphRAG 还需要可用的 Neo4j 账号。数据库密码可通过 YIGRAPH_NEO4J_PASSWORD 或配置中的 password_file 在节点本地提供，不要提交到 Git。NLTK 的词库目录须符合已安装 NLTK 的私有路径检查；共享目录权限与代理下载策略需要分别处理，不要用关闭安全检查的方法消除警告。

后端从 5089 起选择可用端口；静态 out/index.html 存在时，前端和 API 由同一后端端口提供。请转发新进程实际打印的端口，避免访问旧实例。

在网页选择 AMLSim1K，提交以下问题：

```text
WEB-BFS-1111
在完整原始交易图上，以账户 ID 1111 为源执行 bfs_tree，沿原图方向遍历，不限制深度。
将得到的 BFS 树传给 pagerank，alpha=0.85，weight=null。
汇总实际原图和树的节点、边数量，以及树上 PageRank 排名前五的节点和得分。
```

执行结束后运行：

```bash
python tools/runtime_binder/verify_web_run.py --repo "$PWD"
```

检查器读取对应网页运行的完整产物与 Binder 事件，从 BFS 实际输入快照计算参考树和全部 PageRank 得分，并检查下游工作图类型与规模。它不把网页文本或严格 JSON 字符串匹配当成数值计算正确性的证明；报告文字还须核对其给出的参考事实。

截至 2026-11-22，已收到 node1 的框架检查 passed=true（原图 1446 节点、17512 条边；树 778 节点、777 条边）及主模型列表接口成功记录。完整网页流程仍受嵌入缓存权限、Neo4j 凭据和 NLTK 配置影响，尚未收到网页检查器 passed=true 的结果；不能将此发布解释为真实网页验收已全部通过。
