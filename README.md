# dsh-adaptive-tool-router

面向 DeepSeek Harness 大规模工具目录的自适应检索与排序插件。项目将算法核心与
DSH 运行时适配分离：先在公开 ToolRet/MCPToolBench++ 数据上验证工具召回，再通过
Shadow Mode 学习个人与项目偏好，最后按置信度动态披露 Top-K 工具。

## 当前阶段

已实现可复现的 BM25、Dense（E5）和 BM25+Dense RRF 混合召回，以及标准检索
指标。后续对照 Cross-Encoder 重排、公开工具检索插件和 Shadow Mode 个性化。

## 评测路线

1. **ToolRet**：主检索基准，约 7.6K 查询、43K 工具；报告 Recall@K、MRR、nDCG。
2. **MCPToolBench++**：MCP 泛化与相似工具困难负例。
3. **BFCL**：Top-K 工具约束下的函数名、参数 AST 与拒识评测。
4. **DSH Shadow Mode**：推荐但不限制工具，按时间切分评估个人化增量。

主结论采用任务质量与工具 schema 成本的 Pareto，而不只报告压缩比例。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

对本地 JSONL 运行 BM25 基线：

```bash
dsh-tool-router-eval \
  --tools examples/tools.jsonl \
  --queries examples/queries.jsonl \
  --k 1 3
```

直接运行 ToolRet 子任务（需要 `pip install -e ".[toolret]"`）：

```bash
dsh-tool-router-eval --toolret-task autotools-weather --k 1 5 10
```

Dense / Hybrid 需要额外安装 embedding 依赖：

```bash
pip install -e ".[dense,toolret]"
dsh-tool-router-eval --toolret-task autotools-weather --retriever dense --k 1 5 10
dsh-tool-router-eval --toolret-task autotools-weather --retriever hybrid --k 1 5 10
```

默认 Dense 模型是 `intfloat/e5-small-v2`，工具向量会缓存到
`.cache/dsh-tool-router/`，重复评测不会重新编码整个工具库。

当前 BM25 smoke run 在完整 44,453 工具库上的 Recall@10 为 0.0303，说明普通词法
匹配远不足以解决开放工具检索；结果见
`reports/bm25_autotools_weather.json`。该数字仅用于验证数据和评测链路，不作为最终结论。

工具文件格式：

```json
{"id":"weather","name":"weather_forecast","description":"Get a weather forecast","parameters":{"city":"string"}}
```

查询文件格式：

```json
{"id":"q1","query":"Will it rain in Beijing tomorrow?","labels":["weather"]}
```

## 目录规划

```text
src/dsh_tool_router/  算法核心、数据适配与离线评测
plugin/               DeepSeek Harness TypeScript 适配（Shadow Mode 优先）
tests/                单元测试
reports/              基线、消融与失败案例
```

## 原则

- 先观测再限制：第一版只运行 Shadow Mode。
- 公开数据验证通用能力，个人日志只验证个性化增量。
- 训练/测试按工具、Server、来源隔离，避免 schema 与查询近重复泄漏。
- Router 失败时 fail-open，始终保留核心工具与 `tool_search`。
