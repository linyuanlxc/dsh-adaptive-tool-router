# dsh-adaptive-tool-router

DeepSeek Harness **host 插件**：在每一步模型请求前，从当前 Agent 可见的工具目录里召回 Top-K，默认只观察、不限制。

当 MCP / 本地工具变多时，Harness 会把全部 schema 塞进上下文。这个插件把「模型该看见哪些工具」变成一步可配置的路由决策，而不是让模型在完整目录里自己翻。

```text
用户请求
   ↓
agent/pre-step
   ↓
读取 agent.ctx.tools.schemas()
   ↓
BM25 召回 Top-K
   ↓
Shadow：写日志，工具列表不变
Restrict：tools.restrict({ allow: alwaysAllow ∪ Top-K })
   ↓
失败则 fail-open，不阻断 Agent
```

离线评测（ToolRet / JSONL）用来对照召回策略，**不是这个仓库的主交付**。主交付是可以 `link` / `--patch` 进 DSH 的插件。

## 当前行为

- 默认 **Shadow Mode**：推荐 Top-K，但不调用 `restrict()`。
- 打开限制后，只暴露 `alwaysAllow`（默认保留 `tool_search`）加上本轮 Top-K。
- 排序失败、写日志失败、`restrict` 不可用时都 **fail-open**，Agent 继续看到原工具集。
- 运行时排序在插件内完成，不依赖 Python、不调用外部 embedding 服务。

## 安装

插件是纯 host 半，没有 Web UI。本地开发用 `link`，临时验证用 `--patch`。

### 1. 构建检查

```bash
cd plugin
npm install
npm test
```

### 2. 挂到 web profile

在 `~/.dsh/profiles/web/package.json` 里加入：

```json
{
  "dependencies": {
    "dsh-adaptive-tool-router": "link:/absolute/path/to/dsh-adaptive-tool-router"
  }
}
```

仓库根目录的 `package.json` 已声明 `dsh.bundle`，`dsh plugin add` 或 profile 安装后应自动挂上 `cordis.patch.yml`。若使用 `github:` 源、补丁没有写进去，再手动补：

```yaml
# ~/.dsh/profiles/web/cordis.patch.yml
- insert:
    - id: adaptive-tool-router
      name: dsh-adaptive-tool-router
      config:
        shadow: true
        topK: 8
        logPath: /tmp/dsh-adaptive-tool-router.jsonl
        alwaysAllow:
          - tool_search
```

```bash
cd ~/.dsh/profiles/web && pnpm install
dsh web
```

### 3. 不改持久 profile 的验证

```bash
dsh --profile headless --patch ./plugin-test.cordis.yml "查一下北京明天天气"
```

通过时应同时看到：

- 日志行：`[dsh-adaptive-tool-router] shadow topK=8 tools=... recommend=...`
- 若配置了 `logPath`，JSONL 里有本轮 query、候选数和推荐列表
- Agent 正常结束；Shadow 下工具调用不受插件拦截

## 配置

| 字段 | 默认 | 含义 |
|---|---|---|
| `shadow` | `true` | `true` 只推荐；`false` 才 `restrict` |
| `topK` | `8` | 每步推荐 / 披露的工具数 |
| `alwaysAllow` | `["tool_search"]` | 限制模式下始终保留的工具 |
| `logPath` | 无 | 决策 JSONL；不设则只打控制台 |
| `verbose` | `true` | 每步打一行决策摘要 |
| `k1` / `b` | `1.5` / `0.75` | 插件内 BM25 参数 |

覆盖示例：

```yaml
- override:
    - id: adaptive-tool-router
      config:
        shadow: false
        topK: 5
        alwaysAllow:
          - tool_search
          - read
          - bash
```

第一版请先用 Shadow 收集自己的调用日志，确认推荐经常覆盖你真正用到的工具，再打开 `shadow: false`。

## 插件结构

```text
package.json              # 仓库根：dsh plugin / github 安装入口
cordis.patch.yml          # 默认挂载与 Shadow 配置
plugin-test.cordis.yml    # --patch 临时验证
plugin/
  src/index.ts            # apply(ctx)：接 agent/pre-step
  src/rank.ts             # BM25 纯函数
  src/query.ts            # 从 messages 抽查询
  src/config.ts           # 配置与保留工具
  tests/                  # 纯函数 + waterfall 契约测试
eval 代码在 src/dsh_tool_router/，只服务离线对照
```

设计约束：

1. 决策逻辑抽纯函数，契约测试确认 `await next()`，排序异常不能吞掉后续 waterfall。
2. 运行时只从 `ctx` / `agent.ctx.tools` 取能力，不 `import '@deepseek-ai/dsh-tools'`，避免 `link` 进 profile 后模块解析失败。
3. 限制模式每次先解除上一轮 `restrict()`，避免多次 intersect 把工具集收成空集。
4. `alwaysAllow` 只保留当前目录里真实存在的名字，不会因为写了 `tool_search` 就伪造一个工具。

## 开发

```bash
cd plugin && npm test
python3 -m pip install -e ".[dev]" && pytest
```

改召回规则先改 `plugin/src/rank.ts` 和契约测试；Python 侧的 BM25 是同一套离线对照，不是另一套线上逻辑。

## 离线评测

评测用来回答「这套召回在公开工具目录上有多差 / 改进了多少」，用来迭代插件，不替代插件是否挂上、是否 fail-open。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
dsh-tool-router-eval \
  --tools examples/tools.jsonl \
  --queries examples/queries.jsonl \
  --k 1 3
```

ToolRet 子任务需要 `pip install -e ".[toolret]"`：

```bash
dsh-tool-router-eval --toolret-task autotools-weather --k 1 5 10
```

Dense / Hybrid 是离线对照，还没有进插件运行时：

```bash
pip install -e ".[dense,toolret]"
dsh-tool-router-eval --toolret-task autotools-weather --retriever dense --k 1 5 10
```

当前 BM25 smoke 在 44,453 个工具上 Recall@10 = 0.0303，说明词法召回不够，也说明插件默认 Shadow 是必要的。结果见 `reports/bm25_autotools_weather.json`。

## 原则

- 先观测再限制：默认 Shadow，限制是显式打开的第二阶段。
- 插件失败不得阻断 Agent。
- 公开数据验证通用召回，个人 Shadow 日志只验证「对你是否有用」。
- 主结论看任务是否还能做对、schema 是否变少，而不是只报压缩比。
