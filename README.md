# AIAgent

一个用 Rust 编写的自进化 AI Agent 运行时。多 crate 工作区，干净的 trait 边界，内置工具循环、四层记忆、技能/规则注入与半自动进化。支持 OpenAI / DeepSeek / Anthropic Claude。

## 特性

- **Agent 主循环**：`think → tool_call → observe`，流式输出，多轮工具调用，`max_steps` 上限。
- **内置工具**：`file_read` / `file_edit` / `bash` / `grep` / `glob` / `fetch`，以及记忆工具 `remember` / `forget` / `recall` 和进化工具 `propose_rule` / `propose_skill`。
- **工具并行**：一轮内的多个工具调用并发执行、按原顺序产出结果。
- **可靠性**：LLM 瞬时错误（网络 / 限流 / 5xx）自动指数退避重试（尊重 `Retry-After`）；单轮 `token_budget` 预算上限；REPL 内 Ctrl-C 优雅取消（已产生的消息照常持久化）。
- **四层记忆**：会话历史（SQLite）/ 事实笔记（markdown）/ 向量召回（语义检索）/ 早期消息摘要压缩。
- **技能与规则**：markdown 声明式能力包，按触发词注入 system prompt；技能可用 `tools_allowed` **在运行时限定**可调用的工具子集。
- **半自动进化**：任务后自动反思写入记忆；规则 / 技能候选进审批队列，`agent evolution apply` 确认后落地。
- **成本统计**：按模型估算 token 费用，缓存命中（prompt cache）按缓存价计费。
- **分层配置**：默认值 → `/etc/agent` → `~/.config/agent` → `./agent.toml` → `AGENT_*` 环境变量。

## 快速开始

```bash
# 1. 构建
cargo build --release

# 2. 初始化配置目录（生成 config.toml + 起步 skill/rule 模板）
cargo run -p agent-cli -- init

# 3. 设置 API key（按所选 provider，三选一）
export OPENAI_API_KEY=...       # openai（默认）
export DEEPSEEK_API_KEY=...     # deepseek
export ANTHROPIC_API_KEY=...    # claude / anthropic

# 4. 开聊
cargo run -p agent-cli -- chat
```

可选：`cargo install --path apps/agent-cli` 后直接使用 `agent` 命令（下文示例均以 `agent` 代指该二进制）。

## CLI 命令

```text
agent init [--force]                初始化配置目录（已存在的文件默认跳过，--force 覆盖）
agent chat                          启动交互式 REPL（Ctrl-D 或 /quit 退出，运行中 Ctrl-C 取消当前轮）
agent run <prompt>                  执行单条 prompt 后退出
agent sessions [--limit N]          列出历史会话
agent resume <session_id>           恢复某个会话继续对话
agent skills                        列出已加载的 skills 与 rules
agent config                        打印解析后的分层配置
agent memory <子命令>               长期记忆管理（见下）
agent evolution <子命令>            进化候选审批（见下）
```

记忆子命令：

```text
agent memory list [--kind K]              列出事实（K: preference|project|reflection|note）
agent memory search <query> [--limit N]   子串搜索
agent memory show <id>                     查看单条事实
agent memory add <name> <body> [--kind K]  直接新增事实
agent memory forget <id>                   删除事实
agent memory index                         向量化索引所有事实（语义召回前置，需 OPENAI_API_KEY）
```

进化子命令：

```text
agent evolution review            查看待审批的规则 / 技能候选
agent evolution apply <id>        接受候选并写入 rules/ 或 skills/
agent evolution reject <id>       丢弃候选
```

全局参数（可叠加在任意子命令上）：

```text
--provider <openai|deepseek|claude>   覆盖配置中的 provider
--model <model-id>                    指定模型（留空用 provider 默认）
--no-tools                            纯文本模式，禁用全部工具
--config-dir <path>                   覆盖配置目录（默认 ~/.config/agent）
--evolve                              每轮后自动反思一条记忆（多一次 LLM 调用）
```

## Provider 与模型

| provider | 环境变量 | 默认模型 |
|---|---|---|
| `openai`（默认） | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| `claude` / `anthropic` | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |

API key **只从环境变量读取，绝不写进配置文件**。

## 配置

`agent init` 会生成一份带注释的 `~/.config/agent/config.toml`。关键项：

```toml
provider = "openai"        # openai | deepseek | claude
# model = "gpt-4o-mini"    # 留空用 provider 默认
no_tools = false
evolve = false

[agent]
max_steps = 12             # 单轮 think→tool 循环步数上限
# max_tokens = 4096        # 每次 LLM 请求的最大输出 token
summary_threshold = 30     # 历史超过该条数时压缩早期消息；0 关闭
summary_keep_tail = 8
vector_recall = false      # 语义召回（需 OPENAI_API_KEY + 先跑 `agent memory index`）
vector_recall_top_k = 5
vector_recall_min_score = 0.2
max_retries = 2            # LLM 瞬时错误自动重试次数
retry_base_delay_ms = 500  # 重试退避基数，按 base*2^n 增长
# token_budget = 100000    # 单轮累计 token 预算，超出即停止

[permissions]
allow_read = true
allow_write = true
allow_shell = true
allow_network = true
max_runtime_secs = 120     # bash 等长任务的硬超时（秒）
```

加载顺序（后者覆盖前者）：内置默认 → `/etc/agent/config.toml` → `~/.config/agent/config.toml` → `./agent.toml` → `AGENT_*` 环境变量（双下划线分隔层级，如 `AGENT_AGENT__MAX_STEPS=20`）。

完整配置目录布局见 [`docs/architecture.md`](docs/architecture.md)。

## 技能与规则

放在 `~/.config/agent/{skills,rules}/` 下的 markdown 文件，重启即生效。

- **规则（rules/）**：无条件注入 system prompt 的全局约束，只有 `name` + 正文。
- **技能（skills/）**：按触发词命中后才注入的能力包。可声明 `tools_allowed` 限定该技能激活时允许调用的工具——**运行时会强制执行**，未授权的工具调用被拒（记忆 / 进化等基础设施工具始终放行）。

技能示例（`skills/code-review.md`）：

```markdown
---
name: code-review
description: 系统化代码审查流程
triggers:
  - review
  - 代码审查
tools_allowed:
  - file_read
  - bash
---

# Code Review 工作流
当用户要求审查代码时……
```

## 记忆与进化

四层记忆与半自动进化（反思自动写入、规则/技能候选需审批）的详细说明见 [`docs/architecture.md`](docs/architecture.md)。

语义召回默认关闭；开启需在 `config.toml` 设 `vector_recall = true`、设置 `OPENAI_API_KEY`，并先运行 `agent memory index` 把事实向量化。

## 项目结构

```
crates/
  agent-core/        运行时内核：消息、Agent 循环、LLM/Tool/Channel trait
  agent-llm/         LlmProvider 实现：OpenAI / DeepSeek / Anthropic
  agent-tools/       Tool trait + 内置工具
  agent-skills/      Skill / Rule 加载与 prompt 增强
  agent-memory/      四层记忆：会话 / 事实 / 向量 / 摘要
  agent-evolution/   反思 / 总结
  agent-config/      分层 TOML + env 配置
  agent-telemetry/   tracing 初始化 + token/费用统计
apps/
  agent-cli/         CLI 入口（REPL + 一次性命令）
  agent-bot/         stdio JSON 适配器（接 IM/Bot 平台，见 apps/agent-bot/README.md）
```

模块边界、依赖方向与四大扩展点详见 [`docs/architecture.md`](docs/architecture.md)。

## 开发

```bash
cargo build --workspace     # 构建
cargo test --workspace      # 测试
cargo clippy --all-targets  # 静态检查
```

CI（`.github/workflows/ci.yml`）在 push / PR 到 `main` 时跑 clippy + test。

> 注：仓库使用手工调校的代码风格且未提供 `rustfmt.toml`，默认 `cargo fmt` 会与既有代码冲突，因此 **CI 不门禁 fmt**，也请勿用 `cargo fmt` 重排既有代码。

## 许可证

MIT OR Apache-2.0
