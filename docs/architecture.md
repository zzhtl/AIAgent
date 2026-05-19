# 架构总览

本文记录 AIAgent 仓库的模块边界与扩展点。详细规划见 `/home/qingteng/.claude/plans/delightful-pondering-fox.md`。

## 模块布局

```
AIAgent/
├── crates/
│   ├── agent-core/        # 运行时内核：消息、Agent 循环、Channel trait
│   ├── agent-llm/         # LlmProvider trait + OpenAI / DeepSeek / Anthropic
│   ├── agent-tools/       # Tool trait + 内置工具（文件、shell、fetch）
│   ├── agent-skills/      # Skill / Rule 加载（markdown + YAML frontmatter）
│   ├── agent-memory/      # 四层记忆：会话 / 事实 / 向量 / 摘要
│   ├── agent-evolution/   # 自动进化：反思 / 规则提炼 / Skill 合成（半自动）
│   ├── agent-config/      # 分层 TOML + env 配置
│   └── agent-telemetry/   # tracing 初始化 + token/费用统计
└── apps/
    ├── agent-cli/         # CLI 入口（REPL + 一次性命令）
    └── agent-bot/         # IM/Bot 接入占位
```

## 依赖方向

```
                ┌───────────────────┐
                │    agent-core     │  ← 所有模块的依赖根
                └─────────┬─────────┘
                          │
   ┌──────────────────────┼──────────────────────┐
   │            │         │         │            │
agent-llm  agent-tools  agent-skills  agent-memory  agent-config
                          │
                          ▼
              ┌───────────────────────┐
              │   apps/agent-cli      │  agents/agent-bot
              └───────────────────────┘
```

**关键原则**

- `agent-core` 不依赖任何具体 provider/tool/transport
- 横向能力 crate（llm / tools / skills / memory / config / telemetry）互相独立，仅依赖 core
- apps 层做组装与渲染，不放业务逻辑

## 四大扩展点

| 扩展点 | trait | 位置 | 加新能力 |
|---|---|---|---|
| 接入新模型 | `LlmProvider` | `agent-llm` | 写 `providers/<name>.rs` + registry 注册 |
| 加新工具 | `Tool` | `agent-tools` | 写 `builtin/<name>.rs` 或外部 crate，调 `registry.register()` |
| 加新 Skill | （声明式） | `~/.config/agent/skills/*.md` | 写 markdown 文件，重启即生效 |
| 接入新 IM | `Channel` | `apps/agent-bot` | 实现 trait 即可复用 core |

## 记忆系统（四层）

| 层级 | 形态 | 存放位置 | 何时调用 |
|---|---|---|---|
| 短期 | 会话消息历史 | `sessions.db` | 每轮对话自动追加 |
| 中期 | 事实笔记（markdown + frontmatter） | `memory/facts/*.md` + `MEMORY.md` 索引 | Agent 显式 `remember` 或 reflection 自动写入 |
| 长期 | 向量嵌入（sqlite-vec） | `memory/vectors.db` | 语义检索 + 摘要归档 |
| 摘要 | LLM 压缩的早期消息 | `sessions.db.summaries` | 会话 token 超阈值时触发 |

## 自动进化（半自动）

```
任务结束
  ├─ reflection.rs          → 自动写 facts（无须确认）
  ├─ rule_extractor.rs      → 候选规则（需 `agent evolution apply` 确认）
  └─ skill_synthesizer.rs   → 候选 Skill（需 `agent evolution apply` 确认）
```

默认 `evolution.auto_apply = false`，rule/skill 候选先入审批队列，避免 prompt 污染。

## 配置路径

```
~/.config/agent/
├── config.toml          # 主配置
├── rules/*.md           # 全局规则（无条件注入 system prompt）
├── skills/*.md          # 能力包（按触发条件注入）
├── memory/
│   ├── MEMORY.md        # 事实索引
│   ├── facts/*.md       # 跨会话事实笔记
│   └── vectors.db       # 向量库
├── sessions.db          # SQLite 会话存储
└── evolution/queue.json # 待审批的候选规则/Skill
```

加载顺序（后者覆盖前者）：
1. `/etc/agent/config.toml`
2. `~/.config/agent/config.toml`
3. `./agent.toml`（项目本地）
4. `AGENT_*` 环境变量

API key 走环境变量或系统 keyring，**不进配置文件**。

## 当前进度

- [x] 阶段 0：Workspace 骨架
- [x] 阶段 1：core 类型 + OpenAI provider + 最小 chat
- [x] 阶段 2：工具循环（Tool trait / Registry / Agent loop / file_read / file_edit / bash）
- [x] 阶段 3：Skills / Rules + PromptProvider + Anthropic Claude provider
- [x] 阶段 4：SQLite 会话持久化 + Token/费用统计 + tracing
- [x] 阶段 5：记忆系统（事实 + 向量 + Reflector）+ remember/forget/recall 工具 + memory CLI
- [x] 阶段 6：grep/glob/fetch + propose_rule/skill + evolution CLI + agent-bot stdio 适配

主体架构与 MVP 完成。后续可按需扩展：候选自动提炼、向量检索召回、Skill 工具子集裁剪、配置层 figment / 项目本地 `agent.toml`、机器人具体平台适配、Web 入口等。
