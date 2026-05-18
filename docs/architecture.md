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
│   ├── agent-memory/      # SessionStore trait + SQLite 实现
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

## 配置路径

```
~/.config/agent/
├── config.toml          # 主配置
├── rules/*.md           # 全局规则（无条件注入 system prompt）
├── skills/*.md          # 能力包（按触发条件注入）
└── sessions.db          # SQLite 会话存储
```

加载顺序（后者覆盖前者）：
1. `/etc/agent/config.toml`
2. `~/.config/agent/config.toml`
3. `./agent.toml`（项目本地）
4. `AGENT_*` 环境变量

API key 走环境变量或系统 keyring，**不进配置文件**。

## 当前进度

- [x] 阶段 0：Workspace 骨架
- [ ] 阶段 1：core 类型 + OpenAI provider + 最小 chat
- [ ] 阶段 2：工具循环
- [ ] 阶段 3：Skills / Rules + Anthropic
- [ ] 阶段 4：持久化 + 可观测性
- [ ] 阶段 5：机器人占位 + 工具补完
