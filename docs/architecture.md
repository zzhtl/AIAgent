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
│   ├── agent-runtime/     # 应用边界共享装配（依赖具体能力 crate）
│   └── agent-telemetry/   # tracing 初始化 + token/费用统计
└── apps/
    ├── agent-cli/         # CLI 入口（REPL + 一次性命令）
    ├── agent-bot/         # stdio JSON 入口
    └── agent-web/         # HTTP / SSE 入口
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
              │     agent-runtime     │
              └───────────┬───────────┘
                          ▼
                 CLI / bot / web apps
```

**关键原则**

- `agent-core` 不依赖任何具体 provider/tool/transport
- 横向能力 crate（llm / tools / skills / memory / config / telemetry）保持 core 向外的依赖方向
- `agent-runtime` 是唯一允许组合全部具体能力 crate 的可复用应用边界
- apps 只处理参数、transport、会话生命周期与渲染，不复制 Agent 装配

## 入口安全与会话并发

- `agent-web` 的 `/health` 始终公开；`AGENT_WEB_TOKEN` 存在时，其余路由使用 Bearer 鉴权。
- 无 token 时只允许 loopback 监听，除非显式设置 `web.allow_unauthenticated=true`。
- web 未配置专属权限时使用只读安全档，不继承 CLI 面向本机使用的全开放默认值。
- web 为每个 session 建立独立异步锁，锁从读取历史持续到 SSE `done` 折叠；同会话串行、异会话并发。省略 session 时由服务端生成 UUID。
- bot 的 stdin 天然严格顺序，保留缺省 `"default"` 会话，不引入不必要的锁。
- 两个入口均可通过 `persist_sessions=true` 复用 SQLite；持久化失败只告警，进程内会话继续运行。

## 架构决策记录

### ADR-008：共享装配位于 agent-runtime

状态：已接受。core 保持无具体实现依赖，runtime 集中连接 provider、工具、MCP、子 agent、记忆与策略。代价是 web/bot 编译依赖增多，收益是三个入口不再发生能力和安全配置漂移。

### ADR-009：Web 采用安全默认值

状态：已接受。匿名访问仅限 loopback，非 loopback 需要 token 或显式不安全开关；工具缺省只读。该决定有意改变旧版 web 的全开放行为。

### ADR-010：CandidateQueue 仅保证进程内写安全

状态：已接受。clone 共享异步锁覆盖 read-modify-write，原子 rename 防止撕裂读；多个进程同时写同一 queue 文件仍不支持，暂不引入平台相关文件锁。

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
