# agent-bot

stdio JSON 适配器 —— 任何 IM / 机器人平台都可以 spawn 本进程来获得一个完整的 Agent。

## 设计意图

- 每行 stdin 一个 JSON 请求 → 每事件一行 JSON 输出到 stdout
- 业务逻辑全部委托给 `agent-core::Agent`，本 crate 只负责 transport
- 后期接 IM 平台（企业微信 / 钉钉 / Telegram / Slack ...）只需在外层进程写胶水：消息进来 → 写 stdin；事件出来 → 转发给用户

## Wire 协议

请求行（JSON Lines，stdin）：

```json
{"input": "你好"}
{"input": "查看 README", "session": "user-42"}
```

响应行（JSON Lines，stdout）：每行是一个 `AgentEvent`，一次 run 总以 `done` 结尾：

```json
{"kind":"text_delta","delta":"你"}
{"kind":"text_delta","delta":"好"}
{"kind":"usage_report","usage":{...},"model":"gpt-4o-mini"}
{"kind":"done","reason":"end_turn","transcript_delta":[...]}
```

错误（请求 JSON 不合法等）：

```json
{"kind":"error","message":"invalid request: ..."}
```

## 配置与环境变量

- 使用与 CLI 相同的分层 `config.toml`，包括 `[agent]`、`[permissions]`、
  `[tool_policy]`、`[[subagents]]` 和 `[[mcp_servers]]`。
- 配置的 provider 优先；缺少对应 key 时按 OpenAI → Anthropic → DeepSeek 探测。
- `AGENT_BOT_MODEL`（可选，覆盖默认模型）
- `AGENT_CONFIG_DIR`（默认 `~/.config/agent`）

```toml
[bot]
persist_sessions = true  # 默认 false；开启后使用 <config_dir>/sessions.db
```

## 状态

- 进程内按 `session` 隔离 history；缺省 session 保留为 `"default"`，适合单用户管道。
- stdin 严格顺序处理，因此不存在 web 并发同会话的丢更新问题。
- `persist_sessions=false` 时关闭进程即清空；设为 true 后重启可恢复。
- 工具、权限与策略由共享 `agent-runtime` 装配，与 CLI 保持一致。

## 示例

```bash
$ echo '{"input": "ls 当前目录"}' | OPENAI_API_KEY=xxx cargo run -p agent-bot
{"kind":"tool_call_start","call":{...}}
{"kind":"tool_call_result","result":{...}}
{"kind":"text_delta","delta":"..."}
{"kind":"done","reason":"end_turn","transcript_delta":[...]}
```
