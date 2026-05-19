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

## 环境变量

- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`（按顺序探测）
- `AGENT_BOT_MODEL`（可选，覆盖默认模型）
- `AGENT_CONFIG_DIR`（默认 `~/.config/agent`）：从这里加载 skills / rules / memory

## 状态

- 进程内维护 `history`，所有 stdin 行共享同一对话上下文（关闭进程即清空）
- 工具默认全开（file/shell/grep/glob/fetch + remember/recall + propose）
- 暂不持久化 session 到 SQLite —— 如需跨进程持久化，调用方记录 transcript_delta 自行管理

## 示例

```bash
$ echo '{"input": "ls 当前目录"}' | OPENAI_API_KEY=xxx cargo run -p agent-bot
{"kind":"tool_call_start","call":{...}}
{"kind":"tool_call_result","result":{...}}
{"kind":"text_delta","delta":"..."}
{"kind":"done","reason":"end_turn","transcript_delta":[...]}
```
