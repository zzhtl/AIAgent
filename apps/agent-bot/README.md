# agent-bot

机器人 / IM 接入层占位。

## 设计意图

- 未来在此实现各 IM 平台的 `Channel` trait（输入输出适配）
- 业务逻辑全部委托给 `agent-core::Agent`，本 crate 只负责 transport 与渲染
- 一个适配器一个子模块（`src/wechat_work.rs`、`src/dingtalk.rs`、`src/telegram.rs` 等）

## 当前状态

仅占位，`fn main()` 输出说明文字。
