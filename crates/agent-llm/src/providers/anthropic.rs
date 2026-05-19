//! Anthropic Claude provider.
//!
//! Uses the `/v1/messages` endpoint with native streaming SSE. Differences
//! versus OpenAI handled at the wire boundary:
//!
//! - `system` is a top-level field, not a role inside `messages`.
//! - `tool_result` lives inside a `user` message's content blocks.
//! - Tool argument deltas arrive as `input_json_delta` (JSON fragments)
//!   inside a `content_block_delta` event; we concatenate then `from_str`.
//! - Stream events are typed (`message_start`, `content_block_start`,
//!   `content_block_delta`, `content_block_stop`, `message_delta`,
//!   `message_stop`).

use std::collections::HashMap;

use agent_core::llm::{
    ChatRequest, LlmError, LlmEvent, LlmEventStream, LlmProvider, ProviderCapabilities, ToolSchema,
};
use agent_core::{ContentBlock, Message, Role, StopReason, TokenUsage};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, warn};

const DEFAULT_API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub base_url: String,
    pub provider_name: String,
    pub default_model: Option<String>,
    pub api_version: String,
}

impl AnthropicConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
            provider_name: "anthropic".into(),
            default_model: None,
            api_version: DEFAULT_API_VERSION.into(),
        }
    }
}

pub struct AnthropicProvider {
    config: AnthropicConfig,
    http: Client,
}

impl AnthropicProvider {
    pub fn new(config: AnthropicConfig) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&config.api_key)
                .map_err(|_| LlmError::Auth("invalid api key bytes".into()))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_str(&config.api_version)
                .map_err(|_| LlmError::Auth("invalid api version".into()))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| LlmError::Network(e.to_string()))?;
        Ok(Self { config, http })
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.config.provider_name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: true, vision: true, thinking: true }
    }

    async fn chat_stream(&self, request: ChatRequest) -> Result<LlmEventStream, LlmError> {
        let model = if request.model.is_empty() {
            self.config
                .default_model
                .clone()
                .ok_or_else(|| LlmError::Unsupported("no model specified and no default".into()))?
        } else {
            request.model.clone()
        };

        let body = build_request_body(&model, &request);
        debug!(provider = %self.config.provider_name, %model, "sending anthropic request");

        let url = format!("{}/messages", self.config.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(map_http_error(resp).await);
        }

        Ok(parse_sse_to_events(resp.bytes_stream()))
    }
}

fn build_request_body(model: &str, request: &ChatRequest) -> Value {
    // Anthropic requires system at the top level. We pull any leading
    // system messages out and concatenate them.
    let mut system_parts: Vec<String> = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for msg in &request.messages {
        match msg.role {
            Role::System => {
                let text = msg.text();
                if !text.is_empty() {
                    system_parts.push(text);
                }
            }
            Role::User => messages.push(json!({ "role": "user", "content": user_content(msg) })),
            Role::Assistant => {
                messages.push(json!({ "role": "assistant", "content": assistant_content(msg) }))
            }
            Role::Tool => {
                // Anthropic represents tool results as a user message with
                // tool_result content blocks.
                messages.push(json!({ "role": "user", "content": tool_result_content(msg) }));
            }
        }
    }

    let mut body = json!({
        "model": model,
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
        "stream": request.stream,
    });
    if !system_parts.is_empty() {
        body["system"] = json!(system_parts.join("\n\n"));
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = json!(temp);
    }
    if !request.tools.is_empty() {
        body["tools"] = json!(request
            .tools
            .iter()
            .map(tool_schema_to_anthropic)
            .collect::<Vec<_>>());
    }
    body
}

fn tool_schema_to_anthropic(schema: &ToolSchema) -> Value {
    json!({
        "name": schema.name,
        "description": schema.description,
        "input_schema": schema.parameters,
    })
}

fn user_content(msg: &Message) -> Value {
    // Plain text for user messages.
    let text = msg.text();
    if text.is_empty() {
        json!([])
    } else {
        json!([{ "type": "text", "text": text }])
    }
}

fn assistant_content(msg: &Message) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    for block in &msg.content {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                blocks.push(json!({ "type": "text", "text": text }))
            }
            ContentBlock::Text { .. } => {}
            ContentBlock::ToolUse(tu) => blocks.push(json!({
                "type": "tool_use",
                "id": tu.id,
                "name": tu.name,
                "input": tu.input,
            })),
            ContentBlock::ToolResult(_) => {
                warn!("tool_result on assistant message — skipping");
            }
        }
    }
    json!(blocks)
}

fn tool_result_content(msg: &Message) -> Value {
    let mut blocks: Vec<Value> = Vec::new();
    for block in &msg.content {
        if let ContentBlock::ToolResult(tr) = block {
            let mut item = json!({
                "type": "tool_result",
                "tool_use_id": tr.tool_use_id,
                "content": tr.output,
            });
            if tr.is_error {
                item["is_error"] = json!(true);
            }
            blocks.push(item);
        }
    }
    json!(blocks)
}

async fn map_http_error(resp: reqwest::Response) -> LlmError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    match status {
        StatusCode::UNAUTHORIZED => LlmError::Auth(body),
        StatusCode::TOO_MANY_REQUESTS => LlmError::RateLimited { retry_after_secs: None },
        s => LlmError::Provider { status: s.as_u16(), message: body },
    }
}

// ----------------------------------------------------------------------------
// SSE parsing
// ----------------------------------------------------------------------------

fn parse_sse_to_events<S>(byte_stream: S) -> LlmEventStream
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = async_stream::try_stream! {
        let mut byte_stream = Box::pin(byte_stream);
        let mut buffer = String::new();
        let mut state = StreamState::default();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::Network(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // SSE in Anthropic uses `event: <name>\ndata: <json>\n\n`. We
            // ignore the `event:` line (the JSON's own `type` field carries
            // the event kind) and parse each `data:` payload as JSON.
            while let Some(idx) = buffer.find('\n') {
                let line = buffer[..idx].trim_end_matches('\r').to_string();
                buffer.drain(..=idx);
                if line.is_empty() {
                    continue;
                }
                if line.starts_with("event:") {
                    continue;
                }
                let Some(payload) = line.strip_prefix("data:") else {
                    continue;
                };
                let payload = payload.trim();
                if payload.is_empty() {
                    continue;
                }

                let parsed: AnthropicEvent = match serde_json::from_str(payload) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, payload, "failed to parse anthropic event");
                        continue;
                    }
                };

                for ev in state.absorb(parsed) {
                    yield ev;
                }
            }
        }

        // Flush at end-of-stream (in case the server cut off before message_stop).
        for ev in state.finish() {
            yield ev;
        }
    };
    Box::pin(stream)
}

#[derive(Default)]
struct StreamState {
    pending_tool_calls: HashMap<u32, PendingToolCall>,
    usage: Option<TokenUsage>,
    stop_reason: Option<StopReason>,
    done: bool,
}

#[derive(Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    args_buf: String,
}

impl StreamState {
    fn absorb(&mut self, event: AnthropicEvent) -> Vec<LlmEvent> {
        let mut out = Vec::new();
        match event {
            AnthropicEvent::MessageStart { message } => {
                if let Some(u) = message.usage {
                    self.usage = Some(TokenUsage {
                        prompt_tokens: u.input_tokens.unwrap_or(0),
                        completion_tokens: u.output_tokens.unwrap_or(0),
                        cached_tokens: u.cache_read_input_tokens.unwrap_or(0),
                    });
                }
            }
            AnthropicEvent::ContentBlockStart { index, content_block } => match content_block {
                ContentBlockStart::Text { .. } => {}
                ContentBlockStart::ToolUse { id, name, .. } => {
                    let entry = self.pending_tool_calls.entry(index).or_default();
                    entry.id = Some(id);
                    entry.name = Some(name);
                }
            },
            AnthropicEvent::ContentBlockDelta { index, delta } => match delta {
                ContentBlockDelta::TextDelta { text } => {
                    if !text.is_empty() {
                        out.push(LlmEvent::TextDelta { delta: text });
                    }
                }
                ContentBlockDelta::InputJsonDelta { partial_json } => {
                    let entry = self.pending_tool_calls.entry(index).or_default();
                    entry.args_buf.push_str(&partial_json);
                    out.push(LlmEvent::ToolCallDelta {
                        index,
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                        arguments_delta: Some(entry.args_buf.clone()),
                    });
                }
            },
            AnthropicEvent::ContentBlockStop { index } => {
                if let Some(pending) = self.pending_tool_calls.remove(&index) {
                    let (Some(id), Some(name)) = (pending.id, pending.name) else {
                        return out;
                    };
                    let arguments: Value = if pending.args_buf.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&pending.args_buf).unwrap_or_else(|_| json!({}))
                    };
                    out.push(LlmEvent::ToolCallReady { index, id, name, arguments });
                }
            }
            AnthropicEvent::MessageDelta { delta, usage } => {
                if let Some(reason) = delta.stop_reason {
                    self.stop_reason = Some(map_stop_reason(&reason));
                }
                if let Some(u) = usage {
                    let cur = self.usage.unwrap_or_default();
                    self.usage = Some(TokenUsage {
                        prompt_tokens: u.input_tokens.unwrap_or(cur.prompt_tokens),
                        completion_tokens: u.output_tokens.unwrap_or(cur.completion_tokens),
                        cached_tokens: u.cache_read_input_tokens.unwrap_or(cur.cached_tokens),
                    });
                }
            }
            AnthropicEvent::MessageStop => {
                out.extend(self.finish());
            }
            AnthropicEvent::Ping | AnthropicEvent::Other => {}
        }
        out
    }

    fn finish(&mut self) -> Vec<LlmEvent> {
        if self.done {
            return Vec::new();
        }
        self.done = true;
        let mut out = Vec::new();
        // Drain any tool calls that never got a content_block_stop.
        let mut indices: Vec<u32> = self.pending_tool_calls.keys().copied().collect();
        indices.sort();
        for idx in indices {
            if let Some(pending) = self.pending_tool_calls.remove(&idx) {
                let (Some(id), Some(name)) = (pending.id, pending.name) else {
                    continue;
                };
                let arguments: Value = if pending.args_buf.trim().is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&pending.args_buf).unwrap_or_else(|_| json!({}))
                };
                out.push(LlmEvent::ToolCallReady { index: idx, id, name, arguments });
            }
        }
        if let Some(u) = self.usage.take() {
            out.push(LlmEvent::Usage(u));
        }
        out.push(LlmEvent::End(self.stop_reason.unwrap_or(StopReason::EndTurn)));
        out
    }
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

// ---- Wire types ------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicEvent {
    MessageStart {
        message: MessageStartInner,
    },
    ContentBlockStart {
        index: u32,
        content_block: ContentBlockStart,
    },
    ContentBlockDelta {
        index: u32,
        delta: ContentBlockDelta,
    },
    ContentBlockStop {
        index: u32,
    },
    MessageDelta {
        delta: MessageDeltaInner,
        #[serde(default)]
        usage: Option<UsageWire>,
    },
    MessageStop,
    Ping,
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct MessageStartInner {
    #[serde(default)]
    usage: Option<UsageWire>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockStart {
    Text {
        #[serde(default)]
        #[allow(dead_code)]
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        #[allow(dead_code)]
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaInner {
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UsageWire {
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(default)]
    output_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}
