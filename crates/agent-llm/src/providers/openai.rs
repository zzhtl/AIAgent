//! OpenAI / OpenAI-compatible provider.
//!
//! Also serves DeepSeek and other OpenAI-compatible endpoints by changing
//! `base_url`. Tool calling uses OpenAI's `tools` / `tool_calls` shape and is
//! translated to the neutral `agent_core` `ToolUse` / `ToolResult` model on
//! both directions.

use std::collections::HashMap;

use agent_core::llm::{
    ChatRequest, LlmError, LlmEvent, LlmEventStream, LlmProvider, ProviderCapabilities, ToolSchema,
};
use agent_core::{ContentBlock, Message, Role, StopReason, TokenUsage};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{debug, warn};

/// Construction-time configuration. `api_key` is required; everything else
/// has sensible defaults for stock OpenAI.
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub api_key: String,
    /// Defaults to `https://api.openai.com/v1`. Set to e.g.
    /// `https://api.deepseek.com/v1` for DeepSeek.
    pub base_url: String,
    /// Provider identifier returned by `LlmProvider::name()`. Lets the same
    /// implementation serve multiple registry entries (e.g. "openai" and
    /// "deepseek").
    pub provider_name: String,
    pub default_model: Option<String>,
}

impl OpenAiConfig {
    pub fn openai(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".into(),
            provider_name: "openai".into(),
            default_model: None,
        }
    }

    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.deepseek.com/v1".into(),
            provider_name: "deepseek".into(),
            default_model: Some("deepseek-chat".into()),
        }
    }
}

pub struct OpenAiProvider {
    config: OpenAiConfig,
    http: Client,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        let auth_value = format!("Bearer {}", config.api_key);
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&auth_value).map_err(|_| {
                LlmError::Auth("invalid api key: contains non-ascii bytes".into())
            })?,
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
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        &self.config.provider_name
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: true, vision: false, thinking: false }
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
        debug!(provider = %self.config.provider_name, %model, "sending chat request");

        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
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

        // SSE stream of bytes → text lines → typed events.
        let byte_stream = resp.bytes_stream();
        let event_stream = parse_sse_to_events(byte_stream);
        Ok(event_stream)
    }
}

fn build_request_body(model: &str, request: &ChatRequest) -> Value {
    let messages = request
        .messages
        .iter()
        .flat_map(message_to_openai)
        .collect::<Vec<_>>();

    let mut body = json!({
        "model": model,
        "messages": messages,
        "stream": request.stream,
    });

    if let Some(temp) = request.temperature {
        body["temperature"] = json!(temp);
    }
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = json!(max_tokens);
    }
    if !request.tools.is_empty() {
        body["tools"] = json!(request.tools.iter().map(tool_schema_to_openai).collect::<Vec<_>>());
    }
    if request.stream {
        // Ask OpenAI to include the final usage block in the stream.
        body["stream_options"] = json!({ "include_usage": true });
    }
    body
}

fn tool_schema_to_openai(schema: &ToolSchema) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": schema.parameters,
        }
    })
}

/// Convert one neutral `Message` into one or more OpenAI wire messages.
///
/// - `Role::User` / `Role::System` → single message with concatenated text.
/// - `Role::Assistant` carrying `ToolUse` blocks → one message with `tool_calls`.
/// - `Role::Tool` → one OpenAI message per `ToolResult` block (OpenAI requires
///   one `role: "tool"` message per tool result).
fn message_to_openai(msg: &Message) -> Vec<Value> {
    match msg.role {
        Role::System => vec![json!({ "role": "system", "content": msg.text() })],
        Role::User => vec![json!({ "role": "user", "content": msg.text() })],
        Role::Assistant => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text { text } => text_parts.push(text.clone()),
                    ContentBlock::ToolUse(tu) => {
                        tool_calls.push(json!({
                            "id": tu.id,
                            "type": "function",
                            "function": {
                                "name": tu.name,
                                "arguments": tu.input.to_string(),
                            }
                        }));
                    }
                    ContentBlock::ToolResult(_) => {
                        warn!("tool_result block on assistant message — ignoring");
                    }
                }
            }
            let mut out = json!({ "role": "assistant" });
            if !text_parts.is_empty() {
                out["content"] = json!(text_parts.join(""));
            } else {
                // OpenAI requires `content` even if null when only tool_calls present.
                out["content"] = Value::Null;
            }
            if !tool_calls.is_empty() {
                out["tool_calls"] = json!(tool_calls);
            }
            vec![out]
        }
        Role::Tool => msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult(tr) => Some(json!({
                    "role": "tool",
                    "tool_call_id": tr.tool_use_id,
                    "content": tr.output,
                })),
                _ => None,
            })
            .collect(),
    }
}

async fn map_http_error(resp: reqwest::Response) -> LlmError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    match status {
        StatusCode::UNAUTHORIZED => LlmError::Auth(body),
        StatusCode::TOO_MANY_REQUESTS => LlmError::RateLimited { retry_after_secs: None },
        s if s.is_server_error() => LlmError::Provider { status: s.as_u16(), message: body },
        s => LlmError::Provider { status: s.as_u16(), message: body },
    }
}

// ----------------------------------------------------------------------------
// SSE stream parsing
// ----------------------------------------------------------------------------

/// Convert a raw `Stream<Item = Result<Bytes>>` into an `LlmEventStream`.
/// Buffers across chunk boundaries, splits on newlines, strips `data:`
/// prefixes, and dispatches into `OpenAiChunk` decoding.
fn parse_sse_to_events<S>(byte_stream: S) -> LlmEventStream
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
{
    let stream = async_stream::try_stream! {
        let mut byte_stream = Box::pin(byte_stream);
        let mut buffer = crate::sse::SseLineBuffer::new();
        let mut state = StreamState::default();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = chunk.map_err(|e| LlmError::Network(e.to_string()))?;
            buffer.extend(&chunk);

            // SSE events are separated by blank lines, but in practice OpenAI
            // sends one `data:` line per event. Split per-line and process.
            while let Some(line) = buffer.next_line() {
                if line.is_empty() {
                    continue;
                }
                let Some(payload) = line.strip_prefix("data:") else {
                    // OpenAI uses only `data:` lines; ignore comments / others.
                    continue;
                };
                let payload = payload.trim();
                if payload == "[DONE]" {
                    // Stream terminator. Flush any pending tool calls then end.
                    for ready in state.drain_tool_calls() {
                        yield ready;
                    }
                    if let Some(usage) = state.usage.take() {
                        yield LlmEvent::Usage(usage);
                    }
                    let reason = state.stop_reason.unwrap_or(StopReason::EndTurn);
                    yield LlmEvent::End(reason);
                    return;
                }

                let chunk: OpenAiChunk = match serde_json::from_str(payload) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, payload, "failed to parse openai chunk");
                        continue;
                    }
                };

                for event in state.absorb_chunk(chunk) {
                    yield event;
                }
            }
        }

        // Stream ended without [DONE]; emit any accumulated state.
        for ready in state.drain_tool_calls() {
            yield ready;
        }
        if let Some(usage) = state.usage.take() {
            yield LlmEvent::Usage(usage);
        }
        yield LlmEvent::End(state.stop_reason.unwrap_or(StopReason::EndTurn));
    };

    Box::pin(stream)
}

#[derive(Default)]
struct StreamState {
    pending_tool_calls: HashMap<u32, PendingToolCall>,
    usage: Option<TokenUsage>,
    stop_reason: Option<StopReason>,
}

#[derive(Default)]
struct PendingToolCall {
    id: Option<String>,
    name: Option<String>,
    args_buf: String,
}

impl StreamState {
    fn absorb_chunk(&mut self, chunk: OpenAiChunk) -> Vec<LlmEvent> {
        let mut out = Vec::new();

        if let Some(usage) = chunk.usage {
            self.usage = Some(TokenUsage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                cached_tokens: usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|d| d.cached_tokens)
                    .unwrap_or(0),
            });
        }

        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason.as_deref() {
                self.stop_reason = Some(match reason {
                    "stop" => StopReason::EndTurn,
                    "tool_calls" => StopReason::ToolUse,
                    "length" => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                });
                // On finish, flush any complete tool calls.
                out.extend(self.drain_tool_calls());
            }

            if let Some(delta) = choice.delta {
                if let Some(text) = delta.content {
                    if !text.is_empty() {
                        out.push(LlmEvent::TextDelta { delta: text });
                    }
                }
                if let Some(tcs) = delta.tool_calls {
                    for tc in tcs {
                        let entry = self.pending_tool_calls.entry(tc.index).or_default();
                        if let Some(id) = tc.id {
                            entry.id = Some(id);
                        }
                        let mut args_fragment: Option<String> = None;
                        if let Some(func) = tc.function {
                            if let Some(name) = func.name {
                                entry.name = Some(name);
                            }
                            if let Some(args) = func.arguments {
                                if !args.is_empty() {
                                    entry.args_buf.push_str(&args);
                                    args_fragment = Some(args);
                                }
                            }
                        }
                        out.push(LlmEvent::ToolCallDelta {
                            index: tc.index,
                            id: entry.id.clone(),
                            name: entry.name.clone(),
                            arguments_delta: args_fragment,
                        });
                    }
                }
            }
        }

        out
    }

    fn drain_tool_calls(&mut self) -> Vec<LlmEvent> {
        let mut indices: Vec<u32> = self.pending_tool_calls.keys().copied().collect();
        indices.sort();
        let mut out = Vec::with_capacity(indices.len());
        for idx in indices {
            let Some(pending) = self.pending_tool_calls.remove(&idx) else {
                continue;
            };
            let (Some(id), Some(name)) = (pending.id, pending.name) else {
                continue;
            };
            let arguments: Value = if pending.args_buf.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&pending.args_buf).unwrap_or_else(|_| json!(pending.args_buf))
            };
            out.push(LlmEvent::ToolCallReady { index: idx, id, name, arguments });
        }
        out
    }
}

// ---- Wire types ------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAiChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: Option<ChunkDelta>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCall {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<ChunkToolCallFunction>,
}

#[derive(Debug, Deserialize)]
struct ChunkToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkUsage {
    #[serde(default)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    completion_tokens: Option<u32>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

