//! LLM provider abstraction.
//!
//! Trait-level seam between the runtime and any concrete model client.
//! Implementations live in `agent-llm`; the runtime only sees this trait.

use crate::message::{Message, StopReason, TokenUsage};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Errors surfaced by LLM providers. Providers map their transport-specific
/// errors (HTTP, JSON parsing) into this neutral shape.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("network: {0}")]
    Network(String),

    #[error("auth: {0}")]
    Auth(String),

    #[error("rate limited (retry after {retry_after_secs:?}s)")]
    RateLimited { retry_after_secs: Option<u64> },

    #[error("invalid response: {0}")]
    InvalidResponse(String),

    #[error("provider error ({status}): {message}")]
    Provider { status: u16, message: String },

    #[error("unsupported: {0}")]
    Unsupported(String),
}

pub type LlmResult<T> = std::result::Result<T, LlmError>;

/// Capability flags advertised by a provider so the runtime can degrade
/// gracefully (e.g. skip tool injection for providers without tool support).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub streaming: bool,
    pub tools: bool,
    pub vision: bool,
    /// Anthropic-style extended thinking.
    pub thinking: bool,
}

/// Neutral JSON-schema description of a tool that the model may call.
/// Produced from `Tool::parameters()` and forwarded by the runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    /// JSON Schema (draft-07 subset) for the tool's arguments.
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// When `false`, providers should still return one final event but may
    /// buffer internally. Default `true`.
    #[serde(default = "default_stream")]
    pub stream: bool,
}

fn default_stream() -> bool {
    true
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            stream: true,
        }
    }

    pub fn with_tools(mut self, tools: Vec<ToolSchema>) -> Self {
        self.tools = tools;
        self
    }
}

/// Streaming event from a provider. Providers translate their native
/// streaming protocol (OpenAI delta chunks, Anthropic event types) into
/// this neutral shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LlmEvent {
    /// Incremental assistant text.
    TextDelta { delta: String },

    /// Tool-call argument fragments are accumulating. `index` lets multiple
    /// parallel tool calls be assembled in order.
    ToolCallDelta {
        index: u32,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },

    /// A tool call is fully assembled and ready to dispatch. Emitted once
    /// per call after its `ToolCallDelta` stream completes.
    ToolCallReady {
        index: u32,
        id: String,
        name: String,
        arguments: Value,
    },

    /// Usage accounting for the round-trip. Emitted once before `End`.
    Usage(TokenUsage),

    /// Stream terminator. After this no more events are emitted.
    End(StopReason),
}

pub type LlmEventStream = BoxStream<'static, LlmResult<LlmEvent>>;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;

    fn capabilities(&self) -> ProviderCapabilities;

    async fn chat_stream(&self, request: ChatRequest) -> LlmResult<LlmEventStream>;
}
