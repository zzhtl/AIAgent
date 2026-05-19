//! Input/output adapter shared by CLI and bot front-ends.

use crate::message::{Message, StopReason, TokenUsage, ToolResult, ToolUse};
use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// User-side input into the agent. Currently text-only; extensible to
/// attachments / images later via an `attachments` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInput {
    pub text: String,
}

impl UserInput {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Streaming event emitted by `Agent::run`. The front-end (CLI / bot)
/// consumes this stream and renders accordingly.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Incremental assistant text. Front-ends should append to the current
    /// turn's buffer.
    TextDelta { delta: String },

    /// The model has decided to invoke a tool. Arguments may still be
    /// streaming when this is first emitted; rendered as a header.
    ToolCallStart { call: ToolUse },

    /// The tool finished executing.
    ToolCallResult { result: ToolResult },

    /// Token usage for the just-finished LLM round-trip.
    UsageReport { usage: TokenUsage, model: String },

    /// Terminal event. `transcript_delta` carries every message the run
    /// appended (the user turn, intermediate assistant + tool messages, and
    /// the final assistant message). Front-ends merge it into their history.
    Done {
        reason: StopReason,
        transcript_delta: Vec<Message>,
    },

    /// Recoverable warning surfaced to the user (does not end the stream).
    Warning { message: String },
}

/// Bidirectional adapter between a transport (terminal, IM, HTTP) and the
/// agent runtime. CLI and bot implementations both implement this trait.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Pull the next user input. `None` signals the channel has closed and
    /// the agent should exit gracefully.
    async fn recv(&mut self) -> Option<UserInput>;

    /// Push an event to the user side. Must not block the run loop on slow
    /// transports — implementations should buffer internally if needed.
    async fn send(&mut self, event: AgentEvent) -> Result<()>;
}
