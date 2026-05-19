//! Core message types.
//!
//! Models a conversation as `Vec<Message>` where each `Message` carries one
//! or more `ContentBlock`s. The shape is closer to the Anthropic API
//! (first-class `tool_use` / `tool_result` blocks) because that representation
//! is cleaner; OpenAI-style `tool_calls` are converted at the provider edge.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Conversational role of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    /// Tool-result messages. Always paired with a preceding `ToolUse` block.
    Tool,
}

/// A single message in the conversation transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: Role::System, content: vec![ContentBlock::text(text)] }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self { role: Role::User, content: vec![ContentBlock::text(text)] }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: vec![ContentBlock::text(text)] }
    }

    /// Flatten all `Text` blocks into one string. Useful for rendering and
    /// summarisation; ignores tool-use / tool-result blocks.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// A typed chunk of a message. A single message may carry text, tool calls,
/// and tool results in any order. Wire shape matches the Anthropic content-
/// block convention: `{"type": "text", "text": "..."}` etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse(ToolUse),
    #[serde(rename = "tool_result")]
    ToolResult(ToolResult),
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }
}

/// A tool invocation requested by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: Value,
}

/// The result of executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub output: String,
    #[serde(default)]
    pub is_error: bool,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Cancelled,
    Error,
}

/// Token accounting for a single LLM response.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub cached_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}
