use thiserror::Error;

use crate::llm::LlmError;
use crate::tool::ToolError;

/// Top-level error type for the agent runtime. Each module-specific failure
/// surfaces here so callers handle one error type.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("llm: {0}")]
    Llm(#[from] LlmError),

    #[error("tool: {0}")]
    Tool(#[from] ToolError),

    #[error("skill: {0}")]
    Skill(String),

    #[error("memory: {0}")]
    Memory(String),

    #[error("config: {0}")]
    Config(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("agent: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AgentError>;
