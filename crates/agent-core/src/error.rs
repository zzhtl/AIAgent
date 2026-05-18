use thiserror::Error;

/// Top-level error type for the agent runtime. Each module-specific failure
/// surfaces here so callers handle one error type.
#[derive(Debug, Error)]
pub enum AgentError {
    #[error("llm error: {0}")]
    Llm(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("skill error: {0}")]
    Skill(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
