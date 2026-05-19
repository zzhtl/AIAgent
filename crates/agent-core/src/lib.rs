//! agent-core
//!
//! Runtime kernel for the AI agent. Holds the message model, the LLM and
//! Tool trait abstractions, the `Agent` execution loop, and the `Channel`
//! transport abstraction.
//!
//! Concrete provider, tool, and transport implementations live in sibling
//! crates and are wired in at the application boundary.

pub mod agent;
pub mod channel;
pub mod error;
pub mod evolution;
pub mod frontmatter;
pub mod llm;
pub mod memory;
pub mod message;
pub mod prompt;
pub mod session;
pub mod store;
pub mod tool;

pub use channel::{AgentEvent, Channel, UserInput};
pub use error::{AgentError, Result};
pub use llm::{
    ChatRequest, LlmError, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, ToolSchema,
};
pub use message::{
    ContentBlock, Message, Role, StopReason, TokenUsage, ToolResult as MessageToolResult, ToolUse,
};
pub use session::SessionId;
pub use tool::{
    Permissions, Tool, ToolContext, ToolError, ToolOutcome, ToolRegistry,
    ToolResult as ToolInvocationResult,
};

pub use agent::{Agent, AgentBuilder, RunConfig};
pub use prompt::{ChainedPromptProvider, PromptProvider};
pub use evolution::{Candidate, CandidateError, CandidateKind, CandidateQueue};
pub use memory::{
    EmbeddingProvider, Fact, FactId, FactKind, FactStore, MemoryError, MemoryHit, MemoryResult,
    NewFact, VectorStore,
};
pub use store::{SessionStore, SessionStoreError, SessionSummary, StoreResult, UsageSummary};
