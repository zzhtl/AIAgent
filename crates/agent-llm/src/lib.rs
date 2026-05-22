//! agent-llm
//!
//! Concrete LLM provider clients and the `ProviderRegistry` that selects
//! between them at runtime. The abstraction (`LlmProvider` trait,
//! `ChatRequest`, `LlmEvent`, `ToolSchema`, `LlmError`) lives in
//! `agent-core` so the runtime doesn't transitively depend on transport
//! crates.

pub mod providers;
pub mod registry;
pub mod sse;

pub use registry::ProviderRegistry;

// Re-export the abstraction so callers can `use agent_llm::{LlmProvider, ChatRequest, ...}`
// without reaching into `agent-core` for every type.
pub use agent_core::llm::{
    ChatRequest, LlmError, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, ToolSchema,
};
