//! agent-llm
//!
//! LLM provider abstraction (`LlmProvider` trait) and concrete clients for
//! OpenAI, DeepSeek (OpenAI-compatible), and Anthropic Claude.
//!
//! Implementations are added by writing a new module under `providers/` and
//! registering it with the provider registry.
