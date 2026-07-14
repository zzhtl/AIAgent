//! Shared application-boundary assembly for every first-party front-end.
//!
//! `agent-core` deliberately knows nothing about concrete providers, tools,
//! persistence, or transports. This crate is the reusable outer boundary that
//! turns an [`agent_config::AgentConfig`] into the same fully wired runtime for
//! the CLI, stdio bot, and HTTP server.

mod assemble;

pub use assemble::{
    build_runtime, build_runtime_with_provider, open_session_store, resolve_provider,
    AgentRuntime, BuildOptions, ResolvedProvider,
};
