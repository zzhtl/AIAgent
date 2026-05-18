//! agent-core
//!
//! Runtime kernel for the AI agent: message types, the agent loop, and the
//! `Channel` abstraction shared by CLI and bot front-ends.
//!
//! This crate stays free of any concrete provider, tool, or transport
//! implementation — those live in sibling crates and are wired in at the
//! application boundary.

pub mod error;

pub use error::{AgentError, Result};
