//! agent-mcp
//!
//! A minimal MCP (Model Context Protocol) client. Connects to an MCP server
//! over stdio (newline-delimited JSON-RPC 2.0), performs the `initialize`
//! handshake, lists the server's tools, and exposes each one as an
//! `agent_core::Tool` (`McpTool`) the agent loop can dispatch like any other.
//!
//! Scope is intentionally small: stdio transport + `tools/list` / `tools/call`.
//! Resources / prompts / SSE transport are out of scope for this batch.

mod client;
mod tool;

pub use client::McpClient;
pub use tool::McpTool;

use serde_json::Value;

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("spawn failed: {0}")]
    Spawn(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("rpc error: {0}")]
    Rpc(String),
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("request timed out")]
    Timeout,
}

/// A tool definition advertised by an MCP server's `tools/list`.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's arguments (the server's `inputSchema`).
    pub input_schema: Value,
}
