//! Adapter exposing one MCP-server tool as an `agent_core::Tool`.

use agent_core::tool::{Tool, ToolContext, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde_json::Value;

use crate::McpClient;

/// Wraps a single tool from an MCP server. The agent may see a namespaced
/// `name` (e.g. `fs__read_file`) while `remote_name` is what the server's
/// `tools/call` expects. A failed call is surfaced as an error *outcome* (not
/// an `Err`) so the agent loop still feeds a result back to the model.
pub struct McpTool {
    client: McpClient,
    name: String,
    remote_name: String,
    description: String,
    parameters: Value,
}

impl McpTool {
    pub fn new(
        client: McpClient,
        name: impl Into<String>,
        remote_name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            client,
            name: name.into(),
            remote_name: remote_name.into(),
            description: description.into(),
            parameters,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn invoke(&self, args: Value, _ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        match self.client.call_tool(&self.remote_name, args).await {
            Ok((text, true)) => Ok(ToolOutcome::error(text)),
            Ok((text, false)) => Ok(ToolOutcome::ok(text)),
            Err(e) => Ok(ToolOutcome::error(format!("mcp tool `{}` failed: {e}", self.name))),
        }
    }
}
