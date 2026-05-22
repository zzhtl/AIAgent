//! Tool abstraction.
//!
//! Mirrors the LLM trait pattern: `Tool` is a runtime-visible interface,
//! while concrete implementations live in `agent-tools`. The runtime drives
//! the registry; the registry never touches the network or the filesystem.

use crate::evolution::CandidateQueue;
use crate::llm::ToolSchema;
use crate::memory::FactStore;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("unknown tool: {0}")]
    Unknown(String),

    #[error("invalid arguments for {tool}: {detail}")]
    InvalidArguments { tool: String, detail: String },

    #[error("permission denied: {0}")]
    PermissionDenied(String),

    #[error("execution failed: {0}")]
    ExecutionFailed(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

pub type ToolResult<T> = std::result::Result<T, ToolError>;

/// Permission gates that bound what tools may do at runtime. Conservative
/// defaults; the CLI can flip flags via config.
#[derive(Debug, Clone)]
pub struct Permissions {
    pub allow_read: bool,
    pub allow_write: bool,
    pub allow_shell: bool,
    pub allow_network: bool,
    /// Hard cap for `bash` and similar long-running tools.
    pub max_runtime_secs: u64,
}

impl Default for Permissions {
    fn default() -> Self {
        Self {
            allow_read: true,
            allow_write: true,
            allow_shell: true,
            allow_network: true,
            max_runtime_secs: 120,
        }
    }
}

/// Per-invocation context handed to a tool. Tools should treat this as
/// read-only.
#[derive(Clone)]
pub struct ToolContext {
    pub workspace: PathBuf,
    pub permissions: Permissions,
    pub session_id: String,
    /// Optional handle to the long-term fact store. Memory tools
    /// (`remember` / `forget` / `recall`) need it; others ignore it.
    pub fact_store: Option<Arc<dyn FactStore>>,
    /// Optional candidate queue for the evolution flow. `propose_rule`
    /// and `propose_skill` enqueue here.
    pub candidate_queue: Option<CandidateQueue>,
}

impl std::fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolContext")
            .field("workspace", &self.workspace)
            .field("permissions", &self.permissions)
            .field("session_id", &self.session_id)
            .field("fact_store", &self.fact_store.is_some())
            .field("candidate_queue", &self.candidate_queue.is_some())
            .finish()
    }
}

impl ToolContext {
    pub fn new(workspace: PathBuf) -> Self {
        Self {
            workspace,
            permissions: Permissions::default(),
            session_id: String::new(),
            fact_store: None,
            candidate_queue: None,
        }
    }

    pub fn with_permissions(mut self, permissions: Permissions) -> Self {
        self.permissions = permissions;
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    pub fn with_fact_store(mut self, store: Arc<dyn FactStore>) -> Self {
        self.fact_store = Some(store);
        self
    }

    pub fn with_candidate_queue(mut self, queue: CandidateQueue) -> Self {
        self.candidate_queue = Some(queue);
        self
    }
}

/// Outcome of a tool invocation. `is_error == true` signals semantic failure
/// (the tool ran but the operation failed: file not found, command exited
/// non-zero); `Err(ToolError)` signals runtime / permission failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub text: String,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn ok(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: false }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self { text: text.into(), is_error: true }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool's argument object.
    fn parameters(&self) -> Value;

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome>;

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            parameters: self.parameters(),
        }
    }
}

/// Registry of available tools, keyed by name. Cloning the registry is cheap
/// (each tool is behind an `Arc`).
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// All registered tool schemas in stable (alphabetical) order. Stable
    /// ordering matters because the LLM may cache prompts on the prefix.
    pub fn schemas(&self) -> Vec<ToolSchema> {
        let mut names: Vec<&str> = self.tools.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
            .into_iter()
            .filter_map(|n| self.tools.get(n).map(|t| t.schema()))
            .collect()
    }

    pub async fn invoke(
        &self,
        name: &str,
        args: Value,
        ctx: &ToolContext,
    ) -> ToolResult<ToolOutcome> {
        let tool = self
            .get(name)
            .ok_or_else(|| ToolError::Unknown(name.to_string()))?;
        tool.invoke(args, ctx).await
    }
}
