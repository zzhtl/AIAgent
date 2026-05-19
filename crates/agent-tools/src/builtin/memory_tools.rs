//! Memory tools: `remember`, `forget`, `recall`.
//!
//! All three require a `FactStore` in `ToolContext`; without one they
//! return an explanatory error so the agent can recover gracefully.

use agent_core::memory::{FactKind, NewFact};
use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

fn parse_kind(s: &str) -> FactKind {
    match s.to_ascii_lowercase().as_str() {
        "preference" => FactKind::Preference,
        "project" => FactKind::Project,
        "reflection" => FactKind::Reflection,
        _ => FactKind::Note,
    }
}

#[derive(Default)]
pub struct RememberTool;

#[derive(Deserialize)]
struct RememberArgs {
    name: String,
    body: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
}

#[async_trait]
impl Tool for RememberTool {
    fn name(&self) -> &str {
        "remember"
    }

    fn description(&self) -> &str {
        "Persist a fact to long-term memory so future sessions can use it. \
         Use for user preferences, project context, or important notes. \
         `kind` is one of preference / project / reflection / note."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": { "type": "string", "description": "Short title for the fact." },
                "body": { "type": "string", "description": "Full content of the fact." },
                "kind": { "type": "string", "enum": ["preference", "project", "reflection", "note"] },
                "tags": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["name", "body"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let store = ctx
            .fact_store
            .as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("no fact store available".into()))?;
        let args: RememberArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "remember".into(),
            detail: e.to_string(),
        })?;
        let kind = args.kind.as_deref().map(parse_kind).unwrap_or(FactKind::Note);
        let new_fact = NewFact::new(args.name.clone(), args.body)
            .with_kind(kind)
            .with_tags(args.tags);
        let id = store
            .save(new_fact)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        Ok(ToolOutcome::ok(format!("remembered {} (id: {id})", args.name)))
    }
}

#[derive(Default)]
pub struct ForgetTool;

#[derive(Deserialize)]
struct ForgetArgs {
    id: String,
}

#[async_trait]
impl Tool for ForgetTool {
    fn name(&self) -> &str {
        "forget"
    }

    fn description(&self) -> &str {
        "Delete a fact from long-term memory by its id."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "id": { "type": "string", "description": "Fact id (slug)." }
            },
            "required": ["id"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let store = ctx
            .fact_store
            .as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("no fact store available".into()))?;
        let args: ForgetArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "forget".into(),
            detail: e.to_string(),
        })?;
        match store.delete(&agent_core::memory::FactId::from(args.id.as_str())).await {
            Ok(()) => Ok(ToolOutcome::ok(format!("forgot {}", args.id))),
            Err(agent_core::memory::MemoryError::NotFound(_)) => {
                Ok(ToolOutcome::error(format!("no fact with id `{}`", args.id)))
            }
            Err(e) => Err(ToolError::ExecutionFailed(e.to_string())),
        }
    }
}

#[derive(Default)]
pub struct RecallTool;

#[derive(Deserialize)]
struct RecallArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    5
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Search long-term memory for facts that match a query (substring \
         match over name and body). Returns up to `limit` results."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "limit": { "type": "integer", "minimum": 1, "maximum": 50, "default": 5 }
            },
            "required": ["query"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let store = ctx
            .fact_store
            .as_ref()
            .ok_or_else(|| ToolError::ExecutionFailed("no fact store available".into()))?;
        let args: RecallArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "recall".into(),
            detail: e.to_string(),
        })?;
        let hits = store
            .search(&args.query, args.limit)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
        if hits.is_empty() {
            return Ok(ToolOutcome::ok(format!("no facts match `{}`", args.query)));
        }
        let mut body = format!("found {} fact(s):\n", hits.len());
        for f in &hits {
            body.push_str(&format!("- {} (id: {}): {}\n", f.name, f.id, f.one_liner()));
        }
        Ok(ToolOutcome::ok(body))
    }
}
