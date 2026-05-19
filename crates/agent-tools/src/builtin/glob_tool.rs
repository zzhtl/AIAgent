//! `glob` — list paths matching a shell-style glob pattern.

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use glob::glob_with;
use serde::Deserialize;
use serde_json::{json, Value};

use super::path_safety;

const MAX_RESULTS: usize = 200;

#[derive(Default)]
pub struct GlobTool;

#[derive(Deserialize)]
struct Args {
    pattern: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "List paths matching a shell glob (e.g. `src/**/*.rs`). Returns up to \
         200 entries. Pattern is resolved relative to `cwd` (workspace by \
         default)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern":          { "type": "string" },
                "cwd":              { "type": "string", "description": "Working directory." },
                "case_insensitive": { "type": "boolean", "default": false }
            },
            "required": ["pattern"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_read {
            return Err(ToolError::PermissionDenied("glob disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "glob".into(),
            detail: e.to_string(),
        })?;

        let cwd = args
            .cwd
            .as_deref()
            .map(|c| path_safety::resolve(ctx, c))
            .unwrap_or_else(|| ctx.workspace.clone());
        let full_pattern = cwd.join(&args.pattern);
        let pattern_str = full_pattern.to_string_lossy().to_string();

        let options = glob::MatchOptions {
            case_sensitive: !args.case_insensitive,
            require_literal_separator: false,
            require_literal_leading_dot: false,
        };

        let iter = glob_with(&pattern_str, options).map_err(|e| ToolError::InvalidArguments {
            tool: "glob".into(),
            detail: e.to_string(),
        })?;

        let mut paths = Vec::new();
        let mut truncated = false;
        for entry in iter {
            match entry {
                Ok(p) => {
                    let rel = p.strip_prefix(&cwd).unwrap_or(&p).display().to_string();
                    paths.push(rel);
                    if paths.len() >= MAX_RESULTS {
                        truncated = true;
                        break;
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, "glob entry error");
                }
            }
        }

        if paths.is_empty() {
            return Ok(ToolOutcome::ok(format!("no matches for `{}`", args.pattern)));
        }
        let suffix = if truncated { format!("\n[truncated at {MAX_RESULTS}]") } else { String::new() };
        Ok(ToolOutcome::ok(format!(
            "{} path(s):\n{}{suffix}",
            paths.len(),
            paths.join("\n")
        )))
    }
}
