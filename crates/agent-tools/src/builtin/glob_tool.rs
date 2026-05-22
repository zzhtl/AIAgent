//! `glob` — list paths matching a shell-style glob pattern.

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use glob::glob_with;
use serde::Deserialize;
use serde_json::{json, Value};

use super::path_safety;

const MAX_RESULTS: usize = 200;
/// Defensive cap on how many filesystem entries we will visit before
/// admitting defeat — prevents `cwd: "/"` + `pattern: "**/*"` from
/// scanning the entire host.
const MAX_VISITED: usize = 100_000;

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

        // glob_with walks the filesystem synchronously; ship it to a
        // blocking worker so the tokio reactor is never stalled. We also
        // cap the total number of visited entries to keep pathological
        // patterns (`cwd: "/", pattern: "**/*"`) from hanging the agent.
        let cwd_for_walk = cwd.clone();
        let join = tokio::task::spawn_blocking(move || {
            let iter = glob_with(&pattern_str, options).map_err(|e| e.to_string())?;
            let mut paths: Vec<String> = Vec::new();
            let mut visited: usize = 0;
            let mut truncated = false;
            for entry in iter {
                visited += 1;
                if visited > MAX_VISITED {
                    truncated = true;
                    break;
                }
                match entry {
                    Ok(p) => {
                        let rel = p
                            .strip_prefix(&cwd_for_walk)
                            .unwrap_or(&p)
                            .display()
                            .to_string();
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
            Ok::<(Vec<String>, bool), String>((paths, truncated))
        })
        .await;

        let (paths, truncated) = match join {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                return Err(ToolError::InvalidArguments { tool: "glob".into(), detail: e })
            }
            Err(e) => return Ok(ToolOutcome::error(format!("glob join: {e}"))),
        };

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
