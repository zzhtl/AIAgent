//! `file_read` — read a UTF-8 text file, optionally a line range.

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs;

use super::path_safety;

#[derive(Default)]
pub struct FileReadTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "file_read"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file. Optionally skip the first `offset` lines and \
         cap at `limit` lines. Paths are relative to the workspace unless \
         absolute."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path":   { "type": "string",  "description": "File path." },
                "offset": { "type": "integer", "minimum": 0, "description": "Line offset (0-based)." },
                "limit":  { "type": "integer", "minimum": 1, "description": "Max lines to return." }
            },
            "required": ["path"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_read {
            return Err(ToolError::PermissionDenied("file_read disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "file_read".into(),
            detail: e.to_string(),
        })?;
        let path = path_safety::resolve(ctx, &args.path);
        let content = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutcome::error(format!("read {}: {e}", path.display()))),
        };

        let total_lines = content.lines().count();
        let sliced = match (args.offset, args.limit) {
            (None, None) => content,
            (offset, limit) => {
                let off = offset.unwrap_or(0);
                let lim = limit.unwrap_or(usize::MAX);
                content
                    .lines()
                    .skip(off)
                    .take(lim)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        let header = format!("# {} ({} lines)\n", path.display(), total_lines);
        Ok(ToolOutcome::ok(format!("{header}{sliced}")))
    }
}
