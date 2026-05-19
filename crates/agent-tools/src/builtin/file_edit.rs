//! `file_edit` — exact-string replacement (Claude-Code-style).

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs;

use super::path_safety;

#[derive(Default)]
pub struct FileEditTool;

#[derive(Deserialize)]
struct Args {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "file_edit"
    }

    fn description(&self) -> &str {
        "Edit a text file by exact string replacement. Errors out if \
         `old_string` is missing or matches multiple times (unless \
         `replace_all` is true)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path":        { "type": "string" },
                "old_string":  { "type": "string", "description": "Exact text to find." },
                "new_string":  { "type": "string", "description": "Replacement text." },
                "replace_all": { "type": "boolean", "default": false }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_write {
            return Err(ToolError::PermissionDenied("file_edit disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "file_edit".into(),
            detail: e.to_string(),
        })?;
        if args.old_string.is_empty() {
            return Ok(ToolOutcome::error("old_string is empty"));
        }
        if args.old_string == args.new_string {
            return Ok(ToolOutcome::error("old_string and new_string are identical"));
        }

        let path = path_safety::resolve(ctx, &args.path);
        let original = match fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => return Ok(ToolOutcome::error(format!("read {}: {e}", path.display()))),
        };

        let occurrences = original.matches(&args.old_string).count();
        if occurrences == 0 {
            return Ok(ToolOutcome::error(format!(
                "old_string not found in {}",
                path.display()
            )));
        }
        if occurrences > 1 && !args.replace_all {
            return Ok(ToolOutcome::error(format!(
                "old_string matches {occurrences} times in {} — pass replace_all=true or use a more specific snippet",
                path.display()
            )));
        }

        let updated = if args.replace_all {
            original.replace(&args.old_string, &args.new_string)
        } else {
            original.replacen(&args.old_string, &args.new_string, 1)
        };

        fs::write(&path, &updated).await?;
        Ok(ToolOutcome::ok(format!(
            "edited {} ({} replacement{})",
            path.display(),
            if args.replace_all { occurrences } else { 1 },
            if args.replace_all && occurrences > 1 { "s" } else { "" }
        )))
    }
}
