//! `grep` — regex search across files in the workspace.

use std::path::{Path, PathBuf};

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use glob::Pattern;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::fs;

use super::path_safety;

const MAX_MATCHES: usize = 200;
const MAX_LINE_LEN: usize = 240;
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", ".venv", "dist", "build"];

#[derive(Default)]
pub struct GrepTool;

#[derive(Deserialize)]
struct Args {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    glob: Option<String>,
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search files for a regex pattern. Returns up to 200 `file:line: text` \
         matches. Skips `.git`, `target`, `node_modules`, `.venv`, `dist`, \
         `build`. `glob` (e.g. `**/*.rs`) narrows which files are scanned."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern":          { "type": "string", "description": "Regex pattern." },
                "path":             { "type": "string", "description": "Root directory (default: workspace)." },
                "case_insensitive": { "type": "boolean", "default": false },
                "glob":             { "type": "string", "description": "Optional glob filter, e.g. `**/*.rs`." }
            },
            "required": ["pattern"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_read {
            return Err(ToolError::PermissionDenied("grep disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "grep".into(),
            detail: e.to_string(),
        })?;

        let re = RegexBuilder::new(&args.pattern)
            .case_insensitive(args.case_insensitive)
            .build()
            .map_err(|e| ToolError::InvalidArguments {
                tool: "grep".into(),
                detail: format!("regex: {e}"),
            })?;

        let root = match args.path.as_deref() {
            Some(p) => path_safety::resolve(ctx, p),
            None => ctx.workspace.clone(),
        };
        if !root.exists() {
            return Ok(ToolOutcome::error(format!("path not found: {}", root.display())));
        }

        let glob_pattern = args.glob.as_deref().and_then(|g| Pattern::new(g).ok());

        // Filesystem walk is synchronous; hand it to a blocking pool so we
        // don't stall the tokio runtime on large trees.
        let walk_root = root.clone();
        let paths: Vec<PathBuf> = match tokio::task::spawn_blocking(move || {
            let mut out = Vec::new();
            walk_files(&walk_root, &mut out, 10_000);
            out
        })
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(ToolOutcome::error(format!("walk: {e}"))),
        };

        let mut hits = Vec::new();
        'outer: for path in paths {
            if let Some(g) = glob_pattern.as_ref() {
                let rel = path.strip_prefix(&root).unwrap_or(&path);
                if !g.matches_path(rel) {
                    continue;
                }
            }
            let Ok(content) = fs::read_to_string(&path).await else { continue };
            let rel = path.strip_prefix(&root).unwrap_or(&path).display().to_string();
            for (idx, line) in content.lines().enumerate() {
                if re.is_match(line) {
                    let shown = agent_core::text::truncate_with_ellipsis(line, MAX_LINE_LEN);
                    hits.push(format!("{rel}:{}: {shown}", idx + 1));
                    if hits.len() >= MAX_MATCHES {
                        break 'outer;
                    }
                }
            }
        }

        if hits.is_empty() {
            return Ok(ToolOutcome::ok(format!("no matches in {}", root.display())));
        }
        let truncated_note = if hits.len() == MAX_MATCHES {
            format!("\n[truncated at {MAX_MATCHES} matches]")
        } else {
            String::new()
        };
        Ok(ToolOutcome::ok(format!(
            "{} matches:\n{}{truncated_note}",
            hits.len(),
            hits.join("\n")
        )))
    }
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>, cap: usize) {
    if out.len() >= cap {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        if out.len() >= cap {
            return;
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if SKIP_DIRS.contains(&name) || (name.starts_with('.') && name != ".") {
                    continue;
                }
            }
            walk_files(&path, out, cap);
        } else if meta.is_file() {
            out.push(path);
        }
    }
}
