//! `bash` — run a shell command with a hard timeout.

use std::process::Stdio;
use std::time::Duration;

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;
use tokio::time::timeout;

use super::path_safety;

const STDOUT_CAP: usize = 16 * 1024;

#[derive(Default)]
pub struct BashTool;

#[derive(Deserialize)]
struct Args {
    command: String,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a shell command via `bash -c`. Returns stdout/stderr (each \
         capped at 16 KiB) and the exit code. `timeout_secs` is bounded by \
         the runtime permission cap."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command":      { "type": "string" },
                "cwd":          { "type": "string", "description": "Working directory (relative to workspace if not absolute)." },
                "timeout_secs": { "type": "integer", "minimum": 1 }
            },
            "required": ["command"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_shell {
            return Err(ToolError::PermissionDenied("bash disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "bash".into(),
            detail: e.to_string(),
        })?;

        let limit = args
            .timeout_secs
            .unwrap_or(ctx.permissions.max_runtime_secs)
            .min(ctx.permissions.max_runtime_secs);

        let cwd = args
            .cwd
            .as_deref()
            .map(|c| path_safety::resolve(ctx, c))
            .unwrap_or_else(|| ctx.workspace.clone());

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&args.command)
            .current_dir(&cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        let exec = cmd.output();
        let result = match timeout(Duration::from_secs(limit), exec).await {
            Ok(Ok(output)) => output,
            Ok(Err(e)) => return Ok(ToolOutcome::error(format!("spawn: {e}"))),
            Err(_) => {
                return Ok(ToolOutcome::error(format!(
                    "command timed out after {limit}s"
                )))
            }
        };

        let stdout = truncate_utf8(&result.stdout, STDOUT_CAP);
        let stderr = truncate_utf8(&result.stderr, STDOUT_CAP);
        let code = result.status.code();

        let body = format!(
            "exit_code: {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            code.map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
            stdout,
            stderr,
        );

        if code == Some(0) {
            Ok(ToolOutcome::ok(body))
        } else {
            Ok(ToolOutcome::error(body))
        }
    }
}

fn truncate_utf8(bytes: &[u8], cap: usize) -> String {
    if bytes.len() <= cap {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let head = String::from_utf8_lossy(&bytes[..cap]).into_owned();
    format!("{head}\n... [truncated, {} bytes total]", bytes.len())
}
