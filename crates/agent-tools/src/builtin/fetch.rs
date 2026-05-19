//! `fetch` — HTTP GET a URL and return the body (capped).

use std::time::Duration;

use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};

const MAX_BYTES: usize = 1024 * 1024;
const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(Default)]
pub struct FetchTool;

#[derive(Deserialize)]
struct Args {
    url: String,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

#[async_trait]
impl Tool for FetchTool {
    fn name(&self) -> &str {
        "fetch"
    }

    fn description(&self) -> &str {
        "HTTP GET a URL and return the response body (truncated at 1 MiB). \
         Only http/https schemes are allowed; default timeout 30s."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url":          { "type": "string", "format": "uri" },
                "timeout_secs": { "type": "integer", "minimum": 1, "maximum": 120 }
            },
            "required": ["url"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        if !ctx.permissions.allow_network {
            return Err(ToolError::PermissionDenied("fetch disabled".into()));
        }
        let args: Args = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "fetch".into(),
            detail: e.to_string(),
        })?;

        let url = args.url.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Ok(ToolOutcome::error("only http/https URLs are allowed"));
        }

        let timeout = args.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS).min(120);
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .map_err(|e| ToolError::ExecutionFailed(format!("client init: {e}")))?;

        let resp = match client.get(url).send().await {
            Ok(r) => r,
            Err(e) => return Ok(ToolOutcome::error(format!("network: {e}"))),
        };
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => return Ok(ToolOutcome::error(format!("body: {e}"))),
        };
        let truncated = bytes.len() > MAX_BYTES;
        let slice = if truncated { &bytes[..MAX_BYTES] } else { &bytes[..] };
        let body = String::from_utf8_lossy(slice).into_owned();
        let header = format!(
            "GET {url} → {status} ({}){}",
            content_type,
            if truncated { format!(", truncated at {MAX_BYTES} bytes") } else { String::new() }
        );
        let outcome = if status.is_success() {
            ToolOutcome::ok(format!("{header}\n\n{body}"))
        } else {
            ToolOutcome::error(format!("{header}\n\n{body}"))
        };
        Ok(outcome)
    }
}
