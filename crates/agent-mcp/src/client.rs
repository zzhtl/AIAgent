//! MCP client over newline-delimited JSON-RPC 2.0.
//!
//! A dedicated reader task drains the transport and dispatches each response to
//! the waiting caller by request `id` (avoids `select!` cancel-safety hazards on
//! `read_line`). Requests register a `oneshot` in a shared pending map, write the
//! frame, and await the reply with a timeout.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{oneshot, Mutex};

use crate::{McpError, McpToolDef};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const PROTOCOL_VERSION: &str = "2024-11-05";

type Writer = Box<dyn AsyncWrite + Unpin + Send>;
type PendingMap = HashMap<i64, oneshot::Sender<Result<Value, McpError>>>;

struct Inner {
    writer: Mutex<Writer>,
    pending: Mutex<PendingMap>,
    next_id: AtomicI64,
    /// Kept alive so the server process isn't reaped while the client lives.
    _child: Mutex<Option<Child>>,
}

/// A connected MCP client. Cheap to clone (shared `Arc`).
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<Inner>,
}

impl McpClient {
    /// Build a client over arbitrary reader/writer halves. Used directly by
    /// tests (with an in-memory duplex); production goes through
    /// [`McpClient::connect_stdio`].
    pub async fn new(
        reader: impl AsyncRead + Unpin + Send + 'static,
        writer: impl AsyncWrite + Unpin + Send + 'static,
        child: Option<Child>,
    ) -> Result<Self, McpError> {
        let inner = Arc::new(Inner {
            writer: Mutex::new(Box::new(writer)),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicI64::new(1),
            _child: Mutex::new(child),
        });

        // Reader task: dispatch each response to its waiter by id. Exits on EOF.
        let reader_inner = inner.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(reader).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let val: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Notifications carry no `id`; ignore them.
                let Some(id) = val.get("id").and_then(Value::as_i64) else {
                    continue;
                };
                let waiter = reader_inner.pending.lock().await.remove(&id);
                if let Some(reply) = waiter {
                    let res = if let Some(err) = val.get("error") {
                        Err(McpError::Rpc(err.to_string()))
                    } else {
                        Ok(val.get("result").cloned().unwrap_or(Value::Null))
                    };
                    let _ = reply.send(res);
                }
            }
        });

        let client = Self { inner };
        client.initialize().await?;
        Ok(client)
    }

    /// Spawn an MCP server subprocess and connect over its stdio.
    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<Self, McpError> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| McpError::Spawn(e.to_string()))?;
        let stdin = child.stdin.take().ok_or_else(|| McpError::Spawn("no stdin".into()))?;
        let stdout = child.stdout.take().ok_or_else(|| McpError::Spawn("no stdout".into()))?;
        Self::new(stdout, stdin, Some(child)).await
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().await.insert(id, tx);
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        self.write_msg(&msg).await?;
        match tokio::time::timeout(REQUEST_TIMEOUT, rx).await {
            Ok(Ok(res)) => res,
            Ok(Err(_)) => Err(McpError::Transport("reply channel closed".into())),
            Err(_) => {
                self.inner.pending.lock().await.remove(&id);
                Err(McpError::Timeout)
            }
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let msg = json!({"jsonrpc": "2.0", "method": method, "params": params});
        self.write_msg(&msg).await
    }

    async fn write_msg(&self, msg: &Value) -> Result<(), McpError> {
        let mut s = serde_json::to_string(msg).map_err(|e| McpError::Protocol(e.to_string()))?;
        s.push('\n');
        let mut w = self.inner.writer.lock().await;
        w.write_all(s.as_bytes()).await.map_err(|e| McpError::Transport(e.to_string()))?;
        w.flush().await.map_err(|e| McpError::Transport(e.to_string()))?;
        Ok(())
    }

    async fn initialize(&self) -> Result<(), McpError> {
        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "agent-mcp", "version": "0.1.0" }
        });
        self.request("initialize", params).await?;
        self.notify("notifications/initialized", json!({})).await?;
        Ok(())
    }

    /// List the tools the server advertises.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDef>, McpError> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut out = Vec::with_capacity(tools.len());
        for t in tools {
            let name = t.get("name").and_then(Value::as_str).unwrap_or_default().to_string();
            if name.is_empty() {
                continue;
            }
            let description =
                t.get("description").and_then(Value::as_str).unwrap_or_default().to_string();
            let input_schema =
                t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"}));
            out.push(McpToolDef { name, description, input_schema });
        }
        Ok(out)
    }

    /// Call a tool. Returns the concatenated text content and the server's
    /// `isError` flag.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<(String, bool), McpError> {
        let params = json!({ "name": name, "arguments": arguments });
        let result = self.request("tools/call", params).await?;
        let is_error = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        Ok((extract_text(&result), is_error))
    }
}

/// Concatenate the `text` parts of an MCP tool result's `content` array.
fn extract_text(result: &Value) -> String {
    let Some(content) = result.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-process MCP server over a duplex stream: answers
    /// `initialize`, `tools/list`, and `tools/call` for one `echo` tool.
    async fn mock_server(io: tokio::io::DuplexStream) {
        let (read, mut write) = tokio::io::split(io);
        let mut lines = BufReader::new(read).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let req: Value = match serde_json::from_str(line.trim()) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let method = req.get("method").and_then(Value::as_str).unwrap_or("");
            let id = req.get("id").cloned();
            let result = match method {
                "initialize" => json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "serverInfo": { "name": "mock", "version": "0" }
                }),
                "tools/list" => json!({
                    "tools": [{
                        "name": "echo",
                        "description": "Echoes the message",
                        "inputSchema": { "type": "object", "properties": { "msg": { "type": "string" } } }
                    }]
                }),
                "tools/call" => {
                    let msg = req.pointer("/params/arguments/msg").and_then(Value::as_str).unwrap_or("");
                    json!({ "content": [{ "type": "text", "text": format!("echo: {msg}") }], "isError": false })
                }
                _ => continue, // notifications need no reply
            };
            // Only requests (with an id) get a response.
            if let Some(id) = id {
                let resp = json!({"jsonrpc": "2.0", "id": id, "result": result});
                let mut s = serde_json::to_string(&resp).unwrap();
                s.push('\n');
                let _ = write.write_all(s.as_bytes()).await;
                let _ = write.flush().await;
            }
        }
    }

    #[tokio::test]
    async fn handshake_list_and_call() {
        let (client_io, server_io) = tokio::io::duplex(8192);
        let (c_read, c_write) = tokio::io::split(client_io);
        let server = tokio::spawn(mock_server(server_io));

        let client = McpClient::new(c_read, c_write, None).await.expect("connect");

        let tools = client.list_tools().await.expect("list");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let (text, is_err) = client.call_tool("echo", json!({"msg": "hi"})).await.expect("call");
        assert_eq!(text, "echo: hi");
        assert!(!is_err);

        // The client's reader task keeps `Inner` (and thus the writer) alive, so
        // the server never sees EOF — abort it rather than awaiting.
        server.abort();
    }

    #[tokio::test]
    async fn connect_stdio_bad_command_errors() {
        // A non-existent server binary must surface a Spawn error, not hang.
        let res =
            McpClient::connect_stdio("definitely-not-a-real-binary-xyz123", &[], &[]).await;
        assert!(matches!(res, Err(McpError::Spawn(_))));
    }
}
