//! agent-bot: sequential stdio JSON adapter for the shared agent runtime.
//!
//! Send one JSON object per line on stdin and consume one serialised
//! `AgentEvent` per line on stdout. Unlike the concurrent web entry, stdin is
//! processed strictly in order, so no per-session mutex is needed. Omitted
//! session ids intentionally share the single-user `"default"` conversation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use agent_config::AgentConfig;
use agent_core::{AgentEvent, Message, SessionId, SessionStore, SessionStoreError, UserInput};
use agent_memory::SqliteSessionStore;
use agent_runtime::{build_runtime, open_session_store, BuildOptions};
use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug, Deserialize)]
struct BotRequest {
    input: String,
    /// Optional per-user or per-channel identifier. Omission is deliberately
    /// mapped to `default` because one stdio producer is normally one user.
    #[serde(default)]
    session: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let config = AgentConfig::load(None).map_err(|error| anyhow!("config: {error}"))?;
    let options = BuildOptions {
        model_override: non_empty_env("AGENT_BOT_MODEL"),
        workspace: Some(std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        env_probe_fallback: true,
        ..BuildOptions::default()
    };
    let runtime = build_runtime(&config, options).await.context("agent-bot init")?;
    for warning in &runtime.warnings {
        tracing::warn!("{warning}");
    }
    let session_store = if config.bot.persist_sessions {
        Some(Arc::new(open_session_store(&runtime.config_dir).await?))
    } else {
        None
    };
    let agent = runtime.agent;

    let mut stdin = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();
    let mut histories: HashMap<String, Vec<Message>> = HashMap::new();

    while let Some(line) = stdin.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let request: BotRequest = match serde_json::from_str(line) {
            Ok(request) => request,
            Err(error) => {
                emit_error(&mut stdout, &format!("invalid request: {error}")).await?;
                continue;
            }
        };

        let session_key = request.session.unwrap_or_else(|| "default".to_string());
        if !histories.contains_key(&session_key) {
            let sid = SessionId::from(session_key.as_str());
            let history = load_or_create_history(session_store.as_deref(), &sid).await;
            histories.insert(session_key.clone(), history);
        }
        let history = histories.get(&session_key).cloned().unwrap_or_default();
        let sid = SessionId::from(session_key.as_str());
        let mut stream = agent.run(sid.clone(), history, UserInput::new(request.input));
        while let Some(event) = stream.next().await {
            emit_event(&mut stdout, &event).await?;
            if let AgentEvent::Done { transcript_delta, .. } = &event {
                histories
                    .entry(session_key.clone())
                    .or_default()
                    .extend(transcript_delta.clone());
                if let Some(store) = session_store.as_ref() {
                    if let Err(error) = store.append_messages(&sid, transcript_delta).await {
                        tracing::warn!(session = %sid, %error, "failed to persist messages");
                    }
                }
            }
        }
    }
    Ok(())
}

async fn load_or_create_history(
    store: Option<&SqliteSessionStore>,
    sid: &SessionId,
) -> Vec<Message> {
    let Some(store) = store else {
        return Vec::new();
    };
    match store.load_messages(sid).await {
        Ok(messages) => messages,
        Err(SessionStoreError::NotFound(_)) => {
            if let Err(error) = store.ensure_session(sid, None).await {
                tracing::warn!(session = %sid, %error, "failed to create persistent session");
            }
            Vec::new()
        }
        Err(error) => {
            tracing::warn!(session = %sid, %error, "failed to load persistent session");
            Vec::new()
        }
    }
}

async fn emit_event(out: &mut tokio::io::Stdout, event: &AgentEvent) -> Result<()> {
    let line = serde_json::to_string(event).map_err(|error| anyhow!("serialise event: {error}"))?;
    out.write_all(line.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await?;
    Ok(())
}

async fn emit_error(out: &mut tokio::io::Stdout, message: &str) -> Result<()> {
    let payload = serde_json::json!({ "kind": "error", "message": message });
    out.write_all(payload.to_string().as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await?;
    Ok(())
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn persistence_restores_explicit_and_default_sessions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = open_session_store(dir.path()).await.expect("store");
        for name in ["default", "user-42"] {
            let sid = SessionId::from(name);
            assert!(load_or_create_history(Some(&store), &sid).await.is_empty());
            store
                .append_messages(&sid, &[Message::user(format!("hello {name}"))])
                .await
                .expect("append");
            let restored = load_or_create_history(Some(&store), &sid).await;
            assert_eq!(restored.len(), 1);
        }
    }

    #[tokio::test]
    async fn in_memory_mode_starts_empty() {
        let history = load_or_create_history(None, &SessionId::from("default")).await;
        assert!(history.is_empty());
    }
}
