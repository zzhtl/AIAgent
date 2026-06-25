//! agent-web: HTTP / SSE entry for the agent runtime.
//!
//! Exposes the same agent kernel as the CLI/bot over HTTP:
//!
//! - `GET  /health` → `ok`
//! - `POST /chat`   → `{ "input": "...", "session": "user-42" }`, responds with
//!   a Server-Sent Events stream of `AgentEvent`s (one event per SSE `data:`),
//!   ending with a `done` event. `session` isolates per-conversation history in
//!   process memory (callers that need durability persist `transcript_delta`).
//!
//! The reusable seam is [`stream_events`]: it maps an agent event stream to SSE
//! payloads and folds each run's `transcript_delta` back into history. It is
//! transport-agnostic and unit-tested with a synthetic stream (no LLM needed).

use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{BoxStream, Stream};
use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::Mutex;

use agent_core::evolution::CandidateQueue;
use agent_core::{
    Agent, AgentEvent, ChainedPromptProvider, FactStore, LlmProvider, Message, PromptProvider,
    SessionId, ToolRegistry, UserInput,
};
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_memory::{FactsPromptProvider, MarkdownFactStore};
use agent_skills::{Augmenter, RuleSet, SkillRegistry};

/// In-process per-session transcript store, shared across requests.
type Histories = Arc<Mutex<HashMap<String, Vec<Message>>>>;

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    histories: Histories,
}

#[derive(Debug, Deserialize)]
struct ChatReq {
    input: String,
    #[serde(default)]
    session: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let agent = build_agent().context("agent-web init")?;
    let state = AppState {
        agent: Arc::new(agent),
        histories: Arc::new(Mutex::new(HashMap::new())),
    };

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/chat", post(chat))
        .with_state(state);

    let addr = std::env::var("AGENT_WEB_ADDR").unwrap_or_else(|_| "127.0.0.1:8787".to_string());
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("bind {addr}"))?;
    tracing::info!("agent-web listening on http://{addr}");
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

/// `POST /chat` → SSE stream of the run's events.
async fn chat(
    State(state): State<AppState>,
    Json(req): Json<ChatReq>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let session = req.session.unwrap_or_else(|| "default".to_string());
    let history = state.histories.lock().await.get(&session).cloned().unwrap_or_default();
    let sid = SessionId::from(session.as_str());
    let stream = state.agent.run(sid, history, UserInput::new(req.input));

    let events = stream_events(stream, session, state.histories.clone())
        .map(|json| Ok(Event::default().data(json)));
    Sse::new(events).keep_alive(KeepAlive::default())
}

/// Map an agent event stream to SSE JSON payloads, folding the terminal
/// `transcript_delta` back into `histories[session]`. Transport-agnostic so it
/// can be unit-tested without an HTTP server or an LLM.
fn stream_events(
    stream: BoxStream<'static, AgentEvent>,
    session: String,
    histories: Histories,
) -> impl Stream<Item = String> {
    async_stream::stream! {
        let mut stream = stream;
        while let Some(ev) = stream.next().await {
            if let AgentEvent::Done { transcript_delta, .. } = &ev {
                histories
                    .lock()
                    .await
                    .entry(session.clone())
                    .or_default()
                    .extend(transcript_delta.clone());
            }
            yield serde_json::to_string(&ev).unwrap_or_else(|_| "{}".to_string());
        }
    }
}

fn build_agent() -> Result<Agent> {
    let (provider, model) = build_provider()?;
    let mut tools = ToolRegistry::new();
    agent_tools::register_builtins(&mut tools);
    agent_tools::register_memory_tools(&mut tools);
    agent_tools::register_evolution_tools(&mut tools);

    let config_dir = resolve_config_dir();
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;

    let skills = SkillRegistry::load_dir(&config_dir.join("skills")).map_err(|e| anyhow!("skills: {e}"))?;
    let rules = RuleSet::load_dir(&config_dir.join("rules")).map_err(|e| anyhow!("rules: {e}"))?;
    let augmenter = Augmenter::new(rules, skills);

    let fact_store: Arc<dyn FactStore> = Arc::new(MarkdownFactStore::open(config_dir.join("memory")));
    let facts_provider = FactsPromptProvider::new(fact_store.clone());

    let mut chain = ChainedPromptProvider::new();
    if !augmenter.is_empty() {
        chain.push(Arc::new(augmenter));
    }
    chain.push(Arc::new(facts_provider));

    let candidate_queue = CandidateQueue::open(config_dir.join("evolution").join("queue.json"));

    let mut builder = Agent::builder()
        .with_llm(provider)
        .with_model(model)
        .with_tools(tools)
        .with_fact_store(fact_store)
        .with_candidate_queue(candidate_queue)
        .with_workspace(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if !chain.is_empty() {
        let provider_arc: Arc<dyn PromptProvider> = Arc::new(chain);
        builder = builder.with_prompt_provider(provider_arc);
    }
    builder.build().map_err(|e| anyhow!("agent builder: {e}"))
}

fn build_provider() -> Result<(Arc<dyn LlmProvider>, String)> {
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let model = std::env::var("AGENT_WEB_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        let provider: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new(OpenAiConfig::openai(key)).map_err(|e| anyhow!("provider init: {e}"))?);
        return Ok((provider, model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("AGENT_WEB_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".into());
        let provider: Arc<dyn LlmProvider> =
            Arc::new(AnthropicProvider::new(AnthropicConfig::new(key)).map_err(|e| anyhow!("provider init: {e}"))?);
        return Ok((provider, model));
    }
    if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
        let model = std::env::var("AGENT_WEB_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
        let provider: Arc<dyn LlmProvider> =
            Arc::new(OpenAiProvider::new(OpenAiConfig::deepseek(key)).map_err(|e| anyhow!("provider init: {e}"))?);
        return Ok((provider, model));
    }
    Err(anyhow!("agent-web: set one of OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY"))
}

fn resolve_config_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AGENT_CONFIG_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("agent");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".config").join("agent");
    }
    PathBuf::from(".agent")
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::StopReason;

    #[tokio::test]
    async fn stream_events_emits_json_and_updates_history() {
        let histories: Histories = Arc::new(Mutex::new(HashMap::new()));
        let events = vec![
            AgentEvent::TextDelta { delta: "hi".into() },
            AgentEvent::Done {
                reason: StopReason::EndTurn,
                transcript_delta: vec![Message::user("hi"), Message::assistant("hello")],
            },
        ];
        let stream = futures::stream::iter(events).boxed();

        let out: Vec<String> =
            stream_events(stream, "s1".to_string(), histories.clone()).collect().await;

        assert_eq!(out.len(), 2);
        assert!(out[0].contains("text_delta"), "got: {}", out[0]);
        assert!(out[1].contains("\"kind\":\"done\""), "got: {}", out[1]);
        // The Done transcript_delta (2 messages) was folded into history.
        assert_eq!(histories.lock().await.get("s1").map(Vec::len), Some(2));
    }
}
