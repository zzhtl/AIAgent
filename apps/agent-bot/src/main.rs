//! agent-bot: stdio JSON adapter.
//!
//! Spawn this process from any IM / bot platform. Send one JSON object per
//! line on stdin; consume one JSON event per line on stdout. The same agent
//! kernel as `agent-cli` is reused — every adapter only has to translate the
//! transport.
//!
//! ## Wire protocol
//!
//! Request line:
//! ```json
//! {"input": "你好"}
//! ```
//!
//! Response: one or more lines, each is a serialised `AgentEvent`. A run
//! always ends with a `done` event. Example sequence:
//! ```json
//! {"kind":"text_delta","delta":"你"}
//! {"kind":"text_delta","delta":"好"}
//! {"kind":"usage_report","usage":{...},"model":"gpt-4o-mini"}
//! {"kind":"done","reason":"end_turn","transcript_delta":[...]}
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use futures::StreamExt;
use serde::Deserialize;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use agent_core::{
    Agent, AgentEvent, ChainedPromptProvider, FactStore, LlmProvider, Message, PromptProvider,
    SessionId, ToolRegistry, UserInput,
};
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_memory::{FactsPromptProvider, MarkdownFactStore};
use agent_skills::{Augmenter, RuleSet, SkillRegistry};

#[derive(Debug, Deserialize)]
struct BotRequest {
    input: String,
    /// Optional: a session-style identifier. Currently only used as a tag
    /// for log correlation; conversation state lives inside this process.
    #[serde(default)]
    session: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let agent = build_agent().context("agent-bot init")?;

    let mut stdin = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();
    let mut history: Vec<Message> = Vec::new();

    while let Some(line) = stdin.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let req: BotRequest = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                emit_error(&mut stdout, &format!("invalid request: {e}")).await?;
                continue;
            }
        };

        let sid = SessionId::from(req.session.as_deref().unwrap_or("bot"));

        let mut stream = agent.run(sid, history.clone(), UserInput::new(req.input));
        while let Some(event) = stream.next().await {
            emit_event(&mut stdout, &event).await?;
            if let AgentEvent::Done { transcript_delta, .. } = &event {
                history.extend(transcript_delta.clone());
            }
        }
    }
    Ok(())
}

async fn emit_event(out: &mut tokio::io::Stdout, event: &AgentEvent) -> Result<()> {
    let line = serde_json::to_string(event).map_err(|e| anyhow!("serialise event: {e}"))?;
    out.write_all(line.as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await?;
    Ok(())
}

async fn emit_error(out: &mut tokio::io::Stdout, msg: &str) -> Result<()> {
    let payload = serde_json::json!({ "kind": "error", "message": msg });
    out.write_all(payload.to_string().as_bytes()).await?;
    out.write_all(b"\n").await?;
    out.flush().await?;
    Ok(())
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

    let skills = SkillRegistry::load_dir(&config_dir.join("skills"))
        .map_err(|e| anyhow!("skills: {e}"))?;
    let rules = RuleSet::load_dir(&config_dir.join("rules"))
        .map_err(|e| anyhow!("rules: {e}"))?;
    let augmenter = Augmenter::new(rules, skills);

    let fact_store: Arc<dyn FactStore> =
        Arc::new(MarkdownFactStore::open(config_dir.join("memory")));
    let facts_provider = FactsPromptProvider::new(fact_store.clone());

    let mut chain = ChainedPromptProvider::new();
    if !augmenter.is_empty() {
        chain.push(Arc::new(augmenter));
    }
    chain.push(Arc::new(facts_provider));

    let mut builder = Agent::builder()
        .with_llm(provider)
        .with_model(model)
        .with_tools(tools)
        .with_fact_store(fact_store)
        .with_workspace(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if !chain.is_empty() {
        let provider_arc: Arc<dyn PromptProvider> = Arc::new(chain);
        builder = builder.with_prompt_provider(provider_arc);
    }
    builder.build().map_err(|e| anyhow!("agent builder: {e}"))
}

fn build_provider() -> Result<(Arc<dyn LlmProvider>, String)> {
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let model = std::env::var("AGENT_BOT_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        let provider: Arc<dyn LlmProvider> = Arc::new(
            OpenAiProvider::new(OpenAiConfig::openai(key))
                .map_err(|e| anyhow!("provider init: {e}"))?,
        );
        return Ok((provider, model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let model = std::env::var("AGENT_BOT_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-6".into());
        let provider: Arc<dyn LlmProvider> = Arc::new(
            AnthropicProvider::new(AnthropicConfig::new(key))
                .map_err(|e| anyhow!("provider init: {e}"))?,
        );
        return Ok((provider, model));
    }
    if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
        let model = std::env::var("AGENT_BOT_MODEL").unwrap_or_else(|_| "deepseek-chat".into());
        let provider: Arc<dyn LlmProvider> = Arc::new(
            OpenAiProvider::new(OpenAiConfig::deepseek(key))
                .map_err(|e| anyhow!("provider init: {e}"))?,
        );
        return Ok((provider, model));
    }
    Err(anyhow!(
        "agent-bot: set one of OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY"
    ))
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
