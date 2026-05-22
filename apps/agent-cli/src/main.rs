use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};

use agent_config::AgentConfig;
use agent_core::agent::RunConfig;
use agent_core::evolution::{CandidateKind, CandidateQueue};
use agent_core::{
    Agent, AgentEvent, ChainedPromptProvider, FactId, FactKind, FactStore, LlmProvider, Message,
    NewFact, PromptProvider, SessionId, SessionStore, StopReason, TokenUsage, ToolRegistry,
    UserInput,
};
use agent_evolution::{Reflector, Summariser};
use agent_core::memory::{EmbeddingProvider, VectorStore};
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_llm::providers::openai_embeddings::OpenAiEmbeddingProvider;
use agent_memory::{
    FactsPromptProvider, MarkdownFactStore, SimpleVectorStore, SqliteSessionStore,
    VectorRecallPromptProvider,
};
use agent_skills::{Augmenter, RuleSet, SkillRegistry};

#[derive(Parser)]
#[command(name = "agent", version, about = "AI Agent CLI", long_about = None)]
struct Cli {
    /// Provider id: `openai`, `deepseek`, or `claude` / `anthropic`.
    /// Falls back to the value in `config.toml` (default `openai`).
    #[arg(long, global = true)]
    provider: Option<String>,

    /// Model id. Defaults to a provider-specific value when omitted.
    #[arg(long, global = true)]
    model: Option<String>,

    /// Disable built-in tools (text-only mode).
    #[arg(long, global = true)]
    no_tools: bool,

    /// Override the config directory (default: `~/.config/agent`).
    #[arg(long, global = true, env = "AGENT_CONFIG_DIR")]
    config_dir: Option<PathBuf>,

    /// After each run, generate a reflection note and persist it to memory.
    /// Adds one extra LLM call per turn.
    #[arg(long, global = true)]
    evolve: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

/// Effective settings after merging CLI flags on top of the layered config.
struct EffectiveCli {
    provider: String,
    model: Option<String>,
    no_tools: bool,
    evolve: bool,
    config_dir: PathBuf,
    config: AgentConfig,
}

impl EffectiveCli {
    fn from(cli: &Cli) -> Result<Self> {
        let config = AgentConfig::load(cli.config_dir.as_deref())
            .map_err(|e| anyhow!("config: {e}"))?;
        let provider = cli
            .provider
            .clone()
            .unwrap_or_else(|| config.provider.clone());
        let model = cli.model.clone().or_else(|| config.model.clone());
        let evolve = cli.evolve || config.evolve;
        let no_tools = cli.no_tools || config.no_tools;
        let config_dir = cli
            .config_dir
            .clone()
            .unwrap_or_else(|| config.config_dir());
        Ok(Self { provider, model, no_tools, evolve, config_dir, config })
    }
}

#[derive(Subcommand)]
enum Command {
    /// Start an interactive REPL chat.
    Chat,
    /// Run a single prompt and exit.
    Run {
        /// The user prompt.
        prompt: String,
    },
    /// List stored sessions.
    Sessions {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Resume a previous session by id.
    Resume { session_id: String },
    /// List loaded skills and rules.
    Skills,
    /// Long-term memory management.
    Memory {
        #[command(subcommand)]
        action: MemoryCmd,
    },
    /// Review and apply self-evolution candidates proposed by the agent.
    Evolution {
        #[command(subcommand)]
        action: EvolutionCmd,
    },
    /// Print the resolved configuration (not implemented in stage 6).
    Config,
}

#[derive(Subcommand)]
enum EvolutionCmd {
    /// Show pending rule / skill candidates.
    Review,
    /// Accept a candidate and write it to `rules/` or `skills/`.
    Apply { id: String },
    /// Drop a candidate without applying.
    Reject { id: String },
}

#[derive(Subcommand)]
enum MemoryCmd {
    /// List all stored facts.
    List {
        /// Filter by kind (`preference`, `project`, `reflection`, `note`).
        #[arg(long)]
        kind: Option<String>,
    },
    /// Substring search over fact name / body.
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Show the full body of one fact.
    Show { id: String },
    /// Delete one fact by id.
    Forget { id: String },
    /// Add a fact directly from the CLI.
    Add {
        name: String,
        body: String,
        #[arg(long, default_value = "note")]
        kind: String,
    },
    /// Re-build the vector index by embedding every fact and upserting it
    /// into the SQLite `vectors` table. Requires `OPENAI_API_KEY`.
    Index,
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let mut cli = Cli::parse();

    let cmd = cli.command.take();
    let eff = EffectiveCli::from(&cli)?;
    match cmd {
        Some(Command::Chat) => cmd_chat(&eff).await,
        Some(Command::Run { prompt }) => cmd_run(&eff, prompt).await,
        Some(Command::Sessions { limit }) => cmd_sessions(&eff, limit).await,
        Some(Command::Resume { session_id }) => cmd_resume(&eff, session_id).await,
        Some(Command::Skills) => cmd_skills(&eff),
        Some(Command::Memory { action }) => cmd_memory(&eff, action).await,
        Some(Command::Evolution { action }) => cmd_evolution(&eff, action).await,
        Some(Command::Config) => cmd_config(&eff),
        None => {
            println!("agent — AI Agent runtime");
            println!("run `agent --help` to see available commands");
            Ok(())
        }
    }
}

fn cmd_config(eff: &EffectiveCli) -> Result<()> {
    println!("provider:    {}", eff.provider);
    println!(
        "model:       {}",
        eff.model.as_deref().unwrap_or("<provider default>")
    );
    println!("config_dir:  {}", eff.config_dir.display());
    println!("no_tools:    {}", eff.no_tools);
    println!("evolve:      {}", eff.evolve);
    println!();
    println!("[loop]");
    println!("  max_steps:   {}", eff.config.agent.max_steps);
    println!(
        "  max_tokens:  {}",
        eff.config
            .agent
            .max_tokens
            .map(|v| v.to_string())
            .unwrap_or_else(|| "<unset>".into())
    );
    println!(
        "  temperature: {}",
        eff.config
            .agent
            .temperature
            .map(|v| v.to_string())
            .unwrap_or_else(|| "<unset>".into())
    );
    println!();
    println!("[permissions]");
    println!("  allow_read:       {}", eff.config.permissions.allow_read);
    println!("  allow_write:      {}", eff.config.permissions.allow_write);
    println!("  allow_shell:      {}", eff.config.permissions.allow_shell);
    println!("  allow_network:    {}", eff.config.permissions.allow_network);
    println!(
        "  max_runtime_secs: {}",
        eff.config.permissions.max_runtime_secs
    );
    Ok(())
}

async fn cmd_run(eff: &EffectiveCli, prompt: String) -> Result<()> {
    let bundle = build_bundle(eff).await?;
    let title = title_from(&prompt);
    let sid = bundle
        .session_store
        .create_session(Some(&title))
        .await
        .map_err(|e| anyhow!("create_session: {e}"))?;

    let history = drive(&bundle, &sid, Vec::new(), UserInput::new(prompt)).await?;
    println!();
    maybe_reflect(&bundle, &history).await;
    print_session_summary(&bundle.session_store, &sid, &bundle.model).await;
    Ok(())
}

async fn cmd_chat(eff: &EffectiveCli) -> Result<()> {
    let bundle = build_bundle(eff).await?;
    let sid = bundle
        .session_store
        .create_session(None)
        .await
        .map_err(|e| anyhow!("create_session: {e}"))?;

    println!("Connected to {} ({}). Session: {}", eff.provider, bundle.model, sid);
    if bundle.evolve {
        println!("Self-reflection enabled (--evolve).");
    }
    println!("Type your message and press Enter. /quit or Ctrl-D to exit.");

    let history = interactive_loop(&bundle, &sid, Vec::new(), true).await?;
    println!();
    maybe_reflect(&bundle, &history).await;
    print_session_summary(&bundle.session_store, &sid, &bundle.model).await;
    Ok(())
}

async fn cmd_resume(eff: &EffectiveCli, session_id: String) -> Result<()> {
    let bundle = build_bundle(eff).await?;
    let sid = SessionId::from(session_id.as_str());

    let history = bundle
        .session_store
        .load_messages(&sid)
        .await
        .map_err(|e| anyhow!("load_messages: {e}"))?;
    println!(
        "Resumed {} ({} messages, {}). Provider: {} ({}).",
        sid,
        history.len(),
        if history.is_empty() { "empty" } else { "ready" },
        eff.provider,
        bundle.model,
    );

    let history = interactive_loop(&bundle, &sid, history, false).await?;
    println!();
    maybe_reflect(&bundle, &history).await;
    print_session_summary(&bundle.session_store, &sid, &bundle.model).await;
    Ok(())
}

async fn cmd_sessions(eff: &EffectiveCli, limit: usize) -> Result<()> {
    let store = open_session_store(eff).await?;
    let sessions = store
        .list_sessions(limit)
        .await
        .map_err(|e| anyhow!("list_sessions: {e}"))?;
    if sessions.is_empty() {
        println!("No sessions yet. Run `agent chat` to start one.");
        return Ok(());
    }
    println!("{:<36}  {:>4}  {:<19}  title", "id", "msgs", "updated_at");
    for s in sessions {
        let title = s.title.as_deref().unwrap_or("");
        println!(
            "{:<36}  {:>4}  {}  {}",
            s.id,
            s.message_count,
            fmt_unix(s.updated_at),
            truncate(title, 60),
        );
    }
    Ok(())
}

fn cmd_skills(eff: &EffectiveCli) -> Result<()> {
    let config_dir = eff.config_dir.clone();
    let skills_dir = config_dir.join("skills");
    let rules_dir = config_dir.join("rules");

    let skills = SkillRegistry::load_dir(&skills_dir).map_err(|e| anyhow!("skills: {e}"))?;
    let rules = RuleSet::load_dir(&rules_dir).map_err(|e| anyhow!("rules: {e}"))?;

    println!("Config dir: {}", config_dir.display());
    println!();
    if skills.is_empty() {
        println!("No skills loaded.");
        println!("  Place markdown files under {}.", skills_dir.display());
    } else {
        println!("Loaded skills ({}):", skills.len());
        for s in skills.all() {
            println!("  - {}", s.name);
            if !s.description.is_empty() {
                println!("      desc: {}", s.description);
            }
            if !s.triggers.is_empty() {
                println!("      triggers: {:?}", s.triggers);
            }
            if !s.tools_allowed.is_empty() {
                println!("      tools_allowed: {:?}", s.tools_allowed);
            }
        }
    }
    println!();
    if rules.is_empty() {
        println!("No rules loaded.");
        println!("  Place markdown files under {}.", rules_dir.display());
    } else {
        println!("Loaded rules ({}):", rules.len());
        for r in rules.all() {
            println!("  - {}", r.name);
        }
    }
    Ok(())
}

async fn cmd_memory(eff: &EffectiveCli, action: MemoryCmd) -> Result<()> {
    let fact_store = open_fact_store(eff);
    match action {
        MemoryCmd::List { kind } => {
            let kind = kind.as_deref().and_then(parse_kind);
            let facts = fact_store
                .list(kind)
                .await
                .map_err(|e| anyhow!("list: {e}"))?;
            if facts.is_empty() {
                println!("No facts stored.");
                return Ok(());
            }
            println!("{} fact(s):", facts.len());
            for f in facts {
                let k = format_kind(f.kind);
                println!("  - [{k}] {} (id: {})", f.name, f.id);
                let summary = first_line_truncated(&f.body, 120);
                if !summary.is_empty() {
                    println!("      {summary}");
                }
            }
        }
        MemoryCmd::Search { query, limit } => {
            let hits = fact_store
                .search(&query, limit)
                .await
                .map_err(|e| anyhow!("search: {e}"))?;
            if hits.is_empty() {
                println!("No facts match `{query}`.");
                return Ok(());
            }
            for f in hits {
                let k = format_kind(f.kind);
                println!("- [{k}] {} (id: {})", f.name, f.id);
                println!("    {}", first_line_truncated(&f.body, 200));
            }
        }
        MemoryCmd::Show { id } => {
            let fact = fact_store
                .get(&FactId::from(id.as_str()))
                .await
                .map_err(|e| anyhow!("get: {e}"))?;
            println!("# {} ({})", fact.name, fact.id);
            println!("kind: {}", format_kind(fact.kind));
            if !fact.tags.is_empty() {
                println!("tags: {:?}", fact.tags);
            }
            println!();
            println!("{}", fact.body);
        }
        MemoryCmd::Forget { id } => {
            fact_store
                .delete(&FactId::from(id.as_str()))
                .await
                .map_err(|e| anyhow!("forget: {e}"))?;
            println!("forgot {id}");
        }
        MemoryCmd::Add { name, body, kind } => {
            let kind = parse_kind(&kind).unwrap_or(FactKind::Note);
            let id = fact_store
                .save(NewFact::new(name.clone(), body).with_kind(kind))
                .await
                .map_err(|e| anyhow!("save: {e}"))?;
            println!("saved {name} (id: {id})");
        }
        MemoryCmd::Index => cmd_memory_index(eff, fact_store).await?,
    }
    Ok(())
}

async fn cmd_memory_index(eff: &EffectiveCli, fact_store: Arc<dyn FactStore>) -> Result<()> {
    let key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY is required for embeddings")?;
    let embedder = OpenAiEmbeddingProvider::new(key)
        .map_err(|e| anyhow!("embedder init: {e}"))?;

    let store = open_session_store_concrete(eff).await?;
    let vectors = SimpleVectorStore::from_session_store(&store);

    let facts = fact_store
        .list(None)
        .await
        .map_err(|e| anyhow!("list facts: {e}"))?;
    if facts.is_empty() {
        println!("No facts to index.");
        return Ok(());
    }

    println!("Indexing {} fact(s) with model {} …", facts.len(), embedder.model());
    // Embed in small batches to stay friendly to the API.
    const BATCH: usize = 16;
    let mut indexed = 0usize;
    for chunk in facts.chunks(BATCH) {
        let texts: Vec<String> = chunk
            .iter()
            .map(|f| format!("{}\n\n{}", f.name, f.body))
            .collect();
        let embeddings = embedder
            .embed(&texts)
            .await
            .map_err(|e| anyhow!("embed: {e}"))?;
        for (fact, emb) in chunk.iter().zip(embeddings) {
            let key = format!("fact:{}", fact.id);
            let metadata = serde_json::json!({
                "fact_id": fact.id.as_str(),
                "name": fact.name,
                "kind": format_kind(fact.kind),
            });
            vectors
                .upsert(&key, &texts[indexed % BATCH], emb, metadata)
                .await
                .map_err(|e| anyhow!("upsert: {e}"))?;
            indexed += 1;
        }
    }
    println!("Indexed {indexed} fact(s).");
    Ok(())
}

async fn open_session_store_concrete(eff: &EffectiveCli) -> Result<SqliteSessionStore> {
    let config_dir = eff.config_dir.clone();
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;
    let db_path = config_dir.join("sessions.db");
    SqliteSessionStore::open(&db_path)
        .await
        .map_err(|e| anyhow!("open sessions.db: {e}"))
}

async fn cmd_evolution(eff: &EffectiveCli, action: EvolutionCmd) -> Result<()> {
    let config_dir = eff.config_dir.clone();
    let queue = open_candidate_queue(&config_dir);
    match action {
        EvolutionCmd::Review => {
            let all = queue.list().await.map_err(|e| anyhow!("read queue: {e}"))?;
            if all.is_empty() {
                println!("Candidate queue is empty.");
                return Ok(());
            }
            println!("{} pending candidate(s):", all.len());
            for c in all {
                let kind = match c.kind {
                    CandidateKind::Rule => "rule",
                    CandidateKind::Skill => "skill",
                };
                println!("  - [{kind}] {} (id: {})", c.name, c.id);
                println!("      rationale: {}", first_line_truncated(&c.rationale, 200));
            }
        }
        EvolutionCmd::Apply { id } => {
            let popped = queue
                .remove(&id)
                .await
                .map_err(|e| anyhow!("remove: {e}"))?;
            let Some(c) = popped else {
                return Err(anyhow!("no candidate with id `{id}`"));
            };
            let (subdir, kind_label) = match c.kind {
                CandidateKind::Rule => ("rules", "rule"),
                CandidateKind::Skill => ("skills", "skill"),
            };
            let dest_dir = config_dir.join(subdir);
            std::fs::create_dir_all(&dest_dir)
                .with_context(|| format!("create_dir_all {}", dest_dir.display()))?;
            let filename = safe_filename(&c.name);
            if filename.is_empty() {
                return Err(anyhow!("candidate name `{}` has no usable filename", c.name));
            }
            let mut path = dest_dir.join(format!("{filename}.md"));
            let mut suffix = 1;
            while path.exists() {
                path = dest_dir.join(format!("{filename}-{suffix}.md"));
                suffix += 1;
            }
            std::fs::write(&path, &c.body)
                .with_context(|| format!("write {}", path.display()))?;
            println!("applied {kind_label} `{}` → {}", c.name, path.display());
        }
        EvolutionCmd::Reject { id } => {
            let popped = queue
                .remove(&id)
                .await
                .map_err(|e| anyhow!("remove: {e}"))?;
            match popped {
                Some(c) => println!("rejected {} (id: {id})", c.name),
                None => return Err(anyhow!("no candidate with id `{id}`")),
            }
        }
    }
    Ok(())
}

fn open_candidate_queue(config_dir: &Path) -> CandidateQueue {
    CandidateQueue::open(config_dir.join("evolution").join("queue.json"))
}

fn safe_filename(name: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_dash = false;
        } else if c.is_alphanumeric() {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

async fn interactive_loop(
    bundle: &AgentBundle,
    sid: &SessionId,
    mut history: Vec<Message>,
    rename_on_first_turn: bool,
) -> Result<Vec<Message>> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut first_turn = rename_on_first_turn;

    loop {
        print!("\nyou> ");
        io::stdout().flush().ok();

        let Some(line) = reader.next_line().await? else {
            println!();
            break;
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }

        if first_turn {
            let title = title_from(line);
            let _ = bundle.session_store.rename_session(sid, &title).await;
            first_turn = false;
        }

        print!("agent> ");
        io::stdout().flush().ok();

        history = drive(bundle, sid, history, UserInput::new(line.to_string())).await?;
        println!();
    }
    Ok(history)
}

async fn drive(
    bundle: &AgentBundle,
    sid: &SessionId,
    mut history: Vec<Message>,
    input: UserInput,
) -> Result<Vec<Message>> {
    history = maybe_compact_history(bundle, sid, history).await;
    let mut stream = bundle.agent.run(sid.clone(), history.clone(), input);
    let mut pending_usages: Vec<(String, TokenUsage)> = Vec::new();

    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta { delta } => {
                print!("{delta}");
                io::stdout().flush().ok();
            }
            AgentEvent::ToolCallStart { call } => {
                let args = compact_json(&call.input);
                println!("\n  ⟢ tool[{}]({args})", call.name);
            }
            AgentEvent::ToolCallResult { result } => {
                let summary = summarize_result(&result.output);
                let tag = if result.is_error { "error" } else { "ok" };
                println!("  ⟢ {tag}: {summary}");
            }
            AgentEvent::UsageReport { usage, model } => {
                tracing::debug!(%model, ?usage, "usage");
                pending_usages.push((model, usage));
            }
            AgentEvent::Warning { message } => eprintln!("\n[warning] {message}"),
            AgentEvent::Done { reason, transcript_delta } => {
                if !transcript_delta.is_empty() {
                    if let Err(e) = bundle.session_store.append_messages(sid, &transcript_delta).await {
                        eprintln!("\n[warning] failed to persist messages: {e}");
                    }
                }
                history.extend(transcript_delta);
                match reason {
                    StopReason::MaxTokens => {
                        eprintln!("\n[note] response truncated by model max_tokens");
                    }
                    StopReason::MaxSteps => {
                        eprintln!(
                            "\n[note] reached the agent loop cap (max_steps); some work may be incomplete"
                        );
                    }
                    _ => {}
                }
                break;
            }
        }
    }

    for (m, usage) in pending_usages {
        let cost = agent_telemetry::estimate_cost_usd(&m, usage);
        let _ = bundle.session_store.record_usage(sid, &m, usage, cost).await;
    }
    Ok(history)
}

/// If the in-memory transcript has grown past `summary_threshold`, ask the
/// Summariser to compress the early portion and replace it with a single
/// system-prompt summary message. Best-effort: any failure leaves the
/// history untouched.
async fn maybe_compact_history(
    bundle: &AgentBundle,
    sid: &SessionId,
    history: Vec<Message>,
) -> Vec<Message> {
    let threshold = bundle.summary_threshold;
    if threshold == 0 || history.len() <= threshold {
        return history;
    }
    let Some(summariser) = bundle.summariser.as_ref() else {
        return history;
    };
    let keep_tail = bundle.summary_keep_tail.min(history.len());
    let split_at = history.len().saturating_sub(keep_tail);
    if split_at == 0 {
        return history;
    }
    let (head, tail) = history.split_at(split_at);
    eprintln!(
        "[compacting {} earlier messages into a summary …]",
        head.len()
    );
    let Some(summary) = summariser.summarise(head).await else {
        return [head, tail].concat();
    };
    if let Err(e) = bundle.session_store.record_summary(sid, &summary, None).await {
        tracing::debug!(error = %e, "record_summary failed");
    }
    let mut compact = Vec::with_capacity(tail.len() + 1);
    compact.push(Message::system(format!("# Earlier conversation summary\n\n{summary}")));
    compact.extend(tail.iter().cloned());
    compact
}

async fn maybe_reflect(bundle: &AgentBundle, history: &[Message]) {
    if !bundle.evolve {
        return;
    }
    if let Some(reflector) = bundle.reflector.as_ref() {
        eprintln!("[reflecting...]");
        match reflector.reflect(history).await {
            Some(id) => eprintln!("[reflection saved as {id}]"),
            None => eprintln!("[reflection skipped or failed]"),
        }
    }
}

async fn print_session_summary(store: &Arc<dyn SessionStore>, sid: &SessionId, model: &str) {
    if let Ok(summary) = store.session_usage(sid).await {
        if summary.total_tokens() > 0 {
            eprintln!(
                "[session {}] model={} tokens={}+{} (cached {}) ≈ ${:.4}",
                sid,
                model,
                summary.prompt_tokens,
                summary.completion_tokens,
                summary.cached_tokens,
                summary.cost_estimate_usd,
            );
        } else {
            eprintln!("[session {}] no LLM calls recorded", sid);
        }
    }
}

struct AgentBundle {
    agent: Agent,
    session_store: Arc<dyn SessionStore>,
    reflector: Option<Reflector>,
    summariser: Option<Summariser>,
    summary_threshold: usize,
    summary_keep_tail: usize,
    evolve: bool,
    model: String,
}

async fn build_bundle(eff: &EffectiveCli) -> Result<AgentBundle> {
    let (provider, model) = build_provider(eff)?;
    let mut tools = ToolRegistry::new();
    if !eff.no_tools {
        agent_tools::register_builtins(&mut tools);
        agent_tools::register_memory_tools(&mut tools);
        agent_tools::register_evolution_tools(&mut tools);
    }

    let config_dir = eff.config_dir.clone();
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

    let candidate_queue = open_candidate_queue(&config_dir);

    let run_config = RunConfig {
        max_steps: eff.config.agent.max_steps,
        temperature: eff.config.agent.temperature,
        max_tokens: eff.config.agent.max_tokens,
        permissions: eff.config.permissions.to_runtime(),
    };

    let mut builder = Agent::builder()
        .with_llm(provider.clone())
        .with_model(model.clone())
        .with_tools(tools)
        .with_fact_store(fact_store.clone())
        .with_candidate_queue(candidate_queue)
        .with_config(run_config)
        .with_workspace(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if !chain.is_empty() {
        let provider_arc: Arc<dyn PromptProvider> = Arc::new(chain);
        builder = builder.with_prompt_provider(provider_arc);
    }
    let agent = builder.build().map_err(|e| anyhow!("agent builder: {e}"))?;

    let reflector = if eff.evolve {
        Some(Reflector::new(provider.clone(), model.clone(), fact_store.clone()))
    } else {
        None
    };

    let summary_threshold = eff.config.agent.summary_threshold;
    let summary_keep_tail = eff.config.agent.summary_keep_tail.max(1);
    let summariser = if summary_threshold > 0 {
        Some(Summariser::new(provider, model.clone()))
    } else {
        None
    };

    let session_store = open_session_store(eff).await?;

    Ok(AgentBundle {
        agent,
        session_store,
        reflector,
        summariser,
        summary_threshold,
        summary_keep_tail,
        evolve: eff.evolve,
        model,
    })
}

fn build_provider(eff: &EffectiveCli) -> Result<(Arc<dyn LlmProvider>, String)> {
    let provider_id = eff.provider.to_ascii_lowercase();
    match provider_id.as_str() {
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;
            let model = eff.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
            let provider: Arc<dyn LlmProvider> = Arc::new(
                OpenAiProvider::new(OpenAiConfig::openai(key))
                    .map_err(|e| anyhow!("provider init: {e}"))?,
            );
            Ok((provider, model))
        }
        "deepseek" => {
            let key = std::env::var("DEEPSEEK_API_KEY").context("DEEPSEEK_API_KEY is not set")?;
            let model = eff.model.clone().unwrap_or_else(|| "deepseek-chat".into());
            let provider: Arc<dyn LlmProvider> = Arc::new(
                OpenAiProvider::new(OpenAiConfig::deepseek(key))
                    .map_err(|e| anyhow!("provider init: {e}"))?,
            );
            Ok((provider, model))
        }
        "claude" | "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY is not set")?;
            let model = eff
                .model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-5".into());
            let provider: Arc<dyn LlmProvider> = Arc::new(
                AnthropicProvider::new(AnthropicConfig::new(key))
                    .map_err(|e| anyhow!("provider init: {e}"))?,
            );
            Ok((provider, model))
        }
        other => Err(anyhow!(
            "unsupported provider `{other}` (expected `openai`, `deepseek`, or `claude`)"
        )),
    }
}

async fn open_session_store(eff: &EffectiveCli) -> Result<Arc<dyn SessionStore>> {
    let config_dir = eff.config_dir.clone();
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;
    let db_path = config_dir.join("sessions.db");
    let store = SqliteSessionStore::open(&db_path)
        .await
        .map_err(|e| anyhow!("open sessions.db: {e}"))?;
    Ok(Arc::new(store))
}

fn open_fact_store(eff: &EffectiveCli) -> Arc<dyn FactStore> {
    Arc::new(MarkdownFactStore::open(eff.config_dir.join("memory")))
}

fn parse_kind(s: &str) -> Option<FactKind> {
    match s.to_ascii_lowercase().as_str() {
        "preference" | "pref" => Some(FactKind::Preference),
        "project" | "proj" => Some(FactKind::Project),
        "reflection" | "ref" => Some(FactKind::Reflection),
        "note" => Some(FactKind::Note),
        _ => None,
    }
}

fn format_kind(k: FactKind) -> &'static str {
    match k {
        FactKind::Preference => "preference",
        FactKind::Project => "project",
        FactKind::Reflection => "reflection",
        FactKind::Note => "note",
    }
}

fn title_from(input: &str) -> String {
    let trimmed = input.trim();
    let first_line = trimmed.lines().next().unwrap_or("");
    truncate(first_line, 60)
}

use agent_core::text::{first_line_truncated, truncate_with_ellipsis as truncate};

fn fmt_unix(secs: i64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp(secs, 0)
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "—".into())
}

fn compact_json(v: &serde_json::Value) -> String {
    truncate(&v.to_string(), 120)
}

fn summarize_result(text: &str) -> String {
    first_line_truncated(text, 200)
}
