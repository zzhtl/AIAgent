use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};

use agent_core::evolution::{CandidateKind, CandidateQueue};
use agent_core::{
    Agent, AgentEvent, ChainedPromptProvider, FactId, FactKind, FactStore, LlmProvider, Message,
    NewFact, PromptProvider, SessionId, SessionStore, StopReason, TokenUsage, ToolRegistry,
    UserInput,
};
use agent_evolution::Reflector;
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_memory::{FactsPromptProvider, MarkdownFactStore, SqliteSessionStore};
use agent_skills::{Augmenter, RuleSet, SkillRegistry};

#[derive(Parser)]
#[command(name = "agent", version, about = "AI Agent CLI", long_about = None)]
struct Cli {
    /// Provider id: `openai`, `deepseek`, or `claude` / `anthropic`.
    #[arg(long, global = true, default_value = "openai")]
    provider: String,

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
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let mut cli = Cli::parse();

    match cli.command.take() {
        Some(Command::Chat) => cmd_chat(&cli).await,
        Some(Command::Run { prompt }) => cmd_run(&cli, prompt).await,
        Some(Command::Sessions { limit }) => cmd_sessions(&cli, limit).await,
        Some(Command::Resume { session_id }) => cmd_resume(&cli, session_id).await,
        Some(Command::Skills) => cmd_skills(&cli),
        Some(Command::Memory { action }) => cmd_memory(&cli, action).await,
        Some(Command::Evolution { action }) => cmd_evolution(&cli, action).await,
        Some(Command::Config) => {
            println!("[config] not implemented yet");
            Ok(())
        }
        None => {
            println!("agent — AI Agent runtime");
            println!("run `agent --help` to see available commands");
            Ok(())
        }
    }
}

async fn cmd_run(cli: &Cli, prompt: String) -> Result<()> {
    let bundle = build_bundle(cli)?;
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

async fn cmd_chat(cli: &Cli) -> Result<()> {
    let bundle = build_bundle(cli)?;
    let sid = bundle
        .session_store
        .create_session(None)
        .await
        .map_err(|e| anyhow!("create_session: {e}"))?;

    println!("Connected to {} ({}). Session: {}", cli.provider, bundle.model, sid);
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

async fn cmd_resume(cli: &Cli, session_id: String) -> Result<()> {
    let bundle = build_bundle(cli)?;
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
        cli.provider,
        bundle.model,
    );

    let history = interactive_loop(&bundle, &sid, history, false).await?;
    println!();
    maybe_reflect(&bundle, &history).await;
    print_session_summary(&bundle.session_store, &sid, &bundle.model).await;
    Ok(())
}

async fn cmd_sessions(cli: &Cli, limit: usize) -> Result<()> {
    let store = open_session_store(cli).await?;
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

fn cmd_skills(cli: &Cli) -> Result<()> {
    let config_dir = resolve_config_dir(cli);
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

async fn cmd_memory(cli: &Cli, action: MemoryCmd) -> Result<()> {
    let fact_store = open_fact_store(cli);
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
    }
    Ok(())
}

async fn cmd_evolution(cli: &Cli, action: EvolutionCmd) -> Result<()> {
    let config_dir = resolve_config_dir(cli);
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
                if reason == StopReason::MaxTokens {
                    eprintln!("\n[note] response may be truncated (max_tokens / max_steps)");
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
    evolve: bool,
    model: String,
}

fn build_bundle(cli: &Cli) -> Result<AgentBundle> {
    let (provider, model) = build_provider(cli)?;
    let mut tools = ToolRegistry::new();
    if !cli.no_tools {
        agent_tools::register_builtins(&mut tools);
        agent_tools::register_memory_tools(&mut tools);
        agent_tools::register_evolution_tools(&mut tools);
    }

    let config_dir = resolve_config_dir(cli);
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

    let mut builder = Agent::builder()
        .with_llm(provider.clone())
        .with_model(model.clone())
        .with_tools(tools)
        .with_fact_store(fact_store.clone())
        .with_candidate_queue(candidate_queue)
        .with_workspace(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if !chain.is_empty() {
        let provider_arc: Arc<dyn PromptProvider> = Arc::new(chain);
        builder = builder.with_prompt_provider(provider_arc);
    }
    let agent = builder.build().map_err(|e| anyhow!("agent builder: {e}"))?;

    let reflector = if cli.evolve {
        Some(Reflector::new(provider, model.clone(), fact_store.clone()))
    } else {
        None
    };

    // We need a sync handle to the session store as well; build it here so
    // the bundle owns it.
    let session_store = futures::executor::block_on(open_session_store(cli))?;

    Ok(AgentBundle {
        agent,
        session_store,
        reflector,
        evolve: cli.evolve,
        model,
    })
}

fn build_provider(cli: &Cli) -> Result<(Arc<dyn LlmProvider>, String)> {
    let provider_id = cli.provider.to_ascii_lowercase();
    match provider_id.as_str() {
        "openai" => {
            let key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;
            let model = cli.model.clone().unwrap_or_else(|| "gpt-4o-mini".into());
            let provider: Arc<dyn LlmProvider> = Arc::new(
                OpenAiProvider::new(OpenAiConfig::openai(key))
                    .map_err(|e| anyhow!("provider init: {e}"))?,
            );
            Ok((provider, model))
        }
        "deepseek" => {
            let key = std::env::var("DEEPSEEK_API_KEY").context("DEEPSEEK_API_KEY is not set")?;
            let model = cli.model.clone().unwrap_or_else(|| "deepseek-chat".into());
            let provider: Arc<dyn LlmProvider> = Arc::new(
                OpenAiProvider::new(OpenAiConfig::deepseek(key))
                    .map_err(|e| anyhow!("provider init: {e}"))?,
            );
            Ok((provider, model))
        }
        "claude" | "anthropic" => {
            let key = std::env::var("ANTHROPIC_API_KEY").context("ANTHROPIC_API_KEY is not set")?;
            let model = cli
                .model
                .clone()
                .unwrap_or_else(|| "claude-sonnet-4-6".into());
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

async fn open_session_store(cli: &Cli) -> Result<Arc<dyn SessionStore>> {
    let config_dir = resolve_config_dir(cli);
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;
    let db_path = config_dir.join("sessions.db");
    let store = SqliteSessionStore::open(&db_path)
        .await
        .map_err(|e| anyhow!("open sessions.db: {e}"))?;
    Ok(Arc::new(store))
}

fn open_fact_store(cli: &Cli) -> Arc<dyn FactStore> {
    let config_dir = resolve_config_dir(cli);
    Arc::new(MarkdownFactStore::open(config_dir.join("memory")))
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

fn resolve_config_dir(cli: &Cli) -> PathBuf {
    if let Some(p) = cli.config_dir.as_ref() {
        return p.clone();
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

fn title_from(input: &str) -> String {
    let trimmed = input.trim();
    let first_line = trimmed.lines().next().unwrap_or("");
    truncate(first_line, 60)
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars).collect();
    format!("{head}…")
}

fn first_line_truncated(body: &str, max_chars: usize) -> String {
    let line = body.lines().next().unwrap_or("").trim();
    if line.chars().count() <= max_chars {
        line.to_string()
    } else {
        let head: String = line.chars().take(max_chars).collect();
        format!("{head}…")
    }
}

fn fmt_unix(secs: i64) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    let when = UNIX_EPOCH + Duration::from_secs(secs.max(0) as u64);
    match when.duration_since(UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            let (year, month, day, hour, minute, second) = gmtime_components(secs);
            format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}")
        }
        Err(_) => "—".into(),
    }
}

fn gmtime_components(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    let second = (secs % 60) as u32;
    let total_minutes = secs / 60;
    let minute = (total_minutes % 60) as u32;
    let total_hours = total_minutes / 60;
    let hour = (total_hours % 24) as u32;
    let mut days = total_hours / 24;

    let mut year: i32 = 1970;
    loop {
        let ydays = if is_leap(year) { 366 } else { 365 };
        if days >= ydays {
            days -= ydays;
            year += 1;
        } else {
            break;
        }
    }
    let months: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    for (i, &len) in months.iter().enumerate() {
        let len = if i == 1 && is_leap(year) { 29 } else { len };
        if days < len as u64 {
            month = i as u32 + 1;
            break;
        }
        days -= len as u64;
    }
    let day = days as u32 + 1;
    (year, month, day, hour, minute, second)
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn compact_json(v: &serde_json::Value) -> String {
    let s = v.to_string();
    if s.len() > 120 {
        format!("{}…", &s[..120])
    } else {
        s
    }
}

fn summarize_result(text: &str) -> String {
    let first_line = text.lines().next().unwrap_or("");
    if first_line.len() > 200 {
        format!("{}…", &first_line[..200])
    } else {
        first_line.to_string()
    }
}
