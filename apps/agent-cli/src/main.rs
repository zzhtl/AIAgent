use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use futures::StreamExt;
use tokio::io::{AsyncBufReadExt, BufReader};

use agent_config::{AgentConfig, McpServerConfig, SubAgentConfig, ToolPolicyConfig};
use agent_mcp::{McpClient, McpTool};
use agent_tools::policy::{PolicyHook, ToolAccess, ToolPolicy};
use agent_core::agent::RunConfig;
use agent_core::evolution::{CandidateKind, CandidateQueue};
use agent_core::{
    Agent, AgentEvent, ChainedPromptProvider, FactId, FactKind, FactStore, LlmProvider, Message,
    NewFact, PromptProvider, SessionId, SessionStore, StopReason, SubAgentTool, TokenUsage,
    ToolRegistry, UserInput,
};
use agent_evolution::{Extractor, Reflector, Summariser};
use agent_core::memory::{EmbeddingProvider, VectorStore};
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_llm::ProviderRegistry;
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
    /// Scaffold the config directory (config.toml + example skill/rule).
    Init {
        /// Overwrite files that already exist.
        #[arg(long)]
        force: bool,
    },
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
    /// Print the resolved configuration.
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
    /// Mine accumulated reflections for rule/skill candidates (uses the LLM);
    /// proposals land in the same review queue.
    Extract,
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
        Some(Command::Init { force }) => cmd_init(&eff, force),
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
    println!(
        "  summary_threshold: {} (0 = disabled)",
        eff.config.agent.summary_threshold
    );
    println!("  summary_keep_tail: {}", eff.config.agent.summary_keep_tail);
    println!(
        "  vector_recall:     {} (top_k={}, min_score={})",
        eff.config.agent.vector_recall,
        eff.config.agent.vector_recall_top_k,
        eff.config.agent.vector_recall_min_score,
    );
    println!(
        "  max_retries:       {} (base_delay={}ms)",
        eff.config.agent.max_retries, eff.config.agent.retry_base_delay_ms,
    );
    println!(
        "  token_budget:      {}",
        eff.config
            .agent
            .token_budget
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

const CONFIG_TEMPLATE: &str = r#"# AIAgent 配置文件
# 解析顺序（后者覆盖前者）：内置默认 → /etc/agent/config.toml →
# ~/.config/agent/config.toml（本文件）→ ./agent.toml → AGENT_* 环境变量。
# API key 永远不从配置读取，只从环境变量取（见文件末尾）。

# provider: openai | deepseek | claude/anthropic
provider = "openai"
# model：留空用 provider 默认（openai=gpt-4o-mini, deepseek=deepseek-chat,
# claude=claude-sonnet-4-5）。
# model = "gpt-4o-mini"
no_tools = false   # true = 纯文本模式，禁用所有工具
evolve = false     # true = 每轮后自动反思一条记忆（多一次 LLM 调用）

[agent]
max_steps = 12              # 单轮 think→tool 循环步数上限
# max_tokens = 4096         # 每次 LLM 请求的最大输出 token
# temperature = 0.7
summary_threshold = 30      # 历史超过该条数时压缩早期消息；0 关闭
summary_keep_tail = 8       # 压缩时保留最近多少条原文
vector_recall = false       # 语义召回（需 OPENAI_API_KEY + 先跑 `agent memory index`）
vector_recall_top_k = 5
vector_recall_min_score = 0.2
max_retries = 2             # LLM 瞬时错误（网络/限流/5xx）自动重试次数
retry_base_delay_ms = 500   # 重试退避基数，按 base*2^n 增长
# token_budget = 100000     # 单轮累计 token 预算，超出即停止

[permissions]
allow_read = true
allow_write = true
allow_shell = true
allow_network = true
max_runtime_secs = 120      # bash 等长任务的硬超时（秒）

# API key（按所选 provider 设置对应环境变量，不要写进本文件）：
#   openai   → export OPENAI_API_KEY=...
#   deepseek → export DEEPSEEK_API_KEY=...
#   claude   → export ANTHROPIC_API_KEY=...

# 子 agent（可选）：声明后会作为工具暴露给主 agent，由主 agent 自行决定何时委派。
# 每个子 agent 复用主 provider、继承权限，但可独立设置 model / 系统提示 / 步数上限。
# [[subagents]]
# name = "researcher"                      # 工具名（主 agent 用它调用，需唯一）
# description = "深入研究某个主题并给出结论"  # 主 agent 看到的用途说明
# prompt = "你是研究专家，检索并综合信息后给出有依据的结论。"
# model = "gpt-4o"                         # 可选，留空则用主 model
# max_steps = 8                            # 可选，留空则用主 max_steps

# MCP 服务（可选）：启动时通过 stdio 连接，把服务暴露的工具注册进来（名字加
# 前缀 <name>__<tool> 防冲突）。连接失败会告警并跳过，不影响启动。
# [[mcp_servers]]
# name = "fs"                              # 工具名前缀（需唯一）
# command = "npx"                          # 启动 MCP server 的可执行文件
# args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
# [mcp_servers.env]                        # 可选，传给 server 进程的环境变量
# SOME_TOKEN = "..."

# 工具访问策略 / 沙箱（可选）：默认放开所有工具。不信任 / 多租户场景可收紧。
# 仅在确实有限制时才会启用（before_tool hook 拦截），否则零开销。
# [tool_policy]
# default_allow = true                     # 未列出的工具默认是否允许
# deny = ["bash"]                          # 显式拒绝的工具
# allow = []                               # 显式允许（配合 default_allow=false 做白名单）
# bash_allowed_prefixes = ["ls", "cat", "git "]  # 非空时 bash 命令必须以其一开头
"#;

const SKILL_TEMPLATE: &str = r#"---
name: code-review
description: 系统化代码审查流程
triggers:
  - review
  - 代码审查
  - code review
  - 审查代码
tools_allowed:
  - file_read
  - bash
---

# Code Review 工作流

当用户要求审查代码时，按以下步骤执行：

## 步骤

1. **定位目标**：用 `file_read` 读取用户指定的文件（如未指定，先列出当前目录）
2. **结构分析**：识别函数 / 模块边界，标记入口与关键路径
3. **检查清单**：
   - 命名与可读性
   - 错误处理是否完整（特别是边界条件）
   - 资源管理（文件、连接、锁）
   - 并发安全（如涉及）
   - 测试覆盖（是否易测、是否有死代码）
4. **输出格式**：分 high / medium / low 三档列出问题
   - 每条问题给出 `file:line` 锚点（如能确定）
   - 给出一条具体修复建议（不要泛泛而谈）

## 重要

- 不要直接修改文件，只输出建议
- 如需运行编译/测试验证，用 `bash` 工具
"#;

const RULE_TEMPLATE: &str = r#"---
name: coding-style
---

# 全局编码风格

- 默认中文交流，称呼用户 zzhtl
- 代码注释优先跟随仓库现状：英文项目用英文注释
- 写代码追求最小变更，不主动重构无关代码
- 公共 API 必须有 doc 注释
- 错误优先用 `?` 操作符；避免 `unwrap()` 在非测试代码中
- 公开变更前简明说明 what + why
"#;

/// Scaffold the config directory: create the standard subdirectories and drop
/// a commented `config.toml` plus one starter skill / rule so a fresh install
/// has a working baseline. Existing files are skipped unless `--force`.
fn cmd_init(eff: &EffectiveCli, force: bool) -> Result<()> {
    let dir = eff.config_dir.clone();
    for sub in ["", "skills", "rules", "memory", "evolution"] {
        let p = if sub.is_empty() { dir.clone() } else { dir.join(sub) };
        std::fs::create_dir_all(&p).with_context(|| format!("create_dir_all {}", p.display()))?;
    }

    let files = [
        (dir.join("config.toml"), CONFIG_TEMPLATE),
        (dir.join("skills").join("code-review.md"), SKILL_TEMPLATE),
        (dir.join("rules").join("coding-style.md"), RULE_TEMPLATE),
    ];
    let mut wrote = 0usize;
    let mut skipped = 0usize;
    for (path, content) in files {
        if path.exists() && !force {
            println!("  skip (exists): {}", path.display());
            skipped += 1;
            continue;
        }
        std::fs::write(&path, content).with_context(|| format!("write {}", path.display()))?;
        println!("  wrote: {}", path.display());
        wrote += 1;
    }

    println!();
    println!("Initialized config at {}", dir.display());
    println!("  {wrote} file(s) written, {skipped} skipped.");
    if skipped > 0 && !force {
        println!("  Re-run with --force to overwrite skipped files.");
    }
    println!();
    println!("Next steps:");
    println!("  1. 设置 API key（按 provider 选一个）：");
    println!("       export OPENAI_API_KEY=...      # openai");
    println!("       export DEEPSEEK_API_KEY=...    # deepseek");
    println!("       export ANTHROPIC_API_KEY=...   # claude");
    println!("  2. 运行：agent chat");
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
        for ((fact, emb), text) in chunk.iter().zip(embeddings).zip(texts.iter()) {
            let key = format!("fact:{}", fact.id);
            let metadata = serde_json::json!({
                "fact_id": fact.id.as_str(),
                "name": fact.name,
                "kind": format_kind(fact.kind),
            });
            vectors
                .upsert(&key, text, emb, metadata)
                .await
                .map_err(|e| anyhow!("upsert: {e}"))?;
            indexed += 1;
        }
    }
    println!("Indexed {indexed} fact(s).");
    Ok(())
}

/// 构造可选的 `VectorRecallPromptProvider`：仅在 `OPENAI_API_KEY` 可用且
/// 向量表非空时返回 `Some`，否则返回 `None`（语义召回静默跳过，不破坏正常聊天）。
async fn build_vector_recall(
    eff: &EffectiveCli,
) -> Result<Option<Arc<dyn PromptProvider>>> {
    let Ok(key) = std::env::var("OPENAI_API_KEY") else {
        return Ok(None);
    };
    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(
        OpenAiEmbeddingProvider::new(key).map_err(|e| anyhow!("embedder init: {e}"))?,
    );
    let store = open_session_store_concrete(eff).await?;
    let vectors: Arc<dyn VectorStore> = Arc::new(SimpleVectorStore::from_session_store(&store));
    if vectors.is_empty().await.unwrap_or(true) {
        return Ok(None);
    }
    let provider = VectorRecallPromptProvider::new(embedder, vectors)
        .with_top_k(eff.config.agent.vector_recall_top_k)
        .with_min_score(eff.config.agent.vector_recall_min_score);
    Ok(Some(Arc::new(provider)))
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
        EvolutionCmd::Extract => {
            let (provider, model) = build_provider(eff)?;
            let fact_store: Arc<dyn FactStore> =
                Arc::new(MarkdownFactStore::open(config_dir.join("memory")));
            let candidates = Extractor::new(provider, model, fact_store).extract().await;
            if candidates.is_empty() {
                println!(
                    "No candidates extracted (need ≥3 reflections and a recurring pattern)."
                );
                return Ok(());
            }
            let n = candidates.len();
            for c in candidates {
                queue.enqueue(c).await.map_err(|e| anyhow!("enqueue: {e}"))?;
            }
            println!(
                "Extracted {n} candidate(s) → `agent evolution review` to inspect, `apply <id>` to install."
            );
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
    // Ctrl-C during a turn flips this flag; the agent loop observes it at its
    // next checkpoint and ends gracefully with `StopReason::Cancelled`.
    let cancel = Arc::new(AtomicBool::new(false));
    let mut stream =
        bundle
            .agent
            .run_cancellable(sid.clone(), history.clone(), input, cancel.clone());
    let mut pending_usages: Vec<(String, TokenUsage)> = Vec::new();
    let mut cancelling = false;

    loop {
        let event = if cancelling {
            // Already signalled; just drain remaining events to the `Done`.
            match stream.next().await {
                Some(e) => e,
                None => break,
            }
        } else {
            tokio::select! {
                ev = stream.next() => match ev {
                    Some(e) => e,
                    None => break,
                },
                _ = tokio::signal::ctrl_c() => {
                    cancel.store(true, Ordering::Relaxed);
                    cancelling = true;
                    eprintln!("\n[cancelling…]");
                    continue;
                }
            }
        };
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
                    StopReason::BudgetExceeded => {
                        eprintln!(
                            "\n[note] stopped: per-turn token_budget exhausted; some work may be incomplete"
                        );
                    }
                    StopReason::Cancelled => {
                        eprintln!("\n[note] cancelled by user; partial work saved");
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
        // Register declared sub-agents as callable tools. Each reuses the main
        // provider but gets its own model / system prompt; its tool-set is the
        // built-ins only (no nested sub-agents), so the call graph can't recurse.
        for sub in &eff.config.subagents {
            let sub_agent = build_subagent(eff, &provider, &model, sub)?;
            tools.register(Arc::new(SubAgentTool::new(
                sub.name.clone(),
                sub.description.clone(),
                sub_agent,
            )));
        }
        // Connect declared MCP servers and register their tools (namespaced by
        // server name). A server that fails to start is logged and skipped so
        // one bad entry doesn't abort startup.
        for srv in &eff.config.mcp_servers {
            match register_mcp_server(&mut tools, srv).await {
                Ok(n) => tracing::info!("mcp `{}`: registered {n} tool(s)", srv.name),
                Err(e) => eprintln!("[warning] mcp server `{}` skipped: {e}", srv.name),
            }
        }
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

    // 语义召回（可选）：开启后每轮先 embed 用户输入，再从 SQLite 向量表
    // 拉 top-k 注入到 system prompt。需要先跑 `agent memory index`。
    if eff.config.agent.vector_recall {
        match build_vector_recall(eff).await {
            Ok(Some(provider)) => chain.push(provider),
            Ok(None) => {
                eprintln!(
                    "[note] vector_recall enabled but skipped (vector store empty or embedder unavailable; run `agent memory index` and set OPENAI_API_KEY)"
                );
            }
            Err(e) => eprintln!("[warning] vector_recall init failed: {e}"),
        }
    }

    let candidate_queue = open_candidate_queue(&config_dir);

    let run_config = RunConfig {
        max_steps: eff.config.agent.max_steps,
        temperature: eff.config.agent.temperature,
        max_tokens: eff.config.agent.max_tokens,
        permissions: eff.config.permissions.to_runtime(),
        max_retries: eff.config.agent.max_retries,
        retry_base_delay_ms: eff.config.agent.retry_base_delay_ms,
        token_budget: eff.config.agent.token_budget,
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
    // Install the tool-access policy as a before_tool hook, but only when it
    // actually restricts something (zero overhead otherwise).
    let policy = tool_policy_from_config(&eff.config.tool_policy);
    if !policy.is_unrestricted() {
        builder = builder.with_hook(Arc::new(PolicyHook::new(policy)));
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

/// Build one declared sub-agent: reuses the main provider, inherits the main
/// permissions/workspace, but takes its own model / system prompt / step cap.
/// Its tool-set is the built-ins only — no memory, evolution, or nested
/// sub-agent tools — so it stays focused and the call graph cannot recurse.
fn build_subagent(
    eff: &EffectiveCli,
    provider: &Arc<dyn LlmProvider>,
    main_model: &str,
    sub: &SubAgentConfig,
) -> Result<Agent> {
    let mut sub_tools = ToolRegistry::new();
    agent_tools::register_builtins(&mut sub_tools);

    let run_config = RunConfig {
        max_steps: sub.max_steps.unwrap_or(eff.config.agent.max_steps),
        permissions: eff.config.permissions.to_runtime(),
        ..RunConfig::default()
    };

    let mut builder = Agent::builder()
        .with_llm(provider.clone())
        .with_model(sub.model.clone().unwrap_or_else(|| main_model.to_string()))
        .with_tools(sub_tools)
        .with_config(run_config)
        .with_workspace(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    if let Some(p) = &sub.prompt {
        builder = builder.with_system_prompt(p.clone());
    }
    builder.build().map_err(|e| anyhow!("subagent `{}`: {e}", sub.name))
}

/// Translate the declarative `[tool_policy]` config into a runtime `ToolPolicy`.
fn tool_policy_from_config(cfg: &ToolPolicyConfig) -> ToolPolicy {
    let default = if cfg.default_allow { ToolAccess::Allow } else { ToolAccess::Deny };
    let mut policy =
        ToolPolicy::new(default).with_bash_allowed_prefixes(cfg.bash_allowed_prefixes.clone());
    for t in &cfg.deny {
        policy = policy.deny(t.clone());
    }
    for t in &cfg.allow {
        policy = policy.allow(t.clone());
    }
    policy
}

/// Connect one MCP server over stdio, list its tools, and register each as a
/// namespaced `McpTool` (`<server>__<tool>`). Returns the number registered.
async fn register_mcp_server(tools: &mut ToolRegistry, srv: &McpServerConfig) -> Result<usize> {
    let env: Vec<(String, String)> =
        srv.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let client = McpClient::connect_stdio(&srv.command, &srv.args, &env)
        .await
        .map_err(|e| anyhow!("connect: {e}"))?;
    let defs = client.list_tools().await.map_err(|e| anyhow!("list_tools: {e}"))?;
    let count = defs.len();
    for def in defs {
        let exposed = format!("{}__{}", srv.name, def.name);
        tools.register(Arc::new(McpTool::new(
            client.clone(),
            exposed,
            def.name,
            def.description,
            def.input_schema,
        )));
    }
    Ok(count)
}

fn build_provider(eff: &EffectiveCli) -> Result<(Arc<dyn LlmProvider>, String)> {
    let provider_id = eff.provider.to_ascii_lowercase();

    // Resolve the provider through a registry of lazy factories: only the
    // selected provider is constructed, so only its API key must be present.
    // This is the seam for registering additional providers later.
    let mut registry = ProviderRegistry::new();
    register_builtin_providers(&mut registry);

    let provider = registry
        .build(&provider_id)
        .ok_or_else(|| {
            anyhow!(
                "unsupported provider `{provider_id}` (expected `openai`, `deepseek`, or `claude`)"
            )
        })?
        .map_err(|e| anyhow!("{e}"))?;

    let model = eff
        .model
        .clone()
        .unwrap_or_else(|| default_model_for(&provider_id).to_string());

    Ok((provider, model))
}

/// Register the built-in providers as lazy factories. Each reads its API key
/// from the environment only when selected; the error wording matches the
/// previous hard-coded branches exactly.
fn register_builtin_providers(registry: &mut ProviderRegistry) {
    registry.register_factory("openai", || {
        let key =
            std::env::var("OPENAI_API_KEY").map_err(|_| "OPENAI_API_KEY is not set".to_string())?;
        let p = OpenAiProvider::new(OpenAiConfig::openai(key))
            .map_err(|e| format!("provider init: {e}"))?;
        Ok(Arc::new(p) as Arc<dyn LlmProvider>)
    });
    registry.register_factory("deepseek", || {
        let key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| "DEEPSEEK_API_KEY is not set".to_string())?;
        let p = OpenAiProvider::new(OpenAiConfig::deepseek(key))
            .map_err(|e| format!("provider init: {e}"))?;
        Ok(Arc::new(p) as Arc<dyn LlmProvider>)
    });
    for name in ["claude", "anthropic"] {
        registry.register_factory(name, || {
            let key = std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| "ANTHROPIC_API_KEY is not set".to_string())?;
            let p = AnthropicProvider::new(AnthropicConfig::new(key))
                .map_err(|e| format!("provider init: {e}"))?;
            Ok(Arc::new(p) as Arc<dyn LlmProvider>)
        });
    }
}

/// Provider-specific default model when the user didn't pin one with `--model`.
fn default_model_for(provider_id: &str) -> &'static str {
    match provider_id {
        "deepseek" => "deepseek-chat",
        "claude" | "anthropic" => "claude-sonnet-4-5",
        _ => "gpt-4o-mini",
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
