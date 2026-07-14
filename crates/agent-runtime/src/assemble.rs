use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_config::{AgentConfig, McpServerConfig, SubAgentConfig, ToolPolicyConfig};
use agent_core::agent::RunConfig;
use agent_core::evolution::CandidateQueue;
use agent_core::memory::{EmbeddingProvider, VectorStore};
use agent_core::tool::Permissions;
use agent_core::{
    Agent, ChainedPromptProvider, FactStore, LlmProvider, PromptProvider, SubAgentTool,
    ToolRegistry,
};
use agent_evolution::{Reflector, Summariser};
use agent_llm::providers::anthropic::{AnthropicConfig, AnthropicProvider};
use agent_llm::providers::openai::{OpenAiConfig, OpenAiProvider};
use agent_llm::providers::openai_embeddings::OpenAiEmbeddingProvider;
use agent_llm::ProviderRegistry;
use agent_mcp::{McpClient, McpTool};
use agent_memory::{
    FactsPromptProvider, MarkdownFactStore, SimpleVectorStore, SqliteSessionStore,
    VectorRecallPromptProvider,
};
use agent_skills::{Augmenter, RuleSet, SkillRegistry};
use agent_tools::policy::{PolicyHook, ToolAccess, ToolPolicy};
use anyhow::{anyhow, Context, Result};

/// Per-entry overrides layered on top of [`AgentConfig`].
#[derive(Debug, Clone, Default)]
pub struct BuildOptions {
    /// Provider selected by a CLI flag. `None` uses `config.provider`.
    pub provider_override: Option<String>,
    /// Explicit model selected by an app flag or its legacy environment
    /// variable. This remains authoritative if provider probing falls back.
    pub model_override: Option<String>,
    /// Disable tools in addition to `config.no_tools`.
    pub no_tools: bool,
    /// Enable reflection in addition to `config.evolve`.
    pub evolve: bool,
    /// Tool workspace. `None` resolves to the process current directory.
    pub workspace: Option<PathBuf>,
    /// Entry-specific permission profile, used by the web safety boundary.
    pub permissions_override: Option<Permissions>,
    /// Entry-specific policy profile, used by `[web.tool_policy]`.
    pub tool_policy_override: Option<ToolPolicyConfig>,
    /// If the configured provider is missing its API key, probe the legacy
    /// web/bot order: OpenAI, Anthropic, then DeepSeek.
    pub env_probe_fallback: bool,
}

/// Provider selection result reusable by non-agent commands such as
/// `agent evolution extract`.
pub struct ResolvedProvider {
    pub provider: Arc<dyn LlmProvider>,
    pub provider_id: String,
    pub model: String,
    pub warnings: Vec<String>,
}

/// Fully assembled runtime shared by every first-party entry.
pub struct AgentRuntime {
    pub agent: Agent,
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
    pub fact_store: Arc<dyn FactStore>,
    pub candidate_queue: CandidateQueue,
    pub config_dir: PathBuf,
    pub warnings: Vec<String>,
    summary_threshold: usize,
    summary_keep_tail: usize,
    evolve: bool,
}

impl AgentRuntime {
    pub fn summariser(&self) -> Option<Summariser> {
        (self.summary_threshold > 0)
            .then(|| Summariser::new(self.provider.clone(), self.model.clone()))
    }

    pub fn reflector(&self) -> Option<Reflector> {
        self.evolve.then(|| {
            Reflector::new(
                self.provider.clone(),
                self.model.clone(),
                self.fact_store.clone(),
            )
        })
    }

    pub fn summary_threshold(&self) -> usize {
        self.summary_threshold
    }

    pub fn summary_keep_tail(&self) -> usize {
        self.summary_keep_tail
    }

    pub fn evolve_enabled(&self) -> bool {
        self.evolve
    }
}

/// Resolve the configured provider and build a complete runtime.
pub async fn build_runtime(config: &AgentConfig, options: BuildOptions) -> Result<AgentRuntime> {
    let resolved = resolve_provider(config, &options)?;
    let mut runtime = build_runtime_with_provider(
        config,
        options,
        resolved.provider,
        resolved.model,
    )
    .await?;
    runtime.warnings.splice(0..0, resolved.warnings);
    Ok(runtime)
}

/// Build a runtime around an already-created provider. Tests and embedders
/// use this to avoid touching process credentials.
pub async fn build_runtime_with_provider(
    config: &AgentConfig,
    options: BuildOptions,
    provider: Arc<dyn LlmProvider>,
    model: impl Into<String>,
) -> Result<AgentRuntime> {
    let model = model.into();
    let config_dir = config.config_dir();
    std::fs::create_dir_all(&config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;

    let workspace = options
        .workspace
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let permissions = options
        .permissions_override
        .clone()
        .unwrap_or_else(|| config.permissions.to_runtime());
    let policy_config = options
        .tool_policy_override
        .as_ref()
        .unwrap_or(&config.tool_policy);

    let fact_store: Arc<dyn FactStore> =
        Arc::new(MarkdownFactStore::open(config_dir.join("memory")));
    let candidate_queue = CandidateQueue::open(config_dir.join("evolution").join("queue.json"));
    let mut warnings = Vec::new();

    let mut tools = ToolRegistry::new();
    if !(config.no_tools || options.no_tools) {
        agent_tools::register_builtins(&mut tools);
        agent_tools::register_memory_tools(&mut tools);
        agent_tools::register_evolution_tools(&mut tools);

        for sub in &config.subagents {
            let sub_agent = build_subagent(
                config,
                provider.clone(),
                &model,
                sub,
                permissions.clone(),
                workspace.clone(),
            )?;
            tools.register(Arc::new(SubAgentTool::new(
                sub.name.clone(),
                sub.description.clone(),
                sub_agent,
            )));
        }

        for server in &config.mcp_servers {
            match register_mcp_server(&mut tools, server).await {
                Ok(count) => tracing::info!(
                    "mcp `{}`: registered {count} tool(s)",
                    server.name
                ),
                Err(error) => warnings.push(format!(
                    "mcp server `{}` skipped: {error}",
                    server.name
                )),
            }
        }
    }

    let skills = SkillRegistry::load_dir(&config_dir.join("skills"))
        .map_err(|error| anyhow!("skills: {error}"))?;
    let rules = RuleSet::load_dir(&config_dir.join("rules"))
        .map_err(|error| anyhow!("rules: {error}"))?;
    let augmenter = Augmenter::new(rules, skills);

    let mut prompt_chain = ChainedPromptProvider::new();
    if !augmenter.is_empty() {
        prompt_chain.push(Arc::new(augmenter));
    }
    prompt_chain.push(Arc::new(FactsPromptProvider::new(fact_store.clone())));

    if config.agent.vector_recall {
        match build_vector_recall(config, &config_dir).await {
            Ok(Some(recall)) => prompt_chain.push(recall),
            Ok(None) => warnings.push(
                "vector_recall enabled but skipped (vector store empty or embedder unavailable; run `agent memory index` and set OPENAI_API_KEY)"
                    .into(),
            ),
            Err(error) => warnings.push(format!("vector_recall init failed: {error}")),
        }
    }

    let run_config = RunConfig {
        max_steps: config.agent.max_steps,
        temperature: config.agent.temperature,
        max_tokens: config.agent.max_tokens,
        permissions,
        max_retries: config.agent.max_retries,
        retry_base_delay_ms: config.agent.retry_base_delay_ms,
        token_budget: config.agent.token_budget,
    };

    let mut builder = Agent::builder()
        .with_llm(provider.clone())
        .with_model(model.clone())
        .with_tools(tools)
        .with_fact_store(fact_store.clone())
        .with_candidate_queue(candidate_queue.clone())
        .with_config(run_config)
        .with_workspace(workspace);
    if !prompt_chain.is_empty() {
        let prompt_provider: Arc<dyn PromptProvider> = Arc::new(prompt_chain);
        builder = builder.with_prompt_provider(prompt_provider);
    }
    let policy = tool_policy_from_config(policy_config);
    if !policy.is_unrestricted() {
        builder = builder.with_hook(Arc::new(PolicyHook::new(policy)));
    }
    let agent = builder.build().map_err(|error| anyhow!("agent builder: {error}"))?;

    Ok(AgentRuntime {
        agent,
        provider,
        model,
        fact_store,
        candidate_queue,
        config_dir,
        warnings,
        summary_threshold: config.agent.summary_threshold,
        summary_keep_tail: config.agent.summary_keep_tail.max(1),
        evolve: config.evolve || options.evolve,
    })
}

/// Build only the selected provider. The configured provider is authoritative;
/// fallback happens solely when its credential is absent.
pub fn resolve_provider(config: &AgentConfig, options: &BuildOptions) -> Result<ResolvedProvider> {
    let configured = normalize_provider_id(
        options
            .provider_override
            .as_deref()
            .unwrap_or(&config.provider),
    )?;
    let candidates = provider_probe_order(&configured, options.env_probe_fallback, |provider| {
        provider_key_available(provider)
    });
    let selected = candidates.first().cloned().unwrap_or_else(|| configured.clone());

    let mut registry = ProviderRegistry::new();
    register_builtin_providers(&mut registry);
    let provider = registry
        .build(&selected)
        .ok_or_else(|| anyhow!("unsupported provider `{selected}`"))?
        .map_err(|error| anyhow!("{error}"))?;

    let fell_back = selected != configured;
    let model = resolve_model(
        config.model.as_deref(),
        options.model_override.as_deref(),
        fell_back,
        &selected,
    );
    let warnings = if fell_back {
        vec![format!(
            "provider `{configured}` has no API key; using `{selected}` from environment probe"
        )]
    } else {
        Vec::new()
    };

    Ok(ResolvedProvider { provider, provider_id: selected, model, warnings })
}

/// Open the shared SQLite database. The concrete return type lets web/bot
/// ensure externally chosen session ids before appending messages.
pub async fn open_session_store(config_dir: &Path) -> Result<SqliteSessionStore> {
    std::fs::create_dir_all(config_dir)
        .with_context(|| format!("create_dir_all {}", config_dir.display()))?;
    SqliteSessionStore::open(&config_dir.join("sessions.db"))
        .await
        .map_err(|error| anyhow!("open sessions.db: {error}"))
}

fn build_subagent(
    config: &AgentConfig,
    provider: Arc<dyn LlmProvider>,
    main_model: &str,
    sub: &SubAgentConfig,
    permissions: Permissions,
    workspace: PathBuf,
) -> Result<Agent> {
    let mut tools = ToolRegistry::new();
    agent_tools::register_builtins(&mut tools);
    let run_config = RunConfig {
        max_steps: sub.max_steps.unwrap_or(config.agent.max_steps),
        permissions,
        ..RunConfig::default()
    };
    let mut builder = Agent::builder()
        .with_llm(provider)
        .with_model(sub.model.clone().unwrap_or_else(|| main_model.to_string()))
        .with_tools(tools)
        .with_config(run_config)
        .with_workspace(workspace);
    if let Some(prompt) = &sub.prompt {
        builder = builder.with_system_prompt(prompt.clone());
    }
    builder.build().map_err(|error| anyhow!("subagent `{}`: {error}", sub.name))
}

async fn register_mcp_server(
    tools: &mut ToolRegistry,
    server: &McpServerConfig,
) -> Result<usize> {
    let env: Vec<(String, String)> = server
        .env
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let client = McpClient::connect_stdio(&server.command, &server.args, &env)
        .await
        .map_err(|error| anyhow!("connect: {error}"))?;
    let definitions = client
        .list_tools()
        .await
        .map_err(|error| anyhow!("list_tools: {error}"))?;
    let count = definitions.len();
    for definition in definitions {
        let exposed_name = format!("{}__{}", server.name, definition.name);
        tools.register(Arc::new(McpTool::new(
            client.clone(),
            exposed_name,
            definition.name,
            definition.description,
            definition.input_schema,
        )));
    }
    Ok(count)
}

async fn build_vector_recall(
    config: &AgentConfig,
    config_dir: &Path,
) -> Result<Option<Arc<dyn PromptProvider>>> {
    let Ok(key) = std::env::var("OPENAI_API_KEY") else {
        return Ok(None);
    };
    let embedder: Arc<dyn EmbeddingProvider> = Arc::new(
        OpenAiEmbeddingProvider::new(key).map_err(|error| anyhow!("embedder init: {error}"))?,
    );
    let session_store = open_session_store(config_dir).await?;
    let vectors: Arc<dyn VectorStore> =
        Arc::new(SimpleVectorStore::from_session_store(&session_store));
    if vectors.is_empty().await.unwrap_or(true) {
        return Ok(None);
    }
    let recall = VectorRecallPromptProvider::new(embedder, vectors)
        .with_top_k(config.agent.vector_recall_top_k)
        .with_min_score(config.agent.vector_recall_min_score);
    Ok(Some(Arc::new(recall)))
}

fn tool_policy_from_config(config: &ToolPolicyConfig) -> ToolPolicy {
    let default = if config.default_allow {
        ToolAccess::Allow
    } else {
        ToolAccess::Deny
    };
    let mut policy =
        ToolPolicy::new(default).with_bash_allowed_prefixes(config.bash_allowed_prefixes.clone());
    for name in &config.deny {
        policy = policy.deny(name.clone());
    }
    for name in &config.allow {
        policy = policy.allow(name.clone());
    }
    policy
}

fn register_builtin_providers(registry: &mut ProviderRegistry) {
    registry.register_factory("openai", || {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY is not set".to_string())?;
        let provider = OpenAiProvider::new(OpenAiConfig::openai(key))
            .map_err(|error| format!("provider init: {error}"))?;
        Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
    });
    registry.register_factory("deepseek", || {
        let key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| "DEEPSEEK_API_KEY is not set".to_string())?;
        let provider = OpenAiProvider::new(OpenAiConfig::deepseek(key))
            .map_err(|error| format!("provider init: {error}"))?;
        Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
    });
    registry.register_factory("anthropic", || {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY is not set".to_string())?;
        let provider = AnthropicProvider::new(AnthropicConfig::new(key))
            .map_err(|error| format!("provider init: {error}"))?;
        Ok(Arc::new(provider) as Arc<dyn LlmProvider>)
    });
}

fn normalize_provider_id(provider: &str) -> Result<String> {
    match provider.to_ascii_lowercase().as_str() {
        "openai" => Ok("openai".into()),
        "deepseek" => Ok("deepseek".into()),
        "claude" | "anthropic" => Ok("anthropic".into()),
        other => Err(anyhow!(
            "unsupported provider `{other}` (expected `openai`, `deepseek`, or `claude`)"
        )),
    }
}

fn provider_probe_order(
    configured: &str,
    fallback: bool,
    key_available: impl Fn(&str) -> bool,
) -> Vec<String> {
    if !fallback || key_available(configured) {
        return vec![configured.to_string()];
    }
    let available: Vec<String> = ["openai", "anthropic", "deepseek"]
        .into_iter()
        .filter(|provider| key_available(provider))
        .map(str::to_string)
        .collect();
    if available.is_empty() {
        vec![configured.to_string()]
    } else {
        available
    }
}

fn provider_key_available(provider: &str) -> bool {
    let key = match provider {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "deepseek" => "DEEPSEEK_API_KEY",
        _ => return false,
    };
    std::env::var(key).is_ok()
}

fn default_model_for(provider: &str) -> &'static str {
    match provider {
        "deepseek" => "deepseek-chat",
        "anthropic" => "claude-sonnet-4-5",
        _ => "gpt-4o-mini",
    }
}

fn resolve_model(
    configured_model: Option<&str>,
    model_override: Option<&str>,
    fell_back: bool,
    selected_provider: &str,
) -> String {
    model_override
        .map(str::to_string)
        .or_else(|| (!fell_back).then(|| configured_model.map(str::to_string)).flatten())
        .unwrap_or_else(|| default_model_for(selected_provider).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn provider_probe_prefers_configured_key() {
        let available = HashSet::from(["openai"]);
        let order = provider_probe_order("openai", true, |name| available.contains(name));
        assert_eq!(order, vec!["openai"]);
    }

    #[test]
    fn provider_probe_uses_legacy_order_when_configured_key_is_missing() {
        let available = HashSet::from(["anthropic", "deepseek"]);
        let order = provider_probe_order("openai", true, |name| available.contains(name));
        assert_eq!(order, vec!["anthropic", "deepseek"]);
    }

    #[test]
    fn provider_probe_can_be_disabled() {
        let available = HashSet::from(["anthropic"]);
        let order = provider_probe_order("openai", false, |name| available.contains(name));
        assert_eq!(order, vec!["openai"]);
    }

    #[test]
    fn provider_fallback_discards_config_model_but_keeps_explicit_override() {
        assert_eq!(
            resolve_model(Some("gpt-config"), None, true, "anthropic"),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            resolve_model(
                Some("gpt-config"),
                Some("operator-override"),
                true,
                "anthropic"
            ),
            "operator-override"
        );
    }
}
