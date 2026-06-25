//! agent-config
//!
//! Layered configuration loading (figment). Resolution order, later entries
//! winning:
//!
//! 1. Hard-coded defaults
//! 2. `/etc/agent/config.toml`
//! 3. `~/.config/agent/config.toml` (or `$XDG_CONFIG_HOME/agent/config.toml`)
//! 4. `./agent.toml` (project-local)
//! 5. `AGENT_*` environment variables (double-underscore splits sections,
//!    e.g. `AGENT_AGENT__MAX_STEPS=20` ⇒ `agent.max_steps = 20`).
//!
//! API keys are never read from config files — only from environment
//! variables / system keyring at use time.
//!
//! CLI flags should still take precedence over this layer: load the config,
//! then apply any explicit `--flag` overrides on top.

use std::path::{Path, PathBuf};

use figment::providers::{Env, Format, Serialized, Toml};
use figment::Figment;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("figment: {0}")]
    Figment(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

/// Top-level configuration block.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfig {
    /// Provider id: `openai`, `deepseek`, or `claude` / `anthropic`.
    pub provider: String,
    /// Model id. `None` means "use the provider's built-in default".
    pub model: Option<String>,
    /// Disable built-in tools entirely (text-only chat).
    pub no_tools: bool,
    /// After each run, generate a reflection note.
    pub evolve: bool,
    /// Override the config directory. When `None`, the loader uses
    /// `$XDG_CONFIG_HOME/agent` or `~/.config/agent`.
    pub config_dir: Option<PathBuf>,
    pub agent: LoopConfig,
    pub permissions: PermissionsConfig,
    /// Specialist sub-agents exposed to the main agent as callable tools. Each
    /// entry becomes a `SubAgentTool`. Empty by default.
    #[serde(default)]
    pub subagents: Vec<SubAgentConfig>,
    /// MCP servers to connect on startup; each server's tools are registered
    /// alongside the built-ins. Empty by default.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Tool access policy (sandbox). Default allows everything; tighten it for
    /// untrusted / multi-tenant deployments.
    #[serde(default)]
    pub tool_policy: ToolPolicyConfig,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider: "openai".into(),
            model: None,
            no_tools: false,
            evolve: false,
            config_dir: None,
            agent: LoopConfig::default(),
            permissions: PermissionsConfig::default(),
            subagents: Vec::new(),
            mcp_servers: Vec::new(),
            tool_policy: ToolPolicyConfig::default(),
        }
    }
}

/// Tool access policy (sandbox). `default_allow` governs tools not named in
/// `allow`/`deny`; a non-empty `bash_allowed_prefixes` further confines `bash`.
/// Maps to `agent_tools::policy::ToolPolicy` at the application boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolPolicyConfig {
    /// Default decision for tools not listed in `allow`/`deny`.
    pub default_allow: bool,
    /// Tools explicitly denied (override `default_allow`).
    pub deny: Vec<String>,
    /// Tools explicitly allowed (override `default_allow`).
    pub allow: Vec<String>,
    /// If non-empty, `bash` commands must start with one of these prefixes.
    pub bash_allowed_prefixes: Vec<String>,
}

impl Default for ToolPolicyConfig {
    fn default() -> Self {
        Self {
            default_allow: true,
            deny: Vec::new(),
            allow: Vec::new(),
            bash_allowed_prefixes: Vec::new(),
        }
    }
}

/// An MCP server to spawn and connect over stdio. Its advertised tools are
/// registered as agent tools, namespaced by `name`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpServerConfig {
    /// Local identifier, used as a prefix so tool names stay unique.
    pub name: String,
    /// Executable to launch (e.g. `npx`, `uvx`, a path).
    pub command: String,
    /// Arguments passed to `command`.
    #[serde(default)]
    pub args: Vec<String>,
    /// Extra environment variables for the server process.
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
}

/// Declares a specialist sub-agent the main agent can call as a tool. The CLI
/// builds one `Agent` per entry (reusing the main provider) and registers it as
/// a `SubAgentTool` alongside the built-in tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SubAgentConfig {
    /// Tool name the main agent calls. Should be unique and snake_case.
    pub name: String,
    /// Description shown to the main agent — when to delegate to this sub-agent.
    pub description: String,
    /// Model id for the sub-agent. `None` ⇒ reuse the main agent's model.
    #[serde(default)]
    pub model: Option<String>,
    /// System prompt defining the sub-agent's role / specialty.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Per-sub-agent step cap. `None` ⇒ reuse the main loop's `max_steps`.
    #[serde(default)]
    pub max_steps: Option<u32>,
}

/// Tunables for the agent run loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoopConfig {
    pub max_steps: u32,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// When `history.len()` exceeds this many messages, the CLI compresses
    /// the early portion into a system-prompt summary. `0` disables the
    /// behaviour entirely.
    pub summary_threshold: usize,
    /// How many of the most recent messages to keep verbatim after a
    /// summary is generated (the rest get replaced by the summary).
    pub summary_keep_tail: usize,
    /// Enable semantic recall: every turn embeds the user input and pulls
    /// the top-`vector_recall_top_k` closest entries from the vector store.
    /// Requires `OPENAI_API_KEY` (for embeddings) and a populated vector
    /// table — run `agent memory index` first to seed it from your facts.
    pub vector_recall: bool,
    pub vector_recall_top_k: usize,
    pub vector_recall_min_score: f32,
    /// Max automatic retries for transient LLM errors (network / rate-limit /
    /// 5xx) before the turn aborts. `0` disables retrying.
    pub max_retries: u32,
    /// Base backoff (ms) for retries; the delay grows exponentially
    /// (`base * 2^attempt`) unless the provider supplies a `Retry-After`.
    pub retry_base_delay_ms: u64,
    /// Cumulative token budget for a single turn across all loop steps.
    /// When the running total reaches it the loop stops. `None` = unlimited.
    pub token_budget: Option<u32>,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_steps: 12,
            max_tokens: None,
            temperature: None,
            summary_threshold: 30,
            summary_keep_tail: 8,
            vector_recall: false,
            vector_recall_top_k: 5,
            vector_recall_min_score: 0.2,
            max_retries: 2,
            retry_base_delay_ms: 500,
            token_budget: None,
        }
    }
}

/// Permission gates for built-in tools. Maps 1:1 to
/// `agent_core::tool::Permissions`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PermissionsConfig {
    pub allow_read: bool,
    pub allow_write: bool,
    pub allow_shell: bool,
    pub allow_network: bool,
    pub max_runtime_secs: u64,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            allow_read: true,
            allow_write: true,
            allow_shell: true,
            allow_network: true,
            max_runtime_secs: 120,
        }
    }
}

impl PermissionsConfig {
    pub fn to_runtime(&self) -> agent_core::tool::Permissions {
        agent_core::tool::Permissions {
            allow_read: self.allow_read,
            allow_write: self.allow_write,
            allow_shell: self.allow_shell,
            allow_network: self.allow_network,
            max_runtime_secs: self.max_runtime_secs,
        }
    }
}

impl AgentConfig {
    /// Load the layered configuration. Pass an optional CLI override for
    /// the config directory; when provided, it shortcuts the XDG lookup.
    pub fn load(cli_config_dir: Option<&Path>) -> Result<Self> {
        let user_dir = match cli_config_dir {
            Some(p) => Some(p.to_path_buf()),
            None => default_user_config_dir(),
        };

        let mut figment = Figment::from(Serialized::defaults(AgentConfig::default()));

        let etc = PathBuf::from("/etc/agent/config.toml");
        if etc.exists() {
            figment = figment.merge(Toml::file(&etc));
        }
        if let Some(dir) = user_dir.as_ref() {
            let f = dir.join("config.toml");
            if f.exists() {
                figment = figment.merge(Toml::file(&f));
            }
        }
        let local = PathBuf::from("./agent.toml");
        if local.exists() {
            figment = figment.merge(Toml::file(&local));
        }

        figment = figment.merge(Env::prefixed("AGENT_").split("__"));

        let mut cfg: AgentConfig = figment
            .extract()
            .map_err(|e| ConfigError::Figment(e.to_string()))?;

        // Promote the CLI-supplied dir into the struct so downstream code
        // sees a fully-resolved path regardless of how it was passed in.
        if cfg.config_dir.is_none() {
            cfg.config_dir = user_dir;
        }
        Ok(cfg)
    }

    /// Resolved config directory (CLI flag → loaded value → XDG / HOME →
    /// `./.agent`).
    pub fn config_dir(&self) -> PathBuf {
        if let Some(p) = self.config_dir.as_ref() {
            return p.clone();
        }
        default_user_config_dir().unwrap_or_else(|| PathBuf::from(".agent"))
    }
}

/// Resolve `$XDG_CONFIG_HOME/agent` → `~/.config/agent` → `None`.
pub fn default_user_config_dir() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return Some(PathBuf::from(xdg).join("agent"));
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        return Some(PathBuf::from(home).join(".config").join("agent"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_self_consistent() {
        let cfg = AgentConfig::default();
        assert_eq!(cfg.provider, "openai");
        assert_eq!(cfg.agent.max_steps, 12);
        assert!(cfg.permissions.allow_read);
        assert_eq!(cfg.permissions.max_runtime_secs, 120);
    }

    #[test]
    fn permissions_round_trip_to_runtime() {
        let cfg = PermissionsConfig::default();
        let p = cfg.to_runtime();
        assert!(p.allow_read);
        assert_eq!(p.max_runtime_secs, 120);
    }

    #[test]
    fn subagents_default_empty() {
        assert!(AgentConfig::default().subagents.is_empty());
    }

    #[test]
    fn subagents_parse_from_toml() {
        let toml = r#"
provider = "openai"

[[subagents]]
name = "researcher"
description = "Researches topics in depth"
model = "gpt-4o"
prompt = "You are a research specialist."
"#;
        let cfg: AgentConfig = Figment::from(Serialized::defaults(AgentConfig::default()))
            .merge(Toml::string(toml))
            .extract()
            .expect("parse subagents");
        assert_eq!(cfg.subagents.len(), 1);
        assert_eq!(cfg.subagents[0].name, "researcher");
        assert_eq!(cfg.subagents[0].model.as_deref(), Some("gpt-4o"));
        assert!(cfg.subagents[0].max_steps.is_none());
    }

    #[test]
    fn mcp_servers_parse_from_toml() {
        let toml = r#"
provider = "openai"

[[mcp_servers]]
name = "fs"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

[mcp_servers.env]
FOO = "bar"
"#;
        let cfg: AgentConfig = Figment::from(Serialized::defaults(AgentConfig::default()))
            .merge(Toml::string(toml))
            .extract()
            .expect("parse mcp_servers");
        assert_eq!(cfg.mcp_servers.len(), 1);
        assert_eq!(cfg.mcp_servers[0].name, "fs");
        assert_eq!(cfg.mcp_servers[0].command, "npx");
        assert_eq!(cfg.mcp_servers[0].args.len(), 3);
        assert_eq!(cfg.mcp_servers[0].env.get("FOO").map(String::as_str), Some("bar"));
    }

    #[test]
    fn tool_policy_defaults_to_unrestricted() {
        let p = AgentConfig::default().tool_policy;
        assert!(p.default_allow);
        assert!(p.deny.is_empty());
        assert!(p.bash_allowed_prefixes.is_empty());
    }

    #[test]
    fn tool_policy_parses_from_toml() {
        let toml = r#"
provider = "openai"

[tool_policy]
default_allow = true
deny = ["bash"]
bash_allowed_prefixes = ["ls", "git "]
"#;
        let cfg: AgentConfig = Figment::from(Serialized::defaults(AgentConfig::default()))
            .merge(Toml::string(toml))
            .extract()
            .expect("parse tool_policy");
        assert!(cfg.tool_policy.default_allow);
        assert_eq!(cfg.tool_policy.deny, vec!["bash".to_string()]);
        assert_eq!(cfg.tool_policy.bash_allowed_prefixes.len(), 2);
    }
}
