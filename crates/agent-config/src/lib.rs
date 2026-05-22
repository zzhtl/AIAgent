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
        }
    }
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
}
