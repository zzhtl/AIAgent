//! agent-config
//!
//! Layered configuration loading (figment). Resolution order:
//!
//! 1. `/etc/agent/config.toml`
//! 2. `~/.config/agent/config.toml`
//! 3. `./agent.toml` (project-local)
//! 4. `AGENT_*` environment variables
//!
//! API keys are never read from config files — only from env / keyring.
