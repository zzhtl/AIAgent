//! Policy-based tool access control.
//!
//! [`PolicyHook`] is an [`AgentHook`] that enforces a [`ToolPolicy`] at the
//! `before_tool` seam: a denied tool — or a `bash` command outside an allowed
//! prefix set — is blocked before it ever runs. This is the sandbox primitive
//! for untrusted / multi-tenant deployments, built entirely on the hook
//! foundation (no changes to the agent loop).

use std::collections::HashMap;

use agent_core::hook::{AgentHook, HookDecision};
use agent_core::message::ToolUse;
use agent_core::tool::ToolContext;
use async_trait::async_trait;

/// Whether a tool may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolAccess {
    Allow,
    Deny,
}

/// A static tool-access policy. `default` applies to any tool not named in
/// `overrides`. A non-empty `bash_allowed_prefixes` additionally confines the
/// `bash` tool to commands that start with one of the listed prefixes.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    default: ToolAccess,
    overrides: HashMap<String, ToolAccess>,
    bash_allowed_prefixes: Vec<String>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            default: ToolAccess::Allow,
            overrides: HashMap::new(),
            bash_allowed_prefixes: Vec::new(),
        }
    }
}

impl ToolPolicy {
    /// A policy with the given default for unlisted tools.
    pub fn new(default: ToolAccess) -> Self {
        Self { default, ..Default::default() }
    }

    pub fn allow(mut self, name: impl Into<String>) -> Self {
        self.overrides.insert(name.into(), ToolAccess::Allow);
        self
    }

    pub fn deny(mut self, name: impl Into<String>) -> Self {
        self.overrides.insert(name.into(), ToolAccess::Deny);
        self
    }

    pub fn with_bash_allowed_prefixes(mut self, prefixes: Vec<String>) -> Self {
        self.bash_allowed_prefixes = prefixes;
        self
    }

    /// Whether a tool is permitted by name (ignores per-argument rules).
    pub fn tool_allowed(&self, name: &str) -> bool {
        match self.overrides.get(name) {
            Some(access) => *access == ToolAccess::Allow,
            None => self.default == ToolAccess::Allow,
        }
    }

    /// Whether a `bash` command passes the prefix whitelist. An empty whitelist
    /// allows everything.
    pub fn bash_command_allowed(&self, command: &str) -> bool {
        if self.bash_allowed_prefixes.is_empty() {
            return true;
        }
        let cmd = command.trim_start();
        self.bash_allowed_prefixes.iter().any(|p| cmd.starts_with(p.as_str()))
    }

    /// True when the policy restricts nothing — lets callers skip installing
    /// the hook entirely (zero overhead in the common case).
    pub fn is_unrestricted(&self) -> bool {
        self.default == ToolAccess::Allow
            && self.overrides.values().all(|a| *a == ToolAccess::Allow)
            && self.bash_allowed_prefixes.is_empty()
    }
}

/// Enforces a [`ToolPolicy`] via the `before_tool` hook seam.
pub struct PolicyHook {
    policy: ToolPolicy,
}

impl PolicyHook {
    pub fn new(policy: ToolPolicy) -> Self {
        Self { policy }
    }
}

#[async_trait]
impl AgentHook for PolicyHook {
    async fn before_tool(&self, call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
        if !self.policy.tool_allowed(&call.name) {
            return HookDecision::Block(format!("tool `{}` denied by policy", call.name));
        }
        if call.name == "bash" {
            let cmd = call.input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            if !self.policy.bash_command_allowed(cmd) {
                return HookDecision::Block(format!(
                    "bash command denied by policy (not in allowed prefixes): {cmd}"
                ));
            }
        }
        HookDecision::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_allow_with_denylist() {
        let p = ToolPolicy::new(ToolAccess::Allow).deny("bash");
        assert!(p.tool_allowed("file_read"));
        assert!(!p.tool_allowed("bash"));
        assert!(!p.is_unrestricted());
    }

    #[test]
    fn default_deny_with_allowlist() {
        let p = ToolPolicy::new(ToolAccess::Deny).allow("file_read");
        assert!(p.tool_allowed("file_read"));
        assert!(!p.tool_allowed("grep"));
    }

    #[test]
    fn bash_prefix_whitelist() {
        let p = ToolPolicy::default()
            .with_bash_allowed_prefixes(vec!["ls".into(), "git ".into()]);
        assert!(p.bash_command_allowed("ls -la"));
        assert!(p.bash_command_allowed("  git status"));
        assert!(!p.bash_command_allowed("rm -rf /"));
        // An empty whitelist allows everything.
        assert!(ToolPolicy::default().bash_command_allowed("rm -rf /"));
    }

    #[tokio::test]
    async fn hook_blocks_denied_tool() {
        let hook = PolicyHook::new(ToolPolicy::new(ToolAccess::Allow).deny("bash"));
        let mut call = ToolUse { id: "1".into(), name: "bash".into(), input: json!({"command": "ls"}) };
        let d = hook.before_tool(&mut call, &ToolContext::new(".".into())).await;
        assert!(matches!(d, HookDecision::Block(_)));
    }

    #[tokio::test]
    async fn hook_blocks_out_of_policy_bash_command() {
        let hook = PolicyHook::new(
            ToolPolicy::default().with_bash_allowed_prefixes(vec!["ls".into()]),
        );
        let mut ok = ToolUse { id: "1".into(), name: "bash".into(), input: json!({"command": "ls -la"}) };
        assert!(matches!(
            hook.before_tool(&mut ok, &ToolContext::new(".".into())).await,
            HookDecision::Continue
        ));
        let mut bad = ToolUse { id: "2".into(), name: "bash".into(), input: json!({"command": "curl evil"}) };
        assert!(matches!(
            hook.before_tool(&mut bad, &ToolContext::new(".".into())).await,
            HookDecision::Block(_)
        ));
    }
}
