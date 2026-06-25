//! Agent lifecycle hooks (middleware).
//!
//! An optional `AgentHook` lets an embedder observe and influence a run at
//! four seams: before/after the LLM call and before/after each tool. Every
//! method has a default no-op implementation, so a hook overrides only what it
//! needs. When no hook is installed the loop takes a `None` fast-path with zero
//! overhead.
//!
//! This is the seam the upper layers build on: approval / sandbox (`before_tool`
//! returning `Block`), audit and cost accounting (`after_tool`, `after_llm`),
//! context injection and budget guards (`before_llm` returning `Modify` /
//! `Abort`). `on_event` is intentionally not exposed yet — yielding inside the
//! `async_stream` loop resists clean wrapping, and the four seams here cover the
//! known use cases.

use async_trait::async_trait;
use std::sync::Arc;

use crate::llm::ChatRequest;
use crate::message::ToolUse;
use crate::tool::{ToolContext, ToolOutcome};

/// Decision returned by `before_*` hooks.
#[derive(Debug)]
pub enum HookDecision {
    /// Proceed unchanged.
    Continue,
    /// The hook mutated the in-flight value in place (a `ChatRequest`, a tool
    /// `ToolUse.input`); proceed with the modified value.
    Modify,
    /// Skip this step with a substituted result but keep the turn running
    /// (e.g. deny a tool and feed the message back as the tool result).
    Block(String),
    /// Stop the whole turn now (budget guard, policy violation). Honored at
    /// `before_llm`; `before_tool` treats it as `Block` this round (see the
    /// loop's dispatch closure).
    Abort(String),
}

#[async_trait]
pub trait AgentHook: Send + Sync {
    /// Called just before each LLM round-trip. May mutate `request` in place
    /// (return `Modify`) to inject context, cap tokens, swap model, etc.
    async fn before_llm(&self, _request: &mut ChatRequest) -> HookDecision {
        HookDecision::Continue
    }

    /// Called after an LLM round-trip completes, with the assistant text and
    /// any tool calls the model produced. Observe-only.
    async fn after_llm(&self, _assistant_text: &str, _calls: &[ToolUse]) {}

    /// Called before dispatching one tool. May mutate `call.input` (return
    /// `Modify`) or deny it (`Block`). This is the approval / rate-limit /
    /// sandbox seam.
    async fn before_tool(&self, _call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
        HookDecision::Continue
    }

    /// Called after a tool finishes, with its full structured outcome
    /// (including `data`). Observe-only — audit, cost accounting, telemetry.
    async fn after_tool(&self, _call: &ToolUse, _outcome: &ToolOutcome) {}
}

/// Compose multiple hooks into one, mirroring `ChainedPromptProvider`.
///
/// `before_*` runs each child in order: the first child returning `Block` or
/// `Abort` short-circuits and that decision is returned; otherwise a `Modify`
/// from any child is propagated (all children still run so each can amend the
/// value). `after_*` simply fans out to every child.
#[derive(Default, Clone)]
pub struct ChainedHook {
    inner: Vec<Arc<dyn AgentHook>>,
}

impl ChainedHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, hook: Arc<dyn AgentHook>) {
        self.inner.push(hook);
    }

    pub fn with(mut self, hook: Arc<dyn AgentHook>) -> Self {
        self.inner.push(hook);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[async_trait]
impl AgentHook for ChainedHook {
    async fn before_llm(&self, request: &mut ChatRequest) -> HookDecision {
        let mut modified = false;
        for h in &self.inner {
            match h.before_llm(request).await {
                HookDecision::Continue => {}
                HookDecision::Modify => modified = true,
                other => return other,
            }
        }
        if modified {
            HookDecision::Modify
        } else {
            HookDecision::Continue
        }
    }

    async fn after_llm(&self, assistant_text: &str, calls: &[ToolUse]) {
        for h in &self.inner {
            h.after_llm(assistant_text, calls).await;
        }
    }

    async fn before_tool(&self, call: &mut ToolUse, ctx: &ToolContext) -> HookDecision {
        let mut modified = false;
        for h in &self.inner {
            match h.before_tool(call, ctx).await {
                HookDecision::Continue => {}
                HookDecision::Modify => modified = true,
                other => return other,
            }
        }
        if modified {
            HookDecision::Modify
        } else {
            HookDecision::Continue
        }
    }

    async fn after_tool(&self, call: &ToolUse, outcome: &ToolOutcome) {
        for h in &self.inner {
            h.after_tool(call, outcome).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts `before_tool` calls and denies the second one.
    struct DenySecond {
        seen: AtomicUsize,
    }

    #[async_trait]
    impl AgentHook for DenySecond {
        async fn before_tool(&self, _call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
            let n = self.seen.fetch_add(1, Ordering::Relaxed);
            if n == 1 {
                HookDecision::Block("denied".into())
            } else {
                HookDecision::Continue
            }
        }
    }

    /// Rewrites the tool input (Modify).
    struct RewriteArgs;

    #[async_trait]
    impl AgentHook for RewriteArgs {
        async fn before_tool(&self, call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
            call.input = json!({"rewritten": true});
            HookDecision::Modify
        }
    }

    fn call() -> ToolUse {
        ToolUse { id: "1".into(), name: "x".into(), input: json!({}) }
    }

    #[tokio::test]
    async fn chained_modify_is_propagated_and_applied() {
        let chain = ChainedHook::new().with(Arc::new(RewriteArgs));
        let mut c = call();
        let d = chain.before_tool(&mut c, &ToolContext::new(".".into())).await;
        assert!(matches!(d, HookDecision::Modify));
        assert_eq!(c.input, json!({"rewritten": true}));
    }

    #[tokio::test]
    async fn chained_block_short_circuits() {
        let deny = Arc::new(DenySecond { seen: AtomicUsize::new(0) });
        let chain = ChainedHook::new().with(deny);
        let ctx = ToolContext::new(".".into());
        // First call continues, second is blocked.
        assert!(matches!(
            chain.before_tool(&mut call(), &ctx).await,
            HookDecision::Continue
        ));
        assert!(matches!(
            chain.before_tool(&mut call(), &ctx).await,
            HookDecision::Block(_)
        ));
    }
}
