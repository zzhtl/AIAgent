//! Dynamic prompt augmentation.
//!
//! Lets the runtime ask an external component (skills, memory, retrieval,
//! etc.) for extra system-prompt content tailored to the current user
//! input. The trait is async so implementations can hit storage / vector
//! search without blocking the run loop.

use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait PromptProvider: Send + Sync {
    /// Return any additional system-prompt content to inject for this turn.
    /// Empty string means "nothing to add".
    async fn system_prompt_for(&self, input: &str) -> String;

    /// Tools the runtime is allowed to dispatch for this turn. `None` means
    /// "no restriction" (the default); `Some(list)` confines the model to
    /// `list` (plus the runtime's always-on infrastructure tools). Skills
    /// declaring `tools_allowed` drive this.
    async fn tool_whitelist_for(&self, _input: &str) -> Option<Vec<String>> {
        None
    }
}

/// Compose multiple providers into one. Their outputs are concatenated in
/// order with blank lines between non-empty fragments. Stage-3 skills and
/// stage-5 memory providers both plug in via this.
#[derive(Default, Clone)]
pub struct ChainedPromptProvider {
    inner: Vec<Arc<dyn PromptProvider>>,
}

impl ChainedPromptProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, provider: Arc<dyn PromptProvider>) {
        self.inner.push(provider);
    }

    pub fn with(mut self, provider: Arc<dyn PromptProvider>) -> Self {
        self.inner.push(provider);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[async_trait]
impl PromptProvider for ChainedPromptProvider {
    async fn system_prompt_for(&self, input: &str) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(self.inner.len());
        for p in &self.inner {
            let fragment = p.system_prompt_for(input).await;
            if !fragment.trim().is_empty() {
                parts.push(fragment);
            }
        }
        parts.join("\n\n")
    }

    /// Union of every child's whitelist. If no child restricts, returns
    /// `None`; otherwise the de-duplicated union of all `Some` lists.
    async fn tool_whitelist_for(&self, input: &str) -> Option<Vec<String>> {
        let mut union: Vec<String> = Vec::new();
        let mut any = false;
        for p in &self.inner {
            if let Some(list) = p.tool_whitelist_for(input).await {
                any = true;
                for t in list {
                    if !union.contains(&t) {
                        union.push(t);
                    }
                }
            }
        }
        any.then_some(union)
    }
}
