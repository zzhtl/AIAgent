//! `FactsPromptProvider` — injects all known facts into the system prompt
//! for every turn.
//!
//! Stage 5 keeps this trivially simple: dump every fact. When the memory
//! grows past a few dozen entries, swap this for a vector-search variant
//! that ranks facts by relevance to `input`.

use std::sync::Arc;

use agent_core::memory::{FactKind, FactStore};
use agent_core::prompt::PromptProvider;
use async_trait::async_trait;

#[derive(Clone)]
pub struct FactsPromptProvider {
    store: Arc<dyn FactStore>,
    /// Maximum facts to include — keeps prompts bounded.
    max_facts: usize,
}

impl FactsPromptProvider {
    pub fn new(store: Arc<dyn FactStore>) -> Self {
        Self { store, max_facts: 50 }
    }

    pub fn with_max_facts(mut self, max: usize) -> Self {
        self.max_facts = max;
        self
    }
}

#[async_trait]
impl PromptProvider for FactsPromptProvider {
    async fn system_prompt_for(&self, _input: &str) -> String {
        let facts = match self.store.list(None).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "failed to load facts for prompt");
                return String::new();
            }
        };
        if facts.is_empty() {
            return String::new();
        }

        let mut prefs = Vec::new();
        let mut projs = Vec::new();
        let mut reflections = Vec::new();
        let mut notes = Vec::new();
        for f in facts.into_iter().take(self.max_facts) {
            match f.kind {
                FactKind::Preference => prefs.push(f),
                FactKind::Project => projs.push(f),
                FactKind::Reflection => reflections.push(f),
                FactKind::Note => notes.push(f),
            }
        }

        let mut out = String::from("# Persistent memory\n\n");
        out.push_str("The following facts have been remembered across previous sessions. Use them when relevant.\n\n");

        let mut emit = |label: &str, items: &[agent_core::memory::Fact]| {
            if !items.is_empty() {
                out.push_str(&format!("## {label}\n"));
                for f in items {
                    out.push_str(&format!("- **{}** ({}): {}\n", f.name, f.id, f.one_liner()));
                }
                out.push('\n');
            }
        };
        emit("Preferences", &prefs);
        emit("Project context", &projs);
        emit("Past reflections", &reflections);
        emit("Notes", &notes);
        out
    }
}
