//! Prompt providers backed by long-term memory.
//!
//! - [`FactsPromptProvider`] dumps every fact into the system prompt every
//!   turn. Trivially simple; fine until the memory grows past a few dozen
//!   entries.
//! - [`VectorRecallPromptProvider`] embeds the user input, searches the
//!   vector store for the top-K closest entries, and injects only those.
//!   Use it once the corpus is large enough that "include everything"
//!   would blow the prompt window.

use std::sync::Arc;

use agent_core::memory::{EmbeddingProvider, FactKind, FactStore, VectorStore};
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

/// Semantic recall over a [`VectorStore`]. On every turn:
///
/// 1. Embed the user input with `embedder.embed`.
/// 2. Pull the top-`k` closest entries from the vector store.
/// 3. Drop anything below `min_score`.
/// 4. Format the survivors as a bulleted list under a "relevant context"
///    header and return it as system-prompt content.
///
/// Failures (embedding API down, empty store, …) collapse to an empty
/// string so the agent loop never breaks because recall stumbled.
#[derive(Clone)]
pub struct VectorRecallPromptProvider {
    embedder: Arc<dyn EmbeddingProvider>,
    store: Arc<dyn VectorStore>,
    top_k: usize,
    min_score: f32,
}

impl VectorRecallPromptProvider {
    pub fn new(embedder: Arc<dyn EmbeddingProvider>, store: Arc<dyn VectorStore>) -> Self {
        Self { embedder, store, top_k: 5, min_score: 0.2 }
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_min_score(mut self, s: f32) -> Self {
        self.min_score = s;
        self
    }
}

#[async_trait]
impl PromptProvider for VectorRecallPromptProvider {
    async fn system_prompt_for(&self, input: &str) -> String {
        if input.trim().is_empty() {
            return String::new();
        }
        // Skip the round-trip when there's nothing to search.
        match self.store.is_empty().await {
            Ok(true) => return String::new(),
            Ok(false) => {}
            Err(e) => {
                tracing::debug!(error = %e, "vector store len() failed");
                return String::new();
            }
        }

        let embeddings = match self.embedder.embed(&[input.to_string()]).await {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => return String::new(),
            Err(e) => {
                tracing::debug!(error = %e, "embedder failed");
                return String::new();
            }
        };
        let query = &embeddings[0];

        let hits = match self.store.search(query, self.top_k).await {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(error = %e, "vector search failed");
                return String::new();
            }
        };
        let hits: Vec<_> = hits
            .into_iter()
            .filter(|h| h.score >= self.min_score)
            .collect();
        if hits.is_empty() {
            return String::new();
        }

        let mut out = String::from("# Relevant context (semantic recall)\n\n");
        for h in &hits {
            let snippet = agent_core::text::truncate_with_ellipsis(&h.text, 300);
            out.push_str(&format!(
                "- ({score:.2}) {snippet}\n",
                score = h.score,
                snippet = snippet
            ));
        }
        out
    }
}

