//! Long-term memory abstractions.
//!
//! Three layers stacked above the per-session `SessionStore`:
//!
//! - **Facts** — markdown notes (one per topic) usable across sessions.
//!   Written by reflection, by the user, or by the agent's `remember` tool.
//! - **Vectors** — embeddings of arbitrary text plus metadata; used for
//!   semantic recall.
//! - **Embeddings** — `EmbeddingProvider` produces vectors from text. Lives
//!   here so the memory layer can call into whichever LLM the user picked.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("not found: {0}")]
    NotFound(String),

    #[error("backend: {0}")]
    Backend(String),

    #[error("serde: {0}")]
    Serde(String),
}

pub type MemoryResult<T> = std::result::Result<T, MemoryError>;

/// Stable identifier for a fact (slug derived from name on creation).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FactId(pub String);

impl FactId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for FactId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for FactId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// What kind of memory a fact represents. Lets the runtime weight or filter
/// (e.g. always show `Preference`, only inject `Reflection` when relevant).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    /// A user preference or directive ("I prefer Rust over Go").
    Preference,
    /// Context about the current project / codebase / environment.
    Project,
    /// A self-reflection written after a run completes.
    Reflection,
    /// Anything else worth keeping.
    #[default]
    Note,
}

/// A persisted memory note. Storage backend is responsible for choosing the
/// on-disk format (markdown + frontmatter for the default impl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub id: FactId,
    pub name: String,
    pub kind: FactKind,
    #[serde(default)]
    pub tags: Vec<String>,
    pub body: String,
    /// UNIX seconds.
    pub created_at: i64,
    /// UNIX seconds.
    pub updated_at: i64,
}

impl Fact {
    /// Short single-line representation for prompt injection / list views.
    pub fn one_liner(&self) -> String {
        let first_line = self.body.lines().next().unwrap_or("").trim();
        if first_line.is_empty() {
            self.name.clone()
        } else if first_line.len() > 200 {
            format!("{}: {}…", self.name, &first_line[..200])
        } else {
            format!("{}: {}", self.name, first_line)
        }
    }
}

#[async_trait]
pub trait FactStore: Send + Sync {
    /// Insert a new fact (or update if `id` already exists). Returns the id.
    async fn save(&self, fact: NewFact) -> MemoryResult<FactId>;

    /// Fetch one fact.
    async fn get(&self, id: &FactId) -> MemoryResult<Fact>;

    /// All facts (sorted by `updated_at` desc by default).
    async fn list(&self, kind: Option<FactKind>) -> MemoryResult<Vec<Fact>>;

    /// Case-insensitive substring search over name/body. Cheap fallback when
    /// vectors aren't available.
    async fn search(&self, query: &str, limit: usize) -> MemoryResult<Vec<Fact>>;

    async fn delete(&self, id: &FactId) -> MemoryResult<()>;
}

/// Payload for `FactStore::save`. The store assigns `id` from `name` if
/// `id` is `None`.
#[derive(Debug, Clone)]
pub struct NewFact {
    pub id: Option<FactId>,
    pub name: String,
    pub kind: FactKind,
    pub tags: Vec<String>,
    pub body: String,
}

impl NewFact {
    pub fn new(name: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            id: None,
            name: name.into(),
            kind: FactKind::default(),
            tags: Vec::new(),
            body: body.into(),
        }
    }

    pub fn with_kind(mut self, kind: FactKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Embedding-vector hit returned by `VectorStore::search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHit {
    pub key: String,
    pub text: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Persist a text + embedding pair, replacing any prior entry with the
    /// same `key`.
    async fn upsert(
        &self,
        key: &str,
        text: &str,
        embedding: Vec<f32>,
        metadata: serde_json::Value,
    ) -> MemoryResult<()>;

    /// Top-`k` cosine-similar entries.
    async fn search(&self, query_embedding: &[f32], k: usize) -> MemoryResult<Vec<MemoryHit>>;

    async fn delete(&self, key: &str) -> MemoryResult<()>;

    async fn len(&self) -> MemoryResult<usize>;

    async fn is_empty(&self) -> MemoryResult<bool> {
        Ok(self.len().await? == 0)
    }
}

/// Turn text into an embedding vector. Implementations live in `agent-llm`.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn model(&self) -> &str;
    fn dimension(&self) -> usize;

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, crate::llm::LlmError>;
}
