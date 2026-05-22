//! Session persistence abstraction.
//!
//! The runtime never touches a database directly; it talks to this trait.
//! `agent-memory` provides the SQLite implementation, but a test harness or
//! a bot integration could supply an in-memory variant just as easily.

use crate::message::{Message, TokenUsage};
use crate::session::SessionId;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SessionStoreError {
    #[error("session not found: {0}")]
    NotFound(String),

    #[error("backend: {0}")]
    Backend(String),

    #[error("serde: {0}")]
    Serde(String),
}

pub type StoreResult<T> = std::result::Result<T, SessionStoreError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: SessionId,
    pub title: Option<String>,
    /// UNIX seconds.
    pub created_at: i64,
    /// UNIX seconds.
    pub updated_at: i64,
    pub message_count: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageSummary {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
    pub cost_estimate_usd: f64,
}

/// A compressed snapshot of an early portion of a session's transcript.
/// Written by a summariser when conversation length grows past a threshold;
/// the next turn can replace those early messages with the summary body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSummary {
    pub id: i64,
    pub body: String,
    /// Highest message row id covered by this summary, when known. `None`
    /// means "everything up to the time of writing".
    pub cutoff_message_id: Option<i64>,
    /// UNIX seconds.
    pub created_at: i64,
}

impl UsageSummary {
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Create a new session row. `title` is a user-facing label (typically
    /// the first ~40 chars of the opening user message); pass `None` to
    /// let the backend leave it null.
    async fn create_session(&self, title: Option<&str>) -> StoreResult<SessionId>;

    /// Append messages in insertion order. System / assistant / tool rows
    /// are all stored as a serialised `ContentBlock` array under `role`.
    async fn append_messages(&self, sid: &SessionId, msgs: &[Message]) -> StoreResult<()>;

    /// Load the full transcript in insertion order.
    async fn load_messages(&self, sid: &SessionId) -> StoreResult<Vec<Message>>;

    /// Most recent sessions first.
    async fn list_sessions(&self, limit: usize) -> StoreResult<Vec<SessionSummary>>;

    /// Optional: update the human-readable title.
    async fn rename_session(&self, sid: &SessionId, title: &str) -> StoreResult<()>;

    /// Record a single LLM round-trip's token consumption and estimated cost.
    async fn record_usage(
        &self,
        sid: &SessionId,
        model: &str,
        tokens: TokenUsage,
        cost_estimate_usd: f64,
    ) -> StoreResult<()>;

    /// Aggregated usage for this session (all rounds).
    async fn session_usage(&self, sid: &SessionId) -> StoreResult<UsageSummary>;

    /// Persist a transcript summary. `cutoff_message_id` is the inclusive
    /// upper bound of message rows this summary replaces; pass `None` if
    /// not tracking row ids. Returns the new summary's row id.
    ///
    /// Default impl is a no-op so legacy stores remain valid; the SQLite
    /// implementation overrides this with a real insert.
    async fn record_summary(
        &self,
        _sid: &SessionId,
        _body: &str,
        _cutoff_message_id: Option<i64>,
    ) -> StoreResult<i64> {
        Err(SessionStoreError::Backend(
            "summarisation not supported by this store".into(),
        ))
    }

    /// All transcript summaries for a session, oldest first.
    async fn list_summaries(&self, _sid: &SessionId) -> StoreResult<Vec<TranscriptSummary>> {
        Ok(Vec::new())
    }

    /// Most recently written summary, if any.
    async fn latest_summary(&self, sid: &SessionId) -> StoreResult<Option<TranscriptSummary>> {
        Ok(self.list_summaries(sid).await?.into_iter().last())
    }
}
