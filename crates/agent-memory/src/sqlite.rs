//! SQLite-backed `SessionStore` implementation.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::{
    ContentBlock, Message, Role, SessionId, SessionStore, SessionStoreError, SessionSummary,
    StoreResult, TokenUsage, TranscriptSummary, UsageSummary,
};
use async_trait::async_trait;
use sqlx::{
    sqlite::{SqliteConnectOptions, SqlitePoolOptions},
    Pool, Row, Sqlite,
};

#[derive(Clone)]
pub struct SqliteSessionStore {
    pool: Pool<Sqlite>,
}

impl SqliteSessionStore {
    /// Borrow the underlying connection pool. Used by sibling stores
    /// (e.g. `SimpleVectorStore`) that share the same database file.
    pub fn pool(&self) -> &Pool<Sqlite> {
        &self.pool
    }
}

impl SqliteSessionStore {
    /// Open (or create) a SQLite database at `path` and run migrations.
    pub async fn open(path: &Path) -> StoreResult<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    SessionStoreError::Backend(format!(
                        "create_dir_all {}: {e}",
                        parent.display()
                    ))
                })?;
            }
        }

        let options = SqliteConnectOptions::new()
            .filename(path)
            .create_if_missing(true)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("connect: {e}")))?;

        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("migrate: {e}")))?;

        Ok(Self { pool })
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn parse_role(s: &str) -> Option<Role> {
    Some(match s {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => return None,
    })
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn create_session(&self, title: Option<&str>) -> StoreResult<SessionId> {
        let id = SessionId::new();
        let now = now_secs();
        sqlx::query("INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)")
            .bind(id.as_str())
            .bind(title)
            .bind(now)
            .bind(now)
            .execute(&self.pool)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("insert session: {e}")))?;
        Ok(id)
    }

    async fn append_messages(&self, sid: &SessionId, msgs: &[Message]) -> StoreResult<()> {
        if msgs.is_empty() {
            return Ok(());
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| SessionStoreError::Backend(format!("begin tx: {e}")))?;
        let now = now_secs();
        for msg in msgs {
            let content_json = serde_json::to_string(&msg.content)
                .map_err(|e| SessionStoreError::Serde(e.to_string()))?;
            sqlx::query(
                "INSERT INTO messages (session_id, role, content_json, created_at) VALUES (?, ?, ?, ?)",
            )
            .bind(sid.as_str())
            .bind(role_str(msg.role))
            .bind(content_json)
            .bind(now)
            .execute(&mut *tx)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("insert message: {e}")))?;
        }
        sqlx::query("UPDATE sessions SET updated_at = ? WHERE id = ?")
            .bind(now)
            .bind(sid.as_str())
            .execute(&mut *tx)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("touch session: {e}")))?;
        tx.commit()
            .await
            .map_err(|e| SessionStoreError::Backend(format!("commit: {e}")))?;
        Ok(())
    }

    async fn load_messages(&self, sid: &SessionId) -> StoreResult<Vec<Message>> {
        // Ensure the session exists for clearer errors.
        let exists: Option<i64> =
            sqlx::query_scalar("SELECT 1 FROM sessions WHERE id = ?")
                .bind(sid.as_str())
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| SessionStoreError::Backend(format!("check session: {e}")))?;
        if exists.is_none() {
            return Err(SessionStoreError::NotFound(sid.to_string()));
        }

        let rows = sqlx::query("SELECT role, content_json FROM messages WHERE session_id = ? ORDER BY id ASC")
            .bind(sid.as_str())
            .fetch_all(&self.pool)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("select messages: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let role_s: String = row.try_get("role").map_err(serde_err)?;
            let content_s: String = row.try_get("content_json").map_err(serde_err)?;
            let role = parse_role(&role_s)
                .ok_or_else(|| SessionStoreError::Serde(format!("unknown role `{role_s}`")))?;
            let content: Vec<ContentBlock> = serde_json::from_str(&content_s)
                .map_err(|e| SessionStoreError::Serde(e.to_string()))?;
            out.push(Message { role, content });
        }
        Ok(out)
    }

    async fn list_sessions(&self, limit: usize) -> StoreResult<Vec<SessionSummary>> {
        let rows = sqlx::query(
            "SELECT s.id, s.title, s.created_at, s.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS msg_count
             FROM sessions s
             ORDER BY s.updated_at DESC
             LIMIT ?",
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("list sessions: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.try_get("id").map_err(serde_err)?;
            let title: Option<String> = row.try_get("title").map_err(serde_err)?;
            let created_at: i64 = row.try_get("created_at").map_err(serde_err)?;
            let updated_at: i64 = row.try_get("updated_at").map_err(serde_err)?;
            let msg_count: i64 = row.try_get("msg_count").map_err(serde_err)?;
            out.push(SessionSummary {
                id: SessionId(id),
                title,
                created_at,
                updated_at,
                message_count: msg_count,
            });
        }
        Ok(out)
    }

    async fn rename_session(&self, sid: &SessionId, title: &str) -> StoreResult<()> {
        let affected = sqlx::query("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?")
            .bind(title)
            .bind(now_secs())
            .bind(sid.as_str())
            .execute(&self.pool)
            .await
            .map_err(|e| SessionStoreError::Backend(format!("rename: {e}")))?
            .rows_affected();
        if affected == 0 {
            return Err(SessionStoreError::NotFound(sid.to_string()));
        }
        Ok(())
    }

    async fn record_usage(
        &self,
        sid: &SessionId,
        model: &str,
        tokens: TokenUsage,
        cost_estimate_usd: f64,
    ) -> StoreResult<()> {
        sqlx::query(
            "INSERT INTO usages \
             (session_id, model, prompt_tokens, completion_tokens, cached_tokens, cost_estimate_usd, created_at) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
        )
        .bind(sid.as_str())
        .bind(model)
        .bind(tokens.prompt_tokens as i64)
        .bind(tokens.completion_tokens as i64)
        .bind(tokens.cached_tokens as i64)
        .bind(cost_estimate_usd)
        .bind(now_secs())
        .execute(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("insert usage: {e}")))?;
        Ok(())
    }

    async fn session_usage(&self, sid: &SessionId) -> StoreResult<UsageSummary> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(prompt_tokens), 0) AS pt,
                    COALESCE(SUM(completion_tokens), 0) AS ct,
                    COALESCE(SUM(cached_tokens), 0) AS cached,
                    COALESCE(SUM(cost_estimate_usd), 0) AS cost
             FROM usages WHERE session_id = ?",
        )
        .bind(sid.as_str())
        .fetch_one(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("aggregate usage: {e}")))?;

        let pt: i64 = row.try_get("pt").map_err(serde_err)?;
        let ct: i64 = row.try_get("ct").map_err(serde_err)?;
        let cached: i64 = row.try_get("cached").map_err(serde_err)?;
        let cost: f64 = row.try_get("cost").map_err(serde_err)?;

        Ok(UsageSummary {
            prompt_tokens: pt.max(0) as u64,
            completion_tokens: ct.max(0) as u64,
            cached_tokens: cached.max(0) as u64,
            cost_estimate_usd: cost,
        })
    }

    async fn record_summary(
        &self,
        sid: &SessionId,
        body: &str,
        cutoff_message_id: Option<i64>,
    ) -> StoreResult<i64> {
        let now = now_secs();
        let row = sqlx::query(
            "INSERT INTO summaries (session_id, body, cutoff_message_id, created_at) \
             VALUES (?, ?, ?, ?) RETURNING id",
        )
        .bind(sid.as_str())
        .bind(body)
        .bind(cutoff_message_id)
        .bind(now)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("insert summary: {e}")))?;
        let id: i64 = row.try_get("id").map_err(serde_err)?;
        Ok(id)
    }

    async fn list_summaries(&self, sid: &SessionId) -> StoreResult<Vec<TranscriptSummary>> {
        let rows = sqlx::query(
            "SELECT id, body, cutoff_message_id, created_at \
             FROM summaries WHERE session_id = ? ORDER BY id ASC",
        )
        .bind(sid.as_str())
        .fetch_all(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("list summaries: {e}")))?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: i64 = row.try_get("id").map_err(serde_err)?;
            let body: String = row.try_get("body").map_err(serde_err)?;
            let cutoff: Option<i64> = row.try_get("cutoff_message_id").map_err(serde_err)?;
            let created_at: i64 = row.try_get("created_at").map_err(serde_err)?;
            out.push(TranscriptSummary {
                id,
                body,
                cutoff_message_id: cutoff,
                created_at,
            });
        }
        Ok(out)
    }

    async fn latest_summary(&self, sid: &SessionId) -> StoreResult<Option<TranscriptSummary>> {
        let row = sqlx::query(
            "SELECT id, body, cutoff_message_id, created_at \
             FROM summaries WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        )
        .bind(sid.as_str())
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| SessionStoreError::Backend(format!("latest summary: {e}")))?;
        let Some(row) = row else { return Ok(None) };
        let id: i64 = row.try_get("id").map_err(serde_err)?;
        let body: String = row.try_get("body").map_err(serde_err)?;
        let cutoff: Option<i64> = row.try_get("cutoff_message_id").map_err(serde_err)?;
        let created_at: i64 = row.try_get("created_at").map_err(serde_err)?;
        Ok(Some(TranscriptSummary {
            id,
            body,
            cutoff_message_id: cutoff,
            created_at,
        }))
    }
}

fn serde_err(e: sqlx::Error) -> SessionStoreError {
    SessionStoreError::Serde(e.to_string())
}
