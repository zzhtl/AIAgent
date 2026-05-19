//! Minimal SQLite-backed `VectorStore`.
//!
//! Embeddings are stored as JSON arrays; similarity is computed in-process at
//! query time. This is fine for tens of thousands of entries and avoids
//! pulling in `sqlite-vec` / FAISS. Swap the impl later if the corpus grows.

use agent_core::memory::{MemoryError, MemoryHit, MemoryResult, VectorStore};
use async_trait::async_trait;
use serde_json::Value;
use sqlx::{Pool, Row, Sqlite};

use crate::sqlite::SqliteSessionStore;

#[derive(Clone)]
pub struct SimpleVectorStore {
    pool: Pool<Sqlite>,
}

impl SimpleVectorStore {
    /// Borrow the connection pool from a SqliteSessionStore. Both stores
    /// share the same database file (migrations create both tables).
    pub fn from_session_store(store: &SqliteSessionStore) -> Self {
        Self { pool: store.pool().clone() }
    }
}

#[async_trait]
impl VectorStore for SimpleVectorStore {
    async fn upsert(
        &self,
        key: &str,
        text: &str,
        embedding: Vec<f32>,
        metadata: Value,
    ) -> MemoryResult<()> {
        let emb_json = serde_json::to_string(&embedding).map_err(|e| MemoryError::Serde(e.to_string()))?;
        let meta_json = serde_json::to_string(&metadata).map_err(|e| MemoryError::Serde(e.to_string()))?;
        let now = now_secs();
        sqlx::query(
            "INSERT INTO vectors (key, text, embedding_json, metadata_json, created_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(key) DO UPDATE SET
                 text = excluded.text,
                 embedding_json = excluded.embedding_json,
                 metadata_json = excluded.metadata_json,
                 created_at = excluded.created_at",
        )
        .bind(key)
        .bind(text)
        .bind(emb_json)
        .bind(meta_json)
        .bind(now)
        .execute(&self.pool)
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize) -> MemoryResult<Vec<MemoryHit>> {
        let rows = sqlx::query("SELECT key, text, embedding_json, metadata_json FROM vectors")
            .fetch_all(&self.pool)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;

        let mut scored: Vec<MemoryHit> = Vec::with_capacity(rows.len());
        for row in rows {
            let key: String = row.try_get("key").map_err(map_sqlx)?;
            let text: String = row.try_get("text").map_err(map_sqlx)?;
            let emb_json: String = row.try_get("embedding_json").map_err(map_sqlx)?;
            let meta_json: String = row.try_get("metadata_json").map_err(map_sqlx)?;
            let embedding: Vec<f32> = serde_json::from_str(&emb_json)
                .map_err(|e| MemoryError::Serde(e.to_string()))?;
            let metadata: Value = serde_json::from_str(&meta_json).unwrap_or(Value::Null);
            let score = cosine(query, &embedding);
            scored.push(MemoryHit { key, text, score, metadata });
        }

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    async fn delete(&self, key: &str) -> MemoryResult<()> {
        sqlx::query("DELETE FROM vectors WHERE key = ?")
            .bind(key)
            .execute(&self.pool)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        Ok(())
    }

    async fn len(&self) -> MemoryResult<usize> {
        let row = sqlx::query("SELECT COUNT(*) AS c FROM vectors")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let c: i64 = row.try_get("c").map_err(map_sqlx)?;
        Ok(c.max(0) as usize)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn map_sqlx(e: sqlx::Error) -> MemoryError {
    MemoryError::Serde(e.to_string())
}

fn now_secs() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
