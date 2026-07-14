//! Approval queue for proposed rules and skills.
//!
//! Defines the on-disk JSON format and exposes `enqueue` / `list` / `remove`.
//! Candidates are produced by the `propose_rule` / `propose_skill` meta-tools
//! and by `agent_evolution::Extractor` (which mines accumulated reflections);
//! nothing is installed without an explicit human `apply`.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Error)]
pub enum CandidateError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CandidateError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateKind {
    Rule,
    Skill,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub kind: CandidateKind,
    /// What this candidate would be named once approved.
    pub name: String,
    /// Free-form rationale (why was this proposed).
    pub rationale: String,
    /// Body to write to disk on approval.
    pub body: String,
    pub created_at: i64,
}

/// On-disk queue stored as a single JSON file. `enqueue` / `remove` serialise
/// their read-modify-write cycle through a lock shared by every clone of one
/// queue, so concurrent tool calls in the same process (e.g. parallel web
/// requests) never lose updates. Writers in *different processes* (say,
/// `agent evolution apply` while a server is running) remain unsupported;
/// the atomic rename in `write` only prevents torn reads.
#[derive(Clone)]
pub struct CandidateQueue {
    path: PathBuf,
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl CandidateQueue {
    pub fn open(path: PathBuf) -> Self {
        Self { path, write_lock: Arc::new(tokio::sync::Mutex::new(())) }
    }

    pub async fn list(&self) -> Result<Vec<Candidate>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(&self.path).await?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_slice(&bytes)?)
    }

    pub async fn enqueue(&self, c: Candidate) -> Result<()> {
        let _guard = self.write_lock.lock().await;
        let mut all = self.list().await?;
        all.push(c);
        self.write(&all).await
    }

    pub async fn remove(&self, id: &str) -> Result<Option<Candidate>> {
        let _guard = self.write_lock.lock().await;
        let mut all = self.list().await?;
        let pos = all.iter().position(|c| c.id == id);
        let popped = pos.map(|i| all.remove(i));
        self.write(&all).await?;
        Ok(popped)
    }

    async fn write(&self, all: &[Candidate]) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await?;
            }
        }
        let tmp = self.path.with_extension("json.tmp");
        let serialised = serde_json::to_vec_pretty(all)?;
        {
            let mut f = fs::File::create(&tmp).await?;
            f.write_all(&serialised).await?;
            f.flush().await?;
        }
        fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str) -> Candidate {
        Candidate {
            id: id.to_string(),
            kind: CandidateKind::Rule,
            name: format!("rule-{id}"),
            rationale: "test".into(),
            body: "body".into(),
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn concurrent_enqueue_loses_no_candidates() {
        let dir = std::env::temp_dir().join(format!("agent-queue-{}", uuid::Uuid::new_v4()));
        let queue = CandidateQueue::open(dir.join("queue.json"));

        let tasks: Vec<_> = (0..16)
            .map(|i| {
                let q = queue.clone();
                tokio::spawn(async move { q.enqueue(candidate(&i.to_string())).await })
            })
            .collect();
        for t in tasks {
            t.await.expect("task").expect("enqueue");
        }

        assert_eq!(queue.list().await.expect("list").len(), 16);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
