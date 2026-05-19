//! Approval queue for proposed rules and skills.
//!
//! Stage-5 skeleton: defines the on-disk JSON format and exposes
//! `enqueue` / `list` / `pop`. The actual extractor / synthesizer that
//! decides *what* becomes a candidate is left for a later iteration.

use std::path::PathBuf;

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

/// On-disk queue stored as a single JSON file. Concurrent writers should
/// not exist in the CLI use case.
#[derive(Clone)]
pub struct CandidateQueue {
    path: PathBuf,
}

impl CandidateQueue {
    pub fn open(path: PathBuf) -> Self {
        Self { path }
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
        let mut all = self.list().await?;
        all.push(c);
        self.write(&all).await
    }

    pub async fn remove(&self, id: &str) -> Result<Option<Candidate>> {
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
