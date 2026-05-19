//! Markdown-backed `FactStore`.
//!
//! On-disk layout:
//!
//! ```text
//! <root>/
//! ├── MEMORY.md        # one-line index regenerated on every write
//! └── facts/
//!     ├── user-prefers-bun.md
//!     └── …
//! ```
//!
//! Each fact file:
//!
//! ```text
//! ---
//! name: user prefers bun
//! kind: preference
//! tags: [frontend, build]
//! created_at: 1716000000
//! updated_at: 1716000000
//! ---
//!
//! body text…
//! ```

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::frontmatter;
use agent_core::memory::{
    Fact, FactId, FactKind, FactStore, MemoryError, MemoryResult, NewFact,
};
use async_trait::async_trait;
use tokio::fs;
use tokio::io::AsyncWriteExt;

#[derive(Clone)]
pub struct MarkdownFactStore {
    root: PathBuf,
}

impl MarkdownFactStore {
    /// Open (or create) a store rooted at `<root>/memory/`. The directory is
    /// created lazily on the first write to keep `open` cheap.
    pub fn open(root: PathBuf) -> Self {
        Self { root }
    }

    fn facts_dir(&self) -> PathBuf {
        self.root.join("facts")
    }

    fn index_path(&self) -> PathBuf {
        self.root.join("MEMORY.md")
    }

    fn fact_path(&self, id: &FactId) -> PathBuf {
        self.facts_dir().join(format!("{}.md", id.as_str()))
    }

    async fn ensure_dirs(&self) -> MemoryResult<()> {
        fs::create_dir_all(self.facts_dir())
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))
    }

    async fn rebuild_index(&self) -> MemoryResult<()> {
        let mut facts = self.list(None).await?;
        facts.sort_by(|a, b| a.name.cmp(&b.name));
        let mut body = String::from("# Agent memory index\n\n");
        if facts.is_empty() {
            body.push_str("_no facts yet_\n");
        } else {
            for f in &facts {
                body.push_str(&format!(
                    "- [{name}](facts/{id}.md) — {one_liner}\n",
                    name = f.name,
                    id = f.id,
                    one_liner = first_line_truncated(&f.body, 120),
                ));
            }
        }
        write_atomic(&self.index_path(), body.as_bytes()).await
    }
}

#[async_trait]
impl FactStore for MarkdownFactStore {
    async fn save(&self, fact: NewFact) -> MemoryResult<FactId> {
        self.ensure_dirs().await?;
        let id = fact.id.clone().unwrap_or_else(|| FactId(slugify(&fact.name)));
        if id.as_str().is_empty() {
            return Err(MemoryError::Backend("empty fact id".into()));
        }
        let path = self.fact_path(&id);
        let now = now_secs();

        // Preserve `created_at` if the fact already exists.
        let created_at = match read_fact(&path).await {
            Ok(existing) => existing.created_at,
            Err(_) => now,
        };

        let serialised = serialise_fact(&Fact {
            id: id.clone(),
            name: fact.name,
            kind: fact.kind,
            tags: fact.tags,
            body: fact.body,
            created_at,
            updated_at: now,
        });

        write_atomic(&path, serialised.as_bytes()).await?;
        self.rebuild_index().await?;
        Ok(id)
    }

    async fn get(&self, id: &FactId) -> MemoryResult<Fact> {
        read_fact(&self.fact_path(id))
            .await
            .map_err(|e| match e {
                MemoryError::Backend(s) if s.contains("not found") => {
                    MemoryError::NotFound(id.to_string())
                }
                other => other,
            })
    }

    async fn list(&self, kind: Option<FactKind>) -> MemoryResult<Vec<Fact>> {
        let dir = self.facts_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut entries = fs::read_dir(&dir)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        let mut out = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            match read_fact(&path).await {
                Ok(f) => {
                    if kind.map(|k| k == f.kind).unwrap_or(true) {
                        out.push(f);
                    }
                }
                Err(e) => tracing::warn!(path = %path.display(), error = %e, "skipping bad fact file"),
            }
        }
        out.sort_by_key(|f| std::cmp::Reverse(f.updated_at));
        Ok(out)
    }

    async fn search(&self, query: &str, limit: usize) -> MemoryResult<Vec<Fact>> {
        let lower = query.to_lowercase();
        let mut hits = self.list(None).await?;
        hits.retain(|f| {
            f.name.to_lowercase().contains(&lower) || f.body.to_lowercase().contains(&lower)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    async fn delete(&self, id: &FactId) -> MemoryResult<()> {
        let path = self.fact_path(id);
        if !path.exists() {
            return Err(MemoryError::NotFound(id.to_string()));
        }
        fs::remove_file(&path)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        self.rebuild_index().await?;
        Ok(())
    }
}

async fn read_fact(path: &Path) -> MemoryResult<Fact> {
    let content = fs::read_to_string(path)
        .await
        .map_err(|e| MemoryError::Backend(format!("not found: {}: {e}", path.display())))?;
    let (fm, body) = frontmatter::split(&content);
    let fm = fm.unwrap_or_default();
    let id_str = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    let name = fm.get_string("name").unwrap_or(&id_str).to_string();
    let kind = match fm.get_string("kind").unwrap_or("note") {
        "preference" => FactKind::Preference,
        "project" => FactKind::Project,
        "reflection" => FactKind::Reflection,
        _ => FactKind::Note,
    };
    let tags = fm.get_list("tags");
    let created_at = fm
        .get_string("created_at")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(0);
    let updated_at = fm
        .get_string("updated_at")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(created_at);
    Ok(Fact {
        id: FactId(id_str),
        name,
        kind,
        tags,
        body,
        created_at,
        updated_at,
    })
}

fn serialise_fact(f: &Fact) -> String {
    let kind = match f.kind {
        FactKind::Preference => "preference",
        FactKind::Project => "project",
        FactKind::Reflection => "reflection",
        FactKind::Note => "note",
    };
    let mut out = String::from("---\n");
    out.push_str(&format!("name: {}\n", f.name));
    out.push_str(&format!("kind: {kind}\n"));
    if !f.tags.is_empty() {
        out.push_str("tags:\n");
        for t in &f.tags {
            out.push_str(&format!("  - {t}\n"));
        }
    }
    out.push_str(&format!("created_at: {}\n", f.created_at));
    out.push_str(&format!("updated_at: {}\n", f.updated_at));
    out.push_str("---\n\n");
    out.push_str(f.body.trim_end());
    out.push('\n');
    out
}

async fn write_atomic(path: &Path, bytes: &[u8]) -> MemoryResult<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| MemoryError::Backend(e.to_string()))?;
        }
    }
    let tmp = path.with_extension("md.tmp");
    {
        let mut f = fs::File::create(&tmp)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        f.write_all(bytes)
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
        f.flush()
            .await
            .map_err(|e| MemoryError::Backend(e.to_string()))?;
    }
    fs::rename(&tmp, path)
        .await
        .map_err(|e| MemoryError::Backend(e.to_string()))?;
    Ok(())
}

fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            prev_dash = false;
        } else if c.is_alphanumeric() {
            // Keep non-ASCII letters as-is (CJK, etc.).
            out.push(c);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn first_line_truncated(body: &str, max_chars: usize) -> String {
    let line = body.lines().next().unwrap_or("").trim();
    if line.chars().count() <= max_chars {
        line.to_string()
    } else {
        let head: String = line.chars().take(max_chars).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_basic() {
        assert_eq!(slugify("Hello World"), "hello-world");
        assert_eq!(slugify("用户偏好 Bun"), "用户偏好-bun");
        assert_eq!(slugify("foo!!!bar"), "foo-bar");
        assert_eq!(slugify("---a---"), "a");
    }
}
