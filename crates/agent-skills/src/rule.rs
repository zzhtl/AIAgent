//! Rules: markdown documents merged unconditionally into the system prompt
//! (global style guides, safety policies, etc.).

use std::path::Path;

use tracing::warn;

use crate::error::Result;
use crate::frontmatter;

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: String,
}

impl Rule {
    fn from_md(content: &str, fallback_name: &str) -> Self {
        let (fm, body) = frontmatter::split(content);
        let name = fm
            .as_ref()
            .and_then(|f| f.get_string("name"))
            .unwrap_or(fallback_name)
            .to_string();
        Self { name, body }
    }
}

#[derive(Default, Debug, Clone)]
pub struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut rules = Vec::new();
        if !dir.exists() {
            return Ok(Self { rules });
        }
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("md") {
                continue;
            }
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "failed to read rule");
                    continue;
                }
            };
            let fallback = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("rule")
                .to_string();
            rules.push(Rule::from_md(&content, &fallback));
        }
        rules.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { rules })
    }

    pub fn all(&self) -> &[Rule] {
        &self.rules
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Concatenate every rule body with a separator, prefixed with a header.
    /// Returns an empty string if no rules are loaded.
    pub fn merged_system_prompt(&self) -> String {
        if self.rules.is_empty() {
            return String::new();
        }
        let mut out = String::from("# Global Rules\n\n");
        for (i, r) in self.rules.iter().enumerate() {
            if i > 0 {
                out.push_str("\n---\n\n");
            }
            out.push_str(r.body.trim_end());
            out.push('\n');
        }
        out
    }
}
