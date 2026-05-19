//! Skills: markdown capability packs that get injected into the system prompt
//! when a user input matches their trigger keywords.

use std::path::Path;

use tracing::warn;

use crate::error::{Result, SkillError};
use crate::frontmatter;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub triggers: Vec<String>,
    /// Optional whitelist. Empty means "all tools allowed". (Enforcement is
    /// the runtime's job — the registry only carries the list.)
    pub tools_allowed: Vec<String>,
    pub body: String,
}

impl Skill {
    fn from_md(content: &str, fallback_name: &str) -> Self {
        let (fm, body) = frontmatter::split(content);
        let fm = fm.unwrap_or_default();
        let name = fm.get_string("name").unwrap_or(fallback_name).to_string();
        let description = fm.get_string("description").unwrap_or("").to_string();
        let triggers = fm.get_list("triggers");
        let tools_allowed = fm.get_list("tools_allowed");
        Self { name, description, triggers, tools_allowed, body }
    }

    /// Returns `true` if any of the skill's trigger keywords appears in
    /// `input` (case-insensitive substring match).
    pub fn matches(&self, input_lower: &str) -> bool {
        self.triggers
            .iter()
            .any(|t| !t.is_empty() && input_lower.contains(&t.to_lowercase()))
    }

    /// Fragment ready to be appended to a system prompt. Header + body.
    pub fn prompt_fragment(&self) -> String {
        let header = if self.description.is_empty() {
            format!("# Skill: {}\n", self.name)
        } else {
            format!("# Skill: {}\n_{}_\n", self.name, self.description)
        };
        format!("{header}\n{}", self.body.trim_end())
    }
}

#[derive(Default, Debug, Clone)]
pub struct SkillRegistry {
    skills: Vec<Skill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every `*.md` file under `dir`. Missing directory is treated as
    /// "no skills" (returns an empty registry).
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut skills = Vec::new();
        if !dir.exists() {
            return Ok(Self { skills });
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
                    warn!(path = %path.display(), error = %e, "failed to read skill");
                    continue;
                }
            };
            let fallback = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("skill")
                .to_string();
            let skill = Skill::from_md(&content, &fallback);
            if skill.name.is_empty() {
                return Err(SkillError::Parse {
                    file: path.display().to_string(),
                    detail: "empty skill name".into(),
                });
            }
            skills.push(skill);
        }
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { skills })
    }

    pub fn all(&self) -> &[Skill] {
        &self.skills
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Return all skills whose triggers match `input` (case-insensitive).
    pub fn match_for(&self, input: &str) -> Vec<&Skill> {
        let lower = input.to_lowercase();
        self.skills.iter().filter(|s| s.matches(&lower)).collect()
    }
}
