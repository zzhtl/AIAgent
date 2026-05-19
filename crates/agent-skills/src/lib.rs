//! agent-skills
//!
//! Two layers of prompt augmentation:
//!
//! - **Skill** — a markdown file with YAML frontmatter (`name`, `description`,
//!   `triggers`, `tools_allowed`). When a user input matches any of the
//!   skill's trigger keywords (case-insensitive substring), its body is
//!   appended to the system prompt for that turn.
//! - **Rule** — a markdown file merged unconditionally into the system
//!   prompt (global style guides, safety policies, etc.).
//!
//! Files live under `~/.config/agent/skills/*.md` and `~/.config/agent/rules/*.md`.

pub mod augmenter;
pub mod error;
pub mod rule;
pub mod skill;

// Re-export shared util so existing call sites keep working.
pub use agent_core::frontmatter;

pub use augmenter::Augmenter;
pub use error::{Result, SkillError};
pub use rule::{Rule, RuleSet};
pub use skill::{Skill, SkillRegistry};
