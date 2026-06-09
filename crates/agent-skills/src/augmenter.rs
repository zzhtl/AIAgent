//! `PromptProvider` implementation that combines `RuleSet` (always on) and
//! `SkillRegistry` (selected by trigger keywords) into a single prompt
//! fragment.

use agent_core::prompt::PromptProvider;
use async_trait::async_trait;

use crate::rule::RuleSet;
use crate::skill::SkillRegistry;

#[derive(Default, Debug, Clone)]
pub struct Augmenter {
    rules: RuleSet,
    skills: SkillRegistry,
}

impl Augmenter {
    pub fn new(rules: RuleSet, skills: SkillRegistry) -> Self {
        Self { rules, skills }
    }

    pub fn rules(&self) -> &RuleSet {
        &self.rules
    }

    pub fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && self.skills.is_empty()
    }
}

#[async_trait]
impl PromptProvider for Augmenter {
    async fn system_prompt_for(&self, input: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.rules.is_empty() {
            parts.push(self.rules.merged_system_prompt());
        }
        for skill in self.skills.match_for(input) {
            parts.push(skill.prompt_fragment());
        }
        parts.join("\n\n")
    }

    /// Union of `tools_allowed` across the skills triggered by `input`. Only
    /// skills that declare a non-empty list contribute; if none do, returns
    /// `None` (no restriction).
    async fn tool_whitelist_for(&self, input: &str) -> Option<Vec<String>> {
        let mut union: Vec<String> = Vec::new();
        let mut any = false;
        for skill in self.skills.match_for(input) {
            if skill.tools_allowed.is_empty() {
                continue;
            }
            any = true;
            for t in &skill.tools_allowed {
                if !union.contains(t) {
                    union.push(t.clone());
                }
            }
        }
        any.then_some(union)
    }
}
