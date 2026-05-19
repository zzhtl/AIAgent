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
}
