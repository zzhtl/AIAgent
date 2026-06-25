//! Extractor: mine accumulated reflections for recurring patterns and propose
//! them as Rule / Skill candidates onto the review queue.
//!
//! This is the "rule extractor / skill synthesizer" half the evolution module
//! reserved. Auto-written reflections (text-only) are handed to the LLM, which
//! proposes rule/skill candidates as JSON. They enter the *same* human-approval
//! queue as the `propose_*` tools — nothing is installed automatically, so a
//! noisy model can at worst produce queue entries the user rejects.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::evolution::{Candidate, CandidateKind};
use agent_core::llm::{ChatRequest, LlmEvent, LlmProvider};
use agent_core::memory::{Fact, FactKind, FactStore};
use agent_core::message::Message;
use futures::StreamExt;
use serde::Deserialize;
use tracing::{debug, warn};
use uuid::Uuid;

/// Need at least this many reflections before extraction is worthwhile.
const MIN_REFLECTIONS: usize = 3;
/// Cap how many reflections we feed the model (most recent first).
const MAX_REFLECTIONS: usize = 40;

const SYSTEM_PROMPT: &str = "你是一个经验提炼代理。下面是一批由 agent 在历次任务后自动写下的反思笔记。\
请从中找出**反复出现、值得固化**的模式，提议成「规则」或「技能」候选，供人工审核。\n\n\
- 规则(rule)：无条件、全局生效的约定（如编码规范、安全约束、沟通偏好）。\n\
- 技能(skill)：由关键词触发的能力包（如「代码审查」流程），需给出 triggers。\n\n\
只提议确有重复证据的项；没有就返回空数组。最多 3 条。\n\n\
**只输出一个 JSON 数组**，每项形如：\n\
{\"kind\":\"rule|skill\",\"name\":\"短标题\",\"rationale\":\"为什么提议(引用反复出现的证据)\",\
\"body\":\"markdown 正文\",\"triggers\":[\"仅 skill 需要\"]}\n\
不要输出 JSON 以外的任何文字。";

#[derive(Debug, Deserialize)]
struct Proposal {
    kind: String,
    name: String,
    #[serde(default)]
    rationale: String,
    body: String,
    #[serde(default)]
    triggers: Vec<String>,
}

pub struct Extractor {
    llm: Arc<dyn LlmProvider>,
    model: String,
    fact_store: Arc<dyn FactStore>,
}

impl Extractor {
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        model: impl Into<String>,
        fact_store: Arc<dyn FactStore>,
    ) -> Self {
        Self { llm, model: model.into(), fact_store }
    }

    /// Analyze reflections and return proposed candidates (the caller enqueues
    /// them). Best-effort: returns empty on any failure or when there's too
    /// little material.
    pub async fn extract(&self) -> Vec<Candidate> {
        let reflections = match self.fact_store.list(Some(FactKind::Reflection)).await {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, "extractor: listing reflections failed");
                return Vec::new();
            }
        };
        if reflections.len() < MIN_REFLECTIONS {
            debug!(count = reflections.len(), "extractor: too few reflections to extract");
            return Vec::new();
        }

        let corpus = format_reflections(&reflections);
        let request = ChatRequest::new(
            self.model.clone(),
            vec![Message::system(SYSTEM_PROMPT), Message::user(corpus)],
        );

        let text = match collect_text(self.llm.as_ref(), request).await {
            Some(t) => t,
            None => return Vec::new(),
        };

        let proposals = parse_proposals(&text);
        let candidates: Vec<Candidate> = proposals.into_iter().filter_map(to_candidate).collect();
        debug!(count = candidates.len(), "extractor: produced candidates");
        candidates
    }
}

fn format_reflections(reflections: &[Fact]) -> String {
    let mut out = String::from("以下是历次反思笔记：\n\n");
    for (i, f) in reflections.iter().take(MAX_REFLECTIONS).enumerate() {
        let body = agent_core::text::truncate_with_ellipsis(f.body.trim(), 500);
        out.push_str(&format!("## 反思 {} — {}\n{}\n\n", i + 1, f.name, body));
    }
    out
}

async fn collect_text(llm: &dyn LlmProvider, request: ChatRequest) -> Option<String> {
    let mut stream = match llm.chat_stream(request).await {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "extractor: llm call failed");
            return None;
        }
    };
    let mut text = String::new();
    while let Some(ev) = stream.next().await {
        match ev {
            Ok(LlmEvent::TextDelta { delta }) => text.push_str(&delta),
            Ok(LlmEvent::End(_)) => break,
            Ok(_) => {}
            Err(e) => {
                warn!(error = %e, "extractor: stream error");
                return None;
            }
        }
    }
    let text = text.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Pull the first JSON array out of a (possibly fenced) LLM response and parse
/// it into proposals. Returns empty on any parse failure.
fn parse_proposals(text: &str) -> Vec<Proposal> {
    let (Some(start), Some(end)) = (text.find('['), text.rfind(']')) else {
        return Vec::new();
    };
    if end < start {
        return Vec::new();
    }
    match serde_json::from_str::<Vec<Proposal>>(&text[start..=end]) {
        Ok(v) => v,
        Err(e) => {
            warn!(error = %e, "extractor: could not parse proposals JSON");
            Vec::new()
        }
    }
}

fn to_candidate(p: Proposal) -> Option<Candidate> {
    if p.name.trim().is_empty() || p.body.trim().is_empty() {
        return None;
    }
    let kind = match p.kind.to_ascii_lowercase().as_str() {
        "rule" => CandidateKind::Rule,
        "skill" => CandidateKind::Skill,
        _ => return None,
    };
    // For skills, embed name + triggers as frontmatter so `evolution apply`
    // writes a complete skill file straight from the candidate (matches the
    // `propose_skill` tool's format).
    let body = match kind {
        CandidateKind::Rule => p.body.trim_end().to_string(),
        CandidateKind::Skill => {
            let mut b = String::from("---\n");
            b.push_str(&format!("name: {}\n", p.name));
            if !p.triggers.is_empty() {
                b.push_str("triggers:\n");
                for t in &p.triggers {
                    b.push_str(&format!("  - {t}\n"));
                }
            }
            b.push_str("---\n\n");
            b.push_str(p.body.trim_end());
            b.push('\n');
            b
        }
    };
    Some(Candidate {
        id: Uuid::new_v4().to_string(),
        kind,
        name: p.name,
        rationale: p.rationale,
        body,
        created_at: now_secs(),
    })
}

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::llm::{LlmEventStream, LlmResult, ProviderCapabilities};
    use agent_core::memory::{FactId, MemoryError, MemoryResult, NewFact};
    use async_trait::async_trait;

    /// Returns a canned JSON array (wrapped in a markdown fence to exercise the
    /// lenient extractor).
    struct CannedLlm(String);

    #[async_trait]
    impl LlmProvider for CannedLlm {
        fn name(&self) -> &str {
            "fake"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities { streaming: true, ..Default::default() }
        }
        async fn chat_stream(&self, _req: ChatRequest) -> LlmResult<LlmEventStream> {
            let text = self.0.clone();
            let events: Vec<LlmResult<LlmEvent>> = vec![
                Ok(LlmEvent::TextDelta { delta: text }),
                Ok(LlmEvent::End(agent_core::StopReason::EndTurn)),
            ];
            Ok(futures::stream::iter(events).boxed())
        }
    }

    struct FakeFacts {
        facts: Vec<Fact>,
    }

    #[async_trait]
    impl FactStore for FakeFacts {
        async fn save(&self, _f: NewFact) -> MemoryResult<FactId> {
            Ok(FactId::from("x"))
        }
        async fn get(&self, _id: &FactId) -> MemoryResult<Fact> {
            Err(MemoryError::NotFound("x".into()))
        }
        async fn list(&self, _kind: Option<FactKind>) -> MemoryResult<Vec<Fact>> {
            Ok(self.facts.clone())
        }
        async fn search(&self, _q: &str, _l: usize) -> MemoryResult<Vec<Fact>> {
            Ok(Vec::new())
        }
        async fn delete(&self, _id: &FactId) -> MemoryResult<()> {
            Ok(())
        }
    }

    fn reflection(name: &str) -> Fact {
        Fact {
            id: FactId::from(name),
            name: name.to_string(),
            kind: FactKind::Reflection,
            tags: vec![],
            body: format!("body of {name}"),
            created_at: 0,
            updated_at: 0,
        }
    }

    #[tokio::test]
    async fn extracts_rule_and_skill_from_reflections() {
        let json = r#"```json
[
  {"kind":"rule","name":"用中文回复","rationale":"多次出现","body":"始终用中文回复用户。"},
  {"kind":"skill","name":"代码审查","rationale":"反复审查","body":"系统化审查代码。","triggers":["审查","review"]}
]
```"#;
        let facts = vec![reflection("r1"), reflection("r2"), reflection("r3")];
        let ex = Extractor::new(
            Arc::new(CannedLlm(json.to_string())),
            "fake",
            Arc::new(FakeFacts { facts }),
        );

        let cands = ex.extract().await;
        assert_eq!(cands.len(), 2);
        assert_eq!(cands[0].kind, CandidateKind::Rule);
        assert_eq!(cands[0].name, "用中文回复");
        assert_eq!(cands[1].kind, CandidateKind::Skill);
        // Skill body carries frontmatter with triggers.
        assert!(cands[1].body.contains("triggers:"), "got: {}", cands[1].body);
        assert!(cands[1].body.contains("- 审查"));
    }

    #[tokio::test]
    async fn returns_empty_when_too_few_reflections() {
        let ex = Extractor::new(
            Arc::new(CannedLlm("[]".to_string())),
            "fake",
            Arc::new(FakeFacts { facts: vec![reflection("only-one")] }),
        );
        assert!(ex.extract().await.is_empty());
    }

    #[tokio::test]
    async fn tolerates_garbage_llm_output() {
        let facts = vec![reflection("r1"), reflection("r2"), reflection("r3")];
        let ex = Extractor::new(
            Arc::new(CannedLlm("sorry, I can't help".to_string())),
            "fake",
            Arc::new(FakeFacts { facts }),
        );
        assert!(ex.extract().await.is_empty());
    }
}
