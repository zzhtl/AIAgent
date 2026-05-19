//! Reflector: after a run completes, ask the LLM to summarise what happened
//! and write a `Reflection`-kind fact to long-term memory.
//!
//! Reflections are *text only* — they cannot execute, install rules, or
//! modify configuration. Even if the model produces noise, the worst-case
//! outcome is a junk markdown file the user can grep / delete.

use std::sync::Arc;

use agent_core::llm::{ChatRequest, LlmEvent, LlmProvider};
use agent_core::memory::{FactId, FactKind, FactStore, NewFact};
use agent_core::message::{Message, Role};
use futures::StreamExt;
use tracing::{debug, warn};

const SYSTEM_PROMPT: &str = "你是一个反思代理。读完下面这段对话记录，写一段简短的反思笔记，目的是把可重复的经验写下来，让未来的同类任务更顺利。\n\n输出格式（markdown）：\n- 第一行：标题（不超过 12 字，描述本次任务核心）\n- 后续：分要点写\n  - 任务目标\n  - 关键过程或工具调用\n  - 哪里顺利、哪里不顺利\n  - 下次同类任务应注意什么\n\n不要重复对话原文。不要超过 200 字。";

pub struct Reflector {
    llm: Arc<dyn LlmProvider>,
    model: String,
    fact_store: Arc<dyn FactStore>,
}

impl Reflector {
    pub fn new(llm: Arc<dyn LlmProvider>, model: impl Into<String>, fact_store: Arc<dyn FactStore>) -> Self {
        Self { llm, model: model.into(), fact_store }
    }

    /// Generate one reflection and persist it. Returns the fact id on
    /// success; logs and returns `None` on any failure (reflection is
    /// best-effort — it must not break the user's session).
    pub async fn reflect(&self, transcript: &[Message]) -> Option<FactId> {
        if transcript.iter().filter(|m| matches!(m.role, Role::User | Role::Assistant)).count() < 2 {
            // Not enough material to reflect on.
            return None;
        }

        let user_summary = format_transcript(transcript);
        debug!(model = %self.model, "generating reflection");

        let request = ChatRequest::new(
            self.model.clone(),
            vec![
                Message::system(SYSTEM_PROMPT),
                Message::user(user_summary),
            ],
        );

        let mut stream = match self.llm.chat_stream(request).await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "reflection llm call failed");
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
                    warn!(error = %e, "reflection stream error");
                    return None;
                }
            }
        }

        let text = text.trim();
        if text.is_empty() {
            return None;
        }

        let (title, body) = split_title(text);
        let title = title.unwrap_or_else(default_title);
        let new_fact = NewFact::new(title, body)
            .with_kind(FactKind::Reflection)
            .with_tags(vec!["auto-reflection".into()]);
        match self.fact_store.save(new_fact).await {
            Ok(id) => {
                debug!(%id, "reflection saved");
                Some(id)
            }
            Err(e) => {
                warn!(error = %e, "failed to save reflection");
                None
            }
        }
    }
}

fn format_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    out.push_str("下面是一次对话记录，按时间顺序：\n\n");
    for m in messages {
        let label = match m.role {
            Role::System => continue,
            Role::User => "用户",
            Role::Assistant => "助手",
            Role::Tool => "工具结果",
        };
        let text = m.text();
        if text.is_empty() {
            // Look for tool uses on assistant rows.
            let tool_names: Vec<String> = m
                .content
                .iter()
                .filter_map(|b| match b {
                    agent_core::ContentBlock::ToolUse(tu) => Some(tu.name.clone()),
                    _ => None,
                })
                .collect();
            if !tool_names.is_empty() {
                out.push_str(&format!("[{label} 调用工具: {}]\n\n", tool_names.join(", ")));
            }
            continue;
        }
        let trimmed = if text.len() > 600 {
            format!("{}…", &text[..600])
        } else {
            text
        };
        out.push_str(&format!("[{label}]\n{trimmed}\n\n"));
    }
    out
}

fn split_title(text: &str) -> (Option<String>, String) {
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("").trim();
    let title = if first.is_empty() {
        None
    } else {
        let cleaned = first
            .trim_start_matches('#')
            .trim_start_matches('-')
            .trim();
        if cleaned.is_empty() {
            None
        } else {
            Some(cleaned.chars().take(40).collect::<String>())
        }
    };
    let body: String = lines.collect::<Vec<_>>().join("\n");
    let body = if body.trim().is_empty() {
        text.to_string()
    } else {
        body
    };
    (title, body)
}

fn default_title() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("reflection-{now}")
}
