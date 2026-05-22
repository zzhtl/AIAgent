//! Transcript summariser: compress an early window of a conversation into a
//! short narrative so the next turn can keep context without re-sending the
//! whole history to the model.
//!
//! Best-effort: any failure returns `None` and the caller falls back to
//! sending the un-compressed transcript.

use std::sync::Arc;

use agent_core::llm::{ChatRequest, LlmEvent, LlmProvider};
use agent_core::message::{Message, Role};
use futures::StreamExt;
use tracing::{debug, warn};

const SYSTEM_PROMPT: &str =
    "你是会话摘要器。请把下面这段对话压缩成一段简短的中文笔记，保留：\n\
     - 用户的核心目标\n\
     - 关键决策、约束、人物或文件名等具体事实\n\
     - 已经完成或排除的尝试\n\n\
     输出要求：\n\
     - 不超过 400 字，不分章节\n\
     - 不要复述原文，不要写 Markdown 列表前缀\n\
     - 用第三人称，便于下一轮对话作为系统提示注入";

pub struct Summariser {
    llm: Arc<dyn LlmProvider>,
    model: String,
}

impl Summariser {
    pub fn new(llm: Arc<dyn LlmProvider>, model: impl Into<String>) -> Self {
        Self { llm, model: model.into() }
    }

    /// Compress `transcript` into a single narrative paragraph. Returns
    /// `None` when the input is empty / too short or the LLM call fails.
    pub async fn summarise(&self, transcript: &[Message]) -> Option<String> {
        let dialogue = format_transcript(transcript);
        if dialogue.trim().is_empty() {
            return None;
        }
        debug!(model = %self.model, "summarising transcript");

        let request = ChatRequest::new(
            self.model.clone(),
            vec![
                Message::system(SYSTEM_PROMPT),
                Message::user(dialogue),
            ],
        );

        let mut stream = match self.llm.chat_stream(request).await {
            Ok(s) => s,
            Err(e) => {
                warn!(error = %e, "summariser llm call failed");
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
                    warn!(error = %e, "summariser stream error");
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
}

fn format_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let label = match m.role {
            Role::System => continue,
            Role::User => "用户",
            Role::Assistant => "助手",
            Role::Tool => "工具结果",
        };
        let text = m.text();
        let trimmed = agent_core::text::truncate_with_ellipsis(&text, 400);
        if trimmed.is_empty() {
            continue;
        }
        out.push_str(&format!("[{label}]\n{trimmed}\n\n"));
    }
    out
}
