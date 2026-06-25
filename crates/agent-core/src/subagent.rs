//! Wrap an `Agent` so a parent agent can invoke it as a `Tool`.
//!
//! This is the seam multi-agent orchestration grows from: a specialist
//! sub-agent (researcher, reviewer, planner) is exposed to a coordinator agent
//! as just another tool. The sub-agent runs its own loop to completion; only
//! its final text — plus a small structured summary in `ToolOutcome::data` —
//! returns to the parent. The token stream is *collapsed*: the parent sees a
//! single tool result, not the sub-agent's intermediate deltas, so the parent
//! transcript stays clean.
//!
//! This module is a foundation skeleton: it is intentionally not wired into the
//! CLI/bot yet. Cancellation propagation and recursion-depth guarding (via a
//! counter in `ToolContext` extensions) are left for the multi-agent batch.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::agent::Agent;
use crate::channel::{AgentEvent, UserInput};
use crate::extensions::Extensions;
use crate::message::StopReason;
use crate::session::SessionId;
use crate::tool::{Tool, ToolContext, ToolOutcome, ToolResult};

/// Recursion-depth marker carried through `ToolContext` extensions so nested
/// sub-agents can't spin up an unbounded tree. Each `SubAgentTool` reads the
/// parent's depth and runs its child at depth + 1.
#[derive(Debug, Clone, Copy)]
pub struct SubAgentDepth(pub usize);

/// Default cap on how deep sub-agent nesting may go.
const DEFAULT_MAX_DEPTH: usize = 4;

/// A child `Agent` presented to a parent agent as a single-input tool.
pub struct SubAgentTool {
    name: String,
    description: String,
    agent: Agent,
    /// When true, each call runs in a fresh session (stateless). When false,
    /// calls share one session derived from the parent so the sub-agent
    /// remembers prior turns.
    new_session: bool,
    /// Maximum nesting depth before a call is refused (recursion guard).
    max_depth: usize,
}

impl SubAgentTool {
    /// Stateless sub-agent: a fresh session per call.
    pub fn new(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent,
            new_session: true,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Stateful sub-agent: calls share one session so it remembers across
    /// invocations within a parent run.
    pub fn stateful(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent,
            new_session: false,
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Override the recursion-depth cap (default 4).
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }
}

#[async_trait]
impl Tool for SubAgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "task": { "type": "string", "description": "The task for the sub-agent to complete." }
            },
            "required": ["task"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let task = args.get("task").and_then(|v| v.as_str()).unwrap_or_default();
        if task.is_empty() {
            return Ok(ToolOutcome::error("sub-agent: missing `task` argument"));
        }

        // Recursion guard: refuse to nest deeper than `max_depth`.
        let depth = ctx.get_ext::<SubAgentDepth>().map(|d| d.0).unwrap_or(0);
        if depth >= self.max_depth {
            return Ok(ToolOutcome::error(format!(
                "sub-agent `{}`: max nesting depth {} reached",
                self.name, self.max_depth
            )));
        }

        // Fresh session per call, or one derived from the parent for a stateful
        // sub-agent. The sub-agent runs its own ReAct loop to completion.
        let sid = if self.new_session {
            SessionId::new()
        } else {
            SessionId::from(format!("{}::{}", ctx.session_id, self.name))
        };

        // Run the child at depth + 1, inheriting the parent's cancel flag so a
        // cancelled turn stops the whole sub-agent tree. The child's own tools
        // see the incremented depth in their context.
        let mut child_ext = Extensions::new();
        child_ext.insert(SubAgentDepth(depth + 1));
        let child = self.agent.clone_with_extensions(child_ext);
        let cancel = ctx.cancel.clone().unwrap_or_else(|| Arc::new(AtomicBool::new(false)));

        // Collapse the stream: accumulate the final text, count tool calls, and
        // capture the stop reason. The parent never sees intermediate deltas.
        let mut stream = child.run_cancellable(sid, Vec::new(), UserInput::new(task), cancel);
        let mut final_text = String::new();
        let mut tool_calls = 0usize;
        let mut total_tokens: u32 = 0;
        let mut reason: Option<StopReason> = None;
        while let Some(ev) = stream.next().await {
            match ev {
                AgentEvent::TextDelta { delta } => final_text.push_str(&delta),
                AgentEvent::ToolCallResult { .. } => tool_calls += 1,
                AgentEvent::UsageReport { usage, .. } => {
                    total_tokens = total_tokens.saturating_add(usage.total());
                }
                AgentEvent::Done { reason: r, .. } => reason = Some(r),
                _ => {}
            }
        }

        // Structured summary for the parent (programmatic / audit use); the LLM
        // only ever sees `final_text`.
        let data = json!({
            "stop_reason": reason.map(|r| format!("{r:?}")),
            "tool_calls": tool_calls,
            "total_tokens": total_tokens,
        });
        Ok(ToolOutcome::ok(final_text).with_data(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{
        ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult, ProviderCapabilities,
    };
    use crate::message::TokenUsage;
    use crate::tool::ToolRegistry;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn fake_agent() -> Agent {
        Agent::builder()
            .with_llm(Arc::new(DoneLlm))
            .with_model("fake")
            .with_tools(ToolRegistry::new())
            .build()
            .expect("agent builds")
    }

    /// Emits one text delta + usage, then ends — no tools.
    struct DoneLlm;

    #[async_trait]
    impl LlmProvider for DoneLlm {
        fn name(&self) -> &str {
            "fake"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities { streaming: true, tools: true, ..Default::default() }
        }
        async fn chat_stream(&self, _req: ChatRequest) -> LlmResult<LlmEventStream> {
            let events: Vec<LlmResult<LlmEvent>> = vec![
                Ok(LlmEvent::TextDelta { delta: "sub result".into() }),
                Ok(LlmEvent::Usage(TokenUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    cached_tokens: 0,
                })),
                Ok(LlmEvent::End(StopReason::EndTurn)),
            ];
            Ok(futures::stream::iter(events).boxed())
        }
    }

    #[tokio::test]
    async fn subagent_collapses_to_single_outcome() {
        let agent = Agent::builder()
            .with_llm(Arc::new(DoneLlm))
            .with_model("fake")
            .with_tools(ToolRegistry::new())
            .build()
            .expect("agent builds");
        let tool = SubAgentTool::new("researcher", "does research", agent);
        let ctx = ToolContext::new(".".into());

        let outcome = tool.invoke(json!({ "task": "go" }), &ctx).await.expect("invoke ok");
        assert_eq!(outcome.text, "sub result");
        assert!(!outcome.is_error);
        let data = outcome.data.expect("structured summary should be attached");
        assert_eq!(data["total_tokens"], 15, "token usage should be summed");
        assert_eq!(data["stop_reason"], "EndTurn");
    }

    #[tokio::test]
    async fn subagent_rejects_missing_task() {
        let agent = Agent::builder()
            .with_llm(Arc::new(DoneLlm))
            .with_model("fake")
            .with_tools(ToolRegistry::new())
            .build()
            .expect("agent builds");
        let tool = SubAgentTool::new("researcher", "does research", agent);
        let ctx = ToolContext::new(".".into());

        let outcome = tool.invoke(json!({}), &ctx).await.expect("invoke ok");
        assert!(outcome.is_error);
    }

    #[tokio::test]
    async fn subagent_blocks_when_depth_exceeded() {
        let tool = SubAgentTool::new("researcher", "x", fake_agent()).with_max_depth(2);
        // A context already at depth 2 must be refused.
        let mut ext = Extensions::new();
        ext.insert(SubAgentDepth(2));
        let ctx = ToolContext::new(".".into()).with_extensions(ext);

        let outcome = tool.invoke(json!({ "task": "go" }), &ctx).await.expect("invoke ok");
        assert!(outcome.is_error);
        assert!(outcome.text.contains("max nesting depth"), "got: {}", outcome.text);
    }

    #[tokio::test]
    async fn subagent_honors_parent_cancel() {
        let tool = SubAgentTool::new("researcher", "x", fake_agent());
        // An already-cancelled flag stops the child before it produces text.
        let cancel = Arc::new(AtomicBool::new(true));
        let ctx = ToolContext::new(".".into()).with_cancel(cancel);

        let outcome = tool.invoke(json!({ "task": "go" }), &ctx).await.expect("invoke ok");
        let data = outcome.data.expect("summary");
        assert_eq!(data["stop_reason"], "Cancelled");
    }
}
