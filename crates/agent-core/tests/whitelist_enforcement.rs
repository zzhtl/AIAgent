//! End-to-end check that a skill `tools_allowed` whitelist is *enforced* by
//! the agent loop: a non-whitelisted tool call is rejected with an error
//! result and the tool's `invoke` is never reached, while a whitelisted tool
//! runs normally. Uses a fake provider + fake tools so no network/LLM is hit.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use agent_core::tool::{Tool, ToolContext, ToolOutcome, ToolResult};
use agent_core::{
    Agent, AgentEvent, ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, PromptProvider, SessionId, StopReason, ToolRegistry, UserInput,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// Emits two tool calls (one allowed, one blocked) on the first turn, then a
/// plain end-turn so the loop terminates.
struct FakeLlm {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl LlmProvider for FakeLlm {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: true, ..Default::default() }
    }
    async fn chat_stream(&self, _req: ChatRequest) -> LlmResult<LlmEventStream> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst);
        let events: Vec<LlmResult<LlmEvent>> = if n == 0 {
            vec![
                Ok(LlmEvent::ToolCallReady {
                    index: 0,
                    id: "1".into(),
                    name: "file_read".into(),
                    arguments: json!({ "path": "x" }),
                }),
                Ok(LlmEvent::ToolCallReady {
                    index: 1,
                    id: "2".into(),
                    name: "bash".into(),
                    arguments: json!({ "command": "ls" }),
                }),
                Ok(LlmEvent::End(StopReason::ToolUse)),
            ]
        } else {
            vec![
                Ok(LlmEvent::TextDelta { delta: "done".into() }),
                Ok(LlmEvent::End(StopReason::EndTurn)),
            ]
        };
        Ok(futures::stream::iter(events).boxed())
    }
}

/// Records whether it was actually invoked.
struct FlagTool {
    tool_name: &'static str,
    invoked: Arc<AtomicBool>,
}

#[async_trait]
impl Tool for FlagTool {
    fn name(&self) -> &str {
        self.tool_name
    }
    fn description(&self) -> &str {
        "test tool"
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn invoke(&self, _args: Value, _ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        self.invoked.store(true, Ordering::SeqCst);
        Ok(ToolOutcome::ok("ran"))
    }
}

/// Restricts the turn to `file_read` only.
struct RestrictToFileRead;

#[async_trait]
impl PromptProvider for RestrictToFileRead {
    async fn system_prompt_for(&self, _input: &str) -> String {
        String::new()
    }
    async fn tool_whitelist_for(&self, _input: &str) -> Option<Vec<String>> {
        Some(vec!["file_read".into()])
    }
}

#[tokio::test]
async fn non_whitelisted_tool_is_blocked_before_invoke() {
    let fr_flag = Arc::new(AtomicBool::new(false));
    let bash_flag = Arc::new(AtomicBool::new(false));

    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(FlagTool { tool_name: "file_read", invoked: fr_flag.clone() }));
    reg.register(Arc::new(FlagTool { tool_name: "bash", invoked: bash_flag.clone() }));

    let agent = Agent::builder()
        .with_llm(Arc::new(FakeLlm { calls: Arc::new(AtomicUsize::new(0)) }))
        .with_model("fake-model")
        .with_tools(reg)
        .with_prompt_provider(Arc::new(RestrictToFileRead))
        .build()
        .expect("agent builds");

    let mut stream = agent.run(SessionId::from("t"), Vec::new(), UserInput::new("hi"));
    let mut results = Vec::new();
    while let Some(ev) = stream.next().await {
        match ev {
            AgentEvent::ToolCallResult { result } => results.push(result),
            AgentEvent::Done { .. } => break,
            _ => {}
        }
    }

    let fr = results.iter().find(|r| r.tool_use_id == "1").expect("file_read result");
    let bash = results.iter().find(|r| r.tool_use_id == "2").expect("bash result");

    assert!(!fr.is_error, "whitelisted tool should run");
    assert!(bash.is_error, "non-whitelisted tool should be rejected");
    assert!(bash.output.contains("未被当前技能授权"), "got: {}", bash.output);

    assert!(fr_flag.load(Ordering::SeqCst), "file_read should have been invoked");
    assert!(!bash_flag.load(Ordering::SeqCst), "bash must never be invoked");
}
