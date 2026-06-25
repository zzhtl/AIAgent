//! End-to-end checks that the agent loop drives `AgentHook` at the right
//! seams: the four lifecycle methods fire in order across loop steps, and a
//! `before_tool` `Block` substitutes an error result without ever invoking the
//! tool. Fake provider + fake tool, no network.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_core::tool::{Tool, ToolContext, ToolOutcome, ToolResult};
use agent_core::{
    Agent, AgentEvent, AgentHook, ChatRequest, HookDecision, LlmEvent, LlmEventStream, LlmProvider,
    LlmResult, ProviderCapabilities, SessionId, StopReason, ToolRegistry, ToolUse, UserInput,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// One tool call on the first turn, then a plain end-turn on the second.
struct OneToolThenDone {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl LlmProvider for OneToolThenDone {
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

/// A tool that records whether it actually ran.
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

/// Records the order of lifecycle calls.
#[derive(Default)]
struct RecordingHook {
    log: Mutex<Vec<String>>,
}

#[async_trait]
impl AgentHook for RecordingHook {
    async fn before_llm(&self, _req: &mut ChatRequest) -> HookDecision {
        self.log.lock().unwrap().push("before_llm".into());
        HookDecision::Continue
    }
    async fn after_llm(&self, _text: &str, _calls: &[ToolUse]) {
        self.log.lock().unwrap().push("after_llm".into());
    }
    async fn before_tool(&self, call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
        self.log.lock().unwrap().push(format!("before_tool:{}", call.name));
        HookDecision::Continue
    }
    async fn after_tool(&self, call: &ToolUse, _outcome: &ToolOutcome) {
        self.log.lock().unwrap().push(format!("after_tool:{}", call.name));
    }
}

#[tokio::test]
async fn hooks_fire_in_order() {
    let hook = Arc::new(RecordingHook::default());
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(FlagTool {
        tool_name: "file_read",
        invoked: Arc::new(AtomicBool::new(false)),
    }));

    let agent = Agent::builder()
        .with_llm(Arc::new(OneToolThenDone { calls: Arc::new(AtomicUsize::new(0)) }))
        .with_model("fake-model")
        .with_tools(reg)
        .with_hook(hook.clone())
        .build()
        .expect("agent builds");

    let mut stream = agent.run(SessionId::from("t"), Vec::new(), UserInput::new("hi"));
    while let Some(ev) = stream.next().await {
        if matches!(ev, AgentEvent::Done { .. }) {
            break;
        }
    }

    let log = hook.log.lock().unwrap().clone();
    assert_eq!(
        log,
        vec![
            "before_llm".to_string(),
            "after_llm".to_string(),
            "before_tool:file_read".to_string(),
            "after_tool:file_read".to_string(),
            "before_llm".to_string(),
            "after_llm".to_string(),
        ]
    );
}

/// Blocks every tool call.
struct BlockAll;

#[async_trait]
impl AgentHook for BlockAll {
    async fn before_tool(&self, _call: &mut ToolUse, _ctx: &ToolContext) -> HookDecision {
        HookDecision::Block("denied by hook".into())
    }
}

#[tokio::test]
async fn before_tool_block_substitutes_error_without_invoking() {
    let invoked = Arc::new(AtomicBool::new(false));
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(FlagTool { tool_name: "file_read", invoked: invoked.clone() }));

    let agent = Agent::builder()
        .with_llm(Arc::new(OneToolThenDone { calls: Arc::new(AtomicUsize::new(0)) }))
        .with_model("fake-model")
        .with_tools(reg)
        .with_hook(Arc::new(BlockAll))
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

    let r = results.iter().find(|r| r.tool_use_id == "1").expect("tool result");
    assert!(r.is_error, "blocked tool should yield an error result");
    assert!(r.output.contains("denied by hook"), "got: {}", r.output);
    assert!(!invoked.load(Ordering::SeqCst), "blocked tool must never be invoked");
}
