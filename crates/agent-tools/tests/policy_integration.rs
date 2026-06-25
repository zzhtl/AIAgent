//! End-to-end: a `PolicyHook` denying `bash` blocks the call inside the agent
//! loop — the model gets an error result and the tool's `invoke` never runs.
//! Fake provider + fake tool, no network.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use agent_core::tool::{Tool, ToolContext, ToolOutcome, ToolResult};
use agent_core::{
    Agent, AgentEvent, ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, SessionId, StopReason, ToolRegistry, UserInput,
};
use agent_tools::policy::{PolicyHook, ToolAccess, ToolPolicy};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// Requests `bash` on the first turn, then ends.
struct CallBashThenDone {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl LlmProvider for CallBashThenDone {
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
                    name: "bash".into(),
                    arguments: json!({ "command": "rm -rf /" }),
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

struct FlagBash {
    invoked: Arc<AtomicBool>,
}

#[async_trait]
impl Tool for FlagBash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> &str {
        "fake bash"
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn invoke(&self, _args: Value, _ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        self.invoked.store(true, Ordering::SeqCst);
        Ok(ToolOutcome::ok("ran"))
    }
}

#[tokio::test]
async fn policy_blocks_denied_tool_in_loop() {
    let invoked = Arc::new(AtomicBool::new(false));
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(FlagBash { invoked: invoked.clone() }));

    let agent = Agent::builder()
        .with_llm(Arc::new(CallBashThenDone { calls: Arc::new(AtomicUsize::new(0)) }))
        .with_model("fake")
        .with_tools(reg)
        .with_hook(Arc::new(PolicyHook::new(ToolPolicy::new(ToolAccess::Allow).deny("bash"))))
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
    assert!(r.is_error, "denied tool should yield an error result");
    assert!(r.output.contains("denied by policy"), "got: {}", r.output);
    assert!(!invoked.load(Ordering::SeqCst), "denied bash must never be invoked");
}
