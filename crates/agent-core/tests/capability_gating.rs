//! The runtime advertises tools only when the provider reports tool support.
//! A provider with `capabilities().tools == false` must receive an empty tools
//! array; one reporting `true` receives the registered schemas. Fake provider
//! captures the schema count it was handed.

use std::sync::{Arc, Mutex};

use agent_core::tool::{Tool, ToolContext, ToolOutcome, ToolResult};
use agent_core::{
    Agent, AgentEvent, ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, SessionId, StopReason, ToolRegistry, UserInput,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// Captures the number of tool schemas seen on the request, then ends the turn.
struct CapturingLlm {
    tools_supported: bool,
    seen_tools: Arc<Mutex<Option<usize>>>,
}

#[async_trait]
impl LlmProvider for CapturingLlm {
    fn name(&self) -> &str {
        "fake"
    }
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: self.tools_supported, ..Default::default() }
    }
    async fn chat_stream(&self, req: ChatRequest) -> LlmResult<LlmEventStream> {
        *self.seen_tools.lock().unwrap() = Some(req.tools.len());
        let events: Vec<LlmResult<LlmEvent>> = vec![
            Ok(LlmEvent::TextDelta { delta: "ok".into() }),
            Ok(LlmEvent::End(StopReason::EndTurn)),
        ];
        Ok(futures::stream::iter(events).boxed())
    }
}

struct NoopTool;

#[async_trait]
impl Tool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn description(&self) -> &str {
        "noop"
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn invoke(&self, _args: Value, _ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        Ok(ToolOutcome::ok("ok"))
    }
}

async fn schemas_seen_by_provider(tools_supported: bool) -> usize {
    let seen = Arc::new(Mutex::new(None));
    let mut reg = ToolRegistry::new();
    reg.register(Arc::new(NoopTool));

    let agent = Agent::builder()
        .with_llm(Arc::new(CapturingLlm { tools_supported, seen_tools: seen.clone() }))
        .with_model("fake-model")
        .with_tools(reg)
        .build()
        .expect("agent builds");

    let mut stream = agent.run(SessionId::from("t"), Vec::new(), UserInput::new("hi"));
    while let Some(ev) = stream.next().await {
        if matches!(ev, AgentEvent::Done { .. }) {
            break;
        }
    }
    let count = seen.lock().unwrap().expect("chat_stream was called");
    count
}

#[tokio::test]
async fn tools_omitted_when_provider_lacks_tool_support() {
    assert_eq!(schemas_seen_by_provider(false).await, 0, "no tools should be advertised");
}

#[tokio::test]
async fn tools_advertised_when_provider_supports_them() {
    assert_eq!(schemas_seen_by_provider(true).await, 1, "registered tool should be advertised");
}
