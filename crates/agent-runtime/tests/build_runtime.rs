use std::sync::Arc;

use agent_config::{AgentConfig, McpServerConfig, SubAgentConfig};
use agent_core::tool::Permissions;
use agent_core::{
    AgentEvent, ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult,
    ProviderCapabilities, Role, SessionId, StopReason, TokenUsage, UserInput,
};
use agent_runtime::{build_runtime_with_provider, BuildOptions};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

struct TextProvider;

#[async_trait]
impl LlmProvider for TextProvider {
    fn name(&self) -> &str {
        "stub"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, ..Default::default() }
    }

    async fn chat_stream(&self, _request: ChatRequest) -> LlmResult<LlmEventStream> {
        Ok(futures::stream::iter(vec![
            Ok(LlmEvent::TextDelta { delta: "hello".into() }),
            Ok(LlmEvent::End(StopReason::EndTurn)),
        ])
        .boxed())
    }
}

struct BashCallingProvider;

#[async_trait]
impl LlmProvider for BashCallingProvider {
    fn name(&self) -> &str {
        "stub"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: true, ..Default::default() }
    }

    async fn chat_stream(&self, _request: ChatRequest) -> LlmResult<LlmEventStream> {
        Ok(futures::stream::iter(vec![
            Ok(LlmEvent::ToolCallReady {
                index: 0,
                id: "call-1".into(),
                name: "bash".into(),
                arguments: json!({ "command": "echo should-not-run" }),
            }),
            Ok(LlmEvent::Usage(TokenUsage {
                prompt_tokens: 5,
                completion_tokens: 5,
                cached_tokens: 0,
            })),
            Ok(LlmEvent::End(StopReason::ToolUse)),
        ])
        .boxed())
    }
}

struct DelegatingProvider {
    marker: String,
}

#[async_trait]
impl LlmProvider for DelegatingProvider {
    fn name(&self) -> &str {
        "stub"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities { streaming: true, tools: true, ..Default::default() }
    }

    async fn chat_stream(&self, request: ChatRequest) -> LlmResult<LlmEventStream> {
        let is_parent = request.tools.iter().any(|tool| tool.name == "researcher");
        let has_tool_result = request.messages.iter().any(|message| message.role == Role::Tool);
        let events = if has_tool_result {
            vec![
                Ok(LlmEvent::TextDelta { delta: "done".into() }),
                Ok(LlmEvent::End(StopReason::EndTurn)),
            ]
        } else if is_parent {
            vec![
                Ok(LlmEvent::ToolCallReady {
                    index: 0,
                    id: "delegate-1".into(),
                    name: "researcher".into(),
                    arguments: json!({ "task": "check permissions" }),
                }),
                Ok(LlmEvent::End(StopReason::ToolUse)),
            ]
        } else {
            vec![
                Ok(LlmEvent::ToolCallReady {
                    index: 0,
                    id: "bash-1".into(),
                    name: "bash".into(),
                    arguments: json!({
                        "command": format!("printf child-ran > {}", self.marker)
                    }),
                }),
                Ok(LlmEvent::End(StopReason::ToolUse)),
            ]
        };
        Ok(futures::stream::iter(events).boxed())
    }
}

fn config_in(dir: &std::path::Path) -> AgentConfig {
    AgentConfig { config_dir: Some(dir.to_path_buf()), ..AgentConfig::default() }
}

#[tokio::test]
async fn injected_provider_builds_and_runs_one_turn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config_in(dir.path());
    config.no_tools = true;
    let runtime = build_runtime_with_provider(
        &config,
        BuildOptions { workspace: Some(dir.path().to_path_buf()), ..Default::default() },
        Arc::new(TextProvider),
        "stub-model",
    )
    .await
    .expect("runtime");

    let events: Vec<_> = runtime
        .agent
        .run(SessionId::new(), Vec::new(), UserInput::new("hi"))
        .collect()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::TextDelta { delta } if delta == "hello"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::Done { reason: StopReason::EndTurn, .. }
    )));
}

#[tokio::test]
async fn token_budget_and_permission_override_reach_agent_loop() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config_in(dir.path());
    config.agent.token_budget = Some(5);
    let permissions = Permissions {
        allow_read: true,
        allow_write: false,
        allow_shell: false,
        allow_network: false,
        max_runtime_secs: 1,
    };
    let runtime = build_runtime_with_provider(
        &config,
        BuildOptions {
            workspace: Some(dir.path().to_path_buf()),
            permissions_override: Some(permissions),
            ..Default::default()
        },
        Arc::new(BashCallingProvider),
        "stub-model",
    )
    .await
    .expect("runtime");

    let events: Vec<_> = runtime
        .agent
        .run(SessionId::new(), Vec::new(), UserInput::new("run"))
        .collect()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::ToolCallResult { result } if result.is_error && result.output.contains("bash disabled")
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::Done { reason: StopReason::BudgetExceeded, .. }
    )));
}

#[tokio::test]
async fn broken_mcp_server_becomes_warning() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = config_in(dir.path());
    config.mcp_servers.push(McpServerConfig {
        name: "broken".into(),
        command: dir.path().join("missing-command").display().to_string(),
        args: Vec::new(),
        env: Default::default(),
    });
    let runtime = build_runtime_with_provider(
        &config,
        BuildOptions { workspace: Some(dir.path().to_path_buf()), ..Default::default() },
        Arc::new(TextProvider),
        "stub-model",
    )
    .await
    .expect("runtime remains available");
    assert!(runtime
        .warnings
        .iter()
        .any(|warning| warning.contains("mcp server `broken` skipped")));
}

#[tokio::test]
async fn permission_override_propagates_to_declared_subagent() {
    let dir = tempfile::tempdir().expect("tempdir");
    let marker = dir.path().join("child-ran");
    let mut config = config_in(dir.path());
    config.subagents.push(SubAgentConfig {
        name: "researcher".into(),
        description: "test delegate".into(),
        model: None,
        prompt: None,
        max_steps: Some(3),
    });
    let permissions = Permissions {
        allow_read: true,
        allow_write: false,
        allow_shell: false,
        allow_network: false,
        max_runtime_secs: 1,
    };
    let provider = DelegatingProvider { marker: marker.display().to_string() };
    let runtime = build_runtime_with_provider(
        &config,
        BuildOptions {
            workspace: Some(dir.path().to_path_buf()),
            permissions_override: Some(permissions),
            ..Default::default()
        },
        Arc::new(provider),
        "stub-model",
    )
    .await
    .expect("runtime");

    let events: Vec<_> = runtime
        .agent
        .run(SessionId::new(), Vec::new(), UserInput::new("delegate"))
        .collect()
        .await;
    assert!(events.iter().any(|event| matches!(
        event,
        AgentEvent::Done { reason: StopReason::EndTurn, .. }
    )));
    assert!(!marker.exists(), "subagent must inherit the shell denial");
}
