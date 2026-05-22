//! Agent runtime.
//!
//! Wires an `LlmProvider` and a `ToolRegistry` into an executable loop that
//! emits `AgentEvent`s. The loop is the `think → tool_call → observe`
//! pattern: each round-trip sends current messages to the LLM, dispatches
//! any requested tool calls, appends tool results, and iterates until the
//! model signals `end_turn` or `max_steps` is reached.

use std::path::PathBuf;
use std::sync::Arc;

use async_stream::stream;
use futures::stream::BoxStream;
use futures::StreamExt;
use tracing::{debug, warn};

use crate::channel::{AgentEvent, UserInput};
use crate::evolution::CandidateQueue;
use crate::llm::{ChatRequest, LlmEvent, LlmProvider};
use crate::memory::FactStore;
use crate::message::{
    ContentBlock, Message, Role, StopReason, TokenUsage, ToolResult as MessageToolResult, ToolUse,
};
use crate::prompt::PromptProvider;
use crate::session::SessionId;
use crate::tool::{Permissions, ToolContext, ToolRegistry};

/// Knobs for the run loop. Defaults are conservative.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub max_steps: u32,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub permissions: Permissions,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self { max_steps: 12, temperature: None, max_tokens: None, permissions: Permissions::default() }
    }
}

/// Concrete agent. Cheap to clone (every owned field is `Arc` or small).
#[derive(Clone)]
pub struct Agent {
    llm: Arc<dyn LlmProvider>,
    tools: ToolRegistry,
    model: String,
    system_prompt: Option<String>,
    prompt_provider: Option<Arc<dyn PromptProvider>>,
    fact_store: Option<Arc<dyn FactStore>>,
    candidate_queue: Option<CandidateQueue>,
    workspace: PathBuf,
    config: RunConfig,
}

impl Agent {
    pub fn builder() -> AgentBuilder {
        AgentBuilder::default()
    }

    /// Execute one user turn. Returns a stream of events; the caller drives
    /// rendering. `history` is the prior transcript (system messages are
    /// injected automatically — do not pre-include them).
    pub fn run(
        &self,
        session_id: SessionId,
        history: Vec<Message>,
        input: UserInput,
    ) -> BoxStream<'static, AgentEvent> {
        let llm = self.llm.clone();
        let tools = self.tools.clone();
        let model = self.model.clone();
        let system_prompt = self.system_prompt.clone();
        let prompt_provider = self.prompt_provider.clone();
        let fact_store = self.fact_store.clone();
        let candidate_queue = self.candidate_queue.clone();
        let workspace = self.workspace.clone();
        let config = self.config.clone();
        let session_id_str = session_id.to_string();
        let tool_schemas = tools.schemas();

        // Resolve dynamic prompt content up-front so the stream! body stays
        // free of borrow gymnastics. The provider can await storage / vector
        // search; the cost is paid once per turn.
        let input_text = input.text.clone();

        let s = stream! {
            let mut messages = Vec::with_capacity(history.len() + 3);
            if let Some(sys) = system_prompt.as_deref() {
                if !sys.trim().is_empty() {
                    messages.push(Message::system(sys));
                }
            }
            if let Some(provider) = prompt_provider.as_ref() {
                let dynamic_system = provider.system_prompt_for(&input_text).await;
                if !dynamic_system.trim().is_empty() {
                    messages.push(Message::system(dynamic_system));
                }
            }
            messages.extend(history);
            // First index of messages this run will append (user input +
            // assistant/tool messages produced during the loop).
            let delta_start = messages.len();
            messages.push(Message::user(input.text));

            let mut steps = 0u32;
            let stop_reason = loop {
                steps += 1;
                if steps > config.max_steps {
                    yield AgentEvent::Warning {
                        message: format!("max_steps ({}) reached; stopping.", config.max_steps),
                    };
                    break StopReason::MaxSteps;
                }

                debug!(step = steps, "agent: sending chat request");

                let request = ChatRequest {
                    model: model.clone(),
                    messages: messages.clone(),
                    tools: tool_schemas.clone(),
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                    stream: true,
                };

                let mut stream = match llm.chat_stream(request).await {
                    Ok(s) => s,
                    Err(e) => {
                        yield AgentEvent::Warning { message: format!("llm error: {e}") };
                        break StopReason::Error;
                    }
                };

                let mut assistant_text = String::new();
                let mut pending_calls: Vec<ToolUse> = Vec::new();
                let mut round_stop: Option<StopReason> = None;
                let mut round_usage: Option<TokenUsage> = None;

                while let Some(event) = stream.next().await {
                    match event {
                        Ok(LlmEvent::TextDelta { delta }) => {
                            assistant_text.push_str(&delta);
                            yield AgentEvent::TextDelta { delta };
                        }
                        Ok(LlmEvent::ToolCallDelta { .. }) => {
                            // Argument fragments — rendered only at ToolCallReady.
                        }
                        Ok(LlmEvent::ToolCallReady { id, name, arguments, .. }) => {
                            let call = ToolUse { id, name, input: arguments };
                            yield AgentEvent::ToolCallStart { call: call.clone() };
                            pending_calls.push(call);
                        }
                        Ok(LlmEvent::Usage(usage)) => {
                            round_usage = Some(usage);
                        }
                        Ok(LlmEvent::End(reason)) => {
                            round_stop = Some(reason);
                            break;
                        }
                        Err(e) => {
                            yield AgentEvent::Warning { message: format!("stream error: {e}") };
                            round_stop = Some(StopReason::Error);
                            break;
                        }
                    }
                }

                if let Some(usage) = round_usage {
                    yield AgentEvent::UsageReport { usage, model: model.clone() };
                }

                // Append the assistant message (text + any tool_use blocks).
                let mut assistant_blocks: Vec<ContentBlock> = Vec::new();
                if !assistant_text.is_empty() {
                    assistant_blocks.push(ContentBlock::Text { text: assistant_text });
                }
                for call in &pending_calls {
                    assistant_blocks.push(ContentBlock::ToolUse(call.clone()));
                }
                if !assistant_blocks.is_empty() {
                    messages.push(Message { role: Role::Assistant, content: assistant_blocks });
                }

                let reason = round_stop.unwrap_or(StopReason::EndTurn);

                if pending_calls.is_empty() {
                    break reason;
                }

                // Dispatch tools. Each result becomes a Tool-role message.
                let mut ctx = ToolContext::new(workspace.clone())
                    .with_permissions(config.permissions.clone())
                    .with_session_id(session_id_str.clone());
                if let Some(fs) = fact_store.clone() {
                    ctx = ctx.with_fact_store(fs);
                }
                if let Some(q) = candidate_queue.clone() {
                    ctx = ctx.with_candidate_queue(q);
                }

                let mut tool_result_blocks: Vec<ContentBlock> = Vec::new();
                for call in pending_calls {
                    let invocation = invoke_one(&tools, &call, &ctx).await;
                    yield AgentEvent::ToolCallResult { result: invocation.clone() };
                    tool_result_blocks.push(ContentBlock::ToolResult(invocation));
                }
                messages.push(Message { role: Role::Tool, content: tool_result_blocks });

                // Continue the loop so the model can react to tool results.
                if reason != StopReason::ToolUse {
                    // Some providers may signal EndTurn even with tool_calls present;
                    // we still need to feed results back for the next round.
                    debug!(?reason, "tools present despite non-tool stop reason; continuing");
                }
            };

            let transcript_delta = messages.split_off(delta_start);
            yield AgentEvent::Done { reason: stop_reason, transcript_delta };
        };

        Box::pin(s)
    }
}

async fn invoke_one(
    registry: &ToolRegistry,
    call: &ToolUse,
    ctx: &ToolContext,
) -> MessageToolResult {
    match registry.invoke(&call.name, call.input.clone(), ctx).await {
        Ok(outcome) => MessageToolResult {
            tool_use_id: call.id.clone(),
            output: outcome.text,
            is_error: outcome.is_error,
        },
        Err(e) => {
            warn!(tool = %call.name, error = %e, "tool invocation failed");
            MessageToolResult {
                tool_use_id: call.id.clone(),
                output: format!("tool error: {e}"),
                is_error: true,
            }
        }
    }
}

/// Modular builder. Every component plugs in via a fluent setter so the same
/// kernel can be assembled differently in CLI / bot / tests.
#[derive(Default)]
pub struct AgentBuilder {
    llm: Option<Arc<dyn LlmProvider>>,
    tools: Option<ToolRegistry>,
    model: Option<String>,
    system_prompt: Option<String>,
    prompt_provider: Option<Arc<dyn PromptProvider>>,
    fact_store: Option<Arc<dyn FactStore>>,
    candidate_queue: Option<CandidateQueue>,
    workspace: Option<PathBuf>,
    config: Option<RunConfig>,
}

impl AgentBuilder {
    pub fn with_llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_prompt_provider(mut self, provider: Arc<dyn PromptProvider>) -> Self {
        self.prompt_provider = Some(provider);
        self
    }

    pub fn with_fact_store(mut self, store: Arc<dyn FactStore>) -> Self {
        self.fact_store = Some(store);
        self
    }

    pub fn with_candidate_queue(mut self, queue: CandidateQueue) -> Self {
        self.candidate_queue = Some(queue);
        self
    }

    pub fn with_workspace(mut self, workspace: PathBuf) -> Self {
        self.workspace = Some(workspace);
        self
    }

    pub fn with_config(mut self, config: RunConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> Result<Agent, &'static str> {
        let llm = self.llm.ok_or("agent: missing llm provider")?;
        let model = self.model.ok_or("agent: missing model")?;
        Ok(Agent {
            llm,
            tools: self.tools.unwrap_or_default(),
            model,
            system_prompt: self.system_prompt,
            prompt_provider: self.prompt_provider,
            fact_store: self.fact_store,
            candidate_queue: self.candidate_queue,
            workspace: self
                .workspace
                .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
            config: self.config.unwrap_or_default(),
        })
    }
}

