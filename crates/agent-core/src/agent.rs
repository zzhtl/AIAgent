//! Agent runtime.
//!
//! Wires an `LlmProvider` and a `ToolRegistry` into an executable loop that
//! emits `AgentEvent`s. The loop is the `think → tool_call → observe`
//! pattern: each round-trip sends current messages to the LLM, dispatches
//! any requested tool calls, appends tool results, and iterates until the
//! model signals `end_turn` or `max_steps` is reached.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use futures::stream::{BoxStream, FuturesOrdered};
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
    /// Max automatic retries for transient LLM errors before aborting the
    /// turn. `0` disables retrying.
    pub max_retries: u32,
    /// Base backoff (ms) between retries; grows as `base * 2^attempt`.
    pub retry_base_delay_ms: u64,
    /// Cumulative per-turn token budget across all loop steps. `None` =
    /// unlimited.
    pub token_budget: Option<u32>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_steps: 12,
            temperature: None,
            max_tokens: None,
            permissions: Permissions::default(),
            max_retries: 2,
            retry_base_delay_ms: 500,
            token_budget: None,
        }
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
        // Same loop as `run_cancellable`, with a flag that never flips.
        self.run_cancellable(session_id, history, input, Arc::new(AtomicBool::new(false)))
    }

    /// Like [`Agent::run`], but a caller-held `cancel` flag lets the turn be
    /// stopped gracefully. The loop checks it at each step, before tool
    /// dispatch, and between streamed events; on cancel it stops at the next
    /// checkpoint and still emits a final `Done` (carrying whatever was
    /// produced) so the transcript stays consistent. In-flight LLM streaming
    /// or a running `bash` are not force-killed — cancellation takes effect at
    /// the next checkpoint.
    pub fn run_cancellable(
        &self,
        session_id: SessionId,
        history: Vec<Message>,
        input: UserInput,
        cancel: Arc<AtomicBool>,
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
            // Resolve the dynamic system prompt and the active tool whitelist
            // for this turn in one pass.
            let mut tool_whitelist: Option<Vec<String>> = None;
            if let Some(provider) = prompt_provider.as_ref() {
                let dynamic_system = provider.system_prompt_for(&input_text).await;
                if !dynamic_system.trim().is_empty() {
                    messages.push(Message::system(dynamic_system));
                }
                tool_whitelist = provider.tool_whitelist_for(&input_text).await;
            }
            messages.extend(history);
            // First index of messages this run will append (user input +
            // assistant/tool messages produced during the loop).
            let delta_start = messages.len();
            messages.push(Message::user(input.text));

            let mut steps = 0u32;
            let mut total_tokens: u32 = 0;
            let stop_reason = 'agent: loop {
                if cancel.load(Ordering::Relaxed) {
                    break 'agent StopReason::Cancelled;
                }
                steps += 1;
                if steps > config.max_steps {
                    yield AgentEvent::Warning {
                        message: format!("max_steps ({}) reached; stopping.", config.max_steps),
                    };
                    break 'agent StopReason::MaxSteps;
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

                // Open the stream, retrying transient errors (network / rate
                // limit / 5xx) with exponential backoff. Only the initial
                // connect is retried — once text is flowing a retry would
                // duplicate output, so mid-stream errors are not retried.
                let mut stream = {
                    let mut attempt = 0u32;
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            break 'agent StopReason::Cancelled;
                        }
                        match llm.chat_stream(request.clone()).await {
                            Ok(s) => break s,
                            Err(e) => {
                                if attempt >= config.max_retries || !e.is_retryable() {
                                    yield AgentEvent::Warning { message: format!("llm error: {e}") };
                                    break 'agent StopReason::Error;
                                }
                                let backoff = e
                                    .retry_after_secs()
                                    .map(Duration::from_secs)
                                    .unwrap_or_else(|| {
                                        let mult = 1u64 << attempt.min(16);
                                        Duration::from_millis(
                                            config.retry_base_delay_ms.saturating_mul(mult),
                                        )
                                    });
                                yield AgentEvent::Warning {
                                    message: format!(
                                        "llm transient error ({e}); retry {}/{} in {}ms",
                                        attempt + 1,
                                        config.max_retries,
                                        backoff.as_millis(),
                                    ),
                                };
                                tokio::time::sleep(backoff).await;
                                attempt += 1;
                            }
                        }
                    }
                };

                let mut assistant_text = String::new();
                let mut pending_calls: Vec<ToolUse> = Vec::new();
                let mut round_stop: Option<StopReason> = None;
                let mut round_usage: Option<TokenUsage> = None;
                let mut cancelled_mid_stream = false;

                while let Some(event) = stream.next().await {
                    if cancel.load(Ordering::Relaxed) {
                        cancelled_mid_stream = true;
                        break;
                    }
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
                    total_tokens = total_tokens.saturating_add(usage.total());
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

                if cancelled_mid_stream {
                    break 'agent StopReason::Cancelled;
                }

                let reason = round_stop.unwrap_or(StopReason::EndTurn);

                if pending_calls.is_empty() {
                    break 'agent reason;
                }

                if cancel.load(Ordering::Relaxed) {
                    break 'agent StopReason::Cancelled;
                }

                // Dispatch tools concurrently, preserving result order. Each
                // result becomes part of a single Tool-role message.
                let mut ctx = ToolContext::new(workspace.clone())
                    .with_permissions(config.permissions.clone())
                    .with_session_id(session_id_str.clone());
                if let Some(fs) = fact_store.clone() {
                    ctx = ctx.with_fact_store(fs);
                }
                if let Some(q) = candidate_queue.clone() {
                    ctx = ctx.with_candidate_queue(q);
                }

                let mut futs = FuturesOrdered::new();
                for call in pending_calls {
                    let tools = tools.clone();
                    let ctx = ctx.clone();
                    let whitelist = tool_whitelist.clone();
                    futs.push_back(async move {
                        if !is_tool_allowed(&call.name, whitelist.as_deref()) {
                            return MessageToolResult {
                                tool_use_id: call.id.clone(),
                                output: format!(
                                    "tool `{}` 未被当前技能授权 (tools_allowed)",
                                    call.name
                                ),
                                is_error: true,
                            };
                        }
                        invoke_one(&tools, &call, &ctx).await
                    });
                }

                let mut tool_result_blocks: Vec<ContentBlock> = Vec::new();
                while let Some(invocation) = futs.next().await {
                    yield AgentEvent::ToolCallResult { result: invocation.clone() };
                    tool_result_blocks.push(ContentBlock::ToolResult(invocation));
                }
                messages.push(Message { role: Role::Tool, content: tool_result_blocks });

                // Stop if the cumulative per-turn token budget is spent.
                if let Some(budget) = config.token_budget {
                    if total_tokens >= budget {
                        yield AgentEvent::Warning {
                            message: format!(
                                "token budget ({budget}) reached after {total_tokens} tokens; stopping."
                            ),
                        };
                        break 'agent StopReason::BudgetExceeded;
                    }
                }

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

/// Infrastructure tools that a skill `tools_allowed` whitelist never restricts:
/// memory and evolution plumbing the agent relies on regardless of task.
const META_TOOLS: &[&str] = &["remember", "forget", "recall", "propose_rule", "propose_skill"];

/// `None` whitelist ⇒ everything allowed. Otherwise a tool must be listed, or
/// be one of the always-on [`META_TOOLS`].
fn is_tool_allowed(name: &str, whitelist: Option<&[String]>) -> bool {
    match whitelist {
        None => true,
        Some(list) => list.iter().any(|t| t == name) || META_TOOLS.contains(&name),
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

