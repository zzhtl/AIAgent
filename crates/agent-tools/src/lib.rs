//! agent-tools
//!
//! Concrete `Tool` implementations: file read / edit / write, shell exec,
//! search, HTTP fetch. The `Tool` trait, `ToolContext`, `ToolRegistry`,
//! `Permissions`, and error type all live in `agent-core` — this crate only
//! supplies implementations.
//!
//! Register a default set via [`register_builtins`].

use std::sync::Arc;

use agent_core::tool::{Tool, ToolRegistry};

pub mod builtin;

/// Core built-ins: `file_read`, `file_edit`, `bash`, `grep`, `glob`,
/// `fetch`. These never touch the fact store and work without one.
pub fn register_builtins(registry: &mut ToolRegistry) {
    let tools: [Arc<dyn Tool>; 6] = [
        Arc::new(builtin::file_read::FileReadTool),
        Arc::new(builtin::file_edit::FileEditTool),
        Arc::new(builtin::bash::BashTool),
        Arc::new(builtin::grep::GrepTool),
        Arc::new(builtin::glob_tool::GlobTool),
        Arc::new(builtin::fetch::FetchTool),
    ];
    for t in tools {
        registry.register(t);
    }
}

/// Memory built-ins: `remember`, `forget`, `recall`. Each one fails fast
/// when no `FactStore` is plumbed through `ToolContext`, so it's safe to
/// register them unconditionally.
pub fn register_memory_tools(registry: &mut ToolRegistry) {
    let tools: [Arc<dyn Tool>; 3] = [
        Arc::new(builtin::memory_tools::RememberTool),
        Arc::new(builtin::memory_tools::ForgetTool),
        Arc::new(builtin::memory_tools::RecallTool),
    ];
    for t in tools {
        registry.register(t);
    }
}

/// Evolution proposal tools: `propose_rule`, `propose_skill`. Fail fast if
/// no candidate queue is plumbed through `ToolContext`.
pub fn register_evolution_tools(registry: &mut ToolRegistry) {
    let tools: [Arc<dyn Tool>; 2] = [
        Arc::new(builtin::propose::ProposeRuleTool),
        Arc::new(builtin::propose::ProposeSkillTool),
    ];
    for t in tools {
        registry.register(t);
    }
}
