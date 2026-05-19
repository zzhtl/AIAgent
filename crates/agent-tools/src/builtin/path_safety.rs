//! Shared path resolution used by file tools.
//!
//! Relative paths are resolved against `ctx.workspace`. Absolute paths are
//! allowed (the agent may legitimately want to touch system config / temp
//! files) but get logged at debug level.

use std::path::{Path, PathBuf};

use agent_core::tool::ToolContext;

pub fn resolve(ctx: &ToolContext, raw: &str) -> PathBuf {
    let p = Path::new(raw);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        ctx.workspace.join(p)
    }
}
