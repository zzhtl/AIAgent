//! agent-tools
//!
//! Defines the `Tool` trait, `ToolRegistry`, permission gating, and a set of
//! built-in tools (file read/write/edit, shell, search, fetch).
//!
//! Custom tools plug in by implementing `Tool` and calling `registry.register()`.
