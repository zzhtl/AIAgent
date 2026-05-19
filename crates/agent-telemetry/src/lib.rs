//! agent-telemetry
//!
//! `tracing-subscriber` setup and per-model cost estimation. Lives in its
//! own crate so the CLI / bot front-ends can initialise once and reuse.

pub mod init;
pub mod pricing;

pub use init::init_default;
pub use pricing::{estimate_cost_usd, price_for, Price};
