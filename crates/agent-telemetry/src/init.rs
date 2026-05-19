//! `tracing` subscriber initialisation.
//!
//! Pretty output to stderr (so it doesn't pollute the CLI's streamed
//! assistant text on stdout). Filter defaults to `warn` and is overridable
//! via the `RUST_LOG` environment variable.

use std::io;
use tracing_subscriber::EnvFilter;

pub fn init_default() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(io::stderr)
        .with_target(false)
        .try_init();
}
