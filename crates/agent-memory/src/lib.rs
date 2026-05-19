//! agent-memory
//!
//! Concrete persistence implementations:
//!
//! - `SqliteSessionStore` — short-term conversation history (stage 4).
//! - `MarkdownFactStore` — cross-session facts written as markdown files
//!   (stage 5).
//! - `SimpleVectorStore` — semantic recall over an embedding index, sharing
//!   the SQLite file (stage 5).
//!
//! Each store implements a trait defined in `agent-core` so the runtime
//! sees only the abstractions.

pub mod facts;
pub mod prompt;
pub mod sqlite;
pub mod vectors;

pub use facts::MarkdownFactStore;
pub use prompt::FactsPromptProvider;
pub use sqlite::SqliteSessionStore;
pub use vectors::SimpleVectorStore;
