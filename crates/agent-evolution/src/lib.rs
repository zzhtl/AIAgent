//! agent-evolution
//!
//! Stage 5 ships the **reflection** half: after each run, optionally write a
//! short reflection note into long-term memory.
//!
//! Stage 6 adds the **candidate flow**: `propose_rule` / `propose_skill`
//! tools enqueue candidate rules and skills; the user reviews and applies
//! them via `agent evolution review / apply / reject`.
//!
//! Candidate data types and the JSON queue live in `agent-core::evolution`
//! so tools (`agent-tools`) can enqueue without depending on this crate.

pub mod reflection;
pub mod summariser;

pub use agent_core::evolution::{Candidate, CandidateError, CandidateKind, CandidateQueue};
pub use reflection::Reflector;
pub use summariser::Summariser;
