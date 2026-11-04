//! FlashProve proof-format support (D11).
//!
//! Implemented against `spikes/flashprove` (anthropics/FlashProve, Apache 2.0):
//! - `blocks/frame/FORMAT.md` (file layout; descriptive)
//! - `docs/format/FORMAT.md` (ruling spec for hashed bytes)
//! - stored ground truth: `examples/accepted-proof/`, `examples/rejected-proofs/`
//! - oracle: `bin-clang/cp_verify` (local macOS build, see spikes/ notes)
//!
//! Scope of this increment: request/parameter/version/field/proof-head parsing
//! with fail-closed, named errors. This is NOT yet a cryptographic verifier;
//! chain walking and proof verification land in later increments against the
//! same ground truth. Nothing here claims a proof is valid.

pub mod error;
pub mod field;
pub mod policy;
pub mod proof;
pub mod request;
pub mod version;

pub use error::FpError;
pub use field::{FieldElement, GOLDILOCKS_P};
pub use policy::Policy;
pub use proof::{ProofHead, PROOF_TAG};
pub use request::{NextToken, RequestFile};
pub use version::{Version, H821};
