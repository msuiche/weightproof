//! FlashProve proof-format support (D11).
//!
//! Implemented against `spikes/flashprove` (anthropics/FlashProve, Apache 2.0):
//! - `blocks/frame/FORMAT.md` (file layout; descriptive)
//! - `docs/format/FORMAT.md` (ruling spec for hashed bytes)
//! - stored ground truth: `examples/accepted-proof/`, `examples/rejected-proofs/`
//! - oracle: `bin-clang/cp_verify` (local macOS build, see spikes/ notes)
//!
//! Scope of the current increments: request/parameter/version/field parsing;
//! configuration and commitment IDs; group/stage derivation; key/leaf/proof
//! byte-layout composition; complete LINK/leaf and key planning; and exact
//! structural walking from drawn positions. This is NOT yet a cryptographic verifier:
//! transcript execution, field validation and proof verification land against
//! the same ground truth. No layout or parse result claims a proof is valid.

pub mod algebra;
pub mod analysis;
pub mod blake2;
pub mod chain;
pub mod codes;
pub mod digest;
pub mod error;
pub mod field;
pub mod heads;
pub mod key;
pub mod layout;
pub mod plan;
pub mod policy;
pub mod proof;
pub mod request;
pub mod version;
pub mod walk;

pub use error::FpError;
pub use field::{FieldElement, GOLDILOCKS_P};
pub use policy::Policy;
pub use proof::{ProofHead, PROOF_TAG};
pub use request::{NextToken, RequestFile};
pub use version::{Version, H821};
