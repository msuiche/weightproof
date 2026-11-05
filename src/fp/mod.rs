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
//! byte-layout composition; complete LINK/leaf, key, and request-proof planning;
//! exact structural walking from drawn positions; and LINK transcript execution
//! with canonical field checks, proof-of-work, Merkle opening verification, and
//! algebraic claims. Key/request transcript checks remain. No layout or parse
//! result alone claims a proof is valid.

pub mod algebra;
pub mod analysis;
pub mod batt;
pub mod blake2;
pub mod chain;
pub mod claims;
pub mod code_eval;
pub mod codes;
pub mod digest;
pub mod error;
pub mod field;
pub mod heads;
pub mod iopp;
pub mod key;
pub mod key_fixture;
pub mod key_public;
pub mod layout;
pub mod lin;
pub mod link_algebra;
pub mod link_public;
pub mod lkp;
pub mod mask;
pub mod merkle;
pub mod plan;
pub mod policy;
pub mod proof;
pub mod proof_plan;
pub mod proximity;
pub mod psc;
pub mod request;
pub mod subproof;
pub mod tensor;
pub mod transcript;
pub mod verify_key;
pub mod verify_link;
pub mod version;
pub mod walk;
pub mod wf;

pub use error::FpError;
pub use field::{FieldElement, GOLDILOCKS_P};
pub use policy::Policy;
pub use proof::{ProofHead, PROOF_TAG};
pub use request::{NextToken, RequestFile};
pub use version::{Version, H821};
