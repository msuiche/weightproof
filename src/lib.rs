//! weightproof — provable LLM inference.
//!
//! First artifact (D11): a Rust implementation of the FlashProve proof
//! format, developed against `spikes/flashprove` (anthropics/FlashProve,
//! Apache 2.0) with C++↔Rust agreement tests on real proofs.
//!
//! Layout (D6):
//! - `fp` — FlashProve proof-format types, parsing, structural validation.
//!   no_std + alloc, no prover dependencies, no panics on adversarial input.
//! - later: cryptographic verification, then proving (feature `prover`).

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
// The verifier parses adversarial proofs. Indexing/unwrap on untrusted bytes
// is a defect (D8); clippy config enforces this as the crate grows.
#![warn(clippy::indexing_slicing, clippy::unwrap_used, clippy::panic)]

extern crate alloc;

pub mod fp;
