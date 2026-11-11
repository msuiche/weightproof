# Rocq extraction feasibility review

Status: reviewed 2026-11-09. No extraction was run: neither `hax` nor `cargo-hax`
is installed in the current environment.

## Findings

The D11 verifier is a good boundary for incremental formalization, but not a
drop-in hax/coqo-of-rust input.

Positive evidence:

- The crate builds with `no_std` plus `alloc` on `thumbv7em-none-eabi`.
- No `Rc`, `RefCell`, `Arc`, `Mutex`, `RwLock`, `static mut`, `thread_local!`, or
  `unsafe` blocks were found in `src/`.
- Field and tensor code is mostly explicit, total, and already checked against
  C++ vectors and end-to-end proof fixtures.

Extraction blockers:

- `src/fp/lkp.rs` uses `&dyn Summand` and `&dyn Fn(&mut Row, Tensor)`.
- `src/fp/psc.rs` defines `MapAt = Box<dyn Fn(&mut Row, &[E], E)>`.
- `src/fp/layer_checks.rs` passes `&dyn Fn(&WsGrp) -> WsFac` into the claim
  builder.
- Capturing closures require an explicit state machine in Rocq; trait-object
  dispatch and erased closures are not a semantics-preserving extraction target.
- `no_std` removes platform dependencies, but does not remove these higher-order
  abstractions or the work of specifying their semantics.

## Decision

Do not start with full-verifier extraction. Keep the Rust implementation plus
C++ agreement tests as the acceptance oracle while extracting a small arithmetic
kernel first.

Recommended order:

1. Extract `fp::field` and its Goldilocks/cubic-extension operations. Prove the
   canonicality, addition, subtraction, multiplication, and inverse properties.
2. Extract `fp::tensor` product/dense-factor evaluation, `tensor_below`, and
   rows-high rotation. Reuse the existing dense regression vectors.
3. Extract `fp::merkle` leaf and vertex hashing, with the reference domain and
   run-aliasing rules made explicit.
4. Extract `fp::claims` and the remaining proximity algebra only after replacing
   higher-order interfaces with explicit enums or generic implementations.
5. Consider full key/request extraction only after the `dyn` boundaries are gone
   and the generated Rocq code compiles and agrees with the stored transcripts.

## Required refactor before stage 4

- Replace `Summand` trait objects with a `SummandKind` enum or monomorphized
  generic implementations.
- Replace `MapAt` boxed closures with named function pointers or explicit state
  types. Do not encode erased closure state implicitly.
- Replace the `claim` callback in `layer_checks` with a `ClaimKind` enum whose
  variants carry the segment, layer, width, and factor selector.
- Keep `no_std`, explicit `FpError`, and total functions. Avoid introducing new
  interior mutability or unsafe code during the refactor.

## Acceptance evidence for each stage

- The extracted Rocq definitions compile.
- Their Rust counterparts pass the existing unit and integration tests.
- The same field/tensor/merkle vectors evaluate identically in Rocq and Rust.
- The accepted key and accepted request proofs still verify after every refactor.
- A report distinguishes proved arithmetic properties from unproved protocol and
  transcript properties; an extraction artifact alone is not a soundness proof.

Until stages 1–3 exist, the correct choice is the current differential-testing
track, not a claim that the verifier has been formally extracted.
