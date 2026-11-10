# TODOS — weightproof

Captured by /plan-eng-review, 2026-11-03/04. Plan state: D1–D16 decisions applied
(incl. codex outside-voice findings). First work: D11 FlashProve reproduction +
Rust verifier, D3/D14 WHIR security-gated spike, D4/D15 Lean composition track.

## 1. Protocol spec gaps: proof-pending, commitment renewal + identity, versioning

- **Status:** Completed in `spec/proof-lifecycle.md` (2026-11-09): proof-pending
  states, independent commitment certification and renewal, format/parameter
  versioning, and per-request extraction acceptance rules.

- **What:** Write spec sections for (a) proof-pending client semantics during
  post-hoc proving, (b) commitment certification + renewal flow (~16M requests),
  (c) proof-format + parameter versioning.
- **Why:** Three architecture-review P2s plus codex finding #8. A commitment
  authenticates no model identity by itself: if the provider supplies the expected
  commitment, it can commit to substitute weights and prove them honestly.
- **Pros:** Renewal and versioning are breaking-change-prone; specifying them
  before commitments are deployed avoids forced migration. Proof-pending semantics
  defines what clients may assume before a proof lands.
- **Cons:** ~1–2 days spec work that will need revision once the WHIR spike and
  serving integration settle formats.
- **Context:** FlashProve commitments expire after ~16M requests (offline phase
  re-run); its verifier exit-code contract is the model for clear client semantics.
  Start from statement binding H(version ‖ commitment ‖ prompt ‖ cfg ‖ sampling ‖
  tokens). Spec must define: who certifies the commitment (provider self-claim is
  insufficient), what architecture/quantization metadata it binds, and how renewal
  preserves model identity across re-commits.
- **Depends on:** WHIR spike outcome (D3/D14) for commitment format; nothing for
  the versioning skeleton.

## 2. hax / coq-of-rust extraction feasibility (Rocq track)

- **Status:** Completed in `spec/extraction-review.md` (2026-11-09): staged
  extraction plan, concrete `dyn` blockers, and acceptance evidence. No extraction
  was run because hax/cargo-hax is not installed.

- **What:** Evaluate extracting a Rocq model from the real Rust verifier to prove
  implementation correctness, vs. the hand-written Lean twin + CI agreement tests.
- **Why:** FlashProve's admitted gap — "we do not guarantee Lean, paper, and code
  match" — is inherited by any hand-twin design. Extraction proves properties of
  the artifact users actually run.
- **Pros:** Closes the code↔spec trust gap; the D11 Rust-verifier-first artifact
  is a small, natural extraction target.
- **Cons:** hax requires a restricted Rust subset (no trait objects, no interior
  mutability); toolchain younger than Lean 4; substantial proof effort.
- **Context:** Lean 4 remains the default for protocol statements (D4/D15:
  composed-transcript simulator + randomness accounting across adaptive requests).
  Extraction targets verifier implementation correctness — complementary, not a
  replacement. Cheapest first step: keep the D11 verifier's Rust style
  extraction-friendly from its first commit (no trait objects, explicit error
  types, no_std per D6), before any formal spike.
- **Depends on:** D11 Rust verifier existing; D6 no_std boundary already moves
  toward an extraction-friendly subset.

## 3. Verifier distribution + CI

- **What:** Build/publish pipeline for the verifier CLI — GitHub Releases
  (linux/darwin × amd64/arm64), cargo install — plus CI jobs for the no_std check
  (D6), fuzz targets (D8), criterion benches (D9).
- **Why:** The plan introduces a CLI artifact with no distribution story; verifiers
  are meant to be run by auditors/users who won't compile Rust.
- **Pros:** One work item covers release plumbing and the CI that D6/D8/D9 require.
- **Cons:** Release engineering is easy to perpetually defer; cross-platform CI
  runners cost time/money.
- **Context:** Start with a single GitHub Actions workflow when the D11 Rust
  verifier lands. CI jobs (no_std check, fuzz, benches) arrive with the M0 crate.
- **Depends on:** D11 Rust verifier = first publishable artifact.

## 4. Review SF Tensor's PTX-in-Lean formalization (potential reuse, M2+)

- **Status:** Completed in `spec/ptx-review.md` (2026-11-09): reuse the memory
  model and pipeline composition for M2, with explicit Goldilocks semantics,
  progress/UB arguments, non-vacuity, and differential verification.

- **What:** Review https://sf-tensor.com/engineering/formalizing-ptx and the
  paper https://sf-tensor.com/papers/formalizing-ptx.pdf for reuse in
  verifying our GPU prover kernels in Lean.
- **Why:** They formalize PTX semantics + the PTX memory consistency model in
  Lean 4 and prove GPU kernels correct with LLM-generated, machine-checked
  proofs — no `sorry`s, no user axioms. Directly relevant to our M2 GPU
  sumcheck/FRI kernels and the D4/D15 Lean trust track.
- **Pros:** Already proven on Hopper tensor-core GEMM, online-softmax
  FlashAttention, a three-kernel MoE pipeline, and megakernels with
  data-dependent control flow. Their multi-kernel composition interface
  (successor kernel's input = predecessor's verified output) is a precedent for
  our D12 composition gate. User flagged for review 2026-11-04.
- **Cons / caveats to check in review:** (a) floats modeled as reals — semantic,
  not numerical/bitwise correctness; our D13 exact-integer semantics is a
  different (and for ZK, easier) target, but the mismatch must be understood;
  (b) UB-freedom is an external assumption, not proven; (c) no-thin-air and
  Fence-SC axioms omitted (limits what can be proven, doesn't enable false
  proofs); (d) PTX ≠ SASS — compiler reorderings are out of model scope.
- **Context:** Our prover kernels will be integer field arithmetic (Goldilocks)
  on CUDA; if their Lean PTX model is reusable, kernel-level correctness proofs
  could extend our "statements before code" rule to the GPU layer instead of
  relying on differential testing alone.
- **Depends on:** nothing for the review; reuse decisions wait for M2 GPU work.

## 5. M1 dense integer model and proof batch

- **Status:** Completed (2026-11-09): exact dense model, statement binding,
  prover/verifier transcript generation, arithmetic and lookup proofs, compact
  encoding, independent request batches, and CI fuzz coverage.

- **Evidence:** Python/Lean/Rust agreement on the dense case; upstream complete
  proof with output 48; two eight-token batches with outputs 48 and 6; 132 Rust
  tests; no-std build; all five fuzz targets passing locally, including 1,583
  compact-proof cases and 2,778,910 layout cases.
- **Next:** M2 GPU sumcheck/FRI kernels and larger model/window benchmarks.
- **Baseline:** multilinear small-model release benchmarks: Rust 2.94 ms proving
  and 3.11 ms verification; ICICLE CPU 9.02 ms and 6.26 ms after pairing witness
  folds (down from 14.73 ms and 9.26 ms). Native CUDA with persistent per-thread
  buffers/streams and operation-specific transfer paths: 5.82 ms proving and
  5.73 ms verification on the GB10, down from 7.78 ms and 5.80 ms with copies
  alone. The four-layer/f8/window-8 benchmark takes 4.59/4.86 ms locally and
  21.98/17.20 ms on the GB10 over 30 samples;
  it is not an 8B result because no 8B fixture or loader is present.
  The backend uses Rust below 8,192 input elements and CUDA at and above it;
  threshold selection reduces this larger fixture to 8.68/8.80 ms in interleaved
  A/B runs, while preserving byte-identical proof encodings.
  CI includes two-layer and two-request cases.
- **M2 kernel lane:** ICICLE v4 selected; `fp::gpu::sumcheck_fold` and the
  protocol-exact `sumcheck_fold_extension` pass CPU differential tests, including
  cubic `X^3 = 2` reduction, and have an explicit CUDA acceptance test.
  Fusion and host-side splitting reduced the 1,024-element ICICLE CPU fold from
  22.26 ms to 340 us; plain Rust remains faster at this small size.
  `DotProductRoundProver` now emits quadratic multilinear rounds and uses the
  kernels under `icicle-prover`; non-power-of-two witnesses are zero-padded and
  both witness vectors fold after every challenge. `Tensor::folded_dense` and
  proximity `fold_low_once` use the same verified arithmetic. All 133
  ICICLE-enabled tests pass, and Rust/ICICLE multilinear block proofs are
  byte-identical.
  `formal/GpuFold.lean` proves universal fused-fold and quadratic-dot-round
  correctness plus non-vacuity without `sorry` or user axioms. A canonical
  Goldilocks/cubic-extension model and the fixed CUDA launch now also have
  universal per-thread correctness, input-bound, inactive-thread, and
  non-vacuity proofs. Exact unsigned-word addition, subtraction, 128-bit
  reduction, and multiplication are proved against canonical modular arithmetic
  and connected to the field model. This closes the source-level field/index and
  machine-word arithmetic gaps, not PTX semantics; SF Tensor has not published
  its Lean model.
  `native-cuda` builds the fixed NVCC kernel through a scoped FFI; all 138 tests
  pass on the GB10 and native-CUDA proofs are byte-identical to Rust. PTX launch,
  termination, UB, and synchronization evidence is recorded in
  `spec/ptx-review.md`. Large dot-product witnesses stay resident across rounds;
  each fold computes the next round's coefficients from the folded values.
  Generic fold chains also reuse one per-thread context and alternate buffers.
  Next: import an available PTX model for compiler/hardware semantics and scale
  benchmarks to the 8B serving model.
