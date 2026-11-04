# TODOS — zkvm-inference

Captured by /plan-eng-review, 2026-11-03/04. Plan state: D1–D16 decisions applied
(incl. codex outside-voice findings). First work: D11 FlashProve reproduction +
Rust verifier, D3/D14 WHIR security-gated spike, D4/D15 Lean composition track.

## 1. Protocol spec gaps: proof-pending, commitment renewal + identity, versioning

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

- **What:** Review https://sf-tensor.com/engineering/formalizing-ptx (+ linked
  paper) for reuse in verifying our GPU prover kernels in Lean.
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
