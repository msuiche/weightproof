# PLAN — weightproof

Provable LLM inference in Rust: prove that a response came from committed model
weights, without revealing the weights, with cheap CPU verification.

## Ecosystem: weightless (../spark/weightless)

Matt's projective-steering project (GLP vectors, vLLM/SGLang plugins, DGX Spark
+ Modal H100/H200 lanes). Direct synergies, verified against its README/spec:

- **Statement extension:** a steered serving stack computes
  `(base weights, GLP vector, α, hook points) → tokens`, so the proof statement
  binds `H(glp_gguf) + α + hook metadata` alongside the weight commitment.
  Use cases: prove safety/compliance steering WAS applied per request; prove an
  abliterated model was served rather than secretly-stock (users cannot tell
  from outputs alone — the exact trust gap this project exists to close);
  prove NO unapproved intervention ran.
- **Proof cost is small:** GLP projection `h ← (I − α·d̂d̂ᵀ)h` is a rank-1
  LINEAR map for fixed (d̂, α) — fits lincheck/WHIR linear constraints with no
  new nonlinear gadgets. GLP files are few-hundred-KB public artifacts → public
  hash input; only the base weights need hiding.
- **Spec decode needs no drafter proofs:** post-hoc replay proving (prefill
  style) certifies the produced token stream regardless of DFlash2 drafting.
- **D13 divergence eval:** reuse weightless's refusal32/cyber32/benign32 suites
  to measure BEHAVIORAL divergence of the integer reference model, not just
  token mismatch.
- **Lane A hardware:** weightless's Modal cloud lanes (H100/H200, wizard
  deploy) are the reproduction + prover-benchmark deployment path — no new
  infra needed.
- **Commitment metadata (codex #8):** must bind checkpoint revision + quant
  format (NVFP4/MXFP4) + pruning. **REAP-pruned builds are rejected** (confirmed
  2026-11-04): deleting experts changes the circuit — the GLP vector no longer
  matches, the provable computation is a different model, and the degraded
  behavior is not worth supporting. Reject at commitment time with a named
  error, same discipline as weightless's lane rejection. Escape hatch if ever
  needed: bind the pruned topology explicitly as a new model identity, never
  silently prove against the unpruned commitment.


Status: reviewed (plan-eng-review + codex outside voice), all decisions applied.
Supersedes: the original 10-crate tensor-native sketch.
Artifacts: `TODOS.md` (deferred work), `spec/security-parameters.md` (D14 gate),
test plan in `~/.gstack/projects/zkvm-inference/`.

## Evidence base

- FlashProve (Anthropic, archived, Apache 2.0): bespoke IOP, 110 prefill tok/s on
  400B MoE, 1 H100, ~19 MB proofs per 1,024-token request, ~17 s CPU verify,
  Lean soundness proofs, C++/CUDA implementation. Proves the approach works.
- SP1/zkVM: ~10^6 cycles/s proving vs ~10^10 cycles/token → unusable at LLM
  scale. zkVM stays an optional shell for glue logic, never the matmul core.
- worldfnd/whir: Rust WHIR implementation (academic prototype, NOT production
  reviewed). Candidate core primitive, gated by security + benchmark evaluation.

## Current D11 increment (2026-11-09)

- `fp::verify_link` executes the stored LINK transcript, checks canonical field
  encodings and proof-of-work, derives positions from the chain, verifies all
  five Merkle openings, and checks LINK's algebraic claims through BAT, FOLD,
  CS, and BASE against the C++ oracle's accepted leaf.
- `fp::request_frame` executes the accepted request proof through the lookup
  forest, layer claims, output head, LIN second stage, and the request's own
  test, consuming the exact stored transcript. Request fixes cover expert-width
  row factors, embedding token/phi factors, signed RoPE coefficients, bounded
  QK layout points, dense tensor cuts, and dense-variable rotation.
- `fp::verify_key` executes the accepted key's complete transcript: the one-time
  WF/ranges stage, shared LIN stage, and second request-own test, with exact
  cursor exhaustion. Key plans, leaf domains, root-run aliases, opening
  contexts, and work scopes are bound separately from the request's plans.
- `fp::request_frame` executes the accepted request proof through the lookup
  forest, layer claims, output head, LIN second stage, and the request's own
  test, consuming the exact stored transcript. Request fixes cover expert-width
  row factors, embedding token/phi factors, signed RoPE coefficients, bounded
  QK layout points, dense tensor cuts, and dense-variable rotation.
- `fp::field` provides allocation-free Goldilocks and cubic-extension
  arithmetic, checked against C++ differential vectors. `fp::tensor`,
  `fp::claims`, `fp::code_eval`, `fp::proximity`, and `fp::link_algebra`
  implement the verifier-only claim and proximity algebra.
- Verification: all 107 Rust tests pass, including accepted-key and
  accepted-request acceptance, key/request mutation rejection, dense tensor
  regression tests, and key-transcript corruption rejection. The
  `thumbv7em-none-eabi` no-std check passes. Earlier evidence: all four fuzz
  targets completed 1,000 nightly runs without crashes.
- CI: `.github/workflows/verifier.yml` fetches the FlashProve ground-truth
  fixtures, runs the full Rust suite, and checks the no-std build on every push
  and pull request.
- Protocol: `spec/proof-lifecycle.md` defines proof-pending states, independent
  commitment certification and renewal, format/parameter versioning, and
  per-request extraction acceptance rules.
- Extraction: `spec/extraction-review.md` stages field/tensor/Merkle extraction
  before full verifier extraction and names the `dyn` interfaces to refactor.
- M2: `spec/ptx-review.md` records the PTX-in-Lean reuse decision, its partial
  correctness/UB/coverage limits, and the acceptance gate for integer kernels.
  `spec/execution-environments.md` separates local CPU checks from `dgx1` CUDA
  execution and records exact synchronization and verification commands.

## M1 integer model (started 2026-11-09)

- `spec/integer-model.md` fixes the exact dense forward-pass equations, table
  generation, range domain, and evidence requirements for the CPU prover.
- `fp::integer_model` implements floor division, `rq`, `clamp`, `low8`,
  `argmax`, `isqrt`, and the norm without gains. `forward` evaluates the full
  dense path with per-position K/V history and range checks. Public rotary,
  softmax, SiLU, and sigmoid tables match Python reference values. Canonical
  table serialization, byte regions, and mutation-sensitive public digests are
  implemented. Weight commitments, residual execution traces, and a complete
  `BlockStatement` bind one execution to its weights and output token. Rotary,
  gate, epsilon, and signed-range behavior now follows the upstream model; all
  four regenerated table digests match the accepted-proof policy exactly.
  Balanced-limb Goldilocks encoding covers weights and residual streams. A
  generated upstream Python dense-model vector matches the Rust intermediates,
  residual stream, and output token; it exposed and regression-tested an
  `isqrt` initial-guess bug. CPU prover rows encode embeddings, layer weights,
  residuals, and all public lookup tables. `protocol_inputs` maps them to native
  multilinear tensors and lookup forests. The upstream Lean comparator also
  passes the dense case: 28 arrays and 1,143 values, with zero differences.
  `LkpPublic` evaluates the encoded tables, and `BlockStatement` commits to all
  canonical protocol rows with mutation rejection. Dimensions, scales, rotary
  flags, and norm epsilons have a separate parameter commitment. `ProverTr`
  emits canonical messages with verifier-identical Fiat–Shamir challenges.
  `prove_psc_product` generates a complete product-relation proof that
  `psc_verify` accepts; transcript mutation is rejected. Generic round-polynomial
  proving covers a nonconstant linear relation, and the forest-shaped PSC
  composition reproduces the product transcript exactly. Slot-aware prover PSC
  rounds hide endpoints and build affine rows; all 31 initial forest sequences
  execute with zero residual rows. Product-level polynomials, line-challenge
  hand-down, final leaf rounds, and the terminal bit row are implemented. A
  complete generated zero-table forest transcript verifies through `forest_pad`
  with all 52 rows at zero residual. Nonzero SiLU tables also verify, and
  `BlockProof` binds a 64-row forest to the block statement with mutation
  rejection. All gate, attention, projection, rotary, norm, output-head, and
  residual-addition proofs share one chain. Explicit SiLU/softmax keys and
  values feed a combined 66-row forest; lookup-count, individual key/value, and
  execution mutations fail closed, with verification restored afterward. The
  upstream dense model verifies with output 48, 28,552 arithmetic proofs, 66
  rows, and 8 SiLU plus 128 softmax reads. Two-layer and independent request
  proofs verify, with four SiLU and twelve softmax reads and cross-request
  rejection. `prove_block_batch` and `verify_block_batch` expose independent
  requests; large counts split at 4,095 reads per tree. Two upstream eight-token
  proofs verify with outputs 48 and 6, 683,360 arithmetic proofs each, 67 rows,
  and cross-request rejection. Dot-product batching reduces each to 10,930
  arithmetic proofs (62.5× fewer) with identical verification. Residual batching
  leaves 10,675 arithmetic proofs per request; the two-request transcript is
  67,685,664 bytes, dominated by extension-field endpoints. Compact base-field
  endpoints reduce it to 23,967,008 bytes (2.82× smaller) with identical
  verification. `format_version = 1` binds the encoding; bad versions, offsets,
  truncation, and non-canonical values are rejected. Dot-product arithmetic now
  emits quadratic multilinear rounds with epsilon scaling and zero-padded witness
  tensors; `fold_low_pair_once` batches both witness vectors, and ICICLE-backed
  proving is byte-identical to Rust. `formal/GpuFold.lean` proves the fused-fold
  and quadratic-round equations with non-vacuity witnesses.
  `fp_compact_proof` completed
  1,583 libFuzzer cases without a panic and is registered in CI with the other
  parser targets. All five targets pass locally; `fp_layout` was restored in
  `fuzz/Cargo.toml` and completed 2,778,910 cases. All 132 Rust tests pass, and
  no-std passes.

## Decision register

| # | Decision |
|---|---|
| D2 | Single crate until content justifies more; no empty scaffolding |
| D3 | WHIR-reuse spike gates commitment-building work |
| D4 | Lean statements before code, started early in parallel |
| D5 | Sampling seed = H(commitment ‖ prompt ‖ client_nonce); greedy until built |
| D6 | Verifier: no_std core, feature-gated prover, CI no_std check |
| D7 | Concrete Goldilocks first; field generics after a second field exists |
| D8 | Full adversarial test suite from day 1 (see Test requirements) |
| D9 | Criterion benches + memory high-water + baseline report in first milestone |
| D11 | Reproduce FlashProve, then Rust verifier for its proofs, BEFORE rebuilding |
| D12 | Success gate = full transformer block vs committed weights, not lincheck |
| D13 | Exact integer reference-model semantics; decomposed metrics; verifier thresholds; Groth16 wrap deferred until required |
| D14 | WHIR security gate before benchmarks (spec/security-parameters.md) |
| D15 | Lean track covers composed-transcript simulator + randomness accounting across adaptive requests |
| D16 | Batching: per-client extractable verification required; expert-coverage measured |

## Workstreams (parallel lanes)

```
Lane A (needs x86 H100 box)   Lane B (Lean)              Lane C (laptop)
───────────────────────────   ─────────────────────────  ─────────────────────────
1. Reproduce FlashProve       Composed-transcript        1. spec/security-parameters.md
   small-model route          simulator + randomness        (DONE)
2. Rust verifier for          accounting across             (D14)
   FlashProve proofs          adaptive requests         2. WHIR spike: build, run,
   (no_std, D6)              (D4/D15)                      fixed-security benches (D9)
3. C++↔Rust agreement                                    3. Reuse verdict: adopt /
   test corpus on real                                     adapt / reject, written
   proofs                                                4. If reject: hand-roll
                                                              lincheck spike
        └──────────┬────────────┴────────────┬─────────────┘
                   ▼                         ▼
        GATE (D12): complete transformer block against committed weights.
        Every activation, residual, expert selection, KV entry, and output
        token bound to ONE execution and ONE weight commitment.
        Includes preprocessing and memory costs. No kernel scaling before
        this gate passes. Random-matrix lincheck = infra spike, not success.
                   │
                   ▼
        M1 dense GPT-2-class int8, CPU
        M2 8B dense, GPU kernels (ICICLE CPU + native NVCC CUDA)
        M3 MoE, work ∝ active experts, batched windows
        M4 ZK implementation, verifiable sampling (D5), chunked 128k context
```

Conflict flag: A and C want different bench harnesses (different proof formats) —
do NOT unify them prematurely.

## Acceptance metrics (D13) — replaces bare "110 tok/s"

Prover, reported separately at every milestone:
- prefill tokens/s proved; generated tokens proved/s; proof latency per request;
  amortized cost per request (GPU-seconds); memory high-water.

Verifier acceptance thresholds (initial, validated at D12 gate):
- ≤ 60 s per request proof on an 8-core laptop CPU; ≤ 4 GB RAM;
- ≤ 32 MB proof per 1,024-token window (FlashProve: 19 MB, 17 s on 24 cores).
- No Groth16/recursion wrap unless a demonstrated requirement (e.g. on-chain)
  forces smaller/cheaper verification.

## Reference-model semantics (D13) — precedes prover optimization

Written spec of the exact integer model being proved: quantization scales,
rounding, rescaling, overflow behavior, nonlinear approximations (softmax/exp
tables), argmax tie-breaking, and the range constraints that make field
equations imply integer arithmetic. Includes a measured divergence evaluation
between the integer model and the floating-point serving engine (FlashProve's
own limitation: integer-model token divergence, no task-quality benchmark).
A proof of a different computation than the one served is a statement-soundness
failure, the worst bug class in this system.

## Test requirements (D8, applies to every verifier artifact)

- Full reject matrix: tamper each proof section; wrong dims/statement/params;
  truncated; trailing bytes; non-canonical encodings; random bytes.
- Statement-replay MUST fail; params mismatch MUST fail closed with named error
  (critical gap from review — silent accept is a release blocker).
- cargo-fuzz arbitrary bytes → verify(); no panics, size caps not OOM.
- proptest differential field arithmetic vs num-bigint; determinism + roundtrip.
- CLI E2E: exit 0 YES / 1 NO+reason / 2 usage; every rejection names its cause.
- Fuzzing tests robustness, NEVER cryptographic soundness (D14).

## Batching requirements (D16)

- Per-request extractable verification: a client verifies its own request from
  the batch proof WITHOUT other clients' prompts/outputs.
- Expert-coverage instrumentation: distinct experts activated per window; the
  work-∝-active-experts assumption must be validated against realistic traffic.
- Both constrain transcript/commitment design together with D15's simulator
  requirements, before the D12 gate freezes interfaces.

## NOT in scope (explicit deferrals)

10-crate workspace (D2) · Groth16 wrap / recursion until required (D13) · ZK
implementation (M4) · sampled decoding implementation (D5) · >1k context (M4) ·
generic field traits (D7) · SP1 shell unless glue outgrows kernels · on-chain
verification · architecture hiding · releases (TODOS #3, at first artifact).

## Prerequisites / environment

- Lane A reproduction requires x86-64 Linux + one H100 80 GB + CUDA 13.2.86
  (FlashProve's pinned stack), 280–330 GB RAM for full size; small-model route
  needs 10 GB RAM. This workstation (macOS/arm64) cannot run it — provision a
  box or defer Lane A; the Rust verifier can still be developed against
  FlashProve's published spec (frame/FORMAT.md) and stored example proofs.
- Lanes B/C run anywhere with Lean 4 / Rust 1.98+.
