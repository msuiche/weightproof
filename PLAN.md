# PLAN — awesome-weightproof

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
  format (NVFP4/MXFP4) + pruning. REAP-pruned builds change the circuit —
  weightless rejects them for steering; weightproof must reject them for
  proving (or bind the pruned topology explicitly).


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
        M2 8B dense, GPU kernels (evaluate ICICLE before custom cudarc)
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
