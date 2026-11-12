# weightproof

Experimental Rust proofs for LLM inference with committed weights and **GLP steering**. The GLP-specific goal is to bind the base model, a rank-1 project vector, its strength and layer/hook metadata, the resulting execution, and the output token to one statement.

This is a research playground for learning verification, not a production proving service. It is not audited, does not yet provide zero knowledge, and does not claim that the exact integer model is bitwise-equivalent to a floating-point vLLM serving lane.

## What is implemented

- A no-std + alloc verifier core and a std CLI.
- Dense integer-model execution with weight, parameter, residual, lookup-table, and output commitments.
- Multilinear product, dot-product, residual, and lookup-forest transcripts, with independent request batches.
- Rank-1 GLP `project` vectors at `residual_stream_post_layer`:
  - real GGUF v3 parsing, including tensor alignment and metadata skipping;
  - fail-closed rejection of additive, rank-k, alternate-hook, unsupported-version, malformed, and nonfinite inputs;
  - deterministic F32-to-fixed-point quantization;
  - statement binding for the complete GLP artifact, alpha, quantization, and 1-based layer/direction mapping;
  - projection proofs using the existing arithmetic and residual witness streams.
- FlashProve key, link, and request-transcript verification.
- Optional ICICLE CPU and native NVCC CUDA arithmetic backends, plus Lean proofs for the fixed CUDA fold kernel.

The parser has also accepted the published `GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf` artifact as a one-off compatibility check.

## GLP computation

For quantization exponents `sd` and `sa`, each F32 direction coefficient and alpha is multiplied by `2^sd` or `2^sa` and rounded to nearest-even. For integer direction `D`, integer alpha `A`, and post-layer residual row `h`:

```text
p       = sum_i h[i] * D[i]
S       = 2 * sd + sa
term[i] = rq(p * A * D[i], S)
out[i]  = h[i] - term[i]
```

`rq` is the model's round-to-nearest, ties-up integer operator. There is no intermediate rounding of `p` or `p * A`. Layer `direction.N` applies immediately after layer `N`'s second residual addition.

The exact contract and rejection requirements are in [`spec/integer-model.md`](spec/integer-model.md#13-rank-1-glp-steering-g0). The GLP container semantics come from [`weightless/docs/spec/GLP.md`](https://github.com/msuiche/weightless/blob/main/docs/spec/GLP.md). Measured quantization error and the 14-bit recommendation are in [`spec/glp-differential.md`](spec/glp-differential.md).

## Build and test

Requires Rust 1.85 or newer.

```bash
cargo build --locked
cargo test --locked
cargo test --locked glp
cargo check --locked --no-default-features --target thumbv7em-none-eabi
```

Fuzz the GLP parser with `cargo-fuzz` on nightly:

```bash
cd fuzz
cargo fuzz run fp_glp -- -max_total_time=60 -timeout=5
```

CI runs the default, ICICLE, native CUDA, Lean, no-std, and parser-fuzz jobs. See [`spec/execution-environments.md`](spec/execution-environments.md) for backend-specific commands.

## Rust API

```rust
use weightproof::fp::{
    glp::{GlpQuantization, GlpSteering},
    integer_model::{prove_block_forest, small_model, verify_block_forest},
};

// A test GGUF containing one two-wide `direction.1` tensor.
let gguf = std::fs::read("test-vector.gguf")?;
let steering = GlpSteering::from_gguf(
    &gguf,
    GlpQuantization::new(14, 14).unwrap(),
).unwrap();

let mut model = small_model();
model.steering = Some(steering);
let tokens = [0, 1];
let proof = prove_block_forest(&model, &tokens, 4, 4, 1).unwrap();
assert!(verify_block_forest(&model, &tokens, &proof));
```

A proof generated with steering fails verification against an unsteered model, another direction, or another alpha. Mutating the statement's steering commitment also fails closed.

## Verifier CLI

Install locally:

```bash
cargo install --path .
```

The binary returns `0` for verified, `1` for rejected with a named cause, and `2` for usage errors:

```text
weightproof verify-key --policy <path> --layout <path> --kl <path> --key <path>
weightproof verify-request --policy <path> --layout <path> --kl <path> --leaf <path> --statement <path> --request <path> --first-plan <path> --proof <path>
```

Complete fixture commands and release targets are documented in [`spec/execution-environments.md`](spec/execution-environments.md#verifier-cli-and-releases).

## Current limits

- Only rank-1 GLP `project`, spec version 1, and `residual_stream_post_layer` are supported. Rank-k, additive, FFN/attention-writer, and hyper-connection fold-site semantics are rejected.
- No 8B checkpoint loader or 8B performance claim.
- No zero-knowledge implementation; weight hiding remains future work.
- No production claim that the integer reference model matches floating-point serving. That requires differential evaluation against `weightless`.
- Lean covers the fixed arithmetic kernel, not compiler lowering, SASS, or the complete PTX memory model.
- A commitment authenticates a model or vector only when an independent party certifies its identity. Provider self-assertion is insufficient.

## Repository map

- `src/fp/glp.rs`: GLP GGUF parsing, quantization, and statement commitments.
- `src/fp/integer_model.rs`: exact dense execution, projection, statements, and block proofs.
- `src/fp/`: FlashProve transcript verification and multilinear proof machinery.
- `formal/GpuFold.lean`: fixed-kernel arithmetic and launch proofs.
- `spec/`: normative model, security, execution, and lifecycle documents.
- `fuzz/fuzz_targets/fp_glp.rs`: arbitrary-byte GLP parser fuzzing.
- `PLAN.md`: architecture, decisions, milestones, and deferred work.

## License

Apache-2.0.
