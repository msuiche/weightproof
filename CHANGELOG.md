# Changelog

## [0.1.0.0] - 2026-11-09

### Added
- Verify accepted FlashProve key, link, and request transcripts, including canonical parsing, Merkle openings, algebraic claims, compact proof encoding, and mutation rejection.
- Prove exact dense integer-model execution with committed weights, trace, lookup tables, and output tokens, including independent request batches.
- Generate quadratic multilinear dot-product proofs with zero-padded witnesses and paired witness folding.
- Run protocol-exact Goldilocks and cubic-extension folds through ICICLE CPU or a native NVCC CUDA backend, with Lean correctness and non-vacuity proofs.
- Execute the complete 134-test suite on an NVIDIA GB10 and generate `sm_121` PTX.
- Reuse per-thread CUDA buffers and streams to reduce small-model block proving to 7.78 ms and verification to 5.80 ms while preserving byte-identical proofs.
- Document local, CPU-backend, and DGX CUDA execution environments, verification commands, performance baselines, and PTX safety evidence.

### Infrastructure
- Add Criterion block and kernel benchmarks, parser fuzz targets, CI fuzzing, ICICLE tests, and pinned Lean checks.
