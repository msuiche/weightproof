# Changelog

## [0.1.1.3] - 2026-11-12

### Added
- Add BF16 token-level GLP evaluation with an exact 493/512 null-gate baseline and 492/512 full-strength agreement, reported separately from the stronger float32 result.

## [0.1.1.2] - 2026-11-12

### Added
- Add a token-level GLP differential using FlashProve's integer/float32 model pair, including exact null gates and 509/512 top-choice agreement at 14-bit full-strength steering on the synthetic fixture.

## [0.1.1.1] - 2026-11-12

### Added
- Add a NumPy GLP differential harness and measured GLM-5.3 GLP-77 results, with 14-bit quantization preferred after comparing 8-, 12-, and 14-bit shifts over 121,110,528 values per configuration.

## [0.1.1.0] - 2026-11-12

### Added
- Bind rank-1 GLP project vectors, alpha, fixed-point quantization, layer mapping, and post-layer hook metadata into dense block statements.
- Parse GGUF v3 GLP artifacts with fail-closed mode/hook/version checks, deterministic F32 quantization, and parser fuzzing.
- Prove post-layer GLP projections through the existing arithmetic and residual witness streams, with wrong-vector, wrong-alpha, omitted-steering, and commitment-mutation rejection tests.

## [0.1.0.1] - 2026-11-11

### Added
- Expose complete request verification through `verify_request_transcript`, returning named rejection errors instead of escaping proof or parser failures.
- Add the `weightproof verify-key` and `verify-request` CLI commands with exit status 0 for verified, 1 for rejected, and 2 for usage errors, plus key/request integration tests.
- Build and publish verifier binaries with SHA-256 checksums for Linux and macOS on amd64 and arm64, and document CLI usage and installation.

## [0.1.0.0] - 2026-11-09

### Added
- Verify accepted FlashProve key, link, and request transcripts, including canonical parsing, Merkle openings, algebraic claims, compact proof encoding, and mutation rejection.
- Prove exact dense integer-model execution with committed weights, trace, lookup tables, and output tokens, including independent request batches.
- Generate quadratic multilinear dot-product proofs with zero-padded witnesses and paired witness folding.
- Run protocol-exact Goldilocks and cubic-extension folds through ICICLE CPU or a native NVCC CUDA backend, with Lean correctness and non-vacuity proofs.
- Prove canonical Goldilocks extension folding and fixed CUDA launch/index safety in Lean, including active/inactive thread behavior and a concrete non-vacuity witness.
- Prove the CUDA unsigned-word `add`, `sub`, 128-bit `reduce128`, and `mul` implementations against canonical modular arithmetic, with field-level correctness wrappers.
- Model the fixed kernel's emitted PTX instruction subset in Lean and prove `ptxRunThread_correct` and `ptxRunThread_eq_ssa`, connecting active/inactive instruction execution to the independent SSA and canonical fold.
- Keep large dot-product witnesses resident across CUDA rounds, computing the next round's coefficients from folded values in the same launch; reduce an 8,192-pair proof from 2.24 ms to 2.08 ms with byte-identical proofs.
- Reuse a per-thread resident context for generic fold chains and large dense tensors, reducing a thirteen-challenge 8,192-element chain from 208–241 us to 119–130 us.
- Reuse dot-product streams and buffers across proofs, reducing isolated create/upload/drop cycles from 183.8-246.1 us to 21.8-22.3 us.
- Run large dense folds in default Rust as well as native CUDA, fixing `base-linear-claim` verification for 8,192-element tensors and covering both backends with the same regression test.
- Execute the complete 139-test suite on an NVIDIA GB10 and generate `sm_121` PTX.
- Reuse per-thread CUDA buffers and streams, with mapped transfers for proving and copies for verification, to reduce small-model block proving to 5.82 ms and verification to 5.73 ms while preserving byte-identical proofs.
- Measure four-layer, feed-forward-width-8, eight-token-window proofs on Rust and native CUDA without labeling the fixture as an 8B model.
- Select Rust below the measured 8,192-element fold crossover and native CUDA above it, reducing the larger block fixture to 8.68 ms proving and 8.80 ms verification while preserving byte-identical proofs.
- Document local, CPU-backend, and DGX CUDA execution environments, verification commands, performance baselines, and PTX safety evidence.

### Infrastructure
- Add Criterion block and kernel benchmarks, parser fuzz targets, CI fuzzing, ICICLE tests, pinned Lean checks, and a self-hosted GB10 runner restricted to trusted pushes.
