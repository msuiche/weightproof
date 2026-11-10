# Execution environments

## Why CUDA runs on `dgx1`

The development laptop is an Apple Silicon Mac. It has no NVIDIA GPU and current
CUDA toolkits do not support macOS, so installing NVCC locally would neither run
CUDA kernels nor validate GPU memory behavior. ICICLE CPU, Rust, Lean, no-std,
fuzzing, and default benchmarks run locally.

CUDA acceptance runs on the already-configured `dgx1` SSH host:

- Ubuntu 24.04.5 LTS, aarch64
- NVIDIA GB10, driver 580.178.04
- NVCC 13.0.88 at `/usr/local/cuda/bin/nvcc`
- Rust/Cargo under `~/.cargo/bin`

No packages were installed. The only PATH addition required is
`PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH`.

## Backend selection

| Configuration | Where it runs | Purpose |
|---|---|---|
| default | laptop or Linux CPU | Rust verifier and multilinear prover |
| `icicle-prover` | laptop or Linux CPU | ICICLE Goldilocks differential tests and CPU baseline |
| `native-cuda` | `dgx1` GB10 | Fixed NVCC kernel, full proofs, PTX generation |

ICICLE remains the evaluated CPU backend. Its CUDA source repository is not
publicly available and its release binaries target x86 Linux, while `dgx1` is
aarch64. `native-cuda` therefore builds `cuda/fold.cu` directly with NVCC through
`build.rs`. Unsafe Rust is confined to `src/fp/native_cuda.rs`; the rest of the
crate retains `#![deny(unsafe_code)]`.

## Source and fixture synchronization

The laptop checkout is the source of truth. Do not copy `target/` build artifacts
between different architectures. Required test fixtures are embedded with
`include_bytes!`, so `spikes/plans` and the accepted-proof fixture directory must
also be present for `cargo test`.

```bash
rsync -az --exclude .git --exclude target ./ dgx1:/home/msuiche/weightproof/
rsync -az spikes/plans dgx1:/home/msuiche/weightproof/spikes/
rsync -az spikes/flashprove/examples/accepted-proof \
  dgx1:/home/msuiche/weightproof/spikes/flashprove/examples/
```

The full FlashProve checkout is not required for the Rust unit tests. Ground-truth
accepted and mutated request proofs are exercised from these fixture files.

## Verification commands

Local:

```bash
cargo test --locked
cargo test --locked --features icicle-prover
cargo check --locked --no-default-features --target thumbv7em-none-eabi
~/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean formal/GpuFold.lean
```

On `dgx1`:

```bash
cd /home/msuiche/weightproof
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
make -C cuda check
cargo test --locked --features native-cuda
cargo bench --locked --features native-cuda --bench block_proof -- \
  --sample-size 10 --measurement-time 1 --warm-up-time 0.1
```

`make -C cuda check` builds `target/cuda/fold` and `target/cuda/fold.ptx`, then
checks 512 outputs at each stride 1 through 512 against independent full-precision
128-bit host arithmetic. NVCC targets `sm_121` on this machine.

### Self-hosted CI

`dgx1-weightproof` is registered with labels `self-hosted`, `Linux`, `ARM64`,
`cuda`, and `gb10`. It runs as the user-level `weightproof-actions.service`, not
as root. The runner uses `/home/msuiche/actions-runner` and the same CUDA/Rust
PATH shown above. The `cuda` job in `.github/workflows/verifier.yml` runs
`make -C cuda check` and `cargo test --locked --features native-cuda`. Native CI
also clones FlashProve before compiling so every `include_bytes!` fixture is
available. Ubuntu 26 fuzz jobs explicitly use `x86_64-unknown-linux-gnu` because
the default musl target is incompatible with ASan's static libc.

For security, the native job is gated with `github.event_name != 'pull_request'`.
It runs only on pushes and manual dispatch, so pull requests cannot execute
arbitrary code on the persistent GPU host. Check registration and service health
with:

```bash
gh api repos/msuiche/weightproof/actions/runners
ssh dgx1 'systemctl --user status weightproof-actions.service --no-pager'
```

## Verified native-CUDA results (2026-11-09)

- 134 tests pass: 127 unit tests and 7 ground-truth integration tests.
- The suite includes real quadratic multilinear block proving, accepted proofs,
  statement mutations, and request-proof rejection.
- Native CUDA and default Rust produce byte-identical multilinear proofs with
  SHA-256 `04d95d7500f779145f1018a711496f318d1baa257b5ebada780640a0ea92c7ea`.
- Pre-persistent baseline: 9.76 ms proving and 7.74 ms verification for the
  small-model block. Persistent per-thread buffers and streams reduce this to
  **5.82 ms proving** and **5.73 ms verification**. Proving uses mapped host
  memory; verification uses explicit copies. The persistent implementation
  passes the same 134 tests and preserves the byte-identical proof digest above.

### Profiling and transfer selection

Nsight Systems shows that the original copy-heavy path spends about 3.98 us per
`cudaLaunchKernel`, 5.68 us per `cudaMemcpyAsync`, and 1.61 us executing the fold
kernel. Pinned staging alone regressed proving by 21.5% and was rejected. Mapped
memory helped proving but hurt verification, so `prove_block_forest` selects
mapped transfers and `verify_block_forest` selects copies. Both paths are tested
independently. Thirty-sample results are 5.82/5.73 ms for the small block and
21.98/17.20 ms for the larger fixture (prove/verify).

### Rejected host-allocation experiment

A subsequent experiment passed extension slices directly through a `repr(C)`
layout, folded in place, and reused the paired host buffer. It passed all 134
CUDA tests but regressed block verification from 5.80 ms to 7.76 ms; proving was
7.88 ms versus 7.78 ms. The changes were reverted and not included in the pushed
implementation. The 7.78/5.80 ms persistent-device context remains the accepted
baseline. Further work should profile stream/copy latency before trying more
host-layout changes.

### Larger model/window benchmark

`block-4-layer-f8-window8` uses four transformer layers, feed-forward width 8,
an eight-token maximum window, and a four-token input. It is a scaling fixture,
not an 8B model. Every run validates the proof before measuring. Results:

| Backend | Prove | Verify |
|---|---:|---:|
| Local Rust | 4.59 ms | 4.86 ms |
| GB10 native CUDA | 21.80 ms | 15.95 ms |

No 8B weight fixture or loader exists in this repository, so these numbers are
not presented as 8B results.

## Formal and PTX evidence

`formal/GpuFold.lean` proves the fused cubic-extension fold and multilinear
quadratic-round identities without `sorry` or user-added axioms.
`spec/ptx-review.md` records the fixed kernel's launch bounds, termination,
UB-freedom guards, and stream synchronization evidence. It also distinguishes
that reviewed evidence from a future mechanized proof in SF Tensor's PTX model.
