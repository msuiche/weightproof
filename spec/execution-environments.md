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
| `native-cuda` | `dgx1` GB10 | Rust below 8,192 elements; fixed NVCC kernel at and above it |

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
~/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean -s 1048576 formal/GpuFold.lean
```

## Verifier CLI and releases

Build the CLI with `cargo build --release --bin weightproof`. It returns 0 for
`verified`, 1 for `rejected` with a named cause, and 2 for usage errors.

```bash
weightproof verify-key \
  --policy spikes/flashprove/examples/accepted-proof/config/policy.txt \
  --layout spikes/flashprove/examples/accepted-proof/config/layout.txt \
  --kl spikes/flashprove/examples/accepted-proof/config/KL.bin \
  --key spikes/flashprove/examples/accepted-proof/ivk/key.bin

weightproof verify-request \
  --policy spikes/flashprove/examples/accepted-proof/config/policy.txt \
  --layout spikes/flashprove/examples/accepted-proof/config/layout.txt \
  --kl spikes/flashprove/examples/accepted-proof/config/KL.bin \
  --leaf spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin \
  --statement spikes/plans/request-statement.bin \
  --request spikes/flashprove/examples/accepted-proof/example.req \
  --first-plan spikes/plans/request-first-plan.bin \
  --proof spikes/flashprove/examples/accepted-proof/example.proof
```

`tests/cli.rs` covers usage errors, accepted key/request verification, and a
mutated request proof. `.github/workflows/release.yml` builds and publishes the
binary plus SHA-256 checksum on GitHub Releases for `x86_64` and `aarch64` Linux
and macOS. `cargo install --path .` installs the same CLI locally.

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

- 139 tests pass: 132 unit tests and 7 ground-truth integration tests.
- The suite includes real quadratic multilinear block proving, accepted proofs,
  statement mutations, and request-proof rejection.
- Native CUDA and default Rust produce byte-identical multilinear proofs with
  SHA-256 `04d95d7500f779145f1018a711496f318d1baa257b5ebada780640a0ea92c7ea`.
- Pre-persistent baseline: 9.76 ms proving and 7.74 ms verification for the
  small-model block. Persistent per-thread buffers and streams reduce this to
  **5.82 ms proving** and **5.73 ms verification**. Proving uses mapped host
  memory; verification uses explicit copies. The persistent implementation
  passes the same 139 tests and preserves the byte-identical proof digest above.

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

### Rejected scratch and polling experiments

Per-thread `RefCell` conversion buffers passed all 139 CUDA tests but were mostly
0.5–2% slower in repeated low-load A/B pairs, so they were reverted. Replacing
`cudaStreamSynchronize` with a `cudaStreamQuery` polling loop was neutral and was
also reverted. Nsight showed that round-boundary launch/synchronization overhead
dominates the tiny folds, so backend selection is more effective than another
host-layout or synchronization tweak.

### Measured backend crossover (2026-11-10)

`gpu::sumcheck_fold` and `gpu::dense_fold_extension` use Rust below 8,192 input
elements and native CUDA at and above it. The same validation and field equations
apply on both paths. Size-sweep results on the GB10:

| Input elements | Rust | Native CUDA |
|---:|---:|---:|
| 1,024 | 7.99 us | 18.59 us |
| 4,096 | 20.73 us | 35.32 us |
| 8,192 | 63.97 us | 40.95 us |
| 16,384 | 82.93 us | 61.57 us |
| 65,536 | 330.42 us | 190.91 us |
| 262,144 | 2,051.21 us | 737.69 us |

The larger block fixture improved from 18.4–23.8 ms to 8.68 ms proving and from
16.7–20.5 ms to 8.80 ms verification in interleaved A/B runs. The other GPU
workload made baseline samples bimodal, but every threshold sample stayed near
8.7/8.8 ms. Small-block movement was minor by comparison, so the 5.82/5.73 ms
small-block baseline above is retained rather than replaced with a noisy number.
Baseline and threshold arithmetic/forest proof encodings were byte-identical with
SHA-256 `ffd242f078c45cc97c32fd3d4992c6cad04ec159ad59c3180320723114b40c66`.

### Device-resident dot-product rounds (2026-11-10)

At and above the crossover, `DotProductRoundProver` uploads both witnesses once.
Each `dot_round_kernel` launch folds the current pair and computes the following
round's three coefficients from the folded values. The initial coefficient pass
is the only additional launch. No full witness returns until final evaluation.
`Ext.nextDotRound_correct` proves the fold-then-coefficient composition in Lean.

Three interleaved A/B runs for a complete 8,192-pair PSC proof gave baseline
2.24/1.74/2.25 ms and resident 2.09/1.66/2.08 ms (baseline/resident ranges
1.74–2.25 ms and 1.66–2.09 ms). The large proof bytes are identical with SHA-256
`820fc388ddd948216ea3c99e2f0d127f139407240cab6d40d35b255027bd7922`.
All 142 native tests pass: 132 unit tests, 3 CLI tests, and 7 ground-truth
integration tests.

### Generic resident fold chains

`fold_low` and large `Tensor::folded_dense` paths use a per-thread reusable
`SeriesContext`, including its stream and alternate buffers. No stream or buffer
is allocated per chain. Thirteen challenges on an 8,192-element chain took
208–241 us before residency and 119–130 us after, with identical outputs.
The complete suite now has 142 tests: 132 unit tests, 3 CLI tests, and 7
ground-truth integration tests. Non-native builds run the same large dense-fold
loop at every size; skipping it caused `base-linear-claim` failures. The 8,192-element
regression test now runs on both backends, and the default suite has 136
passing tests.

### Cross-proof dot-context residency

Large dot-product proofs also reuse a per-thread `DotContext`, including its
stream, coefficient buffer, and alternate witness buffers. Each proof resets
the logical length and current buffer and uploads fresh witnesses; capacity is
retained until thread exit. Sequential-proof tests check fresh coefficients
and folded outputs after reinitialization.

Five interleaved 200-iteration measurements of an 8,192-pair
create/upload/drop cycle cost 183.8-246.1 us before reuse and 21.8-22.3 us
after. This is an isolated initialization result; shared-GPU full-proof
samples remained bimodal and are not used to claim a new end-to-end timing.

### Larger model/window benchmark

`block-4-layer-f8-window8` uses four transformer layers, feed-forward width 8,
an eight-token maximum window, and a four-token input. It is a scaling fixture,
not an 8B model. Every run validates the proof before measuring. Results:

| Backend | Prove | Verify |
|---|---:|---:|
| Local Rust | 4.59 ms | 4.86 ms |
| GB10 native CUDA before thresholding | 18.4–23.8 ms | 16.7–20.5 ms |
| GB10 threshold-selected backend | 8.68 ms | 8.80 ms |

No 8B weight fixture or loader exists in this repository, so these numbers are
not presented as 8B results.

## Formal and PTX evidence

`formal/GpuFold.lean` proves the fused cubic-extension fold and multilinear
quadratic-round identities without `sorry` or user-added axioms. Its `Ptx`
model covers the fixed kernel's emitted instruction subset;
`ptxRunThread_correct` proves active and inactive PTX threads equal
`Launch.runThread`, and `ptxRunThread_eq_ssa` connects the instruction stream to
the independently written arithmetic SSA. This does not model compiler
lowering, SASS, or the full PTX memory model. `spec/ptx-review.md` records the
fixed kernel's launch bounds, termination, UB-freedom guards, and stream
synchronization evidence.
