# Review: SF Tensor's PTX-in-Lean formalization

Sources reviewed 2026-11-09:

- [Formalizing PTX in Lean to Prove GPU Kernel Correctness](https://sf-tensor.com/engineering/formalizing-ptx)
- [Paper and proof model](https://sf-tensor.com/papers/formalizing-ptx.pdf)

## What is proved

The work models PTX executions in Lean and checks universal correctness and a
separate non-vacuity theorem for every presented kernel. It covers data-dependent
control flow, acquire-release synchronization, asynchronous copies, TMA, proxy
fences, warp shuffles, atomics, and sequential multi-kernel pipelines. Reported
examples include Hopper WGMMA GEMM, online FlashAttention, a three-kernel MoE
pipeline, and an atomic-work-distribution megakernel. The development has no
`sorry` and no user-added axioms.

The model's useful properties are:

- Conservative semantics: ambiguous points admit more outcomes, so an incorrect
  kernel cannot be made correct by weakening the model.
- Every completed consistent execution must satisfy the postcondition, and a
  concrete execution is constructed to rule out vacuity.
- A successor kernel's initial memory is defined from the predecessor's output,
  giving a practical composition interface for proof pipelines.
- Integer operations are modeled exactly, rather than treating every arithmetic
  result as an approximation.

## Limits that matter for weightproof

- **Partial correctness only.** The work does not prove termination of polling or
  acquire-release loops. A separate progress/termination argument is required.
- **UB is external.** ISA-level undefined behavior and correct `ptxas`/hardware
  behavior are assumptions, not proved properties.
- **Two memory-model axioms are omitted.** No-thin-air and Fence-SC are omitted,
  which admits more executions and can prevent a proof, but does not make an
  incorrect theorem true by itself.
- **Floats are reals.** The model does not establish IEEE rounding, quantization
  error, numerical stability, or bit-level float reinterpretation. Our exact
  integer model is a better fit, but it needs its own modular-field semantics.
- **Coverage and scale.** The current interpreter has 81 instruction forms and
  the examples are hand-written. The paper reports 34,034 proof lines for 454
  instructions, so production compiler output will need substantial automation.
- **No direct Goldilocks proof.** The paper does not provide the field modulus,
  canonical reduction, or the cubic extension relation $X^3-2$. Those must be
  specified and proved by us.
- **No SASS model.** SASS is not fully documented. PTX-level compiler reordering
  and optimized output remain proof liabilities.

## Reuse decision

Reuse the memory model and the multi-kernel composition interface for M2, not the
existing kernel proofs. Do not treat this as evidence that a GPU prover is correct
by construction.

The appropriate first target is a small, hand-written integer sumcheck/FRI kernel
with fixed public parameters. The MoE pipeline example is directly relevant to
expert routing and batched expert coverage, but our version must also bind the
exact active-expert set and its commitment.

Before using the model, specify:

1. Goldilocks $p=2^{64}-2^{32}+1$, canonical representatives, and cubic extension
   $E=F[X]/(X^3-2)$.
2. Exact integer tensor operations, table semantics, expert selection, and the
   per-request statement/parameter binding.
3. A progress argument for any polling loop and a UB-freedom argument for every
   instruction and launch shape.
4. A non-vacuity witness and a differential execution against the current C++/Rust
   verifier for the same public input.
5. A termination and memory-visibility argument for every asynchronous stage and
   every kernel-to-kernel handoff.

## Acceptance gate for M2

An M2 kernel is accepted only when all of these hold:

- Universal correctness and non-vacuity are proved in Lean for the kernel's
  integer semantics and the statement it is intended to prove.
- Termination and UB-freedom are proved or discharged by a reviewed, explicit
  assumption with a smaller runtime guard.
- The generated proof uses no `sorry` and no user-added axioms.
- The Lean and Rust paths agree on the accepted key/request fixtures and reject
  the same mutations.
- The proof binds the exact committed weights and active experts; it does not
  merely prove an unrelated mathematical kernel.
- Differential serving tests still compare the integer model's tokens with the
  serving engine. A correct kernel for a different computation is a statement
  failure, not a valid optimization.

Defer adoption until the M2 kernel shape exists. Reusing the PTX model early is
useful; claiming its current examples cover our field arithmetic or protocol is
not.

## Native fold evidence (2026-11-09, `dgx1`)

The first concrete target now exists as `cuda/fold.cu`, built by `build.rs` under
`native-cuda` and linked through the reviewed `fp::native_cuda` FFI boundary.
NVCC 13.0.88 generates PTX 9.0 for `sm_121` on the NVIDIA GB10.

- The kernel has one bounded index computation and no loop, polling, recursion,
  asynchronous copy, or kernel-to-kernel handoff. Every thread returns directly.
- `index >= output_count` returns before any global load/store. A zero `low_half`
  returns before division. Host and FFI guards additionally require nonempty even
  input, exact output length, power-of-two stride, and stride no larger than the
  output half. These guards discharge the identified division and out-of-bounds
  risks for accepted launches.
- Proving uses mapped host input/output; verification uses explicit H2D/D2H copies.
  Each path uses one per-thread CUDA stream. `cudaGetLastError` and
  `cudaStreamSynchronize` run before any host output is returned. Device and
  mapped host buffers are reused, and no later read depends on an unsynchronized
  kernel write.
- The standalone test checks 512 outputs at each stride 1, 2, 4, ..., 512 against
  independent full-precision 128-bit host arithmetic, and checks non-vacuity.
- The Rust `sumcheck_fold_matches_rust_on_cuda` test and the complete 139-test
  `native-cuda` suite pass. This includes multilinear block proofs, large
  device-resident dot-product rounds, cross-proof context reinitialization,
  generic fold chains, and accepted/mutated key/request fixtures, not an
  isolated arithmetic benchmark.
- Rust and native-CUDA multilinear block proofs are byte-identical.

`formal/GpuFold.lean` proves the integer-lift equations and their canonical
Goldilocks/cubic-extension counterparts. `Ext.nextDotRound_correct` additionally
proves the fused fold-then-next-coefficient composition used by
`dot_round_kernel`. `Kernel.Launch` models the
fixed kernel's guards and source-level index arithmetic: every active thread's
two checked loads are proved in bounds, `runThread_correct` proves its output
against the independent reference fold, inactive threads perform no load/store,
and a concrete valid launch proves non-vacuity. `Kernel.Word` additionally models
the unsigned wrapping operations and proves `add_correct`, `sub_correct`,
`reduce128_correct`, and `mul_correct` against canonical modular arithmetic;
`Field.word_add_correct`, `Field.word_sub_correct`, and `Field.word_mul_correct`
connect those results directly to the field operations used by the fold proof.
The `Ptx` model covers the fixed kernel's emitted instruction subset:
`ptxRunThread_eq_ssa` proves the instruction stream equal to an independent
arithmetic SSA, and `ptxRunThread_correct` proves active and inactive PTX
threads equal `Launch.runThread` and therefore the canonical fold. Lean reports
no `sorryAx` or user-added axioms for these theorems. This is a self-contained
source-level field/index/machine-word and instruction-subset model, not a full
PTX memory-model proof: SF Tensor has published its paper but not its Lean
implementation, so no direct reuse is claimed. Compiler lowering, SASS, and
hardware remain covered by the reviewed guards and differential evidence above.
Mechanizing those layers requires an available full PTX model. Scaling to the 8B
serving model and eliminating per-fold allocation/copies are separate
performance work, not correctness assumptions.
