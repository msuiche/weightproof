# WHIR spike report — 2026-11-04

Commit: worldfnd/whir @83bc41744c994aa182c547d43c73ce3d1f76d04d
Machine: macOS/arm64 (Apple Silicon), rustc 1.98.1, release build (51 s).
Raw log: `spikes/results-whir-20261104.log`. Security: `-l 128`, Goldilocks3,
Blake3, rate 1, PoW default 20 (round-by-round analysis printed by binary).

## PCS benchmark matrix (D9 harness: time / size / peak RSS)

| Regime | d (vars) | Prover | Proof | Verifier (1k reps) | Peak RSS |
|---|---|---|---|---|---|
| Johnson (proven, BCSS25) | 16 | 45.5 ms | 125.1 KiB | 1.1 ms | 15.5 MB |
| Johnson | 20 | 124.1 ms | 179.2 KiB | 1.4 ms | 176 MB |
| Johnson | 24 | 1.7 s | 238.4 KiB | 1.6 ms | 2.59 GB |
| Unique (proven) | 16 | 65.6 ms | 230.8 KiB | 1.3 ms | 15.4 MB |
| Unique | 20 | 139.4 ms | 364.6 KiB | 1.8 ms | 177 MB |
| Unique | 24 | 1.6 s | 522.5 KiB | 2.4 ms | 2.59 GB |

Read: at equal security (128-bit, proven regimes only), Johnson ≈ same prover
time with ~2× smaller proofs at d=24. Verifier is ms-scale everywhere — the
"cheap CPU verification" goal is not at risk from this primitive. Peak RSS
grows steeply (~2.6 GB at d=24) — memory, not time, is the scaling watch item,
consistent with the SP1 RAM-cliff lesson.

## Constrained RS path (CLARK-relevant, D12 prerequisite)

| Config | Prover | Proof | Verifier (1k reps) |
|---|---|---|---|
| Johnson d=20, `--linear-constraints 100` | 1.7 s (vs 124 ms unconstrained) | 179.8 KiB (≈ unchanged) | 238.6 ms (vs 1.4 ms) |

Constrained proving works and proof size barely moves, but costs land where the
theory predicts: prover ~14× slower at 100 constraints, verifier ~170× slower
(verifier evaluates the constraints). For the D12 gate this means constraint
count per segment is a first-order design budget — batch constraints via RLC
rather than proving hundreds individually. (Wall time for this run was ~240 s,
dominated by 1000 verifier repetitions, not single verification.)

## Soundness configuration facts (D14)

- Both CLI regimes are proven: Unique (unique decoding), Johnson (BCSS25
  Thm 1.5). Conjectural `Capacity` (STIR Conj 5.6) is API-only, not exposed
  in the CLI — no accidental conjecture-dependent benchmark wins.
- The binary prints a round-by-round soundness analysis (RLC, OOD, prox gaps,
  query error, PoW) — exactly the artifact the Lean track needs for composition
  accounting (D15). Composition losses for OUR protocol remain to be written;
  standalone WHIR at 128 bits ≠ composed protocol at 128 bits.

## ZK path (D15 feasibility check — validated, not assumed)

Requires `--features rs_in_order` (panics without it: "ZK requires --features
rs_in_order"). Same d=16 Johnson config, ZK vs non-ZK:

| Metric | Non-ZK | ZK | Cost |
|---|---|---|---|
| Prover | 45.5 ms | 8.7 s | ~190× |
| Proof size | 125.1 KiB | 3,589.7 KiB (3.5 MiB) | ~29× |
| Verifier (1k reps) | 1.1 ms | 266.3 ms | ~240× |
| Peak RSS | 15.5 MB | 309 MB | ~20× |

Implications: library ZK is functional but far from free. At these overheads,
naive ZK-per-request breaks the batching amortization math (D16) — ZK must be
designed into the composed transcript (D15 simulator track), not bolted on at
M4 by flipping a flag. This is concrete evidence for the statements-before-code
rule: the ZK cost structure changes protocol-level choices.

## Integration assessment

- arkworks-based (ark-ff 0.6, ark-serialize), NOT Plonky3. Mixing with p3-* is
  possible at the byte/transcript boundary but ugly. Decision: treat WHIR as
  the proximity core behind a thin trait; do not force p3 interop inside the
  proof system. Our field/transcript core follows WHIR's choices (Goldilocks3,
  spongefish transcripts) if adopted.
- `src/bin/main.rs` exposes `--linear-constraints` (constrained correlated
  agreement, degree ≤ 2) — the CLARK-style functionality the D12 gate needs.
- Audit cost: academic prototype, no production review. Any adoption beyond
  the D12 prototype requires a security audit of the soundness-parameter
  computation (params/) and the transcript codecs.

## Verdict

**ADAPT** — use WHIR as the proximity core for the D12 transformer-block gate
prototype, behind a thin trait. Do not hand-roll lincheck+FRI. Production
adoption is deferred until (a) the D12 gate validates composition, (b) an audit
scope is costed. Rejected: adopting WHIR blindly as production dependency;
rejected: hand-rolling a duplicate of the technique authors' own implementation.
