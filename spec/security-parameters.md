# Security parameters — WHIR evaluation gate (D14)

Rule: **no benchmark comparison before this document is complete.** A benchmark
winner under different security assumptions is not a valid comparison. Fuzzing
and malformed-proof tests establish implementation robustness, never soundness.

Sources: [worldfnd/whir README](https://github.com/worldfnd/whir),
[WHIR paper (eprint 2024/1586)](https://eprint.iacr.org/2024/1586),
FlashProve soundness statement (lean/THEOREM.md).

## Target

| Parameter | Target | Rationale |
|---|---|---|
| Soundness error | ≤ q/2¹²⁴ + 2⁻¹⁰¹ for q hash queries | Match FlashProve's Lean-proven statement; anything weaker needs explicit justification |
| Model | Random oracle, hash-based — **Blake3 confirmed by user (2026-11-04)** | Post-quantum, matches both candidates; Blake3 also WHIR's default |
| Regime for comparison | **Unique and Johnson — both proven per current source** | The conjectural `Capacity` regime (STIR Conj 5.6) is API-only, not CLI-exposed; any capacity-regime datapoint must be labeled conjectural |
| Extension field | Goldilocks2 / Goldilocks3 evaluated; Field192/Field256 if grinding margins require | Must be stated alongside every number; field choice changes both speed and soundness |
| PoW bits | Explicit, not default: query-phase and proximity-gap PoW recorded per config | CLI default is 20 bits PoW; the binary prints round-by-round soundness incl. achieved PoW per round — record it |

## WHIR configuration surface (verified against source @83bc417, 2026-11-04 — README is outdated)

- `-l/--security-level`: **default 128** now (README's 100 is stale). Spike uses `-l 128` (> 124 target).
- `--decoding-regime`: `Unique` | `Johnson` (**default**). Per `src/protocols/params/regime.rs`:
  - `Unique`: proven unique-decoding regime.
  - `Johnson`: **proven** list-decoding — BCSS25 Theorem 1.5 at canonical slack `η = √ρ/20`, m=10 (`O(n/η⁵)` improvement over BCIKS'20).
  - `Capacity` (STIR Conjecture 5.6): **conjectural**, exists in the params API (`DecodingRegimeParams::Capacity`) but is NOT exposed in the CLI. Conjecture-dependent configurations are therefore opt-in via code, not default — good.
- `-f/--field`: default `Goldilocks3` (README's Goldilocks2 stale); `Field192`/`Field256` available.
- `--linear-constraints N`: constrained correlated agreement for degree ≤ 2 constraints is implemented (`constrained_agreement_bits`, citing ProveKit Prover Whitepaper Appendix A) — the CLARK-style path this project needs, present in the library.
- `--zk`: **validated working** with `--features rs_in_order` (panics without the feature). Measured at d=16: prover ~190× slower, proof ~29× larger (3.5 MiB), verifier ~240× slower than non-ZK. ZK is feasible but expensive — cost structure must feed the D15 composed-transcript design, not be deferred to an M4 flag flip.
- `-p/--pow-bits` default 20, `--reps` default 1000 (verify timing), Blake3 default hash.
- Library warning stands: "academic prototype … NOT ready for production use." Adoption verdict must include audit cost.

## Composition losses (must be written down before the spike ends)

The D12 gate composes WHIR with sumcheck-based linchecks and lookup arguments.
Record: soundness loss per composition step (Fiat–Shamir transform, batching via
random linear combination, per-request extraction per D16), and whether the
composed error still meets q/2¹²⁴ + 2⁻¹⁰¹. A standalone WHIR proof at 124 bits
does not imply a composed protocol at 124 bits.

## Spike protocol

1. Fix two configurations meeting the target: (a) `UniqueDecoding`,
   (b) `ProvableList`, same field and rate; document chosen parameters.
2. Benchmark both at d = 16, 20, 24 variables: prove time, verify time, proof
   bytes, memory high-water (D9 harness).
3. Optional, labeled secondary run: `ConjectureList` — only to quantify the
   speed gap between conjectural and proven settings.
4. Write the reuse verdict: **adopt / adapt (fork+audit) / reject (hand-roll)**,
   including audit cost, arkworks-vs-Plonky3 integration cost (the library is
   arkworks-based; our field/transcript core plan is p3-based — mixing stacks
   is a real integration decision, not a detail), and ZK feasibility.
5. Verdict requires: numbers at fixed security, integration cost, audit cost.
   Benchmarks alone are not a verdict.
