# Spikes — third-party checkouts (gitignored)

## flashprove/ — anthropics/FlashProve @main (cloned 2026-11-04)

Ground truth for the D11 Rust verifier work: format specs, stored accepted
proof, rejected-proof corpus, and a working local C++ oracle.

### C++ oracle on macOS (works, clang build)

`verifier/build.sh` is Linux/GNU-only. To build `bin-clang/cp_verify` here:

1. `brew install coreutils findutils` (GNU stat/find/etc.).
2. Shim GNU tools at `/tmp/fp-shim/` (already set up on this machine):
   `stat→gstat`, `find→gfind`, `sha256sum→gsha256sum`,
   `readlink→greadlink`, `realpath→grealpath`, `mv→gmv`.
   Each shim is a one-line `exec <gnu-tool> "$@"` script on PATH first.
3. Local uncommitted patch to `verifier/build.sh`: on Darwin, drop
   `-Wl,-z,relro,-z,now,-z,noexecstack` and `-fstack-clash-protection`
   (Apple clang/ld rejects them). Marked "macOS spike patch".
4. `PATH="/tmp/fp-shim:$PATH" verifier/build.sh --clang - "$PWD/bin-clang"`

Do not use the default g++ path: `-fno-ipa-modref` is a GCC-only flag and
Apple clang rejects it; `brew install gcc` was not needed once clang worked.

### Ground-truth calls (all verified on this machine)

```sh
# accept: last line YES
bin-clang/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk \
  examples/accepted-proof/example.req examples/accepted-proof/example.y \
  examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)" -q
# wrong reply token (200 instead of 199): NO: reply-differs
printf '\xc8\x00\x00\x00' > /tmp/wrong.y
bin-clang/cp_verify examples/accepted-proof/config examples/accepted-proof/ivk \
  examples/accepted-proof/example.req /tmp/wrong.y \
  examples/accepted-proof/example.proof -id "$(cat examples/accepted-proof/ID.txt)" -q
# malformed request: NO: statement-range
```

Facts pinned in `tests/fp_ground_truth.rs`: request = 66 u32 words (T=64,
r=60, e=64, checked tokens 61..63 end-exclusive), y=199, proof head
`CPPF H821 entry=0`, policy sec=128 salt=40 record=1 listed=0 answers=1,
18 layout call lines. LINK layout: `spikes/plans/leaf-openings.txt` records
all five opening lists, in draw order, from a `PART PASSED: leaf` oracle run;
the Rust planner matches all 43 fields and the 546112-byte maximum, and the
structural walker consumes the stored 536952-byte leaf exactly.

Negative corpus for later increments: `examples/rejected-proofs/`
(`altered_copies.tsv`, `every_kind_of_item.tsv`, `try_altered_copies.py`).

### Format spec map

- `blocks/frame/FORMAT.md` — file layout (descriptive).
- `docs/format/FORMAT.md` (+ 08-parameters, 09-key-and-leaf, 10-proof-file,
  11-*) — ruling spec for hashed bytes; where they differ, this wins.
- `verifier/docs/output.md` — exit codes / verdict lines.

## whir/ — worldfnd/whir @83bc417 (cloned 2026-11-04)

Proximity-core evaluation; results and verdict in `spikes/REPORT.md`.
Build: `cargo build --release [--features rs_in_order]` (feature required
for `--zk`). Benchmark matrix: `spikes/results-whir-20261104.log`.
