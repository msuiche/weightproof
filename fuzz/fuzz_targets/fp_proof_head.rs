#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::ProofHead;

fuzz_target!(|data: &[u8]| {
    let _ = ProofHead::parse(data);
});
