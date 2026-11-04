#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::Policy;

fuzz_target!(|data: &[u8]| {
    let _ = Policy::parse(data);
});
