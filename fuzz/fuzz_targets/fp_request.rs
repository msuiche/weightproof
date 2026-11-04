#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::{NextToken, RequestFile};

fuzz_target!(|data: &[u8]| {
    // D8: no panics on adversarial input; errors are values, never crashes.
    let _ = RequestFile::parse(data);
    let _ = NextToken::parse(data);
});
