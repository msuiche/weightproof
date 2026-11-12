#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::glp::{GlpQuantization, GlpSteering};

fuzz_target!(|data: &[u8]| {
    let quantization = GlpQuantization::new(12, 12).unwrap();
    let _ = GlpSteering::from_gguf(data, quantization);
});
