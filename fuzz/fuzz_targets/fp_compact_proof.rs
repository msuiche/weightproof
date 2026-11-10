#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::integer_model::*;

fuzz_target!(|data: &[u8]| {
    let model = small_model();
    let tokens = [0, 1];
    let mut proof = prove_block_forest(&model, &tokens, 4, 4, 1).unwrap();
    if let Some(&index) = data.first() {
        let index = index as usize % proof.arithmetic.len();
        if let Some(&byte) = data.get(1) {
            let bytes = &mut proof.arithmetic[index].bytes;
            if !bytes.is_empty() {
                let at = data.get(2).copied().unwrap_or(0) as usize % bytes.len();
                bytes[at] ^= byte;
            }
        }
        if let Some(&offset) = data.get(3) {
            if !proof.arithmetic[index].compact_offsets.is_empty() {
                proof.arithmetic[index].compact_offsets[0] = offset as usize;
            }
        }
        if let Some(&version) = data.get(4) {
            proof.statement.format_version = version as u32;
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_block_forest(&model, &tokens, &proof)
    }));
    assert!(result.is_ok(), "verifier panicked on malformed compact proof");
});
