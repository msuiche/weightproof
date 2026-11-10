//! Native NVCC backend for the fixed, Lean-specified fold kernel.

#![allow(unsafe_code)]

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
};
use alloc::{vec, vec::Vec};

#[repr(C)]
#[derive(Clone, Copy)]
struct CExt {
    c: [u64; 3],
}

unsafe extern "C" {
    fn wp_fold_cuda_set_mapped(mapped: bool);
    fn wp_fold_cuda(
        input: *const CExt,
        input_count: usize,
        output: *mut CExt,
        output_count: usize,
        low_half: usize,
        challenge: CExt,
    ) -> i32;
}

pub fn set_mapped(mapped: bool) {
    unsafe { wp_fold_cuda_set_mapped(mapped) }
}

fn call(
    input: &[CExt],
    output: &mut [CExt],
    low_half: usize,
    challenge: CExt,
) -> Result<(), FpError> {
    let status = unsafe {
        wp_fold_cuda(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr(),
            output.len(),
            low_half,
            challenge,
        )
    };
    if status == 0 {
        Ok(())
    } else {
        Err(FpError::Range {
            context: "native-cuda-fold",
        })
    }
}

pub fn fold_base(values: &[F], challenge: F, output: &mut [F]) -> Result<(), FpError> {
    if values.is_empty() || values.len() % 2 != 0 || output.len() != values.len() / 2 {
        return Err(FpError::BadLength {
            context: "native-cuda-base",
        });
    }
    let input: Vec<CExt> = values
        .iter()
        .map(|value| CExt { c: [value.0, 0, 0] })
        .collect();
    let mut folded = vec![CExt { c: [0; 3] }; output.len()];
    call(
        &input,
        &mut folded,
        1,
        CExt {
            c: [challenge.0, 0, 0],
        },
    )?;
    for (out, value) in output.iter_mut().zip(folded) {
        *out = F(value.c[0]);
    }
    Ok(())
}

pub fn fold_extension(
    values: &[E],
    challenge: E,
    low_half: usize,
    output: &mut [E],
) -> Result<(), FpError> {
    if values.is_empty()
        || values.len() % 2 != 0
        || output.len() != values.len() / 2
        || low_half == 0
        || !low_half.is_power_of_two()
        || low_half > output.len()
        || values.len() % (2 * low_half) != 0
    {
        return Err(FpError::BadLength {
            context: "native-cuda-extension",
        });
    }
    let input: Vec<CExt> = values.iter().map(|value| CExt { c: value.0 }).collect();
    let mut folded = vec![CExt { c: [0; 3] }; output.len()];
    call(&input, &mut folded, low_half, CExt { c: challenge.0 })?;
    for (out, value) in output.iter_mut().zip(folded) {
        *out = E(value.c);
    }
    Ok(())
}
