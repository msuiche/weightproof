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

#[repr(C)]
struct DotContext {
    _private: [u8; 0],
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
    fn wp_dot_cuda_create(a: *const CExt, b: *const CExt, count: usize) -> *mut DotContext;
    fn wp_dot_cuda_coefficients(context: *mut DotContext, coefficients: *mut CExt) -> i32;
    fn wp_dot_cuda_round(context: *mut DotContext, challenge: CExt, coefficients: *mut CExt)
        -> i32;
    fn wp_dot_cuda_download(context: *mut DotContext, a: *mut CExt, b: *mut CExt) -> i32;
    fn wp_dot_cuda_destroy(context: *mut DotContext);
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

pub struct DotState {
    context: *mut DotContext,
    count: usize,
    next: [E; 3],
}

impl DotState {
    pub fn new(a: &[E], b: &[E]) -> Result<Self, FpError> {
        if a.is_empty() || a.len() != b.len() {
            return Err(FpError::BadLength {
                context: "native-cuda-dot",
            });
        }
        let a: Vec<CExt> = a.iter().map(|value| CExt { c: value.0 }).collect();
        let b: Vec<CExt> = b.iter().map(|value| CExt { c: value.0 }).collect();
        let context = unsafe { wp_dot_cuda_create(a.as_ptr(), b.as_ptr(), a.len()) };
        if context.is_null() {
            return Err(FpError::Range {
                context: "native-cuda-dot-create",
            });
        }
        Ok(Self {
            context,
            count: a.len(),
            next: [E::ZERO; 3],
        })
    }

    pub fn coefficients(&mut self) -> Result<[E; 3], FpError> {
        let mut coefficients = [CExt { c: [0; 3] }; 3];
        let status = unsafe { wp_dot_cuda_coefficients(self.context, coefficients.as_mut_ptr()) };
        if status != 0 {
            return Err(FpError::Range {
                context: "native-cuda-dot-round",
            });
        }
        Ok(coefficients.map(|value| E(value.c)))
    }

    pub fn cached(&self) -> [E; 3] {
        self.next
    }

    pub fn fold(&mut self, challenge: E) -> Result<[E; 3], FpError> {
        let mut ignored = [CExt { c: [0; 3] }; 3];
        let status = unsafe {
            wp_dot_cuda_round(self.context, CExt { c: challenge.0 }, ignored.as_mut_ptr())
        };
        if status != 0 {
            return Err(FpError::Range {
                context: "native-cuda-dot-round",
            });
        }
        self.next = ignored.map(|value| E(value.c));
        Ok(self.next)
    }

    pub fn download(&self, a: &mut [E], b: &mut [E]) -> Result<(), FpError> {
        if a.len() != self.count || b.len() != self.count {
            return Err(FpError::BadLength {
                context: "native-cuda-dot-download",
            });
        }
        let mut out_a = vec![CExt { c: [0; 3] }; self.count];
        let mut out_b = vec![CExt { c: [0; 3] }; self.count];
        let status =
            unsafe { wp_dot_cuda_download(self.context, out_a.as_mut_ptr(), out_b.as_mut_ptr()) };
        if status != 0 {
            return Err(FpError::Range {
                context: "native-cuda-dot-download",
            });
        }
        for ((out_a, out_b), (a, b)) in out_a
            .into_iter()
            .zip(out_b)
            .zip(a.iter_mut().zip(b.iter_mut()))
        {
            *a = E(out_a.c);
            *b = E(out_b.c);
        }
        Ok(())
    }
}

impl Drop for DotState {
    fn drop(&mut self) {
        unsafe { wp_dot_cuda_destroy(self.context) }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_state_zero_challenge_folds_pairs() {
        let a: Vec<E> = (0..8192u64)
            .map(|i| E::from_base(F::from_u64(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))))
            .collect();
        let b: Vec<E> = (0..8192u64)
            .map(|i| E::from_base(F::from_u64(i.wrapping_mul(0xd1b5_4a32_d192_ed03))))
            .collect();
        let mut state = DotState::new(&a, &b).unwrap();
        state.coefficients().unwrap();
        state.fold(E::ZERO).unwrap();
        let mut actual_a = a.clone();
        let mut actual_b = b.clone();
        state.download(&mut actual_a, &mut actual_b).unwrap();
        assert_eq!(actual_a[0], a[0]);
        assert_eq!(actual_b[0], b[0]);
        let expected_a: Vec<E> = a.chunks_exact(2).map(|pair| pair[0]).collect();
        let expected_b: Vec<E> = b.chunks_exact(2).map(|pair| pair[0]).collect();
        for (index, (actual, expected)) in actual_a.iter().zip(&expected_a).enumerate() {
            assert_eq!(actual, expected, "a index {index}");
        }
        for (index, (actual, expected)) in actual_b.iter().zip(&expected_b).enumerate() {
            assert_eq!(actual, expected, "b index {index}");
        }
    }
}
