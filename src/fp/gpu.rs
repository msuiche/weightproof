//! ICICLE-backed prover kernels. The transcript and statement remain owned by
//! weightproof; ICICLE is used only for field arithmetic on the active device.

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
};
#[cfg(not(feature = "native-cuda"))]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "native-cuda"))]
use icicle_core::{
    bignum::BigNum,
    program::ProgramImpl,
    vec_ops::{add_scalars, scalar_mul, slice, sub_scalars, VecOpsConfig},
};
#[cfg(not(feature = "native-cuda"))]
use icicle_goldilocks::{field::ScalarField, program::goldilocks::Program};
#[cfg(not(feature = "native-cuda"))]
use icicle_runtime::memory::HostSlice;
use icicle_runtime::Device;

/// Measured GB10 crossover: native CUDA wins at 8,192 elements and loses at
/// 4,096. Smaller folds avoid launch, synchronization, and transfer overhead.
#[cfg(feature = "native-cuda")]
pub(crate) const NATIVE_CUDA_MIN: usize = 8192;

fn backend_error(_error: icicle_runtime::IcicleError) -> FpError {
    FpError::Range {
        context: "icicle-backend",
    }
}

#[cfg(not(feature = "native-cuda"))]
fn ensure_active_device() -> Result<(), FpError> {
    if icicle_runtime::get_active_device().is_ok() {
        Ok(())
    } else {
        initialize_device("CPU")
    }
}

/// Load the registered ICICLE backends and select a device for this thread.
/// Call once per proving thread; common device types are `CPU`, `METAL`, and
/// `CUDA`.
pub fn initialize_device(device_type: &str) -> Result<(), FpError> {
    #[cfg(feature = "native-cuda")]
    if device_type == "CUDA" {
        return Ok(());
    }
    let device = Device::new(device_type, 0);
    if !icicle_runtime::is_device_available(&device) {
        icicle_runtime::runtime::load_backend_from_env_or_default().map_err(backend_error)?;
    }
    if !icicle_runtime::is_device_available(&device) {
        return Err(FpError::Range {
            context: "icicle-device-unavailable",
        });
    }
    icicle_runtime::set_device(&device).map_err(backend_error)
}

#[cfg(not(feature = "native-cuda"))]
fn to_icicle(value: F) -> ScalarField {
    ScalarField::from_bytes_le(&value.0.to_le_bytes())
}

#[cfg(not(feature = "native-cuda"))]
fn from_icicle(value: ScalarField) -> F {
    let bytes = value.to_bytes_le();
    let mut word = [0u8; 8];
    word.copy_from_slice(&bytes[..8]);
    F::from_u64(u64::from_le_bytes(word))
}

#[cfg(not(feature = "native-cuda"))]
std::thread_local! {
    static EXTENSION_FOLD_PROGRAM: std::cell::RefCell<Option<Program>> =
        const { std::cell::RefCell::new(None) };
}

/// Fold one base-field multilinear sumcheck round:
/// `out[i] = a[2i] + r * (a[2i+1] - a[2i])`.
///
/// The operation runs on the active ICICLE device. Input and output use
/// canonical Goldilocks representatives, matching `field::FieldElement`.
pub fn sumcheck_fold(values: &[F], challenge: F, output: &mut [F]) -> Result<(), FpError> {
    #[cfg(feature = "native-cuda")]
    {
        if values.len() < NATIVE_CUDA_MIN {
            if values.is_empty() || values.len() % 2 != 0 || output.len() != values.len() / 2 {
                return Err(FpError::BadLength {
                    context: "native-cuda-base",
                });
            }
            for (out, pair) in output.iter_mut().zip(values.chunks_exact(2)) {
                *out = pair[0] + challenge * (pair[1] - pair[0]);
            }
            return Ok(());
        }
        return crate::fp::native_cuda::fold_base(values, challenge, output);
    }
    #[cfg(not(feature = "native-cuda"))]
    {
        if values.is_empty() || values.len() % 2 != 0 {
            return Err(FpError::BadLength {
                context: "icicle-sumcheck-input",
            });
        }
        if output.len() != values.len() / 2 {
            return Err(FpError::BadLength {
                context: "icicle-sumcheck-output",
            });
        }

        ensure_active_device()?;
        let input: Vec<ScalarField> = values.iter().copied().map(to_icicle).collect();
        let half = output.len();
        let mut even = vec![ScalarField::zero(); half];
        let mut odd = vec![ScalarField::zero(); half];
        let mut delta = vec![ScalarField::zero(); half];
        let mut scaled = vec![ScalarField::zero(); half];
        let mut folded = vec![ScalarField::zero(); half];
        let challenge = vec![to_icicle(challenge); half];
        let config = VecOpsConfig::default();

        slice(
            HostSlice::from_slice(&input),
            0,
            2,
            values.len() as u64,
            half as u64,
            &config,
            HostSlice::from_mut_slice(&mut even),
        )
        .map_err(backend_error)?;
        slice(
            HostSlice::from_slice(&input),
            1,
            2,
            values.len() as u64,
            half as u64,
            &config,
            HostSlice::from_mut_slice(&mut odd),
        )
        .map_err(backend_error)?;
        sub_scalars(
            HostSlice::from_slice(&odd),
            HostSlice::from_slice(&even),
            HostSlice::from_mut_slice(&mut delta),
            &config,
        )
        .map_err(backend_error)?;
        scalar_mul(
            HostSlice::from_slice(&delta),
            HostSlice::from_slice(&challenge),
            HostSlice::from_mut_slice(&mut scaled),
            &config,
        )
        .map_err(backend_error)?;
        add_scalars(
            HostSlice::from_slice(&even),
            HostSlice::from_slice(&scaled),
            HostSlice::from_mut_slice(&mut folded),
            &config,
        )
        .map_err(backend_error)?;

        for (out, value) in output.iter_mut().zip(folded) {
            *out = from_icicle(value);
        }
        Ok(())
    }
}

/// Fold one cubic-extension sumcheck round over `E = F[X]/(X^3 - 2)`.
pub fn sumcheck_fold_extension(
    values: &[E],
    challenge: E,
    output: &mut [E],
) -> Result<(), FpError> {
    fold_extension_strided(values, challenge, 1, output)
}

/// Fold one dense tensor variable whose contiguous half-size is `low_half`.
/// This is the layout used by `Tensor::folded_dense` in the block prover.
pub fn dense_fold_extension(
    values: &[E],
    challenge: E,
    low_half: usize,
    output: &mut [E],
) -> Result<(), FpError> {
    fold_extension_strided(values, challenge, low_half, output)
}

/// ICICLE supplies base-field vector operations, not a different
/// extension-field protocol.
fn fold_extension_strided(
    values: &[E],
    challenge: E,
    low_half: usize,
    output: &mut [E],
) -> Result<(), FpError> {
    #[cfg(feature = "native-cuda")]
    {
        if values.len() < NATIVE_CUDA_MIN {
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
            for block in 0..values.len() / (2 * low_half) {
                for offset in 0..low_half {
                    let zero = values[offset + 2 * low_half * block];
                    let one = values[offset + low_half + 2 * low_half * block];
                    output[offset + low_half * block] = zero + challenge * (one - zero);
                }
            }
            return Ok(());
        }
        return crate::fp::native_cuda::fold_extension(values, challenge, low_half, output);
    }
    #[cfg(not(feature = "native-cuda"))]
    {
        if values.is_empty()
            || values.len() % 2 != 0
            || low_half == 0
            || !low_half.is_power_of_two()
            || low_half > values.len() / 2
            || values.len() % (2 * low_half) != 0
        {
            return Err(FpError::BadLength {
                context: "icicle-extension-input",
            });
        }
        if output.len() != values.len() / 2 {
            return Err(FpError::BadLength {
                context: "icicle-extension-output",
            });
        }

        ensure_active_device()?;
        let half = output.len();
        let mut even = vec![ScalarField::zero(); 3 * half];
        let mut odd = vec![ScalarField::zero(); 3 * half];
        for block in 0..values.len() / (2 * low_half) {
            for offset in 0..low_half {
                let zero = values[offset + 2 * low_half * block];
                let one = values[offset + low_half + 2 * low_half * block];
                let out = offset + low_half * block;
                for coordinate in 0..3 {
                    even[coordinate * half + out] = to_icicle(F(zero.0[coordinate]));
                    odd[coordinate * half + out] = to_icicle(F(one.0[coordinate]));
                }
            }
        }
        let mut challenges = vec![ScalarField::zero(); 3 * half];
        for coordinate in 0..3 {
            let challenge = to_icicle(F(challenge.0[coordinate]));
            challenges[coordinate * half..(coordinate + 1) * half].fill(challenge);
        }
        let mut folded = vec![ScalarField::zero(); 3 * half];
        let config = VecOpsConfig::default();

        let inputs = [
            HostSlice::from_slice(&even[..half]),
            HostSlice::from_slice(&even[half..2 * half]),
            HostSlice::from_slice(&even[2 * half..]),
            HostSlice::from_slice(&odd[..half]),
            HostSlice::from_slice(&odd[half..2 * half]),
            HostSlice::from_slice(&odd[2 * half..]),
            HostSlice::from_slice(&challenges[..half]),
            HostSlice::from_slice(&challenges[half..2 * half]),
            HostSlice::from_slice(&challenges[2 * half..]),
        ];
        let (folded0, rest) = folded.split_at_mut(half);
        let (folded1, folded2) = rest.split_at_mut(half);
        let outputs = [
            HostSlice::from_mut_slice(folded0),
            HostSlice::from_mut_slice(folded1),
            HostSlice::from_mut_slice(folded2),
        ];
        let mut data: Vec<&HostSlice<ScalarField>> = inputs
            .iter()
            .copied()
            .chain(outputs.iter().map(|output| &**output))
            .collect();
        EXTENSION_FOLD_PROGRAM.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                *slot = Some(
                    Program::new(
                        |vars| {
                            let d0 = vars[3] - vars[0];
                            let d1 = vars[4] - vars[1];
                            let d2 = vars[5] - vars[2];
                            let two = ScalarField::from(2);
                            vars[9] = vars[0] + d0 * vars[6] + two * (d1 * vars[8] + d2 * vars[7]);
                            vars[10] = vars[1] + d0 * vars[7] + d1 * vars[6] + two * (d2 * vars[8]);
                            vars[11] = vars[2] + d0 * vars[8] + d1 * vars[7] + d2 * vars[6];
                        },
                        12,
                    )
                    .map_err(backend_error)?,
                );
            }
            slot.as_ref()
                .unwrap()
                .execute_program(&mut data, &config)
                .map_err(backend_error)
        })?;

        for (index, out) in output.iter_mut().enumerate() {
            *out = E([
                from_icicle(folded[index]).0,
                from_icicle(folded[half + index]).0,
                from_icicle(folded[2 * half + index]).0,
            ]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_fold_on_device(device_type: &str) {
        let values: Vec<F> = (0..8192u64)
            .map(|i| F::from_u64(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)))
            .collect();
        let challenge = F::from_u64(0xfedc_ba98_7654_3210);
        let expected: Vec<F> = values
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
        assert!(sumcheck_fold(&values[..8191], challenge, &mut [F::ZERO; 4096]).is_err());
        assert!(sumcheck_fold(&values, challenge, &mut [F::ZERO; 4095]).is_err());

        initialize_device(device_type).unwrap();
        let mut actual = vec![F::ZERO; expected.len()];
        sumcheck_fold(&values, challenge, &mut actual).unwrap();
        assert_eq!(actual, expected);

        let extension_values: Vec<E> = (0..8192u64)
            .map(|i| {
                E([
                    F::from_u64(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).0,
                    F::from_u64(i.wrapping_mul(0xd1b5_4a32_d192_ed03)).0,
                    F::from_u64(i.wrapping_mul(0x94d0_49bb_1331_11eb)).0,
                ])
            })
            .collect();
        let extension_challenge = E([7, 11, 13]);
        let extension_expected: Vec<E> = extension_values
            .chunks_exact(2)
            .map(|pair| pair[0] + extension_challenge * (pair[1] - pair[0]))
            .collect();
        let mut extension_actual = vec![E::ZERO; extension_expected.len()];
        assert!(sumcheck_fold_extension(
            &extension_values[..8191],
            extension_challenge,
            &mut extension_actual,
        )
        .is_err());
        sumcheck_fold_extension(
            &extension_values,
            extension_challenge,
            &mut extension_actual,
        )
        .unwrap();
        assert_eq!(extension_actual, extension_expected);

        for low_half in [1, 2, 4, 8, 16, 32, 64, 128] {
            let mut expected = vec![E::ZERO; extension_values.len() / 2];
            for block in 0..extension_values.len() / (2 * low_half) {
                for offset in 0..low_half {
                    let zero = extension_values[offset + 2 * low_half * block];
                    let one = extension_values[offset + low_half + 2 * low_half * block];
                    expected[offset + low_half * block] = zero + extension_challenge * (one - zero);
                }
            }
            let mut actual = vec![E::ZERO; expected.len()];
            dense_fold_extension(
                &extension_values,
                extension_challenge,
                low_half,
                &mut actual,
            )
            .unwrap();
            assert_eq!(actual, expected, "low_half={low_half}");
        }
        assert!(dense_fold_extension(
            &extension_values,
            extension_challenge,
            0,
            &mut extension_actual,
        )
        .is_err());
        assert!(dense_fold_extension(
            &extension_values,
            extension_challenge,
            3,
            &mut extension_actual,
        )
        .is_err());

        #[cfg(feature = "native-cuda")]
        {
            crate::fp::native_cuda::set_mapped(true);
            dense_fold_extension(
                &extension_values,
                extension_challenge,
                1,
                &mut extension_actual,
            )
            .unwrap();
            assert_eq!(extension_actual, extension_expected);
            crate::fp::native_cuda::set_mapped(false);
        }
    }

    #[test]
    fn sumcheck_fold_matches_rust_on_cpu() {
        assert_fold_on_device("CPU");
    }

    /// Requires an installed ICICLE CUDA backend and a CUDA device.
    #[test]
    #[cfg_attr(not(feature = "native-cuda"), ignore)]
    fn sumcheck_fold_matches_rust_on_cuda() {
        assert_fold_on_device("CUDA");
    }
}
