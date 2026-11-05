//! Reed–Solomon coset generator rows and symbol evaluation (`code/code.h`).

use crate::fp::{
    codes::CodeSize,
    field::{ExtensionElement as E, FieldElement as F},
    tensor::{ceil_log2, product_index, product_table},
};
use alloc::vec::Vec;

const GENERATOR: F = F(7);
const ROOT_32: F = F(0xE2AF_7063_5744_FD26);

fn pow(mut base: F, mut exponent: u64) -> F {
    let mut out = F::ONE;
    while exponent != 0 {
        if exponent & 1 != 0 {
            out *= base;
        }
        base *= base;
        exponent >>= 1;
    }
    out
}

fn point(log_n: u32, index: u64) -> F {
    let s = log_n.min(32);
    let mut out = pow(GENERATOR, 1 + (index >> s));
    let mut root = ROOT_32;
    for i in (0..32).rev() {
        if i < s && (index >> i) & 1 != 0 {
            out *= root;
        }
        root *= root;
    }
    out
}

pub fn log_ell(code: &CodeSize) -> u32 {
    ceil_log2(code.ell)
}
pub fn log_r(code: &CodeSize) -> u32 {
    ceil_log2(code.r)
}
pub fn log_dim(code: &CodeSize) -> u32 {
    ceil_log2(code.dim())
}

#[derive(Debug, Clone)]
pub struct GeneratorRow {
    pub message_scale: F,
    pub tail_scale: F,
    pub factors: Vec<F>,
}

pub fn generator_at(code: &CodeSize, position: u64) -> GeneratorRow {
    let mut factors = Vec::with_capacity(log_dim(code) as usize);
    if log_dim(code) != 0 {
        factors.push(point(code.log_n, position));
        for i in 1..log_dim(code) as usize {
            factors.push(factors[i - 1] * factors[i - 1]);
        }
    }
    GeneratorRow {
        message_scale: product_index(&factors, code.r),
        tail_scale: product_index(&factors, 0),
        factors,
    }
}

pub fn generator_row(code: &CodeSize, position: u64) -> (Vec<F>, Vec<F>) {
    let generator = generator_at(code, position);
    let mut message = product_table(
        &generator.factors[..log_ell(code) as usize],
        code.ell as usize,
    );
    let mut tail = product_table(&generator.factors[..log_r(code) as usize], code.r as usize);
    for value in &mut message {
        *value *= generator.message_scale;
    }
    for value in &mut tail {
        *value *= generator.tail_scale;
    }
    (message, tail)
}

fn product_dot(values: &[E], factors: &[F]) -> E {
    let mut out = E::ZERO;
    for (index, value) in values.iter().enumerate() {
        out += *value * product_index(factors, index as u64);
    }
    out
}

pub fn symbol_at(code: &CodeSize, position: u64, message: &[E], tail: &[E]) -> E {
    assert_eq!(message.len() as u64, code.ell);
    assert_eq!(tail.len() as u64, code.r);
    let generator = generator_at(code, position);
    product_dot(message, &generator.factors[..log_ell(code) as usize]) * generator.message_scale
        + product_dot(tail, &generator.factors[..log_r(code) as usize]) * generator.tail_scale
}

pub fn mask_position(_main: &CodeSize, mask: &CodeSize, position: u64) -> u64 {
    position & ((1u64 << mask.log_n) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_and_symbol_match_cpp() {
        let code = CodeSize {
            log_n: 4,
            ell: 3,
            r: 2,
            over_E: true,
        };
        let generator = generator_at(&code, 11);
        assert_eq!(generator.message_scale.0, 0xffff_fffe_cf00_0001);
        assert_eq!(generator.tail_scale.0, 1);
        assert_eq!(
            generator.factors.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![
                0x8fff_ffff_0000_0001,
                0xffff_fffe_cf00_0001,
                0x0961_0000_0000_0000
            ]
        );
        let message = [E([1, 2, 3]), E([4, 5, 6]), E([7, 8, 9])];
        let tail = [E([10, 11, 12]), E([13, 14, 15])];
        assert_eq!(
            symbol_at(&code, 11, &message, &tail).0,
            [
                0xe766_fff9_7940_0010,
                0x9637_fff9_32d0_0011,
                0x4508_fff8_ec60_0012
            ]
        );
    }
}
