//! Goldilocks field F, p = 2^64 − 2^32 + 1 (FORMAT.md §1 rule 1):
//! an element of F is 8 little-endian bytes, refused unless below p.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::fp::error::FpError;

/// p = 2^64 − 2^32 + 1.
pub const GOLDILOCKS_P: u64 = 0xffff_ffff_0000_0001;

/// A canonical field element, guaranteed `< p`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldElement(pub u64);

impl FieldElement {
    /// Parse one 8-byte little-endian element, refusing non-canonical values.
    pub fn parse(bytes: &[u8]) -> Result<Self, FpError> {
        if bytes.len() != 8 {
            return Err(FpError::Truncated {
                context: "field-element",
                needed: 8,
                available: bytes.len(),
            });
        }
        let mut le = [0u8; 8];
        le.copy_from_slice(bytes);
        let value = u64::from_le_bytes(le);
        if value >= GOLDILOCKS_P {
            return Err(FpError::NonCanonicalField);
        }
        Ok(Self(value))
    }

    pub const ZERO: Self = Self(0);
    pub const ONE: Self = Self(1);

    /// Reduce an arbitrary word; every word is less than 2p.
    pub const fn from_u64(value: u64) -> Self {
        Self(if value >= GOLDILOCKS_P {
            value - GOLDILOCKS_P
        } else {
            value
        })
    }

    pub fn pow(self, mut exponent: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base;
            }
            base = base * base;
            exponent >>= 1;
        }
        result
    }

    /// Multiplicative inverse; zero maps to zero, as in the oracle.
    pub fn inverse(self) -> Self {
        self.pow(GOLDILOCKS_P - 2)
    }
}

const EPS: u64 = 0xffff_ffff;

fn reduce128(hi: u64, lo: u64) -> u64 {
    let high = hi >> 32;
    let low = hi & EPS;
    let mut t = lo.wrapping_sub(high);
    t = t.wrapping_sub(EPS & 0u64.wrapping_sub(u64::from(lo < high)));
    let m = (low << 32).wrapping_sub(low);
    let mut r = t.wrapping_add(m);
    r = r.wrapping_add(EPS & 0u64.wrapping_sub(u64::from(r < m)));
    FieldElement::from_u64(r).0
}

fn dot3(a0: u64, b0: u64, a1: u64, b1: u64, a2: u64, b2: u64) -> u64 {
    let first = (a0 as u128) * (b0 as u128);
    let mut hi = (first >> 64) as u64;
    let mut lo = first as u64;
    let mut carry = 0u64;
    for product in [(a1 as u128) * (b1 as u128), (a2 as u128) * (b2 as u128)] {
        let mut product_hi = (product >> 64) as u64;
        let product_lo = product as u64;
        lo = lo.wrapping_add(product_lo);
        product_hi += u64::from(lo < product_lo);
        let previous_hi = hi;
        hi = hi.wrapping_add(product_hi);
        carry += u64::from(hi < previous_hi);
    }
    (FieldElement(reduce128(hi, lo)) - FieldElement(carry << 32)).0
}

impl Add for FieldElement {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let not_rhs = GOLDILOCKS_P - rhs.0;
        let difference = self.0.wrapping_sub(not_rhs);
        Self(if self.0 < not_rhs {
            difference.wrapping_sub(EPS)
        } else {
            difference
        })
    }
}

impl AddAssign for FieldElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for FieldElement {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let difference = self.0.wrapping_sub(rhs.0);
        Self(if self.0 < rhs.0 {
            difference.wrapping_sub(EPS)
        } else {
            difference
        })
    }
}

impl SubAssign for FieldElement {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for FieldElement {
    type Output = Self;
    fn neg(self) -> Self {
        Self(if self.0 == 0 {
            0
        } else {
            GOLDILOCKS_P - self.0
        })
    }
}

impl Mul for FieldElement {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let wide = (self.0 as u128) * (rhs.0 as u128);
        Self(reduce128((wide >> 64) as u64, wide as u64))
    }
}

impl MulAssign for FieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

/// E = F[X]/(X^3 - 2), stored as canonical coordinates of 1, X, X^2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtensionElement(pub [u64; 3]);

impl ExtensionElement {
    pub const ZERO: Self = Self([0; 3]);
    pub const ONE: Self = Self([1, 0, 0]);

    /// Construct from coordinates already known to be canonical (chain draws
    /// and parsed field elements satisfy this invariant).
    pub const fn from_words(words: [u64; 3]) -> Self {
        Self(words)
    }

    pub const fn from_base(value: FieldElement) -> Self {
        Self([value.0, 0, 0])
    }

    pub const fn theta(index: usize) -> Self {
        match index {
            0 => Self::ONE,
            1 => Self([0, 1, 0]),
            2 => Self([0, 0, 1]),
            _ => Self::ZERO,
        }
    }

    pub fn is_zero(self) -> bool {
        self.0 == [0; 3]
    }

    pub fn pow(self, mut exponent: u64) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base;
            }
            base = base * base;
            exponent >>= 1;
        }
        result
    }

    /// Multiplicative inverse; zero maps to zero, as in the oracle.
    pub fn inverse(self) -> Self {
        let a = FieldElement(self.0[0]);
        let b = FieldElement(self.0[1]);
        let c = FieldElement(self.0[2]);
        let nr = FieldElement(2);
        let aa = a * a - nr * (b * c);
        let bb = nr * (c * c) - a * b;
        let cc = b * b - a * c;
        let norm_inverse = (a * aa + nr * (b * cc + c * bb)).inverse();
        Self([
            (aa * norm_inverse).0,
            (bb * norm_inverse).0,
            (cc * norm_inverse).0,
        ])
    }
}

impl Add for ExtensionElement {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            (FieldElement(self.0[0]) + FieldElement(rhs.0[0])).0,
            (FieldElement(self.0[1]) + FieldElement(rhs.0[1])).0,
            (FieldElement(self.0[2]) + FieldElement(rhs.0[2])).0,
        ])
    }
}

impl AddAssign for ExtensionElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for ExtensionElement {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            (FieldElement(self.0[0]) - FieldElement(rhs.0[0])).0,
            (FieldElement(self.0[1]) - FieldElement(rhs.0[1])).0,
            (FieldElement(self.0[2]) - FieldElement(rhs.0[2])).0,
        ])
    }
}

impl SubAssign for ExtensionElement {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for ExtensionElement {
    type Output = Self;
    fn neg(self) -> Self {
        Self([
            (-FieldElement(self.0[0])).0,
            (-FieldElement(self.0[1])).0,
            (-FieldElement(self.0[2])).0,
        ])
    }
}

impl Mul<FieldElement> for ExtensionElement {
    type Output = Self;
    fn mul(self, rhs: FieldElement) -> Self {
        Self([
            (FieldElement(self.0[0]) * rhs).0,
            (FieldElement(self.0[1]) * rhs).0,
            (FieldElement(self.0[2]) * rhs).0,
        ])
    }
}

impl Mul for ExtensionElement {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let [a0, a1, a2] = self.0;
        let [b0, b1, b2] = rhs.0;
        let n1 = (FieldElement(b1) + FieldElement(b1)).0;
        let n2 = (FieldElement(b2) + FieldElement(b2)).0;
        Self([
            dot3(a0, b0, a1, n2, a2, n1),
            dot3(a0, b1, a1, b0, a2, n2),
            dot3(a0, b2, a1, b1, a2, b0),
        ])
    }
}

impl MulAssign for ExtensionElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_elements() {
        assert_eq!(
            FieldElement::parse(&0u64.to_le_bytes()),
            Ok(FieldElement(0))
        );
        assert_eq!(
            FieldElement::parse(&(GOLDILOCKS_P - 1).to_le_bytes()),
            Ok(FieldElement(GOLDILOCKS_P - 1))
        );
    }

    #[test]
    fn rejects_p_and_above() {
        assert_eq!(
            FieldElement::parse(&GOLDILOCKS_P.to_le_bytes()),
            Err(FpError::NonCanonicalField)
        );
        assert_eq!(
            FieldElement::parse(&u64::MAX.to_le_bytes()),
            Err(FpError::NonCanonicalField)
        );
    }

    #[test]
    fn rejects_truncation() {
        assert!(FieldElement::parse(&[0u8; 7]).is_err());
    }

    #[test]
    fn arithmetic_matches_cpp_vectors() {
        let a = FieldElement(0x1234_5678_9abc_def0);
        let b = FieldElement(0xdead_beef_1234_5678);
        assert_eq!((a + b).0, 0xf0e2_1567_acf1_3568);
        assert_eq!((a - b).0, 0x3386_9788_8888_8879);
        assert_eq!((a * b).0, 0x72f6_03e1_3651_634e);
        assert_eq!(a.inverse().0, 0xcc82_4220_76a0_4151);

        let x = ExtensionElement([
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0x0f0e_0d0c_0b0a_0908,
        ]);
        let y = ExtensionElement([
            0xdead_beef_1234_5678,
            0x0123_4567_89ab_cdef,
            0x1111_1111_1111_1111,
        ]);
        assert_eq!(
            (x + y).0,
            [
                0xf0e2_1567_acf1_3568,
                0x0000_0000_ffff_fffe,
                0x201f_1e1d_1c1b_1a19
            ]
        );
        assert_eq!(
            (x - y).0,
            [
                0x3386_9788_8888_8879,
                0xfdb9_7530_eca8_6421,
                0xfdfc_fbf9_f9f8_f7f8
            ]
        );
        assert_eq!(
            (x * y).0,
            [
                0xd119_fec3_5217_b8e5,
                0x10e3_287d_2e45_1102,
                0x1070_14da_78d1_b2ed
            ]
        );
        assert_eq!(
            x.inverse().0,
            [
                0xe9f5_8ce2_eb75_f0c4,
                0xe465_3364_25c1_6346,
                0x58e7_bd49_4afc_9afa
            ]
        );
        assert_eq!(
            x.pow(257).0,
            [
                0x03b7_1760_a855_8691,
                0x358b_a942_4b4e_70de,
                0x1266_5356_e62c_d592
            ]
        );
    }

    #[test]
    fn arithmetic_boundaries_and_inverse_identity() {
        assert_eq!(FieldElement::ZERO + FieldElement::ONE, FieldElement::ONE);
        assert_eq!(
            FieldElement::ZERO - FieldElement::ONE,
            FieldElement(GOLDILOCKS_P - 1)
        );
        assert_eq!(
            FieldElement(GOLDILOCKS_P - 1) * FieldElement(GOLDILOCKS_P - 1),
            FieldElement::ONE
        );
        let x = ExtensionElement([3, 7, 11]);
        assert_eq!(x * x.inverse(), ExtensionElement::ONE);
        assert_eq!(ExtensionElement::ZERO.inverse(), ExtensionElement::ZERO);
    }
}
