//! Goldilocks field F, p = 2^64 − 2^32 + 1 (FORMAT.md §1 rule 1):
//! an element of F is 8 little-endian bytes, refused unless below p.

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_elements() {
        assert_eq!(FieldElement::parse(&0u64.to_le_bytes()), Ok(FieldElement(0)));
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
}
