//! The one version word (FORMAT.md §1 rule 3): four characters —
//! `H`, format number `8`, field number `2` (p = 2^64 − 2^32 + 1),
//! hash number `1` (BLAKE2b). A program serves this one word and refuses
//! every other as `version-unknown`.

use crate::fp::error::FpError;

/// `H821`: format 8, Goldilocks field, BLAKE2b, plain Reed–Solomon codes.
pub const H821: Version = Version(*b"H821");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version(pub [u8; 4]);

impl Version {
    /// Parse a 4-byte version word, refusing anything but `H821`.
    pub fn parse(bytes: &[u8]) -> Result<Self, FpError> {
        if bytes.len() != 4 {
            return Err(FpError::BadLength {
                context: "version-word",
            });
        }
        let mut word = [0u8; 4];
        word.copy_from_slice(bytes);
        if word != H821.0 {
            return Err(FpError::VersionUnknown(word));
        }
        Ok(H821)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_h821() {
        assert_eq!(Version::parse(b"H821"), Ok(H821));
    }

    #[test]
    fn rejects_other_versions() {
        assert!(matches!(
            Version::parse(b"H820"),
            Err(FpError::VersionUnknown(_))
        ));
        assert!(matches!(
            Version::parse(b"H921"),
            Err(FpError::VersionUnknown(_))
        ));
        assert!(matches!(
            Version::parse(b"H831"),
            Err(FpError::VersionUnknown(_))
        ));
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(Version::parse(b"H82").is_err());
        assert!(Version::parse(b"H8210").is_err());
        assert!(Version::parse(b"").is_err());
    }
}
