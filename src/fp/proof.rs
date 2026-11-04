//! Proof-file head (FORMAT.md §4): tag, version, entry number (3 words).
//! The stored example starts `CPPF H821 0000 0000` (entry 0).
//! Chain walking comes in a later increment; this is head parsing only.

use crate::fp::{error::FpError, version::Version};

/// Tag of a proof file: `CPPF`.
pub const PROOF_TAG: [u8; 4] = *b"CPPF";

/// The 12-byte proof-file head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofHead {
    pub version: Version,
    /// Which stored copy (leaf) of the weight table this proof uses.
    pub entry: u32,
}

impl ProofHead {
    /// Parse the head, returning it and the remaining bytes of the file.
    pub fn parse(bytes: &[u8]) -> Result<(Self, &[u8]), FpError> {
        if bytes.len() < 12 {
            return Err(FpError::Truncated {
                context: "proof-head",
                needed: 12,
                available: bytes.len(),
            });
        }
        if bytes[0..4] != PROOF_TAG {
            return Err(FpError::BadTag { expected: "CPPF" });
        }
        let version = Version::parse(&bytes[4..8])?;
        let mut le = [0u8; 4];
        le.copy_from_slice(&bytes[8..12]);
        let entry = u32::from_le_bytes(le);
        Ok((Self { version, entry }, &bytes[12..]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head_bytes(tag: &[u8; 4], version: &[u8; 4], entry: u32) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(tag);
        out[4..8].copy_from_slice(version);
        out[8..12].copy_from_slice(&entry.to_le_bytes());
        out
    }

    #[test]
    fn parses_head() {
        let bytes = head_bytes(b"CPPF", b"H821", 7);
        let (head, rest) = ProofHead::parse(&bytes).unwrap();
        assert_eq!(head.entry, 7);
        assert!(rest.is_empty());
    }

    #[test]
    fn rejects_bad_tag() {
        let bytes = head_bytes(b"XXXX", b"H821", 0);
        assert!(matches!(
            ProofHead::parse(&bytes),
            Err(FpError::BadTag { .. })
        ));
    }

    #[test]
    fn rejects_bad_version() {
        let bytes = head_bytes(b"CPPF", b"H999", 0);
        assert!(matches!(
            ProofHead::parse(&bytes),
            Err(FpError::VersionUnknown(_))
        ));
    }

    #[test]
    fn rejects_truncation() {
        assert!(ProofHead::parse(&[0u8; 11]).is_err());
    }
}
