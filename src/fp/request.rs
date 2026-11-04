//! The request file (`iop/req.h`, example README): T + 2 little-endian
//! 32-bit words — the tokens, then `r` and `e`. The proof is about tokens
//! `r + 1` to `e` (counted from 0), plus optionally the next token after
//! the request (`example.y`, one 32-bit word).

use alloc::vec::Vec;

use crate::fp::error::FpError;

/// A parsed `.req` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestFile {
    /// The T tokens: prompt followed by the reply being proved.
    pub tokens: Vec<u32>,
    /// First checked token is `r + 1` (0-based).
    pub r: u32,
    /// Last checked token is `e` (0-based).
    pub e: u32,
}

impl RequestFile {
    /// Parse the whole file. Refuses partial words, empty token lists,
    /// `r >= e`, and `e` beyond the token list — fail closed, named reasons.
    pub fn parse(bytes: &[u8]) -> Result<Self, FpError> {
        if bytes.len() % 4 != 0 {
            return Err(FpError::BadLength { context: "request-words" });
        }
        let word_count = bytes.len() / 4;
        if word_count < 3 {
            return Err(FpError::Truncated {
                context: "request-tokens-r-e",
                needed: 12,
                available: bytes.len(),
            });
        }
        let mut words = Vec::with_capacity(word_count);
        for chunk in bytes.chunks_exact(4) {
            let mut le = [0u8; 4];
            le.copy_from_slice(chunk);
            words.push(u32::from_le_bytes(le));
        }
        let e = words[word_count - 1];
        let r = words[word_count - 2];
        let tokens = words[..word_count - 2].to_vec();
        let t = tokens.len() as u32;
        if r >= e {
            return Err(FpError::Range { context: "r-before-e" });
        }
        // e - r is safe here (r < e); avoids r + 1 overflowing at u32::MAX
        // (found by cargo-fuzz, 2026-11-04).
        if e - r < 2 {
            return Err(FpError::Range { context: "empty-checked-range" });
        }
        if e > t {
            return Err(FpError::Range { context: "e-within-tokens" });
        }
        Ok(Self { tokens, r, e })
    }

    /// Token count T.
    pub fn t(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// The 0-based indices this proof checks: `r + 1 .. e` (end exclusive).
    /// For the stored example (r = 60, e = 64) this is tokens 61, 62, 63 —
    /// matching the oracle's `tokens=61..63`.
    pub fn checked_indices(&self) -> core::ops::Range<u32> {
        (self.r + 1)..self.e
    }

    /// The checked tokens themselves.
    pub fn checked_tokens(&self) -> &[u32] {
        let start = (self.r + 1) as usize;
        let end = self.e as usize;
        &self.tokens[start..end]
    }
}

/// The token after the request (`example.y`): exactly one 32-bit word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NextToken(pub u32);

impl NextToken {
    pub fn parse(bytes: &[u8]) -> Result<Self, FpError> {
        if bytes.len() < 4 {
            return Err(FpError::Truncated {
                context: "next-token",
                needed: 4,
                available: bytes.len(),
            });
        }
        if bytes.len() > 4 {
            return Err(FpError::TrailingBytes {
                context: "next-token",
                count: bytes.len() - 4,
            });
        }
        let mut le = [0u8; 4];
        le.copy_from_slice(bytes);
        Ok(Self(u32::from_le_bytes(le)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req_bytes(tokens: &[u32], r: u32, e: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for w in tokens.iter().chain([r, e].iter()) {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_and_computes_checked_range() {
        let req = RequestFile::parse(&req_bytes(&[10, 20, 30, 40], 1, 3)).unwrap();
        assert_eq!(req.t(), 4);
        assert_eq!(req.checked_indices(), 2..3);
        assert_eq!(req.checked_tokens(), &[30]);
    }

    #[test]
    fn rejects_partial_word() {
        assert!(matches!(
            RequestFile::parse(&[0u8; 15]),
            Err(FpError::BadLength { .. })
        ));
    }

    #[test]
    fn rejects_empty_and_short() {
        assert!(RequestFile::parse(&[]).is_err());
        assert!(RequestFile::parse(&req_bytes(&[], 0, 0)).is_err());
    }

    #[test]
    fn rejects_r_after_e() {
        assert!(matches!(
            RequestFile::parse(&req_bytes(&[1, 2, 3], 3, 1)),
            Err(FpError::Range { context: "r-before-e" })
        ));
        assert!(RequestFile::parse(&req_bytes(&[1, 2, 3], 2, 2)).is_err());
        assert!(RequestFile::parse(&req_bytes(&[1, 2, 3], 0, 1)).is_err(), "empty checked range must fail");
    }

    #[test]
    fn rejects_e_beyond_tokens() {
        assert!(matches!(
            RequestFile::parse(&req_bytes(&[1, 2, 3], 0, 4)),
            Err(FpError::Range { context: "e-within-tokens" })
        ));
    }

    #[test]
    fn rejects_extreme_r_without_panic() {
        // Regression: r + 1 used to overflow here (fuzz artifact
        // crash-68eeb1a2). r = u32::MAX must be a clean rejection.
        let result = RequestFile::parse(&req_bytes(&[1, 2, 3], u32::MAX, u32::MAX));
        assert!(matches!(result, Err(FpError::Range { context: "r-before-e" })));
        assert!(RequestFile::parse(&req_bytes(&[1, 2, 3], u32::MAX - 1, u32::MAX)).is_err());
    }

    #[test]
    fn next_token_exact_word() {
        assert_eq!(NextToken::parse(&199u32.to_le_bytes()), Ok(NextToken(199)));
        assert!(NextToken::parse(&[0u8; 3]).is_err());
        assert!(NextToken::parse(&[0u8; 5]).is_err());
    }
}
