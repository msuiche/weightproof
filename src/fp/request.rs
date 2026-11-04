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
        // cp_verify usage: T = 2^lt tokens, then r and e with r < e <= T.
        if !t.is_power_of_two() {
            return Err(FpError::Range { context: "t-power-of-two" });
        }
        if r >= e {
            return Err(FpError::Range { context: "r-before-e" });
        }
        if e > t {
            return Err(FpError::Range { context: "e-within-tokens" });
        }
        Ok(Self { tokens, r, e })
    }

    /// Token count T (= 2^lt).
    pub fn t(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// The request tokens this proof covers (cp_verify usage):
    /// positions `r .. e-1`, where position p is about token p + 1, so the
    /// covered indices are `r + 1 ..= min(e, T - 1)`. May be empty in
    /// next-token-only mode (`r = T - 1`, `e = T`).
    /// For the stored example (r = 60, e = 64, T = 64): tokens 61, 62, 63.
    pub fn checked_indices(&self) -> core::ops::RangeInclusive<u32> {
        let end = self.e.min(self.t() - 1);
        (self.r + 1)..=end
    }

    /// The covered tokens themselves (empty when `r = T - 1`, `e = T`).
    pub fn checked_tokens(&self) -> &[u32] {
        let start = (self.r + 1) as usize;
        let end = self.e.min(self.t() - 1) as usize + 1;
        if start >= end {
            return &[];
        }
        &self.tokens[start..end]
    }

    /// Whether this proof also covers y, the token after the request:
    /// cp_verify checks y's argmax iff `e == T`; if `e < T`, y must be 0.
    pub fn covers_next_token(&self) -> bool {
        self.e == self.t()
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
        assert_eq!(req.checked_indices(), 2..=3);
        assert_eq!(req.checked_tokens(), &[30, 40]);
        assert!(!req.covers_next_token());
    }

    #[test]
    fn next_token_only_mode_has_empty_request_range() {
        // cp_verify -next-token: r = T - 1, e = T; the proof is about y alone.
        let req = RequestFile::parse(&req_bytes(&[10, 20, 30, 40], 3, 4)).unwrap();
        assert!(req.covers_next_token());
        assert_eq!(req.checked_tokens(), &[] as &[u32]);
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
    fn rejects_non_power_of_two_t() {
        assert!(matches!(
            RequestFile::parse(&req_bytes(&[1, 2, 3], 0, 2)),
            Err(FpError::Range { context: "t-power-of-two" })
        ));
    }

    #[test]
    fn rejects_r_after_e() {
        assert!(matches!(
            RequestFile::parse(&req_bytes(&[1, 2, 3, 4], 3, 1)),
            Err(FpError::Range { context: "r-before-e" })
        ));
        assert!(RequestFile::parse(&req_bytes(&[1, 2, 3, 4], 2, 2)).is_err());
    }

    #[test]
    fn rejects_e_beyond_tokens() {
        assert!(matches!(
            RequestFile::parse(&req_bytes(&[1, 2, 3, 4], 0, 5)),
            Err(FpError::Range { context: "e-within-tokens" })
        ));
    }

    #[test]
    fn rejects_extreme_r_without_panic() {
        // Regression: r + 1 used to overflow here (fuzz artifact
        // crash-68eeb1a2). r = u32::MAX must be a clean rejection.
        let result = RequestFile::parse(&req_bytes(&[1, 2, 3, 4], u32::MAX, u32::MAX));
        assert!(matches!(result, Err(FpError::Range { context: "r-before-e" })));
        assert!(RequestFile::parse(&req_bytes(&[1, 2, 3, 4], u32::MAX - 1, u32::MAX)).is_err());
    }

    #[test]
    fn next_token_exact_word() {
        assert_eq!(NextToken::parse(&199u32.to_le_bytes()), Ok(NextToken(199)));
        assert!(NextToken::parse(&[0u8; 3]).is_err());
        assert!(NextToken::parse(&[0u8; 5]).is_err());
    }
}
