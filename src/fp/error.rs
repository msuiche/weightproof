//! Fail-closed, named errors (D8): every rejection names its cause.

use core::fmt;

/// Parsing/validation failure. No variant is recoverable by "accepting anyway":
/// callers must treat every error as a rejection, never a warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FpError {
    /// Input ended before the last planned item (FORMAT.md §1 rule 2).
    Truncated {
        context: &'static str,
        needed: usize,
        available: usize,
    },
    /// Bytes remain after the last planned item (FORMAT.md §1 rule 2).
    TrailingBytes { context: &'static str, count: usize },
    /// Version word is not `H821` (FORMAT.md §1 rule 3: refuse every other word).
    VersionUnknown([u8; 4]),
    /// File tag does not match the expected four bytes.
    BadTag { expected: &'static str },
    /// Field element encoding is ≥ p (non-canonical, FORMAT.md §1 rule 1).
    NonCanonicalField,
    /// A length/count is structurally impossible (e.g. not a whole number of words).
    BadLength { context: &'static str },
    /// A value is outside its permitted range, with the rule that failed.
    Range { context: &'static str },
    /// A parameter-file line is malformed or violates a parser rule.
    Policy { context: &'static str },
    /// Parameter text is not valid UTF-8.
    NotUtf8,
}

impl fmt::Display for FpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { context, needed, available } => write!(
                f,
                "truncated-{context}: needs {needed} bytes, has {available}"
            ),
            Self::TrailingBytes { context, count } => {
                write!(f, "trailing-bytes-{context}: {count} extra")
            }
            Self::VersionUnknown(word) => {
                write!(f, "version-unknown: {:02x?}", word)
            }
            Self::BadTag { expected } => write!(f, "bad-tag: expected {expected}"),
            Self::NonCanonicalField => write!(f, "non-canonical-field: element >= p"),
            Self::BadLength { context } => write!(f, "bad-length-{context}"),
            Self::Range { context } => write!(f, "range-{context}"),
            Self::Policy { context } => write!(f, "policy-{context}"),
            Self::NotUtf8 => write!(f, "not-utf8"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FpError {}
