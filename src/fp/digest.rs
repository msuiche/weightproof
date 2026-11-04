//! Digest type (eight 32-bit words, 32 bytes) and hex formatting.

use alloc::string::String;

/// A 32-byte digest as eight little-endian u32 words (blocks/hash/hash.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Digest(pub [u32; 8]);

impl Digest {
    pub const ZERO: Self = Self([0; 8]);

    pub fn from_le_bytes(bytes: &[u8; 32]) -> Self {
        let mut w = [0u32; 8];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            w[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Self(w)
    }

    pub fn to_le_bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (i, word) in self.0.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }
        out
    }

    /// `hex64(w)` print form used by cp_verify for h_par: each native word as
    /// eight hex digits, without the byte serialization used for commitment IDs.
    pub fn to_words_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for word in self.0 {
            s.push_str(&alloc::format!("{:08x}", word));
        }
        s
    }

    /// External print form: the digest's 32 bytes in order, two hex digits each.
    /// Words are stored little-endian, so formatting the `u32`s directly reverses
    /// every four-byte group (hash.h `digest_to_bytes`).
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for byte in self.to_le_bytes() {
            s.push_str(&alloc::format!("{:02x}", byte));
        }
        s
    }

    pub fn is_zero(&self) -> bool {
        self.0 == [0; 8]
    }
}
