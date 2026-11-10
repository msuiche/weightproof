//! BLAKE2b with FlashProve's parameter block (blake2b_f.h), implemented
//! directly: digest length in byte 0, fanout = depth = 1, and a 32-byte
//! Start (16-byte salt + 16-byte personalization). Every hashed string
//! begins with the 16-byte label: tag word (domain | tree << 16), then index.
//! Own implementation (no_std) so Start semantics match exactly.

use crate::fp::digest::Digest;
use alloc::vec::Vec;

/// Start: salt and personalization as eight u32 words; zeros when absent.
pub type Start = [u32; 8];

/// Domain numbers (blocks/hash/domains.h).
pub const CHAIN_START: u64 = 160;
pub const CHAIN_QUERY: u64 = 162;
pub const PARAMETERS: u64 = 169;
pub const RUN: u64 = 172;
pub const PUBLISHED_ID: u64 = 179;

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

fn param_word0(digest_bytes: u32) -> u64 {
    (digest_bytes as u64) | (1 << 16) | (1 << 24)
}

fn begin(start: &Start, digest_bytes: u32) -> [u64; 8] {
    let mut h = IV;
    h[0] ^= param_word0(digest_bytes);
    h[4] ^= (start[0] as u64) | ((start[1] as u64) << 32);
    h[5] ^= (start[2] as u64) | ((start[3] as u64) << 32);
    h[6] ^= (start[4] as u64) | ((start[5] as u64) << 32);
    h[7] ^= (start[6] as u64) | ((start[7] as u64) << 32);
    h
}

#[inline]
fn g(v: &mut [u64; 16], m: &[u64; 16], r: usize, i: usize, a: usize, b: usize, c: usize, d: usize) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(m[SIGMA[r][2 * i]]);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(m[SIGMA[r][2 * i + 1]]);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

#[inline]
fn round(v: &mut [u64; 16], m: &[u64; 16], r: usize) {
    g(v, m, r, 0, 0, 4, 8, 12);
    g(v, m, r, 1, 1, 5, 9, 13);
    g(v, m, r, 2, 2, 6, 10, 14);
    g(v, m, r, 3, 3, 7, 11, 15);
    g(v, m, r, 4, 0, 5, 10, 15);
    g(v, m, r, 5, 1, 6, 11, 12);
    g(v, m, r, 6, 2, 7, 8, 13);
    g(v, m, r, 7, 3, 4, 9, 14);
}

/// h := F(h, m, t_low, t_high, last), RFC 7693 section 3.2.
fn compress(h: &mut [u64; 8], m: &[u64; 16], t_low: u64, t_high: u64, last: bool) {
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t_low;
    v[13] ^= t_high;
    v[14] ^= if last { u64::MAX } else { 0 };
    for r in 0..12 {
        round(&mut v, m, r);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Digest a public table as little-endian `u32` words. This is unkeyed
/// BLAKE2b-512 without a label prefix; every compression call uses
/// `t_high = 72 + 2^32 * word_count`. The first 32 output bytes are returned.
pub fn public_table_digest(words: &[u32]) -> Digest {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let mut h = begin(&[0; 8], 64);
    let high = 72u64 + ((words.len() as u64) << 32);
    let mut offset = 0usize;
    let mut low = 0u64;
    while bytes.len() - offset > 128 {
        let mut message = [0u64; 16];
        for (i, chunk) in bytes[offset..offset + 128].chunks_exact(8).enumerate() {
            message[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        low += 128;
        compress(&mut h, &message, low, high, false);
        offset += 128;
    }
    let mut final_block = [0u8; 128];
    let remaining = bytes.len() - offset;
    final_block[..remaining].copy_from_slice(&bytes[offset..]);
    let mut message = [0u64; 16];
    for (i, chunk) in final_block.chunks_exact(8).enumerate() {
        message[i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    low += remaining as u64;
    compress(&mut h, &message, low, high, true);
    let mut output = [0u8; 32];
    for (i, word) in h[..4].iter().enumerate() {
        output[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    Digest::from_le_bytes(&output)
}

/// One BLAKE2b computation with the label prefix and Start.
/// `outlen` is 32 for ordinary hashes, 64 for chain queries.
pub struct Hasher {
    h: [u64; 8],
    buf: [u8; 128],
    buflen: usize,
    t: u64,
}

impl Hasher {
    pub fn new(domain_word: u64, index: u64, start: &Start, outlen: usize) -> Self {
        debug_assert!(outlen == 32 || outlen == 64);
        let mut h = Self {
            h: begin(start, outlen as u32),
            buf: [0; 128],
            buflen: 0,
            t: 0,
        };
        h.push(&domain_word.to_le_bytes());
        h.push(&index.to_le_bytes());
        h
    }

    pub fn push(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let fill = 128 - self.buflen;
            if bytes.len() > fill {
                self.buf[self.buflen..].copy_from_slice(&bytes[..fill]);
                self.t += 128;
                let block = self.buf;
                let mut words = [0u64; 16];
                for (i, chunk) in block.chunks_exact(8).enumerate() {
                    let mut le = [0u8; 8];
                    le.copy_from_slice(chunk);
                    words[i] = u64::from_le_bytes(le);
                }
                compress(&mut self.h, &words, self.t, 0, false);
                self.buf = [0; 128];
                self.buflen = 0;
                bytes = &bytes[fill..];
            } else {
                self.buf[self.buflen..self.buflen + bytes.len()].copy_from_slice(bytes);
                self.buflen += bytes.len();
                break;
            }
        }
    }

    pub fn push_u32(&mut self, x: u32) {
        self.push(&x.to_le_bytes());
    }

    pub fn push_u64(&mut self, x: u64) {
        self.push(&x.to_le_bytes());
    }

    pub fn push_digest(&mut self, d: &Digest) {
        for word in d.0 {
            self.push_u32(word);
        }
    }

    /// An element of E: three field elements, each 8 little-endian bytes.
    pub fn push_e(&mut self, e: &[u64; 3]) {
        for c in e {
            self.push_u64(*c);
        }
    }

    fn final_words(&self) -> [u64; 8] {
        let mut h = self.h;
        let mut block = self.buf;
        for byte in block.iter_mut().skip(self.buflen) {
            *byte = 0;
        }
        let mut words = [0u64; 16];
        for (i, chunk) in block.chunks_exact(8).enumerate() {
            let mut le = [0u8; 8];
            le.copy_from_slice(chunk);
            words[i] = u64::from_le_bytes(le);
        }
        compress(&mut h, &words, self.t + self.buflen as u64, 0, true);
        h
    }

    /// Finish and take the first 32 bytes as a Digest.
    pub fn finish(self) -> Digest {
        let h = self.final_words();
        let mut bytes = [0u8; 32];
        for (i, w) in h[..4].iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        Digest::from_le_bytes(&bytes)
    }

    /// Finish and take all 64 bytes as eight u64 words (a ChainCall).
    pub fn finish_words(self) -> [u64; 8] {
        self.final_words()
    }

    /// Debug helper: print the buffer and its length (std only).
    #[cfg(feature = "std")]
    pub fn dump_state(&self) {
        let used = self.buflen;
        println!("buflen: {}", used);
        let mut hex = String::new();
        for b in &self.buf[..used.min(32)] {
            hex.push_str(&format!("{:02x}", b));
        }
        println!("first-bytes: {}", hex);
    }
}
