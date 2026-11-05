//! Fiat–Shamir chain (blocks/chain/chain.h): first query, round queries,
//! and verifier-message draws with rejection sampling.

use alloc::vec::Vec;

use crate::fp::{
    blake2::{Hasher, CHAIN_QUERY, CHAIN_START},
    digest::Digest,
    field::GOLDILOCKS_P,
};

/// A round's verifier-message kind (values are part of the query).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainKind {
    Algebraic = 1,
    Epsilon = 2,
    Seed = 3,
    Positions = 4,
    Bits = 5,
    Seal = 6,
    Empty = 7,
}

/// What is fixed before the prover speaks (chain.h `FirstQuery`).
#[derive(Debug, Clone)]
pub struct FirstQuery {
    pub label: u64,
    pub parameters: Digest,
    pub statement: Vec<u8>,
    pub record: Vec<u8>,
    pub salted: bool,
    /// The run's value (first_query_run), only for salted chains.
    pub run: Option<Digest>,
}

/// The label of the record-seed chain: 2^32.
pub const LABEL_RECORD: u64 = 1 << 32;

/// A live chain: current value rho, the open query, and the draw stream.
#[derive(Debug, Clone)]
pub struct Chain {
    rho: Digest,
    round: u64,
    label: u64,
    query_start: Digest,
    query_round: u64,
    kind: u64,
    message: Vec<u64>,
    call_index: u64,
    nonce: u64,
    next: usize,
    words: [u64; 8],
}

impl Chain {
    /// rho = H(CHAIN_START, label; parameters, statement, record, salted[, run]).
    pub fn new(q: FirstQuery) -> Self {
        let mut h = Hasher::new(CHAIN_START, q.label, &[0; 8], 32);
        h.push_digest(&q.parameters);
        h.push_u64(q.statement.len() as u64);
        h.push(&q.statement);
        h.push_u64(q.record.len() as u64);
        h.push(&q.record);
        h.push_u64(q.salted as u64);
        if let Some(run) = q.run {
            h.push_digest(&run);
        }
        let rho = h.finish();
        Self {
            rho,
            round: 1,
            label: q.label,
            query_start: Digest::ZERO,
            query_round: 0,
            kind: 0,
            message: Vec::new(),
            call_index: 1,
            nonce: 0,
            next: 8,
            words: [0; 8],
        }
    }

    /// Current chain value rho (for comparing copies).
    pub fn value(&self) -> Digest {
        self.rho
    }

    /// Chain call j of the open query: label, message, kind | j << 16, nonce 0.
    fn call(&self, j: u64) -> [u64; 8] {
        let mut h = Hasher::new(
            CHAIN_QUERY | (self.label << 16),
            self.query_round,
            &self.query_start.0,
            64,
        );
        for w in &self.message {
            h.push_u64(*w);
        }
        h.push_u64(self.kind | (j << 16));
        h.push_u64(self.nonce);
        h.finish_words()
    }

    /// Close a round without proof of work.
    pub fn close(&mut self, kind: ChainKind, message: &[u64]) {
        let accepted = self.close_with_work(kind, message, 0, 0);
        debug_assert!(accepted);
    }

    /// Close a round with proof of work. Returns false without changing the
    /// chain if the answer's first word has fewer than `work_bits` zero bits.
    pub fn close_with_work(
        &mut self,
        kind: ChainKind,
        message: &[u64],
        work_bits: u32,
        nonce: u64,
    ) -> bool {
        let first = {
            let mut h = Hasher::new(
                CHAIN_QUERY | (self.label << 16),
                self.round,
                &self.rho.0,
                64,
            );
            for w in message {
                h.push_u64(*w);
            }
            h.push_u64(kind as u64 | ((work_bits as u64) << 8));
            h.push_u64(nonce);
            h.finish_words()
        };
        if first[0].trailing_zeros() < work_bits {
            return false;
        }
        self.query_start = self.rho;
        self.query_round = self.round;
        let mut first32 = [0u8; 32];
        let rho_words = if work_bits == 0 {
            &first[..4]
        } else {
            &first[4..]
        };
        for (i, w) in rho_words.iter().enumerate() {
            first32[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
        self.rho = Digest::from_le_bytes(&first32);
        self.kind = kind as u64 | ((work_bits as u64) << 8);
        self.message = message.to_vec();
        self.round += 1;
        self.call_index = 1;
        self.nonce = nonce;
        self.words = first;
        self.next = if work_bits == 0 { 4 } else { 8 };
        true
    }

    /// The next word of the verifier's message: call 0's second half, then
    /// all words of calls 1, 2, …
    pub fn next_word(&mut self) -> u64 {
        if self.next == 8 {
            self.words = self.call(self.call_index);
            self.call_index += 1;
            self.next = 0;
        }
        let w = self.words[self.next];
        self.next += 1;
        w
    }

    /// Element of F: the next word accepted by rejection sampling (< p),
    /// zero after DRAW_TRIES = 6 rejected words.
    pub fn draw_f(&mut self) -> u64 {
        for _ in 0..6 {
            let w = self.next_word();
            if w < GOLDILOCKS_P {
                return w;
            }
        }
        0
    }

    /// Element of E: three elements of F.
    pub fn draw_e(&mut self) -> [u64; 3] {
        [self.draw_f(), self.draw_f(), self.draw_f()]
    }

    /// Nonzero element of E: the first of NONZERO_TRIES = 2 elements of E
    /// that is not zero, else 1.
    pub fn draw_e_nonzero(&mut self) -> [u64; 3] {
        for _ in 0..2 {
            let x = self.draw_e();
            if x != [0, 0, 0] {
                return x;
            }
        }
        [1, 0, 0]
    }

    /// Position in [0, 2^log_n): the low log_n bits of the next word.
    pub fn draw_position(&mut self, log_n: u32) -> u64 {
        self.next_word() & ((1u64 << log_n) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_stream_matches_close_layout() {
        let q = FirstQuery {
            label: LABEL_RECORD,
            parameters: Digest::ZERO,
            statement: b"test-statement".to_vec(),
            record: Vec::new(),
            salted: false,
            run: None,
        };
        let mut c = Chain::new(q);
        c.close(ChainKind::Seed, &[]);
        // The draw stream is deterministic and total: 64 words are available.
        let first = c.next_word();
        let second = c.next_word();
        assert_ne!(
            first, 0,
            "first draw word should be nonzero with high probability"
        );
        let mut words = Vec::new();
        for _ in 0..62 {
            words.push(c.next_word());
        }
        assert_eq!(words.len(), 62);
        // Same first query → same stream.
        let q2 = FirstQuery {
            label: LABEL_RECORD,
            parameters: Digest::ZERO,
            statement: b"test-statement".to_vec(),
            record: Vec::new(),
            salted: false,
            run: None,
        };
        let mut c2 = Chain::new(q2);
        c2.close(ChainKind::Seed, &[]);
        assert_eq!(c2.next_word(), first);
        assert_eq!(c2.next_word(), second);
    }

    #[test]
    fn draws_are_rejection_sampled() {
        let q = FirstQuery {
            label: LABEL_RECORD,
            parameters: Digest::ZERO,
            statement: Vec::new(),
            record: Vec::new(),
            salted: false,
            run: None,
        };
        let mut c = Chain::new(q);
        c.close(ChainKind::Seed, &[]);
        for _ in 0..20 {
            assert!(c.draw_f() < GOLDILOCKS_P);
        }
        let beta = c.draw_e_nonzero();
        assert_ne!(beta, [0, 0, 0]);
    }
}
