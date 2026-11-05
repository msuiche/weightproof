//! Allocation of sampled oracle-message mask entries.

use crate::fp::field::ExtensionElement as E;
use alloc::vec::Vec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskSlot {
    pub oracle: u32,
    pub index: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct MaskTerm {
    pub slot: MaskSlot,
    pub coeff: E,
}

#[derive(Debug, Clone)]
pub struct MaskLedger {
    oracle: u32,
    next: u64,
    left: u64,
    rows: u64,
    log_rows: u32,
    closed: bool,
}

impl MaskLedger {
    pub fn new(oracle: u32, first: u64, count: u64, rows: u64, log_rows: u32) -> Self {
        Self {
            oracle,
            next: first,
            left: count,
            rows,
            log_rows,
            closed: false,
        }
    }

    pub fn take(&mut self) -> MaskSlot {
        assert!(!self.closed, "mask ledger closed");
        assert!(self.left != 0, "mask ledger exhausted");
        while self.next & ((1u64 << self.log_rows) - 1) >= self.rows {
            self.next += 1;
        }
        self.left -= 1;
        let slot = MaskSlot {
            oracle: self.oracle,
            index: self.next,
        };
        self.next += 1;
        slot
    }

    pub fn take_many(&mut self, count: usize) -> Vec<MaskSlot> {
        (0..count).map(|_| self.take()).collect()
    }

    pub fn left(&self) -> u64 {
        if self.closed {
            0
        } else {
            self.left
        }
    }

    pub fn close(&mut self) {
        self.closed = true;
    }
}

pub fn free_mask_entries(mask_rows: u64, ell: u64, answers: u64) -> u64 {
    if mask_rows == 0 || ell < answers {
        0
    } else {
        mask_rows * (ell - answers) + mask_rows - 1
    }
}
