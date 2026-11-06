//! Smallness-test symbol packing helpers.

use alloc::{vec, vec::Vec};

use crate::fp::{field::ExtensionElement as E, tensor::eq_table};

#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub m: u64,
    pub k: u64,
    pub ms: u64,
    pub j: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct Piece {
    pub tree: u32,
    pub row: u32,
    pub r0: u64,
    pub rp: u64,
    pub c0: u64,
    pub cp: u64,
}

pub fn tie_left(shape: &Shape, rows: &[E], lo: &[E]) -> Vec<E> {
    let mut out = vec![E::ZERO; shape.m as usize];
    for i in 0..shape.ms as usize {
        let value = rows[i] * lo[i % shape.j as usize];
        for lane in 0..3 {
            if 3 * i + lane < shape.m as usize {
                out[3 * i + lane] = if lane == 0 {
                    value
                } else {
                    let mut shifted = value;
                    for _ in 0..8 * lane {
                        shifted += shifted;
                    }
                    shifted
                };
            }
        }
    }
    out
}

pub fn tie_right(shape: &Shape, lo: &[E], gam: &[E]) -> Vec<E> {
    let mut out = vec![E::ZERO; (shape.j * shape.k) as usize];
    for c in 0..shape.k as usize {
        for x in 0..shape.j as usize {
            out[c * shape.j as usize + x] = gam[c] * lo[x];
        }
    }
    out
}

pub fn eq_one(point: &[E], index: u64) -> E {
    eq_table(point)[index as usize]
}
