//! Structured coefficient tensors and forms for verifier algebra.

use crate::fp::field::{ExtensionElement as E, FieldElement as F};
use alloc::{vec, vec::Vec};

pub(crate) fn ceil_log2(x: u64) -> u32 {
    if x <= 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub scalar: E,
    pairs: Vec<E>,
    from: usize,
    pick_mask: u64,
    pick_value: u64,
    vars: u32,
    dense_vars: Vec<u32>,
    values: Option<Vec<E>>,
    first: u64,
}

impl Tensor {
    fn real_at(&self, t: u32) -> usize {
        self.from + (!self.pick_mask & ((1u64 << t) - 1)).count_ones() as usize
    }

    fn picks(&self, t: u32) -> bool {
        (self.pick_mask >> t) & 1 != 0
    }

    pub fn eq(z: &[E], scalar: E) -> Self {
        let mut out = Self {
            scalar,
            pairs: vec![],
            from: 0,
            pick_mask: 0,
            pick_value: 0,
            vars: 0,
            dense_vars: vec![],
            values: None,
            first: 0,
        };
        for &x in z {
            out.push(E::ONE - x, x);
        }
        out
    }

    pub fn product(f: &[F], scalar: E) -> Self {
        let mut out = Self {
            scalar,
            pairs: vec![],
            from: 0,
            pick_mask: 0,
            pick_value: 0,
            vars: 0,
            dense_vars: vec![],
            values: None,
            first: 0,
        };
        for &x in f {
            out.push(E::ONE, E::from_base(x));
        }
        out
    }

    pub fn vars(&self) -> u32 {
        self.vars
    }

    pub fn has_dense(&self) -> bool {
        !self.dense_vars.is_empty()
    }

    pub fn push_dense(&mut self, count: u32) {
        assert!(self.values.is_none());
        for _ in 0..count {
            self.dense_vars.push(self.vars);
            self.push(E::ONE, E::ONE);
        }
    }

    pub fn set_values(&mut self, values: Vec<E>, first: u64) {
        assert!(!self.dense_vars.is_empty() && self.values.is_none());
        assert!(first as usize + (1usize << self.dense_vars.len()) <= values.len());
        self.values = Some(values);
        self.first = first;
    }

    fn dense_index(&self, index: u64) -> usize {
        self.dense_vars
            .iter()
            .enumerate()
            .fold(0usize, |out, (i, &variable)| {
                out | (((index >> variable) & 1) as usize) << i
            })
    }

    pub fn lo(&self, t: u32) -> E {
        if self.picks(t) {
            if (self.pick_value >> t) & 1 != 0 {
                E::ZERO
            } else {
                E::ONE
            }
        } else {
            self.pairs[2 * self.real_at(t)]
        }
    }

    pub fn hi(&self, t: u32) -> E {
        if self.picks(t) {
            if (self.pick_value >> t) & 1 != 0 {
                E::ONE
            } else {
                E::ZERO
            }
        } else {
            self.pairs[2 * self.real_at(t) + 1]
        }
    }

    pub fn push(&mut self, lo: E, hi: E) {
        assert!(self.vars < 63);
        let l0 = lo.is_zero();
        let h0 = hi.is_zero();
        if (l0 && hi == E::ONE) || (h0 && lo == E::ONE) {
            self.pick_mask |= 1u64 << self.vars;
            if l0 {
                self.pick_value |= 1u64 << self.vars;
            }
        } else {
            self.pairs.push(lo);
            self.pairs.push(hi);
        }
        self.vars += 1;
    }

    pub fn set_pair(&mut self, t: u32, lo: E, hi: E) {
        let los: Vec<E> = (0..self.vars)
            .map(|i| if i == t { lo } else { self.lo(i) })
            .collect();
        let his: Vec<E> = (0..self.vars)
            .map(|i| if i == t { hi } else { self.hi(i) })
            .collect();
        self.pairs.clear();
        self.from = 0;
        self.vars = 0;
        self.pick_mask = 0;
        self.pick_value = 0;
        for (l, h) in los.into_iter().zip(his) {
            self.push(l, h);
        }
    }

    pub fn insert_choice(&mut self, at: u32, count: u32, value: u64) {
        while self.vars < at {
            self.push(E::ONE, E::ONE);
        }
        assert!(self.vars + count <= 63);
        if count == 0 {
            return;
        }
        let low = (1u64 << at) - 1;
        let ones = (1u64 << count) - 1;
        self.pick_mask = (self.pick_mask & low) | ((self.pick_mask & !low) << count) | (ones << at);
        self.pick_value =
            (self.pick_value & low) | ((self.pick_value & !low) << count) | ((value & ones) << at);
        for variable in &mut self.dense_vars {
            if *variable >= at {
                *variable += count;
            }
        }
        self.vars += count;
    }

    pub fn rotate(&mut self, by: u32) {
        if self.vars == 0 {
            return;
        }
        let los: Vec<E> = (0..self.vars)
            .map(|i| self.lo((i + by) % self.vars))
            .collect();
        let his: Vec<E> = (0..self.vars)
            .map(|i| self.hi((i + by) % self.vars))
            .collect();
        self.pairs.clear();
        self.from = 0;
        self.vars = 0;
        self.pick_mask = 0;
        self.pick_value = 0;
        for (l, h) in los.into_iter().zip(his) {
            self.push(l, h);
        }
    }

    pub fn at(&self, index: u64) -> E {
        if (index & self.pick_mask) != self.pick_value {
            return E::ZERO;
        }
        let mut out = self.scalar;
        let mut pair = self.from;
        for t in 0..self.vars {
            if !self.picks(t) {
                out *= self.pairs[2 * pair + ((index >> t) & 1) as usize];
                pair += 1;
            }
        }
        if !self.has_dense() || out.is_zero() {
            return out;
        }
        out * self.values.as_ref().unwrap()[self.first as usize + self.dense_index(index)]
    }

    fn folded_dense(&self, g: &[E], count: u32) -> (Option<Vec<E>>, E) {
        let n = 1usize << self.dense_vars.len();
        let mut values =
            self.values.as_ref().unwrap()[self.first as usize..self.first as usize + n].to_vec();
        for i in (0..self.dense_vars.len()).rev() {
            let variable = self.dense_vars[i];
            if variable >= count {
                continue;
            }
            let challenge = g[variable as usize];
            let low_half = 1usize << i;
            let mut next = vec![E::ZERO; values.len() / 2];
            for b in 0..values.len() / (2 * low_half) {
                for a in 0..low_half {
                    let zero = values[a + 2 * low_half * b];
                    let one = values[a + low_half + 2 * low_half * b];
                    next[a + low_half * b] = zero + (one - zero) * challenge;
                }
            }
            values = next;
        }
        if values.len() == 1 {
            (None, values[0])
        } else {
            (Some(values), E::ONE)
        }
    }

    pub fn fix_low(&mut self, g: &[E], count: u32) {
        assert!(count <= self.vars);
        let low = (1u64 << count) - 1;
        let mask = self.pick_mask & low;
        let value = self.pick_value & low;
        if mask != 0 {
            let mut product = E::ONE;
            for t in 0..count {
                if (mask >> t) & 1 != 0 {
                    product *= if (value >> t) & 1 != 0 {
                        g[t as usize]
                    } else {
                        E::ONE - g[t as usize]
                    };
                }
            }
            self.scalar *= product;
        }
        for t in 0..count {
            if (mask >> t) & 1 != 0 {
                continue;
            }
            let lo = self.pairs[2 * self.from];
            let hi = self.pairs[2 * self.from + 1];
            self.from += 1;
            if lo == E::ONE && hi == E::ONE {
                continue;
            }
            self.scalar *= lo + g[t as usize] * (hi - lo);
        }
        self.pick_mask >>= count;
        self.pick_value >>= count;
        self.vars -= count;
        if self.dense_vars.iter().any(|&variable| variable < count) {
            let (values, factor) = self.folded_dense(g, count);
            self.dense_vars.retain(|&variable| variable >= count);
            if self.dense_vars.is_empty() {
                self.scalar *= factor;
                self.values = None;
            } else {
                self.values = values;
            }
            self.first = 0;
        }
        for variable in &mut self.dense_vars {
            *variable -= count;
        }
    }

    pub fn add_to(&self, out: &mut [E]) {
        assert_eq!(out.len(), 1usize << self.vars());
        for (index, value) in out.iter_mut().enumerate() {
            *value += self.at(index as u64);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Form {
    pub tensors: Vec<Tensor>,
    pub entries: Vec<(u64, E)>,
}

impl Form {
    pub fn add(&mut self, tensor: Tensor) {
        self.tensors.push(tensor);
    }

    pub fn add_entry(&mut self, index: u64, coefficient: E) {
        self.entries.push((index, coefficient));
    }

    pub fn add_scaled(&mut self, other: &Self, scale: E) {
        for tensor in &other.tensors {
            let mut copy = tensor.clone();
            copy.scalar *= scale;
            self.tensors.push(copy);
        }
        for &(index, coefficient) in &other.entries {
            self.entries.push((index, coefficient * scale));
        }
    }

    pub fn scale(&mut self, scale: E) {
        for tensor in &mut self.tensors {
            tensor.scalar *= scale;
        }
        for entry in &mut self.entries {
            entry.1 *= scale;
        }
    }

    pub fn insert_choice(&mut self, at: u32, count: u32, value: u64) {
        for tensor in &mut self.tensors {
            tensor.insert_choice(at, count, value);
        }
        let low = (1u64 << at) - 1;
        for (index, _) in &mut self.entries {
            *index = (*index & low) | (value << at) | ((*index & !low) << count);
        }
    }

    pub fn at(&self, index: u64) -> E {
        let mut out = E::ZERO;
        for tensor in &self.tensors {
            out += tensor.at(index);
        }
        for &(at, coefficient) in &self.entries {
            if at == index {
                out += coefficient;
            }
        }
        out
    }

    pub fn fix_low(&mut self, g: &[E], count: u32, scale: E) {
        for tensor in &mut self.tensors {
            tensor.scalar *= scale;
            tensor.fix_low(g, count);
        }
        for (index, coefficient) in &mut self.entries {
            for t in 0..count as usize {
                *coefficient *= if (*index >> t) & 1 == 0 {
                    E::ONE - g[t]
                } else {
                    g[t]
                };
            }
            *coefficient *= scale;
            *index >>= count;
        }
        self.entries.sort_by_key(|entry| entry.0);
        let mut merged: Vec<(u64, E)> = Vec::with_capacity(self.entries.len());
        for (index, coefficient) in self.entries.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.0 == index {
                    last.1 += coefficient;
                    continue;
                }
            }
            merged.push((index, coefficient));
        }
        self.entries = merged;
    }

    pub fn mle(&self, point: &[E]) -> E {
        let mut folded = self.clone();
        folded.fix_low(point, point.len() as u32, E::ONE);
        folded.at(0)
    }

    pub fn add_to(&self, out: &mut [E]) {
        for tensor in &self.tensors {
            assert_eq!(out.len(), 1usize << tensor.vars());
            for (index, value) in out.iter_mut().enumerate() {
                *value += tensor.at(index as u64);
            }
        }
        for &(index, coefficient) in &self.entries {
            out[index as usize] += coefficient;
        }
    }
}

pub fn tensor_below(v: &Tensor, from: u32, count: u32, len: u64) -> Vec<Tensor> {
    assert!(from + count <= v.vars() && count < 64 && len <= 1u64 << count);
    if len == 1u64 << count {
        return vec![v.clone()];
    }
    let mut least = 0u64;
    let mut most = 0u64;
    for t in 0..count {
        if v.lo(from + t).is_zero() {
            least |= 1u64 << t;
        }
        if !v.hi(from + t).is_zero() {
            most |= 1u64 << t;
        }
    }
    if most < len {
        return vec![v.clone()];
    }
    if least >= len {
        return vec![];
    }
    let mut out = Vec::new();
    for q in (0..count).rev() {
        if (len >> q) & 1 == 0 {
            continue;
        }
        let mut w = v.clone();
        for t in q + 1..count {
            if (len >> t) & 1 != 0 {
                w.set_pair(from + t, E::ZERO, w.hi(from + t));
            } else {
                w.set_pair(from + t, w.lo(from + t), E::ZERO);
            }
        }
        w.set_pair(from + q, w.lo(from + q), E::ZERO);
        out.push(w);
    }
    out
}

pub fn fold_low_once(values: &mut Vec<E>, challenge: E) {
    assert_eq!(values.len() % 2, 0);
    for i in 0..values.len() / 2 {
        values[i] = values[2 * i] + challenge * (values[2 * i + 1] - values[2 * i]);
    }
    values.truncate(values.len() / 2);
}

pub fn fold_low(mut values: Vec<E>, challenges: &[E]) -> Vec<E> {
    for &challenge in challenges {
        fold_low_once(&mut values, challenge);
    }
    values
}

pub fn inner(a: &[E], b: &[E]) -> E {
    assert_eq!(a.len(), b.len());
    let mut out = E::ZERO;
    for (&x, &y) in a.iter().zip(b) {
        out += x * y;
    }
    out
}

pub fn block_rows(low: &[E], token: &[E]) -> Vec<E> {
    let mut out = vec![E::ZERO; low.len() * token.len()];
    for (c2, &weight) in low.iter().enumerate() {
        for (t, &value) in token.iter().enumerate() {
            out[c2 * token.len() + t] = weight * value;
        }
    }
    out
}

pub fn eq_table(point: &[E]) -> Vec<E> {
    let mut out = vec![E::ONE];
    for &x in point {
        let old_len = out.len();
        out.reserve(old_len);
        for i in 0..old_len {
            let low = out[i] * (E::ONE - x);
            let high = out[i] * x;
            out[i] = low;
            out.push(high);
        }
    }
    out
}

pub fn product_index(factors: &[F], index: u64) -> F {
    let mut out = F::ONE;
    for (t, &factor) in factors.iter().enumerate() {
        if (index >> t) & 1 != 0 {
            out *= factor;
        }
    }
    out
}

pub fn product_table(factors: &[F], len: usize) -> Vec<F> {
    (0..len)
        .map(|index| product_index(factors, index as u64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(x: u64) -> E {
        E::from_base(F(x))
    }

    #[test]
    fn tensor_fold_matches_cpp() {
        let z: Vec<E> = (1..=21).map(e).collect();
        let gamma: Vec<E> = (2..=7).map(e).collect();
        let mut tensor = Tensor::eq(&z, e(29));
        tensor.scalar *= e(11);
        tensor.fix_low(&gamma, 6);
        assert_eq!(tensor.at(32767), e(14_662_351_929_298_044_661));
    }

    #[test]
    fn form_fold_matches_cpp() {
        let z = [e(2), e(3), e(5)];
        let mut form = Form::default();
        form.add(Tensor::eq(&z, e(29)));
        form.add_entry(0, e(7));
        form.add_entry(3, e(11));
        form.add_entry(5, e(13));
        form.add_entry(7, e(17));
        form.insert_choice(1, 2, 2);
        form.fix_low(&[e(2), e(7)], 2, e(11));
        let expected = [
            0,
            18_446_744_069_414_508_223,
            0,
            113_388,
            0,
            93_984,
            0,
            18_446_744_069_414_438_527,
            0,
            18_446_744_069_414_507_761,
            0,
            114_840,
            0,
            95_700,
            0,
            18_446_744_069_414_440_771,
        ];
        for (index, expected) in expected.into_iter().enumerate() {
            assert_eq!(form.at(index as u64), E([expected, 0, 0]), "index {index}");
        }
    }

    #[test]
    fn high_choice_form_fold_matches_cpp() {
        let z: Vec<E> = (1..=21).map(e).collect();
        let gamma: Vec<E> = (2..=7).map(e).collect();
        let mut form = Form::default();
        form.add(Tensor::eq(&z, e(29)));
        form.add(Tensor::eq(&z, e(31)));
        form.insert_choice(5, 3, 5);
        form.fix_low(&gamma, 6, e(11));
        assert_eq!(form.at(2), e(12_906_612_702_051_138_778));
    }

    #[test]
    fn dense_factor_evaluates_and_folds() {
        let mut tensor = Tensor::eq(&[], E([2, 0, 0]));
        tensor.push_dense(2);
        tensor.set_values(
            vec![E([1, 0, 0]), E([3, 0, 0]), E([5, 0, 0]), E([7, 0, 0])],
            0,
        );
        assert_eq!(tensor.at(0), E([2, 0, 0]));
        assert_eq!(tensor.at(3), E([14, 0, 0]));
        tensor.fix_low(&[E([2, 0, 0])], 1);
        assert_eq!(tensor.at(0), E([10, 0, 0]));
        assert_eq!(tensor.at(1), E([18, 0, 0]));
    }
}
