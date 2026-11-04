//! Code sizes and Johnson-regime proximity parameters (numbers/codes.h).
//!
//! Plain Reed–Solomon only (`Setting::circle` is out of scope): dimension
//! `dim = ell + r`, the formulas' `k = dim - 1` (the theorem on codes writes
//! rho = k / n with dimension k + 1). Random tail `tail_for(t) = t`.
//!
//! Ground truth: every group printed by `cp_verify -plan request -shapes`
//! for the stored example (spikes/plans/request-shapes.txt). Tests pin the
//! printed distance (6 decimals), integer list size, and position count.

use crate::fp::error::FpError;

/// sqrt(rho) without std (no_std core has no float math).
fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// base^exp for a non-negative integer exponent, by squaring — exact
/// multiplies, no libm::pow rounding differences vs C++ `std::pow` on integers.
fn pow_u64(mut base: f64, mut exp: u64) -> f64 {
    let mut out = 1.0f64;
    while exp > 0 {
        if exp & 1 == 1 {
            out *= base;
        }
        base *= base;
        exp >>= 1;
    }
    out
}

/// 2^-bits for bits >= 0 (exact power of two).
fn pow2_neg(bits: u32) -> f64 {
    if bits >= 64 {
        // Shift in two steps to avoid overflowing the u64 literal.
        libm::pow(2.0, -(bits as i32) as f64)
    } else {
        1.0 / (1u64 << bits) as f64
    }
}

/// Public no_std math helpers for sibling modules (analysis).
pub fn sqrt_pub(x: f64) -> f64 {
    sqrt(x)
}

pub fn pow2_neg_pub(bits: u32) -> f64 {
    pow2_neg(bits)
}

pub fn ceil_pub(x: f64) -> f64 {
    libm::ceil(x)
}

pub fn pow_u64_pub(base: f64, exp: u64) -> f64 {
    pow_u64(base, exp)
}

/// Anything the formulas read as a code: a length and the k of the rate
/// formula. Two instantiations: `CodeSize` (a code with message+randomness)
/// and `Word` (a code seen through `word_of`/`repeated`).
pub trait Code {
    fn log_n(&self) -> u32;
    fn k(&self) -> u64;
    fn n(&self) -> u64 {
        1u64 << self.log_n()
    }
}

/// A code as the formulas read it: length `2^log_n` and rate parameter k.
/// `repeated` models a short row committed as copies in a longer tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word {
    pub log_n: u32,
    pub k: u64,
}

impl Code for Word {
    fn log_n(&self) -> u32 {
        self.log_n
    }
    fn k(&self) -> u64 {
        self.k
    }
}

/// `word_of`: the code itself as a formula input.
pub fn word_of(code: &CodeSize) -> Word {
    Word { log_n: code.log_n, k: code.k() }
}

/// A code of length `2^log_n` with every codeword written `2^(log_n - code.log_n)`
/// times: same k/n of the formulas, for distances/list sizes/counts.
pub fn repeated(code: &CodeSize, log_n: u32) -> Word {
    if log_n <= code.log_n {
        word_of(code)
    } else {
        Word { log_n, k: code.k() << (log_n - code.log_n) }
    }
}

/// A code's sizes: length `n = 2^log_n`, message length `ell`,
/// `r` random coordinates of the zero-knowledge encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CodeSize {
    pub log_n: u32,
    pub ell: u64,
    pub r: u64,
    /// Symbols/messages/randomness in the extension field E.
    pub over_E: bool,
}

impl Code for CodeSize {
    fn log_n(&self) -> u32 {
        self.log_n
    }
    fn k(&self) -> u64 {
        self.dim() - 1
    }
}

impl CodeSize {
    pub fn n(&self) -> u64 {
        1u64 << self.log_n
    }

    /// What a row consists of when sent in the clear.
    pub fn coordinates(&self) -> u64 {
        self.ell + self.r
    }

    /// Dimension of the code containing all encodings (plain RS: ell + r).
    pub fn dim(&self) -> u64 {
        self.ell + self.r
    }

    /// Random tail needed to hide `t` opened positions (plain RS: t).
    pub fn tail_for(t: u64) -> u64 {
        t
    }
}

/// Where a mutual-correlated-agreement count comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountSource {
    /// Within a third of the minimum distance (proved lemma).
    Third,
    /// ePrint 2025/2055, Theorem 4.6, taken as given.
    Johnson,
}

/// A code at one relative distance, with the theorem-on-codes count and the
/// Johnson counting-form list size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proximity {
    pub rho: f64,
    /// 1 - delta: agreement fraction = sqrt(rho) * (1 + 1/m).
    pub agree: f64,
    pub delta: f64,
    /// Slack below the Johnson radius: sqrt(rho) / m.
    pub eta: f64,
    /// The theorem's integer m (>= 3).
    pub m: u64,
    /// Codewords within the distance (Johnson counting form, floored).
    pub list: u64,
    /// Mutual-correlated-agreement count (ePrint 2025/2055, Thm 4.6, M = 1).
    pub count: f64,
    pub source: CountSource,
}

/// `count = (2 M^5 + 3 M delta rho) / (3 rho^1.5) * n + M / sqrt(rho)`,
/// with M = m + 1/2.
pub fn johnson_count<C: Code>(code: &C, rho: f64, delta: f64, m: u64) -> f64 {
    let big_m = m as f64 + 0.5;
    let root = sqrt(rho);
    (2.0 * pow_u64(big_m, 5) + 3.0 * big_m * delta * rho) / (3.0 * rho * root)
        * code.n() as f64
        + big_m / root
}

/// Johnson counting form: `(agree - lambda) / (agree^2 - lambda)`, floored.
/// `None` if the denominator is not positive (list is unbounded).
pub fn johnson_list(agree: f64, lambda: f64) -> Option<u64> {
    let denominator = agree * agree - lambda;
    if denominator <= 0.0 {
        return None;
    }
    let value = (agree - lambda) / denominator;
    if !(0.0..=u64::MAX as f64).contains(&value) {
        return None;
    }
    Some(libm::floor(value) as u64)
}

/// The code at the distance that `m` gives: `1 - delta = sqrt(rho)(1 + 1/m)`.
/// `None` if m < 3 or the code is trivial (k >= n).
pub fn johnson<C: Code>(code: &C, m: u64) -> Option<Proximity> {
    if m < 3 || code.k() >= code.n() {
        return None;
    }
    let rho = code.k() as f64 / code.n() as f64;
    let root = sqrt(rho);
    let eta = root / m as f64;
    let agree = root + eta;
    let delta = 1.0 - agree;
    let list = johnson_list(agree, rho)?;
    Some(Proximity {
        rho,
        agree,
        delta,
        eta,
        m,
        list,
        count: johnson_count(code, rho, delta, m),
        source: CountSource::Johnson,
    })
}

/// Smallest `t` with `factor * (agree^t + extra) <= 2^-bits`, or `None` if
/// the goal is unreachable (agree >= 1, factor = 0, or t would exceed 2^32).
/// Matches codes.h: upper bounds are compared, so t is never too small.
pub fn positions(agree: f64, bits: u32, extra: f64, factor: u64) -> Option<u64> {
    if factor == 0 || !(agree < 1.0) {
        return None;
    }
    let goal = pow2_neg(bits) / factor as f64;
    if !(extra < goal) {
        return None;
    }
    let ok = |t: u64| factor as f64 * (pow_u64(agree, t) + extra) <= goal;
    let mut hi: u64 = 1;
    while !ok(hi) {
        hi *= 2;
        if hi > (1u64 << 32) {
            return None;
        }
    }
    let mut lo = hi / 2;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ok(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

/// Convert a policy `group` line (`group <name> 1 log_n log_ell mask_log_n mask_log_ell`)
/// into the main code's shape. The mask code is a separate CodeSize over E.
pub fn group_main_code(log_n: u32, log_ell: u32, r: u64, over_E: bool) -> CodeSize {
    CodeSize { log_n, ell: 1u64 << log_ell, r, over_E }
}

/// Convert the mask part of a `group` line (always over E per derive.h).
pub fn group_mask_code(mask_log_n: u32, mask_log_ell: u32, r: u64) -> CodeSize {
    CodeSize { log_n: mask_log_n, ell: 1u64 << mask_log_ell, r, over_E: true }
}

/// Planning errors for derived sizes.
pub fn trivial_code_error() -> FpError {
    FpError::Range { context: "code-k-exceeds-n" }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct GroupFixture {
        name: &'static str,
        log_n: u32,
        ell: u64,
        r: u64,
        m: u64,
        printed_delta: f64,
        printed_list: u64,
        work: u32,
        printed_t: u64,
    }

    // spikes/plans/request-shapes.txt, group lines. work values are the
    // per-scope position work bits (request 20/28, committer 32) recovered
    // from the printed position counts.
    const GROUPS: &[GroupFixture] = &[
        GroupFixture { name: "link", log_n: 18, ell: 131072, r: 218, m: 517, printed_delta: 0.290939, printed_list: 107, work: 20, printed_t: 218 },
        GroupFixture { name: "folded", log_n: 17, ell: 32768, r: 101, m: 483, printed_delta: 0.498201, printed_list: 241, work: 28, printed_t: 101 },
        GroupFixture { name: "Omega", log_n: 18, ell: 16384, r: 3831, m: 295, printed_delta: 0.721371, printed_list: 384, work: 28, printed_t: 55 },
        GroupFixture { name: "own", log_n: 18, ell: 16384, r: 3831, m: 295, printed_delta: 0.721371, printed_list: 384, work: 20, printed_t: 59 },
        GroupFixture { name: "ranges", log_n: 22, ell: 262144, r: 49, m: 158, printed_delta: 0.748395, printed_list: 238, work: 32, printed_t: 49 },
        GroupFixture { name: "once-folded", log_n: 17, ell: 32768, r: 97, m: 483, printed_delta: 0.498231, printed_list: 241, work: 32, printed_t: 97 },
        GroupFixture { name: "once-omega", log_n: 18, ell: 16384, r: 98, m: 277, printed_delta: 0.748356, printed_list: 415, work: 32, printed_t: 49 },
        GroupFixture { name: "once-own", log_n: 18, ell: 16384, r: 98, m: 277, printed_delta: 0.748356, printed_list: 415, work: 32, printed_t: 49 },
        GroupFixture { name: "once-ranges", log_n: 22, ell: 262144, r: 49, m: 158, printed_delta: 0.748395, printed_list: 238, work: 32, printed_t: 49 },
    ];

    #[test]
    fn every_oracle_group_matches() {
        for g in GROUPS {
            let code = CodeSize { log_n: g.log_n, ell: g.ell, r: g.r, over_E: false };
            let x = johnson(&code, g.m).unwrap_or_else(|| panic!("{}: johnson failed", g.name));
            assert!(
                (x.delta - g.printed_delta).abs() < 5e-7,
                "{}: distance {} vs printed {}",
                g.name, x.delta, g.printed_delta
            );
            assert_eq!(x.list, g.printed_list, "{}: list size", g.name);
            let t = positions(x.agree, 128 - g.work, 0.0, 1)
                .unwrap_or_else(|| panic!("{}: positions failed", g.name));
            assert_eq!(t, g.printed_t, "{}: positions", g.name);
        }
    }

    #[test]
    fn random_tails_hide_drawn_positions() {
        for g in GROUPS {
            assert_eq!(CodeSize::tail_for(g.printed_t), g.printed_t);
        }
        // The link group's mask tail equals its drawn position count.
        let link = &GROUPS[0];
        assert_eq!(link.r, CodeSize::tail_for(link.printed_t));
    }

    #[test]
    fn count_formula_is_positive_and_sane() {
        let code = CodeSize { log_n: 18, ell: 131072, r: 218, over_E: false };
        let x = johnson(&code, 517).unwrap();
        // ePrint 2025/2055 Thm 4.6 count for the link group: finite, positive.
        assert!(x.count > 0.0 && x.count.is_finite());
        assert!((x.agree - 0.709061).abs() < 1e-5);
        assert!((x.eta - sqrt(x.rho) / 517.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(johnson(&CodeSize { log_n: 18, ell: 131072, r: 218, over_E: false }, 2).is_none(), "m < 3");
        assert!(johnson(&CodeSize { log_n: 1, ell: 1, r: 3, over_E: false }, 3).is_none(), "k >= n");
        assert!(positions(1.0, 128, 0.0, 1).is_none(), "agree = 1 unreachable");
        assert!(positions(0.5, 128, 0.0, 0).is_none(), "factor = 0");
        assert!(positions(0.5, 128, 1.0, 1).is_none(), "extra >= goal");
        assert!(johnson_list(0.5, 0.5).is_none(), "agree^2 = lambda");
    }

    #[test]
    fn group_lines_shape_codes() {
        // policy: group link 1 18 17 17 13 → main code 2^18, ell = 2^17.
        let main = group_main_code(18, 17, 218, false);
        assert_eq!(main.n(), 262144);
        assert_eq!(main.ell, 131072);
        let mask = group_mask_code(17, 13, 218);
        assert_eq!(mask.n(), 131072);
        assert_eq!(mask.ell, 8192);
        assert!(mask.over_E);
    }
}
