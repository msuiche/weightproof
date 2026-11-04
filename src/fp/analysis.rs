//! Group analysis, sizing and stage solving (numbers/blocks.h, derive.h).
//!
//! The code with the larger rate determines the distance (`mask_rules`).
//! `size_group` picks the largest theorem integer m whose rounds meet the
//! target; `solve_stages` recurses the proximity test's shapes and solves
//! positions/random tails by iteration. Ground truth: every stage printed by
//! `cp_verify -plan request -shapes` for the stored example.

use alloc::vec::Vec;

use crate::fp::codes::{
    johnson, johnson_count, johnson_list, positions, repeated, word_of, Code, CodeSize,
    CountSource, Proximity,
};
use crate::fp::error::FpError;
use crate::fp::field::GOLDILOCKS_P;

/// Scope of a round (the `work positions` line's three values, in order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    Request = 0,
    Shared = 1,
    Commit = 2,
}

/// What sizing needs of the build: error target, OOD answers, work bits.
#[derive(Debug, Clone, Copy)]
pub struct Setting {
    pub target: u32,
    pub samples: u64,
    /// work positions per Scope (policy: `work positions 20 28 32`).
    pub work_positions: [u32; 3],
    /// work closeness per Scope (policy: all zero in the example).
    pub work_closeness: [u32; 3],
    /// work sample per Scope (all zero in the example).
    pub work_sample: [u32; 3],
    /// work field per Scope (all zero in the example).
    pub work_field: [u32; 3],
    /// Whether the verifier fixes one round-polynomial coefficient, so two E
    /// coefficients are sent instead of three (setting.saved_coefficient).
    pub saved_coefficient: bool,
}

impl Setting {
    /// The stored example's setting (policy.txt).
    pub fn example() -> Self {
        Self {
            target: 128,
            samples: 1,
            work_positions: [20, 28, 32],
            work_closeness: [0, 0, 0],
            work_sample: [0, 0, 0],
            work_field: [0, 0, 0],
            saved_coefficient: true,
        }
    }

    /// |E| = p^3 (cubic extension of Goldilocks).
    pub fn card_e(&self) -> f64 {
        let p = GOLDILOCKS_P as f64;
        p * p * p
    }

    /// Largest error that meets the target with w bits of work: 2^(w - target).
    pub fn goal_of(&self, work: [u32; 3], scope: Scope) -> f64 {
        let w = work[scope as usize];
        if w >= self.target {
            return f64::INFINITY;
        }
        crate::fp::codes::pow2_neg_pub(self.target - w)
    }

    pub fn query_work(&self, scope: Scope) -> u32 {
        self.work_positions[scope as usize]
    }
}

/// A group of rows tested together at one distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupSize {
    /// log of main rows (row variables).
    pub k: u32,
    /// log of mask rows.
    pub h: u32,
    /// Rows folded side by side in one main row.
    pub slots: u64,
    pub main: CodeSize,
    pub mask: CodeSize,
    /// The theorem's integer m, chosen by `size_group`.
    pub m: u64,
}

impl GroupSize {
    pub fn has_mask(&self) -> bool {
        self.mask.ell != 0
    }
}

/// True if the mask code has the larger rate — then it determines the distance.
pub fn mask_rules(g: &GroupSize) -> bool {
    if !g.has_mask() {
        return false;
    }
    (g.mask.k() as u128) * (g.main.n() as u128) > (g.main.k() as u128) * (g.mask.n() as u128)
}

/// The analysis of a group: each code's proximity and the joint list size.
#[derive(Debug, Clone, Copy)]
pub struct Analysis {
    pub at: Proximity,
    pub mask_at: Option<Proximity>,
    pub list: u64,
    pub with_mask: bool,
    pub fault: Option<&'static str>,
    /// Main-code proximity is invalid (trivial code or no admissible distance).
    pub at_bad: bool,
}

impl Analysis {
    pub fn bad(&self) -> bool {
        self.at_bad || self.fault.is_some() || (self.with_mask && self.mask_at.is_none())
    }
}

fn shape_fault(g: &GroupSize) -> Option<&'static str> {
    if !g.has_mask() {
        return if g.h != 0 {
            Some("mask rows without a mask code")
        } else {
            None
        };
    }
    if g.mask.log_n > g.main.log_n {
        return Some("the mask code is longer than the main code");
    }
    if g.h > g.k {
        return Some("more row variables for the mask rows than for the main rows");
    }
    if g.k > 0 && g.h == 0 && mask_rules(g) {
        return Some("a single mask row beside folded rows has a code of larger rate");
    }
    None
}

/// `at_distance`: the code at `1 - agree`, with the count of the first source
/// that applies (THIRD below a third of the distance, else JOHNSON).
pub fn at_distance<C: Code>(code: &C, agree: f64) -> Option<Proximity> {
    if code.k() >= code.n() || !(agree <= 1.0) {
        return None;
    }
    let n = code.n();
    let k = code.k();
    let rho = k as f64 / n as f64;
    let delta = 1.0 - agree;
    let errors = (delta * n as f64) as u64; // floor
    if 3u64.saturating_mul(errors) < n - k {
        return Some(Proximity {
            rho,
            agree,
            delta,
            eta: 0.0,
            m: 0,
            list: 1,
            count: (errors + 1) as f64,
            source: CountSource::Third,
        });
    }
    let root = crate::fp::codes::sqrt_pub(rho);
    let eta = agree - root;
    if eta <= 0.0 {
        return None;
    }
    let q = root / eta;
    let m = (crate::fp::codes::ceil_pub(q) as u64).max(3);
    let count = johnson_count(code, rho, delta, m);
    // List only where the third-of-distance form does not cover the count.
    let list = if 2.0 * delta * n as f64 >= (n - k) as f64 {
        johnson_list(agree, rho)?
    } else {
        1
    };
    Some(Proximity {
        rho,
        agree,
        delta,
        eta,
        m,
        list,
        count,
        source: CountSource::Johnson,
    })
}

/// `analysis_at`: the group at `at`, mask code analysed at the same distance
/// as its repetition to the main length.
pub fn analysis_at(g: &GroupSize, at: Proximity) -> Analysis {
    let mut a = Analysis {
        at,
        mask_at: None,
        list: at.list,
        with_mask: g.has_mask(),
        fault: shape_fault(g),
        at_bad: false,
    };
    if g.has_mask() {
        let mask_at = at_distance(&repeated(&g.mask, g.main.log_n), at.agree);
        if let Some(mut z) = mask_at {
            if z.list > a.list {
                a.list = z.list;
            }
            if z.source != CountSource::Third {
                z.count += z.list as f64;
            }
            a.mask_at = Some(z);
        }
    }
    a
}

/// `analysis_of_m`: under the Johnson rule with integer m, the code with the
/// larger rate gives the distance.
pub fn analysis_of_m(g: &GroupSize) -> Analysis {
    if mask_rules(g) {
        let Some(z) = johnson(&repeated(&g.mask, g.main.log_n), g.m) else {
            return Analysis {
                at: Proximity::default_bad(),
                mask_at: None,
                list: 1,
                with_mask: g.has_mask(),
                fault: shape_fault(g),
                at_bad: true,
            };
        };
        let Some(at) = at_distance(&word_of(&g.main), z.agree) else {
            return Analysis {
                at: Proximity::default_bad(),
                mask_at: Some(z),
                list: z.list,
                with_mask: true,
                fault: shape_fault(g),
                at_bad: true,
            };
        };
        let mut a = analysis_at(g, at);
        let mut z = z;
        z.count += z.list as f64;
        a.mask_at = Some(z);
        a
    } else {
        match johnson(&g.main, g.m) {
            Some(at) => analysis_at(g, at),
            None => Analysis {
                at: Proximity::default_bad(),
                mask_at: None,
                list: 1,
                with_mask: g.has_mask(),
                fault: shape_fault(g),
                at_bad: true,
            },
        }
    }
}

impl Proximity {
    fn default_bad() -> Self {
        Self {
            rho: 0.0,
            agree: 0.0,
            delta: 1.0,
            eta: 0.0,
            m: 0,
            list: 1,
            count: f64::NAN,
            source: CountSource::Johnson,
        }
    }
}

/// Mutual-correlated-agreement error: count / |E|.
pub fn mca_error(s: &Setting, at: &Proximity) -> f64 {
    at.count / s.card_e()
}

/// A nonzero polynomial of degree `degree` vanishes at a challenge: degree / |E|.
pub fn field_error(s: &Setting, degree: u64) -> f64 {
    degree as f64 / s.card_e()
}

/// `rows` values combined by powers of a challenge give 0: (rows - 1) / |E|.
pub fn powers_error(s: &Setting, rows: u64) -> f64 {
    rows.saturating_sub(1) as f64 / s.card_e()
}

/// Two different messages of a list of size L share an OOD sample:
/// (L^2 / 2) * max(cube_bits / |E|, 1 / (|E| - 1))^answers.
pub fn sample_error(s: &Setting, list: u64, cube_bits: u32) -> f64 {
    let l = list as f64;
    let per_answer = (cube_bits as f64 / s.card_e()).max(1.0 / (s.card_e() - 1.0));
    l * l / 2.0 * crate::fp::codes::pow_u64_pub(per_answer, s.samples)
}

/// FOLD's round j error (j = 1..k).
pub fn fold_round_error(s: &Setting, g: &GroupSize, a: &Analysis, j: u32) -> f64 {
    let mut e = mca_error(s, &a.at) + a.list as f64 * field_error(s, 2);
    if g.has_mask() && j <= g.h {
        if let Some(mask_at) = &a.mask_at {
            e += 2.0 * mca_error(s, mask_at);
        }
    }
    e
}

/// BASE's first-round error (single-row group).
pub fn base_error(s: &Setting, g: &GroupSize, a: &Analysis) -> f64 {
    let mut e = mca_error(s, &a.at) + a.list as f64 * field_error(s, 1);
    if g.has_mask() {
        if let Some(mask_at) = &a.mask_at {
            e += mca_error(s, mask_at);
        }
    }
    e
}

/// What the rounds outside FOLD/BASE need of a group.
#[derive(Debug, Clone, Copy)]
pub struct GroupLoad {
    pub cube_bits: u32,
    pub messages: u64,
    pub batched_rows: u64,
}

/// True if the distance-dependent rounds (except position draws) meet the target.
pub fn group_holds(s: &Setting, scope: Scope, g: &GroupSize, load: &GroupLoad) -> bool {
    let a = analysis_of_m(g);
    if a.bad() {
        return false;
    }
    let worst = if g.k > 0 {
        fold_round_error(s, g, &a, 1)
    } else {
        base_error(s, g, &a)
    };
    let goal = s.goal_of(s.work_closeness, scope);
    worst <= goal
        && (load.messages as f64 * sample_error(s, a.list, load.cube_bits))
            <= s.goal_of(s.work_sample, scope)
        && (a.list as f64 * (powers_error(s, load.batched_rows) + field_error(s, 1)))
            <= s.goal_of(s.work_field, scope)
}

/// `largest_m`: an m in [3, limit] at which `good` is true, or None.
/// Mirrors codes.h: linear scan to 67, then doubling, then binary search.
pub fn largest_m(mut good: impl FnMut(u64) -> bool) -> Option<u64> {
    let limit = 1u64 << 40;
    let mut lo = 3u64;
    while lo < 67 && !good(lo) {
        lo += 1;
    }
    if lo == 67 {
        return None;
    }
    let mut hi = lo + 1;
    while hi < limit && good(hi) {
        lo = hi;
        hi *= 2;
    }
    if hi >= limit {
        if good(limit) {
            return Some(limit);
        }
        hi = limit;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if good(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

/// The group at the largest m that holds (m = 0 and bad analysis if none).
pub fn size_group(s: &Setting, scope: Scope, shape: &GroupSize, load: &GroupLoad) -> GroupSize {
    let mut g = *shape;
    g.m = largest_m(|m| {
        let mut candidate = *shape;
        candidate.m = m;
        group_holds(s, scope, &candidate, load)
    })
    .unwrap_or(0);
    g
}

/// Solve a group's random tails and position count together (derive.h
/// `solve_group`). The returned stage-1 group and `t` are the inputs to
/// `solve_stages`; both random tails hide `t` opened positions.
pub fn solve_group(
    s: &Setting,
    scope: Scope,
    shape: &GroupSize,
    load: &GroupLoad,
) -> Result<(GroupSize, u64), FpError> {
    let mut t = 0u64;
    for _turn in 0..32 {
        let mut candidate = *shape;
        candidate.main.r = CodeSize::tail_for(t);
        candidate.mask.r = CodeSize::tail_for(t);
        if candidate.main.dim() > candidate.main.n() || candidate.mask.dim() > candidate.mask.n() {
            return Err(FpError::Range {
                context: "group-code-overflow",
            });
        }
        candidate = size_group(s, scope, &candidate, load);
        if !group_holds(s, scope, &candidate, load) {
            return Err(FpError::Range {
                context: "group-no-distance",
            });
        }
        let a = analysis_of_m(&candidate);
        let next = positions(
            a.at.agree,
            s.target - s.query_work(scope),
            crate::fp::codes::pow2_neg_pub(s.target + 30),
            1,
        )
        .ok_or(FpError::Range {
            context: "group-no-positions",
        })?;
        if next <= t {
            return Ok((candidate, t));
        }
        t = next;
    }
    Err(FpError::Range {
        context: "group-not-settled",
    })
}

/// stage_cube_bits: variables of a sample's point on an oracle message.
pub fn stage_cube_bits(g: &GroupSize) -> u32 {
    fn ceil_log2(x: u64) -> u32 {
        if x <= 1 {
            0
        } else {
            64 - (x - 1).leading_zeros()
        }
    }
    let a = g.k + ceil_log2(g.main.ell);
    if g.has_mask() {
        let b = g.h + ceil_log2(g.mask.ell);
        a.max(b) + 1
    } else {
        a
    }
}

/// One stage of a proximity test (stage 1 is the group itself).
#[derive(Debug, Clone, Copy)]
pub struct Stage {
    pub g: GroupSize,
    pub t: u64,
    pub r_ood: u64,
}

/// A caller-chosen stage shape (k, h, log_n, mask_log_n from `stages` lines).
#[derive(Debug, Clone, Copy)]
pub struct StageChoice {
    pub k: u32,
    pub h: u32,
    pub log_n: u32,
    pub mask_log_n: u32,
}

fn ceil_log2(x: u64) -> u32 {
    if x <= 1 {
        0
    } else {
        64 - (x - 1).leading_zeros()
    }
}

fn pow2_at_least(x: u64) -> u64 {
    if x <= 1 {
        1
    } else {
        1u64 << ceil_log2(x)
    }
}

/// `solve_stages`: stage shapes follow the fold; positions/random tails are
/// solved by iteration from below (derive.h, blocks.h).
pub fn solve_stages(
    s: &Setting,
    scope: Scope,
    first: &GroupSize,
    first_t: u64,
    choices: &[StageChoice],
) -> Result<Vec<Stage>, FpError> {
    if choices.is_empty() || first.k == 0 {
        return Err(FpError::Range {
            context: "stages-first-stage-fold",
        });
    }
    let mut stages = Vec::with_capacity(choices.len() + 1);
    stages.push(Stage {
        g: *first,
        t: first_t,
        r_ood: 0,
    });
    for (i, c) in choices.iter().enumerate() {
        let o = &stages[stages.len() - 1].g;
        let mut n = Stage {
            g: *o,
            t: 0,
            r_ood: 0,
        };
        if c.h > c.k
            || c.k > 32
            || (i + 1 == choices.len() && c.k != 0)
            || (i + 1 < choices.len() && c.k == 0)
        {
            return Err(FpError::Range {
                context: "stages-k-h-shape",
            });
        }
        n.g.k = c.k;
        n.g.h = c.h;
        n.g.main.log_n = c.log_n;
        n.g.main.over_E = true;
        n.g.mask.log_n = c.mask_log_n;
        n.g.mask.over_E = true;
        let cut = 1u64 << c.k;
        if o.main.ell * o.slots % cut != 0 {
            return Err(FpError::Range {
                context: "stages-fold-divisibility",
            });
        }
        n.g.main.ell = o.main.ell * o.slots >> c.k;
        let carried = 2 * o.mask.ell + o.main.r * o.slots + 2 * o.mask.r;
        let own_fold = if c.k > 0 { 2 * c.k + 1 - c.h } else { 1 };
        n.g.mask.ell = pow2_at_least((carried + s.samples + (1u64 << c.h) - 1) >> c.h)
            .max(pow2_at_least(own_fold as u64));
        n.r_ood = (n.g.mask.ell << c.h) - carried;
        let mut t: u64 = 0;
        let mut settled = false;
        for _turn in 0..32 {
            n.g.main.r = CodeSize::tail_for(t);
            n.g.mask.r = CodeSize::tail_for(t);
            if n.g.main.dim() > n.g.main.n() || n.g.mask.dim() > n.g.mask.n() {
                return Err(FpError::Range {
                    context: "stages-code-overflow",
                });
            }
            let load = GroupLoad {
                cube_bits: stage_cube_bits(&n.g),
                messages: 1,
                batched_rows: 4 + 3 * o.main.r.max(1024),
            };
            let shape = n.g;
            n.g = size_group(s, scope, &shape, &load);
            let serves = group_holds(s, scope, &n.g, &load);
            let a = analysis_of_m(&n.g);
            let t2 = if serves {
                positions(
                    a.at.agree,
                    s.target - s.query_work(scope),
                    crate::fp::codes::pow2_neg_pub(s.target + 30),
                    1,
                )
            } else {
                None
            };
            match t2 {
                Some(next) if next > t => {
                    t = next;
                    if t > (1u64 << 32) {
                        return Err(FpError::Range {
                            context: "stages-positions-unbounded",
                        });
                    }
                }
                _ => {
                    if !serves {
                        return Err(FpError::Range {
                            context: "stages-no-distance",
                        });
                    }
                    settled = true;
                    break;
                }
            }
        }
        if !settled {
            return Err(FpError::Range {
                context: "stages-not-settled",
            });
        }
        n.t = t;
        stages.push(n);
    }
    Ok(stages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting() -> Setting {
        Setting::example()
    }

    fn group(
        _name: &str,
        log_n: u32,
        ell: u64,
        r: u64,
        mask_log_n: u32,
        mask_ell: u64,
        k: u32,
    ) -> GroupSize {
        GroupSize {
            k,
            h: 0,
            slots: 1,
            main: CodeSize {
                log_n,
                ell,
                r,
                over_E: false,
            },
            mask: CodeSize {
                log_n: mask_log_n,
                ell: mask_ell,
                r,
                over_E: true,
            },
            m: 0,
        }
    }

    #[test]
    fn size_group_recovers_oracle_m_for_link() {
        let g0 = group("link", 18, 131072, 218, 17, 8192, 9);
        let load = GroupLoad {
            cube_bits: stage_cube_bits(&g0),
            messages: 8,
            batched_rows: 1 << 16,
        };
        let g = size_group(&setting(), Scope::Request, &g0, &load);
        assert_eq!(g.m, 517, "link group m from -plan request -shapes");
        let a = analysis_of_m(&g);
        assert!(!a.bad());
        assert_eq!(a.list, 107);
        let t = positions(a.at.agree, 128 - 20, 0.0, 1).unwrap();
        assert_eq!(t, 218);
    }

    #[test]
    fn size_group_recovers_oracle_m_for_committer_groups() {
        // once-folded: code 2^17, message 32768 + 97, m = 483, list 241, t = 97.
        let g0 = group("once-folded", 17, 32768, 97, 13, 64, 5);
        let load = GroupLoad {
            cube_bits: stage_cube_bits(&g0),
            messages: 8,
            batched_rows: 1 << 16,
        };
        let g = size_group(&setting(), Scope::Commit, &g0, &load);
        assert_eq!(g.m, 483);
        let a = analysis_of_m(&g);
        assert_eq!(a.list, 241);
        // once-omega: code 2^18, message 16384 + 98, m = 277, list 415.
        let g0 = group("once-omega", 18, 16384, 98, 13, 64, 2);
        let load = GroupLoad {
            cube_bits: stage_cube_bits(&g0),
            messages: 8,
            batched_rows: 1 << 16,
        };
        let g = size_group(&setting(), Scope::Commit, &g0, &load);
        assert_eq!(g.m, 277);
        assert_eq!(analysis_of_m(&g).list, 415);
        // ranges: code 2^22, message 262144 + 49, m = 158, list 238.
        let g0 = group("ranges", 22, 262144, 49, 16, 1024, 3);
        let load = GroupLoad {
            cube_bits: stage_cube_bits(&g0),
            messages: 8,
            batched_rows: 1 << 16,
        };
        let g = size_group(&setting(), Scope::Commit, &g0, &load);
        assert_eq!(g.m, 158);
        assert_eq!(analysis_of_m(&g).list, 238);
    }

    #[test]
    fn link_stages_match_oracle_shapes_lists_and_positions() {
        // stage 1 = link group, m = 517, t = 218.
        let first = group("link", 18, 131072, 218, 17, 8192, 9);
        let choices = [
            StageChoice {
                k: 4,
                h: 3,
                log_n: 17,
                mask_log_n: 17,
            },
            StageChoice {
                k: 3,
                h: 3,
                log_n: 16,
                mask_log_n: 16,
            },
            StageChoice {
                k: 3,
                h: 3,
                log_n: 15,
                mask_log_n: 15,
            },
            StageChoice {
                k: 0,
                h: 0,
                log_n: 15,
                mask_log_n: 15,
            },
        ];
        let stages = solve_stages(&setting(), Scope::Request, &first, 218, &choices).unwrap();
        assert_eq!(stages.len(), 5);

        // stage 2: 8192 + 55 in 2^17; mask 4096 + 55 in 2^17; list 477, t 55.
        let s2 = &stages[1];
        assert_eq!(s2.g.main.ell, 8192);
        assert_eq!(s2.g.mask.ell, 4096);
        assert_eq!(s2.r_ood, 15730);
        assert_eq!(s2.t, 55);
        assert_eq!(analysis_of_m(&s2.g).list, 477);

        // stage 3: 1024 + 44 in 2^16; mask 2048 + 44; list 599, t 44 (mask-dominant).
        let s3 = &stages[2];
        assert_eq!(s3.g.main.ell, 1024);
        assert_eq!(s3.g.mask.ell, 2048);
        assert_eq!(s3.r_ood, 8027);
        assert_eq!(s3.t, 44);
        assert!(mask_rules(&s3.g), "stage 3 is mask-dominant");
        assert_eq!(analysis_of_m(&s3.g).list, 599);

        // stage 4: 128 + 44 in 2^15; mask 1024 + 44; list 682, t 44.
        let s4 = &stages[3];
        assert_eq!(s4.g.main.ell, 128);
        assert_eq!(s4.g.mask.ell, 1024);
        assert_eq!(s4.r_ood, 3964);
        assert_eq!(s4.t, 44);
        assert_eq!(analysis_of_m(&s4.g).list, 682);

        // stage 5: 128 + 73 in 2^15; mask 4096 + 73; list 469, t 73.
        let s5 = &stages[4];
        assert_eq!(s5.g.main.ell, 128);
        assert_eq!(s5.g.mask.ell, 4096);
        assert_eq!(s5.r_ood, 1916);
        assert_eq!(s5.t, 73);
        assert_eq!(analysis_of_m(&s5.g).list, 469);
    }

    #[test]
    fn at_distance_third_vs_johnson() {
        // High agreement (small delta) takes the proved THIRD count.
        let code = CodeSize {
            log_n: 10,
            ell: 100,
            r: 10,
            over_E: false,
        };
        let x = at_distance(&word_of(&code), 0.999).unwrap();
        assert_eq!(x.source, CountSource::Third);
        assert_eq!(x.m, 0);
        // Low agreement (large delta, but above the Johnson radius) takes
        // Johnson with m >= 3.
        let x = at_distance(&word_of(&code), 0.5).unwrap();
        assert_eq!(x.source, CountSource::Johnson);
        assert!(x.m >= 3);
        // Below the Johnson radius (agree < sqrt(rho)) there is no distance.
        assert!(at_distance(&word_of(&code), 0.05).is_none());
    }

    #[test]
    fn rejects_bad_stage_choices() {
        let first = group("link", 18, 131072, 218, 17, 8192, 9);
        assert!(solve_stages(&setting(), Scope::Request, &first, 218, &[]).is_err());
        let bad = [StageChoice {
            k: 2,
            h: 3,
            log_n: 17,
            mask_log_n: 17,
        }];
        assert!(
            solve_stages(&setting(), Scope::Request, &first, 218, &bad).is_err(),
            "h > k"
        );
        let bad_last = [StageChoice {
            k: 1,
            h: 0,
            log_n: 17,
            mask_log_n: 17,
        }];
        assert!(
            solve_stages(&setting(), Scope::Request, &first, 218, &bad_last).is_err(),
            "last stage must have k = 0"
        );
    }
}
