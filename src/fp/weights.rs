//! Weight-claim merging and pending-row construction (`vf_weights_layout.h`).

use alloc::{vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::ExtensionElement as E,
    key_public::LinShape,
    tensor::eq_table,
    wf::{wf_weights_on_rows, TensorRow},
};

#[derive(Debug, Clone)]
pub struct WsGrp {
    pub name: &'static str,
    pub r: u64,
    pub o: u64,
    pub left: u64,
    pub cls: char,
    pub high: bool,
    pub z: i32,
    pub t: i32,
    pub cb0: i32,
}

impl WsGrp {
    pub fn taken_bit(&self, b: i32) -> bool {
        self.high && b >= self.z - self.t || !self.high && b < self.t
    }
    pub fn taken_at(&self, i: i32) -> i32 {
        if self.high {
            self.z - self.t + i
        } else {
            i
        }
    }
    pub fn a_taken(&self) -> bool {
        self.cls == 'A' && self.t > 0
    }
}

#[derive(Debug, Clone)]
pub struct WsUnit {
    pub x: u32,
    pub l: u32,
    pub nlw: u32,
    pub ncb: i32,
    pub g: Vec<WsGrp>,
    pub rows: u64,
    pub off: u64,
}

#[derive(Debug, Clone)]
pub struct WsLay {
    pub lk: i32,
    pub m0: u64,
    pub u: Vec<WsUnit>,
}

#[derive(Debug, Clone)]
pub struct WsFac {
    pub hv: Vec<E>,
    pub pt: Vec<E>,
    pub av: Vec<E>,
}

#[derive(Debug, Clone)]
pub struct WsClaim {
    pub ui: u32,
    pub f: Vec<WsFac>,
}

#[derive(Debug, Clone)]
pub struct WsRowFac {
    pub hv: Vec<E>,
    pub pl: Vec<E>,
    pub av: Vec<E>,
}

#[derive(Debug, Clone)]
pub struct WsRecClaim {
    pub ui: u32,
    pub ka: E,
    pub f: Vec<WsRowFac>,
}

pub type WsMerged = crate::fp::layer_checks::WeightMerge;

fn e_grp(name: &'static str, r: u64, high: bool) -> WsGrp {
    let mut o = r;
    let mut z = 0;
    while o & 1 == 0 {
        o >>= 1;
        z += 1;
    }
    WsGrp {
        name,
        r,
        o,
        left: 0,
        cls: 'E',
        high,
        z,
        t: 0,
        cb0: 0,
    }
}

fn a_grp(name: &'static str, r: u64) -> WsGrp {
    WsGrp {
        name,
        r,
        o: 0,
        left: 0,
        cls: 'A',
        high: false,
        z: 0,
        t: 0,
        cb0: 0,
    }
}

pub fn ws_lay() -> WsLay {
    let lk = 11;
    let mut m0 = 0u64;
    let mut u = Vec::new();
    let mut add = |x: u32, l: u32, nlw: u32, mut g: Vec<WsGrp>| {
        let mut left = lk;
        let mut cb = 0;
        let mut rows = nlw as u128;
        for x in &mut g {
            let t = if x.cls == 'E' {
                x.z.min(left)
            } else {
                let c = crate::fp::tensor::ceil_log2(x.r) as i32;
                if c <= left {
                    c
                } else {
                    0
                }
            };
            x.t = t;
            x.cb0 = cb;
            left -= t;
            cb += t;
            x.left = if x.cls == 'E' {
                x.r >> t
            } else if x.a_taken() {
                1
            } else {
                x.r
            };
            rows *= x.left as u128;
        }
        let off = m0;
        m0 += rows as u64;
        u.push(WsUnit {
            x,
            l,
            nlw,
            ncb: cb,
            g,
            rows: rows as u64,
            off,
        });
    };
    for l in 0..8u32 {
        let moe = l % 2 == 1;
        add(
            0,
            l,
            2,
            vec![a_grp("hd", 64), a_grp("cd", 128), e_grp("kk", 512, false)],
        );
        add(
            1,
            l,
            2,
            vec![e_grp("kk", 512, false), e_grp("x", 256, true)],
        );
        add(
            2,
            l,
            2,
            vec![e_grp("kk", 512, false), e_grp("x", 512, true)],
        );
        if !moe {
            add(
                3,
                l,
                2,
                vec![e_grp("kk", 512, false), e_grp("x", 1024, true)],
            );
            add(
                4,
                l,
                2,
                vec![e_grp("kk", 512, false), e_grp("x", 1024, true)],
            );
            add(
                9,
                l,
                2,
                vec![e_grp("kk", 1024, false), e_grp("x", 512, true)],
            );
        } else {
            add(
                5,
                l,
                2,
                vec![e_grp("kk", 512, false), e_grp("x", 512, true)],
            );
            add(
                6,
                l,
                2,
                vec![e_grp("kk", 512, false), e_grp("x", 512, true)],
            );
            add(
                7,
                l,
                2,
                vec![
                    e_grp("kk", 512, false),
                    e_grp("e", 4, false),
                    e_grp("x", 512, true),
                ],
            );
            add(
                8,
                l,
                2,
                vec![
                    e_grp("kk", 512, false),
                    e_grp("e", 4, false),
                    e_grp("x", 512, true),
                ],
            );
            add(
                9,
                l,
                2,
                vec![e_grp("kk", 512, false), e_grp("x", 512, true)],
            );
            add(
                10,
                l,
                2,
                vec![
                    e_grp("kk", 512, false),
                    e_grp("e", 4, false),
                    e_grp("x", 512, true),
                ],
            );
            add(11, l, 2, vec![e_grp("kk", 512, false), e_grp("x", 4, true)]);
        }
    }
    add(12, 8, 4, vec![a_grp("v", 512), e_grp("x", 512, true)]);
    add(13, 8, 2, vec![a_grp("x", 512), e_grp("kk", 512, false)]);
    let m0 = m0.div_ceil(2048) * 2048;
    WsLay { lk, m0, u }
}

fn ws_eq1(z: E, x: E) -> E {
    z * x + (E::ONE - z) * (E::ONE - x)
}

pub fn ws_eps_t(lay: &WsLay, claim: &WsClaim, rm: &[E]) -> E {
    let unit = &lay.u[claim.ui as usize];
    let lk = lay.lk as usize;
    let x = |b: usize| rm[lk - 1 - b];
    let mut p = E::ONE;
    for (gi, g) in unit.g.iter().enumerate() {
        let xfac = &claim.f[gi];
        if g.cls == 'E' {
            for i in 0..g.t {
                let pt = xfac.pt[(g.z - 1 - g.taken_at(i)) as usize];
                let coordinate = x((g.cb0 + i) as usize);
                p *= ws_eq1(pt, coordinate);
            }
        } else if g.a_taken() {
            let zz: Vec<E> = (0..g.t)
                .map(|s| x((g.cb0 + g.t - 1 - s) as usize))
                .collect();
            let point: Vec<E> = zz.iter().rev().copied().collect();
            let e = eq_table(&point);
            let mut s = E::ZERO;
            for (i, &coefficient) in xfac.av.iter().enumerate() {
                s += coefficient * e[i];
            }
            p *= s;
        }
    }
    for b in unit.ncb as usize..lk {
        p *= E::ONE - x(b);
    }
    p
}

pub fn ws_row_part(lay: &WsLay, claim: &WsClaim, ka: E) -> WsRecClaim {
    let unit = &lay.u[claim.ui as usize];
    let mut f = Vec::with_capacity(unit.g.len());
    for (gi, g) in unit.g.iter().enumerate() {
        let xfac = &claim.f[gi];
        if g.cls == 'E' {
            let mut pl = Vec::new();
            for b in (0..g.z).rev() {
                if !g.taken_bit(b) {
                    pl.push(xfac.pt[(g.z - 1 - b) as usize]);
                }
            }
            f.push(WsRowFac {
                hv: xfac.hv.clone(),
                pl,
                av: vec![],
            });
        } else if !g.a_taken() {
            f.push(WsRowFac {
                hv: vec![],
                pl: vec![],
                av: xfac.av.clone(),
            });
        } else {
            f.push(WsRowFac {
                hv: vec![],
                pl: vec![],
                av: vec![],
            });
        }
    }
    WsRecClaim {
        ui: claim.ui,
        ka,
        f,
    }
}

fn gamma_add(gam: &mut [E], lay: &WsLay, claim: &WsRecClaim, wgt: E) {
    let unit = &lay.u[claim.ui as usize];
    let mut v = Vec::with_capacity(unit.nlw as usize);
    let mut s = wgt * claim.ka;
    for _ in 0..unit.nlw {
        v.push(s);
        s *= E::from_base(crate::fp::field::FieldElement(256));
    }
    for gi in (0..unit.g.len()).rev() {
        let g = &unit.g[gi];
        let xfac = &claim.f[gi];
        let mut rf = vec![E::ZERO; g.left as usize];
        if g.cls == 'E' {
            let zl = (g.z - g.t) as usize;
            let point: Vec<E> = xfac.pl.iter().rev().copied().collect();
            let e = eq_table(&point);
            for hi in 0..g.o {
                for lo in 0..1 << zl {
                    let at = ((hi << zl) + lo) as usize;
                    rf[at] = xfac.hv[hi as usize] * e[lo as usize];
                }
            }
        } else if g.a_taken() {
            rf[0] = E::ONE;
        } else {
            for (i, &coefficient) in xfac.av.iter().enumerate() {
                rf[i] = coefficient;
            }
        }
        let mut next = Vec::with_capacity(v.len() * rf.len());
        for &a in &v {
            for &b in &rf {
                next.push(a * b);
            }
        }
        v = next;
    }
    if v.len() as u64 != unit.rows || unit.off + unit.rows > gam.len() as u64 {
        return;
    }
    for y in 0..unit.rows as usize {
        gam[unit.off as usize + y] += v[y];
    }
}

pub fn ws_pending(
    lay: &WsLay,
    merged: &WsMerged,
    claims: usize,
    shape: &LinShape,
) -> Result<WsPendingOut, FpError> {
    if lay.m0 != shape.matrix_rows
        || merged.rec.len() != claims
        || merged.rm.len() != lay.lk as usize
    {
        #[cfg(feature = "std")]
        eprintln!(
            "RUST_WEIGHTS_PENDING lay_m0={} shape_rows={} rec_len={} claims={} rm_len={} lk={}",
            lay.m0,
            shape.matrix_rows,
            merged.rec.len(),
            claims,
            merged.rm.len(),
            lay.lk
        );
        return Err(FpError::Range {
            context: "weights-pending",
        });
    }
    let (mut first, mut last) = (u64::MAX, 0);
    for claim in &merged.rec {
        let unit = &lay.u[claim.ui as usize];
        first = first.min(unit.off);
        last = last.max(unit.off + unit.rows);
    }
    if first >= last || last > lay.m0 || shape.k_row() > 62 || lay.m0 > (1u64 << shape.k_row()) {
        return Err(FpError::Range {
            context: "weights-gamma-rows",
        });
    }
    let mut gam = vec![E::ZERO; last as usize];
    for claim in &merged.rec {
        gamma_add(&mut gam, lay, claim, E::ONE);
    }
    let values = &gam[first as usize..last as usize];
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_OWN_BASE").is_some() {
        eprintln!(
            "RUST_WEIGHTS_GAMMA first={} last={} values={:?}",
            first,
            last,
            values
                .iter()
                .take(12)
                .map(|value| value.0[0])
                .collect::<Vec<_>>()
        );
    }
    let mut g = Vec::new();
    wf_weights_on_rows(
        shape.k_row(),
        first,
        values,
        0,
        values.len() as u64,
        E::ONE,
        &mut g,
    );
    let psi = merged.rm.iter().rev().copied().collect();
    Ok(WsPendingOut {
        g,
        psi,
        beta: merged.beta,
        target: merged.fvp,
    })
}

#[derive(Debug)]
pub struct WsPendingOut {
    pub g: Vec<crate::fp::tensor::Tensor>,
    pub psi: Vec<E>,
    pub beta: E,
    pub target: E,
}

pub fn pending_row(
    lay: &WsLay,
    merged: &WsMerged,
    claims: usize,
    shape: &LinShape,
) -> Result<TensorRow, FpError> {
    let pending = ws_pending(lay, merged, claims, shape)?;
    Ok(TensorRow {
        g: pending.g,
        psi: pending.psi,
        masks: vec![],
        target: pending.target,
    })
}
