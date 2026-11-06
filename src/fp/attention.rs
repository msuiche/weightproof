//! Attention sub-proof verification (`vf_attention_check.h`).
//!
//! The accepted example uses the public parameters from `policy.txt`. The
//! protocol order is lookup, score/softmax sumchecks, U3 tables, then the
//! approximate range check.

use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
    forest::{self, Forest},
    layer::HFun,
    model::{psc_begin, psc_ends, psc_row, round_in, Aff, HFork, Psc, RoundForm, RoundName, Tr},
    tensor::eq_table,
};

pub const RM: usize = 0;
pub const RD: usize = 1;
pub const FM: usize = 2;
pub const FD: usize = 3;
pub const WL: usize = 4;
pub const WH: usize = 5;
pub const IX: usize = 6;

fn pw(bits: u32) -> E { E::from_base(F(1u64 << bits)) }
fn ei(x: i64) -> E {
    if x < 0 { -E::from_base(F((-x) as u64)) } else { E::from_base(F(x as u64)) }
}
fn es(x: E, bits: u32) -> E { x * pw(bits) }
fn smul(x: E, s: u64) -> E { x * E::from_base(F(s)) }
fn shift(x: E, bits: u32) -> E {
    let mut y = x;
    for _ in 0..bits { y += y; }
    y
}
fn last(x: &[E], n: usize) -> Vec<E> { x[x.len() - n..].to_vec() }
fn eq_at(z: &[E], n: usize, index: u64) -> E {
    let mut out = E::ONE;
    for v in 0..n {
        let bit = (index >> (n - 1 - v)) & 1 != 0;
        out *= if bit { z[v] } else { E::ONE - z[v] };
    }
    out
}
fn eq_pts(x: &[E], y: &[E]) -> E {
    assert_eq!(x.len(), y.len());
    let mut out = E::ONE;
    for i in 0..x.len() { out *= x[i] * y[i] + (E::ONE - x[i]) * (E::ONE - y[i]); }
    out
}
fn ones_of(z: &[E]) -> E { z.iter().fold(E::ONE, |a, &x| a * x) }
fn id_of(z: &[E]) -> E {
    let n = z.len();
    z.iter().enumerate().fold(E::ZERO, |a, (i, &x)| a + es(x, (n - 1 - i) as u32))
}
fn le_of(x: &[E], y: &[E]) -> E {
    assert_eq!(x.len(), y.len());
    let mut out = E::ONE;
    for v in (0..x.len()).rev() {
        out = (E::ONE - x[v]) * y[v] + (x[v] * y[v] + (E::ONE - x[v]) * (E::ONE - y[v])) * out;
    }
    out
}
fn block_rows(low: &[E], tok: &[E]) -> Vec<E> {
    let mut out = vec![E::ZERO; low.len() * tok.len()];
    for c in 0..low.len() { for t in 0..tok.len() { out[c * tok.len() + t] = low[c] * tok[t]; } }
    out
}
fn cut_last(z: &[E], n: usize, k: usize) -> (&[E], &[E]) {
    assert!(k <= n && n <= z.len());
    (&z[..n - k], &z[n - k..n])
}
fn cover(n_t: u64, p: u64, sg: u64) -> (u64, u64) {
    if sg < p { (p, sg) } else { (n_t - 1 - p, sg - p) }
}

#[derive(Debug, Clone)]
pub struct Piece { pub seg: usize, pub lam: u64, pub a: u64, pub b: u32, pub off: u64, pub weighed: bool }
#[derive(Debug, Clone)]
pub struct CallMeta { pub kind: String, pub sh: u32, pub segs: Vec<usize>, pub pieces: Vec<Piece>, pub tot: u64 }
#[derive(Debug, Clone)]
pub struct AttentionShape {
    pub lt: u32, pub lb: u32, pub lgk: u32, pub ldh: u32,
    pub lam: u64, pub h: u64, pub g: u64, pub ha: u64, pub dh: u64,
    pub t: u64, pub b: u64, pub nt: u64, pub hp: u64,
    pub lnt: u32, pub nm: u32, pub nd: u32, pub lm_s: u32, pub la: u32, pub ls: u32, pub lmb: u32,
    pub nas: u64, pub wc_h: u64, pub wc_o: u64, pub wc_q: u64,
    pub k_as: u64, pub kb_a: u64, pub fill_h: u64, pub fill_o: u64, pub fill_q: u64, pub feat_q: u64,
    pub o_mr: u64, pub o_zr: u64, pub o_ib: u64, pub o_rh: u64, pub o_rhp: u64,
    pub o_q: u64, pub o_k: u64, pub o_v: u64, pub o_o: u64, pub dq: u64, pub dkv: u64,
    pub sh: u32, pub u0: u64, pub c: u64, pub rho: u64, pub nhd: u64, pub om: u64, pub n_w: u64,
    pub calls: Vec<CallMeta>,
}
impl AttentionShape {
    pub fn accepted() -> Self {
        let lt = 6; let lb = 4; let lgk = 1; let ldh = 7;
        let lam = 4; let h = 4; let g = 2; let ha = h / g; let dh = 128;
        let t = 1u64 << lt; let b = 1u64 << lb; let lnt = lt - lb; let nt = 1u64 << lnt; let hp = nt / 2;
        let nm = lgk + 2 * lt - 1; let nd = lgk + lt + lb;
        let lm_s = 11; let la = 8; let ls = la - lt; let lmb = 13;
        let nas = 1; let wc_h = 64; let wc_o = 128; let wc_q = 128;
        let dq = h * dh; let dkv = g * dh; let sh = 12; let u0 = 2307;
        let c = u0 << sh; let rho = 65536 - (1u64 << sh);
        let smax = dh * 2047 * 2047; let nhd = (4 * smax + 1 + rho - 1) / rho; let om = 2 * smax + c;
        let k_as = 120; let kb_a = 1024;
        let o_mr = 0; let o_zr = 4 * lam * h; let o_ib = 8 * lam * h;
        let feat_h = o_ib + lam * lt as u64 * h; let fill_h = (feat_h + wc_h - 1) / wc_h * wc_h;
        let o_rh = 0; let o_rhp = lam * dq; let feat_o = 2 * lam * dq; let fill_o = (feat_o + wc_o - 1) / wc_o * wc_o;
        let o_q = 0; let o_k = 2 * lam * dq; let o_v = o_k + 2 * lam * dkv; let o_o = o_v + 2 * lam * dkv;
        let feat_q = o_o + 2 * lam * dq; let fill_q = (feat_q + wc_q - 1) / wc_q * wc_q;
        let n_w = 3 * k_as + 8 * fill_h + (fill_o * t >> la) + 9 * (fill_q * t >> la) + 3 * kb_a;
        let mut calls = Vec::new();
        for (kind, sh, segs) in [("KR", sh, vec![RM, RD]), ("KFD", sh, vec![FM, FD]), ("R16", 0, vec![WL, IX]), ("WH", 0, vec![WH])] {
            let mut pieces = Vec::new();
            for &seg in &segs {
                let b = match seg { RM | FM => nm, RD | FD => nd, WL => 1 + lgk + lt + ldh, WH => lgk + lt + ldh, IX => lgk + lt, _ => unreachable!() };
                for lam_i in 0..lam { for a in 0..ha { pieces.push(Piece { seg, lam: lam_i, a, b, off: 0, weighed: false }); } }
            }
            pieces.sort_by_key(|p| core::cmp::Reverse(p.b));
            let mut off = 0; for p in &mut pieces { p.off = off; off += 1u64 << p.b; }
            calls.push(CallMeta { kind: kind.into(), sh, segs, pieces, tot: off });
        }
        Self { lt, lb, lgk, ldh, lam, h, g, ha, dh, t, b, nt, hp, lnt, nm, nd, lm_s, la, ls, lmb, nas, wc_h, wc_o, wc_q, k_as, kb_a, fill_h, fill_o, fill_q, feat_q, o_mr, o_zr, o_ib, o_rh, o_rhp, o_q, o_k, o_v, o_o, dq, dkv, sh, u0, c, rho, nhd, om, n_w, calls }
    }
    pub fn np(&self) -> usize { (self.lam * self.ha) as usize }
    pub fn nv_kf(&self, seg: usize) -> usize { if seg == FM { (self.lgk + 2 * self.lnt - 1 + self.ldh) as usize } else { (self.lgk + self.lnt + self.ldh) as usize } }
    pub fn nv_e(&self) -> usize { (self.lgk + self.lt) as usize }
    pub fn nv_vm(&self) -> usize { (self.lgk + 2 * self.lnt - 1 + self.lb) as usize }
    pub fn nv_vd(&self) -> usize { (self.lgk + self.lnt + self.lb) as usize }
    pub fn nv_f(&self) -> usize { (self.lt + self.lgk + self.ldh) as usize }
    pub fn nv_h(&self) -> usize { (self.lt + self.lgk + 4) as usize }
    pub fn scd_degrees(&self) -> Vec<u32> { (0..2 * self.lt + self.lgk).map(|j| if j < self.lt { 2 } else if j < self.lt + self.lgk { self.lt + 2 } else { self.lt + 1 }).collect() }
    pub fn ten_q(&self) -> TenAt { TenAt { off: self.o_q, nl: 2, cnt: self.lam, j: self.dq } }
    pub fn ten_k(&self) -> TenAt { TenAt { off: self.o_k, nl: 2, cnt: self.lam, j: self.dkv } }
    pub fn ten_v(&self) -> TenAt { TenAt { off: self.o_v, nl: 2, cnt: self.lam, j: self.dkv } }
    pub fn ten_o(&self) -> TenAt { TenAt { off: self.o_o, nl: 2, cnt: self.lam, j: self.dq } }
}
#[derive(Debug, Clone, Copy)]
pub struct TenAt { pub off: u64, pub nl: u64, pub cnt: u64, pub j: u64 }
fn ten_cell(x: TenAt, l: u64, i: u64, j: u64) -> u64 { x.off + (l * x.cnt + i) * x.j + j }
fn head_feat(x: TenAt, hd: u64, dh: u64) -> u64 { assert!(dh != 0 && x.j % dh == 0 && hd < x.j / dh); hd * dh }
#[derive(Debug)] pub struct Pw { pub lam: u64, pub a: u64, pub wp: E }
#[derive(Debug)] pub struct PlaneClaim { pub call: usize, pub seg: usize, pub z: Vec<E>, pub is_d: bool, pub pw: Vec<Pw>, pub s0: u32, pub np: usize }
pub struct Sc3 { pub r: Vec<E>, pub s_end: u32, pub nb: usize, pub p: Psc, pub row_u: Aff }
pub struct Scd { pub r: Vec<E>, pub s_end: u32, pub p: Psc, pub row_u: Aff }
pub struct Kf { pub call: usize, pub claim: usize, pub seg: usize, pub pw: Vec<Pw>, pub zg: Vec<E>, pub zp: Vec<E>, pub zs: Vec<E>, pub zi: Vec<E>, pub zj: Vec<E>, pub wi: Vec<E>, pub w: Vec<E>, pub s0: u32, pub sc: Sc3, pub t_m: u64, pub t_q: u64, pub t_k: u64 }
#[derive(Debug, Default)] pub struct TableOffsets { pub t_a: u64, pub t_vm: u64, pub t_vd: u64, pub h_w: u64, pub h_x: u64, pub h_e: u64, pub h_t: u64, pub h_pi: u64, pub h_r2: u64, pub t_o: u64, pub q_o: u64, pub q_vm: u64, pub q_vd: u64, pub q_f: u64, pub q_kx: u64, pub t_bit: [u64; 2], pub t_pin: u64 }
pub struct AttentionRun {
    pub shape: AttentionShape, pub forest: Forest, pub meta: Vec<CallMeta>, pub planes: Vec<PlaneClaim>, pub kfs: Vec<Kf>,
    pub pw_w: Vec<Pw>, pub pw_x: Vec<Pw>, pub pw_h: Vec<Pw>, pub zw: Vec<E>, pub zx: Vec<E>, pub zh: Vec<E>,
    pub s5: u32, pub s_ib: u32, pub s5h: u32, pub sc_e: Sc3, pub vpm: Sc3, pub vpd: Sc3,
    pub gam_p: Vec<E>, pub s_cl: u32, pub points: Vec<E>, pub s_m6: u32, pub sc_f: Sc3, pub scd: Scd, pub sc_h: Sc3, pub w_jv: Vec<E>,
    pub tables: Vec<HFun>, pub tab_len: BTreeMap<u64, u64>, pub offsets: TableOffsets,
    pub row_x: Vec<Vec<E>>, pub vt: Vec<E>, pub vp: Vec<E>, pub n_msg: u32, pub n_lin: u32, pub reads: u64, pub next_table: u64,
}

fn sc3_in(f: &mut HFork, tr: &mut Tr, slot: &mut u32, nv: usize, nb: usize, start: Aff) -> Result<Sc3, FpError> {
    let mut p = psc_begin(f, tr, slot, start)?;
    let mut r = Vec::with_capacity(nv);
    for _ in 0..nv { r.push(round_in(f, tr, slot, RoundForm::Sum3, &mut p, E::ZERO, 0)?); }
    let s_end = psc_ends(f, tr, slot, &mut p, 2, (2 * nb) as u32)?;
    Ok(Sc3 { r, s_end, nb, p, row_u: Aff::default() })
}
fn sc3_a(sc: &Sc3, b: usize) -> Aff { Aff::slot(sc.s_end + (2 * b) as u32) }
fn sc3_b(sc: &Sc3, b: usize) -> Aff { Aff::slot(sc.s_end + (2 * b + 1) as u32) }
fn sc3_row(f: &mut HFork, sc: &mut Sc3, weights: &[E]) -> Result<(), FpError> {
    if weights.len() != sc.nb { return Err(FpError::Range { context: "attention-sc3-weights" }); }
    let mut s0 = E::ZERO;
    for b in 0..sc.nb { s0 += weights[b] * f.shown(sc.s_end + (2 * b) as u32)? * f.shown(sc.s_end + (2 * b + 1) as u32)?; }
    sc.row_u = psc_row(f, &mut sc.p, s0, &Aff::default())?;
    Ok(())
}
fn scd_in(f: &mut HFork, tr: &mut Tr, slot: &mut u32, degrees: &[u32], degree_y: u32, ends: u32, start: Aff) -> Result<Scd, FpError> {
    let mut p = psc_begin(f, tr, slot, start)?;
    let mut r = Vec::with_capacity(degrees.len());
    for &degree in degrees { r.push(round_in(f, tr, slot, RoundForm::Nodes, &mut p, E::ZERO, degree)?); }
    let s_end = psc_ends(f, tr, slot, &mut p, degree_y, ends)?;
    Ok(Scd { r, s_end, p, row_u: Aff::default() })
}
fn sum_w(pw: &[Pw]) -> E { pw.iter().fold(E::ZERO, |a, x| a + x.wp) }
fn times_w(pw: &[Pw], factor: E) -> Vec<E> { pw.iter().map(|x| x.wp * factor).collect() }
fn inner(forest: &Forest, shape: &AttentionShape, q: usize, seg: usize) -> Vec<E> {
    let b = match seg {
        RM | FM => shape.nm,
        RD | FD => shape.nd,
        WL => 1 + shape.lgk + shape.lt + shape.ldh,
        WH => shape.lgk + shape.lt + shape.ldh,
        IX => shape.lgk + shape.lt,
        _ => unreachable!(),
    } as usize;
    last(&forest.calls[q].zk, b)
}
fn pieces(shape: &AttentionShape, meta: &mut [CallMeta], forest: &Forest, q: usize, seg: usize) -> Vec<Pw> {
    let mut out = Vec::new();
    for p in &mut meta[q].pieces {
        if p.seg == seg {
            p.weighed = true;
            out.push(Pw { lam: p.lam, a: p.a, wp: eq_at(&forest.calls[q].zk, (forest.calls[q].n - p.b) as usize, p.off >> p.b) });
        }
    }
    out.sort_by_key(|x| (x.lam, x.a));
    let _ = shape;
    out
}
fn send(f: &mut HFork, tr: &mut Tr, slot: &mut u32, n: u32) -> Result<u32, FpError> { f.claims_in(tr, slot, n) }

/// Reads all messages through the attention SCD/bit sumchecks. U3 and the
/// smallness messages are deliberately left to the caller, since the request
/// frame owns those rounds.
pub fn read_messages(
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
    shape: AttentionShape,
    forest: Forest,
) -> Result<AttentionRun, FpError> {
    let mut meta = shape.calls.clone();
    let mut planes = Vec::new();
    for q in 0..meta.len() {
        if meta[q].kind == "KR" {
            for seg in [RM, RD] {
                if !meta[q].segs.contains(&seg) { continue; }
                let pw = pieces(&shape, &mut meta, &forest, q, seg);
                let s0 = send(f, tr, slot, 3)?;
                planes.push(PlaneClaim { call: q, seg, z: inner(&forest, &shape, q, seg), is_d: seg == RD, pw, s0, np: 3 });
            }
        }
    }
    let mut kfs = Vec::new();
    for q in 0..meta.len() {
        if meta[q].kind != "KFD" { continue; }
        for seg in [FM, FD] {
            if !meta[q].segs.contains(&seg) { continue; }
            let pw = pieces(&shape, &mut meta, &forest, q, seg);
            let z = inner(&forest, &shape, q, seg);
            let mut at = 0;
            let zg = z[at..at + shape.lgk as usize].to_vec(); at += shape.lgk as usize;
            let mut zp = Vec::new(); let mut zs = Vec::new(); let mut zi = Vec::new();
            if seg == FM {
                zp = z[at..at + shape.lnt as usize - 1].to_vec(); at += shape.lnt as usize - 1;
                zs = z[at..at + shape.lnt as usize].to_vec(); at += shape.lnt as usize;
            } else {
                zi = z[at..at + shape.lnt as usize].to_vec(); at += shape.lnt as usize;
            }
            let zi2 = z[at..at + shape.lb as usize].to_vec(); at += shape.lb as usize;
            let zj = z[at..at + shape.lb as usize].to_vec();
            let s0 = send(f, tr, slot, 7)?;
            let claim = planes.len();
            planes.push(PlaneClaim { call: q, seg, z: z.clone(), is_d: seg == FD, pw: Vec::new(), s0, np: 5 });
            planes[claim].pw = pw.iter().map(|x| Pw { lam: x.lam, a: x.a, wp: x.wp }).collect();
            let mut wi = vec![E::ZERO; shape.nt as usize];
            if seg == FM {
                let ep = eq_table(&zp); let esg = eq_table(&zs);
                for p in 0..shape.hp { for sg in 0..shape.nt - 1 {
                    let (i, j) = cover(shape.nt, p, sg);
                    wi[i as usize] += ep[p as usize] * esg[sg as usize];
                }}
            } else { wi = eq_table(&zi); }
            let mut sc = sc3_in(f, tr, slot, shape.nv_kf(seg), pw.len(), Aff::slot(s0 + 6))?;
            let r = &sc.r;
            let mut factor = eq_pts(&zg, &r[..shape.lgk as usize]);
            if seg == FM {
                factor *= eq_pts(&zp, &r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize - 1]);
                let a = &zs;
                let b = &r[shape.lgk as usize + shape.lnt as usize - 1..shape.lgk as usize + 2 * shape.lnt as usize - 1];
                factor *= eq_pts(a, b) - ones_of(a) * ones_of(b);
            } else { factor *= eq_pts(&zi, &r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize]); }
            let w = times_w(&pw, factor);
            sc3_row(f, &mut sc, &w)?;
            kfs.push(Kf { call: q, claim, seg, pw, zg, zp, zs, zi: zi2, zj, wi, w, s0, sc, t_m: 0, t_q: 0, t_k: 0 });
        }
    }
    let q_wl = meta.iter().position(|c| c.segs.contains(&WL)).ok_or(FpError::Range { context: "attention-calls" })?;
    let q_ix = meta.iter().position(|c| c.segs.contains(&IX)).ok_or(FpError::Range { context: "attention-calls" })?;
    let q_wh = meta.iter().position(|c| c.segs.contains(&WH)).ok_or(FpError::Range { context: "attention-calls" })?;
    let pw_w = pieces(&shape, &mut meta, &forest, q_wl, WL);
    let pw_x = pieces(&shape, &mut meta, &forest, q_ix, IX);
    let pw_h = pieces(&shape, &mut meta, &forest, q_wh, WH);
    if pw_w.len() != shape.np() || pw_x.len() != shape.np() || pw_h.len() != shape.np() { return Err(FpError::Range { context: "attention-calls" }); }
    let zw = inner(&forest, &shape, q_wl, WL);
    let zx = inner(&forest, &shape, q_ix, IX);
    let zh = inner(&forest, &shape, q_wh, WH);
    let s5 = send(f, tr, slot, 5)?;
    let s_ib = send(f, tr, slot, shape.lt)?;
    let s5h = send(f, tr, slot, 2)?;
    let s_zo = s5 + 1;
    let zg_w = &zw[1..1 + shape.lgk as usize];
    let zi_w = &zw[1 + shape.lgk as usize..1 + shape.lgk as usize + shape.lt as usize];
    let mut sc_e = sc3_in(f, tr, slot, shape.nv_e(), shape.np(), Aff::slot(s_zo))?;
    let w_e = times_w(&pw_w, eq_pts(zg_w, &sc_e.r[..shape.lgk as usize]) * eq_pts(zi_w, &sc_e.r[shape.lgk as usize..shape.lgk as usize + shape.lt as usize]));
    sc3_row(f, &mut sc_e, &w_e)?;
    let gam_p = tr.batch(2)?;
    let s_cl = send(f, tr, slot, 2)?;
    let mut vpm = sc3_in(f, tr, slot, shape.nv_vm(), shape.np(), Aff::slot(s_cl))?;
    let zihi = &zi_w[..shape.lnt as usize];
    let zilo = &zi_w[shape.lnt as usize..];
    let mut wjv = vec![E::ZERO; shape.nt as usize];
    let ep = eq_table(&vpm.r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize - 1]);
    let esg = eq_table(&vpm.r[shape.lgk as usize + shape.lnt as usize - 1..shape.lgk as usize + 2 * shape.lnt as usize - 1]);
    let eih = eq_table(zihi);
    let mut factor = E::ZERO;
    for p in 0..shape.hp { for sg in 0..shape.nt - 1 {
        let (i, j) = cover(shape.nt, p, sg);
        factor += ep[p as usize] * esg[sg as usize] * eih[i as usize];
        wjv[j as usize] += ep[p as usize] * esg[sg as usize];
    }}
    let w_vm = times_w(&pw_w, factor * eq_pts(zg_w, &vpm.r[..shape.lgk as usize]));
    sc3_row(f, &mut vpm, &w_vm)?;
    let mut vpd = sc3_in(f, tr, slot, shape.nv_vd(), shape.np(), Aff::slot(s_cl + 1))?;
    let w_vd = times_w(&pw_w, eq_pts(zg_w, &vpd.r[..shape.lgk as usize]) * eq_pts(zihi, &vpd.r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize]));
    sc3_row(f, &mut vpd, &w_vd)?;
    let points = tr.point(RoundName::Zero, (2 * shape.lgk + 2 * shape.lt + 4) as usize)?;
    let part = |first: usize, n: usize| points[first..first + n].to_vec();
    let tg = part(0, shape.lgk as usize);
    let ti = part(shape.lgk as usize, shape.lt as usize);
    let tk = part((shape.lgk + shape.lt) as usize, 4);
    let tbg = part((shape.lgk + shape.lt + 4) as usize, shape.lgk as usize);
    let tbi = part((2 * shape.lgk + shape.lt + 4) as usize, shape.lt as usize);
    let s_m6 = send(f, tr, slot, shape.np() as u32)?;
    let om = tr.batch(shape.np())?;
    let mut start_f = Aff::default();
    for b in 0..shape.np() { start_f.add(&Aff::slot(s_m6 + b as u32), om[b]); }
    let mut sc_f = sc3_in(f, tr, slot, shape.nv_f(), shape.np(), start_f)?;
    let f_factor = eq_pts(&tg, &sc_f.r[shape.lt as usize..shape.lt as usize + shape.lgk as usize]) * eq_pts(&ti, &sc_f.r[..shape.lt as usize]);
    let w_f: Vec<E> = om.iter().map(|&x| x * f_factor).collect();
    sc3_row(f, &mut sc_f, &w_f)?;
    let om4 = tr.batch(shape.np())?;
    let mut start_d = Aff::default();
    for b in 0..shape.np() { start_d.add(&sc3_b(&sc_f, b), om4[b]); }
    let ends = (shape.lam + shape.np() as u64 * shape.lt as u64) as u32;
    let mut scd = scd_in(f, tr, slot, &shape.scd_degrees(), shape.lt + 1, ends, start_d)?;
    let pj = &scd.r[..shape.lt as usize];
    let pg = &scd.r[shape.lt as usize..shape.lt as usize + shape.lgk as usize];
    let pi = &scd.r[shape.lt as usize + shape.lgk as usize..];
    let ri = &sc_f.r[..shape.lt as usize];
    let rg = &sc_f.r[shape.lt as usize..shape.lt as usize + shape.lgk as usize];
    let s_kc = scd.s_end;
    let s_ib_at = |b: usize, k: usize| s_kc + (shape.lam as usize + b * shape.lt as usize + k) as u32;
    let mut s0 = E::ZERO;
    for b in 0..shape.np() {
        let mut t = om4[b] * f.shown(s_kc + (b / shape.ha as usize) as u32)?;
        for k in 0..shape.lt as usize {
            let y = pj[shape.lt as usize - 1 - k];
            t *= f.shown(s_ib_at(b, k))? * (y + y - E::ONE) + (E::ONE - y);
        }
        s0 += t;
    }
    scd.row_u = psc_row(f, &mut scd.p, eq_pts(ri, pi) * eq_pts(rg, pg) * s0, &Aff::default())?;
    let om5 = tr.batch(shape.np())?;
    let mut sc_h = sc3_in(f, tr, slot, shape.nv_h(), shape.np(), Aff::default())?;
    let ri2 = &sc_h.r[..shape.lt as usize];
    let rg2 = &sc_h.r[shape.lt as usize..shape.lt as usize + shape.lgk as usize];
    let rk = &sc_h.r[shape.lt as usize + shape.lgk as usize..];
    let h_factor = eq_pts(&tbi, ri2) * eq_pts(&tbg, rg2) * eq_pts(&tk, rk);
    let w_h: Vec<E> = om5.iter().map(|&x| x * h_factor).collect();
    sc3_row(f, &mut sc_h, &w_h)?;
    Ok(AttentionRun {
        shape, forest, meta, planes, kfs, pw_w, pw_x, pw_h, zw, zx, zh,
        s5, s_ib, s5h, sc_e, vpm, vpd, gam_p, s_cl, points, s_m6, sc_f, scd, sc_h, w_jv: wjv,
        tables: vec![], tab_len: BTreeMap::new(), offsets: TableOffsets::default(),
        row_x: vec![], vt: vec![], vp: vec![], n_msg: 0, n_lin: 0, reads: 0, next_table: 0,
    })
}

fn add_table(run: &mut AttentionRun, matrix: usize, l: u64, phi: Vec<E>, elems: u64) -> Result<u64, FpError> {
    if l == 0 || phi.len() != l as usize { return Err(FpError::Range { context: "attention-table-length" }); }
    let at = run.next_table;
    run.tables.push(HFun { matrix, l, phi, at });
    run.tab_len.insert(at, elems);
    run.next_table += elems;
    Ok(at)
}
fn add_as(run: &mut AttentionRun, point: Vec<E>) -> Result<u64, FpError> {
    let first = run.next_table;
    let k_part = run.shape.k_as / run.shape.nas;
    // C++ addAs: eqTab over the last lm_s coordinates of the point.
    let phi = eq_table(&point);
    for q in 0..run.shape.nas as usize { add_table(run, q, 1u64 << run.shape.lm_s, phi.clone(), k_part)?; }
    let total = run.next_table - first;
    run.tab_len.insert(first, total);
    Ok(first)
}
fn tok(hi: &[E], lo: &[E], shape: &AttentionShape) -> Vec<E> {
    let el = eq_table(lo);
    let mut out = vec![E::ZERO; shape.t as usize];
    for i in 0..shape.nt as usize { for x in 0..shape.b as usize { out[i * shape.b as usize + x] = hi[i] * el[x]; } }
    out
}
fn lng(zc: &[E], ptk: &[E], shape: &AttentionShape) -> Vec<E> {
    let (_, lo0) = cut_last(zc, zc.len(), shape.ldh as usize);
    let lo = &lo0[lo0.len() - shape.ls as usize..];
    block_rows(&eq_table(lo), ptk)
}

pub fn build_tables(run: &mut AttentionRun) -> Result<(), FpError> {
    let shape = run.shape.clone();
    let la = 1u64 << shape.la;
    let gso = shape.fill_o * shape.t >> shape.la;
    let gsq = shape.fill_q * shape.t >> shape.la;
    let mb = 1u64 << shape.lmb;
    let i_ah = shape.nas as usize;
    let i_ao = i_ah + 1;
    let i_ba = i_ah + 2;
    let i_qa = i_ah + 3;
    let zg_w = run.zw[1..1 + shape.lgk as usize].to_vec();
    let zi_w = run.zw[1 + shape.lgk as usize..1 + shape.lgk as usize + shape.lt as usize].to_vec();
    let zc_w = run.zw[1 + shape.lgk as usize + shape.lt as usize..].to_vec();
    let zi_x = run.zx[shape.lgk as usize..shape.lgk as usize + shape.lt as usize].to_vec();
    let _ = zg_w;
    let mut pt_m = run.vpm.r[..(shape.lgk + 2 * shape.lnt - 1) as usize].to_vec();
    pt_m.extend_from_slice(&zi_w[shape.lnt as usize..]);
    pt_m.extend_from_slice(&run.vpm.r[run.vpm.r.len() - shape.lb as usize..]);
    let mut pt_d = run.vpd.r[..(shape.lgk + shape.lnt) as usize].to_vec();
    pt_d.extend_from_slice(&zi_w[shape.lnt as usize..]);
    pt_d.extend_from_slice(&run.vpd.r[run.vpd.r.len() - shape.lb as usize..]);
    let t_a = add_as(run, last(&run.forest.zx, shape.lm_s as usize))?;
    let t_vm = add_as(run, last(&pt_m, shape.lm_s as usize))?;
    let t_vd = add_as(run, last(&pt_d, shape.lm_s as usize))?;
    run.offsets.t_a = t_a; run.offsets.t_vm = t_vm; run.offsets.t_vd = t_vd;
    for index in 0..run.kfs.len() {
        let wi = run.kfs[index].wi.clone(); let zi = run.kfs[index].zi.clone();
        let at = add_table(run, i_ah, shape.t, tok(&wi, &zi, &shape), shape.fill_h)?;
        run.kfs[index].t_m = at;
    }
    let h_w = add_table(run, i_ah, shape.t, eq_table(&zi_w), shape.fill_h)?;
    let h_x = add_table(run, i_ah, shape.t, eq_table(&zi_x), shape.fill_h)?;
    let h_e = add_table(run, i_ah, shape.t, eq_table(&run.sc_e.r[shape.lgk as usize..shape.lgk as usize + shape.lt as usize]), shape.fill_h)?;
    let h_t = add_table(run, i_ah, shape.t, eq_table(&run.points[shape.lgk as usize..shape.lgk as usize + shape.lt as usize]), shape.fill_h)?;
    let h_pi = add_table(run, i_ah, shape.t, eq_table(&run.scd.r[shape.lt as usize + shape.lgk as usize..shape.lt as usize + shape.lgk as usize + shape.lt as usize]), shape.fill_h)?;
    let h_r2 = add_table(run, i_ah, shape.t, eq_table(&run.sc_h.r[..shape.lt as usize]), shape.fill_h)?;
    run.offsets.h_w = h_w; run.offsets.h_x = h_x; run.offsets.h_e = h_e; run.offsets.h_t = h_t; run.offsets.h_pi = h_pi; run.offsets.h_r2 = h_r2;
    let t_o = add_table(run, i_ao, la, lng(&zc_w, &eq_table(&zi_w), &shape), gso)?;
    run.offsets.t_o = t_o;
    for index in 0..run.kfs.len() {
        let r = run.kfs[index].sc.r.clone();
        let rc = &r[r.len() - shape.ldh as usize..];
        let mut wir = vec![E::ZERO; shape.nt as usize];
        let mut wjr = vec![E::ZERO; shape.nt as usize];
        if run.kfs[index].seg == FM {
            let ep = eq_table(&r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize - 1]);
            let esg = eq_table(&r[shape.lgk as usize + shape.lnt as usize - 1..shape.lgk as usize + 2 * shape.lnt as usize - 1]);
            for p in 0..shape.hp { for sg in 0..shape.nt - 1 {
                let (i, j) = cover(shape.nt, p, sg);
                wir[i as usize] += ep[p as usize] * esg[sg as usize];
                wjr[j as usize] += ep[p as usize] * esg[sg as usize];
            }}
        } else {
            wir = eq_table(&r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize]);
            wjr = wir.clone();
        }
        let zi = run.kfs[index].zi.clone(); let zj = run.kfs[index].zj.clone();
        let tq = add_table(run, i_qa, la, lng(rc, &tok(&wir, &zi, &shape), &shape), gsq)?;
        let tk = add_table(run, i_qa, la, lng(rc, &tok(&wjr, &zj, &shape), &shape), gsq)?;
        run.kfs[index].t_q = tq; run.kfs[index].t_k = tk;
    }
    run.offsets.q_o = add_table(run, i_qa, la, lng(&zc_w, &eq_table(&run.sc_e.r[shape.lgk as usize..shape.lgk as usize + shape.lt as usize]), &shape), gsq)?;
    run.offsets.q_vm = add_table(run, i_qa, la, lng(&zc_w, &tok(&run.w_jv, &run.vpm.r[run.vpm.r.len() - shape.lb as usize..], &shape), &shape), gsq)?;
    run.offsets.q_vd = add_table(run, i_qa, la, lng(&zc_w, &tok(&eq_table(&run.vpd.r[shape.lgk as usize..shape.lgk as usize + shape.lnt as usize]), &run.vpd.r[run.vpd.r.len() - shape.lb as usize..], &shape), &shape), gsq)?;
    let rc = run.sc_f.r[run.sc_f.r.len() - shape.ldh as usize..].to_vec();
    let ri = run.sc_f.r[..shape.lt as usize].to_vec();
    run.offsets.q_f = add_table(run, i_qa, la, lng(&rc, &eq_table(&ri), &shape), gsq)?;
    let pj = run.scd.r[..shape.lt as usize].to_vec();
    run.offsets.q_kx = add_table(run, i_qa, la, lng(&rc, &eq_table(&pj), &shape), gsq)?;
    run.offsets.t_bit[0] = add_table(run, i_ba, mb, forest::forest_phi_b(&run.forest, shape.lmb, 0), shape.kb_a)?;
    run.offsets.t_bit[1] = add_table(run, i_ba, mb, forest::forest_phi_b(&run.forest, shape.lmb, 1), shape.kb_a)?;
    run.offsets.t_pin = add_table(run, i_ba, mb, forest::forest_phi_p(shape.lmb), shape.kb_a)?;
    if run.next_table != shape.n_w { return Err(FpError::Range { context: "attention-W-length" }); }
    Ok(())
}

impl AttentionRun {
    fn u3at(&self, t: u64, i: u64) -> Result<Aff, FpError> {
        let length = self.tab_len.get(&t).ok_or(FpError::Range { context: "attention-table-index" })?;
        if i >= *length { return Err(FpError::Range { context: "attention-table-index" }); }
        Ok(Aff::entry(0, 0, t + i))
    }
    fn plane(&self, t: u64, x: usize, lam: u64, a: u64, z: &[E], is_d: bool) -> Result<Aff, FpError> {
        let nb = if is_d { self.shape.nd } else { self.shape.nm } as usize;
        let a0 = (if is_d { 5 * self.shape.lam * self.shape.ha << self.shape.nm } else { 0 })
            + (((x as u64 * self.shape.lam + lam) * self.shape.ha + a) << nb);
        let eh = eq_table(&z[..nb - self.shape.lm_s as usize]);
        let mut out = Aff::default();
        for hi in 0..eh.len() { out.add(&self.u3at(t, (a0 >> self.shape.lm_s) + hi as u64)?, eh[hi]); }
        Ok(out)
    }
    fn sh_v(&self, t: u64, off: u64, nl: u64, j: u64, lam: u64, feature: u64) -> Result<Aff, FpError> {
        let mut out = Aff::default();
        for l in 0..nl {
            let at = off + ((l * self.shape.lam + lam) * j + feature);
            out.add(&self.u3at(t, at)?, pw((8 * l) as u32));
        }
        Ok(out)
    }
    fn xl(&self, t: u64, x: TenAt, lam: u64, hd: u64, zc: &[E]) -> Result<Aff, FpError> {
        let (hi, _) = cut_last(zc, zc.len(), self.shape.ldh as usize);
        let ec = eq_table(hi);
        let j0 = head_feat(x, hd, self.shape.dh);
        let s2 = 1u64 << self.shape.ls;
        let mut out = Aff::default();
        for ch in 0..ec.len() {
            for l in 0..x.nl {
                let at = ten_cell(x, l, lam, j0) / s2 + ch as u64;
                out.add(&self.u3at(t, at)?, es(ec[ch], (8 * l) as u32));
            }
        }
        Ok(out)
    }
    fn over_g<FN>(&self, zg: &[E], mut f: FN) -> Aff
    where FN: FnMut(u64) -> Aff {
        let eg = eq_table(zg);
        let mut out = Aff::default();
        for g in 0..self.shape.g { out.add(&f(g), eg[g as usize]); }
        out
    }
    fn lin_planes(&self, f: &mut HFork, pc: &PlaneClaim) -> Result<(), FpError> {
        for x in 0..pc.np {
            let mut form = Aff::slot(pc.s0 + x as u32);
            for w in &pc.pw {
                let plane = self.plane(self.offsets.t_a, x, w.lam, w.a, &pc.z, pc.is_d)?;
                form.add(&plane, -w.wp);
            }
            f.lin(form);
        }
        Ok(())
    }
    fn forest_join(&self, keys: &[Aff]) -> Aff {
        let mut out = Aff::default();
        for q in 0..keys.len() {
            let term = Aff::constant(self.forest.alpha - smul(self.forest.theta, q as u64)).minus(&keys[q]);
            out.add(&term, self.forest.e[q]);
        }
        let ey = eq_table(&self.forest.zy[..self.forest.r_k as usize]);
        for y in self.forest.nt as usize..1usize << self.forest.r_k {
            out.c += ey[y] * (self.forest.alpha - self.forest.padding_key(0));
        }
        out.minus(&self.forest.c_end)
    }

    pub fn add_main_claims(&mut self, f: &mut HFork) -> Result<(), FpError> {
        for row in &self.forest.rows { f.lin(row.clone()); }
        let mut keys = vec![Aff::default(); self.meta.len()];
        let mut joined = vec![false; self.meta.len()];
        for q in 0..self.meta.len() {
            if self.meta[q].kind != "KR" { continue; }
            let mut v = Aff::default();
            for pi in 0..self.planes.len() {
                if self.planes[pi].call != q { continue; }
                let pc = &self.planes[pi];
                self.lin_planes(f, pc)?;
                let lw = sum_w(&pc.pw);
                let l128 = smul(lw, 128);
                v = v + (Aff::slot(pc.s0) + Aff::constant(l128))
                    + (Aff::slot(pc.s0 + 1) + Aff::constant(l128)) * pw(20)
                    + Aff::slot(pc.s0 + 2) * pw(40);
            }
            keys[q] = v;
            joined[q] = true;
        }
        for q in 0..self.meta.len() {
            if self.meta[q].kind != "KFD" { continue; }
            let call = &self.forest.calls[q];
            let wall = self.meta[q].pieces.iter().fold(E::ZERO, |a, p| a + eq_at(&call.zk, (call.n - p.b) as usize, p.off >> p.b));
            let pad = call.tau_at(call.t0);
            let mut v = Aff::constant((E::ONE - wall) * pad);
            for ki in 0..self.kfs.len() {
                if self.kfs[ki].call != q { continue; }
                let pc = self.kfs[ki].claim;
                self.lin_planes(f, &self.planes[pc])?;
                let s = self.kfs[ki].s0;
                let t_m = self.kfs[ki].t_m;
                let pw_list = &self.kfs[ki].pw;
                let mut form = Aff::slot(s + 5);
                for w in pw_list {
                    let values: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(t_m, self.shape.o_mr, 4, self.shape.h, w.lam, g * self.shape.ha + w.a).unwrap()).collect();
                    form.add(&self.over_g(&self.kfs[ki].zg, |g| values[g as usize].clone()), -w.wp);
                }
                f.lin(form);
                f.lin(self.kfs[ki].sc.row_u.clone());
                let rg = self.kfs[ki].sc.r.clone();
                let rc = rg[rg.len() - self.shape.ldh as usize..].to_vec();
                for b in 0..pw_list.len() {
                    let w = &pw_list[b];
                    let q_values: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.kfs[ki].t_q, self.shape.ten_q(), w.lam, g * self.shape.ha + w.a, &rc).unwrap()).collect();
                    f.lin(sc3_a(&self.kfs[ki].sc, b) - self.over_g(&rg[..self.shape.lgk as usize], |g| q_values[g as usize].clone()));
                    let k_values: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.kfs[ki].t_k, self.shape.ten_k(), w.lam, g, &rc).unwrap()).collect();
                    f.lin(sc3_b(&self.kfs[ki].sc, b) - self.over_g(&rg[..self.shape.lgk as usize], |g| k_values[g as usize].clone()));
                }
                let lw = sum_w(pw_list);
                let l128 = smul(lw, 128);
                let dp = Aff::slot(s + 5) - Aff::slot(s + 6) + Aff::constant(if self.kfs[ki].seg == FM { smul(lw, self.shape.c) * ones_of(&self.kfs[ki].zs) } else { smul(lw, self.shape.om) * (E::ONE - le_of(&self.kfs[ki].zj, &self.kfs[ki].zi)) });
                v = v + dp - (Aff::slot(s) + Aff::constant(l128)) - (Aff::slot(s + 1) + Aff::constant(l128)) * pw(8) + Aff::slot(s + 2) * pw(32) + (Aff::slot(s + 3) + Aff::constant(l128) + Aff::slot(s + 4) * pw(8)) * pw(33);
            }
            keys[q] = v;
            joined[q] = true;
        }
        let s_acc = self.s5;
        let s_zo = self.s5 + 1;
        let s_zc = self.s5 + 2;
        let s_rh = self.s5 + 3;
        let s_rhp = self.s5 + 4;
        let zg_w = self.zw[1..1 + self.shape.lgk as usize].to_vec();
        let zi_w = self.zw[1 + self.shape.lgk as usize..1 + self.shape.lgk as usize + self.shape.lt as usize].to_vec();
        let zc_w = self.zw[1 + self.shape.lgk as usize + self.shape.lt as usize..].to_vec();
        let zg_x = self.zx[..self.shape.lgk as usize].to_vec();
        let zi_x = self.zx[self.shape.lgk as usize..self.shape.lgk as usize + self.shape.lt as usize].to_vec();
        let zg_h = self.zh[..self.shape.lgk as usize].to_vec();
        let zc_h = self.zh[self.shape.lgk as usize + self.shape.lt as usize..].to_vec();
        let lww = sum_w(&self.pw_w);
        let lwx = sum_w(&self.pw_x);
        let lwh = sum_w(&self.pw_h);
        let mut zc = Aff::slot(s_zc);
        let mut rh = Aff::slot(s_rh);
        let mut rhp = Aff::slot(s_rhp);
        let mut r2 = Aff::slot(self.s5h);
        let mut r2p = Aff::slot(self.s5h + 1);
        for w in &self.pw_w {
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_w, self.shape.o_zr, 4, self.shape.h, w.lam, g * self.shape.ha + w.a).unwrap()).collect();
            zc.add(&self.over_g(&zg_w, |g| vals[g as usize].clone()), -w.wp);
            let rh_vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.t_o, TenAt { off: self.shape.o_rh, nl: 1, cnt: self.shape.lam, j: self.shape.dq }, w.lam, g * self.shape.ha + w.a, &zc_w).unwrap()).collect();
            rh.add(&self.over_g(&zg_w, |g| rh_vals[g as usize].clone()), -w.wp);
            let rhp_vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.t_o, TenAt { off: self.shape.o_rhp, nl: 1, cnt: self.shape.lam, j: self.shape.dq }, w.lam, g * self.shape.ha + w.a, &zc_w).unwrap()).collect();
            rhp.add(&self.over_g(&zg_w, |g| rhp_vals[g as usize].clone()), -w.wp);
        }
        for w in &self.pw_h {
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.t_o, TenAt { off: self.shape.o_rh, nl: 1, cnt: self.shape.lam, j: self.shape.dq }, w.lam, g * self.shape.ha + w.a, &zc_h).unwrap()).collect();
            r2.add(&self.over_g(&zg_h, |g| vals[g as usize].clone()), -w.wp);
            let vals2: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.t_o, TenAt { off: self.shape.o_rhp, nl: 1, cnt: self.shape.lam, j: self.shape.dq }, w.lam, g * self.shape.ha + w.a, &zc_h).unwrap()).collect();
            r2p.add(&self.over_g(&zg_h, |g| vals2[g as usize].clone()), -w.wp);
        }
        f.lin(zc); f.lin(rh); f.lin(rhp);
        for k in 0..self.shape.lt as usize {
            let mut form = Aff::slot(self.s_ib + k as u32);
            for w in &self.pw_x {
                let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_x, self.shape.o_ib, 1, self.shape.lt as u64 * self.shape.h, w.lam, k as u64 * self.shape.h + g * self.shape.ha + w.a).unwrap()).collect();
                form.add(&self.over_g(&zg_x, |g| vals[g as usize].clone()), -w.wp);
            }
            f.lin(form);
        }
        f.lin(r2); f.lin(r2p);
        let rem = Aff::slot(s_acc) * ei(2) + Aff::slot(s_zc) - Aff::slot(s_zo) * ei(2);
        let mut ib = Aff::default();
        for k in 0..self.shape.lt { ib.add(&Aff::slot(self.s_ib + k), pw(k)); }
        let tw0 = rem.clone() - (Aff::slot(s_rh) + Aff::constant(smul(lww, 128))) * ei(65536);
        let tw1 = Aff::slot(s_zc) * ei(2) - Aff::constant(lww) - rem - (Aff::slot(s_rhp) + Aff::constant(smul(lww, 128))) * ei(65536);
        let zsl = self.zw[0];
        let twl = tw0 * (E::ONE - zsl) + tw1 * zsl;
        let tix = (Aff::constant(lwx * id_of(&zi_x)) - ib) * pw(16 - self.shape.lt);
        for q in 0..self.meta.len() {
            if self.meta[q].kind != "R16" { continue; }
            let mut v = Aff::default();
            if self.meta[q].segs.contains(&WL) { v = v + twl.clone(); }
            if self.meta[q].segs.contains(&IX) { v = v + tix.clone(); }
            keys[q] = v; joined[q] = true;
        }
        let q_wh = self.meta.iter().position(|c| c.segs.contains(&WH)).unwrap();
        keys[q_wh] = (Aff::slot(self.s5h) + Aff::constant(smul(lwh, 128))) + (Aff::slot(self.s5h + 1) + Aff::constant(smul(lwh, 128))) * pw(20);
        joined[q_wh] = true;
        f.lin(self.sc_e.row_u.clone());
        let rg_e = self.sc_e.r[..self.shape.lgk as usize].to_vec();
        for b in 0..self.pw_w.len() {
            let w = &self.pw_w[b];
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_e, self.shape.o_zr, 4, self.shape.h, w.lam, g * self.shape.ha + w.a).unwrap()).collect();
            f.lin(sc3_a(&self.sc_e, b) - self.over_g(&rg_e, |g| vals[g as usize].clone()));
            let vals2: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.q_o, self.shape.ten_o(), w.lam, g * self.shape.ha + w.a, &zc_w).unwrap()).collect();
            f.lin(sc3_b(&self.sc_e, b) - self.over_g(&rg_e, |g| vals2[g as usize].clone()));
        }
        if joined.iter().any(|x| !*x) { return Err(FpError::Range { context: "attention-unclosed-call" }); }
        f.lin(self.forest_join(&keys));
        f.lin(Aff::slot(self.s_cl) + Aff::slot(self.s_cl + 1) - Aff::slot(s_acc) - Aff::slot(s_zc) * self.gam_p[1]);
        f.lin(self.vpm.row_u.clone());
        let pt_m = {
            let mut p = self.vpm.r[..(self.shape.lgk + 2 * self.shape.lnt - 1) as usize].to_vec();
            p.extend_from_slice(&zi_w[self.shape.lnt as usize..]);
            p.extend_from_slice(&self.vpm.r[self.vpm.r.len() - self.shape.lb as usize..]);
            p
        };
        let pt_d = {
            let mut p = self.vpd.r[..(self.shape.lgk + self.shape.lnt) as usize].to_vec();
            p.extend_from_slice(&zi_w[self.shape.lnt as usize..]);
            p.extend_from_slice(&self.vpd.r[self.vpd.r.len() - self.shape.lb as usize..]);
            p
        };
        for b in 0..self.pw_w.len() {
            let w = &self.pw_w[b];
            let a = self.plane(self.offsets.t_vm, 3, w.lam, w.a, &pt_m, false)? + Aff::constant(ei(128)) + self.plane(self.offsets.t_vm, 4, w.lam, w.a, &pt_m, false)? * pw(8);
            f.lin(sc3_a(&self.vpm, b) - a);
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.q_vm, self.shape.ten_v(), w.lam, g, &zc_w).unwrap()).collect();
            f.lin(sc3_b(&self.vpm, b) - (Aff::constant(self.gam_p[1]) + self.over_g(&self.vpm.r[..self.shape.lgk as usize], |g| vals[g as usize].clone())));
        }
        f.lin(self.vpd.row_u.clone());
        for b in 0..self.pw_w.len() {
            let w = &self.pw_w[b];
            let a = self.plane(self.offsets.t_vd, 3, w.lam, w.a, &pt_d, true)? + Aff::constant(ei(128)) + self.plane(self.offsets.t_vd, 4, w.lam, w.a, &pt_d, true)? * pw(8);
            f.lin(sc3_a(&self.vpd, b) - a);
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.q_vd, self.shape.ten_v(), w.lam, g, &zc_w).unwrap()).collect();
            f.lin(sc3_b(&self.vpd, b) - (Aff::constant(self.gam_p[1]) + self.over_g(&self.vpd.r[..self.shape.lgk as usize], |g| vals[g as usize].clone())));
        }
        let tg = self.points[..self.shape.lgk as usize].to_vec();
        let rg = self.sc_f.r[self.shape.lt as usize..self.shape.lt as usize + self.shape.lgk as usize].to_vec();
        let rc = self.sc_f.r[self.sc_f.r.len() - self.shape.ldh as usize..].to_vec();
        for b in 0..self.shape.np() {
            let lam = (b / self.shape.ha as usize) as u64;
            let a = (b % self.shape.ha as usize) as u64;
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_t, self.shape.o_mr, 4, self.shape.h, lam, g * self.shape.ha + a).unwrap()).collect();
            f.lin(Aff::slot(self.s_m6 + b as u32) - self.over_g(&tg, |g| vals[g as usize].clone()));
        }
        f.lin(self.sc_f.row_u.clone());
        for b in 0..self.shape.np() {
            let lam = (b / self.shape.ha as usize) as u64;
            let a = (b % self.shape.ha as usize) as u64;
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.q_f, self.shape.ten_q(), lam, g * self.shape.ha + a, &rc).unwrap()).collect();
            f.lin(sc3_a(&self.sc_f, b) - self.over_g(&rg, |g| vals[g as usize].clone()));
        }
        f.lin(self.scd.row_u.clone());
        let s_kc = self.scd.s_end;
        let pg = self.scd.r[self.shape.lt as usize..self.shape.lt as usize + self.shape.lgk as usize].to_vec();
        for lam in 0..self.shape.lam {
            let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.xl(self.offsets.q_kx, self.shape.ten_k(), lam, g, &rc).unwrap()).collect();
            f.lin(Aff::slot(s_kc + lam as u32) - self.over_g(&pg, |g| vals[g as usize].clone()));
        }
        for b in 0..self.shape.np() {
            let lam = (b / self.shape.ha as usize) as u64;
            let a = (b % self.shape.ha as usize) as u64;
            for k in 0..self.shape.lt as usize {
                let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_pi, self.shape.o_ib, 1, self.shape.lt as u64 * self.shape.h, lam, k as u64 * self.shape.h + g * self.shape.ha + a).unwrap()).collect();
                let at = s_kc + (self.shape.lam as usize + b * self.shape.lt as usize + k) as u32;
                f.lin(Aff::slot(at) - self.over_g(&pg, |g| vals[g as usize].clone()));
            }
        }
        f.lin(self.sc_h.row_u.clone());
        let rg2 = self.sc_h.r[self.shape.lt as usize..self.shape.lt as usize + self.shape.lgk as usize].to_vec();
        let rk = self.sc_h.r[self.shape.lt as usize + self.shape.lgk as usize..].to_vec();
        let ek = eq_table(&rk);
        for b in 0..self.shape.np() {
            let lam = (b / self.shape.ha as usize) as u64;
            let a = (b % self.shape.ha as usize) as u64;
            f.lin(sc3_b(&self.sc_h, b) - sc3_a(&self.sc_h, b) + Aff::constant(E::ONE));
            let mut form = sc3_a(&self.sc_h, b);
            for k in 0..self.shape.lt as usize {
                let vals: Vec<Aff> = (0..self.shape.g).map(|g| self.sh_v(self.offsets.h_r2, self.shape.o_ib, 1, self.shape.lt as u64 * self.shape.h, lam, k as u64 * self.shape.h + g * self.shape.ha + a).unwrap()).collect();
                form.add(&self.over_g(&rg2, |g| vals[g as usize].clone()), -ek[k]);
            }
            f.lin(form);
        }
        let w = self.forest.k_t + 5 - self.shape.lmb;
        let mut at = [Aff::default(), Aff::default()];
        for e in 0..2 {
            let ec = eq_table(&self.forest.zb[e][self.forest.k_t as usize..(self.forest.k_t + self.forest.c_k) as usize]);
            let et = eq_table(&self.forest.zb[e][..w as usize]);
            for c in 0..self.forest.calls.len() {
                for th in 0..1usize << w {
                    let form = self.u3at(self.offsets.t_bit[e], ((c << w) + th) as u64)?;
                    at[e].add(&form, ec[c] * et[th]);
                }
            }
        }
        let padding = self.forest.empty_keys() + self.forest.pub_counts.iter().sum::<u64>();
        let mut pin = Aff::constant(-E::from_base(F((1u64 << self.forest.v()) - padding)));
        for col in 0..(self.forest.calls.len() << w) { pin = pin + self.u3at(self.offsets.t_pin, col as u64)?; }
        let plane_row = psc_row(f, &mut self.forest.leaves, E::ONE, &at[0].clone().scaled(self.forest.leaves_a))?;
        f.lin(plane_row);
        f.lin(Aff::slot(self.forest.s_bit) - at[1].clone());
        f.lin(pin);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Run2 { tree: u32, row: u32, first: u64, coef: E, weights: Vec<E>, w0: u64, count: u64 }

impl AttentionRun {
    pub fn init_trees(&self, f: &mut HFork) {
        if !f.trees.is_empty() { return; }
        let s = &self.shape;
        f.trees.push(crate::fp::model::HTree::new("U3".into(), 'N', 1, s.n_w, 1));
        let matrices = self.matrices();
        for (i, (m, k)) in matrices.iter().take(4).enumerate() {
            f.trees.push(crate::fp::model::HTree::new(format!("M{i}").into(), 'P', (m + 2) / 3, *k, 1));
        }
        f.trees.push(crate::fp::model::HTree::new("ZZ".into(), 'N', 128, 2 * self.mz(), 4));
        f.trees.push(crate::fp::model::HTree::new("BZ".into(), 'N', 32, 2048, 3));
        f.trees.push(crate::fp::model::HTree::new("PXo".into(), 'P', (8192 + 2) / 3, 96, 1));
    }
    pub fn matrices(&self) -> [(u64, u64); 5] {
        let s = &self.shape;
        [(1u64 << s.lm_s, s.k_as), (s.wc_h * s.t, s.fill_h / s.wc_h), (s.wc_o * s.t, s.fill_o / s.wc_o), (1u64 << s.lmb, s.kb_a), (s.wc_q * s.t, s.fill_q / s.wc_q)]
    }
    fn gt_tree(&self, mu: usize) -> u32 { if mu < 4 { 1 + mu as u32 } else { 7 } }
    fn zz_tree(&self) -> u32 { 5 }
    fn bz_tree(&self) -> u32 { 6 }
    fn mz(&self) -> u64 { self.matrices().iter().map(|x| x.0).sum() }
    fn off(&self, mu: usize) -> u64 { self.matrices()[..mu].iter().map(|x| x.0).sum() }
    fn lead(&self, f: &HFork, first: u64) -> E {
        let mut out = E::ONE;
        let mut square = f.theta;
        let mut exponent = (f.plan_claims - f.n_claim) as u64 + first;
        while exponent != 0 { if exponent & 1 != 0 { out *= square; } square *= square; exponent >>= 1; }
        out
    }
    fn t2lin(&self, f: &mut HFork, form: Aff, runs: &[Run2]) {
        let weight = f.kap();
        f.push(weight, &form);
        for run in runs {
            for i in 0..run.count as usize {
                let coefficient = weight * run.coef * run.weights[run.w0 as usize + i];
                let at = (run.first + i as u64) as usize;
                f.trees[run.tree as usize].rows[run.row as usize].l[at] += coefficient;
            }
        }
    }
    fn lin_join9(&self, digits: &Forest, f: &mut HFork) -> Result<(), FpError> {
        let mut form = Aff::default();
        let mut runs = Vec::new();
        for q in 0..2 {
            form.c += digits.e[q] * (digits.alpha - smul(digits.theta, q as u64));
            let call = &digits.calls[q];
            let ec = eq_table(&call.zk[7..]);
            let coef = -digits.e[q] * crate::fp::smallness::eq_one(&call.zk, 0);
            runs.push(Run2 { tree: self.zz_tree(), row: q as u32, first: q as u64 * self.mz(), coef, weights: ec, w0: 0, count: self.mz() });
        }
        let ey = eq_table(&digits.zy);
        for t in digits.nt as usize..1usize << digits.r_k { form.c += ey[t] * (digits.alpha - digits.padding_key(0)); }
        self.t2lin(f, form - digits.c_end.clone(), &runs);
        Ok(())
    }
    fn lin_bits9(&self, digits: &mut Forest, f: &mut HFork) {
        let kb = 1u64 << (digits.k_t + digits.c_k);
        let w0 = eq_table(&digits.zb[0][5..]);
        let w1 = eq_table(&digits.zb[1][5..]);
        let bit_factor = digits.leaves.eps * digits.leaves_a;
        let row = psc_row(f, &mut digits.leaves, E::ONE, &Aff::default()).unwrap();
        self.t2lin(f, row, &[Run2 { tree: self.bz_tree(), row: 0, first: 0, coef: bit_factor, weights: w0, w0: 0, count: kb }]);
        self.t2lin(f, Aff::slot(digits.s_bit), &[Run2 { tree: self.bz_tree(), row: 1, first: 0, coef: -E::ONE, weights: w1, w0: 0, count: kb }]);
        let padding = digits.empty_keys() + digits.pub_counts.iter().sum::<u64>();
        let w: Vec<E> = (0..kb).map(|u| E::from_base(F(1u64 << (u & 31)))).collect();
        self.t2lin(f, Aff::constant(-E::from_base(F((1u64 << digits.v()) - padding))), &[Run2 { tree: self.bz_tree(), row: 2, first: 0, coef: E::ONE, weights: w, w0: 0, count: kb }]);
    }
    pub fn prepare(&mut self, f: &mut HFork, tr: &mut Tr, slot: u32, digits: &Forest) -> Result<(), FpError> {
        let main_lins = 31 + self.forest.v() + self.forest.d() + 1;
        let digit_lins = 31 + digits.v() + digits.d() + 1;
        let np = self.shape.np() as u32;
        let n_krp = 2;
        let n_kfp = 2;
        let sum_np = 16;
        let base = 3 * n_krp + (6 * n_kfp + 2 * sum_np) + (5 + self.shape.lt + 2 * np) + 1 + (1 + 4 * np) + (4 * np + self.shape.lam as u32 + np * self.shape.lt) + 3 + (n_kfp + 5 + 1);
        let reads = 3 * self.shape.nas + (n_kfp as u64 + 6) * self.shape.wc_h + (self.shape.wc_o * self.shape.t >> self.shape.la) + (2 * n_kfp as u64 + 5) * (self.shape.wc_q * self.shape.t >> self.shape.la) + 3;
        self.reads = reads;
        self.n_lin = main_lins + base + digit_lins + 4 + 5 + reads as u32;
        self.n_msg = slot;
        f.finish_slots(slot);
        f.plan_claims = self.n_lin;
        self.init_trees(f);
        f.theta = tr.batch_open()?;
        Ok(())
    }
    pub fn smallness(&mut self, digits: &mut Forest, cf_f: &[i8], f: &mut HFork) -> Result<(), FpError> {
        for row in &digits.rows { f.lin(row.clone()); }
        self.lin_join9(digits, f)?;
        self.lin_bits9(digits, f);
        let mut vt = Vec::with_capacity(128);
        vt.push(E::ONE);
        for i in 1..128 { vt.push(vt[i - 1] * f.theta); }
        let vp = vt.clone();
        let step = self.lead(f, 128);
        self.vt = vt.clone(); self.vp = vp.clone();
        let matrices = self.matrices();
        let mut eq_first = vec![0u64];
        let mut row_x = Vec::new();
        let mut lo_x = Vec::new();
        for &(m, _) in &matrices {
            let ms = (m + 2) / 3;
            eq_first.push(eq_first[eq_first.len() - 1] + (128 << crate::fp::tensor::ceil_log2(ms)));
            let mut row = Vec::with_capacity(ms as usize);
            row.push(E::ONE);
            for i in 1..ms { row.push(row[i as usize - 1] * step); }
            row_x.push(row);
            lo_x.push(vec![E::ONE]);
        }
        let cf_first: Vec<u64> = matrices.iter().scan(0, |at, &(_, k)| { let out = *at; *at += k; Some(out) }).collect();
        let mut gam_x = Vec::new(); let mut gam_v = Vec::new();
        for (mu, &(_, k)) in matrices.iter().enumerate() {
            let mut gx = vec![E::ZERO; k as usize]; let mut gv = vec![E::ZERO; k as usize];
            for c in 0..k as usize {
                let start = ((cf_first[mu] + c as u64) * 128) as usize;
                let cf = &cf_f[start..start + 128];
                for (u, &trit) in cf.iter().enumerate() {
                    if trit != 0 { let value = if trit < 0 { -vt[u] } else { vt[u] }; gx[c] += value; gv[c] += value; }
                }
            }
            gam_x.push(gx); gam_v.push(gv);
        }
        let sx = vp.iter().fold(E::ZERO, |a, &b| a + b);
        let sv = vt.iter().fold(E::ZERO, |a, &b| a + b);
        let mz = self.mz();
        self.row_x = row_x.clone();
        for (mu, &(m, k)) in matrices.iter().enumerate() {
            let shape = crate::fp::smallness::Shape { m, k, ms: (m + 2) / 3, j: 1 };
            let wl = crate::fp::smallness::tie_left(&shape, &row_x[mu], &lo_x[mu]);
            let wr = crate::fp::smallness::tie_right(&shape, &lo_x[mu], &gam_x[mu]);
            let lead = self.lead(f, eq_first[mu]);
            let sw = wl.iter().fold(E::ZERO, |a, &b| a + b);
            self.t2lin(f, Aff::constant(-lead * shift(sx * sw, 17)), &[
                Run2 { tree: self.zz_tree(), row: 3, first: self.off(mu), coef: lead, weights: wl.clone(), w0: 0, count: m },
                Run2 { tree: self.zz_tree(), row: 3, first: mz + self.off(mu), coef: shift(lead, 10), weights: wl.clone(), w0: 0, count: m },
                Run2 { tree: self.gt_tree(mu), row: 0, first: 0, coef: -lead, weights: wr, w0: 0, count: k },
            ]);
        }
        let mut rd_t = vec![0u8; self.shape.n_w as usize];
        let mut n_x4 = 0u64;
        for (mu, &(m, k)) in matrices.iter().enumerate() {
            let gc = &gam_v[mu];
            for fun in &self.tables {
                if fun.matrix != mu || fun.l == 0 || m % fun.l != 0 || fun.phi.len() != fun.l as usize { continue; }
                let nsg = m / fun.l;
                let sphi = fun.phi.iter().fold(E::ZERO, |a, &b| a + b);
                for sg in 0..nsg {
                    let lead = self.lead(f, eq_first[matrices.len()] + n_x4 * 128);
                    let mut form = Aff::constant(-lead * shift(sphi * sv, 17));
                    for c in 0..k {
                        let at = fun.at + c * nsg + sg;
                        if at as usize >= rd_t.len() || rd_t[at as usize] != 0 || gc[c as usize].is_zero() { return Err(FpError::Range { context: "attention-smallness-read" }); }
                        rd_t[at as usize] = 1;
                        form.add(&Aff::entry(0, 0, at), -lead * gc[c as usize]);
                    }
                    self.t2lin(f, form, &[
                        Run2 { tree: self.zz_tree(), row: 2, first: self.off(mu) + sg * fun.l, coef: lead, weights: fun.phi.clone(), w0: 0, count: fun.l },
                        Run2 { tree: self.zz_tree(), row: 2, first: mz + self.off(mu) + sg * fun.l, coef: shift(lead, 10), weights: fun.phi.clone(), w0: 0, count: fun.l },
                    ]);
                    n_x4 += 1;
                }
            }
        }
        if n_x4 != self.reads || rd_t.iter().any(|&x| x == 0) { return Err(FpError::Range { context: "attention-smallness-read" }); }
        Ok(())
    }

    pub fn coefficients(&self, digits: &Forest, f: &mut HFork) {
        f.trees[0].rows[0].coef = vec![E::ONE];
        f.trees[0].rows[0].row_ones = true;
        f.trees[0].rows[0].row_form = true;
        let zz = self.zz_tree() as usize;
        f.trees[zz].rows[2].coef = self.vt.clone();
        f.trees[zz].rows[2].row_form = true;
        f.trees[zz].rows[2].row_lo = vec![E::ONE; 7];
        f.trees[zz].rows[2].row_hi = (0..7).map(|i| { let mut value = f.theta; for _ in 0..i { value *= value; } value }).collect();
        f.trees[zz].rows[3].coef = self.vp.clone();
        f.trees[zz].rows[3].row_form = true;
        f.trees[zz].rows[3].row_lo = f.trees[zz].rows[2].row_lo.clone();
        f.trees[zz].rows[3].row_hi = f.trees[zz].rows[2].row_hi.clone();
        for q in 0..2 {
            let call = &digits.calls[q];
            f.trees[zz].rows[q].coef = eq_table(&call.zk[..7]);
            f.trees[zz].rows[q].row_form = true;
            f.trees[zz].rows[q].row_point = call.zk[..7].to_vec();
        }
        let bz = self.bz_tree() as usize;
        for e in 0..2 {
            f.trees[bz].rows[e].coef = eq_table(&digits.zb[e][..5]);
            f.trees[bz].rows[e].row_form = true;
            f.trees[bz].rows[e].row_point = digits.zb[e][..5].to_vec();
        }
        f.trees[bz].rows[2].coef = vec![E::ONE; 32];
        f.trees[bz].rows[2].row_ones = true;
        f.trees[bz].rows[2].row_form = true;
        for mu in 0..5 {
            let tree = self.gt_tree(mu) as usize;
            f.trees[tree].rows[0].coef = self.row_x[mu].clone();
            f.trees[tree].rows[0].row_form = true;
            let bits = crate::fp::tensor::ceil_log2(f.trees[tree].r);
            f.trees[tree].rows[0].row_lo = vec![E::ONE; bits as usize];
            f.trees[tree].rows[0].row_hi = (0..bits).map(|i| { let mut value = self.lead(f, 128); for _ in 0..i { value *= value; } value }).collect();
        }
    }
    fn array_of(&self, link: usize) -> Vec<(usize, String)> {
        let mut out = vec![(link, "U3".into()), (link, "M0".into()), (link, "M1".into()), (link, "M2".into()), (link, "M3".into()), (link, "ZZ".into()), (link, "BZ".into())];
        out.push((link - 1, "M7".into()));
        out
    }
    fn row_weights(&self, tree: &crate::fp::model::HTree, row: &crate::fp::model::HRow) -> Result<crate::fp::req_rows::IndexWeights, FpError> {
        let bits = crate::fp::tensor::ceil_log2(tree.r);
        let np = row.row_point.len() as u32;
        let given = !row.row_lo.is_empty() || !row.row_hi.is_empty();
        if np > bits || (row.row_ones || given) && (np != 0 || row.row_first != 0) || row.row_first & ((1u64 << np) - 1) != 0 || row.row_first >= tree.r || given && (row.row_ones || row.row_lo.len() != bits as usize || row.row_hi.len() != bits as usize) {
            return Err(FpError::Range { context: "row-form" });
        }
        let mut out = crate::fp::req_rows::IndexWeights::default();
        for t in 0..bits {
            if given { out.lo.push(row.row_lo[t as usize]); out.hi.push(row.row_hi[t as usize]); }
            else if row.row_ones { out.lo.push(E::ONE); out.hi.push(E::ONE); }
            else if t < np { let z = row.row_point[(np - 1 - t) as usize]; out.lo.push(E::ONE - z); out.hi.push(z); }
            else { let one = (row.row_first >> t) & 1 != 0; out.lo.push(if one { E::ZERO } else { E::ONE }); out.hi.push(if one { E::ONE } else { E::ZERO }); }
        }
        Ok(out)
    }
    pub fn door_h(&self, link: usize, f: &HFork) -> Result<crate::fp::req_rows::ReqEquation, FpError> {
        if f.n_claim != f.plan_claims { return Err(FpError::Range { context: "request-claim-count" }); }
        let array_of = self.array_of(link);
        let mut equation = crate::fp::req_rows::ReqEquation { terms: vec![], on_masks: f.ls.clone(), target: -f.c0 };
        for (ti, tree) in f.trees.iter().enumerate() {
            let (born, array) = &array_of[ti];
            for row in &tree.rows {
                if row.coef.len() != tree.r as usize || row.l.len() != tree.k as usize { return Err(FpError::Range { context: "request-row-form" }); }
                if row.row_form {
                    equation.terms.push(crate::fp::req_rows::ReqTerm { born: *born, array: array.clone(), scale: E::ONE, on_rows: self.row_weights(tree, row)?, on_columns: crate::fp::req_rows::IndexWeights::written(row.l.clone()) });
                } else {
                    for (i, &coefficient) in row.coef.iter().enumerate() {
                        if coefficient.is_zero() { continue; }
                        let mut on_rows = crate::fp::req_rows::IndexWeights::default();
                        for b in 0..crate::fp::tensor::ceil_log2(tree.r) {
                            let one = (i >> b) & 1 != 0;
                            on_rows.lo.push(if one { E::ZERO } else { E::ONE });
                            on_rows.hi.push(if one { E::ONE } else { E::ZERO });
                        }
                        equation.terms.push(crate::fp::req_rows::ReqTerm { born: *born, array: array.clone(), scale: coefficient, on_rows, on_columns: crate::fp::req_rows::IndexWeights::written(row.l.clone()) });
                    }
                }
            }
        }
        Ok(equation)
    }
}
