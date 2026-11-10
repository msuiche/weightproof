//! Output-head verification (`vf_output_head_check.inc`).
//!
//! The accepted example uses five-limb residual rows (`xl 5`), T = 64 token
//! positions, d = 512 channels, and the exact-match acceptance policy.

use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
    forest::{self, Forest, ProductLevel},
    layer::{HFun, LayerFixture, LayerRun, TMat, Tables},
    layer_checks::{led_lines5, Row5, WeightMerge},
    model::{self, psc_row, Aff, HFork, HStack, RoundName, TokenSumcheckState, Tr},
    subproof::HeadRounds,
    tensor::eq_table,
};

const D: u64 = 512;
const S2: u64 = 4;
const T: u64 = 64;
const LA: u32 = 8;
const LS: u32 = 2;
const JV: u64 = 512;
const JVA: u64 = 256;
const O_A_W: u64 = 0;
const O_RA: u64 = 1024;
const O_D1: u64 = 2048;
const O_D2: u64 = 3072;
const O_E: u64 = 3584;
const EXC: u64 = 3;
const BIGH: u64 = 1 << 37;
const LEDG: [(u64, u64); 6] = [(0, 5), (5, 1), (6, 5), (11, 5), (16, 2), (18, EXC)];

fn eq_top(point: &[E]) -> Vec<E> {
    crate::fp::tensor::eq_table_rev(point)
}
const LED_NR: u64 = 99;

fn pw(bits: u32) -> E {
    E::from_base(F(1u64 << bits))
}

fn last(point: &[E], count: usize) -> Vec<E> {
    point[point.len() - count..].to_vec()
}

fn order_default(mu: u32) -> Vec<u32> {
    (0..mu).rev().collect()
}

fn sum(values: &[E]) -> E {
    values.iter().fold(E::ZERO, |a, &b| a + b)
}

fn long_phi(point: &[E], ls: u32, lt: u32) -> Vec<E> {
    let low = eq_top(&point[..ls as usize]);
    let high = eq_top(&point[ls as usize..(ls + lt) as usize]);
    let mut out = vec![E::ZERO; low.len() * high.len()];
    for (c2, &a) in low.iter().enumerate() {
        for (t, &b) in high.iter().enumerate() {
            out[c2 * high.len() + t] = a * b;
        }
    }
    out
}

fn seg(name: &str, n: u64) -> crate::fp::layer::Seg {
    crate::fp::layer::Seg {
        name: name.into(),
        n,
        moe: false,
        pos: vec![],
    }
}

fn calls() -> Vec<crate::fp::layer::CallD> {
    vec![
        crate::fp::layer::CallD::new_with_floor(
            "R16a",
            "R16",
            vec![seg("NF", 512), seg("NFH", 512), seg("HD0", 512)],
            14,
        ),
        crate::fp::layer::CallD::new_with_floor("R16b", "R16", vec![seg("HD1", 512)], 14),
        crate::fp::layer::CallD::new_with_floor(
            "R8",
            "R8",
            vec![seg("ROW", 120), seg("HD2", 512)],
            14,
        ),
        crate::fp::layer::CallD::new_with_floor("FEa", "FE", vec![seg("HE0", 256)], 14),
        crate::fp::layer::CallD::new_with_floor("FEb", "FE", vec![seg("HE1", 256)], 14),
        crate::fp::layer::CallD::new_with_floor("FC", "FC", vec![seg("CNF", 512)], 14),
    ]
}

fn mats() -> Vec<TMat> {
    let mat = |name: &str, m: u64, k: u64| TMat {
        name: name.into(),
        m,
        k,
        kind: 0,
    };
    vec![
        mat("hf", 16384, 20),
        mat("hd", 4096, 16),
        mat("hr", 4096, 2),
        mat("bH", 8192, 1536),
        mat("xbL", 8192, 20),
    ]
}

pub struct HeadRun {
    pub run: LayerRun,
    pub lin: crate::fp::key_public::LinShape,
    pub t_n: TokenSumcheckState,
    pub t_h: TokenSumcheckState,
    pub stack: HStack,
    pub s_v: u32,
    pub merged: Option<WeightMerge>,
    pub on: Vec<bool>,
    pub on_c: Vec<bool>,
    pub win: Vec<u32>,
    pub erc: E,
    pub vt: Vec<E>,
    pub vp: Vec<E>,
    pub row_x: Vec<Vec<E>>,
}

impl HeadRun {
    fn call_index(&self, name: &str) -> usize {
        self.run
            .fixture
            .calls
            .iter()
            .position(|call| call.seg(name).is_some())
            .unwrap_or_else(|| panic!("unknown head segment {name}"))
    }

    fn zj(&self, call: usize) -> &[E] {
        &self.run.forest.calls[call].zk[..self.run.fixture.calls[call].l_j as usize]
    }

    fn zt(&self, call: usize) -> &[E] {
        &self.run.forest.calls[call].zk[self.run.fixture.calls[call].l_j as usize..]
    }

    fn table_at(&self, name: &str) -> Result<u64, FpError> {
        self.run.tables.at(name)
    }

    fn u3at(&self, offset: u64, index: u64) -> Result<Aff, FpError> {
        self.run.tables.entry(0, offset, index)
    }

    fn ew_of(&self, segment: &str) -> Vec<E> {
        let call = self.call_index(segment);
        let c = &self.run.fixture.calls[call];
        let e = eq_top(self.zj(call));
        let s = c.seg(segment).unwrap();
        s.pos.iter().map(|&position| e[position as usize]).collect()
    }

    fn ewh(&self, segment: &str) -> Vec<E> {
        let call = self.call_index(segment);
        let c = &self.run.fixture.calls[call];
        let z = self.zj(call);
        let e = eq_top(&z[..z.len() - LS as usize]);
        let s = c.seg(segment).unwrap();
        (0..s.n / S2)
            .map(|xh| e[s.pos[(S2 * xh) as usize] as usize / S2 as usize])
            .collect()
    }

    fn on_sum(&self, point: &[E]) -> E {
        let e = eq_top(point);
        self.on
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .fold(E::ZERO, |acc, (t, _)| acc + e[t])
    }

    fn place_term(&self, ew: &[E], et: &[E]) -> E {
        let mut pre = vec![E::ZERO; ew.len()];
        for v in 1..ew.len() {
            pre[v] = pre[v - 1] + ew[v - 1];
        }
        self.on
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .fold(E::ZERO, |acc, (t, _)| {
                acc + et[t] * pre[self.win[t] as usize]
            })
    }

    fn row5_at(&self, point: &str, i: u64) -> Result<Row5, FpError> {
        let offset = self.table_at(&format!("hr@{point}"))?;
        let mut digits = Vec::with_capacity(99);
        for j in 0..99 {
            digits.push(self.u3at(offset, i * 99 + j)? + Aff::constant(pw(7)));
        }
        let dig = |at: usize, count: usize| {
            let mut out = Aff::default();
            for j in 0..count {
                out.add(&digits[at + j], pw(8 * j as u32));
            }
            out
        };
        let mut c = vec![Aff::default()];
        for s in 1..=8 {
            c.push(dig(10 + 6 * (s - 1), 6) - Aff::constant(pw(47)));
        }
        c.push(digits[9].clone());
        Ok(Row5 {
            digits: digits.clone(),
            c,
            v_lo: dig(0, 6),
            v_hi: dig(6, 4),
            a: dig(58, 1).scaled(pw(1))
                + dig(59, 1)
                + dig(60, 1).scaled(pw(8))
                + dig(61, 1).scaled(pw(16)),
            b: dig(62, 1) + dig(63, 1).scaled(pw(8)),
            sig: dig(58, 1).scaled(pw(1))
                + dig(59, 1)
                + dig(60, 1).scaled(pw(8))
                + dig(61, 1).scaled(pw(16))
                + dig(62, 1).scaled(pw(24))
                + dig(63, 1).scaled(pw(32)),
            s0: dig(59, 1),
            l1: dig(64, 5),
            h1: dig(69, 5),
            b_lo: dig(74, 3),
            b_hi: dig(77, 2),
            kap: dig(79, 1) - Aff::constant(E::from_base(F(4))),
            r_lo: dig(80, 3),
            r_hi: dig(83, 3),
            r: dig(80, 3) + dig(83, 3).scaled(pw(24)),
            q1: dig(86, 3),
            rr: dig(89, 5),
            rq: dig(94, 5),
        })
    }

    fn gap_at(&self, point: &str, i: usize, wsum: E) -> Result<Aff, FpError> {
        let offset = self.table_at(&format!("hr@{point}"))?;
        let (first, count) = LEDG[i];
        let mut out = Aff::default();
        for l in 0..count {
            out.add(
                &(self.u3at(offset, LED_NR + first + l)? + Aff::constant(wsum * pw(7))),
                pw(8 * l as u32),
            );
        }
        Ok(out)
    }

    fn sl(&self, segment: &str, off: u64, nl: u64, j: u64) -> Result<Aff, FpError> {
        let wh = self.ewh(segment);
        let t = self.table_at("hf@Z")?;
        let mut out = Aff::default();
        for (xh, &weight) in wh.iter().enumerate() {
            for l in 0..nl {
                out.add(
                    &self.u3at(t, (off + l * j) / S2 + xh as u64)?,
                    weight * pw(8 * l as u32),
                );
            }
        }
        Ok(out)
    }

    fn xb_a(&self, offset: u64, kh: u64) -> Result<Aff, FpError> {
        let mut out = Aff::default();
        for limb in 0..5u64 {
            out.add(&self.u3at(offset, limb * D / S2 + kh)?, pw(8 * limb as u32));
        }
        Ok(out)
    }

    fn sh2(&self, offset: u64, j: u64) -> Result<Aff, FpError> {
        Ok(self.u3at(offset, j)? + self.u3at(offset, D + j)?.scaled(pw(8)))
    }

    fn line(&self, segment: &str) -> Result<Aff, FpError> {
        let ew = self.ew_of(segment);
        let w = sum(&ew);
        match segment {
            "NF" => Ok(
                Aff::constant(w * (pw(29) - pw(16) * E::from_base(F(32896))))
                    - self.sl(segment, O_A_W, 2, D)?.scaled(pw(30))
                    - self.sl(segment, O_RA, 2, D)?.scaled(pw(16)),
            ),
            "NFH" => Ok((Aff::constant(E::from_base(F(32896)) * w)
                + self.sl(segment, O_RA, 2, D)?)
            .scaled(pw(2))),
            "HD0" | "HD1" | "HD2" => {
                let wx = sum(&ew);
                let w_r = wx * self.on_sum(self.zt(self.call_index(segment)));
                if segment == "HD1" {
                    return Ok(Aff::constant(E::from_base(F(32896)) * w_r)
                        + self.sl(segment, O_D1, 2, JV)?);
                }
                if segment == "HD2" {
                    return Ok(Aff::constant(pw(7) * w_r) + self.sl(segment, O_D2, 1, JV)?);
                }
                let et = eq_top(self.zt(self.call_index("HD0")));
                let iw = pw(23).inverse();
                Ok((Aff::slot(self.t_h.s_cl).scaled(pw(9))
                    + (self.gap_at("phiC", 0, self.erc)?
                        + self
                            .gap_at("phiC", 1, self.erc)?
                            .scaled(E::from_base(F(BIGH))))
                    .scaled(wx)
                    - Aff::constant(self.place_term(&ew, &et))
                    - (Aff::constant(E::from_base(F(32896)) * w_r)
                        + self.sl(segment, O_D1, 2, JV)?)
                    .scaled(pw(8))
                    - (Aff::constant(pw(7) * w_r) + self.sl(segment, O_D2, 1, JV)?))
                .scaled(iw))
            }
            "HE0" | "HE1" => {
                let v0 = if segment == "HE1" { JVA } else { 0 };
                let wx = ew
                    .iter()
                    .take((JV - v0) as usize)
                    .fold(E::ZERO, |acc, &value| acc + value);
                let off_e = E::from_base(F(128 * (((1u64 << (8 * EXC)) - 1) / 255) + 1));
                Ok(Aff::constant(
                    E::from_base(F(32896)) * wx * self.on_sum(self.zt(self.call_index(segment))),
                ) + self.sl(segment, O_D1 + v0, 2, JV)?
                    + (Aff::constant(w * off_e) + self.sl(segment, O_E + v0, EXC, JV)?)
                        .scaled(pw(16)))
            }
            "ROW" => {
                let t = self.table_at("hr@zt_ROW")?;
                let mut out = Aff::default();
                for (idx, &weight) in ew.iter().enumerate() {
                    let mu = if idx < LED_NR as usize {
                        match idx {
                            9 => 4,
                            58 => 2,
                            59 => 128,
                            63 => 2,
                            76 => 2,
                            79 => 32,
                            _ => 1,
                        }
                    } else {
                        1
                    };
                    out.add(
                        &(self.u3at(t, idx as u64)? + Aff::constant(pw(7))),
                        weight * E::from_base(F(mu)),
                    );
                }
                Ok(out)
            }
            "CNF" => {
                let t = self.table_at("hd@zt_FC")?;
                let mut b = Aff::default();
                for x in 0..D {
                    b.add(&self.sh2(t, x)?, ew[x as usize]);
                }
                Ok(self.sl(segment, O_A_W, 2, D)?
                    + b.scaled(pw(28))
                    + Aff::constant(w * pw(15) + w * pw(39)))
            }
            _ => unreachable!(),
        }
    }

    fn key_form(&self, q: usize, laid: &mut [u64]) -> Result<Aff, FpError> {
        let c = &self.run.fixture.calls[q];
        let e = eq_top(self.zj(q));
        let mut weighed = vec![0u8; c.j as usize];
        let mut lines = Aff::default();
        for segment in &c.segs {
            lines = lines + self.line(&segment.name)?;
            let entries = matches!(segment.name.as_str(), "HD0" | "HD1" | "HD2" | "HE0" | "HE1");
            let v0 = if segment.name == "HE1" { JVA } else { 0 };
            for (x, &position) in segment.pos.iter().enumerate() {
                if weighed[position as usize] != 0 {
                    return Err(FpError::Range {
                        context: "head-join-place",
                    });
                }
                weighed[position as usize] = if entries && v0 + x as u64 >= 512 {
                    2
                } else {
                    1
                };
            }
        }
        let mut pad_w = E::ZERO;
        #[cfg(feature = "std")]
        {
            let (n0, n1, n2) =
                weighed
                    .iter()
                    .fold((0usize, 0usize, 0usize), |(a, b, c), &x| match x {
                        0 => (a + 1, b, c),
                        1 => (a, b + 1, c),
                        _ => (a, b, c + 1),
                    });
            eprintln!(
                "RUST_USED q={} used0={} used1={} used2={} j={}",
                q,
                n0,
                n1,
                n2,
                weighed.len()
            );
        }
        for (j, &weight) in e.iter().enumerate() {
            if weighed[j] == 0 {
                pad_w += weight;
            }
            if weighed[j] != 1 {
                let at = (self.run.forest.first[q] as usize
                    + (j >> (self.run.forest.d_t - 6)) as usize) as usize;
                if at < laid.len() {
                    laid[at] += T;
                }
            }
        }
        let mut out =
            Aff::constant(pad_w * self.run.forest.calls[q].tau_at(self.run.forest.calls[q].t0))
                + lines;
        if q == self.call_index("NF") {
            out = out + Aff::slot(self.t_n.s_cl);
        }
        Ok(out)
    }

    fn forest_join(&self, keys: &[Aff]) -> Aff {
        let mut out = Aff::default();
        for (q, key) in keys.iter().enumerate() {
            out.add(
                &(Aff::constant(
                    self.run.forest.alpha - self.run.forest.theta * E::from_base(F(q as u64)),
                ) - key.clone()),
                self.run.forest.e[q],
            );
        }
        let ey = eq_top(&self.run.forest.zy[..self.run.forest.r_k as usize]);
        for y in self.run.forest.nt as usize..1usize << self.run.forest.r_k {
            out.c += ey[y] * (self.run.forest.alpha - self.run.forest.padding_key(0));
        }
        out.minus(&self.run.forest.c_end)
    }

    pub fn read_messages(
        f: &mut HFork,
        tr: &mut Tr,
        slot: &mut u32,
        shape: HeadRounds,
        lin: crate::fp::key_public::LinShape,
        forest: Forest,
        on: Vec<bool>,
        on_c: Vec<bool>,
        win: Vec<u32>,
    ) -> Result<Self, FpError> {
        let fixture = LayerFixture {
            first: false,
            eps: [0, 0],
            ns: 2879,
            ncl: 1,
            nlin: 897,
            n_batched: 0,
            lam_s: shape.l_pd,
            n_u3: 11264,
            calls: calls(),
            tensors: vec![],
            mats: mats(),
        };
        let pv = tr.point(RoundName::Zero, shape.lt as usize)?;
        let half = E::from_base(F(2)).inverse();
        let plain_pd = (0..shape.l_pd).fold(E::ONE, |acc, _| acc * half);
        let s_w = f.claims_in(tr, slot, shape.sums as u32)?;
        let gp = tr.batch(shape.sums as usize)?;
        let mut start = Aff::default();
        for (i, &power) in gp.iter().enumerate() {
            start.add(&Aff::slot(s_w + i as u32), power * plain_pd);
        }
        let mut rho = vec![half; shape.l_pd as usize];
        rho.extend_from_slice(&pv);
        let wpl = forest::read_product_level(
            f,
            tr,
            slot,
            shape.l_pd + shape.lt,
            &rho,
            &mut start,
            &order_default(shape.l_pd + shape.lt),
        )?;
        let s_pv = f.claims_in(tr, slot, shape.products as u32)?;
        let om_v = tr.batch(shape.products as usize)?;
        let mut t_v = model::tsc_in(f, tr, slot, &pv, shape.row_values as u32, 2)?;
        let sh = |j: u32| -> Result<E, FpError> { f.shown(t_v.s_end + j) };
        t_v.row_u = psc_row(
            f,
            &mut t_v.p,
            om_v[0] * sh(0)? * sh(0)?
                + om_v[1] * sh(0)? * sh(1)?
                + om_v[2] * sh(1)? * sh(1)?
                + om_v[3] * sh(2)? * sh(0)?
                + om_v[4] * sh(2)? * sh(1)?
                + om_v[5] * sh(3)? * sh(0)?
                + om_v[6] * sh(3)? * sh(1)?,
            &Aff::default(),
        )?;
        let zt_n = {
            let q = 0;
            let zk = &forest.calls[q].zk;
            zk[14..].to_vec()
        };
        let mut t_n = model::tsc_in(f, tr, slot, &zt_n[..shape.lt as usize], 2, 2)?;
        t_n.row_u = psc_row(
            f,
            &mut t_n.p,
            f.shown(t_n.s_end)? * f.shown(t_n.s_end + 1)?,
            &Aff::default(),
        )?;
        let zt_h = {
            let q = 0;
            let zk = &forest.calls[q].zk;
            zk[14..].to_vec()
        };
        let mut t_h = model::tsc_rounds(f, tr, slot, &zt_h[..shape.lt as usize])?;
        let mut stack = HStack {
            lam: shape.l_pd,
            rs: vec![],
            p: t_h.p.clone(),
        };
        for _ in 0..shape.l_pd {
            stack.rs.push(model::round_in(
                f,
                tr,
                slot,
                model::RoundForm::Eq2,
                &mut t_h.p,
                half,
                0,
            )?);
        }
        stack.rs.reverse();
        stack.p = t_h.p.clone();
        t_h.s_end = model::psc_ends(f, tr, slot, &mut t_h.p, 2, 2)?;
        t_h.row_u = psc_row(
            f,
            &mut t_h.p,
            f.shown(t_h.s_end)? * f.shown(t_h.s_end + 1)?,
            &Aff::default(),
        )?;
        let s_v = t_h.s_end + 1;
        let zl = last(&forest.zx, LA as usize);
        let zv = last(&wpl.r, LA as usize);
        let yn = {
            let c = &forest.calls[0];
            let z = &c.zk[..fixture.calls[0].l_j as usize];
            let mut point = z[z.len() - LS as usize..].to_vec();
            point.extend_from_slice(&t_n.r);
            point
        };
        if yn.len() != LA as usize {
            return Err(FpError::Range {
                context: "head-tensor-layout",
            });
        }
        let mut tables = Tables {
            fun: vec![],
            later_phi: vec![],
            offsets: BTreeMap::new(),
            lengths: BTreeMap::new(),
        };
        let mut position = 0u64;
        let mut add =
            |matrix: usize, length: u64, phi: Vec<E>, name: String| -> Result<(), FpError> {
                let m = &fixture.mats[matrix];
                let first = position;
                tables.fun.push(HFun {
                    matrix,
                    l: length,
                    phi,
                    at: position,
                });
                position += m.k * (m.m / length);
                if position == first {
                    return Err(FpError::Range {
                        context: "head-table-length",
                    });
                }
                tables.offsets.insert(name.clone(), first);
                tables.lengths.insert(first, position - first);
                Ok(())
            };
        add(0, 1 << LA, long_phi(&zl, LS, shape.lt), "hf@Z".into())?;
        let mut phi_c = eq_top(&zt_h[..shape.lt as usize]);
        let mut rho_cv = vec![E::ZERO; T as usize];
        let mut erc = E::ZERO;
        for t in 0..T as usize {
            if on_c[t] {
                erc += phi_c[t];
                rho_cv[t] = E::ONE;
            } else {
                phi_c[t] = E::ZERO;
            }
        }
        let phi_z: Vec<E> = (0..1 << LA).map(|i| phi_c[i & (T as usize - 1)]).collect();
        add(0, 1 << LA, phi_z, "hf@phiZ".into())?;
        add(4, 1 << LA, long_phi(&zv, LS, shape.lt), "xbL@Z_V".into())?;
        add(4, 1 << LA, long_phi(&yn, LS, shape.lt), "xbL@Y_N".into())?;
        let zt_fc = {
            let q = 5;
            let zk = &forest.calls[q].zk;
            zk[14..].to_vec()
        };
        add(1, T, eq_top(&zt_fc), "hd@zt_FC".into())?;
        add(1, T, eq_top(&t_h.r), "hd@r_H".into())?;
        let zt_row = {
            let q = 2;
            let zk = &forest.calls[q].zk;
            zk[14..].to_vec()
        };
        add(2, T, eq_top(&zt_row), "hr@zt_ROW".into())?;
        add(2, T, eq_top(&pv), "hr@P_V".into())?;
        add(2, T, eq_top(&t_v.r), "hr@r_tV".into())?;
        add(2, T, eq_top(&t_n.r), "hr@rt_N".into())?;
        add(2, T, phi_c, "hr@phiC".into())?;
        add(2, T, rho_cv, "hr@rhoC".into())?;
        add(
            3,
            1 << 13,
            forest::forest_phi_b(&forest, 13, 0),
            "bH-bits-0".into(),
        )?;
        add(
            3,
            1 << 13,
            forest::forest_phi_b(&forest, 13, 1),
            "bH-bits-1".into(),
        )?;
        add(3, 1 << 13, forest::forest_phi_p(13), "bH-pin".into())?;
        if position != fixture.n_u3 {
            return Err(FpError::Range {
                context: "head-table-length",
            });
        }
        Ok(Self {
            run: LayerRun {
                fixture,
                forest,
                tables,
                wpl,
                s_w,
                gp,
                s_pv,
                om_v,
                t_v,
                t_t: t_h.clone(),
                kt: Aff::default(),
                tkh: vec![],
                points: BTreeMap::new(),
                tokv: vec![],
                s_sc: 0,
                s_sel: 0,
            },
            lin,
            t_n,
            t_h,
            stack,
            s_v,
            merged: None,
            on,
            on_c,
            win,
            erc,
            vt: vec![],
            vp: vec![],
            row_x: vec![],
        })
    }

    pub fn prepare(&mut self, f: &mut HFork, tr: &mut Tr, slot: &mut u32) -> Result<(), FpError> {
        let rho = tr.batch(1)?;
        let start = Aff::slot(self.s_v);
        let mut p = model::psc_begin(f, tr, slot, start)?;
        let mut rm = Vec::with_capacity(11);
        for _ in 0..11 {
            rm.push(model::round_in(
                f,
                tr,
                slot,
                model::RoundForm::Sum2,
                &mut p,
                E::ZERO,
                0,
            )?);
        }
        let s_fv = model::psc_ends(f, tr, slot, &mut p, 1, 1)?;
        let fvp = f.shown(s_fv)?;
        let beta = p.r[p.r.len() - 1];
        let merged = WeightMerge {
            fvp,
            beta,
            rm,
            p,
            s_fv,
            s_ve: 0,
            rho,
            rec: vec![],
        };
        if *slot != 2879 {
            return Err(FpError::Range {
                context: "head-slot-count",
            });
        }
        self.merged = Some(merged);
        let lay = crate::fp::weights::ws_lay();
        let ew = self.ew_of("HD0");
        let wx = sum(&ew);
        let e_h = eq_top(&self.t_h.r);
        let s_r = self.on_sum(&self.t_h.r);
        let mut av = vec![E::ZERO; 512];
        for (v, &coefficient) in ew.iter().enumerate() {
            av[v] = -s_r * coefficient;
        }
        for t in 0..T as usize {
            if self.on[t] {
                let index = self.win[t] as usize;
                if index >= av.len() {
                    return Err(FpError::Range {
                        context: "header-y",
                    });
                }
                av[index] += wx * e_h[t];
            }
        }
        let pt: Vec<E> = (0..9).map(|i| self.stack.rs[8 - i]).collect();
        let hv = model::eq_low(&self.stack.rs, 9, 9);
        let ui = lay
            .u
            .iter()
            .position(|unit| unit.x == 13)
            .filter(|&ui| ui + 1 == lay.u.len())
            .ok_or(FpError::Range {
                context: "head-claim-unit",
            })? as u32;
        #[cfg(feature = "std")]
        eprintln!("RUST_HEAD_AV av={:?} pt={:?}", &av[..6], &pt);
        let claim = crate::fp::weights::WsClaim {
            ui,
            f: vec![
                crate::fp::weights::WsFac {
                    hv: vec![],
                    pt: vec![],
                    av,
                },
                crate::fp::weights::WsFac { hv, pt, av: vec![] },
            ],
        };
        #[cfg(feature = "std")]
        eprintln!(
            "RUST_HEAD_CLAIM_AV {:?} scalars={:?}",
            &claim.f[0].av[..6],
            claim.f[0]
                .av
                .iter()
                .take(6)
                .map(|x| x.0[0])
                .collect::<Vec<_>>()
        );
        let merged = self.merged.as_mut().unwrap();
        #[cfg(feature = "std")]
        eprintln!(
            "RUST_HEAD_RM {:?}",
            merged.rm.iter().map(|x| x.0[0]).collect::<Vec<_>>()
        );
        merged.rec = vec![crate::fp::weights::ws_row_part(
            &lay,
            &claim,
            merged.rho[0] * crate::fp::weights::ws_eps_t(&lay, &claim, &merged.rm),
        )];
        self.run.fixture.nlin = 897;
        f.plan_claims = 897;
        f.finish_slots(2879);
        f.theta = tr.batch_open()?;
        Ok(())
    }

    pub fn add_main_claims(&mut self, f: &mut HFork) -> Result<(), FpError> {
        crate::fp::layer_checks::LayerCheck::new_head(&mut self.run).init_trees(f);
        for row in &self.run.forest.rows {
            f.lin(row.clone());
        }
        f.lin(self.run.wpl.row.clone());
        f.lin(Aff::slot(self.run.wpl.s_l) - Aff::slot(self.run.wpl.s_r));
        {
            let e = eq_top(&self.run.wpl.r[..self.run.wpl.r.len() - LA as usize]);
            let t = self.table_at("xbL@Z_V")?;
            let mut b = Aff::slot(self.run.wpl.s_l);
            for kh in 0..D / S2 {
                for limb in 0..5u64 {
                    b.add(
                        &self.u3at(t, limb * D / S2 + kh)?,
                        -e[kh as usize] * self.run.gp[limb as usize],
                    );
                }
            }
            f.lin(b);
        }
        let mut batch = Aff::slot(self.run.t_v.s_cl);
        for (i, &power) in self.run.om_v.iter().enumerate() {
            batch.add(&Aff::slot(self.run.s_pv + i as u32), -power);
        }
        f.lin(batch);
        f.lin(self.run.t_v.row_u.clone());
        {
            let r = self.row5_at("r_tV", 0)?;
            f.lin(Aff::slot(self.run.t_v.s_end) - r.a);
            f.lin(Aff::slot(self.run.t_v.s_end + 1) - r.b);
            f.lin(Aff::slot(self.run.t_v.s_end + 2) - r.r_lo);
            f.lin(Aff::slot(self.run.t_v.s_end + 3) - r.r_hi);
            let q = self.row5_at("P_V", 0)?;
            for s in 0..9 {
                f.lin(
                    Aff::slot(self.run.s_w + s as u32) + q.c[s].clone()
                        - q.digits[s].clone()
                        - q.c[s + 1].clone().scaled(pw(8)),
                );
            }
            let products: Vec<Aff> = (0..7)
                .map(|j| Aff::slot(self.run.s_pv + j as u32))
                .collect();
            let two = Aff::constant(pw(44));
            for out in led_lines5(&q, &products, E::ONE, &two) {
                f.lin(out);
            }
        }
        {
            let wh = self.ewh("NF");
            let t = self.table_at("xbL@Y_N")?;
            let mut sx = Aff::default();
            for (kh, &weight) in wh.iter().enumerate() {
                sx.add(&self.xb_a(t, kh as u64)?, weight);
            }
            f.lin(self.t_n.row_u.clone());
            let r = self.row5_at("rt_N", 0)?;
            f.lin(Aff::slot(self.t_n.s_end) - r.r);
            f.lin(Aff::slot(self.t_n.s_end + 1) - sx);
        }
        let mut laid = vec![0u64; self.run.forest.nt as usize];
        let mut keys = Vec::with_capacity(self.run.fixture.calls.len());
        for q in 0..self.run.fixture.calls.len() {
            keys.push(self.key_form(q, &mut laid)?);
        }
        if laid != self.run.forest.pub_counts {
            #[cfg(feature = "std")]
            eprintln!(
                "RUST_HEAD_PADDING laid={:?} expected={:?}",
                laid, self.run.forest.pub_counts
            );
            return Err(FpError::Range {
                context: "head-padding",
            });
        }
        f.lin(self.forest_join(&keys));
        let r_c = E::from_base(F(3));
        f.lin(
            self.gap_at("rhoC", 0, r_c)?
                + self.gap_at("rhoC", 1, r_c)?.scaled(E::from_base(F(256)))
                + self.gap_at("rhoC", 2, r_c)?
                - Aff::constant(r_c * E::from_base(F(256))),
        );
        f.lin(self.gap_at("rhoC", 1, r_c)? + self.gap_at("rhoC", 4, r_c)? - Aff::constant(E::ZERO));
        f.lin(
            self.gap_at("phiC", 0, self.erc)? + self.gap_at("phiC", 3, self.erc)?
                - Aff::constant(self.erc * E::from_base(F(256))),
        );
        {
            let te = self.table_at("hf@phiZ")?;
            let off_e = E::from_base(F(128 * (((1u64 << (8 * EXC)) - 1) / 255) + 1));
            let mut a = self.gap_at("phiC", 5, self.erc)?
                + Aff::constant(
                    self.erc * off_e * E::from_base(F(JV))
                        - self.erc * E::from_base(F((JV - 512 + 1) << 24)),
                );
            for l in 0..EXC {
                for xh in 0..JV / S2 {
                    a.add(&self.u3at(te, (O_E + l * JV) / S2 + xh)?, pw(8 * l as u32));
                }
            }
            f.lin(a);
        }
        {
            let t_b = [self.table_at("bH-bits-0")?, self.table_at("bH-bits-1")?];
            let t_p = self.table_at("bH-pin")?;
            let w = 16 + 5 - 13;
            let mut at = [Aff::default(), Aff::default()];
            for e in 0..2 {
                let ec = eq_top(&self.run.forest.zb[e][16..16 + self.run.forest.c_k as usize]);
                let et = eq_top(&self.run.forest.zb[e][..w]);
                for c in 0..self.run.forest.calls.len() {
                    for th in 0..1usize << w {
                        at[e].add(&self.u3at(t_b[e], ((c << w) + th) as u64)?, ec[c] * et[th]);
                    }
                }
            }
            let padding =
                self.run.forest.empty_keys() + self.run.forest.pub_counts.iter().sum::<u64>();
            let mut pin = Aff::constant(-E::from_base(F((1u64 << self.run.forest.v()) - padding)));
            for column in 0..(self.run.forest.calls.len() << w) {
                pin = pin + self.u3at(t_p, column as u64)?;
            }
            let plane_row = psc_row(
                f,
                &mut self.run.forest.leaves,
                E::ONE,
                &at[0].clone().scaled(self.run.forest.leaves_a),
            )?;
            f.lin(plane_row);
            f.lin(Aff::slot(self.run.forest.s_bit) - at[1].clone());
            f.lin(pin);
        }
        Ok(())
    }

    pub fn add_post_smallness_claims(&mut self, f: &mut HFork) -> Result<(), FpError> {
        {
            let eq = crate::fp::tensor::eq_table(&self.stack.rs);
            let t = self.table_at("hd@r_H")?;
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_POST").is_some() {
                let direct = crate::fp::tensor::eq_table(&self.stack.rs);
                eprintln!(
                    "RUST_ALIN_DIRECT eq={:?}",
                    direct.iter().take(8).map(|x| x.0[0]).collect::<Vec<_>>()
                );
                eprintln!(
                    "RUST_ALIN t={} rs={:?} eq={:?}",
                    t,
                    self.stack.rs.iter().map(|x| x.0[0]).collect::<Vec<_>>(),
                    eq.iter().take(8).map(|x| x.0[0]).collect::<Vec<_>>()
                );
            }
            let mut alin = Aff::default();
            for kk in 0..D {
                alin.add(&self.sh2(t, kk)?, eq[kk as usize]);
            }
            f.lin(self.t_h.row_u.clone());
            f.lin(Aff::slot(self.t_h.s_end) - alin);
        }
        let merged = self.merged.as_mut().ok_or(FpError::Range {
            context: "head-merge-state",
        })?;
        let merge_row = psc_row(f, &mut merged.p, merged.fvp, &Aff::default())?;
        f.lin(merge_row);
        Ok(())
    }

    pub fn pending(&self) -> Result<crate::fp::wf::TensorRow, FpError> {
        let merged = self.merged.as_ref().ok_or(FpError::Range {
            context: "head-merge-state",
        })?;
        crate::fp::weights::pending_row(&crate::fp::weights::ws_lay(), merged, 1, &self.lin)
    }

    pub fn smallness(
        &mut self,
        digits: &mut Forest,
        trits: &[i8],
        f: &mut HFork,
    ) -> Result<(), FpError> {
        let mut check = crate::fp::layer_checks::LayerCheck::new_head(&mut self.run);
        check.smallness(digits, trits, f)?;
        self.vt = check.vt.clone();
        self.vp = check.vp.clone();
        self.row_x = check.row_x.clone();
        #[cfg(feature = "std")]
        eprintln!("RUST_HEAD_SMALLNESS_STATE row_x={}", self.row_x.len());
        Ok(())
    }

    pub fn coefficients(&mut self, digits: &Forest, f: &mut HFork) {
        let mut check = crate::fp::layer_checks::LayerCheck::new_head(&mut self.run);
        check.vt = self.vt.clone();
        check.vp = self.vp.clone();
        check.row_x = self.row_x.clone();
        check.coefficients(digits, f)
    }

    fn array_of(&self, link: usize) -> Vec<(usize, String)> {
        vec![
            (link, "U3".into()),
            (link, "M0".into()),
            (link, "M1".into()),
            (link, "M2".into()),
            (link, "M3".into()),
            (2, "M7".into()),
            (link, "ZZ".into()),
            (link, "BZ".into()),
        ]
    }

    pub fn door_h(
        &self,
        link: usize,
        f: &HFork,
    ) -> Result<crate::fp::req_rows::ReqEquation, FpError> {
        if f.n_claim != f.plan_claims {
            return Err(FpError::Range {
                context: "head-claim-count",
            });
        }
        let (on_masks, constant) = f.settle()?;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!(
                "RUST_HEAD_DOOR raw={:016x} settled={:016x} target={:016x}",
                f.c0.0[0],
                constant.0[0],
                (-constant).0[0]
            );
        }
        let array_of = self.array_of(link);
        let mut equation = crate::fp::req_rows::ReqEquation {
            terms: vec![],
            on_masks,
            target: -constant,
        };
        for (ti, tree) in f.trees.iter().enumerate() {
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_DOOR").is_some() && ti == 0 {
                for row in &tree.rows {
                    let hash = row.l.iter().fold(0u64, |hash, value| hash ^ value.0[0]);
                    eprintln!(
                        "RUST_DOOR_ROW_L hash={:016x} first={:016x}",
                        hash,
                        row.l.first().map_or(0, |value| value.0[0])
                    );
                }
            }
            let (born, array) = &array_of[ti];
            for row in &tree.rows {
                if row.coef.len() != tree.r as usize || row.l.len() != tree.k as usize {
                    return Err(FpError::Range {
                        context: "head-row-form",
                    });
                }
                if row.row_form {
                    equation.terms.push(crate::fp::req_rows::ReqTerm {
                        born: *born,
                        array: array.clone(),
                        scale: E::ONE,
                        on_rows: row_weights(tree, row)?,
                        on_columns: crate::fp::req_rows::IndexWeights::written(row.l.clone()),
                    });
                } else {
                    for (i, &coefficient) in row.coef.iter().enumerate() {
                        if coefficient.is_zero() {
                            continue;
                        }
                        let mut on_rows = crate::fp::req_rows::IndexWeights::default();
                        for b in 0..crate::fp::tensor::ceil_log2(tree.r) {
                            let one = (i >> b) & 1 != 0;
                            on_rows.lo.push(if one { E::ZERO } else { E::ONE });
                            on_rows.hi.push(if one { E::ONE } else { E::ZERO });
                        }
                        equation.terms.push(crate::fp::req_rows::ReqTerm {
                            born: *born,
                            array: array.clone(),
                            scale: coefficient,
                            on_rows,
                            on_columns: crate::fp::req_rows::IndexWeights::written(row.l.clone()),
                        });
                    }
                }
            }
        }
        Ok(equation)
    }
}

fn row_weights(
    tree: &crate::fp::model::HTree,
    row: &crate::fp::model::HRow,
) -> Result<crate::fp::req_rows::IndexWeights, FpError> {
    let bits = crate::fp::tensor::ceil_log2(tree.r);
    let np = row.row_point.len() as u32;
    let given = !row.row_lo.is_empty() || !row.row_hi.is_empty();
    if np > bits
        || (row.row_ones || given) && (np != 0 || row.row_first != 0)
        || row.row_first & ((1u64 << np) - 1) != 0
        || row.row_first >= tree.r
        || given
            && (row.row_ones
                || row.row_lo.len() != bits as usize
                || row.row_hi.len() != bits as usize)
    {
        return Err(FpError::Range {
            context: "head-row-form",
        });
    }
    let mut out = crate::fp::req_rows::IndexWeights::default();
    for t in 0..bits {
        if given {
            out.lo.push(row.row_lo[t as usize]);
            out.hi.push(row.row_hi[t as usize]);
        } else if row.row_ones {
            out.lo.push(E::ONE);
            out.hi.push(E::ONE);
        } else if t < np {
            let z = row.row_point[(np - 1 - t) as usize];
            out.lo.push(E::ONE - z);
            out.hi.push(z);
        } else {
            let one = (row.row_first >> t) & 1 != 0;
            out.lo.push(if one { E::ZERO } else { E::ONE });
            out.hi.push(if one { E::ONE } else { E::ZERO });
        }
    }
    Ok(out)
}
