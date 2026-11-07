//! Layer lookup-segment forms and the inner-product stack.

use alloc::{format, string::String, vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
    forest::ProductLevel,
    layer::{LayerRun, Ten},
    model::{psc_begin, psc_row, stack_rounds, Aff, HFork, HStack, RoundName, Tr},
    tensor::eq_table,
};

const D: u64 = 512;
const F_WIDTH: u64 = 1024;
const FE: u64 = 512;
const DQ: u64 = 512;
const NQK: u64 = 768;
const LAM: u64 = 4;
const LAM_M: u64 = 2;
const NE: u64 = 4;
const S2: u64 = 4;
const LT: u32 = 6;
const LS: u32 = 2;
const LA: u32 = 8;
const LFE: u32 = 9;
const LPF: u32 = 10;

fn eq_top(point: &[E]) -> Vec<E> {
    eq_table(&point.iter().rev().copied().collect::<Vec<_>>())
}

fn kl_words() -> Vec<u32> {
    include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin")
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn kl(layer: u64, index: usize) -> u32 {
    kl_words()[13 * layer as usize + index]
}

fn kl_m(layer: u64, index: usize) -> u32 {
    let words = kl_words();
    words[13 * 8 + 3 * (layer as usize / 2) + index]
}

fn pw(bits: u32) -> E {
    E::from_base(F(1u64 << bits))
}

fn t_of(k: u32) -> u32 {
    if k < 16 { 16 - k } else { 0 }
}

fn scale_index(value: E, index: u64) -> E {
    value * E::from_base(F(index))
}

fn shift(value: E, multiplier: u32) -> E {
    value * E::from_base(F(multiplier as u64))
}

fn sum(values: &[E]) -> E {
    values.iter().fold(E::ZERO, |acc, &value| acc + value)
}

fn sum_range(values: &[E], a: usize, b: usize) -> E {
    sum(&values[a..b])
}

fn last(point: &[E], count: usize) -> Vec<E> {
    point[point.len() - count..].to_vec()
}

fn order_default(mu: u32) -> Vec<u32> {
    (0..mu).rev().collect()
}

fn order_range(mu: u32) -> Vec<u32> {
    let mut out = vec![2, 1, 0, 5, 4, 3, 8, 7, 6];
    out.extend((9..mu).rev());
    out
}

fn form_constant(form: u32) -> E {
    if form == 1 {
        pw(7)
    } else {
        E::from_base(F(32896))
    }
}

pub struct LayerCheck<'a> {
    pub run: &'a mut LayerRun,
    pub i_u3: u32,
    pub head: bool,
    pub vt: Vec<E>,
    pub vp: Vec<E>,
    pub row_x: Vec<Vec<E>>,
    pub before_smallness: u32,
    pub reads_by_matrix: Vec<u64>,
}

impl<'a> LayerCheck<'a> {
    pub fn new(run: &'a mut LayerRun) -> Self {
        let i_u3 = 0;
        Self {
            run,
            i_u3,
            head: false,
            vt: vec![],
            vp: vec![],
            row_x: vec![],
            before_smallness: 0,
            reads_by_matrix: vec![],
        }
    }

    pub fn new_head(run: &'a mut LayerRun) -> Self {
        let mut out = Self::new(run);
        out.head = true;
        out
    }

    fn call_index(&self, name: &str) -> usize {
        self.run
            .fixture
            .calls
            .iter()
            .position(|call| call.seg(name).is_some())
            .unwrap_or_else(|| panic!("unknown layer segment {name}"))
    }

    fn zl(&self, call: usize) -> &[E] {
        &self.run.forest.calls[call].zk[..2]
    }

    fn zj(&self, call: usize) -> &[E] {
        &self.run.forest.calls[call].zk[2..]
    }

    fn table_name(&self, matrix: usize) -> &'static str {
        match matrix {
            0 => "af",
            1 => "ad",
            2 => "ar",
            3 => "qa",
            4 => "ar",
            5 => "b",
            6 => "qa",
            7 => "xbO",
            8 => "xbI",
            _ => unreachable!(),
        }
    }

    fn table_at(&self, matrix: usize, point: &str) -> u64 {
        self.run
            .tables
            .at(&format!("{}@{point}", self.table_name(matrix)))
            .unwrap()
    }

    fn u3at(&self, offset: u64, index: u64) -> Aff {
        self.run
            .tables
            .entry(self.i_u3, offset, index)
            .unwrap_or_else(|_| panic!("bad U3 entry offset={offset} index={index}"))
    }

    fn long_a(&self, offset: u64, tensor: &Ten, i: u64, jh: u64) -> Aff {
        let mut out = Aff::default();
        for limb in 0..tensor.nl {
            out.add(
                &self.u3at(offset, tensor.cell(limb, i, 0) / S2 + jh),
                pw(8 * limb as u32),
            );
        }
        out
    }

    fn short_a(&self, offset: u64, tensor: &Ten, i: u64, j: u64) -> Aff {
        let mut out = Aff::default();
        for limb in 0..tensor.nl {
            out.add(
                &self.u3at(offset, tensor.cell(limb, i, j)),
                pw(8 * limb as u32),
            );
        }
        out
    }

    fn xb_a(&self, offset: u64, kh: u64) -> Aff {
        let mut out = Aff::default();
        for limb in 0..4 {
            out.add(
                &self.u3at(offset, limb * 512 / S2 + kh),
                pw(8 * limb as u32),
            );
        }
        out
    }

    fn u3abs(&self, index: u64) -> Aff {
        if index >= self.run.fixture.n_u3 {
            panic!("bad absolute U3 index {index}");
        }
        Aff::entry(self.i_u3, 0, index)
    }

    fn ew_of(&self, segment: &str) -> Vec<E> {
        let call = self.call_index(segment);
        let c = self.run.fixture.call(segment);
        let eq = eq_top(&self.zj(call)[..c.l_j as usize]);
        let seg = c.seg(segment).unwrap();
        let out: Vec<E> = seg.pos.iter().map(|&position| eq[position as usize]).collect();
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_EW name={} lj={} z0={:016x} positions={:?} values={:?}", segment, c.l_j, self.zj(call)[0].0[0], &seg.pos[..seg.pos.len().min(4)], &out[..out.len().min(4)]);
        }
        out
    }

    fn ewh(&self, segment: &str) -> Vec<E> {
        let call = self.call_index(segment);
        let c = self.run.fixture.call(segment);
        let z = &self.zj(call)[..c.l_j as usize];
        let eq = eq_top(&z[..z.len() - LS as usize]);
        let seg = c.seg(segment).unwrap();
        (0..seg.n / S2)
            .map(|xh| eq[seg.pos[(S2 * xh) as usize] as usize / S2 as usize])
            .collect()
    }

    fn sa(&self, wh: &[E], x: &str, i: u64, point: &str) -> Aff {
        let tensor = self.run.fixture.tensor(x);
        let offset = self.table_at(tensor.mat, point);
        let mut out = Aff::default();
        for (xh, &weight) in wh.iter().enumerate() {
            out.add(&self.long_a(offset, tensor, i, xh as u64), weight);
        }
        out
    }

    fn sba(&self, wh: &[E], a: u64, b: u64, x: &str, i: u64, point: &str, x0: u64) -> Aff {
        let tensor = self.run.fixture.tensor(x);
        let offset = self.table_at(tensor.mat, point);
        let mut out = Aff::default();
        for xh in (a / S2) as usize..(b / S2) as usize {
            out.add(
                &self.long_a(offset, tensor, i, xh as u64 - x0 / S2),
                wh[xh],
            );
        }
        out
    }

    fn sxs_a(&self, wh: &[E], point: &str, sg: u64) -> Aff {
        let matrix = if sg == 0 {
            8
        } else if sg == 2 * LAM {
            7
        } else {
            0
        };
        let offset = self.table_at(matrix, point);
        let tensor = self.run.fixture.tensor("xs");
        let mut out = Aff::default();
        for xh in 0..D / S2 {
            let form = if sg == 0 || sg == 2 * LAM {
                self.xb_a(offset, xh)
            } else {
                self.long_a(offset, tensor, sg - 1, xh)
            };
            out.add(&form, wh[xh as usize]);
        }
        out
    }

    fn sh_a(&self, x: &str, point: &str, i: u64, j: u64) -> Aff {
        let tensor = self.run.fixture.tensor(x);
        let offset = self.table_at(tensor.mat, point);
        self.short_a(offset, tensor, i, j)
    }

    fn site(&self, w: E, sq: Aff, k: u32, form: u32, srh: Aff) -> Aff {
        let mut out = (Aff::constant(w * pw(k - 1)) - sq.clone().scaled(pw(k))).scaled(pw(t_of(k)));
        if form != 0 {
            out = out - (Aff::constant(form_constant(form) * w) + srh.clone()).scaled(pw(16));
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_SITE w={:016x} k={} form={} sq={:016x} srh={:016x} out={:016x}", w.0[0], k, form, sq.c.0[0], srh.c.0[0], out.c.0[0]);
        }
        out
    }

    fn high(&self, w: E, srh: Aff, k: u32, form: u32) -> Aff {
        let s = Aff::constant(form_constant(form) * w) + srh.clone();
        let top = if form == 1 { 24 } else { 32 };
        let out = s.clone().scaled(pw(top - k.max(16)));
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_HIGH w={:016x} k={} form={} srh={:016x} sc={:016x} out={:016x}", w.0[0], k, form, srh.c.0[0], s.c.0[0], out.c.0[0]);
        }
        out
    }

    fn row_at(&self, point: &str, i: u64) -> RowA {
        let offset = self.table_at(4, point);
        let tensor = self.run.fixture.tensor("row");
        let mut digits = Vec::with_capacity(74);
        for j in 0..74 {
            digits.push(self.u3at(offset, tensor.off + i * 74 + j) + Aff::constant(pw(7)));
        }
        let dig = |at: usize, count: usize| {
            let mut out = Aff::default();
            for j in 0..count {
                out.add(&digits[at + j], pw(8 * j as u32));
            }
            out
        };
        let mut c = vec![Aff::default()];
        for s in 1..=6 {
            c.push(dig(8 + 6 * (s - 1), 6) - Aff::constant(pw(47)));
        }
        c.push(digits[7].clone());
        RowA {
            digits: digits.clone(),
            v: dig(0, 8),
            c,
            s0: dig(45, 1),
            sig: dig(44, 1).scaled(pw(1)) + dig(45, 1) + dig(46, 1).scaled(pw(8)) + dig(47, 1).scaled(pw(16)) + dig(48, 1).scaled(pw(24)),
            l1: dig(49, 4),
            h1: dig(53, 4),
            r_lo: dig(57, 3),
            r_hi: dig(60, 3),
            r: dig(57, 3) + dig(60, 3).scaled(pw(24)),
            p1: dig(63, 3),
            rr: dig(66, 4),
            rq: dig(70, 4),
        }
    }

    fn row5_at(&self, point: &str, i: u64) -> Row5 {
        let offset = self.table_at(4, point);
        let tensor = self.run.fixture.tensor("row");
        let mut digits = Vec::with_capacity(99);
        for j in 0..99 {
            digits.push(self.u3at(offset, tensor.off + i * 99 + j) + Aff::constant(pw(7)));
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
        Row5 {
            digits: digits.clone(),
            c,
            v_lo: dig(0, 6),
            v_hi: dig(6, 4),
            a: dig(58, 1).scaled(pw(1)) + dig(59, 1) + dig(60, 1).scaled(pw(8)) + dig(61, 1).scaled(pw(16)),
            b: dig(62, 1) + dig(63, 1).scaled(pw(8)),
            sig: dig(58, 1).scaled(pw(1)) + dig(59, 1) + dig(60, 1).scaled(pw(8)) + dig(61, 1).scaled(pw(16)) + dig(62, 1).scaled(pw(24)) + dig(63, 1).scaled(pw(32)),
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
        }
    }

    fn row_r(&self, point: &str, i: u64) -> Aff {
        self.row5_at(point, i).r
    }

    pub fn line(&self, segment: &str, layer: u64) -> Aff {
        let call = self.call_index(segment);
        let z = "Z";
        let short = "zt";
        let l = u64::from(!self.run.fixture.first) * LAM + layer;
        let ew = self.ew_of(segment);
        let w = sum(&ew);
        let wh = if ew.len() as u64 % S2 == 0 && segment != "ROW" {
            self.ewh(segment)
        } else {
            vec![]
        };
        let s = |x: &str| self.sa(&wh, x, layer, z);
        let layer_m = layer / 2;
        let kq = kl(l, 0);
        let kk = kl(l, 1);
        let kv = kl(l, 2);
        let ko = kl(l, 3);
        let kd = kl(l, 6);
        let kr = if layer % 2 == 1 { kl_m(l, 0) } else { 0 };
        let is_moe = layer % 2 == 1;
        match segment {
            "G" | "U" => {
                let up = segment == "U";
                let form = 1;
                let (qn, rh) = if up { ("u", "ru") } else { ("gi", "rg") };
                let q = |a: u64, b: u64| {
                    if up {
                        self.sba(&wh, a, b, qn, layer, z, 0)
                    } else {
                        self.sba(&wh, a, b, "g0", layer, z, 0)
                            + self.sba(&wh, a, b, "gh", layer, z, 0).scaled(pw(8))
                    }
                };
                if !is_moe {
                    return self.site(w, q(0, F_WIDTH), kl(l, if up { 5 } else { 4 }), form, s(rh));
                }
                let a = self.site(
                    sum_range(&ew, 0, FE as usize),
                    q(0, FE),
                    kl(l, if up { 5 } else { 4 }),
                    form,
                    self.sba(&wh, 0, FE, rh, layer, z, 0),
                );
                return a + self.site(
                    sum_range(&ew, FE as usize, F_WIDTH as usize),
                    q(FE, F_WIDTH),
                    kl_m(l, if up { 2 } else { 1 }),
                    form,
                    self.sba(&wh, FE, F_WIDTH, rh, layer, z, 0),
                );
            }
            "GH" | "UH" => {
                let up = segment == "UH";
                let rh = if up { "ru" } else { "rg" };
                if !is_moe {
                    return self.high(w, s(rh), kl(l, if up { 5 } else { 4 }), 1);
                }
                let a = self.high(
                    sum_range(&ew, 0, FE as usize),
                    self.sba(&wh, 0, FE, rh, layer, z, 0),
                    kl(l, if up { 5 } else { 4 }),
                    1,
                );
                return a + self.high(
                    sum_range(&ew, FE as usize, F_WIDTH as usize),
                    self.sba(&wh, FE, F_WIDTH, rh, layer, z, 0),
                    kl_m(l, if up { 2 } else { 1 }),
                    1,
                );
            }
            "UT" => return s("u") + Aff::constant(pw(15) * w),
            "QK" | "QKH" => {
                let mut out = Aff::default();
                for b in 0..2 {
                    let (a0, b0, k) = if b == 0 { (0, DQ, kq) } else { (DQ, NQK, kk) };
                    let wb = sum_range(&ew, a0 as usize, b0 as usize);
                    let sq = self.sba(&wh, a0, b0, "qkw", layer, z, 0);
                    let srh = self.sba(&wh, a0, b0, "rqk", layer, z, 0);
                    if segment == "QK" {
                        out += self.site(wb, sq, k, 2, srh);
                    } else {
                        out += self.high(wb, srh, k, 2);
                    }
                }
                return out;
            }
            "N1" | "N2" => {
                let n1 = segment == "N1";
                let first = n1;
                return self.site(
                    w,
                    s(if n1 { "aw" } else { "mw" }),
                    kl(l, if first { 8 } else { 10 }),
                    2,
                    s(if n1 { "ra" } else { "rm" }),
                );
            }
            "N1H" | "N2H" => {
                return self.high(
                    w,
                    s(if segment == "N1H" { "ra" } else { "rm" }),
                    kl(l, if segment == "N1H" { 8 } else { 10 }),
                    2,
                );
            }
            "O" => {
                return self.site(
                    w,
                    self.sxs_a(&wh, z, 2 * layer + 1) - self.sxs_a(&wh, z, 2 * layer),
                    ko,
                    1,
                    s("ro"),
                );
            }
            "D" => {
                return self.site(
                    w,
                    self.sxs_a(&wh, z, 2 * layer + 2) - self.sxs_a(&wh, z, 2 * layer + 1),
                    kd,
                    1,
                    s("rd"),
                );
            }
            "V" => return self.site(w, s("vw"), kv, 1, s("rv")),
            "OH" => return self.high(w, s("ro"), ko, 1),
            "DH" => return self.high(w, s("rd"), kd, 1),
            "VH" => return self.high(w, s("rv"), kv, 1),
            "ROW" => {
                let offset = self.table_at(4, short);
                let tensor = self.run.fixture.tensor("row");
                let mut out = Aff::default();
                for x in 0..ew.len() {
                    let np = x / 99;
                    let idx = x % 99;
                    let mu = match idx {
                        9 => 4,
                        58 => 2,
                        59 => 128,
                        63 => 2,
                        76 => 2,
                        79 => 32,
                        _ => 1,
                    };
                    out.add(
                        &(self.u3at(offset, tensor.off + (2 * layer + np as u64) * 99 + idx as u64)
                            + Aff::constant(pw(7))),
                        shift(ew[x], mu),
                    );
                }
                return out;
            }
            "CN1" | "CN2" => {
                let mut b = Aff::default();
                for x in 0..D {
                    b.add(&self.sh_a(if segment == "CN1" { "a" } else { "m" }, short, layer, x), ew[x as usize]);
                }
                return s(if segment == "CN1" { "aw" } else { "mw" })
                    + b.scaled(pw(28))
                    + Aff::constant(w * (pw(15) + pw(39)));
            }
            "CQK" => {
                return s("qkw")
                    + (self.sba(&wh, 0, DQ, "q", layer, z, 0)
                        + self.sba(&wh, DQ, NQK, "kx", layer, z, DQ))
                        .scaled(pw(28))
                    + Aff::constant(w * (pw(15) + pw(39)));
            }
            "CV" => return s("vw") + s("v").scaled(pw(28)) + Aff::constant(w * (pw(15) + pw(39))),
            "F" => {
                return (s("g0") + s("gc").scaled(pw(8)))
                    + s("s").scaled(pw(28))
                    + Aff::constant(w * (pw(15) + pw(40)));
            }
            "G0" => return s("g0") + Aff::constant(w * pw(7)),
            "CG" => return s("gh") + s("gc").scaled(pw(28)) + Aff::constant(w * (pw(14) + pw(15) + pw(35) + pw(40))),
            "XR" => {
                let mut b = Aff::default();
                for x in 0..D {
                    b.add(&self.sh_a("xr", short, layer_m, x), ew[x as usize]);
                }
                return Aff::constant(pw(15) * w) - b.scaled(pw(16));
            }
            "LR" => {
                let mut q = Aff::default();
                for x in 0..NE {
                    q.add(
                        &(self.sh_a("ly", short, layer_m, x).scaled(pw(3)) + self.sh_a("l3", short, layer_m, x)),
                        ew[x as usize],
                    );
                }
                return self.site(w, q, kr, 1, {
                    let mut out = Aff::default();
                    for x in 0..NE {
                        out.add(&self.sh_a("rl", short, layer_m, x), ew[x as usize]);
                    }
                    out
                });
            }
            "LT" => {
                let mut out = Aff::default();
                for x in 0..NE {
                    out.add(&(self.sh_a("lr", short, layer_m, x) + Aff::constant(pw(14))), shift(ew[x as usize], 2));
                }
                return out;
            }
            "DD" => {
                let mut out = Aff::default();
                let mut tail = Aff::default();
                for x in (0..NE).rev() {
                    out.add(&(self.sh_a("lr", short, layer_m, x) + tail.clone()), -ew[x as usize]);
                    tail = tail + self.sh_a("sel", short, layer_m, x);
                }
                return out;
            }
            "LH" => {
                let mut out = Aff::default();
                for x in 0..NE {
                    out.add(&self.sh_a("rl", short, layer_m, x), ew[x as usize]);
                }
                return self.high(w, out, kr, 1);
            }
            "SEL" => {
                let mut out = Aff::default();
                for x in 0..NE {
                    out.add(&self.sh_a("sel", short, layer_m, x), shift(ew[x as usize], 7));
                }
                return out;
            }
            "MR" => return (Aff::constant(pw(14)) + self.sh_a("sc", short, layer_m, 0).scaled(pw(17))).scaled(ew[0]),
            "L3" => {
                let mut out = Aff::default();
                for x in 0..NE {
                    out.add(&self.sh_a("l3", short, layer_m, x), shift(ew[x as usize], 5));
                }
                return out;
            }
            "CLR" => {
                let mut out = Aff::default();
                for x in 0..NE {
                    out.add(
                        &(self.sh_a("ly", short, layer_m, x)
                            + (self.sh_a("lr", short, layer_m, x) - self.sh_a("l3", short, layer_m, x)).scaled(pw(25))),
                        ew[x as usize],
                    );
                }
                return out + Aff::constant(w * (pw(15) + pw(39)));
            }
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct RowA {
    pub digits: Vec<Aff>,
    pub v: Aff,
    pub c: Vec<Aff>,
    pub s0: Aff,
    pub sig: Aff,
    pub l1: Aff,
    pub h1: Aff,
    pub r_lo: Aff,
    pub r_hi: Aff,
    pub r: Aff,
    pub p1: Aff,
    pub rr: Aff,
    pub rq: Aff,
}

#[derive(Debug)]
pub struct Row5 {
    pub digits: Vec<Aff>,
    pub c: Vec<Aff>,
    pub v_lo: Aff,
    pub v_hi: Aff,
    pub a: Aff,
    pub b: Aff,
    pub sig: Aff,
    pub s0: Aff,
    pub l1: Aff,
    pub h1: Aff,
    pub b_lo: Aff,
    pub b_hi: Aff,
    pub kap: Aff,
    pub r_lo: Aff,
    pub r_hi: Aff,
    pub r: Aff,
    pub q1: Aff,
    pub rr: Aff,
    pub rq: Aff,
}

pub fn led_lines5(row: &Row5, products: &[Aff], eps: E, two: &Aff) -> Vec<Aff> {
    let e0 = products[1].clone() - row.b_lo.clone() - row.b_hi.clone().scaled(pw(23));
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_LED0 p1={:016x} blo={:016x} bhi={:016x} vlo={:016x} vhi={:016x} kap={:016x} q1={:016x} l1={:016x} out={:016x}", products[1].c.0[0], row.b_lo.c.0[0], row.b_hi.c.0[0], row.v_lo.c.0[0], row.v_hi.c.0[0], row.kap.c.0[0], row.q1.c.0[0], row.l1.c.0[0], e0.c.0[0]);
    }
    let e6 = two.clone() + row.sig.clone()
        - row.s0.clone()
        - products[3].clone().scaled(E::ONE + E::ONE)
        - row.q1.clone().scaled(pw(25))
        - row.rr.clone().scaled(E::ONE + E::ONE);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_LED6 two={:016x} sig={:016x} s0={:016x} q1={:016x} rr={:016x} out={:016x}", two.c.0[0], row.sig.c.0[0], row.s0.c.0[0], row.q1.c.0[0], row.rr.c.0[0], e6.c.0[0]);
    }
    vec![
        e0,
        row.v_lo.clone() + Aff::constant(eps)
            - products[0].clone()
            - row.b_lo.clone().scaled(pw(25))
            - row.l1.clone()
            + row.kap.clone().scaled(pw(48)),
        row.v_hi.clone() - row.b_hi.clone() - products[2].clone() - row.kap.clone(),
        row.l1.clone() + row.h1.clone() - row.sig.clone().scaled(E::ONE + E::ONE),
        products[4].clone() + products[5].clone() - row.q1.clone(),
        products[6].clone(),
        e6,
        row.rr.clone() + row.rq.clone() - row.sig.clone() + Aff::constant(E::ONE),
    ]
}

#[derive(Debug, Clone)]
pub struct Panel {
    pub sn: String,
    claim: usize,
    ec: E,
    ni: u64,
    ny: u64,
    wy: u32,
    w: u32,
    off: u64,
    tri: bool,
    virt: i64,
    v0: i64,
    wt: Vec<E>,
    weights: Vec<E>,
    sl: Vec<E>,
    xx: Vec<Aff>,
}

pub struct StackOut {
    pub row_s: Aff,
    pub hd: ProductLevel,
    pub hd2: Option<ProductLevel>,
    pub panels: Vec<Panel>,
    pub claims: Vec<crate::fp::weights::WsClaim>,
    pub rs: Vec<E>,
    pub s_a0: u32,
    pub s_b0: u32,
    pub s_v: u32,
    pub s_h1: u32,
    pub a_lin: Aff,
    pub b_lin: Aff,
    pub om: Vec<E>,
    pub next_v: u32,
    pub after_hd: u32,
    pub after_hd2: u32,
}

impl LayerCheck<'_> {
    fn join_part(&self, call: usize) -> Result<(Aff, Vec<u64>), FpError> {
        let c = &self.run.fixture.calls[call];
        let el = eq_top(self.zl(call));
        let e = eq_top(&self.zj(call)[..c.l_j as usize]);
        let mut weighed = vec![false; (LAM * c.j) as usize];
        let mut lines = Aff::default();
        let mut laid = vec![0u64; self.run.forest.nt as usize];
        for layer in 0..LAM {
            for segment in &c.segs {
                if segment.moe && layer % 2 == 0 {
                    continue;
                }
                let value = self.line(&segment.name, layer);
                #[cfg(feature = "std")]
                if std::env::var_os("DUMP_SHARED").is_some() {
                    eprintln!("RUST_LINE name={} lam={} constant={:016x}", segment.name, layer, value.c.0[0]);
                }
                lines.add(&value, -el[layer as usize]);
                for &position in &segment.pos {
                    let at = (layer * c.j + position) as usize;
                    if weighed[at] {
                        return Err(FpError::Range {
                            context: "layer-join-place",
                        });
                    }
                    weighed[at] = true;
                }
            }
        }
        let mut pad_w = E::ZERO;
        for layer in 0..LAM {
            let mut weight = E::ZERO;
            for column in 0..c.j {
                if !weighed[(layer * c.j + column) as usize] {
                    weight += e[column as usize];
                    laid[(self.run.forest.first[call] as u64 + ((layer * c.j + column) >> (self.run.forest.d_t - 6))) as usize] += 64;
                }
            }
            pad_w += el[layer as usize] * weight;
        }
        let call_public = &self.run.forest.calls[call];
        let start = Aff::constant(
            self.run.forest.alpha
                - scale_index(self.run.forest.theta, call as u64)
                - pad_w * call_public.tau_at(call_public.t0),
        ) + lines.clone();
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_JOIN_PART q={} constant={:016x} lines={:016x} padw={:016x} alpha={:016x}", call, start.c.0[0], lines.c.0[0], pad_w.0[0], self.run.forest.alpha.0[0]);
        }
        Ok((start, laid))
    }

    fn k_t(layer: u64, index: usize) -> E {
        pw(t_of(kl(layer, index)))
    }

    fn k_tm(layer: u64, index: usize) -> E {
        pw(t_of(kl_m(layer, index)))
    }

    pub fn stack(&mut self, f: &mut HFork, tr: &mut Tr, slot: &mut u32) -> Result<StackOut, FpError> {
        let mut panels = Vec::new();
        let mut next_v = 0i64;
        let ac = [0, 1, 2];
        let ts = [0, 1, 2, 5];
        for &ci in &ac {
            let c = &self.run.fixture.calls[ci];
            let el = eq_top(self.zl(ci));
            for segment in &c.segs {
                let sn = segment.name.as_str();
                let (ny, wt): (u64, Vec<E>) = match sn {
                    "G" | "U" => {
                        let up = sn == "U";
                        (
                            D,
                            (0..LAM)
                                .map(|layer| el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, if up { 5 } else { 4 }))
                                .collect(),
                        )
                    }
                    "V" => (
                        D,
                        (0..LAM)
                            .map(|layer| el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, 2))
                            .collect(),
                    ),
                    "O" => (
                        DQ,
                        (0..LAM)
                            .map(|layer| el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, 3))
                            .collect(),
                    ),
                    "D" => (
                        F_WIDTH,
                        (0..LAM)
                            .map(|layer| el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, 6))
                            .collect(),
                    ),
                    "LR" => (
                        D,
                        (0..LAM_M)
                            .map(|i| {
                                let layer = 2 * i + 1;
                                el[layer as usize] * Self::k_tm(u64::from(!self.run.fixture.first) * LAM + layer, 0)
                            })
                            .collect(),
                    ),
                    _ => continue,
                };
                let ni = if sn == "LR" { LAM_M } else { LAM };
                let virt = if sn == "D" { 0 } else { -1 };
                let v0 = next_v;
                next_v += ni as i64;
                panels.push(Panel {
                    sn: sn.into(),
                    claim: 0,
                    ec: self.run.forest.e[ci],
                    ni,
                    ny,
                    wy: crate::fp::tensor::ceil_log2(ny),
                    w: crate::fp::tensor::ceil_log2(ni) + crate::fp::tensor::ceil_log2(ny),
                    off: 0,
                    tri: false,
                    virt,
                    v0,
                    wt,
                    sl: vec![],
                    weights: vec![],
                    xx: vec![],
                });
            }
        }
        for &ti in &ts {
            let c = &self.run.fixture.calls[ti];
            let el = eq_top(self.zl(ti));
            for segment in &c.segs {
                let sn = segment.name.as_str();
                match sn {
                    "N1" | "N2" => {
                        let n2 = usize::from(sn == "N2");
                        let wh = self.ewh(sn);
                        panels.push(Panel {
                            sn: sn.into(),
                            claim: 1,
                            ec: self.run.forest.e[ti],
                            ni: LAM,
                            ny: 1,
                            wy: 0,
                            w: crate::fp::tensor::ceil_log2(LAM),
                            off: 0,
                            tri: false,
                            virt: -1,
                            v0: -1,
                            wt: (0..LAM).map(|layer| el[layer as usize]).collect(),
                            sl: vec![],
                            xx: vec![],
                            weights: wh,
                        });
                    }
                    "QK" => {
                        let v0 = next_v;
                        next_v += LAM as i64;
                        panels.push(Panel {
                            sn: sn.into(),
                            claim: 1,
                            ec: self.run.forest.e[ti],
                            ni: LAM,
                            ny: D,
                            wy: crate::fp::tensor::ceil_log2(D),
                            w: crate::fp::tensor::ceil_log2(LAM) + crate::fp::tensor::ceil_log2(D),
                            off: 0,
                            tri: false,
                            virt: -1,
                            v0,
                            wt: (0..LAM)
                                .map(|layer| el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, 0))
                                .collect(),
                            sl: vec![],
                    weights: vec![],
                            xx: vec![],
                        });
                    }
                    "XR" => {
                        let ew = self.ew_of(sn);
                        panels.push(Panel {
                            sn: sn.into(),
                            claim: 1,
                            ec: self.run.forest.e[ti],
                            ni: LAM_M,
                            ny: 1,
                            wy: 0,
                            w: crate::fp::tensor::ceil_log2(LAM_M),
                            off: 0,
                            tri: false,
                            virt: -1,
                            v0: -1,
                            wt: (0..LAM_M).map(|layer| el[(2 * layer + 1) as usize]).collect(),
                            sl: vec![],
                            xx: vec![],
                            weights: ew,
                        });
                    }
                    "DD" | "MR" => {
                        let ew = self.ew_of(sn);
                        let w_s = if sn == "DD" { sum(&ew) } else { ew[0] };
                        panels.push(Panel {
                            sn: sn.into(),
                            claim: 1,
                            ec: self.run.forest.e[ti],
                            ni: LAM_M,
                            ny: NE,
                            wy: crate::fp::tensor::ceil_log2(NE),
                            w: crate::fp::tensor::ceil_log2(LAM_M) + crate::fp::tensor::ceil_log2(NE),
                            off: 0,
                            tri: false,
                            virt: -1,
                            v0: -1,
                            wt: (0..LAM_M).map(|layer| el[(2 * layer + 1) as usize] * w_s).collect(),
                            sl: vec![],
                            xx: vec![],
                            weights: ew,
                        });
                    }
                    "G" | "U" | "D" => {
                        for lm in 0..LAM_M {
                            let layer = 2 * lm + 1;
                            let mut sl = Vec::new();
                            for e in 0..NE {
                                sl.push(f.shown(self.run.s_sel + (lm * NE + e) as u32)?);
                            }
                            let (virt, wt) = if sn == "D" {
                                (
                                    1 + lm as i64,
                                    el[layer as usize] * Self::k_t(u64::from(!self.run.fixture.first) * LAM + layer, 6),
                                )
                            } else {
                                let up = sn == "U";
                                (
                                    -1,
                                    el[layer as usize] * Self::k_tm(u64::from(!self.run.fixture.first) * LAM + layer, if up { 2 } else { 1 }),
                                )
                            };
                            let v0 = next_v;
                            next_v += 1;
                            let ny = if sn == "D" { FE } else { D };
                            let xx = if sn == "D" {
                                vec![]
                            } else {
                                (0..D).map(|k2| self.sh_a("xr", "rt", lm, k2)).collect()
                            };
                            panels.push(Panel {
                                sn: sn.into(),
                                claim: 1,
                                ec: self.run.forest.e[ti],
                                ni: NE,
                                ny,
                                wy: crate::fp::tensor::ceil_log2(ny),
                                w: crate::fp::tensor::ceil_log2(NE) + crate::fp::tensor::ceil_log2(ny),
                                off: 0,
                                tri: true,
                                virt,
                                v0,
                                wt: vec![wt],
                                weights: vec![],
                                sl,
                                xx,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        let mut order: Vec<usize> = (0..panels.len()).collect();
        order.sort_by_key(|&index| core::cmp::Reverse(panels[index].w));
        let mut off = 0u64;
        for index in order {
            panels[index].off = off;
            off += 1u64 << panels[index].w;
        }
        if crate::fp::tensor::ceil_log2(off) != self.run.fixture.lam_s {
            return Err(FpError::Range {
                context: "layer-stack",
            });
        }
        let om = tr.batch(2)?;
        let mut cstart = Aff::default();
        let mut laid = vec![0u64; self.run.forest.nt as usize];
        for call in 0..self.run.fixture.calls.len() {
            let (part, counts) = self.join_part(call)?;
            cstart.add(&part, self.run.forest.e[call]);
            for (i, count) in counts.into_iter().enumerate() {
                laid[i] += count;
            }
        }
        if laid != self.run.forest.pub_counts {
            return Err(FpError::Range {
                context: "layer-padding",
            });
        }
        let ey = eq_top(&self.run.forest.zy);
        for t in self.run.forest.nt as usize..1usize << self.run.forest.r_k {
            cstart.c += ey[t] * (self.run.forest.alpha - self.run.forest.padding_key(0));
        }
        cstart = cstart - Aff::slot(self.run.t_t.s_cl) - self.run.forest.c_end.clone();
        cstart.add(&self.run.kt, om[1]);
        let mut stack = HStack {
            lam: self.run.fixture.lam_s,
            rs: vec![],
            p: psc_begin(f, tr, slot, cstart)?,
        };
        stack_rounds(f, tr, slot, &mut stack)?;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            for (v, value) in stack.rs.iter().enumerate() {
                eprintln!("RUST_STACK_RS v={} value={:016x}", v, value.0[0]);
            }
        }
        let nvirt = 1 + LAM_M;
        // C++: NCLp = hp.NCL - (hasE ? 1 : 0), and hasE holds only for the
        // first layer (g == 0).
        let nclp = self.run.fixture.ncl as u64 - u64::from(self.run.fixture.first);
        if next_v != nclp as i64 {
            return Err(FpError::Range {
                context: "layer-claim-count",
            });
        }
        let s_a0 = crate::fp::model::psc_ends(f, tr, slot, &mut stack.p, 2, (2 + nvirt + nclp) as u32)?;
        let s_h1 = s_a0 + 1;
        let s_b0 = s_h1 + nvirt as u32;
        let s_v = s_b0 + 1;
        let mut a_lin = Aff::default();
        let mut b_lin = Aff::default();
        let mut a_s = f.shown(s_a0)?;
        let mut b_s = f.shown(s_b0)?;
        let mut w_d = 0;
        for panel in &panels {
            let sb = crate::fp::model::eq_low_at(&stack.rs, panel.w, stack.lam, panel.off >> panel.w);
            let ey = crate::fp::model::eq_low(&stack.rs, 0, panel.wy);
            let ei = crate::fp::model::eq_low(&stack.rs, panel.wy, panel.w);
            let co = sb * om[panel.claim] * panel.ec;
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_PANEL_VALUES name={} off={} w={} sb={:016x} ei0={:016x} ei1={:016x} sv0={:016x} sv1={:016x} sb0={:016x}", panel.sn, panel.off, panel.w, sb.0[0], ei[0].0[0], ei[1].0[0], f.shown(s_v)?.0[0], f.shown(s_v + 1)?.0[0], f.shown(s_b0)?.0[0]);
            }
            if panel.tri {
                let s1 = (0..NE).map(|e| panel.sl[e as usize] * ei[e as usize]).fold(E::ZERO, |a, b| a + b);
                if panel.virt >= 0 {
                    a_s += co * panel.wt[0] * s1 * f.shown(s_h1 + panel.virt as u32)?;
                } else {
                    for k2 in 0..panel.ny {
                        a_lin.add(&panel.xx[k2 as usize], co * panel.wt[0] * s1 * ey[k2 as usize]);
                    }
                }
                b_s += sb * f.shown(s_v + panel.v0 as u32)?;
                continue;
            }
            if panel.virt >= 0 {
                if panel.v0 < 0 || w_d != 0 {
                    return Err(FpError::Range {
                        context: "layer-stack",
                    });
                }
                w_d = panel.w;
                a_s += sb * om[panel.claim] * f.shown(s_h1 + panel.virt as u32)?;
                for i in 0..panel.ni {
                    b_s += sb * ei[i as usize] * panel.ec * panel.wt[i as usize] * f.shown(s_v + panel.v0 as u32 + i as u32)?;
                }
                continue;
            }
            for i in 0..panel.ni {
                for yv in 0..panel.ny {
                    let a = self.panel_a(f, panel, i, yv)?;
                    a_lin.add(&a, co * panel.wt[i as usize] * ei[i as usize] * ey[yv as usize]);
                    if panel.v0 < 0 {
                        let b = self.panel_b(panel, i, yv)?;
                        b_lin.add(&b, sb * ei[i as usize] * ey[yv as usize]);
                    }
                }
                if panel.v0 >= 0 {
                    b_s += sb * ei[i as usize] * f.shown(s_v + panel.v0 as u32 + i as u32)?;
                }
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_PANEL_CONTRIB name={} as={:016x} bs={:016x}", panel.sn, a_s.0[0], b_s.0[0]);
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_ALIN_CONTRIB name={} alin={:016x}", panel.sn, a_lin.c.0[0]);
            }
        }
        if w_d != 2 + LPF {
            return Err(FpError::Range {
                context: "layer-stack",
            });
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_STACK_ROW as={:016x} bs={:016x} target={:016x} start={:016x} claim={:016x} eps={:016x}", a_s.0[0], b_s.0[0], stack.p.target.0[0], stack.p.start.c.0[0], stack.p.claim.0[0], stack.p.eps.0[0]);
        }
        let row_s = psc_row(f, &mut stack.p, a_s * b_s, &Aff::default())?;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_STACK_ROW_OUT constant={:016x}", row_s.c.0[0]);
        }
        let mut rho = (0..w_d).rev().map(|v| stack.rs[v as usize]).collect::<Vec<_>>();
        rho.extend_from_slice(&self.run.points["zt"]);
        let hd = crate::fp::forest::read_product_level(
            f,
            tr,
            slot,
            18,
            &rho,
            &mut Aff::slot(s_h1),
            &order_range(18),
        )?;
        self.run.points.insert("Z_a".into(), hd.r[hd.r.len() - 8..].to_vec());
        let after_hd = *slot;
        let hd2 = if true {
            let mut rho = tr.point(RoundName::Join, 1)?;
            let eq = eq_top(&rho);
            let mut start = Aff::default();
            for lm in 0..LAM_M {
                start.add(&Aff::slot(s_h1 + 1 + lm as u32), eq[lm as usize]);
            }
            rho.extend((0..LFE).rev().map(|v| stack.rs[v as usize]));
            rho.extend_from_slice(&self.run.t_t.r);
            let level = crate::fp::forest::read_product_level(
                f,
                tr,
                slot,
                16,
                &rho,
                &mut start,
                &order_range(16),
            )?;
            self.run.points.insert("Z_b".into(), level.r[level.r.len() - 8..].to_vec());
            Some(level)
        } else {
            None
        };
        Ok(StackOut {
            row_s,
            hd,
            hd2,
            panels,
            claims: build_claims(self.run, &stack.rs)?,
            rs: stack.rs.clone(),
            s_a0,
            s_b0,
            s_v,
            s_h1,
            a_lin,
            b_lin,
            om,
            next_v: next_v as u32,
            after_hd,
            after_hd2: *slot,
        })
    }

    fn panel_a(&self, f: &HFork, panel: &Panel, i: u64, yv: u64) -> Result<Aff, FpError> {
        match panel.sn.as_str() {
            "G" | "U" | "LR" => Ok(self.sh_a("m", "zt", i, yv)),
            "V" => Ok(self.sh_a("a", "zt", i, yv)),
            "O" => Ok(self.sh_a("o", "zt", i, yv)),
            "N1" | "N2" => {
                let n2 = u32::from(panel.sn == "N2");
                Ok(Aff::constant(f.shown(self.run.t_t.s_end + (2 * i + n2 as u64) as u32)?))
            }
            "QK" => Ok(self.sh_a("a", "rt", i, yv)),
            "XR" => {
                #[cfg(feature = "std")]
                if std::env::var_os("DUMP_SHARED").is_some() {
                    eprintln!("RUST_XR_SC i={} value={:016x}", i, f.shown(self.run.s_sc + i as u32)?.0[0]);
                }
                Ok(Aff::constant(f.shown(self.run.s_sc + i as u32)?))
            }
            "DD" | "MR" => Ok(Aff::constant(f.shown(self.run.s_sel + (i * NE + yv) as u32)?)),
            _ => unreachable!(),
        }
    }

    fn panel_b(&self, panel: &Panel, i: u64, yv: u64) -> Result<Aff, FpError> {
        match panel.sn.as_str() {
            "N1" | "N2" => {
                let n2 = u32::from(panel.sn == "N2");
                Ok(self.sxs_a(&panel.weights, "Y", 2 * i + n2 as u64))
            }
            "XR" => {
                let mut out = Aff::default();
                for x in 0..D {
                    out.add(&self.sh_a("m", "rt", i, x), panel.weights[x as usize]);
                }
                Ok(out)
            }
            "DD" | "MR" => Ok(self.sh_a("lr", "rt", i, yv)),
            _ => unreachable!(),
        }
    }
}

pub struct WeightMerge {
    pub fvp: E,
    pub beta: E,
    pub rm: Vec<E>,
    pub p: crate::fp::model::Psc,
    pub s_fv: u32,
    pub s_ve: u32,
    pub rho: Vec<E>,
    pub rec: Vec<crate::fp::weights::WsRecClaim>,
}

fn build_claims(run: &LayerRun, rs: &[E]) -> Result<Vec<crate::fp::weights::WsClaim>, FpError> {
    let lay = crate::fp::weights::ws_lay();
    let ui_of = |x: u32, l: u32| -> Option<u32> {
        lay.u.iter().position(|unit| unit.x == x && unit.l == l).map(|i| i as u32)
    };
    let fac_rs = |g: &crate::fp::weights::WsGrp, lo0: i32, wtot: i32| {
        let mut out = crate::fp::weights::WsFac {
            hv: vec![],
            pt: vec![],
            av: vec![],
        };
        for i in 0..g.z {
            out.pt.push(rs[(lo0 + g.z - 1 - i) as usize]);
        }
        out.hv = crate::fp::model::eq_low(rs, (lo0 + g.z) as u32, (lo0 + wtot) as u32);
        out
    };
    let fac_ew = |g: &crate::fp::weights::WsGrp, sn: &str, x0: u64| {
        let q = run.fixture.calls.iter().position(|call| call.seg(sn).is_some()).unwrap();
        let c = &run.fixture.calls[q];
        let z = &run.forest.calls[q].zk[..c.l_j as usize];
        let mut out = crate::fp::weights::WsFac {
            hv: vec![],
            pt: vec![],
            av: vec![],
        };
        for i in 0..g.z {
            out.pt.push(z[(c.l_j as i32 - g.z + i) as usize]);
        }
        let e = eq_top(&z[..(c.l_j as i32 - g.z) as usize]);
        let s = c.seg(sn).unwrap();
        for xh in 0..g.o {
            let at = x0 + ((xh as u64) << g.z);
            out.hv.push(e[(s.pos[at as usize] >> (g.z as u32)) as usize]);
        }
        out
    };
    let mut claims = Vec::new();
    let mut claim = |x: u32, lam: u32, fac: &dyn Fn(&crate::fp::weights::WsGrp) -> crate::fp::weights::WsFac| {
        let ui = ui_of(x, lam).ok_or(FpError::Range {
            context: "layer-claim-unit",
        })?;
        let unit = &lay.u[ui as usize];
        claims.push(crate::fp::weights::WsClaim {
            ui,
            f: unit.g.iter().map(fac).collect(),
        });
        Ok::<(), FpError>(())
    };
    for &ci in &[0usize, 1, 2] {
        let c = &run.forest.calls[ci];
        for segment in &run.fixture.calls[ci].segs {
            let sn = segment.name.as_str();
            if !matches!(sn, "G" | "U" | "V" | "O" | "D" | "LR") {
                continue;
            }
            for i in 0..if sn == "LR" { 2 } else { 4 } {
                let lam = if sn == "LR" { 2 * i + 1 } else { i } as u32;
                let mo = lam % 2 == 1;
                let x = match sn {
                    "G" => if mo { 5 } else { 3 },
                    "U" => if mo { 6 } else { 4 },
                    "V" => 1,
                    "O" => 2,
                    "D" => 9,
                    _ => 11,
                };
                let wy = crate::fp::tensor::ceil_log2(match sn {
                    "O" => DQ,
                    "D" => F_WIDTH,
                    _ => D,
                });
                claim(x, lam, &|g| {
                    if g.name == "kk" {
                        fac_rs(g, 0, wy as i32)
                    } else {
                        fac_ew(g, sn, 0)
                    }
                })?;
            }
        }
    }
    for &ti in &[0usize, 1, 2, 5] {
        for segment in &run.fixture.calls[ti].segs {
            let sn = segment.name.as_str();
            if sn == "QK" {
                let et = eq_top(&run.t_t.r);
                for lam in 0..4u32 {
                    let l = lam;
                    let mut kap = vec![E::ZERO; 128];
                    for i in 0..64u64 {
                        let mut ct = E::ZERO;
                        let mut st = E::ZERO;
                        for t in 0..64u64 {
                            let cv = rope_c(l, t, i, false);
                            let sv = rope_c(l, t, i, true);
                            if cv != 0 {
                                ct += et[t as usize] * E::from_base(F(cv as u64));
                            }
                            if sv != 0 {
                                st += et[t as usize] * E::from_base(F(sv as u64));
                            }
                        }
                        let c = &run.fixture.calls[ti];
                        let z = &run.forest.calls[ti].zk[..c.l_j as usize];
                        let lfc = eq_top(&z[(c.l_j as usize - 7)..]);
                        kap[2 * i as usize] = ct * lfc[2 * i as usize] + st * lfc[2 * i as usize + 1];
                        kap[2 * i as usize + 1] = ct * lfc[2 * i as usize + 1] - st * lfc[2 * i as usize];
                    }
                    claim(0, lam, &|g| {
                        if g.name == "hd" {
                            crate::fp::weights::WsFac {
                                hv: vec![],
                                pt: vec![],
                                av: (0..64).map(|hd| {
                                    if hd >= 6 {
                                        return E::ZERO;
                                    }
                                    let tq = crate::fp::layer::Ten {
                                        name: "q".into(),
                                        mat: 3,
                                        nl: 2,
                                        cnt: 4,
                                        j: 512,
                                        off: 0,
                                    };
                                    let tk = crate::fp::layer::Ten {
                                        name: "kx".into(),
                                        mat: 3,
                                        nl: 2,
                                        cnt: 4,
                                        j: 256,
                                        off: 4096,
                                    };
                                    let at = if hd < 4 {
                                        head_feat(&tq, hd, 128)
                                    } else {
                                        tq.j + head_feat(&tk, hd - 4, 128)
                                    };
                                    let c = &run.fixture.calls[ti];
                                    let seg = c.seg("QK").unwrap();
                                    E::from_base(F(seg.pos[at as usize]))
                                }).collect(),
                            }
                        } else if g.name == "cd" {
                            crate::fp::weights::WsFac {
                                hv: vec![],
                                pt: vec![],
                                av: kap.clone(),
                            }
                        } else {
                            fac_rs(g, 0, 9)
                        }
                    })?;
                }
            } else if matches!(sn, "G" | "U" | "D") {
                for lm in 0..2u32 {
                    let wk = crate::fp::tensor::ceil_log2(match sn {
                        "D" => F_WIDTH,
                        _ => D,
                    });
                    let x = match sn {
                        "G" => 7,
                        "U" => 8,
                        _ => 10,
                    };
                    let lam = 2 * lm + 1;
                    claim(x, lam, &|g| {
                        if g.name == "kk" {
                            fac_rs(g, 0, wk as i32)
                        } else if g.name == "e" {
                            fac_rs(g, wk as i32, 2)
                        } else {
                            fac_ew(g, sn, if sn == "D" { 0 } else { 512 })
                        }
                    })?;
                }
            }
        }
    }
    if run.fixture.first {
        let ui = ui_of(12, 8).ok_or(FpError::Range {
            context: "layer-claim-unit",
        })?;
        let unit = &lay.u[ui as usize];
        let et = eq_top(&run.points["Z_E"]);
        claims.push(crate::fp::weights::WsClaim {
            ui,
            f: unit.g.iter().map(|g| {
                if g.name == "v" {
                    let mut av = vec![E::ZERO; 512];
                    for t in 0..64 {
                        av[run.tokv[t] as usize] += et[t];
                    }
                    crate::fp::weights::WsFac {
                        hv: vec![],
                        pt: vec![],
                        av,
                    }
                } else {
                    fac_rs(g, 0, 9)
                }
            }).collect(),
        });
    }
    if claims.len() as u64 != run.fixture.ncl as u64 {
        return Err(FpError::Range {
            context: "layer-claim-count",
        });
    }
    Ok(claims)
}

fn head_feat(x: &crate::fp::layer::Ten, hd: u64, dh: u64) -> u64 {
    if dh == 0 || x.j % dh != 0 || hd >= x.j / dh {
        return 0;
    }
    hd * dh
}

fn rope_c(layer: u32, t: u64, i: u64, sine: bool) -> i64 {
    let l = layer as u64;
    if l != u64::MAX && crate::fp::layer_checks::kl(l, 12) == 0 {
        return if sine { 0 } else { 4096 };
    }
    let angle = (t as f64) * libm::pow(10000.0, -2.0 * i as f64 / 128.0);
    let value = 4096.0 * if sine { libm::sin(angle) } else { libm::cos(angle) };
    libm::round(value) as i64
}

pub fn ws_fork_merge(
    run: &mut LayerRun,
    stack: &StackOut,
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
) -> Result<WeightMerge, FpError> {
    let nclp = run.fixture.ncl - u32::from(run.fixture.first);
    let mut values = Vec::with_capacity(nclp as usize + usize::from(run.fixture.first));
    for a in 0..nclp {
        values.push(stack.s_v + a);
    }
    let s_ve = if run.fixture.first {
        let slot = f.claims_in(tr, slot, 1)?;
        values.push(slot);
        slot
    } else {
        0
    };
    let rho = tr.batch(values.len())?;
    let mut start = Aff::default();
    for (a, &slot) in values.iter().enumerate() {
        start.add(&Aff::slot(slot), rho[a]);
    }
    let mut p = psc_begin(f, tr, slot, start)?;
    let mut rm = Vec::with_capacity(11);
    for _ in 0..11 {
        rm.push(crate::fp::model::round_in(f, tr, slot, crate::fp::model::RoundForm::Sum2, &mut p, E::ZERO, 0)?);
    }
    let s_fv = crate::fp::model::psc_ends(f, tr, slot, &mut p, 1, 1)?;
    let fvp = f.shown(s_fv)?;
    let beta = p.r[p.r.len() - 1];
    let lay = crate::fp::weights::ws_lay();
    let rec = stack
        .claims
        .iter()
        .enumerate()
        .map(|(a, claim)| crate::fp::weights::ws_row_part(&lay, claim, rho[a] * crate::fp::weights::ws_eps_t(&lay, claim, &rm)))
        .collect();
    Ok(WeightMerge {
        fvp,
        beta,
        rm,
        p,
        s_fv,
        s_ve,
        rho,
        rec,
    })
}

impl LayerCheck<'_> {
    pub fn objects_before_smallness(
        &mut self,
        stack: &StackOut,
        merged: &mut WeightMerge,
        f: &mut HFork,
        tr: &mut Tr,
    ) -> Result<(), FpError> {
        f.plan_claims = self.run.fixture.nlin;
        self.init_trees(f);
        f.theta = tr.batch_open()?;
        for (row_index, row) in self.run.forest.rows.iter().enumerate() {
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && row_index == 55 {
                eprintln!("RUST_FOREST_ROW_ADD index={} n={} slots={:?}", row_index, f.n_claim, row.s.iter().map(|x| x.0).take(5).collect::<Vec<_>>());
            }
            f.lin(row.clone());
        }
        f.lin(self.run.wpl.row.clone());
        f.lin(Aff::slot(self.run.wpl.s_l) - Aff::slot(self.run.wpl.s_r));
        {
            let e = eq_top(&self.run.wpl.r[..self.run.wpl.r.len() - 8]);
            let t_i = self.table_at(8, "Z_V");
            let t_a = self.table_at(0, "Z_V");
            let xs = self.run.fixture.tensor("xs");
            let mut a = Aff::slot(self.run.wpl.s_l);
            for i in 0..8u64 {
                for kh in 0..D / S2 {
                    for limb in 0..4 {
                        let form = if i == 0 {
                            self.u3at(t_i, limb * 512 / S2 + kh)
                        } else {
                            self.u3at(t_a, xs.cell(limb, i - 1, 0) / S2 + kh)
                        };
                        a.add(&form, -e[(i * 512 / S2 + kh) as usize] * self.run.gp[limb as usize]);
                    }
                }
            }
            f.lin(a);
        }
        let mut batch = Aff::slot(self.run.t_v.s_cl);
        for i in 0..self.run.om_v.len() {
            batch.add(&Aff::slot(self.run.s_pv + i as u32), -self.run.om_v[i]);
        }
        f.lin(batch);
        let mut s0 = E::ZERO;
        for i in 0..8u64 {
            let sh = |k: u32| -> Result<E, FpError> {
                f.shown(self.run.t_v.s_end + (4 * i + k as u64) as u32)
            };
            let a = sh(0)?;
            let b = sh(1)?;
            let lo = sh(2)?;
            let hi = sh(3)?;
            s0 += self.run.om_v[(7 * i) as usize] * a * a
                + self.run.om_v[(7 * i + 1) as usize] * a * b
                + self.run.om_v[(7 * i + 2) as usize] * b * b
                + self.run.om_v[(7 * i + 3) as usize] * lo * a
                + self.run.om_v[(7 * i + 4) as usize] * lo * b
                + self.run.om_v[(7 * i + 5) as usize] * hi * a
                + self.run.om_v[(7 * i + 6) as usize] * hi * b;
        }
        let token_row = crate::fp::model::psc_row(f, &mut self.run.t_v.p, s0, &Aff::default())?;
        f.lin(token_row);
        for i in 0..8u64 {
            let row = self.row5_at("r_tV", i);
            f.lin(Aff::slot(self.run.t_v.s_end + (4 * i) as u32) - row.a);
            f.lin(Aff::slot(self.run.t_v.s_end + (4 * i + 1) as u32) - row.b);
            f.lin(Aff::slot(self.run.t_v.s_end + (4 * i + 2) as u32) - row.r_lo);
            f.lin(Aff::slot(self.run.t_v.s_end + (4 * i + 3) as u32) - row.r_hi);
        }
        let mut rows = Vec::new();
        for i in 0..8u64 {
            rows.push(self.row5_at("P_V", i));
        }
        for (i, row) in rows.iter().enumerate() {
            for s in 0..9 {
                let out = Aff::slot(self.run.s_w + (9 * i + s) as u32)
                    + row.c[s].clone()
                    - row.digits[s].clone()
                    - row.c[s + 1].clone().scaled(pw(8));
                #[cfg(feature = "std")]
                if std::env::var_os("DUMP_SHARED").is_some() {
                    eprintln!("RUST_CARRY i={} s={} cs={:016x} dg={:016x} next={:016x} out={:016x}", i, s, row.c[s].c.0[0], row.digits[s].c.0[0], row.c[s + 1].c.0[0], out.c.0[0]);
                }
                f.lin(out);
            }
        }
        for (i, row) in rows.iter().enumerate() {
            let pr: Vec<Aff> = (0..7)
                .map(|j| Aff::slot(self.run.s_pv + (7 * i + j) as u32))
                .collect();
            let n = i % 2;
            let mn = kl(u64::from(!self.run.fixture.first) * LAM + i as u64 / 2, if n == 0 { 7 } else { 9 });
            let two = Aff::constant(pw(mn + 1));
            for out in led_lines5(row, &pr, E::from_base(F(self.run.fixture.eps[n])), &two) {
                f.lin(out);
            }
        }
        for i in 0..8u64 {
            f.lin(Aff::slot(self.run.t_t.s_end + i as u32) - self.row_r("rt", i));
        }
        for lm in 0..LAM_M {
            f.lin(Aff::slot(self.run.s_sc + lm as u32) - self.sh_a("sc", "rt", lm, 0));
        }
        for lm in 0..LAM_M {
            for e in 0..NE {
                f.lin(Aff::slot(self.run.s_sel + (lm * NE + e) as u32) - self.sh_a("sel", "rt", lm, e));
            }
        }
        f.lin(stack.hd.row.clone());
        {
            let e = eq_top(&stack.hd.r[..stack.hd.r.len() - 8]);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                for (i, x) in e.iter().enumerate() {
                    eprintln!("RUST_HD_EQ {} {:016x}", i, x.0[0]);
                }
            }
            let offset = self.table_at(0, "Z_a");
            let ts = self.run.fixture.tensor("s");
            let tu = self.run.fixture.tensor("u");
            let mut a = Aff::slot(stack.hd.s_l);
            let mut b = Aff::slot(stack.hd.s_r);
            for layer in 0..LAM {
                for kh in 0..F_WIDTH / S2 {
                    let weight = e[(layer * (F_WIDTH / S2) + kh) as usize];
                    a.add(&self.long_a(offset, ts, layer, kh), weight);
                    b.add(&self.long_a(offset, tu, layer, kh), weight);
                }
            }
            f.lin(a);
            f.lin(b);
        }
        if let Some(hd2) = &stack.hd2 {
            f.lin(hd2.row.clone());
            let e = eq_top(&hd2.r[..hd2.r.len() - 8]);
            let offset = self.table_at(0, "Z_b");
            let ts = self.run.fixture.tensor("s");
            let tu = self.run.fixture.tensor("u");
            let mut a = Aff::slot(hd2.s_l);
            let mut b = Aff::slot(hd2.s_r);
            for lm in 0..LAM_M {
                for kh in 0..FE / S2 {
                    let weight = e[(lm * (FE / S2) + kh) as usize];
                    a.add(&self.long_a(offset, ts, 2 * lm + 1, FE / S2 + kh), weight);
                    b.add(&self.long_a(offset, tu, 2 * lm + 1, FE / S2 + kh), weight);
                }
            }
            f.lin(a);
            f.lin(b);
        }
        for lm in 0..LAM_M {
            let mut a = Aff::constant(-E::ONE);
            for e in 0..NE {
                a.add(&self.sh_a("sel", "zt", lm, e), E::ONE);
            }
            f.lin(a);
        }
        {
            let t_b = [self.run.tables.at("b-bits-0").unwrap(), self.run.tables.at("b-bits-1").unwrap()];
            let t_p = self.run.tables.at("b-pin").unwrap();
            let w = 16 + 5 - 13;
            let mut at = [Aff::default(), Aff::default()];
            for e in 0..2 {
                let ec = eq_top(&self.run.forest.zb[e][16..]);
                let et = eq_top(&self.run.forest.zb[e][..w]);
                for c in 0..self.run.forest.calls.len() {
                    for th in 0..1usize << w {
                        at[e].add(&self.u3at(t_b[e], ((c << w) + th) as u64), ec[c] * et[th]);
                    }
                }
            }
            let padding = self.run.forest.empty_keys() + self.run.forest.pub_counts.iter().sum::<u64>();
            let mut pin = Aff::constant(-E::from_base(F((1u64 << self.run.forest.v()) - padding)));
            for column in 0..(self.run.forest.calls.len() << w) {
                pin = pin + self.u3at(t_p, column as u64);
            }
            let plane_row = crate::fp::model::psc_row(
                f,
                &mut self.run.forest.leaves,
                E::ONE,
                &at[0].clone().scaled(self.run.forest.leaves_a),
            )?;
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_MARK plane_row n={}", f.n_claim);
            }
            f.lin(plane_row);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_MARK plane_bit n={}", f.n_claim);
            }
            f.lin(Aff::slot(self.run.forest.s_bit) - at[1].clone());
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_MARK plane_pin n={} constant={:016x}", f.n_claim, pin.c.0[0]);
            }
            f.lin(pin);
        }
        if self.run.fixture.first {
            let e = eq_top(&self.run.tkh);
            let offset = self.table_at(8, "Z_E");
            let mut a = Aff::slot(merged.s_ve).scaled(-E::ONE);
            for kh in 0..D / S2 {
                a.add(&self.xb_a(offset, kh), e[kh as usize]);
            }
            f.lin(a);
        }
        Ok(())
    }

    pub fn objects_after_smallness(
        &mut self,
        stack: &StackOut,
        merged: &mut WeightMerge,
        f: &mut HFork,
    ) -> Result<(), FpError> {
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_ALIN alin={:016x} blin={:016x}", stack.a_lin.c.0[0], stack.b_lin.c.0[0]);
        }
        f.lin(stack.row_s.clone());
        f.lin(Aff::slot(stack.s_a0) - stack.a_lin.clone());
        f.lin(Aff::slot(stack.s_b0) - stack.b_lin.clone());
        let merge_row = crate::fp::model::psc_row(f, &mut merged.p, merged.fvp, &Aff::default())?;
        f.lin(merge_row);
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Run2 {
    tree: u32,
    row: u32,
    first: u64,
    coef: E,
    weights: Vec<E>,
    w0: usize,
    count: u64,
}

impl LayerCheck<'_> {
    fn zz_tree(&self) -> u32 {
        if self.head { 6 } else if self.run.fixture.first { 10 } else { 9 }
    }

    fn bz_tree(&self) -> u32 {
        if self.head { 7 } else if self.run.fixture.first { 11 } else { 10 }
    }

    fn gt_tree(&self, matrix: usize) -> u32 {
        if self.head {
            [1, 2, 3, 4, 5][matrix]
        } else if self.run.fixture.first {
            [1, 2, 3, 4, 5, 6, 9, 8, 7][matrix]
        } else {
            [1, 2, 3, 4, 5, 6, 8, 7, 11][matrix]
        }
    }

    pub fn init_trees(&self, f: &mut HFork) {
        if !f.trees.is_empty() {
            return;
        }
        f.trees.push(crate::fp::model::HTree::new("U3".into(), 'N', 1, self.run.fixture.n_u3, 1));
        if self.head {
            for (index, name) in ["hf", "hd", "hr", "bH", "xbL"].iter().enumerate() {
                let m = &self.run.fixture.mats[index];
                f.trees.push(crate::fp::model::HTree::new((*name).into(), 'P', (m.m + 2) / 3, m.k, 1));
            }
            f.trees.push(crate::fp::model::HTree::new("ZZ".into(), 'N', 128, 2 * self.mz(), 4));
            f.trees.push(crate::fp::model::HTree::new("BZ".into(), 'N', 32, 2048, 3));
            return;
        }
        let names: &[&str] = if self.run.fixture.first {
            &["af0", "af1", "af2", "ad0", "ar", "b", "xbI", "xbO", "qa"]
        } else {
            &["af0", "af1", "af2", "ad0", "ar", "b", "xbO", "qa"]
        };
        for (index, name) in names.iter().enumerate() {
            let matrix = if self.run.fixture.first {
                [0, 1, 2, 3, 4, 5, 8, 7, 6][index]
            } else {
                [0, 1, 2, 3, 4, 5, 7, 6, 8][index]
            };
            let m = &self.run.fixture.mats[matrix];
            f.trees.push(crate::fp::model::HTree::new((*name).into(), 'P', (m.m + 2) / 3, m.k, 1));
        }
        f.trees.push(crate::fp::model::HTree::new("ZZ".into(), 'N', 128, 2 * self.mz(), 4));
        f.trees.push(crate::fp::model::HTree::new("BZ".into(), 'N', 32, 2048, 3));
        // array_of lists xbI after BZ for later layers (M8), not among the
        // named matrices.
        if !self.run.fixture.first {
            f.trees.push(crate::fp::model::HTree::new("xbI".into(), 'P', (8192 + 2) / 3, 20, 1));
        }
    }

    fn t2lin(&self, f: &mut HFork, form: Aff, runs: &[Run2]) {
        let weight = f.kap();
        f.push(weight, &form);
        for run in runs {
            for i in 0..run.count as usize {
                let coefficient = weight * run.coef * run.weights[run.w0 + i];
                let at = (run.first + i as u64) as usize;
                if at >= f.trees[run.tree as usize].rows[run.row as usize].l.len() {
                    panic!("bad run tree={} row={} first={} count={} index={} columns={}", run.tree, run.row, run.first, run.count, at, f.trees[run.tree as usize].rows[run.row as usize].l.len());
                }
                f.trees[run.tree as usize].rows[run.row as usize].l[at] += coefficient;
            }
        }
    }

    fn lead(&self, f: &HFork, first: u64) -> E {
        let mut out = E::ONE;
        let mut square = f.theta;
        let mut exponent = (f.plan_claims - f.n_claim) as u64 + first;
        while exponent != 0 {
            if exponent & 1 != 0 {
                out *= square;
            }
            square *= square;
            exponent >>= 1;
        }
        out
    }

    fn off(&self, matrix: usize) -> u64 {
        self.run.fixture.mats[..matrix].iter().map(|m| m.m).sum()
    }

    fn mz(&self) -> u64 {
        self.run.fixture.mats.iter().map(|m| m.m).sum()
    }

    fn lin_join9(&self, forest: &crate::fp::forest::Forest, f: &mut HFork) -> Result<(), FpError> {
        let mut form = Aff::default();
        let mut runs = Vec::new();
        for q in 0..2 {
            form.c += forest.e[q] * (forest.alpha - scale_index(forest.theta, q as u64));
            let call = &forest.calls[q];
            let c0 = q as u64 * self.mz();
            let zr = &call.zk[7..];
            let ec = eq_top(zr);
            let coef = -forest.e[q] * crate::fp::smallness::eq_one(&call.zk, 0);
            runs.push(Run2 {
                tree: self.zz_tree(),
                row: q as u32,
                first: c0,
                coef,
                weights: ec,
                w0: 0,
                count: self.mz(),
            });
        }
        let ey = eq_top(&forest.zy);
        for t in forest.nt as usize..1usize << forest.r_k {
            form.c += ey[t] * (forest.alpha - forest.padding_key(0));
        }
        let laid = forest.pub_counts.clone();
        if laid != forest.pub_counts {
            return Err(FpError::Range {
                context: "smallness-padding",
            });
        }
        self.t2lin(f, form - forest.c_end.clone(), &runs);
        Ok(())
    }

    fn lin_bits9(&self, forest: &mut crate::fp::forest::Forest, f: &mut HFork) {
        let kb = 1u64 << (forest.k_t + forest.c_k);
        let w0 = eq_top(&forest.zb[0][5..]);
        let w1 = eq_top(&forest.zb[1][5..]);
        let bit_factor = forest.leaves.eps * forest.leaves_a;
        let row = crate::fp::model::psc_row(f, &mut forest.leaves, E::ONE, &Aff::default()).unwrap();
        self.t2lin(
            f,
            row,
            &[Run2 {
                tree: self.bz_tree(),
                row: 0,
                first: 0,
                coef: bit_factor,
                weights: w0,
                w0: 0,
                count: kb,
            }],
        );
        self.t2lin(
            f,
            Aff::slot(forest.s_bit),
            &[Run2 {
                tree: self.bz_tree(),
                row: 1,
                first: 0,
                coef: -E::ONE,
                weights: w1,
                w0: 0,
                count: kb,
            }],
        );
        let padding = forest.empty_keys() + forest.pub_counts.iter().sum::<u64>();
        let w: Vec<E> = (0..kb).map(|u| E::from_base(crate::fp::field::FieldElement(1u64 << (u & 31)))).collect();
        self.t2lin(
            f,
            Aff::constant(-E::from_base(crate::fp::field::FieldElement((1u64 << forest.v()) - padding))),
            &[Run2 {
                tree: self.bz_tree(),
                row: 2,
                first: 0,
                coef: E::ONE,
                weights: w,
                w0: 0,
                count: kb,
            }],
        );
    }

    pub fn pending(&self, shape: &crate::fp::key_public::LinShape, merged: &WeightMerge) -> Result<crate::fp::wf::TensorRow, FpError> {
        crate::fp::weights::pending_row(&crate::fp::weights::ws_lay(), merged, merged.rec.len(), shape)
    }

    pub fn smallness(
        &mut self,
        digits: &mut crate::fp::forest::Forest,
        cf_f: &[i8],
        f: &mut HFork,
    ) -> Result<(), FpError> {
        self.before_smallness = f.n_claim;
        self.reads_by_matrix = vec![0; self.run.fixture.mats.len()];
        self.init_trees(f);
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_SMALLNESS_DIGITS n={} rows={} first={:016x}", f.n_claim, digits.rows.len(), digits.rows.first().map(|r| r.c.0[0]).unwrap_or_default());
        }
        for row in &digits.rows {
            f.lin(row.clone());
        }
        self.lin_join9(digits, f)?;
        self.lin_bits9(digits, f);
        let mut vt = Vec::with_capacity(128);
        vt.push(E::ONE);
        for i in 1..128 {
            vt.push(vt[i - 1] * f.theta);
        }
        let vp = vt.clone();
        let step = f.theta.pow(128);
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_ROW_STEP step={:016x}", step.0[0]);
        }
        self.vt = vt.clone();
        self.vp = vp.clone();
        let mut eq_first = vec![0u64];
        let mut row_x = Vec::new();
        let mut lo_x = Vec::new();
        for m in &self.run.fixture.mats {
            let ms = (m.m + 2) / 3;
            eq_first.push(eq_first[eq_first.len() - 1] + (128 << crate::fp::tensor::ceil_log2(ms)));
            let mut row = Vec::with_capacity(ms as usize);
            row.push(E::ONE);
            for i in 1..ms {
                row.push(row[i as usize - 1] * step);
            }
            row_x.push(row);
            lo_x.push(vec![E::ONE]);
        }
        let cf_first: Vec<u64> = self.run.fixture.mats.iter().scan(0, |at, m| {
            let out = *at;
            *at += m.k;
            Some(out)
        }).collect();
        let mut gam_x = Vec::new();
        let mut gam_v = Vec::new();
        for (mu, m) in self.run.fixture.mats.iter().enumerate() {
            let mut gx = vec![E::ZERO; m.k as usize];
            let mut gv = vec![E::ZERO; m.k as usize];
            for c in 0..m.k as usize {
                let cf = &cf_f[(cf_first[mu] + c as u64) as usize * 128..][..128];
                for (u, &trit) in cf.iter().enumerate() {
                    if trit != 0 {
                        let value = if trit < 0 { -vt[u] } else { vt[u] };
                        gx[c] += value;
                        gv[c] += value;
                    }
                }
            }
            gam_x.push(gx.clone());
            gam_v.push(gv.clone());
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_GAM mu={} k={} first={} gx0={:016x} gv0={:016x}", mu, m.k, cf_first[mu], gx[0].0[0], gv[0].0[0]);
            }
        }
        let sx = vp.iter().fold(E::ZERO, |a, &b| a + b);
        let sv = vt.iter().fold(E::ZERO, |a, &b| a + b);
        let mz = self.mz();
        self.row_x = row_x.clone();
        for (mu, m) in self.run.fixture.mats.iter().enumerate() {
            let s = crate::fp::smallness::Shape {
                m: m.m,
                k: m.k,
                ms: (m.m + 2) / 3,
                j: 1,
            };
            let wl = crate::fp::smallness::tie_left(&s, &row_x[mu], &lo_x[mu]);
            let wr = crate::fp::smallness::tie_right(&s, &lo_x[mu], &gam_x[mu]);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_TIE_LEFT mu={} m={} ms={} wl1={:016x} wl2={:016x}", mu, m.m, s.ms, wl[1].0[0], wl[2].0[0]);
            }
            let lead = self.lead(f, eq_first[mu]);
            let sw = wl.iter().fold(E::ZERO, |a, &b| a + b);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_TIE mu={} lead={:016x} sx={:016x} sw={:016x} wl0={:016x} wr0={:016x} target={:016x}", mu, lead.0[0], sx.0[0], sw.0[0], wl[0].0[0], wr[0].0[0], (-lead * sx * sw * pw(17)).0[0]);
            }
            self.t2lin(
                f,
                Aff::constant(-lead * sx * sw * pw(17)),
                &[
                    Run2 {
                        tree: self.zz_tree(),
                        row: 3,
                        first: self.off(mu),
                        coef: lead,
                        weights: wl.clone(),
                        w0: 0,
                        count: m.m,
                    },
                    Run2 {
                        tree: self.zz_tree(),
                        row: 3,
                        first: mz + self.off(mu),
                        coef: lead * pw(10),
                        weights: wl.clone(),
                        w0: 0,
                        count: m.m,
                    },
                    Run2 {
                        tree: self.gt_tree(mu),
                        row: 0,
                        first: 0,
                        coef: -lead,
                        weights: wr,
                        w0: 0,
                        count: m.k,
                    },
                ],
            );
        }
        let mut rd_t = vec![0u8; self.run.fixture.n_u3 as usize];
        let mut n_x4 = 0u64;
        for (mu, m) in self.run.fixture.mats.iter().enumerate() {
            let gc = &gam_v[mu];
            for fun in &self.run.tables.fun {
                if fun.matrix != mu || fun.l == 0 || m.m % fun.l != 0 || fun.phi.len() != fun.l as usize {
                    continue;
                }
                let nsg = m.m / fun.l;
                let sphi = fun.phi.iter().fold(E::ZERO, |a, &b| a + b);
                for sg in 0..nsg {
                    let lead = self.lead(f, eq_first[self.run.fixture.mats.len()] + n_x4 * 128);
                    let mut form = Aff::constant(-lead * sphi * sv * pw(17));
                    for c in 0..m.k {
                        let at = fun.at + c * nsg + sg;
                        if at as usize >= rd_t.len() || rd_t[at as usize] != 0 || gc[c as usize].is_zero() {
                            return Err(FpError::Range {
                                context: "smallness-read",
                            });
                        }
                        rd_t[at as usize] = 1;
                        form.add(&self.u3abs(at), -lead * gc[c as usize]);
                    }
                    self.t2lin(
                        f,
                        form,
                        &[
                            Run2 {
                                tree: self.zz_tree(),
                                row: 2,
                                first: self.off(mu) + sg * fun.l,
                                coef: lead,
                                weights: fun.phi.clone(),
                                w0: 0,
                                count: fun.l,
                            },
                            Run2 {
                                tree: self.zz_tree(),
                                row: 2,
                                first: mz + self.off(mu) + sg * fun.l,
                                coef: lead * pw(10),
                                weights: fun.phi.clone(),
                                w0: 0,
                                count: fun.l,
                            },
                        ],
                    );
                    self.reads_by_matrix[mu] += 1;
                    n_x4 += 1;
                }
            }
        }
        if rd_t.iter().any(|&seen| seen == 0) {
            return Err(FpError::Range {
                context: "smallness-read",
            });
        }
        Ok(())
    }
}

impl LayerCheck<'_> {
    pub fn coefficients(&self, digits: &crate::fp::forest::Forest, f: &mut HFork) {
        f.trees[0].rows[0].coef = vec![E::ONE];
        f.trees[0].rows[0].row_ones = true;
        f.trees[0].rows[0].row_form = true;
        let zz = self.zz_tree() as usize;
        let reads = 2usize;
        let ties = 3usize;
        f.trees[zz].rows[reads].coef = self.vt.clone();
        f.trees[zz].rows[reads].row_form = true;
        f.trees[zz].rows[reads].row_lo = vec![E::ONE; 7];
        f.trees[zz].rows[reads].row_hi = (0..7).map(|i| {
            let mut value = f.theta;
            for _ in 0..i {
                value *= value;
            }
            value
        }).collect();
        f.trees[zz].rows[ties].coef = self.vp.clone();
        f.trees[zz].rows[ties].row_form = true;
        f.trees[zz].rows[ties].row_lo = f.trees[zz].rows[reads].row_lo.clone();
        f.trees[zz].rows[ties].row_hi = f.trees[zz].rows[reads].row_hi.clone();
        for q in 0..2 {
            let call = &digits.calls[q];
            f.trees[zz].rows[q].coef = eq_top(&call.zk[..7]);
            f.trees[zz].rows[q].row_form = true;
            f.trees[zz].rows[q].row_point = call.zk[..7].to_vec();
            f.trees[zz].rows[q].row_first = 0;
        }
        let bz = self.bz_tree() as usize;
        for e in 0..2 {
            f.trees[bz].rows[e].coef = eq_top(&digits.zb[e][..5]);
            f.trees[bz].rows[e].row_form = true;
            f.trees[bz].rows[e].row_point = digits.zb[e][..5].to_vec();
            f.trees[bz].rows[e].row_first = 0;
        }
        f.trees[bz].rows[2].coef = vec![E::ONE; 32];
        f.trees[bz].rows[2].row_ones = true;
        f.trees[bz].rows[2].row_form = true;
        for (mu, tree) in (0..self.run.fixture.mats.len()).map(|mu| (mu, self.gt_tree(mu))) {
            let tree = tree as usize;
            f.trees[tree].rows[0].coef = self.row_x[mu].clone();
            f.trees[tree].rows[0].row_form = true;
            f.trees[tree].rows[0].row_lo = vec![E::ONE; crate::fp::tensor::ceil_log2(f.trees[tree].r) as usize];
            f.trees[tree].rows[0].row_hi = (0..crate::fp::tensor::ceil_log2(f.trees[tree].r)).map(|i| {
                let mut value = self.lead(f, 128);
                for _ in 0..i {
                    value *= value;
                }
                value
            }).collect();
        }
    }

    fn array_of(&self) -> Vec<(usize, String)> {
        let own = if self.run.fixture.first { 0 } else { 2 };
        let mut out = vec![(own, "U3".into())];
        if self.run.fixture.first {
            out.extend([
                (own, "M0".into()),
                (own, "M1".into()),
                (own, "M2".into()),
                (own, "M3".into()),
                (own, "M4".into()),
                (own, "M5".into()),
                (own, "M8".into()),
                (own, "M7".into()),
                (own, "M6".into()),
            ]);
        } else {
            out.extend([
                (own, "M0".into()),
                (own, "M1".into()),
                (own, "M2".into()),
                (own, "M3".into()),
                (own, "M4".into()),
                (own, "M5".into()),
                (own, "M7".into()),
                (own, "M6".into()),
            ]);
        }
        out.push((own, "ZZ".into()));
        out.push((own, "BZ".into()));
        // M8 (xbI) is claimed only in the first layer: its message has life 0,
        // so later layers use its table but have no claim for it.
        out
    }

    fn row_weights(&self, tree: &crate::fp::model::HTree, row: &crate::fp::model::HRow) -> Result<crate::fp::req_rows::IndexWeights, FpError> {
        let bits = crate::fp::tensor::ceil_log2(tree.r);
        let np = row.row_point.len() as u32;
        let given = !row.row_lo.is_empty() || !row.row_hi.is_empty();
        if np > bits
            || (row.row_ones || given) && (np != 0 || row.row_first != 0)
            || row.row_first & ((1u64 << np) - 1) != 0
            || row.row_first >= tree.r
            || given && (row.row_ones || row.row_lo.len() != bits as usize || row.row_hi.len() != bits as usize)
        {
            return Err(FpError::Range {
                context: "row-form",
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

    pub fn door_h(&self, f: &HFork) -> Result<crate::fp::req_rows::ReqEquation, FpError> {
        if f.n_claim != f.plan_claims {
            return Err(FpError::Range {
                context: "request-claim-count",
            });
        }
        let (on_masks, constant) = f.settle()?;
        let array_of = self.array_of();
        let mut equation = crate::fp::req_rows::ReqEquation {
            terms: vec![],
            on_masks,
            target: -constant,
        };
        for (ti, tree) in f.trees.iter().enumerate().take(array_of.len()) {
            let (born, array) = &array_of[ti];
            for row in &tree.rows {
                if row.coef.len() != tree.r as usize || row.l.len() != tree.k as usize {
                    return Err(FpError::Range {
                        context: "request-row-form",
                    });
                }
                if row.row_form {
                    equation.terms.push(crate::fp::req_rows::ReqTerm {
                        born: *born,
                        array: array.clone(),
                        scale: E::ONE,
                        on_rows: self.row_weights(tree, row)?,
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
