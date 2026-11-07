//! Lookup forests used by model subproofs (`vf_lookup_argument.h`).

use alloc::{string::String, vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
    model::{psc_begin, psc_ends, psc_row, round_in, Aff, HFork, Psc, RoundForm, RoundName, Tr},
    tensor::eq_table,
};

fn eq_top(point: &[E]) -> Vec<E> {
    crate::fp::tensor::eq_table_rev(point)
}

#[derive(Debug, Clone)]
pub struct Call {
    pub name: String,
    pub n: u32,
    pub k: u32,
    pub function: u32,
    pub t0: u64,
    pub tau: Option<Vec<u64>>,
    pub alpha: E,
    pub zk: Vec<E>,
}

impl Call {
    fn new(name: &str, n: u32, k: u32, function: u32, t0: u64, tau: Option<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            n,
            k,
            function,
            t0,
            tau,
            alpha: E::ZERO,
            zk: vec![],
        }
    }

    pub fn tau_at(&self, index: u64) -> E {
        E::from_base(F(self.tau.as_ref().map_or(index, |tau| tau[index as usize])))
    }
}

#[derive(Debug)]
pub struct ForestInput {
    pub calls: Vec<Call>,
    pub pub_counts: Vec<u64>,
    pub lid: u32,
}

pub struct Forest {
    pub calls: Vec<Call>,
    pub d_t: u32,
    pub k_t: u32,
    pub nt: u32,
    pub r_k: u32,
    pub c_k: u32,
    pub iota: Vec<u32>,
    pub first: Vec<u32>,
    pub call_of: Vec<u32>,
    pub pub_counts: Vec<u64>,
    pub theta: E,
    pub alpha: E,
    pub a: E,
    pub rows: Vec<Aff>,
    pub leaves: Psc,
    pub leaves_a: E,
    pub s_bit: u32,
    pub zx: Vec<E>,
    pub zy: Vec<E>,
    pub e: Vec<E>,
    pub zb: [Vec<E>; 2],
    pub t_t: Vec<E>,
    pub c_end: Aff,
}

impl Forest {
    pub fn v(&self) -> u32 {
        self.d_t + self.r_k
    }
    pub fn d(&self) -> u32 {
        self.k_t + self.c_k
    }
    pub fn padding_key(&self, call: usize) -> E {
        self.calls[call].tau_at(self.calls[call].t0) + scale_index(self.theta, call as u64)
    }
    pub fn empty_keys(&self) -> u64 {
        ((1u64 << self.r_k) - self.nt as u64) << self.d_t
    }
}

fn scale_index(value: E, index: u64) -> E {
    value * E::from_base(F(index))
}

fn pow2_scale(mut value: E, shift: u64) -> E {
    let mut out = value;
    for _ in 0..shift {
        out += out;
    }
    out
}

fn pow(mut value: E, mut exponent: u64) -> E {
    let mut out = E::ONE;
    while exponent != 0 {
        if exponent & 1 != 0 {
            out *= value;
        }
        value *= value;
        exponent >>= 1;
    }
    out
}

fn order_default(mu: u32) -> Vec<u32> {
    (0..mu).rev().collect()
}

fn order_forest_range(mu: u32, r_k: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for v in [2, 1, 0, 5, 4, 3, 8, 7, 6] {
        out.push(r_k + v);
    }
    for v in (r_k + 9..mu).rev() {
        out.push(v);
    }
    for v in (0..r_k).rev() {
        out.push(v);
    }
    out
}

#[derive(Debug, Clone)]
pub struct ProductLevel {
    pub r: Vec<E>,
    pub s_l: u32,
    pub s_r: u32,
    pub tau: E,
    pub row: Aff,
}

pub fn read_product_level(
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
    mu: u32,
    rho: &[E],
    start: &mut Aff,
    order: &[u32],
) -> Result<ProductLevel, FpError> {
    let mut p = psc_begin(f, tr, slot, start.clone())?;
    let mut r = vec![E::ZERO; mu as usize];
    for &v in order {
        let index = mu - 1 - v;
        r[index as usize] = round_in(f, tr, slot, RoundForm::Eq2, &mut p, rho[index as usize], 0)?;
    }
    let s_l = psc_ends(f, tr, slot, &mut p, 2, 2)?;
    let s_r = s_l + 1;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_PL slot={} nclaim={} psc_first={} eps={:016x} target={:016x} claim={:016x} start={:016x}", slot, f.n_claim, p.first, p.eps.0[0], p.target.0[0], p.claim.0[0], p.start.c.0[0]);
    }
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        let target_before = f.public_part(&start.clone())?;
        eprintln!("RUST_PL_START slot={} public={:016x} slots={:?}", slot, target_before.0[0], start.s);
    }
    let tau = tr.answer(RoundName::Line)?;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_PL_END slot={} s_l={} s_r={} l={:016x} r={:016x} tau={:016x}", slot, s_l, s_r, f.shown(s_l)?.0[0], f.shown(s_r)?.0[0], tau.0[0]);
    }
    let row = psc_row(
        f,
        &mut p,
        f.shown(s_l)? * f.shown(s_r)?,
        &Aff::default(),
    )?;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() && (mu == 25 || mu == 18) {
        eprintln!("RUST_LEVEL first={} mu={} start={:?}", p.first, mu, p.start.s.iter().map(|x| (x.0, x.1.0[0])).collect::<Vec<_>>());
    }
    Ok(ProductLevel { r, s_l, s_r, tau, row })
}

pub fn hand_down(level: &ProductLevel, rho: &mut Vec<E>, start: &mut Aff) {
    *rho = vec![level.tau];
    rho.extend_from_slice(&level.r);
    *start = Aff::slot(level.s_l)
        .scaled(E::ONE - level.tau)
        .plus(&Aff::slot(level.s_r).scaled(level.tau));
}

fn forest_shape(input: &ForestInput) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>, u32, u32, u32, u32, u32), FpError> {
    let calls = &input.calls;
    if calls.is_empty() || calls.len() > 16 {
        return Err(FpError::Range { context: "forest-calls" });
    }
    let d_t = calls.iter().map(|call| call.n).fold(input.lid, core::cmp::min);
    let k_t = calls.iter().map(|call| call.k).max().unwrap_or(0);
    if d_t < 10 {
        return Err(FpError::Range { context: "forest-depth" });
    }
    let iota: Vec<u32> = calls.iter().map(|call| call.n - d_t).collect();
    let nt: u32 = iota.iter().map(|&bits| 1u32 << bits).sum();
    if nt > 256 {
        return Err(FpError::Range { context: "forest-trees" });
    }
    let r_k = crate::fp::tensor::ceil_log2(nt as u64);
    let c_k = crate::fp::tensor::ceil_log2(calls.len() as u64);
    let mut order: Vec<usize> = (0..calls.len()).collect();
    order.sort_by_key(|&index| core::cmp::Reverse(iota[index]));
    let mut first = vec![0u32; calls.len()];
    let mut call_of = vec![u32::MAX; nt as usize];
    let mut at = 0u32;
    for index in order {
        first[index] = at;
        for _ in 0..1u32 << iota[index] {
            call_of[at as usize] = index as u32;
            at += 1;
        }
    }
    Ok((iota, first, call_of, d_t, k_t, nt, r_k, c_k))
}

fn forest_elems(v: u32, d: u32) -> u32 {
    let mut out = 7 + 30 * 6;
    for mu in 0..v {
        out += 5 + 2 * mu;
    }
    for l in 0..d {
        out += 15 + 2 * l;
    }
    out += 1 + 2 * d + 5;
    out + 14 + 2 * d
}

pub fn forest_pad(f: &mut HFork, tr: &mut Tr, slot: &mut u32, mut input: ForestInput) -> Result<Forest, FpError> {
    let (iota, first, call_of, d_t, k_t, nt, r_k, c_k) = forest_shape(&input)?;
    let nc = input.calls.len();
    let v = d_t + r_k;
    let d = k_t + c_k;
    let m = 5 + d;
    if input.pub_counts.len() != nt as usize
        || input.pub_counts.iter().any(|&count| count > 1u64 << d_t)
    {
        return Err(FpError::Range { context: "forest-padding" });
    }
    for (call_index, call) in input.calls.iter().enumerate() {
        let padding: u64 = (0..1u32 << iota[call_index])
            .map(|b| input.pub_counts[(first[call_index] + b) as usize])
            .sum();
        if padding >> (d_t + iota[call_index]) != 0 {
            return Err(FpError::Range { context: "forest-only-padding" });
        }
    }
    let theta = if nc > 1 {
        tr.answer(RoundName::LkpShift)?
    } else {
        E::ZERO
    };
    let alpha = tr.answer(RoundName::Lkp)?;
    let pad_key = |index: usize, theta: E| {
        let call = &input.calls[index];
        call.tau_at(call.t0) + scale_index(theta, index as u64)
    };
    let empty_keys = ((1u64 << r_k) - nt as u64) << d_t;
    let mut a = pow(alpha - pad_key(0, theta), empty_keys);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_FOREST_INIT empty={} nt={} call0={} pub0={} key0={:016x} a0={:016x} theta={:016x}", empty_keys, nt, call_of.first().copied().unwrap_or(0), input.pub_counts.first().copied().unwrap_or(0), pad_key(0, theta).0[0], a.0[0], theta.0[0]);
    }
    for y in 0..nt as usize {
        a *= pow(alpha - pad_key(call_of[y] as usize, theta), input.pub_counts[y]);
    }
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_FOREST_A a={:016x} alpha={:016x}", a.0[0], alpha.0[0]);
    }
    let slot0 = *slot;
    let mut rows = Vec::new();
    let mut s_g = [0u32; 32];
    let mut s_z = 0;
    let mut s_p = 0;
    for j in 0..31 {
        let start = if j == 0 { Aff::default() } else { Aff::slot(s_p) };
        let mut p = psc_begin(f, tr, slot, start)?;
        let at = psc_ends(f, tr, slot, &mut p, 3, if j == 0 { 3 } else { 2 })?;
        if j == 0 {
            s_z = at;
        }
        s_p = if j == 0 { at + 1 } else { at };
        s_g[j] = s_p + 1;
        let power = f.shown(s_p)? * f.shown(s_p)? * f.shown(s_g[j])?;
        let s0 = if j == 0 {
            f.shown(s_z)? - a * power
        } else {
            power
        };
        rows.push(psc_row(f, &mut p, s0, &Aff::default())?);
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() && j == 0 {
            eprintln!("RUST_FOREST_TIE0 sz={:016x} power={:016x} s0={:016x} row={:016x}", f.shown(s_z)?.0[0], power.0[0], s0.0[0], rows.last().unwrap().c.0[0]);
        }
    }
    s_g[31] = s_p;
    let by_variable = tr.point(RoundName::LkpPlanes, 5)?;
    let mut rho_t: Vec<E> = by_variable.into_iter().rev().collect();
    let eq = eq_top(&rho_t);
    let mut ct = Aff::default();
    for j in 0..32 {
        ct.add(&Aff::slot(s_g[j]), eq[j]);
    }
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_RHO_T {:?}", rho_t);
        eprintln!("RUST_CT_EQ {:?}", eq);
    }
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_SG 0={} val={:016x} 1={} val={:016x} 2={} val={:016x} 3={} val={:016x}", s_g[0], f.shown(s_g[0])?.0[0], s_g[1], f.shown(s_g[1])?.0[0], s_g[2], f.shown(s_g[2])?.0[0], s_g[3], f.shown(s_g[3])?.0[0]);
    }
    let mut rho = Vec::new();
    let mut start = Aff::slot(s_z);
    for mu in 0..v {
        let order = if mu + 1 == v {
            order_forest_range(mu, r_k)
        } else {
            order_default(mu)
        };
        let level = read_product_level(f, tr, slot, mu, &rho, &mut start, &order)?;
        hand_down(&level, &mut rho, &mut start);
        rows.push(level.row);
    }
    let zx = rho[..d_t as usize].to_vec();
    let zy = rho[d_t as usize..].to_vec();
    let c_end = start;
    let mut table_start = ct;
    for l in 0..d {
        let level = read_product_level(f, tr, slot, 5 + l, &rho_t, &mut table_start, &order_default(5 + l))?;
        hand_down(&level, &mut rho_t, &mut table_start);
        rows.push(level.row);
    }
    if rho_t.len() != m as usize {
        return Err(FpError::Range {
            context: "forest-point",
        });
    }
    let mut leaves = psc_begin(f, tr, slot, table_start)?;
    let mut zb0 = Vec::with_capacity(m as usize);
    for j in 0..m {
        zb0.push(round_in(
            f,
            tr,
            slot,
            RoundForm::Eq,
            &mut leaves,
            rho_t[j as usize],
            if j < d { 2 } else { 1 },
        )?);
    }
    let by_variable = tr.point(RoundName::Zero, m as usize)?;
    let r5: Vec<E> = by_variable.into_iter().rev().collect();
    let mut bits = psc_begin(f, tr, slot, Aff::default())?;
    let mut zb1 = Vec::with_capacity(m as usize);
    for &point in &r5 {
        zb1.push(round_in(f, tr, slot, RoundForm::Eq2, &mut bits, point, 0)?);
    }
    let s_bit = psc_ends(f, tr, slot, &mut bits, 2, 1)?;
    rows.push(psc_row(
        f,
        &mut bits,
        f.shown(s_bit)? * f.shown(s_bit)? - f.shown(s_bit)?,
        &Aff::default(),
    )?);
    if *slot != slot0 + forest_elems(v, d) || rows.len() != (31 + v + d + 1) as usize {
        return Err(FpError::Range {
            context: "forest-slot-count",
        });
    }
    let mut e = vec![E::ZERO; nc];
    let mut t_t = vec![E::ZERO; nc];
    for (c, call) in input.calls.iter_mut().enumerate() {
        if r_k < iota[c] {
            return Err(FpError::Range {
                context: "forest-place",
            });
        }
        let hi = r_k - iota[c];
        let pre = first[c] >> iota[c];
        let mut weight = E::ONE;
        for i in 0..hi {
            weight *= if (pre >> (hi - 1 - i)) & 1 != 0 {
                zy[i as usize]
            } else {
                E::ONE - zy[i as usize]
            };
        }
        e[c] = weight;
        call.alpha = alpha;
        call.zk = zy[hi as usize..].to_vec();
        call.zk.extend_from_slice(&zx);
        if call.zk.len() != call.n as usize {
            return Err(FpError::Range {
                context: "forest-point",
            });
        }
        let start = (k_t - call.k) as usize;
        let u = &zb0[start..start + call.k as usize];
        let mut table = E::ZERO;
        if let Some(tau) = &call.tau {
            let eq = eq_top(u);
            for (i, &value) in tau.iter().enumerate() {
                table += E::from_base(F(value)) * eq[i];
            }
        } else {
            for (i, &coordinate) in u.iter().enumerate() {
                table += pow2_scale(coordinate, (call.k as usize - 1 - i) as u64);
            }
        }
        t_t[c] = table;
    }
    let ec = eq_top(&zb0[k_t as usize..]);
    let mut row = E::ZERO;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        for c in 0..input.calls.len() {
            eprintln!("RUST_FOREST_TABLE c={} tT={:016x}", c, t_t[c].0[0]);
        }
    }
    for c in 0..1usize << c_k {
        let of = c.min(nc - 1);
        row += ec[c] * (t_t[of] + scale_index(theta, of as u64));
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_FOREST_TERM c={} ec={:016x} tT={:016x} theta={:016x}", c, ec[c].0[0], t_t[of].0[0], scale_index(theta, of as u64).0[0]);
        }
    }
    let leaves_a = alpha - E::ONE - row;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_FOREST_ROW row={:016x} leaves_a={:016x}", row.0[0], leaves_a.0[0]);
    }
    Ok(Forest {
        calls: input.calls,
        d_t,
        k_t,
        nt,
        r_k,
        c_k,
        iota,
        first,
        call_of,
        pub_counts: input.pub_counts,
        theta,
        alpha,
        a,
        rows,
        leaves,
        leaves_a,
        s_bit,
        zx,
        zy,
        e,
        zb: [zb0, zb1],
        t_t,
        c_end,
    })
}

pub fn forest_phi_b(forest: &Forest, lm: u32, e: usize) -> Vec<E> {
    let mut point = forest.zb[e][(forest.k_t + 5 - lm) as usize..forest.k_t as usize].to_vec();
    point.extend_from_slice(&forest.zb[e][(forest.k_t + forest.c_k) as usize..]);
    eq_top(&point)
}

pub fn forest_phi_p(lm: u32) -> Vec<E> {
    (0..1u64 << lm)
        .map(|row| E::from_base(F(1u64 << (row & 31))))
        .collect()
}

fn table(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

pub(crate) fn accepted_forest(link: usize, digits: bool) -> ForestInput {
    let range = |name: &str, n: u32, k: u32| Call::new(name, n, k, 0, 0, None);
    match (link, digits) {
        (0 | 2, false) => ForestInput {
            calls: vec![
                range("R16a", 20, 16),
                range("R16b", 20, 16),
                range("R16c", 20, 16),
                range("R8", 21, 8),
                Call::new("FC", 20, 16, 1, 32768, Some(table(include_bytes!("../../spikes/plans/forest/g_fc.bin")))),
                Call::new("FS", 20, 16, 1, 32768, Some(table(include_bytes!("../../spikes/plans/forest/g_fs.bin")))),
                Call::new("FL", 20, 16, 1, 32768, Some(table(include_bytes!("../../spikes/plans/forest/g_fl.bin")))),
                range("R16d", 20, 16),
            ],
            pub_counts: vec![465408, 465408, 524288, 196608, 588288, 523776, 786304, 786432, 786432],
            lid: 24,
        },
        (0 | 2, true) => ForestInput {
            calls: vec![range("R10z", 23, 10), range("R8d", 23, 8)],
            pub_counts: vec![0, 0],
            lid: 24,
        },
        (1 | 3, false) => ForestInput {
            calls: vec![
                Call::new("KR", 20, 16, 2, 0, Some(table(include_bytes!("../../spikes/plans/forest/kr.bin")))),
                Call::new("KFD", 20, 16, 1, 2307, Some(table(include_bytes!("../../spikes/plans/forest/kfd.bin")))),
                range("R16", 20, 16),
                Call::new("WH", 20, 16, 2, 0, Some(table(include_bytes!("../../spikes/plans/forest/wh.bin")))),
            ],
            pub_counts: vec![999424, 999424, 785408, 917504],
            lid: 24,
        },
        (1 | 3, true) => ForestInput {
            calls: vec![range("R10z", 22, 10), range("R8d", 22, 8)],
            pub_counts: vec![262144, 262144],
            lid: 24,
        },
        (4, false) => ForestInput {
            calls: vec![
                range("R16a", 20, 16),
                range("R16b", 20, 16),
                range("R8", 20, 8),
                Call::new("FEa", 20, 16, 1, 0, Some(table(include_bytes!("../../spikes/plans/forest/h_fea.bin")))),
                Call::new("FEb", 20, 16, 1, 0, Some(table(include_bytes!("../../spikes/plans/forest/h_feb.bin")))),
                Call::new("FC", 20, 16, 1, 32768, Some(table(include_bytes!("../../spikes/plans/forest/h_fc.bin")))),
            ],
            pub_counts: vec![950272, 1015808, 1008128, 1032192, 1032192, 1015808],
            lid: 24,
        },
        (4, true) => ForestInput {
            calls: vec![range("R10z", 23, 10), range("R8d", 23, 8)],
            pub_counts: vec![3145728, 3145728],
            lid: 24,
        },
        _ => unreachable!(),
    }
}
