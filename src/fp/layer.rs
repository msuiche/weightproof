//! Layer subproof public declarations and U3 table construction.

use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};

use crate::fp::{
    error::FpError,
    field::ExtensionElement as E,
    forest::{forest_phi_b, forest_phi_p, Forest},
    model::Aff,
};

#[derive(Debug, Clone)]
pub struct Seg {
    pub name: String,
    pub n: u64,
    pub moe: bool,
    pub pos: Vec<u64>,
}

fn eq_top(point: &[E]) -> Vec<E> {
    crate::fp::tensor::eq_table_rev(point)
}

#[derive(Debug, Clone)]
pub struct CallD {
    pub name: String,
    pub kind: String,
    pub segs: Vec<Seg>,
    pub j: u64,
    pub l_j: u32,
    pub used: Vec<u8>,
}

impl CallD {
    pub fn new(name: &str, kind: &str, segs: Vec<Seg>) -> Self {
        Self::new_with_floor(name, kind, segs, 12)
    }

    pub fn new_with_floor(name: &str, kind: &str, segs: Vec<Seg>, floor_lg: u32) -> Self {
        let mut out = Self {
            name: name.into(),
            kind: kind.into(),
            segs,
            j: 0,
            l_j: 0,
            used: vec![],
        };
        out.place(floor_lg);
        out
    }

    fn place(&mut self, floor_lg: u32) {
        #[derive(Clone)]
        struct Piece {
            segment: usize,
            first: u64,
            size: u64,
        }
        let mut pieces = Vec::new();
        for (index, segment) in self.segs.iter_mut().enumerate() {
            let mut first = 0;
            for b in (0..=40).rev() {
                if (segment.n >> b) & 1 != 0 {
                    pieces.push(Piece {
                        segment: index,
                        first,
                        size: 1u64 << b,
                    });
                    first += 1u64 << b;
                }
            }
            segment.pos = vec![0; segment.n as usize];
        }
        pieces.sort_by_key(|piece| core::cmp::Reverse(piece.size));
        let mut off = 0;
        for piece in pieces {
            for x in 0..piece.size {
                self.segs[piece.segment].pos[(piece.first + x) as usize] = off + x;
            }
            off += piece.size;
        }
        self.l_j = crate::fp::tensor::ceil_log2(off).max(floor_lg);
        self.j = 1u64 << self.l_j;
        self.used = vec![0; self.j as usize];
        for segment in &self.segs {
            for &position in &segment.pos {
                self.used[position as usize] = if segment.moe { 2 } else { 1 };
            }
        }
    }

    pub fn padding(&self, layer: u64, column: u64) -> bool {
        self.used[column as usize] == 0 || self.used[column as usize] == 2 && layer % 2 == 0
    }

    pub fn seg(&self, name: &str) -> Option<&Seg> {
        self.segs.iter().find(|segment| segment.name == name)
    }
}

#[derive(Debug, Clone)]
pub struct Ten {
    pub name: String,
    pub mat: usize,
    pub nl: u64,
    pub cnt: u64,
    pub j: u64,
    pub off: u64,
}

impl Ten {
    pub fn cell(&self, limb: u64, layer: u64, feature: u64) -> u64 {
        self.off + (limb * self.cnt + layer) * self.j + feature
    }
}

#[derive(Debug, Clone)]
pub struct TMat {
    pub name: String,
    pub m: u64,
    pub k: u64,
    pub kind: u32,
}

#[derive(Debug, Clone)]
pub struct LayerFixture {
    pub first: bool,
    pub eps: [u64; 2],
    pub ns: u32,
    pub ncl: u32,
    pub nlin: u32,
    pub n_batched: u64,
    pub lam_s: u32,
    pub n_u3: u64,
    pub calls: Vec<CallD>,
    pub tensors: Vec<Ten>,
    pub mats: Vec<TMat>,
}

impl LayerFixture {
    pub fn tensor(&self, name: &str) -> &Ten {
        self.tensors.iter().find(|tensor| tensor.name == name).unwrap()
    }

    pub fn call(&self, name: &str) -> &CallD {
        self.calls.iter().find(|call| call.seg(name).is_some()).unwrap()
    }
}

fn seg(name: &str, n: u64, moe: bool) -> Seg {
    Seg {
        name: name.into(),
        n,
        moe,
        pos: vec![],
    }
}

fn ten(name: &str, mat: usize, nl: u64, cnt: u64, j: u64, off: u64) -> Ten {
    Ten {
        name: name.into(),
        mat,
        nl,
        cnt,
        j,
        off,
    }
}

fn tensors() -> Vec<Ten> {
    vec![
        ten("g0", 0, 1, 4, 1024, 0),
        ten("gc", 0, 1, 4, 1024, 4096),
        ten("s", 0, 2, 4, 1024, 8192),
        ten("u", 0, 2, 4, 1024, 16384),
        ten("rg", 0, 1, 4, 1024, 24576),
        ten("ru", 0, 1, 4, 1024, 28672),
        ten("xs", 0, 5, 7, 512, 32768),
        ten("ro", 0, 1, 4, 512, 50688),
        ten("rd", 0, 1, 4, 512, 52736),
        ten("aw", 0, 2, 4, 512, 54784),
        ten("ra", 0, 2, 4, 512, 58880),
        ten("mw", 0, 2, 4, 512, 62976),
        ten("rm", 0, 2, 4, 512, 67072),
        ten("qkw", 0, 2, 4, 768, 71168),
        ten("rqk", 0, 2, 4, 768, 77312),
        ten("vw", 0, 2, 4, 256, 83456),
        ten("rv", 0, 1, 4, 256, 85504),
        ten("gh", 0, 2, 4, 1024, 86528),
        ten("a", 1, 2, 4, 512, 0),
        ten("m", 1, 2, 4, 512, 4096),
        ten("xr", 1, 2, 2, 512, 8192),
        ten("lr", 2, 2, 2, 4, 0),
        ten("rl", 2, 1, 2, 4, 16),
        ten("sel", 2, 1, 2, 4, 24),
        ten("sc", 2, 3, 2, 1, 32),
        ten("row", 2, 1, 8, 99, 38),
        ten("ly", 2, 2, 2, 4, 830),
        ten("l3", 2, 1, 2, 4, 846),
        ten("q", 3, 2, 4, 512, 0),
        ten("kx", 3, 2, 4, 256, 4096),
        ten("v", 3, 2, 4, 256, 6144),
        ten("o", 3, 2, 4, 512, 8192),
    ]
}

fn calls() -> Vec<CallD> {
    vec![
        CallD::new("R16a", "R16", vec![seg("G", 1024, false), seg("U", 1024, false)]),
        CallD::new(
            "R16b",
            "R16",
            vec![
                seg("N1", 512, false),
                seg("N1H", 512, false),
                seg("N2", 512, false),
                seg("N2H", 512, false),
                seg("O", 512, false),
                seg("D", 512, false),
                seg("V", 256, false),
            ],
        ),
        CallD::new(
            "R16c",
            "R16",
            vec![
                seg("QK", 768, false),
                seg("QKH", 768, false),
                seg("XR", 512, true),
                seg("LR", 4, true),
                seg("LT", 4, true),
                seg("DD", 4, true),
            ],
        ),
        CallD::new(
            "R8",
            "R8",
            vec![
                seg("GH", 1024, false),
                seg("UH", 1024, false),
                seg("OH", 512, false),
                seg("DH", 512, false),
                seg("VH", 256, false),
                seg("ROW", 198, false),
                seg("LH", 4, true),
                seg("SEL", 4, true),
                seg("L3", 4, true),
                seg("G0", 1024, false),
            ],
        ),
        CallD::new(
            "FC",
            "FC",
            vec![
                seg("CN1", 512, false),
                seg("CN2", 512, false),
                seg("CQK", 768, false),
                seg("CV", 256, false),
                seg("CLR", 4, true),
            ],
        ),
        CallD::new("FS", "FS", vec![seg("MR", 1, true), seg("CG", 1024, false)]),
        CallD::new("FL", "FL", vec![seg("F", 1024, false)]),
        CallD::new("R16d", "R16", vec![seg("UT", 1024, false)]),
    ]
}

pub(crate) fn accepted_layer(first: bool) -> LayerFixture {
    let mat = |name: &str, m: u64, k: u64, kind: u32| TMat {
        name: name.into(),
        m,
        k,
        kind,
    };
    LayerFixture {
        first,
        eps: [1, 1],
        ns: if first { 3204 } else { 3203 },
        ncl: if first { 33 } else { 32 },
        nlin: if first { 1549 } else { 1516 },
        n_batched: if first { 4347789 } else { 4343660 },
        lam_s: 15,
        n_u3: if first { 167328 } else { 166688 },
        calls: calls(),
        tensors: tensors(),
        mats: vec![
            mat("af0", 8192, 247, 0),
            mat("af1", 8192, 247, 0),
            mat("af2", 8192, 247, 0),
            mat("ad0", 4096, 160, 0),
            mat("ar", 4096, 14, 0),
            mat("b", 8192, 2048, 0),
            mat("qa", 8192, 96, 0),
            mat("xbO", 8192, 20, 0),
            mat("xbI", 8192, 20, u32::from(!first)),
        ],
    }
}

#[derive(Debug, Clone)]
pub struct HFun {
    pub matrix: usize,
    pub l: u64,
    pub phi: Vec<E>,
    pub at: u64,
}

#[derive(Debug, Clone)]
pub struct LaterPhi {
    pub first: usize,
    pub count: usize,
    pub point: String,
    pub long: bool,
}

pub struct Tables {
    pub fun: Vec<HFun>,
    pub later_phi: Vec<LaterPhi>,
    pub offsets: BTreeMap<String, u64>,
    pub lengths: BTreeMap<u64, u64>,
}

impl Tables {
    pub fn at(&self, name: &str) -> Result<u64, FpError> {
        self.offsets.get(name).copied().ok_or(FpError::Range {
            context: "layer-table",
        })
    }

    pub fn entry(&self, tree: u32, offset: u64, index: u64) -> Result<Aff, FpError> {
        let length = self.lengths.get(&offset).ok_or(FpError::Range {
            context: "layer-table-index",
        })?;
        if index >= *length {
            return Err(FpError::Range {
                context: "layer-table-index",
            });
        }
        Ok(Aff::entry(tree, 0, offset + index))
    }
}

fn reads_z(segment: &str) -> &'static str {
    match segment {
        "G" | "U" | "V" | "CN1" | "CN2" => "df",
        "O" => "fiQ",
        "D" => "fo",
        "XR" => "d",
        "LR" => "dr",
        "LT" | "DD" | "ROW" | "LH" | "SEL" | "MR" | "L3" | "CLR" => "r",
        "CQK" | "CV" => "fq",
        _ => "f",
    }
}

fn reads_r(segment: &str) -> &'static str {
    match segment {
        "N1" => "rfi",
        "N2" => "rf",
        "QK" => "d",
        "G" | "U" | "XR" => "rd",
        "D" | "DD" | "MR" => "r",
        _ => "",
    }
}

fn point_lists(calls: &[CallD], _first: bool) -> (Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>, Vec<String>) {
    let mut z = Vec::new();
    let mut r = Vec::new();
    for call in calls {
        let mut zs = String::new();
        let mut rs = String::new();
        for segment in &call.segs {
            zs.push_str(reads_z(&segment.name));
            rs.push_str(reads_r(&segment.name));
        }
        z.push(zs);
        r.push(rs);
    }
    let z_any = |wanted: char| z.iter().any(|s| s.contains(wanted));
    let r_any = |wanted: char| r.iter().any(|s| s.contains(wanted));
    let mut af = Vec::new();
    let mut qa = Vec::new();
    let mut xi = Vec::new();
    let mut xo = Vec::new();
    if z_any('f') {
        af.push("Z".into());
    }
    af.push("Z_V".into());
    if r_any('f') {
        af.push("Y".into());
    }
    af.push("Z_a".into());
    af.push("Z_b".into());
    if z_any('q') {
        qa.push("Z".into());
    }
    if r_any('q') {
        qa.push("Y".into());
    }
    if z_any('i') {
        xi.push("Z".into());
    }
    xi.push("Z_V".into());
    if r_any('i') {
        xi.push("Y".into());
    }
    if z_any('o') {
        xo.push("Z".into());
    }
    if r_any('o') {
        xo.push("Y".into());
    }
    let mut ad = Vec::new();
    let mut ar = Vec::new();
    let mut qs = Vec::new();
    if z_any('d') {
        ad.push("zt".into());
    }
    if r_any('d') {
        ad.push("rt".into());
    }
    if z_any('r') {
        ar.push("zt".into());
        ar.push("P_V".into());
        ar.push("r_tV".into());
    }
    if r_any('r') {
        ar.push("rt".into());
    }
    if z_any('Q') {
        qs.push("zt".into());
    }
    if r_any('Q') {
        qs.push("rt".into());
    }
    (af, qa, xi, xo, ad, ar, qs)
}

pub fn build_tables(
    fixture: &LayerFixture,
    forest: &Forest,
) -> Result<Tables, FpError> {
    let (af, qa, xi, xo, ad, ar, qs) = point_lists(&fixture.calls, fixture.first);
    let mut out = Tables {
        fun: vec![],
        later_phi: vec![],
        offsets: BTreeMap::new(),
        lengths: BTreeMap::new(),
    };
    let mut position = 0u64;
    let mut add = |matrix: usize, parts: u64, length: u64, phi: Vec<E>, name: String, later: Option<(String, bool)>| {
        let first = position;
        for q in 0..parts {
            let m = &fixture.mats[matrix + q as usize];
            out.fun.push(HFun {
                matrix: matrix + q as usize,
                l: length,
                phi: phi.clone(),
                at: position,
            });
            position += m.k * (m.m / length);
        }
        if position == first {
            return Err(FpError::Range {
                context: "layer-table-length",
            });
        }
        out.offsets.insert(name.clone(), first);
        out.lengths.insert(first, position - first);
        if let Some((point, long)) = later {
            out.later_phi.push(LaterPhi {
                first: out.fun.len() - parts as usize,
                count: parts as usize,
                point,
                long,
            });
        }
        Ok(())
    };
    let long = 1u64 << 8;
    for point in af {
        add(0, 3, long, vec![], format!("af@{point}"), Some((point, true)))?;
    }
    for point in qa {
        add(6, 1, long, vec![], format!("qa@{point}"), Some((point, true)))?;
    }
    for point in xi {
        add(8, 1, long, vec![], format!("xbI@{point}"), Some((point, true)))?;
    }
    if fixture.first {
        add(8, 1, long, vec![], "xbI@Z_E".into(), Some(("Z_E".into(), true)))?;
    }
    for point in xo {
        add(7, 1, long, vec![], format!("xbO@{point}"), Some((point, true)))?;
    }
    for point in ad {
        add(3, 1, 64, vec![], format!("ad@{point}"), Some((point, false)))?;
    }
    for point in ar {
        add(4, 1, 64, vec![], format!("ar@{point}"), Some((point, false)))?;
    }
    for point in qs {
        add(6, 1, 64, vec![], format!("qa@{point}"), Some((point, false)))?;
    }
    let mb = 1u64 << 13;
    add(5, 1, mb, forest_phi_b(forest, 13, 0), "b-bits-0".into(), None)?;
    add(5, 1, mb, forest_phi_b(forest, 13, 1), "b-bits-1".into(), None)?;
    add(5, 1, mb, forest_phi_p(13), "b-pin".into(), None)?;
    if position != fixture.n_u3 {
        return Err(FpError::Range {
            context: "layer-table-length",
        });
    }
    Ok(out)
}

pub struct LayerRun {
    pub fixture: LayerFixture,
    pub forest: Forest,
    pub tables: Tables,
    pub wpl: crate::fp::forest::ProductLevel,
    pub s_w: u32,
    pub gp: Vec<E>,
    pub s_pv: u32,
    pub om_v: Vec<E>,
    pub t_v: crate::fp::model::TokenSumcheckState,
    pub t_t: crate::fp::model::TokenSumcheckState,
    pub kt: Aff,
    pub tkh: Vec<E>,
    pub points: BTreeMap<String, Vec<E>>,
    pub tokv: Vec<u32>,
    pub s_sc: u32,
    pub s_sel: u32,
}

fn last(point: &[E], count: usize) -> Vec<E> {
    point[point.len() - count..].to_vec()
}

fn order_default(mu: u32) -> Vec<u32> {
    (0..mu).rev().collect()
}

pub fn through_terms(
    f: &mut crate::fp::model::HFork,
    tr: &mut crate::fp::model::Tr,
    slot: &mut u32,
    fixture: LayerFixture,
    forest: Forest,
) -> Result<LayerRun, FpError> {
    const LT: u32 = 6;
    const LS: u32 = 2;
    const L_PD: u32 = 9;
    const NI: u32 = 8;
    const N_SL: u32 = 9;
    const N_PL: u32 = 7;
    let pv = tr.point(crate::fp::model::RoundName::Zero, LT as usize)?;
    let (ze, tkh) = if fixture.first {
        (
            tr.point(crate::fp::model::RoundName::Zero, (LS + LT) as usize)?,
            tr.point(crate::fp::model::RoundName::Zero, (L_PD - LS) as usize)?,
        )
    } else {
        (vec![], vec![])
    };
    let zt_f = last(&forest.zx, LT as usize);
    for call in &forest.calls {
        let start = call.zk.len() - 8;
        if call.zk[start..] != forest.zx[forest.d_t as usize - 8..] {
            return Err(FpError::Range {
                context: "layer-merged-point",
            });
        }
    }
    let s_w = f.claims_in(tr, slot, N_SL * NI)?;
    let gp = tr.batch(N_SL as usize)?;
    let rho_i = tr.point(crate::fp::model::RoundName::Join, 3)?;
    let half = E::from_base(crate::fp::field::FieldElement(2)).inverse();
    let eq_i = eq_top(&rho_i);
    let mut start = Aff::default();
    let mut scale = E::ONE;
    for _ in 0..L_PD {
        scale *= half;
    }
    for i in 0..NI as usize {
        let weight = scale * eq_i[i];
        for s in 0..N_SL as usize {
            start.add(&Aff::slot(s_w + (N_SL as usize * i + s) as u32), weight * gp[s]);
        }
    }
    let mut rho = rho_i;
    rho.extend(vec![half; L_PD as usize]);
    rho.extend_from_slice(&pv);
    let wpl = crate::fp::forest::read_product_level(
        f,
        tr,
        slot,
        18,
        &rho,
        &mut start,
        &order_default(18),
    )?;
    let s_pv = f.claims_in(tr, slot, N_PL * NI)?;
    let om_v = tr.batch((N_PL * NI) as usize)?;
    let tables = build_tables(&fixture, &forest)?;
    let t_v = crate::fp::model::tsc_in(f, tr, slot, &pv, 4 * NI, 2)?;
    let mut points = BTreeMap::new();
    points.insert("Z".into(), last(&forest.zx, 8));
    points.insert("zt".into(), zt_f.clone());
    points.insert("Z_V".into(), last(&wpl.r, 8));
    points.insert("P_V".into(), pv.clone());
    points.insert("r_tV".into(), t_v.r.clone());
    if fixture.first {
        points.insert("Z_E".into(), ze);
    }
    let mut t_t = crate::fp::model::tsc_in(f, tr, slot, &zt_f, 18, 1)?;
    let s_sc = t_t.s_end + 8;
    let s_sel = s_sc + 2;
    points.insert("rt".into(), t_t.r.clone());
    let mut y = forest.zx[forest.d_t as usize - 8..forest.d_t as usize - 6].to_vec();
    y.extend_from_slice(&t_t.r);
    points.insert("Y".into(), y);
    if t_t.p.eps.is_zero() {
        return Err(FpError::Range {
            context: "layer-zero-epsilon",
        });
    }
    let kt_row = crate::fp::model::psc_row(f, &mut t_t.p, E::ZERO, &Aff::default())?;
    let kt = kt_row.scaled(-t_t.p.eps.inverse());
    Ok(LayerRun {
        fixture,
        forest,
        tables,
        wpl,
        s_w,
        gp,
        s_pv,
        om_v,
        t_v,
        t_t,
        kt,
        tkh,
        points,
        tokv: vec![],
        s_sc,
        s_sel,
    })
}

pub fn assign_later_phi(run: &mut LayerRun) -> Result<(), FpError> {
    let later = run.tables.later_phi.clone();
    for x in later {
        let point = run.points.get(&x.point).unwrap_or_else(|| panic!("missing laterPhi point {}", x.point)).clone();
        if point.len() != if x.long { 8 } else { 6 } {
            return Err(FpError::Range {
                context: "layer-table-point",
            });
        }
        if x.long {
            let weights = crate::fp::tensor::block_rows(
                &eq_top(&point[..2]),
                &eq_top(&point[2..]),
            );
            for j in 0..x.count {
                run.tables.fun[x.first + j].phi = weights.clone();
            }
        } else {
            let weights = eq_top(&point);
            for j in 0..x.count {
                run.tables.fun[x.first + j].phi = weights.clone();
            }
        }
    }
    Ok(())
}
