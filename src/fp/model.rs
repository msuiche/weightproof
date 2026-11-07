//! Model-subproof transcript, affine forms, and masked sumchecks.

use alloc::{string::String, vec, vec::Vec};

use crate::fp::{
    chain::{Chain, ChainKind},
    error::FpError,
    field::ExtensionElement as E,
    psc::{hp_form, poly_at, Weight},
    transcript::{elements, words, Cursor},
};

#[derive(Debug, Clone, Copy)]
pub enum RoundName {
    PscEpsilon,
    PscRound,
    PscY,
    Line,
    Zero,
    BatSigma,
    Lkp,
    LkpShift,
    LkpPlanes,
    ForestPoint,
    Join,
    PscValues,
}

pub struct Tr<'a, 'b> {
    cursor: &'a mut Cursor<'b>,
    chain: &'a mut Chain,
    context: String,
    last_words: Vec<u64>,
}

impl<'a, 'b> Tr<'a, 'b> {
    pub fn new(cursor: &'a mut Cursor<'b>, chain: &'a mut Chain, context: &str) -> Self {
        Self {
            cursor,
            chain,
            context: context.into(),
            last_words: vec![],
        }
    }

    pub fn set_context(&mut self, context: &str) {
        self.context = context.into();
    }

    pub fn chain(&mut self) -> &mut Chain {
        self.chain
    }

    pub fn read(&mut self, count: usize) -> Result<Vec<E>, FpError> {
        let bytes = self.cursor.read_scalars(count)?;
        // Every read of a round joins its prover's message (rounds.h `message_`),
        // until the round's close takes the accumulated words.
        self.last_words.extend_from_slice(&words(bytes));
        Ok(elements(bytes))
    }

    pub fn sum(&mut self) -> Result<E, FpError> {
        let bytes = self.cursor.read_scalars(1)?;
        self.last_words.extend_from_slice(&words(bytes));
        Ok(elements(bytes)[0])
    }

    fn close(&mut self, kind: ChainKind) {
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_ROUNDS").is_some() {
            eprintln!("RUST_MODEL_CLOSE context={} kind={}", self.context, kind as u64);
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_MSG").is_some() {
            eprintln!("RUST_MSG context={} kind={} words={:?}", self.context, kind as u64, self.last_words);
        }
        let message = core::mem::take(&mut self.last_words);
        self.chain.close(kind, &message);
    }

    pub fn answer(&mut self, round: RoundName) -> Result<E, FpError> {
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_ROUNDS").is_some() { eprintln!("RUST_ANSWER context={} round={:?}", self.context, round); }
        self.close(match round {
            RoundName::PscEpsilon => ChainKind::Epsilon,
            RoundName::PscValues => ChainKind::Empty,
            _ => ChainKind::Algebraic,
        });
        Ok(E::from_words(self.chain.draw_e()))
    }

    pub fn answer_not_zero(&mut self, round: RoundName) -> Result<E, FpError> {
        self.close(ChainKind::Algebraic);
        Ok(E::from_words(self.chain.draw_e_nonzero()))
    }

    pub fn point(&mut self, round: RoundName, count: usize) -> Result<Vec<E>, FpError> {
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_ROUNDS").is_some() { eprintln!("RUST_POINT context={} count={}", self.context, count); }
        self.close(ChainKind::Algebraic);
        Ok((0..count).map(|_| E::from_words(self.chain.draw_e())).collect())
    }

    pub fn batch(&mut self, rows: usize) -> Result<Vec<E>, FpError> {
        if rows == 0 {
            return Err(FpError::Range {
                context: "model-empty-batch",
            });
        }
        self.close(ChainKind::Algebraic);
        let x = E::from_words(self.chain.draw_e());
        let mut out = Vec::with_capacity(rows);
        out.push(E::ONE);
        for i in 1..rows {
            out.push(out[i - 1] * x);
        }
        Ok(out)
    }

    pub fn batch_open(&mut self) -> Result<E, FpError> {
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_ROUNDS").is_some() { eprintln!("RUST_BATCH_OPEN context={}", self.context); }
        self.close(ChainKind::Algebraic);
        Ok(E::from_words(self.chain.draw_e()))
    }

    pub fn empty(&mut self, _round: RoundName) -> Result<(), FpError> {
        self.close(ChainKind::Empty);
        Ok(())
    }

    pub fn trits(&mut self, columns: usize, combinations: usize) -> Result<Vec<i8>, FpError> {
        self.close(ChainKind::Bits);
        let total = columns * combinations;
        let mut out = Vec::with_capacity(total);
        // C++ BitsDraw packs 32 trits per word, two bits per entry.
        for start in (0..total).step_by(32) {
            let mut word = self.chain.next_word();
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && start == 0 {
                eprintln!("RUST_TRIT_WORD first={:016x} total={}", word, total);
            }
            for j in 0..32 {
                if start + j >= total {
                    break;
                }
                out.push((word & 1) as i8 - (word & (word << 1) & 2) as i8);
                word >>= 2;
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct Ent {
    pub tree: u32,
    pub row: u32,
    pub idx: u64,
    pub coef: E,
}

#[derive(Debug, Clone)]
pub struct Aff {
    pub e: Vec<Ent>,
    pub s: Vec<(u32, E)>,
    pub c: E,
}

impl Default for Aff {
    fn default() -> Self {
        Self {
            e: vec![],
            s: vec![],
            c: E::ZERO,
        }
    }
}

impl Aff {
    pub fn constant(value: E) -> Self {
        Self {
            c: value,
            ..Self::default()
        }
    }

    pub fn slot(slot: u32) -> Self {
        Self {
            s: vec![(slot, E::ONE)],
            ..Self::default()
        }
    }

    pub fn entry(tree: u32, row: u32, idx: u64) -> Self {
        Self {
            e: vec![Ent {
                tree,
                row,
                idx,
                coef: E::ONE,
            }],
            ..Self::default()
        }
    }

    pub fn add(&mut self, other: &Self, scale: E) {
        self.e.extend(other.e.iter().map(|entry| Ent {
            tree: entry.tree,
            row: entry.row,
            idx: entry.idx,
            coef: entry.coef * scale,
        }));
        self.s.extend(other.s.iter().map(|&(slot, coeff)| (slot, coeff * scale)));
        self.c += other.c * scale;
    }

    pub fn scaled(self, scale: E) -> Self {
        let mut out = Self::default();
        out.add(&self, scale);
        out
    }

    pub fn plus(mut self, other: &Self) -> Self {
        self.add(other, E::ONE);
        self
    }

    pub fn minus(mut self, other: &Self) -> Self {
        self.add(other, -E::ONE);
        self
    }
}

#[derive(Debug, Clone)]
pub struct HRow {
    pub coef: Vec<E>,
    pub l: Vec<E>,
    pub row_form: bool,
    pub row_ones: bool,
    pub row_point: Vec<E>,
    pub row_lo: Vec<E>,
    pub row_hi: Vec<E>,
    pub row_first: u64,
}

impl HRow {
    pub fn new(columns: usize) -> Self {
        Self {
            coef: vec![],
            l: vec![E::ZERO; columns],
            row_form: false,
            row_ones: false,
            row_point: vec![],
            row_lo: vec![],
            row_hi: vec![],
            row_first: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HTree {
    pub name: String,
    pub kind: char,
    pub r: u64,
    pub k: u64,
    pub rows: Vec<HRow>,
}

impl HTree {
    pub fn new(name: String, kind: char, r: u64, k: u64, rows: usize) -> Self {
        Self {
            name,
            kind,
            r,
            k,
            rows: (0..rows).map(|_| HRow::new(k as usize)).collect(),
        }
    }
}

#[derive(Debug)]
pub struct HFork {
    pub ns: u32,
    pub trees: Vec<HTree>,
    pub vp: Vec<E>,
    pub mf: Vec<E>,
    pub ls: Vec<E>,
    pub c0: E,
    got: Vec<bool>,
    pub n_claim: u32,
    pub theta: E,
    pub plan_claims: u32,
}

impl Default for HFork {
    fn default() -> Self {
        Self {
            ns: 0,
            trees: vec![],
            vp: vec![],
            mf: vec![],
            ls: vec![],
            c0: E::ZERO,
            got: vec![],
            n_claim: 0,
            theta: E::ZERO,
            plan_claims: 0,
        }
    }
}

impl HFork {
    pub fn slots(&mut self, count: u32) {
        self.ns = count;
        self.vp = vec![E::ZERO; count as usize];
        self.mf = vec![E::ZERO; count as usize];
        self.ls = vec![E::ZERO; count as usize];
        self.got = vec![false; count as usize];
    }

    pub fn finish_slots(&mut self, count: u32) {
        self.ns = count;
        self.vp.truncate(count as usize);
        self.mf.truncate(count as usize);
        self.ls.truncate(count as usize);
        self.got.truncate(count as usize);
    }

    pub fn hide(&mut self, slot: u32, shown: E, factor: E) -> Result<(), FpError> {
        let index = slot as usize;
        if index >= self.ns as usize || self.got.len() != self.ns as usize || self.got[index] {
            return Err(FpError::Range {
                context: "model-slot-order",
            });
        }
        self.got[index] = true;
        self.vp[index] = shown;
        self.mf[index] = factor;
        Ok(())
    }

    pub fn shown(&self, slot: u32) -> Result<E, FpError> {
        if slot as usize >= self.ns as usize || !self.got[slot as usize] {
            return Err(FpError::Range {
                context: "model-slot-order",
            });
        }
        Ok(self.vp[slot as usize])
    }

    pub fn public_part(&self, form: &Aff) -> Result<E, FpError> {
        let mut value = form.c;
        for &(slot, coeff) in &form.s {
            value += coeff * self.shown(slot)?;
        }
        Ok(value)
    }

    pub fn claims_in(&mut self, tr: &mut Tr, slot: &mut u32, count: u32) -> Result<u32, FpError> {
        let values = tr.read(count as usize)?;
        for (i, value) in values.into_iter().enumerate() {
            self.hide(*slot + i as u32, value, E::ONE)?;
        }
        let first = *slot;
        *slot += count;
        Ok(first)
    }

    pub fn lin(&mut self, form: Aff) {
        let weight = self.kap();
        self.push(weight, &form);
    }

    pub fn push(&mut self, weight: E, form: &Aff) {
        for entry in &form.e {
            if let Some(tree) = self.trees.get_mut(entry.tree as usize) {
                if let Some(row) = tree.rows.get_mut(entry.row as usize) {
                    if let Some(coefficient) = row.l.get_mut(entry.idx as usize) {
                        *coefficient += weight * entry.coef;
                    }
                }
            }
        }
        for &(slot, coeff) in &form.s {
            self.ls[slot as usize] += weight * coeff;
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && slot == 1591 {
                eprintln!("RUST_SLOT1591_WRITE weight={:016x} coeff={:016x} nclaim={}", weight.0[0], coeff.0[0], self.n_claim);
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && slot == 12 {
                eprintln!("RUST_SLOT12_WRITE weight={:016x} coeff={:016x} nclaim={}", weight.0[0], coeff.0[0], self.n_claim);
            }
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && slot == 78 {
                eprintln!("RUST_SLOT78_WRITE weight={:016x} coeff={:016x} nclaim={}", weight.0[0], coeff.0[0], self.n_claim);
            }
        }
        self.c0 += weight * form.c;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() && self.n_claim == 76 {
            for &(slot, coeff) in &form.s {
                eprintln!("RUST_CLAIM76_SLOT {} {:016x}", slot, coeff.0[0]);
            }
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() && self.n_claim <= 1600 {
            eprintln!("RUST_PUSH_CLAIM index={} weight={:016x} constant={:016x} total={:016x} sh={:016x}", self.n_claim, weight.0[0], form.c.0[0], self.c0.0[0], form.s.iter().fold(0u64, |a, &(slot, coeff)| a ^ coeff.0[0] ^ slot as u64));
        }
    }

    pub fn kap_at(&self, claim: u32) -> E {
        let mut power = E::ONE;
        let mut square = self.theta;
        let mut exponent = claim;
        while exponent != 0 {
            if exponent & 1 != 0 {
                power *= square;
            }
            square *= square;
            exponent >>= 1;
        }
        power
    }

    pub fn kap(&mut self) -> E {
        let claim = self.n_claim;
        self.n_claim += 1;
        self.kap_at(claim)
    }

    /// C++ HFork::settle: move shown slot values into the equation's constant
    /// and retain only the still-hidden mask factors.
    pub fn settle(&self) -> Result<(Vec<E>, E), FpError> {
        let mut mask_coefficients = vec![E::ZERO; self.ns as usize];
        let mut constant = self.c0;
        let mut debug_sum = E::ZERO;
        for j in 0..self.ns as usize {
            if self.ls[j].is_zero() {
                continue;
            }
            if !self.got[j] {
                return Err(FpError::Range {
                    context: "model-slot-order",
                });
            }
            mask_coefficients[j] = -self.ls[j] * self.mf[j];
            constant += self.ls[j] * self.vp[j];
            debug_sum += self.ls[j] * self.vp[j];
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() && j < 2000 {
                eprintln!("RUST_SETTLE_SLOT j={} ls={:016x} vp={:016x} mf={:016x}", j, self.ls[j].0[0], self.vp[j].0[0], self.mf[j].0[0]);
            }
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_SETTLE_SUM ls_vp={:016x}", debug_sum.0[0]);
        }
        Ok((mask_coefficients, constant))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RoundForm {
    Eq2,
    Sum2,
    Sum3,
    Nodes,
    Eq,
}

fn round_values(form: RoundForm, degree: u32) -> u32 {
    match form {
        RoundForm::Sum3 => 3,
        RoundForm::Nodes | RoundForm::Eq => degree,
        _ => 2,
    }
}

#[derive(Debug, Clone)]
pub struct Psc {
    pub start: Aff,
    pub target: E,
    pub eps: E,
    pub claim: E,
    pub first: u32,
    pub vars: Vec<Weight>,
    pub r: Vec<E>,
    pub ended: bool,
    pub row_made: bool,
    pub started: bool,
}

pub fn psc_begin(f: &HFork, tr: &mut Tr, slot: &mut u32, start: Aff) -> Result<Psc, FpError> {
    let target = f.public_part(&start)?;
    let first = *slot;
    let mu = tr.sum()?;
    let eps = tr.answer(RoundName::PscEpsilon)?;
    *slot += 1;
    Ok(Psc {
        start,
        target,
        eps,
        claim: mu + eps * target,
        first,
        vars: vec![],
        r: vec![],
        ended: false,
        row_made: false,
        started: true,
    })
}

pub fn psc_round(
    tr: &mut Tr,
    slot: &mut u32,
    p: &mut Psc,
    weight: Weight,
    nonzero: bool,
) -> Result<E, FpError> {
    let sent = tr.read(weight.d - 1)?;
    let mut q = vec![E::ZERO; weight.d];
    let mut i = 0;
    for t in 0..weight.d {
        if t != weight.fixed {
            q[t] = sent[i];
            i += 1;
        }
    }
    weight.complete(&mut q, p.claim);
    let r = if nonzero {
        tr.answer_not_zero(RoundName::PscY)?
    } else {
        tr.answer(RoundName::PscRound)?
    };
    p.claim = poly_at(&q, r);
    let degree = weight.d;
    p.vars.push(weight);
    p.r.push(r);
    *slot += (degree - 1) as u32;
    Ok(r)
}

pub fn round_in(
    _f: &HFork,
    tr: &mut Tr,
    slot: &mut u32,
    form: RoundForm,
    p: &mut Psc,
    z: E,
    degree: u32,
) -> Result<E, FpError> {
    if p.ended || (matches!(form, RoundForm::Nodes | RoundForm::Eq) && !(1..=64).contains(&degree)) {
        return Err(FpError::Range {
            context: "model-psc-round",
        });
    }
    let d = round_values(form, degree) + 1;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() && p.first == 1621 {
        eprintln!("RUST_ROUND form={:?} degree={} d={} z={:016x}", form, degree, d, z.0[0]);
    }
    let weight = if matches!(form, RoundForm::Eq2 | RoundForm::Eq) {
        Weight::equality(z, d as usize)
    } else {
        Weight::plain(d as usize)
    };
    psc_round(tr, slot, p, weight, false)
}

pub fn psc_ends(
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
    p: &mut Psc,
    degree_in_y: u32,
    count: u32,
) -> Result<u32, FpError> {
    if p.ended || degree_in_y < 1 || count < 1 {
        return Err(FpError::Range {
            context: "model-psc-end",
        });
    }
    let beta = psc_round(
        tr,
        slot,
        p,
        Weight::at_zero(degree_in_y as usize + 1),
        true,
    )?;
    p.ended = true;
    let first = *slot;
    let values = tr.read(count as usize)?;
    tr.empty(RoundName::PscValues)?;
    for (i, value) in values.into_iter().enumerate() {
        f.hide(first + i as u32, value, beta)?;
    }
    *slot += count;
    Ok(first)
}

pub fn psc_row(f: &mut HFork, p: &mut Psc, s0: E, kept: &Aff) -> Result<Aff, FpError> {
    if p.row_made || !p.started || p.vars.is_empty() {
        return Err(FpError::Range {
            context: "model-psc-row",
        });
    }
    p.row_made = true;
    let form = hp_form(&p.vars, &p.r);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        let xor = form.iter().fold(0u64, |a, x| a ^ x.0[0]);
        eprintln!("RUST_PSC_FORM first={} k={} r0={:016x} xor={:016x}", p.first, p.vars.len(), p.r[0].0[0], xor);
        if p.first == 1621 || p.first == 1583 {
            eprintln!("RUST_PSC_DETAIL d={:?}", p.vars.iter().map(|v| v.d).collect::<Vec<_>>());
            for (i, x) in p.r.iter().enumerate() { eprintln!("RUST_PSC_G {} {:016x} {:016x} {:016x}", i, x.0[0], x.0[1], x.0[2]); }
            for (i, x) in form.iter().enumerate() { eprintln!("RUST_PSC_C {} {:016x}", i, x.0[0]); }
        }
    }
    let mut out = Aff::constant(-(p.claim - p.eps * s0));
    for (i, coefficient) in form.into_iter().enumerate() {
        f.hide(p.first + i as u32, E::ZERO, -E::ONE)?;
        out.add(&Aff::slot(p.first + i as u32), coefficient);
    }
    let mut product = p.eps;
    for (weight, &challenge) in p.vars.iter().zip(&p.r) {
        product *= poly_at(&weight.psi, challenge);
    }
    out.add(&Aff::constant(p.target).minus(&p.start), product);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() && p.first == 1583 {
        eprintln!("RUST_PSC_START_PRODUCT product={:016x} slots={:?}", product.0[0], p.start.s.iter().map(|x| (x.0, x.1.0[0])).collect::<Vec<_>>());
    }
    out.add(kept, p.eps);
    Ok(out)
}

impl core::ops::Add for Aff {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.plus(&rhs)
    }
}

impl core::ops::AddAssign for Aff {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs, E::ONE);
    }
}

impl core::ops::Sub for Aff {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self.minus(&rhs)
    }
}

impl core::ops::Mul<E> for Aff {
    type Output = Self;
    fn mul(self, rhs: E) -> Self {
        self.scaled(rhs)
    }
}

#[derive(Debug, Clone)]
pub struct TokenSumcheckState {
    pub s_cl: u32,
    pub s_end: u32,
    pub r: Vec<E>,
    pub p: Psc,
    pub row_u: Aff,
}

pub fn tsc_rounds(
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
    point: &[E],
) -> Result<TokenSumcheckState, FpError> {
    let s_cl = f.claims_in(tr, slot, 1)?;
    let mut p = psc_begin(f, tr, slot, Aff::slot(s_cl))?;
    let mut r = Vec::with_capacity(point.len());
    for &coordinate in point {
        r.push(round_in(f, tr, slot, RoundForm::Eq2, &mut p, coordinate, 0)?);
    }
    Ok(TokenSumcheckState {
        s_cl,
        s_end: 0,
        r,
        p,
        row_u: Aff::default(),
    })
}

pub fn tsc_in(
    f: &mut HFork,
    tr: &mut Tr,
    slot: &mut u32,
    point: &[E],
    shown: u32,
    degree_in_y: u32,
) -> Result<TokenSumcheckState, FpError> {
    let mut out = tsc_rounds(f, tr, slot, point)?;
    out.s_end = psc_ends(f, tr, slot, &mut out.p, degree_in_y, shown)?;
    Ok(out)
}

pub struct HStack {
    pub lam: u32,
    pub rs: Vec<E>,
    pub p: Psc,
}

pub fn stack_rounds(f: &mut HFork, tr: &mut Tr, slot: &mut u32, stack: &mut HStack) -> Result<(), FpError> {
    stack.rs = vec![E::ZERO; stack.lam as usize];
    for v in (0..stack.lam).rev() {
        stack.rs[v as usize] = round_in(f, tr, slot, RoundForm::Sum2, &mut stack.p, E::ZERO, 0)?;
    }
    Ok(())
}

pub fn eq_low(rs: &[E], lo: u32, hi: u32) -> Vec<E> {
    if lo > hi {
        return vec![];
    }
    if lo == hi {
        return vec![E::ONE];
    }
    crate::fp::tensor::eq_table(&rs[lo as usize..hi as usize])
}

pub fn eq_low_at(rs: &[E], lo: u32, hi: u32, index: u64) -> E {
    eq_low(rs, lo, hi)[index as usize]
}
