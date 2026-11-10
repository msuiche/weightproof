//! Verifier-only FOLD/CS/BASE transitions for LINK and reusable IOPPs.

use crate::fp::{
    claims::{
        pad_places, GroupLayout, GroupRow, KeptForm, OracleId, OracleShape, Part, Row, RowKind,
        Statement,
    },
    code_eval::{generator_at, generator_row, log_ell, mask_position, symbol_at},
    field::{ExtensionElement as E, FieldElement as F},
    tensor::{fold_low, inner, Form, Tensor},
};
use alloc::{vec, vec::Vec};

#[derive(Debug, Clone)]
pub struct SingleRows {
    pub code: u32,
    pub mask_code: u32,
    pub layout: GroupLayout,
    pub gamma: Vec<E>,
    pub masks: u32,
    pub fresh: OracleId,
    pub c: Form,
    pub kept: Vec<KeptForm>,
    pub kept_rows: usize,
    pub d: Vec<Vec<E>>,
    pub target: E,
}

impl SingleRows {
    pub fn rows(&self) -> usize {
        1 + self.kept_rows + self.masks as usize
    }
    pub fn mask_row(&self, index: u32) -> usize {
        1 + self.kept_rows + index as usize
    }
    pub fn oracles(&self) -> Vec<OracleId> {
        let mut out = self.layout.members.clone();
        if self.masks == 2 {
            out.push(self.fresh);
        }
        out
    }
}

pub fn fold_round_claim(claim: E, c0: E, c2: E, challenge: E) -> E {
    let c1 = claim - c0 - c0 - c2;
    c0 + c1 * challenge + c2 * challenge * challenge
}

pub fn fold_mask_form(k: u32, h: u32, mut d: Vec<E>, eps: E, gamma: &[E]) -> Vec<E> {
    d = fold_low(d, &gamma[..h as usize]);
    let mut scale = eps;
    for &challenge in &gamma[h as usize..k as usize] {
        scale *= challenge;
    }
    for value in &mut d {
        *value *= scale;
    }
    d
}

pub fn fold_mask_weights(k: u32, h: u32, len: usize, gamma: &[E]) -> Vec<E> {
    let mut after = vec![E::ONE; k as usize + 1];
    for l in (0..k as usize).rev() {
        after[l] = after[l + 1] * gamma[l];
    }
    let mut out = vec![E::ZERO; len];
    out[0] = after[h as usize];
    for l in 1..=h as usize {
        out[l] = gamma[l - 1] * after[h as usize];
    }
    let mut column = h as usize + 1;
    for l in h as usize + 1..=k as usize {
        for t in if l == h as usize + 1 { 2 } else { 1 }..=2 {
            out[column] = if t == 1 {
                gamma[l - 1]
            } else {
                gamma[l - 1] * gamma[l - 1]
            } * after[l];
            column += 1;
        }
    }
    if h < k {
        out[column] = E::ONE;
    }
    out
}

pub fn fold_output(
    statement: &Statement,
    code: u32,
    mask_code: u32,
    layout: GroupLayout,
    row: GroupRow,
    fresh: OracleId,
    eps: E,
    gamma: Vec<E>,
    claim: E,
) -> SingleRows {
    let k = layout.k;
    let h = layout.h;
    let mut c = row.c;
    c.fix_low(&gamma, k, eps);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some()
        && layout.members.len() == 1
        && layout.members[0] == 35
    {
        eprintln!("RUST_FOLD_TENSORS count={}", c.tensors.len());
        for (index, tensor) in c.tensors.iter().take(12).enumerate() {
            eprintln!(
                "RUST_FOLD_TENSOR index={} scalar={:016x} at0={:016x} at1={:016x}",
                index,
                tensor.scalar.0[0],
                tensor.at(0).0[0],
                tensor.at(1).0[0]
            );
        }
        eprintln!(
            "RUST_FOLD_ENTRIES {:?}",
            &c.entries[..c.entries.len().min(12)]
        );
    }
    let kept_factor = gamma
        .iter()
        .fold(eps, |factor, &challenge| factor * (E::ONE - challenge));
    let mut kept = row.kept;
    for form in &mut kept {
        form.c.scale(kept_factor);
    }
    let kept_rows = layout
        .members
        .iter()
        .enumerate()
        .filter(|(i, _)| layout.kept[*i])
        .map(|(_, &oracle)| statement.oracles[oracle as usize].shape.main_rows as usize)
        .sum();
    let mask_ell = statement.mask[mask_code as usize].ell as usize;
    let out = SingleRows {
        code,
        mask_code,
        layout,
        gamma: gamma.clone(),
        masks: 2,
        fresh,
        c,
        kept,
        kept_rows,
        d: vec![
            fold_mask_form(k, h, row.d, eps, &gamma),
            fold_mask_weights(k, h, mask_ell, &gamma),
        ],
        target: claim,
    };
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!("RUST_FOLD_TARGET {:016x}", claim.0[0]);
        if !out.layout.members.is_empty() {
            eprintln!(
                "RUST_FOLD_FORM16 {:?}",
                (0..16)
                    .map(|index| out.c.at(index).0[0])
                    .collect::<Vec<_>>()
            );
            let vars = out
                .c
                .tensors
                .iter()
                .map(|tensor| tensor.vars())
                .max()
                .unwrap_or(0);
            let point = vec![E::ONE + E::ONE; vars as usize];
            eprintln!("RUST_FOLD_COEF hash={:016x}", out.c.mle(&point).0[0]);
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct OpenedColumn {
    pub main: Vec<E>,
    pub mask: Vec<E>,
}
#[derive(Debug, Clone, Default)]
pub struct Opened {
    pub of: Vec<Vec<OpenedColumn>>,
}

pub fn single_rows_symbols(rows: &SingleRows, opened: &Opened, positions: usize) -> Vec<E> {
    let layout = &rows.layout;
    let mut out = Vec::with_capacity(positions * rows.rows());
    for i in 0..positions {
        let mut main = vec![E::ZERO; 1usize << layout.k];
        let mut mask = vec![E::ZERO; 1usize << layout.h];
        for (w, _) in layout.members.iter().enumerate() {
            let column = &opened.of[w][i];
            for (b, &value) in column.main.iter().enumerate() {
                if !layout.kept[w] {
                    main[layout.main_first[w] as usize + b] = value;
                }
            }
            for (b, &value) in column.mask.iter().enumerate() {
                mask[layout.mask_first[w] as usize + b] = value;
            }
        }
        let folded_main = fold_low(main, &rows.gamma);
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some()
            && layout.k == 4
            && layout.members.len() == 1
            && i == 0
        {
            eprintln!(
                "RUST_SHARED_SYMBOL main0={:016x} main1={:016x} y={:016x}",
                opened.of[0][i].main[0].0[0],
                opened.of[0][i].main.get(1).map_or(E::ZERO, |x| *x).0[0],
                folded_main[0].0[0]
            );
        }
        out.push(folded_main[0]);
        for (w, _) in layout.members.iter().enumerate() {
            if layout.kept[w] {
                out.extend_from_slice(&opened.of[w][i].main);
            }
        }
        if rows.masks >= 1 {
            out.push(fold_low(mask, &rows.gamma[..layout.h as usize])[0]);
        }
        if rows.masks == 2 {
            let fresh = &opened.of[layout.members.len()][i].mask;
            let folded_fresh = fold_low(fresh.clone(), &rows.gamma[..layout.h as usize]);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some()
                && layout.k == 4
                && layout.members.len() == 1
                && i == 0
            {
                eprintln!(
                    "RUST_FRESH_SYMBOL mask0={:016x} y={:016x}",
                    fresh[0].0[0], folded_fresh[0].0[0]
                );
            }
            out.push(folded_fresh[0]);
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct Cut {
    pub part: Part,
    pub bits: u32,
    pub first_row: u64,
}
#[derive(Debug, Clone)]
pub struct KeptTarget {
    pub shape: OracleShape,
    pub kept_again: bool,
}

#[derive(Debug, Clone)]
pub struct SwitchTarget {
    pub shape: OracleShape,
    pub main: Cut,
    pub rest: Cut,
    pub kept: Vec<KeptTarget>,
}

pub fn switch_target_stage(
    code: u32,
    mask_code: u32,
    k: u32,
    h: u32,
    answers: u32,
) -> SwitchTarget {
    SwitchTarget {
        shape: OracleShape {
            code,
            mask_code,
            main_rows: 1u64 << k,
            mask_rows: 1u64 << h,
            main_deg: 3,
            answers,
        },
        main: Cut {
            part: Part::Main,
            bits: k,
            first_row: 0,
        },
        rest: Cut {
            part: Part::Mask,
            bits: h,
            first_row: 0,
        },
        kept: vec![],
    }
}

fn cut_index(cut: Cut, table_row_bits: u32, index: u64) -> u64 {
    (cut.first_row + (index & ((1u64 << cut.bits) - 1))) + ((index >> cut.bits) << table_row_bits)
}

#[derive(Debug, Clone, Default)]
pub struct SwitchParts {
    pub mask_at: Vec<usize>,
    pub main_tail_at: usize,
    pub kept_tail_at: Vec<usize>,
    pub mask_tail_at: Vec<usize>,
    pub old_pad_at: Vec<usize>,
    pub length: usize,
}

pub fn switch_kept(rows: &SingleRows) -> Vec<OracleId> {
    rows.layout
        .members
        .iter()
        .enumerate()
        .filter(|(i, _)| rows.layout.kept[*i])
        .map(|(_, &oracle)| oracle)
        .collect()
}

pub fn switch_parts(statement: &Statement, rows: &SingleRows) -> SwitchParts {
    let mut out = SwitchParts::default();
    for _ in 0..rows.masks {
        out.mask_at.push(out.length);
        out.length += statement.mask[rows.mask_code as usize].ell as usize;
    }
    out.main_tail_at = out.length;
    out.length += statement.base[rows.code as usize].r as usize;
    for _ in 0..rows.kept_rows {
        out.kept_tail_at.push(out.length);
        out.length += statement.base[rows.code as usize].r as usize;
    }
    for _ in 0..rows.masks {
        out.mask_tail_at.push(out.length);
        out.length += statement.mask[rows.mask_code as usize].r as usize;
    }
    for oracle in switch_kept(rows) {
        out.old_pad_at.push(out.length);
        out.length += pad_places(statement, oracle).len();
    }
    out
}

#[derive(Debug, Clone)]
pub struct SwitchPlaces {
    cut: Cut,
    row_bits: u32,
    pads: Vec<u64>,
    size: usize,
}

impl SwitchPlaces {
    pub fn new(statement: &Statement, target: &SwitchTarget, oracle: OracleId) -> Self {
        let row_bits = target.shape.row_bits(target.rest.part);
        let room = match target.rest.part {
            Part::Main => statement.base[target.shape.code as usize].ell << target.rest.bits,
            Part::Mask => statement.mask[target.shape.mask_code as usize].ell << target.rest.bits,
        };
        let mut pads = Vec::new();
        for (part, index, _, _) in pad_places(statement, oracle) {
            let row = index & ((1u64 << row_bits) - 1);
            let place = index >> row_bits;
            if part == target.rest.part
                && row >= target.rest.first_row
                && row < target.rest.first_row + (1u64 << target.rest.bits)
            {
                pads.push((row - target.rest.first_row) + (place << target.rest.bits));
            }
        }
        pads.sort_unstable();
        Self {
            cut: target.rest,
            row_bits,
            size: room as usize - pads.len(),
            pads,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get(&self, index: usize) -> u64 {
        let mut j = index as u64;
        for &pad in &self.pads {
            if pad <= j {
                j += 1;
            }
        }
        cut_index(self.cut, self.row_bits, j)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SwitchPiece {
    pub from: usize,
    pub to: usize,
    pub first: u64,
    pub step: u64,
}

pub fn switch_pieces(places: &SwitchPlaces, first: usize, count: usize) -> Vec<SwitchPiece> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < count {
        let begin = places.get(first + i);
        let step = if i + 1 < count {
            places.get(first + i + 1) - begin
        } else {
            1
        };
        let mut end = i + 1;
        let mut last = begin;
        while end < count && places.get(first + end) == last + step {
            last += step;
            end += 1;
        }
        out.push(SwitchPiece {
            from: i,
            to: end,
            first: begin,
            step,
        });
        i = end;
    }
    out
}

fn switch_add_rest(
    row: &mut Row,
    oracle: OracleId,
    part: Part,
    pieces: &[SwitchPiece],
    coefficients: &[E],
) {
    for piece in pieces {
        row.add_run(
            oracle,
            part,
            piece.first,
            piece.step,
            coefficients[piece.from..piece.to].to_vec(),
        );
    }
}

fn switch_add_rest_base(
    row: &mut Row,
    oracle: OracleId,
    part: Part,
    pieces: &[SwitchPiece],
    coefficients: &[F],
) {
    let converted: Vec<E> = coefficients.iter().copied().map(E::from_base).collect();
    switch_add_rest(row, oracle, part, pieces, &converted);
}

fn switch_lift(mut form: Form, vars: u32, cut: Cut, table_row_bits: u32, table_bits: u32) -> Form {
    let above = table_row_bits - cut.bits;
    form.insert_choice(cut.bits, above, cut.first_row >> cut.bits);
    form.insert_choice(vars + above, table_bits - (vars + above), 0);
    form
}

fn switch_form_on(oracle: OracleId, row: &Row) -> Form {
    let mut out = Form::default();
    for term in &row.terms {
        assert_eq!((term.oracle, term.part), (oracle, Part::Main));
        out.add(term.tensor.clone());
    }
    for unit in &row.units {
        assert_eq!((unit.oracle, unit.part), (oracle, Part::Main));
        out.add_entry(unit.index, unit.coefficient);
    }
    for run in &row.runs {
        assert_eq!((run.oracle, run.part), (oracle, Part::Main));
        for (j, &coefficient) in run.coefficients.iter().enumerate() {
            out.add_entry(run.first + run.step * j as u64, coefficient);
        }
    }
    for form in &row.forms {
        assert_eq!(form.oracle, oracle);
        out.add_scaled(&form.form, E::ONE);
    }
    out
}

fn switch_add_kept(
    row: &mut Row,
    pads: &[(Part, u64, u32, E)],
    anew: OracleId,
    group: OracleId,
    rest_part: Part,
    places: &SwitchPlaces,
    old_pad_at: usize,
    form: &Form,
) {
    row.add_form(anew, form.clone());
    for (i, &(_, index, _, _)) in pads.iter().enumerate() {
        let there = form.at(index);
        row.add_unit(anew, Part::Main, index, -there);
        row.add_unit(group, rest_part, places.get(old_pad_at + i), there);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SwitchBox {
    pub is: bool,
    pub shift: u32,
    pub start: u64,
}

pub fn switch_box(places: &SwitchPlaces, first: usize, vars: u32) -> SwitchBox {
    let count = 1usize << vars;
    if vars == 0 || first + count > places.len() {
        return SwitchBox::default();
    }
    let step = places.get(first + 1) - places.get(first);
    if step == 0 || !step.is_power_of_two() {
        return SwitchBox::default();
    }
    for i in 1..count {
        if places.get(first + i) != places.get(first) + i as u64 * step {
            return SwitchBox::default();
        }
    }
    let shift = step.trailing_zeros();
    let start = places.get(first);
    SwitchBox {
        is: ((start >> shift) & (count as u64 - 1)) == 0,
        shift,
        start,
    }
}

fn switch_on_box(box_: SwitchBox, mut tensor: Tensor, vars: u32, table_bits: u32) -> Tensor {
    tensor.insert_choice(0, box_.shift, box_.start & ((1u64 << box_.shift) - 1));
    tensor.insert_choice(
        box_.shift + vars,
        table_bits - box_.shift - vars,
        box_.start >> (box_.shift + vars),
    );
    tensor
}

pub fn switch_rows(
    statement: &Statement,
    rows: &SingleRows,
    target: &SwitchTarget,
    oracle: OracleId,
    anew: &[OracleId],
    positions: &[u64],
    symbols: &[E],
) -> Vec<Row> {
    let parts = switch_parts(statement, rows);
    let places = SwitchPlaces::new(statement, target, oracle);
    let row_bits = target.shape.row_bits(Part::Main);
    let table_bits = row_bits + log_ell(&statement.base[target.shape.code as usize]);
    let code = &statement.base[rows.code as usize];
    let zk = &statement.mask[rows.mask_code as usize];
    let rest_bits = statement.table_bits(oracle, target.rest.part);
    let zk_is_cube = rows.masks != 0 && zk.ell.is_power_of_two();
    let mut boxes = vec![SwitchBox::default(); rows.masks as usize];
    for j in 0..rows.masks as usize {
        if zk_is_cube {
            boxes[j] = switch_box(&places, parts.mask_at[j], log_ell(zk));
        }
    }
    let main_tail = switch_pieces(&places, parts.main_tail_at, code.r as usize);
    let kept = switch_kept(rows);
    assert_eq!(kept.len(), target.kept.len());
    assert_eq!(kept.len(), anew.len());
    let pads: Vec<Vec<(Part, u64, u32, E)>> = kept
        .iter()
        .map(|&oracle| pad_places(statement, oracle))
        .collect();
    let kept_tail: Vec<Vec<SwitchPiece>> = parts
        .kept_tail_at
        .iter()
        .map(|&at| switch_pieces(&places, at, code.r as usize))
        .collect();
    let mut mask_message = Vec::new();
    let mut mask_tail = Vec::new();
    for j in 0..rows.masks as usize {
        mask_message.push(if boxes[j].is {
            vec![]
        } else {
            switch_pieces(&places, parts.mask_at[j], zk.ell as usize)
        });
        mask_tail.push(switch_pieces(&places, parts.mask_tail_at[j], zk.r as usize));
    }

    let mut out = Vec::new();
    let mut old = Row::default();
    let lifted = switch_lift(
        rows.c.clone(),
        log_ell(code),
        target.main,
        row_bits,
        table_bits,
    );
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        let point = vec![E::ONE + E::ONE; table_bits as usize];
        eprintln!(
            "RUST_SWITCH_OLD_COEF hash={:016x} vars={} row_bits={} above={}",
            lifted.mle(&point).0[0],
            log_ell(code),
            row_bits,
            row_bits - target.main.bits
        );
    }
    old.add_form(oracle, lifted);
    for f in &rows.kept {
        let i = kept
            .iter()
            .position(|&o| o == f.oracle)
            .expect("CS: the claim reads a kept member that the group does not have");
        switch_add_kept(
            &mut old,
            &pads[i],
            anew[i],
            oracle,
            target.rest.part,
            &places,
            parts.old_pad_at[i],
            &f.c,
        );
    }
    for j in 0..rows.masks as usize {
        switch_add_rest(
            &mut old,
            oracle,
            target.rest.part,
            &switch_pieces(&places, parts.mask_at[j], rows.d[j].len()),
            &rows.d[j],
        );
    }
    old.target = rows.target;
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() {
        eprintln!(
            "RUST_SWITCH_CUT target={:016x} bits={} first_row={} table_bits={}",
            old.target.0[0], target.main.bits, target.main.first_row, table_bits
        );
    }
    out.push(old);
    for (i, &old_oracle) in kept.iter().enumerate() {
        for sample in &statement.oracles[old_oracle as usize].sample_rows {
            let mut old = Row::default();
            switch_add_kept(
                &mut old,
                &pads[i],
                anew[i],
                oracle,
                target.rest.part,
                &places,
                parts.old_pad_at[i],
                &switch_form_on(old_oracle, sample),
            );
            old.target = sample.target;
            out.push(old);
        }
    }

    let sorted: Vec<u64> = {
        let mut values = positions.to_vec();
        values.sort_unstable();
        values.dedup();
        values
    };
    for &position in positions {
        let sorted_index = sorted.binary_search(&position).unwrap();
        let at_symbols = &symbols[sorted_index * rows.rows()..(sorted_index + 1) * rows.rows()];
        let generator = generator_at(code, position);
        let on_message = Tensor::product(
            &generator.factors[..log_ell(code) as usize],
            E::from_base(generator.message_scale),
        );
        let (_, tail) = generator_row(code, position);
        let mut main = Row {
            kind: RowKind::Generator,
            target: at_symbols[0],
            ..Row::default()
        };
        let mut form = Form::default();
        form.add(on_message.clone());
        main.add_form(
            oracle,
            switch_lift(form, log_ell(code), target.main, row_bits, table_bits),
        );
        switch_add_rest_base(&mut main, oracle, target.rest.part, &main_tail, &tail);
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_SHARED_CS_GEN row=0 target={:016x}", main.target.0[0]);
        }
        out.push(main);

        let mut kept_row = 0usize;
        for (i, &old_oracle) in kept.iter().enumerate() {
            for b in 0..statement.oracles[old_oracle as usize].shape.main_rows {
                let mut row = Row {
                    kind: RowKind::Generator,
                    target: at_symbols[1 + kept_row],
                    ..Row::default()
                };
                let mut form = Form::default();
                form.add(on_message.clone());
                form.insert_choice(0, statement.row_bits(old_oracle, Part::Main), b);
                switch_add_kept(
                    &mut row,
                    &pads[i],
                    anew[i],
                    oracle,
                    target.rest.part,
                    &places,
                    parts.old_pad_at[i],
                    &form,
                );
                switch_add_rest_base(
                    &mut row,
                    oracle,
                    target.rest.part,
                    &kept_tail[kept_row],
                    &tail,
                );
                out.push(row);
                kept_row += 1;
            }
        }

        for j in 0..rows.masks as usize {
            let position_zk = mask_position(code, zk, position);
            let (message, tail) = generator_row(zk, position_zk);
            let mut row = Row {
                kind: RowKind::Generator,
                target: at_symbols[rows.mask_row(j as u32)],
                ..Row::default()
            };
            if boxes[j].is {
                let generator_zk = generator_at(zk, position_zk);
                row.add(
                    oracle,
                    target.rest.part,
                    switch_on_box(
                        boxes[j],
                        Tensor::product(
                            &generator_zk.factors[..log_ell(zk) as usize],
                            E::from_base(generator_zk.message_scale),
                        ),
                        log_ell(zk),
                        rest_bits,
                    ),
                );
            } else {
                switch_add_rest_base(
                    &mut row,
                    oracle,
                    target.rest.part,
                    &mask_message[j],
                    &message,
                );
            }
            switch_add_rest_base(&mut row, oracle, target.rest.part, &mask_tail[j], &tail);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!(
                    "RUST_SHARED_CS_GEN row={} target={:016x}",
                    rows.mask_row(j as u32),
                    row.target.0[0]
                );
            }
            out.push(row);
        }
    }
    out
}

pub fn last_rows(
    statement: &Statement,
    code: u32,
    mask_code: u32,
    mut layout: GroupLayout,
    mut row: GroupRow,
) -> SingleRows {
    row.c.fix_low(&[E::ZERO], 1, E::ONE);
    #[cfg(feature = "std")]
    if std::env::var_os("DUMP_SHARED").is_some() && row.target.0[0] == 0x89044890d677c990 {
        eprintln!(
            "RUST_LAST_ROWS c0={:016x} c2={:016x} c42={:016x}",
            row.c.at(0).0[0],
            row.c.at(2).0[0],
            row.c.at(42).0[0]
        );
    }
    layout.k = 0;
    let kept_rows = layout
        .members
        .iter()
        .enumerate()
        .filter(|(i, _)| layout.kept[*i])
        .map(|(_, &oracle)| statement.oracles[oracle as usize].shape.main_rows as usize)
        .sum();
    SingleRows {
        code,
        mask_code,
        layout,
        gamma: vec![],
        masks: 1,
        fresh: u32::MAX,
        c: row.c,
        kept: row.kept,
        kept_rows,
        d: vec![row.d],
        target: row.target,
    }
}

pub fn base_linear_left(statement: &Statement, rows: &SingleRows, message: &[E]) -> E {
    let code = &statement.base[rows.code as usize];
    let mut coefficients = vec![E::ZERO; code.ell as usize];
    rows.c.add_to(&mut coefficients);
    let mut left = inner(&coefficients, &message[..code.ell as usize]);
    let mut offset = code.ell as usize;
    for kept in &rows.kept {
        let shape = &statement.oracles[kept.oracle as usize].shape;
        let width = statement.base[shape.code as usize].ell as usize;
        offset += shape.main_rows as usize * width;
    }
    for j in 0..rows.masks as usize {
        let width = statement.mask[rows.mask_code as usize].ell as usize;
        let start = offset + j * width;
        left += inner(&rows.d[j], &message[start..start + width]);
    }
    left
}

pub fn base_symbol(
    statement: &Statement,
    rows: &SingleRows,
    row: usize,
    position: u64,
    messages: &[Vec<E>],
    tails: &[Vec<E>],
) -> E {
    let code = if row == 0 {
        &statement.base[rows.code as usize]
    } else if row <= rows.kept_rows {
        let kept = &rows.kept[row - 1];
        &statement.base[statement.oracles[kept.oracle as usize].shape.code as usize]
    } else {
        &statement.mask[rows.mask_code as usize]
    };
    let position = if row <= rows.kept_rows {
        position
    } else {
        mask_position(&statement.base[rows.code as usize], code, position)
    };
    symbol_at(code, position, &messages[row], &tails[row])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::codes::CodeSize;

    #[test]
    fn base_mask_message_starts_after_main_message() {
        let statement = Statement::new(
            vec![CodeSize {
                log_n: 4,
                ell: 4,
                r: 0,
                over_E: true,
            }],
            vec![CodeSize {
                log_n: 3,
                ell: 2,
                r: 0,
                over_E: true,
            }],
        );
        let mut form = Form::default();
        form.add_entry(0, E::ONE);
        let rows = SingleRows {
            code: 0,
            mask_code: 0,
            layout: GroupLayout {
                members: vec![],
                kept: vec![],
                main_first: vec![],
                mask_first: vec![],
                k: 0,
                h: 0,
                log_ell: 0,
                log_ell_zk: 0,
            },
            gamma: vec![],
            masks: 1,
            fresh: 0,
            c: form,
            kept: vec![],
            kept_rows: 0,
            d: vec![vec![E([100, 0, 0]), E([200, 0, 0])]],
            target: E::ZERO,
        };
        let message = [
            E([10, 0, 0]),
            E([20, 0, 0]),
            E([30, 0, 0]),
            E([40, 0, 0]),
            E([100, 0, 0]),
            E([200, 0, 0]),
        ];
        assert_eq!(
            base_linear_left(&statement, &rows, &message),
            E([10 + 100 * 100 + 200 * 200, 0, 0])
        );
    }

    #[test]
    fn fold_claim_matches_quadratic_evaluation() {
        let claim = E([17, 19, 23]);
        let c0 = E([2, 3, 5]);
        let c2 = E([7, 11, 13]);
        let challenge = E([29, 31, 37]);
        let c1 = claim - c0 - c0 - c2;
        assert_eq!(
            fold_round_claim(claim, c0, c2, challenge),
            c0 + c1 * challenge + c2 * challenge * challenge
        );
    }

    #[test]
    fn mask_weights_match_direct_hiding_polynomial() {
        let gamma = vec![E([2, 3, 5]), E([7, 11, 13]), E([17, 19, 23])];
        let weights = fold_mask_weights(3, 1, 8, &gamma);
        assert_eq!(weights.len(), 8);
        assert_eq!(weights[0], gamma[1] * gamma[2]);
        assert_eq!(weights[1], gamma[0] * gamma[1] * gamma[2]);
        assert_eq!(weights[2], gamma[1] * gamma[1] * gamma[2]);
        assert_eq!(weights[3], gamma[2]);
        assert_eq!(weights[4], gamma[2] * gamma[2]);
        assert_eq!(weights[5], E::ONE);
    }
}
