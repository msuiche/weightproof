//! LINK linear claims and BAT, including kept members.

use crate::fp::{
    code_eval::{generator_at, log_ell, log_r},
    codes::CodeSize,
    field::{ExtensionElement as E, FieldElement as F},
    heads::SampleSeed,
    link_public::IndexName,
    tensor::{ceil_log2, tensor_below, Form, Tensor},
};
use alloc::{vec, vec::Vec};

pub type OracleId = u32;
pub type GroupId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Part {
    Main,
    Mask,
}

#[derive(Debug, Clone)]
pub struct OracleShape {
    pub code: u32,
    pub mask_code: u32,
    pub main_rows: u64,
    pub mask_rows: u64,
    pub main_deg: u32,
    pub answers: u32,
}

impl OracleShape {
    pub fn row_bits(&self, part: Part) -> u32 {
        ceil_log2(match part {
            Part::Main => self.main_rows,
            Part::Mask => self.mask_rows,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    Plain,
    Sample,
    Generator,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub oracle: OracleId,
    pub part: Part,
    pub tensor: Tensor,
}
#[derive(Debug, Clone)]
pub struct Unit {
    pub oracle: OracleId,
    pub part: Part,
    pub index: u64,
    pub coefficient: E,
}
#[derive(Debug, Clone)]
pub struct Run {
    pub oracle: OracleId,
    pub part: Part,
    pub first: u64,
    pub step: u64,
    pub coefficients: Vec<E>,
}
#[derive(Debug, Clone)]
pub struct MainForm {
    pub oracle: OracleId,
    pub form: Form,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub kind: RowKind,
    pub terms: Vec<Term>,
    pub units: Vec<Unit>,
    pub runs: Vec<Run>,
    pub forms: Vec<MainForm>,
    pub target: E,
}

impl Default for Row {
    fn default() -> Self {
        Self {
            kind: RowKind::Plain,
            terms: vec![],
            units: vec![],
            runs: vec![],
            forms: vec![],
            target: E::ZERO,
        }
    }
}

impl Row {
    pub fn add(&mut self, oracle: OracleId, part: Part, tensor: Tensor) {
        self.terms.push(Term {
            oracle,
            part,
            tensor,
        });
    }
    pub fn add_unit(&mut self, oracle: OracleId, part: Part, index: u64, coefficient: E) {
        self.units.push(Unit {
            oracle,
            part,
            index,
            coefficient,
        });
    }
    pub fn add_run(
        &mut self,
        oracle: OracleId,
        part: Part,
        first: u64,
        step: u64,
        coefficients: Vec<E>,
    ) {
        self.runs.push(Run {
            oracle,
            part,
            first,
            step,
            coefficients,
        });
    }
    pub fn add_form(&mut self, oracle: OracleId, form: Form) {
        self.forms.push(MainForm { oracle, form });
    }
}

#[derive(Debug, Clone)]
pub struct Oracle {
    pub shape: OracleShape,
    pub sample_rows: Vec<Row>,
}
#[derive(Debug, Clone, Default)]
pub struct Group {
    pub code: u32,
    pub mask_code: u32,
    pub members: Vec<OracleId>,
    pub kept: Vec<bool>,
    pub rows: Vec<Row>,
}

#[derive(Debug, Clone)]
pub struct GroupLayout {
    pub members: Vec<OracleId>,
    pub kept: Vec<bool>,
    pub main_first: Vec<u64>,
    pub mask_first: Vec<u64>,
    pub k: u32,
    pub h: u32,
    pub log_ell: u32,
    pub log_ell_zk: u32,
}

#[derive(Debug, Clone)]
pub struct KeptForm {
    pub oracle: OracleId,
    pub c: Form,
}

#[derive(Debug, Clone)]
pub struct GroupRow {
    pub c: Form,
    pub kept: Vec<KeptForm>,
    pub d: Vec<E>,
    pub target: E,
}

#[derive(Debug, Clone)]
pub struct Statement {
    pub base: Vec<CodeSize>,
    pub mask: Vec<CodeSize>,
    pub oracles: Vec<Oracle>,
    pub groups: Vec<Group>,
}

impl Statement {
    pub fn new(base: Vec<CodeSize>, mask: Vec<CodeSize>) -> Self {
        Self {
            base,
            mask,
            oracles: vec![],
            groups: vec![],
        }
    }

    pub fn sent(&mut self, shape: OracleShape) -> OracleId {
        self.oracles.push(Oracle {
            shape,
            sample_rows: vec![],
        });
        self.oracles.len() as OracleId - 1
    }

    pub fn new_group(&mut self, code: u32, mask_code: u32) -> GroupId {
        self.groups.push(Group {
            code,
            mask_code,
            ..Group::default()
        });
        self.groups.len() as GroupId - 1
    }

    pub fn join(&mut self, group: GroupId, oracle: OracleId) {
        self.join_kept(group, oracle, false);
    }

    pub fn join_kept(&mut self, group: GroupId, oracle: OracleId, kept: bool) {
        let group = &mut self.groups[group as usize];
        let at = group.members.partition_point(|&id| id < oracle);
        group.members.insert(at, oracle);
        group.kept.insert(at, kept);
    }

    pub fn add_row(&mut self, group: GroupId, row: Row) {
        self.groups[group as usize].rows.push(row);
    }

    pub fn row_bits(&self, oracle: OracleId, part: Part) -> u32 {
        self.oracles[oracle as usize].shape.row_bits(part)
    }

    pub fn place_bits(&self, oracle: OracleId, part: Part) -> u32 {
        let shape = &self.oracles[oracle as usize].shape;
        match part {
            Part::Main => log_ell(&self.base[shape.code as usize]),
            Part::Mask => log_ell(&self.mask[shape.mask_code as usize]),
        }
    }

    pub fn table_bits(&self, oracle: OracleId, part: Part) -> u32 {
        self.row_bits(oracle, part) + self.place_bits(oracle, part)
    }

    pub fn layout(&self, group: GroupId) -> GroupLayout {
        let group = &self.groups[group as usize];
        let mut main_order: Vec<usize> = (0..group.members.len()).collect();
        main_order
            .sort_by_key(|&i| core::cmp::Reverse(self.row_bits(group.members[i], Part::Main)));
        let mut main_first = vec![u64::MAX; group.members.len()];
        let mut main_used = 0u64;
        for i in main_order {
            if group.kept[i] {
                continue;
            }
            let oracle = group.members[i];
            main_first[i] = main_used;
            main_used += 1u64 << self.row_bits(oracle, Part::Main);
        }
        let mut mask_order: Vec<usize> = (0..group.members.len())
            .filter(|&i| self.oracles[group.members[i] as usize].shape.mask_rows != 0)
            .collect();
        mask_order
            .sort_by_key(|&i| core::cmp::Reverse(self.row_bits(group.members[i], Part::Mask)));
        let mut mask_first = vec![0; group.members.len()];
        let mut mask_used = 0u64;
        for i in mask_order {
            let oracle = group.members[i];
            mask_first[i] = mask_used;
            mask_used += 1u64 << self.row_bits(oracle, Part::Mask);
        }
        let h = ceil_log2(mask_used);
        GroupLayout {
            members: group.members.clone(),
            kept: group.kept.clone(),
            main_first,
            mask_first,
            k: ceil_log2(main_used).max(h).max(1),
            h,
            log_ell: log_ell(&self.base[group.code as usize]),
            log_ell_zk: log_ell(&self.mask[group.mask_code as usize]),
        }
    }

    pub fn begin_run(&mut self, group: GroupId) -> (GroupLayout, Vec<Row>) {
        let layout = self.layout(group);
        let group = &mut self.groups[group as usize];
        let mut rows = Vec::new();
        rows.extend(
            group
                .rows
                .iter()
                .filter(|row| row.kind == RowKind::Plain)
                .cloned(),
        );
        for &oracle in &group.members {
            rows.extend(self.oracles[oracle as usize].sample_rows.clone());
        }
        rows.extend(
            group
                .rows
                .iter()
                .filter(|row| row.kind == RowKind::Generator)
                .cloned(),
        );
        (layout, rows)
    }
}

fn group_index(index: u64, row_bits: u32, first: u64, group_bits: u32) -> u64 {
    let row = index & ((1u64 << row_bits) - 1);
    let place = index >> row_bits;
    (first + row) + (place << group_bits)
}

fn kept_form<'a>(out: &'a mut GroupRow, oracle: OracleId) -> &'a mut Form {
    let at = match out.kept.binary_search_by_key(&oracle, |form| form.oracle) {
        Ok(at) => at,
        Err(at) => {
            out.kept.insert(
                at,
                KeptForm {
                    oracle,
                    c: Form::default(),
                },
            );
            at
        }
    };
    &mut out.kept[at].c
}

fn add_to_group_row(
    statement: &Statement,
    layout: &GroupLayout,
    row: &Row,
    scale: E,
    out: &mut GroupRow,
) {
    if out.d.is_empty() {
        out.d
            .resize(1usize << (layout.h + layout.log_ell_zk), E::ZERO);
    }
    let member = |oracle: OracleId| layout.members.iter().position(|&id| id == oracle).unwrap();
    let carry_entry =
        |out: &mut GroupRow, oracle: OracleId, part: Part, index: u64, coefficient: E| {
            let i = member(oracle);
            if part == Part::Main && layout.kept[i] {
                kept_form(out, oracle).add_entry(index, coefficient * scale);
                return;
            }
            let a = statement.row_bits(oracle, part);
            let (first, bits) = match part {
                Part::Main => (layout.main_first[i], layout.k),
                Part::Mask => (layout.mask_first[i], layout.h),
            };
            let at = group_index(index, a, first, bits);
            match part {
                Part::Main => out.c.add_entry(at, coefficient * scale),
                Part::Mask => out.d[at as usize] += coefficient * scale,
            }
        };
    for term in &row.terms {
        let i = member(term.oracle);
        let a = statement.row_bits(term.oracle, term.part);
        let (first, bits) = match term.part {
            Part::Main => (layout.main_first[i], layout.k),
            Part::Mask => (layout.mask_first[i], layout.h),
        };
        let mut tensor = term.tensor.clone();
        tensor.scalar *= scale;
        if term.part == Part::Main && layout.kept[i] {
            kept_form(out, term.oracle).add(tensor);
            continue;
        }
        tensor.insert_choice(a, bits - a, first >> a);
        match term.part {
            Part::Main => out.c.add(tensor),
            Part::Mask => tensor.add_to(&mut out.d),
        }
    }
    for unit in &row.units {
        carry_entry(out, unit.oracle, unit.part, unit.index, unit.coefficient);
    }
    for run in &row.runs {
        for (j, &coefficient) in run.coefficients.iter().enumerate() {
            carry_entry(
                out,
                run.oracle,
                run.part,
                run.first + run.step * j as u64,
                coefficient,
            );
        }
    }
    for main in &row.forms {
        let i = member(main.oracle);
        if layout.kept[i] {
            kept_form(out, main.oracle).add_scaled(&main.form, scale);
            continue;
        }
        let a = statement.row_bits(main.oracle, Part::Main);
        let mut form = main.form.clone();
        form.insert_choice(a, layout.k - a, layout.main_first[i] >> a);
        out.c.add_scaled(&form, scale);
    }
    out.target += row.target * scale;
}

pub fn bat_combine(
    statement: &Statement,
    layout: &GroupLayout,
    rows: &[Row],
    challenge: E,
) -> GroupRow {
    let mut out = GroupRow {
        c: Form::default(),
        kept: vec![],
        d: vec![],
        target: E::ZERO,
    };
    let mut power = E::ONE;
    for row in rows {
        add_to_group_row(statement, layout, row, power, &mut out);
        power *= challenge;
    }
    out
}

#[derive(Debug, Clone)]
pub struct VectorLay {
    pub row_bits: u32,
    pub place_bits: u32,
    pub rows_high: bool,
}

impl VectorLay {
    pub fn bits(&self) -> u32 {
        self.row_bits + self.place_bits
    }
    pub fn index(&self, entry: u64) -> u64 {
        if self.rows_high {
            (entry >> self.place_bits)
                | ((entry & ((1u64 << self.place_bits) - 1)) << self.row_bits)
        } else {
            entry
        }
    }
    pub fn entry_at(&self, index: u64) -> u64 {
        if self.rows_high {
            (index >> self.row_bits) | ((index & ((1u64 << self.row_bits) - 1)) << self.place_bits)
        } else {
            index
        }
    }
    pub fn on_table(&self, mut tensor: Tensor) -> Tensor {
        tensor.insert_choice(tensor.vars(), self.bits() - tensor.vars(), 0);
        if self.rows_high {
            tensor.rotate(self.place_bits);
        }
        tensor
    }
}

pub fn sample_cube(statement: &Statement, oracle: OracleId) -> (bool, bool, u32, u32, u32) {
    let shape = &statement.oracles[oracle as usize].shape;
    let has_main = shape.main_rows != 0;
    let has_mask = shape.mask_rows != 0;
    let main_bits = if has_main {
        statement.table_bits(oracle, Part::Main)
    } else {
        0
    };
    let mask_bits = if has_mask {
        statement.table_bits(oracle, Part::Mask)
    } else {
        0
    };
    let bits = if has_main && has_mask {
        main_bits.max(mask_bits) + 1
    } else {
        main_bits + mask_bits
    };
    (has_main, has_mask, main_bits, mask_bits, bits)
}

pub fn pad_places(statement: &Statement, oracle: OracleId) -> Vec<(Part, u64, u32, E)> {
    let shape = &statement.oracles[oracle as usize].shape;
    let (_, has_mask, _, _, _) = sample_cube(statement, oracle);
    let part = if has_mask { Part::Mask } else { Part::Main };
    let rows = if has_mask {
        shape.mask_rows
    } else {
        shape.main_rows
    };
    let row_bits = shape.row_bits(part);
    let len = match part {
        Part::Main => statement.base[shape.code as usize].ell,
        Part::Mask => statement.mask[shape.mask_code as usize].ell,
    };
    let each = if has_mask || shape.main_deg == 3 {
        1
    } else {
        3
    };
    let mut out = Vec::new();
    for answer in 0..shape.answers {
        for t in 0..each {
            let place = len - each * shape.answers as u64 + each * answer as u64 + t;
            out.push((
                part,
                (rows - 1) + (place << row_bits),
                answer,
                E::theta(t as usize),
            ));
        }
    }
    out
}

pub fn send_form(statement: &Statement, oracle: OracleId, z: &[E], beta: E, answer: u32) -> Row {
    let (has_main, has_mask, main_bits, mask_bits, bits) = sample_cube(statement, oracle);
    let mut on_main = Tensor::eq(&z[..main_bits as usize], E::ONE);
    let mut on_mask = Tensor::eq(&z[..mask_bits as usize], E::ONE);
    if has_main && has_mask {
        let front = |table_bits: u32, mask: bool| {
            let mut factor = if mask {
                z[bits as usize - 1]
            } else {
                E::ONE - z[bits as usize - 1]
            };
            for t in table_bits as usize..bits as usize - 1 {
                factor *= E::ONE - z[t];
            }
            factor
        };
        on_main.scalar = front(main_bits, false);
        on_mask.scalar = front(mask_bits, true);
    }
    let mut row = Row {
        kind: RowKind::Sample,
        ..Row::default()
    };
    for (part, index, which, basis) in pad_places(statement, oracle) {
        let under = match part {
            Part::Main => on_main.at(index),
            Part::Mask => on_mask.at(index),
        };
        row.add_unit(
            oracle,
            part,
            index,
            if which == answer {
                beta * basis
            } else {
                E::ZERO
            } - under,
        );
    }
    if has_main {
        row.add(oracle, Part::Main, on_main);
    }
    if has_mask {
        row.add(oracle, Part::Mask, on_mask);
    }
    row
}

pub fn extension_seed(seed: &SampleSeed) -> (Vec<E>, E) {
    (
        seed.z.iter().copied().map(E::from_words).collect(),
        E::from_words(seed.beta),
    )
}

#[derive(Debug, Clone)]
pub struct FoldedRow {
    pub oracle: OracleId,
    pub lay: VectorLay,
}

pub fn foldeq_record_row(
    table_code: &CodeSize,
    table_v: u32,
    record: &IndexName,
    answer: usize,
    seed: &SampleSeed,
    eq_gamma: &[E],
    at: &FoldedRow,
) -> Row {
    let (z, beta) = extension_seed(seed);
    let plain = table_code.ell - 3;
    let under = Tensor::eq(&z, E::ONE);
    let mut row = Row::default();
    for tensor in tensor_below(&under, 0, log_ell(table_code), table_code.ell) {
        row.add(at.oracle, Part::Main, at.lay.on_table(tensor));
    }
    for a in 0..1u64 {
        for t in 0..3u64 {
            let p = plain + 3 * a + t;
            let coefficient = if a as usize == answer {
                beta * E::theta(t as usize)
            } else {
                E::ZERO
            } - under.at(p);
            row.add_unit(at.oracle, Part::Main, at.lay.index(p), coefficient);
        }
    }
    let rows = 1usize << table_v;
    for r in 0..rows {
        row.target += eq_gamma[r] * E::from_words(record.record[r]);
    }
    row
}

pub fn foldeq_column_row(table_code: &CodeSize, position: u64, target: E, at: &FoldedRow) -> Row {
    let generator = generator_at(table_code, position);
    let mut row = Row {
        kind: RowKind::Generator,
        target,
        ..Row::default()
    };
    let message = Tensor::product(
        &generator.factors[..log_ell(table_code) as usize],
        E::from_base(generator.message_scale),
    );
    for tensor in tensor_below(&message, 0, log_ell(table_code), table_code.ell) {
        row.add(at.oracle, Part::Main, at.lay.on_table(tensor));
    }
    if table_code.r != 0 {
        let tail = Tensor::product(
            &generator.factors[..log_r(table_code) as usize],
            E::from_base(generator.tail_scale),
        );
        let step = 1u64 << log_r(table_code);
        let tail_first = (table_code.ell + step - 1) / step * step;
        for mut tensor in tensor_below(&tail, 0, log_r(table_code), table_code.r) {
            tensor.insert_choice(
                log_r(table_code),
                at.lay.bits() - log_r(table_code),
                tail_first >> log_r(table_code),
            );
            row.add(at.oracle, Part::Main, at.lay.on_table(tensor));
        }
    }
    row
}

pub fn foldeq_symbol(eq_gamma: &[E], column: &[F]) -> E {
    let mut out = E::ZERO;
    for (&weight, &value) in eq_gamma.iter().zip(column) {
        out += weight * value;
    }
    out
}

pub fn link_equality_row(
    table_code: &CodeSize,
    phi: &[E],
    first: &FoldedRow,
    further: &FoldedRow,
) -> Row {
    let mut row = Row::default();
    let cut = table_code.ell - 3;
    for (at, sign) in [(first, E::ONE), (further, -E::ONE)] {
        let tensor = Tensor::eq(phi, sign);
        for piece in tensor_below(&tensor, 0, phi.len() as u32, cut) {
            row.add(at.oracle, Part::Main, at.lay.on_table(piece));
        }
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(log_n: u32, ell: u64, r: u64) -> CodeSize {
        CodeSize {
            log_n,
            ell,
            r,
            over_E: true,
        }
    }

    #[test]
    fn bat_batches_sparse_and_product_coefficients() {
        let mut statement = Statement::new(vec![code(4, 4, 0)], vec![code(3, 2, 0)]);
        let oracle = statement.sent(OracleShape {
            code: 0,
            mask_code: 0,
            main_rows: 1,
            mask_rows: 0,
            main_deg: 3,
            answers: 1,
        });
        let group = statement.new_group(0, 0);
        statement.join(group, oracle);
        let z = vec![E([2, 3, 5]), E([7, 11, 13])];
        let mut first = Row::default();
        first.add(oracle, Part::Main, Tensor::eq(&z, E::ONE));
        first.target = E([17, 19, 23]);
        statement.add_row(group, first.clone());
        let (layout, rows) = statement.begin_run(group);
        let challenge = E([29, 31, 37]);
        let batched = bat_combine(&statement, &layout, &rows, challenge);
        assert_eq!(batched.target, first.target);
        assert_eq!(batched.c.at(0), first.terms[0].tensor.at(0));
    }

    #[test]
    fn bat_keeps_claims_on_their_original_table() {
        let mut statement = Statement::new(vec![code(4, 4, 0)], vec![code(3, 2, 0)]);
        let kept = statement.sent(OracleShape {
            code: 0,
            mask_code: 0,
            main_rows: 1,
            mask_rows: 0,
            main_deg: 3,
            answers: 0,
        });
        let folded = statement.sent(OracleShape {
            code: 0,
            mask_code: 0,
            main_rows: 1,
            mask_rows: 0,
            main_deg: 3,
            answers: 0,
        });
        let group = statement.new_group(0, 0);
        statement.join_kept(group, kept, true);
        statement.join(group, folded);
        let z = vec![E([2, 3, 5]), E([7, 11, 13])];
        let mut row = Row::default();
        row.add(kept, Part::Main, Tensor::eq(&z, E::ONE));
        row.add(folded, Part::Main, Tensor::eq(&z, E::ONE));
        row.add_unit(kept, Part::Main, 3, E([17, 19, 23]));
        statement.add_row(group, row.clone());
        let (layout, rows) = statement.begin_run(group);
        let batched = bat_combine(&statement, &layout, &rows, E::ONE);
        assert!(layout.kept[0]);
        assert_eq!(layout.main_first[0], u64::MAX);
        let mut expected = row.terms[1].tensor.clone();
        expected.insert_choice(0, 1, 0);
        assert_eq!(batched.c.at(2), expected.at(2));
        assert_eq!(batched.kept[0].c.at(3), z[0] * z[1] + E([17, 19, 23]));
    }
}
