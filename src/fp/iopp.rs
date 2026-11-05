//! Reusable verifier-only IOPP transitions over a statement and verified openings.

use crate::fp::{
    analysis::Stage,
    claims::{bat_combine, send_form, GroupId, GroupRow, OracleShape, Statement},
    error::FpError,
    field::ExtensionElement as E,
    proximity::{
        base_linear_left, base_symbol, fold_output, fold_round_claim, last_rows,
        single_rows_symbols, switch_rows, Opened, SwitchTarget,
    },
};
use alloc::{vec, vec::Vec};

#[derive(Debug)]
pub struct Iopp {
    pub statement: Statement,
    pub group: GroupId,
    pub base_ids: Vec<u64>,
    pub mask_ids: Vec<u64>,
    pub last_target: Option<SwitchTarget>,
}

impl Iopp {
    pub fn new(
        base: Vec<crate::fp::codes::CodeSize>,
        mask: Vec<crate::fp::codes::CodeSize>,
    ) -> Self {
        Self {
            statement: Statement::new(base, mask),
            group: 0,
            base_ids: vec![],
            mask_ids: vec![],
            last_target: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_cs(
        &mut self,
        stages: &[Stage],
        samples: u32,
        stage: usize,
        row: GroupRow,
        hiding: E,
        epsilon: E,
        polynomials: &[(E, E)],
        challenges: &[E],
        z: &[E],
        beta: E,
        answer: E,
        positions: &[u64],
        theta: E,
        opened: &Opened,
        anew: &[u32],
        mut target: SwitchTarget,
        masks_ride: bool,
    ) -> Result<GroupRow, FpError> {
        let next = stages[stage + 1];
        let mut claim = hiding + epsilon * row.target;
        for (&(c0, c2), &challenge) in polynomials.iter().zip(challenges) {
            claim = fold_round_claim(claim, c0, c2, challenge);
        }
        let layout = self.statement.layout(self.group);
        let folded = fold_output(
            &self.statement,
            self.base_ids[stage] as u32,
            self.mask_ids[stage] as u32,
            layout,
            row,
            self.statement.oracles.len() as u32 - 1,
            epsilon,
            challenges.to_vec(),
            claim,
        );
        let symbols = single_rows_symbols(&folded, opened, opened.of.first().map_or(0, Vec::len));
        let oracle = self.statement.sent(target.shape.clone());
        if masks_ride {
            let to_last = stage + 2 == stages.len();
            let _with = self.statement.sent(if to_last {
                OracleShape {
                    code: target.shape.code,
                    mask_code: target.shape.mask_code,
                    main_rows: 1,
                    mask_rows: 1,
                    main_deg: 3,
                    answers: 0,
                }
            } else {
                OracleShape {
                    code: 0,
                    mask_code: target.shape.mask_code,
                    main_rows: 0,
                    mask_rows: 1u64 << next.g.h,
                    main_deg: 3,
                    answers: 0,
                }
            });
        }
        self.group = self
            .statement
            .new_group(target.shape.code, target.shape.mask_code);
        self.statement.join(self.group, oracle);
        let mut sample = send_form(&self.statement, oracle, z, beta, 0);
        sample.target = answer;
        self.statement.oracles[oracle as usize].sample_rows = vec![sample];
        for claim in switch_rows(
            &self.statement,
            &folded,
            &target,
            oracle,
            anew,
            positions,
            &symbols,
        ) {
            self.statement.add_row(self.group, claim);
        }
        let (layout, rows) = self.statement.begin_run(self.group);
        target.shape.answers = samples;
        self.last_target = Some(target);
        Ok(bat_combine(&self.statement, &layout, &rows, theta))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn base(
        &mut self,
        stages: &[Stage],
        row: GroupRow,
        target_star: E,
        lambda: E,
        clear: &[E],
        opened: &Opened,
        positions: &[u64],
    ) -> Result<(), FpError> {
        let last = stages.len() - 1;
        let layout = self.statement.layout(self.group);
        let rows = last_rows(
            &self.statement,
            self.base_ids[last] as u32,
            self.mask_ids[last] as u32,
            layout,
            row,
        );
        let mut widths = Vec::new();
        let mut tail_widths = Vec::new();
        widths.push(self.statement.base[rows.code as usize].ell as usize);
        tail_widths.push(self.statement.base[rows.code as usize].r as usize);
        for kept in &rows.kept {
            let shape = &self.statement.oracles[kept.oracle as usize].shape;
            widths.push(
                self.statement.base[shape.code as usize].ell as usize * shape.main_rows as usize,
            );
            tail_widths.push(
                self.statement.base[shape.code as usize].r as usize * shape.main_rows as usize,
            );
        }
        for _ in 0..rows.masks {
            widths.push(self.statement.mask[rows.mask_code as usize].ell as usize);
            tail_widths.push(self.statement.mask[rows.mask_code as usize].r as usize);
        }
        let expected: usize = widths.iter().sum::<usize>() + tail_widths.iter().sum::<usize>();
        if clear.len() != expected {
            return Err(FpError::BadLength {
                context: "base-clear-values",
            });
        }
        let mut messages = Vec::with_capacity(widths.len());
        let mut tails = Vec::with_capacity(widths.len());
        let mut at = 0;
        for &width in &widths {
            messages.push(clear[at..at + width].to_vec());
            at += width;
        }
        for &width in &tail_widths {
            tails.push(clear[at..at + width].to_vec());
            at += width;
        }
        let flat: Vec<E> = messages.concat();
        let left = base_linear_left(&self.statement, &rows, &flat);
        let right = target_star + lambda * rows.target;
        if left != right {
            return Err(FpError::Range {
                context: "base-linear-claim",
            });
        }
        let symbols = single_rows_symbols(&rows, opened, opened.of.first().map_or(0, Vec::len));
        for i in 0..positions.len() {
            let star = &opened.of.last().unwrap()[i];
            for row_index in 0..rows.rows() {
                let star_value = if row_index == 0 {
                    star.main[0]
                } else if row_index <= rows.kept_rows {
                    star.main[row_index]
                } else {
                    star.mask[row_index - rows.kept_rows - 1]
                };
                let masked = star_value + lambda * symbols[i * rows.rows() + row_index];
                let actual = base_symbol(
                    &self.statement,
                    &rows,
                    row_index,
                    positions[i],
                    &messages,
                    &tails,
                );
                if actual != masked {
                    return Err(FpError::Range {
                        context: "base-encoding",
                    });
                }
            }
        }
        Ok(())
    }
}
