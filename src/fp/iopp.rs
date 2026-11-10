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

pub fn execution_stage(
    codes: &crate::fp::key_public::Codes,
    stage: &crate::fp::key_public::IoppStage,
) -> crate::fp::analysis::Stage {
    crate::fp::analysis::Stage {
        g: crate::fp::analysis::GroupSize {
            k: stage.k,
            h: stage.h,
            slots: 1,
            main: codes.base[stage.code as usize],
            mask: codes.mask[stage.mask_code as usize],
            m: 0,
        },
        t: 0,
        r_ood: 0,
    }
}

#[derive(Debug)]
pub struct Iopp {
    pub statement: Statement,
    pub group: GroupId,
    pub base_ids: Vec<u64>,
    pub mask_ids: Vec<u64>,
    pub last_target: Option<SwitchTarget>,
    pub fresh_override: Option<u32>,
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
            fresh_override: None,
        }
    }

    pub fn with_statement(statement: Statement) -> Self {
        let mut out = Self::new(vec![], vec![]);
        out.statement = statement;
        out
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
        target: SwitchTarget,
        masks_ride: bool,
    ) -> Result<GroupRow, FpError> {
        self.fold_cs_inner(
            stages,
            samples,
            stage,
            row,
            hiding,
            epsilon,
            polynomials,
            challenges,
            z,
            beta,
            answer,
            positions,
            theta,
            opened,
            anew,
            None,
            target,
            masks_ride,
            true,
            self.group,
        )
        .map(|(row, _)| row.unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_cs_pre_sent(
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
        oracle: u32,
        target: SwitchTarget,
        to_group: crate::fp::claims::GroupId,
    ) -> Result<(Option<GroupRow>, crate::fp::proximity::SingleRows), FpError> {
        self.fold_cs_inner(
            stages,
            samples,
            stage,
            row,
            hiding,
            epsilon,
            polynomials,
            challenges,
            z,
            beta,
            answer,
            positions,
            theta,
            opened,
            anew,
            Some(oracle),
            target,
            false,
            false,
            to_group,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_cs_inner(
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
        pre_sent: Option<u32>,
        mut target: SwitchTarget,
        masks_ride: bool,
        batch: bool,
        to_group: crate::fp::claims::GroupId,
    ) -> Result<(Option<GroupRow>, crate::fp::proximity::SingleRows), FpError> {
        let next = stages[stage + 1];
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!(
                "RUST_INITIAL_FORM c0={:016x} c1={:016x} c2={:016x} c3={:016x}",
                row.c.at(0).0[0],
                row.c.at(1).0[0],
                row.c.at(2).0[0],
                row.c.at(3).0[0]
            );
        }
        let mut claim = hiding + epsilon * row.target;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!("RUST_FOLD_INPUT sum={:016x} {:016x} {:016x} eps={:016x} {:016x} {:016x} target={:016x} {:016x} {:016x}", hiding.0[0], hiding.0[1], hiding.0[2], epsilon.0[0], epsilon.0[1], epsilon.0[2], row.target.0[0], row.target.0[1], row.target.0[2]);
        }
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            eprintln!(
                "RUST_FOLD_INITIAL {:016x} {:016x} {:016x}",
                claim.0[0], claim.0[1], claim.0[2]
            );
        }
        for (&(c0, c2), &challenge) in polynomials.iter().zip(challenges) {
            claim = fold_round_claim(claim, c0, c2, challenge);
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_FOLD_CLAIM {:016x}", claim.0[0]);
            }
        }
        let layout = self.statement.layout(self.group);
        let folded = fold_output(
            &self.statement,
            self.base_ids[stage] as u32,
            self.mask_ids[stage] as u32,
            layout,
            row,
            self.fresh_override
                .unwrap_or(self.statement.oracles.len() as u32 - 1),
            epsilon,
            challenges.to_vec(),
            claim,
        );
        let symbols = single_rows_symbols(&folded, opened, opened.of.first().map_or(0, Vec::len));
        let oracle = pre_sent.unwrap_or_else(|| self.statement.sent(target.shape.clone()));
        if masks_ride && pre_sent.is_none() {
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
        if pre_sent.is_none() {
            self.group = self
                .statement
                .new_group(target.shape.code, target.shape.mask_code);
        }
        if pre_sent.is_some() {
            self.group = to_group;
        }
        if pre_sent.is_none() {
            self.statement.join(self.group, oracle);
        }
        if pre_sent.is_none() {
            let mut sample = send_form(&self.statement, oracle, z, beta, 0);
            sample.target = answer;
            self.statement.oracles[oracle as usize].sample_rows = vec![sample];
        }
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
        let row = if batch {
            let (layout, rows) = self.statement.begin_run(self.group);
            Some(bat_combine(&self.statement, &layout, &rows, theta))
        } else {
            None
        };
        target.shape.answers = samples;
        self.last_target = Some(target);
        Ok((row, folded))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn base(
        &mut self,
        stages: &[Stage],
        last: usize,
        row: GroupRow,
        target_star: E,
        lambda: E,
        clear: &[E],
        opened: &Opened,
        positions: &[u64],
    ) -> Result<(), FpError> {
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
            #[cfg(feature = "std")]
            eprintln!("BASE_CLEAR expected={} actual={} widths={:?} tails={:?} kept_rows={} masks={} code={} mask_code={} main={:?} mask={:?}", expected, clear.len(), widths, tail_widths, rows.kept_rows, rows.masks, rows.code, rows.mask_code, self.statement.base[rows.code as usize], self.statement.mask[rows.mask_code as usize]);
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
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            let main = crate::fp::tensor::inner(
                &{
                    let mut coefficients =
                        vec![E::ZERO; self.statement.base[rows.code as usize].ell as usize];
                    rows.c.add_to(&mut coefficients);
                    coefficients
                },
                &messages[0],
            );
            let mask = if rows.masks == 0 {
                E::ZERO
            } else {
                crate::fp::tensor::inner(&rows.d[0], &messages[1])
            };
            eprintln!("RUST_SHARED_BASE main={:016x} mask={:016x} target_star={:016x} lambda={:016x} target={:016x} left={:016x} right={:016x} d0={:016x} d1={:016x} d2={:016x} d3={:016x}", main.0[0], mask.0[0], target_star.0[0], lambda.0[0], rows.target.0[0], left.0[0], right.0[0], rows.d[0].get(0).map_or(E::ZERO, |x| *x).0[0], rows.d[0].get(1).map_or(E::ZERO, |x| *x).0[0], rows.d[0].get(2).map_or(E::ZERO, |x| *x).0[0], rows.d[0].get(3).map_or(E::ZERO, |x| *x).0[0]);
            eprintln!("RUST_BASE_FORM c0={:016x} c1={:016x} c2={:016x} c3={:016x} m0={:016x} m1={:016x} m2={:016x} m3={:016x}", rows.c.at(0).0[0], rows.c.at(1).0[0], rows.c.at(2).0[0], rows.c.at(3).0[0], messages[0][0].0[0], messages[0][1].0[0], messages[0][2].0[0], messages[0][3].0[0]);
        }
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
