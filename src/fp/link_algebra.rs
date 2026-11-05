//! Algebraic LINK verification over the live transcript and verified openings.

use crate::fp::{
    claims::{
        bat_combine, foldeq_column_row, foldeq_record_row, foldeq_symbol, link_equality_row,
        send_form, FoldedRow, GroupRow, OracleShape, Statement, VectorLay,
    },
    code_eval::log_ell,
    codes::CodeSize,
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    heads::{record_seeds, table_name},
    link_public::{unique_codes, IndexName},
    plan::{solve_entry_stages, EntryLayoutInput},
    proximity::{
        base_linear_left, base_symbol, fold_output, fold_round_claim, last_rows,
        single_rows_symbols, switch_rows, switch_target_stage, Opened, OpenedColumn, SingleRows,
        SwitchTarget,
    },
    tensor::ceil_log2,
    verify_link::OpeningData,
};
use alloc::{vec, vec::Vec};

const TABLE_V: u32 = 8;

fn table_code() -> CodeSize {
    CodeSize {
        log_n: 23,
        ell: 524_288,
        r: 65_535,
        over_E: false,
    }
}

pub struct LinkAlgebra {
    input: EntryLayoutInput,
    configuration: Digest,
    statement: Statement,
    group: u32,
    base_ids: Vec<u64>,
    mask_ids: Vec<u64>,
    last_target: Option<SwitchTarget>,
}

impl LinkAlgebra {
    pub fn new(input: &EntryLayoutInput, configuration: &Digest) -> Self {
        let stages = solve_entry_stages(input).unwrap();
        let (base, base_ids) = unique_codes(&stages, false);
        let (mask, mask_ids) = unique_codes(&stages, true);
        let mut statement = Statement::new(base, mask);
        let rows = input.folded_message_rows;
        for _ in 0..2 {
            statement.sent(OracleShape {
                code: 0,
                mask_code: 0,
                main_rows: rows,
                mask_rows: 0,
                main_deg: 3,
                answers: input.setting.samples as u32,
            });
        }
        statement.sent(OracleShape {
            code: 0,
            mask_code: 0,
            main_rows: 0,
            mask_rows: 1u64 << input.first_group.h,
            main_deg: 3,
            answers: 0,
        });
        let group = statement.new_group(0, 0);
        statement.join(group, 0);
        statement.join(group, 1);
        Self {
            input: input.clone(),
            configuration: *configuration,
            statement,
            group,
            base_ids,
            mask_ids,
            last_target: None,
        }
    }

    fn folded_row(&self, oracle: u32) -> FoldedRow {
        FoldedRow {
            oracle,
            lay: VectorLay {
                row_bits: ceil_log2(self.input.folded_message_rows),
                place_bits: log_ell(&self.statement.base[0]),
                rows_high: true,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initial(
        &mut self,
        first: &IndexName,
        further: &IndexName,
        gamma: &[E],
        z: &[E],
        beta: E,
        answers: &[E],
        phi: &[E],
        batch: E,
        opened: &[OpeningData],
        positions_first: &[u64],
        positions_further: &[u64],
        bytes: &[u8],
    ) -> Result<GroupRow, FpError> {
        let first_folded = self.folded_row(0);
        let further_folded = self.folded_row(1);
        for (oracle, answer) in [(0usize, answers[0]), (1usize, answers[1])] {
            let mut sample = send_form(&self.statement, oracle as u32, z, beta, 0);
            sample.target = answer;
            self.statement.oracles[oracle].sample_rows = vec![sample];
        }

        let code = table_code();
        let shape = [code.log_n as u64, code.ell, code.r, TABLE_V as u64, 1, 40];
        let first_name = table_name(&first.commitment, &self.configuration, shape);
        let further_name = table_name(&further.commitment, &self.configuration, shape);
        let first_seed = record_seeds(&first_name, 19, 1);
        let further_seed = record_seeds(&further_name, 19, 1);
        let eq_gamma = crate::fp::tensor::eq_table(gamma);
        let first_record = foldeq_record_row(
            &code,
            TABLE_V,
            first,
            0,
            &first_seed[0],
            &eq_gamma,
            &first_folded,
        );
        let further_record = foldeq_record_row(
            &code,
            TABLE_V,
            further,
            0,
            &further_seed[0],
            &eq_gamma,
            &further_folded,
        );
        self.statement.add_row(self.group, first_record);
        self.statement.add_row(self.group, further_record);

        for (opening, at, draws) in [
            (&opened[0], &first_folded, positions_first),
            (&opened[1], &further_folded, positions_further),
        ] {
            for &position in draws {
                let column = opening.column_f(bytes, position, 0, 256);
                let target = foldeq_symbol(&eq_gamma, &column);
                self.statement
                    .add_row(self.group, foldeq_column_row(&code, position, target, at));
            }
        }
        self.statement.add_row(
            self.group,
            link_equality_row(&code, phi, &first_folded, &further_folded),
        );
        let (layout, rows) = self.statement.begin_run(self.group);
        Ok(bat_combine(&self.statement, &layout, &rows, batch))
    }

    fn opened_rows(&self, rows: &SingleRows, opening: &OpeningData, bytes: &[u8]) -> Opened {
        let mut out = Opened::default();
        for &position in &opening.positions {
            let mut columns = Vec::new();
            match &self.last_target {
                None => {
                    let main = self.input.folded_message_rows as usize;
                    columns.push(OpenedColumn {
                        main: opening.column_e(bytes, position, 0, main),
                        mask: vec![],
                    });
                    columns.push(OpenedColumn {
                        main: opening.column_e(bytes, position, main * 3, main),
                        mask: vec![],
                    });
                    if rows.masks == 2 {
                        columns.push(OpenedColumn {
                            main: vec![],
                            mask: opening.column_e(
                                bytes,
                                position,
                                2 * main * 3,
                                1usize << rows.layout.h,
                            ),
                        });
                    }
                }
                Some(target) => {
                    let main = target.shape.main_rows as usize;
                    let mask = target.shape.mask_rows as usize;
                    columns.push(OpenedColumn {
                        main: opening.column_e(bytes, position, 0, main),
                        mask: opening.column_e(bytes, position, main * 3, mask),
                    });
                    if rows.masks == 2 {
                        columns.push(OpenedColumn {
                            main: vec![],
                            mask: opening.column_e(
                                bytes,
                                position,
                                (main + mask) * 3,
                                1usize << rows.layout.h,
                            ),
                        });
                    }
                }
            }
            for (i, column) in columns.into_iter().enumerate() {
                if out.of.len() <= i {
                    out.of.push(vec![]);
                }
                out.of[i].push(column);
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub fn fold_cs(
        &mut self,
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
        opening: &OpeningData,
        bytes: &[u8],
    ) -> Result<GroupRow, FpError> {
        let stages = solve_entry_stages(&self.input)?;
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
        let opened = self.opened_rows(&folded, opening, bytes);
        let symbols = single_rows_symbols(&folded, &opened, opening.positions.len());

        let target = switch_target_stage(
            self.base_ids[stage + 1] as u32,
            self.mask_ids[stage + 1] as u32,
            next.g.k,
            next.g.h,
            self.input.setting.samples as u32,
        );
        let oracle = self.statement.sent(target.shape.clone());
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
            positions,
            &symbols,
        ) {
            self.statement.add_row(self.group, claim);
        }
        let (layout, rows) = self.statement.begin_run(self.group);
        self.last_target = Some(target);
        Ok(bat_combine(&self.statement, &layout, &rows, theta))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn base(
        &mut self,
        row: GroupRow,
        target_star: E,
        lambda: E,
        clear: &[E],
        _positions: &[u64],
        opening: &OpeningData,
        bytes: &[u8],
    ) -> Result<(), FpError> {
        let stages = solve_entry_stages(&self.input)?;
        let last = stages.len() - 1;
        let layout = self.statement.layout(self.group);
        let rows = last_rows(
            &self.statement,
            self.base_ids[last] as u32,
            self.mask_ids[last] as u32,
            layout,
            row,
        );
        let main_ell = self.statement.base[rows.code as usize].ell as usize;
        let mask_ell = self.statement.mask[rows.mask_code as usize].ell as usize;
        let main_r = self.statement.base[rows.code as usize].r as usize;
        let mask_r = self.statement.mask[rows.mask_code as usize].r as usize;
        let expected = main_ell + mask_ell + main_r + mask_r;
        if clear.len() != expected {
            return Err(FpError::BadLength {
                context: "base-clear-values",
            });
        }
        let messages = vec![
            clear[..main_ell].to_vec(),
            clear[main_ell..main_ell + mask_ell].to_vec(),
        ];
        let tails = vec![
            clear[main_ell + mask_ell..main_ell + mask_ell + main_r].to_vec(),
            clear[main_ell + mask_ell + main_r..].to_vec(),
        ];
        let flat: Vec<E> = messages.concat();
        let left = base_linear_left(&self.statement, &rows, &flat);
        let right = target_star + lambda * rows.target;
        if left != right {
            return Err(FpError::Range {
                context: "base-linear-claim",
            });
        }

        let target = self.last_target.clone().unwrap();
        let mut opened = Opened::default();
        for &position in &opening.positions {
            let main = target.shape.main_rows as usize;
            let mask = target.shape.mask_rows as usize;
            let columns = vec![
                OpenedColumn {
                    main: opening.column_e(bytes, position, 0, main),
                    mask: opening.column_e(bytes, position, (main + 1) * 3, mask),
                },
                OpenedColumn {
                    main: opening.column_e(bytes, position, main * 3, 1),
                    mask: opening.column_e(bytes, position, (main + 1 + mask) * 3, 1),
                },
            ];
            for (i, column) in columns.into_iter().enumerate() {
                if opened.of.len() <= i {
                    opened.of.push(vec![]);
                }
                opened.of[i].push(column);
            }
        }
        let symbols = single_rows_symbols(&rows, &opened, opening.positions.len());
        for (i, &position) in opening.positions.iter().enumerate() {
            let star = &opened.of[1][i];
            for row_index in 0..2 {
                let masked = if row_index == 0 {
                    star.main[0]
                } else {
                    star.mask[0]
                } + lambda * symbols[i * 2 + row_index];
                let actual = base_symbol(
                    &self.statement,
                    &rows,
                    row_index,
                    position,
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
