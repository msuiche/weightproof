//! Algebraic LINK verification over the live transcript and verified openings.

use crate::fp::{
    claims::{
        bat_combine, foldeq_column_row, foldeq_record_row, foldeq_symbol, link_equality_row,
        send_form, FoldedRow, GroupRow, OracleShape, VectorLay,
    },
    code_eval::log_ell,
    codes::CodeSize,
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    heads::{record_seeds, table_name},
    iopp::Iopp,
    link_public::{unique_codes, IndexName},
    plan::{solve_entry_stages, EntryLayoutInput},
    proximity::{
        fold_output, fold_round_claim, switch_target_stage, Opened, OpenedColumn, SingleRows,
    },
    tensor::ceil_log2,
    transcript::OpeningData,
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
    iopp: Iopp,
}

impl LinkAlgebra {
    pub fn new(input: &EntryLayoutInput, configuration: &Digest) -> Self {
        let stages = solve_entry_stages(input).unwrap();
        let (base, base_ids) = unique_codes(&stages, false);
        let (mask, mask_ids) = unique_codes(&stages, true);
        let mut iopp = Iopp::new(base, mask);
        iopp.base_ids = base_ids;
        iopp.mask_ids = mask_ids;
        let rows = input.folded_message_rows;
        for _ in 0..2 {
            iopp.statement.sent(OracleShape {
                code: 0,
                mask_code: 0,
                main_rows: rows,
                mask_rows: 0,
                main_deg: 3,
                answers: input.setting.samples as u32,
            });
        }
        iopp.statement.sent(OracleShape {
            code: 0,
            mask_code: 0,
            main_rows: 0,
            mask_rows: 1u64 << input.first_group.h,
            main_deg: 3,
            answers: 0,
        });
        iopp.group = iopp.statement.new_group(0, 0);
        iopp.statement.join(iopp.group, 0);
        iopp.statement.join(iopp.group, 1);
        Self {
            input: input.clone(),
            configuration: *configuration,
            iopp,
        }
    }

    fn folded_row(&self, oracle: u32) -> FoldedRow {
        FoldedRow {
            oracle,
            lay: VectorLay {
                row_bits: ceil_log2(self.input.folded_message_rows),
                place_bits: log_ell(&self.iopp.statement.base[0]),
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
            let mut sample = send_form(&self.iopp.statement, oracle as u32, z, beta, 0);
            sample.target = answer;
            self.iopp.statement.oracles[oracle].sample_rows = vec![sample];
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
        self.iopp.statement.add_row(self.iopp.group, first_record);
        self.iopp.statement.add_row(self.iopp.group, further_record);

        for (opening, at, draws) in [
            (&opened[0], &first_folded, positions_first),
            (&opened[1], &further_folded, positions_further),
        ] {
            for &position in draws {
                let column = opening.column_f(bytes, position, 0, 256);
                let target = foldeq_symbol(&eq_gamma, &column);
                self.iopp.statement.add_row(
                    self.iopp.group,
                    foldeq_column_row(&code, position, target, at),
                );
            }
        }
        self.iopp.statement.add_row(
            self.iopp.group,
            link_equality_row(&code, phi, &first_folded, &further_folded),
        );
        let (layout, rows) = self.iopp.statement.begin_run(self.iopp.group);
        Ok(bat_combine(&self.iopp.statement, &layout, &rows, batch))
    }

    fn opened_rows(&self, rows: &SingleRows, opening: &OpeningData, bytes: &[u8]) -> Opened {
        let mut out = Opened::default();
        for &position in &opening.positions {
            let mut columns = Vec::new();
            match &self.iopp.last_target {
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
        let folded = fold_output(
            &self.iopp.statement,
            self.iopp.base_ids[stage] as u32,
            self.iopp.mask_ids[stage] as u32,
            self.iopp.statement.layout(self.iopp.group),
            row.clone(),
            self.iopp.statement.oracles.len() as u32 - 1,
            epsilon,
            challenges.to_vec(),
            {
                let mut claim = hiding + epsilon * row.target;
                for (&(c0, c2), &challenge) in polynomials.iter().zip(challenges) {
                    claim = fold_round_claim(claim, c0, c2, challenge);
                }
                claim
            },
        );
        let opened = self.opened_rows(&folded, opening, bytes);
        let target = switch_target_stage(
            self.iopp.base_ids[stage + 1] as u32,
            self.iopp.mask_ids[stage + 1] as u32,
            stages[stage + 1].g.k,
            stages[stage + 1].g.h,
            self.input.setting.samples as u32,
        );
        self.iopp.fold_cs(
            &stages,
            self.input.setting.samples as u32,
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
            &opened,
            &[],
            target,
            true,
        )
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
        let target = self.iopp.last_target.clone().unwrap();
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
        self.iopp.base(
            &stages,
            stages.len() - 1,
            row,
            target_star,
            lambda,
            clear,
            &opened,
            &opening.positions,
        )
    }
}
