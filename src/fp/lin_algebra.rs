//! Algebraic shared-LIN verification: FOLDEQ, Omega, and their groups.

use alloc::{vec, vec::Vec};

use crate::fp::{
    claims::{
        bat_combine, foldeq_column_row, foldeq_record_row, foldeq_symbol, send_form, FoldedRow,
        GroupId, GroupRow, OracleShape, VectorLay,
    },
    code_eval::log_ell,
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    heads::{record_seeds, table_name},
    iopp::{execution_stage, Iopp},
    key_public::{Codes, LinShape},
    link_public::{unique_codes, IndexName},
    proximity::Opened,
    tensor::ceil_log2,
    transcript::OpeningData,
};

#[derive(Debug)]
pub struct LinShared {
    pub u: FoldedRow,
    pub omega: u32,
    pub split: Option<u32>,
    pub group_u: GroupId,
    pub group_omega: GroupId,
    pub mask: Option<u32>,
}

pub struct LinSharedAlgebra {
    pub shape: LinShape,
    pub configuration: Digest,
    pub iopp: Iopp,
    pub shared: LinShared,
}

impl LinSharedAlgebra {
    pub fn new(shape: &LinShape, codes: &Codes, configuration: &Digest) -> Self {
        let stages: Vec<_> = shape
            .test_u
            .iter()
            .chain(&shape.test_omega)
            .map(|stage| execution_stage(codes, stage))
            .collect();
        let (base, base_ids) = unique_codes(&stages, false);
        let (mask, mask_ids) = unique_codes(&stages, true);
        let mut iopp = Iopp::new(base, mask);
        iopp.base_ids = base_ids;
        iopp.mask_ids = mask_ids;
        let u = iopp.statement.sent(shape.fold.u.clone());
        let fold_mask = if shape.masks_ahead {
            Some(iopp.statement.sent(OracleShape {
                code: 0,
                mask_code: shape.test_u[0].mask_code,
                main_rows: 0,
                mask_rows: 1u64 << shape.test_u[0].h,
                main_deg: 3,
                answers: 0,
            }))
        } else {
            None
        };
        let omega = iopp.statement.sent(shape.omega.clone());
        let split = if shape.fold.u.code != shape.omega.code {
            Some(iopp.statement.sent(shape.shared_split.clone()))
        } else {
            None
        };
        let group_u = iopp.statement.new_group(shape.fold.u.code, shape.shared_mask_code);
        let group_omega = iopp.statement.new_group(shape.omega.code, shape.shared_mask_code);
        iopp.statement.join(group_u, u);
        iopp.statement.join(group_omega, omega);
        if let Some(split) = split {
            iopp.statement.join(group_u, split);
            iopp.statement.join(group_omega, split);
        }
        iopp.group = group_u;
        let shared = LinShared {
            u: FoldedRow {
                oracle: u,
                lay: VectorLay {
                    row_bits: ceil_log2(shape.fold.u.main_rows),
                    place_bits: log_ell(&iopp.statement.base[shape.fold.u.code as usize]),
                    rows_high: shape.fold.rows_high,
                },
            },
            omega,
            split,
            group_u,
            group_omega,
            mask: fold_mask,
        };
        Self {
            shape: shape.clone(),
            configuration: *configuration,
            iopp,
            shared,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn initial(
        &mut self,
        table: &IndexName,
        gamma: &[E],
        positions: &[u64],
        opening: &OpeningData,
        bytes: &[u8],
    ) -> Result<(), FpError> {
        let code = self.shape.fold.index.code;
        let shape = [
            code.log_n as u64,
            code.ell,
            code.r,
            self.shape.fold.index.v as u64,
            self.shape.fold.index.answers as u64,
            self.shape.fold.index.salt_bytes as u64,
        ];
        let name = table_name(&table.commitment, &self.configuration, shape);
        let seed = record_seeds(&name, log_ell(&code), 1);
        let eq_gamma = crate::fp::tensor::eq_table(gamma);
        let record = foldeq_record_row(
            &code,
            self.shape.fold.index.v,
            table,
            0,
            &seed[0],
            &eq_gamma,
            &self.shared.u,
        );
        self.iopp.statement.add_row(self.shared.group_u, record);
        for &position in positions {
            let column = opening.column_f(bytes, position, 0, 1usize << self.shape.fold.index.v);
            let target = foldeq_symbol(&eq_gamma, &column);
            self.iopp.statement.add_row(
                self.shared.group_u,
                foldeq_column_row(&code, position, target, &self.shared.u),
            );
        }
        Ok(())
    }

    pub fn sample_omega(&mut self, z: &[E], beta: E, answer: E) {
        let mut sample = send_form(&self.iopp.statement, self.shared.omega, z, beta, 0);
        sample.target = answer;
        self.iopp.statement.oracles[self.shared.omega as usize].sample_rows = vec![sample];
    }

    pub fn sample_split(&mut self, z: &[E], beta: E, answer: E) {
        if let Some(split) = self.shared.split {
            let mut sample = send_form(&self.iopp.statement, split, z, beta, 0);
            sample.target = answer;
            self.iopp.statement.oracles[split as usize].sample_rows = vec![sample];
        }
    }

    pub fn combine_u(&mut self, challenge: E) -> GroupRow {
        let (layout, rows) = self.iopp.statement.begin_run(self.shared.group_u);
        bat_combine(&self.iopp.statement, &layout, &rows, challenge)
    }

    pub fn combine_omega(&mut self, challenge: E) -> GroupRow {
        let (layout, rows) = self.iopp.statement.begin_run(self.shared.group_omega);
        bat_combine(&self.iopp.statement, &layout, &rows, challenge)
    }

    pub fn opened_rows(&self, opening: &OpeningData, bytes: &[u8], positions: usize) -> Opened {
        let mut out = Opened::default();
        let shape = &self.shape.fold.u;
        let main = shape.main_rows as usize;
        for &position in &opening.positions {
            let mut columns = Vec::new();
            columns.push(crate::fp::proximity::OpenedColumn {
                main: opening.column_e(bytes, position, 0, main),
                mask: vec![],
            });
            if self.shared.mask.is_some() {
                columns.push(crate::fp::proximity::OpenedColumn {
                    main: vec![],
                    mask: opening.column_e(bytes, position, main * 3, 1),
                });
            }
            for (i, column) in columns.into_iter().enumerate() {
                if out.of.len() <= i {
                    out.of.push(vec![]);
                }
                out.of[i].push(column);
            }
        }
        let _ = positions;
        out
    }
}
