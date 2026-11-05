//! Executable key transcript and WF verification.

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{bat_combine, send_form, Statement},
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    heads::parse_key_record,
    iopp::Iopp,
    key::KeyLayoutInput,
    key_public::{KeyPublic, LABEL_ONE_TIME_CHAIN},
    link_public::IndexName,
    psc::PscChannel,
    transcript::{elements, words, Cursor, OpeningData},
    wf::wf_rows_verify,
};
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};

#[derive(Debug, Clone, Copy)]
enum ReceiveKind {
    Round,
    End,
}

struct KeyChannel<'a, 'b> {
    cursor: &'a mut Cursor<'b>,
    chain: &'a mut Chain,
    context: String,
    expect: ReceiveKind,
    pending: Vec<E>,
    last_words: Vec<u64>,
}

impl<'a, 'b> KeyChannel<'a, 'b> {
    fn new(cursor: &'a mut Cursor<'b>, chain: &'a mut Chain, context: &'a str) -> Self {
        Self {
            cursor,
            chain,
            context: context.into(),
            expect: ReceiveKind::Round,
            pending: vec![],
            last_words: vec![],
        }
    }
}

impl PscChannel for KeyChannel<'_, '_> {
    fn recv(&mut self) -> Result<E, FpError> {
        if self.pending.is_empty() {
            let bytes = match self.expect {
                ReceiveKind::Round => {
                    let first = self.cursor.take("round polynomial", self.context.as_str());
                    match first {
                        Ok(bytes) => bytes,
                        Err(_) => self
                            .cursor
                            .take("round polynomial in Y", self.context.as_str())?,
                    }
                }
                ReceiveKind::End => self
                    .cursor
                    .take("values at the last point", self.context.as_str())?,
            };
            self.last_words = words(bytes);
            self.pending = elements(bytes);
        }
        Ok(self.pending.remove(0))
    }

    fn recv_hiding_sum(&mut self) -> Result<E, FpError> {
        let bytes = self.cursor.take("hiding sum", self.context.as_str())?;
        self.last_words = words(bytes);
        self.expect = ReceiveKind::Round;
        Ok(elements(bytes)[0])
    }

    fn close_epsilon(&mut self) -> Result<E, FpError> {
        let message = core::mem::take(&mut self.last_words);
        self.chain.close(ChainKind::Epsilon, &message);
        self.expect = ReceiveKind::Round;
        Ok(E::from_words(self.chain.draw_e()))
    }

    fn close_algebraic(&mut self, nonzero: bool) -> Result<E, FpError> {
        let message = core::mem::take(&mut self.last_words);
        self.chain.close(ChainKind::Algebraic, &message);
        self.expect = if nonzero {
            ReceiveKind::End
        } else {
            ReceiveKind::Round
        };
        Ok(E::from_words(if nonzero {
            self.chain.draw_e_nonzero()
        } else {
            self.chain.draw_e()
        }))
    }

    fn draw(&mut self, nonzero: bool) -> Result<E, FpError> {
        Ok(E::from_words(if nonzero {
            self.chain.draw_e_nonzero()
        } else {
            self.chain.draw_e()
        }))
    }

    fn close_empty(&mut self) -> Result<(), FpError> {
        let message = core::mem::take(&mut self.last_words);
        self.chain.close(ChainKind::Empty, &message);
        self.expect = ReceiveKind::Round;
        Ok(())
    }

    fn set_context(&mut self, context: &str) {
        self.context = context.into();
    }
}

fn opened_for_rows(
    rows: &crate::fp::proximity::SingleRows,
    statement: &Statement,
    opening: &OpeningData,
    bytes: &[u8],
    base_star: bool,
) -> crate::fp::proximity::Opened {
    let ids = rows.oracles();
    let total_main: usize = ids
        .iter()
        .map(|&oracle| {
            let shape = &statement.oracles[oracle as usize].shape;
            shape.main_rows as usize * shape.main_deg as usize
        })
        .sum();
    let mut out = crate::fp::proximity::Opened::default();
    for &position in &opening.positions {
        let mut main_at = 0;
        let mut mask_at = total_main + if base_star { 3 } else { 0 };
        for (i, &oracle) in ids.iter().enumerate() {
            let shape = &statement.oracles[oracle as usize].shape;
            let main = if shape.main_deg == 1 {
                opening
                    .column_f(bytes, position, main_at, shape.main_rows as usize)
                    .into_iter()
                    .map(E::from_base)
                    .collect()
            } else {
                opening.column_e(bytes, position, main_at, shape.main_rows as usize)
            };
            let mask = opening.column_e(bytes, position, mask_at, shape.mask_rows as usize);
            main_at += shape.main_rows as usize * shape.main_deg as usize;
            mask_at += shape.mask_rows as usize * 3;
            let column = crate::fp::proximity::OpenedColumn { main, mask };
            if out.of.len() <= i {
                out.of.push(vec![]);
            }
            out.of[i].push(column);
        }
        if base_star {
            let column = crate::fp::proximity::OpenedColumn {
                main: opening.column_e(bytes, position, main_at, 1),
                mask: opening.column_e(bytes, position, mask_at, 1),
            };
            if out.of.len() == ids.len() {
                out.of.push(vec![]);
            }
            out.of[ids.len()].push(column);
        }
    }
    out
}

fn sample_seed(chain: &mut Chain, bits: usize) -> (Vec<E>, E) {
    let z = (0..bits).map(|_| E::from_words(chain.draw_e())).collect();
    let beta = E::from_words(chain.draw_e_nonzero());
    (z, beta)
}

fn read_sample(
    cursor: &mut Cursor,
    chain: &mut Chain,
    statement: &mut Statement,
    oracle: u32,
    context: &str,
) -> Result<E, FpError> {
    let root = cursor.take_root(context)?;
    chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
    let bits = crate::fp::claims::sample_cube(statement, oracle).4 as usize;
    let (z, beta) = sample_seed(chain, bits);
    let answer_bytes = cursor.take("sample answers", context)?;
    let answers = elements(answer_bytes);
    chain.close(ChainKind::Empty, &words(answer_bytes));
    let mut sample = send_form(statement, oracle, &z, beta, 0);
    sample.target = answers[0];
    statement.oracles[oracle as usize].sample_rows = vec![sample];
    statement.admit_sample(oracle);
    Ok(answers[0])
}

pub fn verify_key_transcript(
    input: &KeyLayoutInput,
    configuration: &Digest,
    bytes: &[u8],
) -> Result<(), FpError> {
    let plan = crate::fp::key::plan_key(input)?;
    let layout = plan.layout()?;
    let (root, record, _) = parse_key_record(bytes, input.table_answers as usize)?;
    let table = IndexName::new(&root, configuration, record);
    let fixed = crate::fp::key_fixture::accepted_fixed(*configuration);
    let public = KeyPublic::new(
        &fixed,
        &table,
        include_bytes!("../../spikes/plans/key-one-time-plan.bin"),
    );
    let mut cursor = Cursor::new_with_anchor(bytes, &layout, configuration, "table", &root);
    cursor.take("head", "")?;
    cursor.take_root("table")?;
    cursor.take("record", "table")?;
    let salt = Digest::from_le_bytes(
        cursor
            .take("chain salt", "one-time chain")?
            .try_into()
            .map_err(|_| FpError::BadLength {
                context: "key-salt",
            })?,
    );
    let mut chain = Chain::new(public.first_query(&salt, LABEL_ONE_TIME_CHAIN));
    let run = public.first_query(&salt, LABEL_ONE_TIME_CHAIN).run.unwrap();
    let mut statement = Statement::new(fixed.codes.base.clone(), fixed.codes.mask.clone());
    let index_masks = statement.sent(fixed.index_masks.shape.clone());
    read_sample(
        &mut cursor,
        &mut chain,
        &mut statement,
        index_masks,
        "masks of the committer's rows on the table",
    )?;
    let message = statement.sent(fixed.ranges.message.shape.clone());
    read_sample(
        &mut cursor,
        &mut chain,
        &mut statement,
        message,
        "ranges: multiplicities",
    )?;
    let _fold_mask = statement.sent(crate::fp::claims::OracleShape {
        code: 0,
        mask_code: fixed.ranges.stages[0].mask_code,
        main_rows: 0,
        mask_rows: 1u64 << fixed.ranges.stages[0].h,
        main_deg: 3,
        answers: 0,
    });
    let wf = wf_rows_verify(
        &mut statement,
        &mut KeyChannel::new(&mut cursor, &mut chain, "ranges: bytes, tie"),
        &fixed.ranges,
        &fixed.lin,
        index_masks,
    )?;
    let (layout, rows) = statement.begin_run(wf.group);
    let mut row = bat_combine(&statement, &layout, &rows, {
        chain.close(ChainKind::Algebraic, &[]);
        E::from_words(chain.draw_e())
    });
    let mut iopp = Iopp::with_statement(statement);
    iopp.group = wf.group;
    iopp.base_ids = fixed
        .ranges
        .stages
        .iter()
        .map(|stage| stage.code as u64)
        .collect();
    iopp.mask_ids = fixed
        .ranges
        .stages
        .iter()
        .map(|stage| stage.mask_code as u64)
        .collect();
    let stages: Vec<crate::fp::analysis::Stage> = fixed
        .ranges
        .stages
        .iter()
        .enumerate()
        .map(|(index, stage)| {
            let mut out = crate::fp::iopp::execution_stage(&fixed.codes, stage);
            out.t = input.wf.stages[index].t;
            out.r_ood = input.wf.stages[index].r_ood;
            out
        })
        .collect();
    for (index, stage) in stages.iter().enumerate() {
        let context = if index + 1 == stages.len() {
            "ranges, last stage".to_string()
        } else {
            format!("ranges, stage {}", index + 1)
        };
        if index + 1 < stages.len() {
            let hiding = cursor.take("hiding sum", &context)?;
            let hiding_sum = elements(hiding)[0];
            chain.close(ChainKind::Epsilon, &words(hiding));
            let epsilon = E::from_words(chain.draw_e());
            let mut polynomials = Vec::new();
            let mut challenges = Vec::new();
            for _ in 0..stage.g.k {
                let polynomial = cursor.take("round polynomial", &context)?;
                let coefficients = elements(polynomial);
                polynomials.push((coefficients[0], coefficients[1]));
                chain.close(ChainKind::Algebraic, &words(polynomial));
                challenges.push(E::from_words(chain.draw_e()));
            }
            let root = cursor.take_root(&context)?;
            chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
            let cube = crate::fp::analysis::stage_cube_bits(&stages[index + 1].g) as usize;
            let z: Vec<E> = (0..cube).map(|_| E::from_words(chain.draw_e())).collect();
            let beta = E::from_words(chain.draw_e_nonzero());
            let answers = cursor.take("sample answers", &context)?;
            let answer = elements(answers)[0];
            let nonce = cursor.take_nonce(&context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(answers), 32, nonce) {
                return Err(FpError::Range {
                    context: "proof-of-work-cs",
                });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            let theta = E::from_words(chain.draw_e());
            cursor.open_stage(&positions, &run)?;
            let opened = opened_for_rows(
                &crate::fp::proximity::SingleRows {
                    code: iopp.base_ids[index] as u32,
                    mask_code: iopp.mask_ids[index] as u32,
                    layout: iopp.statement.layout(iopp.group),
                    gamma: challenges.clone(),
                    masks: 2,
                    fresh: iopp.statement.oracles.len() as u32 - 1,
                    c: Default::default(),
                    kept: vec![],
                    kept_rows: 0,
                    d: vec![],
                    target: E::ZERO,
                },
                &iopp.statement,
                &cursor.opened.last().unwrap(),
                bytes,
                false,
            );
            let target = crate::fp::proximity::switch_target_stage(
                iopp.base_ids[index + 1] as u32,
                iopp.mask_ids[index + 1] as u32,
                stages[index + 1].g.k,
                stages[index + 1].g.h,
                1,
            );
            row = iopp.fold_cs(
                &stages,
                1,
                index,
                row,
                hiding_sum,
                epsilon,
                &polynomials,
                &challenges,
                &z,
                beta,
                answer,
                &positions,
                theta,
                &opened,
                &[],
                target,
                true,
            )?;
        } else {
            let target = cursor.take("masked target", &context)?;
            let target_star = elements(target)[0];
            chain.close(ChainKind::Algebraic, &words(target));
            let lambda = E::from_words(chain.draw_e());
            let clear = cursor.take("masked message and randomness", &context)?;
            let clear_values = elements(clear);
            let nonce = cursor.take_nonce(&context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(clear), 32, nonce) {
                return Err(FpError::Range {
                    context: "proof-of-work-base",
                });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            cursor.open_stage(&positions, &run)?;
            let opened = opened_for_rows(
                &crate::fp::proximity::SingleRows {
                    code: iopp.base_ids[index] as u32,
                    mask_code: iopp.mask_ids[index] as u32,
                    layout: iopp.statement.layout(iopp.group),
                    gamma: vec![],
                    masks: 1,
                    fresh: u32::MAX,
                    c: Default::default(),
                    kept: vec![],
                    kept_rows: 0,
                    d: vec![],
                    target: E::ZERO,
                },
                &iopp.statement,
                &cursor.opened.last().unwrap(),
                bytes,
                true,
            );
            iopp.base(
                &stages,
                row,
                target_star,
                lambda,
                &clear_values,
                &opened,
                &cursor.opened.last().unwrap().positions,
            )?;
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{heads::configuration_digest, key::accepted_input};

    fn fixture() -> (&'static [u8], crate::fp::digest::Digest) {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let key = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/key.bin");
        (key, configuration_digest(&[policy, layout, kl]))
    }

    #[test]
    fn accepted_key_verifies() {
        let (key, configuration) = fixture();
        verify_key_transcript(&accepted_input(), &configuration, key).unwrap();
    }

    #[test]
    fn mutated_key_is_rejected() {
        let (key, configuration) = fixture();
        for &offset in &[0, key.len() / 4, key.len() / 3] {
            let mut damaged = key.to_vec();
            damaged[offset] ^= 1;
            assert!(
                verify_key_transcript(&accepted_input(), &configuration, &damaged).is_err(),
                "mutation at byte {offset} was accepted"
            );
        }
    }
}
