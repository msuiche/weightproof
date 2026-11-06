//! Executable key transcript and WF verification.

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{bat_combine, Statement},
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    heads::parse_key_record,
    iopp::Iopp,
    key::KeyLayoutInput,
    key_public::{KeyPublic, LABEL_ONE_TIME_CHAIN},
    link_public::IndexName,
    lin::{member_first_verify, read_binding_root},
    psc::PscChannel,
    send::read_samples_after_root,
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

pub(crate) struct KeyChannel<'a, 'b> {
    cursor: &'a mut Cursor<'b>,
    chain: &'a mut Chain,
    context: String,
    expect: ReceiveKind,
    pending: Vec<E>,
    last_words: Vec<u64>,
}

impl<'a, 'b> KeyChannel<'a, 'b> {
    pub(crate) fn new(cursor: &'a mut Cursor<'b>, chain: &'a mut Chain, context: &'a str) -> Self {
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

fn opened_for_shared_rows(
    rows: &crate::fp::proximity::SingleRows,
    statement: &Statement,
    openings: &[OpeningData],
    bytes: &[u8],
    raw_positions: &[u64],
    base_star: bool,
    split: u32,
) -> crate::fp::proximity::Opened {
    let ids = rows.oracles();
    let primary = ids
        .iter()
        .copied()
        .find(|&oracle| oracle != split)
        .unwrap();
    let fresh = if rows.masks == 2 {
        Some(*ids.last().unwrap())
    } else {
        None
    };
    let main_width = |oracle: u32| {
        let shape = &statement.oracles[oracle as usize].shape;
        shape.main_rows as usize * shape.main_deg as usize
    };
    let mask_width = |oracle: u32| statement.oracles[oracle as usize].shape.mask_rows as usize * 3;
    let fresh_main = fresh.map_or(0, &main_width);
    let fresh_mask = fresh.map_or(0, &mask_width);
    let star_main = if base_star { 3 } else { 0 };
    let primary_main = main_width(primary);
    let primary_mask = mask_width(primary);
    let total_main = primary_main + fresh_main + star_main;
    let primary_only_bytes = 8 * (total_main + primary_mask + if base_star { 3 } else { 0 });
    let with_fresh_bytes = primary_only_bytes + 8 * fresh_mask;
    let (primary_opening, fresh_opening) = if let Some(opening) = openings
        .iter()
        .find(|opening| opening.leaf_bytes == with_fresh_bytes)
    {
        (opening, opening)
    } else {
        let primary = openings
            .iter()
            .find(|opening| opening.leaf_bytes == primary_only_bytes)
            .unwrap();
        let fresh = fresh.and_then(|_| {
            openings
                .iter()
                .rfind(|opening| opening.leaf_bytes == 8 * fresh_mask)
        });
        (primary, fresh.unwrap_or(primary))
    };
    let split_opening = if ids.contains(&split) {
        openings
            .iter()
            .find(|opening| opening.tree_number + 1 == split)
    } else {
        None
    };
    let mut out = crate::fp::proximity::Opened::default();
    let mut sorted_positions = raw_positions.to_vec();
    sorted_positions.sort_unstable();
    sorted_positions.dedup();
    for (i, &oracle) in ids.iter().enumerate() {
        let shape = &statement.oracles[oracle as usize].shape;
        let (opening, main_at, mask_at) = if oracle == split {
            (split_opening.unwrap(), 0, 0)
        } else if Some(oracle) == fresh {
            if core::ptr::eq(fresh_opening, primary_opening) {
                (fresh_opening, primary_main, total_main + primary_mask)
            } else {
                (fresh_opening, 0, 0)
            }
        } else {
            (primary_opening, 0, total_main)
        };
        for &raw_position in &sorted_positions {
            let position = raw_position & ((1u64 << opening.log_n) - 1);
            let main = if Some(oracle) == fresh {
                vec![]
            } else if shape.main_deg == 1 {
                opening
                    .column_f(bytes, position, main_at, shape.main_rows as usize)
                    .into_iter()
                    .map(E::from_base)
                    .collect()
            } else {
                opening.column_e(bytes, position, main_at, shape.main_rows as usize)
            };
            let mask = opening.column_e(bytes, position, mask_at, shape.mask_rows as usize);
            let column = crate::fp::proximity::OpenedColumn { main, mask };
            if out.of.len() <= i {
                out.of.push(vec![]);
            }
            out.of[i].push(column);
        }
    }
    if base_star {
        for &raw_position in &sorted_positions {
            let position = raw_position & ((1u64 << primary_opening.log_n) - 1);
            let column = crate::fp::proximity::OpenedColumn {
                main: primary_opening.column_e(bytes, position, primary_main + fresh_main, 1),
                mask: primary_opening.column_e(
                    bytes,
                    position,
                    total_main + primary_mask + fresh_mask,
                    1,
                ),
            };
            if out.of.len() == ids.len() {
                out.of.push(vec![]);
            }
            out.of[ids.len()].push(column);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_iopp_stages(
    cursor: &mut Cursor,
    chain: &mut Chain,
    iopp: &mut Iopp,
    stages: &[crate::fp::analysis::Stage],
    start: usize,
    end: usize,
    mut row: crate::fp::claims::GroupRow,
    run: &Digest,
    context: &str,
    bytes: &[u8],
    work_bits: u32,
    shared_split: Option<u32>,
    initial_fresh: Option<u32>,
    mask_ahead: bool,
) -> Result<(), FpError> {
    for index in start..end {
        let stage = &stages[index];
        let stage_context = if index + 1 == end {
            format!("{context}, last stage")
        } else {
            format!("{context}, stage {}", index - start + 1)
        };
        if index + 1 < end {
            let mut epsilon_message = Vec::new();
            let fresh_override = if !mask_ahead && index == start {
                let root = cursor.take_root(&stage_context)?;
                epsilon_message = words(&root.to_le_bytes());
                let fresh = iopp.statement.sent(crate::fp::claims::OracleShape {
                    code: 0,
                    mask_code: iopp.mask_ids[index] as u32,
                    main_rows: 0,
                    mask_rows: 1u64 << stages[index].g.h,
                    main_deg: 3,
                    answers: 0,
                });
                Some(fresh)
            } else if index == start {
                initial_fresh
            } else {
                None
            };
            let hiding = cursor.take("hiding sum", &stage_context)?;
            let hiding_sum = elements(hiding)[0];
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_OMEGA_FOLD root={:?} sum={:016x}", fresh_override, hiding_sum.0[0]);
            }
            epsilon_message.extend(words(hiding));
            chain.close(ChainKind::Epsilon, &epsilon_message);
            let epsilon = E::from_words(chain.draw_e());
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_FOLD_EPSILON {:016x}", epsilon.0[0]);
            }
            let mut polynomials = Vec::new();
            let mut challenges = Vec::new();
            for _ in 0..stage.g.k {
                let polynomial = cursor.take("round polynomial", &stage_context)?;
                let coefficients = elements(polynomial);
                polynomials.push((coefficients[0], coefficients[1]));
                #[cfg(feature = "std")]
                if std::env::var_os("DUMP_SHARED").is_some() {
                    eprintln!("RUST_FOLD_COEFF c0={:016x} c2={:016x}", coefficients[0].0[0], coefficients[1].0[0]);
                }
                chain.close(ChainKind::Algebraic, &words(polynomial));
                challenges.push(E::from_words(chain.draw_e()));
                #[cfg(feature = "std")]
                if std::env::var_os("DUMP_SHARED").is_some() {
                    eprintln!("RUST_FOLD_GAMMA {:016x}", challenges.last().unwrap().0[0]);
                }
            }
            let root = cursor.take_root(&stage_context)?;
            chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
            let cube = crate::fp::analysis::stage_cube_bits(&stages[index + 1].g) as usize;
            let z: Vec<E> = (0..cube).map(|_| E::from_words(chain.draw_e())).collect();
            let beta = E::from_words(chain.draw_e_nonzero());
            let answers = cursor.take("sample answers", &stage_context)?;
            let answer = elements(answers)[0];
            let nonce = cursor.take_nonce(&stage_context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(answers), work_bits, nonce) {
                return Err(FpError::Range { context: "proof-of-work-cs" });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_STAGE_POSITIONS index={} count={} first={} second={} last={}", index, positions.len(), positions[0], positions[1], positions[positions.len() - 1]);
            }
            let theta = E::from_words(chain.draw_e());
            let opening_start = cursor.opened.len();
            cursor.open_stage(&positions, run)?;
            let opened = if let Some(split) = shared_split {
                opened_for_shared_rows(
                    &crate::fp::proximity::SingleRows {
                        code: iopp.base_ids[index] as u32,
                        mask_code: iopp.mask_ids[index] as u32,
                        layout: iopp.statement.layout(iopp.group),
                        gamma: challenges.clone(),
                        masks: 2,
                        fresh: fresh_override
                            .or(if index == start {
                                initial_fresh
                            } else {
                                Some(iopp.statement.oracles.len() as u32 - 1)
                            })
                            .unwrap(),
                        c: Default::default(),
                        kept: vec![],
                        kept_rows: 0,
                        d: vec![],
                        target: E::ZERO,
                    },
                    &iopp.statement,
                    &cursor.opened[opening_start..],
                    bytes,
                    &positions,
                    false,
                    split,
                )
            } else {
                opened_for_rows(
                    &crate::fp::proximity::SingleRows {
                        code: iopp.base_ids[index] as u32,
                        mask_code: iopp.mask_ids[index] as u32,
                        layout: iopp.statement.layout(iopp.group),
                        gamma: challenges.clone(),
                        masks: 2,
                        fresh: fresh_override
                            .or(if index == start {
                                initial_fresh
                            } else {
                                Some(iopp.statement.oracles.len() as u32 - 1)
                            })
                            .unwrap(),
                        c: Default::default(),
                        kept: vec![],
                        kept_rows: 0,
                        d: vec![],
                        target: E::ZERO,
                    },
                    &iopp.statement,
                    cursor.opened.last().unwrap(),
                    bytes,
                    false,
                )
            };
            let target = crate::fp::proximity::switch_target_stage(
                iopp.base_ids[index + 1] as u32,
                iopp.mask_ids[index + 1] as u32,
                stages[index + 1].g.k,
                stages[index + 1].g.h,
                1,
            );
            iopp.fresh_override = fresh_override;
            row = iopp.fold_cs(
                stages,
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
            let target = cursor.take("masked target", &stage_context)?;
            let target_star = elements(target)[0];
            chain.close(ChainKind::Algebraic, &words(target));
            let lambda = E::from_words(chain.draw_e());
            let clear = cursor.take("masked message and randomness", &stage_context)?;
            let clear_values = elements(clear);
            let nonce = cursor.take_nonce(&stage_context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(clear), work_bits, nonce) {
                return Err(FpError::Range {
                    context: "proof-of-work-base",
                });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            let opening_start = cursor.opened.len();
            cursor.open_stage(&positions, run)?;
            let opened = if let Some(split) = shared_split {
                opened_for_shared_rows(
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
                    &cursor.opened[opening_start..],
                    bytes,
                    &positions,
                    true,
                    split,
                )
            } else {
                opened_for_rows(
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
                    cursor.opened.last().unwrap(),
                    bytes,
                    true,
                )
            };
            iopp.base(
                stages,
                end - 1,
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

fn read_sample_after_root(
    cursor: &mut Cursor,
    chain: &mut Chain,
    statement: &mut Statement,
    oracle: u32,
    context: &str,
) -> Result<E, FpError> {
    Ok(read_samples_after_root(cursor, chain, statement, &[oracle], context)?[0])
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
    read_sample_after_root(cursor, chain, statement, oracle, context)
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
    #[cfg(feature = "std")]
    eprintln!("RUST_ONE_RUN run={} salt={} params={}", run.to_hex(), salt.to_hex(), public.parameters.to_hex());
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
    let mut wf = wf_rows_verify(
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
                stages.len() - 1,
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
    let w = iopp.statement.sent(fixed.lin.w.clone());
    let fold_mask = if fixed.lin.masks_ahead {
        Some(iopp.statement.sent(crate::fp::claims::OracleShape {
            code: 0,
            mask_code: fixed.lin.test_own[0].mask_code,
            main_rows: 0,
            mask_rows: 1u64 << fixed.lin.test_own[0].h,
            main_deg: 3,
            answers: 0,
        }))
    } else {
        None
    };
    let _first = member_first_verify(
        &mut iopp.statement,
        &mut KeyChannel::new(&mut cursor, &mut chain, "ranges: rows on the table"),
        &fixed.lin,
        index_masks,
        &mut wf.pending,
        w,
        fold_mask,
    )?;
    read_sample(&mut cursor, &mut chain, &mut iopp.statement, w, "table, W")?;
    chain.close(ChainKind::Seal, &[]);
    let mut seal_words = [0u32; 8];
    for i in 0..4 {
        let word = chain.next_word();
        seal_words[2 * i] = word as u32;
        seal_words[2 * i + 1] = (word >> 32) as u32;
    }
    let seal = Digest(seal_words);
    let _binding = read_binding_root(&mut cursor, &fixed.binding, configuration, &seal)?;
    let shared_public = crate::fp::key_public::LinPublic::shared(
        &fixed,
        &table,
        include_bytes!("../../spikes/plans/key-shared-plan.bin"),
        &_binding,
    );
    let shared_query = shared_public.first_query(crate::fp::key_public::LIN_LABEL_SHARED);
    #[cfg(feature = "std")]
    eprintln!("SHARED_STATEMENT_MATCH {}", shared_public.statement == include_bytes!("../../spikes/plans/key-shared-statement.bin"));
    let shared_run = shared_query.run.unwrap();
    #[cfg(feature = "std")]
    eprintln!("RUST_SHARED_RUN run={} salt={} params={}", shared_run.to_hex(), _binding.digest.to_hex(), shared_public.parameters.to_hex());
    let mut shared_chain = Chain::new(shared_query);
    let mut shared = crate::fp::lin_algebra::LinSharedAlgebra::new(
        &fixed.lin,
        &fixed.codes,
        configuration,
    );
    shared_chain.close(ChainKind::Algebraic, &[]);
    let psi: Vec<E> = (0..fixed.lin.k_col())
        .map(|_| E::from_words(shared_chain.draw_e()))
        .collect();
    let _u_answer = read_sample(
        &mut cursor,
        &mut shared_chain,
        &mut shared.iopp.statement,
        shared.shared.u.oracle,
        "table, folded row",
    )?;
    let columns_nonce = cursor.take_nonce("table, columns")?;
    if !shared_chain.close_with_work(ChainKind::Positions, &[], 32, columns_nonce) {
        return Err(FpError::Range {
            context: "proof-of-work-lin-columns",
        });
    }
    let columns: Vec<u64> = (0..input.lin.columns)
        .map(|_| shared_chain.draw_position(fixed.lin.fold.index.code.log_n))
        .collect();
    let first_shared_opening = cursor.opened.len();
    #[cfg(feature = "std")]
    eprintln!("RUST_POSITIONS count={} first={} second={} last={}", columns.len(), columns[0], columns[1], columns[columns.len() - 1]);
    cursor.open_stage(&columns, &shared_run)?;
    shared.initial(
        &table,
        &psi[fixed.lin.k_lo as usize..],
        &columns,
        &cursor.opened[first_shared_opening],
        bytes,
    )?;
    let omega_root = cursor.take_root("table, Omega")?;
    let mut omega_message = words(&omega_root.to_le_bytes());
    if shared.shared.split.is_some() {
        let split_root = cursor.take_root("table, Omega")?;
        omega_message.extend(words(&split_root.to_le_bytes()));
    }
    shared_chain.close(ChainKind::Seed, &omega_message);
    let mut sampled = vec![shared.shared.omega];
    if let Some(split) = shared.shared.split {
        sampled.push(split);
    }
    let _omega_answers = read_samples_after_root(
        &mut cursor,
        &mut shared_chain,
        &mut shared.iopp.statement,
        &sampled,
        "table, Omega",
    )?;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let zeta: Vec<E> = (0..fixed.lin.k_row())
        .map(|_| E::from_words(shared_chain.draw_e()))
        .collect();
    let omega_lay = crate::fp::claims::VectorLay {
        row_bits: crate::fp::tensor::ceil_log2(fixed.lin.omega.main_rows),
        place_bits: crate::fp::code_eval::log_ell(&shared.iopp.statement.base[fixed.lin.omega.code as usize]),
        rows_high: fixed.lin.omega_rows_high,
    };
    let fold_row = crate::fp::lin::lin_fold_row(
        &fixed.lin,
        &psi,
        &zeta,
        shared.shared.u.clone(),
        shared.shared.omega,
        omega_lay,
    );
    if let Some(split) = shared.shared.split {
        crate::fp::split::split_verify(
            &mut shared.iopp.statement,
            &mut cursor,
            &mut shared_chain,
            "Omega against the folded row",
            &fold_row,
            shared.shared.group_omega,
            shared.shared.group_u,
            split,
        )?;
    } else {
        shared.iopp.statement.add_row(shared.shared.group_u, fold_row);
    }
    shared.iopp.group = shared.shared.group_u;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let row = shared.combine_u(E::from_words(shared_chain.draw_e()));
    let shared_stages: Vec<crate::fp::analysis::Stage> = input
        .lin
        .folded_stages
        .iter()
        .chain(&input.lin.omega_stages)
        .cloned()
        .collect();
    run_iopp_stages(
        &mut cursor,
        &mut shared_chain,
        &mut shared.iopp,
        &shared_stages,
        0,
        input.lin.folded_stages.len(),
        row,
        &shared_run,
        "folded row",
        bytes,
        32,
        shared.shared.split,
        shared.shared.mask,
        true,
    )?;
    shared.iopp.group = shared.shared.group_omega;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let row = shared.combine_omega(E::from_words(shared_chain.draw_e()));
    run_iopp_stages(
        &mut cursor,
        &mut shared_chain,
        &mut shared.iopp,
        &shared_stages,
        input.lin.folded_stages.len(),
        shared_stages.len(),
        row,
        &shared_run,
        "Omega",
        bytes,
        32,
        shared.shared.split,
        shared.shared.mask,
        false,
    )?;
    shared_chain.close(ChainKind::Seal, &[]);
    let mut shared_seal_words = [0u32; 8];
    for i in 0..4 {
        let word = shared_chain.next_word();
        shared_seal_words[2 * i] = word as u32;
        shared_seal_words[2 * i + 1] = (word >> 32) as u32;
    }
    let _shared_seal = Digest(shared_seal_words);
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
