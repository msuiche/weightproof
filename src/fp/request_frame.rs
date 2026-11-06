//! Request-link frame: SEND, CS^sigma, BAT, FOLD, and the final IOPP.

use alloc::{vec, vec::Vec};

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{bat_combine, GroupId, GroupRow, OracleShape, Part},
    error::FpError,
    field::ExtensionElement as E,
    iopp::{execution_stage, Iopp},
    key_public::{IoppStage, OracleNotice},
    proof_plan::ProofLayoutInput,
    proximity::{fold_output, fold_round_claim, Cut, KeptTarget, SingleRows, SwitchTarget},
    req_rows::{req_row, ReqEquation},
    request_public::{ReqMessage, ReqShape},
    send::OracleBook,
    transcript::{elements, one_element, words, Cursor},
};

#[derive(Debug, Clone)]
struct ReqGroup {
    k: u32,
    h: u32,
    target: SwitchTarget,
    born: Vec<(usize, usize)>,
}

fn block(rows: u64) -> u64 {
    if rows == 0 {
        0
    } else {
        1u64 << crate::fp::tensor::ceil_log2(rows)
    }
}

fn req_groups(shape: &ReqShape) -> Vec<ReqGroup> {
    #[derive(Clone)]
    struct Alive {
        shape: OracleShape,
        left: u32,
        born: (usize, usize),
    }
    let mut alive = Vec::<Alive>::new();
    let mut groups = Vec::with_capacity(shape.links.len() + 1);
    for l in 0..=shape.links.len() {
        let mut main = 0;
        let mut mask = 0;
        let mut target = SwitchTarget {
            shape: OracleShape {
                code: shape.code,
                mask_code: shape.mask_code,
                main_rows: 2,
                mask_rows: 0,
                main_deg: 3,
                answers: shape.answers,
            },
            main: Cut {
                part: Part::Main,
                bits: 0,
                first_row: 0,
            },
            rest: Cut {
                part: Part::Main,
                bits: 0,
                first_row: 1,
            },
            kept: vec![],
        };
        let mut born = vec![];
        if l > 0 {
            main += 2;
        }
        let mut still = Vec::new();
        for message in alive {
            let kept_again = message.left > 1;
            target.kept.push(KeptTarget {
                shape: message.shape.clone(),
                kept_again,
            });
            born.push(message.born);
            // Kept oracles' mask rows are folded as well as their main rows:
            // Statement::layout counts them in the group's mask height.
            mask += block(message.shape.mask_rows);
            if kept_again {
                still.push(Alive {
                    left: message.left - 1,
                    ..message
                });
            } else {
                main += block(message.shape.main_rows);
            }
        }
        alive = still;
        if l < shape.links.len() {
            for (i, message) in shape.links[l].messages.iter().enumerate() {
                if message.life > 0 {
                    alive.push(Alive {
                        shape: message.shape.clone(),
                        left: message.life,
                        born: (l, i),
                    });
                } else {
                    main += block(message.shape.main_rows);
                }
                mask += block(message.shape.mask_rows);
            }
        }
        let h = crate::fp::tensor::ceil_log2(mask);
        let k = crate::fp::tensor::ceil_log2(main).max(h).max(1);
        groups.push(ReqGroup {
            k,
            h,
            target,
            born,
        });
    }
    groups
}

#[derive(Debug)]
struct FoldState {
    row: GroupRow,
    hiding: E,
    epsilon: E,
    polynomials: Vec<(E, E)>,
    challenges: Vec<E>,
    folded: SingleRows,
    mask: u32,
    group: GroupId,
}

#[derive(Debug)]
pub(crate) struct First {
    pub own: Vec<u32>,
    pub plain: Vec<E>,
    pub at_start: Vec<u32>,
}

pub(crate) struct RequestFrame<'a, 'b> {
    pub cursor: &'a mut Cursor<'b>,
    pub chain: &'a mut Chain,
    pub fixed: crate::fp::request_public::RequestFixed,
    pub input: ProofLayoutInput,
    pub run: crate::fp::digest::Digest,
    pub iopp: Iopp,
    pub book: OracleBook,
    groups: Vec<ReqGroup>,
    link: usize,
    moment: u32,
    group: GroupId,
    own: Vec<u32>,
    kept_before: Vec<u32>,
    at_start: Vec<u32>,
    plain: Vec<E>,
    mask: Option<u32>,
    fold: Option<FoldState>,
}

impl<'a, 'b> RequestFrame<'a, 'b> {
    pub fn new(
        cursor: &'a mut Cursor<'b>,
        chain: &'a mut Chain,
        fixed: crate::fp::request_public::RequestFixed,
        input: ProofLayoutInput,
        run: crate::fp::digest::Digest,
    ) -> Self {
        let groups = req_groups(&fixed.shape);
        let mut iopp = Iopp::new(fixed.codes.base.clone(), fixed.codes.mask.clone());
        iopp.base_ids = fixed.shape.last.iter().map(|stage| stage.code as u64).collect();
        iopp.mask_ids = fixed.shape.last.iter().map(|stage| stage.mask_code as u64).collect();
        Self {
            cursor,
            chain,
            fixed,
            input,
            run,
            iopp,
            book: OracleBook::default(),
            groups,
            link: 0,
            moment: 0,
            group: 0,
            own: vec![],
            kept_before: vec![],
            at_start: vec![],
            plain: vec![],
            mask: None,
            fold: None,
        }
    }

    fn begin_link(&mut self) {
        self.group = self
            .iopp
            .statement
            .new_group(self.fixed.shape.code, self.fixed.shape.mask_code);
        self.iopp.group = self.group;
        self.moment = 1;
        self.own.clear();
    }

    fn at_moment(&self, moment: u32) -> Vec<ReqMessage> {
        self.fixed.shape.links[self.link]
            .messages
            .iter()
            .filter(|message| message.moment == moment)
            .cloned()
            .collect()
    }

    fn notices(own: &[ReqMessage]) -> Vec<OracleNotice> {
        own.iter()
            .map(|message| OracleNotice {
                shape: message.shape.clone(),
                leaf_class: message.leaf_class,
            })
            .collect()
    }

    fn fold_notice(&self, link: usize) -> OracleNotice {
        let group = &self.groups[link];
        OracleNotice {
            shape: OracleShape {
                code: u32::MAX,
                mask_code: self.fixed.shape.mask_code,
                main_rows: 0,
                mask_rows: 1u64 << group.h,
                main_deg: 3,
                answers: 0,
            },
            leaf_class: 0,
        }
    }

    fn join(&mut self, own: &[ReqMessage], ids: &[u32]) {
        for (message, &oracle) in own.iter().zip(ids) {
            self.iopp
                .statement
                .join_kept(self.group, oracle, message.life > 0);
            self.own.push(oracle);
        }
    }

    fn position_count(&self, link: usize) -> u64 {
        if link < self.input.links.len() {
            let group = &self.input.links[link].fold;
            crate::fp::codes::positions(
                crate::fp::analysis::analysis_of_m(group).at.agree,
                self.input.setting.target - self.input.setting.query_work(crate::fp::analysis::Scope::Request),
                0.0,
                1,
            )
            .unwrap()
        } else {
            self.input.last_stages[0].t
        }
    }

    pub fn first(&mut self) -> Result<First, FpError> {
        self.begin_link();
        let own = self.at_moment(0);
        self.plain = if self.fixed.shape.links[self.link].plain == 0 {
            vec![]
        } else {
            let bytes = self.cursor.take("plain scalars", &self.fixed.shape.links[self.link].name)?;
            let values = elements(bytes);
            self.chain.close(ChainKind::Empty, &words(bytes));
            values
        };
        if self.link == 0 {
            let mut notices = Self::notices(&own);
            notices.push(self.fold_notice(0));
            notices.extend(self.fixed.shape.at_start.clone());
            let sent = self.book.send(
                self.cursor,
                self.chain,
                &mut self.iopp.statement,
                &notices,
                &self.fixed.shape.links[0].name,
                true,
            )?;
            let own_count = own.len();
            self.mask = Some(sent.ids[own_count]);
            self.at_start = sent.ids[own_count + 1..].to_vec();
            let own_ids = sent.ids[..own_count].to_vec();
            self.join(&own, &own_ids);
            Ok(First {
                own: own_ids,
                plain: self.plain.clone(),
                at_start: self.at_start.clone(),
            })
        } else {
            let previous = self.fold.take().ok_or(FpError::Range {
                context: "request-fold-state",
            })?;
            let target = self.groups[self.link].target.clone();
            #[cfg(feature = "std")]
            if self.fixed.shape.links[self.link].name == "A0" {
                eprintln!("A0 target kept={} born={:?} own={}", target.kept.len(), self.groups[self.link].born, own.len());
            }
            let mut notices = Vec::with_capacity(1 + target.kept.len() + own.len() + 1);
            notices.push(OracleNotice {
                shape: target.shape.clone(),
                leaf_class: 0,
            });
            notices.extend(target.kept.iter().map(|kept| OracleNotice {
                shape: kept.shape.clone(),
                leaf_class: 0,
            }));
            notices.extend(Self::notices(&own));
            notices.push(self.fold_notice(self.link));
            let opening_start = self.cursor.opened.len();
            let sent = self.book.send(
                self.cursor,
                self.chain,
                &mut self.iopp.statement,
                &notices,
                &self.fixed.shape.links[self.link].name,
                false,
            )?;
            let nonce = self
                .cursor
                .take_nonce(&self.fixed.shape.links[self.link - 1].name)?;
            if !self.chain.close_with_work(
                ChainKind::Positions,
                &sent.answer_words,
                self.input.setting.query_work(crate::fp::analysis::Scope::Request),
                nonce,
            ) {
                return Err(FpError::Range {
                    context: "proof-of-work-loc",
                });
            }
            let positions: Vec<u64> = (0..self.position_count(self.link - 1))
                .map(|_| {
                    self.chain.draw_position(
                        self.iopp.statement.base[self.fixed.shape.code as usize].log_n,
                    )
                })
                .collect();
            self.cursor.open_stage(&positions, &self.run)?;
            let opened = self.book.opened(
                &previous.folded,
                &self.iopp.statement,
                &self.cursor.opened[opening_start..],
                self.cursor.bytes(),
                &positions,
                false,
            )?;
            let kept_count = target.kept.len();
            let oracle = sent.ids[0];
            let kept = sent.ids[1..1 + kept_count].to_vec();
            let own_ids = sent.ids[1 + kept_count..1 + kept_count + own.len()].to_vec();
            self.mask = Some(sent.ids[1 + kept_count + own.len()]);
            self.kept_before = kept.clone();
            // The group that this link's fold runs over: the previous fold's
            // output and the oracles kept from earlier links. The fold mask is
            // not a group member (fresh_override handles it).
            self.iopp.statement.join(self.group, oracle);
            for (&id, kept_target) in kept.iter().zip(target.kept.iter()) {
                self.iopp
                    .statement
                    .join_kept(self.group, id, kept_target.kept_again);
            }
            let current = execution_stage(
                &self.fixed.codes,
                &IoppStage {
                    code: self.fixed.shape.code,
                    mask_code: self.fixed.shape.mask_code,
                    k: self.groups[self.link - 1].k,
                    h: self.groups[self.link - 1].h,
                },
            );
            let next = execution_stage(
                &self.fixed.codes,
                &IoppStage {
                    code: target.shape.code,
                    mask_code: target.shape.mask_code,
                    k: self.groups[self.link].k,
                    h: self.groups[self.link].h,
                },
            );
            self.iopp.fresh_override = Some(previous.mask);
            // The fold is over the previous link's group, not the group that
            // begin_link() just opened for this link.
            self.iopp.group = previous.group;
            let folded = self.iopp.fold_cs_pre_sent(
                &[current, next],
                self.fixed.shape.answers,
                0,
                previous.row,
                previous.hiding,
                previous.epsilon,
                &previous.polynomials,
                &previous.challenges,
                &[],
                E::ONE,
                E::ZERO,
                &positions,
                E::ZERO,
                &opened,
                &kept,
                oracle,
                target,
            );
            self.iopp.group = self.group;
            folded?;
            self.join(&own, &own_ids);
            Ok(First {
                own: own_ids,
                plain: self.plain.clone(),
                at_start: self.at_start.clone(),
            })
        }
    }

    pub fn later(&mut self) -> Result<Vec<u32>, FpError> {
        let own = self.at_moment(self.moment);
        self.moment += 1;
        let sent = self.book.send(
            self.cursor,
            self.chain,
            &mut self.iopp.statement,
            &Self::notices(&own),
            &self.fixed.shape.links[self.link].name,
            true,
        )?;
        self.join(&own, &sent.ids);
        Ok(sent.ids)
    }

    pub fn where_oracle(&self, born: usize, message: usize) -> Result<u32, FpError> {
        if born == self.link {
            return self.own.get(message).copied().ok_or(FpError::Range {
                context: "request-oracle",
            });
        }
        self.groups[self.link]
            .born
            .iter()
            .position(|&candidate| candidate == (born, message))
            .and_then(|index| self.kept_before.get(index).copied())
            .ok_or(FpError::Range {
                context: "request-oracle",
            })
    }

    pub fn close(&mut self, equation: &ReqEquation) -> Result<(), FpError> {
        let link = self.link;
        let own = self.own.clone();
        let kept = self.kept_before.clone();
        let groups = &self.groups;
        let row = req_row(
            &mut self.iopp.statement,
            &self.fixed.shape,
            &self.fixed.layout,
            link,
            equation,
            |born, message| {
                if born == link {
                    return own.get(message).copied().ok_or(FpError::Range {
                        context: "request-oracle",
                    });
                }
                groups[link]
                    .born
                    .iter()
                    .position(|&candidate| candidate == (born, message))
                    .and_then(|index| kept.get(index).copied())
                    .ok_or(FpError::Range {
                        context: "request-oracle",
                    })
            },
        )?;
        self.iopp.statement.add_row(self.group, row);
        let (layout, rows) = self.iopp.statement.begin_run(self.group);
        if layout.k != self.groups[link].k || layout.h != self.groups[link].h {
            return Err(FpError::Range {
                context: "request-group-shape",
            });
        }
        self.chain.close(ChainKind::Algebraic, &[]);
        let theta = E::from_words(self.chain.draw_e());
        let row = bat_combine(&self.iopp.statement, &layout, &rows, theta);
        let context = &self.fixed.shape.links[link].name;
        let hiding_bytes = self.cursor.take("hiding sum", context)?;
        let hiding = one_element(hiding_bytes);
        self.chain.close(ChainKind::Epsilon, &words(hiding_bytes));
        let epsilon = E::from_words(self.chain.draw_e());
        let mut polynomials = Vec::with_capacity(self.groups[link].k as usize);
        let mut challenges = Vec::with_capacity(self.groups[link].k as usize);
        for _ in 0..self.groups[link].k {
            let bytes = self.cursor.take("round polynomial", context)?;
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_MSG").is_some() {
                eprintln!("RUST_FOLD link={} round={} words={:?}", link, polynomials.len(), crate::fp::transcript::words(bytes));
            }
            let coefficients = elements(bytes);
            polynomials.push((coefficients[0], coefficients[1]));
            self.chain.close(ChainKind::Algebraic, &words(bytes));
            challenges.push(E::from_words(self.chain.draw_e()));
        }
        let mut claim = hiding + epsilon * row.target;
        for (&(c0, c2), &challenge) in polynomials.iter().zip(&challenges) {
            claim = fold_round_claim(claim, c0, c2, challenge);
        }
        let folded = fold_output(
            &self.iopp.statement,
            self.fixed.shape.code,
            self.fixed.shape.mask_code,
            layout,
            row.clone(),
            self.mask.ok_or(FpError::Range {
                context: "request-fold-mask",
            })?,
            epsilon,
            challenges.clone(),
            claim,
        );
        self.fold = Some(FoldState {
            row,
            hiding,
            epsilon,
            polynomials,
            challenges,
            folded,
            mask: self.mask.unwrap(),
            group: self.group,
        });
        self.mask = None;
        self.link += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{
        heads::{configuration_digest, parse_leaf_record},
        link_public::IndexName,
        proof_plan::{plan_proof, tests::accepted_input},
        request_public::{accepted_request_fixed, RequestPublic},
    };

    #[test]
    fn first_send_consumes_oracle_fields_exactly() {
        let policy = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let leaf = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin");
        let proof = include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.proof");
        let configuration = configuration_digest(&[policy, layout, kl]);
        let (root, record, _) = parse_leaf_record(leaf, 256).unwrap();
        let table = IndexName::new(&root, &configuration, record);
        let public = RequestPublic::new(
            &accepted_request_fixed(configuration, layout),
            &table,
            0,
            include_bytes!("../../spikes/plans/request-statement.bin"),
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.req"),
            include_bytes!("../../spikes/plans/request-first-plan.bin"),
        );
        let input = accepted_input();
        let plan = plan_proof(&input).unwrap();
        let file_layout = plan.layout().unwrap();
        let mut cursor = Cursor::new(proof, &file_layout, &configuration, &table.commitment);
        cursor.take("head", "").unwrap();
        let salt = crate::fp::digest::Digest::from_le_bytes(
            cursor.take("chain salt", "first chain").unwrap().try_into().unwrap(),
        );
        let query = public.first_query(&salt);
        let run = query.run.unwrap();
        let mut chain = Chain::new(query);
        let mut frame = RequestFrame::new(
            &mut cursor,
            &mut chain,
            accepted_request_fixed(configuration, layout),
            input,
            run,
        );
        let first = frame.first().unwrap();
        assert_eq!(first.own.len(), 3);
        assert_eq!(first.at_start.len(), 1);
        assert_eq!(frame.iopp.statement.oracles.len(), 5);
        assert_eq!(frame.book.log_n(first.own[0]), Some(18));
        assert_eq!(frame.book.log_n(first.at_start[0]), Some(13));
        assert_eq!(frame.cursor.index, 5);
    }

    #[test]
    fn g0_lookup_forest_consumes_exact_transcript() {
        let policy = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let leaf = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin");
        let proof = include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.proof");
        let configuration = configuration_digest(&[policy, layout, kl]);
        let (root, record, _) = parse_leaf_record(leaf, 256).unwrap();
        let table = IndexName::new(&root, &configuration, record);
        let public = RequestPublic::new(
            &accepted_request_fixed(configuration, layout),
            &table,
            0,
            include_bytes!("../../spikes/plans/request-statement.bin"),
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.req"),
            include_bytes!("../../spikes/plans/request-first-plan.bin"),
        );
        let input = accepted_input();
        let plan = plan_proof(&input).unwrap();
        let file_layout = plan.layout().unwrap();
        let mut cursor = Cursor::new(proof, &file_layout, &configuration, &table.commitment);
        cursor.take("head", "").unwrap();
        let salt = crate::fp::digest::Digest::from_le_bytes(
            cursor.take("chain salt", "first chain").unwrap().try_into().unwrap(),
        );
        let query = public.first_query(&salt);
        let run = query.run.unwrap();
        let mut chain = Chain::new(query);
        let mut frame = RequestFrame::new(
            &mut cursor,
            &mut chain,
            accepted_request_fixed(configuration, layout),
            input,
            run,
        );
        frame.first().unwrap();
        let mut fork = crate::fp::model::HFork::default();
        fork.slots(3204);
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G0, look-up");
        let mut slot = 0;
        let forest = crate::fp::forest::forest_pad(
            &mut fork,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(0, false),
        )
        .unwrap();
        assert_eq!(slot, 1582);
        assert_eq!(forest.rows.len(), 75);
        assert_eq!(forest.zx.len(), 20);
        assert_eq!(forest.zy.len(), 4);
        assert_eq!(forest.calls[3].zk.len(), 21);
        let tables = crate::fp::layer::build_tables(&crate::fp::layer::accepted_layer(true), &forest).unwrap();
        assert_eq!(tables.fun.len(), 31, "{:?}", tables.offsets.keys().collect::<Vec<_>>());
        assert_eq!(tables.at("b-pin").unwrap() + 2048, 167328, "{:?}", tables.fun);
        let run = crate::fp::layer::through_terms(
            &mut fork,
            &mut tr,
            &mut slot,
            crate::fp::layer::accepted_layer(true),
            forest,
        )
        .unwrap();
        assert_eq!(slot, 1832);
        assert_eq!(run.tables.fun.len(), 31);
        let mut run = run;
        let mut check = crate::fp::layer_checks::LayerCheck::new(&mut run);
        let stack = check.stack(&mut fork, &mut tr, &mut slot).unwrap();
        assert_eq!(stack.panels.len(), 18, "{:?}", stack.panels.iter().map(|panel| &panel.sn).collect::<Vec<_>>());
        assert_eq!(stack.next_v, 32);
        assert_eq!(stack.s_a0, 1865);
        assert_eq!(stack.after_hd, 1943);
        assert_eq!(stack.after_hd2, 1980);
        drop(check);
        let mut merged = crate::fp::layer_checks::ws_fork_merge(&mut run, &stack, &mut fork, &mut tr, &mut slot).unwrap();
        assert_eq!(merged.rho.len(), 33);
        crate::fp::layer::assign_later_phi(&mut run).unwrap();
        assert_eq!(slot, 2006);
        let next_root = file_layout
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.message.name == "root" && field.message.context == "G0")
            .nth(2)
            .unwrap()
            .0;
        drop(tr);
        assert_eq!(frame.cursor.index, next_root);
        frame.later().unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G0, smallness test");
        let trits = tr.trits(3099, 128).unwrap();
        assert_eq!(trits.len(), 3099 * 128);
        drop(tr);
        frame.later().unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G0, smallness test, digits");
        let mut digits = crate::fp::forest::forest_pad(
            &mut fork,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(0, true),
        )
        .unwrap();
        assert_eq!(digits.rows.len(), 31 + 24 + 11 + 1);
        assert_eq!(slot, 3204);
        let mut check = crate::fp::layer_checks::LayerCheck::new(&mut run);
        check
            .objects_before_smallness(&stack, &mut merged, &mut fork, &mut tr)
            .unwrap();
        assert!(fork.n_claim < 1549);
        check.smallness(&mut digits, &trits, &mut fork).unwrap();
        assert_eq!(fork.n_claim, 1549, "before={} reads={:?}", check.before_smallness, check.reads_by_matrix);
        check.coefficients(&digits, &mut fork);
        let equation = check.door_h(&fork).unwrap();
        assert_eq!(equation.on_masks.len(), 3204);
        assert_eq!(equation.target, -fork.c0);
        frame.close(&equation).unwrap();
        let close = file_layout
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.message.name == "hiding sum" && field.message.context == "G0")
            .last()
            .unwrap()
            .0;
        assert_eq!(frame.cursor.index, close + 10);

        frame.first().unwrap();
        let mut a0 = crate::fp::model::HFork::default();
        a0.slots(10000);
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "A0, look-up");
        let mut slot = 0;
        let forest = crate::fp::forest::forest_pad(
            &mut a0,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(1, false),
        )
        .unwrap();
        let mut attention = crate::fp::attention::read_messages(
            &mut a0,
            &mut tr,
            &mut slot,
            crate::fp::attention::AttentionShape::accepted(),
            forest,
        )
        .unwrap();
        drop(tr);
        frame.later().unwrap();
        crate::fp::attention::build_tables(&mut attention).unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "A0, smallness test");
        let trits = tr.trits(1276, 128).unwrap();
        drop(tr);
        frame.later().unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "A0, smallness test, digits");
        let mut digits = crate::fp::forest::forest_pad(
            &mut a0,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(1, true),
        )
        .unwrap();
        attention.prepare(&mut a0, &mut tr, slot, &digits).unwrap();
        attention.add_main_claims(&mut a0).unwrap();
        attention.smallness(&mut digits, &trits, &mut a0).unwrap();
        assert_eq!(a0.n_claim, attention.n_lin);
        attention.coefficients(&digits, &mut a0);
        let equation = attention.door_h(1, &a0).unwrap();
        frame.close(&equation).unwrap();
        let close_a0 = file_layout
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.message.name == "hiding sum" && field.message.context == "A0")
            .last()
            .unwrap()
            .0;
        assert_eq!(frame.cursor.index, close_a0 + 8);

        frame.first().unwrap();
        let mut g1 = crate::fp::model::HFork::default();
        g1.slots(3203);
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G1, look-up");
        let mut slot = 0;
        let forest = crate::fp::forest::forest_pad(
            &mut g1,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(2, false),
        )
        .unwrap();
        assert_eq!(slot, 1582);
        let mut run = crate::fp::layer::through_terms(
            &mut g1,
            &mut tr,
            &mut slot,
            crate::fp::layer::accepted_layer(false),
            forest,
        )
        .unwrap();
        let mut check = crate::fp::layer_checks::LayerCheck::new(&mut run);
        let stack = check.stack(&mut g1, &mut tr, &mut slot).unwrap();
        drop(check);
        assert!(run.points.contains_key("Z_a"), "points={:?}", run.points.keys());
        let mut merged = crate::fp::layer_checks::ws_fork_merge(&mut run, &stack, &mut g1, &mut tr, &mut slot).unwrap();
        assert_eq!(merged.rho.len(), 31);
        assert!(run.points.contains_key("Z_a"), "points after merge={:?}", run.points.keys());
        crate::fp::layer::assign_later_phi(&mut run).unwrap();
        drop(tr);
        frame.later().unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G1, smallness test");
        let trits = tr.trits(3099, 128).unwrap();
        drop(tr);
        frame.later().unwrap();
        let mut tr = crate::fp::model::Tr::new(frame.cursor, frame.chain, "G1, smallness test, digits");
        let mut digits = crate::fp::forest::forest_pad(
            &mut g1,
            &mut tr,
            &mut slot,
            crate::fp::forest::accepted_forest(2, true),
        )
        .unwrap();
        let mut check = crate::fp::layer_checks::LayerCheck::new(&mut run);
        check.objects_before_smallness(&stack, &mut merged, &mut g1, &mut tr).unwrap();
        check.smallness(&mut digits, &trits, &mut g1).unwrap();
        // cp_verify: hp.nLin = 1516 for G1 (1549 for G0).
        assert_eq!(g1.n_claim, 1516);
        check.coefficients(&digits, &mut g1);
        let equation = check.door_h(&g1).unwrap();
        assert_eq!(equation.on_masks.len(), 3203);
        frame.close(&equation).unwrap();
        let close_g1 = file_layout
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| field.message.name == "hiding sum" && field.message.context == "G1")
            .last()
            .unwrap()
            .0;
        assert_eq!(frame.cursor.index, close_g1 + 10);
    }
}
