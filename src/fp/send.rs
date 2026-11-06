//! SEND sampling and oracle-tree bookkeeping for request proofs.

use alloc::{collections::BTreeMap, vec, vec::Vec};

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{send_form, Statement},
    error::FpError,
    field::ExtensionElement as E,
    key_public::OracleNotice,
    proximity::{Opened, OpenedColumn, SingleRows},
    transcript::{elements, words, Cursor, OpeningData},
};

pub(crate) fn sample_seed(chain: &mut Chain, bits: usize) -> (Vec<E>, E) {
    let z = (0..bits).map(|_| E::from_words(chain.draw_e())).collect();
    let beta = E::from_words(chain.draw_e_nonzero());
    (z, beta)
}

fn read_samples_inner(
    cursor: &mut Cursor,
    chain: &mut Chain,
    statement: &mut Statement,
    oracles: &[u32],
    context: &str,
    close_answers: bool,
) -> Result<(Vec<E>, Vec<u64>), FpError> {
    let bits = oracles
        .iter()
        .map(|&oracle| crate::fp::claims::sample_cube(statement, oracle).4 as usize)
        .max()
        .unwrap_or(0);
    let answer_bytes = cursor.take("sample answers", context)?;
    let answer_words = words(answer_bytes);
    let values = elements(answer_bytes);
    let mut answers = Vec::with_capacity(oracles.len());
    let mut at = 0;
    let most = oracles
        .iter()
        .map(|&oracle| statement.oracles[oracle as usize].shape.answers)
        .max()
        .unwrap_or(0);
    for answer in 0..most {
        let (z, beta) = sample_seed(chain, bits);
        for &oracle in oracles {
            let shape = &statement.oracles[oracle as usize].shape;
            if answer >= shape.answers {
                continue;
            }
            let mut sample = send_form(statement, oracle, &z, beta, answer);
            sample.target = *values.get(at).ok_or(FpError::BadLength {
                context: "sample-answers",
            })?;
            statement.oracles[oracle as usize].sample_rows.push(sample);
            statement.admit_sample(oracle);
            if answer == 0 {
                answers.push(values[at]);
            }
            at += 1;
        }
    }
    if at != values.len() {
        return Err(FpError::BadLength {
            context: "sample-answers",
        });
    }
    if close_answers {
        chain.close(ChainKind::Empty, &answer_words);
    }
    Ok((answers, answer_words))
}

pub(crate) fn read_samples_after_root(
    cursor: &mut Cursor,
    chain: &mut Chain,
    statement: &mut Statement,
    oracles: &[u32],
    context: &str,
) -> Result<Vec<E>, FpError> {
    read_samples_inner(cursor, chain, statement, oracles, context, true).map(|(answers, _)| answers)
}

#[derive(Debug, Clone, Copy)]
struct OracleLocation {
    tree: u32,
    main_at: usize,
    mask_at: usize,
}

#[derive(Debug, Clone)]
struct Tree {
    number: u32,
    log_n: u32,
}

#[derive(Debug, Default)]
pub(crate) struct OracleBook {
    trees: Vec<Tree>,
    locations: BTreeMap<u32, OracleLocation>,
}

#[derive(Debug)]
pub(crate) struct Sent {
    pub ids: Vec<u32>,
    pub answers: Vec<E>,
    pub answer_words: Vec<u64>,
}

impl OracleBook {
    pub fn send(
        &mut self,
        cursor: &mut Cursor,
        chain: &mut Chain,
        statement: &mut Statement,
        notices: &[OracleNotice],
        context: &str,
        close_answers: bool,
    ) -> Result<Sent, FpError> {
        let ids: Vec<u32> = notices
            .iter()
            .map(|notice| statement.sent(notice.shape.clone()))
            .collect();
        // (leaf class, main log_n, mask-only) -> local tree index. Mask-only
        // messages join the first tree of their class, as in lay_trees().
        let mut classes: Vec<(u32, u32, bool)> = vec![];
        let mut tree_for = vec![0usize; notices.len()];
        for mask_only in [false, true] {
            for (i, notice) in notices.iter().enumerate() {
                if (notice.shape.main_rows == 0) != mask_only {
                    continue;
                }
                let log_n = if mask_only {
                    0
                } else {
                    statement.base[notice.shape.code as usize].log_n
                };
                let at = classes
                    .iter()
                    .position(|&(class, tree_log_n, _)| {
                        class == notice.leaf_class && (mask_only || tree_log_n == log_n)
                    })
                    .unwrap_or_else(|| {
                        classes.push((notice.leaf_class, log_n, mask_only));
                        classes.len() - 1
                });
                tree_for[i] = at;
            }
        }
        let first_tree = self.trees.len() as u32;
        let mut root_words = Vec::new();
        for (local, _) in classes.iter().enumerate() {
            let number = first_tree + local as u32;
            let root = cursor.take_root_at(context, Some(number))?;
            root_words.extend(words(&root.to_le_bytes()));
            let log_n = notices
                .iter()
                .zip(&tree_for)
                .find(|(_, tree)| **tree == local)
                .map(|(notice, _)| {
                    if notice.shape.main_rows != 0 {
                        statement.base[notice.shape.code as usize].log_n
                    } else {
                        statement.mask[notice.shape.mask_code as usize].log_n
                    }
                })
                .unwrap_or(0);
            self.trees.push(Tree { number, log_n });
        }
        for local in 0..classes.len() {
            let total_main: usize = notices
                .iter()
                .zip(&tree_for)
                .filter(|(_, tree)| **tree == local)
                .map(|(notice, _)| notice.shape.main_rows as usize * notice.shape.main_deg as usize)
                .sum();
            let mut main_at = 0;
            let mut mask_at = total_main;
            for (i, notice) in notices.iter().enumerate() {
                if tree_for[i] != local {
                    continue;
                }
                self.locations.insert(
                    ids[i],
                    OracleLocation {
                        tree: first_tree + local as u32,
                        main_at,
                        mask_at,
                    },
                );
                main_at += notice.shape.main_rows as usize * notice.shape.main_deg as usize;
                mask_at += notice.shape.mask_rows as usize * 3;
            }
        }
        chain.close(ChainKind::Seed, &root_words);
        let sampled: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&oracle| statement.oracles[oracle as usize].shape.answers != 0)
            .collect();
        let (answers, answer_words) = read_samples_inner(
            cursor,
            chain,
            statement,
            &sampled,
            context,
            close_answers,
        )?;
        #[cfg(feature = "std")]
        if context == "G0" || context == "A0" {
            eprintln!("{} sampled={:?} shapes={:?} words={:?}", context, sampled, ids.iter().map(|&id| statement.oracles[id as usize].shape.answers).collect::<Vec<_>>(), answer_words);
        }
        Ok(Sent {
            ids,
            answers,
            answer_words,
        })
    }

    pub fn opened(
        &self,
        rows: &SingleRows,
        statement: &Statement,
        openings: &[OpeningData],
        bytes: &[u8],
        raw_positions: &[u64],
        base_star: bool,
    ) -> Result<Opened, FpError> {
        let ids = rows.oracles();
        let mut sorted_positions = raw_positions.to_vec();
        sorted_positions.sort_unstable();
        sorted_positions.dedup();
        let mut out = Opened::default();
        for (i, &oracle) in ids.iter().enumerate() {
            let location = self.locations.get(&oracle).ok_or(FpError::Range {
                context: "oracle-tree",
            })?;
            let opening = openings
                .iter()
                .find(|opening| opening.tree_number == location.tree)
                .ok_or(FpError::Range {
                    context: "oracle-opening",
                })?;
            let shape = &statement.oracles[oracle as usize].shape;
            for &raw_position in &sorted_positions {
                let position = raw_position & ((1u64 << opening.log_n) - 1);
                let main = if shape.main_deg == 1 {
                    opening
                        .column_f(bytes, position, location.main_at, shape.main_rows as usize)
                        .into_iter()
                        .map(E::from_base)
                        .collect()
                } else {
                    opening.column_e(bytes, position, location.main_at, shape.main_rows as usize)
                };
                let mask = opening.column_e(bytes, position, location.mask_at, shape.mask_rows as usize);
                if out.of.len() <= i {
                    out.of.push(vec![]);
                }
                out.of[i].push(OpenedColumn { main, mask });
            }
        }
        if base_star {
            let total_main: usize = ids
                .iter()
                .map(|&oracle| {
                    let shape = &statement.oracles[oracle as usize].shape;
                    shape.main_rows as usize * shape.main_deg as usize
                })
                .sum();
            let total_mask: usize = ids
                .iter()
                .map(|&oracle| statement.oracles[oracle as usize].shape.mask_rows as usize * 3)
                .sum();
            let primary = ids.first().ok_or(FpError::Range {
                context: "base-star",
            })?;
            let location = self.locations.get(primary).ok_or(FpError::Range {
                context: "base-star",
            })?;
            let opening = openings
                .iter()
                .find(|opening| opening.tree_number == location.tree)
                .ok_or(FpError::Range {
                    context: "base-star",
                })?;
            for &raw_position in &sorted_positions {
                let position = raw_position & ((1u64 << opening.log_n) - 1);
                let column = OpenedColumn {
                    main: opening.column_e(bytes, position, total_main, 1),
                    mask: opening.column_e(bytes, position, total_main + total_mask, 1),
                };
                if out.of.len() == ids.len() {
                    out.of.push(vec![]);
                }
                out.of[ids.len()].push(column);
            }
        }
        Ok(out)
    }

    pub fn log_n(&self, oracle: u32) -> Option<u32> {
        let location = self.locations.get(&oracle)?;
        self.trees
            .iter()
            .find(|tree| tree.number == location.tree)
            .map(|tree| tree.log_n)
    }
}
