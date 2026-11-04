//! Protocol-chain byte planners (numbers/blocks.h, frame/link_frame.h).
//!
//! These functions compute layout only. Soundness analysis lives in
//! `fp::analysis`; no byte plan is itself a proof-verification result.

use alloc::{string::String, vec, vec::Vec};

use crate::fp::{
    analysis::{
        analysis_of_m, powers_error, solve_group, solve_stages, stage_cube_bits, GroupLoad,
        GroupSize, Scope, Setting, Stage, StageChoice,
    },
    codes::positions,
    error::FpError,
    layout::{ChainPlan, FilePlan, MessageKind, OpeningSize, PlanMessage},
};

const E_BYTES: u64 = 24;
const F_BYTES: u64 = 8;
const DIGEST_BYTES: u64 = 32;
const SALT_BYTES: u64 = 40;
const NONCE_BYTES: u64 = 8;

fn scalars(name: &str, context: &str, count: u64) -> Result<PlanMessage, FpError> {
    let bytes = count.checked_mul(E_BYTES).ok_or(FpError::Range {
        context: "plan-scalar-bytes",
    })?;
    Ok(PlanMessage::fixed(name, context, MessageKind::Scalars, bytes).in_chain())
}

fn answers(context: &str, count: u64) -> Result<PlanMessage, FpError> {
    let bytes = count.checked_mul(E_BYTES).ok_or(FpError::Range {
        context: "plan-answer-bytes",
    })?;
    Ok(PlanMessage::fixed("sample answers", context, MessageKind::Answers, bytes).in_chain())
}

fn root(context: &str) -> PlanMessage {
    PlanMessage::fixed("root", context, MessageKind::Root, DIGEST_BYTES).in_chain()
}

fn nonce(context: &str, work_bits: u32) -> Option<PlanMessage> {
    (work_bits != 0).then(|| PlanMessage::fixed("nonce", context, MessageKind::Nonce, NONCE_BYTES))
}

/// Sizes of one Merkle tree opened at a list of positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeSize {
    pub name: String,
    pub log_n: u32,
    pub leaf_bytes: u64,
    pub one_list: bool,
}

impl TreeSize {
    pub fn new(name: impl Into<String>, log_n: u32, leaf_bytes: u64) -> Self {
        Self {
            name: name.into(),
            log_n,
            leaf_bytes,
            one_list: false,
        }
    }
}

/// Mean number of distinct leaves hit by independent uniform draws.
pub fn expected_distinct(log_n: u32, t: u64) -> f64 {
    let n = libm::ldexp(1.0, log_n as i32);
    -n * libm::expm1(t as f64 * libm::log1p(-1.0 / n))
}

/// Maximum sibling digests in a pruned opening (numbers/blocks.h).
pub fn path_digests_most(log_n: u32, mut t: u64) -> u64 {
    if t == 0 || log_n == 0 {
        return 0;
    }
    t = t.min(1u64 << (log_n - 1));
    let mut a = 0u32;
    while (2u64 << a) <= t {
        a += 1;
    }
    t * (log_n - a - 1) as u64 + (2u64 << a) - t
}

/// Expected sibling digests after path pruning.
pub fn path_digests_expected(log_n: u32, t: u64) -> f64 {
    let mut digests = 0.0;
    for j in 1..=log_n {
        let n = libm::ldexp(1.0, j as i32);
        let one = libm::exp(t as f64 * libm::log1p(-1.0 / n));
        let two = if j == 1 {
            0.0
        } else {
            libm::exp(t as f64 * libm::log1p(-2.0 / n))
        };
        digests += n * (one - two);
    }
    digests
}

/// Append symbols, salts and pruned paths for one tree opening.
pub fn plan_open(
    messages: &mut Vec<PlanMessage>,
    context: &str,
    tree: &TreeSize,
    t: u64,
) -> Result<(), FpError> {
    let where_tree = alloc::format!("{context}, {}", tree.name);
    let distinct = expected_distinct(tree.log_n, t);
    let symbols = t.checked_mul(tree.leaf_bytes).ok_or(FpError::Range {
        context: "opening-symbol-bytes",
    })?;
    let salts = t.checked_mul(SALT_BYTES).ok_or(FpError::Range {
        context: "opening-salt-bytes",
    })?;
    let paths = path_digests_most(tree.log_n, t)
        .checked_mul(DIGEST_BYTES)
        .ok_or(FpError::Range {
            context: "opening-path-bytes",
        })?;
    let opening = OpeningSize {
        log_n: tree.log_n,
        leaf_bytes: tree.leaf_bytes,
        draws: t,
    };
    messages.push(
        PlanMessage::position_dependent(
            "opened symbols",
            &where_tree,
            MessageKind::Columns,
            symbols,
            distinct * tree.leaf_bytes as f64,
        )
        .with_opening(opening),
    );
    messages.push(
        PlanMessage::position_dependent(
            "salts",
            &where_tree,
            MessageKind::Salts,
            salts,
            distinct * SALT_BYTES as f64,
        )
        .with_opening(opening),
    );
    messages.push(
        PlanMessage::position_dependent(
            "paths",
            &where_tree,
            MessageKind::Paths,
            paths,
            path_digests_expected(tree.log_n, t) * DIGEST_BYTES as f64,
        )
        .with_opening(opening),
    );
    Ok(())
}

fn cs_positions(
    setting: &Setting,
    scope: Scope,
    from: &Stage,
    to: &Stage,
    symbols: u64,
) -> Result<u64, FpError> {
    let agree = analysis_of_m(&from.g).at.agree;
    let list = analysis_of_m(&to.g).list;
    let mut t = 0u64;
    loop {
        let rows = 1u64
            + setting.samples
            + symbols.checked_mul(t).ok_or(FpError::Range {
                context: "cs-batch-rows",
            })?;
        let extra = list as f64 * powers_error(setting, rows);
        let next = positions(agree, setting.target - setting.query_work(scope), extra, 1).ok_or(
            FpError::Range {
                context: "cs-no-positions",
            },
        )?;
        if next == t {
            return Ok(t);
        }
        t = next;
    }
}

fn base_positions(setting: &Setting, scope: Scope, stage: &Stage) -> Result<u64, FpError> {
    positions(
        analysis_of_m(&stage.g).at.agree,
        setting.target - setting.query_work(scope),
        0.0,
        1,
    )
    .ok_or(FpError::Range {
        context: "base-no-positions",
    })
}

/// Byte layout of a full zero-knowledge proximity test (`plan_iopp`).
pub fn plan_iopp(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    where_: &str,
    stages: &[Stage],
    first_trees: &[TreeSize],
    first_mask_roots: bool,
    masks_ride: bool,
) -> Result<(), FpError> {
    if stages.len() < 2 {
        return Err(FpError::Range {
            context: "iopp-stage-count",
        });
    }
    if let Some(n) = nonce(where_, setting.work_field[scope as usize]) {
        messages.push(n); // BAT before the first fold
    }
    let mut trees = first_trees.to_vec();
    for i in 0..stages.len() - 1 {
        let stage = &stages[i];
        let next = &stages[i + 1];
        let g = &stage.g;
        let n = &next.g;
        let context = alloc::format!("{where_}, stage {}", i + 1);
        let fresh_root = if i == 0 {
            first_mask_roots
        } else {
            !masks_ride
        };
        if fresh_root {
            messages.push(root(&context));
        }
        messages.push(scalars("hiding sum", &context, 1)?);
        if let Some(nonce) = nonce(&context, setting.work_field[scope as usize]) {
            messages.push(nonce);
        }
        let coefficients = if setting.saved_coefficient { 2 } else { 3 };
        for _j in 1..=g.k {
            messages.push(scalars("round polynomial", &context, coefficients)?);
            if let Some(nonce) = nonce(&context, setting.work_closeness[scope as usize]) {
                messages.push(nonce);
            }
        }
        if g.has_mask() && fresh_root {
            let mut mask = TreeSize::new(
                "fresh mask",
                g.mask.log_n,
                E_BYTES.checked_shl(g.h).ok_or(FpError::Range {
                    context: "mask-leaf-bytes",
                })?,
            );
            mask.one_list = true;
            trees.push(mask);
        }
        let symbols = g.slots + if g.has_mask() { 2 } else { 0 };
        let t = cs_positions(setting, scope, stage, next, symbols)?;
        if t > stage.t {
            return Err(FpError::Range {
                context: "cs-positions-exceed-tail",
            });
        }
        messages.push(root(&context));
        if let Some(nonce) = nonce(&context, setting.work_sample[scope as usize]) {
            messages.push(nonce);
        }
        messages.push(answers(&context, setting.samples)?);
        if let Some(nonce) = nonce(&context, setting.query_work(scope)) {
            messages.push(nonce);
        }
        for tree in &trees {
            plan_open(messages, &context, tree, t)?;
        }
        let main = E_BYTES.checked_shl(n.k).ok_or(FpError::Range {
            context: "switch-leaf-bytes",
        })?;
        let mask = E_BYTES
            .checked_shl(n.h)
            .and_then(|x| x.checked_mul(if masks_ride { 2 } else { 1 }))
            .ok_or(FpError::Range {
                context: "switch-leaf-bytes",
            })?;
        let last_extra = if masks_ride && i + 2 == stages.len() {
            E_BYTES
        } else {
            0
        };
        let leaf = main
            .checked_add(mask)
            .and_then(|x| x.checked_add(last_extra))
            .ok_or(FpError::Range {
                context: "switch-leaf-bytes",
            })?;
        let mut switch = TreeSize::new("switch oracle", n.main.log_n, leaf);
        switch.one_list = masks_ride;
        trees = vec![switch];
    }
    let last = stages.last().unwrap();
    let g = &last.g;
    let context = alloc::format!("{where_}, last stage");
    if !masks_ride {
        let mut mask = TreeSize::new("masking codeword", g.main.log_n, 2 * E_BYTES);
        mask.one_list = true;
        trees.push(mask);
        messages.push(root(&context));
    }
    messages.push(scalars("masked target", &context, 1)?);
    if let Some(nonce) = nonce(&context, setting.work_closeness[scope as usize]) {
        messages.push(nonce);
    }
    let clear = g
        .main
        .coordinates()
        .checked_mul(g.slots)
        .and_then(|x| {
            x.checked_add(if g.has_mask() {
                g.mask.coordinates()
            } else {
                0
            })
        })
        .ok_or(FpError::Range {
            context: "base-clear-values",
        })?;
    messages.push(scalars("masked message and randomness", &context, clear)?);
    let t = base_positions(setting, scope, last)?;
    if t > last.t {
        return Err(FpError::Range {
            context: "base-positions-exceed-tail",
        });
    }
    if let Some(nonce) = nonce(&context, setting.query_work(scope)) {
        messages.push(nonce);
    }
    for tree in &trees {
        plan_open(messages, &context, tree, t)?;
    }
    Ok(())
}

/// Public sizes needed to plan a leaf/LINK file. Distances and table-column
/// counts come from the committed-table analysis; folded row shapes come from
/// LIN. No field offsets or file lengths are inputs.
#[derive(Debug, Clone)]
pub struct EntryLayoutInput {
    pub setting: Setting,
    pub table_answers: u64,
    pub table_tree: TreeSize,
    pub columns_first: u64,
    pub columns_further: u64,
    /// Number of rows in each table's folded message before padding.
    pub folded_message_rows: u64,
    pub first_group: GroupSize,
    pub stage_choices: Vec<StageChoice>,
    pub masks_ahead: bool,
    pub masks_ride: bool,
}

/// Plan a complete leaf file (`file_entry` + `link_plan`).
pub fn plan_entry(input: &EntryLayoutInput) -> Result<FilePlan, FpError> {
    if input.columns_first == 0 || input.columns_further == 0 {
        return Err(FpError::Range {
            context: "link-no-columns",
        });
    }
    let setting = &input.setting;
    let scope = Scope::Commit;
    let rows = 2u64
        .checked_mul(setting.samples)
        .and_then(|x| x.checked_add(input.columns_first))
        .and_then(|x| x.checked_add(input.columns_further))
        .and_then(|x| x.checked_add(1))
        .and_then(|x| x.checked_add(2 * setting.samples))
        .ok_or(FpError::Range {
            context: "link-batched-rows",
        })?;
    let load = GroupLoad {
        cube_bits: stage_cube_bits(&input.first_group),
        messages: 2,
        batched_rows: rows,
    };
    let (first, first_t) = solve_group(setting, scope, &input.first_group, &load)?;
    let stages = solve_stages(setting, scope, &first, first_t, &input.stage_choices)?;

    let mut messages = vec![PlanMessage::fixed(
        "chain salt",
        "link's chain",
        MessageKind::Salts,
        DIGEST_BYTES,
    )];
    messages.push(root("link"));
    if let Some(nonce) = nonce("link", setting.work_sample[scope as usize]) {
        messages.push(nonce);
    }
    messages.push(answers("link", 2 * setting.samples)?);
    if let Some(nonce) = nonce("link", setting.query_work(scope)) {
        messages.push(nonce);
    }
    plan_open(
        &mut messages,
        "link, first table",
        &input.table_tree,
        input.columns_first,
    )?;
    plan_open(
        &mut messages,
        "link, further table",
        &input.table_tree,
        input.columns_further,
    )?;

    let mask_rows = 1u64.checked_shl(first.h).ok_or(FpError::Range {
        context: "link-mask-rows",
    })?;
    let folded = input
        .folded_message_rows
        .checked_mul(2 * F_BYTES)
        .and_then(|x| x.checked_add(mask_rows * 3 * F_BYTES))
        .ok_or(FpError::Range {
            context: "link-first-tree-leaf",
        })?;
    let first_tree = TreeSize::new(
        "tree of the first table's folded row; the further table's folded row; fold mask",
        first.main.log_n,
        folded,
    );
    plan_iopp(
        &mut messages,
        setting,
        scope,
        "the two folded rows",
        &stages,
        core::slice::from_ref(&first_tree),
        !input.masks_ahead,
        input.masks_ride,
    )?;
    FilePlan::entry(
        input.table_answers,
        ChainPlan::new("link's chain", messages),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{analysis::GroupSize, codes::CodeSize};

    fn example_input() -> EntryLayoutInput {
        EntryLayoutInput {
            setting: Setting::example(),
            table_answers: 256,
            table_tree: TreeSize::new("table", 23, 2048),
            columns_first: 51,
            columns_further: 51,
            folded_message_rows: 54,
            first_group: GroupSize {
                k: 6,
                h: 0,
                slots: 1,
                main: CodeSize {
                    log_n: 17,
                    ell: 32768,
                    r: 0,
                    over_E: true,
                },
                mask: CodeSize {
                    log_n: 13,
                    ell: 64,
                    r: 0,
                    over_E: true,
                },
                m: 0,
            },
            stage_choices: vec![
                StageChoice {
                    k: 4,
                    h: 2,
                    log_n: 14,
                    mask_log_n: 12,
                },
                StageChoice {
                    k: 0,
                    h: 0,
                    log_n: 14,
                    mask_log_n: 14,
                },
            ],
            masks_ahead: true,
            masks_ride: true,
        }
    }

    #[test]
    fn merkle_path_bounds_match_oracle_openings() {
        assert_eq!(path_digests_most(23, 51), 880);
        assert_eq!(path_digests_most(17, 97), 1001);
        assert_eq!(path_digests_most(14, 66), 524);
        assert!(path_digests_expected(23, 51) <= 880.0);
    }

    #[test]
    fn complete_entry_layout_matches_all_oracle_offsets_and_lengths() {
        let plan = plan_entry(&example_input()).unwrap();
        let layout = plan.layout().unwrap();
        assert_eq!(layout.fields().len(), 43);
        assert_eq!(layout.maximum_bytes(), 546112);

        let fields = layout.fields();
        let expected = [
            (0, 8),
            (8, 32),
            (40, 6144),
            (6184, 32),
            (6216, 32),
            (6248, 48),
            (6296, 8),
            (6304, 104448),
            (110752, 2040),
            (112792, 28160),
            (140952, 104448),
            (245400, 2040),
            (247440, 28160),
            (275600, 24),
            (275624, 48),
            (275672, 48),
            (275720, 48),
            (275768, 48),
            (275816, 48),
            (275864, 48),
            (275912, 32),
            (275944, 24),
            (275968, 8),
            (275976, 86136),
            (362112, 3880),
            (365992, 32032),
            (398024, 24),
            (398048, 48),
            (398096, 48),
            (398144, 48),
            (398192, 48),
            (398240, 32),
            (398272, 24),
            (398296, 8),
            (398304, 38016),
            (436320, 2640),
            (438960, 16768),
            (455728, 24),
            (455752, 64608),
            (520360, 8),
            (520368, 6336),
            (526704, 2640),
            (529344, 16768),
        ];
        for (field, (offset, bytes)) in fields.iter().zip(expected) {
            assert_eq!(
                (field.offset, field.message.bytes),
                (offset, bytes),
                "field {}",
                field.message.name
            );
            assert_eq!(field.end(), offset + bytes);
        }
    }

    #[test]
    fn entry_layout_rejects_more_than_maximum_but_allows_pruned_openings() {
        let plan = plan_entry(&example_input()).unwrap();
        let layout = plan.layout().unwrap();
        layout.validate_len(546112).unwrap();
        layout.validate_len(400000).unwrap();
        assert!(matches!(
            layout.validate_len(546113),
            Err(FpError::TrailingBytes { count: 1, .. })
        ));
    }
}
