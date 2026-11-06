//! Key-file planning (frame/files.h, lin_frame.h, wf_frame.h).
//!
//! Inputs are solved public sizes from LIN and WF. This module composes their
//! byte layouts; it does not re-derive model shapes or execute transcripts.

use alloc::{string::String, vec, vec::Vec};

use crate::fp::{
    algebra::{plan_chain_start, plan_lookup, plan_member_rows, plan_send, plan_split, LookupSize},
    analysis::{Scope, Setting, Stage},
    error::FpError,
    layout::{ChainPlan, FilePlan, PlanMessage},
    plan::{nonce, plan_iopp, plan_iopp_at, plan_open, TreeSize},
};

/// Solved trees and proximity stages used by LIN's three chains.
#[derive(Debug, Clone)]
pub struct LinLayoutInput {
    pub folded_stages: Vec<Stage>,
    pub omega_stages: Vec<Stage>,
    pub own_stages: Vec<Stage>,
    /// `lin_message_u`: folded row and, for a split, its shared mask.
    pub folded_trees: Vec<TreeSize>,
    /// `lin_message_omega`: Omega and, for a split, the shared mask.
    pub omega_trees: Vec<TreeSize>,
    pub w_trees: Vec<TreeSize>,
    pub input_trees: Vec<TreeSize>,
    pub table_tree: TreeSize,
    pub columns: u64,
    pub split: bool,
    pub one_time: bool,
    pub masks_ahead: bool,
    pub masks_ride: bool,
}

/// One named range lookup.
#[derive(Debug, Clone)]
pub struct WfLookup {
    pub name: String,
    pub size: LookupSize,
}

/// Solved range-proof sizes used by the key's one-time chain.
#[derive(Debug, Clone)]
pub struct WfLayoutInput {
    pub stages: Vec<Stage>,
    pub trees: Vec<TreeSize>,
    pub lookups: Vec<WfLookup>,
    pub pending: u64,
    pub k_col: u32,
    pub masks_ride: bool,
}

#[derive(Debug, Clone)]
pub struct KeyLayoutInput {
    pub setting: Setting,
    pub table_answers: u64,
    pub lin: LinLayoutInput,
    pub wf: WfLayoutInput,
}

fn scopes(lin: &LinLayoutInput) -> (Scope, Scope) {
    if lin.one_time {
        (Scope::Commit, Scope::Commit)
    } else {
        (Scope::Shared, Scope::Request)
    }
}

/// LIN's first message, W and its sample.
pub(crate) fn lin_first(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    lin: &LinLayoutInput,
) -> Result<(), FpError> {
    let (_, request) = scopes(lin);
    plan_send(messages, setting, request, "table, W", lin.w_trees.len(), 1)
}

/// LIN's shared chain: folded row, table columns, Omega, optional split and
/// the folded/Omega proximity tests.
pub(crate) fn lin_shared(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    lin: &LinLayoutInput,
) -> Result<(), FpError> {
    let (shared, _) = scopes(lin);
    if let Some(nonce) = nonce("table", setting.work_closeness[shared as usize]) {
        messages.push(nonce);
    }
    plan_send(
        messages,
        setting,
        shared,
        "table, folded row",
        lin.folded_trees.len(),
        1,
    )?;
    if let Some(nonce) = nonce("table, columns", setting.query_work(shared)) {
        messages.push(nonce);
    }
    plan_open(messages, "table", &lin.table_tree, lin.columns)?;
    plan_send(
        messages,
        setting,
        shared,
        "table, Omega",
        lin.omega_trees.len(),
        if lin.split { 2 } else { 1 },
    )?;
    if let Some(nonce) = nonce("table", setting.work_field[shared as usize]) {
        messages.push(nonce); // LIN zeta
    }

    let mut folded_trees = lin.folded_trees.clone();
    if lin.split {
        plan_split(messages, "Omega against the folded row")?;
        let shared_mask = lin.omega_trees.get(1).ok_or(FpError::Range {
            context: "lin-shared-mask-tree",
        })?;
        folded_trees.push(shared_mask.clone());
        plan_iopp(
            messages,
            setting,
            shared,
            "folded row",
            &lin.folded_stages,
            &folded_trees,
            !lin.masks_ahead,
            lin.masks_ride,
        )?;
        plan_iopp_at(
            messages,
            setting,
            shared,
            "Omega",
            &lin.omega_stages,
            &lin.omega_trees,
            true,
            lin.masks_ride,
            5,
        )?;
    } else {
        let omega = lin.omega_trees.first().ok_or(FpError::Range {
            context: "lin-omega-tree",
        })?;
        folded_trees.push(omega.clone());
        plan_iopp(
            messages,
            setting,
            shared,
            "folded row and Omega",
            &lin.folded_stages,
            &folded_trees,
            !lin.masks_ahead,
            lin.masks_ride,
        )?;
    }
    Ok(())
}

/// LIN's request-own second chain.
pub(crate) fn lin_second(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    lin: &LinLayoutInput,
) -> Result<(), FpError> {
    let (_, request) = scopes(lin);
    let mut trees = lin.input_trees.clone();
    trees.extend(lin.w_trees.iter().cloned());
    let omega = lin.omega_trees.first().ok_or(FpError::Range {
        context: "lin-omega-tree",
    })?;
    trees.push(omega.clone());
    plan_iopp_at(
        messages,
        setting,
        request,
        "a request's own test",
        &lin.own_stages,
        &trees,
        !lin.masks_ahead,
        lin.masks_ride,
        7,
    )
}

/// WF's multiplicities, lookups, proximity test and pending-claim batching.
fn wf(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    wf: &WfLayoutInput,
) -> Result<(), FpError> {
    plan_send(
        messages,
        setting,
        Scope::Commit,
        "ranges: multiplicities",
        wf.trees.len(),
        1,
    )?;
    for lookup in &wf.lookups {
        plan_lookup(messages, setting, Scope::Commit, &lookup.name, &lookup.size)?;
    }
    plan_iopp(
        messages,
        setting,
        Scope::Commit,
        "ranges",
        &wf.stages,
        &wf.trees,
        !wf.masks_ride,
        wf.masks_ride,
    )?;
    plan_member_rows(
        messages,
        setting,
        Scope::Commit,
        "ranges: rows on the table",
        wf.pending,
        wf.k_col,
    )
}

/// Plan all three chains of a key file.
pub fn plan_key(input: &KeyLayoutInput) -> Result<FilePlan, FpError> {
    let mut one_time = vec![];
    plan_chain_start(&mut one_time, "one-time chain", true);
    plan_send(
        &mut one_time,
        &input.setting,
        Scope::Commit,
        "masks of the committer's rows on the table",
        input.lin.input_trees.len(),
        1,
    )?;
    wf(&mut one_time, &input.setting, &input.wf)?;
    lin_first(&mut one_time, &input.setting, &input.lin)?;

    let mut shared = vec![];
    plan_chain_start(&mut shared, "shared chain", false);
    lin_shared(&mut shared, &input.setting, &input.lin)?;

    let mut second = vec![];
    plan_chain_start(&mut second, "second chain", true);
    lin_second(&mut second, &input.setting, &input.lin)?;

    FilePlan::key(
        input.table_answers,
        [
            ChainPlan::new("one-time chain", one_time),
            ChainPlan::new("shared chain", shared),
            ChainPlan::new("second chain", second),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{analysis::GroupSize, codes::CodeSize};

    fn stage(
        k: u32,
        h: u32,
        main: (u32, u64, u64, bool),
        mask: (u32, u64, u64, bool),
        m: u64,
        t: u64,
    ) -> Stage {
        Stage {
            g: GroupSize {
                k,
                h,
                slots: 1,
                main: CodeSize {
                    log_n: main.0,
                    ell: main.1,
                    r: main.2,
                    over_E: main.3,
                },
                mask: CodeSize {
                    log_n: mask.0,
                    ell: mask.1,
                    r: mask.2,
                    over_E: mask.3,
                },
                m,
            },
            t,
            r_ood: 0,
        }
    }

    fn tree(name: &str, log_n: u32, leaf_bytes: u64) -> TreeSize {
        TreeSize::new(name, log_n, leaf_bytes)
    }

    pub(crate) fn accepted_input() -> KeyLayoutInput {
        let folded_stages = vec![
            stage(5, 0, (17, 32768, 97, true), (13, 64, 146, true), 483, 97),
            stage(4, 2, (14, 2048, 66, true), (12, 256, 66, true), 600, 66),
            stage(0, 0, (14, 2048, 66, true), (14, 1024, 66, true), 600, 66),
        ];
        let omega_stages = vec![
            stage(2, 0, (18, 16384, 98, true), (13, 64, 146, true), 277, 49),
            stage(4, 3, (14, 1024, 49, true), (13, 128, 49, true), 489, 49),
            stage(0, 0, (15, 1024, 39, true), (15, 512, 39, true), 345, 39),
        ];
        let own_stages = vec![
            stage(3, 0, (18, 16384, 98, true), (13, 256, 49, true), 277, 49),
            stage(4, 3, (14, 1024, 49, true), (13, 128, 49, true), 489, 49),
            stage(0, 0, (15, 1024, 39, true), (15, 512, 39, true), 345, 39),
        ];
        KeyLayoutInput {
            setting: Setting::example(),
            table_answers: 256,
            lin: LinLayoutInput {
                folded_stages,
                omega_stages,
                own_stages,
                folded_trees: vec![tree("tree of the folded row; fold mask", 17, 456)
                    .numbered(0)
                    .rooted("table, folded row")],
                omega_trees: vec![
                    tree("tree of Omega", 18, 96)
                        .numbered(1)
                        .rooted("table, Omega"),
                    tree(
                        "tree of the mask of the split between Omega and the folded row",
                        13,
                        24,
                    )
                    .numbered(2)
                    .rooted("table, Omega"),
                ],
                w_trees: vec![tree("tree of W; fold mask", 18, 48)
                    .numbered(5)
                    .rooted("table, W")],
                input_trees: vec![tree("tree of a request's masks", 13, 24)
                    .rooted("masks of the committer's rows on the table")],
                table_tree: tree("table", 23, 2048).stored("table"),
                columns: 51,
                split: true,
                one_time: true,
                masks_ahead: true,
                masks_ride: true,
            },
            wf: WfLayoutInput {
                stages: vec![
                    stage(3, 3, (22, 262144, 49, false), (16, 1024, 49, true), 158, 49),
                    stage(4, 3, (19, 16384, 39, true), (17, 512, 39, true), 195, 39),
                    stage(3, 3, (17, 2048, 33, true), (17, 256, 33, true), 210, 33),
                    stage(0, 0, (20, 2048, 22, true), (20, 1024, 22, true), 74, 22),
                ],
                trees: vec![tree(
                    "tree of multiplicities of the ranges' look-ups; fold mask",
                    22,
                    368,
                )
                .numbered(1)
                .rooted("ranges: multiplicities")],
                lookups: vec![
                    WfLookup {
                        name: "ranges: bytes".into(),
                        size: LookupSize::new(8, 5, 28, vec![25, 25, 25, 25]),
                    },
                    WfLookup {
                        name: "ranges: weights".into(),
                        size: LookupSize::new(13, 5, 27, vec![24, 24, 24, 24]),
                    },
                ],
                pending: 8,
                k_col: 11,
                masks_ride: true,
            },
        }
    }

    #[test]
    fn complete_key_layout_matches_every_oracle_field() {
        let plan = plan_key(&accepted_input()).unwrap();
        let layout = plan.layout().unwrap();

        let expected: Vec<(u64, u64, String)> =
            include_str!("../../spikes/plans/request-fields.txt")
                .lines()
                .filter(|line| line.starts_with("field: key "))
                .map(|line| {
                    let rest = line.strip_prefix("field: key ").unwrap();
                    let (location, label) = rest.split_once(": ").unwrap();
                    let mut words = location.split_whitespace();
                    let _mode = words.next().unwrap();
                    let offset = words.next().unwrap().parse().unwrap();
                    let bytes = words.next().unwrap().parse().unwrap();
                    let _kind = words.next().unwrap();
                    (offset, bytes, label.to_string())
                })
                .collect();
        for (index, (field, (offset, bytes, label))) in
            layout.fields().iter().zip(expected.iter()).enumerate()
        {
            let actual_label = if field.message.context.is_empty() {
                field.message.name.clone()
            } else {
                alloc::format!("{}: {}", field.message.context, field.message.name)
            };
            assert_eq!(
                (field.offset, field.message.bytes),
                (*offset, *bytes),
                "field {index}: {label}"
            );
            assert_eq!(actual_label.as_str(), label.as_str(), "field {index}");
        }
        assert_eq!(layout.maximum_bytes(), 1120888);
        assert_eq!(layout.fields().len(), 3583);
        assert_eq!(expected.len(), layout.fields().len());
    }

    #[test]
    fn stored_key_is_consumed_exactly_with_oracle_positions() {
        use crate::fp::{
            error::FpError,
            walk::{walk_file, OpeningActual},
        };

        let key = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/key.bin")
            .as_slice();
        assert_eq!(key.len(), 1096408);
        let plan = plan_key(&accepted_input()).unwrap();
        let layout = plan.layout().unwrap();
        let lists: Vec<Vec<u64>> = include_str!("../../spikes/plans/key-openings.txt")
            .lines()
            .map(|line| {
                let mut words = line.split_whitespace();
                assert_eq!(words.next(), Some("OPEN"));
                let _log_n: u32 = words.next().unwrap().parse().unwrap();
                let draws: usize = words.next().unwrap().parse().unwrap();
                let positions: Vec<u64> = words.map(|word| word.parse().unwrap()).collect();
                assert_eq!(positions.len(), draws);
                positions
            })
            .collect();
        let contexts = [
            "ranges, stage 1, tree of multiplicities of the ranges' look-ups; fold mask",
            "ranges, stage 2, switch oracle",
            "ranges, stage 3, switch oracle",
            "ranges, last stage, switch oracle",
            "table, table",
            "folded row, stage 1, tree of the folded row; fold mask",
            "folded row, stage 1, tree of the mask of the split between Omega and the folded row",
            "folded row, stage 2, switch oracle",
            "folded row, last stage, switch oracle",
            "Omega, stage 1, tree of Omega",
            "Omega, stage 1, tree of the mask of the split between Omega and the folded row",
            "Omega, stage 1, fresh mask",
            "Omega, stage 2, switch oracle",
            "Omega, last stage, switch oracle",
            "a request's own test, stage 1, tree of a request's masks",
            "a request's own test, stage 1, tree of W; fold mask",
            "a request's own test, stage 1, tree of Omega",
            "a request's own test, stage 2, switch oracle",
            "a request's own test, last stage, switch oracle",
        ];
        let openings: Vec<OpeningActual<'_>> = contexts
            .iter()
            .zip(&lists)
            .map(|(context, positions)| OpeningActual { context, positions })
            .collect();
        let walked = walk_file(key, &layout, &openings).expect("stored key consumes exactly");
        assert_eq!(walked.bytes(), key.len() as u64);
        assert_eq!(walked.fields.len(), 3583);

        let mut longer = key.to_vec();
        longer.push(0);
        assert!(matches!(
            walk_file(&longer, &layout, &openings),
            Err(FpError::TrailingBytes { count: 1, .. })
        ));
        let mut shorter = key.to_vec();
        shorter.pop();
        assert!(matches!(
            walk_file(&shorter, &layout, &openings),
            Err(FpError::Truncated { .. })
        ));
    }
}

#[cfg(test)]
pub(crate) use tests::accepted_input;
