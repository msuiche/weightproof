//! Request/proof-file planning (iop/req.h, frame/files.h).

use alloc::{string::String, vec, vec::Vec};

use crate::fp::{
    algebra::{
        plan_bat_sigma, plan_chain_start, plan_lookup, plan_member_rows, plan_rng, plan_send,
        LookupSize,
    },
    analysis::{analysis_of_m, GroupSize, Scope, Setting, Stage},
    codes::positions,
    error::FpError,
    key::{lin_first, lin_second, lin_shared, LinLayoutInput},
    layout::{ChainPlan, FilePlan, PlanMessage},
    plan::{nonce, plan_iopp, plan_open, scalars, TreeSize},
    subproof::Subproof,
};

/// One request segment and its solved public sizes.
#[derive(Debug, Clone)]
pub struct RequestLinkInput {
    /// Frame name (`G0`, `A0`, ...), used for SEND/FOLD/openings.
    pub name: String,
    /// Model planner name (`attention 4`, `head`, ...).
    pub subproof_name: String,
    pub subproof: Subproof,
    pub lookup: LookupSize,
    pub digits: LookupSize,
    pub initial_roots: usize,
    pub initial_sampled: u64,
    /// Trees opened at this segment's later positions, excluding `at_start`.
    pub initial_trees: Vec<TreeSize>,
    pub later_trees: [TreeSize; 2],
    pub plain: u64,
    pub fold: GroupSize,
}

#[derive(Debug, Clone)]
pub struct ProofLayoutInput {
    pub setting: Setting,
    pub links: Vec<RequestLinkInput>,
    pub last_stages: Vec<Stage>,
    pub last_tree: TreeSize,
    pub pending_rows: u64,
    pub batch_slots: u64,
    pub lin: LinLayoutInput,
}

fn loc(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    group: &GroupSize,
    trees: &[TreeSize],
) -> Result<(), FpError> {
    let t = positions(
        analysis_of_m(group).at.agree,
        setting.target - setting.query_work(Scope::Request),
        0.0,
        1,
    )
    .ok_or(FpError::Range {
        context: "loc-no-positions",
    })?;
    if let Some(nonce) = nonce(context, setting.query_work(Scope::Request)) {
        messages.push(nonce);
    }
    for tree in trees {
        plan_open(messages, context, tree, t)?;
    }
    Ok(())
}

fn fold(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    group: &GroupSize,
) -> Result<(), FpError> {
    messages.push(scalars("hiding sum", context, 1)?);
    if let Some(nonce) = nonce(context, setting.work_field[Scope::Request as usize]) {
        messages.push(nonce);
    }
    let coefficients = if setting.saved_coefficient { 2 } else { 3 };
    for _ in 1..=group.k {
        messages.push(scalars("round polynomial", context, coefficients)?);
        if let Some(nonce) = nonce(context, setting.work_closeness[Scope::Request as usize]) {
            messages.push(nonce);
        }
    }
    Ok(())
}

fn plan_link(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    link: &RequestLinkInput,
    previous: Option<(&str, &GroupSize, &[TreeSize])>,
    next_tree: &mut u32,
) -> Result<Vec<TreeSize>, FpError> {
    // Tree numbers are global over the request, in the order the trees are
    // committed: the leaf hash's domain tag and the root of each opening use
    // them. The input leaves them at zero.
    let mut initial_trees = link.initial_trees.clone();
    for tree in &mut initial_trees {
        tree.number = *next_tree;
        *next_tree += 1;
    }
    // The initial send commits further trees that the input does not model
    // (the fold mask): they take numbers as well.
    *next_tree += (link.initial_roots as u32).saturating_sub(initial_trees.len() as u32);
    let mut later_trees = link.later_trees.clone();
    for tree in &mut later_trees {
        tree.number = *next_tree;
        *next_tree += 1;
    }
    if link.plain != 0 {
        messages.push(scalars("plain scalars", &link.name, link.plain)?);
    }
    plan_send(
        messages,
        setting,
        Scope::Request,
        &link.name,
        link.initial_roots,
        link.initial_sampled,
    )?;
    if let Some((name, group, trees)) = previous {
        loc(messages, setting, name, group, trees)?;
    }
    plan_lookup(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{}, look-up", link.subproof_name),
        &link.lookup,
    )?;
    link.subproof
        .plan_own(messages, setting, &link.subproof_name)?;

    let mut trees = initial_trees;
    for moment in 0..2 {
        plan_send(messages, setting, Scope::Request, &link.name, 1, 1)?;
        trees.push(later_trees[moment].clone());
        if moment == 0 {
            plan_rng(
                messages,
                setting,
                Scope::Request,
                &alloc::format!("{}, smallness test", link.subproof_name),
            );
        } else {
            plan_lookup(
                messages,
                setting,
                Scope::Request,
                &alloc::format!("{}, smallness test, digits", link.subproof_name),
                &link.digits,
            )?;
            link.subproof
                .plan_end(messages, setting, &link.subproof_name)?;
        }
    }
    plan_bat_sigma(messages, setting, Scope::Request, &link.name);
    fold(messages, setting, &link.name, &link.fold)?;
    Ok(trees)
}

/// Plan all three chains of a request proof.
pub fn plan_proof(input: &ProofLayoutInput) -> Result<FilePlan, FpError> {
    if input.links.is_empty() {
        return Err(FpError::Range {
            context: "request-links",
        });
    }
    let mut first = vec![];
    plan_chain_start(&mut first, "first chain", true);
    let mut previous_trees = Vec::new();
    let mut next_tree = 0u32;
    for (index, link) in input.links.iter().enumerate() {
        previous_trees = if index == 0 {
            plan_link(&mut first, &input.setting, link, None, &mut next_tree)?
        } else {
            let previous = &input.links[index - 1];
            plan_link(
                &mut first,
                &input.setting,
                link,
                Some((&previous.name, &previous.fold, &previous_trees)),
                &mut next_tree,
            )?
        };
    }
    plan_send(
        &mut first,
        &input.setting,
        Scope::Request,
        "last group",
        1,
        1,
    )?;
    if let Some(previous) = input.links.last() {
        loc(
            &mut first,
            &input.setting,
            &previous.name,
            &previous.fold,
            &previous_trees,
        )?;
    }
    let last_tree = TreeSize {
        number: next_tree,
        root_context: Some("last group".into()),
        ..input.last_tree.clone()
    };
    plan_iopp(
        &mut first,
        &input.setting,
        Scope::Request,
        "last group",
        &input.last_stages,
        core::slice::from_ref(&last_tree),
        false,
        true,
    )?;
    plan_member_rows(
        &mut first,
        &input.setting,
        Scope::Request,
        "rows on the table",
        input.pending_rows,
        11,
    )?;
    lin_first(&mut first, &input.setting, &input.lin)?;

    let mut shared = vec![];
    plan_chain_start(&mut shared, "shared chain", false);
    lin_shared(&mut shared, &input.setting, &input.lin)?;
    let mut second = vec![];
    plan_chain_start(&mut second, "second chain", true);
    lin_second(&mut second, &input.setting, &input.lin, 23)?;
    FilePlan::proof(
        [
            ChainPlan::new("first chain", first),
            ChainPlan::new("shared chain", shared),
            ChainPlan::new("second chain", second),
        ],
        input.batch_slots,
    )
}

pub fn accepted_input() -> ProofLayoutInput {
    tests::accepted_input()
}

pub(crate) mod tests {
    use super::*;
    use crate::fp::{
        codes::CodeSize,
        subproof::{AttentionRounds, HeadRounds, LayerRounds},
    };

    fn stage(
        k: u32,
        h: u32,
        main: (u32, u64, u64),
        mask: (u32, u64, u64),
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
                    over_E: true,
                },
                mask: CodeSize {
                    log_n: mask.0,
                    ell: mask.1,
                    r: mask.2,
                    over_E: true,
                },
                m,
            },
            t,
            r_ood: 0,
        }
    }

    fn link_group(_name: &str, k: u32, slots: u64) -> GroupSize {
        GroupSize {
            k,
            h: 0,
            slots,
            main: CodeSize {
                log_n: 18,
                ell: 131072,
                r: 218,
                over_E: true,
            },
            mask: CodeSize {
                log_n: 17,
                ell: 8192,
                r: 218,
                over_E: true,
            },
            m: 517,
        }
    }

    fn tree(name: &str, leaf: u64) -> TreeSize {
        TreeSize::new(name, 18, leaf)
    }

    fn layer(first: bool) -> Subproof {
        Subproof::Layer(LayerRounds {
            lt: 6,
            ls: 2,
            l_ni: 3,
            l_pd: 9,
            ll: 2,
            l_pf: 10,
            l_fe: 9,
            lam_s: 15,
            lk_w: 11,
            ni: 8,
            ns: 9,
            np: 7,
            no: 4,
            with_terms: true,
            shown_by_the_terms: 18,
            never_sent: 3,
            weight_claims: 32,
            first_group: first,
            routed: true,
        })
    }

    fn attention() -> Subproof {
        Subproof::Attention(AttentionRounds {
            lt: 6,
            lgk: 1,
            pairs: 8,
            lam: 4,
            plane_messages: 2,
            score: vec![(11, 8), (10, 8)],
            e: 7,
            value_m: 8,
            value_d: 7,
            attained: 14,
            bits: 11,
            rows: 2074535,
        })
    }

    fn link(
        name: &str,
        subproof_name: &str,
        subproof: Subproof,
        lookup: LookupSize,
        digits: LookupSize,
        initial_roots: usize,
        initial_sampled: u64,
        first_leaf: u64,
        later_leaves: [u64; 2],
        plain: u64,
        k: u32,
        slots: u64,
    ) -> RequestLinkInput {
        let first_label: String = match name {
            "G0" => "tree of G0, moment 1, group 1; G0, moment 1, group 2; G0, moment 1, group 3; fold mask".into(),
            "A0" => "tree of carried; G0, moment 1, group 2; G0, moment 1, group 3; A0, moment 1, group 1; fold mask".into(),
            "G1" => "tree of carried; G0, moment 1, group 3; G1, moment 1, group 1; G1, moment 1, group 2; G1, moment 1, group 3; fold mask".into(),
            "A1" => "tree of carried; G1, moment 1, group 2; G1, moment 1, group 3; A1, moment 1, group 1; fold mask".into(),
            "H" => "tree of carried; G1, moment 1, group 3; H, moment 1, group 1; fold mask".into(),
            _ => "first".into(),
        };
        RequestLinkInput {
            name: name.into(),
            subproof_name: subproof_name.into(),
            subproof,
            lookup,
            digits,
            initial_roots,
            initial_sampled,
            initial_trees: vec![tree(&first_label, first_leaf)],
            later_trees: [
                tree(
                    &alloc::format!("tree of {name}, moment 2, group 1"),
                    later_leaves[0],
                ),
                tree(
                    &alloc::format!("tree of {name}, moment 3, group 1"),
                    later_leaves[1],
                ),
            ],
            plain,
            fold: link_group(name, k, slots),
        }
    }

    pub fn accepted_input() -> ProofLayoutInput {
        let g_lookup = LookupSize::new(19, 5, 0, vec![24]);
        let a_lookup = LookupSize::new(18, 5, 0, vec![22]);
        let h_lookup = LookupSize::new(19, 5, 0, vec![23]);
        let g_digits = LookupSize::new(11, 5, 0, vec![24]);
        let a_digits = LookupSize::new(11, 5, 0, vec![23]);
        ProofLayoutInput {
            setting: Setting::example(),
            links: vec![
                link(
                    "G0",
                    "G0",
                    layer(true),
                    g_lookup.clone(),
                    g_digits.clone(),
                    2,
                    4,
                    568,
                    [48, 1032],
                    0,
                    9,
                    8,
                ),
                link(
                    "A0",
                    "attention 4",
                    attention(),
                    a_lookup.clone(),
                    a_digits.clone(),
                    1,
                    4,
                    312,
                    [24, 488],
                    0,
                    7,
                    2,
                ),
                link(
                    "G1",
                    "G1",
                    layer(false),
                    g_lookup,
                    g_digits,
                    1,
                    5,
                    616,
                    [48, 1032],
                    0,
                    9,
                    8,
                ),
                link(
                    "A1",
                    "attention 5",
                    attention(),
                    a_lookup,
                    a_digits,
                    1,
                    4,
                    312,
                    [24, 488],
                    0,
                    7,
                    2,
                ),
                link(
                    "H",
                    "head",
                    Subproof::Head(HeadRounds {
                        lt: 6,
                        l_pd: 9,
                        lk_w: 11,
                        sums: 9,
                        products: 7,
                        row_values: 4,
                        rows: 2712833,
                    }),
                    h_lookup,
                    LookupSize::new(11, 5, 0, vec![24]),
                    1,
                    3,
                    376,
                    [24, 648],
                    1,
                    8,
                    1,
                ),
            ],
            last_stages: vec![
                stage(1, 0, (18, 131072, 218), (17, 8192, 218), 517, 218),
                stage(4, 3, (17, 8192, 55), (17, 4096, 55), 319, 55),
                stage(3, 3, (16, 1024, 44), (16, 2048, 44), 260, 44),
                stage(3, 3, (15, 128, 44), (15, 1024, 44), 300, 44),
                stage(0, 0, (15, 128, 73), (15, 4096, 73), 520, 73),
            ],
            last_tree: tree("tree of carried; fold mask", 72),
            pending_rows: 13,
            batch_slots: 64,
            lin: LinLayoutInput {
                folded_stages: vec![
                    stage(5, 0, (17, 32768, 101), (13, 64, 156), 483, 101),
                    stage(4, 2, (14, 2048, 68), (12, 256, 68), 600, 68),
                    stage(0, 0, (14, 2048, 68), (14, 1024, 68), 600, 68),
                ],
                omega_stages: vec![
                    stage(2, 0, (18, 16384, 3831), (13, 64, 156), 295, 55),
                    stage(4, 3, (14, 1024, 69), (13, 1024, 69), 527, 69),
                    stage(0, 0, (15, 1024, 68), (15, 4096, 68), 520, 68),
                ],
                own_stages: vec![
                    stage(3, 0, (18, 16384, 3831), (13, 256, 59), 295, 59),
                    stage(4, 3, (14, 1024, 75), (13, 1024, 75), 528, 75),
                    stage(0, 0, (15, 1024, 73), (15, 4096, 73), 520, 73),
                ],
                folded_trees: vec![TreeSize::new("tree of the folded row; fold mask", 17, 456)],
                omega_trees: vec![
                    tree("tree of Omega", 96)
                        .numbered(1)
                        .rooted("Omega, stage 1"),
                    TreeSize::new(
                        "tree of the mask of the split between Omega and the folded row",
                        13,
                        24,
                    )
                    .numbered(2),
                ],
                w_trees: vec![tree("tree of W; fold mask", 48)
                    .rooted("table, W")
                    .numbered(21)],
                input_trees: vec![TreeSize::new("tree of a request's masks", 13, 24)
                    .rooted("G0")
                    .numbered(1)],
                table_tree: TreeSize::new("table", 23, 2048).stored("table"),
                columns: 53,
                split: true,
                one_time: false,
                masks_ahead: true,
                masks_ride: true,
            },
        }
    }

    #[test]
    fn complete_proof_layout_matches_every_oracle_offset_and_length() {
        let plan = plan_proof(&accepted_input()).unwrap();
        let layout = plan.layout().unwrap();
        let expected: Vec<(u64, u64, String)> =
            include_str!("../../spikes/plans/request-fields.txt")
                .lines()
                .filter(|line| line.starts_with("field: proof "))
                .map(|line| {
                    let rest = line.strip_prefix("field: proof ").unwrap();
                    let (location, label) = rest.split_once(": ").unwrap();
                    let mut words = location.split_whitespace();
                    let _mode = words.next().unwrap();
                    let offset = words.next().unwrap().parse().unwrap();
                    let bytes = words.next().unwrap().parse().unwrap();
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
        assert_eq!(layout.fields().len(), 7772);
        assert_eq!(expected.len(), layout.fields().len());
        assert_eq!(layout.maximum_bytes(), 4279884);
    }

    #[test]
    fn stored_proof_is_consumed_exactly_with_oracle_positions() {
        use crate::fp::{
            error::FpError,
            walk::{walk_file, OpeningActual},
        };

        let proof = include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.proof")
            .as_slice();
        assert_eq!(proof.len(), 4145300);
        let plan = plan_proof(&accepted_input()).unwrap();
        let layout = plan.layout().unwrap();
        let lists: Vec<Vec<u64>> = include_str!("../../spikes/plans/proof-openings.txt")
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
        let mut contexts = Vec::new();
        for field in layout
            .fields()
            .iter()
            .filter(|field| field.message.position_dependent)
        {
            let context = field.message.context.as_str();
            if contexts.last().copied() != Some(context) {
                contexts.push(context);
            }
        }
        assert_eq!(contexts.len(), 35);
        assert_eq!(lists.len(), contexts.len());
        let openings: Vec<OpeningActual<'_>> = contexts
            .iter()
            .zip(&lists)
            .map(|(context, positions)| OpeningActual { context, positions })
            .collect();
        let walked = walk_file(proof, &layout, &openings).expect("stored proof consumes exactly");
        assert_eq!(walked.bytes(), proof.len() as u64);
        assert_eq!(walked.fields.len(), 7772);
        assert_eq!(walked.proof_entry, Some(0));

        let mut longer = proof.to_vec();
        longer.push(0);
        assert!(matches!(
            walk_file(&longer, &layout, &openings),
            Err(FpError::TrailingBytes { count: 1, .. })
        ));
        let mut shorter = proof.to_vec();
        shorter.pop();
        assert!(matches!(
            walk_file(&shorter, &layout, &openings),
            Err(FpError::Truncated { .. })
        ));
    }
}
