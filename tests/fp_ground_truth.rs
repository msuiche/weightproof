//! Ground-truth tests against the stored FlashProve example (D8/D11).
//! Files live in `spikes/flashprove` (gitignored clone). If the spike is not
//! checked out (e.g. CI without it), these tests skip with a note — unit
//! tests in `src/fp/` always run.

use std::{fs, path::PathBuf};

use weightproof::fp::{NextToken, Policy, ProofHead, RequestFile};

fn spike(path: &str) -> Option<Vec<u8>> {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("spikes/flashprove")
        .join(path);
    match fs::read(&full) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!("skipping ground-truth test: {} not found (clone anthropics/FlashProve into spikes/)", full.display());
            None
        }
    }
}

#[test]
fn stored_request_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/example.req") else {
        return;
    };
    assert_eq!(bytes.len(), 264, "T + 2 = 66 words of 4 bytes");
    let req = RequestFile::parse(&bytes).expect("stored request parses");
    assert_eq!(req.t(), 64);
    assert_eq!(req.r, 60);
    assert_eq!(req.e, 64);
    assert_eq!(
        req.checked_indices(),
        61..=63,
        "oracle prints tokens=61..63"
    );
    assert_eq!(req.checked_tokens().len(), 3);
    assert_eq!(req.tokens[0], 243, "first word from xxd: 0xf3");
}

#[test]
fn stored_next_token_is_199() {
    let Some(bytes) = spike("examples/accepted-proof/example.y") else {
        return;
    };
    let y = NextToken::parse(&bytes).expect("stored y parses");
    assert_eq!(y.0, 199, "example README: token 64 = 199");
}

#[test]
fn stored_policy_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/config/policy.txt") else {
        return;
    };
    let policy = Policy::parse(&bytes).expect("stored policy parses");
    assert!(policy.h821());
    assert_eq!(policy.sec(), Some(128));
    assert_eq!(policy.salt_len(), Some(40));
    assert!(policy.record(), "oracle prints record=1");
    assert_eq!(policy.get("listed"), Some(&[0][..]));
    assert_eq!(policy.get("answers"), Some(&[1][..]));
    assert_eq!(
        policy.get_named("group", "link"),
        Some(&[1, 18, 17, 17, 13][..]),
        "first group line"
    );
    assert_eq!(
        policy.calls.len(),
        18,
        "8 call + 4 callA + 6 callH lines in the stored file"
    );
    assert_eq!(policy.calls[0].name, "R16a");
    assert_eq!(policy.calls[0].target, "R16");
    assert_eq!(policy.calls[0].symbols, ["G", "U"]);
}

#[test]
fn stored_proof_head_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/example.proof") else {
        return;
    };
    assert_eq!(bytes.len(), 4_145_300);
    let (head, rest) = ProofHead::parse(&bytes).expect("stored proof head parses");
    assert_eq!(head.entry, 0, "third word is zero in the stored file");
    assert_eq!(rest.len(), bytes.len() - 12);
}

#[test]
fn tampered_stored_files_fail_closed() {
    // Tampering the stored request's length must fail parsing, not "verify".
    if let Some(mut bytes) = spike("examples/accepted-proof/example.req") {
        bytes.truncate(bytes.len() - 1);
        assert!(
            RequestFile::parse(&bytes).is_err(),
            "partial word must fail"
        );
    }
    // Flipping the policy's `listed 0` to `listed 1` must be refused.
    if let Some(bytes) = spike("examples/accepted-proof/config/policy.txt") {
        let text = String::from_utf8(bytes).unwrap();
        let tampered = text.replacen("listed 0", "listed 1", 1);
        assert!(
            Policy::parse(tampered.as_bytes()).is_err(),
            "listed != 0 must fail"
        );
        // And a version-word flip must be refused as version-unknown.
        let tampered = text.replacen("version H821", "version H822", 1);
        assert!(
            Policy::parse(tampered.as_bytes()).is_err(),
            "unknown version must fail"
        );
    }
    // A version flip inside the real proof's head must be refused.
    if let Some(mut bytes) = spike("examples/accepted-proof/example.proof") {
        bytes[7] = b'2';
        assert!(ProofHead::parse(&bytes).is_err(), "H822 must fail");
    }
}

#[test]
fn configuration_and_commitment_id_match_oracle() {
    use weightproof::fp::heads::{
        configuration_digest, parse_key_record, published_id, record_seeds, table_name,
    };

    let (Some(policy), Some(kl), Some(layout), Some(key)) = (
        spike("examples/accepted-proof/config/policy.txt"),
        spike("examples/accepted-proof/config/KL.bin"),
        spike("examples/accepted-proof/config/layout.txt"),
        spike("examples/accepted-proof/ivk/key.bin"),
    ) else {
        return;
    };
    let config = configuration_digest(&[&policy, &layout, &kl]);
    assert_eq!(
        config.to_words_hex(),
        "0adb76eef05acef273d2b9a9adc0aa985a2cfcfa64b52f4cbca9b63b565a53b2",
        "h_par from cp_verify -plan id"
    );

    // index_name's shape: (code.log_n, code.ell, code.r, v, answers, salt_bytes).
    let shape = [23, 524288, 65535, 8, 1, 40];
    let (root, answers, _rest) = parse_key_record(&key, 256).expect("key record: 256 E answers");
    let name = table_name(&root, &config, shape);
    assert_eq!(name.len(), 112);

    let seeds = record_seeds(&name, 19, 1);
    assert_eq!(seeds.len(), 1);
    assert_eq!(seeds[0].z.len(), 19);
    assert_ne!(seeds[0].beta, [0, 0, 0]);

    let id = published_id(&config, &name, &seeds, &answers);
    assert_eq!(
        id.to_hex(),
        "79a5a2bc7a469b40df73dedf40abeb740671e8ae8077738bd3743897aeb22183",
        "commitment ID from ID.txt and cp_verify -plan id"
    );
}

#[test]
fn stored_leaf_is_consumed_exactly_with_oracle_positions() {
    use weightproof::fp::{
        analysis::{GroupSize, Setting, StageChoice},
        codes::CodeSize,
        layout::MessageKind,
        plan::{plan_entry, EntryLayoutInput, TreeSize},
        walk::{walk_file, OpeningActual},
        FpError,
    };

    let position_fixture =
        fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spikes/plans/leaf-openings.txt"))
            .ok();
    let (Some(leaf), Some(position_text)) = (
        spike("examples/accepted-proof/ivk/leaf/0.bin"),
        position_fixture,
    ) else {
        return;
    };
    assert_eq!(leaf.len(), 536952);
    let input = EntryLayoutInput {
        setting: Setting::example(),
        table_answers: 256,
        table_tree: TreeSize::new("table", 23, 2048),
        columns_first: 51,
        columns_further: 51,
        folded_message_rows: 18,
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
    };
    let plan = plan_entry(&input).expect("public LINK sizes produce the oracle plan");
    let layout = plan.layout().unwrap();
    assert_eq!(layout.maximum_bytes(), 546112);

    let position_text = String::from_utf8(position_text).unwrap();
    let lists: Vec<Vec<u64>> = position_text
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
        "link, first table, table",
        "link, further table, table",
        "the two folded rows, stage 1, tree of the first table's folded row; the further table's folded row; fold mask",
        "the two folded rows, stage 2, switch oracle",
        "the two folded rows, last stage, switch oracle",
    ];
    let openings: Vec<OpeningActual<'_>> = contexts
        .iter()
        .zip(&lists)
        .map(|(context, positions)| OpeningActual { context, positions })
        .collect();
    let walked = walk_file(&leaf, &layout, &openings).expect("stored leaf consumes exactly");
    assert_eq!(walked.bytes(), leaf.len() as u64);
    assert_eq!(walked.fields.len(), 43);
    assert!(
        walked
            .fields
            .iter()
            .any(|field| field.message.kind == MessageKind::Paths
                && field.bytes < field.message.bytes)
    );

    let mut longer = leaf.clone();
    longer.push(0);
    assert!(matches!(
        walk_file(&longer, &layout, &openings),
        Err(FpError::TrailingBytes { count: 1, .. })
    ));
    let mut shorter = leaf;
    shorter.pop();
    assert!(matches!(
        walk_file(&shorter, &layout, &openings),
        Err(FpError::Truncated { .. })
    ));
}
