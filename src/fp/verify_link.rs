//! Executable LINK transcript and Merkle verification.
//!
//! This checks public-input encoding, every round message, proof-of-work,
//! positions, Merkle openings, and LINK's algebraic claims.

use alloc::vec::Vec;

use crate::fp::{
    chain::{Chain, ChainKind},
    digest::Digest,
    error::FpError,
    field::ExtensionElement,
    link_algebra::LinkAlgebra,
    link_public::{IndexName, LinkPublic},
    plan::EntryLayoutInput,
    transcript::{elements, one_element, words, Cursor},
    walk::{walk_file, OpeningActual},
};

/// Result of transcript, Merkle, and algebraic LINK verification.
#[derive(Debug, Clone)]
pub struct LinkTranscriptEvidence {
    pub fields: usize,
    pub openings: usize,
    pub rho: Digest,
}

/// Execute the stored leaf's LINK transcript and verify every Merkle opening.
pub fn verify_link_transcript(
    input: &EntryLayoutInput,
    configuration: &Digest,
    first: &IndexName,
    further: &IndexName,
    bytes: &[u8],
) -> Result<LinkTranscriptEvidence, FpError> {
    let plan = crate::fp::plan::plan_entry(input)?;
    let layout = plan.layout()?;
    let public = LinkPublic::new(input, configuration, first, further)?;
    let mut cursor = Cursor::new(bytes, &layout, configuration, &first.commitment);
    cursor.take("head", "")?;
    cursor.take_root("further table")?;
    cursor.take("record", "further table")?;
    let salt = Digest::from_le_bytes(
        cursor
            .take("chain salt", "link's chain")?
            .try_into()
            .map_err(|_| FpError::BadLength {
                context: "link-salt",
            })?,
    );
    let run = public.first_query(&salt).run.unwrap();
    let mut chain = Chain::new(public.first_query(&salt));
    let mut algebra = LinkAlgebra::new(input, configuration);

    chain.close(ChainKind::Algebraic, &[]);
    let gamma: Vec<ExtensionElement> = (0..8)
        .map(|_| ExtensionElement::from_words(chain.draw_e()))
        .collect();
    let root = cursor.take_root("link")?;
    chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
    let seed_z: Vec<ExtensionElement> = (0..20)
        .map(|_| ExtensionElement::from_words(chain.draw_e()))
        .collect();
    let beta = ExtensionElement::from_words(chain.draw_e_nonzero());
    let answers = cursor.take("sample answers", "link")?;
    let sample_answers = elements(answers);
    chain.close(ChainKind::Empty, &words(answers));
    let nonce = cursor.take_nonce("link")?;
    if !chain.close_with_work(ChainKind::Positions, &[], 32, nonce) {
        return Err(FpError::Range {
            context: "proof-of-work-link",
        });
    }
    let positions: Vec<u64> = (0..input.columns_first + input.columns_further)
        .map(|_| chain.draw_position(input.table_tree.log_n))
        .collect();
    chain.close(ChainKind::Algebraic, &[]);
    let phi: Vec<ExtensionElement> = (0..19)
        .map(|_| ExtensionElement::from_words(chain.draw_e()))
        .collect();
    chain.close(ChainKind::Algebraic, &[]);
    let batch = ExtensionElement::from_words(chain.draw_e());
    cursor.open_next(&positions[..input.columns_first as usize], &run)?;
    cursor.open_next(&positions[input.columns_first as usize..], &run)?;
    let mut row = algebra.initial(
        first,
        further,
        &gamma,
        &seed_z,
        beta,
        &sample_answers,
        &phi,
        batch,
        &cursor.opened,
        &positions[..input.columns_first as usize],
        &positions[input.columns_first as usize..],
        bytes,
    )?;

    let stages = crate::fp::plan::solve_entry_stages(input)?;
    for (index, stage) in stages.iter().enumerate() {
        let context = if index + 1 == stages.len() {
            "the two folded rows, last stage".into()
        } else {
            alloc::format!("the two folded rows, stage {}", index + 1)
        };
        if index + 1 < stages.len() {
            let hiding = cursor.take("hiding sum", &context)?;
            let hiding_sum = one_element(hiding);
            chain.close(ChainKind::Epsilon, &words(hiding));
            let epsilon = ExtensionElement::from_words(chain.draw_e());
            let mut polynomials = Vec::with_capacity(stage.g.k as usize);
            let mut challenges = Vec::with_capacity(stage.g.k as usize);
            for _ in 1..=stage.g.k {
                let polynomial = cursor.take("round polynomial", &context)?;
                let coefficients = elements(polynomial);
                polynomials.push((coefficients[0], coefficients[1]));
                chain.close(ChainKind::Algebraic, &words(polynomial));
                challenges.push(ExtensionElement::from_words(chain.draw_e()));
            }
            let root = cursor.take_root(&context)?;
            chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
            let cube = crate::fp::analysis::stage_cube_bits(&stages[index + 1].g) as usize;
            let z: Vec<ExtensionElement> = (0..cube)
                .map(|_| ExtensionElement::from_words(chain.draw_e()))
                .collect();
            let beta = ExtensionElement::from_words(chain.draw_e_nonzero());
            let answers = cursor.take("sample answers", &context)?;
            let answer = one_element(answers);
            let nonce = cursor.take_nonce(&context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(answers), 32, nonce) {
                return Err(FpError::Range {
                    context: "proof-of-work-cs",
                });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            let theta = ExtensionElement::from_words(chain.draw_e());
            cursor.open_stage(&positions, &run)?;
            row = algebra.fold_cs(
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
                &cursor.opened[index + 2],
                bytes,
            )?;
        } else {
            let target = cursor.take("masked target", &context)?;
            let target_star = one_element(target);
            chain.close(ChainKind::Algebraic, &words(target));
            let lambda = ExtensionElement::from_words(chain.draw_e());
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
            algebra.base(
                row,
                target_star,
                lambda,
                &clear_values,
                &positions,
                &cursor.opened[index + 2],
                bytes,
            )?;
            break;
        }
    }
    if cursor.index != layout.fields().len() {
        return Err(FpError::Range {
            context: "transcript-unconsumed-fields",
        });
    }
    let openings: Vec<OpeningActual<'_>> = cursor
        .openings
        .iter()
        .map(|(context, positions)| OpeningActual { context, positions })
        .collect();
    let walked = walk_file(bytes, &layout, &openings)?;
    Ok(LinkTranscriptEvidence {
        fields: walked.fields.len(),
        openings: cursor.openings.len(),
        rho: chain.value(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{
        heads::{configuration_digest, parse_key_record, parse_leaf_record},
        plan::accepted_example_entry_input,
    };

    fn stored() -> (Digest, IndexName, IndexName, Vec<u8>) {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let key = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/key.bin");
        let leaf = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin");
        let configuration = configuration_digest(&[policy, layout, kl]);
        let (first_root, first_record, _) = parse_key_record(key, 256).unwrap();
        let (further_root, further_record, _) = parse_leaf_record(leaf, 256).unwrap();
        (
            configuration,
            IndexName::new(&first_root, &configuration, first_record),
            IndexName::new(&further_root, &configuration, further_record),
            leaf.to_vec(),
        )
    }

    fn mutate_named(leaf: &mut [u8], name: &str, occurrence: usize) {
        let input = accepted_example_entry_input();
        let plan = crate::fp::plan::plan_entry(&input).unwrap();
        let layout = plan.layout().unwrap();
        let field = layout
            .fields()
            .iter()
            .filter(|field| field.message.name == name)
            .nth(occurrence)
            .unwrap();
        assert!(!field.message.position_dependent);
        leaf[field.offset as usize] ^= 1;
    }

    #[test]
    fn sample_answer_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        mutate_named(&mut leaf, "sample answers", 0);
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn round_polynomial_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        mutate_named(&mut leaf, "round polynomial", 0);
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn masked_message_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        mutate_named(&mut leaf, "masked message and randomness", 0);
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn stored_leaf_transcript_and_merkle_openings_verify() {
        let (configuration, first, further, leaf) = stored();
        let evidence = verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf,
        )
        .unwrap();
        assert_eq!(evidence.fields, 43);
        assert_eq!(evidence.openings, 5);
        assert_eq!(
            evidence.rho.to_hex(),
            "a5fc707d0714114c4440eddb25390d868522a819cbd3ebfe845cf5e9d4416fc3"
        );
    }

    #[test]
    fn one_byte_root_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        leaf[6216] ^= 1;
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn one_byte_path_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        leaf[112792] ^= 1;
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn noncanonical_table_record_is_rejected() {
        let (configuration, first, mut further, leaf) = stored();
        further.record[0][0] = crate::fp::field::GOLDILOCKS_P;
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }

    #[test]
    fn one_byte_nonce_mutation_is_rejected() {
        let (configuration, first, further, mut leaf) = stored();
        leaf[6296] ^= 1;
        assert!(verify_link_transcript(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
            &leaf
        )
        .is_err());
    }
}
