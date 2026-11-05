//! Executable LINK transcript and Merkle verification.
//!
//! This checks public-input encoding, every round message, proof-of-work,
//! positions, Merkle openings, and LINK's algebraic claims.

use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec, vec::Vec};
use core::ops::Range;

use crate::fp::{
    chain::{Chain, ChainKind},
    digest::Digest,
    error::FpError,
    field::{ExtensionElement, FieldElement},
    layout::{FileLayout, MessageKind},
    link_algebra::LinkAlgebra,
    link_public::{IndexName, LinkPublic},
    merkle::verify_opening,
    plan::EntryLayoutInput,
    walk::{actual_opening, walk_file, OpeningActual},
};

/// One verified opening's sorted positions and interleaved record range.
#[derive(Debug, Clone)]
pub struct OpeningData {
    pub positions: Vec<u64>,
    pub records: Range<usize>,
    pub leaf_bytes: usize,
}

impl OpeningData {
    fn record<'a>(&self, bytes: &'a [u8], position: u64) -> &'a [u8] {
        let index = self.positions.binary_search(&position).unwrap();
        let stride = self.leaf_bytes + 40;
        let start = self.records.start + index * stride;
        &bytes[start..start + self.leaf_bytes]
    }

    pub fn column_f(
        &self,
        bytes: &[u8],
        position: u64,
        start: usize,
        count: usize,
    ) -> Vec<FieldElement> {
        let column = self.record(bytes, position);
        column[start * 8..(start + count) * 8]
            .chunks_exact(8)
            .map(|word| FieldElement(u64::from_le_bytes(word.try_into().unwrap())))
            .collect()
    }

    pub fn column_e(
        &self,
        bytes: &[u8],
        position: u64,
        start: usize,
        count: usize,
    ) -> Vec<ExtensionElement> {
        self.column_f(bytes, position, start, count * 3)
            .chunks_exact(3)
            .map(|coordinates| {
                ExtensionElement::from_words([coordinates[0].0, coordinates[1].0, coordinates[2].0])
            })
            .collect()
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    layout: &'a FileLayout<'a>,
    configuration: Digest,
    index: usize,
    offset: usize,
    roots: BTreeMap<String, Digest>,
    openings: Vec<(String, Vec<u64>)>,
    opened: Vec<OpeningData>,
}

impl<'a> Cursor<'a> {
    fn new(
        bytes: &'a [u8],
        layout: &'a FileLayout<'a>,
        configuration: &Digest,
        anchor_root: &Digest,
    ) -> Self {
        let mut roots = BTreeMap::new();
        roots.insert("first table".into(), *anchor_root);
        Self {
            bytes,
            layout,
            configuration: *configuration,
            index: 0,
            offset: 0,
            roots,
            openings: vec![],
            opened: vec![],
        }
    }

    fn take(&mut self, name: &str, context: &str) -> Result<&'a [u8], FpError> {
        let field = self.layout.fields().get(self.index).ok_or(FpError::Range {
            context: "transcript-field-count",
        })?;
        if field.message.position_dependent
            || field.message.name != name
            || field.message.context != context
        {
            return Err(FpError::Range {
                context: "transcript-field-order",
            });
        }
        let end = self
            .offset
            .checked_add(field.message.bytes as usize)
            .ok_or(FpError::Range {
                context: "transcript-overflow",
            })?;
        if end > self.bytes.len() {
            return Err(FpError::Truncated {
                context: "transcript-field",
                needed: end,
                available: self.bytes.len(),
            });
        }
        let bytes = &self.bytes[self.offset..end];
        if matches!(
            field.message.kind,
            MessageKind::Scalars | MessageKind::Answers | MessageKind::Columns
        ) {
            for word in bytes.chunks_exact(8) {
                if u64::from_le_bytes(word.try_into().unwrap()) >= crate::fp::field::GOLDILOCKS_P {
                    return Err(FpError::NonCanonicalField);
                }
            }
        }
        self.index += 1;
        self.offset = end;
        Ok(bytes)
    }

    fn take_root(&mut self, context: &str) -> Result<Digest, FpError> {
        let bytes = self.take("root", context)?;
        let root = Digest::from_le_bytes(bytes.try_into().map_err(|_| FpError::BadLength {
            context: "transcript-root",
        })?);
        self.roots.insert(context.into(), root);
        Ok(root)
    }

    fn take_nonce(&mut self, context: &str) -> Result<u64, FpError> {
        let bytes = self.take("nonce", context)?;
        Ok(u64::from_le_bytes(bytes.try_into().map_err(|_| {
            FpError::BadLength {
                context: "transcript-nonce",
            }
        })?))
    }

    fn next_opening_context(&self) -> Option<&str> {
        self.layout.fields().get(self.index).and_then(|field| {
            if field.message.position_dependent {
                Some(field.message.context.as_str())
            } else {
                None
            }
        })
    }

    fn open_next(&mut self, positions: &[u64], run: &Digest) -> Result<(), FpError> {
        let context = self
            .next_opening_context()
            .ok_or(FpError::Range {
                context: "opening-context",
            })?
            .to_owned();
        let opening_start = self.offset;
        let first = self.layout.fields()[self.index]
            .message
            .opening
            .as_ref()
            .ok_or(FpError::Range {
                context: "opening-shape-missing",
            })?;
        let (distinct, path_digests) = actual_opening(first.log_n, positions, first.draws)?;
        let record_bytes = first.leaf_bytes as usize + 40;
        let opening_len = (distinct as usize)
            .checked_mul(record_bytes)
            .ok_or(FpError::Range {
                context: "opening-overflow",
            })?;
        let path_len = (path_digests as usize)
            .checked_mul(32)
            .ok_or(FpError::Range {
                context: "opening-overflow",
            })?;
        let path_start = opening_start
            .checked_add(opening_len)
            .ok_or(FpError::Range {
                context: "opening-overflow",
            })?;
        let end = path_start.checked_add(path_len).ok_or(FpError::Range {
            context: "opening-overflow",
        })?;
        if end > self.bytes.len() {
            return Err(FpError::Truncated {
                context: "opening-bytes",
                needed: end,
                available: self.bytes.len(),
            });
        }
        for field in &self.layout.fields()[self.index..self.index + 3] {
            if !field.message.position_dependent
                || field.message.context != context
                || field.message.opening.as_ref() != Some(first)
            {
                return Err(FpError::Range {
                    context: "opening-field-sequence",
                });
            }
        }
        if !matches!(
            self.layout.fields()[self.index].message.kind,
            MessageKind::Columns
        ) || !matches!(
            self.layout.fields()[self.index + 1].message.kind,
            MessageKind::Salts
        ) || !matches!(
            self.layout.fields()[self.index + 2].message.kind,
            MessageKind::Paths
        ) {
            return Err(FpError::Range {
                context: "opening-field-sequence",
            });
        }
        let root = self.roots.get(&first.root_context).ok_or(FpError::Range {
            context: "opening-root-missing",
        })?;
        let leaf_start = if first.stored {
            &self.configuration
        } else {
            run
        };
        verify_opening(
            root,
            first,
            leaf_start,
            &self.configuration,
            positions,
            &self.bytes[opening_start..path_start],
            &self.bytes[path_start..end],
        )?;
        self.offset = end;
        self.index += 3;
        let mut sorted = positions.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        self.opened.push(OpeningData {
            positions: sorted,
            records: opening_start..path_start,
            leaf_bytes: first.leaf_bytes as usize,
        });
        self.openings.push((context, positions.to_vec()));
        Ok(())
    }

    fn open_stage(&mut self, positions: &[u64], start: &Digest) -> Result<(), FpError> {
        while self.next_opening_context().is_some() {
            self.open_next(positions, start)?;
        }
        Ok(())
    }
}

fn words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn elements(bytes: &[u8]) -> Vec<ExtensionElement> {
    bytes
        .chunks_exact(24)
        .map(|element| {
            let mut coordinates = [0u64; 3];
            for (i, word) in element.chunks_exact(8).enumerate() {
                coordinates[i] = u64::from_le_bytes(word.try_into().unwrap());
            }
            ExtensionElement::from_words(coordinates)
        })
        .collect()
}

fn one_element(bytes: &[u8]) -> ExtensionElement {
    let values = elements(bytes);
    assert_eq!(values.len(), 1);
    values[0]
}

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
