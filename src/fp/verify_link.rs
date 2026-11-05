//! Executable LINK transcript and Merkle verification.
//!
//! This checks public-input encoding, every round message, proof-of-work,
//! positions and Merkle openings. It does not yet evaluate LINK's algebraic
//! claims, so a successful result is not a full proof-verification verdict.

use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec, vec::Vec};

use crate::fp::{
    chain::{Chain, ChainKind},
    digest::Digest,
    error::FpError,
    layout::{FileLayout, MessageKind},
    link_public::{IndexName, LinkPublic},
    merkle::verify_opening,
    plan::EntryLayoutInput,
    walk::{actual_opening, walk_file, OpeningActual},
};

struct Cursor<'a> {
    bytes: &'a [u8],
    layout: &'a FileLayout<'a>,
    configuration: Digest,
    index: usize,
    offset: usize,
    roots: BTreeMap<String, Digest>,
    openings: Vec<(String, Vec<u64>)>,
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

/// Result of transcript and Merkle checks, not of LINK's algebraic claims.
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

    chain.close(ChainKind::Algebraic, &[]);
    let _gamma = chain.draw_e();
    let root = cursor.take_root("link")?;
    chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
    let _seed_z: Vec<[u64; 3]> = (0..20).map(|_| chain.draw_e()).collect();
    let _beta = chain.draw_e_nonzero();
    let answers = cursor.take("sample answers", "link")?;
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
    let _phi: Vec<[u64; 3]> = (0..19).map(|_| chain.draw_e()).collect();
    chain.close(ChainKind::Algebraic, &[]);
    let _batch = chain.draw_e();
    cursor.open_next(&positions[..input.columns_first as usize], &run)?;
    cursor.open_next(&positions[input.columns_first as usize..], &run)?;

    let stages = crate::fp::plan::solve_entry_stages(input)?;
    for (index, stage) in stages.iter().enumerate() {
        let context = if index + 1 == stages.len() {
            "the two folded rows, last stage".into()
        } else {
            alloc::format!("the two folded rows, stage {}", index + 1)
        };
        if index + 1 < stages.len() {
            let hiding = cursor.take("hiding sum", &context)?;
            chain.close(ChainKind::Epsilon, &words(hiding));
            let _epsilon = chain.draw_e();
            for _ in 1..=stage.g.k {
                let polynomial = cursor.take("round polynomial", &context)?;
                chain.close(ChainKind::Algebraic, &words(polynomial));
                let _round = chain.draw_e();
            }
            let root = cursor.take_root(&context)?;
            chain.close(ChainKind::Seed, &words(&root.to_le_bytes()));
            let cube = crate::fp::analysis::stage_cube_bits(&stages[index + 1].g) as usize;
            let _z: Vec<[u64; 3]> = (0..=cube).map(|_| chain.draw_e()).collect();
            let _beta = chain.draw_e_nonzero();
            let answers = cursor.take("sample answers", &context)?;
            let nonce = cursor.take_nonce(&context)?;
            if !chain.close_with_work(ChainKind::Positions, &words(answers), 32, nonce) {
                return Err(FpError::Range {
                    context: "proof-of-work-cs",
                });
            }
            let positions: Vec<u64> = (0..stage.t)
                .map(|_| chain.draw_position(stage.g.main.log_n))
                .collect();
            let _batch = chain.draw_e();
            cursor.open_stage(&positions, &run)?;
        } else {
            let target = cursor.take("masked target", &context)?;
            chain.close(ChainKind::Algebraic, &words(target));
            let _lambda = chain.draw_e();
            let clear = cursor.take("masked message and randomness", &context)?;
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
