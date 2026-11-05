//! Configuration digest, table names, record seeds, and the commitment ID
//! (frame/heads.h, foldeq.h, rounds.h).

use alloc::vec::Vec;

use crate::fp::{
    blake2::{Hasher, PARAMETERS, PUBLISHED_ID},
    chain::{Chain, ChainKind, LABEL_RECORD},
    digest::Digest,
    error::FpError,
    version::Version,
};

/// Digest of every byte of the public configuration files, in a fixed order
/// (parameter file first): H(PARAMETERS, 2; count, then per file length+bytes).
pub fn configuration_digest(files: &[&[u8]]) -> Digest {
    let mut h = Hasher::new(PARAMETERS, 2, &[0; 8], 32);
    h.push_u64(files.len() as u64);
    for file in files {
        h.push_u64(file.len() as u64);
        h.push(file);
    }
    h.finish()
}

/// A table's name: root (32) ‖ configuration digest (32) ‖ shape (6 × u64 LE):
/// `(code.log_n, code.ell, code.r, v, answers, salt_bytes)` (iop/foldeq.h
/// `index_name`, NOT the raw `table` parameter line's order).
/// 112 bytes; no digest of these bytes (frame/FORMAT.md §4).
pub fn table_name(root: &Digest, configuration: &Digest, shape: [u64; 6]) -> [u8; 112] {
    let mut name = [0u8; 112];
    name[..32].copy_from_slice(&root.to_le_bytes());
    name[32..64].copy_from_slice(&configuration.to_le_bytes());
    for (i, x) in shape.iter().enumerate() {
        name[64 + i * 8..64 + i * 8 + 8].copy_from_slice(&x.to_le_bytes());
    }
    name
}

/// The seed of an out-of-domain sample: point z in E^variables and factor beta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleSeed {
    pub z: Vec<[u64; 3]>,
    pub beta: [u64; 3],
}

/// The record's seeds: a function of the commitment (the table's name) alone,
/// derived through the LABEL_RECORD chain with one SEED round (rounds.h).
pub fn record_seeds(commitment: &[u8], variables: u32, answers: u32) -> Vec<SampleSeed> {
    let q = crate::fp::chain::FirstQuery {
        label: LABEL_RECORD,
        parameters: Digest::ZERO,
        statement: commitment.to_vec(),
        record: Vec::new(),
        salted: false,
        run: None,
    };
    let mut chain = Chain::new(q);
    chain.close(ChainKind::Seed, &[]);
    (0..answers)
        .map(|_| SampleSeed {
            z: (0..variables).map(|_| chain.draw_e()).collect(),
            beta: chain.draw_e_nonzero(),
        })
        .collect()
}

/// The commitment ID (heads.h `published_id`):
/// H(PUBLISHED_ID, 0; configuration, name length+name, seed elements,
/// seed z coordinates and beta, answers length+answers, empty list).
pub fn published_id(
    configuration: &Digest,
    first_table_name: &[u8],
    record_seeds: &[SampleSeed],
    record_answers: &[[u64; 3]],
) -> Digest {
    let mut h = Hasher::new(PUBLISHED_ID, 0, &[0; 8], 32);
    h.push_digest(configuration);
    h.push_u64(first_table_name.len() as u64);
    h.push(first_table_name);
    let elements: u64 = record_seeds.iter().map(|s| s.z.len() as u64 + 1).sum();
    h.push_u64(elements);
    for seed in record_seeds {
        for x in &seed.z {
            h.push_e(x);
        }
        h.push_e(&seed.beta);
    }
    h.push_u64(record_answers.len() as u64);
    for y in record_answers {
        h.push_e(y);
    }
    h.push_u64(0);
    h.finish()
}

/// The key file's table root and stored record answers, plus the rest of the
/// file (the one-time, shared and second chains, planned in later increments).
/// Layout: head (tag `CPIK`, version) ‖ root (32) ‖ expected_answers × 24 bytes
/// ‖ remaining chains. Extra bytes are NOT an error: they are those chains.
pub fn parse_key_record(
    bytes: &[u8],
    expected_answers: usize,
) -> Result<(Digest, Vec<[u64; 3]>, &[u8]), FpError> {
    parse_table_record(bytes, b"CPIK", "CPIK", expected_answers)
}

/// The leaf file's table root and stored record answers.
pub fn parse_leaf_record(
    bytes: &[u8],
    expected_answers: usize,
) -> Result<(Digest, Vec<[u64; 3]>, &[u8]), FpError> {
    parse_table_record(bytes, b"CPEN", "CPEN", expected_answers)
}

fn parse_table_record<'a>(
    bytes: &'a [u8],
    tag: &[u8; 4],
    tag_name: &'static str,
    expected_answers: usize,
) -> Result<(Digest, Vec<[u64; 3]>, &'a [u8]), FpError> {
    if bytes.len() < 40 {
        return Err(FpError::Truncated {
            context: "key-head",
            needed: 40,
            available: bytes.len(),
        });
    }
    if &bytes[0..4] != tag {
        return Err(FpError::BadTag { expected: tag_name });
    }
    Version::parse(&bytes[4..8])?;
    let root = Digest::from_le_bytes(bytes[8..40].try_into().map_err(|_| FpError::Truncated {
        context: "key-head",
        needed: 40,
        available: bytes.len(),
    })?);
    let body = &bytes[40..];
    let needed = expected_answers * 24;
    if body.len() < needed {
        return Err(FpError::Truncated {
            context: "key-record",
            needed,
            available: body.len(),
        });
    }
    let answers = body[..needed]
        .chunks_exact(24)
        .map(|chunk| {
            let mut e = [0u64; 3];
            for (i, part) in chunk.chunks_exact(8).enumerate() {
                let mut le = [0u8; 8];
                le.copy_from_slice(part);
                e[i] = u64::from_le_bytes(le);
            }
            e
        })
        .collect();
    Ok((root, answers, &body[needed..]))
}
