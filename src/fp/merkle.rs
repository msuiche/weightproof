//! Merkle leaf and opening verification (tree/opening.h, tree/tree.h).

use alloc::vec::Vec;

use crate::fp::{
    blake2::Hasher, digest::Digest, error::FpError, field::GOLDILOCKS_P, layout::OpeningSize,
};

const LEAF_MAIN: u64 = 164;
const LEAF_STORED_MAIN: u64 = 174;
const TREE_NODE: u64 = 166;
const SALT_BYTES: usize = 40;

fn leaf_salt_at(symbols: u64) -> u64 {
    let before = 16 + 8 * symbols;
    if before % 128 + SALT_BYTES as u64 <= 128 {
        before
    } else {
        (before / 128 + 1) * 128
    }
}

/// Hash one column and its salt under the tree's domain, number and start.
pub fn hash_leaf(
    tree: &OpeningSize,
    start: &Digest,
    position: u64,
    column: &[u8],
    salt: &[u8],
) -> Result<Digest, FpError> {
    if column.len() as u64 != tree.leaf_bytes || column.len() % 8 != 0 || salt.len() != SALT_BYTES {
        return Err(FpError::BadLength {
            context: "merkle-leaf",
        });
    }
    let symbols = tree.leaf_bytes / 8;
    let domain = if tree.stored {
        LEAF_STORED_MAIN
    } else {
        LEAF_MAIN
    };
    let tag = domain
        | ((tree.tree_number as u64) << 16)
        | (symbols << 32)
        | ((SALT_BYTES as u64 / 8) << 56);
    let index = (1u64 << tree.log_n) + position;
    let mut h = Hasher::new(tag, index, &start.0, 32);
    h.push(column);
    for _ in (16 + 8 * symbols)..leaf_salt_at(symbols) {
        h.push(&[0]);
    }
    h.push(salt);
    Ok(h.finish())
}

/// Hash two children under TREE_NODE with the configuration start.
pub fn hash_vertex(start: &Digest, left: &Digest, right: &Digest) -> Digest {
    let mut h = Hasher::new(TREE_NODE, 0, &start.0, 32);
    h.push_digest(left);
    h.push_digest(right);
    h.finish()
}

/// Verify interleaved `(column, salt)` records and the pruned sibling path.
/// Leaves use `leaf_start`; inner vertices always use the configuration digest.
pub fn verify_opening(
    root: &Digest,
    tree: &OpeningSize,
    leaf_start: &Digest,
    configuration: &Digest,
    positions: &[u64],
    opening: &[u8],
    paths: &[u8],
) -> Result<(), FpError> {
    if positions.is_empty() || positions.len() as u64 != tree.draws {
        return Err(FpError::Range {
            context: "opening-draw-count",
        });
    }
    let mut sorted = positions.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let record_bytes = tree.leaf_bytes as usize + SALT_BYTES;
    if sorted.len().checked_mul(record_bytes) != Some(opening.len()) || paths.len() % 32 != 0 {
        return Err(FpError::BadLength {
            context: "opening-parts",
        });
    }
    if sorted[0] >= (1u64 << tree.log_n)
        || sorted
            .windows(2)
            .any(|pair| pair[1] >= (1u64 << tree.log_n))
    {
        return Err(FpError::Range {
            context: "opening-position-range",
        });
    }

    let mut now = Vec::with_capacity(sorted.len());
    for (index, position) in sorted.iter().copied().enumerate() {
        let record = &opening[index * record_bytes..(index + 1) * record_bytes];
        let column = &record[..tree.leaf_bytes as usize];
        for word in column.chunks_exact(8) {
            if u64::from_le_bytes(word.try_into().unwrap()) >= GOLDILOCKS_P {
                return Err(FpError::NonCanonicalField);
            }
        }
        let salt = &record[tree.leaf_bytes as usize..];
        now.push((
            position,
            hash_leaf(tree, leaf_start, position, column, salt)?,
        ));
    }
    let mut path_index = 0usize;
    for _depth in (1..=tree.log_n).rev() {
        let mut next = Vec::new();
        let mut index = 0usize;
        while index < now.len() {
            let (place, value) = now[index];
            let parent = place >> 1;
            if place & 1 == 0 && index + 1 < now.len() && now[index + 1].0 == place + 1 {
                next.push((
                    parent,
                    hash_vertex(configuration, &value, &now[index + 1].1),
                ));
                index += 2;
            } else {
                let sibling = path_index;
                path_index += 1;
                if sibling >= paths.len() / 32 {
                    return Err(FpError::Truncated {
                        context: "merkle-path",
                        needed: (sibling + 1) * 32,
                        available: paths.len(),
                    });
                }
                let other = Digest::from_le_bytes(
                    paths[sibling * 32..(sibling + 1) * 32].try_into().unwrap(),
                );
                let node = if place & 1 == 1 {
                    hash_vertex(configuration, &other, &value)
                } else {
                    hash_vertex(configuration, &value, &other)
                };
                next.push((parent, node));
                index += 1;
            }
        }
        now = next;
    }
    if path_index * 32 != paths.len() {
        return Err(FpError::TrailingBytes {
            context: "merkle-path",
            count: paths.len() - path_index * 32,
        });
    }
    if now.len() != 1 || now[0].1 != *root {
        return Err(FpError::Range {
            context: "merkle-root",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_and_vertex_hash_are_deterministic() {
        let tree = OpeningSize {
            log_n: 3,
            leaf_bytes: 16,
            draws: 1,
            root_context: "test".into(),
            tree_number: 0,
            stored: false,
        };
        let start = Digest([1; 8]);
        let root = hash_leaf(&tree, &start, 0, &[2; 16], &[3; 40]).unwrap();
        let other = hash_leaf(&tree, &start, 1, &[2; 16], &[3; 40]).unwrap();
        let parent = hash_vertex(&start, &root, &other);
        assert_ne!(root, other);
        assert_ne!(parent, root);
    }

    #[test]
    fn simple_vertex_matches_cpp() {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let configuration = crate::fp::heads::configuration_digest(&[policy, layout, kl]);
        let left = Digest([1, 0, 0, 0, 0, 0, 0, 0]);
        let right = Digest([2, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            hash_vertex(&configuration, &left, &right).to_hex(),
            "655d56901f3db8dd29d5bda3c57797726b2c4d07c7475bb6c1e36811fbcb0dd9"
        );
    }

    #[test]
    fn first_stored_leaf_matches_cpp() {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let configuration = crate::fp::heads::configuration_digest(&[policy, layout, kl]);
        let leaf = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin");
        let tree = OpeningSize {
            log_n: 23,
            leaf_bytes: 2048,
            draws: 51,
            root_context: "first table".into(),
            tree_number: 0,
            stored: true,
        };
        let digest = hash_leaf(
            &tree,
            &configuration,
            14899,
            &leaf[6304..8352],
            &leaf[8352..8392],
        )
        .unwrap();
        let sibling = Digest::from_le_bytes(leaf[112792..112824].try_into().unwrap());
        let parent = hash_vertex(&configuration, &sibling, &digest);
        assert_eq!(
            parent.to_hex(),
            "07082bcb4478b8789992f4cf6712a3522d5327e335608b9914d75169225e70a8"
        );
    }
}
