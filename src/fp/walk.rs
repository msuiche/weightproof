//! Structural file walking against a `FileLayout`.
//!
//! The caller supplies the positions drawn by the hash chain in opening order.
//! Given those positions, every field has an exact length and the walker
//! requires exact file exhaustion (FORMAT.md §1 rule 2).

use alloc::{collections::BTreeSet, vec::Vec};

use crate::fp::{
    error::FpError,
    layout::{FileKind, FileLayout, MessageKind, PlanMessage},
    version::Version,
};

/// Positions for one `plan_open` call, including duplicates in draw order.
#[derive(Debug, Clone, Copy)]
pub struct OpeningActual<'a> {
    /// Must equal the planned opening's full context (stage and tree name).
    pub context: &'a str,
    pub positions: &'a [u64],
}

/// One field at its actual (not maximum) offset in a file.
#[derive(Debug, Clone, Copy)]
pub struct ActualField<'a> {
    pub offset: u64,
    pub bytes: u64,
    pub message: &'a PlanMessage,
}

#[derive(Debug)]
pub struct WalkedFile<'a> {
    pub fields: Vec<ActualField<'a>>,
    pub proof_entry: Option<u32>,
}

impl WalkedFile<'_> {
    pub fn bytes(&self) -> u64 {
        self.fields
            .last()
            .map_or(0, |field| field.offset + field.bytes)
    }
}

/// Walk a complete file, returning actual offsets. `openings` follows the
/// order of first appearance of each position-dependent context.
pub fn walk_file<'a>(
    bytes: &[u8],
    layout: &'a FileLayout<'a>,
    openings: &[OpeningActual<'_>],
) -> Result<WalkedFile<'a>, FpError> {
    let head_bytes = (4 * layout.kind().head_words()) as usize;
    if bytes.len() < head_bytes {
        return Err(FpError::Truncated {
            context: "file-head",
            needed: head_bytes,
            available: bytes.len(),
        });
    }
    let expected_tag = match layout.kind() {
        FileKind::Key => b"CPIK",
        FileKind::Entry => b"CPEN",
        FileKind::Proof => b"CPPF",
    };
    if bytes[..4] != expected_tag[..] {
        return Err(FpError::BadTag {
            expected: match layout.kind() {
                FileKind::Key => "CPIK",
                FileKind::Entry => "CPEN",
                FileKind::Proof => "CPPF",
            },
        });
    }
    Version::parse(&bytes[4..8])?;
    let proof_entry = if layout.kind() == FileKind::Proof {
        Some(u32::from_le_bytes(bytes[8..12].try_into().map_err(
            |_| FpError::BadLength {
                context: "proof-entry",
            },
        )?))
    } else {
        None
    };

    let mut fields = Vec::with_capacity(layout.fields().len());
    let mut offset = 0u64;
    let mut next_opening = 0usize;
    let mut active_context: Option<&str> = None;
    let mut active_sizes = (0u64, 0u64); // distinct leaves, pruned path digests

    for planned in layout.fields() {
        let message = planned.message;
        let actual = if message.position_dependent {
            if active_context != Some(message.context.as_str()) {
                let supplied = openings.get(next_opening).ok_or(FpError::Range {
                    context: "opening-missing",
                })?;
                if supplied.context != message.context {
                    return Err(FpError::Range {
                        context: "opening-order",
                    });
                }
                let opening = message.opening.ok_or(FpError::Range {
                    context: "opening-shape-missing",
                })?;
                active_sizes = actual_opening(opening.log_n, supplied.positions, opening.draws)?;
                active_context = Some(supplied.context);
                next_opening += 1;
            }
            let opening = message.opening.ok_or(FpError::Range {
                context: "opening-shape-missing",
            })?;
            match message.kind {
                MessageKind::Columns => {
                    active_sizes
                        .0
                        .checked_mul(opening.leaf_bytes)
                        .ok_or(FpError::Range {
                            context: "actual-opening-bytes",
                        })?
                }
                MessageKind::Salts => active_sizes.0.checked_mul(40).ok_or(FpError::Range {
                    context: "actual-opening-bytes",
                })?,
                MessageKind::Paths => active_sizes.1.checked_mul(32).ok_or(FpError::Range {
                    context: "actual-opening-bytes",
                })?,
                _ => {
                    return Err(FpError::Range {
                        context: "opening-kind",
                    })
                }
            }
        } else {
            message.bytes
        };
        if actual > message.bytes {
            return Err(FpError::Range {
                context: "actual-opening-exceeds-plan",
            });
        }
        let end = offset.checked_add(actual).ok_or(FpError::Range {
            context: "actual-file-overflow",
        })?;
        if end > bytes.len() as u64 {
            return Err(FpError::Truncated {
                context: "file-walk",
                needed: end as usize,
                available: bytes.len(),
            });
        }
        fields.push(ActualField {
            offset,
            bytes: actual,
            message,
        });
        offset = end;
    }
    if next_opening != openings.len() {
        return Err(FpError::Range {
            context: "opening-count",
        });
    }
    if offset != bytes.len() as u64 {
        return Err(FpError::TrailingBytes {
            context: "file-walk",
            count: bytes.len() - offset as usize,
        });
    }
    Ok(WalkedFile {
        fields,
        proof_entry,
    })
}

/// Return distinct leaves and pruned sibling-digest count for actual positions.
fn actual_opening(log_n: u32, positions: &[u64], draws: u64) -> Result<(u64, u64), FpError> {
    if log_n >= 64 || positions.len() as u64 != draws {
        return Err(FpError::Range {
            context: "opening-draw-count",
        });
    }
    let leaves = 1u64 << log_n;
    let mut current = BTreeSet::new();
    for &position in positions {
        if position >= leaves {
            return Err(FpError::Range {
                context: "opening-position-range",
            });
        }
        current.insert(position);
    }
    let distinct = current.len() as u64;
    let mut digests = 0u64;
    for _level in 0..log_n {
        let mut parents = BTreeSet::new();
        for &position in &current {
            if !current.contains(&(position ^ 1)) {
                digests = digests.checked_add(1).ok_or(FpError::Range {
                    context: "actual-path-overflow",
                })?;
            }
            parents.insert(position / 2);
        }
        current = parents;
    }
    Ok((distinct, digests))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::{
        layout::{ChainPlan, FilePlan, PlanMessage},
        plan::{plan_open, TreeSize},
    };

    fn entry_with_opening() -> (FilePlan, Vec<u8>) {
        let mut messages = vec![PlanMessage::fixed(
            "chain salt",
            "link's chain",
            MessageKind::Salts,
            32,
        )];
        plan_open(
            &mut messages,
            "test opening",
            &TreeSize::new("tree", 3, 16),
            3,
        )
        .unwrap();
        messages.push(PlanMessage::fixed(
            "after",
            "test",
            MessageKind::Scalars,
            24,
        ));
        let plan = FilePlan::entry(0, ChainPlan::new("link's chain", messages)).unwrap();
        let layout = plan.layout().unwrap();
        let mut bytes = vec![0u8; layout.maximum_bytes() as usize];
        bytes[..4].copy_from_slice(b"CPEN");
        bytes[4..8].copy_from_slice(b"H821");
        (plan, bytes)
    }

    #[test]
    fn duplicate_positions_shrink_openings_and_move_later_offsets_exactly() {
        let (plan, mut bytes) = entry_with_opening();
        let layout = plan.layout().unwrap();
        bytes.truncate(layout.maximum_bytes() as usize - 88); // 16 + 40 + 32 bytes pruned below.
        let walked = walk_file(
            &bytes,
            &layout,
            &[OpeningActual {
                context: "test opening, tree",
                positions: &[1, 1, 3],
            }],
        )
        .unwrap();
        let opening = walked
            .fields
            .iter()
            .find(|field| field.message.name == "opened symbols")
            .unwrap();
        assert_eq!(opening.bytes, 32, "two distinct leaves");
        assert_eq!(
            walked
                .fields
                .iter()
                .find(|field| field.message.name == "salts")
                .unwrap()
                .bytes,
            80
        );
        assert_eq!(
            walked
                .fields
                .iter()
                .find(|field| field.message.name == "paths")
                .unwrap()
                .bytes,
            96
        );
        let after = walked
            .fields
            .iter()
            .find(|field| field.message.name == "after")
            .unwrap();
        assert_eq!(after.offset, opening.offset + 32 + 80 + 96);
        assert_eq!(walked.bytes(), after.offset + 24);

        bytes.truncate(walked.bytes() as usize);
        walk_file(
            &bytes,
            &layout,
            &[OpeningActual {
                context: "test opening, tree",
                positions: &[1, 1, 3],
            }],
        )
        .unwrap();
        bytes.push(0);
        assert!(matches!(
            walk_file(
                &bytes,
                &layout,
                &[OpeningActual {
                    context: "test opening, tree",
                    positions: &[1, 1, 3]
                }]
            ),
            Err(FpError::TrailingBytes { count: 1, .. })
        ));
        bytes.pop();
        bytes.pop();
        assert!(matches!(
            walk_file(
                &bytes,
                &layout,
                &[OpeningActual {
                    context: "test opening, tree",
                    positions: &[1, 1, 3]
                }]
            ),
            Err(FpError::Truncated { .. })
        ));
    }

    #[test]
    fn root_side_openings_keep_higher_level_paths() {
        assert_eq!(actual_opening(3, &[0, 0, 0], 3).unwrap(), (1, 3));
        assert_eq!(actual_opening(3, &[0, 1, 1], 3).unwrap(), (2, 2));
        assert_eq!(actual_opening(3, &[0, 0, 0], 3).unwrap().1 * 32, 96);
    }

    #[test]
    fn bad_position_lists_fail_closed() {
        let (plan, bytes) = entry_with_opening();
        let layout = plan.layout().unwrap();
        for positions in [&[1, 2][..], &[1, 2, 8][..]] {
            assert!(walk_file(
                &bytes,
                &layout,
                &[OpeningActual {
                    context: "test opening, tree",
                    positions
                }]
            )
            .is_err());
        }
        assert!(walk_file(&bytes, &layout, &[]).is_err());
    }

    #[test]
    fn bad_file_tag_is_rejected_before_walking() {
        let (plan, mut bytes) = entry_with_opening();
        bytes[0] = b'X';
        let layout = plan.layout().unwrap();
        assert!(matches!(
            walk_file(
                &bytes,
                &layout,
                &[OpeningActual {
                    context: "test opening, tree",
                    positions: &[1, 1, 3]
                }]
            ),
            Err(FpError::BadTag { expected: "CPEN" })
        ));
    }
}
