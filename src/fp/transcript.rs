//! Shared transcript cursor, canonical field decoding, and verified openings.

use alloc::{borrow::ToOwned, collections::BTreeMap, string::String, vec, vec::Vec};
use core::ops::Range;

use crate::fp::{
    digest::Digest,
    error::FpError,
    field::{ExtensionElement, FieldElement},
    layout::{FileLayout, MessageKind},
    merkle::verify_opening,
    walk::actual_opening,
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

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    layout: &'a FileLayout<'a>,
    configuration: Digest,
    pub(crate) index: usize,
    offset: usize,
    roots: BTreeMap<String, Digest>,
    pub(crate) openings: Vec<(String, Vec<u64>)>,
    pub(crate) opened: Vec<OpeningData>,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(
        bytes: &'a [u8],
        layout: &'a FileLayout<'a>,
        configuration: &Digest,
        anchor_root: &Digest,
    ) -> Self {
        Self::new_with_anchor(bytes, layout, configuration, "first table", anchor_root)
    }

    pub(crate) fn new_with_anchor(
        bytes: &'a [u8],
        layout: &'a FileLayout<'a>,
        configuration: &Digest,
        anchor: &str,
        anchor_root: &Digest,
    ) -> Self {
        let mut roots = BTreeMap::new();
        roots.insert(anchor.into(), *anchor_root);
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

    pub(crate) fn take(&mut self, name: &str, context: &str) -> Result<&'a [u8], FpError> {
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

    pub(crate) fn take_root(&mut self, context: &str) -> Result<Digest, FpError> {
        let bytes = self.take("root", context)?;
        let root = Digest::from_le_bytes(bytes.try_into().map_err(|_| FpError::BadLength {
            context: "transcript-root",
        })?);
        self.roots.insert(context.into(), root);
        Ok(root)
    }

    pub(crate) fn take_nonce(&mut self, context: &str) -> Result<u64, FpError> {
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

    pub(crate) fn open_next(&mut self, positions: &[u64], run: &Digest) -> Result<(), FpError> {
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

    pub(crate) fn open_stage(&mut self, positions: &[u64], start: &Digest) -> Result<(), FpError> {
        while self.next_opening_context().is_some() {
            self.open_next(positions, start)?;
        }
        Ok(())
    }
}

pub(crate) fn words(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

pub(crate) fn elements(bytes: &[u8]) -> Vec<ExtensionElement> {
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

pub(crate) fn one_element(bytes: &[u8]) -> ExtensionElement {
    let values = elements(bytes);
    assert_eq!(values.len(), 1);
    values[0]
}
