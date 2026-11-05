//! Key, leaf (`entry`) and proof file-layout composition (frame/files.h).
//!
//! A plan describes bytes; it is not a verification result. Fixed fields have
//! exact offsets and lengths. Position-dependent openings (`Columns`, `Salts`,
//! `Paths`) have maximum lengths until a structural walk supplies the drawn
//! positions and Merkle-path pruning.

use alloc::{string::String, vec::Vec};

use crate::fp::error::FpError;

/// The byte meaning of one planned message (numbers/plan.h `MessageKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageKind {
    Scalars,
    Answers,
    Root,
    Nonce,
    Columns,
    Salts,
    Paths,
    Head,
}

impl MessageKind {
    /// Whether drawn positions determine the actual byte count in a given run.
    pub fn is_position_dependent(self) -> bool {
        matches!(self, Self::Columns | Self::Salts | Self::Paths)
    }
}

/// Tree and draw information needed to compute an opening's actual size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpeningSize {
    pub log_n: u32,
    pub leaf_bytes: u64,
    pub draws: u64,
    pub root_context: String,
    pub tree_number: u32,
    pub stored: bool,
}

/// One run of bytes in a chain plan. `count` represents compact repeated
/// rounds; layout expansion borrows the message rather than copying its text.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanMessage {
    pub name: String,
    pub context: String,
    pub kind: MessageKind,
    /// Maximum bytes in one copy.
    pub bytes: u64,
    /// Expected bytes in one copy over the drawn positions.
    pub expected: f64,
    pub count: u64,
    /// Whether these bytes enter the Fiat-Shamir chain before its next draw.
    pub in_chain: bool,
    /// Whether drawn positions determine this run's actual byte count.
    pub position_dependent: bool,
    /// Present on position-dependent opening fields.
    pub opening: Option<OpeningSize>,
}

impl PlanMessage {
    pub fn fixed(
        name: impl Into<String>,
        context: impl Into<String>,
        kind: MessageKind,
        bytes: u64,
    ) -> Self {
        Self {
            name: name.into(),
            context: context.into(),
            kind,
            bytes,
            expected: bytes as f64,
            count: 1,
            in_chain: false,
            position_dependent: false,
            opening: None,
        }
    }

    pub fn position_dependent(
        name: impl Into<String>,
        context: impl Into<String>,
        kind: MessageKind,
        maximum: u64,
        expected: f64,
    ) -> Self {
        assert!(
            kind.is_position_dependent(),
            "fixed message kind used for a position-dependent opening"
        );
        assert!(
            expected >= 0.0 && expected <= maximum as f64,
            "invalid expected opening size"
        );
        Self {
            name: name.into(),
            context: context.into(),
            kind,
            bytes: maximum,
            expected,
            count: 1,
            in_chain: false,
            position_dependent: true,
            opening: None,
        }
    }

    pub fn with_count(mut self, count: u64) -> Self {
        self.count = count;
        self
    }

    pub fn in_chain(mut self) -> Self {
        self.in_chain = true;
        self
    }

    pub fn with_opening(mut self, opening: OpeningSize) -> Self {
        assert!(
            self.position_dependent,
            "opening metadata on a fixed message"
        );
        self.opening = Some(opening);
        self
    }
}

/// One chain's ordered messages. The chain's transcript rounds are built by
/// the protocol planners; this type holds their byte-layout projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainPlan {
    pub name: String,
    pub messages: Vec<PlanMessage>,
}

impl ChainPlan {
    pub fn new(name: impl Into<String>, messages: Vec<PlanMessage>) -> Self {
        Self {
            name: name.into(),
            messages,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Key,
    Entry,
    Proof,
}

impl FileKind {
    pub fn head_words(self) -> u64 {
        match self {
            Self::Key | Self::Entry => 2,
            Self::Proof => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Entry => "entry",
            Self::Proof => "proof",
        }
    }
}

/// File-level composition, independent of the protocol-specific planners that
/// produce each chain's messages.
#[derive(Debug, Clone, PartialEq)]
pub struct FilePlan {
    kind: FileKind,
    head: PlanMessage,
    table: Option<[PlanMessage; 2]>,
    place: Option<[PlanMessage; 3]>,
    chains: Vec<ChainPlan>,
    place_depth: Option<u32>,
}

impl FilePlan {
    pub fn key(table_answers: u64, chains: [ChainPlan; 3]) -> Result<Self, FpError> {
        check_chains(&chains, &["one-time chain", "shared chain", "second chain"])?;
        Self::new(
            FileKind::Key,
            Some(table_answers),
            chains.into_iter().collect(),
            Some(0),
        )
    }

    pub fn entry(table_answers: u64, chain: ChainPlan) -> Result<Self, FpError> {
        check_chains(core::slice::from_ref(&chain), &["link's chain"])?;
        Self::new(
            FileKind::Entry,
            Some(table_answers),
            alloc::vec![chain],
            None,
        )
    }

    pub fn proof(chains: [ChainPlan; 3], slots: u64) -> Result<Self, FpError> {
        check_chains(&chains, &["first chain", "shared chain", "second chain"])?;
        if slots == 0 {
            return Err(FpError::Range {
                context: "proof-batch-slots",
            });
        }
        let depth = 63 - slots.leading_zeros(); // floor(log2(slots)); ceil below if not a power of two.
        let depth = depth + u32::from(!slots.is_power_of_two());
        Self::new(
            FileKind::Proof,
            None,
            chains.into_iter().collect(),
            Some(depth),
        )
    }

    fn new(
        kind: FileKind,
        table_answers: Option<u64>,
        chains: Vec<ChainPlan>,
        place_depth: Option<u32>,
    ) -> Result<Self, FpError> {
        let table = match table_answers {
            Some(answers) => {
                let record = answers.checked_mul(24).ok_or(FpError::Range {
                    context: "table-record-bytes",
                })?;
                let context = if kind == FileKind::Entry {
                    "further table"
                } else {
                    "table"
                };
                Some([
                    PlanMessage::fixed("root", context, MessageKind::Root, 32),
                    PlanMessage::fixed("record", context, MessageKind::Answers, record),
                ])
            }
            None => None,
        };
        let place = match place_depth {
            Some(depth) => {
                let path_bytes = (depth as u64).checked_mul(32).ok_or(FpError::Range {
                    context: "batch-path-bytes",
                })?;
                Some([
                    PlanMessage::fixed("place in the batch", "", MessageKind::Head, 8),
                    PlanMessage::fixed("salt of the place's leaf", "", MessageKind::Salts, 40),
                    PlanMessage::fixed("path in the batch", "", MessageKind::Paths, path_bytes),
                ])
            }
            None => None,
        };
        Ok(Self {
            kind,
            head: PlanMessage::fixed("head", "", MessageKind::Head, 4 * kind.head_words()),
            table,
            place,
            chains,
            place_depth,
        })
    }

    pub fn kind(&self) -> FileKind {
        self.kind
    }

    pub fn chains(&self) -> &[ChainPlan] {
        &self.chains
    }

    pub fn place_depth(&self) -> Option<u32> {
        self.place_depth
    }

    /// Expand compact messages into their exact ordered byte layout.
    pub fn layout(&self) -> Result<FileLayout<'_>, FpError> {
        let mut builder = LayoutBuilder::default();
        builder.push(&self.head)?;
        if let Some(table) = &self.table {
            for message in table {
                builder.push(message)?;
            }
        }
        for (index, chain) in self.chains.iter().enumerate() {
            for message in &chain.messages {
                builder.push_repeated(message)?;
            }
            if index == 0 {
                if let Some(place) = &self.place {
                    for message in place {
                        builder.push(message)?;
                    }
                }
            }
        }
        Ok(FileLayout {
            kind: self.kind,
            fields: builder.fields,
            maximum_bytes: builder.offset,
            position_dependent: builder.position_dependent,
        })
    }
}

fn check_chains(chains: &[ChainPlan], names: &[&str]) -> Result<(), FpError> {
    if chains.len() != names.len()
        || !chains
            .iter()
            .zip(names)
            .all(|(chain, name)| chain.name == *name)
    {
        return Err(FpError::Range {
            context: "file-chain-sequence",
        });
    }
    Ok(())
}

/// One expanded field. Its message is borrowed from the file plan.
#[derive(Debug, Clone, Copy)]
pub struct LayoutField<'a> {
    pub offset: u64,
    pub message: &'a PlanMessage,
}

impl LayoutField<'_> {
    pub fn end(&self) -> u64 {
        self.offset + self.message.bytes // checked by LayoutBuilder
    }
}

#[derive(Default)]
struct LayoutBuilder<'a> {
    fields: Vec<LayoutField<'a>>,
    offset: u64,
    position_dependent: bool,
}

impl<'a> LayoutBuilder<'a> {
    fn push(&mut self, message: &'a PlanMessage) -> Result<(), FpError> {
        self.push_repeated(message)
    }

    fn push_repeated(&mut self, message: &'a PlanMessage) -> Result<(), FpError> {
        if message.count == 0 {
            return Err(FpError::Range {
                context: "plan-message-count",
            });
        }
        if message.bytes == 0 {
            return Ok(()); // Plan::file emits no zero-byte item (e.g. a depth-zero path).
        }
        let total = message
            .bytes
            .checked_mul(message.count)
            .ok_or(FpError::Range {
                context: "plan-message-bytes",
            })?;
        self.offset.checked_add(total).ok_or(FpError::Range {
            context: "plan-offset-overflow",
        })?;
        for _ in 0..message.count {
            self.fields.push(LayoutField {
                offset: self.offset,
                message,
            });
            self.offset += message.bytes;
            self.position_dependent |= message.position_dependent;
        }
        Ok(())
    }
}

/// Expanded layout and its checked maximum byte count.
#[derive(Debug)]
pub struct FileLayout<'a> {
    kind: FileKind,
    fields: Vec<LayoutField<'a>>,
    maximum_bytes: u64,
    position_dependent: bool,
}

impl FileLayout<'_> {
    pub fn kind(&self) -> FileKind {
        self.kind
    }

    pub fn fields(&self) -> &[LayoutField<'_>] {
        &self.fields
    }

    pub fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub fn is_fixed_length(&self) -> bool {
        !self.position_dependent
    }

    /// Validate what offsets alone establish. Fixed layouts require exact
    /// equality; layouts with position-dependent openings enforce their upper
    /// bound here and require the structural walker for exact consumption.
    pub fn validate_len(&self, len: usize) -> Result<(), FpError> {
        let actual = len as u64;
        if actual > self.maximum_bytes {
            return Err(FpError::TrailingBytes {
                context: "file-plan",
                count: (actual - self.maximum_bytes) as usize,
            });
        }
        if !self.position_dependent && actual < self.maximum_bytes {
            return Err(FpError::Truncated {
                context: "file-plan",
                needed: self.maximum_bytes as usize,
                available: len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_chain(name: &str, bytes: u64) -> ChainPlan {
        ChainPlan::new(
            name,
            alloc::vec![PlanMessage::fixed(
                "chain salt",
                name,
                MessageKind::Salts,
                bytes
            )],
        )
    }

    #[test]
    fn key_places_table_record_before_chains_and_binding_place_after_first() {
        let plan = FilePlan::key(
            256,
            [
                fixed_chain("one-time chain", 32),
                fixed_chain("shared chain", 32),
                fixed_chain("second chain", 32),
            ],
        )
        .unwrap();
        let layout = plan.layout().unwrap();
        let fields = layout.fields();
        assert_eq!((fields[0].offset, fields[0].message.bytes), (0, 8));
        assert_eq!((fields[1].offset, fields[1].message.bytes), (8, 32));
        assert_eq!((fields[2].offset, fields[2].message.bytes), (40, 6144));
        assert_eq!(fields[3].offset, 6184, "one-time chain follows the record");
        assert_eq!(fields[4].message.name, "place in the batch");
        assert_eq!(fields[4].offset, 6216);
        assert_eq!(fields[5].message.bytes, 40);
        assert_eq!(
            fields[6].offset, 6264,
            "depth-zero path is omitted; shared chain follows the place"
        );
        assert_eq!(layout.maximum_bytes(), 6328);
        assert_eq!(layout.fields().last().unwrap().end(), 6328);
    }

    #[test]
    fn proof_uses_three_word_head_and_ceil_log_batch_depth() {
        let plan = FilePlan::proof(
            [
                fixed_chain("first chain", 32),
                fixed_chain("shared chain", 32),
                fixed_chain("second chain", 32),
            ],
            65,
        )
        .unwrap();
        assert_eq!(plan.place_depth(), Some(7));
        let layout = plan.layout().unwrap();
        let fields = layout.fields();
        assert_eq!(fields[0].message.bytes, 12);
        assert_eq!(fields[1].offset, 12);
        assert_eq!(fields[2].offset, 44);
        assert_eq!(fields[3].offset, 52);
        assert_eq!((fields[4].offset, fields[4].message.bytes), (92, 224));
        assert_eq!(fields[5].offset, 316);
    }

    #[test]
    fn fixed_layout_rejects_short_and_long_inputs() {
        let plan = FilePlan::entry(1, fixed_chain("link's chain", 32)).unwrap();
        let layout = plan.layout().unwrap();
        assert!(layout.is_fixed_length());
        assert!(matches!(
            layout.validate_len((layout.maximum_bytes() - 1) as usize),
            Err(FpError::Truncated { .. })
        ));
        assert!(matches!(
            layout.validate_len((layout.maximum_bytes() + 1) as usize),
            Err(FpError::TrailingBytes { .. })
        ));
        layout
            .validate_len(layout.maximum_bytes() as usize)
            .unwrap();
    }

    #[test]
    fn position_dependent_layout_enforces_maximum_without_claiming_exact_length() {
        let mut chain = fixed_chain("link's chain", 32);
        chain.messages.push(PlanMessage::position_dependent(
            "paths",
            "opening",
            MessageKind::Paths,
            128,
            91.5,
        ));
        let plan = FilePlan::entry(1, chain).unwrap();
        let layout = plan.layout().unwrap();
        assert!(!layout.is_fixed_length());
        layout.validate_len(0).unwrap();
        layout
            .validate_len(layout.maximum_bytes() as usize)
            .unwrap();
        assert!(matches!(
            layout.validate_len((layout.maximum_bytes() + 1) as usize),
            Err(FpError::TrailingBytes { .. })
        ));
    }

    #[test]
    fn arithmetic_overflow_is_refused() {
        let mut chain = fixed_chain("link's chain", 32);
        chain
            .messages
            .push(PlanMessage::fixed("huge", "test", MessageKind::Columns, u64::MAX).with_count(2));
        let plan = FilePlan::entry(1, chain).unwrap();
        assert!(matches!(
            plan.layout(),
            Err(FpError::Range {
                context: "plan-message-bytes"
            })
        ));
    }

    #[test]
    fn constructors_reject_wrong_chain_sequences_and_empty_batches() {
        assert!(FilePlan::entry(1, fixed_chain("wrong", 32)).is_err());
        assert!(FilePlan::proof(
            [
                fixed_chain("first chain", 32),
                fixed_chain("shared chain", 32),
                fixed_chain("second chain", 32)
            ],
            0,
        )
        .is_err());
    }
}
