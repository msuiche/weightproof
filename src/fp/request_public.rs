//! Public request shape, array layout, and canonical first-query encoding.

use alloc::{
    collections::BTreeMap,
    string::String,
    vec,
    vec::Vec,
};

use crate::fp::{
    claims::OracleShape,
    error::FpError,
    key_public::{IoppChoice, IoppStage, OracleNotice},
};

#[derive(Debug, Clone)]
pub struct ReqMessage {
    pub shape: OracleShape,
    pub leaf_class: u32,
    pub moment: u32,
    pub life: u32,
}

#[derive(Debug, Clone)]
pub struct ReqLink {
    pub name: String,
    pub messages: Vec<ReqMessage>,
    pub plain: u32,
}

impl ReqLink {
    pub fn moments(&self) -> u32 {
        self.messages.last().map_or(0, |message| message.moment + 1)
    }
}

#[derive(Debug, Clone)]
pub struct ReqShape {
    pub code: u32,
    pub mask_code: u32,
    pub links: Vec<ReqLink>,
    pub at_start: Vec<OracleNotice>,
    pub answers: u32,
    pub last: Vec<IoppStage>,
    pub choice: IoppChoice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaidBlock {
    pub first_i: u64,
    pub first_c: u64,
    pub first_row: u64,
    pub first_place: u64,
    pub bits_i: u32,
    pub bits_c: u32,
    pub row_bits: u32,
    pub place_bits: u32,
    pub i_is_low: bool,
}

#[derive(Debug, Clone)]
pub struct ReqArray {
    pub name: String,
    pub message: usize,
    pub size_i: u64,
    pub size_c: u64,
    pub blocks: Vec<LaidBlock>,
}

#[derive(Debug, Clone)]
pub struct ReqLayout {
    pub links: Vec<Vec<ReqArray>>,
}

impl ReqLayout {
    pub fn find(&self, link: usize, name: &str) -> Option<&ReqArray> {
        self.links
            .get(link)?
            .iter()
            .find(|array| array.name == name)
    }
}

/// Names and dimensions declared by the model subproofs, indexed as in the
/// layout file: link, commit point, array. Array extents are checked against
/// the blocks' complete cover after parsing.
#[derive(Debug, Clone)]
pub struct ReqDeclarations {
    pub links: Vec<ReqLinkDeclaration>,
}

#[derive(Debug, Clone)]
pub struct ReqLinkDeclaration {
    pub name: String,
    pub plain: u32,
    pub moments: Vec<Vec<ReqArrayDeclaration>>,
}

#[derive(Debug, Clone)]
pub struct ReqArrayDeclaration {
    pub name: String,
    pub rows: u64,
    pub columns: u64,
    pub over_e: bool,
    pub life: u32,
}

fn number(token: Option<&str>, context: &'static str) -> Result<u64, FpError> {
    token
        .ok_or(FpError::BadLength { context })?
        .parse::<u64>()
        .map_err(|_| FpError::Range { context })
}

fn one_based(token: Option<&str>, context: &'static str) -> Result<usize, FpError> {
    let value = number(token, context)?;
    if value == 0 || value >= (1 << 20) {
        return Err(FpError::Range { context });
    }
    Ok(value as usize - 1)
}

/// Parse `layout.txt` and combine it with the model's public array
/// declarations. This mirrors `layout_read()` + `req_frame()`; block overlap,
/// alignment, and pad exclusion are checked when claims are constructed.
pub fn request_layout(
    bytes: &[u8],
    declarations: &ReqDeclarations,
    code: u32,
    mask_code: u32,
    answers: u32,
) -> Result<(Vec<ReqLink>, ReqLayout), FpError> {
    const CONTEXT: &'static str = "request-layout";
    let text = core::str::from_utf8(bytes).map_err(|_| FpError::Range { context: CONTEXT })?;
    let mut links: Vec<ReqLink> = declarations
        .links
        .iter()
        .map(|link| ReqLink {
            name: link.name.clone(),
            messages: vec![],
            plain: link.plain,
        })
        .collect();
    let mut layout = ReqLayout {
        links: declarations.links.iter().map(|_| vec![]).collect(),
    };
    // (link, moment, group) -> message index in the link.
    let mut messages = BTreeMap::new();
    // (link, moment, array) -> array index in the link's layout.
    let mut arrays = BTreeMap::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.len() {
            6 => {
                let link = one_based(tokens.first().copied(), CONTEXT)?;
                let moment = one_based(tokens.get(1).copied(), CONTEXT)?;
                let group = one_based(tokens.get(2).copied(), CONTEXT)?;
                let class = tokens[3];
                let rows = number(tokens.get(4).copied(), CONTEXT)?;
                let life_value = number(tokens.get(5).copied(), CONTEXT)?;
                if life_value >= (1 << 20) {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let life = life_value as u32;
                if link >= declarations.links.len()
                    || moment >= declarations.links[link].moments.len()
                    || rows == 0
                    || rows >> 32 != 0
                    || !matches!(class, "N" | "B" | "W")
                    || (life != 0) != (class == "B")
                {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let next_nonempty = links
                    .get(link + 1)
                    .is_some_and(|next| !next.messages.is_empty());
                let sent = &mut links[link].messages;
                if sent.last().is_some_and(|last| last.moment as usize > moment) || next_nonempty {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let moment_groups = messages
                    .iter()
                    .filter(|((l, m, _), _)| *l == link && *m == moment)
                    .count();
                if group != moment_groups || (sent.is_empty() && life != 0) {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let index = sent.len();
                sent.push(ReqMessage {
                    shape: OracleShape {
                        code,
                        mask_code,
                        main_rows: rows,
                        mask_rows: u64::from(index == 0),
                        main_deg: if class == "W" { 3 } else { 1 },
                        answers,
                    },
                    leaf_class: 0,
                    moment: moment as u32,
                    life,
                });
                messages.insert((link, moment, group), index);
            }
            14 => {
                let link = one_based(tokens.first().copied(), CONTEXT)?;
                let moment = one_based(tokens.get(1).copied(), CONTEXT)?;
                let group = one_based(tokens.get(2).copied(), CONTEXT)?;
                let array = number(tokens.get(3).copied(), CONTEXT)? as usize;
                if link >= declarations.links.len()
                    || moment >= declarations.links[link].moments.len()
                    || array >= declarations.links[link].moments[moment].len()
                {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let Some(&message) = messages.get(&(link, moment, group)) else {
                    return Err(FpError::Range { context: CONTEXT });
                };
                let direction = tokens[8];
                let final_zero = number(tokens.get(13).copied(), CONTEXT)?;
                let block = LaidBlock {
                    first_i: number(tokens.get(4).copied(), CONTEXT)?,
                    bits_i: number(tokens.get(5).copied(), CONTEXT)? as u32,
                    first_c: number(tokens.get(6).copied(), CONTEXT)?,
                    bits_c: number(tokens.get(7).copied(), CONTEXT)? as u32,
                    i_is_low: direction == "r",
                    first_row: number(tokens.get(9).copied(), CONTEXT)?,
                    first_place: number(tokens.get(10).copied(), CONTEXT)?,
                    row_bits: number(tokens.get(11).copied(), CONTEXT)? as u32,
                    place_bits: number(tokens.get(12).copied(), CONTEXT)? as u32,
                };
                if !matches!(direction, "r" | "c")
                    || final_zero != 0
                    || [block.bits_i, block.bits_c, block.row_bits, block.place_bits]
                        .iter()
                        .any(|&bits| bits >= 48)
                    || block.row_bits + block.place_bits != block.bits_i + block.bits_c
                {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let declared = &declarations.links[link].moments[moment][array];
                let sent = &links[link].messages[message];
                if (sent.shape.main_deg == 3) != declared.over_e || sent.life < declared.life {
                    return Err(FpError::Range { context: CONTEXT });
                }
                let output = &mut layout.links[link];
                let at = if let Some(&at) = arrays.get(&(link, moment, array)) {
                    at
                } else {
                    let at = output.len();
                    output.push(ReqArray {
                        name: declared.name.clone(),
                        message,
                        size_i: declared.rows,
                        size_c: declared.columns,
                        blocks: vec![],
                    });
                    arrays.insert((link, moment, array), at);
                    at
                };
                if output[at].message != message {
                    return Err(FpError::Range { context: CONTEXT });
                }
                output[at].blocks.push(block);
            }
            _ => return Err(FpError::Range { context: CONTEXT }),
        }
    }
    for (link, declaration) in declarations.links.iter().enumerate() {
        for (moment, declared_arrays) in declaration.moments.iter().enumerate() {
            for (array, _) in declared_arrays.iter().enumerate() {
                if !arrays.contains_key(&(link, moment, array)) {
                    return Err(FpError::Range { context: CONTEXT });
                }
            }
        }
    }
    Ok((links, layout))
}

/// Array declarations of the checked-in small-model example.
pub(crate) fn accepted_declarations() -> ReqDeclarations {
    fn array(name: &str, rows: u64, columns: u64, over_e: bool, life: u32) -> ReqArrayDeclaration {
        ReqArrayDeclaration {
            name: name.into(),
            rows,
            columns,
            over_e,
            life,
        }
    }
    fn base(name: &str, rows: u64, columns: u64) -> ReqArrayDeclaration {
        array(name, rows, columns, false, 0)
    }
    fn layer(first: bool, u3_columns: u64) -> ReqLinkDeclaration {
        let mut initial = vec![
            base("M0", 2731, 247),
            base("M1", 2731, 247),
            base("M2", 2731, 247),
            base("M3", 1366, 160),
            base("M4", 1366, 14),
            base("M5", 2731, 2048),
        ];
        if first {
            initial.push(base("M8", 2731, 20));
        }
        initial.push(array("M6", 2731, 96, false, 1));
        initial.push(array("M7", 2731, 20, false, 2));
        ReqLinkDeclaration {
            name: if first { "G0".into() } else { "G1".into() },
            plain: 0,
            moments: vec![
                initial,
                vec![array("U3", 1, u3_columns, true, 0)],
                vec![base("ZZ", 128, 131072), base("BZ", 32, 2048)],
            ],
        }
    }
    fn attention(name: &str) -> ReqLinkDeclaration {
        ReqLinkDeclaration {
            name: name.into(),
            plain: 0,
            moments: vec![
                vec![
                    base("M0", 683, 120),
                    base("M1", 1366, 4),
                    base("M2", 2731, 32),
                    base("M3", 2731, 1024),
                ],
                vec![array("U3", 1, 34152, true, 0)],
                vec![base("ZZ", 128, 61440), base("BZ", 32, 2048)],
            ],
        }
    }
    ReqDeclarations {
        links: vec![
            layer(true, 167328),
            attention("A0"),
            layer(false, 166688),
            attention("A1"),
            ReqLinkDeclaration {
                name: "H".into(),
                plain: 1,
                moments: vec![
                    vec![
                        base("M0", 5462, 20),
                        base("M1", 1366, 16),
                        base("M2", 1366, 2),
                        base("M3", 2731, 1536),
                    ],
                    vec![array("U3", 1, 11264, true, 0)],
                    vec![base("ZZ", 128, 81920), base("BZ", 32, 2048)],
                ],
            },
        ],
    }
}

/// Fixed public inputs of one request proof.
#[derive(Debug, Clone)]
pub struct RequestFixed {
    pub configuration: crate::fp::digest::Digest,
    pub codes: crate::fp::key_public::Codes,
    pub lin: crate::fp::key_public::LinShape,
    pub shape: ReqShape,
    pub layout: ReqLayout,
    pub identifiers: Vec<u64>,
    pub pending_rows: u64,
    pub binding: crate::fp::key_public::LinBindingShape,
}

pub fn accepted_request_fixed(
    configuration: crate::fp::digest::Digest,
    layout_bytes: &[u8],
) -> RequestFixed {
    fn code(log_n: u32, ell: u64, r: u64) -> crate::fp::codes::CodeSize {
        crate::fp::codes::CodeSize {
            log_n,
            ell,
            r,
            over_E: true,
        }
    }
    fn stage(code: u32, mask_code: u32, k: u32, h: u32) -> IoppStage {
        IoppStage {
            code,
            mask_code,
            k,
            h,
        }
    }
    let key = crate::fp::key_fixture::accepted_fixed(configuration);
    let (links, layout) = request_layout(layout_bytes, &accepted_declarations(), 0, 0, 1).unwrap();
    let mut lin = key.lin;
    lin.fold.columns = 53;
    lin.fold.u.code = 5;
    lin.omega.code = 7;
    lin.w.code = 7;
    lin.w.mask_code = 5;
    lin.shared_split.mask_code = 5;
    lin.own_split.mask_code = 10;
    lin.shared_mask_code = 5;
    lin.own_mask_code = 10;
    lin.test_u = vec![stage(5, 5, 5, 0), stage(6, 6, 4, 2), stage(6, 7, 0, 0)];
    lin.test_omega = vec![stage(7, 5, 2, 0), stage(8, 8, 4, 3), stage(9, 9, 0, 0)];
    lin.test_own = vec![stage(7, 10, 3, 0), stage(10, 11, 4, 3), stage(11, 4, 0, 0)];
    RequestFixed {
        configuration,
        codes: crate::fp::key_public::Codes {
            base: vec![
                code(18, 131072, 218),
                code(17, 8192, 55),
                code(16, 1024, 44),
                code(15, 128, 44),
                code(15, 128, 73),
                code(17, 32768, 101),
                code(14, 2048, 68),
                code(18, 16384, 3831),
                code(14, 1024, 69),
                code(15, 1024, 68),
                code(14, 1024, 75),
                code(15, 1024, 73),
            ],
            mask: vec![
                code(17, 8192, 218),
                code(17, 4096, 55),
                code(16, 2048, 44),
                code(15, 1024, 44),
                code(15, 4096, 73),
                code(13, 64, 156),
                code(12, 256, 68),
                code(14, 1024, 68),
                code(13, 1024, 69),
                code(15, 4096, 68),
                code(13, 256, 59),
                code(13, 1024, 75),
            ],
        },
        lin,
        shape: ReqShape {
            code: 0,
            mask_code: 0,
            links,
            at_start: vec![OracleNotice {
                shape: OracleShape {
                    code: u32::MAX,
                    mask_code: 10,
                    main_rows: 0,
                    mask_rows: 1,
                    main_deg: 3,
                    answers: 1,
                },
                leaf_class: 1,
            }],
            answers: 1,
            last: vec![
                stage(0, 0, 1, 0),
                stage(1, 1, 4, 3),
                stage(2, 2, 3, 3),
                stage(3, 3, 3, 3),
                stage(4, 4, 0, 0),
            ],
            choice: IoppChoice { masks_ride: true },
        },
        layout,
        identifiers: vec![1, 4, 2, 5, 3],
        pending_rows: 3,
        binding: crate::fp::key_public::LinBindingShape {
            log_leaves: 6,
            salt_bytes: 40,
        },
    }
}

fn req_message(out: &mut Vec<u8>, value: &ReqMessage) {
    crate::fp::key_public::put_u64(out, value.shape.main_rows);
    crate::fp::key_public::put_u64(
        out,
        if value.shape.main_rows != 0 {
            value.shape.main_deg as u64
        } else {
            1
        },
    );
    crate::fp::key_public::put_u64(out, value.shape.mask_rows);
    crate::fp::key_public::put_u64(out, value.leaf_class as u64);
    crate::fp::key_public::put_u64(out, value.moment as u64);
    crate::fp::key_public::put_u64(out, value.life as u64);
}

fn req_shape(out: &mut Vec<u8>, value: &ReqShape) {
    crate::fp::key_public::put_u64(out, value.code as u64);
    crate::fp::key_public::put_u64(out, value.mask_code as u64);
    crate::fp::key_public::put_vec(out, &value.links, |out, link| {
        crate::fp::key_public::put_u64(out, link.plain as u64);
        crate::fp::key_public::put_vec(out, &link.messages, req_message);
    });
    crate::fp::key_public::put_vec(out, &value.at_start, |out, notice| {
        super::key_public::notice(out, notice);
    });
    crate::fp::key_public::put_u64(out, value.answers as u64);
    crate::fp::key_public::put_vec(out, &value.last[usize::from(!value.last.is_empty())..], |out, stage| {
        super::key_public::stage(out, stage);
    });
    crate::fp::key_public::put_u64(out, value.choice.masks_ride as u64);
}

fn req_layout(out: &mut Vec<u8>, value: &ReqLayout) {
    crate::fp::key_public::put_vec(out, &value.links, |out, arrays| {
        crate::fp::key_public::put_vec(out, arrays, |out, array| {
            crate::fp::key_public::put_u64(out, array.name.len() as u64);
            crate::fp::key_public::put_bytes(out, array.name.as_bytes());
            crate::fp::key_public::put_u64(out, array.message as u64);
            crate::fp::key_public::put_u64(out, array.size_i);
            crate::fp::key_public::put_u64(out, array.size_c);
            crate::fp::key_public::put_vec(out, &array.blocks, |out, block| {
                for value in [
                    block.first_i,
                    block.first_c,
                    block.first_row,
                    block.first_place,
                    block.bits_i as u64,
                    block.bits_c as u64,
                    block.row_bits as u64,
                    block.place_bits as u64,
                    block.i_is_low as u64,
                ] {
                    crate::fp::key_public::put_u64(out, value);
                }
            });
        });
    });
}

pub const LABEL_FIRST_CHAIN: u64 = 0x46524d01;
pub const LABEL_PROOF_SHARED_CHAIN: u64 = 0x46524d03;
pub const LABEL_PROOF_SECOND_CHAIN: u64 = 0x46524d04;

pub struct RequestPublic {
    pub parameter_bytes: Vec<u8>,
    pub parameters: crate::fp::digest::Digest,
    pub statement: Vec<u8>,
}

impl RequestPublic {
    pub fn new(
        fixed: &RequestFixed,
        table: &crate::fp::link_public::IndexName,
        entry: u32,
        request: &[u8],
        file_words: &[u8],
        plan_bytes: &[u8],
    ) -> Self {
        let mut parameter_bytes = Vec::new();
        crate::fp::key_public::put_bytes(&mut parameter_bytes, plan_bytes);
        crate::fp::key_public::item(
            &mut parameter_bytes,
            "configuration",
            &fixed.configuration.to_le_bytes(),
        );
        crate::fp::key_public::item(&mut parameter_bytes, "codes", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_vec(&mut out, &fixed.codes.base, crate::fp::key_public::code);
            crate::fp::key_public::put_vec(&mut out, &fixed.codes.mask, crate::fp::key_public::code);
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "the test on the table", &{
            let mut out = Vec::new();
            crate::fp::key_public::lin_shape(&mut out, &fixed.lin);
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "request's shape", &{
            let mut out = Vec::new();
            req_shape(&mut out, &fixed.shape);
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "layout", &{
            let mut out = Vec::new();
            req_layout(&mut out, &fixed.layout);
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "links' identifiers", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_vec(&mut out, &fixed.identifiers, |out, identifier| {
                crate::fp::key_public::put_u64(out, *identifier);
            });
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "pending rows", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_u64(&mut out, fixed.pending_rows);
            out
        });
        crate::fp::key_public::item(&mut parameter_bytes, "binding tree", &{
            let mut out = Vec::new();
            crate::fp::key_public::binding(&mut out, &fixed.binding);
            out
        });
        let parameters = {
            let mut h = crate::fp::blake2::Hasher::new(crate::fp::blake2::PARAMETERS, 1, &[0; 8], 32);
            h.push(&parameter_bytes);
            h.finish()
        };
        let mut statement = Vec::new();
        crate::fp::key_public::item(
            &mut statement,
            "table",
            &crate::fp::key_public::index_name_bytes(table),
        );
        crate::fp::key_public::item(&mut statement, "stored copy", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_u64(&mut out, entry as u64);
            out
        });
        crate::fp::key_public::item(&mut statement, "request", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_u64(&mut out, request.len() as u64);
            crate::fp::key_public::put_bytes(&mut out, request);
            out
        });
        crate::fp::key_public::item(&mut statement, "the request file's words", &{
            let mut out = Vec::new();
            crate::fp::key_public::put_u64(&mut out, file_words.len() as u64);
            crate::fp::key_public::put_bytes(&mut out, file_words);
            out
        });
        Self {
            parameter_bytes,
            parameters,
            statement,
        }
    }

    pub fn first_query(&self, salt: &crate::fp::digest::Digest) -> crate::fp::chain::FirstQuery {
        let run = {
            let mut h = crate::fp::blake2::Hasher::new(
                crate::fp::blake2::RUN,
                LABEL_FIRST_CHAIN,
                &[0; 8],
                32,
            );
            h.push_digest(&self.parameters);
            h.push_digest(salt);
            h.finish()
        };
        crate::fp::chain::FirstQuery {
            label: LABEL_FIRST_CHAIN,
            parameters: self.parameters,
            statement: self.statement.clone(),
            record: Vec::new(),
            salted: true,
            run: Some(run),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepted_layout_matches_oracle_shapes_and_arrays() {
        let bytes = include_bytes!(
            "../../spikes/flashprove/examples/accepted-proof/config/layout.txt"
        );
        let (links, layout) = request_layout(bytes, &accepted_declarations(), 0, 0, 1).unwrap();
        assert_eq!(links.len(), 5);
        assert_eq!(
            links.iter().map(|link| link.messages.len()).collect::<Vec<_>>(),
            [5, 3, 5, 3, 3]
        );
        assert_eq!(links[0].messages[0].shape.main_rows, 61);
        assert_eq!(links[0].messages[0].shape.mask_rows, 1);
        assert_eq!(links[0].messages[1].life, 1);
        assert_eq!(links[0].messages[2].life, 2);
        assert_eq!(links[4].plain, 1);
        assert_eq!(layout.links[0].len(), 12);
        let kept = layout.find(0, "M7").unwrap();
        assert_eq!((kept.message, kept.size_i, kept.size_c), (2, 2731, 20));
        assert_eq!(kept.blocks.len(), 14);
        let embedding = layout.find(0, "M0").unwrap();
        assert_eq!(embedding.blocks.len(), 49);
        assert_eq!(embedding.blocks[0].first_place, 114688);
        assert!(embedding.blocks[0].i_is_low);
        assert_eq!(layout.find(4, "ZZ").unwrap().size_c, 81920);
    }

    #[test]
    fn malformed_layout_indices_and_blocks_are_rejected() {
        let bytes = include_bytes!(
            "../../spikes/flashprove/examples/accepted-proof/config/layout.txt"
        );
        let declarations = accepted_declarations();
        let mut zero = bytes.to_vec();
        zero[0] = b'0';
        assert!(request_layout(&zero, &declarations, 0, 0, 1).is_err());
        let mut malformed_group = bytes.to_vec();
        let first_group = malformed_group.iter().position(|&byte| byte == b'\n').unwrap() + 1;
        malformed_group[first_group] = b'2';
        assert!(request_layout(&malformed_group, &declarations, 0, 0, 1).is_err());
    }

    #[test]
    fn accepted_first_query_matches_oracle() {
        let policy = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let leaf = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin");
        let request_file = include_bytes!("../../spikes/flashprove/examples/accepted-proof/example.req");
        let configuration = crate::fp::heads::configuration_digest(&[policy, layout, kl]);
        let (root, record, _) = crate::fp::heads::parse_leaf_record(leaf, 256).unwrap();
        let table = crate::fp::link_public::IndexName::new(&root, &configuration, record);
        let fixed = accepted_request_fixed(configuration, layout);
        let public = RequestPublic::new(
            &fixed,
            &table,
            0,
            include_bytes!("../../spikes/plans/request-statement.bin"),
            request_file,
            include_bytes!("../../spikes/plans/request-first-plan.bin"),
        );
        let expected = include_bytes!("../../spikes/plans/request-first-parameters.bin");
        let mismatch = public.parameter_bytes.iter().zip(expected.iter()).position(|(a, b)| a != b);
        assert_eq!(public.parameter_bytes.len(), expected.len());
        assert_eq!(mismatch, None, "first mismatch at {mismatch:?}");
        let expected_statement = include_bytes!("../../spikes/plans/request-first-statement.bin");
        let statement_mismatch = public.statement.iter().zip(expected_statement.iter()).position(|(a, b)| a != b);
        assert_eq!(public.statement.len(), expected_statement.len());
        assert_eq!(statement_mismatch, None, "statement mismatch at {statement_mismatch:?}");
    }
}

