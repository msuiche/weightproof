//! LINK public inputs and first-query encoding (chain/public.h, iop/link.h).

use alloc::vec::Vec;

use crate::fp::{
    analysis::{stage_cube_bits, Stage},
    blake2::{Hasher, PARAMETERS, RUN},
    chain::FirstQuery,
    codes::CodeSize,
    digest::Digest,
    error::FpError,
    field::GOLDILOCKS_P,
    plan::{solve_entry_stages, EntryLayoutInput},
};

pub const LINK_LABEL: u64 = 0x4c4e4b01;

#[derive(Debug, Clone)]
pub struct IndexName {
    pub commitment: Digest,
    pub record: Vec<[u64; 3]>,
}

impl IndexName {
    pub fn new(root: &Digest, _configuration: &Digest, record: Vec<[u64; 3]>) -> Self {
        Self {
            commitment: *root,
            record,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct PlanRound {
    kind: u64,
    count: u64,
    work: u64,
    message_bytes: u64,
    roots: u64,
    answers: u64,
    draw_e: u64,
    draw_nonzero: u64,
    positions: u64,
    log_n: u64,
    draw_trits: u64,
    rows: u64,
}

fn round(kind: u64) -> PlanRound {
    PlanRound {
        kind,
        count: 1,
        ..PlanRound::default()
    }
}

fn ceil_log2(x: u64) -> u64 {
    if x <= 1 {
        0
    } else {
        64 - (x - 1).leading_zeros() as u64
    }
}

fn rounds(input: &EntryLayoutInput, stages: &[Stage]) -> Vec<PlanRound> {
    let mut out = Vec::with_capacity(25);
    out.push(PlanRound {
        kind: 7,
        draw_e: 20,
        draw_nonzero: 1,
        ..round(7)
    });
    out.push(PlanRound {
        kind: 1,
        draw_e: 8,
        ..round(1)
    });
    out.push(PlanRound {
        kind: 3,
        message_bytes: 32,
        roots: 1,
        draw_e: input.setting.samples
            * (ceil_log2(input.folded_message_rows) + ceil_log2(stages[0].g.main.ell) + 1),
        draw_nonzero: input.setting.samples,
        ..round(3)
    });
    out.push(PlanRound {
        kind: 0,
        message_bytes: 48,
        answers: 2 * input.setting.samples,
        ..round(0)
    });
    out.push(PlanRound {
        kind: 4,
        work: 32,
        positions: input.columns_first + input.columns_further,
        log_n: 23,
        ..round(4)
    });
    out.push(PlanRound {
        kind: 1,
        draw_e: 19,
        ..round(1)
    });
    out.push(PlanRound {
        kind: 1,
        draw_e: 1,
        rows: 107,
        ..round(1)
    });
    for i in 0..stages.len() {
        let g = &stages[i].g;
        if i + 1 < stages.len() {
            out.push(PlanRound {
                kind: 2,
                message_bytes: 24,
                draw_e: 1,
                ..round(2)
            });
            for _ in 1..=g.k {
                out.push(PlanRound {
                    kind: 1,
                    message_bytes: 48,
                    draw_e: 1,
                    ..round(1)
                });
            }
        }
        if i + 1 < stages.len() {
            let next = &stages[i + 1].g;
            let t = stages[i].t;
            out.push(PlanRound {
                kind: 3,
                message_bytes: 32,
                roots: 1,
                draw_e: input.setting.samples * (stage_cube_bits(next) as u64 + 1),
                draw_nonzero: input.setting.samples,
                ..round(3)
            });
            out.push(PlanRound {
                kind: 4,
                work: 32,
                message_bytes: 24,
                answers: 1,
                draw_e: 1,
                positions: t,
                log_n: g.main.log_n as u64,
                rows: 1 + input.setting.samples + (g.slots + 2) * t,
                ..round(4)
            });
        } else {
            let clear = g.main.coordinates() * g.slots + g.mask.coordinates();
            out.push(PlanRound {
                kind: 1,
                message_bytes: 24,
                draw_e: 1,
                ..round(1)
            });
            out.push(PlanRound {
                kind: 4,
                work: 32,
                message_bytes: clear * 24,
                positions: stages[i].t,
                log_n: g.main.log_n as u64,
                ..round(4)
            });
        }
    }
    out
}

fn put_u64(out: &mut Vec<u8>, x: u64) {
    out.extend_from_slice(&x.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
}

fn put_vec<T>(out: &mut Vec<u8>, values: &[T], mut put: impl FnMut(&mut Vec<u8>, &T)) {
    put_u64(out, values.len() as u64);
    for value in values {
        put(out, value);
    }
}

fn item(out: &mut Vec<u8>, name: &str, value: &[u8]) {
    put_u64(out, name.len() as u64);
    put_bytes(out, name.as_bytes());
    put_u64(out, value.len() as u64);
    put_bytes(out, value);
}

fn put_code(out: &mut Vec<u8>, code: &CodeSize) {
    put_u64(out, code.log_n as u64);
    put_u64(out, code.ell);
    put_u64(out, code.r);
}

fn write_plan(out: &mut Vec<u8>, input: &EntryLayoutInput, stages: &[Stage]) {
    item(out, "the format's version", &(0x31323848u64).to_le_bytes());
    for x in [2, 1, 40, 8, 0, input.setting.samples, 1, 25] {
        put_u64(out, x);
    }
    for r in rounds(input, stages) {
        let name = match r.kind {
            0 => "SEND answer",
            1 => {
                if r.draw_e == 8 {
                    "LINK point"
                } else if r.draw_e == 19 {
                    "LINK phi"
                } else if r.rows == 107 {
                    "BAT"
                } else if r.positions == 0 && r.message_bytes == 24 {
                    "BASE lambda"
                } else {
                    "FOLD round"
                }
            }
            2 => "FOLD epsilon",
            3 => "SEND seed",
            4 => {
                if r.positions == 102 {
                    "LINK"
                } else if r.message_bytes == 24 {
                    "CS"
                } else {
                    "BASE"
                }
            }
            7 => "RECORD",
            _ => unreachable!(),
        };
        put_u64(out, name.len() as u64);
        put_bytes(out, name.as_bytes());
        for x in [
            r.count,
            r.work,
            r.message_bytes,
            r.roots,
            r.answers,
            r.draw_e,
            r.draw_nonzero,
            r.positions,
            r.log_n,
            r.draw_trits,
            r.rows,
        ] {
            put_u64(out, x);
        }
    }
}

pub(crate) fn unique_codes(stages: &[Stage], mask: bool) -> (Vec<CodeSize>, Vec<u64>) {
    let mut unique = Vec::new();
    let mut ids = Vec::with_capacity(stages.len());
    for stage in stages {
        let code = if mask { stage.g.mask } else { stage.g.main };
        let id = match unique.iter().position(|candidate| *candidate == code) {
            Some(id) => id,
            None => {
                unique.push(code);
                unique.len() - 1
            }
        };
        ids.push(id as u64);
    }
    (unique, ids)
}

fn write_link_shape(out: &mut Vec<u8>, input: &EntryLayoutInput, stages: &[Stage]) {
    // IndexShape: table code, v, answers, salt bytes.
    put_u64(out, 23);
    put_u64(out, 524288);
    put_u64(out, 65535);
    put_u64(out, 8);
    put_u64(out, 1);
    put_u64(out, 40);
    put_u64(out, input.columns_first);
    put_u64(out, input.columns_further);
    put_u64(out, 0); // first group's code id
    put_u64(out, 0); // first group's mask-code id
    put_u64(out, input.folded_message_rows);
    put_u64(out, 1); // rows_high
    put_u64(out, 0); // leaf_class
    put_u64(out, u64::from(input.masks_ahead));
    put_u64(out, (stages.len() - 1) as u64);
    let (_, base_ids) = unique_codes(stages, false);
    let (_, mask_ids) = unique_codes(stages, true);
    for (index, stage) in stages.iter().enumerate().skip(1) {
        put_u64(out, base_ids[index]);
        put_u64(out, mask_ids[index]);
        put_u64(out, stage.g.k as u64);
        put_u64(out, stage.g.h as u64);
    }
    put_u64(out, u64::from(input.masks_ride));
}

fn write_index_name(out: &mut Vec<u8>, table: &IndexName) {
    put_u64(out, 32);
    put_bytes(out, &table.commitment.to_le_bytes());
    put_vec(out, &table.record, |out, element| {
        for coordinate in element {
            put_u64(out, *coordinate);
        }
    });
}

/// All bytes fixed before LINK's prover speaks.
#[derive(Debug, Clone)]
pub struct LinkPublic {
    pub parameter_bytes: Vec<u8>,
    pub parameters: Digest,
    pub statement: Vec<u8>,
}

impl LinkPublic {
    pub fn new(
        input: &EntryLayoutInput,
        configuration: &Digest,
        first: &IndexName,
        further: &IndexName,
    ) -> Result<Self, FpError> {
        for element in first.record.iter().chain(further.record.iter()) {
            for coordinate in element {
                if *coordinate >= GOLDILOCKS_P {
                    return Err(FpError::NonCanonicalField);
                }
            }
        }
        let stages = solve_entry_stages(input)?;
        let mut bytes = Vec::new();
        write_plan(&mut bytes, input, &stages);
        item(&mut bytes, "configuration", &configuration.to_le_bytes());
        let mut codes = Vec::new();
        let (base, _) = unique_codes(&stages, false);
        let (mask, _) = unique_codes(&stages, true);
        put_vec(&mut codes, &base, put_code);
        put_vec(&mut codes, &mask, put_code);
        item(&mut bytes, "codes", &codes);
        let mut shape = Vec::new();
        write_link_shape(&mut shape, input, &stages);
        item(&mut bytes, "the link", &shape);
        let parameters = {
            let mut h = Hasher::new(PARAMETERS, 1, &[0; 8], 32);
            h.push(&bytes);
            h.finish()
        };
        let mut statement = Vec::new();
        item(&mut statement, "the first table", &{
            let mut value = Vec::new();
            write_index_name(&mut value, first);
            value
        });
        item(&mut statement, "the further table", &{
            let mut value = Vec::new();
            write_index_name(&mut value, further);
            value
        });
        Ok(Self {
            parameter_bytes: bytes,
            parameters,
            statement,
        })
    }

    pub fn first_query(&self, salt: &Digest) -> FirstQuery {
        let run = {
            let mut h = Hasher::new(RUN, LINK_LABEL, &[0; 8], 32);
            h.push_digest(&self.parameters);
            h.push_digest(salt);
            h.finish()
        };
        FirstQuery {
            label: LINK_LABEL,
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
    use crate::fp::{
        chain::Chain,
        heads::{configuration_digest, parse_key_record, parse_leaf_record},
        plan::accepted_example_entry_input,
    };

    #[test]
    fn public_encoding_matches_oracle_first_query() {
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
        let first = IndexName::new(&first_root, &configuration, first_record);
        let further = IndexName::new(&further_root, &configuration, further_record);
        let public = LinkPublic::new(
            &accepted_example_entry_input(),
            &configuration,
            &first,
            &further,
        )
        .unwrap();
        assert_eq!(
            public.parameters.to_hex(),
            "024a450069cd46ebdeb048d9ec5c3860e8eb16cb4be3a42c0d72f3c33e7ce6d7"
        );
        assert_eq!(public.statement.len(), 12448);

        let salt = Digest::from_le_bytes(leaf[6184..6216].try_into().unwrap());
        let query = public.first_query(&salt);
        assert_eq!(
            query.run.unwrap().to_hex(),
            "5ad247006102748530196e7ba735803c465ed420e2a88a60db72a9d1eeadb009"
        );
        let chain = Chain::new(query);
        assert_eq!(
            chain.value().to_hex(),
            "df33726b2f68c483aa03de095223767d844818c0bf5365331ed87d66617d6e0a"
        );
    }
}
