//! Public key/LIN/WF shapes and their canonical first-query encodings.

use crate::fp::{
    blake2::{Hasher, PARAMETERS, RUN},
    chain::FirstQuery,
    codes::CodeSize,
    digest::Digest,
    field::FieldElement,
    link_public::IndexName,
};
use alloc::{string::String, vec::Vec};

pub const LABEL_ONE_TIME_CHAIN: u64 = 0x46524d02;
pub const LIN_LABEL_SHARED: u64 = 0x4c494e01;
pub const LIN_LABEL_SECOND: u64 = 0x4c494e02;

#[derive(Debug, Clone)]
pub struct OracleNotice {
    pub shape: crate::fp::claims::OracleShape,
    pub leaf_class: u32,
}

#[derive(Debug, Clone)]
pub struct IoppStage {
    pub code: u32,
    pub mask_code: u32,
    pub k: u32,
    pub h: u32,
}

#[derive(Debug, Clone)]
pub struct IoppChoice {
    pub masks_ride: bool,
}

#[derive(Debug, Clone)]
pub struct IndexShape {
    pub code: CodeSize,
    pub v: u32,
    pub answers: u32,
    pub salt_bytes: u32,
}

#[derive(Debug, Clone)]
pub struct FoldeqShape {
    pub index: IndexShape,
    pub u: crate::fp::claims::OracleShape,
    pub rows_high: bool,
    pub leaf_class: u32,
    pub columns: u64,
}

#[derive(Debug, Clone)]
pub struct LinShape {
    pub fold: FoldeqShape,
    pub k_lo: u32,
    pub matrix_rows: u64,
    pub omega: crate::fp::claims::OracleShape,
    pub w: crate::fp::claims::OracleShape,
    pub omega_rows_high: bool,
    pub w_rows_high: bool,
    pub shared_split: crate::fp::claims::OracleShape,
    pub own_split: crate::fp::claims::OracleShape,
    pub omega_leaf_class: u32,
    pub shared_split_leaf_class: u32,
    pub w_leaf_class: u32,
    pub own_split_leaf_class: u32,
    pub shared_mask_code: u32,
    pub own_mask_code: u32,
    pub masks_ahead: bool,
    pub test_u: Vec<IoppStage>,
    pub test_omega: Vec<IoppStage>,
    pub test_own: Vec<IoppStage>,
    pub test_own_omega: Vec<IoppStage>,
    pub choice: IoppChoice,
}

impl LinShape {
    pub fn k_col(&self) -> u32 {
        self.k_lo + self.fold.index.v
    }
    pub fn k_row(&self) -> u32 {
        crate::fp::tensor::ceil_log2(self.matrix_rows)
    }
}

#[derive(Debug, Clone)]
pub struct WfRun {
    pub first_place: u64,
    pub first_row: u64,
    pub places: u64,
    pub stride: u64,
}

#[derive(Debug, Clone)]
pub struct WfPart {
    pub log_places: u32,
    pub runs: Vec<WfRun>,
}

#[derive(Debug, Clone)]
pub struct WfLookup {
    pub name: String,
    pub d: u32,
    pub mu: u32,
    pub live: u32,
    pub digit_weight: Vec<FieldElement>,
    pub offset: FieldElement,
    pub padding_key: FieldElement,
    pub parts: Vec<WfPart>,
    pub first_bound: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct WfPublic {
    pub lookups: Vec<WfLookup>,
    pub message: OracleNotice,
    pub message_id: u32,
    pub stages: Vec<IoppStage>,
    pub choice: IoppChoice,
}

#[derive(Debug, Clone)]
pub struct LinBindingShape {
    pub log_leaves: u32,
    pub salt_bytes: u32,
}

#[derive(Debug, Clone)]
pub struct Codes {
    pub base: Vec<CodeSize>,
    pub mask: Vec<CodeSize>,
}

#[derive(Debug, Clone)]
pub struct KeyFixed {
    pub configuration: Digest,
    pub codes: Codes,
    pub lin: LinShape,
    pub index_masks: OracleNotice,
    pub ranges: WfPublic,
    pub binding: LinBindingShape,
}

pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
pub fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
}
pub fn put_vec<T>(out: &mut Vec<u8>, values: &[T], mut write: impl FnMut(&mut Vec<u8>, &T)) {
    put_u64(out, values.len() as u64);
    for value in values {
        write(out, value);
    }
}
pub fn item(out: &mut Vec<u8>, name: &str, value: &[u8]) {
    put_u64(out, name.len() as u64);
    put_bytes(out, name.as_bytes());
    put_u64(out, value.len() as u64);
    put_bytes(out, value);
}

fn code(out: &mut Vec<u8>, value: &CodeSize) {
    put_u64(out, value.log_n as u64);
    put_u64(out, value.ell);
    put_u64(out, value.r);
}
fn oracle_shape(out: &mut Vec<u8>, value: &crate::fp::claims::OracleShape) {
    put_u64(
        out,
        if value.main_rows != 0 {
            value.code as u64
        } else {
            u32::MAX as u64
        },
    );
    put_u64(
        out,
        if value.mask_rows != 0 {
            value.mask_code as u64
        } else {
            u32::MAX as u64
        },
    );
    put_u64(out, value.main_rows);
    put_u64(
        out,
        if value.main_rows != 0 {
            value.main_deg as u64
        } else {
            1
        },
    );
    put_u64(out, value.mask_rows);
    put_u64(out, value.answers as u64);
}
fn notice(out: &mut Vec<u8>, value: &OracleNotice) {
    oracle_shape(out, &value.shape);
    put_u64(out, value.leaf_class as u64);
}
fn stage(out: &mut Vec<u8>, value: &IoppStage) {
    put_u64(out, value.code as u64);
    put_u64(out, value.mask_code as u64);
    put_u64(out, value.k as u64);
    put_u64(out, value.h as u64);
}
fn choice(out: &mut Vec<u8>, value: &IoppChoice) {
    put_u64(out, value.masks_ride as u64);
}
fn index_shape(out: &mut Vec<u8>, value: &IndexShape) {
    code(out, &value.code);
    put_u64(out, value.v as u64);
    put_u64(out, value.answers as u64);
    put_u64(out, value.salt_bytes as u64);
}
fn foldeq_shape(out: &mut Vec<u8>, value: &FoldeqShape) {
    index_shape(out, &value.index);
    oracle_shape(out, &value.u);
    put_u64(out, value.rows_high as u64);
    put_u64(out, value.leaf_class as u64);
    put_u64(out, value.columns);
}
fn lin_shape(out: &mut Vec<u8>, value: &LinShape) {
    foldeq_shape(out, &value.fold);
    put_u64(out, value.k_lo as u64);
    put_u64(out, value.matrix_rows);
    oracle_shape(out, &value.omega);
    oracle_shape(out, &value.w);
    put_u64(out, value.omega_rows_high as u64);
    put_u64(out, value.w_rows_high as u64);
    oracle_shape(out, &value.shared_split);
    oracle_shape(out, &value.own_split);
    put_u64(out, value.omega_leaf_class as u64);
    put_u64(out, value.shared_split_leaf_class as u64);
    put_u64(out, value.w_leaf_class as u64);
    put_u64(out, value.own_split_leaf_class as u64);
    put_u64(out, value.shared_mask_code as u64);
    put_u64(out, value.own_mask_code as u64);
    put_u64(out, value.masks_ahead as u64);
    put_vec(out, &value.test_u, stage);
    put_vec(out, &value.test_omega, stage);
    put_vec(out, &value.test_own, stage);
    put_vec(out, &value.test_own_omega, stage);
    choice(out, &value.choice);
}
fn wf_run(out: &mut Vec<u8>, value: &WfRun) {
    put_u64(out, value.first_place);
    put_u64(out, value.first_row);
    put_u64(out, value.places);
    put_u64(out, value.stride);
}
fn wf_part(out: &mut Vec<u8>, value: &WfPart) {
    put_u64(out, value.log_places as u64);
    put_vec(out, &value.runs, wf_run);
}
fn wf_lookup(out: &mut Vec<u8>, value: &WfLookup) {
    put_u64(out, value.d as u64);
    put_u64(out, value.mu as u64);
    put_u64(out, value.live as u64);
    put_vec(out, &value.digit_weight, |out, x| put_u64(out, x.0));
    put_u64(out, value.offset.0);
    put_u64(out, value.padding_key.0);
    put_vec(out, &value.parts, wf_part);
    put_vec(out, &value.first_bound, |out, x| put_u64(out, *x as u64));
}
fn wf_public(out: &mut Vec<u8>, value: &WfPublic) {
    put_vec(out, &value.lookups, wf_lookup);
    notice(out, &value.message);
    put_u64(out, value.message_id as u64);
    put_vec(out, &value.stages, stage);
    choice(out, &value.choice);
}
fn binding(out: &mut Vec<u8>, value: &LinBindingShape) {
    put_u64(out, value.log_leaves as u64);
    put_u64(out, value.salt_bytes as u64);
}

pub fn key_public_bytes(value: &KeyFixed) -> Vec<u8> {
    let mut out = Vec::new();
    put_bytes(&mut out, &value.configuration.to_le_bytes());
    put_vec(&mut out, &value.codes.base, code);
    put_vec(&mut out, &value.codes.mask, code);
    lin_shape(&mut out, &value.lin);
    notice(&mut out, &value.index_masks);
    wf_public(&mut out, &value.ranges);
    binding(&mut out, &value.binding);
    out
}

pub fn index_name_bytes(value: &IndexName) -> Vec<u8> {
    let mut out = Vec::new();
    put_u64(&mut out, 32);
    put_bytes(&mut out, &value.commitment.to_le_bytes());
    put_vec(&mut out, &value.record, |out, element| {
        for coordinate in element {
            put_u64(out, *coordinate);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fp::heads::configuration_digest;

    #[test]
    fn one_time_first_query_matches_oracle() {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let configuration = configuration_digest(&[policy, layout, kl]);
        let key = include_bytes!("../../spikes/flashprove/examples/accepted-proof/ivk/key.bin");
        let (root, record, _) = crate::fp::heads::parse_key_record(key, 256).unwrap();
        let fixed = crate::fp::key_fixture::accepted_fixed(configuration);
        let name = IndexName::new(&root, &configuration, record);
        let public = KeyPublic::new(
            &fixed,
            &name,
            include_bytes!("../../spikes/plans/key-one-time-plan.bin"),
        );
        assert_eq!(
            public.parameter_bytes.as_slice(),
            include_bytes!("../../spikes/plans/key-one-time-parameters.bin")
        );
        assert_eq!(
            public.statement.as_slice(),
            include_bytes!("../../spikes/plans/key-one-time-statement.bin")
        );
    }

    #[test]
    fn accepted_public_bytes_match_oracle() {
        let policy =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/policy.txt");
        let layout =
            include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/layout.txt");
        let kl = include_bytes!("../../spikes/flashprove/examples/accepted-proof/config/KL.bin");
        let configuration = configuration_digest(&[policy, layout, kl]);
        let fixed = crate::fp::key_fixture::accepted_fixed(configuration);
        assert_eq!(
            &key_public_bytes(&fixed)[32..],
            include_bytes!("../../spikes/plans/key-public.bin")
        );
    }
}

pub struct KeyPublic {
    pub parameter_bytes: Vec<u8>,
    pub parameters: Digest,
    pub statement: Vec<u8>,
}

impl KeyPublic {
    pub fn new(fixed: &KeyFixed, table: &IndexName, plan_bytes: &[u8]) -> Self {
        let mut parameter_bytes = Vec::new();
        put_bytes(&mut parameter_bytes, plan_bytes);
        item(
            &mut parameter_bytes,
            "configuration",
            &fixed.configuration.to_le_bytes(),
        );
        item(&mut parameter_bytes, "index-mask message", &{
            let mut out = Vec::new();
            notice(&mut out, &fixed.index_masks);
            out
        });
        item(&mut parameter_bytes, "ranges", &{
            let mut out = Vec::new();
            wf_public(&mut out, &fixed.ranges);
            out
        });
        item(&mut parameter_bytes, "binding tree", &{
            let mut out = Vec::new();
            binding(&mut out, &fixed.binding);
            out
        });
        item(&mut parameter_bytes, "codes", &{
            let mut out = Vec::new();
            put_vec(&mut out, &fixed.codes.base, code);
            put_vec(&mut out, &fixed.codes.mask, code);
            out
        });
        item(&mut parameter_bytes, "the test on the table", &{
            let mut out = Vec::new();
            lin_shape(&mut out, &fixed.lin);
            out
        });
        let parameters = {
            let mut h = Hasher::new(PARAMETERS, 1, &[0; 8], 32);
            h.push(&parameter_bytes);
            h.finish()
        };
        let mut statement = Vec::new();
        item(&mut statement, "table", &index_name_bytes(table));
        Self {
            parameter_bytes,
            parameters,
            statement,
        }
    }

    pub fn first_query(&self, salt: &Digest, label: u64) -> FirstQuery {
        let run = {
            let mut h = Hasher::new(RUN, label, &[0; 8], 32);
            h.push_digest(&self.parameters);
            h.push_digest(salt);
            h.finish()
        };
        FirstQuery {
            label,
            parameters: self.parameters,
            statement: self.statement.clone(),
            record: Vec::new(),
            salted: true,
            run: Some(run),
        }
    }
}
