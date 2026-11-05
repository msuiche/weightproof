//! Committer range proof (WF) built from LKP.

use crate::fp::{
    claims::{GroupId, OracleId, Part, Row, Statement},
    error::FpError,
    field::ExtensionElement as E,
    key_public::{LinShape, WfLookup, WfPart, WfPublic},
    lkp::{lkp_verify, LkpKeyBlock, LkpOutput, LkpPadding, LkpPublic, LkpTablePart},
    mask::MaskTerm,
    psc::PscChannel,
    tensor::{eq_table, Tensor},
};
use alloc::{format, vec, vec::Vec};

#[derive(Debug, Clone, Copy)]
pub struct WfMatrix {
    pub k_col: u32,
    pub rows: u64,
}
impl WfMatrix {
    pub fn k_row(&self) -> u32 {
        crate::fp::tensor::ceil_log2(self.rows)
    }
}
pub fn wf_matrix_of(table: &LinShape) -> WfMatrix {
    WfMatrix {
        k_col: table.k_lo + table.fold.index.v,
        rows: table.matrix_rows,
    }
}

pub fn wf_lookups(x: &WfPublic, matrix: WfMatrix) -> Vec<LkpPublic> {
    x.lookups
        .iter()
        .map(|lookup| {
            let mut padding = Vec::new();
            let mut parts = Vec::new();
            for (number, part) in lookup.parts.iter().enumerate() {
                parts.push(part.log_places + matrix.k_col);
                let stretch = |from: u64, to: u64, padding: &mut Vec<LkpPadding>| {
                    if from < to {
                        padding.push(LkpPadding {
                            part: number,
                            first: from << matrix.k_col,
                            log_step: 0,
                            length: (to - from) << matrix.k_col,
                            row: lookup.padding_key.0,
                        });
                    }
                };
                let mut upto = 0;
                for run in &part.runs {
                    stretch(upto, run.first_place, &mut padding);
                    upto = run.first_place + run.places;
                }
                stretch(upto, 1u64 << part.log_places, &mut padding);
            }
            LkpPublic {
                context: lookup.name.clone(),
                parts,
                d: lookup.d,
                mu: lookup.mu,
                live: lookup.live,
                h: 0,
                g: 0,
                shifted: false,
                read_from: vec![x.message_id],
                keys: vec![LkpKeyBlock {
                    first: 0,
                    log_trees: 0,
                    function: 0,
                    padding_row: 0,
                }],
                tables: vec![LkpTablePart {
                    first: 0,
                    log_rows: lookup.d,
                    function: 0,
                    rows: vec![],
                }],
                padding,
                last_key_level_order: lookup.first_bound.clone(),
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
pub struct WfBlock {
    pub lookup: usize,
    pub planes: (u32, u32),
    pub bits: u32,
    pub row: u64,
    pub first_place: u64,
}

pub fn lkp_live_blocks(live: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut first = 0;
    while first < live {
        let mut bits = 0;
        while first + (2 << bits) <= live && (first & ((2 << bits) - 1)) == 0 {
            bits += 1;
        }
        out.push((first, bits));
        first += 1 << bits;
    }
    out
}

pub fn wf_layout(x: &WfPublic, ell: u64) -> Vec<WfBlock> {
    let mut blocks = Vec::new();
    for (i, lookup) in x.lookups.iter().enumerate() {
        for planes in lkp_live_blocks(if lookup.live == 0 {
            1 << lookup.mu
        } else {
            lookup.live
        }) {
            blocks.push(WfBlock {
                lookup: i,
                planes,
                bits: planes.1 + lookup.d,
                row: 0,
                first_place: 0,
            });
        }
    }
    blocks.sort_by_key(|block| core::cmp::Reverse(block.bits));
    let mut row = 0;
    let mut place = 0;
    for block in &mut blocks {
        if place + (1u64 << block.bits) > ell {
            row += 1;
            place = 0;
        }
        block.row = row;
        block.first_place = place;
        place += 1u64 << block.bits;
    }
    blocks
}

pub fn lkp_on_block(v: &Tensor, mu: u32, block: (u32, u32)) -> Tensor {
    let mut part = Tensor::eq(&[], v.scalar);
    for t in 0..v.vars() {
        if t >= block.1 && t < mu {
            part.scalar *= if (block.0 >> t) & 1 != 0 {
                v.hi(t)
            } else {
                v.lo(t)
            };
        } else {
            part.push(v.lo(t), v.hi(t));
        }
    }
    part
}

pub fn wf_planes_at(
    x: &WfPublic,
    codes: &[crate::fp::codes::CodeSize],
    index: usize,
) -> impl Fn(&mut Row, Tensor) {
    let shape = &x.message.shape;
    let log_ell = crate::fp::code_eval::log_ell(&codes[shape.code as usize]);
    let row_bits = crate::fp::tensor::ceil_log2(shape.main_rows);
    let mu = x.lookups[index].mu;
    let oracle = x.message_id;
    let mine: Vec<WfBlock> = wf_layout(x, codes[shape.code as usize].ell)
        .into_iter()
        .filter(|block| block.lookup == index)
        .collect();
    move |row, tensor| {
        for block in &mine {
            let mut part = lkp_on_block(&tensor, mu, block.planes);
            part.insert_choice(
                block.bits,
                log_ell - block.bits,
                block.first_place >> block.bits,
            );
            part.insert_choice(0, row_bits, block.row);
            row.add(oracle, Part::Main, part);
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorRow {
    pub g: Vec<Tensor>,
    pub psi: Vec<E>,
    pub masks: Vec<MaskTerm>,
    pub target: E,
}

fn wf_weights_on_rows(
    k_row: u32,
    first_row: u64,
    values: &[E],
    first: u64,
    count: u64,
    scale: E,
    out: &mut Vec<Tensor>,
) {
    let end = first_row + count;
    let mut at = first_row;
    while at < end {
        let mut d = 0;
        while d < k_row && (at & ((2u64 << d) - 1)) == 0 && at + (2u64 << d) <= end {
            d += 1;
        }
        let mut tensor = Tensor::eq(&[], scale);
        if d == 0 {
            tensor.scalar = values[(first + at - first_row) as usize] * scale;
            tensor.insert_choice(0, k_row, at);
        } else {
            tensor.push_dense(d);
            tensor.insert_choice(d, k_row - d, at >> d);
            tensor.set_values(values.to_vec(), first + at - first_row);
        }
        out.push(tensor);
        at += 1u64 << d;
    }
}

pub fn wf_pending_row(
    matrix: WfMatrix,
    lookup: &WfLookup,
    part: &WfPart,
    claim: crate::fp::lkp::LkpKeys,
) -> TensorRow {
    assert_eq!(
        claim.point.len(),
        matrix.k_col as usize + part.log_places as usize
    );
    let e = eq_table(&claim.point[matrix.k_col as usize..]);
    let mut covered = E::ZERO;
    let mut row = TensorRow {
        g: vec![],
        psi: vec![],
        masks: claim.masks,
        target: claim.target,
    };
    for run in &part.runs {
        for i in 0..run.places {
            covered += e[(run.first_place + i) as usize];
        }
        for (q, &weight) in lookup.digit_weight.iter().enumerate() {
            wf_weights_on_rows(
                matrix.k_row(),
                run.first_row + q as u64 * run.stride,
                &e,
                run.first_place,
                run.places,
                E::from_base(weight),
                &mut row.g,
            );
        }
    }
    let mut handed = E::ZERO;
    for tensor in &claim.weights.tensors {
        let mut sum = tensor.scalar;
        for i in 0..tensor.vars() {
            sum *= tensor.lo(i) + tensor.hi(i);
        }
        handed += sum;
    }
    for &(_, coefficient) in &claim.weights.entries {
        handed += coefficient;
    }
    assert_eq!(handed, covered);
    row.psi = claim.point[..matrix.k_col as usize].to_vec();
    row.target -= covered * E::from_base(lookup.offset);
    row
}

#[derive(Debug)]
pub struct WfOutput {
    pub pending: Vec<TensorRow>,
    pub message: OracleId,
    pub group: GroupId,
}

pub fn wf_rows_verify<C: PscChannel>(
    statement: &mut Statement,
    channel: &mut C,
    fixed: &WfPublic,
    table: &LinShape,
    index_masks: OracleId,
) -> Result<WfOutput, FpError> {
    let matrix = wf_matrix_of(table);
    let group = statement.new_group(fixed.message.shape.code, fixed.message.shape.mask_code);
    statement.join(group, fixed.message_id);
    let lookups = wf_lookups(fixed, matrix);
    let codes = statement.base.clone();
    let mut pending = Vec::new();
    for (i, lookup) in lookups.iter().enumerate() {
        channel.set_context(&format!("{}, tie", fixed.lookups[i].name));
        let output: LkpOutput = lkp_verify(
            statement,
            channel,
            lookup,
            wf_planes_at(fixed, &codes, i),
            fixed.message_id,
            group,
            index_masks,
        )?;
        for (part, keys) in output.keys.into_iter().enumerate() {
            pending.push(wf_pending_row(
                matrix,
                &fixed.lookups[i],
                &fixed.lookups[i].parts[part],
                keys,
            ));
        }
    }
    Ok(WfOutput {
        pending,
        message: fixed.message_id,
        group,
    })
}
