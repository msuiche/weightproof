//! LIN's request-first and shared-claim construction.

use crate::fp::{
    batt::batt_verify,
    blake2::Hasher,
    claims::{GroupId, OracleId, Part, Row, Statement},
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    key_public::{LinBindingShape, LinShape},
    psc::PscChannel,
    tensor::{tensor_below, Tensor},
    transcript::Cursor,
    wf::TensorRow,
};
use alloc::vec::Vec;

#[derive(Debug, Clone)]
pub struct LinClaim {
    pub g: Vec<Tensor>,
    pub psi: Vec<E>,
    pub masks: Vec<crate::fp::claims::Unit>,
    pub target: E,
}

#[derive(Debug)]
pub struct LinFirst {
    pub w: OracleId,
    pub split: Option<OracleId>,
    pub own: GroupId,
    pub mask: Option<OracleId>,
    pub claim: LinClaim,
}

pub fn member_claim(row: TensorRow) -> LinClaim {
    LinClaim {
        g: row.g,
        psi: row.psi,
        masks: row
            .masks
            .into_iter()
            .map(|mask| crate::fp::claims::Unit {
                oracle: mask.slot.oracle,
                part: Part::Mask,
                index: mask.slot.index,
                coefficient: mask.coeff,
            })
            .collect(),
        target: row.target,
    }
}

pub fn lin_eval_row(
    _shape: &LinShape,
    claim: &LinClaim,
    w: OracleId,
    lay: crate::fp::claims::VectorLay,
) -> Row {
    let mut row = Row::default();
    row.units = claim.masks.clone();
    row.target = claim.target;
    row.add(w, Part::Main, lay.on_table(Tensor::eq(&claim.psi, E::ONE)));
    row
}

pub fn lin_first_verify<C: PscChannel>(
    statement: &mut Statement,
    _channel: &mut C,
    shape: &LinShape,
    claim: LinClaim,
    w: OracleId,
    mask: Option<OracleId>,
) -> Result<LinFirst, FpError> {
    let own = statement.new_group(shape.w.code, shape.own_mask_code);
    let mut inputs: Vec<OracleId> = claim.masks.iter().map(|unit| unit.oracle).collect();
    inputs.sort_unstable();
    inputs.dedup();
    for oracle in inputs {
        statement.join(own, oracle);
    }
    statement.join(own, w);
    let lay = crate::fp::claims::VectorLay {
        row_bits: crate::fp::tensor::ceil_log2(shape.w.main_rows),
        place_bits: crate::fp::code_eval::log_ell(&statement.base[shape.w.code as usize]),
        rows_high: shape.w_rows_high,
    };
    statement.add_row(own, lin_eval_row(shape, &claim, w, lay));
    Ok(LinFirst {
        w,
        split: None,
        own,
        mask,
        claim,
    })
}

pub fn member_first_verify<C: PscChannel>(
    statement: &mut Statement,
    channel: &mut C,
    shape: &LinShape,
    index_masks: OracleId,
    pending: &mut [TensorRow],
    w: OracleId,
    mask: Option<OracleId>,
) -> Result<LinFirst, FpError> {
    let mask_shape = &statement.oracles[index_masks as usize].shape;
    if mask_shape.main_rows != 0
        || mask_shape.mask_rows == 0
        || mask_shape.answers == 0
        || mask_shape.mask_code != shape.own_mask_code
    {
        return Err(FpError::Range {
            context: "member-index-masks",
        });
    }
    if pending.is_empty() || pending[0].psi.len() != shape.k_col() as usize {
        return Err(FpError::Range {
            context: "member-pending-shape",
        });
    }
    for row in pending.iter() {
        if row.psi.len() != shape.k_col() as usize
            || row.g.iter().any(|tensor| tensor.vars() != shape.k_row())
            || row.masks.iter().any(|mask| mask.slot.oracle != index_masks)
        {
            return Err(FpError::Range {
                context: "member-pending-shape",
            });
        }
    }
    let one = if pending.len() > 1 {
        batt_verify(channel, pending, statement.masks(index_masks))?
    } else {
        pending[0].clone()
    };
    lin_first_verify(statement, channel, shape, member_claim(one), w, mask)
}

#[derive(Debug, Clone, Copy)]
pub struct LinBindingRoot {
    pub digest: Digest,
    pub places: u64,
}

fn binding_leaf(
    shape: &LinBindingShape,
    configuration: &Digest,
    place: u64,
    seal: &Digest,
    salt: &[u8],
) -> Digest {
    let tag = 173 | (4u64 << 32) | ((shape.salt_bytes as u64 / 8) << 56);
    let mut h = Hasher::new(
        tag,
        (1u64 << shape.log_leaves) + place,
        &configuration.0,
        32,
    );
    h.push_digest(seal);
    h.push(salt);
    h.finish()
}

fn binding_vertex(configuration: &Digest, left: &Digest, right: &Digest) -> Digest {
    let mut h = Hasher::new(178, 0, &configuration.0, 32);
    h.push_digest(left);
    h.push_digest(right);
    h.finish()
}

pub(crate) fn read_binding_root(
    cursor: &mut Cursor,
    shape: &LinBindingShape,
    configuration: &Digest,
    seal: &Digest,
) -> Result<LinBindingRoot, FpError> {
    let place = u64::from_le_bytes(cursor.take("place in the batch", "")?.try_into().map_err(
        |_| FpError::BadLength {
            context: "binding-place",
        },
    )?);
    if place >= (1u64 << shape.log_leaves) {
        return Err(FpError::Range {
            context: "binding-place",
        });
    }
    let salt = cursor.take("salt of the place's leaf", "")?;
    if salt.len() != shape.salt_bytes as usize {
        return Err(FpError::BadLength {
            context: "binding-salt",
        });
    }
    let path = if shape.log_leaves == 0 {
        &[][..]
    } else {
        cursor.take("path in the batch", "")?
    };
    if path.len() != shape.log_leaves as usize * 32 {
        return Err(FpError::BadLength {
            context: "binding-path",
        });
    }
    let mut root = binding_leaf(shape, configuration, place, seal, salt);
    let mut at = place;
    for sibling in path.chunks_exact(32) {
        let sibling = Digest::from_le_bytes(sibling.try_into().unwrap());
        root = if at & 1 != 0 {
            binding_vertex(configuration, &sibling, &root)
        } else {
            binding_vertex(configuration, &root, &sibling)
        };
        at >>= 1;
    }
    Ok(LinBindingRoot {
        digest: root,
        places: 1u64 << shape.log_leaves,
    })
}

pub fn lin_fold_row(
    shape: &LinShape,
    psi: &[E],
    zeta: &[E],
    u: crate::fp::claims::FoldedRow,
    omega: OracleId,
    omega_lay: crate::fp::claims::VectorLay,
) -> Row {
    let mut row = Row::default();
    for tensor in tensor_below(
        &Tensor::eq(zeta, E::ONE),
        0,
        shape.k_row(),
        shape.matrix_rows,
    ) {
        row.add(omega, Part::Main, omega_lay.on_table(tensor));
    }
    let mut on_u = Tensor::eq(&psi[..shape.k_lo as usize], -E::ONE);
    for &challenge in zeta {
        on_u.push(E::ONE - challenge, challenge);
    }
    for tensor in tensor_below(&on_u, shape.k_lo, shape.k_row(), shape.matrix_rows) {
        row.add(u.oracle, Part::Main, u.lay.on_table(tensor));
    }
    row
}

pub fn lin_dot_row(
    shape: &LinShape,
    weights: &[Tensor],
    psi: &[E],
    omega: OracleId,
    omega_lay: crate::fp::claims::VectorLay,
    w: OracleId,
    w_lay: crate::fp::claims::VectorLay,
) -> Row {
    let mut row = Row::default();
    for tensor in weights {
        for piece in tensor_below(tensor, 0, shape.k_row(), shape.matrix_rows) {
            row.add(omega, Part::Main, omega_lay.on_table(piece));
        }
    }
    row.add(w, Part::Main, w_lay.on_table(Tensor::eq(psi, -E::ONE)));
    row
}
