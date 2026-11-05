//! LIN's request-first and shared-claim construction.

use crate::fp::{
    claims::{GroupId, OracleId, Part, Row, Statement},
    error::FpError,
    field::ExtensionElement as E,
    key_public::LinShape,
    psc::PscChannel,
    tensor::{tensor_below, Tensor},
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
    statement.join(own, 0);
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
