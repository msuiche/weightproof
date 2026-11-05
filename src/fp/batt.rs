//! Batching of matrix claims (BATT).

use crate::fp::{
    error::FpError,
    field::ExtensionElement as E,
    mask::MaskLedger,
    psc::{psc_verify, PscChannel, PscOutput, SumClaim, Summand, Weight},
    wf::TensorRow,
};
use alloc::{vec, vec::Vec};

fn eq_eval(point: &[E], at: &[E]) -> E {
    point.iter().zip(at).fold(E::ONE, |acc, (&x, &y)| {
        acc * (x * y + (E::ONE - x) * (E::ONE - y))
    })
}

struct BattSummand {
    chi: Vec<E>,
    psi: Vec<Vec<E>>,
}
impl Summand for BattSummand {
    fn shown(&self) -> usize {
        0
    }
    fn degree(&self) -> usize {
        0
    }
    fn kept(&self) -> usize {
        self.chi.len()
    }
    fn at(&self, point: &[E], _values: &[E]) -> Vec<E> {
        let mut out = vec![E::ZERO];
        for (chi, psi) in self.chi.iter().zip(&self.psi) {
            out.push(*chi * eq_eval(psi, point));
        }
        out
    }
}

pub struct BattClaim {
    pub chi: Vec<E>,
    pub psi: Vec<Vec<E>>,
    pub claim: SumClaim,
}

pub fn batt_claim(rows: &mut [TensorRow], x: E) -> BattClaim {
    let k_col = rows[0].psi.len();
    let mut chi = Vec::new();
    let mut psi = Vec::new();
    let mut power = E::ONE;
    let mut claim = SumClaim {
        vars: vec![Weight::plain(3); k_col],
        masks: vec![],
        target: E::ZERO,
    };
    for row in rows {
        assert_eq!(row.psi.len(), k_col);
        chi.push(power);
        psi.push(row.psi.clone());
        claim.target += power * row.target;
        for mask in &row.masks {
            claim.masks.push(crate::fp::mask::MaskTerm {
                slot: mask.slot,
                coeff: power * mask.coeff,
            });
        }
        power *= x;
    }
    BattClaim { chi, psi, claim }
}

pub fn batt_row(rows: &[TensorRow], out: PscOutput) -> TensorRow {
    let mut row = TensorRow {
        g: vec![],
        psi: out.point,
        masks: out.row_masks,
        target: out.row_target,
    };
    for term in out.row_maps {
        for tensor in &rows[term.map].g {
            let mut tensor = tensor.clone();
            tensor.scalar *= term.coeff;
            row.g.push(tensor);
        }
    }
    row
}

pub fn batt_verify<C: PscChannel>(
    channel: &mut C,
    rows: &mut [TensorRow],
    ledger: &mut MaskLedger,
) -> Result<TensorRow, FpError> {
    let k_col = rows[0].psi.len();
    let x = channel.close_algebraic(false)?;
    let claim = batt_claim(rows, x);
    let fresh = ledger.take_many(1 + 2 * k_col);
    let output = psc_verify(
        channel,
        claim.claim,
        &BattSummand {
            chi: claim.chi.clone(),
            psi: claim.psi.clone(),
        },
        fresh,
    )?;
    Ok(batt_row(rows, output))
}
