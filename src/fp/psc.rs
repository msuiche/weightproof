//! Hiding-polynomial sumcheck (PSC) verification.

use crate::fp::{
    claims::{Part, Row},
    error::FpError,
    field::ExtensionElement as E,
    mask::{MaskSlot, MaskTerm},
};
use alloc::{boxed::Box, vec, vec::Vec};

#[derive(Debug, Clone)]
pub struct Weight {
    pub d: usize,
    pub moment: Vec<E>,
    pub psi: Vec<E>,
    pub fixed: usize,
    pub fixed_inv: E,
}

fn inv(value: E) -> E {
    value.inverse()
}

pub fn poly_at(values: &[E], point: E) -> E {
    values
        .iter()
        .rev()
        .fold(E::ZERO, |acc, &coefficient| acc * point + coefficient)
}

impl Weight {
    fn checked(mut self) -> Self {
        assert!(!self.moment[self.fixed].is_zero());
        self.fixed_inv = inv(self.moment[self.fixed]);
        assert_eq!(self.apply(&self.psi), E::ONE);
        self
    }

    pub fn plain(d: usize) -> Self {
        let mut moment = vec![E::ONE; d];
        moment[0] = E::ONE + E::ONE;
        Self {
            d,
            moment,
            psi: vec![E::ZERO, E::ONE],
            fixed: 1,
            fixed_inv: E::ZERO,
        }
        .checked()
    }

    pub fn equality(z: E, d: usize) -> Self {
        let mut moment = vec![z; d];
        moment[0] = E::ONE;
        Self {
            d,
            moment,
            psi: vec![E::ONE],
            fixed: 0,
            fixed_inv: E::ZERO,
        }
        .checked()
    }

    pub fn at_zero(d: usize) -> Self {
        Self::equality(E::ZERO, d)
    }

    pub fn apply(&self, values: &[E]) -> E {
        values
            .iter()
            .zip(&self.moment)
            .fold(E::ZERO, |acc, (&x, &weight)| acc + weight * x)
    }

    pub fn complete(&self, values: &mut [E], claim: E) {
        let rest = values
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.fixed)
            .fold(E::ZERO, |acc, (i, &x)| acc + self.moment[i] * x);
        values[self.fixed] = (claim - rest) * self.fixed_inv;
    }
}

pub fn hp_entries(vars: &[Weight]) -> usize {
    1 + vars.iter().map(|v| v.d - 1).sum::<usize>()
}

pub fn hp_place(vars: &[Weight], l: usize, t: usize) -> usize {
    let at: usize = vars[..l].iter().map(|v| v.d - 1).sum();
    at + if l + 1 == vars.len() { t } else { t - 1 }
}

pub fn hp_form(vars: &[Weight], gamma: &[E]) -> Vec<E> {
    let mut out = vec![E::ZERO; hp_entries(vars)];
    let mut tail = E::ONE;
    for l in (0..vars.len()).rev() {
        let last = l + 1 == vars.len();
        let mut power = if last { tail } else { tail * gamma[l] };
        for t in if last { 0 } else { 1 }..vars[l].d {
            out[hp_place(vars, l, t)] = power;
            power *= gamma[l];
        }
        tail *= poly_at(&vars[l].psi, gamma[l]);
    }
    out
}

#[derive(Debug, Clone)]
pub struct SumClaim {
    pub vars: Vec<Weight>,
    pub masks: Vec<MaskTerm>,
    pub target: E,
}

pub trait Summand {
    fn shown(&self) -> usize;
    fn degree(&self) -> usize;
    fn kept(&self) -> usize {
        0
    }
    fn at(&self, point: &[E], end_values: &[E]) -> Vec<E>;
}

#[derive(Debug, Clone)]
pub struct KeptTerm {
    pub map: usize,
    pub coeff: E,
}

#[derive(Debug, Clone)]
pub struct PointClaim {
    pub point: Vec<E>,
    pub masks: Vec<MaskTerm>,
    pub value: E,
}

#[derive(Debug, Clone)]
pub struct PscOutput {
    pub point: Vec<E>,
    pub row_masks: Vec<MaskTerm>,
    pub row_maps: Vec<KeptTerm>,
    pub row_target: E,
    pub ends: Vec<PointClaim>,
}

pub trait PscChannel {
    fn recv(&mut self) -> Result<E, FpError>;
    fn recv_hiding_sum(&mut self) -> Result<E, FpError> {
        self.recv()
    }
    fn close_epsilon(&mut self) -> Result<E, FpError>;
    fn close_algebraic(&mut self, nonzero: bool) -> Result<E, FpError>;
    fn draw(&mut self, nonzero: bool) -> Result<E, FpError>;
    fn close_empty(&mut self) -> Result<(), FpError>;
    fn set_context(&mut self, _context: &str) {}
}

pub type MapAt<'a> = Box<dyn Fn(&mut Row, &[E], E) + 'a>;

pub fn line<C: PscChannel>(
    channel: &mut C,
    half0: PointClaim,
    half1: PointClaim,
) -> Result<PointClaim, FpError> {
    assert_eq!(half0.point, half1.point);
    let rho = channel.close_algebraic(false)?;
    let one_minus = E::ONE - rho;
    let mut point = half0.point;
    point.push(rho);
    let mut masks = Vec::new();
    for mask in half0.masks {
        masks.push(MaskTerm {
            slot: mask.slot,
            coeff: one_minus * mask.coeff,
        });
    }
    for mask in half1.masks {
        masks.push(MaskTerm {
            slot: mask.slot,
            coeff: rho * mask.coeff,
        });
    }
    Ok(PointClaim {
        point,
        masks,
        value: one_minus * half0.value + rho * half1.value,
    })
}

pub fn at_point(claim: PointClaim, degrees: &[usize]) -> SumClaim {
    assert_eq!(claim.point.len(), degrees.len());
    SumClaim {
        vars: claim
            .point
            .iter()
            .zip(degrees)
            .map(|(&point, &degree)| Weight::equality(point, degree))
            .collect(),
        masks: claim.masks,
        target: claim.value,
    }
}

pub fn psc_row_u(out: &mut PscOutput, kept: &[MapAt<'_>], name: &str) -> Row {
    assert_eq!(kept.len(), out.row_maps.len());
    assert!(!out.row_masks.is_empty());
    let mut row = Row::default();
    for mask in out.row_masks.drain(..) {
        row.add_unit(mask.slot.oracle, Part::Mask, mask.slot.index, mask.coeff);
    }
    for term in &out.row_maps {
        kept[term.map](&mut row, &out.point, term.coeff);
    }
    row.target = out.row_target;
    let _ = name;
    row
}

pub fn point_row(claim: PointClaim, map: MapAt<'_>, name: &str) -> Row {
    let mut row = Row::default();
    map(&mut row, &claim.point, E::ONE);
    for mask in claim.masks {
        row.add_unit(mask.slot.oracle, Part::Mask, mask.slot.index, mask.coeff);
    }
    row.target = claim.value;
    let _ = name;
    row
}

pub fn psc_vars(claim: &SumClaim, summand: &dyn Summand) -> Vec<Weight> {
    let mut vars = claim.vars.clone();
    if summand.shown() != 0 {
        vars.push(Weight::at_zero(summand.degree() + 1));
    }
    vars
}

pub fn psc_masks(claim: &SumClaim, summand: &dyn Summand) -> usize {
    hp_entries(&psc_vars(claim, summand)) + summand.shown()
}

pub fn psc_rows(
    claim: SumClaim,
    summand: &dyn Summand,
    vars: &[Weight],
    fresh: Vec<MaskSlot>,
    eps: E,
    challenges: &[E],
    last: E,
    end_values: &[E],
) -> PscOutput {
    let point = challenges[..claim.vars.len()].to_vec();
    let a = summand.at(&point, end_values);
    assert_eq!(a.len(), 1 + summand.kept());
    let form = hp_form(vars, challenges);
    let mut row_masks = Vec::new();
    for (i, coefficient) in form.iter().enumerate() {
        row_masks.push(MaskTerm {
            slot: fresh[i],
            coeff: *coefficient,
        });
    }
    let mut psis = eps;
    for (i, weight) in vars.iter().enumerate() {
        psis *= poly_at(&weight.psi, challenges[i]);
    }
    for mask in claim.masks {
        row_masks.push(MaskTerm {
            slot: mask.slot,
            coeff: psis * mask.coeff,
        });
    }
    let row_maps = (0..summand.kept())
        .map(|i| KeptTerm {
            map: i,
            coeff: eps * a[1 + i],
        })
        .collect();
    let ends = (0..summand.shown())
        .map(|j| PointClaim {
            point: point.clone(),
            masks: vec![MaskTerm {
                slot: fresh[form.len() + j],
                coeff: *challenges.last().unwrap(),
            }],
            value: end_values[j],
        })
        .collect();
    PscOutput {
        point,
        row_masks,
        row_maps,
        row_target: last - eps * a[0],
        ends,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Product;
    impl Summand for Product {
        fn shown(&self) -> usize {
            2
        }
        fn degree(&self) -> usize {
            2
        }
        fn at(&self, _point: &[E], end_values: &[E]) -> Vec<E> {
            vec![end_values[0] * end_values[1]]
        }
    }

    struct Channel {
        values: Vec<E>,
        challenges: Vec<E>,
        next: usize,
        round: usize,
    }
    impl PscChannel for Channel {
        fn recv(&mut self) -> Result<E, FpError> {
            let value = self.values[self.next];
            self.next += 1;
            Ok(value)
        }
        fn close_epsilon(&mut self) -> Result<E, FpError> {
            Ok(E([17, 19, 23]))
        }
        fn close_algebraic(&mut self, _nonzero: bool) -> Result<E, FpError> {
            let value = self.challenges[self.round % self.challenges.len()];
            self.round += 1;
            Ok(value)
        }
        fn draw(&mut self, nonzero: bool) -> Result<E, FpError> {
            self.close_algebraic(nonzero)
        }

        fn close_empty(&mut self) -> Result<(), FpError> {
            Ok(())
        }
    }

    #[test]
    fn hp_form_tracks_hiding_polynomial_variables() {
        let vars = vec![Weight::equality(E([2, 3, 5]), 2), Weight::at_zero(3)];
        let form = hp_form(&vars, &[E([7, 11, 13]), E([17, 19, 23])]);
        assert_eq!(form.len(), 4);
        assert_eq!(form[0], E([7, 11, 13]));
        assert_eq!(form[1], E::ONE);
        assert_eq!(form[2], E([17, 19, 23]));
        assert_eq!(form[3], E([17, 19, 23]) * E([17, 19, 23]));
    }

    #[test]
    fn psc_emits_end_claims_and_masked_target() {
        let claim = SumClaim {
            vars: vec![Weight::equality(E([2, 3, 5]), 2)],
            masks: vec![],
            target: E([29, 31, 37]),
        };
        let fresh = (0..6).map(|index| MaskSlot { oracle: 0, index }).collect();
        let mut channel = Channel {
            values: vec![
                E([41, 43, 47]),
                E([53, 59, 61]),
                E([67, 71, 73]),
                E([79, 83, 89]),
                E([97, 101, 103]),
                E([107, 109, 113]),
                E([127, 131, 137]),
            ],
            challenges: vec![E([139, 149, 151]), E([157, 163, 167])],
            next: 0,
            round: 0,
        };
        let output = psc_verify(&mut channel, claim, &Product, fresh).unwrap();
        assert_eq!(output.point, vec![E([139, 149, 151])]);
        assert_eq!(output.ends.len(), 2);
        assert_eq!(output.ends[0].value, E([97, 101, 103]));
        assert_eq!(output.ends[1].value, E([107, 109, 113]));
        assert_eq!(output.row_masks.len(), 4);
        let mut running = E([41, 43, 47]) + E([17, 19, 23]) * E([29, 31, 37]);
        let mut q0 = vec![E::ZERO, E([53, 59, 61])];
        Weight::equality(E([2, 3, 5]), 2).complete(&mut q0, running);
        running = poly_at(&q0, E([139, 149, 151]));
        let mut qy = vec![E::ZERO, E([67, 71, 73]), E([79, 83, 89])];
        Weight::at_zero(3).complete(&mut qy, running);
        let last = poly_at(&qy, E([157, 163, 167]));
        assert_eq!(
            output.row_target,
            last - E([17, 19, 23]) * (E([97, 101, 103]) * E([107, 109, 113]))
        );
    }
}

pub fn psc_verify<C: PscChannel>(
    channel: &mut C,
    claim: SumClaim,
    summand: &dyn Summand,
    fresh: Vec<MaskSlot>,
) -> Result<PscOutput, FpError> {
    let vars = psc_vars(&claim, summand);
    assert_eq!(fresh.len(), hp_entries(&vars) + summand.shown());
    let k = claim.vars.len();
    let mu = channel.recv_hiding_sum()?;
    let eps = channel.close_epsilon()?;
    let mut running = mu + eps * claim.target;
    let mut challenges = Vec::with_capacity(vars.len());
    for j in 0..vars.len() {
        let mut q = vec![E::ZERO; vars[j].d];
        for t in 0..vars[j].d {
            if t != vars[j].fixed {
                q[t] = channel.recv()?;
            }
        }
        vars[j].complete(&mut q, running);
        let challenge = channel.close_algebraic(j >= k)?;
        challenges.push(challenge);
        running = poly_at(&q, challenge);
    }
    let mut end_values = Vec::with_capacity(summand.shown());
    for _ in 0..summand.shown() {
        end_values.push(channel.recv()?);
    }
    if summand.shown() != 0 {
        channel.close_empty()?;
    }
    Ok(psc_rows(
        claim,
        summand,
        &vars,
        fresh,
        eps,
        &challenges,
        running,
        &end_values,
    ))
}
