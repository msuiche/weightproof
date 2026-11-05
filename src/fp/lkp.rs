//! Lookup argument (LKP) verification built on PSC.

use crate::fp::{
    claims::{GroupId, OracleId, Row, Statement},
    error::FpError,
    field::{ExtensionElement as E, FieldElement as F},
    mask::MaskSlot,
    psc::{
        at_point, line, point_row, psc_row_u, psc_verify, MapAt, PointClaim, PscChannel, PscOutput,
        SumClaim, Summand, Weight,
    },
    tensor::{Form, Tensor},
};
use alloc::{boxed::Box, format, string::String, vec, vec::Vec};

#[derive(Debug, Clone)]
pub struct LkpKeyBlock {
    pub first: u64,
    pub log_trees: u32,
    pub function: i64,
    pub padding_row: u64,
}
#[derive(Debug, Clone)]
pub struct LkpTablePart {
    pub first: u64,
    pub log_rows: u32,
    pub function: i64,
    pub rows: Vec<E>,
}
#[derive(Debug, Clone)]
pub struct LkpPadding {
    pub part: usize,
    pub first: u64,
    pub log_step: u32,
    pub length: u64,
    pub row: u64,
}
#[derive(Debug, Clone)]
pub struct LkpPublic {
    pub context: String,
    pub parts: Vec<u32>,
    pub d: u32,
    pub mu: u32,
    pub live: u32,
    pub h: u32,
    pub g: u32,
    pub shifted: bool,
    pub read_from: Vec<OracleId>,
    pub keys: Vec<LkpKeyBlock>,
    pub tables: Vec<LkpTablePart>,
    pub padding: Vec<LkpPadding>,
    pub last_key_level_order: Vec<u32>,
}
impl LkpPublic {
    pub fn planes(&self) -> usize {
        if self.live == 0 {
            1usize << self.mu
        } else {
            self.live as usize
        }
    }
    pub fn all_keys(&self) -> u64 {
        self.parts.iter().map(|&bits| 1u64 << bits).sum()
    }
}

#[derive(Debug, Clone)]
pub struct LkpKeys {
    pub point: Vec<E>,
    pub weights: Form,
    pub masks: Vec<crate::fp::mask::MaskTerm>,
    pub target: E,
}
#[derive(Debug, Clone)]
pub struct LkpOutput {
    pub theta: E,
    pub alpha: E,
    pub keys: Vec<LkpKeys>,
    pub rows: Vec<Row>,
}

#[derive(Debug, Clone)]
pub struct LkpShape {
    pub d: u32,
    pub mu: u32,
    pub planes: usize,
    pub part_bits: Vec<u32>,
    pub all_keys: u64,
    pub padding: Vec<(u64, E)>,
    pub last_key_level_order: Vec<u32>,
}

pub fn lkp_default_order(variables: usize) -> Vec<usize> {
    (0..variables).rev().collect()
}
pub fn lkp_order_beginning(first: &[u32], variables: usize) -> Vec<usize> {
    let mut out: Vec<usize> = first.iter().map(|&x| x as usize).collect();
    for j in (0..variables).rev() {
        if !first.contains(&(j as u32)) {
            out.push(j);
        }
    }
    out
}
pub fn lkp_by_variable(by_round: &[E], order: &[usize]) -> Vec<E> {
    let mut point = vec![E::ZERO; by_round.len()];
    for (j, &variable) in order.iter().enumerate() {
        point[variable] = by_round[j];
    }
    point
}

pub fn lkp_row(fixed: &LkpPublic, theta: E, row: u64) -> E {
    for part in &fixed.tables {
        if (fixed.h != 0 && (row & ((1u64 << fixed.h) - 1)) != part.first)
            || (fixed.h == 0 && row >> part.log_rows != part.first >> part.log_rows)
        {
            continue;
        }
        let own = (if fixed.h != 0 { row >> fixed.h } else { row }) & ((1u64 << part.log_rows) - 1);
        let base = if part.rows.is_empty() {
            E::from_base(F(own))
        } else {
            part.rows[own as usize]
        };
        return base + E::from_base(F(part.function as u64)) * theta;
    }
    E::ZERO
}

pub fn lkp_table_at(fixed: &LkpPublic, theta: E, point: &[E]) -> E {
    let mut sum = E::ZERO;
    for part in &fixed.tables {
        let mut weight = E::ONE;
        let start = if fixed.h != 0 {
            0
        } else {
            part.log_rows as usize
        };
        let end = if fixed.h != 0 {
            fixed.h as usize
        } else {
            fixed.d as usize
        };
        for j in start..end {
            weight *= if (part.first >> j) & 1 != 0 {
                point[j]
            } else {
                E::ONE - point[j]
            };
        }
        let z = &point[fixed.h as usize..];
        let own = if part.rows.is_empty() {
            let mut out = E::ZERO;
            let mut power = E::ONE;
            for j in 0..part.log_rows as usize {
                out += power * z[j];
                power += power;
            }
            out
        } else {
            let mut fold = part.rows.clone();
            for &challenge in z.iter().take(part.log_rows as usize) {
                for i in 0..fold.len() / 2 {
                    fold[i] = fold[2 * i] + challenge * (fold[2 * i + 1] - fold[2 * i]);
                }
                fold.truncate(fold.len() / 2);
            }
            fold[0]
        };
        sum += weight * (own + E::from_base(F(part.function as u64)) * theta);
    }
    sum
}

pub fn lkp_shape(fixed: &LkpPublic, theta: E) -> LkpShape {
    let mut padding = fixed
        .padding
        .iter()
        .map(|p| (p.length, lkp_row(fixed, theta, p.row)))
        .collect::<Vec<_>>();
    for block in &fixed.keys {
        if block.function < 0 {
            padding.push((
                1u64 << (fixed.parts[0] - fixed.g + block.log_trees),
                lkp_row(fixed, theta, block.padding_row),
            ));
        }
    }
    LkpShape {
        d: fixed.d,
        mu: fixed.mu,
        planes: fixed.planes(),
        part_bits: fixed.parts.clone(),
        all_keys: fixed.all_keys(),
        padding,
        last_key_level_order: fixed.last_key_level_order.clone(),
    }
}

pub fn lkp_padding_count(fixed: &LkpPublic) -> u64 {
    fixed.padding.iter().map(|p| p.length).sum::<u64>()
        + fixed
            .keys
            .iter()
            .filter(|b| b.function < 0)
            .map(|b| 1u64 << (fixed.parts[0] - fixed.g + b.log_trees))
            .sum::<u64>()
}

#[derive(Debug, Clone, Copy)]
pub struct LkpCube {
    pub mask: u64,
    pub value: u64,
}
pub fn lkp_cubes(padding: &LkpPadding, bits: usize) -> Vec<LkpCube> {
    let all = (1u64 << bits) - 1;
    let low = (1u64 << padding.log_step) - 1;
    let mut out = Vec::new();
    let mut x = padding.first >> padding.log_step;
    let mut left = padding.length;
    while left != 0 {
        let mut b = 0;
        while b < 62 && (x & ((2u64 << b) - 1)) == 0 && (2u64 << b) <= left {
            b += 1;
        }
        out.push(LkpCube {
            mask: all & !(((1u64 << b) - 1) << padding.log_step),
            value: (padding.first & low) | (x << padding.log_step),
        });
        x += 1u64 << b;
        left -= 1u64 << b;
    }
    out
}
pub fn lkp_cube(fixed: &LkpPublic, block: &LkpKeyBlock) -> LkpCube {
    LkpCube {
        mask: ((1u64 << fixed.g) - 1) & !((1u64 << block.log_trees) - 1),
        value: block.first,
    }
}
pub fn lkp_cube_weight(cube: LkpCube, point: &[E]) -> E {
    let mut weight = E::ONE;
    for (t, &z) in point.iter().enumerate() {
        if (cube.mask >> t) & 1 != 0 {
            weight *= if (cube.value >> t) & 1 != 0 {
                z
            } else {
                E::ONE - z
            };
        }
    }
    weight
}
pub fn lkp_cube_tensor(cube: LkpCube, point: &[E], scalar: E) -> Tensor {
    let mut tensor = Tensor::eq(&[], scalar);
    for (t, &z) in point.iter().enumerate() {
        let picked = (cube.mask >> t) & 1 != 0;
        let one = (cube.value >> t) & 1 != 0;
        tensor.push(
            if picked && one { E::ZERO } else { E::ONE - z },
            if picked && !one { E::ZERO } else { z },
        );
    }
    tensor
}

struct Product;
impl Summand for Product {
    fn shown(&self) -> usize {
        2
    }
    fn degree(&self) -> usize {
        2
    }
    fn at(&self, _point: &[E], values: &[E]) -> Vec<E> {
        vec![values[0] * values[1]]
    }
}
struct Tie {
    first: bool,
    a: E,
}
impl Summand for Tie {
    fn shown(&self) -> usize {
        if self.first {
            3
        } else {
            2
        }
    }
    fn degree(&self) -> usize {
        3
    }
    fn at(&self, _point: &[E], values: &[E]) -> Vec<E> {
        if self.first {
            vec![values[0] - self.a * values[1] * values[1] * values[2]]
        } else {
            vec![values[0] * values[0] * values[1]]
        }
    }
}
struct Leaves<'a> {
    fixed: &'a LkpPublic,
    theta: E,
    alpha: E,
    order: &'a [usize],
}
impl Summand for Leaves<'_> {
    fn shown(&self) -> usize {
        0
    }
    fn degree(&self) -> usize {
        0
    }
    fn kept(&self) -> usize {
        1
    }
    fn at(&self, point: &[E], _values: &[E]) -> Vec<E> {
        let by_round = lkp_by_variable(point, self.order);
        vec![
            E::ONE,
            self.alpha
                - E::ONE
                - lkp_table_at(self.fixed, self.theta, &by_round[self.fixed.mu as usize..]),
        ]
    }
}
struct Bits;
impl Summand for Bits {
    fn shown(&self) -> usize {
        1
    }
    fn degree(&self) -> usize {
        2
    }
    fn at(&self, _point: &[E], values: &[E]) -> Vec<E> {
        vec![values[0] * values[0] - values[0]]
    }
}

struct Side<'a, C: PscChannel> {
    statement: &'a mut Statement,
    channel: &'a mut C,
    masks: OracleId,
    key_masks: OracleId,
    group: GroupId,
    rows: &'a mut Vec<Row>,
}
impl<C: PscChannel> Side<'_, C> {
    fn fresh(&mut self, count: usize, keys: bool) -> Vec<MaskSlot> {
        if keys {
            let mut fresh = self.statement.masks(self.masks).take_many(count - 2);
            fresh.extend(self.statement.masks(self.key_masks).take_many(2));
            fresh
        } else {
            self.statement.masks(self.masks).take_many(count)
        }
    }
    fn psc(
        &mut self,
        claim: SumClaim,
        summand: &dyn Summand,
        keys: bool,
    ) -> Result<PscOutput, FpError> {
        let count = crate::fp::psc::psc_masks(&claim, summand);
        let fresh = self.fresh(count, keys);
        psc_verify(self.channel, claim, summand, fresh)
    }
    fn leave(&mut self, row: Row) {
        self.rows.push(row.clone());
        self.statement.add_row(self.group, row);
    }
}

fn level_chain<C: PscChannel>(
    side: &mut Side<'_, C>,
    fixed: &LkpPublic,
    _part: usize,
    levels: usize,
    more: usize,
    keys: bool,
    mut claim: PointClaim,
) -> Result<PointClaim, FpError> {
    let product = Product;
    for l in 0..levels {
        let variables = more + l;
        let last_of_keys = keys && l + 1 == levels;
        let order = if last_of_keys {
            lkp_order_beginning(&fixed.last_key_level_order, variables)
        } else {
            lkp_default_order(variables)
        };
        let sum = SumClaim {
            vars: (0..variables)
                .map(|j| Weight::equality(claim.point[order[j]], 3))
                .collect(),
            masks: claim.masks.clone(),
            target: claim.value,
        };
        let mut out = side.psc(sum, &product, last_of_keys)?;
        side.leave(psc_row_u(&mut out, &[], "LKP level"));
        for end in &mut out.ends {
            end.point = lkp_by_variable(&end.point, &order);
        }
        claim = line(side.channel, out.ends.remove(0), out.ends.remove(0))?;
    }
    Ok(claim)
}

fn bits_at<'a>(planes: &'a dyn Fn(&mut Row, Tensor)) -> MapAt<'a> {
    Box::new(move |row, point, coefficient| {
        planes(row, Tensor::eq(point, coefficient));
    })
}

pub fn lkp_verify<C: PscChannel>(
    statement: &mut Statement,
    channel: &mut C,
    fixed: &LkpPublic,
    planes: impl Fn(&mut Row, Tensor),
    masks: OracleId,
    group: GroupId,
    key_masks: OracleId,
) -> Result<LkpOutput, FpError> {
    let mut rows = Vec::new();
    let mut side = Side {
        statement,
        channel,
        masks,
        key_masks,
        group,
        rows: &mut rows,
    };
    let theta = if fixed.shifted {
        side.channel.close_algebraic(false)?
    } else {
        E::ZERO
    };
    let shape = lkp_shape(fixed, theta);
    let alpha = side.channel.close_algebraic(false)?;
    let mut a = E::ONE;
    let mut padding = 0u64;
    for &(count, value) in &shape.padding {
        let mut power = alpha - value;
        let mut e = count;
        while e != 0 {
            if e & 1 != 0 {
                a *= power;
            }
            power *= power;
            e >>= 1;
        }
        padding += count;
    }
    assert_eq!(padding, lkp_padding_count(fixed));
    let mut on_root = PointClaim {
        point: vec![],
        masks: vec![],
        value: E::ZERO,
    };
    let mut on_power = on_root.clone();
    let mut on_plane = Vec::new();
    for j in 0..shape.planes.saturating_sub(1) {
        let sum = if j == 0 {
            SumClaim {
                vars: vec![],
                masks: vec![],
                target: E::ZERO,
            }
        } else {
            at_point(on_power, &[])
        };
        let mut out = side.psc(sum, &Tie { first: j == 0, a }, false)?;
        side.leave(psc_row_u(&mut out, &[], "LKP tie"));
        if j == 0 {
            on_root = out.ends[0].clone();
        }
        on_power = out.ends[if j == 0 { 1 } else { 0 }].clone();
        on_plane.push(out.ends[if j == 0 { 2 } else { 1 }].clone());
    }
    on_plane.push(on_power);
    side.channel
        .set_context(&format!("{}, parts", fixed.context));
    let mut on_roots = Vec::new();
    for _ in 0..shape.part_bits.len().saturating_sub(1) {
        let mut out = side.psc(at_point(on_root, &[]), &Product, false)?;
        side.leave(psc_row_u(&mut out, &[], "LKP parts"));
        on_roots.push(out.ends.remove(0));
        on_root = out.ends.remove(0);
    }
    on_roots.push(on_root);
    side.channel
        .set_context(&format!("{}, keys", fixed.context));
    let mut on_table = PointClaim {
        point: {
            let first = side.channel.close_algebraic(false)?;
            let mut point = Vec::with_capacity(shape.mu as usize);
            point.push(first);
            for _ in 1..shape.mu {
                point.push(side.channel.draw(false)?);
            }
            point
        },
        masks: vec![],
        value: E::ZERO,
    };
    for j in 0..(1usize << shape.mu) {
        let mut weight = E::ONE;
        for t in 0..shape.mu as usize {
            weight *= if (j >> t) & 1 != 0 {
                on_table.point[t]
            } else {
                E::ONE - on_table.point[t]
            };
        }
        if j >= shape.planes {
            on_table.value += weight;
            continue;
        }
        for mask in &on_plane[j].masks {
            on_table.masks.push(crate::fp::mask::MaskTerm {
                slot: mask.slot,
                coeff: weight * mask.coeff,
            });
        }
        on_table.value += weight * on_plane[j].value;
    }
    for part in 0..shape.part_bits.len() {
        on_roots[part] = level_chain(
            &mut side,
            fixed,
            part,
            shape.part_bits[part] as usize,
            0,
            true,
            on_roots[part].clone(),
        )?;
    }
    side.channel
        .set_context(&format!("{}, table", fixed.context));
    on_table = level_chain(
        &mut side,
        fixed,
        0,
        shape.d as usize,
        shape.mu as usize,
        false,
        on_table.clone(),
    )?;
    let all = (shape.mu + shape.d) as usize;
    side.channel
        .set_context(&format!("{}, leaves", fixed.context));
    let order = lkp_default_order(all);
    let sum = SumClaim {
        vars: (0..all)
            .map(|j| {
                Weight::equality(
                    on_table.point[order[j]],
                    if order[j] < shape.mu as usize { 2 } else { 3 },
                )
            })
            .collect(),
        masks: on_table.masks.clone(),
        target: on_table.value,
    };
    let mut leaves = side.psc(
        sum,
        &Leaves {
            fixed,
            theta,
            alpha,
            order: &order,
        },
        false,
    )?;
    leaves.point = lkp_by_variable(&leaves.point, &order);
    side.leave(psc_row_u(&mut leaves, &[bits_at(&planes)], "LKP leaves"));
    side.channel
        .set_context(&format!("{}, bits", fixed.context));
    let point: Vec<E> = {
        let first = side.channel.close_algebraic(false)?;
        let mut point = Vec::with_capacity(all);
        point.push(first);
        for _ in 1..all {
            point.push(side.channel.draw(false)?);
        }
        point
    };
    let sum = SumClaim {
        vars: (0..all)
            .map(|j| Weight::equality(point[order[j]], 3))
            .collect(),
        masks: vec![],
        target: E::ZERO,
    };
    let mut bits = side.psc(sum, &Bits, false)?;
    side.leave(psc_row_u(&mut bits, &[], "LKP bits"));
    bits.ends[0].point = lkp_by_variable(&bits.ends[0].point, &order);
    side.leave(point_row(
        bits.ends.remove(0),
        bits_at(&planes),
        "LKP bits at the point",
    ));
    let mut sum = Row::default();
    let mut tensor = Tensor::eq(&[], E::ONE);
    let mut two_to = E::ONE + E::ONE;
    for _ in 0..shape.mu {
        tensor.push(E::ONE, two_to);
        two_to *= two_to;
    }
    for _ in 0..shape.d {
        tensor.push(E::ONE, E::ONE);
    }
    planes(&mut sum, tensor);
    sum.target = E::from_base(F(shape.all_keys - padding));
    side.leave(sum);
    let mut keys = Vec::new();
    for part in 0..shape.part_bits.len() {
        let on_leaves = &on_roots[part];
        let mut weights = Form::default();
        weights.add(Tensor::eq(&on_leaves.point, E::ONE));
        let mut key = LkpKeys {
            point: on_leaves.point.clone(),
            weights,
            masks: on_leaves
                .masks
                .iter()
                .map(|m| crate::fp::mask::MaskTerm {
                    slot: m.slot,
                    coeff: -m.coeff,
                })
                .collect(),
            target: alpha - on_leaves.value,
        };
        let take_out = |cube: LkpCube, row: u64, key: &mut LkpKeys| {
            let weight = lkp_cube_weight(cube, &key.point);
            key.target -= weight * lkp_row(fixed, theta, row);
            key.weights.add(lkp_cube_tensor(cube, &key.point, -E::ONE));
            weight
        };
        let mut real = vec![E::ZERO; fixed.keys.len()];
        for (b, block) in fixed.keys.iter().enumerate() {
            if block.function < 0 {
                take_out(lkp_cube(fixed, block), block.padding_row, &mut key);
            } else {
                real[b] = lkp_cube_weight(lkp_cube(fixed, block), &key.point);
            }
        }
        for padding_run in &fixed.padding {
            if padding_run.part != part {
                continue;
            }
            let b = fixed
                .keys
                .iter()
                .position(|block| {
                    (padding_run.first & ((1u64 << fixed.g) - 1)) >> block.log_trees
                        == block.first >> block.log_trees
                })
                .unwrap();
            for cube in lkp_cubes(padding_run, key.point.len()) {
                real[b] -= take_out(cube, padding_run.row, &mut key);
            }
        }
        for (b, block) in fixed.keys.iter().enumerate() {
            if block.function > 0 {
                key.target -= theta * E::from_base(F(block.function as u64)) * real[b];
            }
        }
        keys.push(key);
    }
    Ok(LkpOutput {
        theta,
        alpha,
        keys,
        rows,
    })
}
