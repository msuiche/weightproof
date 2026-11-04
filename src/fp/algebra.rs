//! Algebraic block byte planners (numbers/blocks.h): PSC, LKP and BATT.
//!
//! Only message sizes and nonce positions are computed here; field challenges
//! and soundness accounting are separate verifier responsibilities.

use alloc::{vec, vec::Vec};

use crate::fp::{
    analysis::{Scope, Setting},
    error::FpError,
    field::GOLDILOCKS_P,
    layout::{MessageKind, PlanMessage},
    plan::{nonce, scalars},
};

/// Sumcheck sizes. `degree[j]` is the strict degree bound of round `j`'s
/// polynomial; `with_y = false` is the affine form with no Y round or values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PscSize {
    pub degree: Vec<u32>,
    pub degree_y: u32,
    pub values: u64,
    pub with_y: bool,
}

impl Default for PscSize {
    fn default() -> Self {
        Self {
            degree: vec![],
            degree_y: 2,
            values: 2,
            with_y: true,
        }
    }
}

fn field_nonce(messages: &mut Vec<PlanMessage>, setting: &Setting, scope: Scope, context: &str) {
    if let Some(nonce) = nonce(context, setting.work_field[scope as usize]) {
        messages.push(nonce);
    }
}

/// Append one PSC, coalescing equal-degree rounds in the stored plan. Layout
/// expansion produces the same fields as the oracle's uncompact print.
pub fn plan_psc(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    x: &PscSize,
) -> Result<(), FpError> {
    let saved = u32::from(setting.saved_coefficient);
    messages.push(scalars("hiding sum", context, 1)?);
    field_nonce(messages, setting, scope, context);
    let mut j = 0usize;
    while j < x.degree.len() {
        let mut run = 1usize;
        while j + run < x.degree.len() && x.degree[j + run] == x.degree[j] {
            run += 1;
        }
        let coefficients = x.degree[j].checked_sub(saved).ok_or(FpError::Range {
            context: "psc-degree",
        })?;
        messages.push(
            scalars("round polynomial", context, coefficients as u64)?.with_count(run as u64),
        );
        for _ in 0..run {
            field_nonce(messages, setting, scope, context);
        }
        j += run;
    }
    if !x.with_y {
        return Ok(());
    }
    let coefficients = (x.degree_y + 1).checked_sub(saved).ok_or(FpError::Range {
        context: "psc-y-degree",
    })?;
    messages.push(scalars(
        "round polynomial in Y",
        context,
        coefficients as u64,
    )?);
    field_nonce(messages, setting, scope, context);
    messages.push(scalars("values at the last point", context, x.values)?);
    Ok(())
}

/// Product-tree levels from the root down, each PSC followed by LINE.
pub fn plan_product_tree(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    levels: u32,
    more: u32,
) -> Result<(), FpError> {
    for level in 0..levels {
        plan_psc(
            messages,
            setting,
            scope,
            context,
            &PscSize {
                degree: vec![3; (level + more) as usize],
                ..PscSize::default()
            },
        )?;
        field_nonce(messages, setting, scope, context);
    }
    Ok(())
}

/// Public sizes of one LKP lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupSize {
    pub v: u32,
    pub d: u32,
    pub mu: u32,
    pub shifted: bool,
    /// Used bit positions; zero means all `2^mu`.
    pub live: u32,
    /// Log2 of each part's key count; empty means one part of `2^v` keys.
    pub parts: Vec<u32>,
}

impl LookupSize {
    pub fn new(d: u32, mu: u32, live: u32, parts: Vec<u32>) -> Self {
        Self {
            v: 0,
            d,
            mu,
            shifted: false,
            live,
            parts,
        }
    }
}

/// Append one complete LKP lookup: tie, parts, key/table product trees,
/// leaves and bit checks.
pub fn plan_lookup(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    x: &LookupSize,
) -> Result<(), FpError> {
    if x.mu == 0 || x.mu > 6 {
        return Err(FpError::Range {
            context: "lookup-mu",
        });
    }
    let slots = 1u32 << x.mu;
    let w = if x.live != 0 { x.live } else { slots };
    if w < 2 || w > slots {
        return Err(FpError::Range {
            context: "lookup-live",
        });
    }
    let default_part;
    let parts: &[u32] = if x.parts.is_empty() {
        default_part = [x.v];
        &default_part
    } else {
        &x.parts
    };
    let mut keys = 0u128;
    for &v in parts {
        if v >= 63 {
            return Err(FpError::Range {
                context: "lookup-part-size",
            });
        }
        keys = keys.checked_add(1u128 << v).ok_or(FpError::Range {
            context: "lookup-keys-overflow",
        })?;
    }
    if keys >= (1u128 << w) || (1u128 << x.d) * ((1u128 << w) - 1) >= GOLDILOCKS_P as u128 {
        return Err(FpError::Range {
            context: "lookup-range",
        });
    }
    if x.shifted {
        field_nonce(messages, setting, scope, context);
    }
    field_nonce(messages, setting, scope, context); // R_LKP

    for j in 0..w - 1 {
        plan_psc(
            messages,
            setting,
            scope,
            &alloc::format!("{context}, tie"),
            &PscSize {
                degree_y: 3,
                values: if j == 0 { 3 } else { 2 },
                ..PscSize::default()
            },
        )?;
    }
    for _ in 0..parts.len().saturating_sub(1) {
        plan_psc(
            messages,
            setting,
            scope,
            &alloc::format!("{context}, parts"),
            &PscSize::default(),
        )?;
    }
    field_nonce(messages, setting, scope, context); // JOIN
    for &v in parts {
        plan_product_tree(
            messages,
            setting,
            scope,
            &alloc::format!("{context}, keys"),
            v,
            0,
        )?;
    }
    plan_product_tree(
        messages,
        setting,
        scope,
        &alloc::format!("{context}, table"),
        x.d,
        x.mu,
    )?;

    let mut leaf_degrees = vec![3; x.d as usize];
    leaf_degrees.extend(vec![2; x.mu as usize]);
    plan_psc(
        messages,
        setting,
        scope,
        &alloc::format!("{context}, leaves"),
        &PscSize {
            degree: leaf_degrees,
            with_y: false,
            ..PscSize::default()
        },
    )?;
    field_nonce(messages, setting, scope, &alloc::format!("{context}, bits")); // ZERO
    plan_psc(
        messages,
        setting,
        scope,
        &alloc::format!("{context}, bits"),
        &PscSize {
            degree: vec![3; (x.mu + x.d) as usize],
            values: 1,
            ..PscSize::default()
        },
    )?;
    Ok(())
}

/// Batch pending table claims and run their affine sumcheck.
pub fn plan_batt(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    k_col: u32,
) -> Result<(), FpError> {
    field_nonce(messages, setting, scope, context); // BAT-sigma
    plan_psc(
        messages,
        setting,
        scope,
        context,
        &PscSize {
            degree: vec![3; k_col as usize],
            with_y: false,
            ..PscSize::default()
        },
    )
}

/// BATT is present exactly when more than one pending claim remains.
pub fn plan_member_rows(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    rows: u64,
    k_col: u32,
) -> Result<(), FpError> {
    if rows > 1 {
        plan_batt(messages, setting, scope, context, k_col)?;
    }
    Ok(())
}

/// SPLIT's target; its empty verifier round adds no bytes.
pub fn plan_split(messages: &mut Vec<PlanMessage>, context: &str) -> Result<(), FpError> {
    messages.push(scalars("split target", context, 1)?);
    Ok(())
}

/// SEND's roots and sample answers, including work nonces at their exact
/// positions. The caller supplies the number of sampled oracle messages.
pub fn plan_send(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    scope: Scope,
    context: &str,
    roots: usize,
    sampled_messages: u64,
) -> Result<(), FpError> {
    for _ in 0..roots {
        messages.push(crate::fp::plan::root(context));
    }
    if let Some(nonce) = nonce(context, setting.work_sample[scope as usize]) {
        messages.push(nonce);
    }
    let count = setting
        .samples
        .checked_mul(sampled_messages)
        .ok_or(FpError::Range {
            context: "send-answer-count",
        })?;
    messages.push(crate::fp::plan::answers(context, count)?);
    Ok(())
}

/// A fixed scalar list that is not an oracle sample.
pub fn plan_scalars(
    messages: &mut Vec<PlanMessage>,
    name: &str,
    context: &str,
    count: u64,
) -> Result<(), FpError> {
    messages.push(scalars(name, context, count)?);
    Ok(())
}

/// A chain's optional salt, outside every prover message.
pub fn plan_chain_start(messages: &mut Vec<PlanMessage>, context: &str, salted: bool) {
    if salted {
        messages.push(PlanMessage::fixed(
            "chain salt",
            context,
            MessageKind::Salts,
            crate::fp::plan::DIGEST_BYTES,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_byte_counts_match_oracle_wf_fields() {
        let setting = Setting::example();
        let mut bytes = vec![];
        plan_lookup(
            &mut bytes,
            &setting,
            Scope::Commit,
            "ranges: bytes",
            &LookupSize::new(8, 5, 28, vec![25, 25, 25, 25]),
        )
        .unwrap();
        let count = |label: &str| {
            bytes
                .iter()
                .filter(|message| message.name == label)
                .map(|message| message.bytes * message.count)
                .sum::<u64>()
        };
        assert_eq!(
            count("round polynomial"),
            1200 * 48 + 68 * 48 + (8 * 48 + 5 * 24) + 13 * 48
        );
        assert_eq!(
            bytes
                .iter()
                .filter(|message| message.context == "ranges: bytes, tie")
                .count(),
            27 * 3
        );
        assert_eq!(
            bytes
                .iter()
                .filter(|message| message.context == "ranges: bytes, parts")
                .count(),
            3 * 3
        );

        let mut weights = vec![];
        plan_lookup(
            &mut weights,
            &setting,
            Scope::Commit,
            "ranges: weights",
            &LookupSize::new(13, 5, 27, vec![24, 24, 24, 24]),
        )
        .unwrap();
        let rounds = weights
            .iter()
            .filter(|message| message.name == "round polynomial")
            .map(|message| message.bytes * message.count)
            .sum::<u64>();
        assert_eq!(rounds, (1104 + 143 + 13 + 18) * 48 + 5 * 24);
    }

    #[test]
    fn member_rows_has_affine_sumcheck_only_for_multiple_claims() {
        let setting = Setting::example();
        let mut messages = vec![];
        plan_member_rows(&mut messages, &setting, Scope::Commit, "rows", 1, 11).unwrap();
        assert!(messages.is_empty());
        plan_member_rows(&mut messages, &setting, Scope::Commit, "rows", 8, 11).unwrap();
        assert_eq!(messages.len(), 2); // hiding sum + one compact run of eleven polynomials
        assert_eq!(
            messages.iter().map(|message| message.count).sum::<u64>(),
            12
        );
        assert!(messages
            .iter()
            .all(|message| message.name != "round polynomial in Y"));
        assert_eq!(messages[1].count, 11);
    }
}
