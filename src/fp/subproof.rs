//! Model subproof byte planners (prover/{prover,attn,head}/plan_rounds.h).

use alloc::{vec, vec::Vec};

use crate::fp::{
    algebra::{plan_bat_sigma, plan_join, plan_line, plan_psc, plan_zero, PscSize},
    analysis::{Scope, Setting},
    error::FpError,
    layout::PlanMessage,
    plan::scalars,
};

#[derive(Debug, Clone)]
pub struct LayerRounds {
    pub lt: u32,
    pub ls: u32,
    pub l_ni: u32,
    pub l_pd: u32,
    pub ll: u32,
    pub l_pf: u32,
    pub l_fe: u32,
    pub lam_s: u32,
    pub lk_w: u32,
    pub ni: u64,
    pub ns: u64,
    pub np: u64,
    pub no: u64,
    pub with_terms: bool,
    pub shown_by_the_terms: u64,
    pub never_sent: u64,
    pub weight_claims: u64,
    pub first_group: bool,
    pub routed: bool,
}

#[derive(Debug, Clone)]
pub struct AttentionRounds {
    pub lt: u32,
    pub lgk: u32,
    pub pairs: u32,
    pub lam: u32,
    pub plane_messages: u32,
    pub score: Vec<(u32, u32)>,
    pub e: u32,
    pub value_m: u32,
    pub value_d: u32,
    pub attained: u32,
    pub bits: u32,
    pub rows: u64,
}

#[derive(Debug, Clone)]
pub struct HeadRounds {
    pub lt: u32,
    pub l_pd: u32,
    pub lk_w: u32,
    pub sums: u64,
    pub products: u64,
    pub row_values: u64,
    pub rows: u64,
}

#[derive(Debug, Clone)]
pub enum Subproof {
    Layer(LayerRounds),
    Attention(AttentionRounds),
    Head(HeadRounds),
}

fn product_layer(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    variables: u32,
) -> Result<(), FpError> {
    plan_psc(
        messages,
        setting,
        Scope::Request,
        context,
        &PscSize {
            degree: vec![3; variables as usize],
            ..PscSize::default()
        },
    )?;
    plan_line(messages, setting, Scope::Request, context);
    Ok(())
}

fn token_sumcheck(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    lt: u32,
    degree_y: u32,
    shown: u64,
) -> Result<(), FpError> {
    messages.push(scalars("claimed sum", context, 1)?);
    plan_psc(
        messages,
        setting,
        Scope::Request,
        context,
        &PscSize {
            degree: vec![3; lt as usize],
            degree_y,
            values: shown,
            ..PscSize::default()
        },
    )
}

fn token_sum(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    lt: u32,
    shown: u64,
    degree_y: u32,
    more: u32,
) -> Result<(), FpError> {
    messages.push(scalars("token sum", context, 1)?);
    plan_psc(
        messages,
        setting,
        Scope::Request,
        context,
        &PscSize {
            degree: vec![3; (lt + more) as usize],
            degree_y,
            values: shown,
            with_y: shown > 0,
            ..PscSize::default()
        },
    )
}

fn layer(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    x: &LayerRounds,
) -> Result<(), FpError> {
    plan_zero(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, the norms' token point"),
    );
    if x.first_group {
        plan_zero(
            messages,
            setting,
            Scope::Request,
            &alloc::format!("{context}, the embedding's point"),
        );
        plan_zero(
            messages,
            setting,
            Scope::Request,
            &alloc::format!("{context}, the embedding's feature point"),
        );
    }
    messages.push(scalars("sums of limb products", context, x.ns * x.ni)?);
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, limbs"),
    );
    if x.l_ni != 0 {
        plan_join(
            messages,
            setting,
            Scope::Request,
            &alloc::format!("{context}, norm instances"),
        );
    }
    product_layer(
        messages,
        setting,
        &alloc::format!("{context}, sums of squares"),
        x.l_ni + x.l_pd + x.lt,
    )?;
    messages.push(scalars("row products", context, x.np * x.ni)?);
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, row products"),
    );
    token_sumcheck(
        messages,
        setting,
        &alloc::format!("{context}, row products"),
        x.lt,
        2,
        x.no * x.ni,
    )?;
    if x.with_terms {
        token_sumcheck(
            messages,
            setting,
            &alloc::format!("{context}, terms"),
            x.lt,
            1,
            x.shown_by_the_terms,
        )?;
        plan_bat_sigma(
            messages,
            setting,
            Scope::Request,
            &alloc::format!("{context}, start of the inner products"),
        );
    }
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, inner products"),
        &PscSize {
            degree: vec![3; x.lam_s as usize],
            values: 2 + x.never_sent + x.weight_claims,
            ..PscSize::default()
        },
    )?;
    product_layer(
        messages,
        setting,
        &alloc::format!("{context}, gated products"),
        x.ll + x.l_pf + x.lt,
    )?;
    if x.routed {
        if x.ll > 1 {
            plan_join(
                messages,
                setting,
                Scope::Request,
                &alloc::format!("{context}, layers with routed experts"),
            );
        }
        product_layer(
            messages,
            setting,
            &alloc::format!("{context}, routed gated products"),
            x.ll - 1 + x.l_fe + x.lt,
        )?;
    }
    if x.first_group {
        messages.push(scalars("the embedding's value", context, 1)?);
    }
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, weight claims"),
    );
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, weight claims"),
        &PscSize {
            degree: vec![3; x.lk_w as usize],
            degree_y: 1,
            values: 1,
            ..PscSize::default()
        },
    )
}

fn attention_sc3(nv: u32, pairs: u32) -> PscSize {
    PscSize {
        degree: vec![4; nv as usize],
        values: 2 * pairs as u64,
        ..PscSize::default()
    }
}

fn attention_scd(lt: u32, lgk: u32, pairs: u32, lam: u32) -> PscSize {
    let mut degree = vec![3; lt as usize];
    degree.extend(vec![lt + 3; lgk as usize]);
    degree.extend(vec![lt + 2; lt as usize]);
    PscSize {
        degree,
        degree_y: lt + 1,
        values: (lam + pairs * lt) as u64,
        ..PscSize::default()
    }
}

fn attention(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    x: &AttentionRounds,
) -> Result<(), FpError> {
    if x.plane_messages != 0 {
        messages.push(scalars(
            "plane values",
            context,
            3 * x.plane_messages as u64,
        )?);
    }
    for &(variables, pieces) in &x.score {
        messages.push(scalars("plane values and the scores' sum", context, 7)?);
        plan_psc(
            messages,
            setting,
            Scope::Request,
            &alloc::format!("{context}, scores"),
            &attention_sc3(variables, pieces),
        )?;
    }
    messages.push(scalars(
        "sums, index bits, remainders",
        context,
        5 + x.lt as u64 + 2,
    )?);
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, Z o"),
        &attention_sc3(x.e, x.pairs),
    )?;
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, numerator and denominator"),
    );
    messages.push(scalars("the value product's two sums", context, 2)?);
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, value product, off the diagonal"),
        &attention_sc3(x.value_m, x.pairs),
    )?;
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, value product, diagonal"),
        &attention_sc3(x.value_d, x.pairs),
    )?;
    plan_zero(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, the maximum is attained; the index bits are bits"),
    );
    messages.push(scalars("the maxima at the point", context, x.pairs as u64)?);
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, maxima"),
    );
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, q k at the index"),
        &attention_sc3(x.attained, x.pairs),
    )?;
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, keys at the index"),
    );
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, the key at the index"),
        &attention_scd(x.lt, x.lgk, x.pairs, x.lam),
    )?;
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, index bits"),
    );
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, index bits"),
        &attention_sc3(x.bits, x.pairs),
    )
}

fn head(
    messages: &mut Vec<PlanMessage>,
    setting: &Setting,
    context: &str,
    x: &HeadRounds,
) -> Result<(), FpError> {
    plan_zero(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, the norm's point"),
    );
    messages.push(scalars("sums of limb products", context, x.sums)?);
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, limbs"),
    );
    plan_psc(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, sum of squares"),
        &PscSize {
            degree: vec![3; (x.l_pd + x.lt) as usize],
            ..PscSize::default()
        },
    )?;
    plan_line(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, sum of squares"),
    );
    messages.push(scalars("row products", context, x.products)?);
    plan_bat_sigma(
        messages,
        setting,
        Scope::Request,
        &alloc::format!("{context}, row products"),
    );
    token_sum(
        messages,
        setting,
        &alloc::format!("{context}, row products"),
        x.lt,
        x.row_values,
        2,
        0,
    )?;
    token_sum(
        messages,
        setting,
        &alloc::format!("{context}, stream times reciprocal"),
        x.lt,
        2,
        2,
        0,
    )?;
    token_sum(
        messages,
        setting,
        &alloc::format!("{context}, activations times head"),
        x.lt,
        2,
        2,
        x.l_pd,
    )
}

impl Subproof {
    pub fn plan_own(
        &self,
        messages: &mut Vec<PlanMessage>,
        setting: &Setting,
        context: &str,
    ) -> Result<(), FpError> {
        match self {
            Self::Layer(x) => layer(messages, setting, context, x),
            Self::Attention(x) => attention(messages, setting, context, x),
            Self::Head(x) => head(messages, setting, context, x),
        }
    }

    pub fn plan_end(
        &self,
        messages: &mut Vec<PlanMessage>,
        setting: &Setting,
        context: &str,
    ) -> Result<(), FpError> {
        match self {
            Self::Layer(_) => plan_bat_sigma(
                messages,
                setting,
                Scope::Request,
                &alloc::format!("{context}, claims"),
            ),
            Self::Attention(_) => plan_bat_sigma(
                messages,
                setting,
                Scope::Request,
                &alloc::format!("{context}, all rows"),
            ),
            Self::Head(x) => {
                plan_bat_sigma(
                    messages,
                    setting,
                    Scope::Request,
                    &alloc::format!("{context}, merge"),
                );
                plan_psc(
                    messages,
                    setting,
                    Scope::Request,
                    &alloc::format!("{context}, merge"),
                    &PscSize {
                        degree: vec![3; x.lk_w as usize],
                        degree_y: 1,
                        values: 1,
                        ..PscSize::default()
                    },
                )?;
                plan_bat_sigma(
                    messages,
                    setting,
                    Scope::Request,
                    &alloc::format!("{context}, all rows"),
                );
            }
        }
        Ok(())
    }
}
