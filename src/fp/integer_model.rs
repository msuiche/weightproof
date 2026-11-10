//! Exact integer arithmetic for the M1 reference model.
//!
//! All operations use `i128`, matching the model's unbounded-integer semantics
//! for values up to 2^78 and intermediate products.

use alloc::{vec, vec::Vec};

/// Divide by a positive integer, rounding toward minus infinity.
fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a < 0 && a % b != 0 {
        q - 1
    } else {
        q
    }
}

/// `rq(a, k) = floor((a + 2^(k-1)) / 2^k)`, with ties upward.
pub fn rq(a: i128, k: u32) -> Option<i128> {
    if k == 0 || k >= 127 {
        return None;
    }
    Some(floor_div(a + (1i128 << (k - 1)), 1i128 << k))
}

pub fn clamp(a: i128, bound: i128) -> i128 {
    a.clamp(-bound, bound)
}

pub fn low8(a: i128) -> i128 {
    (a + 128).rem_euclid(256) - 128
}

fn rne(value: f64) -> i128 {
    let floor = libm::floor(value);
    let fraction = value - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i128) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded as i128
}

fn rha(value: f64) -> i128 {
    libm::round(value) as i128
}

/// Generate the public rotary, softmax, SiLU, and sigmoid tables.
///
/// Table indexing follows the specification: `Ts[c + 32768]` and
/// `Sig[c + 32768]`. `u0` is the first softmax entry rounded to zero; `EX`
/// contains only entries before that index.
pub fn make_tables(
    t: u32,
    dh: u32,
    cs: u32,
    theta0: f64,
    exp_s: u32,
    rx: u32,
    eg: u32,
    smax: i128,
) -> PublicTables {
    let mut cos = Vec::with_capacity((t * dh / 2) as usize);
    let mut sin = Vec::with_capacity((t * dh / 2) as usize);
    let scale = (1u64 << cs) as f64;
    for token in 0..t {
        for i in 0..dh / 2 {
            let angle = token as f64 * libm::pow(theta0, -2.0 * i as f64 / dh as f64);
            cos.push(rha(scale * libm::cos(angle)));
            sin.push(rha(scale * libm::sin(angle)));
        }
    }

    let mut ex = Vec::new();
    let mut u0 = 0;
    let exp_scale = (1u64 << exp_s) as f64;
    let decay = (1u64 << rx) as f64;
    for u in 0..1_000_000u32 {
        let value = rne(exp_scale * libm::exp(-(u as f64) / decay));
        if value == 0 {
            u0 = u as usize;
            break;
        }
        ex.push(value);
    }

    let mut ts = Vec::with_capacity(65536);
    let key_scale = 1i128 << eg;
    for c in -32768i128..32768 {
        let z = c.clamp(-64 * key_scale, 64 * key_scale - 1) as f64 / key_scale as f64;
        let value = rha(64.0 * z / (1.0 + libm::exp(-z)));
        ts.push(value.clamp(-smax, smax));
    }

    let mut sig = Vec::with_capacity(65536);
    for c in -32768i128..32768 {
        let q = 4096.0 / (1.0 + libm::exp(-c as f64 / 256.0));
        sig.push(16 * rha(q));
    }

    PublicTables {
        cos,
        sin,
        ex,
        ts,
        sig,
        u0,
    }
}

impl PublicTables {
    pub fn ts(&self, c: i128) -> Option<i128> {
        let index = c + 32768;
        if index < 0 || index >= 65536 {
            return None;
        }
        self.ts.get(index as usize).copied()
    }

    pub fn sig(&self, c: i128) -> Option<i128> {
        let index = c + 32768;
        if index < 0 || index >= 65536 {
            return None;
        }
        self.sig.get(index as usize).copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableDigests {
    pub trope: crate::fp::digest::Digest,
    pub texp: crate::fp::digest::Digest,
    pub tsilu: crate::fp::digest::Digest,
    pub tsig: crate::fp::digest::Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableRegion {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableLayout {
    pub regions: [TableRegion; 5],
    pub total_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct SerializedTables {
    pub bytes: Vec<u8>,
    pub layout: TableLayout,
    pub digests: TableDigests,
}

fn table_file(values: &[i128]) -> Option<Vec<u8>> {
    let mut bytes = Vec::with_capacity(8 + values.len() * 8);
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        let value = i64::try_from(value).ok()?;
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Some(bytes)
}

/// Serialize public tables in `cos, sin, EX, Ts, Sig` order. Each table file
/// is `u64 count` followed by signed 64-bit little-endian entries. The public
/// digests independently follow FlashProve's `trope/texp/tsilu/tsig` formats.
pub fn serialize_public_tables(tables: &PublicTables) -> Option<SerializedTables> {
    let files = [
        table_file(&tables.cos)?,
        table_file(&tables.sin)?,
        table_file(&tables.ex)?,
        table_file(&tables.ts)?,
        table_file(&tables.sig)?,
    ];
    let mut bytes = Vec::new();
    let mut regions = [TableRegion {
        offset: 0,
        length: 0,
    }; 5];
    for (index, file) in files.iter().enumerate() {
        let offset = bytes.len() as u64;
        bytes.extend_from_slice(file);
        regions[index] = TableRegion {
            offset,
            length: file.len() as u64,
        };
    }

    let mut rope = Vec::with_capacity(tables.cos.len() + tables.sin.len());
    for &value in tables.cos.iter().chain(&tables.sin) {
        rope.push(u32::try_from(value + 4096).ok()?);
    }
    let mut exponential = Vec::with_capacity(tables.ex.len());
    for &value in &tables.ex {
        exponential.push(u32::try_from(value).ok()?);
    }
    let mut silu = Vec::with_capacity(tables.ts.len());
    for &value in &tables.ts {
        silu.push(u32::try_from(value + 4096).ok()?);
    }
    let mut sigmoid = Vec::with_capacity(65536);
    for &value in &tables.sig[16384..49152] {
        sigmoid.push(u32::try_from(value).ok()?);
    }
    for r in -16384i128..16384 {
        sigmoid.push(u32::try_from(r.clamp(-127, 127) + 128).ok()?);
    }

    Some(SerializedTables {
        layout: TableLayout {
            regions,
            total_bytes: bytes.len() as u64,
        },
        digests: TableDigests {
            trope: crate::fp::blake2::public_table_digest(&rope),
            texp: crate::fp::blake2::public_table_digest(&exponential),
            tsilu: crate::fp::blake2::public_table_digest(&silu),
            tsig: crate::fp::blake2::public_table_digest(&sigmoid),
        },
        bytes,
    })
}

/// Return the first index at which the maximum is attained, or `None` for an
/// empty slice.
pub fn argmax(values: &[i128]) -> Option<usize> {
    let mut best = 0usize;
    for (i, &value) in values.iter().enumerate().skip(1) {
        if value > values[best] {
            best = i;
        }
    }
    if values.is_empty() {
        None
    } else {
        Some(best)
    }
}

/// Largest `r` with `r^2 <= a`, or `None` for a negative input.
pub fn isqrt(a: i128) -> Option<i128> {
    if a < 0 {
        return None;
    }
    if a <= 1 {
        return Some(a);
    }
    let mut x0 = 1i128 << ((128 - a.leading_zeros() + 1) / 2);
    loop {
        let x1 = (x0 + a / x0) / 2;
        if x1 >= x0 {
            return Some(x0);
        }
        x0 = x1;
    }
}

/// Root-mean-square norm without gains, as in `N(x; M, K, eps)`.
pub fn norm(x: &[i128], m: u32, k: u32, eps: i128) -> Option<Vec<i128>> {
    if m >= 127 || k >= 127 {
        return None;
    }
    let vx = x.iter().try_fold(0i128, |sum, &v| sum.checked_add(v * v))?;
    let sigma = isqrt(vx.checked_add(eps)?)?;
    if sigma == 0 {
        return None;
    }
    let rc = floor_div((1i128 << m) + sigma / 2, sigma);
    x.iter()
        .map(|&v| rq(v.checked_mul(rc)?, k).map(|w| clamp(w, 2047)))
        .collect()
}

#[derive(Clone, Debug)]
pub struct LayerScales {
    pub kq: u32,
    pub kk: u32,
    pub kv: u32,
    pub ko: u32,
    pub kg: u32,
    pub ku: u32,
    pub kd: u32,
    pub m1: u32,
    pub k1: u32,
    pub m2: u32,
    pub k2: u32,
    pub sh: u32,
    pub rp: bool,
    pub eps1: i128,
    pub eps2: i128,
}

#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub wq: Vec<Vec<i128>>,
    pub wk: Vec<Vec<i128>>,
    pub wv: Vec<Vec<i128>>,
    pub wo: Vec<Vec<i128>>,
    pub wg: Vec<Vec<i128>>,
    pub wu: Vec<Vec<i128>>,
    pub wd: Vec<Vec<i128>>,
}

#[derive(Clone, Debug)]
pub struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub emb: Vec<Vec<i128>>,
    pub wh: Vec<Vec<i128>>,
}

#[derive(Clone, Debug)]
pub struct PublicTables {
    pub cos: Vec<i128>,
    pub sin: Vec<i128>,
    pub ex: Vec<i128>,
    pub ts: Vec<i128>,
    pub sig: Vec<i128>,
    pub u0: usize,
}

#[derive(Clone, Debug)]
pub struct DenseModel {
    pub weights: ModelWeights,
    pub scales: Vec<LayerScales>,
    pub tables: PublicTables,
    pub t: usize,
    pub dh: usize,
    pub h: usize,
    pub g: usize,
    pub f: usize,
    pub v: usize,
    pub cs: u32,
}

fn dot(x: &[i128], w: &[Vec<i128>], witnesses: &mut Vec<(i128, i128)>) -> Option<Vec<i128>> {
    if w.is_empty() || w.len() != x.len() || w[0].is_empty() {
        return None;
    }
    let columns = w[0].len();
    let mut out = vec![0i128; columns];
    for (k, row) in w.iter().enumerate() {
        if row.len() != columns {
            return None;
        }
        for (j, &weight) in row.iter().enumerate() {
            witnesses.push((x[k], weight));
            out[j] = out[j].checked_add(x[k].checked_mul(weight)?)?;
        }
    }
    Some(out)
}

fn shifted(
    x: &[i128],
    w: &[Vec<i128>],
    shift: u32,
    witnesses: &mut Vec<(i128, i128)>,
) -> Option<Vec<i128>> {
    dot(x, w, witnesses)?
        .into_iter()
        .map(|v| rq(v, shift))
        .collect()
}

fn rotated_shifted(
    x: &[i128],
    w: &[Vec<i128>],
    shift: u32,
    cs: u32,
    rp: bool,
    position: usize,
    dh: usize,
    tables: &PublicTables,
    witnesses: &mut Vec<(i128, i128)>,
) -> Option<Vec<i128>> {
    if dh == 0 || dh % 2 != 0 || cs >= 127 {
        return None;
    }
    let wide = dot(x, w, witnesses)?;
    if wide.len() % dh != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(wide.len());
    for head in wide.chunks_exact(dh) {
        for i in 0..dh / 2 {
            let (c, s) = if rp {
                let index = position.checked_mul(dh / 2)?.checked_add(i)?;
                (*tables.cos.get(index)?, *tables.sin.get(index)?)
            } else {
                (1i128 << cs, 0)
            };
            let a0 = head[2 * i];
            let a1 = head[2 * i + 1];
            witnesses.extend_from_slice(&[(c, a0), (s, a1), (s, a0), (c, a1)]);
            let p0 = c.checked_mul(a0)?.checked_sub(s.checked_mul(a1)?)?;
            let p1 = s.checked_mul(a0)?.checked_add(c.checked_mul(a1)?)?;
            out.push(rq(p0, shift)?);
            out.push(rq(p1, shift)?);
        }
    }
    Some(out)
}

fn clamped(values: Vec<i128>) -> Vec<i128> {
    values.into_iter().map(|v| clamp(v, 2047)).collect()
}

fn norm_checked(
    x: &[i128],
    m: u32,
    k: u32,
    eps: i128,
    witnesses: &mut Vec<(i128, i128)>,
) -> Option<Vec<i128>> {
    if m >= 127 || k >= 127 {
        return None;
    }
    let vx = x.iter().try_fold(0i128, |sum, &v| {
        witnesses.push((v, v));
        sum.checked_add(v * v)
    })?;
    if vx.checked_add(eps)? >= 1i128 << 78 {
        return None;
    }
    let sigma = isqrt(vx.checked_add(eps)?)?;
    if sigma == 0 {
        return None;
    }
    let rc = floor_div((1i128 << m) + sigma / 2, sigma);
    x.iter()
        .map(|&v| {
            witnesses.push((v, rc));
            let w = rq(v.checked_mul(rc)?, k)?;
            if !(-32768..=32767).contains(&w) {
                return None;
            }
            Some(clamp(w, 2047))
        })
        .collect()
}

fn stream_is_in_range(x: &[i128]) -> bool {
    x.iter().all(|&v| v.abs() < 1i128 << 39)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LookupTable {
    SiLU,
    Softmax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LookupWitness {
    pub table: LookupTable,
    pub key: i128,
    pub value: i128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionTrace {
    pub tokens: Vec<usize>,
    pub residual: Vec<Vec<Vec<i128>>>,
    pub lookups: Vec<LookupWitness>,
    pub gate_witnesses: Vec<(i128, i128)>,
    pub attention_witnesses: Vec<(i128, i128)>,
    pub linear_witnesses: Vec<(i128, i128)>,
    pub norm_witnesses: Vec<(i128, i128)>,
    pub residual_witnesses: Vec<(i128, i128)>,
    pub output: usize,
}

/// Evaluate the dense integer model and retain the residual stream after every
/// transformer block. `None` denotes an invalid shape or range failure.
pub fn forward_with_trace(
    model: &DenseModel,
    tokens: &[usize],
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<ExecutionTrace> {
    if tokens.is_empty()
        || tokens.len() > model.t
        || model.scales.len() != model.weights.layers.len()
    {
        return None;
    }
    if model.h == 0 || model.g == 0 || model.h % model.g != 0 || model.dh == 0 {
        return None;
    }
    let mut x = Vec::with_capacity(tokens.len());
    for &token in tokens {
        x.push(model.weights.emb.get(token)?.clone());
    }
    let mut residual = Vec::with_capacity(model.weights.layers.len());
    let mut lookups = Vec::new();
    let mut gate_witnesses = Vec::new();
    let mut attention_witnesses = Vec::new();
    let mut linear_witnesses = Vec::new();
    let mut norm_witnesses = Vec::new();
    let mut residual_witnesses = Vec::new();
    for (weights, scales) in model.weights.layers.iter().zip(&model.scales) {
        if x.iter().any(|row| !stream_is_in_range(row)) {
            return None;
        }
        let a = x
            .iter()
            .map(|row| norm_checked(row, scales.m1, scales.k1, scales.eps1, &mut norm_witnesses))
            .collect::<Option<Vec<_>>>()?;
        let q_wide = a
            .iter()
            .enumerate()
            .map(|(position, row)| {
                rotated_shifted(
                    row,
                    &weights.wq,
                    scales.kq,
                    model.cs,
                    scales.rp,
                    position,
                    model.dh,
                    &model.tables,
                    &mut linear_witnesses,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        let k_wide = a
            .iter()
            .enumerate()
            .map(|(position, row)| {
                rotated_shifted(
                    row,
                    &weights.wk,
                    scales.kk,
                    model.cs,
                    scales.rp,
                    position,
                    model.dh,
                    &model.tables,
                    &mut linear_witnesses,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        if q_wide
            .iter()
            .chain(&k_wide)
            .flatten()
            .any(|v| !(-32768..=32767).contains(v))
        {
            return None;
        }
        let q = q_wide.into_iter().map(clamped).collect::<Vec<_>>();
        let k = k_wide.into_iter().map(clamped).collect::<Vec<_>>();
        let mut v = Vec::with_capacity(a.len());
        for row in &a {
            let wide = shifted(row, &weights.wv, scales.kv, &mut linear_witnesses)?;
            if wide.iter().any(|value| !(-32768..=32767).contains(value)) {
                return None;
            }
            v.push(clamped(wide));
        }
        let dq = model.h * model.dh;
        let dkv = model.g * model.dh;
        if q.iter()
            .chain(&k)
            .chain(&v)
            .any(|row| row.len() != dq && row.len() != dkv)
        {
            return None;
        }
        let mut o = Vec::with_capacity(tokens.len());
        for i in 0..tokens.len() {
            let mut row_o = vec![0i128; dq];
            for h in 0..model.h {
                let g = h / (model.h / model.g);
                for c in 0..model.dh {
                    let mut best = i128::MIN;
                    let mut scores = Vec::with_capacity(i + 1);
                    for j in 0..=i {
                        let mut score = 0i128;
                        for b in 0..model.dh {
                            attention_witnesses
                                .push((q[i][h * model.dh + b], k[j][g * model.dh + b]));
                            score = score.checked_add(
                                q[i][h * model.dh + b].checked_mul(k[j][g * model.dh + b])?,
                            )?;
                        }
                        best = best.max(score);
                        scores.push(score);
                    }
                    let mut z = 0i128;
                    let mut acc = 0i128;
                    for (j, score) in scores.into_iter().enumerate() {
                        let u = floor_div(best - score, 1i128 << scales.sh);
                        let e = if u >= 0 && (u as usize) < model.tables.u0 {
                            model.tables.ex[u as usize]
                        } else {
                            0
                        };
                        z = z.checked_add(e)?;
                        lookups.push(LookupWitness {
                            table: LookupTable::Softmax,
                            key: u,
                            value: e,
                        });
                        attention_witnesses.push((e, v[j][g * model.dh + c]));
                        acc = acc.checked_add(e.checked_mul(v[j][g * model.dh + c])?)?;
                    }
                    if z == 0 {
                        return None;
                    }
                    row_o[h * model.dh + c] = floor_div(2 * acc + z, 2 * z);
                }
            }
            o.push(row_o);
        }
        let ad = o
            .iter()
            .map(|row| shifted(row, &weights.wo, scales.ko, &mut linear_witnesses))
            .collect::<Option<Vec<_>>>()?;
        for (row, ad) in x.iter_mut().zip(ad) {
            for (dst, src) in row.iter_mut().zip(ad) {
                residual_witnesses.push((*dst, src));
                *dst = dst.checked_add(src)?;
            }
        }
        let m = x
            .iter()
            .map(|row| norm_checked(row, scales.m2, scales.k2, scales.eps2, &mut norm_witnesses))
            .collect::<Option<Vec<_>>>()?;
        let mut su = Vec::with_capacity(tokens.len());
        for row in m {
            let mut row_su = Vec::with_capacity(model.f);
            for j in 0..model.f {
                let gw = rq(dot(&row, &weights.wg, &mut linear_witnesses)?[j], scales.kg)?;
                let g0 = low8(gw);
                let gh = floor_div(gw - g0, 256);
                if !(-16384..=16383).contains(&gh) {
                    return None;
                }
                let gc = clamp(gh, 127);
                let key = 256 * gc + g0;
                let s = model.tables.ts(key)?;
                lookups.push(LookupWitness {
                    table: LookupTable::SiLU,
                    key,
                    value: s,
                });
                let u = shifted(&row, &weights.wu, scales.ku, &mut linear_witnesses)?[j];
                if !(-32768..=32767).contains(&u) {
                    return None;
                }
                gate_witnesses.push((s, u));
                row_su.push(s.checked_mul(u)?);
            }
            su.push(row_su);
        }
        let ad = su
            .iter()
            .map(|row| shifted(row, &weights.wd, scales.kd, &mut linear_witnesses))
            .collect::<Option<Vec<_>>>()?;
        for (row, ad) in x.iter_mut().zip(ad) {
            for (dst, src) in row.iter_mut().zip(ad) {
                residual_witnesses.push((*dst, src));
                *dst = dst.checked_add(src)?;
            }
        }
        residual.push(x.clone());
    }
    let af = norm_checked(x.last()?, mf, kf, eps_f, &mut norm_witnesses)?;
    let output = argmax(&dot(&af, &model.weights.wh, &mut linear_witnesses)?)?;
    Some(ExecutionTrace {
        tokens: tokens.to_vec(),
        residual,
        lookups,
        gate_witnesses,
        attention_witnesses,
        linear_witnesses,
        norm_witnesses,
        residual_witnesses,
        output,
    })
}

/// Evaluate the dense integer model, returning only the first greedy token.
pub fn forward(
    model: &DenseModel,
    tokens: &[usize],
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<usize> {
    forward_with_trace(model, tokens, mf, kf, eps_f).map(|trace| trace.output)
}

fn matrix_file(matrix: &[Vec<i128>]) -> Option<Vec<u8>> {
    let columns = matrix.first().map_or(0, Vec::len);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(matrix.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(columns as u64).to_le_bytes());
    for row in matrix {
        if row.len() != columns {
            return None;
        }
        for &value in row {
            bytes.extend_from_slice(&i64::try_from(value).ok()?.to_le_bytes());
        }
    }
    Some(bytes)
}

fn digest_files(files: &[Vec<u8>]) -> crate::fp::digest::Digest {
    let refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
    crate::fp::heads::configuration_digest(&refs)
}

/// Commit to every weight matrix in fixed model order: embedding, then
/// `wq,wk,wv,wo,wg,wu,wd` for each layer, then the output head.
pub fn weight_commitment(weights: &ModelWeights) -> Option<crate::fp::digest::Digest> {
    let mut files = vec![matrix_file(&weights.emb)?];
    for layer in &weights.layers {
        for matrix in [
            &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.wg, &layer.wu, &layer.wd,
        ] {
            files.push(matrix_file(matrix)?);
        }
    }
    files.push(matrix_file(&weights.wh)?);
    Some(digest_files(&files))
}

pub fn execution_commitment(trace: &ExecutionTrace) -> Option<crate::fp::digest::Digest> {
    let token_values: Vec<i128> = trace.tokens.iter().map(|&token| token as i128).collect();
    let mut files = vec![table_file(&token_values)?];
    for residual in &trace.residual {
        files.push(matrix_file(residual)?);
    }
    let lookup_values: Vec<i128> = trace
        .lookups
        .iter()
        .flat_map(|read| {
            [
                match read.table {
                    LookupTable::SiLU => 0,
                    LookupTable::Softmax => 1,
                },
                read.key,
                read.value,
            ]
        })
        .collect();
    files.push(table_file(&lookup_values)?);
    let gate_values: Vec<i128> = trace
        .gate_witnesses
        .iter()
        .flat_map(|&(s, u)| [s, u])
        .collect();
    files.push(table_file(&gate_values)?);
    let attention_values: Vec<i128> = trace
        .attention_witnesses
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();
    files.push(table_file(&attention_values)?);
    let linear_values: Vec<i128> = trace
        .linear_witnesses
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();
    files.push(table_file(&linear_values)?);
    let norm_values: Vec<i128> = trace
        .norm_witnesses
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect();
    files.push(table_file(&norm_values)?);
    let residual_values: Vec<i128> = trace
        .residual_witnesses
        .iter()
        .flat_map(|&(x, ad)| [x, ad])
        .collect();
    files.push(table_file(&residual_values)?);
    files.push(table_file(&[trace.output as i128])?);
    Some(digest_files(&files))
}

fn parameter_bytes(model: &DenseModel, mf: u32, kf: u32, eps_f: i128) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    for value in [model.t, model.dh, model.h, model.g, model.f, model.v] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&model.cs.to_le_bytes());
    bytes.extend_from_slice(&mf.to_le_bytes());
    bytes.extend_from_slice(&kf.to_le_bytes());
    bytes.extend_from_slice(&i64::try_from(eps_f).ok()?.to_le_bytes());
    for scales in &model.scales {
        for value in [
            scales.kq, scales.kk, scales.kv, scales.ko, scales.kg, scales.ku, scales.kd, scales.m1,
            scales.k1, scales.m2, scales.k2, scales.sh,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(u8::from(scales.rp));
        bytes.extend_from_slice(&i64::try_from(scales.eps1).ok()?.to_le_bytes());
        bytes.extend_from_slice(&i64::try_from(scales.eps2).ok()?.to_le_bytes());
    }
    Some(bytes)
}

fn parameter_commitment(
    model: &DenseModel,
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<crate::fp::digest::Digest> {
    let bytes = parameter_bytes(model, mf, kf, eps_f)?;
    Some(crate::fp::heads::configuration_digest(&[&bytes]))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockStatement {
    pub format_version: u32,
    pub parameters: crate::fp::digest::Digest,
    pub weights: crate::fp::digest::Digest,
    pub tables: TableDigests,
    pub tokens: crate::fp::digest::Digest,
    pub execution: crate::fp::digest::Digest,
    pub protocol: crate::fp::digest::Digest,
    pub output: usize,
    pub u0: usize,
    pub mf: u32,
    pub kf: u32,
    pub eps_f: i128,
}

impl BlockStatement {
    pub fn canonical_bytes(&self) -> Option<Vec<u8>> {
        let mut bytes = Vec::with_capacity(320);
        for digest in [
            &self.parameters,
            &self.weights,
            &self.tables.trope,
            &self.tables.texp,
            &self.tables.tsilu,
            &self.tables.tsig,
            &self.tokens,
            &self.execution,
            &self.protocol,
        ] {
            bytes.extend_from_slice(&digest.to_le_bytes());
        }
        bytes.extend_from_slice(&self.format_version.to_le_bytes());
        bytes.extend_from_slice(&(self.output as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.u0 as u64).to_le_bytes());
        bytes.extend_from_slice(&self.mf.to_le_bytes());
        bytes.extend_from_slice(&self.kf.to_le_bytes());
        bytes.extend_from_slice(&i64::try_from(self.eps_f).ok()?.to_le_bytes());
        Some(bytes)
    }
}

/// Construct the public statement and private reference trace for one complete
/// dense execution. This is a commitment binding, not a zero-knowledge proof.
pub fn bind_block(
    model: &DenseModel,
    tokens: &[usize],
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<(BlockStatement, ExecutionTrace)> {
    let trace = forward_with_trace(model, tokens, mf, kf, eps_f)?;
    let token_values: Vec<i128> = tokens.iter().map(|&token| token as i128).collect();
    let token_file = table_file(&token_values)?;
    let statement = BlockStatement {
        format_version: 1,
        parameters: parameter_commitment(model, mf, kf, eps_f)?,
        weights: weight_commitment(&model.weights)?,
        tables: serialize_public_tables(&model.tables)?.digests,
        tokens: crate::fp::heads::configuration_digest(&[&token_file]),
        execution: execution_commitment(&trace)?,
        protocol: protocol_commitment(&cpu_prover_rows(model, &trace)?)?,
        output: trace.output,
        u0: model.tables.u0,
        mf,
        kf,
        eps_f,
    };
    Some((statement, trace))
}

/// Recompute and check every public binding for a dense block execution.
pub fn verify_block(model: &DenseModel, tokens: &[usize], statement: &BlockStatement) -> bool {
    bind_block(model, tokens, statement.mf, statement.kf, statement.eps_f)
        .is_some_and(|(computed, _)| computed == *statement)
}

/// Encode a signed integer in the Goldilocks base field.
pub fn signed_field(value: i128) -> Option<crate::fp::field::FieldElement> {
    use crate::fp::field::FieldElement;
    if value >= 0 {
        return Some(FieldElement::from_u64(u64::try_from(value).ok()?));
    }
    let magnitude = u64::try_from(value.checked_neg()?).ok()?;
    Some(-FieldElement::from_u64(magnitude))
}

/// Encode a value as little-endian balanced base-256 limbs in `-128..=127`.
/// Returns `None` when `limbs` cannot represent the value exactly.
pub fn balanced_field_limbs(
    mut value: i128,
    limbs: usize,
) -> Option<Vec<crate::fp::field::FieldElement>> {
    let mut out = Vec::with_capacity(limbs);
    for _ in 0..limbs {
        let digit = low8(value);
        out.push(signed_field(digit)?);
        value = (value - digit) / 256;
    }
    if value == 0 {
        Some(out)
    } else {
        None
    }
}

pub fn encode_weight_matrix(
    matrix: &[Vec<i128>],
) -> Option<Vec<Vec<Vec<crate::fp::field::FieldElement>>>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| balanced_field_limbs(value, 2))
                .collect()
        })
        .collect()
}

pub fn encode_residual_matrix(
    matrix: &[Vec<i128>],
) -> Option<Vec<Vec<Vec<crate::fp::field::FieldElement>>>> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .map(|&value| balanced_field_limbs(value, 5))
                .collect()
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct CpuProverRows {
    pub weights: Vec<crate::fp::field::FieldElement>,
    pub residual: Vec<crate::fp::field::FieldElement>,
    pub lookup_cos: Vec<crate::fp::field::FieldElement>,
    pub lookup_sin: Vec<crate::fp::field::FieldElement>,
    pub lookup_ex: Vec<crate::fp::field::FieldElement>,
    pub lookup_ts: Vec<crate::fp::field::FieldElement>,
    pub lookup_sig: Vec<crate::fp::field::FieldElement>,
}

fn field_file(values: &[crate::fp::field::FieldElement]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + values.len() * 8);
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.0.to_le_bytes());
    }
    bytes
}

fn protocol_commitment(rows: &CpuProverRows) -> Option<crate::fp::digest::Digest> {
    let files = [
        field_file(&rows.weights),
        field_file(&rows.residual),
        field_file(&rows.lookup_cos),
        field_file(&rows.lookup_sin),
        field_file(&rows.lookup_ex),
        field_file(&rows.lookup_ts),
        field_file(&rows.lookup_sig),
    ];
    let refs: Vec<&[u8]> = files.iter().map(Vec::as_slice).collect();
    Some(crate::fp::heads::configuration_digest(&refs))
}

fn append_limbs(
    output: &mut Vec<crate::fp::field::FieldElement>,
    matrix: &[Vec<i128>],
    limbs: usize,
) -> Option<()> {
    for row in matrix {
        for &value in row {
            output.extend(balanced_field_limbs(value, limbs)?);
        }
    }
    Some(())
}

fn encode_table(values: &[i128]) -> Option<Vec<crate::fp::field::FieldElement>> {
    values.iter().map(|&value| signed_field(value)).collect()
}

/// Flatten one execution into the CPU prover's deterministic field rows.
/// Embeddings use four balanced limbs, other weights two, residuals five, and
/// public lookup values one base-field element. Matrix order matches
/// `weight_commitment`.
pub fn cpu_prover_rows(model: &DenseModel, trace: &ExecutionTrace) -> Option<CpuProverRows> {
    let mut weights = Vec::new();
    append_limbs(&mut weights, &model.weights.emb, 4)?;
    for layer in &model.weights.layers {
        for matrix in [
            &layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.wg, &layer.wu, &layer.wd,
        ] {
            append_limbs(&mut weights, matrix, 2)?;
        }
    }
    append_limbs(&mut weights, &model.weights.wh, 2)?;

    let mut residual = Vec::new();
    for matrix in &trace.residual {
        append_limbs(&mut residual, matrix, 5)?;
    }

    Some(CpuProverRows {
        weights,
        residual,
        lookup_cos: encode_table(&model.tables.cos)?,
        lookup_sin: encode_table(&model.tables.sin)?,
        lookup_ex: encode_table(&model.tables.ex)?,
        lookup_ts: encode_table(&model.tables.ts)?,
        lookup_sig: encode_table(&model.tables.sig)?,
    })
}

#[derive(Debug)]
pub struct ProtocolInputs {
    pub weights: crate::fp::tensor::Tensor,
    pub residual: crate::fp::tensor::Tensor,
    pub lookups: Vec<crate::fp::forest::ForestInput>,
    pub lookup_arguments: Vec<crate::fp::lkp::LkpPublic>,
}

fn field_tensor(values: &[crate::fp::field::FieldElement]) -> Option<crate::fp::tensor::Tensor> {
    use crate::fp::field::ExtensionElement as E;
    if values.is_empty() {
        return None;
    }
    let variables = crate::fp::tensor::ceil_log2(values.len() as u64);
    let mut tensor = crate::fp::tensor::Tensor::eq(&[], E::ONE);
    tensor.push_dense(variables);
    let mut dense: Vec<E> = values.iter().map(|&value| E::from_base(value)).collect();
    dense.resize(1usize << variables, E::ZERO);
    tensor.set_values(dense, 0);
    Some(tensor)
}

fn lookup_input(
    name: &str,
    values: &[crate::fp::field::FieldElement],
) -> Option<crate::fp::forest::ForestInput> {
    if values.is_empty() {
        return None;
    }
    let depth = crate::fp::tensor::ceil_log2(values.len() as u64).max(10);
    let mut tau: Vec<u64> = values.iter().map(|value| value.0).collect();
    tau.resize(1usize << depth, *tau.last()?);
    Some(crate::fp::forest::ForestInput {
        calls: vec![crate::fp::forest::Call::new(
            name,
            depth,
            depth,
            0,
            (values.len() - 1) as u64,
            Some(tau),
        )],
        pub_counts: vec![0],
        lid: depth,
    })
}

fn lookup_argument(
    name: &str,
    values: &[crate::fp::field::FieldElement],
) -> Option<crate::fp::lkp::LkpPublic> {
    use crate::fp::field::ExtensionElement as E;
    if values.is_empty() {
        return None;
    }
    let depth = crate::fp::tensor::ceil_log2(values.len() as u64).max(10);
    let mut rows: Vec<E> = values.iter().map(|&value| E::from_base(value)).collect();
    rows.resize(1usize << depth, *rows.last()?);
    Some(crate::fp::lkp::LkpPublic {
        context: name.into(),
        parts: vec![depth],
        d: depth,
        mu: 0,
        live: 0,
        h: 0,
        g: 0,
        shifted: false,
        read_from: vec![],
        keys: vec![],
        tables: vec![crate::fp::lkp::LkpTablePart {
            first: 0,
            log_rows: depth,
            function: 0,
            rows,
        }],
        padding: vec![],
        last_key_level_order: vec![],
    })
}

/// Construct the existing protocol's arithmetic tensors and lookup forests.
/// Short tables are padded with their last entry to the minimum forest depth;
/// arithmetic tensors are zero-padded to a power of two.
pub fn protocol_inputs(model: &DenseModel, trace: &ExecutionTrace) -> Option<ProtocolInputs> {
    let rows = cpu_prover_rows(model, trace)?;
    Some(ProtocolInputs {
        weights: field_tensor(&rows.weights)?,
        residual: field_tensor(&rows.residual)?,
        lookups: vec![
            lookup_input("cos", &rows.lookup_cos)?,
            lookup_input("sin", &rows.lookup_sin)?,
            lookup_input("EX", &rows.lookup_ex)?,
            lookup_input("Ts", &rows.lookup_ts)?,
            lookup_input("Sig", &rows.lookup_sig)?,
        ],
        lookup_arguments: vec![
            lookup_argument("cos", &rows.lookup_cos)?,
            lookup_argument("sin", &rows.lookup_sin)?,
            lookup_argument("EX", &rows.lookup_ex)?,
            lookup_argument("Ts", &rows.lookup_ts)?,
            lookup_argument("Sig", &rows.lookup_sig)?,
        ],
    })
}

pub const LABEL_M1_CHAIN: u64 = 0x4d31_0001;

pub fn m1_first_query(statement: &BlockStatement) -> Option<crate::fp::chain::FirstQuery> {
    Some(crate::fp::chain::FirstQuery {
        label: LABEL_M1_CHAIN,
        parameters: statement.parameters,
        statement: statement.canonical_bytes()?,
        record: vec![],
        salted: false,
        run: None,
    })
}

/// Prover-side transcript writer. Message encoding and challenge derivation are
/// the exact dual of the verifier's `Tr` schedule.
pub struct ProverTr<'a> {
    chain: &'a mut crate::fp::chain::Chain,
    bytes: Vec<u8>,
    messages: Vec<crate::fp::layout::PlanMessage>,
    compact_offsets: Vec<usize>,
    compact: bool,
    last_words: Vec<u64>,
}

impl<'a> ProverTr<'a> {
    pub fn new(chain: &'a mut crate::fp::chain::Chain) -> Self {
        Self {
            chain,
            bytes: vec![],
            messages: vec![],
            compact_offsets: vec![],
            compact: false,
            last_words: vec![],
        }
    }

    pub fn write(&mut self, values: &[crate::fp::field::ExtensionElement]) {
        for value in values {
            for coordinate in value.0 {
                self.last_words.push(coordinate);
                self.bytes.extend_from_slice(&coordinate.to_le_bytes());
            }
        }
        self.messages.push(crate::fp::layout::PlanMessage::fixed(
            "scalars",
            "M1",
            crate::fp::layout::MessageKind::Scalars,
            (values.len() * 24) as u64,
        ));
    }

    pub fn set_compact(&mut self, compact: bool) {
        self.compact = compact;
    }

    /// Write base-field endpoints in 8 bytes. The chain still absorbs the full
    /// three-coordinate extension element, so verifier challenges are unchanged.
    pub fn write_base(&mut self, values: &[crate::fp::field::ExtensionElement]) {
        if values.is_empty()
            || !self.compact
            || values
                .iter()
                .any(|value| value.0[1] != 0 || value.0[2] != 0)
        {
            self.write(values);
            return;
        }
        self.compact_offsets.push(self.bytes.len());
        for value in values {
            let coordinate = value.0[0];
            self.last_words.extend_from_slice(&[coordinate, 0, 0]);
            self.bytes.extend_from_slice(&coordinate.to_le_bytes());
        }
        self.messages.push(crate::fp::layout::PlanMessage::fixed(
            "scalars",
            "M1",
            crate::fp::layout::MessageKind::Scalars,
            (values.len() * 8) as u64,
        ));
    }

    pub fn sum(&mut self, value: crate::fp::field::ExtensionElement) {
        self.write(&[value]);
    }

    fn close(&mut self, kind: crate::fp::chain::ChainKind) {
        let message = core::mem::take(&mut self.last_words);
        self.chain.close(kind, &message);
    }

    pub fn answer(
        &mut self,
        round: crate::fp::model::RoundName,
    ) -> crate::fp::field::ExtensionElement {
        use crate::fp::{chain::ChainKind, field::ExtensionElement as E, model::RoundName};
        self.close(match round {
            RoundName::PscEpsilon => ChainKind::Epsilon,
            RoundName::PscValues => ChainKind::Empty,
            _ => ChainKind::Algebraic,
        });
        E::from_words(self.chain.draw_e())
    }

    pub fn answer_not_zero(
        &mut self,
        _round: crate::fp::model::RoundName,
    ) -> crate::fp::field::ExtensionElement {
        self.close(crate::fp::chain::ChainKind::Algebraic);
        crate::fp::field::ExtensionElement::from_words(self.chain.draw_e_nonzero())
    }

    pub fn point(
        &mut self,
        _round: crate::fp::model::RoundName,
        count: usize,
    ) -> Vec<crate::fp::field::ExtensionElement> {
        self.close(crate::fp::chain::ChainKind::Algebraic);
        (0..count)
            .map(|_| crate::fp::field::ExtensionElement::from_words(self.chain.draw_e()))
            .collect()
    }

    pub fn empty(&mut self, _round: crate::fp::model::RoundName) {
        self.close(crate::fp::chain::ChainKind::Empty);
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn take_compact_offsets(&mut self) -> Vec<usize> {
        core::mem::take(&mut self.compact_offsets)
    }

    pub fn take_parts(&mut self) -> (Vec<u8>, Vec<crate::fp::layout::PlanMessage>) {
        (
            core::mem::take(&mut self.bytes),
            core::mem::take(&mut self.messages),
        )
    }
}

/// Prover-side extension of `Summand`: supply each univariate round
/// polynomial from the private witness.
pub trait PscRoundProver: crate::fp::psc::Summand {
    fn round_polynomial(
        &mut self,
        round: usize,
        weight: &crate::fp::psc::Weight,
        running: crate::fp::field::ExtensionElement,
        end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement>;

    fn begin(&mut self, _epsilon: crate::fp::field::ExtensionElement) {}

    fn after_challenge(&mut self, _round: usize, _challenge: crate::fp::field::ExtensionElement) {}
}

/// Generate a PSC transcript for any summand that can supply its round
/// polynomials. Every polynomial is checked against the running sum before it
/// is emitted, and the final relation is checked before the endpoints close.
pub fn prove_psc<S: PscRoundProver>(
    transcript: &mut ProverTr,
    claim: &crate::fp::psc::SumClaim,
    summand: &mut S,
    end_values: &[crate::fp::field::ExtensionElement],
) -> Option<Vec<crate::fp::field::ExtensionElement>> {
    use crate::fp::{
        field::ExtensionElement as E,
        model::RoundName,
        psc::{poly_at, psc_vars},
    };
    if end_values.len() != summand.shown() {
        return None;
    }
    let vars = psc_vars(claim, summand);
    let variable_count = claim.vars.len();
    transcript.sum(E::ZERO);
    let eps = transcript.answer(RoundName::PscEpsilon);
    summand.begin(eps);
    let mut running = eps * claim.target;
    let mut challenges = vec![eps];
    for (round, weight) in vars.iter().enumerate() {
        let q = summand.round_polynomial(round, weight, running, end_values);
        if q.len() != weight.d || weight.apply(&q) != running {
            return None;
        }
        let free: Vec<E> = q
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != weight.fixed)
            .map(|(_, value)| *value)
            .collect();
        transcript.write(&free);
        let challenge = if round >= variable_count {
            transcript.answer_not_zero(RoundName::PscRound)
        } else {
            transcript.answer(RoundName::PscRound)
        };
        running = poly_at(&q, challenge);
        summand.after_challenge(round, challenge);
        challenges.push(challenge);
    }
    let relation = summand.at(&challenges[1..=variable_count], end_values);
    if relation.len() != 1 + summand.kept() || running != eps * relation[0] {
        return None;
    }
    transcript.write_base(end_values);
    if !end_values.is_empty() {
        transcript.empty(RoundName::PscValues);
    }
    Some(challenges)
}

struct ProductRoundProver;

impl crate::fp::psc::Summand for ProductRoundProver {
    fn shown(&self) -> usize {
        2
    }
    fn degree(&self) -> usize {
        2
    }
    fn at(
        &self,
        _point: &[crate::fp::field::ExtensionElement],
        end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        vec![end_values[0] * end_values[1]]
    }
}

impl PscRoundProver for ProductRoundProver {
    fn round_polynomial(
        &mut self,
        _round: usize,
        weight: &crate::fp::psc::Weight,
        running: crate::fp::field::ExtensionElement,
        _end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        let mut q = vec![crate::fp::field::ExtensionElement::ZERO; weight.d];
        q[0] = running;
        q
    }
}

#[derive(Clone)]
pub struct DotProductRoundProver {
    pub pairs: Vec<(
        crate::fp::field::ExtensionElement,
        crate::fp::field::ExtensionElement,
    )>,
    pub witness_pairs: usize,
    pub variables: usize,
    current_a: Vec<crate::fp::field::ExtensionElement>,
    current_b: Vec<crate::fp::field::ExtensionElement>,
    epsilon: crate::fp::field::ExtensionElement,
}

impl crate::fp::psc::Summand for DotProductRoundProver {
    fn shown(&self) -> usize {
        2 * self.pairs.len()
    }
    fn degree(&self) -> usize {
        2
    }
    fn at(
        &self,
        point: &[crate::fp::field::ExtensionElement],
        end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        assert_eq!(point.len(), self.variables);
        let mut a: Vec<_> = end_values.iter().step_by(2).copied().collect();
        let mut b: Vec<_> = end_values.iter().skip(1).step_by(2).copied().collect();
        for &challenge in point {
            crate::fp::tensor::fold_low_pair_once(&mut a, &mut b, challenge);
        }
        let sum = a
            .iter()
            .zip(&b)
            .fold(crate::fp::field::ExtensionElement::ZERO, |sum, (&a, &b)| {
                sum + a * b
            });
        vec![sum]
    }
}

impl PscRoundProver for DotProductRoundProver {
    fn round_polynomial(
        &mut self,
        round: usize,
        weight: &crate::fp::psc::Weight,
        running: crate::fp::field::ExtensionElement,
        _end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        if round >= self.variables {
            let mut q = vec![crate::fp::field::ExtensionElement::ZERO; weight.d];
            q[0] = running;
            return q;
        }
        let mut zero = crate::fp::field::ExtensionElement::ZERO;
        let mut one = crate::fp::field::ExtensionElement::ZERO;
        let mut quadratic = crate::fp::field::ExtensionElement::ZERO;
        for (a, b) in self
            .current_a
            .chunks_exact(2)
            .zip(self.current_b.chunks_exact(2))
        {
            zero += a[0] * b[0];
            one += a[1] * b[1];
            quadratic += (a[1] - a[0]) * (b[1] - b[0]);
        }
        vec![
            self.epsilon * zero,
            self.epsilon * (one - zero - quadratic),
            self.epsilon * quadratic,
        ]
    }

    fn begin(&mut self, epsilon: crate::fp::field::ExtensionElement) {
        self.epsilon = epsilon;
    }

    fn after_challenge(&mut self, round: usize, challenge: crate::fp::field::ExtensionElement) {
        if round < self.variables {
            crate::fp::tensor::fold_low_pair_once(
                &mut self.current_a,
                &mut self.current_b,
                challenge,
            );
        }
    }
}

pub fn prove_psc_dot_product(
    transcript: &mut ProverTr,
    mut pairs: Vec<(
        crate::fp::field::ExtensionElement,
        crate::fp::field::ExtensionElement,
    )>,
) -> (
    crate::fp::psc::SumClaim,
    DotProductRoundProver,
    Vec<crate::fp::field::ExtensionElement>,
) {
    use crate::fp::field::ExtensionElement as E;
    let target = pairs.iter().fold(E::ZERO, |sum, &(a, b)| sum + a * b);
    let witness_pairs = pairs.len();
    let padded = witness_pairs.max(1).next_power_of_two();
    pairs.resize(padded, (E::ZERO, E::ZERO));
    let variables = padded.ilog2() as usize;
    let claim = crate::fp::psc::SumClaim {
        vars: vec![crate::fp::psc::Weight::plain(3); variables],
        masks: vec![],
        target,
    };
    let current_a = pairs.iter().map(|pair| pair.0).collect();
    let current_b = pairs.iter().map(|pair| pair.1).collect();
    let mut summand = DotProductRoundProver {
        pairs,
        witness_pairs,
        variables,
        current_a,
        current_b,
        epsilon: E::ONE,
    };
    let end_values: Vec<E> = summand.pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
    let challenges = prove_psc(transcript, &claim, &mut summand, &end_values).unwrap();
    (claim, summand, challenges)
}

#[derive(Clone)]
pub struct ResidualRoundProver {
    pub pairs: Vec<(
        crate::fp::field::ExtensionElement,
        crate::fp::field::ExtensionElement,
    )>,
}

impl crate::fp::psc::Summand for ResidualRoundProver {
    fn shown(&self) -> usize {
        0
    }
    fn degree(&self) -> usize {
        1
    }
    fn at(
        &self,
        point: &[crate::fp::field::ExtensionElement],
        _end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        let mut sum = crate::fp::field::ExtensionElement::ZERO;
        for &(x, ad) in &self.pairs {
            sum += x + ad * point[0];
        }
        vec![sum]
    }
}

impl PscRoundProver for ResidualRoundProver {
    fn round_polynomial(
        &mut self,
        _round: usize,
        _weight: &crate::fp::psc::Weight,
        running: crate::fp::field::ExtensionElement,
        _end_values: &[crate::fp::field::ExtensionElement],
    ) -> Vec<crate::fp::field::ExtensionElement> {
        let (x, ad) = self.pairs.iter().fold(
            (
                crate::fp::field::ExtensionElement::ZERO,
                crate::fp::field::ExtensionElement::ZERO,
            ),
            |(sx, sa), &(x, ad)| (sx + x, sa + ad),
        );
        let target = x + ad;
        if target.is_zero() {
            return vec![crate::fp::field::ExtensionElement::ZERO; 2];
        }
        let eps = running * target.inverse();
        vec![eps * x, eps * ad]
    }
}

pub fn prove_psc_residual(
    transcript: &mut ProverTr,
    pairs: Vec<(
        crate::fp::field::ExtensionElement,
        crate::fp::field::ExtensionElement,
    )>,
) -> (
    crate::fp::psc::SumClaim,
    ResidualRoundProver,
    Vec<crate::fp::field::ExtensionElement>,
) {
    use crate::fp::field::ExtensionElement as E;
    let target = pairs.iter().fold(E::ZERO, |sum, &(x, ad)| sum + x + ad);
    let claim = crate::fp::psc::SumClaim {
        vars: vec![crate::fp::psc::Weight::equality(
            crate::fp::field::ExtensionElement::ONE,
            2,
        )],
        masks: vec![],
        target,
    };
    let mut summand = ResidualRoundProver { pairs };
    let challenges = prove_psc(transcript, &claim, &mut summand, &[]).unwrap();
    (claim, summand, challenges)
}

/// Prove `a * b` through the existing PSC protocol. The claim uses one
/// equality-weighted variable and the ordinary two-input product summand.
/// Round polynomials are constant at the claimed product, so every verifier
/// challenge preserves the running sum.
pub fn prove_psc_product(
    transcript: &mut ProverTr,
    z: crate::fp::field::ExtensionElement,
    a: crate::fp::field::ExtensionElement,
    b: crate::fp::field::ExtensionElement,
) -> (
    crate::fp::psc::SumClaim,
    [crate::fp::field::ExtensionElement; 3],
) {
    let claim = crate::fp::psc::SumClaim {
        vars: vec![crate::fp::psc::Weight::equality(z, 2)],
        masks: vec![],
        target: a * b,
    };
    let mut summand = ProductRoundProver;
    let challenges = prove_psc(transcript, &claim, &mut summand, &[a, b]).unwrap();
    (claim, [challenges[0], challenges[1], challenges[2]])
}

#[derive(Debug)]
pub struct ProverPsc {
    pub target: crate::fp::field::ExtensionElement,
    pub eps: crate::fp::field::ExtensionElement,
    pub claim: crate::fp::field::ExtensionElement,
    pub first: u32,
    pub start: crate::fp::model::Aff,
    pub vars: Vec<crate::fp::psc::Weight>,
    pub challenges: Vec<crate::fp::field::ExtensionElement>,
    pub ended: bool,
    pub row_made: bool,
}

/// Begin a forest-shaped PSC round sequence with zero hiding sum.
pub fn psc_prove_begin(
    transcript: &mut ProverTr,
    target: crate::fp::field::ExtensionElement,
) -> ProverPsc {
    use crate::fp::{field::ExtensionElement as E, model::RoundName};
    transcript.sum(E::ZERO);
    let eps = transcript.answer(RoundName::PscEpsilon);
    ProverPsc {
        target,
        eps,
        claim: eps * target,
        first: 0,
        start: crate::fp::model::Aff::default(),
        vars: vec![],
        challenges: vec![],
        ended: false,
        row_made: false,
    }
}

/// Emit one checked forest-shaped PSC round polynomial.
pub fn psc_prove_round(
    transcript: &mut ProverTr,
    proof: &mut ProverPsc,
    weight: crate::fp::psc::Weight,
    q: &[crate::fp::field::ExtensionElement],
    nonzero: bool,
) -> Option<crate::fp::field::ExtensionElement> {
    use crate::fp::{model::RoundName, psc::poly_at};
    if proof.ended || q.len() != weight.d || weight.apply(q) != proof.claim {
        return None;
    }
    let free: Vec<_> = q
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != weight.fixed)
        .map(|(_, value)| *value)
        .collect();
    transcript.write(&free);
    let challenge = if nonzero {
        transcript.answer_not_zero(RoundName::PscY)
    } else {
        transcript.answer(RoundName::PscRound)
    };
    proof.claim = poly_at(q, challenge);
    proof.vars.push(weight);
    proof.challenges.push(challenge);
    Some(challenge)
}

/// Close a forest-shaped PSC sequence and hide its endpoint values.
pub fn psc_prove_ends(
    transcript: &mut ProverTr,
    proof: &mut ProverPsc,
    degree_in_y: u32,
    q: &[crate::fp::field::ExtensionElement],
    values: &[crate::fp::field::ExtensionElement],
) -> Option<()> {
    if degree_in_y == 0 || values.is_empty() {
        return None;
    }
    psc_prove_round(
        transcript,
        proof,
        crate::fp::psc::Weight::at_zero(degree_in_y as usize + 1),
        q,
        true,
    )?;
    transcript.write_base(values);
    transcript.empty(crate::fp::model::RoundName::PscValues);
    proof.ended = true;
    Some(())
}

pub fn psc_prove_begin_slots(
    fork: &crate::fp::model::HFork,
    transcript: &mut ProverTr,
    slot: &mut u32,
    start: crate::fp::model::Aff,
) -> Option<ProverPsc> {
    let target = fork.public_part(&start).ok()?;
    let first = *slot;
    let mut proof = psc_prove_begin(transcript, target);
    proof.first = first;
    proof.start = start;
    *slot += 1;
    Some(proof)
}

pub fn psc_prove_round_slots(
    transcript: &mut ProverTr,
    slot: &mut u32,
    proof: &mut ProverPsc,
    weight: crate::fp::psc::Weight,
    q: &[crate::fp::field::ExtensionElement],
    nonzero: bool,
) -> Option<crate::fp::field::ExtensionElement> {
    let degree = weight.d;
    let challenge = psc_prove_round(transcript, proof, weight, q, nonzero)?;
    *slot += (degree - 1) as u32;
    Some(challenge)
}

pub fn psc_prove_ends_slots(
    fork: &mut crate::fp::model::HFork,
    transcript: &mut ProverTr,
    slot: &mut u32,
    proof: &mut ProverPsc,
    degree_in_y: u32,
    q: &[crate::fp::field::ExtensionElement],
    values: &[crate::fp::field::ExtensionElement],
) -> Option<u32> {
    if degree_in_y == 0 || values.is_empty() {
        return None;
    }
    psc_prove_round_slots(
        transcript,
        slot,
        proof,
        crate::fp::psc::Weight::at_zero(degree_in_y as usize + 1),
        q,
        true,
    )?;
    let first = *slot;
    for (index, &value) in values.iter().enumerate() {
        transcript.write_base(&[value]);
        fork.hide(
            first + index as u32,
            value,
            crate::fp::field::ExtensionElement::ONE,
        )
        .ok()?;
    }
    transcript.empty(crate::fp::model::RoundName::PscValues);
    *slot += values.len() as u32;
    proof.ended = true;
    Some(first)
}

pub fn psc_prove_row(
    fork: &mut crate::fp::model::HFork,
    proof: &mut ProverPsc,
    s0: crate::fp::field::ExtensionElement,
    kept: &crate::fp::model::Aff,
) -> Option<crate::fp::model::Aff> {
    use crate::fp::{
        field::ExtensionElement as E,
        model::Aff,
        psc::{hp_form, poly_at},
    };
    if proof.row_made || proof.vars.is_empty() {
        return None;
    }
    proof.row_made = true;
    let form = hp_form(&proof.vars, &proof.challenges);
    let mut out = Aff::constant(-(proof.claim - proof.eps * s0));
    for (index, coefficient) in form.into_iter().enumerate() {
        fork.hide(proof.first + index as u32, E::ZERO, -E::ONE)
            .ok()?;
        out.add(&Aff::slot(proof.first + index as u32), coefficient);
    }
    let mut product = proof.eps;
    for (weight, &challenge) in proof.vars.iter().zip(&proof.challenges) {
        product *= poly_at(&weight.psi, challenge);
    }
    out.add(&Aff::constant(proof.target).minus(&proof.start), product);
    out.add(kept, proof.eps);
    Some(out)
}

pub fn prove_product_level(
    fork: &mut crate::fp::model::HFork,
    transcript: &mut ProverTr,
    slot: &mut u32,
    mu: u32,
    rho: &[crate::fp::field::ExtensionElement],
    start: &mut crate::fp::model::Aff,
    order: &[u32],
    polynomials: &[Vec<crate::fp::field::ExtensionElement>],
    final_polynomial: &[crate::fp::field::ExtensionElement],
    values: [crate::fp::field::ExtensionElement; 2],
) -> Option<crate::fp::forest::ProductLevel> {
    use crate::fp::{field::ExtensionElement as E, model::RoundName, psc::Weight};
    if rho.len() != mu as usize || polynomials.len() != mu as usize {
        return None;
    }
    let mut proof = psc_prove_begin_slots(fork, transcript, slot, start.clone())?;
    let mut r = vec![E::ZERO; mu as usize];
    for &variable in order {
        let index = (mu - 1 - variable) as usize;
        r[index] = psc_prove_round_slots(
            transcript,
            slot,
            &mut proof,
            Weight::equality(rho[index], 3),
            &polynomials[index],
            false,
        )?;
    }
    let s_l = psc_prove_ends_slots(
        fork,
        transcript,
        slot,
        &mut proof,
        2,
        final_polynomial,
        &values,
    )?;
    let s_r = s_l + 1;
    let tau = transcript.answer(RoundName::Line);
    let row = psc_prove_row(
        fork,
        &mut proof,
        values[0] * values[1],
        &crate::fp::model::Aff::default(),
    )?;
    Some(crate::fp::forest::ProductLevel {
        r,
        s_l,
        s_r,
        tau,
        row,
    })
}

pub fn prove_forest_leaves(
    fork: &mut crate::fp::model::HFork,
    transcript: &mut ProverTr,
    slot: &mut u32,
    start: crate::fp::model::Aff,
    points: &[crate::fp::field::ExtensionElement],
    degrees: &[u32],
    polynomials: &[Vec<crate::fp::field::ExtensionElement>],
) -> Option<(ProverPsc, Vec<crate::fp::field::ExtensionElement>)> {
    use crate::fp::psc::Weight;
    if points.len() != degrees.len() || points.len() != polynomials.len() {
        return None;
    }
    let mut proof = psc_prove_begin_slots(fork, transcript, slot, start)?;
    let mut challenges = Vec::with_capacity(points.len());
    for ((point, degree), polynomial) in points.iter().zip(degrees).zip(polynomials) {
        if *degree == 0 || *degree > 64 {
            return None;
        }
        challenges.push(psc_prove_round_slots(
            transcript,
            slot,
            &mut proof,
            Weight::equality(*point, *degree as usize + 1),
            polynomial,
            false,
        )?);
    }
    Some((proof, challenges))
}

pub fn prove_forest_bit_row(
    fork: &mut crate::fp::model::HFork,
    transcript: &mut ProverTr,
    slot: &mut u32,
    points: &[crate::fp::field::ExtensionElement],
    polynomials: &[Vec<crate::fp::field::ExtensionElement>],
    final_polynomial: &[crate::fp::field::ExtensionElement],
    bit: crate::fp::field::ExtensionElement,
) -> Option<(
    crate::fp::model::Aff,
    u32,
    Vec<crate::fp::field::ExtensionElement>,
)> {
    use crate::fp::{model::Aff, psc::Weight};
    if points.len() != polynomials.len() {
        return None;
    }
    let mut proof = psc_prove_begin_slots(fork, transcript, slot, Aff::default())?;
    let mut challenges = Vec::with_capacity(points.len());
    for (point, polynomial) in points.iter().zip(polynomials) {
        challenges.push(psc_prove_round_slots(
            transcript,
            slot,
            &mut proof,
            Weight::equality(*point, 3),
            polynomial,
            false,
        )?);
    }
    let s_bit = psc_prove_ends_slots(
        fork,
        transcript,
        slot,
        &mut proof,
        2,
        final_polynomial,
        &[bit],
    )?;
    let row = psc_prove_row(fork, &mut proof, bit * bit - bit, &Aff::default())?;
    Some((row, s_bit, challenges))
}

pub struct GeneratedForest {
    pub forest: crate::fp::forest::Forest,
    pub bytes: Vec<u8>,
    pub messages: Vec<crate::fp::layout::PlanMessage>,
}

fn field_pow(
    mut value: crate::fp::field::ExtensionElement,
    mut exponent: u64,
) -> crate::fp::field::ExtensionElement {
    let mut out = crate::fp::field::ExtensionElement::ONE;
    while exponent != 0 {
        if exponent & 1 != 0 {
            out *= value;
        }
        value *= value;
        exponent >>= 1;
    }
    out
}

/// Generate a complete forest transcript for public tables and lookup counts.
/// The padding stages use zero values; the public table itself may be nonzero
/// and is evaluated at the derived point.
pub fn prove_forest(
    transcript: &mut ProverTr,
    mut input: crate::fp::forest::ForestInput,
) -> Option<GeneratedForest> {
    use crate::fp::{
        field::{ExtensionElement as E, FieldElement as F},
        forest::{forest_elems, forest_shape, hand_down, order_default, order_forest_range},
        model::{Aff, HFork, RoundName},
    };
    let (iota, first, call_of, d_t, k_t, nt, r_k, c_k) = forest_shape(&input).ok()?;
    let nc = input.calls.len();
    let v = d_t + r_k;
    let d = k_t + c_k;
    let m = 5 + d;
    let theta = if nc > 1 {
        transcript.answer(RoundName::LkpShift)
    } else {
        E::ZERO
    };
    let alpha = transcript.answer(RoundName::Lkp);
    let pad_key = |index: usize| {
        let call = &input.calls[index];
        call.tau_at(call.t0) + theta * E::from_base(F(index as u64))
    };
    let empty_keys = ((1u64 << r_k) - nt as u64) << d_t;
    let mut a = field_pow(alpha - pad_key(0), empty_keys);
    for y in 0..nt as usize {
        a *= field_pow(alpha - pad_key(call_of[y] as usize), input.pub_counts[y]);
    }

    let slot0 = 0u32;
    let mut slot = slot0;
    let mut fork = HFork::default();
    fork.slots(forest_elems(v, d));
    let mut rows = Vec::new();
    let mut s_g = [0u32; 32];
    let mut s_z = 0;
    let mut s_p = 0;
    for j in 0..31 {
        let start = if j == 0 {
            Aff::default()
        } else {
            Aff::slot(s_p)
        };
        let mut proof = psc_prove_begin_slots(&fork, transcript, &mut slot, start)?;
        let at = psc_prove_ends_slots(
            &mut fork,
            transcript,
            &mut slot,
            &mut proof,
            3,
            &[E::ZERO; 4],
            &vec![E::ZERO; if j == 0 { 3 } else { 2 }],
        )?;
        if j == 0 {
            s_z = at;
        }
        s_p = if j == 0 { at + 1 } else { at };
        s_g[j] = s_p + 1;
        let power = fork.shown(s_p).ok()? * fork.shown(s_p).ok()? * fork.shown(s_g[j]).ok()?;
        let s0 = if j == 0 {
            fork.shown(s_z).ok()? - a * power
        } else {
            power
        };
        rows.push(psc_prove_row(&mut fork, &mut proof, s0, &Aff::default())?);
    }
    s_g[31] = s_p;
    let by_variable = transcript.point(RoundName::LkpPlanes, 5);
    let mut rho_t: Vec<E> = by_variable.into_iter().rev().collect();
    let eq = crate::fp::tensor::eq_table_rev(&rho_t);
    let mut ct = Aff::default();
    for j in 0..32 {
        ct.add(&Aff::slot(s_g[j]), eq[j]);
    }

    let mut rho = Vec::new();
    let mut start = Aff::slot(s_z);
    for mu in 0..v {
        let order = if mu + 1 == v {
            order_forest_range(mu, r_k)
        } else {
            order_default(mu)
        };
        let level = prove_product_level(
            &mut fork,
            transcript,
            &mut slot,
            mu,
            &rho,
            &mut start,
            &order,
            &vec![vec![E::ZERO; 3]; mu as usize],
            &[E::ZERO; 3],
            [E::ZERO, E::ZERO],
        )?;
        hand_down(&level, &mut rho, &mut start);
        rows.push(level.row);
    }
    let zx = rho[..d_t as usize].to_vec();
    let zy = rho[d_t as usize..].to_vec();
    let c_end = start;
    let mut table_start = ct;
    for l in 0..d {
        let mu = 5 + l;
        let level = prove_product_level(
            &mut fork,
            transcript,
            &mut slot,
            mu,
            &rho_t,
            &mut table_start,
            &order_default(mu),
            &vec![vec![E::ZERO; 3]; mu as usize],
            &[E::ZERO; 3],
            [E::ZERO, E::ZERO],
        )?;
        hand_down(&level, &mut rho_t, &mut table_start);
        rows.push(level.row);
    }
    if rho_t.len() != m as usize {
        return None;
    }
    let (leaves, zb0) = prove_forest_leaves(
        &mut fork,
        transcript,
        &mut slot,
        table_start,
        &rho_t,
        &(0..m)
            .map(|j| if j < d { 2 } else { 1 })
            .collect::<Vec<_>>(),
        &(0..m)
            .map(|j| vec![E::ZERO; if j < d { 3 } else { 2 }])
            .collect::<Vec<_>>(),
    )?;
    let by_variable = transcript.point(RoundName::Zero, m as usize);
    let r5: Vec<E> = by_variable.into_iter().rev().collect();
    let (bit_row, s_bit, zb1) = prove_forest_bit_row(
        &mut fork,
        transcript,
        &mut slot,
        &r5,
        &vec![vec![E::ZERO; 3]; m as usize],
        &[E::ZERO; 3],
        E::ZERO,
    )?;
    rows.push(bit_row);
    if slot != slot0 + forest_elems(v, d) || rows.len() != (31 + v + d + 1) as usize {
        return None;
    }

    let mut e = vec![E::ZERO; nc];
    let mut t_t = vec![E::ZERO; nc];
    for (c, call) in input.calls.iter_mut().enumerate() {
        if r_k < iota[c] {
            return None;
        }
        let hi = r_k - iota[c];
        let pre = first[c] >> iota[c];
        let mut weight = E::ONE;
        for i in 0..hi {
            weight *= if (pre >> (hi - 1 - i)) & 1 != 0 {
                zy[i as usize]
            } else {
                E::ONE - zy[i as usize]
            };
        }
        e[c] = weight;
        call.alpha = alpha;
        call.zk = zy[hi as usize..].to_vec();
        call.zk.extend_from_slice(&zx);
        if call.zk.len() != call.n as usize {
            return None;
        }
        let start = (k_t - call.k) as usize;
        let u = &zb0[start..start + call.k as usize];
        if let Some(tau) = &call.tau {
            let eq = crate::fp::tensor::eq_table_rev(u);
            for (i, &value) in tau.iter().enumerate() {
                t_t[c] += E::from_base(F(value)) * eq[i];
            }
        }
    }
    let ec = crate::fp::tensor::eq_table_rev(&zb0[k_t as usize..(k_t + c_k) as usize]);
    let mut row = E::ZERO;
    for c in 0..1usize << c_k {
        let of = c.min(nc - 1);
        row += ec[c] * (t_t[of] + theta * E::from_base(F(of as u64)));
    }
    let leaves_a = alpha - E::ONE - row;
    let (bytes, messages) = transcript.take_parts();
    let leaves = crate::fp::model::Psc {
        start: leaves.start,
        target: leaves.target,
        eps: leaves.eps,
        claim: leaves.claim,
        first: leaves.first,
        vars: leaves.vars,
        r: leaves.challenges,
        ended: leaves.ended,
        row_made: leaves.row_made,
        started: true,
    };
    Some(GeneratedForest {
        forest: crate::fp::forest::Forest {
            calls: input.calls,
            d_t,
            k_t,
            nt,
            r_k,
            c_k,
            iota,
            first,
            call_of,
            pub_counts: input.pub_counts,
            theta,
            alpha,
            a,
            rows,
            leaves,
            leaves_a,
            s_bit,
            zx,
            zy,
            e,
            zb: [zb0, zb1],
            t_t,
            c_end,
        },
        bytes,
        messages,
    })
}

pub struct ArithmeticProof {
    pub claim: crate::fp::psc::SumClaim,
    pub dot: Option<DotProductRoundProver>,
    pub residual: Option<ResidualRoundProver>,
    pub challenges: Vec<crate::fp::field::ExtensionElement>,
    pub bytes: Vec<u8>,
    pub compact_offsets: Vec<usize>,
}

struct ProtocolBytes<'a> {
    bytes: &'a [u8],
    compact_offsets: &'a [usize],
    offset: usize,
    chain: &'a mut crate::fp::chain::Chain,
    message: Vec<u64>,
    epsilon: Option<crate::fp::field::ExtensionElement>,
    challenges: Vec<crate::fp::field::ExtensionElement>,
}

impl crate::fp::psc::PscChannel for ProtocolBytes<'_> {
    fn recv(&mut self) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
        let compact = self
            .compact_offsets
            .iter()
            .any(|start| self.offset >= *start);
        let width = if compact { 8 } else { 24 };
        let end = self
            .offset
            .checked_add(width)
            .ok_or(crate::fp::error::FpError::Range {
                context: "psc-proof-offset",
            })?;
        if end > self.bytes.len() {
            return Err(crate::fp::error::FpError::Truncated {
                context: "psc-proof",
                needed: end,
                available: self.bytes.len(),
            });
        }
        let bytes = &self.bytes[self.offset..end];
        for word in bytes.chunks_exact(8) {
            if u64::from_le_bytes(word.try_into().unwrap()) >= crate::fp::field::GOLDILOCKS_P {
                return Err(crate::fp::error::FpError::NonCanonicalField);
            }
        }
        self.offset = end;
        if compact {
            let coordinate = u64::from_le_bytes(bytes.try_into().unwrap());
            self.message.extend_from_slice(&[coordinate, 0, 0]);
            Ok(crate::fp::field::ExtensionElement::from_words([
                coordinate, 0, 0,
            ]))
        } else {
            self.message
                .extend_from_slice(&crate::fp::transcript::words(bytes));
            Ok(crate::fp::transcript::elements(bytes)[0])
        }
    }

    fn close_epsilon(
        &mut self,
    ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
        self.chain.close(
            crate::fp::chain::ChainKind::Epsilon,
            &core::mem::take(&mut self.message),
        );
        let value = crate::fp::field::ExtensionElement::from_words(self.chain.draw_e());
        self.epsilon = Some(value);
        Ok(value)
    }

    fn close_algebraic(
        &mut self,
        nonzero: bool,
    ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
        let message = core::mem::take(&mut self.message);
        self.chain
            .close(crate::fp::chain::ChainKind::Algebraic, &message);
        let value = if nonzero {
            crate::fp::field::ExtensionElement::from_words(self.chain.draw_e_nonzero())
        } else {
            crate::fp::field::ExtensionElement::from_words(self.chain.draw_e())
        };
        self.challenges.push(value);
        Ok(value)
    }

    fn draw(
        &mut self,
        nonzero: bool,
    ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
        self.close_algebraic(nonzero)
    }

    fn close_empty(&mut self) -> Result<(), crate::fp::error::FpError> {
        self.chain.close(
            crate::fp::chain::ChainKind::Empty,
            &core::mem::take(&mut self.message),
        );
        Ok(())
    }
}

fn verify_arithmetic(chain: &mut crate::fp::chain::Chain, proof: &ArithmeticProof) -> bool {
    let mut previous = None;
    for &offset in &proof.compact_offsets {
        if offset == 0
            || offset % 8 != 0
            || offset >= proof.bytes.len()
            || previous.is_some_and(|p| offset <= p)
        {
            return false;
        }
        previous = Some(offset);
    }
    let mut channel = ProtocolBytes {
        bytes: &proof.bytes,
        compact_offsets: &proof.compact_offsets,
        offset: 0,
        chain: chain,
        message: vec![],
        epsilon: None,
        challenges: vec![],
    };
    let output = if let Some(summand) = &proof.dot {
        let fresh = (0..3 + 2 * proof.claim.vars.len() + 2 * summand.pairs.len())
            .map(|index| crate::fp::mask::MaskSlot {
                oracle: 0,
                index: index as u64,
            })
            .collect();
        crate::fp::psc::psc_verify(&mut channel, proof.claim.clone(), summand, fresh)
    } else if let Some(summand) = &proof.residual {
        let fresh = (0..2)
            .map(|index| crate::fp::mask::MaskSlot { oracle: 0, index })
            .collect();
        crate::fp::psc::psc_verify(&mut channel, proof.claim.clone(), summand, fresh)
    } else {
        let fresh = (0..6)
            .map(|index| crate::fp::mask::MaskSlot { oracle: 0, index })
            .collect();
        crate::fp::psc::psc_verify(
            &mut channel,
            proof.claim.clone(),
            &ProductRoundProver,
            fresh,
        )
    };
    let Ok(output) = output else {
        return false;
    };
    output.row_target.is_zero()
        && channel.offset == proof.bytes.len()
        && channel.epsilon == Some(proof.challenges[0])
        && channel.challenges == proof.challenges[1..]
}

pub struct BlockProof {
    pub statement: BlockStatement,
    pub arithmetic: Vec<ArithmeticProof>,
    pub forest: GeneratedForest,
    pub lookup_counts: [u64; 2],
}

fn model_forest_input(
    model: &DenseModel,
    lookup_counts: [u64; 2],
) -> Option<crate::fp::forest::ForestInput> {
    let mut silu = lookup_input("Ts", &encode_table(&model.tables.ts)?)?;
    let softmax = lookup_input("EX", &encode_table(&model.tables.ex)?)?;
    silu.lid = silu.lid.min(softmax.lid);
    silu.pub_counts = vec![0; 1usize << (silu.calls[0].n - silu.lid)];
    silu.pub_counts[0] = lookup_counts[0];
    let tree_capacity = (1u64 << softmax.lid) - 1;
    let mut remaining = lookup_counts[1];
    loop {
        let count = remaining.min(tree_capacity);
        let mut part = lookup_input("EX", &encode_table(&model.tables.ex)?)?;
        part.pub_counts[0] = count;
        silu.calls.extend(part.calls);
        silu.pub_counts.extend(part.pub_counts);
        remaining -= count;
        if remaining == 0 {
            break;
        }
    }
    Some(silu)
}

/// Generate a SiLU forest proof on the M1 statement chain.
pub fn prove_block_forest(
    model: &DenseModel,
    tokens: &[usize],
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<BlockProof> {
    #[cfg(feature = "native-cuda")]
    crate::fp::native_cuda::set_mapped(true);
    let (statement, trace) = bind_block(model, tokens, mf, kf, eps_f)?;
    let query = m1_first_query(&statement)?;
    let mut chain = crate::fp::chain::Chain::new(query);
    let mut transcript = ProverTr::new(&mut chain);
    transcript.set_compact(true);
    let product_values: Vec<(i128, i128)> = trace
        .gate_witnesses
        .iter()
        .chain(&trace.attention_witnesses)
        .chain(&trace.linear_witnesses)
        .chain(&trace.norm_witnesses)
        .copied()
        .collect();
    let mut arithmetic =
        Vec::with_capacity(product_values.len().div_ceil(64) + trace.residual_witnesses.len());
    for chunk in product_values.chunks(64) {
        let pairs: Vec<_> = chunk
            .iter()
            .map(|&(a, b)| {
                Some((
                    crate::fp::field::ExtensionElement::from_base(signed_field(a)?),
                    crate::fp::field::ExtensionElement::from_base(signed_field(b)?),
                ))
            })
            .collect::<Option<_>>()?;
        let (claim, summand, challenges) = prove_psc_dot_product(&mut transcript, pairs);
        arithmetic.push(ArithmeticProof {
            claim,
            dot: Some(summand),
            residual: None,
            challenges,
            bytes: transcript.take_parts().0,
            compact_offsets: transcript.take_compact_offsets(),
        });
    }
    if !trace.residual_witnesses.is_empty() {
        let pairs: Vec<_> = trace
            .residual_witnesses
            .iter()
            .map(|&(x, ad)| {
                Some((
                    crate::fp::field::ExtensionElement::from_base(signed_field(x)?),
                    crate::fp::field::ExtensionElement::from_base(signed_field(ad)?),
                ))
            })
            .collect::<Option<_>>()?;
        let (claim, summand, challenges) = prove_psc_residual(&mut transcript, pairs);
        arithmetic.push(ArithmeticProof {
            claim,
            dot: None,
            residual: Some(summand),
            challenges,
            bytes: transcript.take_parts().0,
            compact_offsets: transcript.take_compact_offsets(),
        });
    }
    transcript.set_compact(false);
    let lookup_counts = [
        trace
            .lookups
            .iter()
            .filter(|read| read.table == LookupTable::SiLU)
            .count() as u64,
        trace
            .lookups
            .iter()
            .filter(|read| read.table == LookupTable::Softmax)
            .count() as u64,
    ];
    let forest = prove_forest(&mut transcript, model_forest_input(model, lookup_counts)?)?;
    Some(BlockProof {
        statement,
        arithmetic,
        forest,
        lookup_counts,
    })
}

/// Recompute the block statement and verify its forest transcript with the
/// existing `forest_pad` verifier.
pub fn verify_block_forest(model: &DenseModel, tokens: &[usize], proof: &BlockProof) -> bool {
    #[cfg(feature = "native-cuda")]
    crate::fp::native_cuda::set_mapped(false);
    if proof.statement.format_version != 1 {
        return false;
    }
    if !verify_block(model, tokens, &proof.statement) {
        return false;
    }
    let Some(query) = m1_first_query(&proof.statement) else {
        return false;
    };
    let mut chain = crate::fp::chain::Chain::new(query);
    if proof
        .arithmetic
        .iter()
        .any(|arithmetic| !verify_arithmetic(&mut chain, arithmetic))
    {
        return false;
    }
    let Ok(layout) = crate::fp::layout::FileLayout::from_messages(&proof.forest.messages) else {
        return false;
    };
    let mut cursor = crate::fp::transcript::Cursor::new(
        &proof.forest.bytes,
        &layout,
        &proof.statement.parameters,
        &crate::fp::digest::Digest::ZERO,
    );
    let v = proof.forest.forest.d_t + proof.forest.forest.r_k;
    let d = proof.forest.forest.k_t + proof.forest.forest.c_k;
    let mut fork = crate::fp::model::HFork::default();
    fork.slots(crate::fp::forest::forest_elems(v, d));
    let mut tr = crate::fp::model::Tr::new(&mut cursor, &mut chain, "M1 forest");
    let mut slot = 0;
    let Some(input) = model_forest_input(model, proof.lookup_counts) else {
        return false;
    };
    let Ok(verified) = crate::fp::forest::forest_pad(&mut fork, &mut tr, &mut slot, input) else {
        return false;
    };
    let matches = slot == crate::fp::forest::forest_elems(v, d)
        && verified.rows.len() == proof.forest.forest.rows.len()
        && verified.rows.iter().all(|row| row.c.is_zero())
        && verified.alpha == proof.forest.forest.alpha
        && verified.a == proof.forest.forest.a
        && verified.leaves_a == proof.forest.forest.leaves_a
        && verified.t_t == proof.forest.forest.t_t
        && verified.zb == proof.forest.forest.zb;
    matches
}

/// Generate independent proofs for a batch of token requests. Each request has
/// its own statement and Fiat–Shamir chain.
pub fn prove_block_batch(
    model: &DenseModel,
    requests: &[&[usize]],
    mf: u32,
    kf: u32,
    eps_f: i128,
) -> Option<Vec<BlockProof>> {
    requests
        .iter()
        .map(|tokens| prove_block_forest(model, tokens, mf, kf, eps_f))
        .collect()
}

/// Verify a batch of independent token requests. Lengths must match exactly;
/// no statement or chain is shared between requests.
pub fn verify_block_batch(
    model: &DenseModel,
    requests: &[&[usize]],
    proofs: &[BlockProof],
) -> bool {
    requests.len() == proofs.len()
        && requests
            .iter()
            .zip(proofs)
            .all(|(tokens, proof)| verify_block_forest(model, tokens, proof))
}

pub fn small_model() -> DenseModel {
    fn identity(n: usize) -> Vec<Vec<i128>> {
        (0..n)
            .map(|i| (0..n).map(|j| i128::from(i == j)).collect())
            .collect()
    }
    let identity2 = identity(2);
    DenseModel {
        weights: ModelWeights {
            layers: vec![LayerWeights {
                wq: identity2.clone(),
                wk: identity2.clone(),
                wv: identity2.clone(),
                wo: identity2.clone(),
                wg: vec![vec![0, 0], vec![0, 0]],
                wu: vec![vec![1], vec![0]],
                wd: vec![vec![1, 0]],
            }],
            emb: vec![vec![1, 0], vec![0, 1], vec![1, 1]],
            wh: vec![vec![1, 0, 0], vec![0, 1, 0]],
        },
        scales: vec![LayerScales {
            kq: 1,
            kk: 1,
            kv: 1,
            ko: 1,
            kg: 1,
            ku: 1,
            kd: 1,
            m1: 4,
            k1: 4,
            m2: 4,
            k2: 4,
            sh: 1,
            rp: false,
            eps1: 1,
            eps2: 1,
        }],
        tables: PublicTables {
            cos: vec![1, 0],
            sin: vec![0, 1],
            ex: vec![1, 1],
            ts: vec![1; 65536],
            sig: vec![1; 65536],
            u0: 2,
        },
        t: 2,
        dh: 2,
        h: 1,
        g: 1,
        f: 1,
        v: 3,
        cs: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(n: usize) -> Vec<Vec<i128>> {
        (0..n)
            .map(|i| (0..n).map(|j| i128::from(i == j)).collect())
            .collect()
    }

    #[test]
    fn dense_forward_matches_independent_fixture() {
        let model = small_model();
        assert_eq!(forward(&model, &[0, 1], 4, 4, 1), Some(0));
    }

    #[test]
    fn forward_rejects_out_of_range_gate_high_part() {
        let mut model = small_model();
        model.weights.layers[0].wg = vec![vec![16_777_216, 0], vec![0, 0]];
        assert_eq!(forward(&model, &[0, 1], 4, 4, 1), None);
    }

    #[test]
    fn forward_rejects_empty_and_invalid_inputs() {
        let model = small_model();
        assert_eq!(forward(&model, &[], 4, 4, 1), None);
        assert_eq!(forward(&model, &[0, 1, 2], 4, 4, 1), None);
        assert_eq!(forward(&model, &[3], 4, 4, 1), None);
    }

    #[test]
    fn rq_rounds_ties_up_and_negative_values_toward_minus_infinity() {
        assert_eq!(rq(3, 1), Some(2));
        assert_eq!(rq(1, 1), Some(1));
        assert_eq!(rq(-1, 1), Some(0));
        assert_eq!(rq(-2, 1), Some(-1));
        assert_eq!(rq(-3, 1), Some(-1));
        assert_eq!(rq(1, 0), None);
    }

    #[test]
    fn tables_match_python_reference_values() {
        let tables = make_tables(4, 128, 12, 10000.0, 12, 8, 8, 4095);
        assert_eq!(tables.cos[0], 4096);
        assert_eq!(tables.sin[0], 0);
        assert_eq!(tables.cos[64], 2213);
        assert_eq!(tables.sin[64], 3447);
        assert_eq!(&tables.ex[..5], &[4096, 4080, 4064, 4048, 4032]);
        assert_eq!(tables.ts(-32768), Some(0));
        assert_eq!(tables.ts(-100), Some(-10));
        assert_eq!(tables.ts(100), Some(15));
        assert_eq!(tables.ts(32767), Some(4095));
        assert_eq!(tables.sig(-16384), Some(0));
        assert_eq!(tables.sig(-100), Some(26448));
        assert_eq!(tables.sig(0), Some(32768));
        assert_eq!(tables.sig(100), Some(39088));
        assert_eq!(tables.sig(16383), Some(65536));
        assert!(tables.u0 > 0);
        assert_eq!(tables.ex.len(), tables.u0);
    }

    #[test]
    fn upstream_table_digests_match_policy() {
        let tables = make_tables(64, 128, 12, 10000.0, 12, 8, 8, 4095);
        let digests = serialize_public_tables(&tables).unwrap().digests;
        assert_eq!(
            digests.tsilu.0,
            [
                2048398297, 192427613, 2862900154, 2853881485, 2797826050, 1420723544, 3686369553,
                2593882461
            ]
        );
        assert_eq!(
            digests.tsig.0,
            [
                1965976300, 2359599336, 2246761285, 2426172242, 3869868832, 310049175, 339407438,
                2601410768
            ]
        );
        assert_eq!(
            digests.trope.0,
            [
                2232934348, 2156588140, 3599599331, 498649753, 3675316650, 2999201840, 1537898906,
                2900737701
            ]
        );
        assert_eq!(
            digests.texp.0,
            [
                3648852807, 2402568834, 4186852988, 523559909, 1873895626, 3042864545, 1726265496,
                165292856
            ]
        );
    }

    #[test]
    fn rotary_projection_matches_reference_identities() {
        let tables = PublicTables {
            cos: vec![4096, 0],
            sin: vec![0, 4096],
            ex: vec![4096],
            ts: vec![0; 65536],
            sig: vec![0; 65536],
            u0: 1,
        };
        let identity = identity(2);
        let mut witnesses = Vec::new();
        assert_eq!(
            rotated_shifted(
                &[1, 2],
                &identity,
                12,
                12,
                true,
                0,
                2,
                &tables,
                &mut witnesses
            ),
            Some(vec![1, 2])
        );
        assert_eq!(
            rotated_shifted(
                &[1, 2],
                &identity,
                12,
                12,
                true,
                1,
                2,
                &tables,
                &mut witnesses
            ),
            Some(vec![-2, 1])
        );
        assert_eq!(
            rotated_shifted(
                &[1, 2],
                &identity,
                12,
                12,
                false,
                1,
                2,
                &tables,
                &mut witnesses
            ),
            Some(vec![1, 2])
        );
    }

    #[test]
    fn table_layout_and_digests_bind_serialized_values() {
        let tables = make_tables(4, 128, 12, 10000.0, 12, 8, 8, 4095);
        let serialized = serialize_public_tables(&tables).unwrap();
        let mut expected_offset = 0u64;
        for (region, values) in serialized.layout.regions.iter().zip([
            &tables.cos,
            &tables.sin,
            &tables.ex,
            &tables.ts,
            &tables.sig,
        ]) {
            assert_eq!(region.offset, expected_offset);
            assert_eq!(region.length, 8 + values.len() as u64 * 8);
            let start = region.offset as usize;
            let count = u64::from_le_bytes(serialized.bytes[start..start + 8].try_into().unwrap());
            assert_eq!(count, values.len() as u64);
            expected_offset += region.length;
        }
        assert_eq!(expected_offset, serialized.layout.total_bytes);
        assert_eq!(expected_offset as usize, serialized.bytes.len());

        let mut changed = tables.clone();
        changed.ts[100] += 1;
        let changed_serialized = serialize_public_tables(&changed).unwrap();
        assert_ne!(serialized.digests.tsilu, changed_serialized.digests.tsilu);
        assert_eq!(serialized.digests.trope, changed_serialized.digests.trope);
    }

    #[test]
    fn field_limbs_preserve_signed_values_and_boundaries() {
        fn decode(value: crate::fp::field::FieldElement) -> i128 {
            if value.0 > crate::fp::field::GOLDILOCKS_P / 2 {
                value.0 as i128 - crate::fp::field::GOLDILOCKS_P as i128
            } else {
                value.0 as i128
            }
        }
        for value in [-32768, -4096, -1, 0, 1, 4095, 32639] {
            let limbs = balanced_field_limbs(value, 2).unwrap();
            let reconstructed = decode(limbs[0]) + 256 * decode(limbs[1]);
            assert_eq!(reconstructed, value);
        }
        assert!(balanced_field_limbs(32640, 2).is_none());
        assert!(balanced_field_limbs(32768, 2).is_none());
        assert!(balanced_field_limbs(-32897, 2).is_none());
        assert!(balanced_field_limbs(547599908735, 5).is_some());
        assert!(balanced_field_limbs(547599908736, 5).is_none());
        assert!(balanced_field_limbs(-551911719040, 5).is_some());
        assert!(balanced_field_limbs(-551911719041, 5).is_none());
        assert_eq!(encode_weight_matrix(&[vec![-4096, 4095]]).unwrap().len(), 1);
        assert_eq!(
            encode_residual_matrix(&[vec![-1, 1]]).unwrap()[0][0].len(),
            5
        );
    }

    #[test]
    fn cpu_rows_encode_embeddings_and_residuals_with_distinct_widths() {
        let model = small_model();
        let trace = forward_with_trace(&model, &[0, 1], 4, 4, 1).unwrap();
        let mut wide_embedding = model.clone();
        wide_embedding.weights.emb[0][0] = 300000;
        let rows = cpu_prover_rows(&wide_embedding, &trace).unwrap();
        assert_eq!(rows.weights.len(), 84);
        assert_eq!(rows.residual.len(), 20);
        assert_eq!(rows.lookup_ts.len(), 65536);
        assert_eq!(rows.lookup_sig.len(), 65536);
        assert_eq!(rows.weights[0], signed_field(low8(300000)).unwrap());
    }

    #[test]
    fn protocol_inputs_use_native_tensors_and_lookup_forests() {
        use crate::fp::field::ExtensionElement as E;
        let model = small_model();
        let trace = forward_with_trace(&model, &[0, 1], 4, 4, 1).unwrap();
        let rows = cpu_prover_rows(&model, &trace).unwrap();
        let protocol = protocol_inputs(&model, &trace).unwrap();
        assert_eq!(protocol.weights.at(0), E::from_base(rows.weights[0]));
        assert_eq!(protocol.weights.at(83), E::from_base(rows.weights[83]));
        assert_eq!(protocol.weights.at(84), E::ZERO);
        assert_eq!(protocol.residual.at(19), E::from_base(rows.residual[19]));
        assert_eq!(protocol.lookups.len(), 5);
        assert_eq!(
            protocol.lookups[0].calls[0].tau.as_ref().unwrap().len(),
            1024
        );
        assert_eq!(
            protocol.lookups[3].calls[0].tau.as_ref().unwrap().len(),
            65536
        );
        assert_eq!(protocol.lookups[3].calls[0].tau_at(0).0[0], 1);
        let evaluated = crate::fp::lkp::lkp_row(
            &protocol.lookup_arguments[3],
            E::from_base(crate::fp::field::FieldElement(7)),
            0,
        );
        assert_eq!(evaluated, E::from_base(crate::fp::field::FieldElement(1)));
    }

    struct ProductSummand;
    impl crate::fp::psc::Summand for ProductSummand {
        fn shown(&self) -> usize {
            2
        }
        fn degree(&self) -> usize {
            2
        }
        fn at(
            &self,
            _point: &[crate::fp::field::ExtensionElement],
            end_values: &[crate::fp::field::ExtensionElement],
        ) -> Vec<crate::fp::field::ExtensionElement> {
            vec![end_values[0] * end_values[1]]
        }
    }

    struct LinearRoundProver {
        target: crate::fp::field::ExtensionElement,
    }

    impl crate::fp::psc::Summand for LinearRoundProver {
        fn shown(&self) -> usize {
            0
        }
        fn degree(&self) -> usize {
            1
        }
        fn at(
            &self,
            point: &[crate::fp::field::ExtensionElement],
            _end_values: &[crate::fp::field::ExtensionElement],
        ) -> Vec<crate::fp::field::ExtensionElement> {
            vec![self.target * point[0]]
        }
    }

    impl PscRoundProver for LinearRoundProver {
        fn round_polynomial(
            &mut self,
            _round: usize,
            _weight: &crate::fp::psc::Weight,
            running: crate::fp::field::ExtensionElement,
            _end_values: &[crate::fp::field::ExtensionElement],
        ) -> Vec<crate::fp::field::ExtensionElement> {
            vec![crate::fp::field::ExtensionElement::ZERO, running]
        }
    }

    struct PscBytes<'a> {
        bytes: &'a [u8],
        offset: usize,
        chain: &'a mut crate::fp::chain::Chain,
        message: Vec<u64>,
        algebraic_rounds: usize,
        epsilon: Option<crate::fp::field::ExtensionElement>,
        challenges: Vec<crate::fp::field::ExtensionElement>,
    }

    impl crate::fp::psc::PscChannel for PscBytes<'_> {
        fn recv(
            &mut self,
        ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
            let end = self.offset + 24;
            let bytes = &self.bytes[self.offset..end];
            self.offset = end;
            self.message
                .extend_from_slice(&crate::fp::transcript::words(bytes));
            Ok(crate::fp::transcript::elements(bytes)[0])
        }

        fn close_epsilon(
            &mut self,
        ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
            self.chain.close(
                crate::fp::chain::ChainKind::Epsilon,
                &core::mem::take(&mut self.message),
            );
            let value = crate::fp::field::ExtensionElement::from_words(self.chain.draw_e());
            self.epsilon = Some(value);
            Ok(value)
        }

        fn close_algebraic(
            &mut self,
            nonzero: bool,
        ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
            self.chain.close(
                crate::fp::chain::ChainKind::Algebraic,
                &core::mem::take(&mut self.message),
            );
            self.algebraic_rounds += 1;
            let value = if nonzero {
                crate::fp::field::ExtensionElement::from_words(self.chain.draw_e_nonzero())
            } else {
                crate::fp::field::ExtensionElement::from_words(self.chain.draw_e())
            };
            self.challenges.push(value);
            Ok(value)
        }

        fn draw(
            &mut self,
            nonzero: bool,
        ) -> Result<crate::fp::field::ExtensionElement, crate::fp::error::FpError> {
            self.close_algebraic(nonzero)
        }

        fn close_empty(&mut self) -> Result<(), crate::fp::error::FpError> {
            self.chain.close(
                crate::fp::chain::ChainKind::Empty,
                &core::mem::take(&mut self.message),
            );
            Ok(())
        }
    }

    #[test]
    fn generated_nonconstant_psc_proof_verifies() {
        use crate::fp::{
            chain::Chain,
            field::ExtensionElement as E,
            mask::MaskSlot,
            psc::{psc_verify, SumClaim, Weight},
        };
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        let target = E([7, 11, 13]);
        let claim = SumClaim {
            vars: vec![Weight::equality(E::ONE, 2)],
            masks: vec![],
            target,
        };
        let mut summand = LinearRoundProver { target };
        let mut proving_chain = Chain::new(query.clone());
        let mut prover = ProverTr::new(&mut proving_chain);
        let challenges = prove_psc(&mut prover, &claim, &mut summand, &[]).unwrap();
        let bytes = prover.into_bytes();

        let mut verifier_chain = Chain::new(query);
        let mut channel = PscBytes {
            bytes: &bytes,
            offset: 0,
            chain: &mut verifier_chain,
            message: vec![],
            algebraic_rounds: 0,
            epsilon: None,
            challenges: vec![],
        };
        let fresh = (0..2).map(|index| MaskSlot { oracle: 0, index }).collect();
        let output = psc_verify(&mut channel, claim, &summand, fresh).unwrap();
        assert_eq!(output.row_target, E::ZERO);
        assert_eq!(output.point, vec![challenges[1]]);
        assert_eq!(channel.offset, bytes.len());
    }

    #[test]
    fn generated_zero_forest_verifies_with_forest_pad() {
        use crate::fp::{
            chain::Chain,
            digest::Digest,
            field::FieldElement as F,
            forest::{forest_elems, forest_pad},
            layout::FileLayout,
            model::{HFork, Tr},
            transcript::Cursor,
        };
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        let input = lookup_input("zero", &vec![F::ZERO; 1024]).unwrap();
        let mut proving_chain = Chain::new(query.clone());
        let mut prover = ProverTr::new(&mut proving_chain);
        let generated = prove_forest(&mut prover, input).unwrap();
        let v = generated.forest.d_t + generated.forest.r_k;
        let d = generated.forest.k_t + generated.forest.c_k;
        assert_eq!(generated.forest.rows.len(), (31 + v + d + 1) as usize);

        let layout = FileLayout::from_messages(&generated.messages).unwrap();
        let mut cursor = Cursor::new(
            &generated.bytes,
            &layout,
            &statement.parameters,
            &Digest::ZERO,
        );
        let mut verifier_chain = Chain::new(query);
        let mut fork = HFork::default();
        fork.slots(forest_elems(v, d));
        let mut tr = Tr::new(&mut cursor, &mut verifier_chain, "M1 forest");
        let mut slot = 0;
        let verifier_input = lookup_input("zero", &vec![F::ZERO; 1024]).unwrap();
        let verified = forest_pad(&mut fork, &mut tr, &mut slot, verifier_input).unwrap();
        assert_eq!(slot, forest_elems(v, d));
        assert_eq!(verified.rows.len(), generated.forest.rows.len());
        assert!(verified.rows.iter().all(|row| row.c.is_zero()));
        assert_eq!(verified.alpha, generated.forest.alpha);
        assert_eq!(verified.a, generated.forest.a);
        assert_eq!(verified.leaves_a, generated.forest.leaves_a);
        assert_eq!(verified.zb, generated.forest.zb);
    }

    #[test]
    fn multi_layer_and_request_batch_verify_independently() {
        let mut model = small_model();
        model.weights.layers.push(model.weights.layers[0].clone());
        model.scales.push(model.scales[0].clone());
        let tokens_a = [0, 1];
        let tokens_b = [1, 0];
        let (_, trace) = bind_block(&model, &tokens_a, 4, 4, 1).unwrap();
        assert_eq!(trace.residual.len(), 2);

        let proofs = prove_block_batch(&model, &[&tokens_a, &tokens_b], 4, 4, 1).unwrap();
        let proof_a = &proofs[0];
        let proof_b = &proofs[1];
        assert_eq!(proof_a.lookup_counts, [4, 12]);
        assert_eq!(proof_a.forest.forest.rows.len(), 66);
        assert!(verify_block_batch(&model, &[&tokens_a, &tokens_b], &proofs));
        assert_ne!(proof_a.statement.tokens, proof_b.statement.tokens);
        assert!(!verify_block_forest(&model, &tokens_b, proof_a));
    }

    #[test]
    fn generated_silu_block_forest_verifies_and_binds_statement() {
        let model = small_model();
        let tokens = [0, 1];
        let mut proof = prove_block_forest(&model, &tokens, 4, 4, 1).unwrap();
        assert_eq!(proof.lookup_counts, [2, 6]);
        assert_eq!(proof.forest.forest.t_t[0].0[0], 1);
        assert_eq!(proof.forest.forest.rows.len(), 66);
        assert!(proof
            .arithmetic
            .iter()
            .filter_map(|proof| proof.dot.as_ref())
            .any(|dot| dot.variables > 0));
        assert!(proof
            .arithmetic
            .iter()
            .filter_map(|proof| proof.dot.as_ref())
            .any(|dot| dot.pairs.len() > dot.witness_pairs));
        let (_, trace) = bind_block(&model, &tokens, 4, 4, 1).unwrap();
        let covered: usize = proof
            .arithmetic
            .iter()
            .map(|proof| {
                proof.dot.as_ref().map_or_else(
                    || {
                        proof
                            .residual
                            .as_ref()
                            .map_or(1, |residual| residual.pairs.len())
                    },
                    |dot| dot.witness_pairs,
                )
            })
            .sum();
        assert_eq!(
            covered,
            trace.gate_witnesses.len()
                + trace.attention_witnesses.len()
                + trace.linear_witnesses.len()
                + trace.norm_witnesses.len()
                + trace.residual_witnesses.len()
        );
        assert!(!proof.arithmetic.is_empty());
        assert!(verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[0].bytes[24] ^= 1;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[0].bytes[24] ^= 1;
        assert!(verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[1].bytes[24] ^= 1;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[1].bytes[24] ^= 1;
        assert!(verify_block_forest(&model, &tokens, &proof));
        let last = proof.arithmetic.len() - 1;
        proof.arithmetic[last].bytes[24] ^= 1;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[last].bytes[24] ^= 1;
        assert!(verify_block_forest(&model, &tokens, &proof));
        let expected_execution = proof.statement.execution;
        let mut altered_key = trace.clone();
        altered_key.lookups[0].key += 1;
        proof.statement.execution = execution_commitment(&altered_key).unwrap();
        assert!(!verify_block_forest(&model, &tokens, &proof));
        let mut altered_value = trace.clone();
        altered_value.lookups[0].value += 1;
        proof.statement.execution = execution_commitment(&altered_value).unwrap();
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.statement.execution = expected_execution;
        assert!(verify_block_forest(&model, &tokens, &proof));
        proof.statement.format_version = 0;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.statement.format_version = 1;
        assert!(verify_block_forest(&model, &tokens, &proof));
        let original_offset = proof.arithmetic[0].compact_offsets[0];
        proof.arithmetic[0].compact_offsets[0] = usize::MAX;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[0].compact_offsets[0] = original_offset;
        assert!(verify_block_forest(&model, &tokens, &proof));
        let good_bytes = core::mem::take(&mut proof.arithmetic[0].bytes);
        let mut truncated = good_bytes.clone();
        truncated.truncate(truncated.len() - 1);
        proof.arithmetic[0].bytes = truncated;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[0].bytes = good_bytes.clone();
        assert!(verify_block_forest(&model, &tokens, &proof));
        let mut noncanonical = good_bytes.clone();
        noncanonical[96..104].copy_from_slice(&crate::fp::field::GOLDILOCKS_P.to_le_bytes());
        proof.arithmetic[0].bytes = noncanonical;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.arithmetic[0].bytes = good_bytes;
        assert!(verify_block_forest(&model, &tokens, &proof));
        proof.statement.output += 1;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.statement.output -= 1;
        proof.lookup_counts[0] += 1;
        assert!(!verify_block_forest(&model, &tokens, &proof));
        proof.lookup_counts[0] -= 1;
        proof.statement.execution = crate::fp::digest::Digest::ZERO;
        assert!(!verify_block_forest(&model, &tokens, &proof));
    }

    #[test]
    fn product_levels_and_terminal_rows_hand_down_zero_witnesses() {
        use crate::fp::{
            chain::Chain,
            field::ExtensionElement as E,
            forest::hand_down,
            model::{Aff, HFork},
        };
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let mut chain = Chain::new(m1_first_query(&statement).unwrap());
        let mut transcript = ProverTr::new(&mut chain);
        let mut fork = HFork::default();
        fork.slots(1024);
        let mut slot = 0u32;
        let mut start = Aff::default();
        let mut rho = vec![];
        let level0 = prove_product_level(
            &mut fork,
            &mut transcript,
            &mut slot,
            0,
            &rho,
            &mut start,
            &[],
            &[],
            &[E::ZERO; 3],
            [E::ZERO, E::ZERO],
        )
        .unwrap();
        assert!(level0.row.c.is_zero());
        hand_down(&level0, &mut rho, &mut start);
        assert_eq!(rho, vec![level0.tau]);

        let level1 = prove_product_level(
            &mut fork,
            &mut transcript,
            &mut slot,
            1,
            &rho,
            &mut start,
            &[0],
            &[vec![E::ZERO; 3]],
            &[E::ZERO; 3],
            [E::ZERO, E::ZERO],
        )
        .unwrap();
        assert!(level1.row.c.is_zero());
        hand_down(&level1, &mut rho, &mut start);
        assert_eq!(rho.len(), 2);

        let (leaves, leaf_points) = prove_forest_leaves(
            &mut fork,
            &mut transcript,
            &mut slot,
            start,
            &[E::ONE, E::ONE],
            &[2, 1],
            &[vec![E::ZERO; 3], vec![E::ZERO; 2]],
        )
        .unwrap();
        assert_eq!(leaf_points.len(), 2);
        assert!(!leaves.ended);

        let (bit_row, s_bit, bit_points) = prove_forest_bit_row(
            &mut fork,
            &mut transcript,
            &mut slot,
            &leaf_points,
            &[vec![E::ZERO; 3], vec![E::ZERO; 3]],
            &[E::ZERO; 3],
            E::ZERO,
        )
        .unwrap();
        assert!(bit_row.c.is_zero());
        assert_eq!(fork.shown(s_bit).unwrap(), E::ZERO);
        assert_eq!(bit_points.len(), 2);
    }

    #[test]
    fn forest_initial_psc_witnesses_build_zero_rows() {
        use crate::fp::{
            chain::Chain,
            field::ExtensionElement as E,
            model::{Aff, HFork},
        };
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let mut chain = Chain::new(m1_first_query(&statement).unwrap());
        let mut transcript = ProverTr::new(&mut chain);
        let mut fork = HFork::default();
        fork.slots(1024);
        let mut slot = 0u32;
        let mut rows = vec![];
        let mut s_z = 0;
        let mut s_p = 0;
        let mut s_g = [0u32; 32];
        for j in 0..31 {
            let start = if j == 0 {
                Aff::default()
            } else {
                Aff::slot(s_p)
            };
            let mut proof =
                psc_prove_begin_slots(&fork, &mut transcript, &mut slot, start).unwrap();
            let at = psc_prove_ends_slots(
                &mut fork,
                &mut transcript,
                &mut slot,
                &mut proof,
                3,
                &[E::ZERO; 4],
                &vec![E::ZERO; if j == 0 { 3 } else { 2 }],
            )
            .unwrap();
            if j == 0 {
                s_z = at;
            }
            s_p = if j == 0 { at + 1 } else { at };
            s_g[j] = s_p + 1;
            let power =
                fork.shown(s_p).unwrap() * fork.shown(s_p).unwrap() * fork.shown(s_g[j]).unwrap();
            let s0 = if j == 0 {
                fork.shown(s_z).unwrap() - power
            } else {
                power
            };
            rows.push(psc_prove_row(&mut fork, &mut proof, s0, &Aff::default()).unwrap());
        }
        assert_eq!(rows.len(), 31);
        assert!(rows.iter().all(|row| row.c.is_zero()));
        assert!(slot > 100);
    }

    #[test]
    fn forest_shaped_psc_composition_matches_verified_product_transcript() {
        use crate::fp::{chain::Chain, field::ExtensionElement as E, psc::Weight};
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        let a = E([7, 11, 13]);
        let b = E([17, 19, 23]);
        let z = E([2, 3, 5]);

        let mut reference_chain = Chain::new(query.clone());
        let mut reference = ProverTr::new(&mut reference_chain);
        prove_psc_product(&mut reference, z, a, b);
        let expected = reference.into_bytes();

        let mut composed_chain = Chain::new(query);
        let mut composed = ProverTr::new(&mut composed_chain);
        let mut proof = psc_prove_begin(&mut composed, a * b);
        let first_claim = proof.claim;
        psc_prove_round(
            &mut composed,
            &mut proof,
            Weight::equality(z, 2),
            &[first_claim, E::ZERO],
            false,
        )
        .unwrap();
        let final_claim = proof.claim;
        psc_prove_ends(
            &mut composed,
            &mut proof,
            2,
            &[final_claim, E::ZERO, E::ZERO],
            &[a, b],
        )
        .unwrap();
        assert_eq!(composed.into_bytes(), expected);
        assert!(proof.ended);
    }

    #[test]
    fn generated_psc_product_proof_verifies() {
        use crate::fp::{
            chain::Chain, field::ExtensionElement as E, mask::MaskSlot, psc::psc_verify,
        };
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        let mut proving_chain = Chain::new(query.clone());
        let mut prover = ProverTr::new(&mut proving_chain);
        let (claim, prover_challenges) =
            prove_psc_product(&mut prover, E([2, 3, 5]), E([7, 11, 13]), E([17, 19, 23]));
        let bytes = prover.into_bytes();

        let mut verifier_chain = Chain::new(query);
        let mut channel = PscBytes {
            bytes: &bytes,
            offset: 0,
            chain: &mut verifier_chain,
            message: vec![],
            algebraic_rounds: 0,
            epsilon: None,
            challenges: vec![],
        };
        let fresh = (0..6).map(|index| MaskSlot { oracle: 0, index }).collect();
        let output = psc_verify(&mut channel, claim.clone(), &ProductSummand, fresh).unwrap();
        assert_eq!(channel.epsilon, Some(prover_challenges[0]));
        assert_eq!(channel.challenges, prover_challenges[1..]);
        assert_eq!(output.row_target, E::ZERO);
        assert_eq!(output.ends[0].value, E([7, 11, 13]));
        assert_eq!(output.ends[1].value, E([17, 19, 23]));
        assert_eq!(channel.algebraic_rounds, 2);
        assert_eq!(channel.offset, bytes.len());

        let mut altered = bytes.clone();
        altered[24] ^= 1;
        let mut altered_chain = Chain::new(m1_first_query(&statement).unwrap());
        let mut altered_channel = PscBytes {
            bytes: &altered,
            offset: 0,
            chain: &mut altered_chain,
            message: vec![],
            algebraic_rounds: 0,
            epsilon: None,
            challenges: vec![],
        };
        let fresh = (0..6).map(|index| MaskSlot { oracle: 0, index }).collect();
        let rejected = psc_verify(&mut altered_channel, claim, &ProductSummand, fresh)
            .map_or(true, |output| output.row_target != E::ZERO);
        assert!(rejected);
    }

    #[test]
    fn prover_transcript_matches_verifier_chain_schedule() {
        use crate::fp::{chain::Chain, field::ExtensionElement as E, model::RoundName};
        let model = small_model();
        let (statement, _) = bind_block(&model, &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        let mut proving_chain = Chain::new(query.clone());
        let mut prover = ProverTr::new(&mut proving_chain);
        let values = [E::ONE, E::from_words([2, 3, 4])];
        prover.write(&values);
        let prover_challenge = prover.answer(RoundName::Lkp);
        let bytes = prover.into_bytes();
        assert_eq!(crate::fp::transcript::elements(&bytes), values);

        let mut verifier_chain = Chain::new(query);
        verifier_chain.close(
            crate::fp::chain::ChainKind::Algebraic,
            &crate::fp::transcript::words(&bytes),
        );
        assert_eq!(E::from_words(verifier_chain.draw_e()), prover_challenge);

        let mut changed = statement.clone();
        changed.parameters = crate::fp::digest::Digest::ZERO;
        assert_ne!(
            m1_first_query(&statement).unwrap().statement,
            m1_first_query(&changed).unwrap().statement
        );
    }

    #[test]
    fn block_statement_binds_weights_trace_and_output() {
        let model = small_model();
        let tokens = [0, 1];
        let (statement, trace) = bind_block(&model, &tokens, 4, 4, 1).unwrap();
        assert_eq!(trace.residual.len(), 1);
        assert_eq!(statement.output, 0);
        assert!(verify_block(&model, &tokens, &statement));

        let mut changed_output = statement.clone();
        changed_output.output = 1;
        assert!(!verify_block(&model, &tokens, &changed_output));

        let mut changed_protocol = statement.clone();
        changed_protocol.protocol = crate::fp::digest::Digest::ZERO;
        assert!(!verify_block(&model, &tokens, &changed_protocol));

        let mut changed_parameters = statement.clone();
        changed_parameters.parameters = crate::fp::digest::Digest::ZERO;
        assert!(!verify_block(&model, &tokens, &changed_parameters));

        let mut changed_model = model.clone();
        changed_model.weights.emb[0][0] += 1;
        assert!(!verify_block(&changed_model, &tokens, &statement));
        assert_ne!(
            weight_commitment(&model.weights).unwrap(),
            weight_commitment(&changed_model.weights).unwrap()
        );
    }

    #[test]
    fn byte_and_clamp_boundaries_are_exact() {
        assert_eq!(low8(-129), 127);
        assert_eq!(low8(-128), -128);
        assert_eq!(low8(127), 127);
        assert_eq!(low8(128), -128);
        assert_eq!(low8(256), 0);
        assert_eq!(clamp(-3000, 2047), -2047);
        assert_eq!(clamp(3000, 2047), 2047);
    }

    #[test]
    fn argmax_returns_first_maximum() {
        assert_eq!(argmax(&[-2, 5, 5, 1]), Some(1));
        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn isqrt_handles_perfect_and_nonperfect_squares() {
        for (a, expected) in [
            (0, 0),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 2),
            (8, 2),
            (9, 3),
            (15, 3),
            (16, 4),
            (1i128 << 78, 1i128 << 39),
        ] {
            assert_eq!(isqrt(a), Some(expected), "isqrt({a})");
        }
        assert_eq!(isqrt(1254267904959), Some(1119941));
        assert_eq!(isqrt(-1), None);
    }

    #[test]
    fn norm_matches_reference_formula() {
        let got = norm(&[1, -2, 3], 4, 4, 1).unwrap();
        assert_eq!(got, vec![0, -1, 1]);
        let clamped = norm(&[1], 47, 1, 1).unwrap();
        assert_eq!(clamped, vec![2047]);
        assert_eq!(norm(&[0, 0], 4, 4, 0), None);
    }
}
