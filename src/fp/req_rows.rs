//! Request subproof equations and laid-out linear claims (`req_rows.h`, `laid.h`).

use alloc::{string::String, vec::Vec};

use crate::fp::{
    claims::{Part, Row, Statement},
    error::FpError,
    field::ExtensionElement as E,
    request_public::{LaidBlock, ReqLayout, ReqShape},
    tensor::{Form, Tensor},
};

#[derive(Debug, Clone, Default)]
pub struct IndexWeights {
    pub lo: Vec<E>,
    pub hi: Vec<E>,
    pub values: Option<Vec<E>>,
}

impl IndexWeights {
    pub fn eq(point: &[E]) -> Self {
        Self {
            lo: point.iter().map(|&z| E::ONE - z).collect(),
            hi: point.to_vec(),
            values: None,
        }
    }

    pub fn powers(mut base: E, bits: u32) -> Self {
        let mut out = Self::default();
        for _ in 0..bits {
            out.lo.push(E::ONE);
            out.hi.push(base);
            base *= base;
        }
        out
    }

    pub fn written(values: Vec<E>) -> Self {
        Self {
            values: Some(values),
            ..Self::default()
        }
    }

    fn written_out(&self) -> bool {
        self.values.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct ReqTerm {
    pub born: usize,
    pub array: String,
    pub scale: E,
    pub on_rows: IndexWeights,
    pub on_columns: IndexWeights,
}

#[derive(Debug, Clone)]
pub struct ReqEquation {
    pub terms: Vec<ReqTerm>,
    pub on_masks: Vec<E>,
    pub target: E,
}

fn aligned(value: u64, bits: u32) -> bool {
    bits >= 64 || value & ((1u64 << bits) - 1) == 0
}

pub fn laid_block_ok(block: &LaidBlock, table_row_bits: u32, table_place_bits: u32) -> bool {
    block.row_bits + block.place_bits == block.bits_i + block.bits_c
        && aligned(block.first_i, block.bits_i)
        && aligned(block.first_c, block.bits_c)
        && aligned(block.first_row, block.row_bits)
        && aligned(block.first_place, block.place_bits)
        && block.first_row >> table_row_bits == 0
        && block.first_place >> table_place_bits == 0
}

fn meet(a: u64, a_bits: u32, b: u64, b_bits: u32) -> bool {
    a < b + (1u64 << b_bits) && b < a + (1u64 << a_bits)
}

/// Check full matrix cover and pairwise table-placement disjointness.
pub fn laid_ok(
    matrices: &[(&[LaidBlock], u64, u64)],
    table_row_bits: u32,
    table_place_bits: u32,
) -> bool {
    let mut all = Vec::new();
    for &(blocks, size_i, size_c) in matrices {
        let mut covered = 0u128;
        for (x, block) in blocks.iter().enumerate() {
            if !laid_block_ok(block, table_row_bits, table_place_bits)
                || block.first_i + (1u64 << block.bits_i) > size_i
                || block.first_c + (1u64 << block.bits_c) > size_c
            {
                return false;
            }
            for previous in &blocks[..x] {
                if meet(block.first_i, block.bits_i, previous.first_i, previous.bits_i)
                    && meet(block.first_c, block.bits_c, previous.first_c, previous.bits_c)
                {
                    return false;
                }
            }
            covered += 1u128 << (block.bits_i + block.bits_c);
            all.push(block);
        }
        if covered != size_i as u128 * size_c as u128 {
            return false;
        }
    }
    for (x, block) in all.iter().enumerate() {
        for previous in &all[..x] {
            if meet(block.first_row, block.row_bits, previous.first_row, previous.row_bits)
                && meet(block.first_place, block.place_bits, previous.first_place, previous.place_bits)
            {
                return false;
            }
        }
    }
    true
}

pub fn laid_tensor(
    block: &LaidBlock,
    table_row_bits: u32,
    table_place_bits: u32,
    on_i: &IndexWeights,
    on_c: &IndexWeights,
    scalar: E,
) -> Result<Tensor, FpError> {
    if !laid_block_ok(block, table_row_bits, table_place_bits)
        || on_i.written_out() && on_c.written_out()
    {
        return Err(FpError::Range { context: "laid-block" });
    }
    let mut tensor = Tensor::eq(&[], scalar);
    let low_bits = if block.i_is_low { block.bits_i } else { block.bits_c };
    let mut dense = Vec::new();
    let mut push_bit = |tensor: &mut Tensor, x_bit: u32| -> Result<(), FpError> {
        let of_i = (x_bit < low_bits) == block.i_is_low;
        let bit = if x_bit < low_bits { x_bit } else { x_bit - low_bits };
        let weights = if of_i { on_i } else { on_c };
        if weights.written_out() {
            dense.push((bit, tensor.vars()));
            tensor.push(E::ONE, E::ONE);
        } else {
            if bit as usize >= weights.lo.len() || bit as usize >= weights.hi.len() {
                return Err(FpError::Range { context: "laid-weights" });
            }
            tensor.push(weights.lo[bit as usize], weights.hi[bit as usize]);
        }
        Ok(())
    };
    for x in block.place_bits..block.place_bits + block.row_bits {
        push_bit(&mut tensor, x)?;
    }
    tensor.insert_choice(
        tensor.vars(),
        table_row_bits - block.row_bits,
        block.first_row >> block.row_bits,
    );
    for x in 0..block.place_bits {
        push_bit(&mut tensor, x)?;
    }
    tensor.insert_choice(
        tensor.vars(),
        table_place_bits - block.place_bits,
        block.first_place >> block.place_bits,
    );
    for (weights, first, bits) in [(on_i, block.first_i, block.bits_i), (on_c, block.first_c, block.bits_c)] {
        if weights.written_out() {
            continue;
        }
        if weights.lo.len() < 64 && first >> weights.lo.len() != 0 {
            return Err(FpError::Range { context: "laid-weights" });
        }
        for i in bits as usize..weights.lo.len() {
            tensor.scalar *= if (first >> i) & 1 != 0 {
                weights.hi[i]
            } else {
                weights.lo[i]
            };
        }
    }
    if let Some((values, first)) = if on_i.values.is_some() {
        on_i.values.as_ref().map(|values| (values, block.first_i))
    } else {
        on_c.values.as_ref().map(|values| (values, block.first_c))
    } {
        if !dense.is_empty() {
            tensor.set_dense_map(&dense);
            tensor.set_values(values.clone(), first);
        } else if tensor.has_dense() {
            tensor.set_values(values.clone(), first);
        } else {
            tensor.scalar *= *values.get(first as usize).ok_or(FpError::Range {
                context: "laid-values",
            })?;
        }
    }
    Ok(tensor)
}

pub fn laid_form(
    blocks: &[LaidBlock],
    table_row_bits: u32,
    table_place_bits: u32,
    on_i: &IndexWeights,
    on_c: &IndexWeights,
    scalar: E,
) -> Result<Form, FpError> {
    let mut form = Form::default();
    for block in blocks {
        let tensor = laid_tensor(
            block,
            table_row_bits,
            table_place_bits,
            on_i,
            on_c,
            scalar,
        )?;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            let vars = tensor.vars();
            let point = vec![E::ONE + E::ONE; vars as usize];
            let mut folded = tensor.clone();
            folded.fix_low(&point, vars);
            eprintln!("RUST_LAID_BLOCK first_i={} first_c={} value={:016x}", block.first_i, block.first_c, folded.at(0).0[0]);
        }
        form.add(tensor);
    }
    Ok(form)
}

pub fn req_row(
    statement: &mut Statement,
    shape: &ReqShape,
    layout: &ReqLayout,
    link: usize,
    equation: &ReqEquation,
    mut where_oracle: impl FnMut(usize, usize) -> Result<u32, FpError>,
) -> Result<Row, FpError> {
    let mut row = Row::default();
    for (term_index, term) in equation.terms.iter().enumerate() {
        let array = layout.find(term.born, &term.array).ok_or(FpError::Range {
            context: "request-array",
        })?;
        if term_index == 0 {
            #[cfg(feature = "std")]
            if std::env::var_os("DUMP_SHARED").is_some() {
                eprintln!("RUST_EQ_TERM blocks={} first={:?} on_i_lo={:?} on_i_hi={:?} on_c_values={:?} scalar={:016x}", array.blocks.len(), array.blocks.first().map(|b| (b.first_i, b.bits_i, b.first_c, b.bits_c, b.first_row, b.row_bits, b.first_place, b.place_bits, b.i_is_low)), term.on_rows.lo.first().map(|x| x.0[0]), term.on_rows.hi.first().map(|x| x.0[0]), term.on_columns.values.as_ref().and_then(|v| v.first().map(|x| x.0[0])), term.scale.0[0]);
            }
        }
        let oracle = where_oracle(term.born, array.message)?;
        let shape = &statement.oracles[oracle as usize].shape;
        let row_bits = crate::fp::tensor::ceil_log2(shape.main_rows);
        let place_bits = crate::fp::code_eval::log_ell(&statement.base[shape.code as usize]);
        let matrices: Vec<_> = layout.links[term.born]
            .iter()
            .filter(|candidate| candidate.message == array.message)
            .map(|candidate| (candidate.blocks.as_slice(), candidate.size_i, candidate.size_c))
            .collect();
        if !laid_ok(&matrices, row_bits, place_bits) {
            return Err(FpError::Range { context: "request-layout" });
        }
        let form = laid_form(
            &array.blocks,
            row_bits,
            place_bits,
            &term.on_rows,
            &term.on_columns,
            term.scale,
        )?;
        #[cfg(feature = "std")]
        if std::env::var_os("DUMP_SHARED").is_some() {
            let vars = form.tensors.iter().map(|tensor| tensor.vars()).max().unwrap_or(0);
            let point = vec![E::ONE + E::ONE; vars as usize];
            eprintln!("RUST_EQ_FORM oracle={} hash={:016x}", oracle, form.mle(&point).0[0]);
        }
        row.add_form(oracle, form);
    }
    if !equation.on_masks.is_empty() {
        let messages = &shape.links[link].messages;
        let message = messages
            .iter()
            .position(|message| message.shape.mask_rows != 0)
            .ok_or(FpError::Range {
                context: "request-mask-row",
            })?;
        if messages[message].shape.mask_rows != 1 {
            return Err(FpError::Range {
                context: "request-mask-row",
            });
        }
        let oracle = where_oracle(link, message)?;
        let taken = statement.masks(oracle).take_many(equation.on_masks.len());
        if taken.first().map(|slot| slot.index) != Some(0)
            || taken.last().map(|slot| slot.index + 1) != Some(taken.len() as u64)
        {
            return Err(FpError::Range {
                context: "request-mask-order",
            });
        }
        row.add_run(oracle, Part::Mask, 0, 1, equation.on_masks.clone());
    }
    row.target = equation.target;
    Ok(row)
}
