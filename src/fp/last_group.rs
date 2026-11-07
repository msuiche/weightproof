//! Final IOPP, member rows, shared chain and request-second chain.

use alloc::{vec, vec::Vec};

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{OracleShape, Statement},
    digest::Digest,
    error::FpError,
    field::ExtensionElement as E,
    key_public::{LinPublic, LIN_LABEL_SECOND, LIN_LABEL_SHARED},
    link_public::IndexName,
    lin::{lin_dot_row, member_first_verify, read_binding_root},
    lin_algebra::LinSharedAlgebra,
    output_head::HeadRun,
    proof_plan::ProofLayoutInput,
    request_frame::RequestFrame,
    request_public::RequestFixed,
    send::{OracleBook, read_samples_after_root},
    transcript::words,
    verify_key::{read_sample, run_iopp_stages, KeyChannel},
};

fn adopt(
    statement: &mut Statement,
    book: &mut OracleBook,
    from_statement: &Statement,
    from_book: &OracleBook,
    theirs: u32,
) -> Result<u32, FpError> {
    let shape = from_statement.oracles[theirs as usize].shape.clone();
    let location = from_book.location(theirs).ok_or(FpError::Range {
        context: "lin-adopt-tree",
    })?;
    let mine = statement.sent(shape);
    book.adopt(mine, location);
    statement.oracles[mine as usize].sample_rows = from_statement.oracles[theirs as usize].sample_rows.clone();
    Ok(mine)
}

/// Run the final IOPP, member-row verification, shared chain and request-second chain.
pub fn verify_last_group(
    frame: &mut RequestFrame,
    head: &mut HeadRun,
    pending: Vec<crate::fp::wf::TensorRow>,
    input: &ProofLayoutInput,
    fixed: &RequestFixed,
    configuration: &Digest,
    table: &IndexName,
    bytes: &[u8],
) -> Result<(), FpError> {
    let _first = frame.first()?;
    let fresh_override = frame.mask;
    let (layout, rows) = frame.iopp.statement.begin_run(frame.iopp.group);
    frame.chain.close(ChainKind::Algebraic, &[]);
    let row = crate::fp::claims::bat_combine(&frame.iopp.statement, &layout, &rows, E::from_words(frame.chain.draw_e()));
    let row = run_iopp_stages(
        frame.cursor,
        frame.chain,
        &mut frame.iopp,
        &input.last_stages,
        0,
        input.last_stages.len(),
        row,
        &frame.run,
        "last group",
        bytes,
        &input.setting,
        crate::fp::analysis::Scope::Request,
        None,
        fresh_override,
        true,
    )?;
    let mut pending = pending;
    pending.push(head.pending()?);
    let index_masks = frame.at_start[0];
    pending.last_mut().unwrap().masks.push(crate::fp::mask::MaskTerm {
        slot: frame.iopp.statement.masks(index_masks).take(),
        coeff: head.merged.as_ref().unwrap().beta,
    });
    let w = frame.iopp.statement.sent(fixed.lin.w.clone());
    let fold_mask = if fixed.lin.masks_ahead {
        Some(frame.iopp.statement.sent(OracleShape {
            code: 0,
            mask_code: fixed.lin.test_own[0].mask_code,
            main_rows: 0,
            mask_rows: 1u64 << fixed.lin.test_own[0].h,
            main_deg: 3,
            answers: 0,
        }))
    } else {
        None
    };
    let mut pending = pending;
    let first = member_first_verify(
        &mut frame.iopp.statement,
        &mut KeyChannel::new(frame.cursor, frame.chain, "rows on the table"),
        &fixed.lin,
        index_masks,
        &mut pending[..],
        w,
        fold_mask,
    )?;
    read_sample(frame.cursor, frame.chain, &mut frame.iopp.statement, w, "table, W")?;
    frame.chain.close(ChainKind::Seal, &[]);
    let mut seal_words = [0u32; 8];
    for i in 0..4 {
        let word = frame.chain.next_word();
        seal_words[2 * i] = word as u32;
        seal_words[2 * i + 1] = (word >> 32) as u32;
    }
    let seal = Digest(seal_words);
    let binding = read_binding_root(frame.cursor, &fixed.binding, configuration, &seal)?;

    let shared_public = LinPublic::shared_parts(
        &fixed.codes,
        &fixed.lin,
        &fixed.configuration,
        table,
        include_bytes!("../../spikes/plans/key-shared-plan.bin"),
        &binding,
    );
    let shared_query = shared_public.first_query(LIN_LABEL_SHARED);
    let shared_run = shared_query.run.unwrap();
    let mut shared_chain = Chain::new(shared_query);
    let mut shared = LinSharedAlgebra::new(&fixed.lin, &fixed.codes, configuration);
    shared_chain.close(ChainKind::Algebraic, &[]);
    let psi: Vec<E> = (0..fixed.lin.k_col())
        .map(|_| E::from_words(shared_chain.draw_e()))
        .collect();
    let _u_answer = read_sample(
        frame.cursor,
        &mut shared_chain,
        &mut shared.iopp.statement,
        shared.shared.u.oracle,
        "table, folded row",
    )?;
    let columns_nonce = frame.cursor.take_nonce("table, columns")?;
    if !shared_chain.close_with_work(ChainKind::Positions, &[], 32, columns_nonce) {
        return Err(FpError::Range {
            context: "proof-of-work-lin-columns",
        });
    }
    let columns: Vec<u64> = (0..input.lin.columns)
        .map(|_| shared_chain.draw_position(fixed.lin.fold.index.code.log_n))
        .collect();
    let first_shared_opening = frame.cursor.opened.len();
    frame.cursor.open_stage(&columns, &shared_run)?;
    shared.initial(
        table,
        &psi[fixed.lin.k_lo as usize..],
        &columns,
        &frame.cursor.opened[first_shared_opening],
        bytes,
    )?;
    let omega_root = frame.cursor.take_root("table, Omega")?;
    let mut omega_message = words(&omega_root.to_le_bytes());
    if shared.shared.split.is_some() {
        let split_root = frame.cursor.take_root("table, Omega")?;
        omega_message.extend(words(&split_root.to_le_bytes()));
    }
    shared_chain.close(ChainKind::Seed, &omega_message);
    let mut sampled = vec![shared.shared.omega];
    if let Some(split) = shared.shared.split {
        sampled.push(split);
    }
    let _omega_answers = read_samples_after_root(
        frame.cursor,
        &mut shared_chain,
        &mut shared.iopp.statement,
        &sampled,
        "table, Omega",
    )?;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let zeta: Vec<E> = (0..fixed.lin.k_row())
        .map(|_| E::from_words(shared_chain.draw_e()))
        .collect();
    let omega_lay = crate::fp::claims::VectorLay {
        row_bits: crate::fp::tensor::ceil_log2(fixed.lin.omega.main_rows),
        place_bits: crate::fp::code_eval::log_ell(&shared.iopp.statement.base[fixed.lin.omega.code as usize]),
        rows_high: fixed.lin.omega_rows_high,
    };
    let fold_row = crate::fp::lin::lin_fold_row(
        &fixed.lin,
        &psi,
        &zeta,
        shared.shared.u.clone(),
        shared.shared.omega,
        omega_lay.clone(),
    );
    if let Some(split) = shared.shared.split {
        crate::fp::split::split_verify(
            &mut shared.iopp.statement,
            frame.cursor,
            &mut shared_chain,
            "Omega against the folded row",
            &fold_row,
            shared.shared.group_omega,
            shared.shared.group_u,
            split,
        )?;
    } else {
        shared.iopp.statement.add_row(shared.shared.group_u, fold_row);
    }
    shared.iopp.group = shared.shared.group_u;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let row = shared.combine_u(E::from_words(shared_chain.draw_e()));
    let shared_stages: Vec<crate::fp::analysis::Stage> = input
        .lin
        .folded_stages
        .iter()
        .chain(&input.lin.omega_stages)
        .cloned()
        .collect();
    run_iopp_stages(
        frame.cursor,
        &mut shared_chain,
        &mut shared.iopp,
        &shared_stages,
        0,
        input.lin.folded_stages.len(),
        row,
        &shared_run,
        "folded row",
        bytes,
        &input.setting,
        crate::fp::analysis::Scope::Shared,
        shared.shared.split,
        shared.shared.mask,
        true,
    )?;
    shared.iopp.group = shared.shared.group_omega;
    shared_chain.close(ChainKind::Algebraic, &[]);
    let row = shared.combine_omega(E::from_words(shared_chain.draw_e()));
    run_iopp_stages(
        frame.cursor,
        &mut shared_chain,
        &mut shared.iopp,
        &shared_stages,
        input.lin.folded_stages.len(),
        shared_stages.len(),
        row,
        &shared_run,
        "Omega",
        bytes,
        &input.setting,
        crate::fp::analysis::Scope::Shared,
        shared.shared.split,
        shared.shared.mask,
        false,
    )?;
    shared_chain.close(ChainKind::Seal, &[]);
    let mut shared_seal_words = [0u32; 8];
    for i in 0..4 {
        let word = shared_chain.next_word();
        shared_seal_words[2 * i] = word as u32;
        shared_seal_words[2 * i + 1] = (word >> 32) as u32;
    }
    let _shared_seal = Digest(shared_seal_words);

    // The second chain reuses the first chain's statement and book. Its Omega
    // is adopted from the shared chain's statement.
    let second_public = LinPublic::shared_parts(
        &fixed.codes,
        &fixed.lin,
        &fixed.configuration,
        table,
        include_bytes!("../../spikes/plans/key-second-plan.bin"),
        &binding,
    );
    let second_query = second_public.first_query(LIN_LABEL_SECOND);
    let second_run = second_query.run.unwrap();
    let mut second_chain = Chain::new(second_query);
    let mut second_book = OracleBook::default();
    let omega = adopt(
        &mut frame.iopp.statement,
        &mut second_book,
        &shared.iopp.statement,
        &frame.book,
        shared.shared.omega,
    )?;
    let dot = lin_dot_row(
        &fixed.lin,
        &first.claim.g,
        &psi,
        omega,
        omega_lay,
        first.w,
        crate::fp::claims::VectorLay {
            row_bits: crate::fp::tensor::ceil_log2(fixed.lin.w.main_rows),
            place_bits: crate::fp::code_eval::log_ell(&frame.iopp.statement.base[fixed.lin.w.code as usize]),
            rows_high: fixed.lin.w_rows_high,
        },
    );
    frame.iopp.statement.add_row(first.own, dot);
    frame.iopp.group = first.own;
    second_chain.close(ChainKind::Algebraic, &[]);
    let (layout, rows) = frame.iopp.statement.begin_run(first.own);
    let row = crate::fp::claims::bat_combine(&frame.iopp.statement, &layout, &rows, E::from_words(second_chain.draw_e()));
    run_iopp_stages(
        frame.cursor,
        &mut second_chain,
        &mut frame.iopp,
        &input.lin.own_stages,
        0,
        input.lin.own_stages.len(),
        row,
        &second_run,
        "request's own",
        bytes,
        &input.setting,
        crate::fp::analysis::Scope::Request,
        shared.shared.split,
        shared.shared.mask,
        true,
    )?;
    Ok(())
}
