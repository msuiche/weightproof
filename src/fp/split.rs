//! SPLIT: divide one linear claim between two groups with a shared mask entry.

use crate::fp::{
    chain::{Chain, ChainKind},
    claims::{GroupId, OracleId, Part, Row, Statement},
    error::FpError,
    field::ExtensionElement as E,
    transcript::{elements, words, Cursor},
};

fn member(statement: &Statement, group: GroupId, oracle: OracleId) -> bool {
    statement.groups[group as usize].members.contains(&oracle)
}

pub fn split_verify(
    statement: &mut Statement,
    cursor: &mut Cursor,
    chain: &mut Chain,
    context: &str,
    row: &Row,
    first_group: GroupId,
    second_group: GroupId,
    mask_message: OracleId,
) -> Result<(), FpError> {
    if first_group == second_group
        || row.kind != crate::fp::claims::RowKind::Plain
        || statement.oracles[mask_message as usize].shape.main_rows != 0
        || !member(statement, first_group, mask_message)
        || !member(statement, second_group, mask_message)
    {
        return Err(FpError::Range { context: "split-shape" });
    }
    let side = |oracle| -> Result<bool, FpError> {
        let first = member(statement, first_group, oracle);
        let second = member(statement, second_group, oracle);
        if first == second {
            Err(FpError::Range { context: "split-membership" })
        } else {
            Ok(first)
        }
    };
    let mut first = Row::default();
    let mut second = Row::default();
    for term in &row.terms {
        if side(term.oracle)? {
            first.terms.push(term.clone());
        } else {
            second.terms.push(term.clone());
        }
    }
    for unit in &row.units {
        if side(unit.oracle)? {
            first.units.push(unit.clone());
        } else {
            second.units.push(unit.clone());
        }
    }
    for run in &row.runs {
        if side(run.oracle)? {
            first.runs.push(run.clone());
        } else {
            second.runs.push(run.clone());
        }
    }
    for form in &row.forms {
        if side(form.oracle)? {
            first.forms.push(form.clone());
        } else {
            second.forms.push(form.clone());
        }
    }
    let share_bytes = cursor.take("split target", context)?;
    let share = elements(share_bytes)[0];
    chain.close(ChainKind::Empty, &words(share_bytes));
    let kappa = statement.masks(mask_message).take();
    first.add_unit(kappa.oracle, Part::Mask, kappa.index, E::ONE);
    second.add_unit(kappa.oracle, Part::Mask, kappa.index, -E::ONE);
    first.target = share;
    second.target = row.target - share;
    statement.add_row(first_group, first);
    statement.add_row(second_group, second);
    Ok(())
}
