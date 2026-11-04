#![no_main]

use libfuzzer_sys::fuzz_target;
use weightproof::fp::{
    layout::{ChainPlan, FilePlan, MessageKind, PlanMessage},
    plan::{plan_open, TreeSize},
    walk::{walk_file, OpeningActual},
};

fuzz_target!(|data: &[u8]| {
    let mut messages = vec![PlanMessage::fixed("chain salt", "link's chain", MessageKind::Salts, 32)];
    plan_open(&mut messages, "fuzz opening", &TreeSize::new("tree", 3, 16), 3).unwrap();
    messages.push(PlanMessage::fixed("after", "fuzz", MessageKind::Scalars, 24));
    let plan = FilePlan::entry(0, ChainPlan::new("link's chain", messages)).unwrap();
    let layout = plan.layout().unwrap();

    let positions = [
        data.first().copied().unwrap_or(0) as u64 % 8,
        data.get(1).copied().unwrap_or(0) as u64 % 8,
        data.get(2).copied().unwrap_or(0) as u64 % 8,
    ];
    let mut file = data.get(3..).unwrap_or(&[]).to_vec();
    if file.len() > 4096 {
        file.truncate(4096);
    }
    if file.len() < 8 {
        file.resize(8, 0);
    }
    file[..4].copy_from_slice(b"CPEN");
    file[4..8].copy_from_slice(b"H821");
    let _ = walk_file(&file, &layout, &[OpeningActual { context: "fuzz opening, tree", positions: &positions }]);
});
