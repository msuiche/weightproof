use criterion::{black_box, criterion_group, criterion_main, Criterion};
use weightproof::fp::{
    field::{ExtensionElement as E, FieldElement as F},
    gpu::{initialize_device, sumcheck_fold_extension},
};

fn extension_fold_benchmark(c: &mut Criterion) {
    initialize_device("CPU").unwrap();
    let values: Vec<E> = (0..1024u64)
        .map(|i| {
            E([
                F::from_u64(i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).0,
                F::from_u64(i.wrapping_mul(0xd1b5_4a32_d192_ed03)).0,
                F::from_u64(i.wrapping_mul(0x94d0_49bb_1331_11eb)).0,
            ])
        })
        .collect();
    let challenge = E([7, 11, 13]);
    let mut output = vec![E::ZERO; values.len() / 2];

    c.bench_function("sumcheck-fold-extension-1024/rust", |b| {
        b.iter(|| {
            for (out, pair) in output.iter_mut().zip(values.chunks_exact(2)) {
                *out = pair[0] + challenge * (pair[1] - pair[0]);
            }
            black_box(&output);
        })
    });
    c.bench_function("sumcheck-fold-extension-1024/icicle-cpu", |b| {
        b.iter(|| {
            sumcheck_fold_extension(&values, challenge, &mut output).unwrap();
            black_box(&output);
        })
    });
}

criterion_group!(benches, extension_fold_benchmark);
criterion_main!(benches);
