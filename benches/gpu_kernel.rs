use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(feature = "native-cuda")]
use weightproof::fp::{
    chain::Chain,
    integer_model::{bind_block, m1_first_query, prove_psc_dot_product, small_model, ProverTr},
};
use weightproof::fp::{
    field::{ExtensionElement as E, FieldElement as F},
    gpu::{initialize_device, sumcheck_fold_extension},
};

fn extension_fold_benchmark(c: &mut Criterion) {
    initialize_device("CPU").unwrap();
    #[cfg(feature = "native-cuda")]
    let backend = "native-cuda";
    #[cfg(not(feature = "native-cuda"))]
    let backend = "icicle-cpu";

    for size in [1024usize, 4096, 8192, 16384, 65536, 262144] {
        let values: Vec<E> = (0..size as u64)
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

        c.bench_function(&format!("sumcheck-fold-extension-{size}/rust"), |b| {
            b.iter(|| {
                for (out, pair) in output.iter_mut().zip(values.chunks_exact(2)) {
                    *out = pair[0] + challenge * (pair[1] - pair[0]);
                }
                black_box(&output);
            })
        });
        c.bench_function(&format!("sumcheck-fold-extension-{size}/{backend}"), |b| {
            b.iter(|| {
                sumcheck_fold_extension(&values, challenge, &mut output).unwrap();
                black_box(&output);
            })
        });
    }
    #[cfg(feature = "native-cuda")]
    {
        let pairs: Vec<(E, E)> = (0..8192u64)
            .map(|i| {
                (
                    E::from_base(F::from_u64(i.wrapping_mul(0x9e37_79b9_7f4a_7c15))),
                    E::from_base(F::from_u64(i.wrapping_mul(0xd1b5_4a32_d192_ed03))),
                )
            })
            .collect();
        let (statement, _) = bind_block(&small_model(), &[0, 1], 4, 4, 1).unwrap();
        let query = m1_first_query(&statement).unwrap();
        c.bench_function("dot-product-8192/prove", |b| {
            b.iter(|| {
                let mut chain = Chain::new(query.clone());
                let mut transcript = ProverTr::new(&mut chain);
                let proven = prove_psc_dot_product(&mut transcript, pairs.clone());
                black_box(proven)
            })
        });
    }
}

criterion_group!(benches, extension_fold_benchmark);
criterion_main!(benches);
