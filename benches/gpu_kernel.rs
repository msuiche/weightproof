use criterion::{black_box, criterion_group, criterion_main, Criterion};
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
}

criterion_group!(benches, extension_fold_benchmark);
criterion_main!(benches);
