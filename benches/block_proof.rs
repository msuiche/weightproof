use criterion::{black_box, criterion_group, criterion_main, Criterion};
use weightproof::fp::integer_model::{
    prove_block_batch, prove_block_forest, small_model, verify_block_batch, verify_block_forest,
};

fn block_benchmark(c: &mut Criterion) {
    let model = small_model();
    let tokens = [0, 1];
    let proof = prove_block_forest(&model, &tokens, 4, 4, 1).unwrap();
    c.bench_function("block/prove", |b| {
        b.iter(|| black_box(prove_block_forest(&model, &tokens, 4, 4, 1).unwrap()))
    });
    c.bench_function("block/verify", |b| {
        b.iter(|| black_box(verify_block_forest(&model, &tokens, &proof)))
    });

    let mut two_layer = small_model();
    two_layer
        .weights
        .layers
        .push(two_layer.weights.layers[0].clone());
    two_layer.scales.push(two_layer.scales[0].clone());
    let two_layer_proof = prove_block_forest(&two_layer, &tokens, 4, 4, 1).unwrap();
    c.bench_function("block-2-layer/prove", |b| {
        b.iter(|| black_box(prove_block_forest(&two_layer, &tokens, 4, 4, 1).unwrap()))
    });
    c.bench_function("block-2-layer/verify", |b| {
        b.iter(|| black_box(verify_block_forest(&two_layer, &tokens, &two_layer_proof)))
    });

    let other = [1, 0];
    let requests: &[&[usize]] = &[&tokens, &other];
    let batch = prove_block_batch(&two_layer, requests, 4, 4, 1).unwrap();
    c.bench_function("block-2-layer-batch-2/prove", |b| {
        b.iter(|| black_box(prove_block_batch(&two_layer, requests, 4, 4, 1).unwrap()))
    });
    c.bench_function("block-2-layer-batch-2/verify", |b| {
        b.iter(|| black_box(verify_block_batch(&two_layer, requests, &batch)))
    });
}

criterion_group!(benches, block_benchmark);
criterion_main!(benches);
