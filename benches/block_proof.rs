use criterion::{black_box, criterion_group, criterion_main, Criterion};
use weightproof::fp::integer_model::{
    prove_block_batch, prove_block_forest, small_model, verify_block_batch, verify_block_forest,
    DenseModel,
};

fn scaled_model() -> DenseModel {
    let mut model = small_model();
    let layers = 4;
    let width = 8;
    let window = 8;
    let vocab = 16;
    let base = model.weights.layers[0].clone();
    model.weights.layers = (0..layers)
        .map(|_| {
            let mut layer = base.clone();
            layer.wg = vec![vec![0; 2 * width], vec![0; 2 * width]];
            layer.wu = (0..2)
                .map(|i| (0..width).map(|j| i128::from(i == j)).collect())
                .collect();
            layer.wd = (0..width)
                .map(|i| (0..2).map(|j| i128::from(i == j)).collect())
                .collect();
            layer
        })
        .collect();
    model.scales = vec![model.scales[0].clone(); layers];
    model.weights.emb = (0..vocab).map(|_| vec![1, 0]).collect();
    model.weights.wh = vec![vec![1, 0, 0], vec![0, 1, 0]];
    model.tables.cos = vec![1; window];
    model.tables.sin = vec![0; window];
    model.t = window;
    model.f = width;
    model.v = vocab;
    model
}

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

    let scaled = scaled_model();
    let window = [0, 1, 2, 3];
    assert!(weightproof::fp::integer_model::forward(&small_model(), &[0, 1], 4, 4, 1).is_some());
    assert!(weightproof::fp::integer_model::forward(&scaled, &window, 4, 4, 1).is_some());
    let scaled_proof = prove_block_forest(&scaled, &window, 4, 4, 1).unwrap();
    assert!(verify_block_forest(&scaled, &window, &scaled_proof));
    c.bench_function("block-4-layer-f8-window8/prove", |b| {
        b.iter(|| black_box(prove_block_forest(&scaled, &window, 4, 4, 1).unwrap()))
    });
    c.bench_function("block-4-layer-f8-window8/verify", |b| {
        b.iter(|| black_box(verify_block_forest(&scaled, &window, &scaled_proof)))
    });
}

criterion_group!(benches, block_benchmark);
criterion_main!(benches);
