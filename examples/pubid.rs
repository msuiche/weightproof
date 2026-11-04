use std::{fs, path::PathBuf};
use weightproof::fp::heads::{
    configuration_digest, parse_key_record, published_id, record_seeds, table_name,
};

fn main() {
    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("spikes/flashprove/examples/accepted-proof");
    let read = |p: &str| fs::read(dir.join(p)).unwrap();
    let policy = read("config/policy.txt");
    let kl = read("config/KL.bin");
    let layout = read("config/layout.txt");
    let key = read("ivk/key.bin");

    let config = configuration_digest(&[&policy, &layout, &kl]);
    println!("h_par: {}", config.to_words_hex());

    let shape = [23, 524288, 65535, 8, 1, 40];
    let (root, answers, _rest) = parse_key_record(&key, 256).unwrap();
    let name = table_name(&root, &config, shape);
    println!("name: {}", hex(&name));

    let seeds = record_seeds(&name, 19, 1);
    println!(
        "z0: {:016x} {:016x} {:016x}",
        seeds[0].z[0][0], seeds[0].z[0][1], seeds[0].z[0][2]
    );
    println!(
        "beta: {:016x} {:016x} {:016x}",
        seeds[0].beta[0], seeds[0].beta[1], seeds[0].beta[2]
    );
    println!(
        "a0: {:016x} {:016x} {:016x}",
        answers[0][0], answers[0][1], answers[0][2]
    );

    let id = published_id(&config, &name, &seeds, &answers);
    println!("id: {}", id.to_hex());
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::new();
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}
