use weightproof::fp::{blake2::Hasher, heads::configuration_digest};

fn main() {
    let out = Hasher::new(0, 0, &[0; 8], 64).finish_words();
    println!("empty-512: {}", hex(&out));

    let mut h = Hasher::new(0, 0, &[0; 8], 64);
    h.push(b"abc");
    println!("abc-512:   {}", hex(&h.finish_words()));

    let mut h = Hasher::new(0, 0, &[0; 8], 32);
    h.push(b"abc");
    println!("abc-256:   {}", h.finish().to_hex());

    // 128 zero bytes (one full block): exercises compress alone.
    let mut h = Hasher::new(0, 0, &[0; 8], 64);
    h.push(&[0u8; 128]);
    println!("zeros128:  {}", hex(&h.finish_words()));

    // 19 zero bytes (label of zeros + abc, but all zeros).
    let mut h = Hasher::new(0, 0, &[0; 8], 64);
    h.push(&[0u8; 3]);
    println!("zeros19:   {}", hex(&h.finish_words()));

    // Dump the final block of the abc case to check input construction.
    let mut h = Hasher::new(0, 0, &[0; 8], 64);
    h.push(b"abc");
    h.dump_state();
}

fn hex(words: &[u64; 8]) -> String {
    let mut s = String::new();
    for w in words {
        for b in w.to_le_bytes() {
            s.push_str(&format!("{:02x}", b));
        }
    }
    s
}
