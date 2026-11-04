const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

fn g(v: &mut [u64; 16], r: usize, i: usize, a: usize, b: usize, c: usize, d: usize) {
    println!(
        "in-a: {:016x} in-b: {:016x} m0: {:016x}",
        v[a],
        v[b],
        v[SIGMA[r][2 * i]]
    );
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(v[SIGMA[r][2 * i]]);
    println!("a1:  {:016x}", v[a]);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    println!("d1:  {:016x}", v[d]);
    v[c] = v[c].wrapping_add(v[d]);
    println!("c1:  {:016x}", v[c]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    println!("b1:  {:016x}", v[b]);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(v[SIGMA[r][2 * i + 1]]);
    println!("a2:  {:016x}", v[a]);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    println!("d2:  {:016x}", v[d]);
    v[c] = v[c].wrapping_add(v[d]);
    println!("c2:  {:016x}", v[c]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
    println!("b2:  {:016x}", v[b]);
}

fn round(v: &mut [u64; 16], r: usize) {
    g(v, r, 0, 0, 4, 8, 12);
    g(v, r, 1, 1, 5, 9, 13);
    g(v, r, 2, 2, 6, 10, 14);
    g(v, r, 3, 3, 7, 11, 15);
    g(v, r, 4, 0, 5, 10, 15);
    g(v, r, 5, 1, 6, 11, 12);
    g(v, r, 6, 2, 7, 8, 13);
    g(v, r, 7, 3, 4, 9, 14);
}

fn main() {
    let mut h = IV;
    h[0] ^= 0x01010040u64;
    let mut block = [0u8; 128];
    block[16] = b'a';
    block[17] = b'b';
    block[18] = b'c';
    let mut m = [0u64; 16];
    for (i, chunk) in block.chunks_exact(8).enumerate() {
        let mut le = [0u8; 8];
        le.copy_from_slice(chunk);
        m[i] = u64::from_le_bytes(le);
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(&h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= 19u64;
    v[14] ^= u64::MAX;
    g(&mut v, 0, 0, 0, 4, 8, 12);
    for (i, w) in v.iter().enumerate() {
        print!("{:016x} ", w);
        if i == 15 {
            println!();
        }
    }
    round(&mut v, 0);
    for (i, w) in v.iter().enumerate() {
        print!("{:016x} ", w);
        if i == 15 {
            println!();
        }
    }
}
