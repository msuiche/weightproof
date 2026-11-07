use weightproof::fp::{digest::Digest, request_public::accepted_request_fixed};

fn main() {
    let fixed = accepted_request_fixed(Digest::ZERO, include_bytes!("../spikes/flashprove/examples/accepted-proof/config/layout.txt"));
    for (link, arrays) in fixed.layout.links.iter().enumerate() {
        for array in arrays {
            println!("link={} name={} message={} size_i={} size_c={}", link, array.name, array.message, array.size_i, array.size_c);
        }
    }
}
