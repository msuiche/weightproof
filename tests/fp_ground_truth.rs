//! Ground-truth tests against the stored FlashProve example (D8/D11).
//! Files live in `spikes/flashprove` (gitignored clone). If the spike is not
//! checked out (e.g. CI without it), these tests skip with a note — unit
//! tests in `src/fp/` always run.

use std::{fs, path::PathBuf};

use weightproof::fp::{NextToken, Policy, ProofHead, RequestFile};

fn spike(path: &str) -> Option<Vec<u8>> {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("spikes/flashprove")
        .join(path);
    match fs::read(&full) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!("skipping ground-truth test: {} not found (clone anthropics/FlashProve into spikes/)", full.display());
            None
        }
    }
}

#[test]
fn stored_request_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/example.req") else { return };
    assert_eq!(bytes.len(), 264, "T + 2 = 66 words of 4 bytes");
    let req = RequestFile::parse(&bytes).expect("stored request parses");
    assert_eq!(req.t(), 64);
    assert_eq!(req.r, 60);
    assert_eq!(req.e, 64);
    assert_eq!(req.checked_indices(), 61..=63, "oracle prints tokens=61..63");
    assert_eq!(req.checked_tokens().len(), 3);
    assert_eq!(req.tokens[0], 243, "first word from xxd: 0xf3");
}

#[test]
fn stored_next_token_is_199() {
    let Some(bytes) = spike("examples/accepted-proof/example.y") else { return };
    let y = NextToken::parse(&bytes).expect("stored y parses");
    assert_eq!(y.0, 199, "example README: token 64 = 199");
}

#[test]
fn stored_policy_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/config/policy.txt") else { return };
    let policy = Policy::parse(&bytes).expect("stored policy parses");
    assert!(policy.h821());
    assert_eq!(policy.sec(), Some(128));
    assert_eq!(policy.salt_len(), Some(40));
    assert!(policy.record(), "oracle prints record=1");
    assert_eq!(policy.get("listed"), Some(&[0][..]));
    assert_eq!(policy.get("answers"), Some(&[1][..]));
    assert_eq!(policy.get_named("group", "link"), Some(&[1, 18, 17, 17, 13][..]), "first group line");
    assert_eq!(policy.calls.len(), 18, "8 call + 4 callA + 6 callH lines in the stored file");
    assert_eq!(policy.calls[0].name, "R16a");
    assert_eq!(policy.calls[0].target, "R16");
    assert_eq!(policy.calls[0].symbols, ["G", "U"]);
}

#[test]
fn stored_proof_head_matches_oracle_facts() {
    let Some(bytes) = spike("examples/accepted-proof/example.proof") else { return };
    assert_eq!(bytes.len(), 4_145_300);
    let (head, rest) = ProofHead::parse(&bytes).expect("stored proof head parses");
    assert_eq!(head.entry, 0, "third word is zero in the stored file");
    assert_eq!(rest.len(), bytes.len() - 12);
}

#[test]
fn tampered_stored_files_fail_closed() {
    // Tampering the stored request's length must fail parsing, not "verify".
    if let Some(mut bytes) = spike("examples/accepted-proof/example.req") {
        bytes.truncate(bytes.len() - 1);
        assert!(RequestFile::parse(&bytes).is_err(), "partial word must fail");
    }
    // Flipping the policy's `listed 0` to `listed 1` must be refused.
    if let Some(bytes) = spike("examples/accepted-proof/config/policy.txt") {
        let text = String::from_utf8(bytes).unwrap();
        let tampered = text.replacen("listed 0", "listed 1", 1);
        assert!(Policy::parse(tampered.as_bytes()).is_err(), "listed != 0 must fail");
        // And a version-word flip must be refused as version-unknown.
        let tampered = text.replacen("version H821", "version H822", 1);
        assert!(Policy::parse(tampered.as_bytes()).is_err(), "unknown version must fail");
    }
    // A version flip inside the real proof's head must be refused.
    if let Some(mut bytes) = spike("examples/accepted-proof/example.proof") {
        bytes[7] = b'2';
        assert!(ProofHead::parse(&bytes).is_err(), "H822 must fail");
    }
}
