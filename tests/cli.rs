use std::{fs, process::{Command, Output}};

fn run<S: AsRef<str>>(args: &[S]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_weightproof"))
        .args(args.iter().map(AsRef::as_ref))
        .output()
        .unwrap()
}

fn request_args(proof: &str) -> Vec<String> {
    vec![
        "verify-request".into(),
        "--policy".into(),
        "spikes/flashprove/examples/accepted-proof/config/policy.txt".into(),
        "--layout".into(),
        "spikes/flashprove/examples/accepted-proof/config/layout.txt".into(),
        "--kl".into(),
        "spikes/flashprove/examples/accepted-proof/config/KL.bin".into(),
        "--leaf".into(),
        "spikes/flashprove/examples/accepted-proof/ivk/leaf/0.bin".into(),
        "--statement".into(),
        "spikes/plans/request-statement.bin".into(),
        "--request".into(),
        "spikes/flashprove/examples/accepted-proof/example.req".into(),
        "--first-plan".into(),
        "spikes/plans/request-first-plan.bin".into(),
        "--proof".into(),
        proof.into(),
    ]
}

#[test]
fn usage_errors_have_exit_code_two() {
    for args in [
        Vec::<&str>::new(),
        vec!["unknown"],
        vec!["verify-key", "--policy"],
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "args={args:?}");
    }
}

#[test]
fn verify_key_accepts() {
    let output = run(&[
        "verify-key",
        "--policy",
        "spikes/flashprove/examples/accepted-proof/config/policy.txt",
        "--layout",
        "spikes/flashprove/examples/accepted-proof/config/layout.txt",
        "--kl",
        "spikes/flashprove/examples/accepted-proof/config/KL.bin",
        "--key",
        "spikes/flashprove/examples/accepted-proof/ivk/key.bin",
    ]);
    assert_eq!(output.status.code(), Some(0), "stderr={}", String::from_utf8_lossy(&output.stderr));
}

#[test]
fn verify_request_accepts_and_rejects() {
    let output = run(&request_args("spikes/flashprove/examples/accepted-proof/example.proof"));
    assert_eq!(output.status.code(), Some(0), "stderr={}", String::from_utf8_lossy(&output.stderr));

    let mut proof = fs::read("spikes/flashprove/examples/accepted-proof/example.proof").unwrap();
    let last = proof.len() - 1;
    proof[last] ^= 1;
    let path = std::path::PathBuf::from("target/mutated-example.proof");
    fs::write(&path, proof).unwrap();
    let output = run(&request_args(path.to_str().unwrap()));
    assert_eq!(output.status.code(), Some(1), "stderr={}", String::from_utf8_lossy(&output.stderr));
    assert!(String::from_utf8_lossy(&output.stderr).contains("rejected:"));
}
