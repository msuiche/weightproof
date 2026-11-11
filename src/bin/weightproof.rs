use std::{collections::BTreeMap, env, fs, process};

use weightproof::fp::{
    self,
    heads::{configuration_digest, parse_leaf_record},
    link_public::IndexName,
};

const USAGE: &str = "Usage:
  weightproof verify-key --policy <path> --layout <path> --kl <path> --key <path>
  weightproof verify-request --policy <path> --layout <path> --kl <path> --leaf <path> --statement <path> --request <path> --first-plan <path> --proof <path>

Exit codes: 0 verified, 1 rejected, 2 usage error.";

enum CliError {
    Usage(String),
    Reject(String),
}

fn usage(message: impl Into<String>) -> CliError {
    CliError::Usage(message.into())
}

fn reject(message: impl Into<String>) -> CliError {
    CliError::Reject(message.into())
}

fn read(path: &str) -> Result<Vec<u8>, CliError> {
    fs::read(path).map_err(|error| reject(format!("read-{path}: {error}")))
}

fn options(args: Vec<String>, expected: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
    let mut values = BTreeMap::new();
    let mut iter = args.into_iter();
    while let Some(flag) = iter.next() {
        if !expected.contains(&flag.as_str()) {
            return Err(usage(format!("unknown-option: {flag}\n{USAGE}")));
        }
        if values.contains_key(&flag) {
            return Err(usage(format!("duplicate-option: {flag}\n{USAGE}")));
        }
        let value = iter
            .next()
            .ok_or_else(|| usage(format!("missing-value: {flag}\n{USAGE}")))?;
        if value.starts_with("--") {
            return Err(usage(format!("missing-value: {flag}\n{USAGE}")));
        }
        values.insert(flag, value);
    }
    for flag in expected {
        if !values.contains_key(*flag) {
            return Err(usage(format!("missing-option: {flag}\n{USAGE}")));
        }
    }
    Ok(values)
}

fn run() -> Result<(), CliError> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(|| usage(USAGE))?;
    let rest = args.collect::<Vec<_>>();
    match command.as_str() {
        "verify-key" => {
            let values = options(rest, &["--policy", "--layout", "--kl", "--key"])?;
            let policy = read(&values["--policy"])?;
            let layout = read(&values["--layout"])?;
            let kl = read(&values["--kl"])?;
            let key = read(&values["--key"])?;
            let configuration = configuration_digest(&[&policy, &layout, &kl]);
            fp::verify_key::verify_key_transcript(&fp::key::accepted_input(), &configuration, &key)
                .map_err(|error| reject(format!("rejected: {error}")))?;
            println!("verified");
            Ok(())
        }
        "verify-request" => {
            let values = options(
                rest,
                &[
                    "--policy",
                    "--layout",
                    "--kl",
                    "--leaf",
                    "--statement",
                    "--request",
                    "--first-plan",
                    "--proof",
                ],
            )?;
            let policy = read(&values["--policy"])?;
            let layout = read(&values["--layout"])?;
            let kl = read(&values["--kl"])?;
            let leaf = read(&values["--leaf"])?;
            let statement = read(&values["--statement"])?;
            let request = read(&values["--request"])?;
            let first_plan = read(&values["--first-plan"])?;
            let proof = read(&values["--proof"])?;
            let configuration = configuration_digest(&[&policy, &layout, &kl]);
            let (root, record, _) = parse_leaf_record(&leaf, 256)
                .map_err(|error| reject(format!("rejected: {error}")))?;
            let table = IndexName::new(&root, &configuration, record);
            fp::request_frame::verify_request_transcript(
                &fp::proof_plan::accepted_input(),
                &configuration,
                &layout,
                &table,
                &statement,
                &request,
                &first_plan,
                &proof,
            )
            .map_err(|error| reject(format!("rejected: {error}")))?;
            println!("verified");
            Ok(())
        }
        _ => Err(usage(format!("unknown-command: {command}\n{USAGE}"))),
    }
}

fn main() {
    if let Err(error) = run() {
        match error {
            CliError::Usage(message) => {
                eprintln!("{message}");
                process::exit(2);
            }
            CliError::Reject(message) => {
                eprintln!("{message}");
                process::exit(1);
            }
        }
    }
}
