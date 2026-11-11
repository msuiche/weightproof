use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=cuda/fold.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_none() {
        return;
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let library = out.join("libwpfold.a");
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let status = Command::new(nvcc)
        .args([
            "--std=c++17",
            "-O3",
            "-arch=native",
            "-DWP_FOLD_LIBRARY",
            "--lib",
            "cuda/fold.cu",
            "-o",
        ])
        .arg(&library)
        .status()
        .expect("failed to execute NVCC");
    assert!(status.success(), "NVCC failed to build cuda/fold.cu");

    let cuda_home = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-search=native={cuda_home}/lib64");
    println!("cargo:rustc-link-lib=static=wpfold");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
