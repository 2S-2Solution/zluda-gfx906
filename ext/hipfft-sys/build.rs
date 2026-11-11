use std::env::VarError;
use std::{env, path::PathBuf};

fn main() -> Result<(), VarError> {
    if cfg!(windows) {
        // ROCm 5.7 esclusivo (regola 4-bis Vega). hipfft.lib in
        // C:\Program Files\AMD\ROCm\5.7\lib (vendor AMD).
        println!("cargo:rustc-link-lib=dylib=hipfft");
        let env = env::var("CARGO_CFG_TARGET_ENV")?;
        if env == "msvc" {
            // Override locale via ext/hipfft-sys/lib/ se presente
            let mut local = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
            local.push("lib");
            if local.exists() {
                println!("cargo:rustc-link-search=native={}", local.display());
            } else {
                println!("cargo:rustc-link-search=native=C:\\Program Files\\AMD\\ROCm\\5.7\\lib");
            }
        } else {
            println!("cargo:rustc-link-search=native=C:\\Program Files\\AMD\\ROCm\\5.7\\lib");
        }
    } else {
        println!("cargo:rustc-link-lib=dylib=hipfft");
        println!("cargo:rustc-link-search=native=/opt/rocm/lib/");
    }
    Ok(())
}
