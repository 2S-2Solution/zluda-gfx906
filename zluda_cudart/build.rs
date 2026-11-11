// VEGA-PATCH 2026-06-30 TASK A worker3/5
// build.rs per zluda_cudart: quando la feature `cuda12` è attiva, rinomina
// l'output da cudart64_110.dll a cudart64_12.dll (post-link).
// Non possiamo cambiare `[lib].name` da Cargo.toml in base a feature
// (Cargo non supporta nome lib condizionale), quindi facciamo il rename
// nella fase POST-build leggendo cargo:rustc-env.
//
// Strategia:
//   1. Cargo produce sempre `cudart64_110.dll` (nome canonical) in target/<profile>/.
//   2. Lo script di build seguente (eseguito ALL'INIZIO del build, non DOPO) imposta solo
//      una env var `ZLUDA_CUDART_TARGET=cudart64_12` quando feature `cuda12` è attiva.
//   3. Il rename effettivo viene fatto dallo script wrapper `build_cudart.cmd`
//      che chiamiamo dopo `cargo build --features cuda12`.
//
// Inoltre: questo build.rs propaga il flag `--cfg cudart_12` al codice Rust per
// abilitare il modulo cuda12_extensions e bumpare CUDART_VERSION.

use std::env;

fn main() {
    // Re-run se l'ambiente cambia (no input file watch necessario qui).
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA12");

    // VEGA-FIX TASK A worker1/5 2026-06-30 T08:15: dichiara il cfg custom `cudart_12`
    // per evitare warning `unexpected_cfgs` introdotto in Rust 1.80+ (cargo +nightly).
    // Senza questa dichiarazione il build emette warning ma non e' fatal.
    println!("cargo:rustc-check-cfg=cfg(cudart_12)");

    if env::var("CARGO_FEATURE_CUDA12").is_ok() {
        // Propaga al codice Rust: i blocchi `#[cfg(cudart_12)]` saranno compilati in.
        println!("cargo:rustc-cfg=cudart_12");
        // Hint per il wrapper script: leggere questa env var dopo build per sapere
        // se rinominare l'output.
        println!("cargo:warning=zluda_cudart: feature cuda12 ATTIVA -> cudart_12 cfg ON, post-build rename cudart64_110.dll -> cudart64_12.dll necessario");
    } else {
        println!("cargo:warning=zluda_cudart: feature cuda12 DISATTIVA -> build cudart64_110.dll (default torch 2.3/2.4+cu118)");
    }
}
