// VEGA-PATCH 2026-06-28 sprint-orch v2 cublasLt-10: build.rs gemello a zluda_blaslt
fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg=delayimp.lib");
        println!("cargo:rustc-link-arg=/DELAYLOAD:hipblaslt.dll");
    }
}
