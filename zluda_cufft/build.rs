// VEGA-PATCH 2026-06-30 TASK 5 worker 2/3 — zluda_cufft
// Backend hipfft 5.7 via delay-load. Se hipfft.dll non c'e' a runtime,
// la cdylib si carica comunque ma le fn cufft* ritornano CUFFT_ERROR_NOT_SUPPORTED
// tramite il delaylink hook (vedi mod windows in src/lib.rs).
fn main() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg=delayimp.lib");
        println!("cargo:rustc-link-arg=/DELAYLOAD:hipfft.dll");
    }
}
