// VEGA-PATCH 2026-06-30 GAP G8 worker 2/5 — zluda_nvjitlink build.rs
// hiprtc0507 caricato via libloading (delay-load implicito), no DELAYLOAD
// link arg necessario perche' dynamic-loading manuale lazy con OnceLock.
fn main() {
    // riservato per futuri linker tweaks (es. /MANIFEST:NO se cdylib clash).
}
