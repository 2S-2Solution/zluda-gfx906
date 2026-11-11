// VEGA-PATCH 2026-06-30 G6 — no-op build.rs.
//
// Il crate usa libloading per dispatch dinamico verso rocsolver.dll
// (ROCm 5.7 install: C:\Program Files\AMD\ROCm\5.7\bin\rocsolver.dll).
// Niente link-time coupling: zero `rocsolver.lib` richiesta a build time,
// la DLL viene risolta al primo OnceCell::get_or_init().
//
// Vantaggio: zero blocker su nodi senza ROCm SDK installato — la DLL si
// risolve solo se cusolver* effettivamente chiamata da PyTorch/CuPy.
// In assenza/errore di load la funzione torna NOT_SUPPORTED (caller fa
// fallback MAGMA/CPU). Regola 4-bis ROCm 5.7 ESCLUSIVO.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
}
