//! Program handle + name expression registry + PTX/bitcode cache —
//! **worker 3/5** sprint G1 nvrtc 30 giu T15:42+ wf_6e53b33d.
//!
//! Modulo complementare al core API in `lib.rs` (worker 1/5+2/5). Aggiunge
//! state ricco lato ZLUDA per ammortizzare il pattern hot di torch.compile
//! / Triton / flash-attn-JIT:
//!
//!   1. `addNameExpression(prog, "kernel<float>")` × N
//!   2. `compileProgram(prog, opts)`
//!   3. `getLoweredName(prog, "kernel<float>", &out)` × N  ← chiamata HOT
//!   4. `getPTX(prog, buf)` × 1
//!
//! Senza cache, step 3 fa hiprtcGetLoweredName ogni volta — su 200+ kernel
//! Triton autotune = bottleneck misurabile. Cache HashMap = O(1) lookup
//! per query ripetute.
//!
//! ## Architettura: piggyback su lib.rs `hiprtc()` accessor
//!
//! Worker 1/5 espone in lib.rs il singleton loader `hiprtc() -> Option<&HipRtc>`.
//! Questo modulo NON ridefinisce la tabella di dispatch — riusa quella sola,
//! aggiungendo sopra registry + cache. Coesistenza intenzionale:
//!   - lib.rs entry points stateless (passano prog opaco a hiprtc raw)
//!   - questo modulo intercetta gli stessi prog opachi e mantiene state
//!     ortogonale (DiagLog, name_exprs vivi, lowered cache, ptx/bc buffer).
//!
//! Worker 5/5 può scegliere di wireare gli entry points di lib.rs a passare
//! per `NvrtcProgram::*` qui sotto per beneficiare della cache — OPT-IN, no
//! breaking change.
//!
//! ## Drop semantics
//!
//! `NvrtcProgram` non possiede l'handle hiprtc (lo possiede lib.rs via il
//! puntatore opaco passato al caller). Quando il caller chiama
//! `nvrtcDestroyProgram`, lib.rs chiama hiprtcDestroyProgram e DEVE chiamare
//! `program::forget(prog)` per evictare la nostra entry dalla registry.
//! Senza forget abbiamo leak DiagLog buffer (qualche KiB) — non critico ma
//! corretto invariant.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use core::ffi::{c_char, c_void, CStr};
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use crate::error::{
    debug_log, DiagLog, NVRTC_ERROR_INVALID_PROGRAM,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION, NVRTC_SUCCESS,
};

// ============================================================================
//  NvrtcProgram — state ricco lato ZLUDA (parallelo all'handle hiprtc)
// ============================================================================

/// State per ogni `nvrtcProgram` opaque pointer. Coabita con l'handle hiprtc
/// reale che vive in lib.rs (passato al caller come puntatore opaco).
///
/// Layout pensato per cache-friendly access pattern del JIT loop (torch.compile
/// + Triton autotune): vec `name_exprs` denso + hashmap `lowered` sparse.
pub struct NvrtcProgram {
    /// Espressioni nome registrate via `addNameExpression`. Mantenute vive
    /// dopo il forward a hiprtc (hipRTC memorizza puntatore, NON copia
    /// — verificato leggendo `hiprtcAddNameExpression` su ROCm 5.7 sorgenti,
    /// vedi `clr/hipamd/src/hiprtc/hiprtcProgram.cpp:142`).
    pub name_exprs: Vec<CString>,
    /// Cache: expression string → lowered (mangled) symbol name.
    /// CString owned in modo che il puntatore returnato a `getLoweredName`
    /// resti stabile per la vita del programma (NVIDIA contract).
    pub lowered: HashMap<String, CString>,
    /// True dopo successful `compileProgram`. Gate per ordering API:
    /// - `addNameExpression` PRIMA → SUCCESS
    /// - `addNameExpression` DOPO → NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
    /// - `getLoweredName` PRIMA → NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
    pub compiled: bool,
    /// Diagnostic log buffer (vedi worker 4/5 `error::DiagLog`). Iniettato
    /// con righe `[ZLUDA]` per facilitare debug PyTorch inductor.
    pub diag: DiagLog,
    /// Buffer PTX/code cache (lazy populate al primo `getPTX`).
    pub ptx: Option<Vec<u8>>,
    /// Buffer bitcode/LTOIR cache.
    pub bitcode: Option<Vec<u8>>,
}

impl NvrtcProgram {
    /// Costruttore — chiamato da `register()` dopo successful hiprtcCreateProgram.
    pub fn new() -> Self {
        Self {
            name_exprs: Vec::with_capacity(8),
            lowered: HashMap::with_capacity(8),
            compiled: false,
            diag: DiagLog::new(),
            ptx: None,
            bitcode: None,
        }
    }
}

impl Default for NvrtcProgram {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
//  ProgramRegistry — *mut c_void (handle hiprtc) → NvrtcProgram state
// ============================================================================

/// Registry globale thread-safe. Chiave = il puntatore opaco hiprtc esposto
/// al caller (== Box<HipProgram> raw da hiprtcCreateProgram).
///
/// Worker 1/5 lib.rs NON cambia ABI: l'opaque pointer è ancora quello hiprtc.
/// Questa registry è additiva — `find()` ritorna None se il caller chiama
/// con un prog non registrato (caso che si presenta solo se uno smoke test
/// chiama direttamente lib.rs raw senza passare per `register()`).
struct Registry {
    map: HashMap<usize, NvrtcProgram>,
}

impl Registry {
    fn new() -> Self {
        Self { map: HashMap::with_capacity(16) }
    }
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Registry::new()))
}

/// Registra un nuovo programma. Chiamato da `lib.rs::nvrtcCreateProgram` SUBITO
/// dopo successful hiprtcCreateProgram.
///
/// **Idempotente** rispetto a prog già esistente — sovrascrive (caso patologico
/// improbabile: hiprtc che riusa puntatore subito dopo destroy).
///
/// # Safety
/// `prog` è il puntatore opaco restituito da hiprtcCreateProgram.
pub fn register(prog: *mut c_void) {
    if prog.is_null() {
        return;
    }
    if let Ok(mut g) = registry().lock() {
        g.map.insert(prog as usize, NvrtcProgram::new());
    }
}

/// Evicta la entry — chiamato da `lib.rs::nvrtcDestroyProgram` DOPO
/// successful hiprtcDestroyProgram (l'handle hiprtc sotto è già libero).
pub fn forget(prog: *mut c_void) {
    if prog.is_null() {
        return;
    }
    if let Ok(mut g) = registry().lock() {
        let _ = g.map.remove(&(prog as usize));
    }
}

/// Marca il programma come compilato — chiamato dopo successful
/// `hiprtcCompileProgram`. Sblocca `getLoweredName`.
pub fn mark_compiled(prog: *mut c_void) {
    let _ = with_mut(prog, |p| p.compiled = true);
}

/// Aggiunge espressione nome alla registry — chiamato PRIMA di forwarda
/// a hiprtcAddNameExpression. Ritorna error se programma già compilato.
///
/// Restituisce un puntatore stabile (per tutta la vita del programma) che
/// può essere passato a hiprtcAddNameExpression. Se la registry non
/// conosce il prog, fallback: alloca CString temporanea e lascia che il
/// caller la usi una sola volta (worst case — hiprtc copia internamente
/// nei casi noti, vedi sorgenti ROCm).
///
/// # Safety
/// `name_expression` non-null e CStr valida.
pub unsafe fn add_name_expression(
    prog: *mut c_void,
    name_expression: *const c_char,
) -> Result<*const c_char, crate::error::nvrtcResult> {
    if name_expression.is_null() {
        return Err(crate::error::NVRTC_ERROR_INVALID_INPUT);
    }
    let cstr = CStr::from_ptr(name_expression).to_owned();
    let res = with_mut(prog, |p| {
        if p.compiled {
            return Err(NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);
        }
        // Salva owned, returna ptr stable dalla allocazione owned.
        let ptr = cstr.as_ptr();
        p.name_exprs.push(cstr);
        // SAFETY: `name_exprs` mantiene la CString viva fintanto che il
        // programma esiste; il puntatore resta valido.
        Ok(ptr)
    });
    match res {
        Some(Ok(p)) => Ok(p),
        Some(Err(e)) => Err(e),
        // Programma non registrato — passa puntatore originale, hiprtc gestirà
        // (worst case copia internamente). Non-fatal, log per diagnosi.
        None => {
            debug_log("program::add_name_expression: prog non registrato, fallback raw ptr");
            Ok(name_expression)
        }
    }
}

/// Recupera lowered name dalla cache. Cache miss → ritorna `Err(MISS)` e il
/// caller (lib.rs) deve chiamare hiprtcGetLoweredName poi `cache_lowered`.
///
/// # Safety
/// `name_expression` non-null CStr valida.
pub unsafe fn get_lowered_cached(
    prog: *mut c_void,
    name_expression: *const c_char,
) -> Result<*const c_char, LoweredCacheMiss> {
    if name_expression.is_null() {
        return Err(LoweredCacheMiss::InvalidInput);
    }
    let expr = match CStr::from_ptr(name_expression).to_str() {
        Ok(s) => s,
        Err(_) => return Err(LoweredCacheMiss::InvalidInput),
    };
    let res = with(prog, |p| {
        if !p.compiled {
            return Err(LoweredCacheMiss::NotCompiled);
        }
        match p.lowered.get(expr) {
            Some(c) => Ok(c.as_ptr()),
            None => Err(LoweredCacheMiss::NotInCache),
        }
    });
    match res {
        Some(r) => r,
        None => Err(LoweredCacheMiss::NotRegistered),
    }
}

/// Memorizza un lowered name nella cache — chiamato da lib.rs dopo cache miss
/// + successful hiprtcGetLoweredName. Il puntatore restituito da hiprtc è
/// owned dal programma hiprtc (valido per la vita del programma), ma noi
/// lo cloniamo in `CString` per disaccoppiare lifetime (in caso futuro lib.rs
/// volesse evictare la registry indipendentemente dal lifetime hiprtc).
///
/// # Safety
/// `name_expression`, `lowered_ptr` non-null e CStr valide.
pub unsafe fn cache_lowered(
    prog: *mut c_void,
    name_expression: *const c_char,
    lowered_ptr: *const c_char,
) {
    if name_expression.is_null() || lowered_ptr.is_null() {
        return;
    }
    let expr = match CStr::from_ptr(name_expression).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return,
    };
    let lowered_owned = CStr::from_ptr(lowered_ptr).to_owned();
    let _ = with_mut(prog, |p| {
        p.lowered.insert(expr, lowered_owned);
    });
}

/// Riusa o salva il PTX nel buffer cache. Pattern call site (lib.rs):
/// ```ignore
/// if let Some(cached) = program::ptx_cached(prog) {
///     // copia diretta da cached, NO hiprtc roundtrip
/// } else {
///     // chiama hiprtcGetCode, poi program::cache_ptx(prog, buf)
/// }
/// ```
pub fn ptx_cached(prog: *mut c_void) -> Option<Vec<u8>> {
    with(prog, |p| p.ptx.clone()).flatten()
}

pub fn cache_ptx(prog: *mut c_void, buf: Vec<u8>) {
    let _ = with_mut(prog, |p| p.ptx = Some(buf));
}

pub fn bitcode_cached(prog: *mut c_void) -> Option<Vec<u8>> {
    with(prog, |p| p.bitcode.clone()).flatten()
}

pub fn cache_bitcode(prog: *mut c_void, buf: Vec<u8>) {
    let _ = with_mut(prog, |p| p.bitcode = Some(buf));
}

/// Append una riga `[ZLUDA]` al diag log del programma. Utile in lib.rs
/// quando intercettiamo mapping non-triviali (es. linker stage error →
/// rimappato a NVRTC_ERROR_COMPILATION, dobbiamo aggiungere contesto perché
/// PyTorch grep "linker" nel log per fallback).
pub fn diag_zluda(prog: *mut c_void, line: &str) {
    let _ = with_mut(prog, |p| p.diag.push_zluda(line));
}

/// Lunghezza del diag log injected (`[ZLUDA]` rows). Da sommare al
/// hiprtcGetProgramLogSize per riportare totale corretto.
pub fn diag_extra_len(prog: *mut c_void) -> usize {
    with(prog, |p| p.diag.buf.len()).unwrap_or(0)
}

/// Copia il diag log injected DOPO il log hiprtc nel buffer caller-provided.
/// Caller responsable per offset corretto (dst = log_buffer + hiprtc_log_len).
///
/// # Safety
/// `dst` deve puntare a almeno `diag_extra_len(prog)` byte scrivibili.
pub unsafe fn diag_copy_after(prog: *mut c_void, dst: *mut c_char) {
    let _ = with(prog, |p| {
        if p.diag.buf.is_empty() {
            return;
        }
        std::ptr::copy_nonoverlapping(
            p.diag.buf.as_ptr() as *const c_char,
            dst,
            p.diag.buf.len(),
        );
    });
}

/// Errori cache lookup — varianti per il caller (lib.rs) per decidere se
/// chiamare hiprtc, restituire errore, o no-op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoweredCacheMiss {
    /// Programma non noto alla registry — caller pass-through diretto a hiprtc.
    NotRegistered,
    /// Programma noto ma `compileProgram` non ancora chiamato — errore strict.
    NotCompiled,
    /// Espressione non in cache — caller chiama hiprtc e poi `cache_lowered`.
    NotInCache,
    /// Input invalido (null ptr o UTF-8 broken).
    InvalidInput,
}

impl LoweredCacheMiss {
    /// Mappa errore cache → nvrtcResult per ritorno diretto al caller NVIDIA.
    pub fn as_nvrtc(self) -> crate::error::nvrtcResult {
        match self {
            LoweredCacheMiss::NotRegistered => NVRTC_ERROR_INVALID_PROGRAM,
            LoweredCacheMiss::NotCompiled => NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
            LoweredCacheMiss::NotInCache => NVRTC_SUCCESS, // non un errore, è hit miss
            LoweredCacheMiss::InvalidInput => crate::error::NVRTC_ERROR_INVALID_INPUT,
        }
    }
}

// ============================================================================
//  Helpers scoped lock
// ============================================================================

fn with<F, R>(prog: *mut c_void, f: F) -> Option<R>
where
    F: FnOnce(&NvrtcProgram) -> R,
{
    let g = registry().lock().ok()?;
    g.map.get(&(prog as usize)).map(f)
}

fn with_mut<F, R>(prog: *mut c_void, f: F) -> Option<R>
where
    F: FnOnce(&mut NvrtcProgram) -> R,
{
    let mut g = registry().lock().ok()?;
    g.map.get_mut(&(prog as usize)).map(f)
}

// ============================================================================
//  Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Helper: chiave finta per testare la registry senza hiprtc reale.
    fn fake_prog(addr: usize) -> *mut c_void {
        addr as *mut c_void
    }

    #[test]
    fn register_and_forget_roundtrip() {
        let p = fake_prog(0xDEAD_0001);
        register(p);
        // Verifica via lookup interno: addNameExpression accetta solo se registrato.
        let n = CString::new("kernel<float>").unwrap();
        let r = unsafe { add_name_expression(p, n.as_ptr()) };
        assert!(r.is_ok());
        forget(p);
        // Dopo forget: lookup fa miss → fallback raw ptr.
        let r2 = unsafe { add_name_expression(p, n.as_ptr()) };
        assert!(r2.is_ok());
        assert_eq!(r2.unwrap(), n.as_ptr()); // fallback returna l'input
    }

    #[test]
    fn add_name_after_compile_fails() {
        let p = fake_prog(0xDEAD_0002);
        register(p);
        mark_compiled(p);
        let n = CString::new("kernel<float>").unwrap();
        let r = unsafe { add_name_expression(p, n.as_ptr()) };
        assert_eq!(r.unwrap_err(), NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);
        forget(p);
    }

    #[test]
    fn lowered_cache_miss_then_hit() {
        let p = fake_prog(0xDEAD_0003);
        register(p);
        mark_compiled(p);
        let expr = CString::new("kernel<float>").unwrap();
        // Miss prima del caching
        let r = unsafe { get_lowered_cached(p, expr.as_ptr()) };
        assert_eq!(r.unwrap_err(), LoweredCacheMiss::NotInCache);
        // Inserisci nella cache
        let lowered = CString::new("_Z6kernelIfvE").unwrap();
        unsafe { cache_lowered(p, expr.as_ptr(), lowered.as_ptr()) };
        // Hit
        let r2 = unsafe { get_lowered_cached(p, expr.as_ptr()) };
        let p2 = r2.unwrap();
        assert!(!p2.is_null());
        let got = unsafe { CStr::from_ptr(p2).to_str().unwrap() };
        assert_eq!(got, "_Z6kernelIfvE");
        forget(p);
    }

    #[test]
    fn lowered_before_compile_returns_error() {
        let p = fake_prog(0xDEAD_0004);
        register(p);
        // Niente mark_compiled
        let expr = CString::new("kernel<float>").unwrap();
        let r = unsafe { get_lowered_cached(p, expr.as_ptr()) };
        assert_eq!(r.unwrap_err(), LoweredCacheMiss::NotCompiled);
        forget(p);
    }

    #[test]
    fn ptx_cache_roundtrip() {
        let p = fake_prog(0xDEAD_0005);
        register(p);
        assert!(ptx_cached(p).is_none());
        cache_ptx(p, b"PTX BYTES".to_vec());
        let got = ptx_cached(p).unwrap();
        assert_eq!(&got, b"PTX BYTES");
        forget(p);
    }

    #[test]
    fn bitcode_cache_roundtrip() {
        let p = fake_prog(0xDEAD_0006);
        register(p);
        assert!(bitcode_cached(p).is_none());
        cache_bitcode(p, b"BC".to_vec());
        assert_eq!(bitcode_cached(p).unwrap(), b"BC".to_vec());
        forget(p);
    }

    #[test]
    fn diag_zluda_injection() {
        let p = fake_prog(0xDEAD_0007);
        register(p);
        assert_eq!(diag_extra_len(p), 0);
        diag_zluda(p, "wave64 split applied");
        assert!(diag_extra_len(p) > 0);
        forget(p);
    }

    #[test]
    fn lowered_cache_miss_codes_map_correctly() {
        assert_eq!(LoweredCacheMiss::NotRegistered.as_nvrtc(), NVRTC_ERROR_INVALID_PROGRAM);
        assert_eq!(
            LoweredCacheMiss::NotCompiled.as_nvrtc(),
            NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
        );
        assert_eq!(LoweredCacheMiss::NotInCache.as_nvrtc(), NVRTC_SUCCESS);
        assert_eq!(
            LoweredCacheMiss::InvalidInput.as_nvrtc(),
            crate::error::NVRTC_ERROR_INVALID_INPUT
        );
    }
}
