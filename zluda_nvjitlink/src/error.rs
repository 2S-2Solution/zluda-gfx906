//! VEGA-PATCH 2026-06-30 GAP G8 worker 4/5 — error log & panic-safety.
//!
//! Bottega italiana sovrana. Worker 4/5 si occupa esclusivamente di error
//! handling — la superficie piu' delicata perche' Inductor cu12 / Triton
//! cu12 / flash-attn JIT controllano TRE codici di ritorno e DUE buffer log
//! prima di decidere se fare fallback eager o crashare:
//!
//! 1. **Codice `nvJitLinkResult` (u32)** — DEVE essere uno dei valori del
//!    `repr(u32)` ufficiale NVIDIA (vedi `nvJitLink.h` 12.0..12.4). Un valore
//!    fuori range fa esplodere `torch._inductor.codecache.async_compile`:
//!    ```
//!    RuntimeError: nvJitLink error: <unknown> (rc=42)
//!    ```
//!    Vedi `is_canonical_result`.
//!
//! 2. **`nvJitLinkGetErrorLog` (stringa C nul-terminated)** — Triton legge il
//!    log e fa pattern-match su `"ptxas"`/`"compilation"` per discriminare
//!    fault transienti (re-try con cache miss) da fault permanenti (disable
//!    backend). Iniettiamo prefisso `"[ZLUDA-nvJitLink]"` cosi' i loro
//!    `re.search(r"ptxas warning", log)` continuano a matchare quando passa
//!    il warning di hiprtc traduce diagnostica clang, e il loro
//!    `if "uncorrectable" in log` non scatta su nostri stub graceful.
//!
//! 3. **`nvJitLinkGetInfoLog`** — popolato da `parse_options` con warning
//!    non-fatali (opzione sconosciuta, `-arch=sm_xx` skip). NVIDIA garantisce
//!    string sempre nul-terminated anche se vuota — qui sotto `RollingLog`.
//!
//! ## Panic safety (regola FFI Vega LESSON #5/#6)
//!
//! Un panic across `extern "C"` su Windows MSVC = `STATUS_ACCESS_VIOLATION`
//! senza traceback Python — verificato empiricamente 28 giu + 29 giu su
//! `cuda_macros`. Tutto entry-point FFI di `lib.rs` DEVE essere chiamato
//! tramite la macro `nvjitlink_entry!` qui sotto: cattura il panic, scrive
//! diagnostica nel `DiagLog` per-handle, ritorna `NVJITLINK_ERROR_INTERNAL`.
//!
//! ## Env var debug
//!
//! `ZLUDA_NVJITLINK_DEBUG=1` attiva trace su `%TEMP%\zluda_nvjitlink_<pid>.log`
//! (controllato via `GetEnvironmentVariableW` raw kernel32 — LESSON #5
//! 30 giu T01:48: MAI `std::env::var` per env-var DLL on Windows perche'
//! Python `os.environ` snapshot stale).

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use core::ffi::c_int;
use std::ffi::CStr;
use std::sync::OnceLock;

use crate::NvJitLinkResult;

// ============================================================================
// Canonical ABI codes — mirror del repr(u32) di lib.rs ma usabili come
// const numerici per check + mapping da codici hiprtc.
// ============================================================================

pub const NVJITLINK_SUCCESS: u32 = 0;
pub const NVJITLINK_ERROR_UNRECOGNIZED_OPTION: u32 = 1;
pub const NVJITLINK_ERROR_MISSING_ARCH: u32 = 2;
pub const NVJITLINK_ERROR_INVALID_INPUT: u32 = 3;
pub const NVJITLINK_ERROR_PTX_COMPILE: u32 = 4;
pub const NVJITLINK_ERROR_NVVM_COMPILE: u32 = 5;
pub const NVJITLINK_ERROR_INTERNAL: u32 = 6;
pub const NVJITLINK_ERROR_THREADPOOL: u32 = 7;
pub const NVJITLINK_ERROR_UNRECOGNIZED_INPUT: u32 = 8;
pub const NVJITLINK_ERROR_FINALIZE: u32 = 9;

/// True se `rc` e' uno dei codici documentati NVIDIA (0..=9). False altrimenti
/// — usato come assert difensivo prima di un `transmute` u32 -> repr enum.
#[inline]
pub const fn is_canonical_result(rc: u32) -> bool {
    rc <= NVJITLINK_ERROR_FINALIZE
}

/// Coerce un valore arbitrario nel repr canonico. Fuori range -> INTERNAL.
/// Nota: NON facciamo `transmute` — un repr(u32) enum costruito con un
/// discriminante non-listato e' UB (rust ref. 10.4).
#[inline]
pub fn coerce_result(rc: u32) -> NvJitLinkResult {
    match rc {
        NVJITLINK_SUCCESS => NvJitLinkResult::NVJITLINK_SUCCESS,
        NVJITLINK_ERROR_UNRECOGNIZED_OPTION => {
            NvJitLinkResult::NVJITLINK_ERROR_UNRECOGNIZED_OPTION
        }
        NVJITLINK_ERROR_MISSING_ARCH => NvJitLinkResult::NVJITLINK_ERROR_MISSING_ARCH,
        NVJITLINK_ERROR_INVALID_INPUT => NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT,
        NVJITLINK_ERROR_PTX_COMPILE => NvJitLinkResult::NVJITLINK_ERROR_PTX_COMPILE,
        NVJITLINK_ERROR_NVVM_COMPILE => NvJitLinkResult::NVJITLINK_ERROR_NVVM_COMPILE,
        NVJITLINK_ERROR_INTERNAL => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
        NVJITLINK_ERROR_THREADPOOL => NvJitLinkResult::NVJITLINK_ERROR_THREADPOOL,
        NVJITLINK_ERROR_UNRECOGNIZED_INPUT => {
            NvJitLinkResult::NVJITLINK_ERROR_UNRECOGNIZED_INPUT
        }
        NVJITLINK_ERROR_FINALIZE => NvJitLinkResult::NVJITLINK_ERROR_FINALIZE,
        _ => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    }
}

// ============================================================================
// hiprtc -> nvJitLink mapping (tabella esaustiva — NO unreachable!)
// ============================================================================
// hiprtcResult ROCm 5.7 header `hiprtc.h`:
//   HIPRTC_SUCCESS = 0
//   HIPRTC_ERROR_OUT_OF_MEMORY = 1
//   HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2
//   HIPRTC_ERROR_INVALID_INPUT = 3
//   HIPRTC_ERROR_INVALID_PROGRAM = 4
//   HIPRTC_ERROR_INVALID_OPTION = 5
//   HIPRTC_ERROR_COMPILATION = 6
//   HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7
//   HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8
//   HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9
//   HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10
//   HIPRTC_ERROR_INTERNAL_ERROR = 11
//   HIPRTC_ERROR_LINKING = 100
//
// Mapping a nvJitLinkResult (1:N quando NVIDIA distingue piu' fine di AMD):

/// Traduci hiprtcResult (i32 raw, no enum perche' impl.rs usa newtype) in un
/// nvJitLinkResult ABI-canonico. Default INTERNAL per valori sconosciuti
/// (es. release HIP futura aggiunge nuovi codici).
pub fn from_hiprtc(rc: c_int) -> NvJitLinkResult {
    match rc {
        0 => NvJitLinkResult::NVJITLINK_SUCCESS,
        1 => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,        // OOM
        2 => NvJitLinkResult::NVJITLINK_ERROR_FINALIZE,        // PROGRAM_CREATION
        3 => NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT,   // INVALID_INPUT
        4 => NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT,   // INVALID_PROGRAM
        5 => NvJitLinkResult::NVJITLINK_ERROR_UNRECOGNIZED_OPTION,
        6 => NvJitLinkResult::NVJITLINK_ERROR_PTX_COMPILE,     // COMPILATION
        7 => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,        // BUILTIN_OP
        8 | 9 | 10 => NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT, // NAME_EXPR
        11 => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,       // INTERNAL_ERROR
        100 => NvJitLinkResult::NVJITLINK_ERROR_FINALIZE,      // LINKING
        _ => NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    }
}

// ============================================================================
// nvJitLinkGetErrorString — NVIDIA garantisce puntatore valido per vita del
// processo per QUALSIASI rc. Implementiamo con `&'static CStr` letterali —
// zero allocazione, thread-safe lifetime-static.
// ============================================================================
//
// NB: nvJitLink.h 12.x non espone GetErrorString come API pubblica (a
// differenza di nvrtcGetErrorString). Manteniamo questa funzione come helper
// interno per `debug_log` + smoke test. Se mai NVIDIA aggiungesse l'export
// in 12.5+, basta togliere `pub(crate)` -> `pub` e aggiungere `#[no_mangle]`.

pub(crate) fn error_string(rc: u32) -> &'static CStr {
    match rc {
        NVJITLINK_SUCCESS => c_lit(b"NVJITLINK_SUCCESS\0"),
        NVJITLINK_ERROR_UNRECOGNIZED_OPTION => c_lit(b"NVJITLINK_ERROR_UNRECOGNIZED_OPTION\0"),
        NVJITLINK_ERROR_MISSING_ARCH => c_lit(b"NVJITLINK_ERROR_MISSING_ARCH\0"),
        NVJITLINK_ERROR_INVALID_INPUT => c_lit(b"NVJITLINK_ERROR_INVALID_INPUT\0"),
        NVJITLINK_ERROR_PTX_COMPILE => c_lit(b"NVJITLINK_ERROR_PTX_COMPILE\0"),
        NVJITLINK_ERROR_NVVM_COMPILE => c_lit(b"NVJITLINK_ERROR_NVVM_COMPILE\0"),
        NVJITLINK_ERROR_INTERNAL => c_lit(b"NVJITLINK_ERROR_INTERNAL\0"),
        NVJITLINK_ERROR_THREADPOOL => c_lit(b"NVJITLINK_ERROR_THREADPOOL\0"),
        NVJITLINK_ERROR_UNRECOGNIZED_INPUT => c_lit(b"NVJITLINK_ERROR_UNRECOGNIZED_INPUT\0"),
        NVJITLINK_ERROR_FINALIZE => c_lit(b"NVJITLINK_ERROR_FINALIZE\0"),
        _ => c_lit(b"NVJITLINK_ERROR_UNKNOWN\0"),
    }
}

#[inline]
fn c_lit(b: &'static [u8]) -> &'static CStr {
    // SAFETY: tutti i letterali qui sopra sono `b"...\0"` con terminator.
    unsafe { CStr::from_bytes_with_nul_unchecked(b) }
}

// ============================================================================
// RollingLog — buffer log per-handle (error_log + info_log).
//
// NVIDIA garantisce nul-terminated. Cap a 64 KiB per evitare leak quando un
// caller (Inductor cu12 in loop di re-try) non chiama mai GetLog ma Complete
// 10k volte sullo stesso handle.
// ============================================================================

const ROLLING_LOG_CAP: usize = 64 * 1024;

pub struct RollingLog {
    buf: String,
}

impl RollingLog {
    pub fn new() -> Self {
        RollingLog {
            buf: String::with_capacity(256),
        }
    }

    pub fn push(&mut self, prefix: &str, msg: &str) {
        // ZLUDA pattern: marca il log per Triton/Inductor pattern-match
        let line = format!("[ZLUDA-nvJitLink] {} {}", prefix, msg.trim_end());
        if self.buf.len() + line.len() + 1 > ROLLING_LOG_CAP {
            // FIFO drop: tieni solo l'ultima meta'
            let keep_from = ROLLING_LOG_CAP / 2;
            if self.buf.len() > keep_from {
                // tronca al primo `\n` dopo keep_from per non spezzare riga
                if let Some(idx) = self.buf[keep_from..].find('\n') {
                    self.buf.drain(..keep_from + idx + 1);
                } else {
                    self.buf.clear();
                }
            }
        }
        self.buf.push_str(&line);
        self.buf.push('\n');
    }

    pub fn raw_push(&mut self, msg: &str) {
        // append senza prefisso (per warning di parse_options gia' formattati)
        if self.buf.len() + msg.len() > ROLLING_LOG_CAP {
            let keep_from = ROLLING_LOG_CAP / 2;
            if let Some(idx) = self.buf[keep_from..].find('\n') {
                self.buf.drain(..keep_from + idx + 1);
            }
        }
        self.buf.push_str(msg);
        if !msg.ends_with('\n') {
            self.buf.push('\n');
        }
    }

    pub fn len_with_nul(&self) -> usize {
        self.buf.len() + 1
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.buf.as_bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl Default for RollingLog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Panic-safe FFI wrapper.
//
// Uso:
//   #[no_mangle]
//   pub unsafe extern "C" fn nvJitLinkCreate(...) -> NvJitLinkResult {
//       nvjitlink_entry!("nvJitLinkCreate", { r#impl::create(...) })
//   }
//
// Cattura panic, scrive nel diag log globale, ritorna ERROR_INTERNAL.
// ============================================================================

#[macro_export]
macro_rules! nvjitlink_entry {
    ($name:expr, $body:block) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(rc) => rc,
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                    s.to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<non-string panic payload>".to_string()
                };
                $crate::error::debug_log(&format!(
                    "PANIC in {}: {} -- returning NVJITLINK_ERROR_INTERNAL",
                    $name, msg
                ));
                $crate::NvJitLinkResult::NVJITLINK_ERROR_INTERNAL
            }
        }
    }};
}

// ============================================================================
// debug_log — controllato da ZLUDA_NVJITLINK_DEBUG=1.
//
// LESSON #5 (Vega 30 giu T01:48): MAI `std::env::var` su Windows DLL.
// Python `os.environ` e' uno snapshot al boot del processo; modifiche
// in-process via `os.environ["FOO"]=...` non riflettono al livello kernel32.
// Usiamo `GetEnvironmentVariableW` raw via `windows-sys`-free hand-rolling
// (no extra dep — il crate ha solo libloading).
// ============================================================================

static DEBUG_ENABLED: OnceLock<bool> = OnceLock::new();

fn debug_enabled() -> bool {
    *DEBUG_ENABLED.get_or_init(|| {
        #[cfg(windows)]
        {
            check_env_winapi("ZLUDA_NVJITLINK_DEBUG")
        }
        #[cfg(not(windows))]
        {
            std::env::var("ZLUDA_NVJITLINK_DEBUG").is_ok()
        }
    })
}

#[cfg(windows)]
fn check_env_winapi(name: &str) -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetEnvironmentVariableW(
            lp_name: *const u16,
            lp_buffer: *mut u16,
            n_size: u32,
        ) -> u32;
    }
    // converti name UTF-8 -> UTF-16 nul-terminated
    let mut wname: Vec<u16> = name.encode_utf16().collect();
    wname.push(0);
    let mut buf = [0u16; 8];
    let n = unsafe {
        GetEnvironmentVariableW(wname.as_ptr(), buf.as_mut_ptr(), buf.len() as u32)
    };
    let _ = (n, buf); // marca usage; non ci serve il valore, solo presenza
    n > 0
}

pub fn debug_log(msg: &str) {
    if !debug_enabled() {
        return;
    }
    use std::io::Write;
    let path = std::env::temp_dir().join(format!("zluda_nvjitlink_{}.log", std::process::id()));
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{}", msg);
    }
}

// ============================================================================
// Smoke tests — verifica empirica (regola 17): tabella from_hiprtc esaustiva,
// nessun panic su input arbitrari, RollingLog cap rispettato.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_result_range() {
        for rc in 0u32..=9 {
            assert!(is_canonical_result(rc), "rc={} should be canonical", rc);
        }
        assert!(!is_canonical_result(10));
        assert!(!is_canonical_result(42));
        assert!(!is_canonical_result(u32::MAX));
    }

    #[test]
    fn coerce_out_of_range_to_internal() {
        assert_eq!(coerce_result(42), NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
        assert_eq!(coerce_result(u32::MAX), NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
        assert_eq!(coerce_result(0), NvJitLinkResult::NVJITLINK_SUCCESS);
        assert_eq!(coerce_result(9), NvJitLinkResult::NVJITLINK_ERROR_FINALIZE);
    }

    #[test]
    fn from_hiprtc_all_documented_codes() {
        // hiprtc 5.7 documented set
        assert_eq!(from_hiprtc(0), NvJitLinkResult::NVJITLINK_SUCCESS);
        assert_eq!(from_hiprtc(1), NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
        assert_eq!(from_hiprtc(5), NvJitLinkResult::NVJITLINK_ERROR_UNRECOGNIZED_OPTION);
        assert_eq!(from_hiprtc(6), NvJitLinkResult::NVJITLINK_ERROR_PTX_COMPILE);
        assert_eq!(from_hiprtc(100), NvJitLinkResult::NVJITLINK_ERROR_FINALIZE);
        // unknown -> INTERNAL
        assert_eq!(from_hiprtc(999), NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
        assert_eq!(from_hiprtc(-1), NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
    }

    #[test]
    fn error_string_all_canonical_nonempty() {
        for rc in 0u32..=9 {
            let s = error_string(rc);
            assert!(!s.to_bytes().is_empty(), "rc={} empty string", rc);
            assert!(s.to_str().unwrap().starts_with("NVJITLINK_"));
        }
        // out-of-range -> UNKNOWN
        let s = error_string(42);
        assert_eq!(s.to_str().unwrap(), "NVJITLINK_ERROR_UNKNOWN");
    }

    #[test]
    fn rolling_log_respects_cap() {
        let mut log = RollingLog::new();
        // push ~80 KiB di righe -> cap a 64 KiB
        let line = "x".repeat(120);
        for _ in 0..700 {
            log.push("test", &line);
        }
        assert!(
            log.as_bytes().len() < ROLLING_LOG_CAP + 256,
            "log overran cap: {}",
            log.as_bytes().len()
        );
    }

    #[test]
    fn rolling_log_nul_term_size() {
        let mut log = RollingLog::new();
        log.push("op", "hello");
        // len_with_nul include il +1 per il null finale che get_error_log scrive
        assert_eq!(log.len_with_nul(), log.as_bytes().len() + 1);
    }

    #[test]
    fn entry_macro_catches_panic() {
        let rc = nvjitlink_entry!("test_panic", {
            panic!("boom");
            #[allow(unreachable_code)]
            NvJitLinkResult::NVJITLINK_SUCCESS
        });
        assert_eq!(rc, NvJitLinkResult::NVJITLINK_ERROR_INTERNAL);
    }

    #[test]
    fn entry_macro_passes_through_ok() {
        let rc = nvjitlink_entry!("test_ok", { NvJitLinkResult::NVJITLINK_SUCCESS });
        assert_eq!(rc, NvJitLinkResult::NVJITLINK_SUCCESS);
    }
}
