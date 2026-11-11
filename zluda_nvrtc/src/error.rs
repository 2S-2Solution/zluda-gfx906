//! Error handling — worker 4/5 G1 nvrtc.
//!
//! Bottega italiana sovrana: il modulo gestisce tre superfici distinte di
//! errore che PyTorch / Triton / Inductor / flash-attn-JIT controllano:
//!
//! 1. **Valore di ritorno `nvrtcResult`** — codice intero ABI NVIDIA. Mai
//!    rilasciare codici fuori dall'enum ufficiale; un valore non documentato
//!    fa esplodere `torch._inductor.codecache` (vedi traceback noto
//!    `RuntimeError: NVRTC error: <unknown>`). Tutti i mapping passano per
//!    `NvrtcResult::from_hiprtc` con tabella esaustiva.
//!
//! 2. **`nvrtcGetErrorString(rc)`** — restituisce stringa C statica.
//!    NVIDIA garantisce puntatore valido per la vita del processo per
//!    *qualsiasi* `rc`, anche fuori range, in tal caso "NVRTC_ERROR_UNKNOWN".
//!    Implementiamo con `&'static CStr` letterali, **zero allocazione**, per
//!    rispettare il contratto thread-safe lifetime-static.
//!
//! 3. **`nvrtcGetProgramLog`** — buffer log compilazione per-programma. Il
//!    log è alimentato sia da hiprtc (passato traduce diagnostica clang)
//!    sia da messaggi "Vega" iniettati da noi (es. "wave64 split applied",
//!    "PTX 8.3 feature downgraded"). Lo gestiamo nel modulo `program` ma
//!    qui definiamo il tipo `DiagLog` riusato.
//!
//! Inoltre esponiamo un `ZLUDA_NVRTC_DEBUG=1` (env var) che attiva trace su
//! `%TEMP%\zluda_nvrtc_<pid>.log` — utile per debug PyTorch JIT cache miss.
//!
//! ## Panic safety
//!
//! Ogni entry-point FFI **deve** essere avvolto in `catch_unwind`. Un panic
//! che attraversa il boundary `extern "C"` su Windows MSVC = `STATUS_ACCESS_VIOLATION`
//! che torna a Python come crash hard senza traceback (regola 17: verificato
//! empiricamente con `cargo test -- --test-threads=1` su panic indotto in
//! `cuda_macros` 28 giu). La macro `nvrtc_entry!` qui sotto è il wrapper
//! canonico — workers 1/2/3 devono usarla per OGNI funzione `#[no_mangle]`.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::cell::RefCell;
use std::ffi::c_char;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

// ============================================================================
//  nvrtcResult — ABI NVIDIA (nvrtc.h release 12.6)
// ============================================================================

/// `nvrtcResult` ABI NVIDIA. **Mai** cambiare i valori — `torch._inductor`
/// matcha letterali interi.
pub type nvrtcResult = u32;

pub const NVRTC_SUCCESS: nvrtcResult = 0;
pub const NVRTC_ERROR_OUT_OF_MEMORY: nvrtcResult = 1;
pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: nvrtcResult = 2;
pub const NVRTC_ERROR_INVALID_INPUT: nvrtcResult = 3;
pub const NVRTC_ERROR_INVALID_PROGRAM: nvrtcResult = 4;
pub const NVRTC_ERROR_INVALID_OPTION: nvrtcResult = 5;
pub const NVRTC_ERROR_COMPILATION: nvrtcResult = 6;
pub const NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: nvrtcResult = 7;
pub const NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: nvrtcResult = 8;
pub const NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: nvrtcResult = 9;
pub const NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID: nvrtcResult = 10;
pub const NVRTC_ERROR_INTERNAL_ERROR: nvrtcResult = 11;
pub const NVRTC_ERROR_TIME_FILE_WRITE_FAILED: nvrtcResult = 12;
pub const NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED: nvrtcResult = 13;
pub const NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED: nvrtcResult = 14;
pub const NVRTC_ERROR_PCH_CREATE: nvrtcResult = 15;
pub const NVRTC_ERROR_CANCELLED: nvrtcResult = 16;

// ============================================================================
//  hiprtcResult — ABI ROCm 5.7 (hiprtc.h)
// ============================================================================

/// Tipo opaco per il valore intero che hiprtc restituisce. Worker 2/5 importa
/// `hiprtcResult` da `hiprtc-sys`; qui definiamo solo le costanti per evitare
/// dipendenza forte (sostituibili con `hiprtcResult as u32`).
pub type hiprtcResult = u32;

pub const HIPRTC_SUCCESS: hiprtcResult = 0;
pub const HIPRTC_ERROR_OUT_OF_MEMORY: hiprtcResult = 1;
pub const HIPRTC_ERROR_PROGRAM_CREATION_FAILURE: hiprtcResult = 2;
pub const HIPRTC_ERROR_INVALID_INPUT: hiprtcResult = 3;
pub const HIPRTC_ERROR_INVALID_PROGRAM: hiprtcResult = 4;
pub const HIPRTC_ERROR_INVALID_OPTION: hiprtcResult = 5;
pub const HIPRTC_ERROR_COMPILATION: hiprtcResult = 6;
pub const HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE: hiprtcResult = 7;
pub const HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION: hiprtcResult = 8;
pub const HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION: hiprtcResult = 9;
pub const HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID: hiprtcResult = 10;
pub const HIPRTC_ERROR_INTERNAL_ERROR: hiprtcResult = 11;
pub const HIPRTC_ERROR_LINKING: hiprtcResult = 100;

// ============================================================================
//  Translation hiprtc -> nvrtc
// ============================================================================

/// Mapping esaustivo. Da 0..=11 i codici coincidono per puro ordine storico
/// (NVIDIA è il padre dell'ABI, hipRTC ha copiato). Da 12 in su nvrtc estende
/// con PCH / cancel / time-file: hiprtc non li emette mai, perciò sono solo
/// emissioni dirette nostre.
///
/// **Caso speciale `HIPRTC_ERROR_LINKING` (100)**: nvrtc non ha un equivalente
/// diretto perché il linker lo fa cuLink (gestito dal worker GAP-9 chiuso 30 giu).
/// Quando hiprtc emette 100 in compileProgram (può capitare con `-rdc=true`)
/// rimappiamo a `NVRTC_ERROR_COMPILATION` + aggiungiamo riga al log
/// "[ZLUDA] linker stage failure (HIPRTC_ERROR_LINKING)". PyTorch ricerca
/// "linker" nel log per fallback, così la chain JIT non si rompe.
///
/// Regola 17 — verificato empiricamente: codici 12..=16 non sono mai stati
/// osservati nei log autotune notte 25-30 giu, ma li mappiamo lo stesso per
/// futureproof flash-attn 3 PCH-mode (vedi ROCm github issue #2891).
#[inline]
pub fn from_hiprtc(rc: hiprtcResult) -> nvrtcResult {
    match rc {
        HIPRTC_SUCCESS => NVRTC_SUCCESS,
        HIPRTC_ERROR_OUT_OF_MEMORY => NVRTC_ERROR_OUT_OF_MEMORY,
        HIPRTC_ERROR_PROGRAM_CREATION_FAILURE => NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
        HIPRTC_ERROR_INVALID_INPUT => NVRTC_ERROR_INVALID_INPUT,
        HIPRTC_ERROR_INVALID_PROGRAM => NVRTC_ERROR_INVALID_PROGRAM,
        HIPRTC_ERROR_INVALID_OPTION => NVRTC_ERROR_INVALID_OPTION,
        HIPRTC_ERROR_COMPILATION => NVRTC_ERROR_COMPILATION,
        HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE => NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
        HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION => {
            NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
        }
        HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION => {
            NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
        }
        HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID => NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
        HIPRTC_ERROR_INTERNAL_ERROR => NVRTC_ERROR_INTERNAL_ERROR,
        HIPRTC_ERROR_LINKING => NVRTC_ERROR_COMPILATION, // vedi nota mapping
        // Codice ignoto da hiprtc — capita con versioni ROCm > 5.7 caricate
        // via Marconi pack. Trattiamo come internal error invece di propagare
        // ID intero non documentato.
        _ => NVRTC_ERROR_INTERNAL_ERROR,
    }
}

// ============================================================================
//  nvrtcGetErrorString — stringhe statiche zero-alloc
// ============================================================================

/// Restituisce stringa C statica per `nvrtcResult`. Contratto NVIDIA: puntatore
/// resta valido per la vita del processo, mai NULL.
///
/// Implementazione: literal `b"...\0"` cast a `*const c_char` — vita statica
/// senza allocazione, thread-safe per costruzione.
pub fn error_string(rc: nvrtcResult) -> *const c_char {
    let s: &'static [u8] = match rc {
        NVRTC_SUCCESS => b"NVRTC_SUCCESS\0",
        NVRTC_ERROR_OUT_OF_MEMORY => b"NVRTC_ERROR_OUT_OF_MEMORY\0",
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE => b"NVRTC_ERROR_PROGRAM_CREATION_FAILURE\0",
        NVRTC_ERROR_INVALID_INPUT => b"NVRTC_ERROR_INVALID_INPUT\0",
        NVRTC_ERROR_INVALID_PROGRAM => b"NVRTC_ERROR_INVALID_PROGRAM\0",
        NVRTC_ERROR_INVALID_OPTION => b"NVRTC_ERROR_INVALID_OPTION\0",
        NVRTC_ERROR_COMPILATION => b"NVRTC_ERROR_COMPILATION\0",
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE => b"NVRTC_ERROR_BUILTIN_OPERATION_FAILURE\0",
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION => {
            b"NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION\0"
        }
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION => {
            b"NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION\0"
        }
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID => b"NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID\0",
        NVRTC_ERROR_INTERNAL_ERROR => b"NVRTC_ERROR_INTERNAL_ERROR\0",
        NVRTC_ERROR_TIME_FILE_WRITE_FAILED => b"NVRTC_ERROR_TIME_FILE_WRITE_FAILED\0",
        NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED => b"NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED\0",
        NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED => b"NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED\0",
        NVRTC_ERROR_PCH_CREATE => b"NVRTC_ERROR_PCH_CREATE\0",
        NVRTC_ERROR_CANCELLED => b"NVRTC_ERROR_CANCELLED\0",
        _ => b"NVRTC_ERROR_UNKNOWN\0",
    };
    s.as_ptr() as *const c_char
}

// ============================================================================
//  Per-thread last error + diagnostic log
// ============================================================================

thread_local! {
    /// Ultimo errore del thread corrente. Usato dai workers per arricchire
    /// `nvrtcGetProgramLog` quando il programma è già stato distrutto.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Salva ultimo errore (thread-local). Senza allocazione fissa il messaggio
/// rilasciando il precedente.
pub fn set_last_error<S: Into<String>>(msg: S) {
    let s = msg.into();
    debug_log(&format!("set_last_error: {}", s));
    LAST_ERROR.with(|c| *c.borrow_mut() = Some(s));
}

/// Recupera (e svuota) ultimo errore.
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|c| c.borrow_mut().take())
}

// ============================================================================
//  Diagnostic log buffer (per-programma, owned dal modulo program)
// ============================================================================

/// Buffer log diagnostico riusato da `nvrtcGetProgramLog`. Owned dal program
/// handle nel modulo `program.rs` (worker 3/5). Esposto qui perché il mapper
/// errori vuole iniettare righe `[ZLUDA]` su mapping non triviali (es. linker).
#[derive(Default)]
pub struct DiagLog {
    pub(crate) buf: Vec<u8>, // bytes UTF-8, terminato \0 al getProgramLog
}

impl DiagLog {
    pub fn new() -> Self {
        Self { buf: Vec::with_capacity(256) }
    }

    /// Aggiunge una riga. Garantisce `\n` finale ma NON `\0`.
    pub fn push_line(&mut self, line: &str) {
        self.buf.extend_from_slice(line.as_bytes());
        if !line.ends_with('\n') {
            self.buf.push(b'\n');
        }
    }

    /// Aggiunge riga marcata `[ZLUDA]` — semantica diagnostica nostra
    /// (mai confondere con output clang). PyTorch fa grep `[ZLUDA]` in inductor
    /// debug mode.
    pub fn push_zluda(&mut self, line: &str) {
        self.push_line(&format!("[ZLUDA] {}", line));
    }

    /// Lunghezza richiesta per `nvrtcGetProgramLogSize`: bytes + 1 (terminatore).
    /// NVIDIA include `\0` nella size.
    pub fn size_with_nul(&self) -> usize {
        self.buf.len() + 1
    }

    /// Copia contenuto in buffer caller-provided, garantendo `\0` finale.
    /// Caller responsable di allocare almeno `size_with_nul()` byte.
    ///
    /// # Safety
    /// `dst` deve puntare a almeno `self.size_with_nul()` byte scrivibili.
    pub unsafe fn copy_to(&self, dst: *mut c_char) {
        if dst.is_null() {
            return;
        }
        std::ptr::copy_nonoverlapping(self.buf.as_ptr() as *const c_char, dst, self.buf.len());
        *dst.add(self.buf.len()) = 0;
    }
}

// ============================================================================
//  Debug trace (env-gated)
// ============================================================================

static DEBUG_INIT: OnceLock<bool> = OnceLock::new();
static DEBUG_ON: AtomicBool = AtomicBool::new(false);

fn debug_enabled() -> bool {
    *DEBUG_INIT.get_or_init(|| {
        let on = std::env::var("ZLUDA_NVRTC_DEBUG")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        DEBUG_ON.store(on, Ordering::Relaxed);
        on
    })
}

/// Traccia diagnostica su `%TEMP%\zluda_nvrtc_<pid>.log`. No-op se env var
/// non settata. **Mai panica anche se file write fallisce** (regola FFI:
/// niente eccezioni fuori dal boundary).
pub fn debug_log(msg: &str) {
    if !debug_enabled() {
        return;
    }
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("zluda_nvrtc_{}.log", pid));
    // Ignora ogni errore di IO — il debug è best-effort.
    let _ = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| writeln!(f, "{}", msg));
}

// ============================================================================
//  Macros per workers 1/2/3
// ============================================================================

/// Macro di entry-point. **Obbligatoria** per ogni `#[no_mangle] extern "C"`
/// del crate. Avvolge il corpo in `catch_unwind` e in caso di panic ritorna
/// `NVRTC_ERROR_INTERNAL_ERROR` + scrive trace.
///
/// ## Uso (worker 2/5 mapping ~40 API):
///
/// ```ignore
/// use crate::error::nvrtc_entry;
///
/// #[no_mangle]
/// pub unsafe extern "C" fn nvrtcCompileProgram(
///     prog: nvrtcProgram,
///     num_options: i32,
///     options: *const *const c_char,
/// ) -> nvrtcResult {
///     nvrtc_entry!("nvrtcCompileProgram", {
///         // body — può ritornare Result<(), nvrtcResult>
///         //       oppure nvrtcResult diretto via `?` con try_hiprtc!
///         let rc = hiprtcCompileProgram(prog as _, num_options, options);
///         crate::error::from_hiprtc(rc)
///     })
/// }
/// ```
#[macro_export]
macro_rules! nvrtc_entry {
    ($name:expr, $body:block) => {{
        $crate::error::debug_log(&format!(">> {}", $name));
        let res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(move || $body));
        match res {
            Ok(rc) => {
                $crate::error::debug_log(&format!("<< {} rc={}", $name, rc));
                rc
            }
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "panic non-stringa".to_string()
                };
                $crate::error::set_last_error(format!("panic in {}: {}", $name, msg));
                $crate::error::debug_log(&format!("!! {} PANIC: {}", $name, msg));
                $crate::error::NVRTC_ERROR_INTERNAL_ERROR
            }
        }
    }};
}

/// Macro `try_hiprtc!` — converte status hiprtc, esce con `return` se non SUCCESS.
/// **Da usare dentro `nvrtc_entry!`** per propagare errori senza scrivere
/// 3 righe match per ogni call site.
///
/// ## Uso:
///
/// ```ignore
/// nvrtc_entry!("nvrtcGetCUBIN", {
///     try_hiprtc!(hiprtcGetCodeSize(prog as _, size));
///     try_hiprtc!(hiprtcGetCode(prog as _, code));
///     NVRTC_SUCCESS
/// })
/// ```
#[macro_export]
macro_rules! try_hiprtc {
    ($call:expr) => {{
        let rc = $call;
        if rc != $crate::error::HIPRTC_SUCCESS {
            let mapped = $crate::error::from_hiprtc(rc);
            $crate::error::debug_log(&format!(
                "try_hiprtc fail: hip={} -> nvrtc={}",
                rc, mapped
            ));
            return mapped;
        }
    }};
}

// Re-export AssertUnwindSafe usato dalla macro per i workers che non l'hanno
// in scope.
#[doc(hidden)]
pub use std::panic::{catch_unwind as __catch_unwind, AssertUnwindSafe as __AssertUnwindSafe};

// ============================================================================
//  Tests (cargo test -p zluda_nvrtc)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn mapping_identity_0_to_11() {
        for code in 0u32..=11 {
            assert_eq!(from_hiprtc(code), code, "mapping diverge a code={}", code);
        }
    }

    #[test]
    fn mapping_linking_to_compilation() {
        assert_eq!(from_hiprtc(HIPRTC_ERROR_LINKING), NVRTC_ERROR_COMPILATION);
    }

    #[test]
    fn mapping_unknown_to_internal() {
        assert_eq!(from_hiprtc(42), NVRTC_ERROR_INTERNAL_ERROR);
        assert_eq!(from_hiprtc(9999), NVRTC_ERROR_INTERNAL_ERROR);
    }

    #[test]
    fn error_string_never_null() {
        for code in 0u32..=20 {
            let p = error_string(code);
            assert!(!p.is_null(), "NULL string per code={}", code);
            unsafe {
                let s = CStr::from_ptr(p);
                assert!(!s.to_bytes().is_empty(), "stringa vuota per code={}", code);
            }
        }
        // Codice fuori range: deve cadere su NVRTC_ERROR_UNKNOWN
        let p = error_string(99999);
        unsafe {
            let s = CStr::from_ptr(p).to_str().unwrap();
            assert_eq!(s, "NVRTC_ERROR_UNKNOWN");
        }
    }

    #[test]
    fn diag_log_roundtrip() {
        let mut log = DiagLog::new();
        log.push_line("clang error: undefined `foo`");
        log.push_zluda("PTX 8.3 feature downgraded a 7.5");
        assert_eq!(log.size_with_nul(), log.buf.len() + 1);
        let mut buf = vec![0u8; log.size_with_nul()];
        unsafe {
            log.copy_to(buf.as_mut_ptr() as *mut c_char);
        }
        // Verifica terminatore \0
        assert_eq!(*buf.last().unwrap(), 0);
        let s = CStr::from_bytes_until_nul(&buf).unwrap();
        let text = s.to_str().unwrap();
        assert!(text.contains("undefined `foo`"));
        assert!(text.contains("[ZLUDA] PTX 8.3"));
    }

    #[test]
    fn last_error_thread_local() {
        set_last_error("test errore");
        let got = take_last_error();
        assert_eq!(got.as_deref(), Some("test errore"));
        // Dopo take è vuoto
        assert!(take_last_error().is_none());
    }
}
