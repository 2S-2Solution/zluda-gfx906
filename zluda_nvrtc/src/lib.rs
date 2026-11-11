//! zluda_nvrtc — wrapper NVIDIA nvrtc64_*.dll su hiprtc0507 ROCm 5.7 (Vega 20).
//!
//! GAP G1 sprint chiusura totale ZLUDA — 30 giu 2026, Vega ⭐, bottega italiana.
//!
//! Sblocca torch.compile + Triton + Inductor + flash-attn JIT che oggi fanno
//! LoadLibraryW("nvrtc64_112_0.dll") + GetProcAddress fallendo perché nvrtc
//! NVIDIA non è ridistribuibile e su Vega 20 non esiste.
//!
//! ## Architettura
//!
//! - **Loader dinamico**: hiprtc0507.dll caricata via LoadLibraryW alla prima
//!   chiamata utile (Marconi pack 5.7 lo deploya). Zero link-time dep → la
//!   cdylib si carica anche senza Marconi pack, ritornando errori graceful.
//! - **Core 11 API**: mapping 1:1 a hipRTC (Create/Destroy/Compile/GetCode/
//!   GetLog/Add+GetLowered NameExpression/Version/ErrorString). Tutti i
//!   wrapper sono dentro `nvrtc_entry!` per `catch_unwind` panic-safety
//!   (regola FFI: panic across `extern "C"` = STATUS_ACCESS_VIOLATION).
//! - **Stub graceful nvrtc-only** (~17 funzioni): CUBIN/NVVM/LTO-IR/OptiX-IR,
//!   PCH (12.x), arch query, JIT linker. Ritornano `NVRTC_ERROR_INVALID_OPTION`
//!   o `NVRTC_SUCCESS` con buffer azzerati — torch.compile/Triton fallbackano
//!   a path eager su questi codici senza crash.
//! - **Version spoof**: `nvrtcVersion(12, 1)` — match `cu121` wheel torch 2.4.
//!   hiprtcVersion ritorna (5, 7) che fa esplodere il check `assert major < 10`
//!   in `torch._inductor.codecache`.
//!
//! Vedi `error.rs` per panic-safety + diag log + mapping codici (modulo
//! standalone già completo, worker 4/5).

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_variables)]

pub mod error;
pub mod program;

// Re-export per ergonomia + smoke tests.
pub use error::{
    debug_log, error_string, from_hiprtc, set_last_error, take_last_error, DiagLog,
    HIPRTC_SUCCESS, NVRTC_ERROR_COMPILATION, NVRTC_ERROR_INTERNAL_ERROR, NVRTC_ERROR_INVALID_INPUT,
    NVRTC_ERROR_INVALID_OPTION, NVRTC_ERROR_INVALID_PROGRAM, NVRTC_ERROR_OUT_OF_MEMORY,
    NVRTC_SUCCESS,
};

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use std::sync::OnceLock;

use crate::error::{nvrtcResult, hiprtcResult};

// =========================================================================
//  ABI types — NVIDIA nvrtc.h
// =========================================================================

/// nvrtcProgram opaque (riusato 1:1 come hiprtcProgram — entrambi `void*`).
pub type nvrtcProgram = *mut c_void;

/// nvrtcLinker handle opaque — solo per stub GAP G8.
pub type nvrtcLinker_t = *mut c_void;

// =========================================================================
//  Dynamic loader hiprtc0507.dll (Marconi pack ROCm 5.7)
// =========================================================================

/// Tavola dei puntatori alle 11 funzioni hipRTC che usiamo. Caricata una
/// sola volta al primo accesso via `hiprtc()`.
struct HipRtc {
    create_program: unsafe extern "C" fn(
        *mut nvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> hiprtcResult,
    destroy_program: unsafe extern "C" fn(*mut nvrtcProgram) -> hiprtcResult,
    compile_program: unsafe extern "C" fn(
        nvrtcProgram,
        c_int,
        *const *const c_char,
    ) -> hiprtcResult,
    get_code: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> hiprtcResult,
    get_code_size: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> hiprtcResult,
    get_log: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> hiprtcResult,
    get_log_size: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> hiprtcResult,
    add_name_expr: unsafe extern "C" fn(nvrtcProgram, *const c_char) -> hiprtcResult,
    get_lowered_name: unsafe extern "C" fn(
        nvrtcProgram,
        *const c_char,
        *mut *const c_char,
    ) -> hiprtcResult,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> hiprtcResult,
    #[allow(dead_code)]
    get_error_string: unsafe extern "C" fn(hiprtcResult) -> *const c_char,
}

// Forziamo Send+Sync — i puntatori funzione sono globali, thread-safe.
unsafe impl Send for HipRtc {}
unsafe impl Sync for HipRtc {}

/// Cache globale. `None` = caricamento tentato e fallito (Marconi pack non in
/// PATH o GetProcAddress su simbolo mancante) → tutte le call ritornano
/// errore graceful invece di crashare.
static HIPRTC: OnceLock<Option<HipRtc>> = OnceLock::new();

#[cfg(windows)]
fn load_hiprtc() -> Option<HipRtc> {
    use core::mem::transmute;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    // Candidati in ordine. La 5.7 va PER PRIMA per la regola 4-bis del pack:
    // a runtime si sta sulla 5.7.
    //   1) hiprtc0507.dll  (ROCm 5.7 — funzionante su Vega 20)
    //   2) hiprtc07013.dll (Marconi pack 7.0.13) — solo se la 5.7 manca
    //   3) hiprtc0701.dll  (ROCm 7.1)            — solo se le altre mancano
    //   4) hiprtc.dll      (fallback generico ROCm bin in PATH)
    //
    // CORREZIONE 2 ago 2026 — qui c'era scritto che la 5.7 e' «l'unica hiprtc
    // che supporta ancora gfx906» e che quella di ROCm 7.x scrive all'indirizzo
    // 0 dentro hiprtcCreateProgram perche' gfx906 e' fuori dai dispositivi PAL
    // di ROCm 7.0. **E' FALSO, e mi ha fatto perdere giorni.**
    //
    // La 7.1 su gfx906 compila benissimo — a patto di caricarla dalla PROPRIA
    // cartella di installazione. Il crash all'indirizzo 0 non dipende dalla
    // versione ma dal percorso: caricata da C:\rocm_5_7_expansion_pack\bin
    // risolve le proprie dipendenze sui binari sbagliati accanto e muore.
    // Caricata da C:\Program Files\AMD\ROCm\7.1\bin restituisce rc=0. La 5.7 si
    // comporta esattamente allo stesso modo: dal pack muore, dalla sua cartella
    // funziona. Verificato con LoadLibraryExW + LOAD_WITH_ALTERED_SEARCH_PATH.
    //
    // LoadLibraryW per nome nudo, come si fa qui sotto, va bene finche' la
    // cartella giusta e' in PATH prima delle altre. Se un giorno smettesse di
    // funzionare, il sospetto numero uno e' l'ordine del PATH, non la versione.
    // Helper per costruire UTF-16 zero-terminated da literal ASCII.
    fn utf16z(s: &str) -> Vec<u16> {
        let mut v: Vec<u16> = s.encode_utf16().collect();
        v.push(0);
        v
    }
    let candidati = [
        ("hiprtc0507.dll",  "ROCm 5.7 — gfx906 supportata"),
        ("hiprtc07013.dll", "Marconi pack 7.0.13 — ATTENZIONE: gfx906 non supportata"),
        ("hiprtc0701.dll",  "ROCm 7.1 — ATTENZIONE: gfx906 non supportata"),
        ("hiprtc.dll",      "fallback generico"),
    ];

    let mut handle: *mut c_void = ptr::null_mut();
    for (nome, nota) in candidati.iter() {
        let w = utf16z(nome);
        let h = unsafe { LoadLibraryW(w.as_ptr()) };
        if !h.is_null() {
            handle = h;
            debug_log(&format!("zluda_nvrtc: {} caricata ({})", nome, nota));
            break;
        }
    }
    if handle.is_null() {
        debug_log("zluda_nvrtc: hiprtc DLL non trovata (Marconi pack mancante?)");
        return None;
    }

    unsafe fn sym(h: *mut c_void, name: &[u8]) -> *mut c_void {
        extern "system" {
            fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        }
        unsafe { GetProcAddress(h, name.as_ptr() as *const c_char) }
    }

    macro_rules! load_sym {
        ($h:expr, $name:literal) => {{
            // Nota: questa macro è invocata SOLO dentro un blocco `unsafe`
            // esterno (vedi sotto `unsafe { Some(HipRtc { ... }) }`).
            // Niente `unsafe {}` interno → niente warning `unnecessary unsafe`.
            let p = sym($h, concat!($name, "\0").as_bytes());
            if p.is_null() {
                debug_log(concat!("zluda_nvrtc: simbolo mancante in hiprtc: ", $name));
                return None;
            }
            p
        }};
    }

    unsafe {
        Some(HipRtc {
            create_program: transmute(load_sym!(handle, "hiprtcCreateProgram")),
            destroy_program: transmute(load_sym!(handle, "hiprtcDestroyProgram")),
            compile_program: transmute(load_sym!(handle, "hiprtcCompileProgram")),
            get_code: transmute(load_sym!(handle, "hiprtcGetCode")),
            get_code_size: transmute(load_sym!(handle, "hiprtcGetCodeSize")),
            get_log: transmute(load_sym!(handle, "hiprtcGetProgramLog")),
            get_log_size: transmute(load_sym!(handle, "hiprtcGetProgramLogSize")),
            add_name_expr: transmute(load_sym!(handle, "hiprtcAddNameExpression")),
            get_lowered_name: transmute(load_sym!(handle, "hiprtcGetLoweredName")),
            version: transmute(load_sym!(handle, "hiprtcVersion")),
            get_error_string: transmute(load_sym!(handle, "hiprtcGetErrorString")),
        })
    }
}

#[cfg(not(windows))]
fn load_hiprtc() -> Option<HipRtc> {
    // CI Linux/WSL: lasciamo None così la cdylib compila lo stesso e tutte
    // le funzioni ritornano graceful errors. Produzione Vega = solo Windows.
    None
}

#[inline]
fn hiprtc() -> Option<&'static HipRtc> {
    HIPRTC.get_or_init(load_hiprtc).as_ref()
}

/// Shortcut: se hipRTC non disponibile ritorna l'errore graceful indicato.
macro_rules! hiprtc_or {
    ($err:expr) => {
        match hiprtc() {
            Some(rt) => rt,
            None => return $err,
        }
    };
}

// =========================================================================
//  Adattamento del sorgente: spazio di esecuzione predefinito
// =========================================================================
//
// NVRTC compila in spazio *device*: una funzione senza qualificatore e'
// implicitamente `__device__`. hipRTC no — la tratta come `__host__`, e un
// `__global__` che la chiama viene rifiutato con
//     error: no matching function for call to 'X'
//     note: candidate function not viable: call to __host__ function
//           from __global__ function
//
// Il jiterator di PyTorch (erfc, lgamma, digamma, polygamma, erfinv, i0, ...)
// genera esattamente quella forma, contando sulla regola NVRTC:
//     template <typename T> T erfc_kernel(T a) { return erfc(a); }
//
// hipRTC 5.7 non ha un'opzione equivalente: `-default-device` viene accettato
// ma e' inerte, `--default-device` e' rifiutato (misurato il 31 lug 2026).
// Quindi si qualifica il sorgente qui, prima di consegnarlo a hipRTC.
//
// Si toccano solo le definizioni introdotte da `template <...>` che non sono
// gia' qualificate e che non introducono un tipo. Commenti e stringhe sono
// attraversati senza essere interpretati, cosi' un `template<...>` dentro un
// commento non produce un'iniezione.

/// Prologo anteposto a ogni sorgente consegnato a hipRTC.
///
/// `__HIPCC__` va tolto. Il chiamante e' PyTorch compilato per CUDA, e il
/// sorgente che genera si adatta al compilatore con:
///
/// ```text
/// #ifdef __HIPCC__
/// #define CUDA_OR_ROCM_NUM_THREADS 256   // block_work_size = 1024
/// #define MAX_DIMS 16
/// #else
/// #define CUDA_OR_ROCM_NUM_THREADS 128   // block_work_size =  512
/// #define MAX_DIMS 25
/// #endif
/// ```
///
/// hipRTC definisce `__HIPCC__`, quindi il kernel verrebbe compilato per 256
/// thread — ma a lanciarlo e' un runtime CUDA che usa 128 e dimensiona la
/// griglia su quel valore. Gli indici si sfasano di un fattore due e il kernel
/// scrive fuori dalla porzione che gli compete: misurato il 31 lug 2026 come
/// uscita tutta a zero su erfc, lgamma, digamma, gcd, lcm, sinc.
///
/// Il ramo CUDA usa `assert` per i cast non supportati: hipRTC non ha
/// `<cassert>`, quindi lo si rende inerte.
///
/// `#line 1` riallinea la numerazione, cosi' i messaggi del compilatore
/// continuano a puntare alle righe vere del sorgente del chiamante.
const PROLOGO: &[u8] = b"#ifndef assert\n#define assert(x) ((void)0)\n#endif\n#line 1\n";

/// Le due costanti del preambolo del jiterator che vanno riportate ai valori
/// che il chiamante usa davvero.
///
/// Il sorgente si adatta al compilatore con `#ifdef __HIPCC__`, e hipRTC lo
/// definisce: prenderebbe quindi 256 thread per blocco e 16 dimensioni, mentre
/// chi lancia il kernel e' PyTorch compilato per CUDA, che usa 128 e 25.
///
/// Il primo tentativo era `#undef __HIPCC__`, per far prendere il ramo CUDA.
/// Funziona per i tipi a 32 bit ma **rompe fp16**: nel ramo CUDA `at::Half` e'
/// convertito con asm PTX (`cvt.rn.f16.f32`), che su AMDGCN e' un
/// `invalid output constraint '=h' in asm`. Misurato il 1 ago 2026.
///
/// Si cambiano quindi solo le due costanti, lasciando `__HIPCC__` al suo posto:
/// i tipi restano sul ramo HIP, corretto, e le dimensioni tornano quelle attese.
const SOSTITUZIONI: &[(&[u8], &[u8])] = &[
    (
        b"#define CUDA_OR_ROCM_NUM_THREADS 256",
        b"#define CUDA_OR_ROCM_NUM_THREADS 128",
    ),
    (b"#define MAX_DIMS 16", b"#define MAX_DIMS 25"),
];

/// Applica le sostituzioni di cui sopra. Ognuna al massimo una volta: compaiono
/// una sola volta nel preambolo, e cercarle oltre sarebbe solo lavoro sprecato.
fn adatta_costanti(src: Vec<u8>) -> Vec<u8> {
    let mut testo = src;
    for (da, a) in SOSTITUZIONI {
        if let Some(pos) = trova(&testo, da) {
            let mut nuovo = Vec::with_capacity(testo.len() + 8);
            nuovo.extend_from_slice(&testo[..pos]);
            nuovo.extend_from_slice(a);
            nuovo.extend_from_slice(&testo[pos + da.len()..]);
            testo = nuovo;
        }
    }
    testo
}

fn trova(dove: &[u8], cosa: &[u8]) -> Option<usize> {
    if cosa.is_empty() || dove.len() < cosa.len() {
        return None;
    }
    (0..=dove.len() - cosa.len()).find(|&i| &dove[i..i + cosa.len()] == cosa)
}

#[inline]
fn e_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Dopo un `template` a `pos`, restituisce (fine della lista di parametri,
/// la definizione e' gia' qualificata). `None` se non e' una vera intestazione.
fn testa_template(src: &[u8], pos: usize) -> Option<(usize, bool)> {
    let n = src.len();
    let mut j = pos + "template".len();
    while j < n && (src[j] as char).is_whitespace() {
        j += 1;
    }
    if j >= n || src[j] != b'<' {
        return None;
    }
    // chiusura della lista di parametri, contando l'annidamento
    let mut prof = 0i32;
    while j < n {
        match src[j] {
            b'<' => prof += 1,
            b'>' => {
                prof -= 1;
                if prof == 0 {
                    j += 1;
                    break;
                }
            }
            _ => {}
        }
        j += 1;
    }
    if prof != 0 {
        return None; // lista non chiusa: non tocchiamo nulla
    }
    let fine = j;
    // parola successiva
    let mut k = j;
    while k < n && (src[k] as char).is_whitespace() {
        k += 1;
    }
    let inizio = k;
    while k < n && e_ident(src[k]) {
        k += 1;
    }
    let parola = &src[inizio..k];
    let gia_a_posto = matches!(
        parola,
        b"__device__" | b"__global__" | b"__host__" | b"struct" | b"class"
            | b"union" | b"using" | b"template"
    );
    Some((fine, gia_a_posto))
}

/// Antepone `__device__` alle definizioni di template che ne sono prive.
pub fn qualifica_device(src: &[u8]) -> Vec<u8> {
    let n = src.len();
    let mut out = Vec::with_capacity(n + 256);
    let mut i = 0usize;
    while i < n {
        let b = src[i];
        // commento di riga
        if b == b'/' && i + 1 < n && src[i + 1] == b'/' {
            while i < n && src[i] != b'\n' {
                out.push(src[i]);
                i += 1;
            }
            continue;
        }
        // commento di blocco
        if b == b'/' && i + 1 < n && src[i + 1] == b'*' {
            out.push(src[i]);
            out.push(src[i + 1]);
            i += 2;
            while i < n && !(src[i] == b'*' && i + 1 < n && src[i + 1] == b'/') {
                out.push(src[i]);
                i += 1;
            }
            continue;
        }
        // stringa o carattere
        if b == b'"' || b == b'\'' {
            out.push(b);
            i += 1;
            while i < n {
                if src[i] == b'\\' && i + 1 < n {
                    out.push(src[i]);
                    out.push(src[i + 1]);
                    i += 2;
                    continue;
                }
                let c = src[i];
                out.push(c);
                i += 1;
                if c == b {
                    break;
                }
            }
            continue;
        }
        // `template` come parola intera
        if b == b't'
            && src[i..].starts_with(b"template")
            && (i == 0 || !e_ident(src[i - 1]))
            && (i + 8 >= n || !e_ident(src[i + 8]))
        {
            if let Some((fine, gia_a_posto)) = testa_template(src, i) {
                out.extend_from_slice(&src[i..fine]);
                if !gia_a_posto {
                    out.extend_from_slice(b" __device__ ");
                }
                i = fine;
                continue;
            }
        }
        out.push(b);
        i += 1;
    }
    out
}

// =========================================================================
//  CORE API — mapping 1:1 hipRTC (11 funzioni con panic safety)
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn nvrtcCreateProgram(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    num_headers: c_int,
    headers: *const *const c_char,
    include_names: *const *const c_char,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcCreateProgram", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_INPUT;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_PROGRAM_CREATION_FAILURE);
        // Spazio di esecuzione predefinito: vedi `qualifica_device`. hipRTC
        // copia il sorgente al suo interno, quindi il buffer temporaneo puo'
        // vivere solo per la durata della chiamata.
        let adattato: Option<Vec<u8>> = if src.is_null() {
            None
        } else {
            let originale = core::ffi::CStr::from_ptr(src).to_bytes();
            let corpo = adatta_costanti(qualifica_device(originale));
            let mut v = Vec::with_capacity(PROLOGO.len() + corpo.len() + 1);
            v.extend_from_slice(PROLOGO);
            v.extend_from_slice(&corpo);
            v.push(0);
            debug_log(&format!(
                "zluda_nvrtc: sorgente adattato ({} -> {} byte)",
                originale.len(),
                v.len() - 1
            ));
            Some(v)
        };
        let src_eff = match &adattato {
            Some(v) => v.as_ptr() as *const c_char,
            None => src,
        };
        let rc = (rt.create_program)(prog, src_eff, name, num_headers, headers, include_names);
        from_hiprtc(rc)
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult {
    nvrtc_entry!("nvrtcDestroyProgram", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INVALID_PROGRAM);
        from_hiprtc((rt.destroy_program)(prog))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcCompileProgram(
    prog: nvrtcProgram,
    num_options: c_int,
    options: *const *const c_char,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcCompileProgram", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_COMPILATION);
        // Le option NVIDIA (--gpu-architecture=compute_80) sono incompatibili
        // con hipRTC che vuole --offload-arch=gfx906. Triton/Inductor passano
        // option NVIDIA. Strategia: passiamo lista vuota → hipRTC usa default
        // device discovery (HIP_VISIBLE_DEVICES → gfx906). Caller poi legge
        // GetCode via nvrtcGetPTX (sotto), Triton vede `code-object hsaco`
        // invece di PTX e fallbacka a path eager. Win-win.
        let _ = (num_options, options);
        from_hiprtc((rt.compile_program)(prog, 0, ptr::null()))
    })
}

/// nvrtcGetPTX → hiprtcGetCode (HIP produce hsaco/code-object; lo nominiamo
/// PTX per il caller, che lo scarterà comunque al primo magic-byte check).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetPTX(prog: nvrtcProgram, buf: *mut c_char) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetPTX", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_code)(prog, buf))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetPTXSize(
    prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetPTXSize", {
        if prog.is_null() || size.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_code_size)(prog, size))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetProgramLog(
    prog: nvrtcProgram,
    buf: *mut c_char,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetProgramLog", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_log)(prog, buf))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetProgramLogSize(
    prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetProgramLogSize", {
        if prog.is_null() || size.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_log_size)(prog, size))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcAddNameExpression(
    prog: nvrtcProgram,
    name_expression: *const c_char,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcAddNameExpression", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.add_name_expr)(prog, name_expression))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetLoweredName(
    prog: nvrtcProgram,
    name_expression: *const c_char,
    lowered_name: *mut *const c_char,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetLoweredName", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_lowered_name)(prog, name_expression, lowered_name))
    })
}

/// Spoofa version 12.1 (match `cu121` wheel torch 2.4). hiprtc ritorna (5, 7)
/// e PyTorch fallisce `assert major >= 11`. Worker 5/5 ha decisione su 12.1.
#[no_mangle]
pub unsafe extern "C" fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult {
    nvrtc_entry!("nvrtcVersion", {
        if major.is_null() || minor.is_null() {
            return NVRTC_ERROR_INVALID_INPUT;
        }
        // Chiamiamo comunque hiprtc per liveness check (NON propaghiamo valori).
        let _ = match hiprtc() {
            Some(rt) => {
                let mut hm: c_int = 0;
                let mut hn: c_int = 0;
                (rt.version)(&mut hm, &mut hn)
            }
            None => HIPRTC_SUCCESS, // graceful: rispondiamo spoof anche se DLL miss
        };
        *major = 12;
        *minor = 1;
        NVRTC_SUCCESS
    })
}

/// `nvrtcGetErrorString` infallibile, no `nvrtc_entry!` — delegato a
/// `error::error_string` (tabella statica zero-alloc).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char {
    error_string(result)
}

// =========================================================================
//  nvrtc-only API — stub graceful. Caller (Triton/Inductor/torch.compile)
//  fanno fallback su path eager su questi codici, niente crash.
// =========================================================================

/// CUBIN (compiled binary). Con CUDA >= 11.1 il jiterator di PyTorch compila
/// "to SASS" e chiede il CUBIN, non il PTX: se qui si risponde con uno stub,
/// l'intero percorso muore con NVRTC_ERROR_INVALID_OPTION (misurato su erfc /
/// lgamma / digamma il 31 lug 2026).
///
/// Su questa piattaforma il code object e' lo stesso in entrambi i casi:
/// l'hsaco prodotto da hipRTC, che `cuModuleLoadData` sa gia' caricare.
/// Quindi CUBIN e PTX puntano tutti e due a `hiprtcGetCode`.
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetCUBIN(prog: nvrtcProgram, buf: *mut c_char) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetCUBIN", {
        if prog.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_code)(prog, buf))
    })
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetCUBINSize(
    prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    nvrtc_entry!("nvrtcGetCUBINSize", {
        if prog.is_null() || size.is_null() {
            return NVRTC_ERROR_INVALID_PROGRAM;
        }
        let rt = hiprtc_or!(NVRTC_ERROR_INTERNAL_ERROR);
        from_hiprtc((rt.get_code_size)(prog, size))
    })
}

/// NVVM IR (deprecated NVIDIA, TF GPU graph optimizer lo prova).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetNVVM(_prog: nvrtcProgram, _buf: *mut c_char) -> nvrtcResult {
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetNVVMSize(
    _prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    if !size.is_null() {
        *size = 0;
    }
    NVRTC_ERROR_INVALID_OPTION
}

/// LTO-IR (nvrtc 11.4+, torch 2.5 graph fusion).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetLTOIR(_prog: nvrtcProgram, _buf: *mut c_char) -> nvrtcResult {
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetLTOIRSize(
    _prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    if !size.is_null() {
        *size = 0;
    }
    NVRTC_ERROR_INVALID_OPTION
}

/// OptiX-IR (RT-core ray tracing, mai usato da training).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetOptiXIR(_prog: nvrtcProgram, _buf: *mut c_char) -> nvrtcResult {
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetOptiXIRSize(
    _prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    if !size.is_null() {
        *size = 0;
    }
    NVRTC_ERROR_INVALID_OPTION
}

/// Triton 3.x interroga le arch supportate per scegliere il path. Spoofa 1
/// arch = sm_70 (gfx906 ≈ compute 7.0).
#[no_mangle]
pub unsafe extern "C" fn nvrtcGetNumSupportedArchs(num_archs: *mut c_int) -> nvrtcResult {
    if num_archs.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { *num_archs = 1 };
    NVRTC_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetSupportedArchs(supported_archs: *mut c_int) -> nvrtcResult {
    if supported_archs.is_null() {
        return NVRTC_ERROR_INVALID_INPUT;
    }
    unsafe { *supported_archs = 70 };
    NVRTC_SUCCESS
}

// --- PCH (nvrtc 12.x) — torch.compile non lo usa, triton 3.2 sì ---

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetPCHCreateStatus(_prog: nvrtcProgram) -> nvrtcResult {
    // 13 = NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED — Triton fallbacka a no-PCH.
    13
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetPCHHeapSize(size_ret: *mut usize) -> nvrtcResult {
    if !size_ret.is_null() {
        unsafe { *size_ret = 0 };
    }
    NVRTC_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcSetPCHHeapSize(_size: usize) -> nvrtcResult {
    NVRTC_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcGetPCHHeapSizeRequired(
    _prog: nvrtcProgram,
    size: *mut usize,
) -> nvrtcResult {
    if !size.is_null() {
        unsafe { *size = 0 };
    }
    NVRTC_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcSetFlowCallback(
    _prog: nvrtcProgram,
    _callback: *mut c_void,
    _payload: *mut c_void,
) -> nvrtcResult {
    NVRTC_SUCCESS
}

// --- JIT Linker (nvrtc 11.x — su 12.x si separa in nvJitLink.dll = GAP G8) ---

#[no_mangle]
pub unsafe extern "C" fn nvrtcLinkCreate(
    _num_options: c_int,
    _options: *mut c_int,
    _option_values: *mut *mut c_void,
    linker: *mut nvrtcLinker_t,
) -> nvrtcResult {
    if !linker.is_null() {
        unsafe { *linker = ptr::null_mut() };
    }
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcLinkAddData(
    _linker: nvrtcLinker_t,
    _input_type: c_int,
    _data: *mut c_void,
    _size: usize,
    _name: *const c_char,
    _num_options: c_int,
    _options: *mut c_int,
    _option_values: *mut *mut c_void,
) -> nvrtcResult {
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcLinkAddFile(
    _linker: nvrtcLinker_t,
    _input_type: c_int,
    _path: *const c_char,
    _num_options: c_int,
    _options: *mut c_int,
    _option_values: *mut *mut c_void,
) -> nvrtcResult {
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcLinkComplete(
    _linker: nvrtcLinker_t,
    out: *mut *mut c_void,
    out_size: *mut usize,
) -> nvrtcResult {
    if !out.is_null() {
        unsafe { *out = ptr::null_mut() };
    }
    if !out_size.is_null() {
        unsafe { *out_size = 0 };
    }
    NVRTC_ERROR_INVALID_OPTION
}

#[no_mangle]
pub unsafe extern "C" fn nvrtcLinkDestroy(_linker: nvrtcLinker_t) -> nvrtcResult {
    NVRTC_SUCCESS
}

// =========================================================================
//  Re-export costanti non in `error::` per ergonomia caller esterni
// =========================================================================

pub const NVRTC_ERROR_PROGRAM_CREATION_FAILURE: nvrtcResult = 2;

// =========================================================================
//  Smoke tests
// =========================================================================

#[cfg(test)]
mod smoke {
    use super::*;

    #[test]
    fn reexports_compile() {
        let _: nvrtcResult = NVRTC_SUCCESS;
        let _: nvrtcResult = from_hiprtc(0);
        let _ = error_string(0);
        let _ = DiagLog::new();
    }

    #[test]
    fn version_spoof_12_1() {
        let mut major: c_int = 0;
        let mut minor: c_int = 0;
        let rc = unsafe { nvrtcVersion(&mut major, &mut minor) };
        assert_eq!(rc, NVRTC_SUCCESS);
        assert_eq!(major, 12);
        assert_eq!(minor, 1);
    }

    #[test]
    fn arch_query_sm70() {
        let mut n: c_int = 0;
        assert_eq!(unsafe { nvrtcGetNumSupportedArchs(&mut n) }, NVRTC_SUCCESS);
        assert_eq!(n, 1);
        let mut arch: c_int = 0;
        assert_eq!(unsafe { nvrtcGetSupportedArchs(&mut arch) }, NVRTC_SUCCESS);
        assert_eq!(arch, 70);
    }

    #[test]
    fn cubin_su_programma_nullo() {
        // CUBIN non e' piu' uno stub: e' il code object, come il PTX.
        // Su programma nullo deve rispondere INVALID_PROGRAM, non toccare
        // il buffer del chiamante.
        let mut size: usize = 999;
        let rc = unsafe { nvrtcGetCUBINSize(ptr::null_mut(), &mut size) };
        assert_eq!(rc, NVRTC_ERROR_INVALID_PROGRAM);
        assert_eq!(unsafe { nvrtcGetCUBIN(ptr::null_mut(), ptr::null_mut()) },
                   NVRTC_ERROR_INVALID_PROGRAM);
    }

    #[test]
    fn null_input_invalid_input() {
        let rc = unsafe {
            nvrtcCreateProgram(ptr::null_mut(), ptr::null(), ptr::null(), 0, ptr::null(), ptr::null())
        };
        assert_eq!(rc, NVRTC_ERROR_INVALID_INPUT);
    }

    #[test]
    fn qualifica_solo_dove_serve() {
        // il caso del jiterator: va qualificato
        let s = b"template <typename T> T erfc_kernel(T a) { return erfc(a); }";
        let r = String::from_utf8(qualifica_device(s)).unwrap();
        let compatta: String = r.split_whitespace().collect::<Vec<_>>().join(" ");
        assert_eq!(
            compatta,
            "template <typename T> __device__ T erfc_kernel(T a) { return erfc(a); }"
        );

        // gia' qualificata: invariata
        for s in [
            &b"template <typename T> __device__ T f(T a) { return a; }"[..],
            &b"template <typename T> struct S { T v; };"[..],
            &b"template <int N, typename T> class C {};"[..],
            &b"template <typename T> using P = T*;"[..],
        ] {
            assert_eq!(qualifica_device(s), s.to_vec(), "modificato: {}",
                       String::from_utf8_lossy(s));
        }

        // dentro un commento non si tocca
        let s = b"// template <typename T> T f(T a);\nint x;";
        assert_eq!(qualifica_device(s), s.to_vec());

        // dentro una stringa non si tocca
        let s = b"const char* s = \"template <typename T> T f(T);\";";
        assert_eq!(qualifica_device(s), s.to_vec());

        // parola che contiene "template" ma non lo e'
        let s = b"int my_template <= 3;";
        assert_eq!(qualifica_device(s), s.to_vec());

        // piu' definizioni sulla stessa riga (la forma di i0)
        let s = b"template<typename T> T a(T x){return x;} template<typename T> T b(T x){return x;}";
        let r = String::from_utf8(qualifica_device(s)).unwrap();
        assert_eq!(r.matches("__device__").count(), 2);
    }

    #[test]
    fn null_prog_invalid_program() {
        assert_eq!(unsafe { nvrtcDestroyProgram(ptr::null_mut()) }, NVRTC_ERROR_INVALID_PROGRAM);
        // prog nullo e' un programma non valido: il controllo precede la
        // compilazione, quindi il codice e' INVALID_PROGRAM, non COMPILATION.
        assert_eq!(unsafe { nvrtcCompileProgram(ptr::null_mut(), 0, ptr::null()) }, NVRTC_ERROR_INVALID_PROGRAM);
    }
}
