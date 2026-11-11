// VEGA-PATCH 2026-06-30 GAP G8 worker 2/5 — zluda_nvjitlink cdylib
//
// nvJitLink_120_0.dll wrap CUDA 12 PTX/LTO link API.
// BUG: PyTorch Inductor cu12 fail su `extern "C" __device__` JIT link;
//      Triton cu12 backend AOT che chiama nvJitLink* dopo ptxas;
//      pipeline NVCC offline LTO che usa nvJitLink invece di cuLink legacy.
//
// Architettura:
//   - 12 export `nvJitLink*` ABI cdecl x64 (System V su Linux, MS C su Win)
//   - Backend: riusa cu_link.rs (hiprtc 5.7) per AddData/Complete; aggiunge
//     wrapper per AddFile + GetLinkedCubin/PTX + error/info log buffer-per-handle
//   - Handle opaco nvJitLinkHandle = *mut JitLinkEntry (registry interno)
//
// Pattern Marconi italiano sovrano: ROCm 5.7 ESCLUSIVO (regola 4-bis).
// hiprtc0507.dll caricato lazy via libloading + OnceLock (vedi impl.rs).
//
// Limitazioni note (documentate per chiamante CUDA 12):
//   - LTO-IR NVIDIA non e' AMDGCN LLVM bitcode -> AddData con LTO-IR ritorna
//     ERROR_NVVM_COMPILE (NVPTX -> AMDGCN cross-translate non implementata)
//   - PTX raw: hiprtc lo accetta nominalmente ma il backend AMD rifiuta a
//     runtime; ritorniamo ERROR_PTX_COMPILE. Chiamanti come Triton-cu12
//     dovrebbero ricevere errore preciso per fallback CPU/Vulkan/altro path.
//   - CUBIN NVIDIA: ELF SM_xx -> AMDGCN cross-translate impossibile per
//     architettura; ritorniamo ERROR_UNRECOGNIZED_INPUT.
//   - Path felice: AddData con HIPRTC_JIT_INPUT_LLVM_BITCODE (100) AMDGCN
//     pre-compilato + Complete + GetLinkedCubin -> AMDGCN ELF caricabile
//     da cuModuleLoadData (zluda module loader gia' gestisce ELF AMDGCN).

mod r#impl;

// VEGA-PATCH 2026-06-30 GAP G8 worker 4/5 — error handling.
// Modulo standalone: panic-safety wrapper (nvjitlink_entry! macro),
// hiprtc -> nvJitLink mapping esaustivo (from_hiprtc), RollingLog cap 64 KiB
// per error/info buffer per-handle, debug_log via GetEnvironmentVariableW
// raw kernel32 (LESSON #5 — mai std::env::var su Windows DLL).
pub mod error;

use std::ffi::{c_char, c_void};
use std::os::raw::c_uint;

// ---------- ABI mirror nvJitLink (subset CUDA 12.0+) ----------
// Riferimento: docs.nvidia.com/cuda/nvjitlink/

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct NvJitLinkHandle(pub *mut c_void);

unsafe impl Send for NvJitLinkHandle {}
unsafe impl Sync for NvJitLinkHandle {}

#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvJitLinkResult {
    NVJITLINK_SUCCESS = 0,
    NVJITLINK_ERROR_UNRECOGNIZED_OPTION = 1,
    NVJITLINK_ERROR_MISSING_ARCH = 2,
    NVJITLINK_ERROR_INVALID_INPUT = 3,
    NVJITLINK_ERROR_PTX_COMPILE = 4,
    NVJITLINK_ERROR_NVVM_COMPILE = 5,
    NVJITLINK_ERROR_INTERNAL = 6,
    NVJITLINK_ERROR_THREADPOOL = 7,
    NVJITLINK_ERROR_UNRECOGNIZED_INPUT = 8,
    NVJITLINK_ERROR_FINALIZE = 9,
}

#[repr(u32)]
#[allow(non_camel_case_types, dead_code)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NvJitLinkInputType {
    NVJITLINK_INPUT_NONE = 0,
    NVJITLINK_INPUT_CUBIN = 1,        // ELF SM_xx -> non traducibile
    NVJITLINK_INPUT_PTX = 2,          // testo NVPTX -> rifiuto runtime AMD
    NVJITLINK_INPUT_LTOIR = 3,        // LTO-IR NVIDIA -> non traducibile
    NVJITLINK_INPUT_FATBIN = 4,       // multi-arch ELF wrapper
    NVJITLINK_INPUT_OBJECT = 5,       // host .o con embedded device code
    NVJITLINK_INPUT_LIBRARY = 6,      // .a archive
    NVJITLINK_INPUT_INDEX = 7,        // index file (CUDA 12.3+)
    NVJITLINK_INPUT_ANY = 10,         // probe per estensione/header
}

// ---------- 12 export ABI ----------
// Tutti ritornano NvJitLinkResult come u32 plain. ABI: extern "C" su Win x64
// (MS calling convention) e Linux x64 (System V) — bindings cudart12+ usano
// sempre `extern "C"`. NO `extern "system"` (che mappa a stdcall su Win32 x86
// ma su x64 e' identico a C — comunque per chiarezza usiamo "C").

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkCreate(
    handle: *mut NvJitLinkHandle,
    num_options: u32,
    options: *const *const c_char,
) -> NvJitLinkResult {
    r#impl::create(handle, num_options, options)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkDestroy(handle: *mut NvJitLinkHandle) -> NvJitLinkResult {
    r#impl::destroy(handle)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkAddData(
    handle: NvJitLinkHandle,
    input_type: NvJitLinkInputType,
    data: *const c_void,
    size: usize,
    name: *const c_char,
) -> NvJitLinkResult {
    r#impl::add_data(handle, input_type, data, size, name)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkAddFile(
    handle: NvJitLinkHandle,
    input_type: NvJitLinkInputType,
    file_name: *const c_char,
) -> NvJitLinkResult {
    r#impl::add_file(handle, input_type, file_name)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkComplete(handle: NvJitLinkHandle) -> NvJitLinkResult {
    r#impl::complete(handle)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetErrorLogSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    r#impl::get_error_log_size(handle, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetErrorLog(
    handle: NvJitLinkHandle,
    log: *mut c_char,
) -> NvJitLinkResult {
    r#impl::get_error_log(handle, log)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetInfoLogSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    r#impl::get_info_log_size(handle, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetInfoLog(
    handle: NvJitLinkHandle,
    log: *mut c_char,
) -> NvJitLinkResult {
    r#impl::get_info_log(handle, log)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetLinkedCubinSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    r#impl::get_linked_cubin_size(handle, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetLinkedCubin(
    handle: NvJitLinkHandle,
    cubin: *mut c_void,
) -> NvJitLinkResult {
    r#impl::get_linked_cubin(handle, cubin)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetLinkedPtxSize(
    handle: NvJitLinkHandle,
    size: *mut usize,
) -> NvJitLinkResult {
    r#impl::get_linked_ptx_size(handle, size)
}

#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkGetLinkedPtx(
    handle: NvJitLinkHandle,
    ptx: *mut c_char,
) -> NvJitLinkResult {
    r#impl::get_linked_ptx(handle, ptx)
}

// Bonus: nvJitLinkVersion (CUDA 12.2+) per query check chiamante
#[cfg_attr(not(test), no_mangle)]
pub unsafe extern "C" fn nvJitLinkVersion(major: *mut c_uint, minor: *mut c_uint) -> NvJitLinkResult {
    if major.is_null() || minor.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    *major = 12;
    *minor = 0;
    NvJitLinkResult::NVJITLINK_SUCCESS
}


// ---------------------------------------------------------------------------
//  Diagnostica su file — SPENTA salvo richiesta esplicita.
//
//  Questi log nascono come strumento di indagine, ma erano incondizionati: ogni
//  chiamata apriva, scriveva e chiudeva un file su disco. Il 31 lug 2026
//  `zluda_call_trace.log` era arrivato a 64 GB e `nvcuda_trace.log` a 7,5 GB,
//  col disco al 91%. Il costo ricadeva su ogni singola chiamata CUDA.
//
//  Ora si accendono impostando ZLUDA_DIAG (e ZLUDA_DIAG_DIR per la cartella).
//  La variabile e' letta una volta sola: a diagnostica spenta il costo e' la
//  lettura di un bool.
//
//  Il tipo di ritorno resta `std::io::Result<File>` di proposito, cosi' i
//  chiamanti gia' scritti come `if let Ok(mut f) = ...` restano invariati.
// ---------------------------------------------------------------------------
#[allow(dead_code)]
pub(crate) fn diag(nome: &str) -> std::io::Result<std::fs::File> {
    static ATTIVA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ATTIVA.get_or_init(|| std::env::var_os("ZLUDA_DIAG").is_some()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "diagnostica su file spenta",
        ));
    }
    let dir = diag_dir();
    let _ = std::fs::create_dir_all(&dir);
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::path::Path::new(&dir).join(nome))
}

/// Cartella dei file di diagnostica: `ZLUDA_DIAG_DIR`, altrimenti `%TEMP%\zluda_diag`.
/// Si usa solo a diagnostica accesa (`ZLUDA_DIAG`).
#[allow(dead_code)]
pub(crate) fn diag_dir() -> std::path::PathBuf {
    match std::env::var_os("ZLUDA_DIAG_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => std::env::temp_dir().join("zluda_diag"),
    }
}
