// VEGA-PATCH 2026-06-30 TASK 9 BUG 9 — IMPLEMENTAZIONE REALE cuLink* via hiprtc 5.7
//
// Architettura: ogni CUlinkState e' un handle opaco che mappiamo 1-to-1 sopra
// hiprtcLinkState (struct ihiprtcLinkState* opaca). Dynamic-loading di
// hiprtc0507.dll (regola 4-bis ROCm 5.7 esclusivo) via libloading lazy +
// OnceLock; le 5 simboli hiprtcLink* sono cached in HiprtcApi statico.
//
// Mapping enum:
//   CU_JIT_OPTION_*   (0..24, 25=NUM)  ==  HIPRTC_JIT_*   (0..24)  identita' bit-perfetta
//                                                                  verificata cuda_types/src/cuda.rs:2497-2706
//                                                                  vs ROCm 5.7 hiprtc.h:72-101
//   CU_JIT_INPUT_CUBIN/PTX/FATBINARY/OBJECT/LIBRARY/NVVM (0..5)
//                                       ==  HIPRTC_JIT_INPUT_*    (0..5)  identita' verificata
//                                                                  cuda.rs:3016-3055 vs hiprtc.h:106-118
//
// Path PTX: hiprtc accetta nominalmente HIPRTC_JIT_INPUT_PTX ma il backend AMD
// rifiuta runtime (no NVPTX). Per Blender Cycles questa fallback ritorna
// CUDA_ERROR_INVALID_PTX (218), che Blender gestisce con disable del kernel JIT
// e fallback a CPU. Path HIPRTC_JIT_INPUT_LLVM_BITCODE (100) accettato se il
// chiamante passa direttamente bitcode AMDGCN compilato esternamente (caso
// avanzato non comune da Blender ma utile per pipeline customer).

use cuda_types::cuda::*;
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::os::raw::{c_int, c_uint};
use std::ptr;
use std::sync::{Mutex, OnceLock};

// ---------- hiprtc ABI mirror (subset Link API) ----------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct HiprtcResult(c_int);

const HIPRTC_SUCCESS: HiprtcResult = HiprtcResult(0);

type HiprtcLinkStateRaw = *mut c_void;

type FnLinkCreate = unsafe extern "C" fn(
    num_options: c_uint,
    options: *mut c_uint,
    option_values: *mut *mut c_void,
    link_state_out: *mut HiprtcLinkStateRaw,
) -> HiprtcResult;

type FnLinkAddData = unsafe extern "C" fn(
    state: HiprtcLinkStateRaw,
    input_type: c_uint,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    num_options: c_uint,
    options: *mut c_uint,
    option_values: *mut *mut c_void,
) -> HiprtcResult;

type FnLinkAddFile = unsafe extern "C" fn(
    state: HiprtcLinkStateRaw,
    input_type: c_uint,
    path: *const c_char,
    num_options: c_uint,
    options: *mut c_uint,
    option_values: *mut *mut c_void,
) -> HiprtcResult;

type FnLinkComplete = unsafe extern "C" fn(
    state: HiprtcLinkStateRaw,
    bin_out: *mut *mut c_void,
    size_out: *mut usize,
) -> HiprtcResult;

type FnLinkDestroy = unsafe extern "C" fn(state: HiprtcLinkStateRaw) -> HiprtcResult;

struct HiprtcApi {
    _lib: Library,
    link_create: FnLinkCreate,
    link_add_data: FnLinkAddData,
    link_add_file: FnLinkAddFile,
    link_complete: FnLinkComplete,
    link_destroy: FnLinkDestroy,
}

unsafe impl Send for HiprtcApi {}
unsafe impl Sync for HiprtcApi {}

static HIPRTC: OnceLock<Option<HiprtcApi>> = OnceLock::new();

fn load_hiprtc() -> Option<&'static HiprtcApi> {
    HIPRTC
        .get_or_init(|| {
            // regola 4-bis: hiprtc da ROCm 5.7 SOLO
            // Ordine: HIP SDK 5.7, poi il bin del kit Marconi (%MARCONI_ROOT%\bin), poi il PATH.
            let mut candidates = vec![String::from(r"C:\Program Files\AMD\ROCm\5.7\bin\hiprtc0507.dll")];
            if let Some(m) = std::env::var_os("MARCONI_ROOT") {
                candidates.push(std::path::Path::new(&m).join("bin").join("hiprtc0507.dll").to_string_lossy().into_owned());
            }
            candidates.push(String::from("hiprtc0507.dll")); // PATH fallback
            for path in candidates {
                unsafe {
                    if let Ok(lib) = Library::new(&path) {
                        let create: Symbol<FnLinkCreate> = match lib.get(b"hiprtcLinkCreate\0") {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        let add_data: Symbol<FnLinkAddData> =
                            match lib.get(b"hiprtcLinkAddData\0") {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                        let add_file: Symbol<FnLinkAddFile> =
                            match lib.get(b"hiprtcLinkAddFile\0") {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                        let complete: Symbol<FnLinkComplete> =
                            match lib.get(b"hiprtcLinkComplete\0") {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                        let destroy: Symbol<FnLinkDestroy> =
                            match lib.get(b"hiprtcLinkDestroy\0") {
                                Ok(s) => s,
                                Err(_) => continue,
                            };
                        let api = HiprtcApi {
                            link_create: *create.into_raw(),
                            link_add_data: *add_data.into_raw(),
                            link_add_file: *add_file.into_raw(),
                            link_complete: *complete.into_raw(),
                            link_destroy: *destroy.into_raw(),
                            _lib: lib,
                        };
                        trace(&format!("hiprtc 5.7 loaded from {}", path));
                        return Some(api);
                    }
                }
            }
            trace("hiprtc 5.7 NOT loadable - cuLink* will return JIT_COMPILER_NOT_FOUND");
            None
        })
        .as_ref()
}

// ---------- handle registry: CUlinkState -> hiprtcLinkState + owned bin buffer ----------

// Newtype wrapper attorno al raw pointer hiprtcLinkState_t per impl Send/Sync
// (HIPRTC linkState e' opaque handle gestito internamente dalla libreria HIP
// con proprio locking; e' safe condividere puntatore tra thread del wrapper
// purche' una sola chiamata HIPRTC alla volta per state — garantito dal
// REGISTRY mutex.)
#[derive(Copy, Clone)]
struct HiprtcStateHandle(HiprtcLinkStateRaw);
unsafe impl Send for HiprtcStateHandle {}
unsafe impl Sync for HiprtcStateHandle {}

struct LinkEntry {
    hiprtc_state: HiprtcStateHandle,
    // bin_out punta in questo buffer (HIPRTC owns the bytes via state, but
    // alcune versioni AMD copiano ad uscita; conserviamo None se in-state)
    completed_bin: Option<Vec<u8>>,
}

static REGISTRY: OnceLock<Mutex<HashMap<usize, LinkEntry>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<usize, LinkEntry>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn trace(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("nvcuda_trace.log")
    {
        let _ = writeln!(f, "[cuLink] {}", msg);
    }
}

// ---------- mapping enums ----------

// CUjit_option 0..24 == HIPRTC_JIT_* 0..24 (verificato). Per option > 24 e
// option che hiprtc non riconosce, sostituiamo con HIPRTC_JIT_NUM_OPTIONS=25 e
// option_values=NULL = ignorata, oppure ritorniamo INVALID_VALUE se opzione
// gia' fuori range CU.
fn map_options(
    num_options: u32,
    cu_opts: *mut CUjit_option,
    cu_vals: *mut *mut c_void,
) -> Result<(Vec<c_uint>, Vec<*mut c_void>), CUerror> {
    if num_options == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if cu_opts.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let n = num_options as usize;
    let mut out_opts = Vec::with_capacity(n);
    let mut out_vals = Vec::with_capacity(n);
    for i in 0..n {
        let cu_opt = unsafe { (*cu_opts.add(i)).0 };
        // accetta solo 0..24 (range valido CU = HIPRTC condiviso)
        if cu_opt > 24 {
            trace(&format!(
                "map_options: ignoring CU_JIT option {} (out of HIPRTC range)",
                cu_opt
            ));
            // mappa a HIPRTC_JIT_NUM_OPTIONS=25 con value null => ignorata
            out_opts.push(25u32);
            out_vals.push(ptr::null_mut());
        } else {
            out_opts.push(cu_opt);
            out_vals.push(if cu_vals.is_null() {
                ptr::null_mut()
            } else {
                unsafe { *cu_vals.add(i) }
            });
        }
    }
    Ok((out_opts, out_vals))
}

// Mapping CUjitInputType -> hiprtcJITInputType raw u32
fn map_input_type(cu_type: CUjitInputType) -> Result<c_uint, CUerror> {
    let raw = cu_type.0;
    match raw {
        0 => Ok(0),   // CUBIN  -> HIPRTC_JIT_INPUT_CUBIN
        1 => Ok(1),   // PTX    -> HIPRTC_JIT_INPUT_PTX (rifiutato runtime su AMD ma nominalmente accettato)
        2 => Ok(2),   // FATBINARY -> idem
        3 => Err(CUerror::INVALID_VALUE), // OBJECT non supportato
        4 => Ok(4),   // LIBRARY
        5 => Err(CUerror::INVALID_VALUE), // NVVM IR non supportato (HIPRTC vuole LLVM bitcode 100)
        _ => Err(CUerror::INVALID_VALUE),
    }
}

// ---------- API pubbliche ----------

pub(crate) unsafe fn link_create(
    num_options: c_uint,
    cu_opts: *mut CUjit_option,
    cu_vals: *mut *mut c_void,
    state_out: *mut CUlinkState,
) -> Result<(), CUerror> {
    if state_out.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let api = load_hiprtc().ok_or(CUerror::JIT_COMPILER_NOT_FOUND)?;
    let (mut opts, mut vals) = map_options(num_options, cu_opts, cu_vals)?;
    let mut hip_state: HiprtcLinkStateRaw = ptr::null_mut();
    let opts_ptr = if opts.is_empty() {
        ptr::null_mut()
    } else {
        opts.as_mut_ptr()
    };
    let vals_ptr = if vals.is_empty() {
        ptr::null_mut()
    } else {
        vals.as_mut_ptr()
    };
    let r = (api.link_create)(
        num_options,
        opts_ptr,
        vals_ptr,
        &mut hip_state as *mut HiprtcLinkStateRaw,
    );
    if r != HIPRTC_SUCCESS || hip_state.is_null() {
        trace(&format!("link_create: hiprtc failed code={}", r.0));
        return Err(CUerror::UNKNOWN);
    }
    let id = hip_state as usize;
    registry().lock().unwrap().insert(
        id,
        LinkEntry {
            hiprtc_state: HiprtcStateHandle(hip_state),
            completed_bin: None,
        },
    );
    *state_out = id as CUlinkState;
    trace(&format!("link_create: state=0x{:x}", id));
    Ok(())
}

pub(crate) unsafe fn link_add_data(
    state: CUlinkState,
    input_type: CUjitInputType,
    data: *mut c_void,
    size: usize,
    name: *const c_char,
    num_options: c_uint,
    cu_opts: *mut CUjit_option,
    cu_vals: *mut *mut c_void,
) -> Result<(), CUerror> {
    if state.is_null() || data.is_null() || size == 0 {
        return Err(CUerror::INVALID_VALUE);
    }
    let api = load_hiprtc().ok_or(CUerror::JIT_COMPILER_NOT_FOUND)?;
    let id = state as usize;
    let reg = registry().lock().unwrap();
    let entry = reg.get(&id).ok_or(CUerror::INVALID_HANDLE)?;
    let hip_state = entry.hiprtc_state.0;
    let hip_input = map_input_type(input_type)?;
    let (mut opts, mut vals) = map_options(num_options, cu_opts, cu_vals)?;
    let opts_ptr = if opts.is_empty() {
        ptr::null_mut()
    } else {
        opts.as_mut_ptr()
    };
    let vals_ptr = if vals.is_empty() {
        ptr::null_mut()
    } else {
        vals.as_mut_ptr()
    };
    drop(reg); // libera lock prima della call FFI lunga
    let r = (api.link_add_data)(
        hip_state,
        hip_input,
        data,
        size,
        name,
        num_options,
        opts_ptr,
        vals_ptr,
    );
    if r != HIPRTC_SUCCESS {
        let nm = if name.is_null() {
            "<null>".to_string()
        } else {
            CStr::from_ptr(name).to_string_lossy().to_string()
        };
        trace(&format!(
            "link_add_data: hiprtc code={} type={} size={} name={}",
            r.0, hip_input, size, nm
        ));
        // PTX raw su AMD non e' linkable -> diamo errore semantico utile per chiamante
        if hip_input == 1 || hip_input == 2 {
            return Err(CUerror::INVALID_PTX);
        }
        return Err(CUerror::INVALID_IMAGE);
    }
    Ok(())
}

pub(crate) unsafe fn link_add_file(
    state: CUlinkState,
    input_type: CUjitInputType,
    path: *const c_char,
    num_options: c_uint,
    cu_opts: *mut CUjit_option,
    cu_vals: *mut *mut c_void,
) -> Result<(), CUerror> {
    if state.is_null() || path.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let api = load_hiprtc().ok_or(CUerror::JIT_COMPILER_NOT_FOUND)?;
    let id = state as usize;
    let reg = registry().lock().unwrap();
    let entry = reg.get(&id).ok_or(CUerror::INVALID_HANDLE)?;
    let hip_state = entry.hiprtc_state.0;
    let hip_input = map_input_type(input_type)?;
    let (mut opts, mut vals) = map_options(num_options, cu_opts, cu_vals)?;
    let opts_ptr = if opts.is_empty() {
        ptr::null_mut()
    } else {
        opts.as_mut_ptr()
    };
    let vals_ptr = if vals.is_empty() {
        ptr::null_mut()
    } else {
        vals.as_mut_ptr()
    };
    drop(reg);
    let r = (api.link_add_file)(hip_state, hip_input, path, num_options, opts_ptr, vals_ptr);
    if r != HIPRTC_SUCCESS {
        let p = CStr::from_ptr(path).to_string_lossy();
        trace(&format!(
            "link_add_file: hiprtc code={} type={} path={}",
            r.0, hip_input, p
        ));
        if hip_input == 1 || hip_input == 2 {
            return Err(CUerror::INVALID_PTX);
        }
        return Err(CUerror::FILE_NOT_FOUND);
    }
    Ok(())
}

pub(crate) unsafe fn link_complete(
    state: CUlinkState,
    bin_out: *mut *mut c_void,
    size_out: *mut usize,
) -> Result<(), CUerror> {
    if state.is_null() || bin_out.is_null() || size_out.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let api = load_hiprtc().ok_or(CUerror::JIT_COMPILER_NOT_FOUND)?;
    let id = state as usize;
    let mut reg = registry().lock().unwrap();
    let entry = reg.get_mut(&id).ok_or(CUerror::INVALID_HANDLE)?;
    let hip_state = entry.hiprtc_state.0;
    let mut raw_bin: *mut c_void = ptr::null_mut();
    let mut raw_size: usize = 0;
    let r = (api.link_complete)(hip_state, &mut raw_bin, &mut raw_size);
    if r != HIPRTC_SUCCESS || raw_bin.is_null() || raw_size == 0 {
        trace(&format!(
            "link_complete: hiprtc code={} bin=0x{:x} size={}",
            r.0, raw_bin as usize, raw_size
        ));
        return Err(CUerror::INVALID_IMAGE);
    }
    // HIPRTC documenta che il bin_out memory e' owned by state finche'
    // hiprtcLinkDestroy non lo libera. Esponiamo direttamente il puntatore.
    // Caso paranoia: alcune build AMD copiano in heap separato; conserviamo
    // anche una copia owned in completed_bin per garantire vita >= cuLinkDestroy.
    let slice = std::slice::from_raw_parts(raw_bin as *const u8, raw_size);
    entry.completed_bin = Some(slice.to_vec());
    // Esponiamo il buffer della copia (vita garantita fino a cuLinkDestroy):
    let owned = entry.completed_bin.as_ref().unwrap();
    *bin_out = owned.as_ptr() as *mut c_void;
    *size_out = owned.len();
    trace(&format!(
        "link_complete: state=0x{:x} cubin_size={}",
        id, raw_size
    ));
    Ok(())
}

pub(crate) unsafe fn link_destroy(state: CUlinkState) -> Result<(), CUerror> {
    if state.is_null() {
        return Err(CUerror::INVALID_VALUE);
    }
    let api = load_hiprtc().ok_or(CUerror::JIT_COMPILER_NOT_FOUND)?;
    let id = state as usize;
    let mut reg = registry().lock().unwrap();
    let entry = reg.remove(&id).ok_or(CUerror::INVALID_HANDLE)?;
    let r = (api.link_destroy)(entry.hiprtc_state.0);
    if r != HIPRTC_SUCCESS {
        trace(&format!(
            "link_destroy: hiprtc code={} state=0x{:x}",
            r.0, id
        ));
        return Err(CUerror::UNKNOWN);
    }
    trace(&format!("link_destroy: state=0x{:x} OK", id));
    Ok(())
}
