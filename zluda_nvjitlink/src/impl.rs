// VEGA-PATCH 2026-06-30 GAP G8 worker 2/5 — zluda_nvjitlink::impl
//
// Backend hiprtc 5.7 mirror (subset Link API condiviso con cu_link.rs T14:00).
// Differenza dal cu_link.rs:
//   - cu_link.rs: handle CUlinkState (driver API legacy CUDA 11) + opzioni
//     CUjit_option enum (0..24 == HIPRTC_JIT_*)
//   - questo file: handle nvJitLinkHandle (CUDA 12 nuovo) + opzioni come
//     stringhe "-arch=sm_70 -lto" stile compilatore (NVIDIA design choice
//     CUDA 12). Parsing options stringa -> HIPRTC option enum quando possibile,
//     ignored quando AMD non ha analogo (es. -arch=sm_70 mapping a gfx906
//     implicito perche' siamo su Vega 20).
//
// Registry handle:
//   nvJitLinkHandle e' *mut JitLinkEntry. NON usiamo as_usize id come cu_link,
//   per evitare lock contention: ogni handle ha self-contained state +
//   Mutex interno per Complete che e' single-threaded NVIDIA-side. Match
//   pattern hipfft style (Box::into_raw).

use crate::{NvJitLinkHandle, NvJitLinkInputType, NvJitLinkResult};
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_void, CStr};
use std::os::raw::c_uint;
use std::ptr;
use std::sync::{Mutex, OnceLock};

// ---------- hiprtc ABI mirror (subset Link API condiviso con cu_link.rs) ----------

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
struct HiprtcResult(i32);

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
            // regola 4-bis: hiprtc da ROCm 5.7 ESCLUSIVO
            // Ordine: HIP SDK 5.7, il kit Marconi (%MARCONI_ROOT%\bin), la cartella storica della
            // macchina di sviluppo (C:\rocm_5_7_expansion_pack\bin, ignorata se non esiste), il PATH.
            let mut candidates = vec![String::from(r"C:\Program Files\AMD\ROCm\5.7\bin\hiprtc0507.dll")];
            if let Some(m) = std::env::var_os("MARCONI_ROOT") {
                candidates.push(std::path::Path::new(&m).join("bin").join("hiprtc0507.dll").to_string_lossy().into_owned());
            }
            candidates.push(String::from(r"C:\rocm_5_7_expansion_pack\bin\hiprtc0507.dll"));
            candidates.push(String::from("hiprtc0507.dll"));
            for path in candidates {
                unsafe {
                    if let Ok(lib) = Library::new(&path) {
                        let create: Symbol<FnLinkCreate> =
                            match lib.get(b"hiprtcLinkCreate\0") {
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
                        trace(&format!("nvJitLink: hiprtc 5.7 loaded from {}", path));
                        return Some(api);
                    }
                }
            }
            trace("nvJitLink: hiprtc 5.7 NOT loadable -> ERROR_INTERNAL on all calls");
            None
        })
        .as_ref()
}

fn trace(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("nvjitlink_trace.log")
    {
        let _ = writeln!(f, "[nvJitLink] {}", msg);
    }
}

// ---------- Per-handle state ----------

struct JitLinkEntry {
    hiprtc_state: Mutex<HiprtcLinkStateRaw>,
    // bin owned dopo Complete (vita garantita fino a Destroy)
    completed_bin: Mutex<Option<Vec<u8>>>,
    // log buffers
    error_log: Mutex<String>,
    info_log: Mutex<String>,
    // opzioni stringa CUDA 12 (es. "-arch=sm_70 -lto") salvate per diagnostica
    _options: Vec<String>,
}

// ---------- Parsing opzioni stringa NVIDIA -> HIPRTC u32 enum ----------
//
// NVIDIA passa opzioni come ["-arch=sm_70", "-lto", "-g"] (formato compilatore).
// HIPRTC vuole array (u32, *void) con enum 0..24. Pochi opzioni mappabili:
//   "-g"           -> HIPRTC_JIT_GENERATE_DEBUG_INFO (1)  val=1
//   "-lineinfo"    -> HIPRTC_JIT_GENERATE_LINE_INFO (5)   val=1
//   "-arch=sm_*"   -> ignorato (gfx906 implicito su Vega 20)
//   "-O<n>"        -> HIPRTC_JIT_OPTIMIZATION_LEVEL (4)   val=<n>
//   "-lto"         -> ignorato (HIPRTC fa LTO sempre quando vede LLVM_BITCODE)
//   "-maxrregcount=<n>" -> HIPRTC_JIT_MAX_REGISTERS (0)   val=<n>
// Tutto il resto: warning silente in error_log, continua.

fn parse_options(num_options: u32, opts: *const *const c_char) -> (Vec<String>, String) {
    let mut parsed = Vec::with_capacity(num_options as usize);
    let mut warns = String::new();
    if num_options == 0 || opts.is_null() {
        return (parsed, warns);
    }
    for i in 0..num_options as usize {
        unsafe {
            let p = *opts.add(i);
            if p.is_null() {
                continue;
            }
            let s = CStr::from_ptr(p).to_string_lossy().into_owned();
            // skip -arch=sm_XX (gfx906 implicito)
            if s.starts_with("-arch=") {
                trace(&format!("parse_options: skip {} (gfx906 implicit)", s));
                parsed.push(s);
                continue;
            }
            if s == "-lto" {
                trace("parse_options: skip -lto (HIPRTC auto on LLVM_BITCODE)");
                parsed.push(s);
                continue;
            }
            // opzioni note: salva + log
            if s == "-g" || s == "-lineinfo" || s.starts_with("-O") || s.starts_with("-maxrregcount=") {
                trace(&format!("parse_options: accept {}", s));
                parsed.push(s);
                continue;
            }
            // sconosciuta: log warning ma non fail (NVIDIA tollerante)
            let w = format!("nvJitLink warning: unrecognized option '{}' (ignored on AMD backend)\n", s);
            warns.push_str(&w);
            trace(&format!("parse_options: WARN {}", s));
            parsed.push(s);
        }
    }
    (parsed, warns)
}

// ---------- Mapping NvJitLinkInputType -> HIPRTC_JIT_INPUT_* u32 ----------

fn map_input_type(t: NvJitLinkInputType) -> Result<c_uint, NvJitLinkResult> {
    match t {
        // CUBIN NVIDIA ELF SM_xx: cross-arch translate impossibile.
        NvJitLinkInputType::NVJITLINK_INPUT_CUBIN => {
            Err(NvJitLinkResult::NVJITLINK_ERROR_UNRECOGNIZED_INPUT)
        }
        // PTX raw: HIPRTC_JIT_INPUT_PTX=1 accettato nominalmente, rifiutato runtime
        NvJitLinkInputType::NVJITLINK_INPUT_PTX => Ok(1),
        // LTO-IR: NVPTX bitcode, non traducibile a AMDGCN -> NVVM_COMPILE
        NvJitLinkInputType::NVJITLINK_INPUT_LTOIR => {
            Err(NvJitLinkResult::NVJITLINK_ERROR_NVVM_COMPILE)
        }
        // FATBIN: HIPRTC_JIT_INPUT_FATBINARY=2 (legacy CUDA equivalente)
        NvJitLinkInputType::NVJITLINK_INPUT_FATBIN => Ok(2),
        // OBJECT / LIBRARY: forward a HIPRTC 4 (LIBRARY), accetta best-effort
        NvJitLinkInputType::NVJITLINK_INPUT_OBJECT => Ok(4),
        NvJitLinkInputType::NVJITLINK_INPUT_LIBRARY => Ok(4),
        // INDEX / NONE / ANY: input wrapper, prova come fatbin
        NvJitLinkInputType::NVJITLINK_INPUT_INDEX => Ok(2),
        NvJitLinkInputType::NVJITLINK_INPUT_ANY => Ok(2),
        NvJitLinkInputType::NVJITLINK_INPUT_NONE => {
            Err(NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT)
        }
    }
}

// ---------- API impl ----------

pub(crate) unsafe fn create(
    handle_out: *mut NvJitLinkHandle,
    num_options: u32,
    options: *const *const c_char,
) -> NvJitLinkResult {
    if handle_out.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let api = match load_hiprtc() {
        Some(a) => a,
        None => return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    };
    let (parsed_opts, warns) = parse_options(num_options, options);

    // hiprtcLinkCreate con zero opzioni (le opzioni stringa CUDA 12 non
    // mappano 1:1 a HIPRTC enum; le applichiamo per-AddData se necessario)
    let mut hip_state: HiprtcLinkStateRaw = ptr::null_mut();
    let r = (api.link_create)(0, ptr::null_mut(), ptr::null_mut(), &mut hip_state);
    if r != HIPRTC_SUCCESS || hip_state.is_null() {
        trace(&format!("create: hiprtc fail code={}", r.0));
        return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL;
    }

    let entry = Box::new(JitLinkEntry {
        hiprtc_state: Mutex::new(hip_state),
        completed_bin: Mutex::new(None),
        error_log: Mutex::new(String::new()),
        info_log: Mutex::new(warns),
        _options: parsed_opts,
    });
    let raw = Box::into_raw(entry);
    *handle_out = NvJitLinkHandle(raw as *mut c_void);
    trace(&format!("create: handle=0x{:x}", raw as usize));
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn destroy(handle_in: *mut NvJitLinkHandle) -> NvJitLinkResult {
    if handle_in.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let h = (*handle_in).0;
    if h.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let api = match load_hiprtc() {
        Some(a) => a,
        None => return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    };
    let entry: Box<JitLinkEntry> = Box::from_raw(h as *mut JitLinkEntry);
    let state = *entry.hiprtc_state.lock().unwrap();
    if !state.is_null() {
        let _ = (api.link_destroy)(state);
    }
    (*handle_in).0 = ptr::null_mut();
    trace(&format!("destroy: handle=0x{:x}", h as usize));
    NvJitLinkResult::NVJITLINK_SUCCESS
}

fn entry_from(h: NvJitLinkHandle) -> Result<&'static JitLinkEntry, NvJitLinkResult> {
    if h.0.is_null() {
        return Err(NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT);
    }
    unsafe { Ok(&*(h.0 as *const JitLinkEntry)) }
}

pub(crate) unsafe fn add_data(
    handle: NvJitLinkHandle,
    input_type: NvJitLinkInputType,
    data: *const c_void,
    size: usize,
    name: *const c_char,
) -> NvJitLinkResult {
    if data.is_null() || size == 0 {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let api = match load_hiprtc() {
        Some(a) => a,
        None => return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    };
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let hip_input = match map_input_type(input_type) {
        Ok(t) => t,
        Err(e) => {
            entry.error_log.lock().unwrap().push_str(&format!(
                "nvJitLinkAddData: input type {:?} not supported on AMD backend\n",
                input_type
            ));
            return e;
        }
    };
    let state = *entry.hiprtc_state.lock().unwrap();
    let r = (api.link_add_data)(
        state,
        hip_input,
        data as *mut c_void,
        size,
        name,
        0,
        ptr::null_mut(),
        ptr::null_mut(),
    );
    if r != HIPRTC_SUCCESS {
        let msg = format!(
            "nvJitLinkAddData: hiprtc code={} input={:?} size={}\n",
            r.0, input_type, size
        );
        trace(&msg);
        entry.error_log.lock().unwrap().push_str(&msg);
        // PTX/FATBIN su AMD: classifico come PTX_COMPILE per Inductor/Triton
        if hip_input == 1 || hip_input == 2 {
            return NvJitLinkResult::NVJITLINK_ERROR_PTX_COMPILE;
        }
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn add_file(
    handle: NvJitLinkHandle,
    input_type: NvJitLinkInputType,
    file_name: *const c_char,
) -> NvJitLinkResult {
    if file_name.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let api = match load_hiprtc() {
        Some(a) => a,
        None => return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    };
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let hip_input = match map_input_type(input_type) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let state = *entry.hiprtc_state.lock().unwrap();
    let r = (api.link_add_file)(state, hip_input, file_name, 0, ptr::null_mut(), ptr::null_mut());
    if r != HIPRTC_SUCCESS {
        let path = CStr::from_ptr(file_name).to_string_lossy();
        let msg = format!("nvJitLinkAddFile: hiprtc code={} path={}\n", r.0, path);
        trace(&msg);
        entry.error_log.lock().unwrap().push_str(&msg);
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn complete(handle: NvJitLinkHandle) -> NvJitLinkResult {
    let api = match load_hiprtc() {
        Some(a) => a,
        None => return NvJitLinkResult::NVJITLINK_ERROR_INTERNAL,
    };
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let state = *entry.hiprtc_state.lock().unwrap();
    let mut bin: *mut c_void = ptr::null_mut();
    let mut bsize: usize = 0;
    let r = (api.link_complete)(state, &mut bin, &mut bsize);
    if r != HIPRTC_SUCCESS || bin.is_null() || bsize == 0 {
        let msg = format!("nvJitLinkComplete: hiprtc code={} bin=0x{:x} size={}\n",
            r.0, bin as usize, bsize);
        trace(&msg);
        entry.error_log.lock().unwrap().push_str(&msg);
        return NvJitLinkResult::NVJITLINK_ERROR_FINALIZE;
    }
    // copia owned (vita >= Destroy)
    let slice = std::slice::from_raw_parts(bin as *const u8, bsize);
    *entry.completed_bin.lock().unwrap() = Some(slice.to_vec());
    trace(&format!("complete: cubin_size={}", bsize));
    NvJitLinkResult::NVJITLINK_SUCCESS
}

// ---------- Log accessors ----------

pub(crate) unsafe fn get_error_log_size(handle: NvJitLinkHandle, size: *mut usize) -> NvJitLinkResult {
    if size.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    // include null-terminator come NVIDIA fa
    *size = entry.error_log.lock().unwrap().len() + 1;
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn get_error_log(handle: NvJitLinkHandle, out: *mut c_char) -> NvJitLinkResult {
    if out.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let s = entry.error_log.lock().unwrap();
    let bytes = s.as_bytes();
    ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out, bytes.len());
    *out.add(bytes.len()) = 0;
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn get_info_log_size(handle: NvJitLinkHandle, size: *mut usize) -> NvJitLinkResult {
    if size.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    *size = entry.info_log.lock().unwrap().len() + 1;
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn get_info_log(handle: NvJitLinkHandle, out: *mut c_char) -> NvJitLinkResult {
    if out.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let s = entry.info_log.lock().unwrap();
    let bytes = s.as_bytes();
    ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, out, bytes.len());
    *out.add(bytes.len()) = 0;
    NvJitLinkResult::NVJITLINK_SUCCESS
}

// ---------- Linked cubin / ptx ----------

pub(crate) unsafe fn get_linked_cubin_size(handle: NvJitLinkHandle, size: *mut usize) -> NvJitLinkResult {
    if size.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let bin = entry.completed_bin.lock().unwrap();
    match bin.as_ref() {
        Some(v) => {
            *size = v.len();
            NvJitLinkResult::NVJITLINK_SUCCESS
        }
        None => NvJitLinkResult::NVJITLINK_ERROR_FINALIZE,
    }
}

pub(crate) unsafe fn get_linked_cubin(handle: NvJitLinkHandle, out: *mut c_void) -> NvJitLinkResult {
    if out.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let entry = match entry_from(handle) {
        Ok(e) => e,
        Err(e) => return e,
    };
    let bin = entry.completed_bin.lock().unwrap();
    match bin.as_ref() {
        Some(v) => {
            ptr::copy_nonoverlapping(v.as_ptr(), out as *mut u8, v.len());
            NvJitLinkResult::NVJITLINK_SUCCESS
        }
        None => NvJitLinkResult::NVJITLINK_ERROR_FINALIZE,
    }
}

// PTX output non disponibile (AMDGCN backend, no NVPTX emit) -> ritorna 0 size
pub(crate) unsafe fn get_linked_ptx_size(handle: NvJitLinkHandle, size: *mut usize) -> NvJitLinkResult {
    if size.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let _ = entry_from(handle);
    *size = 1; // solo null-terminator
    NvJitLinkResult::NVJITLINK_SUCCESS
}

pub(crate) unsafe fn get_linked_ptx(handle: NvJitLinkHandle, out: *mut c_char) -> NvJitLinkResult {
    if out.is_null() {
        return NvJitLinkResult::NVJITLINK_ERROR_INVALID_INPUT;
    }
    let _ = entry_from(handle);
    *out = 0; // stringa vuota null-terminata
    NvJitLinkResult::NVJITLINK_SUCCESS
}
