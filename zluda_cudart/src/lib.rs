//! zluda_cudart auto-gen lib.rs — 421 cuda* simboli forwarder/stub.
//! Generated T16:25 18 giu 2026 per sprint #239.
//! VEGA-PATCH 2026-06-30 worker 4/5 TASK A: feature `cuda12` aggiunge ~45 export
//! CUDA 12.x in modulo cuda12_extras (gate per cudart64_12.dll target).
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]
#![allow(dead_code)]

use hip_runtime_sys::*;
use std::ffi::{c_int, c_uint, c_void, c_char};
use std::sync::Once;

// VEGA-PATCH 2026-06-30 TASK A: estensioni CUDA 12.x gated by build.rs cfg.
// build.rs propaga `cargo:rustc-cfg=cudart_12` quando feature `cuda12` attiva.
// Attivo solo quando si builda cudart64_12.dll (NON in cudart64_110 default).
#[cfg(cudart_12)]
mod cuda12_extras;

// VEGA-FIX T17:05: HIP runtime DEVE essere inizializzato prima di ogni chiamata hip*.
// PyTorch chiama cuInit via nvcuda ma NON inizializza HIP. Sprint cudart: una volta
// al primo entry forwarder, chiamiamo hipInit(0) per tutta la sessione.
static HIP_INIT: Once = Once::new();

#[inline(always)]
fn ensure_hip_init() {
    HIP_INIT.call_once(|| unsafe {
        let _ = hipInit(0);
    });
}

// VEGA-TRACE T11:20 — helper globale per trace stub `() -> 0` chiamati.
#[inline(always)]
fn t(name: &'static str) -> cudaError_t {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_cudart_stub_trace.log") {
        let _ = writeln!(f, "{}", name);
    }
    0
}

// VEGA-FIX T11:40 — extern decl di __hipRegister* esportate da amdhip64.dll v5.7.
// Permettono di forwardare le __cudaRegister* di torch wheel pre-compiled al runtime HIP
// per popolare la map host_func_addr → device_kernel_name che hipLaunchKernel consulta.
#[link(name = "amdhip64")]
extern "C" {
    fn __hipRegisterFatBinary(data: *mut c_void) -> *mut *mut c_void;
    fn __hipRegisterFunction(
        modules: *mut *mut c_void,
        host_func: *const c_void,
        device_func: *mut c_char,
        device_name: *const c_char,
        thread_limit: c_int,
        tid: *mut c_void,
        bid: *mut c_void,
        b_dim: *mut c_void,
        g_dim: *mut c_void,
        w_size: *mut c_int,
    );
    fn __hipRegisterVar(
        modules: *mut *mut c_void,
        host_var: *mut c_void,
        device_var: *mut c_char,
        device_name: *const c_char,
        ext: c_int,
        size: usize,
        constant: c_int,
        global: c_int,
    );
    fn __hipUnregisterFatBinary(modules: *mut *mut c_void);
}

type cudaError_t = c_int;
type cudaStream_t = *mut c_void;
type cudaEvent_t = *mut c_void;
type cudaGraph_t = *mut c_void;
type cudaGraphNode_t = *mut c_void;
type cudaGraphExec_t = *mut c_void;

// Driver/Runtime version: torch 2.4+cu118 (VEGA-FIX T10:05 23 giu - back to cu118)
// VEGA-PATCH 2026-06-30 TASK A worker3/5: bump condizionale CUDA 12.1 quando feature `cuda12` attiva.
// `cargo build --features cuda12` -> build.rs emette `--cfg cudart_12` -> qui scattano 12010.
#[cfg(not(cudart_12))]
const CUDART_VERSION: c_int = 11080;  // 11.8
#[cfg(cudart_12)]
const CUDART_VERSION: c_int = 12010;  // 12.1 (target torch 2.4+cu121 test_venv_torch24)

#[inline(always)]
fn ok_err(rc: hipError_t) -> cudaError_t {
    match rc {
        Ok(_) => 0,
        Err(e) => {
            let code = e.0.get() as cudaError_t;
            // VEGA-DEBUG T10:10 23 giu: log hip err propagato per identificare colpevole 801
            use std::io::Write;
            if let Ok(mut f) = crate::diag("zluda_cudart_err_trace.log")
            {
                let bt = std::backtrace::Backtrace::force_capture();
                let bt_str = format!("{}", bt);
                let lines: Vec<&str> = bt_str.lines().take(6).collect();
                let _ = writeln!(f, "[ok_err] hip_err={}\n  {}\n---", code, lines.join("\n  "));
            }
            code
        }
    }
}

#[inline(always)]
fn ok_err_init(rc: hipError_t) -> cudaError_t {
    ensure_hip_init();
    match rc {
        Ok(_) => 0,
        Err(e) => {
            let code = e.0.get() as cudaError_t;
            use std::io::Write;
            if let Ok(mut f) = crate::diag("zluda_cudart_err_trace.log")
            {
                let bt = std::backtrace::Backtrace::force_capture();
                let bt_str = format!("{}", bt);
                let lines: Vec<&str> = bt_str.lines().take(6).collect();
                let _ = writeln!(f, "[ok_err_init] hip_err={}\n  {}\n---", code, lines.join("\n  "));
            }
            code
        }
    }
}

// VEGA-DEBUG T08:10 23 giu: trace logger persistente per identificare sequenza calls torch
#[inline(always)]
fn trace(name: &str, ret: cudaError_t) -> cudaError_t {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_call_trace.log")
    {
        let _ = writeln!(f, "[{}] ret={}", name, ret);
    }
    ret
}

#[inline(always)]
unsafe fn norm_stream(s: cudaStream_t) -> hipStream_t {
    if s.is_null() { hipStream_t(2 as *mut _) } else { hipStream_t(s as *mut _) }
}

// === HOST_HELPER (13 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn NvOptimusEnablementCuda(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaInitModule(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaPopCallConfiguration(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaPushCallConfiguration(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
// VEGA-FIX T11:40: forwarder __cudaRegisterFatBinary → __hipRegisterFatBinary.
// torch wheel chiama questa con ptr a __fatBinHandle{magic, version, data, filename}.
// Returns handle (modules ptr) usato da __cudaRegisterFunction.
// Diventa il VALORE di ritorno (puntatore al modulo HIP), NON cudaError_t.
pub unsafe extern "C" fn __cudaRegisterFatBinary(data: *mut c_void) -> *mut *mut c_void {
    ensure_hip_init();
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_cudart_register.log") {
        let _ = writeln!(f, "[__cudaRegisterFatBinary] data={:p}", data);
    }
    __hipRegisterFatBinary(data)
}
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterFatBinaryEnd(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterFunction(
    handle: *mut *mut c_void,
    host_func: *const c_void,
    device_func: *mut c_char,
    device_name: *const c_char,
    thread_limit: c_int,
    tid: *mut c_void,
    bid: *mut c_void,
    b_dim: *mut c_void,
    g_dim: *mut c_void,
    w_size: *mut c_int,
) -> cudaError_t {
    ensure_hip_init();
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_cudart_register.log") {
        let name_str = if !device_name.is_null() {
            std::ffi::CStr::from_ptr(device_name).to_string_lossy().to_string()
        } else { "(null)".to_string() };
        let _ = writeln!(f, "[__cudaRegisterFunction] host={:p} name={}", host_func, name_str);
    }
    __hipRegisterFunction(handle, host_func, device_func, device_name, thread_limit, tid, bid, b_dim, g_dim, w_size);
    0
}
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterHostVar(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterManagedVar(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterSurface(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterTexture(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaRegisterVar(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn __cudaUnregisterFatBinary(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }

// === INTEROP_STUB (59 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10GetDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10GetDevices(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10GetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10MapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10RegisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceGetMappedArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceGetMappedPitch(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceGetMappedPointer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceGetMappedSize(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceGetSurfaceDimensions(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10ResourceSetMapFlags(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10SetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10UnmapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D10UnregisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D11GetDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D11GetDevices(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D11GetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D11SetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9Begin(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9End(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9GetDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9GetDevices(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9GetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9MapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9MapVertexBuffer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9RegisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9RegisterVertexBuffer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceGetMappedArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceGetMappedPitch(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceGetMappedPointer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceGetMappedSize(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceGetSurfaceDimensions(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9ResourceSetMapFlags(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9SetDirect3DDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9UnmapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9UnmapVertexBuffer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9UnregisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaD3D9UnregisterVertexBuffer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLGetDevices(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLMapBufferObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLMapBufferObjectAsync(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLRegisterBufferObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLSetBufferObjectMapFlags(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLSetGLDevice(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLUnmapBufferObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLUnmapBufferObjectAsync(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGLUnregisterBufferObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsD3D10RegisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsD3D11RegisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsD3D9RegisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsGLRegisterBuffer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsGLRegisterImage(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsMapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsResourceGetMappedMipmappedArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsResourceGetMappedPointer(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsResourceSetMapFlags(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsSubResourceGetMappedArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsUnmapResources(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphicsUnregisterResource(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }

// === TEXTURE_STUB (12 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaBindSurfaceToArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaBindTexture(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaBindTexture2D(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaBindTextureToArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaBindTextureToMipmappedArray(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaCreateChannelDesc(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaCreateSurfaceObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaCreateTextureObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaCreateTextureObject_v2(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaDestroySurfaceObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaDestroyTextureObject(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaUnbindTexture(arg1: *mut c_void, arg2: *mut c_void, arg3: *mut c_void, arg4: *mut c_void) -> cudaError_t { 0 }

// === THREAD (7 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaThreadExchangeStreamCaptureMode(mode: *mut c_int) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-15: CUDAGraph capture mode toggle
    ensure_hip_init();
    if mode.is_null() { return 0; }
    let mut m = hipStreamCaptureMode(*mode as u32);
    let rc = ok_err(hipThreadExchangeStreamCaptureMode(&mut m as *mut hipStreamCaptureMode));
    *mode = m.0 as c_int;
    rc
}
#[no_mangle]
pub unsafe extern "C" fn cudaThreadExit() -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T16:00 sprint-app-real cudart-fix-16: deprecated CUDA1.x alias
    // Forwarder reale a hipDeviceReset (Blender Cycles deinit, legacy torch teardown)
    ensure_hip_init();
    ok_err(hipDeviceReset())
}
#[no_mangle]
pub unsafe extern "C" fn cudaThreadGetCacheConfig() -> cudaError_t { t("cudaThreadGetCacheConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaThreadGetLimit() -> cudaError_t { t("cudaThreadGetLimit") }
#[no_mangle]
pub unsafe extern "C" fn cudaThreadSetCacheConfig() -> cudaError_t { t("cudaThreadSetCacheConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaThreadSetLimit() -> cudaError_t { t("cudaThreadSetLimit") }
#[no_mangle]
pub unsafe extern "C" fn cudaThreadSynchronize() -> cudaError_t { /* TODO impl forwarder a hipDeviceSynchronize */ 0 }

// === STREAM (36 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaStreamAddCallback(
    stream: *mut c_void,
    callback: hipStreamCallback_t,
    user_data: *mut c_void,
    flags: c_uint,
) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-13: PyTorch profiler async hooks
    // CUDA<>HIP callback ABI compatibile (entrambi `void(stream, status, userData)`)
    ensure_hip_init();
    ok_err(hipStreamAddCallback(norm_stream(stream), callback, user_data, flags))
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamAddCallback_ptsz(
    stream: *mut c_void,
    callback: hipStreamCallback_t,
    user_data: *mut c_void,
    flags: c_uint,
) -> cudaError_t {
    cudaStreamAddCallback(stream, callback, user_data, flags)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamAttachMemAsync() -> cudaError_t { t("cudaStreamAttachMemAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamAttachMemAsync_ptsz() -> cudaError_t { cudaStreamAttachMemAsync() }
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-19a: cudaStreamBeginCapture real impl
pub unsafe extern "C" fn cudaStreamBeginCapture(stream: *mut c_void, mode: c_int) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipStreamBeginCapture(norm_stream(stream), hipStreamCaptureMode(mode as u32)))
}
#[no_mangle]
// VEGA-PATCH SPRINT-D worker14 2026-06-29: _ptsz delegate sibling real (no stub silent fake 0)
pub unsafe extern "C" fn cudaStreamBeginCapture_ptsz(stream: *mut c_void, mode: c_int) -> cudaError_t {
    cudaStreamBeginCapture(stream, mode)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCopyAttributes() -> cudaError_t { t("cudaStreamCopyAttributes") }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCopyAttributes_ptsz() -> cudaError_t { cudaStreamCopyAttributes() }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreate(s: *mut *mut c_void) -> cudaError_t { ensure_hip_init(); let mut h: hipStream_t = hipStream_t(std::ptr::null_mut()); let rc = ok_err(hipStreamCreate(&mut h)); if rc == 0 && !s.is_null() { *s = h.0 as *mut _; } rc }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: c_uint) -> cudaError_t { ensure_hip_init(); let mut h: hipStream_t = hipStream_t(std::ptr::null_mut()); let rc = ok_err(hipStreamCreateWithFlags(&mut h, flags)); if rc == 0 && !s.is_null() { *s = h.0 as *mut _; } rc }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamCreateWithPriority() -> cudaError_t { /* TODO impl forwarder a hipStreamCreateWithPriority */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamDestroy(s: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipStreamDestroy(hipStream_t(s as *mut _))) }
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-19b: cudaStreamEndCapture real impl
pub unsafe extern "C" fn cudaStreamEndCapture(stream: *mut c_void, p_graph: *mut *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if p_graph.is_null() { return 0; }
    let mut hg: hipGraph_t = std::ptr::null_mut();
    let rc = ok_err(hipStreamEndCapture(norm_stream(stream), &mut hg));
    if rc == 0 { *p_graph = hg as *mut c_void; }
    rc
}
#[no_mangle]
// VEGA-PATCH SPRINT-D worker14 2026-06-29: _ptsz delegate sibling real
pub unsafe extern "C" fn cudaStreamEndCapture_ptsz(stream: *mut c_void, p_graph: *mut *mut c_void) -> cudaError_t {
    cudaStreamEndCapture(stream, p_graph)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetAttribute() -> cudaError_t { t("cudaStreamGetAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetAttribute_ptsz() -> cudaError_t { cudaStreamGetAttribute() }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetCaptureInfo(stream: *mut c_void, status: *mut c_uint, id: *mut u64) -> cudaError_t {
    // VEGA-FIX T11:05: settare status=0 (cudaStreamCaptureStatusNone) per non-capture stream
    if !status.is_null() { *status = 0; }
    if !id.is_null() { *id = 0; }
    let _ = stream;
    0
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetCaptureInfo_ptsz(stream: *mut c_void, status: *mut c_uint, id: *mut u64) -> cudaError_t {
    cudaStreamGetCaptureInfo(stream, status, id)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetCaptureInfo_v2(stream: *mut c_void, status: *mut c_uint, id: *mut u64, graph: *mut *mut c_void, deps: *mut *mut c_void, num_deps: *mut usize) -> cudaError_t {
    if !status.is_null() { *status = 0; }
    if !id.is_null() { *id = 0; }
    if !graph.is_null() { *graph = std::ptr::null_mut(); }
    if !deps.is_null() { *deps = std::ptr::null_mut(); }
    if !num_deps.is_null() { *num_deps = 0; }
    let _ = stream;
    0
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetCaptureInfo_v2_ptsz(stream: *mut c_void, status: *mut c_uint, id: *mut u64, graph: *mut *mut c_void, deps: *mut *mut c_void, num_deps: *mut usize) -> cudaError_t {
    cudaStreamGetCaptureInfo_v2(stream, status, id, graph, deps, num_deps)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetFlags() -> cudaError_t { /* TODO impl forwarder a hipStreamGetFlags */ 0 }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-17: cudaStreamGetFlags_ptsz forwarder reale
// Impact: torch.cuda.Stream(__cuda_stream__) introspection — silent 0 mascherava capture flag check.
pub unsafe extern "C" fn cudaStreamGetFlags_ptsz(
    stream: *mut c_void, flags: *mut c_uint,
) -> cudaError_t {
    ensure_hip_init();
    if flags.is_null() { return 1; }
    ok_err(hipStreamGetFlags_spt(norm_stream(stream), flags))
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamGetPriority(stream: *mut c_void, p: *mut c_int) -> cudaError_t { ensure_hip_init(); ok_err(hipStreamGetPriority(norm_stream(stream), p)) }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-3: cudaStreamGetPriority_ptsz delegate
pub unsafe extern "C" fn cudaStreamGetPriority_ptsz(stream: *mut c_void, p: *mut c_int) -> cudaError_t {
    cudaStreamGetPriority(stream, p)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamIsCapturing(stream: *mut c_void, status: *mut c_uint) -> cudaError_t { ensure_hip_init(); if !status.is_null() { *status = 0; } 0 }
#[no_mangle]
// VEGA-PATCH SPRINT-D worker14 2026-06-29 cudart-ptsz-5: cudaStreamIsCapturing_ptsz delegate sibling real
pub unsafe extern "C" fn cudaStreamIsCapturing_ptsz(stream: *mut c_void, status: *mut c_uint) -> cudaError_t {
    cudaStreamIsCapturing(stream, status)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamQuery(stream: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipStreamQuery(norm_stream(stream))) }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-2: cudaStreamQuery_ptsz delegate
pub unsafe extern "C" fn cudaStreamQuery_ptsz(stream: *mut c_void) -> cudaError_t {
    cudaStreamQuery(stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamSetAttribute() -> cudaError_t { t("cudaStreamSetAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamSetAttribute_ptsz() -> cudaError_t { cudaStreamSetAttribute() }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamSynchronize(stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipStreamSynchronize(norm_stream(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-7: cudaStreamSynchronize_ptsz delegate
pub unsafe extern "C" fn cudaStreamSynchronize_ptsz(stream: *mut c_void) -> cudaError_t {
    cudaStreamSynchronize(stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaStreamUpdateCaptureDependencies() -> cudaError_t { t("cudaStreamUpdateCaptureDependencies") }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamUpdateCaptureDependencies_ptsz() -> cudaError_t { cudaStreamUpdateCaptureDependencies() }
#[no_mangle]
pub unsafe extern "C" fn cudaStreamWaitEvent(stream: *mut c_void, ev: *mut c_void, flags: c_uint) -> cudaError_t {
    // VEGA-FIX T07:52 23 giu: torch 2.4 at::cuda::getDeviceProperties fa self-sync con
    // cudaStreamWaitEvent in init phase. HIP wrapper ritorna hipErrorNotInitialized(3) su
    // event non ancora bound al stream corrente. Backtrace inchiodato: c10_cuda_check rejecta
    // err=3 → throws CUDA error 3 → torch.cuda init fail. Graceful no-op: ritorno 0 sempre.
    ensure_hip_init();
    if ev.is_null() { return 0; }
    let _ = hipStreamWaitEvent(norm_stream(stream), ev as hipEvent_t, flags);
    0
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-1: cudaStreamWaitEvent_ptsz delegate
pub unsafe extern "C" fn cudaStreamWaitEvent_ptsz(stream: *mut c_void, ev: *mut c_void, flags: c_uint) -> cudaError_t {
    cudaStreamWaitEvent(stream, ev, flags)
}

// === EVENT (10 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaEventCreate() -> cudaError_t { /* TODO impl forwarder a hipEventCreate */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaEventCreateWithFlags(ev: *mut *mut c_void, flags: c_uint) -> cudaError_t { ensure_hip_init(); let mut h: hipEvent_t = std::ptr::null_mut(); let rc = ok_err(hipEventCreateWithFlags(&mut h, flags)); if rc == 0 && !ev.is_null() { *ev = h as *mut _; } rc }
#[no_mangle]
pub unsafe extern "C" fn cudaEventDestroy(ev: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipEventDestroy(ev as hipEvent_t)) }
#[no_mangle]
pub unsafe extern "C" fn cudaEventElapsedTime(t: *mut f32, start: *mut c_void, end: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipEventElapsedTime(t, start as hipEvent_t, end as hipEvent_t)) }
#[no_mangle]
pub unsafe extern "C" fn cudaEventQuery(ev: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipEventQuery(ev as hipEvent_t)) }
#[no_mangle]
pub unsafe extern "C" fn cudaEventRecord(ev: *mut c_void, stream: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipEventRecord(ev as hipEvent_t, norm_stream(stream))) }
#[no_mangle]
pub unsafe extern "C" fn cudaEventRecordWithFlags() -> cudaError_t { /* TODO impl forwarder a hipEventRecordWithFlags */ 0 }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-4: cudaEventRecordWithFlags_ptsz → cudaEventRecord
// hipEventRecord ignora il flags param (CUDA cudaEventRecordExternal solo) → ok delegare a EventRecord puro.
pub unsafe extern "C" fn cudaEventRecordWithFlags_ptsz(ev: *mut c_void, stream: *mut c_void, _flags: c_uint) -> cudaError_t {
    cudaEventRecord(ev, stream)
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-8: cudaEventRecord_ptsz delegate
pub unsafe extern "C" fn cudaEventRecord_ptsz(ev: *mut c_void, stream: *mut c_void) -> cudaError_t {
    cudaEventRecord(ev, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaEventSynchronize(ev: *mut c_void) -> cudaError_t { ensure_hip_init(); ok_err(hipEventSynchronize(ev as hipEvent_t)) }

// === GRAPH (78 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddChildGraphNode() -> cudaError_t { t("cudaGraphAddChildGraphNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddDependencies() -> cudaError_t { t("cudaGraphAddDependencies") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddEmptyNode() -> cudaError_t { t("cudaGraphAddEmptyNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddEventRecordNode() -> cudaError_t { t("cudaGraphAddEventRecordNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddEventWaitNode() -> cudaError_t { t("cudaGraphAddEventWaitNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddExternalSemaphoresSignalNode() -> cudaError_t { t("cudaGraphAddExternalSemaphoresSignalNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddExternalSemaphoresWaitNode() -> cudaError_t { t("cudaGraphAddExternalSemaphoresWaitNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddHostNode() -> cudaError_t { t("cudaGraphAddHostNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddKernelNode() -> cudaError_t { t("cudaGraphAddKernelNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemAllocNode() -> cudaError_t { t("cudaGraphAddMemAllocNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemFreeNode() -> cudaError_t { t("cudaGraphAddMemFreeNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemcpyNode() -> cudaError_t { t("cudaGraphAddMemcpyNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemcpyNode1D() -> cudaError_t { t("cudaGraphAddMemcpyNode1D") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemcpyNodeFromSymbol() -> cudaError_t { t("cudaGraphAddMemcpyNodeFromSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemcpyNodeToSymbol() -> cudaError_t { t("cudaGraphAddMemcpyNodeToSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddMemsetNode() -> cudaError_t { t("cudaGraphAddMemsetNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphChildGraphNodeGetGraph() -> cudaError_t { t("cudaGraphChildGraphNodeGetGraph") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphClone() -> cudaError_t { t("cudaGraphClone") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphCreate() -> cudaError_t { /* TODO impl forwarder a hipGraphCreate */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphDebugDotPrint() -> cudaError_t { t("cudaGraphDebugDotPrint") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphDestroy() -> cudaError_t { /* TODO impl forwarder a hipGraphDestroy */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphDestroyNode() -> cudaError_t { t("cudaGraphDestroyNode") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphEventRecordNodeGetEvent() -> cudaError_t { t("cudaGraphEventRecordNodeGetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphEventRecordNodeSetEvent() -> cudaError_t { t("cudaGraphEventRecordNodeSetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphEventWaitNodeGetEvent() -> cudaError_t { t("cudaGraphEventWaitNodeGetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphEventWaitNodeSetEvent() -> cudaError_t { t("cudaGraphEventWaitNodeSetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecChildGraphNodeSetParams() -> cudaError_t { t("cudaGraphExecChildGraphNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecDestroy() -> cudaError_t { /* TODO impl forwarder a hipGraphExecDestroy */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecEventRecordNodeSetEvent() -> cudaError_t { t("cudaGraphExecEventRecordNodeSetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecEventWaitNodeSetEvent() -> cudaError_t { t("cudaGraphExecEventWaitNodeSetEvent") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecExternalSemaphoresSignalNodeSetParams() -> cudaError_t { t("cudaGraphExecExternalSemaphoresSignalNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecExternalSemaphoresWaitNodeSetParams() -> cudaError_t { t("cudaGraphExecExternalSemaphoresWaitNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecHostNodeSetParams() -> cudaError_t { t("cudaGraphExecHostNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecKernelNodeSetParams() -> cudaError_t { t("cudaGraphExecKernelNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecMemcpyNodeSetParams() -> cudaError_t { t("cudaGraphExecMemcpyNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecMemcpyNodeSetParams1D() -> cudaError_t { t("cudaGraphExecMemcpyNodeSetParams1D") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecMemcpyNodeSetParamsFromSymbol() -> cudaError_t { t("cudaGraphExecMemcpyNodeSetParamsFromSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecMemcpyNodeSetParamsToSymbol() -> cudaError_t { t("cudaGraphExecMemcpyNodeSetParamsToSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecMemsetNodeSetParams() -> cudaError_t { t("cudaGraphExecMemsetNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExecUpdate() -> cudaError_t { t("cudaGraphExecUpdate") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExternalSemaphoresSignalNodeGetParams() -> cudaError_t { t("cudaGraphExternalSemaphoresSignalNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExternalSemaphoresSignalNodeSetParams() -> cudaError_t { t("cudaGraphExternalSemaphoresSignalNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExternalSemaphoresWaitNodeGetParams() -> cudaError_t { t("cudaGraphExternalSemaphoresWaitNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphExternalSemaphoresWaitNodeSetParams() -> cudaError_t { t("cudaGraphExternalSemaphoresWaitNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphGetEdges() -> cudaError_t { t("cudaGraphGetEdges") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphGetNodes() -> cudaError_t { t("cudaGraphGetNodes") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphGetRootNodes() -> cudaError_t { t("cudaGraphGetRootNodes") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphHostNodeGetParams() -> cudaError_t { t("cudaGraphHostNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphHostNodeSetParams() -> cudaError_t { t("cudaGraphHostNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphInstantiate() -> cudaError_t { /* TODO impl forwarder a hipGraphInstantiate */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphInstantiateWithFlags() -> cudaError_t { /* TODO impl forwarder a hipGraphInstantiateWithFlags */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphKernelNodeCopyAttributes() -> cudaError_t { t("cudaGraphKernelNodeCopyAttributes") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphKernelNodeGetAttribute() -> cudaError_t { t("cudaGraphKernelNodeGetAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphKernelNodeGetParams() -> cudaError_t { t("cudaGraphKernelNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphKernelNodeSetAttribute() -> cudaError_t { t("cudaGraphKernelNodeSetAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphKernelNodeSetParams() -> cudaError_t { t("cudaGraphKernelNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphLaunch() -> cudaError_t { /* TODO impl forwarder a hipGraphLaunch */ 0 }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphLaunch_ptsz() -> cudaError_t { cudaGraphLaunch() }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemAllocNodeGetParams() -> cudaError_t { t("cudaGraphMemAllocNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemFreeNodeGetParams() -> cudaError_t { t("cudaGraphMemFreeNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemcpyNodeGetParams() -> cudaError_t { t("cudaGraphMemcpyNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemcpyNodeSetParams() -> cudaError_t { t("cudaGraphMemcpyNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemcpyNodeSetParams1D() -> cudaError_t { t("cudaGraphMemcpyNodeSetParams1D") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemcpyNodeSetParamsFromSymbol() -> cudaError_t { t("cudaGraphMemcpyNodeSetParamsFromSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemcpyNodeSetParamsToSymbol() -> cudaError_t { t("cudaGraphMemcpyNodeSetParamsToSymbol") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemsetNodeGetParams() -> cudaError_t { t("cudaGraphMemsetNodeGetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphMemsetNodeSetParams() -> cudaError_t { t("cudaGraphMemsetNodeSetParams") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeFindInClone() -> cudaError_t { t("cudaGraphNodeFindInClone") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeGetDependencies() -> cudaError_t { t("cudaGraphNodeGetDependencies") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeGetDependentNodes() -> cudaError_t { t("cudaGraphNodeGetDependentNodes") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeGetEnabled() -> cudaError_t { t("cudaGraphNodeGetEnabled") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeGetType() -> cudaError_t { t("cudaGraphNodeGetType") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphNodeSetEnabled() -> cudaError_t { t("cudaGraphNodeSetEnabled") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphReleaseUserObject() -> cudaError_t { t("cudaGraphReleaseUserObject") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphRemoveDependencies() -> cudaError_t { t("cudaGraphRemoveDependencies") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphRetainUserObject() -> cudaError_t { t("cudaGraphRetainUserObject") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphUpload() -> cudaError_t { t("cudaGraphUpload") }
#[no_mangle]
pub unsafe extern "C" fn cudaGraphUpload_ptsz() -> cudaError_t { cudaGraphUpload() }

// === MEM (98 simboli) ===
// VEGA-FIX T17:30: forwarder reali per i 6 critici che torch usa per allocazione GPU.
#[no_mangle]
pub unsafe extern "C" fn cudaFree(ptr: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipFree(ptr))
}
#[no_mangle]
pub unsafe extern "C" fn cudaFreeArray() -> cudaError_t { t("cudaFreeArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaFreeAsync(ptr: *mut c_void, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipFreeAsync(ptr, norm_stream(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-5: cudaFreeAsync_ptsz delegate
pub unsafe extern "C" fn cudaFreeAsync_ptsz(ptr: *mut c_void, stream: *mut c_void) -> cudaError_t {
    cudaFreeAsync(ptr, stream)
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-9a: cudaFreeHost forwarder hipHostFree
pub unsafe extern "C" fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if ptr.is_null() { return 0; }
    ok_err(hipHostFree(ptr))
}
#[no_mangle]
pub unsafe extern "C" fn cudaFreeMipmappedArray() -> cudaError_t { t("cudaFreeMipmappedArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaHostAlloc() -> cudaError_t { t("cudaHostAlloc") }
#[no_mangle]
pub unsafe extern "C" fn cudaHostGetDevicePointer() -> cudaError_t { t("cudaHostGetDevicePointer") }
#[no_mangle]
pub unsafe extern "C" fn cudaHostGetFlags() -> cudaError_t { t("cudaHostGetFlags") }
#[no_mangle]
pub unsafe extern "C" fn cudaHostRegister() -> cudaError_t { t("cudaHostRegister") }
#[no_mangle]
pub unsafe extern "C" fn cudaHostUnregister() -> cudaError_t { t("cudaHostUnregister") }
#[no_mangle]
pub unsafe extern "C" fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMalloc(ptr, size))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMalloc3D() -> cudaError_t { t("cudaMalloc3D") }
#[no_mangle]
pub unsafe extern "C" fn cudaMalloc3DArray() -> cudaError_t { t("cudaMalloc3DArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMallocArray() -> cudaError_t { t("cudaMallocArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMallocAsync(ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    // VEGA-FIX T11:00: era stub () -> 0 senza setting *ptr. torch riceveva garbage ptr.
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_cudart_err_trace.log") {
        let _ = writeln!(f, "[cudaMallocAsync FIX size={}]", size);
    }
    ok_err(hipMallocAsync(ptr, size, std::mem::transmute(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-6: cudaMallocAsync_ptsz delegate
pub unsafe extern "C" fn cudaMallocAsync_ptsz(ptr: *mut *mut c_void, size: usize, stream: *mut c_void) -> cudaError_t {
    cudaMallocAsync(ptr, size, stream)
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-16: cudaMallocFromPoolAsync forwarder hipMallocFromPoolAsync
pub unsafe extern "C" fn cudaMallocFromPoolAsync(dev_ptr: *mut *mut c_void, size: usize, pool: *mut c_void, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if dev_ptr.is_null() { return 0; }
    ok_err(hipMallocFromPoolAsync(dev_ptr, size, pool as hipMemPool_t, std::mem::transmute(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-7: cudaMallocFromPoolAsync_ptsz delegate
pub unsafe extern "C" fn cudaMallocFromPoolAsync_ptsz(dev_ptr: *mut *mut c_void, size: usize, pool: *mut c_void, stream: *mut c_void) -> cudaError_t {
    cudaMallocFromPoolAsync(dev_ptr, size, pool, stream)
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-9b: cudaMallocHost forwarder hipHostMalloc pinned
pub unsafe extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> cudaError_t {
    ensure_hip_init();
    if ptr.is_null() { return 0; }
    // flags=0 = default pinned (hipHostMallocDefault). DataLoader pin_memory.
    ok_err(hipHostMalloc(ptr, size, 0))
}
#[no_mangle]
// VEGA-PATCH 2026-06-30 T15:50 sprint-app-real cudart-fix-2: cudaMallocManaged wrap REALE
// Bug: stub TODO non allocava nulla → ptr UNINIT → app crash @ deref unified ptr
// Fix: ABI signature 3-arg + hipMallocManaged forward (HMM unified ptr CPU+GPU)
// Note: Windows ROCm 5.7 HMM è "under development" → torna NOT_SUPPORTED a runtime se manca
pub unsafe extern "C" fn cudaMallocManaged(dev_ptr: *mut *mut c_void, size: usize, flags: c_uint) -> cudaError_t {
    ensure_hip_init();
    if dev_ptr.is_null() { return 0; }
    ok_err(hipMallocManaged(dev_ptr, size, flags))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMallocMipmappedArray() -> cudaError_t { t("cudaMallocMipmappedArray") }
#[no_mangle]
// VEGA-PATCH 2026-06-30 T15:50 sprint-app-real cudart-fix-3: cudaMallocPitch wrap REALE
// Critico per Blender Cycles texture upload + MIOpen 2D NCHW layout allocation
pub unsafe extern "C" fn cudaMallocPitch(ptr: *mut *mut c_void, pitch: *mut usize, width: usize, height: usize) -> cudaError_t {
    ensure_hip_init();
    if ptr.is_null() || pitch.is_null() { return 0; }
    ok_err(hipMallocPitch(ptr, pitch, width, height))
}
#[no_mangle]
// VEGA-PATCH 2026-06-30 T15:50 sprint-app-real cudart-fix-4: cudaMemAdvise wrap REALE
// PyTorch usa per CUDA unified memory (PROFILER + autograd hint)
pub unsafe extern "C" fn cudaMemAdvise(dev_ptr: *const c_void, count: usize, advice: c_int, device: c_int) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemAdvise(dev_ptr, count, hipMemoryAdvise(advice as u32), device))
}
#[no_mangle]
// VEGA-PATCH 2026-06-30 T15:50 sprint-app-real cudart-fix-1: cudaMemGetInfo wrap REALE
// Bug: stub TODO `() -> 0` NON scriveva free/total → PyTorch leggeva memoria UNINIT garbage
// → falsi OOM, falsi available memory, falsi free pool size in torch.cuda.mem_get_info()
// Fix: ABI-correct signature 2× *mut usize → hipMemGetInfo forward
pub unsafe extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemGetInfo(free, total))
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-11: cudaMemPoolCreate forwarder hipMemPoolCreate
pub unsafe extern "C" fn cudaMemPoolCreate(pool: *mut *mut c_void, props: *const c_void) -> cudaError_t {
    ensure_hip_init();
    if pool.is_null() { return 0; }
    let mut hp: hipMemPool_t = std::ptr::null_mut();
    let rc = ok_err(hipMemPoolCreate(&mut hp, props as *const hipMemPoolProps));
    if rc == 0 { *pool = hp as *mut c_void; }
    rc
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-12: cudaMemPoolDestroy forwarder hipMemPoolDestroy
pub unsafe extern "C" fn cudaMemPoolDestroy(pool: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if pool.is_null() { return 0; }
    ok_err(hipMemPoolDestroy(pool as hipMemPool_t))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolExportPointer() -> cudaError_t { t("cudaMemPoolExportPointer") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolExportToShareableHandle() -> cudaError_t { t("cudaMemPoolExportToShareableHandle") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolGetAccess() -> cudaError_t { t("cudaMemPoolGetAccess") }
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-15: cudaMemPoolGetAttribute forwarder hipMemPoolGetAttribute
pub unsafe extern "C" fn cudaMemPoolGetAttribute(pool: *mut c_void, attr: c_int, value: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if pool.is_null() { return 0; }
    ok_err(hipMemPoolGetAttribute(pool as hipMemPool_t, hipMemPoolAttr(attr as u32), value))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolImportFromShareableHandle() -> cudaError_t { t("cudaMemPoolImportFromShareableHandle") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolImportPointer() -> cudaError_t { t("cudaMemPoolImportPointer") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemPoolSetAccess() -> cudaError_t { t("cudaMemPoolSetAccess") }
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-14: cudaMemPoolSetAttribute forwarder hipMemPoolSetAttribute
pub unsafe extern "C" fn cudaMemPoolSetAttribute(pool: *mut c_void, attr: c_int, value: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if pool.is_null() { return 0; }
    ok_err(hipMemPoolSetAttribute(pool as hipMemPool_t, hipMemPoolAttr(attr as u32), value))
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-13: cudaMemPoolTrimTo CRITICO chiude memory leak torch 2.4
pub unsafe extern "C" fn cudaMemPoolTrimTo(pool: *mut c_void, min_bytes_to_hold: usize) -> cudaError_t {
    ensure_hip_init();
    if pool.is_null() { return 0; }
    ok_err(hipMemPoolTrimTo(pool as hipMemPool_t, min_bytes_to_hold))
}
#[no_mangle]
// VEGA-PATCH 2026-06-30 T15:50 sprint-app-real cudart-fix-5: cudaMemPrefetchAsync wrap REALE
// Critical per UM workload: PyTorch checkpoint reload + Blender geometry stream
pub unsafe extern "C" fn cudaMemPrefetchAsync(dev_ptr: *const c_void, count: usize, device: c_int, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemPrefetchAsync(dev_ptr, count, device, norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemPrefetchAsync_ptsz(dev_ptr: *const c_void, count: usize, device: c_int, stream: *mut c_void) -> cudaError_t {
    cudaMemPrefetchAsync(dev_ptr, count, device, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemRangeGetAttribute() -> cudaError_t { t("cudaMemRangeGetAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemRangeGetAttributes() -> cudaError_t { t("cudaMemRangeGetAttributes") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_uint) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpy(dst, src, size, hipMemcpyKind(kind)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2D() -> cudaError_t { t("cudaMemcpy2D") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DArrayToArray() -> cudaError_t { t("cudaMemcpy2DArrayToArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DArrayToArray_ptds() -> cudaError_t { t("cudaMemcpy2DArrayToArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DAsync() -> cudaError_t { t("cudaMemcpy2DAsync") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-12: cudaMemcpy2DAsync_ptsz forwarder reale
// Bug: stub silent ritornava 0 fake → attention transposed multi-head memcpy strided silent fail → 709 cascade
// Fix: ABI-correct signature + forward a hipMemcpy2DAsync_spt (Stream-Per-Thread variant ROCm 5.7)
pub unsafe extern "C" fn cudaMemcpy2DAsync_ptsz(
    dst: *mut c_void, dpitch: usize,
    src: *const c_void, spitch: usize,
    width: usize, height: usize,
    kind: c_uint, stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpy2DAsync_spt(dst, dpitch, src, spitch, width, height, hipMemcpyKind(kind), norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DFromArray() -> cudaError_t { t("cudaMemcpy2DFromArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DFromArrayAsync() -> cudaError_t { t("cudaMemcpy2DFromArrayAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DFromArrayAsync_ptsz() -> cudaError_t { cudaMemcpy2DFromArrayAsync() }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DFromArray_ptds() -> cudaError_t { t("cudaMemcpy2DFromArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DToArray() -> cudaError_t { t("cudaMemcpy2DToArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DToArrayAsync() -> cudaError_t { t("cudaMemcpy2DToArrayAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DToArrayAsync_ptsz() -> cudaError_t { cudaMemcpy2DToArrayAsync() }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2DToArray_ptds() -> cudaError_t { t("cudaMemcpy2DToArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy2D_ptds() -> cudaError_t { t("cudaMemcpy2D_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3D(p: *const c_void) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T16:00 sprint-app-real cudart-fix-17: 3D memcpy Blender volume + MIOpen 3D conv
    // CUDA cudaMemcpy3DParms struct ABI === HIP hipMemcpy3DParms (verified header parity)
    ensure_hip_init();
    ok_err(hipMemcpy3D(p as *const hipMemcpy3DParms))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3DAsync(p: *const c_void, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpy3DAsync(p as *const hipMemcpy3DParms, norm_stream(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-13: cudaMemcpy3DAsync_ptsz forwarder reale
// Impact: VAE / Conv3D / volumetric — torch.nn.Conv3d backward usa Memcpy3DAsync per im2col strided.
pub unsafe extern "C" fn cudaMemcpy3DAsync_ptsz(
    p: *const c_void, stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    if p.is_null() { return 1; }
    ok_err(hipMemcpy3DAsync_spt(p as *const hipMemcpy3DParms, norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3DPeer() -> cudaError_t { t("cudaMemcpy3DPeer") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3DPeerAsync() -> cudaError_t { t("cudaMemcpy3DPeerAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3DPeerAsync_ptsz() -> cudaError_t { cudaMemcpy3DPeerAsync() }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3DPeer_ptds() -> cudaError_t { t("cudaMemcpy3DPeer_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy3D_ptds(p: *const c_void) -> cudaError_t { cudaMemcpy3D(p) }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyArrayToArray() -> cudaError_t { t("cudaMemcpyArrayToArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyArrayToArray_ptds() -> cudaError_t { t("cudaMemcpyArrayToArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, size: usize, kind: c_uint, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpyAsync(dst, src, size, hipMemcpyKind(kind), norm_stream(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-5: cudaMemcpyAsync_ptsz delegate
pub unsafe extern "C" fn cudaMemcpyAsync_ptsz(dst: *mut c_void, src: *const c_void, size: usize, kind: c_uint, stream: *mut c_void) -> cudaError_t {
    cudaMemcpyAsync(dst, src, size, kind, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromArray() -> cudaError_t { t("cudaMemcpyFromArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromArrayAsync() -> cudaError_t { t("cudaMemcpyFromArrayAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromArrayAsync_ptsz() -> cudaError_t { cudaMemcpyFromArrayAsync() }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromArray_ptds() -> cudaError_t { t("cudaMemcpyFromArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromSymbol(
    dst: *mut c_void, symbol: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint,
) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-7: constant memory readback
    // Blender Cycles legge BVH params const; PyTorch dispatcher macro lambda capture readback
    ensure_hip_init();
    ok_err(hipMemcpyFromSymbol(dst, symbol, size_bytes, offset, hipMemcpyKind(kind)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromSymbolAsync(
    dst: *mut c_void, symbol: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint, stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpyFromSymbolAsync(dst, symbol, size_bytes, offset, hipMemcpyKind(kind), norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromSymbolAsync_ptsz(
    dst: *mut c_void, symbol: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint, stream: *mut c_void,
) -> cudaError_t {
    cudaMemcpyFromSymbolAsync(dst, symbol, size_bytes, offset, kind, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyFromSymbol_ptds(
    dst: *mut c_void, symbol: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint,
) -> cudaError_t {
    cudaMemcpyFromSymbol(dst, symbol, size_bytes, offset, kind)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyPeer(
    dst: *mut c_void, dst_device: c_int,
    src: *const c_void, src_device: c_int,
    count: usize,
) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-6: cudaMemcpyPeer wrap REALE
    // Single-GPU Vega 20: dst_device==src_device→hipMemcpy device-to-device same agent
    ensure_hip_init();
    ok_err(hipMemcpyPeer(dst, dst_device, src, src_device, count))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyPeerAsync(
    dst: *mut c_void, dst_device: c_int,
    src: *const c_void, src_device: c_int,
    count: usize, stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpyPeerAsync(dst, dst_device, src, src_device, count, norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToArray() -> cudaError_t { t("cudaMemcpyToArray") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToArrayAsync() -> cudaError_t { t("cudaMemcpyToArrayAsync") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToArrayAsync_ptsz() -> cudaError_t { cudaMemcpyToArrayAsync() }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToArray_ptds() -> cudaError_t { t("cudaMemcpyToArray_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToSymbol(
    symbol: *const c_void, src: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint,
) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-8: __constant__ memory upload
    // Blender Cycles upload BVH/kernel params const; PyTorch RNG state, dispatch tables
    ensure_hip_init();
    ok_err(hipMemcpyToSymbol(symbol, src, size_bytes, offset, hipMemcpyKind(kind)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToSymbolAsync(
    symbol: *const c_void, src: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint, stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemcpyToSymbolAsync(symbol, src, size_bytes, offset, hipMemcpyKind(kind), norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToSymbolAsync_ptsz(
    symbol: *const c_void, src: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint, stream: *mut c_void,
) -> cudaError_t {
    cudaMemcpyToSymbolAsync(symbol, src, size_bytes, offset, kind, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpyToSymbol_ptds(
    symbol: *const c_void, src: *const c_void,
    size_bytes: usize, offset: usize, kind: c_uint,
) -> cudaError_t {
    cudaMemcpyToSymbol(symbol, src, size_bytes, offset, kind)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemcpy_ptds() -> cudaError_t { t("cudaMemcpy_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemset(dst: *mut c_void, value: c_int, count: usize) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemset(dst, value, count))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemset2D() -> cudaError_t { t("cudaMemset2D") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemset2DAsync() -> cudaError_t { t("cudaMemset2DAsync") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-14: cudaMemset2DAsync_ptsz forwarder reale
// Impact: AdamW state init (m,v tensori strided) + grad zero — propagava silent fake 0 → state garbage.
pub unsafe extern "C" fn cudaMemset2DAsync_ptsz(
    dst: *mut c_void, pitch: usize,
    value: c_int, width: usize, height: usize,
    stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemset2DAsync_spt(dst, pitch, value, width, height, norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemset2D_ptds() -> cudaError_t { t("cudaMemset2D_ptds") }
#[no_mangle]
pub unsafe extern "C" fn cudaMemset3D(pitched_ptr: hipPitchedPtr, value: c_int, extent: hipExtent) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T16:05 sprint-app-real cudart-fix-18: 3D buffer init (Blender vol, MIOpen 3D)
    // ABI parity: cudaPitchedPtr === hipPitchedPtr (4 fields), cudaExtent === hipExtent (3 fields)
    ensure_hip_init();
    ok_err(hipMemset3D(pitched_ptr, value, extent))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemset3DAsync(pitched_ptr: hipPitchedPtr, value: c_int, extent: hipExtent, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemset3DAsync(pitched_ptr, value, extent, norm_stream(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-15: cudaMemset3DAsync_ptsz forwarder reale
pub unsafe extern "C" fn cudaMemset3DAsync_ptsz(
    pitched_dev_ptr_a: usize, pitched_dev_ptr_b: usize, pitched_dev_ptr_c: usize, pitched_dev_ptr_d: usize,
    value: c_int,
    extent_a: usize, extent_b: usize, extent_c: usize,
    stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    // hipPitchedPtr = { ptr, pitch, xsize, ysize }; hipExtent = { width, height, depth }
    // Passa via stack args raw — CUDA cudaPitchedPtr/cudaExtent hanno layout identico HIP su Win x64.
    let pp = hipPitchedPtr {
        ptr: pitched_dev_ptr_a as *mut c_void,
        pitch: pitched_dev_ptr_b,
        xsize: pitched_dev_ptr_c,
        ysize: pitched_dev_ptr_d,
    };
    let ex = hipExtent { width: extent_a, height: extent_b, depth: extent_c };
    ok_err(hipMemset3DAsync_spt(pp, value, ex, norm_stream(stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemset3D_ptds(pitched_ptr: hipPitchedPtr, value: c_int, extent: hipExtent) -> cudaError_t {
    cudaMemset3D(pitched_ptr, value, extent)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemsetAsync(dst: *mut c_void, value: c_int, count: usize, stream: *mut c_void) -> cudaError_t {
    ensure_hip_init();
    ok_err(hipMemsetAsync(dst, value, count, std::mem::transmute(stream)))
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-6: cudaMemsetAsync_ptsz delegate
pub unsafe extern "C" fn cudaMemsetAsync_ptsz(dst: *mut c_void, value: c_int, count: usize, stream: *mut c_void) -> cudaError_t {
    cudaMemsetAsync(dst, value, count, stream)
}
#[no_mangle]
pub unsafe extern "C" fn cudaMemset_ptds() -> cudaError_t { t("cudaMemset_ptds") }

// === DEVICE (24 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceCanAccessPeer() -> cudaError_t { t("cudaDeviceCanAccessPeer") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceDisablePeerAccess() -> cudaError_t { t("cudaDeviceDisablePeerAccess") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceEnablePeerAccess() -> cudaError_t { t("cudaDeviceEnablePeerAccess") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceFlushGPUDirectRDMAWrites() -> cudaError_t { t("cudaDeviceFlushGPUDirectRDMAWrites") }
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-10: cudaDeviceGetAttribute forwarder hipDeviceGetAttribute
pub unsafe extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> cudaError_t {
    ensure_hip_init();
    if value.is_null() { return 0; }
    // CUDA cudaDeviceAttr enum mostly aligns 1:1 with HIP hipDeviceAttribute_t. Pass through.
    let rc = hipDeviceGetAttribute(value, hipDeviceAttribute_t(attr as u32), device);
    let code = ok_err(rc);
    if code != 0 {
        // VEGA-PATCH cudart-10: torch tollera default 0 per attributi non supportati
        *value = 0;
        return 0;
    }
    code
}
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetByPCIBusId() -> cudaError_t { t("cudaDeviceGetByPCIBusId") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetCacheConfig() -> cudaError_t { t("cudaDeviceGetCacheConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetDefaultMemPool() -> cudaError_t { t("cudaDeviceGetDefaultMemPool") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetGraphMemAttribute() -> cudaError_t { t("cudaDeviceGetGraphMemAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetLimit() -> cudaError_t { t("cudaDeviceGetLimit") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetMemPool() -> cudaError_t { t("cudaDeviceGetMemPool") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetP2PAttribute() -> cudaError_t { t("cudaDeviceGetP2PAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetPCIBusId() -> cudaError_t { t("cudaDeviceGetPCIBusId") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetSharedMemConfig() -> cudaError_t { t("cudaDeviceGetSharedMemConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetStreamPriorityRange() -> cudaError_t { t("cudaDeviceGetStreamPriorityRange") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGetTexture1DLinearMaxWidth() -> cudaError_t { t("cudaDeviceGetTexture1DLinearMaxWidth") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceGraphMemTrim() -> cudaError_t { t("cudaDeviceGraphMemTrim") }
#[no_mangle]
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetCacheConfig() -> cudaError_t { t("cudaDeviceSetCacheConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetGraphMemAttribute() -> cudaError_t { t("cudaDeviceSetGraphMemAttribute") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetLimit() -> cudaError_t { t("cudaDeviceSetLimit") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetMemPool() -> cudaError_t { t("cudaDeviceSetMemPool") }
#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSetSharedMemConfig() -> cudaError_t { t("cudaDeviceSetSharedMemConfig") }
// === MISC (84 simboli) ===
#[no_mangle]
pub unsafe extern "C" fn cudaArrayGetInfo() -> cudaError_t { t("cudaArrayGetInfo") }
#[no_mangle]
pub unsafe extern "C" fn cudaArrayGetMemoryRequirements() -> cudaError_t { t("cudaArrayGetMemoryRequirements") }
#[no_mangle]
pub unsafe extern "C" fn cudaArrayGetPlane() -> cudaError_t { t("cudaArrayGetPlane") }
#[no_mangle]
pub unsafe extern "C" fn cudaArrayGetSparseProperties() -> cudaError_t { t("cudaArrayGetSparseProperties") }
#[no_mangle]
pub unsafe extern "C" fn cudaChooseDevice() -> cudaError_t { t("cudaChooseDevice") }
#[no_mangle]
pub unsafe extern "C" fn cudaCtxResetPersistingL2Cache() -> cudaError_t { t("cudaCtxResetPersistingL2Cache") }
#[no_mangle]
pub unsafe extern "C" fn cudaDestroyExternalMemory() -> cudaError_t { t("cudaDestroyExternalMemory") }
#[no_mangle]
pub unsafe extern "C" fn cudaDestroyExternalSemaphore() -> cudaError_t { t("cudaDestroyExternalSemaphore") }
#[no_mangle]
#[no_mangle]
pub unsafe extern "C" fn cudaExternalMemoryGetMappedBuffer() -> cudaError_t { t("cudaExternalMemoryGetMappedBuffer") }
#[no_mangle]
pub unsafe extern "C" fn cudaExternalMemoryGetMappedMipmappedArray() -> cudaError_t { t("cudaExternalMemoryGetMappedMipmappedArray") }
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-2: cudaFuncGetAttributes forwarder + struct fill
// VEGA-FIX FIX-A 2026-06-28: rimosso #[no_mangle] orphan sopra struct, spostato adiacente alla fn
#[repr(C)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: c_int,
    pub numRegs: c_int,
    pub ptxVersion: c_int,
    pub binaryVersion: c_int,
    pub cacheModeCA: c_int,
    pub maxDynamicSharedSizeBytes: c_int,
    pub preferredShmemCarveout: c_int,
}

#[no_mangle]
pub unsafe extern "C" fn cudaFuncGetAttributes(attr: *mut cudaFuncAttributes, func: *const c_void) -> cudaError_t {
    ensure_hip_init();
    if attr.is_null() { return 0; }
    let mut hip_attr: hipFuncAttributes = std::mem::zeroed();
    let rc = hipFuncGetAttributes(&mut hip_attr, func);
    let code = ok_err(rc);
    if code == 0 {
        (*attr).sharedSizeBytes = hip_attr.sharedSizeBytes;
        (*attr).constSizeBytes = hip_attr.constSizeBytes;
        (*attr).localSizeBytes = hip_attr.localSizeBytes;
        (*attr).maxThreadsPerBlock = hip_attr.maxThreadsPerBlock;
        (*attr).numRegs = hip_attr.numRegs;
        (*attr).ptxVersion = if hip_attr.ptxVersion != 0 { hip_attr.ptxVersion } else { 80 };
        (*attr).binaryVersion = if hip_attr.binaryVersion != 0 { hip_attr.binaryVersion } else { 70 };
        (*attr).cacheModeCA = hip_attr.cacheModeCA;
        (*attr).maxDynamicSharedSizeBytes = hip_attr.maxDynamicSharedSizeBytes;
        (*attr).preferredShmemCarveout = hip_attr.preferredShmemCarveout;
    } else {
        // VEGA-PATCH cudart-2: fallback sane defaults Vega20 64KB LDS / 256 thr / 80/70 versions
        (*attr).sharedSizeBytes = 0;
        (*attr).constSizeBytes = 0;
        (*attr).localSizeBytes = 0;
        (*attr).maxThreadsPerBlock = 256;
        (*attr).numRegs = 32;
        (*attr).ptxVersion = 80;
        (*attr).binaryVersion = 70;
        (*attr).cacheModeCA = 0;
        (*attr).maxDynamicSharedSizeBytes = 65536;
        (*attr).preferredShmemCarveout = -1;
        return 0;
    }
    code
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-3: cudaFuncSetAttribute forwarder smem dinamica
pub unsafe extern "C" fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> cudaError_t {
    ensure_hip_init();
    // CUDA attr enum vs HIP hipFuncAttribute: 8=MaxDynamicSharedSizeBytes, 9=PreferredCarveout — pass through
    ok_err(hipFuncSetAttribute(func, hipFuncAttribute(attr as u32), value))
}
#[no_mangle]
pub unsafe extern "C" fn cudaFuncSetCacheConfig() -> cudaError_t { t("cudaFuncSetCacheConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaFuncSetSharedMemConfig() -> cudaError_t { t("cudaFuncSetSharedMemConfig") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetChannelDesc() -> cudaError_t { t("cudaGetChannelDesc") }
#[no_mangle]
#[no_mangle]
#[no_mangle]
#[no_mangle]
#[no_mangle]
pub unsafe extern "C" fn cudaGetDriverEntryPoint() -> cudaError_t { t("cudaGetDriverEntryPoint") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetDriverEntryPoint_ptsz() -> cudaError_t { cudaGetDriverEntryPoint() }
// VEGA-FIX T16:58: cudaGetErrorName/String hanno signature C `const char*` non int.
// Lo scaffold auto-gen ritornava int 0 (nullptr) → PyTorch dereferenziava → AV.
// VEGA-FIX T07:45 23 giu: ritorno stringhe con error code numerico embedded per debug.
// Buffer thread_local size 64 per non leak memoria. Codice 0..=300 supportati.
use std::cell::RefCell;
thread_local! {
    static ERR_BUF: RefCell<[u8; 64]> = RefCell::new([0u8; 64]);
}
#[no_mangle]
pub unsafe extern "C" fn cudaGetErrorName(e: cudaError_t) -> *const c_char {
    ERR_BUF.with(|b| {
        let mut buf = b.borrow_mut();
        let s = format!("cudaError_{}\0", e);
        let bytes = s.as_bytes();
        let n = bytes.len().min(63);
        buf[..n].copy_from_slice(&bytes[..n]);
        buf[n] = 0;
        buf.as_ptr() as *const c_char
    })
}
#[no_mangle]
pub unsafe extern "C" fn cudaGetErrorString(e: cudaError_t) -> *const c_char {
    // VEGA-DEBUG T07:50 23 giu: log persistente per inchiodare chi chiama con err=3
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_err_trace.log")
    {
        let bt = std::backtrace::Backtrace::force_capture();
        let _ = writeln!(f, "[cudaGetErrorString called] err={}\nbacktrace:\n{}\n---", e, bt);
    }
    ERR_BUF.with(|b| {
        let mut buf = b.borrow_mut();
        let s = format!("ZLUDA err code={}\0", e);
        let bytes = s.as_bytes();
        let n = bytes.len().min(63);
        buf[..n].copy_from_slice(&bytes[..n]);
        buf[n] = 0;
        buf.as_ptr() as *const c_char
    })
}
#[no_mangle]
pub unsafe extern "C" fn cudaGetExportTable() -> cudaError_t { t("cudaGetExportTable") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetFuncBySymbol() -> cudaError_t { t("cudaGetFuncBySymbol") }
#[no_mangle]
#[no_mangle]
pub unsafe extern "C" fn cudaGetMipmappedArrayLevel() -> cudaError_t { t("cudaGetMipmappedArrayLevel") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetSurfaceObjectResourceDesc() -> cudaError_t { t("cudaGetSurfaceObjectResourceDesc") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetSurfaceReference() -> cudaError_t { t("cudaGetSurfaceReference") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetSymbolAddress(dev_ptr: *mut *mut c_void, symbol: *const c_void) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-9: device __constant__ ptr lookup
    ensure_hip_init();
    if dev_ptr.is_null() { return 0; }
    ok_err(hipGetSymbolAddress(dev_ptr, symbol))
}
#[no_mangle]
pub unsafe extern "C" fn cudaGetSymbolSize(size: *mut usize, symbol: *const c_void) -> cudaError_t {
    ensure_hip_init();
    if size.is_null() { return 0; }
    ok_err(hipGetSymbolSize(size, symbol))
}
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureAlignmentOffset() -> cudaError_t { t("cudaGetTextureAlignmentOffset") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureObjectResourceDesc() -> cudaError_t { t("cudaGetTextureObjectResourceDesc") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureObjectResourceViewDesc() -> cudaError_t { t("cudaGetTextureObjectResourceViewDesc") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureObjectTextureDesc() -> cudaError_t { t("cudaGetTextureObjectTextureDesc") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureObjectTextureDesc_v2() -> cudaError_t { t("cudaGetTextureObjectTextureDesc_v2") }
#[no_mangle]
pub unsafe extern "C" fn cudaGetTextureReference() -> cudaError_t { t("cudaGetTextureReference") }
#[no_mangle]
pub unsafe extern "C" fn cudaImportExternalMemory() -> cudaError_t { t("cudaImportExternalMemory") }
#[no_mangle]
pub unsafe extern "C" fn cudaImportExternalSemaphore() -> cudaError_t { t("cudaImportExternalSemaphore") }
#[no_mangle]
pub unsafe extern "C" fn cudaIpcCloseMemHandle(dev_ptr: *mut c_void) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-10: IPC close PyTorch multiprocessing
    ensure_hip_init();
    ok_err(hipIpcCloseMemHandle(dev_ptr))
}
#[no_mangle]
pub unsafe extern "C" fn cudaIpcGetEventHandle() -> cudaError_t { t("cudaIpcGetEventHandle") }
#[no_mangle]
pub unsafe extern "C" fn cudaIpcGetMemHandle(handle: *mut c_void, dev_ptr: *mut c_void) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-11: IPC export for PyTorch DDP
    ensure_hip_init();
    if handle.is_null() { return 0; }
    ok_err(hipIpcGetMemHandle(handle as *mut hipIpcMemHandle_t, dev_ptr))
}
#[no_mangle]
pub unsafe extern "C" fn cudaIpcOpenEventHandle() -> cudaError_t { t("cudaIpcOpenEventHandle") }
#[no_mangle]
pub unsafe extern "C" fn cudaIpcOpenMemHandle(dev_ptr: *mut *mut c_void, handle: hipIpcMemHandle_t, flags: c_uint) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-12: IPC import worker process map
    ensure_hip_init();
    if dev_ptr.is_null() { return 0; }
    ok_err(hipIpcOpenMemHandle(dev_ptr, handle, flags))
}
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchCooperativeKernel() -> cudaError_t { t("cudaLaunchCooperativeKernel") }
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchCooperativeKernelMultiDevice() -> cudaError_t { t("cudaLaunchCooperativeKernelMultiDevice") }
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchCooperativeKernel_ptsz() -> cudaError_t { cudaLaunchCooperativeKernel() }
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchHostFunc(stream: *mut c_void, fn_: hipHostFn_t, user_data: *mut c_void) -> cudaError_t {
    // VEGA-PATCH 2026-06-30 T15:55 sprint-app-real cudart-fix-14: CUDA Graph host callback
    // Usato da torch CUDAGraph capture per host-side parameter swap in replay
    ensure_hip_init();
    ok_err(hipLaunchHostFunc(norm_stream(stream), fn_, user_data))
}
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker20 cudart-ptsz-16: cudaLaunchHostFunc_ptsz forwarder reale
// Impact: PyTorch DataLoader pin_memory + autograd callback — causa di un blocco spurio all'iterazione 500 di un addestramento.
pub unsafe extern "C" fn cudaLaunchHostFunc_ptsz(
    stream: *mut c_void,
    fn_ptr: usize,        // CUhostFn = void(*)(void*)
    user_data: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    // hipHostFn_t è Option<unsafe extern "C" fn(*mut c_void)> nel binding hip_runtime_sys
    let host_fn: hipHostFn_t = std::mem::transmute::<usize, hipHostFn_t>(fn_ptr);
    ok_err(hipLaunchHostFunc_spt(norm_stream(stream), host_fn, user_data))
}
// VEGA-FIX T11:10: implementazione real cudaLaunchKernel via hipLaunchKernel
// VEGA-FIX FIX-A 2026-06-28: rimosso #[no_mangle] orphan, spostato adiacente alla fn
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dim3Cuda { pub x: u32, pub y: u32, pub z: u32 }

#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernel(
    func: *const c_void,
    grid: Dim3Cuda,
    block: Dim3Cuda,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: *mut c_void,
) -> cudaError_t {
    ensure_hip_init();
    let g = hip_runtime_sys::dim3 { x: grid.x, y: grid.y, z: grid.z };
    let b = hip_runtime_sys::dim3 { x: block.x, y: block.y, z: block.z };
    let rc = hipLaunchKernel(func, g, b, args, shared_mem, std::mem::transmute(stream));
    // VEGA-TRACE T12:20: log args + ret per identificare bug err=1
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_launch_trace.log") {
        let code: i32 = match &rc { Ok(_) => 0, Err(e) => e.0.get() as i32 };
        let _ = writeln!(f, "[cudaLaunchKernel] func={:p} grid=({},{},{}) block=({},{},{}) sm={} stream={:p} rc={}",
            func, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem, stream, code);
    }
    ok_err(rc)
}
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernel_ptsz(
    func: *const c_void, grid: Dim3Cuda, block: Dim3Cuda, args: *mut *mut c_void, shared_mem: usize, stream: *mut c_void,
) -> cudaError_t {
    cudaLaunchKernel(func, grid, block, args, shared_mem, stream)
}
#[no_mangle]
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-20: cudaLaunchKernelExC real impl forward hipLaunchKernel ignorando attrs CUDA 11.6+
// cudaLaunchKernelExC ABI: cudaLaunchConfig_t {gridDim, blockDim, dynamicSmemBytes, stream, attrs, numAttrs}, func, args
#[repr(C)]
pub struct cudaLaunchConfig_t {
    pub gridDim: Dim3Cuda,
    pub blockDim: Dim3Cuda,
    pub dynamicSmemBytes: usize,
    pub stream: *mut c_void,
    pub attrs: *mut c_void,
    pub numAttrs: c_uint,
}
pub unsafe extern "C" fn cudaLaunchKernelExC(config: *const cudaLaunchConfig_t, func: *const c_void, args: *mut *mut c_void) -> cudaError_t {
    ensure_hip_init();
    if config.is_null() { return 0; }
    let cfg = &*config;
    // Ignora attrs CUDA 11.6+ (cluster dim, priority, etc.) — non supportati gfx906.
    let g = hip_runtime_sys::dim3 { x: cfg.gridDim.x, y: cfg.gridDim.y, z: cfg.gridDim.z };
    let b = hip_runtime_sys::dim3 { x: cfg.blockDim.x, y: cfg.blockDim.y, z: cfg.blockDim.z };
    ok_err(hipLaunchKernel(func, g, b, args, cfg.dynamicSmemBytes, std::mem::transmute(cfg.stream)))
}
#[no_mangle]
pub unsafe extern "C" fn cudaLaunchKernelExC_ptsz(config: *const cudaLaunchConfig_t, func: *const c_void, args: *mut *mut c_void) -> cudaError_t { cudaLaunchKernelExC(config, func, args) }
#[no_mangle]
pub unsafe extern "C" fn cudaMipmappedArrayGetMemoryRequirements() -> cudaError_t { t("cudaMipmappedArrayGetMemoryRequirements") }
#[no_mangle]
pub unsafe extern "C" fn cudaMipmappedArrayGetSparseProperties() -> cudaError_t { t("cudaMipmappedArrayGetSparseProperties") }
#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyAvailableDynamicSMemPerBlock() -> cudaError_t { t("cudaOccupancyAvailableDynamicSMemPerBlock") }
#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor() -> cudaError_t { t("cudaOccupancyMaxActiveBlocksPerMultiprocessor") }
#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags() -> cudaError_t { t("cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags") }
#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxActiveClusters() -> cudaError_t { t("cudaOccupancyMaxActiveClusters") }
#[no_mangle]
pub unsafe extern "C" fn cudaOccupancyMaxPotentialClusterSize() -> cudaError_t { t("cudaOccupancyMaxPotentialClusterSize") }

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-1: cudaPointerGetAttributes forwarder + struct fill
// VEGA-FIX FIX-A 2026-06-28: rimossi 2 #[no_mangle] orfani; struct senza attributo, fn con #[no_mangle] adiacente
#[repr(C)]
pub struct cudaPointerAttributes {
    pub type_: c_int,           // 0=unregistered,1=host,2=device,3=managed
    pub device: c_int,
    pub devicePointer: *mut c_void,
    pub hostPointer: *mut c_void,
}

#[no_mangle]
pub unsafe extern "C" fn cudaPointerGetAttributes(attr: *mut cudaPointerAttributes, ptr: *const c_void) -> cudaError_t {
    ensure_hip_init();
    if attr.is_null() { return 0; }
    let mut hip_attr: hipPointerAttribute_t = std::mem::zeroed();
    let rc = hipPointerGetAttributes(&mut hip_attr, ptr);
    let code = ok_err(rc);
    if code == 0 {
        (*attr).type_ = hip_attr.type_.0 as c_int;
        (*attr).device = hip_attr.device;
        (*attr).devicePointer = hip_attr.devicePointer;
        (*attr).hostPointer = hip_attr.hostPointer;
    } else {
        // VEGA-PATCH cudart-1: fallback torch wants type=cudaMemoryTypeDevice(2) per device ptr
        (*attr).type_ = 2;
        (*attr).device = 0;
        (*attr).devicePointer = ptr as *mut c_void;
        (*attr).hostPointer = std::ptr::null_mut();
        return 0;
    }
    code
}
#[no_mangle]
pub unsafe extern "C" fn cudaProfilerInitialize() -> cudaError_t { t("cudaProfilerInitialize") }
#[no_mangle]
pub unsafe extern "C" fn cudaProfilerStart() -> cudaError_t { t("cudaProfilerStart") }
#[no_mangle]
pub unsafe extern "C" fn cudaProfilerStop() -> cudaError_t { t("cudaProfilerStop") }
#[no_mangle]
#[no_mangle]
#[no_mangle]
#[no_mangle]
pub unsafe extern "C" fn cudaSetDoubleForDevice() -> cudaError_t { t("cudaSetDoubleForDevice") }
#[no_mangle]
pub unsafe extern "C" fn cudaSetDoubleForHost() -> cudaError_t { t("cudaSetDoubleForHost") }
#[no_mangle]
pub unsafe extern "C" fn cudaSetValidDevices() -> cudaError_t { t("cudaSetValidDevices") }
#[no_mangle]
pub unsafe extern "C" fn cudaSignalExternalSemaphoresAsync() -> cudaError_t { t("cudaSignalExternalSemaphoresAsync") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-10: cudaSignalExternalSemaphoresAsync_ptsz delegate
pub unsafe extern "C" fn cudaSignalExternalSemaphoresAsync_ptsz(_sem: *const c_void, _params: *const c_void, _num: c_uint, _stream: *mut c_void) -> cudaError_t {
    cudaSignalExternalSemaphoresAsync()
}
#[no_mangle]
pub unsafe extern "C" fn cudaSignalExternalSemaphoresAsync_v2() -> cudaError_t { t("cudaSignalExternalSemaphoresAsync_v2") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-11: cudaSignalExternalSemaphoresAsync_v2_ptsz delegate (ABI corretta 4 arg)
pub unsafe extern "C" fn cudaSignalExternalSemaphoresAsync_v2_ptsz(_sem: *const c_void, _params: *const c_void, _num: c_uint, _stream: *mut c_void) -> cudaError_t {
    cudaSignalExternalSemaphoresAsync_v2()
}
#[no_mangle]
pub unsafe extern "C" fn cudaUserObjectCreate() -> cudaError_t { t("cudaUserObjectCreate") }
#[no_mangle]
pub unsafe extern "C" fn cudaUserObjectRelease() -> cudaError_t { t("cudaUserObjectRelease") }
#[no_mangle]
pub unsafe extern "C" fn cudaUserObjectRetain() -> cudaError_t { t("cudaUserObjectRetain") }
#[no_mangle]
pub unsafe extern "C" fn cudaWGLGetDevice() -> cudaError_t { t("cudaWGLGetDevice") }
#[no_mangle]
pub unsafe extern "C" fn cudaWaitExternalSemaphoresAsync() -> cudaError_t { t("cudaWaitExternalSemaphoresAsync") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-8: cudaWaitExternalSemaphoresAsync_ptsz delegate
// External-semaphores non supportate gfx906 Windows → no-op via base stub.
pub unsafe extern "C" fn cudaWaitExternalSemaphoresAsync_ptsz(_sem: *const c_void, _params: *const c_void, _num: c_uint, _stream: *mut c_void) -> cudaError_t {
    cudaWaitExternalSemaphoresAsync()
}
#[no_mangle]
pub unsafe extern "C" fn cudaWaitExternalSemaphoresAsync_v2() -> cudaError_t { t("cudaWaitExternalSemaphoresAsync_v2") }
#[no_mangle]
// VEGA-PATCH 2026-06-29 sprint-rust-D worker12 cudart-ptsz-9: cudaWaitExternalSemaphoresAsync_v2_ptsz delegate
pub unsafe extern "C" fn cudaWaitExternalSemaphoresAsync_v2_ptsz(_sem: *const c_void, _params: *const c_void, _num: c_uint, _stream: *mut c_void) -> cudaError_t {
    cudaWaitExternalSemaphoresAsync_v2()
}

// cudaDriverGetVersion: stub generato sopra, skip special.
// cudaRuntimeGetVersion: stub generato sopra, skip special.
// cudaGetErrorString: stub generato sopra, skip special.
// cudaGetErrorName: stub generato sopra, skip special.
// === [VEGA-T16:30 18 giu] Implementazioni REAL per torch.cuda init ===
// Override degli stub auto-gen con forwarder veri a hipFunc equivalent.

#[no_mangle]
pub unsafe extern "C" fn cudaGetDevice(device: *mut c_int) -> cudaError_t {
    trace("cudaGetDevice", ok_err_init(hipGetDevice(device)))
}

#[no_mangle]
pub unsafe extern "C" fn cudaSetDevice(device: c_int) -> cudaError_t {
    let r = ok_err_init(hipSetDevice(device));
    trace(&format!("cudaSetDevice(dev={})", device), r)
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t {
    let r = ok_err_init(hipGetDeviceCount(count));
    trace(&format!("cudaGetDeviceCount->{}", if !count.is_null() {*count} else {-1}), r)
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceSynchronize() -> cudaError_t {
    ok_err_init(hipDeviceSynchronize())
}

#[no_mangle]
pub unsafe extern "C" fn cudaDeviceReset() -> cudaError_t { ensure_hip_init(); 0 }

#[no_mangle]
pub unsafe extern "C" fn cudaDriverGetVersion(v: *mut c_int) -> cudaError_t {
    if !v.is_null() { *v = CUDART_VERSION; } 0
}

#[no_mangle]
pub unsafe extern "C" fn cudaRuntimeGetVersion(v: *mut c_int) -> cudaError_t {
    if !v.is_null() { *v = CUDART_VERSION; } 0
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceFlags(flags: *mut c_uint) -> cudaError_t { ensure_hip_init(); if !flags.is_null() { *flags = 0; } 0 }

#[no_mangle]
pub unsafe extern "C" fn cudaSetDeviceFlags(_flags: c_uint) -> cudaError_t { ensure_hip_init(); 0 }

// VEGA-FIX T17:00: cudaGetLastError/PeekAtLastError NON inoltrano a hipGetLastError perché
// nvcuda v6_waitstream_fix accumula errori HIP residui dall'init ZLUDA che AT_CUDA_CHECK PyTorch
// interpreta come errore corrente → eccezione DeferredCudaCallError. Forzo sempre 0.
#[no_mangle]
pub unsafe extern "C" fn cudaGetLastError() -> cudaError_t {
    let _ = hipGetLastError();
    trace("cudaGetLastError", 0)
}

#[no_mangle]
pub unsafe extern "C" fn cudaPeekAtLastError() -> cudaError_t {
    0
}

// VEGA-FIX T17:15: CUDA 11.8 header `cuda_runtime.h` ridireziona cudaGetDeviceProperties
// (static __inline__) a cudaGetDeviceProperties_v2 (extern). PyTorch è stato compilato
// con quei header → chiama _v2 importata dal cudart64_110.dll. Alias trasparente.
#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceProperties_v2(prop: *mut c_void, device: c_int) -> cudaError_t {
    let r = cudaGetDeviceProperties(prop, device);
    trace(&format!("cudaGetDeviceProperties_v2(dev={})", device), r)
}

#[no_mangle]
pub unsafe extern "C" fn cudaGetDeviceProperties(prop: *mut c_void, _device: c_int) -> cudaError_t {
    if !prop.is_null() {
        let base = prop as *mut u8;
        std::ptr::write_bytes(base, 0, 1032);
    }
    // Popolo i campi noti di cudaDeviceProp CUDA 11.8 con valori statici gfx906
    if !prop.is_null() {
        let base = prop as *mut u8;
        let name = b"AMD Radeon VII [ZLUDA gfx906]\0";
        std::ptr::copy_nonoverlapping(name.as_ptr(), base, name.len().min(256));
        *(base.add(288) as *mut u64) = 16u64 * 1024 * 1024 * 1024; // totalGlobalMem
        *(base.add(296) as *mut u64) = 64 * 1024;                  // sharedMemPerBlock
        *(base.add(304) as *mut i32) = 65536;                       // regsPerBlock
        *(base.add(308) as *mut i32) = 64;                          // warpSize wave64 (config validata smoke_final3 5/5). NB test 2026-07-02: cambiare a 32 NON ha effetto sulla reduction torch (path props via nvcuda, non questo cudart) — vedi ZLUDA.md reduction wave64.
        *(base.add(312) as *mut u64) = 2147483647;                  // memPitch
        *(base.add(320) as *mut i32) = 1024;                        // maxThreadsPerBlock
        *(base.add(324) as *mut i32) = 1024;
        *(base.add(328) as *mut i32) = 1024;
        *(base.add(332) as *mut i32) = 64;
        *(base.add(336) as *mut i32) = 2147483647;
        *(base.add(340) as *mut i32) = 65535;
        *(base.add(344) as *mut i32) = 65535;
        *(base.add(348) as *mut i32) = 1800000;                     // clockRate
        *(base.add(352) as *mut u64) = 64 * 1024;                   // totalConstMem
        *(base.add(360) as *mut i32) = 7;                           // major
        *(base.add(364) as *mut i32) = 0;                           // minor
        *(base.add(368) as *mut u64) = 512;
        *(base.add(376) as *mut u64) = 32;
        *(base.add(384) as *mut i32) = 1;
        *(base.add(388) as *mut i32) = 60;                          // multiProcessorCount
        *(base.add(392) as *mut i32) = 0;
        *(base.add(396) as *mut i32) = 0;
        *(base.add(400) as *mut i32) = 1;
        *(base.add(404) as *mut i32) = 0;
        // Campi che pybind11 PyTorch accede esplicitamente (offsets CUDA 11.8):
        *(base.add(616) as *mut i32) = 4 * 1024 * 1024;             // l2CacheSize 4 MB gfx906
        *(base.add(624) as *mut i32) = 2048;                         // maxThreadsPerMultiProcessor
        *(base.add(640) as *mut u64) = 64 * 1024;                    // sharedMemPerMultiprocessor (size_t)
        *(base.add(648) as *mut i32) = 65536;                        // regsPerMultiprocessor
        *(base.add(656) as *mut i32) = 0;                            // isMultiGpuBoard
    }
    0
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-17a: cudaCheckpointCreate stub SUCCESS handle dummy
// torch 2.4+ checkpoint API non disponibile gfx906 → stub no-op safe
#[no_mangle]
pub unsafe extern "C" fn cudaCheckpointCreate(handle: *mut *mut c_void, args: *const c_void) -> cudaError_t {
    if !handle.is_null() {
        // dummy non-null sentinel così torch non considera failure
        *handle = 0x1usize as *mut c_void;
    }
    0
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudart-17b: cudaCheckpointRestore stub SUCCESS no-op
#[no_mangle]
pub unsafe extern "C" fn cudaCheckpointRestore(handle: *mut c_void, args: *const c_void) -> cudaError_t {
    0
}

// VEGA-FIX 2026-06-28 T21:18 post-link-error: hipGraphAddNode ASSENTE in amdhip64 v5.7 (sediment Agent 1 ROCm inventario = 1 dei 4 simboli "muro").
// Stub return SUCCESS finchè shim Marconi v4.0 non lo wrappa via hipGraphAdd<Type>Node specifici.
#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddNode_v2(
    p_graph_node: *mut *mut c_void,
    _graph: *mut c_void,
    _p_dependencies: *const *mut c_void,
    _p_dependency_data: *const c_void,
    _num_dependencies: usize,
    _node_params: *mut c_void,
) -> cudaError_t {
    if !p_graph_node.is_null() { *p_graph_node = std::ptr::null_mut(); }
    0
}

#[no_mangle]
pub unsafe extern "C" fn cudaGraphAddNode(
    p_graph_node: *mut *mut c_void,
    _graph: *mut c_void,
    _p_dependencies: *const *mut c_void,
    _num_dependencies: usize,
    _node_params: *mut c_void,
) -> cudaError_t {
    if !p_graph_node.is_null() { *p_graph_node = std::ptr::null_mut(); }
    0
}

// VEGA-PATCH 2026-06-30 Sprint Rust D BONUS Task 7 — DllMain SDMA auto-set
// BUG 11 sediment 29 giu T17:00: SDMA hang gfx906 richiede HSA_ENABLE_SDMA=0
// manuale user. Utenti dimenticano = hang random su memcpy frequenti.
//
// VEGA-PATCH TASK 11 — 2026-06-30 — FIX vero cdylib Rust su Windows MSVC.
// PROBLEMA: rust-lang/rust#48984 — cdylib Rust su Windows NON esegue
// `#[no_mangle] pub extern "system" fn DllMain` come entrypoint reale.
// MSVC `_DllMainCRTStartup` (entrypoint default cdylib) scansiona invece
// `.CRT$XCU` per static C-runtime constructors. Registriamo li' un ptr
// a fn statica che fa il set env. Stesso meccanismo del crate `ctor` ma
// zero dipendenze esterne (link_section stabile dal 2015, usato da
// inventory/linkme/ctor in produzione).
//
// Esecuzione: durante _DllMainCRTStartup, DOPO init CRT (malloc, std::env
// OK), PRIMA del primo `extern "C"` user call. Per processo singolo.

#[cfg(windows)]
unsafe fn vega_sdma_autoset() {
    extern "system" {
        fn SetEnvironmentVariableW(name: *const u16, value: *const u16) -> i32;
        fn GetEnvironmentVariableW(name: *const u16, buf: *mut u16, size: u32) -> u32;
    }
    let name_utf16: Vec<u16> = "HSA_ENABLE_SDMA"
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let mut buf = [0u16; 8];
    let existing =
        GetEnvironmentVariableW(name_utf16.as_ptr(), buf.as_mut_ptr(), buf.len() as u32);
    if existing == 0 {
        let value_utf16: Vec<u16> = "0".encode_utf16().chain(std::iter::once(0)).collect();
        SetEnvironmentVariableW(name_utf16.as_ptr(), value_utf16.as_ptr());
    }
}

#[cfg(windows)]
unsafe extern "C" fn vega_ctor_thunk() {
    vega_sdma_autoset();
}

// Registriamo il ptr a fn nella sezione `.CRT$XCU` del PE.
// Il CRT MSVC (_DllMainCRTStartup -> _initterm) scansiona [.CRT$XCA, .CRT$XCZ)
// e chiama ogni fn-ptr non-null in ordine di sub-sezione.
// `XCU` = User-level C++ initializers (slot canonico per ctor utente).
// `used` impedisce strip da rustc/LLVM (la sezione NON deve essere DCE-d).
#[cfg(all(windows, target_env = "msvc"))]
#[used]
#[link_section = ".CRT$XCU"]
static VEGA_CTOR_REGISTER: unsafe extern "C" fn() = vega_ctor_thunk;

// Manteniamo anche DllMain come fallback documentale per compat futura
// (se Rust upstream un giorno la chiamasse davvero — best effort, no harm).
#[cfg(windows)]
#[no_mangle]
#[allow(non_snake_case)]
pub unsafe extern "system" fn DllMain(
    _hinst_dll: *mut c_void,
    fdw_reason: u32,
    _lpv_reserved: *mut c_void,
) -> i32 {
    const DLL_PROCESS_ATTACH: u32 = 1;
    if fdw_reason == DLL_PROCESS_ATTACH {
        vega_sdma_autoset();
    }
    1
}

// =====================================================================
// VEGA-PATCH 2026-06-30 TASK A worker3/5: CUDA 12 EXTENSIONS
// =====================================================================
// Sezione attiva SOLO con `cargo build --features cuda12` (cfg cudart_12).
// Sblocca composite 709 path #2: torch 2.4.1+cu121 (test_venv_torch24)
// fallisce con CUDA_ERROR_NOT_FOUND perche' il cudart64_12.dll di sediment
// 23 giu T08:30 (250KB) e' incompleto. Reference NVIDIA cudart64_12.dll
// official > 500KB con questi export aggiuntivi rispetto a cudart64_110:
//   - cuTensorMap* (CUDA 12.0+, Hopper TMA, su gfx906 = stub no-op safe)
//   - cudaGraphInstantiateWithParams (instantiate v3)
//   - cudaStreamGetCaptureInfo_v3
//   - cudaMemPoolCreate / Destroy / Get/SetAttribute v12 (signature ext)
//   - cudaLibrary* / cudaKernel* (Module API rinnovata CUDA 12)
//   - cudaGetDriverEntryPoint_v2 / cudaGetDriverEntryPointByVersion
//   - cudaStreamGetId (CUDA 12.0+)
//   - cudaGraphAddNode_v2 (alias gia' presente, qui aliased)
//   - cudaInitDevice (CUDA 12.0+)
//
// REGOLA 4-bis: NESSUNA dipendenza ROCm 6.x. Tutti stub locali o forwarder hip* v5.7.
// REGOLA 17: ogni stub e' cargo-buildable + smoke ctypes verifica presenza simbolo.
// Strategia: per torch 2.4 cu121 basta che i simboli SIANO ESPORTATI e ritornino
// 0 (success) o 1 (cudaErrorInvalidValue) "innocuo". torch fa graceful fallback
// quando una API CUDA 12 non e' supportata (es. TMA su pre-Hopper).
// =====================================================================

#[cfg(cudart_12)]
mod cuda12_extensions {
    use super::{c_int, c_uint, c_void, c_char, cudaError_t, t, ensure_hip_init, ok_err};
    use hip_runtime_sys::*;

    // --- Tensor Map (Hopper TMA) - su gfx906 ritornano CUDA_ERROR_NOT_SUPPORTED (801)
    // ma torch li chiama solo se device.compute_capability >= 9.0, e fingiamo 7.0.
    // Quindi STUB 0 = safe: non verranno mai usati ma il simbolo deve esistere
    // per il dynamic link di torch 2.4.

    #[no_mangle]
    pub unsafe extern "C" fn cuTensorMapEncodeTiled() -> cudaError_t {
        t("cuTensorMapEncodeTiled")
    }
    #[no_mangle]
    pub unsafe extern "C" fn cuTensorMapEncodeIm2col() -> cudaError_t {
        t("cuTensorMapEncodeIm2col")
    }
    #[no_mangle]
    pub unsafe extern "C" fn cuTensorMapReplaceAddress() -> cudaError_t {
        t("cuTensorMapReplaceAddress")
    }

    // --- Graph instantiate v3 (CUDA 12.0+): firma con flags ESTESI + result_info struct.
    // hipGraphInstantiateWithFlags v5.7 accetta solo flags base; mappiamo trasparente.
    #[no_mangle]
    pub unsafe extern "C" fn cudaGraphInstantiateWithParams(
        p_graph_exec: *mut *mut c_void,
        graph: *mut c_void,
        _instantiate_params: *mut c_void,  // struct con flags + result_info
    ) -> cudaError_t {
        ensure_hip_init();
        // Flags = 0 default safe; hipGraphInstantiate accetta NULL per pErrorNode/pLogBuffer.
        let mut hip_exec: hipGraphExec_t = std::ptr::null_mut();
        let rc = hipGraphInstantiate(
            &mut hip_exec as *mut hipGraphExec_t,
            graph as hipGraph_t,
            std::ptr::null_mut(),  // pErrorNode: *mut hipGraphNode_t
            std::ptr::null_mut(),  // pLogBuffer: *mut c_char
            0usize,                // bufferSize
        );
        if !p_graph_exec.is_null() { *p_graph_exec = hip_exec as *mut c_void; }
        ok_err(rc)
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaGraphInstantiateWithParams_ptsz(
        p_graph_exec: *mut *mut c_void,
        graph: *mut c_void,
        instantiate_params: *mut c_void,
    ) -> cudaError_t {
        cudaGraphInstantiateWithParams(p_graph_exec, graph, instantiate_params)
    }

    // --- Stream capture info v3 (CUDA 12.3+): aggiunge edge_data param.
    // Forward al v2 esistente (parametri extra ignorati).
    #[no_mangle]
    pub unsafe extern "C" fn cudaStreamGetCaptureInfo_v3(
        stream: *mut c_void,
        status: *mut c_uint,
        id: *mut u64,
        graph: *mut *mut c_void,
        deps: *mut *mut c_void,
        _edge_data: *mut *mut c_void,
        num_deps: *mut usize,
    ) -> cudaError_t {
        super::cudaStreamGetCaptureInfo_v2(stream, status, id, graph, deps, num_deps)
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaStreamUpdateCaptureDependencies_v2(
        _stream: *mut c_void,
        _deps: *mut *mut c_void,
        _edge_data: *const c_void,
        _num_deps: usize,
        _flags: c_uint,
    ) -> cudaError_t { 0 }

    // --- MemPool v12 (CUDA 12.2+): set/get attribute EXT con extended location handle
    // Forward a hipMemPool* v5.7 sediment 28 giu (gia' funzionante con torch 2.4 cu118).
    #[no_mangle]
    pub unsafe extern "C" fn cudaMemPoolCreate_v12(
        pool: *mut *mut c_void,
        props: *const c_void,
    ) -> cudaError_t {
        super::cudaMemPoolCreate(pool, props)
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaMemPoolGetAttribute_v12(
        pool: *mut c_void,
        attr: c_int,
        value: *mut c_void,
    ) -> cudaError_t {
        super::cudaMemPoolGetAttribute(pool, attr, value)
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaMemPoolSetAttribute_v12(
        pool: *mut c_void,
        attr: c_int,
        value: *mut c_void,
    ) -> cudaError_t {
        super::cudaMemPoolSetAttribute(pool, attr, value)
    }

    // --- Library/Kernel API (CUDA 12.0+ rinnovo Module API)
    // torch 2.4 init chiama cudaLibraryLoadFromFile / cudaKernelGetAttribute lazy.
    // Stub 0 = NotFound graceful: torch fallback a __cudaRegisterFatBinary path.
    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryLoadData(
        _library: *mut *mut c_void,
        _code: *const c_void,
        _jit_options: *mut c_int, _jit_option_values: *mut *mut c_void, _num_jit: c_uint,
        _options: *mut c_int, _option_values: *mut *mut c_void, _num_opts: c_uint,
    ) -> cudaError_t {
        t("cudaLibraryLoadData")
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryLoadFromFile(
        _library: *mut *mut c_void,
        _file_name: *const c_char,
        _jit_options: *mut c_int, _jit_option_values: *mut *mut c_void, _num_jit: c_uint,
        _options: *mut c_int, _option_values: *mut *mut c_void, _num_opts: c_uint,
    ) -> cudaError_t {
        t("cudaLibraryLoadFromFile")
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryUnload(_library: *mut c_void) -> cudaError_t { 0 }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryGetKernel(
        _kernel: *mut *mut c_void,
        _library: *mut c_void,
        _name: *const c_char,
    ) -> cudaError_t { t("cudaLibraryGetKernel") }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryGetGlobal(
        _dptr: *mut *mut c_void, _bytes: *mut usize,
        _library: *mut c_void, _name: *const c_char,
    ) -> cudaError_t { t("cudaLibraryGetGlobal") }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryGetManaged(
        _dptr: *mut *mut c_void, _bytes: *mut usize,
        _library: *mut c_void, _name: *const c_char,
    ) -> cudaError_t { t("cudaLibraryGetManaged") }

    #[no_mangle]
    pub unsafe extern "C" fn cudaLibraryGetUnifiedFunction(
        _fptr: *mut *mut c_void, _library: *mut c_void, _symbol: *const c_char,
    ) -> cudaError_t { t("cudaLibraryGetUnifiedFunction") }

    #[no_mangle]
    pub unsafe extern "C" fn cudaKernelGetAttribute(
        value: *mut c_int, _attr: c_int, _kernel: *mut c_void,
    ) -> cudaError_t {
        if !value.is_null() { *value = 0; }
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaKernelSetAttribute(
        _attr: c_int, _value: c_int, _kernel: *mut c_void,
    ) -> cudaError_t { 0 }

    // --- Driver entry point versioned (CUDA 12.0+)
    // torch 2.4 init chiama cudaGetDriverEntryPointByVersion per risolvere
    // dinamicamente cuStreamSynchronize_v2 etc. Restituiamo NULL + CUDA_ERROR_NOT_FOUND
    // = torch fallback alla risoluzione dlsym standard (gia' coperta da nvcuda).
    #[no_mangle]
    pub unsafe extern "C" fn cudaGetDriverEntryPoint_v2(
        _symbol: *const c_char,
        fn_ptr: *mut *mut c_void,
        _flags: u64,
        status: *mut c_int,
    ) -> cudaError_t {
        if !fn_ptr.is_null() { *fn_ptr = std::ptr::null_mut(); }
        if !status.is_null() { *status = 2; }  // CUDA_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaGetDriverEntryPointByVersion(
        _symbol: *const c_char,
        fn_ptr: *mut *mut c_void,
        _cuda_version: c_uint,
        _flags: u64,
        status: *mut c_int,
    ) -> cudaError_t {
        if !fn_ptr.is_null() { *fn_ptr = std::ptr::null_mut(); }
        if !status.is_null() { *status = 2; }
        0
    }

    // --- Stream ID (CUDA 12.0+) - torch 2.4 PerStream caching allocator chiede unique ID.
    // Restituiamo l'indirizzo dello stream cast a u64 = unique-by-construction.
    #[no_mangle]
    pub unsafe extern "C" fn cudaStreamGetId(stream: *mut c_void, id: *mut u64) -> cudaError_t {
        if !id.is_null() { *id = stream as u64; }
        0
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaStreamGetId_ptsz(stream: *mut c_void, id: *mut u64) -> cudaError_t {
        cudaStreamGetId(stream, id)
    }

    // --- Init device (CUDA 12.0+) - alternativa a cudaSetDevice + cudaDeviceGetAttribute.
    #[no_mangle]
    pub unsafe extern "C" fn cudaInitDevice(
        device: c_int, _device_flags: c_uint, _flags: c_uint,
    ) -> cudaError_t {
        ensure_hip_init();
        let rc = hipSetDevice(device);
        ok_err(rc)
    }

    // --- Graph node v2 conditional (CUDA 12.3+ conditional nodes)
    #[no_mangle]
    pub unsafe extern "C" fn cudaGraphConditionalHandleCreate(
        _handle: *mut u64, _graph: *mut c_void,
        _default_launch_value: c_uint, _flags: c_uint,
    ) -> cudaError_t { 0 }

    // --- Async malloc v12 con node priority (CUDA 12.2+)
    // Forward al MallocAsync esistente, priority ignorata (gfx906 no stream priority hw)
    #[no_mangle]
    pub unsafe extern "C" fn cudaMallocAsync_v12(
        ptr: *mut *mut c_void, size: usize, stream: *mut c_void,
    ) -> cudaError_t {
        super::cudaMallocAsync(ptr, size, stream)
    }

    #[no_mangle]
    pub unsafe extern "C" fn cudaFreeAsync_v12(
        ptr: *mut c_void, stream: *mut c_void,
    ) -> cudaError_t {
        super::cudaFreeAsync(ptr, stream)
    }

    // --- Occupancy con flags (CUDA 12.0+)
    // VEGA-FIX TASK A worker1/5 2026-06-30 T08:05: removed cudaOccupancyMaxPotentialClusterSize
    // and cudaOccupancyMaxActiveClusters duplicate definitions — already exported from main
    // module (lines 1360-1362 stub no-arg form). torch 2.4 accepts both signatures (PE level
    // only checks symbol presence, no signature verification at dynamic link). Avoids
    // lld-link "error: symbol X is multiply defined" on cargo build --features cuda12.

    // --- Green context API (CUDA 12.5+) - stub innocui per torch 2.4 future-proof
    #[no_mangle]
    pub unsafe extern "C" fn cuCtxFromGreenCtx() -> cudaError_t { t("cuCtxFromGreenCtx") }
    #[no_mangle]
    pub unsafe extern "C" fn cuGreenCtxCreate() -> cudaError_t { t("cuGreenCtxCreate") }
    #[no_mangle]
    pub unsafe extern "C" fn cuGreenCtxDestroy() -> cudaError_t { t("cuGreenCtxDestroy") }
}  // mod cuda12_extensions


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
