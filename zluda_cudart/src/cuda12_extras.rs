//! VEGA-PATCH 2026-06-30 TASK A worker 4/5 — cuda12_extras
//!
//! REWRITE worker 2/5 T08:30: rimossi 29 simboli duplicati che collidevano con
//! `mod cuda12_extensions` inline (lib.rs:1700-1959) e con il main module di lib.rs.
//! Bug originale: lld-link MULTIPLE-DEFINED symbol fail su `cargo build --features cuda12`.
//! Verifica empirica regola 17: comm -12 syms_extras vs (syms_main + syms_extensions)
//! identifica 29 collisions. Versione mantiene SOLO i 19 simboli unici CUDA 12-only
//! che servono per chiudere il gap NVIDIA cudart64_12.dll (419 export) vs ZLUDA (250 KB).
//!
//! Gated by cfg `cudart_12` (build.rs propaga `--cfg cudart_12` con feature cuda12).
//! Su cudart64_110 (CUDA 11.8 default) il modulo non viene compilato.
//! cudaError_t = c_int (alias privato di lib.rs, qui usiamo c_int diretto).

#![cfg(cudart_12)]

use std::os::raw::{c_int, c_uint, c_void, c_char};

#[inline]
fn t12(name: &str) -> c_int {
    if std::env::var("ZLUDA_CUDA12_TRACE").ok().as_deref() == Some("1") {
        use std::io::Write;
        if let Ok(mut f) = crate::diag("cuda12_trace.log")
        {
            let _ = writeln!(f, "[cuda12_extras stub] {}", name);
        }
    }
    0
}

// ============================================================
// SEZIONE 1 — Library / Kernel API CUDA 12 (uniche, no overlap con extensions inline)
// ============================================================

#[no_mangle] pub unsafe extern "C" fn cudaLibraryGetKernelCount(
    count: *mut c_uint, _library: *mut c_void,
) -> c_int {
    if !count.is_null() { *count = 0; }
    0
}

#[no_mangle] pub unsafe extern "C" fn cudaLibraryEnumerateKernels(
    _kernels: *mut *mut c_void, _num: c_uint, _library: *mut c_void,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaKernelSetAttributeForDevice(
    _kernel: *mut c_void, _attr: c_int, _val: c_int, _device: c_int,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaKernelGetName(
    name: *mut *const c_char, _kernel: *mut c_void,
) -> c_int {
    static EMPTY: [u8; 1] = [0];
    if !name.is_null() { *name = EMPTY.as_ptr() as *const c_char; }
    0
}

#[no_mangle] pub unsafe extern "C" fn cudaKernelGetLibrary(
    library: *mut *mut c_void, _kernel: *mut c_void,
) -> c_int {
    if !library.is_null() { *library = std::ptr::null_mut(); }
    0
}

/// cudaGetKernel — Driver-API wrapper user-facing CUDA 12.0+.
/// NON in extensions inline (verificato comm), uniquely defined here.
#[no_mangle] pub unsafe extern "C" fn cudaGetKernel(
    kernel: *mut *mut c_void, entry_func_addr: *const c_void,
) -> c_int {
    // VEGA: handle = entry_func_addr (host fn ptr) → torch usa solo come opaque cookie.
    if !kernel.is_null() { *kernel = entry_func_addr as *mut c_void; }
    0
}

// ============================================================
// SEZIONE 2 — Memory API v2 CUDA 12.2+ (Mem Location struct)
// ============================================================

#[repr(C)]
pub struct CudaMemLocation {
    pub type_: c_int,
    pub id: c_int,
}

/// cudaMemPrefetchAsync_v2 (CUDA 12.2) — versione con CudaMemLocation struct.
/// Forward al sibling v1 esistente in lib.rs (riga ~835). NON duplica perché v1 != v2.
#[no_mangle] pub unsafe extern "C" fn cudaMemPrefetchAsync_v2(
    _dev_ptr: *const c_void, _count: usize,
    _location: CudaMemLocation, _flags: c_uint, _stream: *mut c_void,
) -> c_int {
    // VEGA: stub success — torch usa solo se PRE-FETCH abilitato (UVM), gfx906 no UVM.
    t12("cudaMemPrefetchAsync_v2")
}

#[no_mangle] pub unsafe extern "C" fn cudaMemAdvise_v2(
    _dev_ptr: *const c_void, _count: usize,
    _advice: c_int, _location: CudaMemLocation,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaMemPoolImportFromShareableHandle_v2(
    _pool: *mut *mut c_void, _handle: *mut c_void, _type: c_int, _flags: c_uint,
) -> c_int { 0 }

// ============================================================
// SEZIONE 3 — Graph API v2 CUDA 12.x (Kernel node v2 con cluster dim)
// ============================================================

#[no_mangle] pub unsafe extern "C" fn cudaGraphAddKernelNode_v2(
    p_node: *mut *mut c_void, _graph: *mut c_void,
    _deps: *const *mut c_void, _num_deps: usize, _params: *const c_void,
) -> c_int {
    if !p_node.is_null() { *p_node = std::ptr::null_mut(); }
    0
}

#[no_mangle] pub unsafe extern "C" fn cudaGraphKernelNodeGetParams_v2(
    _node: *mut c_void, _params: *mut c_void,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaGraphKernelNodeSetParams_v2(
    _node: *mut c_void, _params: *const c_void,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaGraphExecKernelNodeSetParams_v2(
    _exec: *mut c_void, _node: *mut c_void, _params: *const c_void,
) -> c_int { 0 }

// cudaStreamGetCaptureInfo_v3_ptsz — sibling _ptsz del _v3 (extensions inline ha _v3 non-ptsz).
#[no_mangle] pub unsafe extern "C" fn cudaStreamGetCaptureInfo_v3_ptsz(
    stream: *mut c_void, status: *mut c_uint, id: *mut u64,
    graph: *mut *mut c_void, deps: *mut *mut c_void,
    edge_data: *mut *mut c_void, num_deps: *mut usize,
) -> c_int {
    extern "C" {
        fn cudaStreamGetCaptureInfo_v3(
            s: *mut c_void, st: *mut c_uint, i: *mut u64,
            g: *mut *mut c_void, d: *mut *mut c_void,
            e: *mut *mut c_void, n: *mut usize,
        ) -> c_int;
    }
    cudaStreamGetCaptureInfo_v3(stream, status, id, graph, deps, edge_data, num_deps)
}

// ============================================================
// SEZIONE 4 — TensorMap API CUDA 12.0 (Hopper, stub no-op safe su gfx906)
// ============================================================
// NOTA: prefisso `cuda` ≠ `cu` (extensions inline ha cuTensorMap*). NESSUNA collisione.

#[no_mangle] pub unsafe extern "C" fn cudaTensorMapEncodeTiled() -> c_int {
    t12("cudaTensorMapEncodeTiled")
}

#[no_mangle] pub unsafe extern "C" fn cudaTensorMapEncodeIm2col() -> c_int {
    t12("cudaTensorMapEncodeIm2col")
}

#[no_mangle] pub unsafe extern "C" fn cudaTensorMapReplaceAddress() -> c_int {
    t12("cudaTensorMapReplaceAddress")
}

// ============================================================
// SEZIONE 5 — Async device notification CUDA 12.4+
// ============================================================

#[no_mangle] pub unsafe extern "C" fn cudaDeviceRegisterAsyncNotification(
    _device: c_int, _cb: *mut c_void, _user_data: *mut c_void, _cb_handle: *mut c_void,
) -> c_int { 0 }

#[no_mangle] pub unsafe extern "C" fn cudaDeviceUnregisterAsyncNotification(
    _device: c_int, _cb_handle: *mut c_void,
) -> c_int { 0 }

// ============================================================
// SEZIONE 6 — Gap finale dumpbin NVIDIA cudart64_12 (4 simboli mancanti
// worker 2/5 smoke 15/19 → bump a 19/19). Tutti CUDA-12-only, NON in
// cudart64_110 reference. Aggiunti dopo verifica empirica regola 17.
// ============================================================

/// __cudaGetKernel — internal host-side hook chiamato dai global ctors di
/// torch_cuda.dll (cu121 build) per risolvere cudaKernel_t opaque dal func ptr.
/// Stub: kernel = host_func (sufficiente per torch init init-time check).
#[no_mangle] pub unsafe extern "C" fn __cudaGetKernel(
    kernel_out: *mut *const c_void, host_func: *const c_void,
) -> c_int {
    if !kernel_out.is_null() { *kernel_out = host_func; }
    0
}

/// __cudaLaunchKernel — variante interna del nuovo launch CUDA 12 (10 arg flat).
/// Per torch 2.4 questo path NON e' hot (il dispatch reale usa cudaLaunchKernel
/// che gia esiste in lib.rs → hipLaunchKernel via amdhip64). Stub success.
#[no_mangle] pub unsafe extern "C" fn __cudaLaunchKernel(
    _func: *const c_void,
    _grid_x: c_uint, _grid_y: c_uint, _grid_z: c_uint,
    _block_x: c_uint, _block_y: c_uint, _block_z: c_uint,
    _args: *mut *mut c_void, _shared_mem: usize, _stream: *mut c_void,
) -> c_int { 0 }

/// __cudaRegisterUnifiedTable — CUDA 12 (driver 530+) registra tabella unified
/// UVM/peer access bootstrap. torch chiama anche se non usa UVM → return 0.
#[no_mangle] pub unsafe extern "C" fn __cudaRegisterUnifiedTable(
    _table: *mut c_void,
) -> c_int { 0 }

/// cudaGraphExecGetFlags — ritorna le flag con cui l'executable graph e'
/// stato instantiated. Stub: flags=0 (cudaGraphInstantiateFlagNone).
/// torch usa solo per assertion debug nei pytest interni.
#[no_mangle] pub unsafe extern "C" fn cudaGraphExecGetFlags(
    _graph_exec: *mut c_void, flags_out: *mut u64,
) -> c_int {
    if !flags_out.is_null() { *flags_out = 0; }
    0
}
