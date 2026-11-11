use crate::r#impl::{
    context,
    driver::{self, global_state},
};
use cuda_types::cuda::{CUerror, CUresult, CUresultConsts};
use hip_runtime_sys::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::{mem, ptr};

// VEGA-PATCH 2026-06-30 GAP G7 worker 2/5 — VMM API (cuMemCreate + cuMemRelease)
// torch >=2.1 con PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True chiama
// cuMemCreate/cuMemRelease/cuMemMap/... per allocazioni virtuali espandibili.
// Senza export ZLUDA torna UNIMPLEMENTED → torch caching allocator si rompe.
//
// ABI mismatch chiave:
//   CUDA CUmemGenericAllocationHandle = c_ulonglong (u64)
//   HIP  hipMemGenericAllocationHandle_t = *mut ihipMemGenericAllocationHandle (opaque ptr)
//
// Soluzione: registry HashMap<u64, ptr> guarded da Mutex + counter atomico.
// All'esterno il caller vede un ID u64 stabile e opaco; internamente lo
// mappiamo al puntatore opaco HIP per il dispatch al runtime ROCm 5.7.
//
// Fallback graceful: ROCm 5.7 doc dichiara hipMemCreate "under development on
// Microsoft Windows". Quando hipMemCreate restituisce hipErrorNotSupported
// (o qualsiasi altro errore non-Success), traduciamo a CUDA_ERROR_NOT_SUPPORTED.
// PyTorch 2.1+ ha path esplicito di fallback: se cuMemCreate ritorna NOT_SUPPORTED
// disabilita expandable_segments per il device e ricasca su hipMalloc legacy
// (gia' wired in alloc_v2 sopra). Zero rischio regressione.

// Newtype wrapper Send+Sync per il puntatore opaco HIP — necessario per static Mutex
// HIP handle e' un puntatore a struct opaca runtime-allocated; il runtime ROCm
// e' thread-safe (HIP design contract) quindi marcare Send+Sync e' sicuro qui.
#[derive(Copy, Clone)]
struct HipVmmHandle(hipMemGenericAllocationHandle_t);
unsafe impl Send for HipVmmHandle {}
unsafe impl Sync for HipVmmHandle {}

struct VmmHandleRegistry {
    map: HashMap<u64, HipVmmHandle>,
}

fn vmm_registry() -> &'static Mutex<VmmHandleRegistry> {
    static REGISTRY: OnceLock<Mutex<VmmHandleRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(VmmHandleRegistry {
            map: HashMap::new(),
        })
    })
}

// id=0 e' riservato come "invalid handle" (CUDA convention) — partiamo da 1
static VMM_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

fn vmm_register(hip_handle: hipMemGenericAllocationHandle_t) -> u64 {
    let id = VMM_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    if let Ok(mut reg) = vmm_registry().lock() {
        reg.map.insert(id, HipVmmHandle(hip_handle));
    }
    id
}

#[allow(dead_code)]
fn vmm_lookup(id: u64) -> Option<hipMemGenericAllocationHandle_t> {
    vmm_registry().lock().ok()?.map.get(&id).map(|h| h.0)
}

fn vmm_remove(id: u64) -> Option<hipMemGenericAllocationHandle_t> {
    vmm_registry().lock().ok()?.map.remove(&id).map(|h| h.0)
}

/// Converti CUDA prop → HIP prop. Stesso layout di campi (verified
/// cuda_types/src/cuda.rs:6283 vs hip_runtime-sys/src/lib.rs:4593):
///   type_                  (CUmemAllocationType / hipMemAllocationType)
///   requestedHandleType[s] (CUmemAllocationHandleType / hipMemAllocationHandleType)
///   location               (CUmemLocation / hipMemLocation)
///   win32HandleMetaData    (*mut c_void)
///   allocFlags             (compressionType u8 + gpuDirectRDMACapable u8 + usage u16)
/// Tutti i campi sono repr(C) e gli enum sono c_uint con valori bit-perfetti
/// (PINNED=1 etc, DEVICE=1 etc). Costruiamo l'equivalente field-by-field per
/// stabilita' anche se ABI futuri divergono.
unsafe fn cuda_prop_to_hip(
    cu_prop: &cuda_types::cuda::CUmemAllocationProp,
) -> hipMemAllocationProp {
    hipMemAllocationProp {
        type_: hipMemAllocationType(cu_prop.type_.0),
        requestedHandleType: hipMemAllocationHandleType(cu_prop.requestedHandleTypes.0),
        location: hipMemLocation {
            type_: mem::transmute::<u32, hipMemLocationType>(cu_prop.location.type_.0),
            id: cu_prop.location.id,
        },
        win32HandleMetaData: cu_prop.win32HandleMetaData,
        allocFlags: hipMemAllocationProp__bindgen_ty_1 {
            compressionType: cu_prop.allocFlags.compressionType,
            gpuDirectRDMACapable: cu_prop.allocFlags.gpuDirectRDMACapable,
            usage: cu_prop.allocFlags.usage,
        },
    }
}

/// cuMemCreate — VMM allocation primitive (CUDA Driver API 10.2+).
/// Worker G7 2/5 focus principale.
///
/// Signature CUDA:
///   CUresult cuMemCreate(
///       CUmemGenericAllocationHandle *handle,
///       size_t size,
///       const CUmemAllocationProp *prop,
///       unsigned long long flags);
pub(crate) unsafe fn create(
    handle: &mut cuda_types::cuda::CUmemGenericAllocationHandle,
    size: usize,
    prop: &cuda_types::cuda::CUmemAllocationProp,
    flags: ::core::ffi::c_ulonglong,
) -> CUresult {
    if size == 0 {
        return CUresult::ERROR_INVALID_VALUE;
    }
    // Tracing minimale per debug expandable_segments
    use std::io::Write;
    if let Ok(mut f) = crate::diag("vmm_trace.log")
    {
        let _ = writeln!(
            f,
            "[cuMemCreate] size={} type={} loc_id={} flags={}",
            size,
            prop.type_.0,
            prop.location.id,
            flags
        );
    }
    let hip_prop = cuda_prop_to_hip(prop);
    let mut hip_handle: hipMemGenericAllocationHandle_t = ptr::null_mut();
    let rc = hipMemCreate(&mut hip_handle, size, &hip_prop, flags);
    match rc {
        Ok(()) => {
            let id = vmm_register(hip_handle);
            *handle = id;
            Ok(())
        }
        Err(hipErrorCode_t::NotSupported) => {
            // Path fallback graceful — torch caching allocator cattura
            // CUDA_ERROR_NOT_SUPPORTED e disabilita expandable_segments,
            // ricasca su hipMalloc legacy (alloc_v2 sopra)
            *handle = 0;
            CUresult::ERROR_NOT_SUPPORTED
        }
        Err(e) => {
            *handle = 0;
            // From<hipErrorCode_t> for CUerror gia' esistente in cuda_types/cuda.rs:9399
            Err(CUerror::from(e))
        }
    }
}

/// cuMemRelease — rilascia un handle VMM precedentemente creato con cuMemCreate.
/// Idempotente per ID sconosciuto (ritorna INVALID_VALUE invece di crash).
pub(crate) unsafe fn release(
    handle: cuda_types::cuda::CUmemGenericAllocationHandle,
) -> CUresult {
    if handle == 0 {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_handle = match vmm_remove(handle) {
        Some(h) => h,
        None => return CUresult::ERROR_INVALID_VALUE,
    };
    use std::io::Write;
    if let Ok(mut f) = crate::diag("vmm_trace.log")
    {
        let _ = writeln!(f, "[cuMemRelease] id={} hip_ptr={:p}", handle, hip_handle);
    }
    let rc = hipMemRelease(hip_handle);
    match rc {
        Ok(()) => Ok(()),
        Err(hipErrorCode_t::NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        Err(e) => Err(CUerror::from(e)),
    }
}

pub(crate) unsafe fn alloc_v2(dptr: &mut hipDeviceptr_t, bytesize: usize) -> CUresult {
    let global = global_state()?;
    let context = context::get_current_context()?;
    hipMalloc(ptr::from_mut(dptr).cast(), bytesize)?;
    add_allocation(dptr.0, bytesize, context)?;
    let mut status = mem::zeroed();
    hipStreamIsCapturing(hipStream_t(ptr::null_mut()), &mut status)?;
    if global.should_zero_allocations
        && status == hipStreamCaptureStatus::hipStreamCaptureStatusNone
    {
        hipMemsetD8(*dptr, 0, bytesize)?;
    }
    Ok(())
}

pub(crate) unsafe fn free_v2(dptr: hipDeviceptr_t) -> CUresult {
    let hip_result = hipFree(dptr.0);
    remove_allocation(dptr.0)?;
    Ok(hip_result?)
}

pub(crate) fn copy_dto_h_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyDtoH(dst_host, src_device, byte_count) }
}

pub(crate) fn copy_hto_d_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe { hipMemcpyHtoD(dst_device, src_host.cast_mut(), byte_count) }
}

pub(crate) fn copy_hto_d_v2_ptds(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
) -> hipError_t {
    unsafe {
        hipMemcpy_spt(
            dst_device.0.cast(),
            src_host.cast_mut(),
            byte_count,
            hipMemcpyKind::hipMemcpyHostToDevice,
        )
    }
}

pub(crate) fn copy_dto_h_v2_ptds(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
) -> hipError_t {
    unsafe {
        hipMemcpy_spt(
            dst_host.cast(),
            src_device.0.cast(),
            byte_count,
            hipMemcpyKind::hipMemcpyDeviceToHost,
        )
    }
}

pub(crate) fn get_address_range_v2(
    pbase: *mut hipDeviceptr_t,
    psize: *mut usize,
    dptr: hipDeviceptr_t,
) -> hipError_t {
    unsafe { hipMemGetAddressRange(pbase, psize, dptr) }
}

pub(crate) fn set_d32_v2(dst: hipDeviceptr_t, ui: ::core::ffi::c_uint, n: usize) -> hipError_t {
    unsafe { hipMemsetD32(dst, ui as std::ffi::c_int, n) }
}

pub(crate) fn set_d8_v2(dst: hipDeviceptr_t, value: ::core::ffi::c_uchar, n: usize) -> hipError_t {
    unsafe { hipMemsetD8(dst, value, n) }
}
pub(crate) fn get_info_v2(free: *mut usize, total: *mut usize) -> hipError_t {
    unsafe { hipMemGetInfo(free, total) }
}

pub(crate) unsafe fn free_host(ptr: *mut ::core::ffi::c_void) -> CUresult {
    let hip_result = hipFreeHost(ptr);
    remove_allocation(ptr)?;
    Ok(hip_result?)
}

pub(crate) unsafe fn host_alloc(
    pp: &mut *mut ::core::ffi::c_void,
    bytesize: usize,
    flags: ::std::os::raw::c_uint,
) -> CUresult {
    let context = context::get_current_context()?;
    hipHostMalloc(pp, bytesize, flags)?;
    add_allocation(*pp, bytesize, context)?;
    Ok(())
}

pub(crate) fn add_allocation(
    dptr: *mut ::core::ffi::c_void,
    bytesize: usize,
    context: cuda_types::cuda::CUcontext,
) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.insert(dptr as usize, bytesize, context);
    Ok(())
}

pub(crate) fn remove_allocation(ptr: *mut std::ffi::c_void) -> Result<(), CUerror> {
    let global_state = driver::global_state()?;
    let mut allocations = global_state
        .allocations
        .lock()
        .map_err(|_| CUerror::UNKNOWN)?;
    allocations.remove(ptr as usize);
    Ok(())
}

pub(crate) unsafe fn retain_allocation_handle(
    _handle: *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    _addr: *mut ::core::ffi::c_void,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

pub(crate) unsafe fn copy_hto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_host: *const ::core::ffi::c_void,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyHtoDAsync(dst_device, src_host.cast_mut(), byte_count, stream)
}

pub(crate) unsafe fn copy_dto_h_async_v2(
    dst_host: *mut ::core::ffi::c_void,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyDtoHAsync(dst_host, src_device, byte_count, stream)
}

pub(crate) unsafe fn copy_dto_d_async_v2(
    dst_device: hipDeviceptr_t,
    src_device: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyDtoDAsync(dst_device, src_device, byte_count, stream)
}

pub(crate) unsafe fn copy_async(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemcpyAsync(
        dst.0,
        src.0,
        byte_count,
        hipMemcpyKind::hipMemcpyDefault,
        stream,
    )
}

// VEGA-PATCH 2026-06-28 sprint-orchestration: cuMemcpyAsync_ptsz delegate a cuMemcpyAsync
// PyTorch CUDA 12+ usa la variante per-thread stream (_ptsz) come simbolo principale per
// `cudaMemcpyAsync`. Senza questo export l'app riceve UNIMPLEMENTED e fallisce backward
// matmul .T non-contig. Semanticamente identica: stesso prototype, stesso hipMemcpyAsync
// con stream esplicito (lo stream zero per-thread e' gestito dal chiamante).
pub(crate) unsafe fn copy_async_ptsz(
    dst: hipDeviceptr_t,
    src: hipDeviceptr_t,
    byte_count: usize,
    stream: hipStream_t,
) -> hipError_t {
    copy_async(dst, src, byte_count, stream)
}

pub(crate) unsafe fn set_d8_async(
    dst_device: hipDeviceptr_t,
    uc: ::core::ffi::c_uchar,
    n: usize,
    stream: hipStream_t,
) -> hipError_t {
    hipMemsetD8Async(dst_device, uc, n, stream)
}

pub(crate) fn get_allocation_granularity(
    _granularity: &mut usize,
    _property: &cuda_types::cuda::CUmemAllocationProp,
    _option: cuda_types::cuda::CUmemAllocationGranularity_flags,
) -> CUresult {
    CUresult::ERROR_NOT_SUPPORTED
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-1: cuMemcpy2DAsync_v2 forwarder a hipMemcpy2DAsync
// ROOT CAUSE bug 999 backward .T non-contig: PyTorch CUDA invoca cuMemcpy2DAsync_v2 (pitched 2D copy)
// per gestire tensori strided/transposed durante autograd. Senza export ZLUDA torna UNIMPLEMENTED
// e la backward su matmul transposte fallisce con CUDA error 999. Mapping diretto al binding HIP
// gia' presente in hip_runtime-sys (hipMemcpy2DAsync). Estraiamo i campi rilevanti dal descrittore
// CUDA_MEMCPY2D (srcDevice, srcPitch, dstDevice, dstPitch, WidthInBytes, Height) e deleghiamo a HIP
// con hipMemcpyKind::hipMemcpyDefault (HIP risolve il tipo via puntatori, come fa copy_async).
pub(crate) unsafe fn memcpy_2d_async_v2(
    p_copy: *const cuda_types::cuda::CUDA_MEMCPY2D,
    stream: hipStream_t,
) -> hipError_t {
    if p_copy.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let copy = &*p_copy;
    hipMemcpy2DAsync(
        copy.dstDevice.0,
        copy.dstPitch,
        copy.srcDevice.0.cast_const(),
        copy.srcPitch,
        copy.WidthInBytes,
        copy.Height,
        hipMemcpyKind::hipMemcpyDefault,
        stream,
    )
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-1: cuMemcpy2DAsync_v2_ptsz delegato a memcpy_2d_async_v2
// Variante per-thread stream usata da PyTorch CUDA 12+. Semanticamente identica, stream esplicito.
pub(crate) unsafe fn memcpy_2d_async_v2_ptsz(
    p_copy: *const cuda_types::cuda::CUDA_MEMCPY2D,
    stream: hipStream_t,
) -> hipError_t {
    memcpy_2d_async_v2(p_copy, stream)
}

// ============================================================================
// VEGA-PATCH 2026-06-30 GAP G7 VMM API completion (worker G7 1/5)
// ----------------------------------------------------------------------------
// ROOT CAUSE: torch 2.1+ con expandable_segments=True (default su CUDA backend)
// invoca cuMemCreate / cuMemMap / cuMemAddressReserve / cuMemSetAccess per
// allocare il pool memoria con virtual memory management (VMM). Senza export
// reali in nvcuda.dll PyTorch torna CUDA_ERROR_NOT_SUPPORTED e l'allocator
// fa fallback al path legacy cudaMalloc, perdendo l'ottimizzazione e in
// alcuni scenari (LSTM a sequenza lunga) crashando con OOM falso.
//
// STRATEGIA: i tipi CUDA (CUmemAllocationProp, CUmemAccessDesc, CUmemLocation,
// CUmemGenericAllocationHandle, CUmemAllocationHandleType) hanno layout
// binario IDENTICO ai corrispondenti hip_runtime-sys (verificato T15:50
// confrontando offset campi type_/requestedHandleType(s)/location/
// win32HandleMetaData/allocFlags). Possiamo quindi fare cast diretto
// `*const CUmemAllocationProp as *const hipMemAllocationProp` e delegare
// alle funzioni HIP native ROCm 5.7 esposte da amdhip64.dll.
//
// FALLBACK: se hipMem* ritorna hipErrorNotSupported (Windows ROCm 5.7 per
// alcune API VMM marcate "under development on Microsoft Windows" nei doc
// HIP) restituiamo CUDA_ERROR_NOT_SUPPORTED graceful in modo che PyTorch
// fallback automaticamente a expandable_segments=False senza crash.
//
// API coperte (8):
//   1. cuMemCreate                       -> hipMemCreate
//   2. cuMemRelease                      -> hipMemRelease
//   3. cuMemMap                          -> hipMemMap
//   4. cuMemUnmap                        -> hipMemUnmap
//   5. cuMemAddressReserve               -> hipMemAddressReserve
//   6. cuMemAddressFree                  -> hipMemAddressFree
//   7. cuMemSetAccess                    -> hipMemSetAccess
//   8. cuMemGetAllocationPropertiesFromHandle -> hipMemGetAllocationPropertiesFromHandle
// (cuMemGetAllocationGranularity gia' presente come stub a riga 208 e wired
//  in lib.rs riga 933, lo sostituiamo con impl reale qui sotto)
// (cuMemRetainAllocationHandle gia' presente come stub a riga 136, idem)
// ============================================================================

/// VMM API #1: cuMemCreate -> hipMemCreate
/// Crea una memoria allocation handle descritta da `prop` di dimensione `size`.
/// Layout CUmemAllocationProp identico a hipMemAllocationProp (5 campi,
/// stesso ordine, stessa size). Cast diretto sicuro.
pub(crate) unsafe fn mem_create(
    handle: *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    size: usize,
    prop: *const cuda_types::cuda::CUmemAllocationProp,
    flags: ::core::ffi::c_ulonglong,
) -> CUresult {
    if handle.is_null() || prop.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_handle = handle as *mut hipMemGenericAllocationHandle_t;
    let hip_prop = prop as *const hipMemAllocationProp;
    let rc = hipMemCreate(hip_handle, size, hip_prop, flags);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #2: cuMemRelease -> hipMemRelease
/// Rilascia un handle ottenuto da cuMemCreate.
pub(crate) unsafe fn mem_release(
    handle: cuda_types::cuda::CUmemGenericAllocationHandle,
) -> CUresult {
    let rc = hipMemRelease(handle as hipMemGenericAllocationHandle_t);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #3: cuMemMap -> hipMemMap
/// Mappa un handle di allocation a un range di indirizzi virtuali riservati
/// precedentemente con cuMemAddressReserve.
pub(crate) unsafe fn mem_map(
    ptr: cuda_types::cuda::CUdeviceptr,
    size: usize,
    offset: usize,
    handle: cuda_types::cuda::CUmemGenericAllocationHandle,
    flags: ::core::ffi::c_ulonglong,
) -> CUresult {
    let rc = hipMemMap(
        ptr.0 as *mut ::core::ffi::c_void,
        size,
        offset,
        handle as hipMemGenericAllocationHandle_t,
        flags,
    );
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #4: cuMemUnmap -> hipMemUnmap
/// Smappa un range precedentemente mappato con cuMemMap.
pub(crate) unsafe fn mem_unmap(
    ptr: cuda_types::cuda::CUdeviceptr,
    size: usize,
) -> CUresult {
    let rc = hipMemUnmap(ptr.0 as *mut ::core::ffi::c_void, size);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #5: cuMemAddressReserve -> hipMemAddressReserve
/// Riserva un range di indirizzi virtuali (senza fisicamente allocare memoria).
pub(crate) unsafe fn mem_address_reserve(
    ptr: *mut cuda_types::cuda::CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: cuda_types::cuda::CUdeviceptr,
    flags: ::core::ffi::c_ulonglong,
) -> CUresult {
    if ptr.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let mut hip_ptr: *mut ::core::ffi::c_void = ptr::null_mut();
    let rc = hipMemAddressReserve(
        &mut hip_ptr as *mut *mut ::core::ffi::c_void,
        size,
        alignment,
        addr.0 as *mut ::core::ffi::c_void,
        flags,
    );
    match rc {
        hipError_t::Success => {
            (*ptr).0 = hip_ptr.cast();
            Ok(())
        }
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #6: cuMemAddressFree -> hipMemAddressFree
/// Libera un range riservato con cuMemAddressReserve.
pub(crate) unsafe fn mem_address_free(
    ptr: cuda_types::cuda::CUdeviceptr,
    size: usize,
) -> CUresult {
    let rc = hipMemAddressFree(ptr.0 as *mut ::core::ffi::c_void, size);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #7: cuMemSetAccess -> hipMemSetAccess
/// Imposta le permission flags di accesso per un range di indirizzi virtuali.
/// Layout CUmemAccessDesc identico a hipMemAccessDesc (2 campi: location +
/// flags). Cast diretto sicuro su array.
pub(crate) unsafe fn mem_set_access(
    ptr: cuda_types::cuda::CUdeviceptr,
    size: usize,
    desc: *const cuda_types::cuda::CUmemAccessDesc,
    count: usize,
) -> CUresult {
    if desc.is_null() && count > 0 {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_desc = desc as *const hipMemAccessDesc;
    let rc = hipMemSetAccess(
        ptr.0 as *mut ::core::ffi::c_void,
        size,
        hip_desc.cast_mut(),
        count,
    );
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API #8: cuMemGetAllocationPropertiesFromHandle ->
///   hipMemGetAllocationPropertiesFromHandle
/// Recupera la struttura properties associata a un handle.
pub(crate) unsafe fn mem_get_allocation_properties_from_handle(
    prop: *mut cuda_types::cuda::CUmemAllocationProp,
    handle: cuda_types::cuda::CUmemGenericAllocationHandle,
) -> CUresult {
    if prop.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_prop = prop as *mut hipMemAllocationProp;
    let rc = hipMemGetAllocationPropertiesFromHandle(
        hip_prop,
        handle as hipMemGenericAllocationHandle_t,
    );
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API EXTRA: upgrade cuMemGetAllocationGranularity da stub a impl reale.
/// La firma esistente (riga 208) torna sempre ERROR_NOT_SUPPORTED, qui deleghiamo
/// a hipMemGetAllocationGranularity. Da wirare in lib.rs sostituendo l'entry
/// nel dispatch table.
pub(crate) unsafe fn mem_get_allocation_granularity_real(
    granularity: *mut usize,
    prop: *const cuda_types::cuda::CUmemAllocationProp,
    option: cuda_types::cuda::CUmemAllocationGranularity_flags,
) -> CUresult {
    if granularity.is_null() || prop.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_prop = prop as *const hipMemAllocationProp;
    let hip_option = hipMemAllocationGranularity_flags(option.0);
    let rc = hipMemGetAllocationGranularity(granularity, hip_prop, hip_option);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}

/// VMM API EXTRA: upgrade cuMemRetainAllocationHandle da stub a impl reale.
/// La firma esistente (riga 136) torna sempre ERROR_NOT_SUPPORTED, qui
/// deleghiamo a hipMemRetainAllocationHandle.
pub(crate) unsafe fn retain_allocation_handle_real(
    handle: *mut cuda_types::cuda::CUmemGenericAllocationHandle,
    addr: *mut ::core::ffi::c_void,
) -> CUresult {
    if handle.is_null() {
        return CUresult::ERROR_INVALID_VALUE;
    }
    let hip_handle = handle as *mut hipMemGenericAllocationHandle_t;
    let rc = hipMemRetainAllocationHandle(hip_handle, addr);
    match rc {
        hipError_t::Success => Ok(()),
        Err(hipErrorCode_t::r#NotSupported) => CUresult::ERROR_NOT_SUPPORTED,
        _ => Ok(rc?),
    }
}
