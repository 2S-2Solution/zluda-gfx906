//! Funzioni del driver CUDA che ZLUDA non implementava (2S, 22 ott 2026).
//!
//! ZLUDA esportava 139 funzioni su 672: bastavano a PyTorch, non agli altri programmi. Qui le piu' comuni,
//! ciascuna inoltrata alla funzione HIP equivalente di ROCm 5.7. Esportate a mano (come texture e GL in lib.rs)
//! ed escluse dal blocco `unimplemented` col marcatore `estese_external` in `cuda_function_declarations!`.
//!
//! Cosa NON e' un inoltro diretto (ed e' convertito qui):
//! - CUDA_MEMCPY2D / CUDA_MEMCPY3D: i codici dei tipi di memoria differiscono (CUDA array 3, unified 4;
//!   HIP array 10, unified 11) e il descrittore 3D di CUDA ha due campi riservati che HIP non ha. La copia 2D
//!   gia' presente in memory.rs (memcpy_2d_async_v2, mai collegata) ignorava tipi e scostamenti.
//! - cuModuleLoad: legge il file e passa da cuModuleLoadData, cosi' il PTX passa dal traduttore di ZLUDA.
//! - Una sola scheda: l'accesso fra schede risponde come CUDA su una macchina con una GPU.
//! - Configurazioni di cache L1/memoria condivisa: su AMD non esistono, si accettano senza effetto.

use crate::r#impl::{context, memory};
use cuda_types::cuda::*;
use hip_runtime_sys::*;
use std::ffi::{c_char, c_int, c_uint, c_ushort, c_void};
use std::ptr;

macro_rules! esporta {
    ($( fn $nome:ident ( $($a:ident : $t:ty),* $(,)? ) $corpo:block )*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes_definitions, non_snake_case, unused_unsafe)]
            pub unsafe extern "system" fn $nome($($a: $t),*) -> CUresult {
                if !crate::initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                let esito: CUresult = (|| -> CUresult { unsafe { $corpo } })();
                if let Ok(mut f) = crate::diag("nvcuda_trace.log") {
                    use std::io::Write;
                    let codice = match &esito { Ok(_) => 0u32, Err(e) => e.0.get() };
                    let _ = writeln!(f, "[{} ESTESA] ret={}", stringify!($nome), codice);
                }
                esito
            }
        )*
    };
}

#[inline]
fn s(x: CUstream) -> hipStream_t {
    // CU_STREAM_LEGACY (0x1) = stream nullo di HIP (vedi zluda_common, FromCuda<CUstream>)
    if x.0 as usize == 1 {
        hipStream_t(ptr::null_mut())
    } else {
        unsafe { std::mem::transmute(x) }
    }
}

#[inline]
fn d(x: CUdeviceptr) -> hipDeviceptr_t {
    hipDeviceptr_t(x.0)
}

#[inline]
fn h(r: hipError_t) -> CUresult {
    r.map_err(CUerror::from)
}

// [2S, 12 nov 2026] Le funzioni che passano un hipMemoryType o un HIP_MEMCPY3D vanno prese SEMPRE
// dallo shim (amdhip64_7_shim.dll). Lo shim traduce l'ABI HIP 7.x di questi binding verso la runtime
// 5.7 del driver: rinumera hipMemoryType e ricostruisce HIP_MEMCPY3D, che in 5.7 ha campi a 32 bit.
// Importate per nome, il linker risolveva le versioni Async da amdhip64.dll (la 5.7, senza
// traduzione): le copie 2D/3D asincrone partivano con tipi e campi sbagliati.
mod dallo_shim {
    use hip_runtime_sys::*;
    use std::ffi::c_void;
    use std::sync::OnceLock;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(nome: *const u16) -> *mut c_void;
        fn GetProcAddress(modulo: *mut c_void, nome: *const u8) -> *mut c_void;
    }

    fn simbolo(nome: &'static [u8]) -> *mut c_void {
        static SHIM: OnceLock<usize> = OnceLock::new();
        let m = *SHIM.get_or_init(|| {
            let n: Vec<u16> = "amdhip64_7_shim.dll\0".encode_utf16().collect();
            unsafe { LoadLibraryW(n.as_ptr()) as usize }
        });
        if m == 0 {
            return std::ptr::null_mut();
        }
        unsafe { GetProcAddress(m as *mut c_void, nome.as_ptr()) }
    }

    type F2d = unsafe extern "C" fn(*const hip_Memcpy2D, hipStream_t) -> hipError_t;
    type F3d = unsafe extern "C" fn(*const HIP_MEMCPY3D, hipStream_t) -> hipError_t;

    pub unsafe fn hipMemcpyParam2DAsync(p: *const hip_Memcpy2D, s: hipStream_t) -> hipError_t {
        static F: OnceLock<usize> = OnceLock::new();
        let f = *F.get_or_init(|| simbolo(b"hipMemcpyParam2DAsync\0") as usize);
        if f == 0 {
            return hipError_t::ErrorNotSupported;
        }
        std::mem::transmute::<usize, F2d>(f)(p, s)
    }

    pub unsafe fn hipDrvMemcpy3DAsync(p: *const HIP_MEMCPY3D, s: hipStream_t) -> hipError_t {
        static F: OnceLock<usize> = OnceLock::new();
        let f = *F.get_or_init(|| simbolo(b"hipDrvMemcpy3DAsync\0") as usize);
        if f == 0 {
            return hipError_t::ErrorNotSupported;
        }
        std::mem::transmute::<usize, F3d>(f)(p, s)
    }
}

fn tipo_memoria(t: CUmemorytype) -> hipMemoryType {
    match t.0 {
        1 => hipMemoryType::hipMemoryTypeHost,
        2 => hipMemoryType::hipMemoryTypeDevice,
        3 => hipMemoryType::hipMemoryTypeArray,
        4 => hipMemoryType::hipMemoryTypeUnified,
        x => hipMemoryType(x),
    }
}

unsafe fn copia_2d(p: *const CUDA_MEMCPY2D) -> Result<hip_Memcpy2D, CUerror> {
    let c = p.as_ref().ok_or(CUerror::INVALID_VALUE)?;
    Ok(hip_Memcpy2D {
        srcXInBytes: c.srcXInBytes,
        srcY: c.srcY,
        srcMemoryType: tipo_memoria(c.srcMemoryType),
        srcHost: c.srcHost,
        srcDevice: d(c.srcDevice),
        srcArray: c.srcArray.cast(),
        srcPitch: c.srcPitch,
        dstXInBytes: c.dstXInBytes,
        dstY: c.dstY,
        dstMemoryType: tipo_memoria(c.dstMemoryType),
        dstHost: c.dstHost,
        dstDevice: d(c.dstDevice),
        dstArray: c.dstArray.cast(),
        dstPitch: c.dstPitch,
        WidthInBytes: c.WidthInBytes,
        Height: c.Height,
    })
}

unsafe fn copia_3d(p: *const CUDA_MEMCPY3D) -> Result<HIP_MEMCPY3D, CUerror> {
    let c = p.as_ref().ok_or(CUerror::INVALID_VALUE)?;
    Ok(HIP_MEMCPY3D {
        srcXInBytes: c.srcXInBytes,
        srcY: c.srcY,
        srcZ: c.srcZ,
        srcLOD: c.srcLOD,
        srcMemoryType: tipo_memoria(c.srcMemoryType),
        srcHost: c.srcHost,
        srcDevice: d(c.srcDevice),
        srcArray: c.srcArray.cast(),
        srcPitch: c.srcPitch,
        srcHeight: c.srcHeight,
        dstXInBytes: c.dstXInBytes,
        dstY: c.dstY,
        dstZ: c.dstZ,
        dstLOD: c.dstLOD,
        dstMemoryType: tipo_memoria(c.dstMemoryType),
        dstHost: c.dstHost,
        dstDevice: d(c.dstDevice),
        dstArray: c.dstArray.cast(),
        dstPitch: c.dstPitch,
        dstHeight: c.dstHeight,
        WidthInBytes: c.WidthInBytes,
        Height: c.Height,
        Depth: c.Depth,
    })
}

/// memset 2D a 16 o 32 bit: HIP ha solo la versione a 8 bit, si fa riga per riga.
unsafe fn set_2d(dst: CUdeviceptr, pitch: usize, valore: u32, byte: usize, larghezza: usize, altezza: usize,
                 flusso: Option<hipStream_t>) -> CUresult {
    for y in 0..altezza {
        let riga = hipDeviceptr_t((dst.0 as *mut u8).add(y * pitch).cast());
        let r = match (byte, flusso) {
            (2, None) => hipMemsetD16(riga, valore as c_ushort, larghezza),
            (2, Some(st)) => hipMemsetD16Async(riga, valore as c_ushort, larghezza, st),
            (_, None) => hipMemsetD32(riga, valore as c_int, larghezza),
            (_, Some(st)) => hipMemsetD32Async(riga, valore as c_int, larghezza, st),
        };
        h(r)?;
    }
    Ok(())
}

esporta! {
    // ------------------------------------------------------------------ stream
    fn cuStreamQuery(hStream: CUstream) { h(hipStreamQuery(s(hStream))) }
    fn cuStreamQuery_ptsz(hStream: CUstream) { h(hipStreamQuery(s(hStream))) }
    fn cuStreamGetPriority(hStream: CUstream, priority: *mut c_int) { h(hipStreamGetPriority(s(hStream), priority)) }
    fn cuStreamGetFlags(hStream: CUstream, flags: *mut c_uint) { h(hipStreamGetFlags(s(hStream), flags)) }
    fn cuStreamAddCallback(hStream: CUstream, callback: CUstreamCallback, userData: *mut c_void, flags: c_uint) {
        // stessa ABI su x64: (stream, esito come intero, dati); CUresult e hipError_t sono Result con nicchia sullo 0
        h(hipStreamAddCallback(s(hStream), std::mem::transmute::<CUstreamCallback, hipStreamCallback_t>(callback),
                               userData, flags))
    }
    fn cuLaunchHostFunc(hStream: CUstream, fn_: CUhostFn, userData: *mut c_void) {
        h(hipLaunchHostFunc(s(hStream), std::mem::transmute::<CUhostFn, hipHostFn_t>(fn_), userData))
    }

    // ------------------------------------------------------------------ copie e memset
    fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) {
        h(hipMemcpy(dst.0, src.0, ByteCount, hipMemcpyKind::hipMemcpyDefault))
    }
    fn cuMemcpyDtoD_v2(dstDevice: CUdeviceptr, srcDevice: CUdeviceptr, ByteCount: usize) {
        h(hipMemcpyDtoD(d(dstDevice), d(srcDevice), ByteCount))
    }
    fn cuMemsetD16_v2(dstDevice: CUdeviceptr, us: c_ushort, N: usize) { h(hipMemsetD16(d(dstDevice), us, N)) }
    fn cuMemsetD16Async(dstDevice: CUdeviceptr, us: c_ushort, N: usize, hStream: CUstream) {
        h(hipMemsetD16Async(d(dstDevice), us, N, s(hStream)))
    }
    fn cuMemsetD32Async(dstDevice: CUdeviceptr, ui: c_uint, N: usize, hStream: CUstream) {
        h(hipMemsetD32Async(d(dstDevice), ui as c_int, N, s(hStream)))
    }
    fn cuMemsetD2D8_v2(dstDevice: CUdeviceptr, dstPitch: usize, uc: u8, Width: usize, Height: usize) {
        h(hipMemset2D(dstDevice.0, dstPitch, uc as c_int, Width, Height))
    }
    fn cuMemsetD2D8Async(dstDevice: CUdeviceptr, dstPitch: usize, uc: u8, Width: usize, Height: usize, hStream: CUstream) {
        h(hipMemset2DAsync(dstDevice.0, dstPitch, uc as c_int, Width, Height, s(hStream)))
    }
    fn cuMemsetD2D16_v2(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize) {
        set_2d(dstDevice, dstPitch, us as u32, 2, Width, Height, None)
    }
    fn cuMemsetD2D16Async(dstDevice: CUdeviceptr, dstPitch: usize, us: c_ushort, Width: usize, Height: usize, hStream: CUstream) {
        set_2d(dstDevice, dstPitch, us as u32, 2, Width, Height, Some(s(hStream)))
    }
    fn cuMemsetD2D32_v2(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize) {
        set_2d(dstDevice, dstPitch, ui, 4, Width, Height, None)
    }
    fn cuMemsetD2D32Async(dstDevice: CUdeviceptr, dstPitch: usize, ui: c_uint, Width: usize, Height: usize, hStream: CUstream) {
        set_2d(dstDevice, dstPitch, ui, 4, Width, Height, Some(s(hStream)))
    }
    fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) { h(hipMemcpyParam2D(&copia_2d(pCopy)?)) }
    fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) { h(hipMemcpyParam2D(&copia_2d(pCopy)?)) }
    fn cuMemcpy2DAsync_v2(pCopy: *const CUDA_MEMCPY2D, hStream: CUstream) {
        h(dallo_shim::hipMemcpyParam2DAsync(&copia_2d(pCopy)?, s(hStream)))
    }
    fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) { h(hipDrvMemcpy3D(&copia_3d(pCopy)?)) }
    fn cuMemcpy3DAsync_v2(pCopy: *const CUDA_MEMCPY3D, hStream: CUstream) {
        h(dallo_shim::hipDrvMemcpy3DAsync(&copia_3d(pCopy)?, s(hStream)))
    }

    // ------------------------------------------------------------------ allocazioni
    fn cuMemAllocPitch_v2(dptr: *mut CUdeviceptr, pPitch: *mut usize, WidthInBytes: usize, Height: usize,
                          ElementSizeBytes: c_uint) {
        if dptr.is_null() || pPitch.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        let contesto = context::get_current_context()?;
        h(hipMemAllocPitch(dptr.cast(), pPitch, WidthInBytes, Height, ElementSizeBytes))?;
        memory::add_allocation((*dptr).0, *pPitch * Height, contesto)
    }
    fn cuMemAllocManaged(dptr: *mut CUdeviceptr, bytesize: usize, flags: c_uint) {
        if dptr.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        let contesto = context::get_current_context()?;
        h(hipMallocManaged(dptr.cast(), bytesize, flags))?;
        memory::add_allocation((*dptr).0, bytesize, contesto)
    }
    fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) {
        if pp.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        let contesto = context::get_current_context()?;
        h(hipHostMalloc(pp, bytesize, 0))?;
        memory::add_allocation(*pp, bytesize, contesto)
    }
    fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, Flags: c_uint) { h(hipHostRegister(p, bytesize, Flags)) }
    fn cuMemHostUnregister(p: *mut c_void) { h(hipHostUnregister(p)) }
    fn cuMemHostGetDevicePointer_v2(pdptr: *mut CUdeviceptr, p: *mut c_void, Flags: c_uint) {
        h(hipHostGetDevicePointer(pdptr.cast(), p, Flags))
    }
    fn cuMemHostGetFlags(pFlags: *mut c_uint, p: *mut c_void) { h(hipHostGetFlags(pFlags, p)) }
    fn cuMemAllocAsync(dptr: *mut CUdeviceptr, bytesize: usize, hStream: CUstream) {
        h(hipMallocAsync(dptr.cast(), bytesize, s(hStream)))
    }
    fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) { h(hipFreeAsync(dptr.0, s(hStream))) }
    fn cuMemPrefetchAsync(devPtr: CUdeviceptr, count: usize, dstDevice: CUdevice, hStream: CUstream) {
        h(hipMemPrefetchAsync(devPtr.0, count, dstDevice, s(hStream)))
    }
    fn cuMemAdvise(devPtr: CUdeviceptr, count: usize, advice: CUmem_advise, device: CUdevice) {
        // valori 1..6 identici in CUDA e HIP (SET/UNSET READ_MOSTLY, PREFERRED_LOCATION, ACCESSED_BY)
        h(hipMemAdvise(devPtr.0, count, hipMemoryAdvise(advice.0), device))
    }
    fn cuDeviceGetDefaultMemPool(pool_out: *mut CUmemoryPool, dev: CUdevice) {
        h(hipDeviceGetDefaultMemPool(pool_out.cast(), dev))
    }
    fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) { h(hipMemPoolTrimTo(pool.cast(), minBytesToKeep)) }
    fn cuMemPoolSetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void) {
        // attributi 1..8 nello stesso ordine in CUDA e HIP
        h(hipMemPoolSetAttribute(pool.cast(), hipMemPoolAttr(attr.0), value))
    }
    fn cuMemPoolGetAttribute(pool: CUmemoryPool, attr: CUmemPool_attribute, value: *mut c_void) {
        h(hipMemPoolGetAttribute(pool.cast(), hipMemPoolAttr(attr.0), value))
    }

    // ------------------------------------------------------------------ array
    fn cuArrayCreate_v2(pHandle: *mut CUarray, pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR) {
        // stesso layout (Width, Height, Format, NumChannels) e stessi codici di formato
        h(hipArrayCreate(pHandle.cast(), pAllocateArray.cast()))
    }
    fn cuArrayDestroy(hArray: CUarray) { h(hipArrayDestroy(hArray.cast())) }
    fn cuMemcpyHtoA_v2(dstArray: CUarray, dstOffset: usize, srcHost: *const c_void, ByteCount: usize) {
        h(hipMemcpyHtoA(dstArray.cast(), dstOffset, srcHost, ByteCount))
    }
    fn cuMemcpyAtoH_v2(dstHost: *mut c_void, srcArray: CUarray, srcOffset: usize, ByteCount: usize) {
        h(hipMemcpyAtoH(dstHost, srcArray.cast(), srcOffset, ByteCount))
    }

    // ------------------------------------------------------------------ eventi
    fn cuEventRecordWithFlags(hEvent: CUevent, hStream: CUstream, flags: c_uint) {
        if flags != 0 {
            return Err(CUerror::NOT_SUPPORTED); // HIP 5.7 non ha hipEventRecordWithFlags
        }
        h(hipEventRecord(hEvent.cast::<ihipEvent_t>(), s(hStream)))
    }

    // ------------------------------------------------------------------ schede e contesto
    fn cuDeviceCanAccessPeer(canAccessPeer: *mut c_int, dev: CUdevice, peerDev: CUdevice) {
        h(hipDeviceCanAccessPeer(canAccessPeer, dev, peerDev))
    }
    fn cuCtxEnablePeerAccess(_peerContext: CUcontext, _Flags: c_uint) { Err(CUerror::PEER_ACCESS_UNSUPPORTED) }
    fn cuCtxDisablePeerAccess(_peerContext: CUcontext) { Err(CUerror::PEER_ACCESS_NOT_ENABLED) }
    fn cuDeviceGetPCIBusId(pciBusId: *mut c_char, len: c_int, dev: CUdevice) {
        h(hipDeviceGetPCIBusId(pciBusId, len, dev))
    }
    fn cuDeviceGetByPCIBusId(dev: *mut CUdevice, pciBusId: *const c_char) { h(hipDeviceGetByPCIBusId(dev, pciBusId)) }
    fn cuCtxGetFlags(flags: *mut c_uint) {
        *flags.as_mut().ok_or(CUerror::INVALID_VALUE)? = 0;
        Ok(())
    }
    fn cuCtxResetPersistingL2Cache() { Ok(()) }
    fn cuCtxSetCacheConfig(_config: CUfunc_cache) { Ok(()) }
    fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) {
        *pconfig.as_mut().ok_or(CUerror::INVALID_VALUE)? = CUfunc_cache::CU_FUNC_CACHE_PREFER_NONE;
        Ok(())
    }
    fn cuFuncSetCacheConfig(_hfunc: CUfunction, _config: CUfunc_cache) { Ok(()) }
    fn cuFuncSetSharedMemConfig(_hfunc: CUfunction, _config: CUsharedconfig) { Ok(()) }
    fn cuCtxSetSharedMemConfig(_config: CUsharedconfig) { Ok(()) }
    fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) {
        *pConfig.as_mut().ok_or(CUerror::INVALID_VALUE)? = CUsharedconfig::CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE;
        Ok(())
    }

    // ------------------------------------------------------------------ moduli e lanci
    fn cuModuleLoad(module: *mut CUmodule, fname: *const c_char) {
        if module.is_null() || fname.is_null() {
            return Err(CUerror::INVALID_VALUE);
        }
        let percorso = std::ffi::CStr::from_ptr(fname).to_str().map_err(|_| CUerror::INVALID_VALUE)?;
        let mut dati = std::fs::read(percorso).map_err(|_| CUerror::FILE_NOT_FOUND)?;
        dati.push(0); // il PTX testuale va terminato
        // l'immagine resta viva quanto il processo: ZLUDA puo' tenerne riferimenti per la compilazione pigra
        let dati: &'static [u8] = Box::leak(dati.into_boxed_slice());
        crate::cuModuleLoadData(module, dati.as_ptr().cast())
    }
    fn cuLaunchCooperativeKernel(f: CUfunction, gridDimX: c_uint, gridDimY: c_uint, gridDimZ: c_uint,
                                 blockDimX: c_uint, blockDimY: c_uint, blockDimZ: c_uint, sharedMemBytes: c_uint,
                                 hStream: CUstream, kernelParams: *mut *mut c_void) {
        h(hipModuleLaunchCooperativeKernel(std::mem::transmute::<CUfunction, hipFunction_t>(f), gridDimX, gridDimY,
                                           gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, s(hStream),
                                           kernelParams))
    }
}
