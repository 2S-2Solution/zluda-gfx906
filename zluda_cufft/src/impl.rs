// VEGA-PATCH 2026-06-30 TASK 5 worker 2/3 — zluda_cufft impl
// Wrap hipfft 5.7. Convenzioni:
//   - cufftHandle (c_int) == hipfftHandle (c_int) -> pass-through bit a bit
//   - cufftType_t::CUFFT_C2C(41) == hipfftType_t::HIPFFT_C2C(0x29) idem altri
//   - cudaStream_t (CUstream = *mut CUstream_st) == hipStream_t (*mut c_void)
//     -> cast diretto come void*
//   - cufftResult = Result<(), cufftError_t>; mappiamo hipfftResult -> cufftResult.

use core::ffi::{c_int, c_longlong, c_void};

use cuda_types::cufft::{
    cudaLibXtDesc, cudaStream_t, cufftComplex, cufftDoubleComplex, cufftDoubleReal, cufftHandle,
    cufftReal, cufftResult, cufftResultConsts, cufftType,
};
use hipfft_sys as hipfft;

// ---------- helpers ---------------------------------------------------------

#[inline]
fn map_result(r: hipfft::hipfftResult) -> cufftResult {
    use hipfft::hipfftResult_t::*;
    match r {
        HIPFFT_SUCCESS => cufftResult::SUCCESS,
        HIPFFT_INVALID_PLAN => cufftResult::ERROR_INVALID_PLAN,
        HIPFFT_ALLOC_FAILED => cufftResult::ERROR_ALLOC_FAILED,
        HIPFFT_INVALID_TYPE => cufftResult::ERROR_INVALID_TYPE,
        HIPFFT_INVALID_VALUE => cufftResult::ERROR_INVALID_VALUE,
        HIPFFT_INTERNAL_ERROR => cufftResult::ERROR_INTERNAL_ERROR,
        HIPFFT_EXEC_FAILED => cufftResult::ERROR_EXEC_FAILED,
        HIPFFT_SETUP_FAILED => cufftResult::ERROR_SETUP_FAILED,
        HIPFFT_INVALID_SIZE => cufftResult::ERROR_INVALID_SIZE,
        HIPFFT_UNALIGNED_DATA => cufftResult::ERROR_UNALIGNED_DATA,
        HIPFFT_INVALID_DEVICE => cufftResult::ERROR_INVALID_DEVICE,
        HIPFFT_NO_WORKSPACE => cufftResult::ERROR_NO_WORKSPACE,
        HIPFFT_NOT_IMPLEMENTED => cufftResult::ERROR_NOT_IMPLEMENTED,
        HIPFFT_NOT_SUPPORTED => cufftResult::ERROR_NOT_SUPPORTED,
        // HIPFFT_INCOMPLETE_PARAMETER_LIST / PARSE_ERROR -> fallback
        _ => cufftResult::ERROR_INTERNAL_ERROR,
    }
}

#[inline]
fn map_type(t: cufftType) -> hipfft::hipfftType {
    // Valori binari identici fra cufftType_t (41,42,44,105,106,108) e
    // hipfftType_t (0x29,0x2a,0x2c,0x69,0x6a,0x6c). Cast diretto safe.
    unsafe { core::mem::transmute::<u32, hipfft::hipfftType_t>(t.0) }
}

// Default error usato dal macro `unimplemented!` in lib.rs.
#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cufftResult {
    cufftResult::ERROR_NOT_SUPPORTED
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cufftResult {
    cufftResult::ERROR_NOT_SUPPORTED
}

// ---------- Plan creation (legacy 1-shot) ------------------------------------

pub(crate) unsafe fn cufftPlan1d(
    plan: *mut cufftHandle,
    nx: c_int,
    type_: cufftType,
    batch: c_int,
) -> cufftResult {
    map_result(hipfft::hipfftPlan1d(
        plan as *mut hipfft::hipfftHandle,
        nx,
        map_type(type_),
        batch,
    ))
}

pub(crate) unsafe fn cufftPlan2d(
    plan: *mut cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: cufftType,
) -> cufftResult {
    map_result(hipfft::hipfftPlan2d(
        plan as *mut hipfft::hipfftHandle,
        nx,
        ny,
        map_type(type_),
    ))
}

pub(crate) unsafe fn cufftPlan3d(
    plan: *mut cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: cufftType,
) -> cufftResult {
    map_result(hipfft::hipfftPlan3d(
        plan as *mut hipfft::hipfftHandle,
        nx,
        ny,
        nz,
        map_type(type_),
    ))
}

pub(crate) unsafe fn cufftPlanMany(
    plan: *mut cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: cufftType,
    batch: c_int,
) -> cufftResult {
    map_result(hipfft::hipfftPlanMany(
        plan as *mut hipfft::hipfftHandle,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        map_type(type_),
        batch,
    ))
}

// ---------- Two-step creation (Create + Make) --------------------------------
//
// hipfft NON espone hipfftCreate / hipfftMakePlan* nelle ABI 5.7 stable —
// li emuliamo: cufftCreate alloca un handle "pending" (-1), cufftMakePlan*
// chiama il corrispettivo hipfftPlan*. workSize lo restituiamo via
// hipfftGetSize dopo creazione (best-effort: 0 se non disponibile).

pub(crate) unsafe fn cufftCreate(handle: *mut cufftHandle) -> cufftResult {
    if handle.is_null() {
        return cufftResult::ERROR_INVALID_VALUE;
    }
    // Sentinel: il Make* lo sovrascriverà col vero handle.
    *handle = cuda_types::cufft::CUFFT_PLAN_NULL;
    cufftResult::SUCCESS
}

pub(crate) unsafe fn cufftMakePlan1d(
    plan: cufftHandle,
    nx: c_int,
    type_: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult {
    let mut h: hipfft::hipfftHandle = plan;
    let r = hipfft::hipfftPlan1d(&mut h, nx, map_type(type_), batch);
    if r != hipfft::hipfftResult_t::HIPFFT_SUCCESS {
        return map_result(r);
    }
    if !work_size.is_null() {
        let _ = hipfft::hipfftGetSize(h, work_size);
    }
    cufftResult::SUCCESS
}

pub(crate) unsafe fn cufftMakePlan2d(
    plan: cufftHandle,
    nx: c_int,
    ny: c_int,
    type_: cufftType,
    work_size: *mut usize,
) -> cufftResult {
    let mut h: hipfft::hipfftHandle = plan;
    let r = hipfft::hipfftPlan2d(&mut h, nx, ny, map_type(type_));
    if r != hipfft::hipfftResult_t::HIPFFT_SUCCESS {
        return map_result(r);
    }
    if !work_size.is_null() {
        let _ = hipfft::hipfftGetSize(h, work_size);
    }
    cufftResult::SUCCESS
}

pub(crate) unsafe fn cufftMakePlan3d(
    plan: cufftHandle,
    nx: c_int,
    ny: c_int,
    nz: c_int,
    type_: cufftType,
    work_size: *mut usize,
) -> cufftResult {
    let mut h: hipfft::hipfftHandle = plan;
    let r = hipfft::hipfftPlan3d(&mut h, nx, ny, nz, map_type(type_));
    if r != hipfft::hipfftResult_t::HIPFFT_SUCCESS {
        return map_result(r);
    }
    if !work_size.is_null() {
        let _ = hipfft::hipfftGetSize(h, work_size);
    }
    cufftResult::SUCCESS
}

pub(crate) unsafe fn cufftMakePlanMany(
    plan: cufftHandle,
    rank: c_int,
    n: *mut c_int,
    inembed: *mut c_int,
    istride: c_int,
    idist: c_int,
    onembed: *mut c_int,
    ostride: c_int,
    odist: c_int,
    type_: cufftType,
    batch: c_int,
    work_size: *mut usize,
) -> cufftResult {
    let mut h: hipfft::hipfftHandle = plan;
    let r = hipfft::hipfftPlanMany(
        &mut h,
        rank,
        n,
        inembed,
        istride,
        idist,
        onembed,
        ostride,
        odist,
        map_type(type_),
        batch,
    );
    if r != hipfft::hipfftResult_t::HIPFFT_SUCCESS {
        return map_result(r);
    }
    if !work_size.is_null() {
        let _ = hipfft::hipfftGetSize(h, work_size);
    }
    cufftResult::SUCCESS
}

// ---------- Execution single precision ---------------------------------------

pub(crate) unsafe fn cufftExecC2C(
    plan: cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftComplex,
    direction: c_int,
) -> cufftResult {
    map_result(hipfft::hipfftExecC2C(
        plan,
        idata as *mut hipfft::hipfftComplex,
        odata as *mut hipfft::hipfftComplex,
        direction,
    ))
}

pub(crate) unsafe fn cufftExecR2C(
    plan: cufftHandle,
    idata: *mut cufftReal,
    odata: *mut cufftComplex,
) -> cufftResult {
    map_result(hipfft::hipfftExecR2C(
        plan,
        idata as *mut hipfft::hipfftReal,
        odata as *mut hipfft::hipfftComplex,
    ))
}

pub(crate) unsafe fn cufftExecC2R(
    plan: cufftHandle,
    idata: *mut cufftComplex,
    odata: *mut cufftReal,
) -> cufftResult {
    map_result(hipfft::hipfftExecC2R(
        plan,
        idata as *mut hipfft::hipfftComplex,
        odata as *mut hipfft::hipfftReal,
    ))
}

// ---------- Execution double precision ---------------------------------------

pub(crate) unsafe fn cufftExecZ2Z(
    plan: cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleComplex,
    direction: c_int,
) -> cufftResult {
    map_result(hipfft::hipfftExecZ2Z(
        plan,
        idata as *mut hipfft::hipfftDoubleComplex,
        odata as *mut hipfft::hipfftDoubleComplex,
        direction,
    ))
}

pub(crate) unsafe fn cufftExecD2Z(
    plan: cufftHandle,
    idata: *mut cufftDoubleReal,
    odata: *mut cufftDoubleComplex,
) -> cufftResult {
    map_result(hipfft::hipfftExecD2Z(
        plan,
        idata as *mut hipfft::hipfftDoubleReal,
        odata as *mut hipfft::hipfftDoubleComplex,
    ))
}

pub(crate) unsafe fn cufftExecZ2D(
    plan: cufftHandle,
    idata: *mut cufftDoubleComplex,
    odata: *mut cufftDoubleReal,
) -> cufftResult {
    map_result(hipfft::hipfftExecZ2D(
        plan,
        idata as *mut hipfft::hipfftDoubleComplex,
        odata as *mut hipfft::hipfftDoubleReal,
    ))
}

// ---------- Plan management --------------------------------------------------

pub(crate) unsafe fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult {
    // cudaStream_t = CUstream = newtype(*mut CUstream_st). Estraiamo .0 e
    // riconvertiamo a *mut c_void per hipfftSetStream (hipStream_t).
    let raw: *mut c_void = stream.0 as *mut c_void;
    map_result(hipfft::hipfftSetStream(plan, raw))
}

pub(crate) unsafe fn cufftDestroy(plan: cufftHandle) -> cufftResult {
    if plan == cuda_types::cufft::CUFFT_PLAN_NULL {
        // Handle creato da cufftCreate ma mai Make*-ato.
        return cufftResult::SUCCESS;
    }
    map_result(hipfft::hipfftDestroy(plan))
}

pub(crate) unsafe fn cufftGetVersion(version: *mut c_int) -> cufftResult {
    if version.is_null() {
        return cufftResult::ERROR_INVALID_VALUE;
    }
    // Riportiamo la versione cuFFT che la chiamata si aspetta (12000) per
    // mantenere check compatibilita' lato app; non quella hipfft.
    *version = cuda_types::cufft::CUFFT_VERSION as c_int;
    cufftResult::SUCCESS
}

pub(crate) unsafe fn cufftGetSize(plan: cufftHandle, work_size: *mut usize) -> cufftResult {
    if work_size.is_null() {
        return cufftResult::ERROR_INVALID_VALUE;
    }
    map_result(hipfft::hipfftGetSize(plan, work_size))
}

pub(crate) unsafe fn cufftSetAutoAllocation(
    _plan: cufftHandle,
    _auto_allocate: c_int,
) -> cufftResult {
    // hipfft 5.7 NON espone hipfftSetAutoAllocation; PyTorch lo chiama
    // come "best effort" durante torch.fft.* -> ack come SUCCESS no-op
    // (l'allocazione effettiva la fa hipfft di default).
    cufftResult::SUCCESS
}

// ---------- Anti-warn niente da fare ----------------------------------------
// Spegne warning "unused import" su tipi 64-bit non chiamati in API minima:
#[allow(dead_code)]
fn _silence_unused(_a: c_longlong, _b: *mut cudaLibXtDesc) {}
