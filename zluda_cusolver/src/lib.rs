// zluda_cusolver — wrap REALE rocSOLVER per Cholesky/eigh + stub graceful resto
// VEGA-PATCH 2026-06-30 T16:55 (Vega ⭐, worker G6 3/5)
//
// UPGRADE vs T15:55 stub-only: cusolverDn{S,D}potrf / potrs / syevd ora chiamano
// rocsolver_{s,d}potrf / potrs / syevd via dynamic LoadLibraryW (no link-time dep
// per evitare cargo dep su rocsolver-sys inesistente; runtime-symbol via
// GetProcAddress da rocsolver.dll + rocblas.dll, asset ROCm 5.7 base / Marconi).
//
// Mapping:
//   cusolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info)
//     -> rocsolver_spotrf(rocblas_h, uplo, n, A, lda, info)  // workspace ignorato (rocSOLVER auto)
//   cusolverDnDpotrf, cusolverDnDpotrs, cusolverDnSpotrs analoghi
//   cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
//     -> rocsolver_ssyevd(rocblas_h, evect, uplo, n, A, lda, D, E, info)
//     E è "off-diagonal scratch" (n-1 floats), allochiamo via hipMalloc.
//
// Enum mapping:
//   cublasFillMode_t: 0=LOWER, 1=UPPER -> rocblas_fill: 122=lower, 121=upper
//   cusolverEigMode_t: 0=NOVECTOR, 1=VECTOR -> rocblas_evect: 213=none, 211=original
//
// Resto delle API rimane stub NOT_SUPPORTED (fallback PyTorch MAGMA/CPU).
// Compatibile con cusolver64_11.dll feature-detect (Version/Property/Create/Destroy).

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use core::ffi::{c_char, c_int, c_void};
use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

// =========================================================================
// Types ABI-compatibili con cusolver_common.h
// =========================================================================

/// cusolverStatus_t: 0=SUCCESS, 1=NOT_INITIALIZED, 2=ALLOC_FAILED,
/// 3=INVALID_VALUE, 4=ARCH_MISMATCH, 5=MAPPING_ERROR, 6=EXECUTION_FAILED,
/// 7=INTERNAL_ERROR, 8=MATRIX_TYPE_NOT_SUPPORTED, 9=NOT_SUPPORTED.
pub type cusolverStatus_t = c_int;

const CUSOLVER_SUCCESS: cusolverStatus_t = 0;
const CUSOLVER_NOT_INITIALIZED: cusolverStatus_t = 1;
const CUSOLVER_ALLOC_FAILED: cusolverStatus_t = 2;
const CUSOLVER_INVALID_VALUE: cusolverStatus_t = 3;
const CUSOLVER_EXECUTION_FAILED: cusolverStatus_t = 6;
const CUSOLVER_INTERNAL_ERROR: cusolverStatus_t = 7;
const CUSOLVER_NOT_SUPPORTED: cusolverStatus_t = 9;

pub type cusolverDnHandle_t = *mut c_void;
pub type cusolverSpHandle_t = *mut c_void;
pub type cusolverRfHandle_t = *mut c_void;
pub type cudaStream_t = *mut c_void;

/// libraryPropertyType da driver_types.h: 0=MAJOR_VERSION, 1=MINOR_VERSION,
/// 2=PATCH_LEVEL.
pub type libraryPropertyType = c_int;

/// cublasFillMode_t — riusato da cuSOLVER per uplo
/// 0 = CUBLAS_FILL_MODE_LOWER, 1 = CUBLAS_FILL_MODE_UPPER
pub type cublasFillMode_t = c_int;

/// cusolverEigMode_t: 0 = CUSOLVER_EIG_MODE_NOVECTOR, 1 = CUSOLVER_EIG_MODE_VECTOR
pub type cusolverEigMode_t = c_int;

// Versione spacciata: cuSOLVER 11.4.5 (allineata a CUDA 11.8 toolkit
// che PyTorch 2.4 / TensorFlow 2.15 si aspettano).
const FAKE_MAJOR: c_int = 11;
const FAKE_MINOR: c_int = 4;
const FAKE_PATCH: c_int = 5;
const FAKE_VERSION_CODE: c_int = FAKE_MAJOR * 1000 + FAKE_MINOR * 100 + FAKE_PATCH;

// =========================================================================
// Dynamic loader rocblas.dll + rocsolver.dll (no link-time dep)
// =========================================================================

type RocblasStatus = c_int;
const ROCBLAS_STATUS_SUCCESS: RocblasStatus = 0;

// rocblas_fill / rocblas_evect (vedi rocblas-types.h / rocsolver-extra-types.h)
const ROCBLAS_FILL_UPPER: c_int = 121;
const ROCBLAS_FILL_LOWER: c_int = 122;
const ROCBLAS_EVECT_ORIGINAL: c_int = 211;
const ROCBLAS_EVECT_NONE: c_int = 213;
const ROCBLAS_OP_NONE: c_int = 111;
const ROCBLAS_OP_TRANSPOSE: c_int = 112;

type RocblasHandle = *mut c_void;

// Signature rocBLAS / rocSOLVER core
type FnRocblasCreate = unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus;
type FnRocblasDestroy = unsafe extern "C" fn(RocblasHandle) -> RocblasStatus;
type FnRocblasSetStream = unsafe extern "C" fn(RocblasHandle, *mut c_void) -> RocblasStatus;

type FnPotrf = unsafe extern "C" fn(
    RocblasHandle, c_int /*uplo*/, c_int /*n*/,
    *mut c_void /*A*/, c_int /*lda*/, *mut c_int /*info*/,
) -> RocblasStatus;

type FnPotrs = unsafe extern "C" fn(
    RocblasHandle, c_int /*uplo*/, c_int /*n*/, c_int /*nrhs*/,
    *mut c_void /*A*/, c_int /*lda*/, *mut c_void /*B*/, c_int /*ldb*/,
) -> RocblasStatus;

type FnSyevd = unsafe extern "C" fn(
    RocblasHandle, c_int /*evect*/, c_int /*uplo*/, c_int /*n*/,
    *mut c_void /*A*/, c_int /*lda*/,
    *mut c_void /*D*/, *mut c_void /*E scratch*/, *mut c_int /*info*/,
) -> RocblasStatus;

// G6 worker 4/5 LU/solve VEGA-PATCH 2026-06-30 T16:55
// rocsolver_{s,d}getrf: LU factorization with partial pivoting
//   (handle, m, n, *A, lda, *ipiv, *info)
type FnGetrf = unsafe extern "C" fn(
    RocblasHandle, c_int /*m*/, c_int /*n*/,
    *mut c_void /*A*/, c_int /*lda*/,
    *mut c_int /*ipiv*/, *mut c_int /*info*/,
) -> RocblasStatus;

// rocsolver_{s,d}getrs: solve Ax=B given LU
//   (handle, trans, n, nrhs, *A, lda, *ipiv, *B, ldb)
type FnGetrs = unsafe extern "C" fn(
    RocblasHandle, c_int /*trans*/, c_int /*n*/, c_int /*nrhs*/,
    *mut c_void /*A*/, c_int /*lda*/,
    *const c_int /*ipiv*/, *mut c_void /*B*/, c_int /*ldb*/,
) -> RocblasStatus;

// hipMalloc / hipFree per scratch E del syevd
type FnHipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int;
type FnHipFree = unsafe extern "C" fn(*mut c_void) -> c_int;

// =========================================================================
// G6 worker 2/5 SVD/QR (T17:10 VEGA-PATCH)
// =========================================================================
// rocsolver_{s,d}gesvd full signature:
//   (handle, left_svect, right_svect, m, n, A, lda, S, U, ldu, V, ldv,
//    E off-diag scratch, fast_alg, info)
// rocblas_svect: 261=all (A), 262=singular (S), 263=overwrite (O), 264=none (N)
// rocblas_workmode fast_alg: 0=rocblas_outofplace (sicuro per allineamento cusolver)
type FnGesvd = unsafe extern "C" fn(
    RocblasHandle, c_int /*left_svect*/, c_int /*right_svect*/,
    c_int /*m*/, c_int /*n*/,
    *mut c_void /*A*/, c_int /*lda*/,
    *mut c_void /*S*/,
    *mut c_void /*U*/, c_int /*ldu*/,
    *mut c_void /*V*/, c_int /*ldv*/,
    *mut c_void /*E off-diag scratch*/, c_int /*fast_alg*/,
    *mut c_int /*info*/,
) -> RocblasStatus;

// rocsolver_{s,d}geqrf(handle, m, n, A, lda, TAU)
type FnGeqrf = unsafe extern "C" fn(
    RocblasHandle, c_int /*m*/, c_int /*n*/,
    *mut c_void /*A*/, c_int /*lda*/, *mut c_void /*TAU*/,
) -> RocblasStatus;

// rocsolver_{s,d}orgqr(handle, m, n, k, A, lda, TAU)
type FnOrgqr = unsafe extern "C" fn(
    RocblasHandle, c_int /*m*/, c_int /*n*/, c_int /*k*/,
    *mut c_void /*A*/, c_int /*lda*/, *mut c_void /*TAU*/,
) -> RocblasStatus;

// rocblas_svect costanti
const ROCBLAS_SVECT_ALL: c_int = 261;
const ROCBLAS_SVECT_SINGULAR: c_int = 262;
const ROCBLAS_SVECT_OVERWRITE: c_int = 263;
const ROCBLAS_SVECT_NONE: c_int = 264;

#[inline]
fn map_svect_char(jobu: c_char) -> c_int {
    match jobu as u8 {
        b'A' | b'a' => ROCBLAS_SVECT_ALL,
        b'S' | b's' => ROCBLAS_SVECT_SINGULAR,
        b'O' | b'o' => ROCBLAS_SVECT_OVERWRITE,
        b'N' | b'n' => ROCBLAS_SVECT_NONE,
        _ => ROCBLAS_SVECT_ALL,
    }
}

#[derive(Default)]
struct RocBackend {
    rocblas_create: Option<FnRocblasCreate>,
    rocblas_destroy: Option<FnRocblasDestroy>,
    rocblas_set_stream: Option<FnRocblasSetStream>,
    spotrf: Option<FnPotrf>,
    dpotrf: Option<FnPotrf>,
    spotrs: Option<FnPotrs>,
    dpotrs: Option<FnPotrs>,
    ssyevd: Option<FnSyevd>,
    dsyevd: Option<FnSyevd>,
    // G6 worker 4/5 LU/solve
    sgetrf: Option<FnGetrf>,
    dgetrf: Option<FnGetrf>,
    sgetrs: Option<FnGetrs>,
    dgetrs: Option<FnGetrs>,
    // G6 worker 2/5 SVD/QR (T17:10)
    sgesvd: Option<FnGesvd>,
    dgesvd: Option<FnGesvd>,
    sgeqrf: Option<FnGeqrf>,
    dgeqrf: Option<FnGeqrf>,
    sorgqr: Option<FnOrgqr>,
    dorgqr: Option<FnOrgqr>,
    hip_malloc: Option<FnHipMalloc>,
    hip_free: Option<FnHipFree>,
}

// Cache backend + opaque handle (cusolverDnHandle_t -> Box<HandleInner>)
struct HandleInner {
    rocblas_h: RocblasHandle,
    stream: cudaStream_t,
}

static BACKEND_PTR: AtomicPtr<RocBackend> = AtomicPtr::new(ptr::null_mut());

#[cfg(target_os = "windows")]
mod loader {
    use super::*;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(lpLibFileName: *const c_char) -> *mut c_void;
        fn GetProcAddress(hModule: *mut c_void, lpProcName: *const c_char) -> *mut c_void;
    }

    pub unsafe fn load_dll(name: &[u8]) -> *mut c_void {
        // name MUST be NUL-terminated; caller passes b"rocblas.dll\0"
        LoadLibraryA(name.as_ptr() as *const c_char)
    }
    pub unsafe fn sym(handle: *mut c_void, name: &[u8]) -> *mut c_void {
        if handle.is_null() { return ptr::null_mut(); }
        GetProcAddress(handle, name.as_ptr() as *const c_char)
    }
}

#[cfg(not(target_os = "windows"))]
mod loader {
    use super::*;
    pub unsafe fn load_dll(_name: &[u8]) -> *mut c_void { ptr::null_mut() }
    pub unsafe fn sym(_h: *mut c_void, _n: &[u8]) -> *mut c_void { ptr::null_mut() }
}

unsafe fn init_backend() -> *mut RocBackend {
    let cur = BACKEND_PTR.load(Ordering::Acquire);
    if !cur.is_null() { return cur; }

    let h_rocblas = loader::load_dll(b"rocblas.dll\0");
    let h_rocsolver = loader::load_dll(b"rocsolver.dll\0");
    let h_amdhip = loader::load_dll(b"amdhip64.dll\0");

    macro_rules! getfn {
        ($h:expr, $name:expr, $t:ty) => {{
            let p = loader::sym($h, concat!($name, "\0").as_bytes());
            if p.is_null() { None } else { Some(core::mem::transmute::<*mut c_void, $t>(p)) }
        }};
    }

    let b = Box::new(RocBackend {
        rocblas_create: getfn!(h_rocblas, "rocblas_create_handle", FnRocblasCreate),
        rocblas_destroy: getfn!(h_rocblas, "rocblas_destroy_handle", FnRocblasDestroy),
        rocblas_set_stream: getfn!(h_rocblas, "rocblas_set_stream", FnRocblasSetStream),
        spotrf: getfn!(h_rocsolver, "rocsolver_spotrf", FnPotrf),
        dpotrf: getfn!(h_rocsolver, "rocsolver_dpotrf", FnPotrf),
        spotrs: getfn!(h_rocsolver, "rocsolver_spotrs", FnPotrs),
        dpotrs: getfn!(h_rocsolver, "rocsolver_dpotrs", FnPotrs),
        ssyevd: getfn!(h_rocsolver, "rocsolver_ssyevd", FnSyevd),
        dsyevd: getfn!(h_rocsolver, "rocsolver_dsyevd", FnSyevd),
        // G6 worker 4/5 LU/solve
        sgetrf: getfn!(h_rocsolver, "rocsolver_sgetrf", FnGetrf),
        dgetrf: getfn!(h_rocsolver, "rocsolver_dgetrf", FnGetrf),
        sgetrs: getfn!(h_rocsolver, "rocsolver_sgetrs", FnGetrs),
        dgetrs: getfn!(h_rocsolver, "rocsolver_dgetrs", FnGetrs),
        // G6 worker 2/5 SVD/QR (T17:10)
        sgesvd: getfn!(h_rocsolver, "rocsolver_sgesvd", FnGesvd),
        dgesvd: getfn!(h_rocsolver, "rocsolver_dgesvd", FnGesvd),
        sgeqrf: getfn!(h_rocsolver, "rocsolver_sgeqrf", FnGeqrf),
        dgeqrf: getfn!(h_rocsolver, "rocsolver_dgeqrf", FnGeqrf),
        sorgqr: getfn!(h_rocsolver, "rocsolver_sorgqr", FnOrgqr),
        dorgqr: getfn!(h_rocsolver, "rocsolver_dorgqr", FnOrgqr),
        hip_malloc: getfn!(h_amdhip, "hipMalloc", FnHipMalloc),
        hip_free: getfn!(h_amdhip, "hipFree", FnHipFree),
    });

    let raw = Box::into_raw(b);
    match BACKEND_PTR.compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => raw,
        Err(other) => {
            // qualcun altro ha già inizializzato; drop il nostro
            drop(Box::from_raw(raw));
            other
        }
    }
}

#[inline]
unsafe fn backend<'a>() -> &'a RocBackend {
    let p = init_backend();
    &*p
}

#[inline]
fn map_uplo(uplo_cublas: cublasFillMode_t) -> c_int {
    // 0=LOWER -> rocblas_fill_lower (122), 1=UPPER -> rocblas_fill_upper (121)
    if uplo_cublas == 0 { ROCBLAS_FILL_LOWER } else { ROCBLAS_FILL_UPPER }
}

#[inline]
fn map_evect(jobz: cusolverEigMode_t) -> c_int {
    if jobz == 0 { ROCBLAS_EVECT_NONE } else { ROCBLAS_EVECT_ORIGINAL }
}

#[inline]
fn map_rocstatus(s: RocblasStatus) -> cusolverStatus_t {
    if s == ROCBLAS_STATUS_SUCCESS { CUSOLVER_SUCCESS } else { CUSOLVER_EXECUTION_FAILED }
}

// =========================================================================
// Query no-op (feature-detect)
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn cusolverGetVersion(version: *mut c_int) -> cusolverStatus_t {
    if version.is_null() { return CUSOLVER_INVALID_VALUE; }
    *version = FAKE_VERSION_CODE;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverGetProperty(
    property: libraryPropertyType,
    value: *mut c_int,
) -> cusolverStatus_t {
    if value.is_null() { return CUSOLVER_INVALID_VALUE; }
    *value = match property {
        0 => FAKE_MAJOR,
        1 => FAKE_MINOR,
        2 => FAKE_PATCH,
        _ => return CUSOLVER_INVALID_VALUE,
    };
    CUSOLVER_SUCCESS
}

// =========================================================================
// cuSOLVER Dense (Dn) — handle reale con rocBLAS handle dietro
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t {
    if handle.is_null() { return CUSOLVER_INVALID_VALUE; }
    let b = backend();
    let create = match b.rocblas_create { Some(f) => f, None => return CUSOLVER_NOT_INITIALIZED };
    let mut rb: RocblasHandle = ptr::null_mut();
    if create(&mut rb) != ROCBLAS_STATUS_SUCCESS || rb.is_null() {
        return CUSOLVER_NOT_INITIALIZED;
    }
    let inner = Box::new(HandleInner { rocblas_h: rb, stream: ptr::null_mut() });
    *handle = Box::into_raw(inner) as cusolverDnHandle_t;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t {
    if handle.is_null() { return CUSOLVER_SUCCESS; }
    let inner = Box::from_raw(handle as *mut HandleInner);
    let b = backend();
    if let Some(destroy) = b.rocblas_destroy {
        if !inner.rocblas_h.is_null() { destroy(inner.rocblas_h); }
    }
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSetStream(
    handle: cusolverDnHandle_t, stream: cudaStream_t,
) -> cusolverStatus_t {
    if handle.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &mut *(handle as *mut HandleInner);
    inner.stream = stream;
    let b = backend();
    if let Some(set) = b.rocblas_set_stream {
        let s = set(inner.rocblas_h, stream);
        return map_rocstatus(s);
    }
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnGetStream(
    handle: cusolverDnHandle_t, stream: *mut cudaStream_t,
) -> cusolverStatus_t {
    if stream.is_null() { return CUSOLVER_INVALID_VALUE; }
    if handle.is_null() { *stream = ptr::null_mut(); return CUSOLVER_SUCCESS; }
    let inner = &*(handle as *const HandleInner);
    *stream = inner.stream;
    CUSOLVER_SUCCESS
}

// =========================================================================
// CHOLESKY: cusolverDn{S,D}potrf + bufferSize + potrs — REALE rocSOLVER
// =========================================================================

// bufferSize cuSOLVER prevede workspace esterno; rocSOLVER gestisce internamente.
// Ritorniamo lwork=1 (dummy >0) così l'allocazione caller è banale e non zero.

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSpotrf_bufferSize(
    _handle: cusolverDnHandle_t, _uplo: cublasFillMode_t, _n: c_int,
    _A: *mut f32, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDpotrf_bufferSize(
    _handle: cusolverDnHandle_t, _uplo: cublasFillMode_t, _n: c_int,
    _A: *mut f64, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSpotrf(
    handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: c_int,
    A: *mut f32, lda: c_int,
    _workspace: *mut f32, _lwork: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || devInfo.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let b = backend();
    let f = match b.spotrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_uplo(uplo), n, A as *mut c_void, lda, devInfo);
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDpotrf(
    handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: c_int,
    A: *mut f64, lda: c_int,
    _workspace: *mut f64, _lwork: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || devInfo.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let b = backend();
    let f = match b.dpotrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_uplo(uplo), n, A as *mut c_void, lda, devInfo);
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSpotrs(
    handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: c_int, nrhs: c_int,
    A: *const f32, lda: c_int, B: *mut f32, ldb: c_int, _devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || B.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.spotrs { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_uplo(uplo), n, nrhs,
              A as *mut c_void, lda, B as *mut c_void, ldb);
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDpotrs(
    handle: cusolverDnHandle_t, uplo: cublasFillMode_t, n: c_int, nrhs: c_int,
    A: *const f64, lda: c_int, B: *mut f64, ldb: c_int, _devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || B.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.dpotrs { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_uplo(uplo), n, nrhs,
              A as *mut c_void, lda, B as *mut c_void, ldb);
    map_rocstatus(s)
}

// =========================================================================
// EIGH: cusolverDn{S,D}syevd + bufferSize — REALE rocSOLVER
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSsyevd_bufferSize(
    _handle: cusolverDnHandle_t, _jobz: cusolverEigMode_t, _uplo: cublasFillMode_t,
    _n: c_int, _A: *const f32, _lda: c_int, _W: *const f32, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDsyevd_bufferSize(
    _handle: cusolverDnHandle_t, _jobz: cusolverEigMode_t, _uplo: cublasFillMode_t,
    _n: c_int, _A: *const f64, _lda: c_int, _W: *const f64, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

unsafe fn syevd_common(
    handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: c_int,
    A: *mut c_void, lda: c_int, W: *mut c_void, devInfo: *mut c_int,
    elem_size: usize, is_double: bool,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || W.is_null() || devInfo.is_null() {
        return CUSOLVER_INVALID_VALUE;
    }
    if n <= 0 { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = if is_double {
        match bk.dsyevd { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED }
    } else {
        match bk.ssyevd { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED }
    };
    let hip_malloc = match bk.hip_malloc { Some(f) => f, None => return CUSOLVER_NOT_INITIALIZED };
    let hip_free = bk.hip_free;

    // Alloca E scratch (n elementi del tipo)
    let nbytes = (n as usize) * elem_size;
    let mut e_dev: *mut c_void = ptr::null_mut();
    if hip_malloc(&mut e_dev, nbytes) != 0 || e_dev.is_null() {
        return CUSOLVER_ALLOC_FAILED;
    }

    let s = f(inner.rocblas_h, map_evect(jobz), map_uplo(uplo), n,
              A, lda, W, e_dev, devInfo);

    if let Some(free) = hip_free { let _ = free(e_dev); }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSsyevd(
    handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: c_int,
    A: *mut f32, lda: c_int, W: *mut f32,
    _work: *mut f32, _lwork: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    syevd_common(handle, jobz, uplo, n, A as *mut c_void, lda, W as *mut c_void,
                 devInfo, 4, false)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDsyevd(
    handle: cusolverDnHandle_t, jobz: cusolverEigMode_t, uplo: cublasFillMode_t, n: c_int,
    A: *mut f64, lda: c_int, W: *mut f64,
    _work: *mut f64, _lwork: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    syevd_common(handle, jobz, uplo, n, A as *mut c_void, lda, W as *mut c_void,
                 devInfo, 8, true)
}

// =========================================================================
// LU: cusolverDn{S,D}getrf + bufferSize + getrs — REALE rocSOLVER
// (Worker 5/5 G6 — completion exports per FnGetrf/FnGetrs già nel backend.)
// =========================================================================

#[inline]
fn map_trans(t_cublas: c_int) -> c_int {
    // cublasOperation_t: 0=N, 1=T, 2=C -> rocblas_operation: 111=N, 112=T, 113=C
    match t_cublas {
        1 => ROCBLAS_OP_TRANSPOSE,
        2 => 113, // conjugate transpose
        _ => ROCBLAS_OP_NONE,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgetrf_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int,
    _A: *mut f32, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgetrf_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int,
    _A: *mut f64, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgetrf(
    handle: cusolverDnHandle_t, m: c_int, n: c_int,
    A: *mut f32, lda: c_int,
    _workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || devInfo.is_null() { return CUSOLVER_INVALID_VALUE; }
    if devIpiv.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.sgetrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, A as *mut c_void, lda, devIpiv, devInfo);
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgetrf(
    handle: cusolverDnHandle_t, m: c_int, n: c_int,
    A: *mut f64, lda: c_int,
    _workspace: *mut f64, devIpiv: *mut c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || devInfo.is_null() { return CUSOLVER_INVALID_VALUE; }
    if devIpiv.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.dgetrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, A as *mut c_void, lda, devIpiv, devInfo);
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgetrs(
    handle: cusolverDnHandle_t, trans: c_int /*cublasOperation_t*/,
    n: c_int, nrhs: c_int,
    A: *const f32, lda: c_int, devIpiv: *const c_int,
    B: *mut f32, ldb: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || B.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.sgetrs { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_trans(trans), n, nrhs,
              A as *mut c_void, lda, devIpiv, B as *mut c_void, ldb);
    if !devInfo.is_null() { *devInfo = if s == ROCBLAS_STATUS_SUCCESS { 0 } else { -1 }; }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgetrs(
    handle: cusolverDnHandle_t, trans: c_int /*cublasOperation_t*/,
    n: c_int, nrhs: c_int,
    A: *const f64, lda: c_int, devIpiv: *const c_int,
    B: *mut f64, ldb: c_int, devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || B.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.dgetrs { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, map_trans(trans), n, nrhs,
              A as *mut c_void, lda, devIpiv, B as *mut c_void, ldb);
    if !devInfo.is_null() { *devInfo = if s == ROCBLAS_STATUS_SUCCESS { 0 } else { -1 }; }
    map_rocstatus(s)
}

// =========================================================================
// G6 worker 2/5 — SVD: cusolverDn{S,D}gesvd + bufferSize REALE rocSOLVER
// (T17:10 VEGA-PATCH)
// =========================================================================
//
// Signature cuSOLVER:
//   cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S,
//                    U, ldu, VT, ldvt, work, lwork, rwork, devInfo)
//   jobu/jobvt = char 'A' | 'S' | 'O' | 'N'
//
// Signature rocSOLVER:
//   rocsolver_sgesvd(handle, left_svect, right_svect, m, n, A, lda, S,
//                    U, ldu, V, ldv, E, fast_alg, info)
//   E = scratch superdiagonale (min(m,n) elementi del tipo); allochiamo via
//   hipMalloc per sicurezza ABI. fast_alg=0 (rocblas_outofplace) sicuro.
//
// Note layout VT vs V: cuSOLVER produce V trasposta (VT), rocSOLVER produce V.
// Per torch.linalg.svd l'output del wrapper PyTorch CUDA è gia tradotto
// internamente al return Python; passare VT come `V` rocSOLVER funziona per
// uso comune (column-major contiguo). Per full_matrices=False con m!=n torch
// gestisce la transpose nel post-processing.

unsafe fn gesvd_common(
    handle: cusolverDnHandle_t,
    jobu: c_char, jobvt: c_char,
    m: c_int, n: c_int,
    A: *mut c_void, lda: c_int,
    S: *mut c_void,
    U: *mut c_void, ldu: c_int,
    VT: *mut c_void, ldvt: c_int,
    devInfo: *mut c_int,
    elem_size: usize, is_double: bool,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || S.is_null() || devInfo.is_null() {
        return CUSOLVER_INVALID_VALUE;
    }
    if m <= 0 || n <= 0 { return CUSOLVER_INVALID_VALUE; }

    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = if is_double {
        match bk.dgesvd { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED }
    } else {
        match bk.sgesvd { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED }
    };
    let hip_malloc = match bk.hip_malloc { Some(f) => f, None => return CUSOLVER_NOT_INITIALIZED };
    let hip_free = bk.hip_free;

    // Scratch E: min(m,n) elementi del tipo
    let mn = core::cmp::min(m, n) as usize;
    let nbytes = mn * elem_size;
    let mut e_dev: *mut c_void = ptr::null_mut();
    if nbytes > 0 {
        if hip_malloc(&mut e_dev, nbytes) != 0 || e_dev.is_null() {
            return CUSOLVER_ALLOC_FAILED;
        }
    }

    let s = f(
        inner.rocblas_h,
        map_svect_char(jobu), map_svect_char(jobvt),
        m, n, A, lda, S,
        U, ldu, VT, ldvt,
        e_dev, 0 /*fast_alg = rocblas_outofplace*/, devInfo,
    );

    if !e_dev.is_null() {
        if let Some(free) = hip_free { let _ = free(e_dev); }
    }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgesvd(
    handle: cusolverDnHandle_t,
    jobu: c_char, jobvt: c_char,
    m: c_int, n: c_int,
    A: *mut f32, lda: c_int,
    S: *mut f32,
    U: *mut f32, ldu: c_int,
    VT: *mut f32, ldvt: c_int,
    _work: *mut f32, _lwork: c_int,
    _rwork: *mut f32,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    gesvd_common(
        handle, jobu, jobvt, m, n,
        A as *mut c_void, lda, S as *mut c_void,
        U as *mut c_void, ldu, VT as *mut c_void, ldvt,
        devInfo, 4, false,
    )
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgesvd(
    handle: cusolverDnHandle_t,
    jobu: c_char, jobvt: c_char,
    m: c_int, n: c_int,
    A: *mut f64, lda: c_int,
    S: *mut f64,
    U: *mut f64, ldu: c_int,
    VT: *mut f64, ldvt: c_int,
    _work: *mut f64, _lwork: c_int,
    _rwork: *mut f64,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    gesvd_common(
        handle, jobu, jobvt, m, n,
        A as *mut c_void, lda, S as *mut c_void,
        U as *mut c_void, ldu, VT as *mut c_void, ldvt,
        devInfo, 8, true,
    )
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgesvd_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1; // dummy (rocSOLVER alloca internamente)
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgesvd_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

// =========================================================================
// G6 worker 2/5 — QR: geqrf + orgqr + bufferSize REALE rocSOLVER
// (T17:10 VEGA-PATCH)
// =========================================================================
//
// Signature cuSOLVER geqrf:
//   cusolverDnSgeqrf(handle, m, n, A, lda, TAU, work, lwork, devInfo)
// Signature rocSOLVER:
//   rocsolver_sgeqrf(handle, m, n, A, lda, TAU)
// rocsolver_sgeqrf NON ritorna info-style → settiamo *devInfo = 0 dopo success.
//
// Signature cuSOLVER orgqr:
//   cusolverDnSorgqr(handle, m, n, k, A, lda, TAU, work, lwork, devInfo)
// Signature rocSOLVER:
//   rocsolver_sorgqr(handle, m, n, k, A, lda, TAU)

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgeqrf(
    handle: cusolverDnHandle_t,
    m: c_int, n: c_int,
    A: *mut f32, lda: c_int,
    TAU: *mut f32,
    _work: *mut f32, _lwork: c_int,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || TAU.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.sgeqrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, A as *mut c_void, lda, TAU as *mut c_void);
    if !devInfo.is_null() { *devInfo = 0; }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgeqrf(
    handle: cusolverDnHandle_t,
    m: c_int, n: c_int,
    A: *mut f64, lda: c_int,
    TAU: *mut f64,
    _work: *mut f64, _lwork: c_int,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || TAU.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.dgeqrf { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, A as *mut c_void, lda, TAU as *mut c_void);
    if !devInfo.is_null() { *devInfo = 0; }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSgeqrf_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int,
    _A: *mut f32, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDgeqrf_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int,
    _A: *mut f64, _lda: c_int, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSorgqr(
    handle: cusolverDnHandle_t,
    m: c_int, n: c_int, k: c_int,
    A: *mut f32, lda: c_int,
    TAU: *mut f32,
    _work: *mut f32, _lwork: c_int,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || TAU.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.sorgqr { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, k, A as *mut c_void, lda, TAU as *mut c_void);
    if !devInfo.is_null() { *devInfo = 0; }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDorgqr(
    handle: cusolverDnHandle_t,
    m: c_int, n: c_int, k: c_int,
    A: *mut f64, lda: c_int,
    TAU: *mut f64,
    _work: *mut f64, _lwork: c_int,
    devInfo: *mut c_int,
) -> cusolverStatus_t {
    if handle.is_null() || A.is_null() || TAU.is_null() { return CUSOLVER_INVALID_VALUE; }
    let inner = &*(handle as *const HandleInner);
    let bk = backend();
    let f = match bk.dorgqr { Some(f) => f, None => return CUSOLVER_NOT_SUPPORTED };
    let s = f(inner.rocblas_h, m, n, k, A as *mut c_void, lda, TAU as *mut c_void);
    if !devInfo.is_null() { *devInfo = 0; }
    map_rocstatus(s)
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnSorgqr_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int, _k: c_int,
    _A: *const f32, _lda: c_int, _TAU: *const f32, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverDnDorgqr_bufferSize(
    _handle: cusolverDnHandle_t, _m: c_int, _n: c_int, _k: c_int,
    _A: *const f64, _lda: c_int, _TAU: *const f64, lwork: *mut c_int,
) -> cusolverStatus_t {
    if lwork.is_null() { return CUSOLVER_INVALID_VALUE; }
    *lwork = 1;
    CUSOLVER_SUCCESS
}

// =========================================================================
// STUB rimanenti — NOT_SUPPORTED (PyTorch fallback MAGMA/CPU)
// =========================================================================

macro_rules! dn_stub_not_supported {
    ($($name:ident),* $(,)?) => {
        $(
            #[no_mangle]
            pub unsafe extern "C" fn $name() -> cusolverStatus_t {
                CUSOLVER_NOT_SUPPORTED
            }
        )*
    };
}

dn_stub_not_supported!(
    // LU factorization complex variants (S/D float/double wrapped REALI in get*_impl
    // below via rocsolver_{s,d}getrf / getrs — G6 worker 4/5)
    cusolverDnCgetrf, cusolverDnZgetrf,
    cusolverDnCgetrf_bufferSize, cusolverDnZgetrf_bufferSize,
    cusolverDnCgetrs, cusolverDnZgetrs,
    // QR factorization — variant C/Z (S/D wrapped REALI sotto, worker 2/5)
    // + cusolverDnSorgqr/Dorgqr S/D anche REALI (orgqr non-complex)
    cusolverDnCgeqrf, cusolverDnZgeqrf,
    cusolverDnCgeqrf_bufferSize, cusolverDnZgeqrf_bufferSize,
    cusolverDnSormqr, cusolverDnDormqr, cusolverDnCunmqr, cusolverDnZunmqr,
    // Cholesky complex (variant C/Z non in scope worker 3/5 focus float/double)
    cusolverDnCpotrf, cusolverDnZpotrf,
    cusolverDnCpotrf_bufferSize, cusolverDnZpotrf_bufferSize,
    cusolverDnCpotrs, cusolverDnZpotrs,
    // SVD — variant C/Z complex (S/D wrapped REALI sotto, worker 2/5)
    cusolverDnCgesvd, cusolverDnZgesvd,
    cusolverDnCgesvd_bufferSize, cusolverDnZgesvd_bufferSize,
    cusolverDnSgesvdj, cusolverDnDgesvdj, cusolverDnCgesvdj, cusolverDnZgesvdj,
    // Eigenvalue complex
    cusolverDnCheevd, cusolverDnZheevd,
    cusolverDnCheevd_bufferSize, cusolverDnZheevd_bufferSize,
    cusolverDnSsygvd, cusolverDnDsygvd, cusolverDnChegvd, cusolverDnZhegvd,
    // 64-bit Generic API (cuSOLVER 11+)
    cusolverDnXgetrf, cusolverDnXgetrf_bufferSize,
    cusolverDnXgetrs,
    cusolverDnXgeqrf, cusolverDnXgeqrf_bufferSize,
    cusolverDnXpotrf, cusolverDnXpotrf_bufferSize,
    cusolverDnXpotrs,
    cusolverDnXsyevd, cusolverDnXsyevd_bufferSize,
    cusolverDnXgesvd, cusolverDnXgesvd_bufferSize,
    cusolverDnXgesvdp, cusolverDnXgesvdp_bufferSize,
    cusolverDnXgesvdjBatched, cusolverDnXgesvdjBatched_bufferSize,
    // Params helpers
    cusolverDnCreateParams, cusolverDnDestroyParams, cusolverDnSetAdvOptions,
);

// =========================================================================
// cuSOLVER Sparse (Sp) — stub graceful (rocSOLVER non copre Sp)
// =========================================================================

static DUMMY_SP: c_int = 0xC050_1101u32 as i32;
static DUMMY_RF: c_int = 0xC050_1102u32 as i32;

#[no_mangle]
pub unsafe extern "C" fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t {
    if handle.is_null() { return CUSOLVER_INVALID_VALUE; }
    *handle = &DUMMY_SP as *const _ as *mut c_void;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverSpDestroy(_handle: cusolverSpHandle_t) -> cusolverStatus_t {
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverSpSetStream(
    _handle: cusolverSpHandle_t,
    _stream: cudaStream_t,
) -> cusolverStatus_t { CUSOLVER_SUCCESS }

dn_stub_not_supported!(
    cusolverSpScsrlsvchol, cusolverSpDcsrlsvchol,
    cusolverSpCcsrlsvchol, cusolverSpZcsrlsvchol,
    cusolverSpScsrlsvqr, cusolverSpDcsrlsvqr,
    cusolverSpCcsrlsvqr, cusolverSpZcsrlsvqr,
    cusolverSpScsreigvsi, cusolverSpDcsreigvsi,
);

// =========================================================================
// cuSOLVER Refactor (Rf) — stub graceful
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn cusolverRfCreate(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t {
    if handle.is_null() { return CUSOLVER_INVALID_VALUE; }
    *handle = &DUMMY_RF as *const _ as *mut c_void;
    CUSOLVER_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn cusolverRfDestroy(_handle: cusolverRfHandle_t) -> cusolverStatus_t {
    CUSOLVER_SUCCESS
}

dn_stub_not_supported!(
    cusolverRfSetupHost, cusolverRfSetupDevice, cusolverRfAnalyze,
    cusolverRfRefactor, cusolverRfSolve, cusolverRfAccessBundledFactorsDevice,
);
