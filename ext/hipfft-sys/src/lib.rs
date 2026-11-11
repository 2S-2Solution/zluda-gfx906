//! Minimal FFI bindings su hipfft ROCm 5.7 (Vega 20 gfx906).
//!
//! Solo le API richieste per supportare cuFFT minimo PyTorch / Whisper / SD-XL VAE:
//! - cufftPlan1d/2d/3d, cufftPlanMany
//! - cufftExecC2C / R2C / C2R (single precision — i workload target sono fp32)
//! - cufftExecZ2Z / D2Z / Z2D (double precision per safety / fallback)
//! - cufftSetStream, cufftDestroy, cufftGetVersion
//!
//! ABI ROCm 5.7: hipfft.h dichiara `hipfftHandle = int` (4 byte) e tutte le
//! `hipfft*` ritornano `hipfftResult` (u32). Stream è `hipStream_t = void*`.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::c_void;
use std::os::raw::c_int;

pub type hipStream_t = *mut c_void;

/// Plan handle: in hipfft.h è `typedef int hipfftHandle;`.
pub type hipfftHandle = c_int;

/// hipfftResult — mirror 1:1 di cufftResult (ABI sicura: enum u32).
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum hipfftResult_t {
    HIPFFT_SUCCESS = 0,
    HIPFFT_INVALID_PLAN = 1,
    HIPFFT_ALLOC_FAILED = 2,
    HIPFFT_INVALID_TYPE = 3,
    HIPFFT_INVALID_VALUE = 4,
    HIPFFT_INTERNAL_ERROR = 5,
    HIPFFT_EXEC_FAILED = 6,
    HIPFFT_SETUP_FAILED = 7,
    HIPFFT_INVALID_SIZE = 8,
    HIPFFT_UNALIGNED_DATA = 9,
    HIPFFT_INCOMPLETE_PARAMETER_LIST = 10,
    HIPFFT_INVALID_DEVICE = 11,
    HIPFFT_PARSE_ERROR = 12,
    HIPFFT_NO_WORKSPACE = 13,
    HIPFFT_NOT_IMPLEMENTED = 14,
    HIPFFT_NOT_SUPPORTED = 16,
}
pub type hipfftResult = hipfftResult_t;

/// hipfftType — mirror 1:1 di cufftType (NVIDIA & AMD usano gli stessi valori
/// numerici 0x29 / 0x2A / 0x2C / 0x69 / 0x6A / 0x6C dalla rispettiva spec).
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum hipfftType_t {
    HIPFFT_R2C = 0x2a, // 42
    HIPFFT_C2R = 0x2c, // 44
    HIPFFT_C2C = 0x29, // 41
    HIPFFT_D2Z = 0x6a, // 106
    HIPFFT_Z2D = 0x6c, // 108
    HIPFFT_Z2Z = 0x69, // 105
}
pub type hipfftType = hipfftType_t;

/// Direzione (HIPFFT_FORWARD = -1, HIPFFT_BACKWARD = 1) — stessi valori cufft.
pub const HIPFFT_FORWARD: c_int = -1;
pub const HIPFFT_BACKWARD: c_int = 1;

/// Complex types (layout identico a cuComplex / cuDoubleComplex / float2 / double2).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hipfftComplex {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct hipfftDoubleComplex {
    pub x: f64,
    pub y: f64,
}

pub type hipfftReal = f32;
pub type hipfftDoubleReal = f64;

extern "C" {
    // --- Plan creation -------------------------------------------------------
    pub fn hipfftPlan1d(
        plan: *mut hipfftHandle,
        nx: c_int,
        type_: hipfftType,
        batch: c_int,
    ) -> hipfftResult;

    pub fn hipfftPlan2d(
        plan: *mut hipfftHandle,
        nx: c_int,
        ny: c_int,
        type_: hipfftType,
    ) -> hipfftResult;

    pub fn hipfftPlan3d(
        plan: *mut hipfftHandle,
        nx: c_int,
        ny: c_int,
        nz: c_int,
        type_: hipfftType,
    ) -> hipfftResult;

    pub fn hipfftPlanMany(
        plan: *mut hipfftHandle,
        rank: c_int,
        n: *mut c_int,
        inembed: *mut c_int,
        istride: c_int,
        idist: c_int,
        onembed: *mut c_int,
        ostride: c_int,
        odist: c_int,
        type_: hipfftType,
        batch: c_int,
    ) -> hipfftResult;

    // --- Execution -----------------------------------------------------------
    pub fn hipfftExecC2C(
        plan: hipfftHandle,
        idata: *mut hipfftComplex,
        odata: *mut hipfftComplex,
        direction: c_int,
    ) -> hipfftResult;

    pub fn hipfftExecR2C(
        plan: hipfftHandle,
        idata: *mut hipfftReal,
        odata: *mut hipfftComplex,
    ) -> hipfftResult;

    pub fn hipfftExecC2R(
        plan: hipfftHandle,
        idata: *mut hipfftComplex,
        odata: *mut hipfftReal,
    ) -> hipfftResult;

    pub fn hipfftExecZ2Z(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleComplex,
        odata: *mut hipfftDoubleComplex,
        direction: c_int,
    ) -> hipfftResult;

    pub fn hipfftExecD2Z(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleReal,
        odata: *mut hipfftDoubleComplex,
    ) -> hipfftResult;

    pub fn hipfftExecZ2D(
        plan: hipfftHandle,
        idata: *mut hipfftDoubleComplex,
        odata: *mut hipfftDoubleReal,
    ) -> hipfftResult;

    // --- Plan management -----------------------------------------------------
    pub fn hipfftSetStream(plan: hipfftHandle, stream: hipStream_t) -> hipfftResult;

    pub fn hipfftDestroy(plan: hipfftHandle) -> hipfftResult;

    pub fn hipfftGetVersion(version: *mut c_int) -> hipfftResult;

    pub fn hipfftGetSize(plan: hipfftHandle, workSize: *mut usize) -> hipfftResult;
}
