//! Minimal FFI bindings su hiprand ROCm 5.7 (Vega 20 gfx906).
//! Solo le 6 API richieste per supportare cuRAND minimo PyTorch.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::c_void;

pub type hipStream_t = *mut c_void;

#[repr(C)]
pub struct hiprandGenerator_st {
    _opaque: [u8; 0],
}
pub type hiprandGenerator_t = *mut hiprandGenerator_st;

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum hiprandStatus_t {
    HIPRAND_STATUS_SUCCESS = 0,
    HIPRAND_STATUS_VERSION_MISMATCH = 100,
    HIPRAND_STATUS_NOT_INITIALIZED = 101,
    HIPRAND_STATUS_ALLOCATION_FAILED = 102,
    HIPRAND_STATUS_TYPE_ERROR = 103,
    HIPRAND_STATUS_OUT_OF_RANGE = 104,
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    HIPRAND_STATUS_LAUNCH_FAILURE = 201,
    HIPRAND_STATUS_PREEXISTING_FAILURE = 202,
    HIPRAND_STATUS_INITIALIZATION_FAILED = 203,
    HIPRAND_STATUS_ARCH_MISMATCH = 204,
    HIPRAND_STATUS_INTERNAL_ERROR = 999,
    HIPRAND_STATUS_NOT_IMPLEMENTED = 1000,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum hiprandRngType_t {
    HIPRAND_RNG_PSEUDO_DEFAULT = 400,
    HIPRAND_RNG_PSEUDO_XORWOW = 401,
    HIPRAND_RNG_PSEUDO_MRG32K3A = 402,
    HIPRAND_RNG_PSEUDO_MTGP32 = 403,
    HIPRAND_RNG_PSEUDO_MT19937 = 404,
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405,
}

extern "C" {
    pub fn hiprandCreateGenerator(
        generator: *mut hiprandGenerator_t,
        rng_type: hiprandRngType_t,
    ) -> hiprandStatus_t;

    pub fn hiprandDestroyGenerator(generator: hiprandGenerator_t) -> hiprandStatus_t;

    pub fn hiprandSetPseudoRandomGeneratorSeed(
        generator: hiprandGenerator_t,
        seed: u64,
    ) -> hiprandStatus_t;

    pub fn hiprandSetGeneratorOffset(
        generator: hiprandGenerator_t,
        offset: u64,
    ) -> hiprandStatus_t;

    pub fn hiprandSetStream(
        generator: hiprandGenerator_t,
        stream: hipStream_t,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerate(
        generator: hiprandGenerator_t,
        output_data: *mut u32,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniform(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateUniformDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateNormalDouble(
        generator: hiprandGenerator_t,
        output_data: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> hiprandStatus_t;

    pub fn hiprandGenerateLogNormal(
        generator: hiprandGenerator_t,
        output_data: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> hiprandStatus_t;

    pub fn hiprandGetVersion(version: *mut i32) -> hiprandStatus_t;
}
