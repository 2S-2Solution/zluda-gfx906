// zluda_nvjpeg — stub graceful per nvjpeg64_11.dll
// VEGA-PATCH 2026-06-30 T16:00 (Vega ⭐, bottega italiana sovrana)
//
// Obiettivo: superare LoadLibraryW/GetProcAddress feature-detect di
// NVIDIA DALI, torchvision GPU decoder, OpenCV CUDA backend.
// Stub graceful = pass per query/Create, NOT_SUPPORTED per Decode/Encode
// così caller fa fallback libjpeg-turbo CPU.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use core::ffi::{c_int, c_void};

// =========================================================================
// Types ABI-compatibili con nvjpeg.h (NVJPEG 11.x)
// =========================================================================

/// nvjpegStatus_t: 0=SUCCESS, 1=NOT_INITIALIZED, 2=INVALID_PARAMETER,
/// 3=BAD_JPEG, 4=JPEG_NOT_SUPPORTED, 5=ALLOCATOR_FAILURE,
/// 6=EXECUTION_FAILED, 7=ARCH_MISMATCH, 8=INTERNAL_ERROR,
/// 9=IMPLEMENTATION_NOT_SUPPORTED, 10=INCOMPLETE_BITSTREAM.
pub type nvjpegStatus_t = c_int;

const NVJPEG_SUCCESS: nvjpegStatus_t = 0;
const NVJPEG_INVALID_PARAMETER: nvjpegStatus_t = 2;
const NVJPEG_JPEG_NOT_SUPPORTED: nvjpegStatus_t = 4;
const NVJPEG_IMPL_NOT_SUPPORTED: nvjpegStatus_t = 9;

pub type nvjpegHandle_t = *mut c_void;
pub type nvjpegJpegState_t = *mut c_void;
pub type nvjpegEncoderState_t = *mut c_void;
pub type nvjpegEncoderParams_t = *mut c_void;
pub type nvjpegBackend_t = c_int;
pub type cudaStream_t = *mut c_void;
pub type libraryPropertyType = c_int;

// Versione spacciata: nvJPEG 11.6.0 (allineata CUDA 11.8 toolkit).
const FAKE_MAJOR: c_int = 11;
const FAKE_MINOR: c_int = 6;
const FAKE_PATCH: c_int = 0;

static DUMMY_HANDLE: c_int = 0xC09E_6000u32 as i32;
static DUMMY_STATE: c_int = 0xC09E_6001u32 as i32;
static DUMMY_ENC_STATE: c_int = 0xC09E_6002u32 as i32;
static DUMMY_ENC_PARAMS: c_int = 0xC09E_6003u32 as i32;

// =========================================================================
// Query
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn nvjpegGetProperty(
    property: libraryPropertyType,
    value: *mut c_int,
) -> nvjpegStatus_t {
    if value.is_null() { return NVJPEG_INVALID_PARAMETER; }
    *value = match property {
        0 => FAKE_MAJOR,
        1 => FAKE_MINOR,
        2 => FAKE_PATCH,
        _ => return NVJPEG_INVALID_PARAMETER,
    };
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegGetCudartProperty(
    property: libraryPropertyType,
    value: *mut c_int,
) -> nvjpegStatus_t {
    nvjpegGetProperty(property, value)
}

// =========================================================================
// Handle lifecycle
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn nvjpegCreate(
    _backend: nvjpegBackend_t,
    _dev_allocator: *mut c_void,
    handle: *mut nvjpegHandle_t,
) -> nvjpegStatus_t {
    if handle.is_null() { return NVJPEG_INVALID_PARAMETER; }
    *handle = &DUMMY_HANDLE as *const _ as *mut c_void;
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegCreateSimple(handle: *mut nvjpegHandle_t) -> nvjpegStatus_t {
    if handle.is_null() { return NVJPEG_INVALID_PARAMETER; }
    *handle = &DUMMY_HANDLE as *const _ as *mut c_void;
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegCreateEx(
    _backend: nvjpegBackend_t,
    _dev_allocator: *mut c_void,
    _pinned_allocator: *mut c_void,
    _flags: u32,
    handle: *mut nvjpegHandle_t,
) -> nvjpegStatus_t {
    if handle.is_null() { return NVJPEG_INVALID_PARAMETER; }
    *handle = &DUMMY_HANDLE as *const _ as *mut c_void;
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegCreateExV2(
    _backend: nvjpegBackend_t,
    _dev_allocator: *mut c_void,
    _pinned_allocator: *mut c_void,
    _flags: u32,
    handle: *mut nvjpegHandle_t,
) -> nvjpegStatus_t {
    nvjpegCreateEx(_backend, _dev_allocator, _pinned_allocator, _flags, handle)
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegDestroy(_handle: nvjpegHandle_t) -> nvjpegStatus_t {
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegSetDeviceMemoryPadding(
    _padding: usize,
    _handle: nvjpegHandle_t,
) -> nvjpegStatus_t { NVJPEG_SUCCESS }

#[no_mangle]
pub unsafe extern "C" fn nvjpegSetPinnedMemoryPadding(
    _padding: usize,
    _handle: nvjpegHandle_t,
) -> nvjpegStatus_t { NVJPEG_SUCCESS }

#[no_mangle]
pub unsafe extern "C" fn nvjpegGetDeviceMemoryPadding(
    padding: *mut usize,
    _handle: nvjpegHandle_t,
) -> nvjpegStatus_t {
    if !padding.is_null() { *padding = 0; }
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegGetPinnedMemoryPadding(
    padding: *mut usize,
    _handle: nvjpegHandle_t,
) -> nvjpegStatus_t {
    if !padding.is_null() { *padding = 0; }
    NVJPEG_SUCCESS
}

// =========================================================================
// JpegState (decoder) lifecycle
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn nvjpegJpegStateCreate(
    _handle: nvjpegHandle_t,
    state: *mut nvjpegJpegState_t,
) -> nvjpegStatus_t {
    if state.is_null() { return NVJPEG_INVALID_PARAMETER; }
    *state = &DUMMY_STATE as *const _ as *mut c_void;
    NVJPEG_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn nvjpegJpegStateDestroy(_state: nvjpegJpegState_t) -> nvjpegStatus_t {
    NVJPEG_SUCCESS
}

// =========================================================================
// Operazioni reali — NOT_SUPPORTED (fallback CPU)
// =========================================================================

macro_rules! nvjpeg_stub_not_supported {
    ($($name:ident),* $(,)?) => {
        $(
            #[no_mangle]
            pub unsafe extern "C" fn $name() -> nvjpegStatus_t {
                NVJPEG_IMPL_NOT_SUPPORTED
            }
        )*
    };
}

nvjpeg_stub_not_supported!(
    // Image info / decode
    nvjpegGetImageInfo,
    nvjpegDecode,
    nvjpegDecodeBatched,
    nvjpegDecodeBatchedInitialize,
    nvjpegDecodeBatchedSupported,
    nvjpegDecodeBatchedSupportedEx,
    // Phased decode
    nvjpegDecodePhaseOne, nvjpegDecodePhaseTwo, nvjpegDecodePhaseThree,
    // Streaming decode (nvJPEG 11.5+)
    nvjpegBufferPinnedCreate, nvjpegBufferPinnedDestroy,
    nvjpegBufferDeviceCreate, nvjpegBufferDeviceDestroy,
    nvjpegStateAttachPinnedBuffer, nvjpegStateAttachDeviceBuffer,
    nvjpegJpegStreamCreate, nvjpegJpegStreamDestroy,
    nvjpegJpegStreamParse, nvjpegJpegStreamParseHeader,
    nvjpegJpegStreamGetFrameDimensions, nvjpegJpegStreamGetComponentsNum,
    nvjpegJpegStreamGetComponentDimensions, nvjpegJpegStreamGetChromaSubsampling,
    nvjpegDecodeParamsCreate, nvjpegDecodeParamsDestroy,
    nvjpegDecodeParamsSetOutputFormat, nvjpegDecodeParamsSetROI,
    nvjpegDecodeParamsSetAllowCMYK, nvjpegDecodeParamsSetScaleFactor,
    nvjpegDecodeJpeg, nvjpegDecodeJpegHost, nvjpegDecodeJpegTransferToDevice,
    nvjpegDecodeJpegDevice,
    nvjpegDecoderCreate, nvjpegDecoderDestroy,
    nvjpegDecoderJpegSupported,
    // Encoder
    nvjpegEncoderStateCreate, nvjpegEncoderStateDestroy,
    nvjpegEncoderParamsCreate, nvjpegEncoderParamsDestroy,
    nvjpegEncoderParamsSetQuality, nvjpegEncoderParamsSetEncoding,
    nvjpegEncoderParamsSetOptimizedHuffman, nvjpegEncoderParamsSetSamplingFactors,
    nvjpegEncodeYUV, nvjpegEncodeImage, nvjpegEncodeRetrieveBitstream,
    nvjpegEncodeGetBufferSize,
);
