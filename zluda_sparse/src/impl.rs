use cuda_types::cusparse::*;

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cusparseStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cusparseStatus_t {
    cusparseStatus_t::ERROR_NOT_SUPPORTED
}

// VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 worker-cublas+sparse:
// Stub graceful per app reali (PyTorch capability probe, Blender/AutoCAD splash).
// PyTorch ROCm/CUDA wheels chiamano cusparseCreate al import per "feature detect".
// Senza questo, torch.cuda.is_available() ritorna False perche' check imports.
// Path no-op: handle = Box leak, descr = Box leak. NO operazioni reali su matrici sparse.
// Per SpMM/SpMV reali servirebbe rocsparse-sys binding (non esistente nel workspace).

use std::ffi::c_char;
use std::ptr;

// Fake context (handle opaco). Box leak per pointer stabile, valid finche' cusparseDestroy free.
#[repr(C)]
struct FakeContext {
    cookie: u64,
    stream: *mut std::ffi::c_void,
    pointer_mode: u32,
}

const COOKIE_CTX: u64 = 0x5350_4152_5345_4356; // "SPARSECV" in hex
const COOKIE_MAT: u64 = 0x4D41_5444_4553_4352; // "MATDESCR"

// VEGA-PATCH cusparse-1: cusparseCreate stub graceful → handle opaco non-null
pub(crate) fn create(handle: *mut cusparseHandle_t) -> cusparseStatus_t {
    if handle.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    let ctx = Box::new(FakeContext {
        cookie: COOKIE_CTX,
        stream: ptr::null_mut(),
        pointer_mode: 0, // HOST default
    });
    unsafe { *handle = Box::into_raw(ctx) as cusparseHandle_t; }
    Ok(())
}

// VEGA-PATCH cusparse-2: cusparseDestroy stub graceful → free Box
pub(crate) fn destroy(handle: cusparseHandle_t) -> cusparseStatus_t {
    if handle.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let ctx = &*(handle as *const FakeContext);
        if ctx.cookie != COOKIE_CTX {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        drop(Box::from_raw(handle as *mut FakeContext));
    }
    Ok(())
}

// VEGA-PATCH cusparse-3: cusparseGetVersion → riporta CUDA 11.8 (11800) come cublas
pub(crate) fn get_version(
    _handle: cusparseHandle_t,
    version: *mut ::core::ffi::c_int,
) -> cusparseStatus_t {
    if version.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe { *version = 11800; }
    Ok(())
}

// VEGA-PATCH cusparse-4: cusparseGetProperty → versione MAJOR/MINOR/PATCH
pub(crate) fn get_property(
    type_: cuda_types::cusparse::libraryPropertyType,
    value: *mut ::core::ffi::c_int,
) -> cusparseStatus_t {
    if value.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    // MAJOR=0, MINOR=1, PATCH=2 nella libraryPropertyType nvidia
    let v = match type_.0 {
        0 => 11,  // MAJOR
        1 => 8,   // MINOR
        2 => 0,   // PATCH
        _ => return cusparseStatus_t::ERROR_INVALID_VALUE,
    };
    unsafe { *value = v; }
    Ok(())
}

// VEGA-PATCH cusparse-5: cusparseSetStream no-op (memorizza in fake ctx)
pub(crate) fn set_stream(
    handle: cusparseHandle_t,
    stream_id: cudaStream_t,
) -> cusparseStatus_t {
    if handle.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let ctx = &mut *(handle as *mut FakeContext);
        if ctx.cookie != COOKIE_CTX {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        // cudaStream_t = CUstream(*mut CUstream_st), accedo a .0 field
        ctx.stream = stream_id.0 as *mut std::ffi::c_void;
    }
    Ok(())
}

// VEGA-PATCH cusparse-6: cusparseGetStream legge da fake ctx
pub(crate) fn get_stream(
    handle: cusparseHandle_t,
    stream_id: *mut cudaStream_t,
) -> cusparseStatus_t {
    if handle.is_null() || stream_id.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let ctx = &*(handle as *const FakeContext);
        if ctx.cookie != COOKIE_CTX {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        // cudaStream_t e' alias di CUstream (tuple struct); costruisco via field .0
        (*stream_id).0 = ctx.stream as *mut _;
    }
    Ok(())
}

// VEGA-PATCH cusparse-7: cusparseSetPointerMode no-op (memorizza)
pub(crate) fn set_pointer_mode(
    handle: cusparseHandle_t,
    mode: cusparsePointerMode_t,
) -> cusparseStatus_t {
    if handle.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let ctx = &mut *(handle as *mut FakeContext);
        if ctx.cookie != COOKIE_CTX {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        ctx.pointer_mode = mode.0;
    }
    Ok(())
}

// VEGA-PATCH cusparse-8: cusparseGetPointerMode legge da fake ctx
pub(crate) fn get_pointer_mode(
    handle: cusparseHandle_t,
    mode: *mut cusparsePointerMode_t,
) -> cusparseStatus_t {
    if handle.is_null() || mode.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let ctx = &*(handle as *const FakeContext);
        if ctx.cookie != COOKIE_CTX {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        *mode = cusparsePointerMode_t(ctx.pointer_mode);
    }
    Ok(())
}

// MatDescr: stub graceful (storage minimo)
#[repr(C)]
struct FakeMatDescr {
    cookie: u64,
    mat_type: u32,
    fill_mode: u32,
    diag_type: u32,
    index_base: u32,
}

// VEGA-PATCH cusparse-9: cusparseCreateMatDescr → Box leak descrittore opaco
pub(crate) fn create_mat_descr(
    descr_a: *mut cusparseMatDescr_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    let d = Box::new(FakeMatDescr {
        cookie: COOKIE_MAT,
        mat_type: 0,  // GENERAL
        fill_mode: 0, // LOWER
        diag_type: 0, // NON_UNIT
        index_base: 0, // ZERO
    });
    unsafe { *descr_a = Box::into_raw(d) as cusparseMatDescr_t; }
    Ok(())
}

// VEGA-PATCH cusparse-10: cusparseDestroyMatDescr free Box
pub(crate) fn destroy_mat_descr(
    descr_a: cusparseMatDescr_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let d = &*(descr_a as *const FakeMatDescr);
        if d.cookie != COOKIE_MAT {
            return cusparseStatus_t::ERROR_INVALID_VALUE;
        }
        drop(Box::from_raw(descr_a as *mut FakeMatDescr));
    }
    Ok(())
}

// VEGA-PATCH cusparse-11: cusparseSetMatType no-op (memorizza)
pub(crate) fn set_mat_type(
    descr_a: cusparseMatDescr_t,
    type_: cusparseMatrixType_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let d = &mut *(descr_a as *mut FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseStatus_t::ERROR_INVALID_VALUE; }
        d.mat_type = type_.0;
    }
    Ok(())
}

// VEGA-PATCH cusparse-12: cusparseSetMatFillMode no-op
pub(crate) fn set_mat_fill_mode(
    descr_a: cusparseMatDescr_t,
    fill_mode: cusparseFillMode_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let d = &mut *(descr_a as *mut FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseStatus_t::ERROR_INVALID_VALUE; }
        d.fill_mode = fill_mode.0;
    }
    Ok(())
}

// VEGA-PATCH cusparse-13: cusparseSetMatDiagType no-op
pub(crate) fn set_mat_diag_type(
    descr_a: cusparseMatDescr_t,
    diag_type: cusparseDiagType_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let d = &mut *(descr_a as *mut FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseStatus_t::ERROR_INVALID_VALUE; }
        d.diag_type = diag_type.0;
    }
    Ok(())
}

// VEGA-PATCH cusparse-14: cusparseSetMatIndexBase no-op
pub(crate) fn set_mat_index_base(
    descr_a: cusparseMatDescr_t,
    base: cusparseIndexBase_t,
) -> cusparseStatus_t {
    if descr_a.is_null() {
        return cusparseStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe {
        let d = &mut *(descr_a as *mut FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseStatus_t::ERROR_INVALID_VALUE; }
        d.index_base = base.0;
    }
    Ok(())
}

// VEGA-PATCH cusparse-15: get_error_name → C-string statica
pub(crate) fn get_error_name(
    status: cuda_types::cusparse::cusparseStatus_t,
) -> *const c_char {
    let s: &'static [u8] = match status {
        Ok(()) => b"CUSPARSE_STATUS_SUCCESS\0",
        Err(e) => match e.0.get() {
            1 => b"CUSPARSE_STATUS_NOT_INITIALIZED\0",
            2 => b"CUSPARSE_STATUS_ALLOC_FAILED\0",
            3 => b"CUSPARSE_STATUS_INVALID_VALUE\0",
            4 => b"CUSPARSE_STATUS_ARCH_MISMATCH\0",
            5 => b"CUSPARSE_STATUS_MAPPING_ERROR\0",
            6 => b"CUSPARSE_STATUS_EXECUTION_FAILED\0",
            7 => b"CUSPARSE_STATUS_INTERNAL_ERROR\0",
            8 => b"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED\0",
            9 => b"CUSPARSE_STATUS_ZERO_PIVOT\0",
            10 => b"CUSPARSE_STATUS_NOT_SUPPORTED\0",
            11 => b"CUSPARSE_STATUS_INSUFFICIENT_RESOURCES\0",
            _ => b"CUSPARSE_STATUS_UNKNOWN\0",
        },
    };
    s.as_ptr() as *const c_char
}

// VEGA-PATCH cusparse-16: get_error_string (alias semantico get_error_name)
pub(crate) fn get_error_string(
    status: cuda_types::cusparse::cusparseStatus_t,
) -> *const c_char {
    let s: &'static [u8] = match status {
        Ok(()) => b"The operation completed successfully.\0",
        Err(e) => match e.0.get() {
            1 => b"The cuSPARSE library was not initialized.\0",
            2 => b"Resource allocation failed.\0",
            3 => b"An invalid value or parameter was used.\0",
            4 => b"Architecture mismatch / function not supported on device.\0",
            5 => b"GPU memory mapping error.\0",
            6 => b"Execution failed on the GPU.\0",
            7 => b"Internal cuSPARSE error.\0",
            8 => b"Matrix type not supported.\0",
            9 => b"Zero pivot encountered in factorization.\0",
            10 => b"Operation not supported (ZLUDA cuSPARSE stub).\0",
            11 => b"Insufficient resources.\0",
            _ => b"Unknown cuSPARSE status.\0",
        },
    };
    s.as_ptr() as *const c_char
}

// VEGA-PATCH cusparse-17: get_mat_type legge da fake descr (no piu' todo!())
pub(crate) fn get_mat_type(
    descr_a: cuda_types::cusparse::cusparseMatDescr_t,
) -> cuda_types::cusparse::cusparseMatrixType_t {
    if descr_a.is_null() {
        return cusparseMatrixType_t(0); // GENERAL fallback
    }
    unsafe {
        let d = &*(descr_a as *const FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseMatrixType_t(0); }
        cusparseMatrixType_t(d.mat_type)
    }
}

// VEGA-PATCH cusparse-18: get_mat_fill_mode (no piu' todo!())
pub(crate) fn get_mat_fill_mode(
    descr_a: cuda_types::cusparse::cusparseMatDescr_t,
) -> cuda_types::cusparse::cusparseFillMode_t {
    if descr_a.is_null() {
        return cusparseFillMode_t(0);
    }
    unsafe {
        let d = &*(descr_a as *const FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseFillMode_t(0); }
        cusparseFillMode_t(d.fill_mode)
    }
}

// VEGA-PATCH cusparse-19: get_mat_diag_type (no piu' todo!())
pub(crate) fn get_mat_diag_type(
    descr_a: cuda_types::cusparse::cusparseMatDescr_t,
) -> cuda_types::cusparse::cusparseDiagType_t {
    if descr_a.is_null() {
        return cusparseDiagType_t(0);
    }
    unsafe {
        let d = &*(descr_a as *const FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseDiagType_t(0); }
        cusparseDiagType_t(d.diag_type)
    }
}

// VEGA-PATCH cusparse-20: get_mat_index_base (no piu' todo!())
pub(crate) fn get_mat_index_base(
    descr_a: cuda_types::cusparse::cusparseMatDescr_t,
) -> cuda_types::cusparse::cusparseIndexBase_t {
    if descr_a.is_null() {
        return cusparseIndexBase_t(0);
    }
    unsafe {
        let d = &*(descr_a as *const FakeMatDescr);
        if d.cookie != COOKIE_MAT { return cusparseIndexBase_t(0); }
        cusparseIndexBase_t(d.index_base)
    }
}
