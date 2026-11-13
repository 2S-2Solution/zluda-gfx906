use cuda_types::cublas::*;
use hip_runtime_sys::*;
use libloading::Library;
use rocblas_sys::*;
use std::ffi::OsStr;
use std::{mem, ptr, sync::OnceLock};
use zluda_common::os;
use zluda_common::{constants, from_cuda_object, ZludaObject};

fn rocblas() -> Result<&'static super::RocblasVtable, rocblas_error> {
    static LOCK: OnceLock<Result<super::RocblasVtable, rocblas_error>> = OnceLock::new();
    let unwrapped: &Result<super::RocblasVtable, rocblas_error> =
        LOCK.get_or_init(|| unsafe { super::RocblasVtable::new() });
    unwrapped.as_ref().map_err(|x| *x)
}

// This is repr(c) because it needs to be compatible with BlasHandle in zluda_blaslt.
#[repr(C)]
pub struct Handle {
    blas_lt: usize,
    blas_lt_library: Library,
    rocm_handle: rocblas_handle,
}

impl ZludaObject for Handle {
    const COOKIE: usize = constants::BLAS_HANDLE_COOKIE;
    type Error = cublasError_t;
    type CudaHandle = cublasHandle_t;
    fn drop_checked(&mut self) -> cublasStatus_t {
        unsafe {
            self.blas_lt_library
                .get::<unsafe extern "system" fn(usize) -> cublasStatus_t>(b"cublasLtDestroy\0")
        }
        // Ignore any errors from destroying embedded blas_lt handle. It may have already been destroyed.
        .map(|cublaslt_destroy| {
            if self.blas_lt != 0 {
                unsafe { cublaslt_destroy(self.blas_lt) };
            }
        })
        .ok();
        unsafe { rocblas()?.rocblas_destroy_handle(self.rocm_handle) }?;
        Ok(())
    }
}

from_cuda_object!(Handle);

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cublasStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cublasStatus_t {
    cublasStatus_t::ERROR_NOT_SUPPORTED
}

#[cfg(windows)]
pub(crate) fn get_cublaslt_file_name(file: &OsStr) -> Option<String> {
    let file = file.to_string_lossy().to_lowercase();
    if file.starts_with("cublas") {
        Some("cublasLt".to_owned() + &file["cublas".len()..])
    } else {
        None
    }
}

#[cfg(unix)]
pub(crate) fn get_cublaslt_file_name(_file: &OsStr) -> Option<&'static str> {
    Some("libcublaslt.so\0")
}

pub(crate) fn create_v2(handle: &mut cublasHandle_t) -> cublasStatus_t {
    let mut self_path = os::self_path().ok_or(cublasError_t::INTERNAL_ERROR)?;
    let file = self_path.file_name().ok_or(cublasError_t::INTERNAL_ERROR)?;
    let cublaslt_file_name = get_cublaslt_file_name(file).ok_or(cublasError_t::INTERNAL_ERROR)?;
    self_path.pop();
    self_path.push(cublaslt_file_name);
    let blas_lt_library = unsafe { libloading::Library::new(self_path) }
        .map_err(|_| cublasError_t::INTERNAL_ERROR)?;
    let mut blas_lt: usize = 0;
    let cublaslt_create = unsafe {
        blas_lt_library
            .get::<unsafe extern "system" fn(*mut usize) -> cublasStatus_t>(b"cublasLtCreate\0")
    }
    .map_err(|_| cublasError_t::INTERNAL_ERROR)?;
    // 2 Solution (gfx906): hipBLASLt is not part of our stack, so cublasLtCreate normally fails.
    // The embedded cuBLASLt handle (upstream #587) is then left out (0) instead of failing
    // cublasCreate, which would make every cuBLAS call unusable.
    if unsafe { (cublaslt_create)(&mut blas_lt) }.is_err() {
        blas_lt = 0;
    }
    let mut rocm_handle = rocblas_handle(ptr::null_mut());
    unsafe { rocblas()?.rocblas_create_handle(&mut rocm_handle) }?;
    *handle = Handle::wrap(Handle {
        blas_lt: blas_lt as usize,
        blas_lt_library,
        rocm_handle,
    });
    Ok(())
}

pub(crate) fn get_status_name(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    match status {
        cublasStatus_t::SUCCESS => c"CUBLAS_STATUS_SUCCESS".as_ptr(),
        cublasStatus_t::ERROR_NOT_INITIALIZED => c"CUBLAS_STATUS_NOT_INITIALIZED".as_ptr(),
        cublasStatus_t::ERROR_ALLOC_FAILED => c"CUBLAS_STATUS_ALLOC_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INVALID_VALUE => c"CUBLAS_STATUS_INVALID_VALUE".as_ptr(),
        cublasStatus_t::ERROR_ARCH_MISMATCH => c"CUBLAS_STATUS_ARCH_MISMATCH".as_ptr(),
        cublasStatus_t::ERROR_MAPPING_ERROR => c"CUBLAS_STATUS_MAPPING_ERROR".as_ptr(),
        cublasStatus_t::ERROR_EXECUTION_FAILED => c"CUBLAS_STATUS_EXECUTION_FAILED".as_ptr(),
        cublasStatus_t::ERROR_INTERNAL_ERROR => c"CUBLAS_STATUS_INTERNAL_ERROR".as_ptr(),
        cublasStatus_t::ERROR_NOT_SUPPORTED => c"CUBLAS_STATUS_NOT_SUPPORTED".as_ptr(),
        cublasStatus_t::ERROR_LICENSE_ERROR => c"CUBLAS_STATUS_LICENSE_ERROR".as_ptr(),
        _ => c"CUBLAS_STATUS_UNKNOWN".as_ptr(),
    }
}

pub(crate) fn get_status_string(status: cublasStatus_t) -> *const ::core::ffi::c_char {
    if cfg!(windows) && status.is_err() && status.err() == rocblas().err().map(Into::into) {
        return c"rocblas.dll could not be found. Please install HIP SDK: https://zluda.readthedocs.io/latest/hip_sdk.html".as_ptr();
    }
    get_status_name(status)
}

// VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 cublas-xerbla: no-op (Fortran-style error reporter)
// LAPACK heritage: cuBLAS reference impl. Reali app PyTorch/AutoCAD chiamano in caso di internal error;
// no-op evita panic todo!() su path retry/fallback senza alterare semantica.
pub(crate) fn xerbla(_sr_name: *const ::core::ffi::c_char, _info: ::core::ffi::c_int) -> () {
    // intentionally no-op (NVIDIA reference impl writes to stderr; ZLUDA suppresses noise)
}

// VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 cublas-cudart-version:
// Riporta CUDA 11.8 (numero convenzionale matched cudart64_110 wrap deployato).
// Era todo!() panic durante torch.cuda.is_available() probe in app PyTorch reali.
pub(crate) fn get_cudart_version() -> usize {
    11080  // CUDA 11.8 (11.080 / 1000 = 11.08 → wrapper cudart64_110)
}

pub(crate) fn set_math_mode(handle: &Handle, mode: rocblas_math_mode) -> cublasStatus_t {
    unsafe { rocblas()?.rocblas_set_math_mode(handle.rocm_handle, mode) }?;
    Ok(())
}

pub(crate) fn sgemm_strided_batched(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const f32,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const f32,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const f32,
    c: *mut f32,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe {
        rocblas()?.rocblas_sgemm_strided_batched(
            handle.rocm_handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        )
    }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-N: helper materialize non-contig matrix into packed buffer (fixes OP_T+non-contig 999 rocBLAS Tensile gfx906 bug)
unsafe fn materialize_contig(
    handle: rocblas_handle,
    src: *const ::core::ffi::c_void,
    src_lda: ::core::ffi::c_int,
    rows: ::core::ffi::c_int,
    cols: ::core::ffi::c_int,
    elem_size: usize,
) -> Option<(*mut ::core::ffi::c_void, ::core::ffi::c_int)> {
    let rows_u = rows as usize;
    let cols_u = cols as usize;
    let dpitch = rows_u * elem_size;
    let spitch = (src_lda as usize) * elem_size;
    let total = dpitch * cols_u;
    let mut buf: *mut ::core::ffi::c_void = core::ptr::null_mut();
    if hipMalloc(&mut buf, total).is_err() {
        return None;
    }
    let mut stream: hipStream_t = hipStream_t(core::ptr::null_mut());
    let _ = rocblas_get_stream(handle, &mut stream);
    if hipMemcpy2DAsync(
        buf,
        dpitch,
        src,
        spitch,
        dpitch,
        cols_u,
        hipMemcpyKind::hipMemcpyDeviceToDevice,
        stream,
    ).is_err() {
        let _ = hipFree(buf);
        return None;
    }
    Some((buf, rows))
}

pub(crate) fn sgemm_v2(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const f32,
    lda: ::core::ffi::c_int,
    b: *const f32,
    ldb: ::core::ffi::c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: ::core::ffi::c_int,
) -> cublasStatus_t {
    // VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-N: PATCH A check is_contiguous, materialize non-contig A (fix OP_T 999 Tensile gfx906)
    let a_rows = if transa == rocblas_operation::rocblas_operation_none { m } else { k };
    let a_cols = if transa == rocblas_operation::rocblas_operation_none { k } else { m };
    let need_pack_a = lda > a_rows;
    let mut owned_a: *mut ::core::ffi::c_void = core::ptr::null_mut();
    let (a_use, lda_use) = if need_pack_a {
        match unsafe {
            materialize_contig(handle.rocm_handle, a as *const _, lda, a_rows, a_cols, core::mem::size_of::<f32>())
        } {
            Some((p, ld)) => { owned_a = p; (p as *const f32, ld) }
            None => (a, lda),
        }
    } else { (a, lda) };
    let status = unsafe {
        rocblas()?.rocblas_sgemm(
            handle.rocm_handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a_use,
            lda_use,
            b,
            ldb,
            beta,
            c,
            ldc,
        )
    };
    if !owned_a.is_null() { unsafe { hipFree(owned_a); } }
    status?;
    Ok(())
}

pub(crate) fn destroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    zluda_common::drop_checked::<Handle>(handle)
}

pub(crate) unsafe fn set_pointer_mode_v2(
    handle: &Handle,
    mode: rocblas_pointer_mode,
) -> rocblas_status {
    rocblas()?.rocblas_set_pointer_mode(handle.rocm_handle, mode)
}

pub(crate) unsafe fn set_stream_v2(handle: &Handle, stream: hipStream_t) -> rocblas_status {
    rocblas()?.rocblas_set_stream(handle.rocm_handle, stream)
}

pub(crate) unsafe fn set_workspace_v2(
    handle: &Handle,
    workspace: *mut ::core::ffi::c_void,
    size: usize,
) -> rocblas_status {
    rocblas()?.rocblas_set_workspace(handle.rocm_handle, workspace, size)
}

pub(crate) unsafe fn get_math_mode(handle: &Handle, mode: &mut cublasMath_t) -> rocblas_status {
    let mut roc_mode = mem::zeroed();
    rocblas()?.rocblas_get_math_mode(handle.rocm_handle, &mut roc_mode)?;
    *mode = zluda_common::FromCuda::from_cuda(&roc_mode)?;
    Ok(())
}

pub(crate) unsafe fn gemm_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    beta: *const ::core::ffi::c_void,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    // VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-N: PATCH A mixed-precision, materialize non-contig A (fix OP_T 999 Tensile gfx906)
    let elem = match a_type {
        rocblas_datatype::rocblas_datatype_f16_r | rocblas_datatype::rocblas_datatype_bf16_r => 2usize,
        rocblas_datatype::rocblas_datatype_f32_r => 4usize,
        rocblas_datatype::rocblas_datatype_f64_r => 8usize,
        _ => 0usize,
    };
    let a_rows = if transa == rocblas_operation::rocblas_operation_none { m } else { k };
    let a_cols = if transa == rocblas_operation::rocblas_operation_none { k } else { m };
    let need_pack_a = elem > 0 && lda > a_rows;
    let mut owned_a: *mut ::core::ffi::c_void = core::ptr::null_mut();
    let (a_use, lda_use) = if need_pack_a {
        match materialize_contig(handle.rocm_handle, a, lda, a_rows, a_cols, elem) {
            Some((p, ld)) => { owned_a = p; (p as *const ::core::ffi::c_void, ld) }
            None => (a, lda),
        }
    } else { (a, lda) };
    let status = rocblas()?.rocblas_gemm_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_use,
        a_type,
        lda_use,
        b,
        b_type,
        ldb,
        beta,
        c,
        c_type,
        ldc,
        c,
        c_type,
        ldc,
        compute_type,
        algo,
        0,
        0,
    );
    if !owned_a.is_null() { let _ = hipFree(owned_a); }
    status
}

pub(crate) unsafe fn hgemm(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const cuda_types::cublas::__half,
    a: *const cuda_types::cublas::__half,
    lda: ::core::ffi::c_int,
    b: *const cuda_types::cublas::__half,
    ldb: ::core::ffi::c_int,
    beta: *const cuda_types::cublas::__half,
    c: *mut cuda_types::cublas::__half,
    ldc: ::core::ffi::c_int,
) -> rocblas_status {
    // VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-N: PATCH A check is_contiguous fp16, materialize non-contig A (fix OP_T 999 Tensile gfx906)
    let a_rows = if transa == rocblas_operation::rocblas_operation_none { m } else { k };
    let a_cols = if transa == rocblas_operation::rocblas_operation_none { k } else { m };
    let need_pack_a = lda > a_rows;
    let mut owned_a: *mut ::core::ffi::c_void = core::ptr::null_mut();
    let (a_use, lda_use) = if need_pack_a {
        match materialize_contig(handle.rocm_handle, a as *const _, lda, a_rows, a_cols, 2) {
            Some((p, ld)) => { owned_a = p; (p as *const cuda_types::cublas::__half, ld) }
            None => (a, lda),
        }
    } else { (a, lda) };
    let status = rocblas()?.rocblas_hgemm(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha.cast(),
        a_use.cast(),
        lda_use,
        b.cast(),
        ldb,
        beta.cast(),
        c.cast(),
        ldc,
    );
    if !owned_a.is_null() { let _ = hipFree(owned_a); }
    status
}

pub(crate) unsafe fn gemm_batched_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a_array: *const *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    b_array: *const *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    beta: *const ::core::ffi::c_void,
    c_array: *const *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    batch_count: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    rocblas()?.rocblas_gemm_batched_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_array.cast(),
        a_type,
        lda,
        b_array.cast(),
        b_type,
        ldb,
        beta,
        c_array.cast(),
        c_type,
        ldc,
        c_array.cast_mut().cast(),
        c_type,
        ldc,
        batch_count,
        compute_type,
        algo,
        -1,
        0,
    )
}

pub(crate) unsafe fn gemm_strided_batched_ex(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    k: ::core::ffi::c_int,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: ::core::ffi::c_int,
    stride_a: ::core::ffi::c_longlong,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: ::core::ffi::c_int,
    stride_b: ::core::ffi::c_longlong,
    beta: *const ::core::ffi::c_void,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: ::core::ffi::c_int,
    stride_c: ::core::ffi::c_longlong,
    batch_count: ::core::ffi::c_int,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    // VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-N: PATCH A batched, materialize non-contig A per-batch (fix OP_T 999 Tensile gfx906)
    let elem = match a_type {
        rocblas_datatype::rocblas_datatype_f16_r | rocblas_datatype::rocblas_datatype_bf16_r => 2usize,
        rocblas_datatype::rocblas_datatype_f32_r => 4usize,
        rocblas_datatype::rocblas_datatype_f64_r => 8usize,
        _ => 0usize,
    };
    let a_rows = if transa == rocblas_operation::rocblas_operation_none { m } else { k };
    let a_cols = if transa == rocblas_operation::rocblas_operation_none { k } else { m };
    let need_pack_a = elem > 0 && lda > a_rows && batch_count > 0;
    let mut owned_a: *mut ::core::ffi::c_void = core::ptr::null_mut();
    let (a_use, lda_use, stride_a_use) = if need_pack_a {
        let rows_u = a_rows as usize;
        let cols_u = a_cols as usize;
        let new_stride = (rows_u * cols_u) as ::core::ffi::c_longlong;
        let total = (rows_u * cols_u) * (batch_count as usize) * elem;
        let mut buf: *mut ::core::ffi::c_void = core::ptr::null_mut();
        if hipMalloc(&mut buf, total).is_ok() {
            let mut stream: hipStream_t = hipStream_t(core::ptr::null_mut());
            let _ = rocblas_get_stream(handle.rocm_handle, &mut stream);
            let dpitch = rows_u * elem;
            let spitch = (lda as usize) * elem;
            let mut ok = true;
            for bi in 0..batch_count as usize {
                let src = (a as *const u8).add(bi * (stride_a as usize) * elem) as *const ::core::ffi::c_void;
                let dst = (buf as *mut u8).add(bi * (new_stride as usize) * elem) as *mut ::core::ffi::c_void;
                if hipMemcpy2DAsync(dst, dpitch, src, spitch, dpitch, cols_u, hipMemcpyKind::hipMemcpyDeviceToDevice, stream).is_err() {
                    ok = false; break;
                }
            }
            if ok { owned_a = buf; (buf as *const ::core::ffi::c_void, a_rows, new_stride) }
            else { let _ = hipFree(buf); (a, lda, stride_a) }
        } else { (a, lda, stride_a) }
    } else { (a, lda, stride_a) };
    let status = rocblas()?.rocblas_gemm_strided_batched_ex(
        handle.rocm_handle,
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        a_use,
        a_type,
        lda_use,
        stride_a_use,
        b,
        b_type,
        ldb,
        stride_b,
        beta,
        c,
        c_type,
        ldc,
        stride_c,
        c,
        c_type,
        ldc,
        stride_c,
        batch_count,
        compute_type,
        algo,
        -1,
        0,
    );
    if !owned_a.is_null() { let _ = hipFree(owned_a); }
    status
}

// VEGA-PATCH 2026-06-29 sprint-D worker3/20: cublasGemmStridedBatchedEx_64 (i64 signature)
// Bug: PyTorch ATen 2.x bmm_out_cuda_impl chiama _64 variant con int64_t m,n,k,lda,ldb,ldc.
// ZLUDA esportava solo 32-bit → stack corruption silent → ATen fallback CPU O(N^3) o NaN.
// Fix: delega a rocblas_gemm_strided_batched_ex_64 nativo (rocBLAS 5.7 ce l'ha già).
// Quando ATen chiama via macro implemented!{} viene mappato a gemm_strided_batched_ex_64.
pub(crate) unsafe fn gemm_strided_batched_ex_64(
    handle: &Handle,
    transa: rocblas_operation,
    transb: rocblas_operation,
    m: i64,
    n: i64,
    k: i64,
    alpha: *const ::core::ffi::c_void,
    a: *const ::core::ffi::c_void,
    a_type: rocblas_datatype,
    lda: i64,
    stride_a: ::core::ffi::c_longlong,
    b: *const ::core::ffi::c_void,
    b_type: rocblas_datatype,
    ldb: i64,
    stride_b: ::core::ffi::c_longlong,
    beta: *const ::core::ffi::c_void,
    c: *mut ::core::ffi::c_void,
    c_type: rocblas_datatype,
    ldc: i64,
    stride_c: ::core::ffi::c_longlong,
    batch_count: i64,
    compute_type: rocblas_datatype,
    algo: rocblas_gemm_algo,
) -> rocblas_status {
    // VEGA-DEBUG: trace persistente per identificare chiamate ATen 2.x _64
    use std::io::Write;
    if let Ok(mut f) = crate::diag("zluda_blas_64_trace.log")
    {
        let _ = writeln!(f, "[gemm_strided_batched_ex_64] m={} n={} k={} batch={} lda={} ldb={} ldc={}",
            m, n, k, batch_count, lda, ldb, ldc);
    }

    // Pack non-contiguous A (stessa patch del path 32-bit, dimensioni i64 native)
    let elem = match a_type {
        rocblas_datatype::rocblas_datatype_f16_r | rocblas_datatype::rocblas_datatype_bf16_r => 2usize,
        rocblas_datatype::rocblas_datatype_f32_r => 4usize,
        rocblas_datatype::rocblas_datatype_f64_r => 8usize,
        _ => 0usize,
    };
    let a_rows: i64 = if transa == rocblas_operation::rocblas_operation_none { m } else { k };
    let a_cols: i64 = if transa == rocblas_operation::rocblas_operation_none { k } else { m };
    let need_pack_a = elem > 0 && lda > a_rows && batch_count > 0 && a_rows > 0 && a_cols > 0;
    let mut owned_a: *mut ::core::ffi::c_void = core::ptr::null_mut();
    let (a_use, lda_use, stride_a_use): (*const ::core::ffi::c_void, i64, ::core::ffi::c_longlong) =
        if need_pack_a {
            let rows_u = a_rows as usize;
            let cols_u = a_cols as usize;
            let new_stride: ::core::ffi::c_longlong = (rows_u * cols_u) as ::core::ffi::c_longlong;
            let total = (rows_u * cols_u) * (batch_count as usize) * elem;
            let mut buf: *mut ::core::ffi::c_void = core::ptr::null_mut();
            if hipMalloc(&mut buf, total).is_ok() {
                let mut stream: hipStream_t = hipStream_t(core::ptr::null_mut());
                let _ = rocblas_get_stream(handle.rocm_handle, &mut stream);
                let dpitch = rows_u * elem;
                let spitch = (lda as usize) * elem;
                let mut ok = true;
                for bi in 0..batch_count as usize {
                    let src = (a as *const u8).add(bi * (stride_a as usize) * elem) as *const ::core::ffi::c_void;
                    let dst = (buf as *mut u8).add(bi * (new_stride as usize) * elem) as *mut ::core::ffi::c_void;
                    if hipMemcpy2DAsync(dst, dpitch, src, spitch, dpitch, cols_u, hipMemcpyKind::hipMemcpyDeviceToDevice, stream).is_err() {
                        ok = false; break;
                    }
                }
                if ok { owned_a = buf; (buf as *const ::core::ffi::c_void, a_rows, new_stride) }
                else { let _ = hipFree(buf); (a, lda, stride_a) }
            } else { (a, lda, stride_a) }
        } else { (a, lda, stride_a) };

    let status = rocblas_gemm_strided_batched_ex_64(
        handle.rocm_handle,
        transa,
        transb,
        m, n, k,
        alpha,
        a_use, a_type, lda_use, stride_a_use,
        b,     b_type, ldb,     stride_b,
        beta,
        c,     c_type, ldc,     stride_c,
        c,     c_type, ldc,     stride_c,   // d aliasato a c (cuBLAS convention)
        batch_count,
        compute_type,
        algo,
        -1,  // solution_index default
        0,   // flags
    );
    if !owned_a.is_null() { let _ = hipFree(owned_a); }
    status
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-5: cublasSgemv forwarder rocblas_sgemv (matrix-vector single, decodifica LLM a lotto 1)
pub(crate) fn sgemv_v2(
    handle: &Handle,
    trans: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    a: *const f32,
    lda: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    beta: *const f32,
    y: *mut f32,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_sgemv(handle.rocm_handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-6: cublasDgemv forwarder rocblas_dgemv
pub(crate) fn dgemv_v2(
    handle: &Handle,
    trans: rocblas_operation,
    m: ::core::ffi::c_int,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    a: *const f64,
    lda: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    beta: *const f64,
    y: *mut f64,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_dgemv(handle.rocm_handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-7: cublasSaxpy forwarder rocblas_saxpy (Adam scalar)
pub(crate) fn saxpy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    x: *const f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_saxpy(handle.rocm_handle, n, alpha, x, incx, y, incy) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-8: cublasSscal forwarder rocblas_sscal
pub(crate) fn sscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_sscal(handle.rocm_handle, n, alpha, x, incx) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-9: cublasSdot forwarder rocblas_sdot
pub(crate) fn sdot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    y: *const f32,
    incy: ::core::ffi::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    unsafe { rocblas_sdot(handle.rocm_handle, n, x, incx, y, incy, result) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-10: cublasSnrm2 forwarder rocblas_snrm2
pub(crate) fn snrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    unsafe { rocblas_snrm2(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-10: cublasSasum forwarder rocblas_sasum
pub(crate) fn sasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut f32,
) -> cublasStatus_t {
    unsafe { rocblas_sasum(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

pub(crate) unsafe fn set_vector(
    n: ::core::ffi::c_int,
    elem_size: ::core::ffi::c_int,
    x: *const ::core::ffi::c_void,
    incx: ::core::ffi::c_int,
    device_ptr: *mut ::core::ffi::c_void,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_set_vector(n, elem_size, x, incx, device_ptr, incy)
}

pub(crate) unsafe fn get_vector(
    n: ::core::ffi::c_int,
    elem_size: ::core::ffi::c_int,
    x: *const ::core::ffi::c_void,
    incx: ::core::ffi::c_int,
    y: *mut ::core::ffi::c_void,
    incy: ::core::ffi::c_int,
) -> rocblas_status {
    rocblas()?.rocblas_get_vector(n, elem_size, x, incx, y, incy)
}

#[cfg(test)]
mod tests {
    use crate::tests::CublasApi;
    use crate::tests::CublasLtApi;
    use cuda_macros::test_cuda;

    #[test_cuda]
    fn create_destroy(api: impl CublasApi) {
        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);
        api.cublasDestroy_v2(handle);
    }

    #[test_cuda]
    fn can_pass_into_blaslt(api: impl CublasApi) {
        use cuda_types::cublas::cublasComputeType_t;
        use cuda_types::cuda::cudaDataType_t;

        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);

        let mut pref = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatmulPreferenceCreate(&mut pref);

        let mut matmul_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatmulDescCreate(
            &mut matmul_desc,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        );

        let m: u64 = 64;
        let n: u64 = 64;
        let k: u64 = 64;

        let mut a_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut a_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            k,
            m as i64,
        );

        let mut b_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut b_desc,
            cudaDataType_t::CUDA_R_32F,
            k,
            n,
            k as i64,
        );

        let mut c_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut c_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            n,
            m as i64,
        );

        let mut d_desc = unsafe { std::mem::zeroed() };
        api.blaslt().cublasLtMatrixLayoutCreate(
            &mut d_desc,
            cudaDataType_t::CUDA_R_32F,
            m,
            n,
            m as i64,
        );

        let mut heuristic_result = unsafe { std::mem::zeroed() };
        let mut return_algo_count: i32 = 0;
        api.blaslt().cublasLtMatmulAlgoGetHeuristic(
            unsafe { std::mem::transmute(handle) },
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            pref,
            1,
            &mut heuristic_result,
            &mut return_algo_count,
        );

        api.blaslt().cublasLtMatrixLayoutDestroy(d_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(c_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(b_desc);
        api.blaslt().cublasLtMatrixLayoutDestroy(a_desc);
        api.blaslt().cublasLtMatmulDescDestroy(matmul_desc);
        api.blaslt().cublasLtMatmulPreferenceDestroy(pref);
        api.cublasDestroy_v2(handle);
    }

    #[test_cuda]
    fn lt_destroy_blas_handle(api: impl CublasApi) {
        let mut handle = unsafe { std::mem::zeroed() };
        api.cublasCreate_v2(&mut handle);
        api.blaslt()
            .cublasLtDestroy(unsafe { std::mem::transmute(handle) });
        let mut math_mode = unsafe { std::mem::zeroed() };
        api.cublasGetMathMode(handle, &mut math_mode);
        api.cublasDestroy_v2(handle);
    }
}

// === VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 — Extension cublas L1 BLAS double + index ===
// Motivo: addestramento LLM FP32 Adam/AdamW chiama ddot/daxpy/dscal/dnrm2/dasum su weight tensor.
// Backward path PyTorch ROCm/CUDA fallback usa idamax per gradient clipping max-norm.
// Senza questi i wrapper torch.optim.AdamW + clip_grad_norm fallivano con CUBLAS_STATUS_NOT_SUPPORTED.

// VEGA-PATCH cublas-11: cublasDaxpy_v2 → rocblas_daxpy (Adam beta1 update)
pub(crate) fn daxpy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    x: *const f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_daxpy(handle.rocm_handle, n, alpha, x, incx, y, incy) }?;
    Ok(())
}

// VEGA-PATCH cublas-12: cublasDscal_v2 → rocblas_dscal (Adam bias correction)
pub(crate) fn dscal_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    alpha: *const f64,
    x: *mut f64,
    incx: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_dscal(handle.rocm_handle, n, alpha, x, incx) }?;
    Ok(())
}

// VEGA-PATCH cublas-13: cublasDdot_v2 → rocblas_ddot
pub(crate) fn ddot_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    y: *const f64,
    incy: ::core::ffi::c_int,
    result: *mut f64,
) -> cublasStatus_t {
    unsafe { rocblas_ddot(handle.rocm_handle, n, x, incx, y, incy, result) }?;
    Ok(())
}

// VEGA-PATCH cublas-14: cublasDnrm2_v2 → rocblas_dnrm2 (clip_grad_norm L2)
pub(crate) fn dnrm2_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> cublasStatus_t {
    unsafe { rocblas_dnrm2(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

// VEGA-PATCH cublas-15: cublasDasum_v2 → rocblas_dasum
pub(crate) fn dasum_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut f64,
) -> cublasStatus_t {
    unsafe { rocblas_dasum(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

// VEGA-PATCH cublas-16: cublasScopy_v2 → rocblas_scopy (FA1 momentum buffer copy)
pub(crate) fn scopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    y: *mut f32,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_scopy(handle.rocm_handle, n, x, incx, y, incy) }?;
    Ok(())
}

// VEGA-PATCH cublas-17: cublasDcopy_v2 → rocblas_dcopy
pub(crate) fn dcopy_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    y: *mut f64,
    incy: ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_dcopy(handle.rocm_handle, n, x, incx, y, incy) }?;
    Ok(())
}

// VEGA-PATCH cublas-18: cublasIsamax_v2 → rocblas_isamax (per gradient clipping max-norm fp32)
pub(crate) fn isamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f32,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_isamax(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

// VEGA-PATCH cublas-19: cublasIdamax_v2 → rocblas_idamax (gradient clipping max-norm fp64)
pub(crate) fn idamax_v2(
    handle: &Handle,
    n: ::core::ffi::c_int,
    x: *const f64,
    incx: ::core::ffi::c_int,
    result: *mut ::core::ffi::c_int,
) -> cublasStatus_t {
    unsafe { rocblas_idamax(handle.rocm_handle, n, x, incx, result) }?;
    Ok(())
}

// VEGA-PATCH cublas-20: cublasGetStream_v2 forwarder rocblas_get_stream
pub(crate) fn get_stream_v2(
    handle: &Handle,
    stream_id: *mut hipStream_t,
) -> cublasStatus_t {
    unsafe { rocblas_get_stream(handle.rocm_handle, stream_id) }?;
    Ok(())
}

// VEGA-PATCH cublas-21: cublasGetPointerMode_v2 forwarder
pub(crate) fn get_pointer_mode_v2(
    handle: &Handle,
    mode: *mut rocblas_pointer_mode,
) -> cublasStatus_t {
    unsafe { rocblas_get_pointer_mode(handle.rocm_handle, mode) }?;
    Ok(())
}

// VEGA-PATCH cublas-22: cublasGetVersion_v2 → riporta CUDA 11.8 (matched cudart64_110 wrap)
pub(crate) fn get_version_v2(
    _handle: &Handle,
    version: *mut ::core::ffi::c_int,
) -> cublasStatus_t {
    if version.is_null() {
        return cublasStatus_t::ERROR_INVALID_VALUE;
    }
    unsafe { *version = 11800; }
    Ok(())
}

// VEGA-PATCH cublas-23: cublasGetProperty → MAJOR/MINOR/PATCH consistency
pub(crate) fn get_property(
    type_: cuda_types::cublas::libraryPropertyType,
    value: *mut ::core::ffi::c_int,
) -> cublasStatus_t {
    if value.is_null() {
        return cublasStatus_t::ERROR_INVALID_VALUE;
    }
    let v = match type_.0 {
        0 => 11,
        1 => 8,
        2 => 0,
        _ => return cublasStatus_t::ERROR_INVALID_VALUE,
    };
    unsafe { *value = v; }
    Ok(())
}
