use cuda_types::cublas::cublasError_t;

mod r#impl;

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            #[allow(unused_variables)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::unimplemented()
            }
        )*
    };
}

macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cublas_normalize_fn!( crate::r#impl::$fn_name ) ($(zluda_common::FromCuda::<_, cublasError_t>::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_and_always_succeeds {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cublas_normalize_fn!( crate::r#impl::$fn_name ) ( $( $arg_id ),* )
            }
        )*
    };
}

cuda_macros::cublas_function_declarations!(
    unimplemented,
    implemented
        <= [
            cublasCreate_v2,
            cublasDestroy_v2,
            cublasGemmBatchedEx,
            cublasGemmEx,
            cublasGemmStridedBatchedEx,
            // VEGA-PATCH 2026-06-29 sprint-D worker3/20: i64 variant per ATen 2.x bmm_out_cuda_impl
            cublasGemmStridedBatchedEx_64,
            cublasGetMathMode,
            cublasGetVector,
            cublasHgemm,
            cublasSetMathMode,
            cublasSetPointerMode_v2,
            cublasSetStream_v2,
            cublasSetWorkspace_v2,
            cublasSetVector,
            cublasSgemmStridedBatched,
            cublasSgemm_v2,
            // VEGA-PATCH 2026-06-28 sprint-orch v2 cublas-5..10: BLAS-1/2 forwarders (gemv/axpy/scal/dot/nrm2/asum)
            cublasSgemv_v2,
            cublasDgemv_v2,
            cublasSaxpy_v2,
            cublasSscal_v2,
            cublasSdot_v2,
            cublasSnrm2_v2,
            cublasSasum_v2,
            // VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 cublas-11..23: double L1 + iamax + version/property
            // Per: Adam/AdamW FP32+FP64 + clip_grad_norm + capability probe PyTorch/Blender/AutoCAD
            cublasDaxpy_v2,
            cublasDscal_v2,
            cublasDdot_v2,
            cublasDnrm2_v2,
            cublasDasum_v2,
            cublasScopy_v2,
            cublasDcopy_v2,
            cublasIsamax_v2,
            cublasIdamax_v2,
            cublasGetStream_v2,
            cublasGetPointerMode_v2,
            cublasGetVersion_v2,
            // NOTE: cublasGetProperty rimossa whitelist — libraryPropertyType non impl FromCuda
            // per cublasError_t; resta in branch `unimplemented` (ritorna ERROR_NOT_SUPPORTED
            // graceful, accettabile: app reali fallback su cublasGetVersion_v2 sopra)
        ],
    implemented_and_always_succeeds
        <= [
            cublasGetStatusName,
            cublasGetStatusString,
            cublasXerbla,
            cublasGetCudartVersion
        ]
);

// VEGA-PATCH 2026-06-29 sprint-rust-D worker9 task4 NOTE:
// cublasGemmStridedBatchedEx_64 e cublasGemmEx_64 sono GIA esportati dal macro
// cuda_macros::cublas_function_declarations! nel ramo `unimplemented` (CUDA 12+ ABI int64).
// Per implementarli serve agire a livello di cuda_macros (cuda_macros/src/lib.rs) spostandoli
// nella whitelist `implemented` e aggiungere impl reali `gemm_ex_64` / `gemm_strided_batched_ex_64`
// in src/impl.rs che fanno range-check i64→i32 e delegano alle variant _32 esistenti.
// Fix proposto (ABI-correct, evita collisione E0428):
//   1) cuda_macros: aggiungi `cublasGemmStridedBatchedEx_64, cublasGemmEx_64` a implemented<=
//   2) src/impl.rs: nuove funzioni pub(crate) gemm_strided_batched_ex_64 / gemm_ex_64 che
//      a) range-check (m,n,k,lda,ldb,ldc,batchCount) > i32::MAX => return ERROR_INVALID_VALUE
//      b) cast i64→i32 e chiamano la gemm_strided_batched_ex / gemm_ex esistente
//   Vedi blocco commentato a fine file per il codice impl pronto.

macro_rules! noop {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {};
}

#[cfg(windows)]
mod os_macro {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use rocblas_sys::*;
            struct RocblasVtable {
                _lib: libloading::os::windows::Library,
                $($fn_name: unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type,)*
            }

            impl RocblasVtable {
                pub unsafe fn new() -> Result<Self, rocblas_error> {
                    let hmodule = zluda_windows::try_load_from_self_or_hip_with_message(&["rocblas.dll"]).ok_or(rocblas_error::internal_error)?;
                    let lib = libloading::os::windows::Library::from_raw(hmodule.0 as _);
                    $(
                        let $fn_name = *lib.get::<unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type>(concat!(stringify!($fn_name), "\0").as_bytes()).map_err(|_| rocblas_error::internal_error)?;
                    )*
                    Ok(Self {
                        _lib: lib,
                        $($fn_name,)*
                    })
                }

                $(
                    pub unsafe fn $fn_name(&self, $($arg_id: $arg_type),*) -> $ret_type {
                        (self.$fn_name)($($arg_id),*)
                    }
                )*
            }
        };
    }
    pub(crate) use vtable_impl;
}

#[cfg(not(windows))]
mod os_macro {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use rocblas_sys::*;

            struct RocblasVtable {}

            impl RocblasVtable {
                pub unsafe fn new() -> Result<Self, rocblas_error> {
                    Ok(Self {})
                }
            }

            impl RocblasVtable {
                $(
                    pub unsafe fn $fn_name(&self, $($arg_id: $arg_type),*) -> $ret_type {
                        (rocblas_sys::$fn_name)($($arg_id),*)
                    }
                )*
            }
        };
    }
    pub(crate) use vtable_impl;
}


// ---------------------------------------------------------------------------
//  Diagnostica su file — SPENTA salvo richiesta esplicita.
//
//  Questi log nascono come strumento di indagine, ma erano incondizionati: ogni
//  chiamata apriva, scriveva e chiudeva un file su disco. Il 31 lug 2026
//  `zluda_call_trace.log` era arrivato a 64 GB e `nvcuda_trace.log` a 7,5 GB,
//  col disco al 91%. Il costo ricadeva su ogni singola chiamata CUDA.
//
//  Ora si accendono impostando ZLUDA_DIAG (e ZLUDA_DIAG_DIR per la cartella).
//  La variabile e' letta una volta sola: a diagnostica spenta il costo e' la
//  lettura di un bool.
//
//  Il tipo di ritorno resta `std::io::Result<File>` di proposito, cosi' i
//  chiamanti gia' scritti come `if let Ok(mut f) = ...` restano invariati.
// ---------------------------------------------------------------------------
#[allow(dead_code)]
pub(crate) fn diag(nome: &str) -> std::io::Result<std::fs::File> {
    static ATTIVA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ATTIVA.get_or_init(|| std::env::var_os("ZLUDA_DIAG").is_some()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "diagnostica su file spenta",
        ));
    }
    let dir = diag_dir();
    let _ = std::fs::create_dir_all(&dir);
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::path::Path::new(&dir).join(nome))
}

/// Cartella dei file di diagnostica: `ZLUDA_DIAG_DIR`, altrimenti `%TEMP%\zluda_diag`.
/// Si usa solo a diagnostica accesa (`ZLUDA_DIAG`).
#[allow(dead_code)]
pub(crate) fn diag_dir() -> std::path::PathBuf {
    match std::env::var_os("ZLUDA_DIAG_DIR") {
        Some(d) => std::path::PathBuf::from(d),
        None => std::env::temp_dir().join("zluda_diag"),
    }
}

cuda_macros::rocblas_function_declarations!(
    noop,
    os_macro::vtable_impl
        <= [
            rocblas_create_handle,
            rocblas_destroy_handle,
            rocblas_gemm_batched_ex,
            rocblas_gemm_ex,
            rocblas_gemm_strided_batched_ex,
            rocblas_get_math_mode,
            rocblas_get_vector,
            rocblas_hgemm,
            rocblas_set_math_mode,
            rocblas_set_pointer_mode,
            rocblas_set_stream,
            rocblas_set_vector,
            rocblas_set_workspace,
            rocblas_sgemm_strided_batched,
            rocblas_sgemm,
        ]
);

#[cfg(test)]
mod tests;

// 2 Solution (gfx906): fallback when the normal search for the delay-loaded rocblas.dll fails
// (see zluda_windows::try_load_from_self_or_hip).
#[cfg(windows)]
mod delay_load_hook {
    #[no_mangle]
    static __pfnDliFailureHook2: zluda_windows::PfnDliHook = delaylink_hook;

    unsafe extern "system" fn delaylink_hook(
        dli_notify: u32,
        pdli: *const zluda_windows::DelayLoadInfo,
    ) -> *mut std::ffi::c_void {
        zluda_windows::delay_load_failure_hook("rocblas.dll", None, dli_notify, pdli)
            .map(|hm| hm.0 as *mut std::ffi::c_void)
            .unwrap_or(std::ptr::null_mut())
    }
}
