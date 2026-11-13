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
                cuda_macros::cublaslt_normalize_fn!( crate::r#impl::$fn_name )  ($(zluda_common::FromCuda::<_, cuda_types::cublas::cublasError_t>::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_unmapped {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cublaslt_normalize_fn!( crate::r#impl::$fn_name ) ( $( $arg_id ),* )
            }
        )*
    };
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cublasLt-1..9: wired 9 nuove fn real/stub
cuda_macros::cublaslt_function_declarations!(
    unimplemented,
    implemented
        <= [
            cublasLtCreate,
            cublasLtDestroy,
            cublasLtMatmul,
            cublasLtMatmulAlgoGetHeuristic,
            cublasLtMatmulDescCreate,
            cublasLtMatmulDescDestroy,
            cublasLtMatmulDescSetAttribute,
            cublasLtMatmulPreferenceCreate,
            cublasLtMatmulPreferenceDestroy,
            cublasLtMatmulPreferenceInit_internal,
            cublasLtMatmulPreferenceSetAttribute,
            cublasLtMatrixLayoutCreate,
            cublasLtMatrixLayoutDestroy,
            cublasLtMatrixLayoutSetAttribute,
        ],
    implemented_unmapped
        <= [
            cublasLtDisableCpuInstructionsSetMask,
            cublasLtGetCudartVersion,
            cublasLtGetStatusName,
            cublasLtGetStatusString,
            cublasLtGetVersion,
            cublasLtLoggerSetCallback,
            cublasLtLoggerSetFile,
            cublasLtLoggerOpenFile,
            cublasLtLoggerSetLevel,
            cublasLtLoggerSetMask,
            cublasLtLoggerForceDisable,
            cublasLtMatmulAlgoCapGetAttribute,
            cublasLtMatmulAlgoConfigSetAttribute,
            cublasLtMatmulAlgoConfigGetAttribute,
        ]
);

macro_rules! noop {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {};
}

#[cfg(windows)]
mod os {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use hipblaslt_sys::*;
            struct HipblasltVtable {
                _lib: libloading::os::windows::Library,
                $($fn_name: unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type,)*
            }

            impl HipblasltVtable {
                pub unsafe fn new() -> Result<Self, hipblasLtError> {
                    // 2 Solution (gfx906): no error dialog, hipBLASLt is usually absent on this stack
                    let hmodule = zluda_windows::try_load_from_self_or_hip("hipblaslt.dll").or_else(|| zluda_windows::try_load_from_self_or_hip("libhipblaslt.dll")).ok_or(hipblasLtError::INTERNAL_ERROR)?;
                    let lib = libloading::os::windows::Library::from_raw(hmodule.0 as _);
                    $(
                        let $fn_name = *lib.get::<unsafe extern "C" fn($($arg_id: $arg_type),*) -> $ret_type>(concat!(stringify!($fn_name), "\0").as_bytes()).map_err(|_| hipblasLtError::INTERNAL_ERROR)?;
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
mod os {
    macro_rules! vtable_impl {
        ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
            use hipblaslt_sys::*;

            struct HipblasltVtable {}

            impl HipblasltVtable {
                pub unsafe fn new() -> Result<Self, hipblasLtError> {
                    Ok(Self {})
                }
            }

            impl HipblasltVtable {
                $(
                    pub unsafe fn $fn_name(&self, $($arg_id: $arg_type),*) -> $ret_type {
                        (hipblaslt_sys::$fn_name)($($arg_id),*)
                    }
                )*
            }
        };
    }
    pub(crate) use vtable_impl;
}

cuda_macros::hipblaslt_function_declarations!(
    noop,
    os::vtable_impl
        <= [
            hipblasLtCreate,
            hipblasLtDestroy,
            hipblasLtMatmul,
            hipblasLtMatmulAlgoGetHeuristic,
            hipblasLtMatmulDescCreate,
            hipblasLtMatmulDescDestroy,
            hipblasLtMatmulDescSetAttribute,
            hipblasLtMatmulPreferenceCreate,
            hipblasLtMatmulPreferenceDestroy,
            hipblasLtMatmulPreferenceSetAttribute,
            hipblasLtMatrixLayoutCreate,
            hipblasLtMatrixLayoutDestroy,
            hipblasLtMatrixLayoutSetAttribute,
        ]
);

// 2 Solution (gfx906): fallback when the normal search for the delay-loaded hipblaslt.dll fails
// (see zluda_windows::try_load_from_self_or_hip).
#[cfg(windows)]
mod delay_load_hook {
    #[no_mangle]
    static __pfnDliFailureHook2: zluda_windows::PfnDliHook = delaylink_hook;

    unsafe extern "system" fn delaylink_hook(
        dli_notify: u32,
        pdli: *const zluda_windows::DelayLoadInfo,
    ) -> *mut std::ffi::c_void {
        zluda_windows::delay_load_failure_hook("hipblaslt.dll", None, dli_notify, pdli)
            .map(|hm| hm.0 as *mut std::ffi::c_void)
            .unwrap_or(std::ptr::null_mut())
    }
}
