// VEGA-PATCH 2026-06-30 TASK 5 worker 2/3 — zluda_cufft cdylib
// BUG 6: cuFFT mancante -> torch.fft.* / Whisper / SD-XL VAE fail.
// API minime richieste: Plan{1,2,3}d / PlanMany / Exec{C2C,R2C,C2R} +
// Z2Z/D2Z/Z2D (double per safety) + SetStream / Destroy / GetVersion.
// Backend: hipfft ROCm 5.7 (regola 4-bis Vega). MAI ROCm 6.x/7.x come dep.
//
// Pattern di delega:
//   - export `pub unsafe extern "system" fn cufft*` (ABI Windows stdcall x64 = C)
//     dichiarati dal macro `cuda_macros::cufft_function_declarations!`
//   - le whitelist `implemented` <= [..] forward al modulo `r#impl`
//   - le NON whitelistate restano `unimplemented` -> CUFFT_ERROR_NOT_SUPPORTED
//
// Restituiamo `cufftResult` (= Result<(), cufftError_t>) come `repr(transparent)`
// u32 grazie a NonZeroU32 nel cuda_types::cufft.

mod r#impl;

use cuda_types::cufft::cufftError_t;

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
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
                crate::r#impl::$fn_name ( $($arg_id),* )
            }
        )*
    };
}

cuda_macros::cufft_function_declarations!(
    unimplemented,
    implemented
        <= [
            // Plan creation (legacy: allocate + make in un colpo)
            cufftPlan1d,
            cufftPlan2d,
            cufftPlan3d,
            cufftPlanMany,
            // Two-step (Create handle + Make plan)
            cufftCreate,
            cufftMakePlan1d,
            cufftMakePlan2d,
            cufftMakePlan3d,
            cufftMakePlanMany,
            // Execution single precision (workload target: torch fp32)
            cufftExecC2C,
            cufftExecR2C,
            cufftExecC2R,
            // Execution double precision
            cufftExecZ2Z,
            cufftExecD2Z,
            cufftExecZ2D,
            // Plan mgmt
            cufftSetStream,
            cufftDestroy,
            cufftGetVersion,
            cufftGetSize,
            // No-op safe defaults (workSize=0): evita fallback NOT_SUPPORTED
            cufftSetAutoAllocation,
        ]
);

// ----- Delay-load hook hipfft.dll (assente -> handler che ritorna NULL ->
// le chiamate cufft* falliranno con CUFFT_ERROR_NOT_SUPPORTED via panic-safe path).
#[cfg(windows)]
mod windows {
    use zluda_windows;
    #[no_mangle]
    static __pfnDliFailureHook2: zluda_windows::PfnDliHook = delaylink_hook;

    unsafe extern "system" fn delaylink_hook(
        dli_notify: u32,
        pdli: *const zluda_windows::DelayLoadInfo,
    ) -> *mut std::ffi::c_void {
        zluda_windows::delay_load_failure_hook("hipfft.dll", dli_notify, pdli)
            .map(|hm| hm.0 as *mut std::ffi::c_void)
            .unwrap_or(std::ptr::null_mut())
    }
}

// Silenzia warning su cufftError_t unused se cfg(test) toglie alcuni branch.
#[allow(dead_code)]
const _UNUSED_ERR: cufftError_t = cufftError_t::r#NOT_SUPPORTED;
