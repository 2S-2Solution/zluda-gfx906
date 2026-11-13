macro_rules! export_dnn8_unmangled {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[no_mangle]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                zluda_dnn::dnn8::$fn_name( $( $arg_id ),* )
            }
        )*
    };
}

cuda_macros::cudnn8_function_declarations! {
    export_dnn8_unmangled
}

#[cfg(windows)]
mod windows {
    use zluda_windows;

    #[no_mangle]
    static __pfnDliNotifyHook2: zluda_windows::PfnDliHook =
        zluda_windows::open_already_loaded_amdhip;

    // 2 Solution (gfx906): zluda_dnn calls MIOpen directly (miopen-sys with raw-dylib), so
    // MIOpen.dll stays delay-loaded as before the merge, with this fallback when the normal
    // search fails (see zluda_windows::try_load_from_self_or_hip).
    #[no_mangle]
    static __pfnDliFailureHook2: zluda_windows::PfnDliHook = delaylink_hook;

    unsafe extern "system" fn delaylink_hook(
        dli_notify: u32,
        pdli: *const zluda_windows::DelayLoadInfo,
    ) -> *mut std::ffi::c_void {
        zluda_windows::delay_load_failure_hook("MIOpen.dll", None, dli_notify, pdli)
            .map(|hm| hm.0 as *mut std::ffi::c_void)
            .unwrap_or(std::ptr::null_mut())
    }
}
