mod r#impl;

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
                cuda_macros::cusparse_normalize_fn!( crate::r#impl::$fn_name ) ( $( $arg_id ),* )
            }
        )*
    };
}

cuda_macros::cusparse_function_declarations!(
    unimplemented,
    implemented
        <= [
            cusparseGetErrorName,
            cusparseGetErrorString,
            cusparseGetMatIndexBase,
            cusparseGetMatDiagType,
            cusparseGetMatFillMode,
            cusparseGetMatType,
            // VEGA-PATCH 2026-06-30 sprint-rust-final-T15:42 cusparse stub graceful
            // per app reali (PyTorch capability probe, Blender splash, AutoCAD init).
            // Senza questi torch.cuda.is_available() falliva durante import perche'
            // PyTorch fa probing carica cusparse all'avvio. Vedi impl.rs commenti dettagli.
            cusparseCreate,
            cusparseDestroy,
            cusparseGetVersion,
            cusparseGetProperty,
            cusparseSetStream,
            cusparseGetStream,
            cusparseSetPointerMode,
            cusparseGetPointerMode,
            cusparseCreateMatDescr,
            cusparseDestroyMatDescr,
            cusparseSetMatType,
            cusparseSetMatFillMode,
            cusparseSetMatDiagType,
            cusparseSetMatIndexBase
        ]
);
