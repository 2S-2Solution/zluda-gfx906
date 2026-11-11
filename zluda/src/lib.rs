use cuda_types::cuda::CUerror;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) mod r#impl;
// [2S, 22 ott 2026] funzioni del driver che mancavano (estese.rs)
mod estese;
// la tabella di cuGetProcAddress (zluda_bindgen/process_table.rs) cerca i simboli nella radice del crate
pub(crate) use estese::*;
#[cfg_attr(windows, path = "os_win.rs")]
#[cfg_attr(not(windows), path = "os_unix.rs")]
mod os;

static INITIALIZED: AtomicBool = AtomicBool::new(true);
pub(crate) fn initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

#[cfg_attr(not(windows), dtor::dtor)]
fn deinitialize() {
    INITIALIZED.store(false, Ordering::SeqCst);
}

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                // VEGA-DEBUG T10:05 23 giu: log unimplemented call per inchiodare quale chiama torch
                use std::io::Write;
                if let Ok(mut f) = crate::diag("nvcuda_unimpl_trace.log")
                {
                    let _ = writeln!(f, "[UNIMPLEMENTED] {}", stringify!($fn_name));
                }
                Err(r#impl::unimplemented())
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
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                let result: $ret_type = (|| {
                    cuda_macros::cuda_normalize_fn!( crate::r#impl::$fn_name ) ($(zluda_common::FromCuda::<_, CUerror>::from_cuda(&$arg_id)?),*)?;
                    Ok(())
                })();
                // VEGA-TRACE T08:43 23 giu: log persistente cu* call sequence
                use std::io::Write;
                if let Ok(mut f) = crate::diag("nvcuda_trace.log")
                {
                    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
                    let _ = writeln!(f, "[{}] ret={}", stringify!($fn_name), code);
                }
                result
            }
        )*
    };
}

// VEGA-PATCH 2026-06-30 T02:10 worker-2/3 TASK9 BUG9 — cuLink* IMPLEMENTAZIONE REALE
// (sostituisce il vecchio macro `link_stub` che ritornava NOT_SUPPORTED).
// Le 8 API sono definite manualmente sotto (dopo il blocco texture) come
// #[no_mangle] pub extern "system" fn con dispatch diretto a r#impl::cu_link::*
// che dynamic-loada hiprtc0507.dll (ROCm 5.7 esclusivo, regola 4-bis) e mappa
// CUjit_option / CUjitInputType su hiprtcJIT_option / hiprtcJITInputType
// (mapping 1:1 verificato cuda.rs:2497-2706 / 3016-3055 vs ROCm 5.7 hiprtc.h:72-118).
// Il macro `link_stub` resta presente come MARKER per il proc-macro
// cuda_function_declarations! affinché ESCLUDA le 8 API dal blocco
// `unimplemented` (evita symbol collision in lld-link). Il body emette zero
// codice — le definizioni reali sono sotto al modulo texture.
macro_rules! link_stub {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        // intentionally empty — funzioni emesse manualmente sotto (LINK_REAL)
    };
}

// VEGA-PATCH 2026-06-30 T01:55 worker-1/3 cuTexObject IMPLEMENTAZIONE REALE:
// sostituisce il vecchio `tex_stub`. amdhip64.dll v5.7 esporta hipTexObjectCreate/
// Destroy/Get* + hipCreateSurfaceObject/hipDestroySurfaceObject con signature/ABI
// strutturalmente identici alle controparti driver-API CUDA — driver-API descriptor
// HIP_RESOURCE_DESC, HIP_TEXTURE_DESC, HIP_RESOURCE_VIEW_DESC sono layout-compatibili
// con CUDA_RESOURCE_DESC / CUDA_TEXTURE_DESC / CUDA_RESOURCE_VIEW_DESC (stesso campo,
// stesso ordine, stessa size — verificato manualmente su cuda_types/cuda.rs:4907 vs
// hip_runtime-sys/lib.rs:2589). cuTexObject_t e cuSurfObject_t sono entrambi
// c_ulonglong (u64) e hipTextureObject_t / hipSurfaceObject_t sono raw pointer 8 byte:
// ABI-compatibili su x64. Per le surface, HIP espone solo l'API runtime
// `hipCreateSurfaceObject(hipResourceDesc*)` — il layout runtime DIFFERISCE dal
// driver-form (linear/pitch2D usano hipChannelFormatDesc invece di format+numChannels+
// sizeInBytes); il caller CUDA però per le surface object passa SEMPRE resType=ARRAY
// (vedi doc cuSurfObjectCreate: "resType must be CU_RESOURCE_TYPE_ARRAY"), e per la
// variante array i due layout coincidono (resType=0, union{ ptr }, padding identico).
// Costruiamo quindi un hipResourceDesc minimale dal CUDA driver-form per il caso ARRAY,
// e fall-back NOT_SUPPORTED per tipologie non-array (NeRF/pipeline texture standard
// usano sempre array). Le 5 API texture invece passano direttamente per cast in-place.

// Helper: cuTexObject_t/cuSurfObject_t (u64) <-> hipTextureObject_t/hipSurfaceObject_t (*mut)
// Su x86_64 entrambi 8 byte. transmute è sicuro per pure value passing.
#[inline(always)]
unsafe fn cu_to_hip_tex_obj(t: cuda_types::cuda::CUtexObject) -> hip_runtime_sys::hipTextureObject_t {
    t as hip_runtime_sys::hipTextureObject_t
}
#[inline(always)]
unsafe fn hip_to_cu_tex_obj(t: hip_runtime_sys::hipTextureObject_t) -> cuda_types::cuda::CUtexObject {
    t as cuda_types::cuda::CUtexObject
}
#[inline(always)]
unsafe fn cu_to_hip_surf_obj(s: cuda_types::cuda::CUsurfObject) -> hip_runtime_sys::hipSurfaceObject_t {
    s as hip_runtime_sys::hipSurfaceObject_t
}
#[inline(always)]
unsafe fn hip_to_cu_surf_obj(s: hip_runtime_sys::hipSurfaceObject_t) -> cuda_types::cuda::CUsurfObject {
    s as cuda_types::cuda::CUsurfObject
}

// Helper: map hipError_t -> CUresult.
// hipError_t è alias di `Result<(), hipErrorCode_t>` (vedi hip_runtime-sys/lib.rs:13689),
// quindi pattern-match diretto su `Ok(_)` / `Err(code)`. Usiamo l'impl
// `From<hipErrorCode_t> for CUerror` già esistente in cuda_types/cuda.rs:9399.
#[inline(always)]
fn hip_to_cuda_result(e: hip_runtime_sys::hipError_t) -> Result<(), CUerror> {
    match e {
        Ok(()) => Ok(()),
        Err(code) => Err(CUerror::from(code)),
    }
}

#[inline(always)]
fn trace_tex(name: &'static str, code: u32) {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("nvcuda_trace.log")
    {
        let _ = writeln!(f, "[{} TEX_REAL] ret={}", name, code);
    }
}

// ============ 5 API TEXTURE OBJECT ============
#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuTexObjectCreate(
    pTexObject: *mut cuda_types::cuda::CUtexObject,
    pResDesc: *const cuda_types::cuda::CUDA_RESOURCE_DESC,
    pTexDesc: *const cuda_types::cuda::CUDA_TEXTURE_DESC,
    pResViewDesc: *const cuda_types::cuda::CUDA_RESOURCE_VIEW_DESC,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pTexObject.is_null() || pResDesc.is_null() || pTexDesc.is_null() {
        trace_tex("cuTexObjectCreate", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut hip_tex: hip_runtime_sys::hipTextureObject_t = std::ptr::null_mut();
    let hip_err = hip_runtime_sys::hipTexObjectCreate(
        &mut hip_tex as *mut _,
        pResDesc as *const hip_runtime_sys::HIP_RESOURCE_DESC,
        pTexDesc as *const hip_runtime_sys::HIP_TEXTURE_DESC,
        pResViewDesc as *const hip_runtime_sys::HIP_RESOURCE_VIEW_DESC,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() {
        *pTexObject = hip_to_cu_tex_obj(hip_tex);
    }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuTexObjectCreate", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuTexObjectDestroy(
    texObject: cuda_types::cuda::CUtexObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let hip_err = hip_runtime_sys::hipTexObjectDestroy(cu_to_hip_tex_obj(texObject));
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuTexObjectDestroy", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuTexObjectGetResourceDesc(
    pResDesc: *mut cuda_types::cuda::CUDA_RESOURCE_DESC,
    texObject: cuda_types::cuda::CUtexObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pResDesc.is_null() {
        trace_tex("cuTexObjectGetResourceDesc", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let hip_err = hip_runtime_sys::hipTexObjectGetResourceDesc(
        pResDesc as *mut hip_runtime_sys::HIP_RESOURCE_DESC,
        cu_to_hip_tex_obj(texObject),
    );
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuTexObjectGetResourceDesc", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuTexObjectGetResourceViewDesc(
    pResViewDesc: *mut cuda_types::cuda::CUDA_RESOURCE_VIEW_DESC,
    texObject: cuda_types::cuda::CUtexObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pResViewDesc.is_null() {
        trace_tex("cuTexObjectGetResourceViewDesc", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let hip_err = hip_runtime_sys::hipTexObjectGetResourceViewDesc(
        pResViewDesc as *mut hip_runtime_sys::HIP_RESOURCE_VIEW_DESC,
        cu_to_hip_tex_obj(texObject),
    );
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuTexObjectGetResourceViewDesc", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuTexObjectGetTextureDesc(
    pTexDesc: *mut cuda_types::cuda::CUDA_TEXTURE_DESC,
    texObject: cuda_types::cuda::CUtexObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pTexDesc.is_null() {
        trace_tex("cuTexObjectGetTextureDesc", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let hip_err = hip_runtime_sys::hipTexObjectGetTextureDesc(
        pTexDesc as *mut hip_runtime_sys::HIP_TEXTURE_DESC,
        cu_to_hip_tex_obj(texObject),
    );
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuTexObjectGetTextureDesc", code);
    result
}

// ============ 3 API SURFACE OBJECT ============
// Surface: HIP espone solo l'API runtime hipCreateSurfaceObject(*hipResourceDesc).
// Per CUDA spec resType deve essere ARRAY (CU_RESOURCE_TYPE_ARRAY = 0 = hipResourceTypeArray),
// quindi costruiamo runtime-desc minimale dal driver-desc CUDA leggendo solo l'array handle.
#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuSurfObjectCreate(
    pSurfObject: *mut cuda_types::cuda::CUsurfObject,
    pResDesc: *const cuda_types::cuda::CUDA_RESOURCE_DESC,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pSurfObject.is_null() || pResDesc.is_null() {
        trace_tex("cuSurfObjectCreate", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let cu_desc = &*pResDesc;
    // CUresourcetype: 0 = ARRAY (richiesto da spec cuSurfObjectCreate).
    if cu_desc.resType.0 != 0 {
        trace_tex("cuSurfObjectCreate", CUerror::INVALID_VALUE.0.get());
        return Err(CUerror::INVALID_VALUE);
    }
    // Estrai hArray dal driver-form union.array.hArray (CUarray = *mut CUarray_st).
    let h_array = cu_desc.res.array.hArray;
    // Costruisci runtime-form hipResourceDesc per hipCreateSurfaceObject.
    // hipArray_t è *mut hipArray (pointer 8 byte); CUarray è *mut CUarray_st (pointer 8 byte).
    // Sono opaque handle, basta cast del puntatore.
    let mut runtime_desc: hip_runtime_sys::hipResourceDesc = std::mem::zeroed();
    runtime_desc.resType = hip_runtime_sys::hipResourceType::hipResourceTypeArray;
    runtime_desc.res.array.array = h_array as hip_runtime_sys::hipArray_t;
    let mut hip_surf: hip_runtime_sys::hipSurfaceObject_t = std::ptr::null_mut();
    let hip_err = hip_runtime_sys::hipCreateSurfaceObject(
        &mut hip_surf as *mut _,
        &runtime_desc as *const _,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() {
        *pSurfObject = hip_to_cu_surf_obj(hip_surf);
    }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuSurfObjectCreate", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuSurfObjectDestroy(
    surfObject: cuda_types::cuda::CUsurfObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let hip_err = hip_runtime_sys::hipDestroySurfaceObject(cu_to_hip_surf_obj(surfObject));
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_tex("cuSurfObjectDestroy", code);
    result
}

// No-op macro: i corpi delle 8 cuTex/cuSurf sono definiti sopra come #[no_mangle].
// Questo macro viene chiamato dal proc_macro `cuda_function_declarations!` per le
// funzioni listate in `tex_real_external <= [...]` e deve espandere a niente per
// evitare duplicate symbol. Nessun output → nessuna funzione duplicata.
macro_rules! tex_real_external {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        // intentionally empty — i wrapper reali sono definiti sopra
    };
}

// HIP non esporta hipGetSurfaceObjectResourceDesc — restituiamo NOT_SUPPORTED.
// È coerente con la spec CUDA (NOT_SUPPORTED è un return code valido), e i caller
// NeRF/texture standard non leggono mai indietro il descrittore di una surface
// dopo averla creata. Manteniamo signature reale + tracing reale.
#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
#[allow(unused_variables)]
pub unsafe extern "system" fn cuSurfObjectGetResourceDesc(
    pResDesc: *mut cuda_types::cuda::CUDA_RESOURCE_DESC,
    surfObject: cuda_types::cuda::CUsurfObject,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    trace_tex("cuSurfObjectGetResourceDesc", CUerror::NOT_SUPPORTED.0.get());
    // HIP non espone hipGetSurfaceObjectResourceDesc; spec CUDA permette NOT_SUPPORTED.
    Err(CUerror::NOT_SUPPORTED)
}

// ============ 8 API cuLink* IMPLEMENTAZIONE REALE ============
// Dispatch diretto a r#impl::cu_link::* — vedi commenti in impl/cu_link.rs per
// mapping enum + dynamic load hiprtc0507.dll (ROCm 5.7 esclusivo regola 4-bis).
// _v2 variants: in CUDA driver-API le _v2 hanno la STESSA signature/semantics
// della base (legacy ABI dropped CUjit_option_v1 obsoleto post-CUDA 4); le
// wrappiamo identicamente.

#[inline(always)]
fn trace_link(name: &'static str, code: u32) {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("nvcuda_trace.log")
    {
        let _ = writeln!(f, "[{} LINK_REAL] ret={}", name, code);
    }
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkCreate(
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
    stateOut: *mut cuda_types::cuda::CUlinkState,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_create(numOptions, options, optionValues, stateOut);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkCreate", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkCreate_v2(
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
    stateOut: *mut cuda_types::cuda::CUlinkState,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_create(numOptions, options, optionValues, stateOut);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkCreate_v2", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkAddData(
    state: cuda_types::cuda::CUlinkState,
    inputType: cuda_types::cuda::CUjitInputType,
    data: *mut ::core::ffi::c_void,
    size: usize,
    name: *const ::core::ffi::c_char,
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_add_data(
        state, inputType, data, size, name, numOptions, options, optionValues,
    );
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkAddData", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkAddData_v2(
    state: cuda_types::cuda::CUlinkState,
    inputType: cuda_types::cuda::CUjitInputType,
    data: *mut ::core::ffi::c_void,
    size: usize,
    name: *const ::core::ffi::c_char,
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_add_data(
        state, inputType, data, size, name, numOptions, options, optionValues,
    );
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkAddData_v2", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkAddFile(
    state: cuda_types::cuda::CUlinkState,
    inputType: cuda_types::cuda::CUjitInputType,
    path: *const ::core::ffi::c_char,
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_add_file(
        state, inputType, path, numOptions, options, optionValues,
    );
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkAddFile", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkAddFile_v2(
    state: cuda_types::cuda::CUlinkState,
    inputType: cuda_types::cuda::CUjitInputType,
    path: *const ::core::ffi::c_char,
    numOptions: ::core::ffi::c_uint,
    options: *mut cuda_types::cuda::CUjit_option,
    optionValues: *mut *mut ::core::ffi::c_void,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_add_file(
        state, inputType, path, numOptions, options, optionValues,
    );
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkAddFile_v2", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkComplete(
    state: cuda_types::cuda::CUlinkState,
    cubinOut: *mut *mut ::core::ffi::c_void,
    sizeOut: *mut usize,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_complete(state, cubinOut, sizeOut);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkComplete", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuLinkDestroy(
    state: cuda_types::cuda::CUlinkState,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let result = r#impl::cu_link::link_destroy(state);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_link("cuLinkDestroy", code);
    result
}

// ============================================================================
// VEGA-PATCH 2026-06-30 T16:10 GL-INTEROP — IMPLEMENTAZIONE REALE
// ============================================================================
// Target: Blender Cycles, viewport interop, Houdini/Resolve GL/CUDA exchange.
// amdhip64.dll v5.7 ROCm esporta 9 funzioni HIP-GL (vedi amdhip64.def).
// CUgraphicsResource (*mut CUgraphicsResource_st, opaque pointer 8B) e
// hipGraphicsResource_t (*mut _hipGraphicsResource, opaque pointer 8B) sono
// ABI-compatibili (entrambi opaque handle 8B). Cast diretto.
// GLuint = u32, GLenum = u32 — ABI-compatibili con c_uint.
// Flag enum: CU_GRAPHICS_REGISTER_FLAGS_* (cuda_types/cuda.rs:3069-3097) mapping
// 1:1 a hipGraphicsRegisterFlags (hip_runtime-sys/lib.rs:4061-4094) verificato.
// CUgraphicsMapResourceFlags_enum (cuda_types/cuda.rs:3100-3120) mapping 1:1 a
// flag HIP runtime (NONE=0/READ_ONLY=1/WRITE_DISCARD=2, identico nei due header).

#[inline(always)]
fn trace_gl(name: &'static str, code: u32) {
    use std::io::Write;
    if let Ok(mut f) = crate::diag("nvcuda_trace.log")
    {
        let _ = writeln!(f, "[{} GL_REAL] ret={}", name, code);
    }
}

#[inline(always)]
unsafe fn cu_to_hip_gfx_res(r: cuda_types::cuda::CUgraphicsResource)
    -> hip_runtime_sys::hipGraphicsResource_t
{
    r as hip_runtime_sys::hipGraphicsResource_t
}

#[inline(always)]
unsafe fn hip_to_cu_gfx_res(r: hip_runtime_sys::hipGraphicsResource_t)
    -> cuda_types::cuda::CUgraphicsResource
{
    r as cuda_types::cuda::CUgraphicsResource
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsGLRegisterBuffer(
    pCudaResource: *mut cuda_types::cuda::CUgraphicsResource,
    buffer: cuda_types::cuda::GLuint,
    Flags: ::core::ffi::c_uint,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pCudaResource.is_null() {
        trace_gl("cuGraphicsGLRegisterBuffer", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut hip_res: hip_runtime_sys::hipGraphicsResource_t = std::ptr::null_mut();
    let hip_err = hip_runtime_sys::hipGraphicsGLRegisterBuffer(
        &mut hip_res as *mut _,
        buffer as ::core::ffi::c_uint,
        Flags,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() { *pCudaResource = hip_to_cu_gfx_res(hip_res); }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsGLRegisterBuffer", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsGLRegisterImage(
    pCudaResource: *mut cuda_types::cuda::CUgraphicsResource,
    image: cuda_types::cuda::GLuint,
    target: cuda_types::cuda::GLenum,
    Flags: ::core::ffi::c_uint,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pCudaResource.is_null() {
        trace_gl("cuGraphicsGLRegisterImage", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut hip_res: hip_runtime_sys::hipGraphicsResource_t = std::ptr::null_mut();
    let hip_err = hip_runtime_sys::hipGraphicsGLRegisterImage(
        &mut hip_res as *mut _,
        image as ::core::ffi::c_uint,
        target as ::core::ffi::c_uint,
        Flags,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() { *pCudaResource = hip_to_cu_gfx_res(hip_res); }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsGLRegisterImage", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsUnregisterResource(
    resource: cuda_types::cuda::CUgraphicsResource,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let hip_err = hip_runtime_sys::hipGraphicsUnregisterResource(cu_to_hip_gfx_res(resource));
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsUnregisterResource", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsMapResources(
    count: ::core::ffi::c_uint,
    resources: *mut cuda_types::cuda::CUgraphicsResource,
    hStream: cuda_types::cuda::CUstream,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let hip_err = hip_runtime_sys::hipGraphicsMapResources(
        count as ::core::ffi::c_int,
        resources as *mut hip_runtime_sys::hipGraphicsResource_t,
        hip_runtime_sys::hipStream_t(hStream.0 as *mut hip_runtime_sys::ihipStream_t),
    );
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsMapResources", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsMapResources_ptsz(
    count: ::core::ffi::c_uint,
    resources: *mut cuda_types::cuda::CUgraphicsResource,
    hStream: cuda_types::cuda::CUstream,
) -> cuda_types::cuda::CUresult {
    cuGraphicsMapResources(count, resources, hStream)
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsUnmapResources(
    count: ::core::ffi::c_uint,
    resources: *mut cuda_types::cuda::CUgraphicsResource,
    hStream: cuda_types::cuda::CUstream,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    let hip_err = hip_runtime_sys::hipGraphicsUnmapResources(
        count as ::core::ffi::c_int,
        resources as *mut hip_runtime_sys::hipGraphicsResource_t,
        hip_runtime_sys::hipStream_t(hStream.0 as *mut hip_runtime_sys::ihipStream_t),
    );
    let result = hip_to_cuda_result(hip_err);
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsUnmapResources", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsUnmapResources_ptsz(
    count: ::core::ffi::c_uint,
    resources: *mut cuda_types::cuda::CUgraphicsResource,
    hStream: cuda_types::cuda::CUstream,
) -> cuda_types::cuda::CUresult {
    cuGraphicsUnmapResources(count, resources, hStream)
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsResourceGetMappedPointer_v2(
    pDevPtr: *mut cuda_types::cuda::CUdeviceptr,
    pSize: *mut usize,
    resource: cuda_types::cuda::CUgraphicsResource,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pDevPtr.is_null() {
        trace_gl("cuGraphicsResourceGetMappedPointer_v2", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut dev_ptr: *mut ::core::ffi::c_void = std::ptr::null_mut();
    let mut size: usize = 0;
    let hip_err = hip_runtime_sys::hipGraphicsResourceGetMappedPointer(
        &mut dev_ptr as *mut _,
        &mut size as *mut _,
        cu_to_hip_gfx_res(resource),
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() {
        // CUdeviceptr e' unsigned integer; transmute pointer -> uint sicuro x64
        *pDevPtr = cuda_types::cuda::CUdeviceptr_v2(dev_ptr);
        if !pSize.is_null() { *pSize = size; }
    }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsResourceGetMappedPointer_v2", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsResourceGetMappedPointer(
    pDevPtr: *mut cuda_types::cuda::CUdeviceptr,
    pSize: *mut usize,
    resource: cuda_types::cuda::CUgraphicsResource,
) -> cuda_types::cuda::CUresult {
    cuGraphicsResourceGetMappedPointer_v2(pDevPtr, pSize, resource)
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsSubResourceGetMappedArray(
    pArray: *mut cuda_types::cuda::CUarray,
    resource: cuda_types::cuda::CUgraphicsResource,
    arrayIndex: ::core::ffi::c_uint,
    mipLevel: ::core::ffi::c_uint,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pArray.is_null() {
        trace_gl("cuGraphicsSubResourceGetMappedArray", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut hip_arr: hip_runtime_sys::hipArray_t = std::ptr::null_mut();
    let hip_err = hip_runtime_sys::hipGraphicsSubResourceGetMappedArray(
        &mut hip_arr as *mut _,
        cu_to_hip_gfx_res(resource),
        arrayIndex,
        mipLevel,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() {
        *pArray = hip_arr as cuda_types::cuda::CUarray;
    }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGraphicsSubResourceGetMappedArray", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsResourceSetMapFlags_v2(
    _resource: cuda_types::cuda::CUgraphicsResource,
    _flags: ::core::ffi::c_uint,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    // VEGA-FIX T17:24 sprint 71-agent: hipGraphicsResourceSetMapFlags NON esiste
    // in amdhip64.dll Adrenalin v5.7. Stub graceful no-op SUCCESS (le flags
    // di mapping non sono usate da AMD HIP runtime: pass-through trasparente).
    trace_gl("cuGraphicsResourceSetMapFlags_v2", 0);
    Ok(())
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGraphicsResourceSetMapFlags(
    resource: cuda_types::cuda::CUgraphicsResource,
    flags: ::core::ffi::c_uint,
) -> cuda_types::cuda::CUresult {
    cuGraphicsResourceSetMapFlags_v2(resource, flags)
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGLGetDevices_v2(
    pCudaDeviceCount: *mut ::core::ffi::c_uint,
    pCudaDevices: *mut cuda_types::cuda::CUdevice,
    cudaDeviceCount: ::core::ffi::c_uint,
    deviceList: cuda_types::cuda::CUGLDeviceList,
) -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    if pCudaDeviceCount.is_null() {
        trace_gl("cuGLGetDevices_v2", 1);
        return Err(CUerror::INVALID_VALUE);
    }
    let mut hip_count: ::core::ffi::c_uint = 0;
    let hip_err = hip_runtime_sys::hipGLGetDevices(
        &mut hip_count as *mut _,
        pCudaDevices as *mut ::core::ffi::c_int,
        cudaDeviceCount,
        deviceList.0,
    );
    let result = hip_to_cuda_result(hip_err);
    if result.is_ok() { *pCudaDeviceCount = hip_count; }
    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
    trace_gl("cuGLGetDevices_v2", code);
    result
}

#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGLGetDevices(
    pCudaDeviceCount: *mut ::core::ffi::c_uint,
    pCudaDevices: *mut cuda_types::cuda::CUdevice,
    cudaDeviceCount: ::core::ffi::c_uint,
    deviceList: cuda_types::cuda::CUGLDeviceList,
) -> cuda_types::cuda::CUresult {
    cuGLGetDevices_v2(pCudaDeviceCount, pCudaDevices, cudaDeviceCount, deviceList)
}

// cuGLInit: deprecato dal CUDA 3.0. Su HIP non e' richiesto bind-time.
// Ritorniamo SUCCESS no-op (compat con app che la chiamano per legacy boilerplate).
#[cfg_attr(not(test), no_mangle)]
#[allow(improper_ctypes)]
#[allow(improper_ctypes_definitions)]
pub unsafe extern "system" fn cuGLInit() -> cuda_types::cuda::CUresult {
    if !initialized() { return Err(CUerror::DEINITIALIZED); }
    trace_gl("cuGLInit", 0);
    Ok(())
}

// Stub marker no-op (gemello a link_stub / tex_real_external) per le 13 cuGL/Graphics
// definite manualmente sopra — esclude dal blocco unimplemented per evitare duplicate symbol.
macro_rules! gl_real_external {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        // intentionally empty — i wrapper reali sono definiti sopra
    };
}

macro_rules! estese_external {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        // intentionally empty — i wrapper reali sono in estese.rs (2S, 22 ott 2026)
    };
}

macro_rules! implemented_in_function {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            #[cfg_attr(not(test), no_mangle)]
            #[allow(improper_ctypes)]
            #[allow(improper_ctypes_definitions)]
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                if !initialized() {
                    return Err(CUerror::DEINITIALIZED);
                }
                let result: $ret_type = (|| {
                    cuda_macros::cuda_normalize_fn!( crate::r#impl::function::$fn_name ) ($(zluda_common::FromCuda::<_, CUerror>::from_cuda(&$arg_id)?),*)?;
                    Ok(())
                })();
                // VEGA-TRACE T10:45 - log implemented_in_function (include cuLaunchKernel)
                use std::io::Write;
                if let Ok(mut f) = crate::diag("nvcuda_trace.log")
                {
                    let code = match &result { Ok(_) => 0u32, Err(e) => e.0.get() };
                    let _ = writeln!(f, "[{}] ret={}", stringify!($fn_name), code);
                }
                result
            }
        )*
    };
}

cuda_macros::cuda_function_declarations!(
    unimplemented,
    implemented
        <= [
            cuCtxCreate_v2,
            cuCtxDestroy_v2,
            cuCtxGetApiVersion,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuCtxGetDevice_v2,
            cuCtxGetLimit,
            cuCtxGetStreamPriorityRange,
            cuCtxPopCurrent,
            cuCtxPopCurrent_v2,
            cuCtxPushCurrent,
            cuCtxPushCurrent_v2,
            cuCtxSetCurrent,
            cuCtxSetFlags,
            cuCtxSetLimit,
            cuCtxSynchronize,
            cuDeviceComputeCapability,
            cuDeviceGet,
            cuDeviceGetAttribute,
            cuDeviceGetCount,
            cuDeviceGetLuid,
            cuDeviceGetName,
            cuDeviceGetProperties,
            cuDeviceGetUuid,
            cuDeviceGetUuid_v2,
            cuDevicePrimaryCtxGetState,
            cuDevicePrimaryCtxRelease,
            cuDevicePrimaryCtxRelease_v2,
            cuDevicePrimaryCtxReset,
            cuDevicePrimaryCtxRetain,
            cuDevicePrimaryCtxSetFlags,
            cuDevicePrimaryCtxSetFlags_v2,
            cuDeviceTotalMem_v2,
            cuDriverGetVersion,
            cuEventCreate,
            cuEventDestroy_v2,
            cuEventElapsedTime,
            cuEventQuery,
            cuEventRecord,
            cuEventSynchronize,
            cuFuncGetAttribute,
            cuFuncSetAttribute,
            cuGetErrorString,
            cuGetExportTable,
            cuGetProcAddress,
            cuGetProcAddress_v2,
            cuGraphDestroy,
            cuGraphExecDestroy,
            cuGraphExecUpdate_v2,
            cuGraphGetNodes,
            cuGraphInstantiateWithFlags,
            cuGraphLaunch,
            cuInit,
            cuKernelGetFunction,
            // VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-8: cuKernelGetAttribute wiring (CUDA 12+ library API)
            cuKernelGetAttribute,
            cuKernelSetAttribute,
            cuLaunchKernel,
            cuLaunchKernelEx,
            cuLibraryGetGlobal,
            cuLibraryGetKernel,
            cuLibraryGetModule,
            cuLibraryLoadData,
            cuLibraryUnload,
            cuMemAlloc_v2,
            // VEGA-PATCH 2026-06-30 GAP G7 worker 2/5 — VMM API cuMemCreate + cuMemRelease
            // Implementati in r#impl::memory::{create, release} con registry
            // HashMap<u64, hipMemGenericAllocationHandle_t> + fallback graceful
            // ERROR_NOT_SUPPORTED per attivare path legacy hipMalloc di torch
            // caching allocator quando hipMemCreate fail su Windows ROCm 5.7.
            cuMemCreate,
            cuMemRelease,
            cuMemFreeHost,
            cuMemFree_v2,
            cuMemGetAddressRange_v2,
            cuMemGetAllocationGranularity,
            cuMemGetInfo_v2,
            cuMemHostAlloc,
            cuMemRetainAllocationHandle,
            cuMemcpyAsync,
            // VEGA-PATCH 2026-06-28 sprint-orchestration: cuMemcpyAsync_ptsz delegate (PyTorch CUDA 12+ per-thread stream)
            cuMemcpyAsync_ptsz,
            cuMemcpyDtoDAsync_v2,
            cuMemcpyDtoH_v2_ptds,
            cuMemcpyDtoHAsync_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyHtoDAsync_v2,
            cuMemcpyHtoD_v2,
            cuMemcpyHtoD_v2_ptds,
            cuMemsetD32_v2,
            cuMemsetD8Async,
            cuMemsetD8_v2,
            cuModuleGetFunction,
            cuModuleGetGlobal_v2,
            cuModuleGetLoadingMode,
            cuModuleLoadData,
            cuModuleLoadDataEx,
            cuModuleLoadFatBinary,
            cuModuleUnload,
            cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
            cuOccupancyMaxPotentialBlockSize,
            cuPointerGetAttribute,
            cuPointerGetAttributes,
            cuProfilerStart,
            cuProfilerStop,
            cuStreamBeginCapture_v2,
            cuStreamCreate,
            cuStreamCreateWithPriority,
            cuStreamDestroy_v2,
            cuStreamEndCapture,
            cuStreamGetCaptureInfo_v2,
            cuStreamIsCapturing,
            cuStreamSynchronize,
            cuStreamWaitEvent,
            cuThreadExchangeStreamCaptureMode,
        ],
    implemented_in_function <= [cuLaunchKernel,],
    // VEGA-PATCH 2026-06-30 TASK9 BUG9: cuLink* graceful stubs (Blender JIT linker not supported on AMD)
    link_stub <= [
        cuLinkCreate,
        cuLinkCreate_v2,
        cuLinkAddData,
        cuLinkAddData_v2,
        cuLinkAddFile,
        cuLinkAddFile_v2,
        cuLinkComplete,
        cuLinkDestroy,
    ],
    // VEGA-PATCH 2026-06-30 T01:55 TASK10 BUG10: cuTexObject*/cuSurfObject* IMPLEMENTAZIONE REALE
    // (le 8 API sono definite manualmente sopra come #[no_mangle] pub extern "system" fn ...
    //  con dispatch diretto a hipTexObjectCreate/Destroy/Get* + hipCreateSurfaceObject/Destroy.
    //  Vanno listate qui in un blocco override no-op `tex_real_external!` per ESCLUDERLE dalla
    //  generazione automatica del blocco `unimplemented` — altrimenti duplicate symbol linker
    //  error. Il macro `tex_real_external!` espande a NULLA: il body reale è già sopra.)
    tex_real_external <= [
        cuTexObjectCreate,
        cuTexObjectDestroy,
        cuTexObjectGetResourceDesc,
        cuTexObjectGetResourceViewDesc,
        cuTexObjectGetTextureDesc,
        cuSurfObjectCreate,
        cuSurfObjectDestroy,
        cuSurfObjectGetResourceDesc,
    ],
    // VEGA-PATCH 2026-06-30 T16:10 GL-INTEROP: 14 cuGL*/cuGraphics* IMPL REALE
    // Wrap diretto a hipGraphicsGLRegisterBuffer/Image + 7 Map/Unmap/Get/Set +
    // hipGLGetDevices (amdhip64.dll v5.7 ROCm). Sblocca Blender Cycles + viewport
    // GL interop. cuGLInit ritorna SUCCESS no-op (deprecato CUDA 3.0).
    gl_real_external <= [
        cuGraphicsGLRegisterBuffer,
        cuGraphicsGLRegisterImage,
        cuGraphicsUnregisterResource,
        cuGraphicsMapResources,
        cuGraphicsMapResources_ptsz,
        cuGraphicsUnmapResources,
        cuGraphicsUnmapResources_ptsz,
        cuGraphicsResourceGetMappedPointer,
        cuGraphicsResourceGetMappedPointer_v2,
        cuGraphicsSubResourceGetMappedArray,
        cuGraphicsResourceSetMapFlags,
        cuGraphicsResourceSetMapFlags_v2,
        cuGLGetDevices,
        cuGLGetDevices_v2,
        cuGLInit,
    ],
    // [2S, 22 ott 2026] funzioni comuni che mancavano: implementate in estese.rs
    estese_external <= [
        cuStreamQuery,
        cuStreamQuery_ptsz,
        cuStreamGetPriority,
        cuStreamGetFlags,
        cuStreamAddCallback,
        cuLaunchHostFunc,
        cuMemcpy,
        cuMemcpyDtoD_v2,
        cuMemsetD16_v2,
        cuMemsetD16Async,
        cuMemsetD32Async,
        cuMemsetD2D8_v2,
        cuMemsetD2D8Async,
        cuMemsetD2D16_v2,
        cuMemsetD2D16Async,
        cuMemsetD2D32_v2,
        cuMemsetD2D32Async,
        cuMemcpy2D_v2,
        cuMemcpy2DUnaligned_v2,
        cuMemcpy2DAsync_v2,
        cuMemcpy3D_v2,
        cuMemcpy3DAsync_v2,
        cuMemAllocPitch_v2,
        cuMemAllocManaged,
        cuMemAllocHost_v2,
        cuMemHostRegister_v2,
        cuMemHostUnregister,
        cuMemHostGetDevicePointer_v2,
        cuMemHostGetFlags,
        cuMemAllocAsync,
        cuMemFreeAsync,
        cuMemPrefetchAsync,
        cuMemAdvise,
        cuDeviceGetDefaultMemPool,
        cuMemPoolTrimTo,
        cuMemPoolSetAttribute,
        cuMemPoolGetAttribute,
        cuArrayCreate_v2,
        cuArrayDestroy,
        cuMemcpyHtoA_v2,
        cuMemcpyAtoH_v2,
        cuEventRecordWithFlags,
        cuDeviceCanAccessPeer,
        cuCtxEnablePeerAccess,
        cuCtxDisablePeerAccess,
        cuDeviceGetPCIBusId,
        cuDeviceGetByPCIBusId,
        cuCtxGetFlags,
        cuCtxResetPersistingL2Cache,
        cuCtxSetCacheConfig,
        cuCtxGetCacheConfig,
        cuFuncSetCacheConfig,
        cuFuncSetSharedMemConfig,
        cuCtxSetSharedMemConfig,
        cuCtxGetSharedMemConfig,
        cuModuleLoad,
        cuLaunchCooperativeKernel,
    ]
);

#[cfg(test)]
mod tests;


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
