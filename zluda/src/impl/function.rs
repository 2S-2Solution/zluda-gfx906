use cuda_types::cuda::CUfunction_attribute;
use hip_runtime_sys::*;
use std::mem;

pub(crate) fn get_attribute(
    pi: &mut i32,
    cu_attrib: CUfunction_attribute,
    func: hipFunction_t,
) -> hipError_t {
    match cu_attrib {
        // Architettura virtuale del codice caricato. Va riportata quella VERA, letta
        // dal `.target` del PTX: le librerie a template la usano per scegliere i
        // parametri dell'algoritmo, e devono scegliere gli stessi che il kernel ha
        // gia' compilato dentro di se'.
        //
        // cub interroga questo attributo su `EmptyKernel`, lo moltiplica per 10 e ne
        // ricava la policy. Con la risposta fissa 120 otteneva 1200 -> Policy600
        // (15 elementi per thread), mentre il PTX che PyTorch distribuisce e' `sm_37`
        // e il kernel compila Policy350 (12 elementi). L'host dimensionava allora la
        // griglia per un tile piu' grande di quello elaborato davvero, e la coda dei
        // dati restava non scritta da nessuno: misurato il 31 lug 2026, `cumsum` su
        // 4096 int64 lasciava intatti gli ultimi 256 elementi. Stessa causa per
        // unique, randperm, nonzero, topk, scatter_add, masked_select.
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            let arch = super::module::ARCH_PTX_CARICATO.load(std::sync::atomic::Ordering::Relaxed);
            // Se non e' ancora stato caricato alcun PTX si resta prudenti: 3.5 e' la
            // soglia piu' bassa delle policy di cub, quindi non promette nulla che il
            // codice caricato poi non sappia fare.
            *pi = if arch > 0 { arch } else { 35 };
            return Ok(());
        }
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE => {
            *pi = 0;
            return Ok(());
        }
        _ => {}
    }
    unsafe { hipFuncGetAttribute(pi, mem::transmute(cu_attrib), func) }?;
    if cu_attrib == CUfunction_attribute::CU_FUNC_ATTRIBUTE_NUM_REGS {
        *pi = (*pi).max(1);
    }
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-6: costanti CUDA per extra array launch_kernel.
// CUDA usa array `extra` come terminated-list di coppie (tag, value) con questi tag noti:
//   CU_LAUNCH_PARAM_END             = 0x00  (terminatore, value ignorato)
//   CU_LAUNCH_PARAM_BUFFER_POINTER  = 0x01  (value = ptr a buffer parametri impacchettati)
//   CU_LAUNCH_PARAM_BUFFER_SIZE     = 0x02  (value = ptr a size_t con dimensione buffer)
// HIP supporta lo stesso protocollo con valori bit-compatibili, quindi il passthrough è
// generalmente safe; tuttavia validiamo qui i tag per evitare di girare ad HIP array malformati.
const CU_LAUNCH_PARAM_END_TAG: usize = 0x00;
const CU_LAUNCH_PARAM_BUFFER_POINTER_TAG: usize = 0x01;
const CU_LAUNCH_PARAM_BUFFER_SIZE_TAG: usize = 0x02;

pub(crate) fn launch_kernel(
    f: hipFunction_t,
    grid_dim_x: ::core::ffi::c_uint,
    grid_dim_y: ::core::ffi::c_uint,
    grid_dim_z: ::core::ffi::c_uint,
    block_dim_x: ::core::ffi::c_uint,
    block_dim_y: ::core::ffi::c_uint,
    block_dim_z: ::core::ffi::c_uint,
    shared_mem_bytes: ::core::ffi::c_uint,
    stream: hipStream_t,
    kernel_params: *mut *mut ::core::ffi::c_void,
    extra: *mut *mut ::core::ffi::c_void,
) -> hipError_t {
    // Traccia delle dimensioni del lancio: serve a vedere con quale griglia arrivano
    // i kernel di cub. Spenta salvo ZLUDA_DIAG.
    if let Ok(mut reg) = crate::diag("zluda_launch_dim.log") {
        use std::io::Write;
        let _ = writeln!(
            reg,
            "[lancio] griglia=({},{},{}) blocco=({},{},{}) shmem={}",
            grid_dim_x, grid_dim_y, grid_dim_z,
            block_dim_x, block_dim_y, block_dim_z, shared_mem_bytes
        );
    }
    // VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-6: validazione array extra prima di
    // delegare ad HIP. Se extra è non-null, scorriamo la lista di tag terminata da
    // CU_LAUNCH_PARAM_END accettando solo i tag noti. Se troviamo un tag sconosciuto
    // ritorniamo InvalidValue piuttosto che propagare puntatori potenzialmente unsafe.
    // kernel_params ed extra sono mutuamente esclusivi per spec CUDA: se entrambi
    // non-null lasciamo che HIP gestisca l'errore standard.
    if !extra.is_null() && kernel_params.is_null() {
        unsafe {
            let mut idx = 0isize;
            loop {
                let tag_ptr = *extra.offset(idx);
                let tag = tag_ptr as usize;
                match tag {
                    CU_LAUNCH_PARAM_END_TAG => break,
                    CU_LAUNCH_PARAM_BUFFER_POINTER_TAG
                    | CU_LAUNCH_PARAM_BUFFER_SIZE_TAG => {
                        // tag valido: salta tag + value
                        idx += 2;
                    }
                    _ => {
                        // tag sconosciuto: rifiuta senza dereferenziare oltre
                        return Err(hipErrorCode_t::InvalidValue);
                    }
                }
                // safety net: evita loop infiniti se array non terminato (max 64 coppie)
                if idx > 128 {
                    return Err(hipErrorCode_t::InvalidValue);
                }
            }
        }
    }
    unsafe {
        hipModuleLaunchKernel(
            f,
            grid_dim_x,
            grid_dim_y,
            grid_dim_z,
            block_dim_x,
            block_dim_y,
            block_dim_z,
            shared_mem_bytes,
            stream,
            kernel_params,
            extra,
        )
    }
}

pub(crate) unsafe fn set_attribute(
    func: hipFunction_t,
    attribute: CUfunction_attribute,
    value: i32,
) -> hipError_t {
    match attribute {
        CUfunction_attribute::CU_FUNC_ATTRIBUTE_PTX_VERSION
        | CUfunction_attribute::CU_FUNC_ATTRIBUTE_BINARY_VERSION => {
            return hipError_t::ErrorNotSupported;
        }
        _ => {}
    }
    hipFuncSetAttribute(func.0.cast(), hipFuncAttribute(attribute.0), value)
}
