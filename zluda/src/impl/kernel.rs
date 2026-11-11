use cuda_types::cuda::{CUfunction_attribute, CUresult};
use hip_runtime_sys::*;

use crate::r#impl::function;

pub(crate) unsafe fn get_function(func: &mut hipFunction_t, kernel: hipFunction_t) -> CUresult {
    *func = kernel;
    Ok(())
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-8: cuKernelGetAttribute forwarder (CUDA 12+ library API)
// Stessa semantica di cuFuncGetAttribute ma con (kernel, dev) per scope device-wide. ZLUDA su gfx906
// tratta CUkernel==hipFunction_t e ignora il device handle (kernel valido per il context corrente).
pub(crate) unsafe fn get_attribute(
    pi: &mut ::core::ffi::c_int,
    attrib: CUfunction_attribute,
    kernel: hipFunction_t,
    _dev: hipDevice_t,
) -> hipError_t {
    function::get_attribute(pi, attrib, kernel)
}

pub(crate) unsafe fn set_attribute(
    attrib: CUfunction_attribute,
    val: ::core::ffi::c_int,
    kernel: hipFunction_t,
    _dev: hipDevice_t,
) -> hipError_t {
    function::set_attribute(kernel, attrib, val)
}
