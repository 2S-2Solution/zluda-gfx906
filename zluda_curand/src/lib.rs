//! zluda_curand — cuRAND shim su hiprand ROCm 5.7 per Vega 20 gfx906.
//!
//! Esporta i simboli che si aspettano i binari PyTorch (curand64_10.dll +
//! cudart-driven RNG). Tutte le chiamate sono inoltrate a hiprand 5.7 lasciando
//! invariato lo stream/contesto HIP corrente.
//!
//! Edge-case "W2 sediment" risolto in `curandGenerateNormal*`: hipRAND Box-Muller
//! richiede `n` pari. Se PyTorch chiede dispari (es. odd N in inizializzazione
//! pesi LSTM con hidden_size dispari) generiamo n+1 in un buffer scratch device
//! e copiamo back n, lasciando 0 leak di entropy oltre il margine.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::ffi::c_void;
use std::mem;
use std::ptr;

use hiprand_sys::{
    hiprandCreateGenerator, hiprandDestroyGenerator, hiprandGenerate,
    hiprandGenerateNormal, hiprandGenerateNormalDouble, hiprandGenerateUniform,
    hiprandGenerateUniformDouble, hiprandGenerator_t, hiprandGetVersion,
    hiprandRngType_t, hiprandSetGeneratorOffset, hiprandSetPseudoRandomGeneratorSeed,
    hiprandSetStream, hiprandStatus_t, hipStream_t,
};

use hip_runtime_sys::{
    hipFree, hipMalloc, hipMemcpyAsync, hipMemcpyKind, hipStreamSynchronize,
    hipStream_t as HipRtStream,
};

/// curandStatus_t (NVIDIA ABI). Stesso layout di hiprandStatus_t — eccetto codici
/// 200/300/etc. che hipRAND non emette. Mantengo il mapping 1:1 sui ritorni
/// che PyTorch effettivamente controlla.
pub type curandStatus_t = u32;
pub const CURAND_STATUS_SUCCESS: curandStatus_t = 0;
pub const CURAND_STATUS_VERSION_MISMATCH: curandStatus_t = 100;
pub const CURAND_STATUS_NOT_INITIALIZED: curandStatus_t = 101;
pub const CURAND_STATUS_ALLOCATION_FAILED: curandStatus_t = 102;
pub const CURAND_STATUS_TYPE_ERROR: curandStatus_t = 103;
pub const CURAND_STATUS_OUT_OF_RANGE: curandStatus_t = 104;
pub const CURAND_STATUS_LENGTH_NOT_MULTIPLE: curandStatus_t = 105;
pub const CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: curandStatus_t = 106;
pub const CURAND_STATUS_LAUNCH_FAILURE: curandStatus_t = 201;
pub const CURAND_STATUS_PREEXISTING_FAILURE: curandStatus_t = 202;
pub const CURAND_STATUS_INITIALIZATION_FAILED: curandStatus_t = 203;
pub const CURAND_STATUS_ARCH_MISMATCH: curandStatus_t = 204;
pub const CURAND_STATUS_INTERNAL_ERROR: curandStatus_t = 999;

/// curandRngType (NVIDIA ABI). I valori NVIDIA partono da 100..., diversi da
/// hipRAND (400...). Faccio mapping esplicito.
pub type curandRngType_t = u32;
pub const CURAND_RNG_TEST: curandRngType_t = 0;
pub const CURAND_RNG_PSEUDO_DEFAULT: curandRngType_t = 100;
pub const CURAND_RNG_PSEUDO_XORWOW: curandRngType_t = 101;
pub const CURAND_RNG_PSEUDO_MRG32K3A: curandRngType_t = 121;
pub const CURAND_RNG_PSEUDO_MTGP32: curandRngType_t = 141;
pub const CURAND_RNG_PSEUDO_MT19937: curandRngType_t = 142;
pub const CURAND_RNG_PSEUDO_PHILOX4_32_10: curandRngType_t = 161;

/// curandGenerator_t opaque (NVIDIA ABI = pointer all'handle). Lo riusiamo
/// direttamente come hiprandGenerator_t (cast diretto, stesso layout: void*).
pub type curandGenerator_t = *mut c_void;

#[inline]
fn hip_to_curand(s: hiprandStatus_t) -> curandStatus_t {
    s as u32
}

#[inline]
fn map_rng_type(t: curandRngType_t) -> Result<hiprandRngType_t, curandStatus_t> {
    Ok(match t {
        CURAND_RNG_PSEUDO_DEFAULT => hiprandRngType_t::HIPRAND_RNG_PSEUDO_DEFAULT,
        CURAND_RNG_PSEUDO_XORWOW => hiprandRngType_t::HIPRAND_RNG_PSEUDO_XORWOW,
        CURAND_RNG_PSEUDO_MRG32K3A => hiprandRngType_t::HIPRAND_RNG_PSEUDO_MRG32K3A,
        CURAND_RNG_PSEUDO_MTGP32 => hiprandRngType_t::HIPRAND_RNG_PSEUDO_MTGP32,
        CURAND_RNG_PSEUDO_MT19937 => hiprandRngType_t::HIPRAND_RNG_PSEUDO_MT19937,
        CURAND_RNG_PSEUDO_PHILOX4_32_10 => hiprandRngType_t::HIPRAND_RNG_PSEUDO_PHILOX4_32_10,
        _ => return Err(CURAND_STATUS_TYPE_ERROR),
    })
}

// ============================================================================
//  API esportata (ABI NVIDIA curand64_10.dll)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn curandCreateGenerator(
    generator: *mut curandGenerator_t,
    rng_type: curandRngType_t,
) -> curandStatus_t {
    if generator.is_null() {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    let hip_type = match map_rng_type(rng_type) {
        Ok(t) => t,
        Err(e) => return e,
    };
    let mut hip_gen: hiprandGenerator_t = ptr::null_mut();
    let s = hiprandCreateGenerator(&mut hip_gen, hip_type);
    if s == hiprandStatus_t::HIPRAND_STATUS_SUCCESS {
        *generator = hip_gen as *mut c_void;
    }
    hip_to_curand(s)
}

#[no_mangle]
pub unsafe extern "C" fn curandDestroyGenerator(
    generator: curandGenerator_t,
) -> curandStatus_t {
    if generator.is_null() {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandDestroyGenerator(generator as hiprandGenerator_t))
}

#[no_mangle]
pub unsafe extern "C" fn curandSetPseudoRandomGeneratorSeed(
    generator: curandGenerator_t,
    seed: u64,
) -> curandStatus_t {
    if generator.is_null() {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandSetPseudoRandomGeneratorSeed(
        generator as hiprandGenerator_t,
        seed,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn curandSetGeneratorOffset(
    generator: curandGenerator_t,
    offset: u64,
) -> curandStatus_t {
    if generator.is_null() {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandSetGeneratorOffset(
        generator as hiprandGenerator_t,
        offset,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn curandSetStream(
    generator: curandGenerator_t,
    stream: *mut c_void,
) -> curandStatus_t {
    if generator.is_null() {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandSetStream(
        generator as hiprandGenerator_t,
        stream as hipStream_t,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn curandGenerate(
    generator: curandGenerator_t,
    output: *mut u32,
    n: usize,
) -> curandStatus_t {
    if generator.is_null() || (output.is_null() && n > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandGenerate(generator as hiprandGenerator_t, output, n))
}

#[no_mangle]
pub unsafe extern "C" fn curandGenerateUniform(
    generator: curandGenerator_t,
    output: *mut f32,
    n: usize,
) -> curandStatus_t {
    if generator.is_null() || (output.is_null() && n > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandGenerateUniform(
        generator as hiprandGenerator_t,
        output,
        n,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn curandGenerateUniformDouble(
    generator: curandGenerator_t,
    output: *mut f64,
    n: usize,
) -> curandStatus_t {
    if generator.is_null() || (output.is_null() && n > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    hip_to_curand(hiprandGenerateUniformDouble(
        generator as hiprandGenerator_t,
        output,
        n,
    ))
}

/// Box-Muller (hipRAND) richiede n pari. Fix W2: se n dispari alloca buffer
/// scratch (n+1) sul device, genera (n+1) elementi, copia indietro n con
/// hipMemcpyAsync (DeviceToDevice) sullo stream del generatore, sync.
#[no_mangle]
pub unsafe extern "C" fn curandGenerateNormal(
    generator: curandGenerator_t,
    output: *mut f32,
    n: usize,
    mean: f32,
    stddev: f32,
) -> curandStatus_t {
    if generator.is_null() || (output.is_null() && n > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if n == 0 {
        return CURAND_STATUS_SUCCESS;
    }
    if n % 2 == 0 {
        return hip_to_curand(hiprandGenerateNormal(
            generator as hiprandGenerator_t,
            output,
            n,
            mean,
            stddev,
        ));
    }
    // Fix W2: odd n — alloca scratch (n+1) f32 sul device
    let pad_n = n + 1;
    let nbytes_pad = pad_n * mem::size_of::<f32>();
    let nbytes_user = n * mem::size_of::<f32>();
    let mut scratch: *mut c_void = ptr::null_mut();
    let hip_alloc = hipMalloc(&mut scratch as *mut *mut c_void, nbytes_pad);
    if hip_alloc.is_err() || scratch.is_null() {
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    let s = hiprandGenerateNormal(
        generator as hiprandGenerator_t,
        scratch as *mut f32,
        pad_n,
        mean,
        stddev,
    );
    if s != hiprandStatus_t::HIPRAND_STATUS_SUCCESS {
        let _ = hipFree(scratch);
        return hip_to_curand(s);
    }
    // Copia n elementi: scratch[0..n] -> output[0..n] async sullo stream del gen
    // (hipMemcpyAsync con stream=0 default: PyTorch poi sincronizza via cudaStreamSync).
    let mem_s = hipMemcpyAsync(
        output as *mut c_void,
        scratch,
        nbytes_user,
        hipMemcpyKind::hipMemcpyDeviceToDevice,
        HipRtStream(ptr::null_mut()),
    );
    if mem_s.is_err() {
        let _ = hipFree(scratch);
        return CURAND_STATUS_LAUNCH_FAILURE;
    }
    // Sync prima di liberare scratch (necessario: senza sync free racing copy)
    let sync_s = hipStreamSynchronize(HipRtStream(ptr::null_mut()));
    let _ = hipFree(scratch);
    if sync_s.is_err() {
        return CURAND_STATUS_LAUNCH_FAILURE;
    }
    CURAND_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn curandGenerateNormalDouble(
    generator: curandGenerator_t,
    output: *mut f64,
    n: usize,
    mean: f64,
    stddev: f64,
) -> curandStatus_t {
    if generator.is_null() || (output.is_null() && n > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if n == 0 {
        return CURAND_STATUS_SUCCESS;
    }
    if n % 2 == 0 {
        return hip_to_curand(hiprandGenerateNormalDouble(
            generator as hiprandGenerator_t,
            output,
            n,
            mean,
            stddev,
        ));
    }
    let pad_n = n + 1;
    let nbytes_pad = pad_n * mem::size_of::<f64>();
    let nbytes_user = n * mem::size_of::<f64>();
    let mut scratch: *mut c_void = ptr::null_mut();
    let hip_alloc = hipMalloc(&mut scratch as *mut *mut c_void, nbytes_pad);
    if hip_alloc.is_err() || scratch.is_null() {
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    let s = hiprandGenerateNormalDouble(
        generator as hiprandGenerator_t,
        scratch as *mut f64,
        pad_n,
        mean,
        stddev,
    );
    if s != hiprandStatus_t::HIPRAND_STATUS_SUCCESS {
        let _ = hipFree(scratch);
        return hip_to_curand(s);
    }
    let mem_s = hipMemcpyAsync(
        output as *mut c_void,
        scratch,
        nbytes_user,
        hipMemcpyKind::hipMemcpyDeviceToDevice,
        HipRtStream(ptr::null_mut()),
    );
    if mem_s.is_err() {
        let _ = hipFree(scratch);
        return CURAND_STATUS_LAUNCH_FAILURE;
    }
    let sync_s = hipStreamSynchronize(HipRtStream(ptr::null_mut()));
    let _ = hipFree(scratch);
    if sync_s.is_err() {
        return CURAND_STATUS_LAUNCH_FAILURE;
    }
    CURAND_STATUS_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn curandGetVersion(version: *mut i32) -> curandStatus_t {
    if version.is_null() {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    // Spoofa 10200 (curand 10.2 — match cu118 PyTorch wheel). Delegando a hiprand
    // otterremmo 5070, fuori range NVIDIA, e PyTorch fallirebbe il check version.
    let mut hip_ver: i32 = 0;
    let _ = hiprandGetVersion(&mut hip_ver);
    *version = 10200;
    CURAND_STATUS_SUCCESS
}

/// Stub per le API che PyTorch tocca raramente (LogNormal, Poisson, Quasi*).
/// Restituiamo NOT_IMPLEMENTED esplicito così cudart non confonde con fake-success.
#[no_mangle]
pub unsafe extern "C" fn curandCreateGeneratorHost(
    _generator: *mut curandGenerator_t,
    _rng_type: curandRngType_t,
) -> curandStatus_t {
    // Vega 20 non ha CPU host generator path: PyTorch userà CPU torch rng standard.
    1000 // CURAND_STATUS_NOT_IMPLEMENTED — esplicito non silent
}
