// zluda_nccl — stub graceful per nccl64_2.dll
// VEGA-PATCH 2026-06-30 T15:50 (Vega ⭐, pattern bottega italiana sovrana)
//
// Obiettivo: superare il LoadLibraryW/GetProcAddress feature-detect che
// PyTorch (torch.distributed), TensorFlow, JAX, DeepSpeed e Megatron fanno
// in early-init anche su singola GPU. Senza questo stub la DLL manca e
// alcuni code path crashano su GetProcAddress = NULL.
//
// Strategia ritorni:
//   - Query no-op (Version, GetUniqueId, CommCount, ...) → SUCCESS (0)
//   - Init reale comm singola → SUCCESS con handle dummy stabile
//   - Op collective reali (AllReduce, Broadcast, ...) → ncclInternalError (3)
//     così PyTorch DDP rileva "no NCCL" e fa fallback CPU/skip
//
// NOTA: nessuna dipendenza esterna. Zero macro CUDA. Build leggera (~50 KB).

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]

use core::ffi::{c_char, c_int, c_void};

// =========================================================================
// Types ABI-compatibili con nccl.h (NCCL 2.x)
// =========================================================================

/// ncclResult_t — enum int.
/// 0=SUCCESS, 1=UnhandledCudaError, 2=SystemError, 3=InternalError,
/// 4=InvalidArgument, 5=InvalidUsage, 6=RemoteError, 7=InProgress.
pub type ncclResult_t = c_int;

const NCCL_SUCCESS: ncclResult_t = 0;
const NCCL_INTERNAL_ERROR: ncclResult_t = 3;
const NCCL_INVALID_ARGUMENT: ncclResult_t = 4;

/// ncclComm_t — opaque pointer.
pub type ncclComm_t = *mut c_void;

/// ncclUniqueId — 128 byte fissi (struct { char internal[128]; }).
#[repr(C)]
pub struct ncclUniqueId {
    pub internal: [c_char; 128],
}

/// NCCL_VERSION_CODE = (MAJOR*1000) + (MINOR*100) + PATCH.
/// Spacciamo 2.18.5 = 21805 (ultimo NCCL 2.x stabile noto a app PyTorch 2.4).
const FAKE_VERSION: c_int = 21805;

/// Handle dummy stabile per ncclComm_t — usiamo un puntatore a static dummy.
static DUMMY_COMM_SLOT: c_int = 0xDEAD_BEEFu32 as i32;

// =========================================================================
// Query no-op — ritornano SUCCESS con valori plausibili
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn ncclGetVersion(version: *mut c_int) -> ncclResult_t {
    if version.is_null() { return NCCL_INVALID_ARGUMENT; }
    *version = FAKE_VERSION;
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn pncclGetVersion(version: *mut c_int) -> ncclResult_t {
    ncclGetVersion(version)
}

#[no_mangle]
pub unsafe extern "C" fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t {
    if uniqueId.is_null() { return NCCL_INVALID_ARGUMENT; }
    // Riempi con pattern deterministico (no entropy ma deterministico).
    let bytes = uniqueId as *mut u8;
    for i in 0..128 {
        *bytes.add(i) = (i as u8).wrapping_mul(7);
    }
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn pncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t {
    ncclGetUniqueId(uniqueId)
}

#[no_mangle]
pub unsafe extern "C" fn ncclGetErrorString(result: ncclResult_t) -> *const c_char {
    static MSG_SUCCESS: &[u8] = b"no error (zluda stub)\0";
    static MSG_INTERNAL: &[u8] = b"zluda-nccl: collective ops not supported on single-GPU stub\0";
    static MSG_UNKNOWN: &[u8] = b"zluda-nccl: unknown error code\0";
    let msg: &[u8] = match result {
        0 => MSG_SUCCESS,
        3 => MSG_INTERNAL,
        _ => MSG_UNKNOWN,
    };
    msg.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn ncclGetLastError(_comm: ncclComm_t) -> *const c_char {
    static MSG_NONE: &[u8] = b"\0";
    MSG_NONE.as_ptr() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommGetAsyncError(
    _comm: ncclComm_t,
    asyncError: *mut ncclResult_t,
) -> ncclResult_t {
    if !asyncError.is_null() {
        *asyncError = NCCL_SUCCESS;
    }
    NCCL_SUCCESS
}

// =========================================================================
// Comm init — produce handle dummy stabile per singola GPU
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn ncclCommInitAll(
    comms: *mut ncclComm_t,
    ndev: c_int,
    _devlist: *const c_int,
) -> ncclResult_t {
    if comms.is_null() || ndev <= 0 { return NCCL_INVALID_ARGUMENT; }
    for i in 0..ndev as usize {
        // Tutti i comm puntano al dummy slot; mai dereferenziato nei nostri stub.
        *comms.add(i) = &DUMMY_COMM_SLOT as *const _ as *mut c_void;
    }
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommInitRank(
    comm: *mut ncclComm_t,
    _nranks: c_int,
    _commId: ncclUniqueId,
    _rank: c_int,
) -> ncclResult_t {
    if comm.is_null() { return NCCL_INVALID_ARGUMENT; }
    *comm = &DUMMY_COMM_SLOT as *const _ as *mut c_void;
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommInitRankConfig(
    comm: *mut ncclComm_t,
    _nranks: c_int,
    _commId: ncclUniqueId,
    _rank: c_int,
    _config: *mut c_void,
) -> ncclResult_t {
    ncclCommInitRank(comm, _nranks, _commId, _rank)
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommDestroy(_comm: ncclComm_t) -> ncclResult_t {
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommAbort(_comm: ncclComm_t) -> ncclResult_t {
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommFinalize(_comm: ncclComm_t) -> ncclResult_t {
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommCount(_comm: ncclComm_t, count: *mut c_int) -> ncclResult_t {
    if !count.is_null() { *count = 1; }
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommCuDevice(_comm: ncclComm_t, device: *mut c_int) -> ncclResult_t {
    if !device.is_null() { *device = 0; }
    NCCL_SUCCESS
}

#[no_mangle]
pub unsafe extern "C" fn ncclCommUserRank(_comm: ncclComm_t, rank: *mut c_int) -> ncclResult_t {
    if !rank.is_null() { *rank = 0; }
    NCCL_SUCCESS
}

// =========================================================================
// Op collective reali — ritornano InternalError per innescare fallback
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn ncclAllReduce(
    _sendbuff: *const c_void, _recvbuff: *mut c_void,
    _count: usize, _datatype: c_int, _op: c_int,
    _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclBroadcast(
    _sendbuff: *const c_void, _recvbuff: *mut c_void,
    _count: usize, _datatype: c_int, _root: c_int,
    _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclBcast(
    _buff: *mut c_void, _count: usize, _datatype: c_int,
    _root: c_int, _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclReduce(
    _sendbuff: *const c_void, _recvbuff: *mut c_void,
    _count: usize, _datatype: c_int, _op: c_int, _root: c_int,
    _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclAllGather(
    _sendbuff: *const c_void, _recvbuff: *mut c_void,
    _sendcount: usize, _datatype: c_int,
    _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclReduceScatter(
    _sendbuff: *const c_void, _recvbuff: *mut c_void,
    _recvcount: usize, _datatype: c_int, _op: c_int,
    _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclSend(
    _sendbuff: *const c_void, _count: usize, _datatype: c_int,
    _peer: c_int, _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclRecv(
    _recvbuff: *mut c_void, _count: usize, _datatype: c_int,
    _peer: c_int, _comm: ncclComm_t, _stream: *mut c_void,
) -> ncclResult_t { NCCL_INTERNAL_ERROR }

#[no_mangle]
pub unsafe extern "C" fn ncclGroupStart() -> ncclResult_t { NCCL_SUCCESS }

#[no_mangle]
pub unsafe extern "C" fn ncclGroupEnd() -> ncclResult_t { NCCL_SUCCESS }

// =========================================================================
// Memory allocator hooks (NCCL 2.19+) — no-op fallback delega CUDA caller
// =========================================================================

#[no_mangle]
pub unsafe extern "C" fn ncclMemAlloc(_ptr: *mut *mut c_void, _size: usize) -> ncclResult_t {
    NCCL_INTERNAL_ERROR
}

#[no_mangle]
pub unsafe extern "C" fn ncclMemFree(_ptr: *mut c_void) -> ncclResult_t {
    NCCL_SUCCESS
}
