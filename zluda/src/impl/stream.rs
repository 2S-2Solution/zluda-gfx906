use hip_runtime_sys::*;

pub(crate) fn synchronize(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamSynchronize(stream) }
}

pub(crate) fn create(stream: *mut hipStream_t, flags: ::core::ffi::c_uint) -> hipError_t {
    unsafe { hipStreamCreateWithFlags(stream, flags) }
}

pub(crate) fn create_with_priority(
    stream: *mut hipStream_t,
    flags: ::core::ffi::c_uint,
    priority: ::core::ffi::c_int,
) -> hipError_t {
    unsafe { hipStreamCreateWithPriority(stream, flags, priority) }
}

pub(crate) fn destroy_v2(stream: hipStream_t) -> hipError_t {
    unsafe { hipStreamDestroy(stream) }
}

pub(crate) fn begin_capture_v2(stream: hipStream_t, mode: hipStreamCaptureMode) -> hipError_t {
    unsafe { hipStreamBeginCapture(stream, mode) }
}

pub(crate) fn end_capture(stream: hipStream_t, graph: *mut hipGraph_t) -> hipError_t {
    unsafe { hipStreamEndCapture(stream, graph) }
}

pub(crate) fn is_capturing(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
) -> hipError_t {
    unsafe { hipStreamIsCapturing(stream, capture_status) }
}

pub(crate) fn get_capture_info_v2(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-9: cuStreamGetCaptureInfo_v3 forwarder
// CUDA 12+ API: estende v2 con edge_data_out (per dipendenze tipizzate del grafo).
// HIP non espone _v3 nativo: forwardiamo a _v2 e azzeriamo edge_data_out (best-effort,
// torch usa edge_data solo per graph capture avanzato non supportato su gfx906).
pub(crate) fn get_capture_info_v3(
    stream: hipStream_t,
    capture_status: *mut hipStreamCaptureStatus,
    id: *mut ::core::ffi::c_ulonglong,
    graph_out: *mut hipGraph_t,
    dependencies_out: *mut *const hipGraphNode_t,
    edge_data_out: *mut *const u64,
    num_dependencies_out: *mut usize,
) -> hipError_t {
    if !edge_data_out.is_null() {
        unsafe { *edge_data_out = ::core::ptr::null() };
    }
    unsafe {
        hipStreamGetCaptureInfo_v2(
            stream,
            capture_status,
            id,
            graph_out,
            dependencies_out,
            num_dependencies_out,
        )
    }
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-9: cuStreamUpdateCaptureDependencies forwarder
// CUDA 11.3+ API: aggiorna lista dipendenze durante graph capture in corso.
// HIP espone hipStreamUpdateCaptureDependencies con stessa signature (deps, num, flags).
pub(crate) fn update_capture_dependencies(
    stream: hipStream_t,
    dependencies: *mut hipGraphNode_t,
    num_dependencies: usize,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    unsafe {
        hipStreamUpdateCaptureDependencies(stream, dependencies, num_dependencies, flags)
    }
}

/// Stream BLOCCANTI di servizio, uno per dispositivo, su cui far passare le attese
/// destinate allo stream nullo. Creati alla prima necessita' e mai distrutti.
static PONTI: std::sync::Mutex<Vec<(i32, hipStream_t)>> = std::sync::Mutex::new(Vec::new());

fn ponte_dispositivo_corrente() -> Result<hipStream_t, hipErrorCode_t> {
    let mut dispositivo: ::core::ffi::c_int = 0;
    unsafe { hipGetDevice(&mut dispositivo) }?;
    let mut ponti = PONTI.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = ponti.iter().find(|(d, _)| *d == dispositivo) {
        return Ok(*s);
    }
    let mut s = hipStream_t(std::ptr::null_mut());
    // flag 0 = hipStreamDefault: stream BLOCCANTE. E' questo che lo lega allo stream
    // nullo: per specifica HIP il nullo aspetta gli stream bloccanti, e viceversa.
    unsafe { hipStreamCreateWithFlags(&mut s, 0) }?;
    ponti.push((dispositivo, s));
    Ok(s)
}

pub(crate) fn wait_event(
    stream: hipStream_t,
    event: hipEvent_t,
    flags: ::core::ffi::c_uint,
) -> hipError_t {
    // [VEGA-PATCH stream-nullo, 17 ott 2026]
    //
    // 1. Il runtime HIP 5.7.31541 del driver rifiuta hipStreamWaitEvent quando chi
    //    aspetta e' lo stream NULLO (hipErrorInvalidValue): entra nel ramo di cattura
    //    dei grafi senza alcuna cattura in corso (hip_stream.cpp:516, build 23.19).
    //    Gia' corretto nel CLR attuale, ma nessun runtime piu' recente vede gfx906.
    //    Rimedio: l'attesa si mette su uno stream BLOCCANTE di servizio, che lo stream
    //    nullo aspetta per specifica. Nessun blocco della CPU.
    //    Vale anche per CU_STREAM_LEGACY (0x1), che zluda_common ora traduce nel nullo.
    //
    // 2. Prima qui si scartava OGNI esito ("let _ = ...; Ok(())", VEGA-FIX 23 giu),
    //    per un hipErrorNotInitialized(3) all'avvio di torch 2.4
    //    (at::cuda::getDeviceProperties). Cosi' pero' un'attesa fallita veniva
    //    dichiarata riuscita e non accodata. Ora si scarta SOLO quell'errore.
    let esito = if stream.0.is_null() {
        match ponte_dispositivo_corrente() {
            Ok(ponte) => unsafe { hipStreamWaitEvent(ponte, event, flags) },
            Err(e) => Err(e),
        }
    } else {
        unsafe { hipStreamWaitEvent(stream, event, flags) }
    };
    match esito {
        Err(e) if e == hipErrorCode_t::NotInitialized => Ok(()),
        altro => altro,
    }
}
