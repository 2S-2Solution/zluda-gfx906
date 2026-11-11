use cuda_types::cudnn9::*;
use hip_runtime_sys::*;
use miopen_sys::*;
use rustc_hash::FxHashMap;
use std::{collections::VecDeque, mem, ptr, sync::Mutex};
use zluda_common::{from_cuda_object, ZludaObject};

pub(crate) struct Context {
    base: miopenHandle_t,
    search_workspace: Mutex<ContextCache>,
}

from_cuda_object!(Context);

impl Context {
    fn new(base: miopenHandle_t) -> Self {
        Self {
            base,
            search_workspace: Mutex::new(ContextCache {
                beta_buffer_cache: TemporaryBufferAllocator::new(),
                algo_cache: FxHashMap::default(),
                algo_cache_bwd_dati: FxHashMap::default(),
                algo_cache_bwd_pesi: FxHashMap::default(),
                pesi_rnn: None,
            }),
        }
    }
}

impl Context {
    /// Restituisce il buffer d'appoggio per i pesi RNN, allargandolo se serve.
    unsafe fn buffer_pesi_rnn(
        &self,
        byte: usize,
    ) -> Result<*mut ::core::ffi::c_void, cudnnError_t> {
        let mut g = self
            .search_workspace
            .lock()
            .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
        if let Some((punt, quanti)) = g.pesi_rnn {
            if quanti >= byte {
                return Ok(punt);
            }
            hipFree(punt).ok();
            g.pesi_rnn = None;
        }
        let mut punt: *mut ::core::ffi::c_void = ptr::null_mut();
        hipMalloc(&mut punt, byte).map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
        g.pesi_rnn = Some((punt, byte));
        Ok(punt)
    }
}

impl ZludaObject for Context {
    const COOKIE: usize = 0x575f48767c76029a;

    type Error = cudnnError_t;
    type CudaHandle = cudnnHandle_t;

    fn drop_checked(&mut self) -> Result<(), cudnnError_t> {
        let result1 = unsafe { miopenDestroy(self.base) }.map_err(Into::into);
        let result2 = if let Ok(search_workspace) = self.search_workspace.get_mut() {
            search_workspace
                .drop_checked()
                .map_err(|_| cudnnError_t::INTERNAL_ERROR)
        } else {
            Ok(())
        };
        result1.and(result2)
    }
}

struct ContextCache {
    beta_buffer_cache: TemporaryBufferAllocator,
    algo_cache: FxHashMap<ConvolutionOpCacheKey, miopenConvAlgoPerf_t>,
    // 2 ago 2026 — la ricerca MIOpen all'indietro costava ~230 ms per passo su
    // una rete minuscola, perche' rigirava a ogni chiamata. La prima volta va
    // fatta (e' lei a registrare l'invoker), dalle successive si riusa.
    algo_cache_bwd_dati: FxHashMap<ConvolutionOpCacheKey, miopenConvBwdDataAlgorithm_t>,
    algo_cache_bwd_pesi: FxHashMap<ConvolutionOpCacheKey, miopenConvBwdWeightsAlgorithm_t>,
    /// Buffer d'appoggio per riordinare i pesi delle RNN fra la disposizione di
    /// cuDNN e quella di MIOpen. Cresce quando serve e vive quanto il contesto.
    pesi_rnn: Option<(*mut ::core::ffi::c_void, usize)>,
}

impl ContextCache {
    fn drop_checked(&mut self) -> Result<(), hipErrorCode_t> {
        let mut err = None;
        while let Some(mut buffer) = self.beta_buffer_cache.buffers.pop_front() {
            if let Err(e) = buffer.drop_checked() {
                err = Some(e);
            }
        }
        if let Some((punt, _)) = self.pesi_rnn.take() {
            if let Err(e) = unsafe { hipFree(punt) } {
                err = Some(e);
            }
        }
        err.map_or(Ok(()), Err)
    }
}

unsafe impl Send for ContextCache {}
unsafe impl Sync for ContextCache {}

// At various point of the execution we need temporary buffers to hold device
// data to support various MIOpen workarounds.
// Normally, their use does not overlap, but it's still possible.
struct TemporaryBufferAllocator {
    buffers: VecDeque<BetaBuffer>,
}

impl TemporaryBufferAllocator {
    fn new() -> Self {
        Self {
            buffers: VecDeque::new(),
        }
    }

    fn get_or_allocate(&mut self, size: usize) -> Result<&BetaBuffer, hipErrorCode_t> {
        let mut buffer = self.scavange_or_allocate_buffer(size)?;
        if let Some(event) = buffer.free.take() {
            unsafe { hipEventDestroy(event) }.ok();
        }
        self.buffers.push_front(buffer);
        Ok(&self.buffers.front().unwrap())
    }

    fn scavange_or_allocate_buffer(&mut self, size: usize) -> Result<BetaBuffer, hipErrorCode_t> {
        let mut result = None;
        let mut err = None;
        let mut update_buffer_if_biggest = |buffer: BetaBuffer| {
            if result
                .as_ref()
                .map(|b: &BetaBuffer| buffer.size > b.size)
                .unwrap_or(true)
            {
                result = Some(buffer);
            }
        };
        while let Some(buffer) = self.buffers.pop_front() {
            match buffer.free {
                None => {
                    update_buffer_if_biggest(buffer);
                }
                Some(event) => match unsafe { hipEventQuery(event) } {
                    hipError_t::ErrorNotReady => {
                        self.buffers.push_front(buffer);
                        break;
                    }
                    hipError_t::Success => {
                        update_buffer_if_biggest(buffer);
                    }
                    Err(err_code) => {
                        err = Some(err_code);
                        break;
                    }
                },
            }
        }
        match err {
            Some(err_code) => Err(err_code),
            None => match result {
                Some(buffer) if buffer.size >= size => Ok(buffer),
                _ => Ok(BetaBuffer::new(size)?),
            },
        }
    }

    fn with_async_buffer(
        &mut self,
        size: usize,
        stream: hipStream_t,
        fn_: impl FnOnce(*mut ::std::os::raw::c_void) -> Result<(), miopenError_t>,
    ) -> Result<(), miopenError_t> {
        let mut buffer = self
            .scavange_or_allocate_buffer(size)
            .map_err(|_| miopenError_t::InternalError)?;
        if buffer.free.is_none() {
            let mut event = std::ptr::null_mut();
            unsafe {
                hipEventCreateWithFlags(
                    &mut event,
                    hipEventDisableTiming | hipEventDisableSystemFence,
                )
                .map_err(|_| miopenError_t::InternalError)?
            };
            buffer.free = Some(event);
        }
        fn_(buffer.data)?;
        unsafe { hipEventRecord(buffer.free.unwrap(), stream) }
            .map_err(|_| miopenError_t::InternalError)?;
        self.buffers.push_back(buffer);
        Ok(())
    }
}

struct BetaBuffer {
    size: usize,
    data: *mut ::std::os::raw::c_void,
    free: Option<hipEvent_t>,
}

impl BetaBuffer {
    fn new(size: usize) -> Result<Self, hipErrorCode_t> {
        let mut data = std::ptr::null_mut();
        unsafe { hipMalloc(&mut data, size)? };
        Ok(Self {
            size,
            data,
            free: None,
        })
    }

    fn drop_checked(&mut self) -> Result<(), hipErrorCode_t> {
        let result1 = unsafe { hipFree(self.data) };
        let result2 = self
            .free
            .map(|e| unsafe { hipEventDestroy(e) })
            .unwrap_or(Ok(()));
        result1.and(result2)
    }
}

impl Drop for BetaBuffer {
    fn drop(&mut self) {
        self.drop_checked().ok();
    }
}

#[cfg(debug_assertions)]
pub(crate) fn unimplemented() -> cudnnStatus_t {
    unimplemented!()
}

#[cfg(not(debug_assertions))]
pub(crate) fn unimplemented() -> cudnnStatus_t {
    cudnnStatus_t::ERROR_NOT_SUPPORTED
}

// Interrogazioni di versione. Erano `todo!()`, e un panic che attraversa un
// confine `extern "C"` non si puo' srotolare: **aborta il processo**.
//
// PyTorch chiama `cudnnGetVersion()` prima ancora di creare l'handle, dentro
// `getCudnnHandle`. Il risultato, misurato il 1 ago 2026: QUALUNQUE convoluzione
// faceva morire il processo con «panic in a function that cannot unwind» — non
// un errore da gestire, proprio un abort.
//
// Qui si risponde con numeri coerenti con la piattaforma che stiamo emulando.
// Non serve che siano esatti: serve che siano plausibili e stabili, perche' chi
// li legge li usa per scegliere un percorso, non per calcolare.

/// Versione cuDNN riportata: 8.9.7. E' quella che accompagna le distribuzioni
/// CUDA 11.8 di PyTorch, quindi non fa scattare ne' i percorsi delle 9.x ne' gli
/// avvisi di versione troppo vecchia.
///
/// ATTENZIONE AL FORMATO. Le 8.x si codificano `major*1000 + minor*100 + patch`;
/// e' dalle 9.x che si passa a `major*10000 + minor*100 + patch`. Scrivendo per
/// sbaglio il formato nuovo su una 8.x viene fuori 80907, e PyTorch — che legge
/// col formato vecchio — si rifiuta di partire:
///   «cuDNN version incompatibility: compiled against (8,7,0), found (0,128,0)»
/// Misurato il 2 ago 2026.
const VERSIONE_CUDNN: usize = 8 * 1000 + 9 * 100 + 7;

/// Versione del runtime CUDA: 11.8, nel formato `major*1000 + minor*10`.
const VERSIONE_CUDART: usize = 11 * 1000 + 8 * 10;

pub(crate) fn get_version() -> usize {
    VERSIONE_CUDNN
}

pub(crate) fn get_max_device_version() -> usize {
    VERSIONE_CUDNN
}

pub(crate) fn get_cudart_version() -> usize {
    VERSIONE_CUDART
}

/// Major/minor/patch della cuDNN che diciamo di essere.
///
/// PyTorch NON usa `cudnnGetVersion` per il controllo di compatibilita' che fa
/// all'avvio: `at::native::getRuntimeVersion()` chiama tre volte
/// `cudnnGetProperty`. Finche' questa restava uno stub che non scriveva nulla,
/// PyTorch leggeva memoria non inizializzata e si fermava con
///   «cuDNN version incompatibility: compiled against (8,7,0), found (0,128,0)»
/// prima ancora di provare una convoluzione. Misurato il 2 ago 2026.
///
/// `libraryPropertyType` e' lo stesso tipo in cudnn8 e cudnn9 (entrambi
/// rimandano a `cuda::libraryPropertyType`), quindi un'unica impl serve i due
/// rami. 0 = MAJOR_VERSION, 1 = MINOR_VERSION, 2 = PATCH_LEVEL.
pub(crate) unsafe extern "C" fn cudnnGetProperty(
    type_: libraryPropertyType,
    value: *mut ::core::ffi::c_int,
) -> cudnnStatus_t {
    if value.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let v = match type_.0 {
        0 => (VERSIONE_CUDNN / 1000) as i32,
        1 => ((VERSIONE_CUDNN % 1000) / 100) as i32,
        2 => (VERSIONE_CUDNN % 100) as i32,
        _ => return Err(cudnnError_t::BAD_PARAM),
    };
    unsafe { *value = v };
    Ok(())
}

/// Scrive il messaggio dell'ultimo errore. Non teniamo uno stato per thread:
/// si restituisce una stringa vuota, che e' la risposta onesta.
pub(crate) fn get_last_error_string(message: *mut ::core::ffi::c_char, max_size: usize) -> () {
    if !message.is_null() && max_size > 0 {
        unsafe { *message = 0 };
    }
}

pub(crate) unsafe fn create(handle: &mut cudnnHandle_t) -> miopenStatus_t {
    let mut miopen_handle = mem::zeroed();
    miopenCreate(&mut miopen_handle)?;
    *handle = Context::new(miopen_handle).wrap();
    Ok(())
}

pub(crate) fn destroy(handle: cudnnHandle_t) -> cudnnStatus_t {
    zluda_common::drop_checked::<Context>(handle)?;
    Ok(())
}

pub(crate) fn create_tensor_descriptor(
    tensor_desc: &mut miopenTensorDescriptor_t,
) -> miopenStatus_t {
    unsafe { miopenCreateTensorDescriptor(tensor_desc) }
}

pub(crate) fn create_filter_descriptor(
    filter_desc: &mut miopenTensorDescriptor_t,
) -> miopenStatus_t {
    unsafe { miopenCreateTensorDescriptor(filter_desc) }
}

pub(crate) unsafe fn set_tensor4d_descriptor(
    tensor_desc: miopenTensorDescriptor_t,
    format: miopenTensorLayout_t,
    data_type: miopenDataType_t,
    n: ::std::os::raw::c_int,
    c: ::std::os::raw::c_int,
    h: ::std::os::raw::c_int,
    w: ::std::os::raw::c_int,
) -> miopenStatus_t {
    // Even if the layout is NHWC, miopenSetNdTensorDescriptorWithLayout still expects NCHW order
    let lens = [n, c, h, w];
    miopenSetNdTensorDescriptorWithLayout(
        tensor_desc,
        data_type,
        format,
        lens.as_ptr(),
        lens.len() as i32,
    )
}

pub(crate) unsafe fn set_filter4d_descriptor(
    filter_desc: miopenTensorDescriptor_t,
    data_type: miopenDataType_t,
    format: miopenTensorLayout_t,
    n: ::std::os::raw::c_int,
    c: ::std::os::raw::c_int,
    h: ::std::os::raw::c_int,
    w: ::std::os::raw::c_int,
) -> miopenStatus_t {
    set_tensor4d_descriptor(filter_desc, format, data_type, n, c, h, w)
}

pub(crate) fn create_convolution_descriptor(
    conv_desc: &mut miopenConvolutionDescriptor_t,
) -> miopenStatus_t {
    unsafe { miopenCreateConvolutionDescriptor(conv_desc) }
}

pub(crate) unsafe fn set_convolution2d_descriptor(
    conv_desc: miopenConvolutionDescriptor_t,
    pad_h: ::std::os::raw::c_int,
    pad_w: ::std::os::raw::c_int,
    u: ::std::os::raw::c_int,
    v: ::std::os::raw::c_int,
    dilation_h: ::std::os::raw::c_int,
    dilation_w: ::std::os::raw::c_int,
    mode: miopenConvolutionMode_t,
    _compute_type: miopenDataType_t,
) -> miopenStatus_t {
    miopenInitConvolutionDescriptor(conv_desc, mode, pad_h, pad_w, u, v, dilation_h, dilation_w)
}

pub(crate) unsafe fn set_convolution_math_type(
    _conv_desc: miopenConvolutionDescriptor_t,
    _math_type: cudnnMathType_t,
) -> miopenStatus_t {
    // miopen does not have an equivalent function to set math type
    miopenStatus_t::Success
}

// ---------------------------------------------------------------------------
// La famiglia Nd — 2 ago 2026
//
// PyTorch non chiama MAI le varianti 2d/4d qui sopra: `at::native::TensorDescriptor`
// e `FilterDescriptor` usano sempre `cudnnSetTensorNdDescriptorEx` /
// `cudnnSetFilterNdDescriptor`, e `ConvolutionDescriptor` usa
// `cudnnSetConvolutionNdDescriptor` + `cudnnSetConvolutionGroupCount`. Finche'
// queste erano stub, qualunque convoluzione moriva prima di arrivare a MIOpen —
// registro MIOpen a livello 6 durante una conv di torch: zero righe.
//
// Sono scritte in stile "Family B" (passaggio diretto, niente FromCuda
// automatica sugli argomenti) perche' le liste di dimensioni sono array: la
// conversione automatica le tratterebbe come riferimenti a un singolo intero.
//
// Le controparti MIOpen esistono tutte e la corrispondenza e' 1:1.
// ---------------------------------------------------------------------------

/// Converte un tipo di dato cuDNN in quello MIOpen riusando la tabella che sta
/// gia' in zluda_common, invece di riscriverla qui e farla divergere.
unsafe fn tipo_dato(t: ct9::cudnnDataType_t) -> Result<miopenDataType_t, cudnnError_t> {
    zluda_common::FromCuda::<_, cudnnError_t>::from_cuda(&t)
}

unsafe fn disposizione(f: ct9::cudnnTensorFormat_t) -> Result<miopenTensorLayout_t, cudnnError_t> {
    zluda_common::FromCuda::<_, cudnnError_t>::from_cuda(&f)
}

unsafe fn modo_conv(m: ct9::cudnnConvolutionMode_t) -> Result<miopenConvolutionMode_t, cudnnError_t> {
    zluda_common::FromCuda::<_, cudnnError_t>::from_cuda(&m)
}

/// PyTorch chiama `cudnnSetStream` subito dopo aver creato l'handle, dentro
/// `getCudnnHandle()`, per legarlo allo stream corrente. Finche' era uno stub
/// che restituiva errore, `AT_CUDNN_CHECK` faceva fallire **qualunque**
/// operazione cuDNN prima ancora che toccasse MIOpen: nel registro di MIOpen si
/// vedeva l'handle creato e poi il silenzio.
///
/// `cudaStream_t` di cuDNN e `miopenAcceleratorQueue_t` sono entrambi
/// `hipStream_t` sotto ZLUDA: passaggio diretto.
pub(crate) unsafe extern "C" fn cudnnSetStream(
    handle: ct9::cudnnHandle_t,
    stream_id: ct9::cudaStream_t,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenSetStream(mh, mem::transmute(stream_id)))
}

pub(crate) unsafe extern "C" fn cudnnGetStream(
    handle: ct9::cudnnHandle_t,
    stream_id: *mut ct9::cudaStream_t,
) -> ct9::cudnnStatus_t {
    if stream_id.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenGetStream(mh, stream_id.cast()))
}

// ---------------------------------------------------------------------------
// Ricerca algoritmo e spazio di lavoro per il passaggio all'indietro
//
// Finche' queste erano stub, la conv in avanti passava e il backward faceva
// **cadere il processo** con access violation: PyTorch riceveva l'errore dalla
// ricerca, ripiegava sull'algoritmo predefinito, e poi chiamava
// `cudnnConvolutionBackwardData` con una dimensione di spazio di lavoro mai
// scritta — cioe' spazzatura presa dallo stack.
//
// Si riferisce sempre UN solo algoritmo, lo 0. Non e' pigrizia: le impl
// `cudnnConvolutionBackwardData` / `...Filter` piu' sotto ignorano comunque
// l'algoritmo richiesto e usano GEMM di MIOpen, quindi annunciarne altri
// sarebbe una bugia. Lo spazio di lavoro e' quello dichiarato da MIOpen, che e'
// il massimo fra i suoi solver: abbondante ma sempre sufficiente.
// ---------------------------------------------------------------------------

pub(crate) unsafe extern "C" fn cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: ct9::cudnnHandle_t,
    w_desc: ct9::cudnnFilterDescriptor_t,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    _algo: ct9::cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenConvolutionBackwardDataGetWorkSpaceSize(
        mh,
        mem::transmute(dy_desc),
        mem::transmute(w_desc),
        mem::transmute(conv_desc),
        mem::transmute(dx_desc),
        size_in_bytes,
    ))
}

pub(crate) unsafe extern "C" fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: ct9::cudnnHandle_t,
    filter_desc: ct9::cudnnFilterDescriptor_t,
    diff_desc: ct9::cudnnTensorDescriptor_t,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    grad_desc: ct9::cudnnTensorDescriptor_t,
    requested_algo_count: ::core::ffi::c_int,
    returned_algo_count: *mut ::core::ffi::c_int,
    perf_results: *mut ct9::cudnnConvolutionBwdDataAlgoPerf_t,
) -> ct9::cudnnStatus_t {
    if requested_algo_count <= 0 || returned_algo_count.is_null() || perf_results.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mut memoria: usize = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        filter_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &mut memoria,
    )?;
    *returned_algo_count = 1;
    *perf_results = ct9::cudnnConvolutionBwdDataAlgoPerf_t {
        algo: cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        status: Ok(()),
        time: -1.0,
        memory: memoria,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0, 0, 0],
    };
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle: ct9::cudnnHandle_t,
    x_desc: ct9::cudnnTensorDescriptor_t,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    grad_desc: ct9::cudnnFilterDescriptor_t,
    _algo: ct9::cudnnConvolutionBwdFilterAlgo_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        mh,
        mem::transmute(dy_desc),
        mem::transmute(x_desc),
        mem::transmute(conv_desc),
        mem::transmute(grad_desc),
        size_in_bytes,
    ))
}

pub(crate) unsafe extern "C" fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: ct9::cudnnHandle_t,
    src_desc: ct9::cudnnTensorDescriptor_t,
    diff_desc: ct9::cudnnTensorDescriptor_t,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    grad_desc: ct9::cudnnFilterDescriptor_t,
    requested_algo_count: ::core::ffi::c_int,
    returned_algo_count: *mut ::core::ffi::c_int,
    perf_results: *mut ct9::cudnnConvolutionBwdFilterAlgoPerf_t,
) -> ct9::cudnnStatus_t {
    if requested_algo_count <= 0 || returned_algo_count.is_null() || perf_results.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mut memoria: usize = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle,
        src_desc,
        diff_desc,
        conv_desc,
        grad_desc,
        cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        &mut memoria,
    )?;
    *returned_algo_count = 1;
    *perf_results = ct9::cudnnConvolutionBwdFilterAlgoPerf_t {
        algo: cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        status: Ok(()),
        time: -1.0,
        memory: memoria,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0, 0, 0],
    };
    Ok(())
}

// ===========================================================================
//  Lettura dei descrittori, attivazioni, pooling, conv+bias+attivazione
//  2 ago 2026 — prima tornata verso la copertura completa dei simboli che
//  PyTorch importa da cudnn64_8.dll.
// ===========================================================================

/// Il contrario della tabella in zluda_common: da tipo MIOpen a tipo cuDNN.
/// Serve a `cudnnGetTensorNdDescriptor` e `cudnnGetFilterNdDescriptor`, che
/// devono restituire al chiamante il tipo nella sua nomenclatura.
fn tipo_dato_indietro(t: miopenDataType_t) -> Result<ct9::cudnnDataType_t, cudnnError_t> {
    Ok(match t {
        miopenDataType_t::miopenHalf => ct9::cudnnDataType_t::CUDNN_DATA_HALF,
        miopenDataType_t::miopenFloat => ct9::cudnnDataType_t::CUDNN_DATA_FLOAT,
        miopenDataType_t::miopenInt32 => ct9::cudnnDataType_t::CUDNN_DATA_INT32,
        miopenDataType_t::miopenInt8 => ct9::cudnnDataType_t::CUDNN_DATA_INT8,
        miopenDataType_t::miopenBFloat16 => ct9::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        miopenDataType_t::miopenDouble => ct9::cudnnDataType_t::CUDNN_DATA_DOUBLE,
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    })
}

/// Legge dimensioni e passi di un descrittore. MIOpen non dice quante
/// dimensioni ha in un colpo solo: prima la lunghezza, poi il contenuto.
unsafe fn leggi_descrittore(
    desc: miopenTensorDescriptor_t,
    quante_chieste: ::core::ffi::c_int,
    dim_a: *mut ::core::ffi::c_int,
    stride_a: *mut ::core::ffi::c_int,
) -> Result<(miopenDataType_t, ::core::ffi::c_int), cudnnError_t> {
    let mut n: ::core::ffi::c_int = 0;
    miopen_to_cudnn_status(miopenGetTensorDescriptorSize(desc, &mut n))?;
    if n <= 0 || n as usize > 8 {
        return Err(cudnnError_t::INTERNAL_ERROR);
    }
    let mut tipo: miopenDataType_t = mem::zeroed();
    let mut dimensioni = [0i32; 8];
    let mut passi = [0i32; 8];
    miopen_to_cudnn_status(miopenGetTensorDescriptor(
        desc,
        &mut tipo,
        dimensioni.as_mut_ptr(),
        passi.as_mut_ptr(),
    ))?;
    // Si copia solo quanto il chiamante ha dichiarato di poter ricevere: cuDNN
    // tronca invece di sbordare.
    let da_copiare = n.min(quante_chieste) as usize;
    if !dim_a.is_null() {
        ptr::copy_nonoverlapping(dimensioni.as_ptr(), dim_a, da_copiare);
    }
    if !stride_a.is_null() {
        ptr::copy_nonoverlapping(passi.as_ptr(), stride_a, da_copiare);
    }
    Ok((tipo, n))
}

pub(crate) unsafe extern "C" fn cudnnGetTensorNdDescriptor(
    tensor_desc: ct9::cudnnTensorDescriptor_t,
    nb_dims_requested: ::core::ffi::c_int,
    data_type: *mut ct9::cudnnDataType_t,
    nb_dims: *mut ::core::ffi::c_int,
    dim_a: *mut ::core::ffi::c_int,
    stride_a: *mut ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if nb_dims_requested <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let (tipo, n) = leggi_descrittore(
        mem::transmute(tensor_desc),
        nb_dims_requested,
        dim_a,
        stride_a,
    )?;
    if !data_type.is_null() {
        *data_type = tipo_dato_indietro(tipo)?;
    }
    if !nb_dims.is_null() {
        *nb_dims = n;
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetFilterNdDescriptor(
    filter_desc: ct9::cudnnFilterDescriptor_t,
    nb_dims_requested: ::core::ffi::c_int,
    data_type: *mut ct9::cudnnDataType_t,
    format: *mut ct9::cudnnTensorFormat_t,
    nb_dims: *mut ::core::ffi::c_int,
    filter_dim_a: *mut ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if nb_dims_requested <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let (tipo, n) = leggi_descrittore(
        mem::transmute(filter_desc),
        nb_dims_requested,
        filter_dim_a,
        ptr::null_mut(),
    )?;
    if !data_type.is_null() {
        *data_type = tipo_dato_indietro(tipo)?;
    }
    if !nb_dims.is_null() {
        *nb_dims = n;
    }
    if !format.is_null() {
        // MIOpen non conserva la disposizione in modo interrogabile: i suoi
        // descrittori tengono dimensioni e passi, non un'etichetta di layout.
        // Si riferisce NCHW, che e' quella con cui PyTorch li imposta sempre su
        // questo percorso.
        *format = ct9::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    }
    Ok(())
}

// --- attivazioni: i descrittori (le operazioni erano gia' cablate) ---------

pub(crate) unsafe extern "C" fn cudnnCreateActivationDescriptor(
    activation_desc: *mut ct9::cudnnActivationDescriptor_t,
) -> ct9::cudnnStatus_t {
    if activation_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreateActivationDescriptor(activation_desc.cast()))
}

pub(crate) unsafe extern "C" fn cudnnDestroyActivationDescriptor(
    activation_desc: ct9::cudnnActivationDescriptor_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenDestroyActivationDescriptor(mem::transmute(
        activation_desc,
    )))
}

pub(crate) unsafe extern "C" fn cudnnSetActivationDescriptor(
    activation_desc: ct9::cudnnActivationDescriptor_t,
    mode: ct9::cudnnActivationMode_t,
    _relu_nan_opt: ct9::cudnnNanPropagation_t,
    coef: f64,
) -> ct9::cudnnStatus_t {
    // Si confrontano i valori numerici e non le costanti: i binding generati non
    // hanno un nome per tutte (mancano CLIPPED_RELU e IDENTITY lato cuDNN,
    // PASTHRU e LOGISTIC lato MIOpen), ma i numeri sono quelli degli header.
    //
    // MIOpen: 0 PASTHRU, 1 LOGISTIC, 2 TANH, 3 RELU, 4 SOFTRELU, 5 ABS,
    //         6 POWER, 7 CLIPPEDRELU, 8 LEAKYRELU, 9 ELU
    // cuDNN:  0 SIGMOID, 1 RELU, 2 TANH, 3 CLIPPED_RELU, 4 ELU, 5 IDENTITY,
    //         6 SWISH
    //
    // `coef` in cuDNN e' la soglia di CLIPPED_RELU o l'alpha di ELU; in MIOpen
    // e' l'alpha. Per TANH, MIOpen calcola beta*tanh(alpha*x): con alpha=beta=1
    // e' la tanh semplice, che e' quello che cuDNN intende.
    let (m, alfa) = match mode.0 {
        0 => (1u32, 1.0),  // SIGMOID      -> LOGISTIC
        1 => (3u32, 1.0),  // RELU         -> RELU
        2 => (2u32, 1.0),  // TANH         -> TANH
        3 => (7u32, coef), // CLIPPED_RELU -> CLIPPEDRELU, soglia = coef
        4 => (9u32, coef), // ELU          -> ELU, alpha = coef
        5 => (0u32, 1.0),  // IDENTITY     -> PASTHRU
        // SWISH (x*sigmoid(x)) non ha equivalente in MIOpen. Non si finge: si
        // dichiara, cosi' PyTorch usa il proprio kernel invece di ricevere
        // numeri sbagliati.
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    };
    miopen_to_cudnn_status(miopenSetActivationDescriptor(
        mem::transmute(activation_desc),
        mem::transmute(m),
        alfa,
        1.0,
        1.0,
    ))
}

// --- pooling: i descrittori ------------------------------------------------

pub(crate) unsafe extern "C" fn cudnnCreatePoolingDescriptor(
    pooling_desc: *mut ct9::cudnnPoolingDescriptor_t,
) -> ct9::cudnnStatus_t {
    if pooling_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreatePoolingDescriptor(pooling_desc.cast()))
}

pub(crate) unsafe extern "C" fn cudnnSetPooling2dDescriptor(
    pooling_desc: ct9::cudnnPoolingDescriptor_t,
    mode: ct9::cudnnPoolingMode_t,
    _maxpooling_nan_opt: ct9::cudnnNanPropagation_t,
    window_height: ::core::ffi::c_int,
    window_width: ::core::ffi::c_int,
    vertical_padding: ::core::ffi::c_int,
    horizontal_padding: ::core::ffi::c_int,
    vertical_stride: ::core::ffi::c_int,
    horizontal_stride: ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    // cuDNN:  0 MAX, 1 AVERAGE_COUNT_INCLUDE_PADDING,
    //         2 AVERAGE_COUNT_EXCLUDE_PADDING, 3 MAX_DETERMINISTIC
    // MIOpen: 0 Max, 1 Average (esclude il bordo), 2 AverageInclusive
    let m: u32 = match mode.0 {
        0 | 3 => 0,
        1 => 2,
        2 => 1,
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    };
    miopen_to_cudnn_status(miopenSet2dPoolingDescriptor(
        mem::transmute(pooling_desc),
        mem::transmute(m),
        window_height,
        window_width,
        vertical_padding,
        horizontal_padding,
        vertical_stride,
        horizontal_stride,
    ))
}

// --- convoluzione + polarizzazione + attivazione, fusa ---------------------

pub(crate) unsafe extern "C" fn cudnnConvolutionBiasActivationForward(
    handle: ct9::cudnnHandle_t,
    alpha1: *const ::core::ffi::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    _algo: ct9::cudnnConvolutionFwdAlgo_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
    alpha2: *const ::core::ffi::c_void,
    z_desc: ct9::cudnnTensorDescriptor_t,
    z: *const ::core::ffi::c_void,
    bias_desc: ct9::cudnnTensorDescriptor_t,
    bias: *const ::core::ffi::c_void,
    activation_desc: ct9::cudnnActivationDescriptor_t,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    // MIOpen ha la stessa operazione con la stessa identica firma: unico caso
    // finora. L'algoritmo lo sceglie MIOpen (0 = GEMM), come per le altre.
    let mh = ctx_handle(handle);
    let algo: miopenConvFwdAlgorithm_t = mem::transmute(0u32);
    miopen_to_cudnn_status(miopenConvolutionBiasActivationForward(
        mh,
        alpha1,
        mem::transmute(x_desc),
        x,
        mem::transmute(w_desc),
        w,
        mem::transmute(conv_desc),
        algo,
        workspace,
        workspace_size,
        alpha2,
        mem::transmute(z_desc),
        z,
        mem::transmute(bias_desc),
        bias,
        mem::transmute(activation_desc),
        mem::transmute(y_desc),
        y,
    ))
}

// ===========================================================================
//  Dropout e CTC loss — 2 ago 2026
//  MIOpen ha entrambe le famiglie complete; la corrispondenza e' quasi 1:1.
// ===========================================================================

pub(crate) unsafe extern "C" fn cudnnCreateDropoutDescriptor(
    dropout_desc: *mut ct9::cudnnDropoutDescriptor_t,
) -> ct9::cudnnStatus_t {
    if dropout_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreateDropoutDescriptor(dropout_desc.cast()))
}

pub(crate) unsafe extern "C" fn cudnnDestroyDropoutDescriptor(
    dropout_desc: ct9::cudnnDropoutDescriptor_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenDestroyDropoutDescriptor(mem::transmute(dropout_desc)))
}

pub(crate) unsafe extern "C" fn cudnnDropoutGetStatesSize(
    handle: ct9::cudnnHandle_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenDropoutGetStatesSize(ctx_handle(handle), size_in_bytes))
}

/// I tre parametri che MIOpen ha in piu':
/// - `use_mask`: se vero riusa una maschera fornita invece di generarla. cuDNN
///   non ha il concetto: falso.
/// - `state_evo`: fa evolvere lo stato del generatore fra una chiamata e
///   l'altra. cuDNN riparte dal seme: falso.
/// - `rng_mode`: 0 e' l'unico generatore che MIOpen implementa.
pub(crate) unsafe extern "C" fn cudnnSetDropoutDescriptor(
    dropout_desc: ct9::cudnnDropoutDescriptor_t,
    handle: ct9::cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    state_size_in_bytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> ct9::cudnnStatus_t {
    // PyTorch crea SEMPRE un descrittore di dropout per le RNN e, quando la
    // probabilita' e' zero, lo imposta con `states = nullptr` e dimensione 0
    // (`DropoutDescriptor::set_no_dropout`). cuDNN lo accetta senza fare nulla;
    // MIOpen invece prova a inizializzare il generatore pseudocasuale e lancia
    // un kernel su ZERO elementi:
    //   [run] kernel_name = InitKernelStateHIP, global_work_dim = { 0, 1, 1 }
    //   Failed to launch kernel: invalid argument   (hipoc_kernel.cpp:106)
    // e ogni LSTM falliva prima ancora di partire. Misurato il 2 ago 2026.
    //
    // Un descrittore MIOpen appena creato ha gia' dropout=0, pstates=nullptr e
    // stateSizeInBytes=0: e' esattamente "nessun dropout". Quindi in quel caso
    // non c'e' niente da impostare.
    if dropout == 0.0 && states.is_null() && state_size_in_bytes == 0 {
        return Ok(());
    }
    miopen_to_cudnn_status(miopenSetDropoutDescriptor(
        mem::transmute(dropout_desc),
        ctx_handle(handle),
        dropout,
        states,
        state_size_in_bytes,
        seed,
        false,
        false,
        mem::transmute(0u32),
    ))
}

pub(crate) unsafe extern "C" fn cudnnRestoreDropoutDescriptor(
    dropout_desc: ct9::cudnnDropoutDescriptor_t,
    handle: ct9::cudnnHandle_t,
    dropout: f32,
    states: *mut ::core::ffi::c_void,
    state_size_in_bytes: usize,
    seed: ::core::ffi::c_ulonglong,
) -> ct9::cudnnStatus_t {
    // PyTorch crea SEMPRE un descrittore di dropout per le RNN e, quando la
    // probabilita' e' zero, lo imposta con `states = nullptr` e dimensione 0
    // (`DropoutDescriptor::set_no_dropout`). cuDNN lo accetta senza fare nulla;
    // MIOpen invece prova a inizializzare il generatore pseudocasuale e lancia
    // un kernel su ZERO elementi:
    //   [run] kernel_name = InitKernelStateHIP, global_work_dim = { 0, 1, 1 }
    //   Failed to launch kernel: invalid argument   (hipoc_kernel.cpp:106)
    // e ogni LSTM falliva prima ancora di partire. Misurato il 2 ago 2026.
    //
    // Un descrittore MIOpen appena creato ha gia' dropout=0, pstates=nullptr e
    // stateSizeInBytes=0: e' esattamente "nessun dropout". Quindi in quel caso
    // non c'e' niente da impostare.
    if dropout == 0.0 && states.is_null() && state_size_in_bytes == 0 {
        return Ok(());
    }
    miopen_to_cudnn_status(miopenRestoreDropoutDescriptor(
        mem::transmute(dropout_desc),
        ctx_handle(handle),
        dropout,
        states,
        state_size_in_bytes,
        seed,
        false,
        false,
        mem::transmute(0u32),
    ))
}

// --- CTC loss --------------------------------------------------------------

pub(crate) unsafe extern "C" fn cudnnCreateCTCLossDescriptor(
    ctc_loss_desc: *mut ct9::cudnnCTCLossDescriptor_t,
) -> ct9::cudnnStatus_t {
    if ctc_loss_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreateCTCLossDescriptor(ctc_loss_desc.cast()))
}

pub(crate) unsafe extern "C" fn cudnnDestroyCTCLossDescriptor(
    ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenDestroyCTCLossDescriptor(mem::transmute(ctc_loss_desc)))
}

/// MIOpen chiede due cose che cuDNN non ha nella forma base:
/// - `blank_label_id`: cuDNN usa sempre 0, ed e' anche il valore predefinito di
///   `torch.nn.CTCLoss`.
/// - `apply_softmax_layer`: nella forma base cuDNN vuole probabilita' gia'
///   normalizzate, quindi falso. La forma `Ex` lo sceglie (vedi sotto).
pub(crate) unsafe extern "C" fn cudnnSetCTCLossDescriptor(
    ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
    comp_type: ct9::cudnnDataType_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenSetCTCLossDescriptor(
        mem::transmute(ctc_loss_desc),
        tipo_dato(comp_type)?,
        0,
        false,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetCTCLossDescriptorEx(
    ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
    comp_type: ct9::cudnnDataType_t,
    norm_mode: ct9::cudnnLossNormalizationMode_t,
    _grad_mode: ct9::cudnnNanPropagation_t,
) -> ct9::cudnnStatus_t {
    // cudnnLossNormalizationMode_t: 0 = NONE, 1 = SOFTMAX. Corrisponde esatto al
    // flag `apply_softmax_layer` di MIOpen.
    miopen_to_cudnn_status(miopenSetCTCLossDescriptor(
        mem::transmute(ctc_loss_desc),
        tipo_dato(comp_type)?,
        0,
        norm_mode.0 == 1,
    ))
}

pub(crate) unsafe extern "C" fn cudnnGetCTCLossWorkspaceSize(
    handle: ct9::cudnnHandle_t,
    probs_desc: ct9::cudnnTensorDescriptor_t,
    gradients_desc: ct9::cudnnTensorDescriptor_t,
    labels: *const ::core::ffi::c_int,
    label_lengths: *const ::core::ffi::c_int,
    input_lengths: *const ::core::ffi::c_int,
    _algo: ct9::cudnnCTCLossAlgo_t,
    ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenGetCTCLossWorkspaceSize(
        ctx_handle(handle),
        mem::transmute(probs_desc),
        mem::transmute(gradients_desc),
        labels,
        label_lengths,
        input_lengths,
        mem::transmute(0u32),
        mem::transmute(ctc_loss_desc),
        size_in_bytes,
    ))
}

pub(crate) unsafe extern "C" fn cudnnCTCLoss(
    handle: ct9::cudnnHandle_t,
    probs_desc: ct9::cudnnTensorDescriptor_t,
    probs: *const ::core::ffi::c_void,
    host_labels: *const ::core::ffi::c_int,
    host_label_lengths: *const ::core::ffi::c_int,
    host_input_lengths: *const ::core::ffi::c_int,
    costs: *mut ::core::ffi::c_void,
    gradients_desc: ct9::cudnnTensorDescriptor_t,
    gradients: *mut ::core::ffi::c_void,
    _algo: ct9::cudnnCTCLossAlgo_t,
    ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
) -> ct9::cudnnStatus_t {
    // Come in cuDNN, le tre liste (etichette e lunghezze) stanno in memoria host.
    miopen_to_cudnn_status(miopenCTCLoss(
        ctx_handle(handle),
        mem::transmute(probs_desc),
        probs,
        host_labels,
        host_label_lengths,
        host_input_lengths,
        costs,
        mem::transmute(gradients_desc),
        gradients,
        mem::transmute(0u32),
        mem::transmute(ctc_loss_desc),
        workspace,
        workspace_size,
    ))
}

// ===========================================================================
//  RNN / LSTM / GRU — 2 ago 2026
//
//  Queste funzioni sono state RIMOSSE in cuDNN 9: esistono solo nel ramo 8, ed
//  e' quello che PyTorch 2.3 chiama. I TIPI pero' esistono anche in cudnn9,
//  quindi si scrivono qui come le altre e il wrapper utile e' quello dnn8_b.
//
//  ATTENZIONE — L'ORDINE DELLE PORTE NON COINCIDE.
//
//    LSTM   cuDNN: 0 ingresso, 1 dimenticanza, 2 nuova memoria, 3 uscita
//           MIOpen: 0 ingresso, 1 dimenticanza, 2 uscita,        3 nuova memoria
//    GRU    cuDNN: 0 azzeramento, 1 aggiornamento, 2 nuova memoria
//           MIOpen: 0 aggiornamento, 1 azzeramento, 2 nuova memoria
//
//  Passare l'indice cosi' com'e' darebbe una rete che si addestra senza dare
//  errori e calcola con le porte scambiate. Le due tabelle qui sotto lo evitano.
// ===========================================================================

/// Da indice di porta cuDNN a indice MIOpen, secondo il modo della rete.
/// Il modo si rilegge dal descrittore: non lo teniamo da parte.
unsafe fn indice_porta(
    rnn_desc: miopenRNNDescriptor_t,
    id: ::core::ffi::c_int,
) -> Result<::core::ffi::c_int, cudnnError_t> {
    let (modo, _) = leggi_modo_rnn(rnn_desc)?;
    // LSTM: si scambiano 2 e 3 in entrambe le meta' (ingresso e ricorrente).
    const LSTM: [i32; 8] = [0, 1, 3, 2, 4, 5, 7, 6];
    // GRU: si scambiano 0 e 1 in entrambe le meta'.
    const GRU: [i32; 6] = [1, 0, 2, 4, 3, 5];
    let tabella: &[i32] = match modo {
        2 => &LSTM,
        3 => &GRU,
        _ => &[0, 1], // RELU e TANH: una matrice d'ingresso e una ricorrente
    };
    tabella
        .get(id as usize)
        .copied()
        .ok_or(cudnnError_t::BAD_PARAM)
}

/// Rilegge modo e tipo di dato dal descrittore RNN. Il tipo serve a convertire
/// in byte l'offset che MIOpen restituisce in ELEMENTI.
unsafe fn leggi_modo_rnn(
    rnn_desc: miopenRNNDescriptor_t,
) -> Result<(u32, miopenDataType_t), cudnnError_t> {
    let mut hs = 0;
    let mut nl = 0;
    let mut drop: miopenDropoutDescriptor_t = mem::zeroed();
    let mut inm: miopenRNNInputMode_t = mem::zeroed();
    let mut dir: miopenRNNDirectionMode_t = mem::zeroed();
    let mut modo: miopenRNNMode_t = mem::zeroed();
    let mut bias: miopenRNNBiasMode_t = mem::zeroed();
    let mut algo: miopenRNNAlgo_t = mem::zeroed();
    let mut tipo: miopenDataType_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetRNNDescriptor_V2(
        rnn_desc, &mut hs, &mut nl, &mut drop, &mut inm, &mut dir, &mut modo,
        &mut bias, &mut algo, &mut tipo,
    ))?;
    Ok((modo.0, tipo))
}

fn byte_per_elemento(t: miopenDataType_t) -> usize {
    match t {
        miopenDataType_t::miopenHalf | miopenDataType_t::miopenBFloat16 => 2,
        miopenDataType_t::miopenDouble => 8,
        miopenDataType_t::miopenInt8 => 1,
        _ => 4,
    }
}

// ===========================================================================
//  Traduzione della disposizione dei pesi RNN — 2 ago 2026
//
//  PyTorch pretende che il blocco unico dei pesi abbia ESATTAMENTE la
//  disposizione di cuDNN, e lo verifica: `at::native` cammina i parametri
//  chiamando cudnnGetRNNLinLayerMatrixParams / ...BiasParams e asserisce
//  `cur_offset == offset`. MIOpen dispone le stesse cose in modo diverso, per
//  due motivi indipendenti — misurati il 2 ago 2026 interrogando MIOpen:
//
//  1) L'ORDINE DELLE PORTE
//       LSTM  cuDNN: ingresso, dimenticanza, nuova memoria, uscita
//             MIOpen: ingresso, dimenticanza, uscita, nuova memoria
//       GRU   cuDNN: azzeramento, aggiornamento, nuova memoria
//             MIOpen: aggiornamento, azzeramento, nuova memoria
//
//  2) LA DISPOSIZIONE COMPLESSIVA, quando gli strati sono piu' di uno
//       cuDNN:  [strato0: matrici, polarizzazioni][strato1: matrici, pol.]...
//       MIOpen: [tutte le matrici di tutti gli strati][tutte le polarizzazioni]
//     Con un solo strato le due coincidono; da due in su no.
//
//  Rimappare i soli indici non basta e rompe pure il controllo di contiguita'.
//  Quello che si fa qui e': a PyTorch si presenta la disposizione cuDNN
//  (contigua, come se la dettasse lui), e al momento di chiamare MIOpen il
//  blocco viene ricopiato nell'ordine che MIOpen si aspetta. Al ritorno dei
//  gradienti sui pesi si fa il percorso inverso.
// ===========================================================================

/// Un pezzo del blocco dei pesi: dove sta secondo cuDNN, dove secondo MIOpen.
#[derive(Clone, Copy)]
struct BloccoPesi {
    cudnn: usize,   // in elementi
    miopen: usize,  // in elementi
    elementi: usize,
}

/// Numero di porte per modo: LSTM 4, GRU 3, RNN semplice 1.
fn porte_per_modo(modo: u32) -> usize {
    match modo {
        2 => 4,
        3 => 3,
        _ => 1,
    }
}

/// Costruisce la corrispondenza fra le due disposizioni interrogando MIOpen.
/// Restituisce i blocchi, il totale in elementi e la dimensione dell'elemento.
unsafe fn mappa_pesi_rnn(
    mh: miopenHandle_t,
    rnn: miopenRNNDescriptor_t,
    x_desc: miopenTensorDescriptor_t,
) -> Result<(Vec<BloccoPesi>, usize, usize), cudnnError_t> {
    let mut hs = 0;
    let mut strati = 0;
    let mut drop: miopenDropoutDescriptor_t = mem::zeroed();
    let mut inm: miopenRNNInputMode_t = mem::zeroed();
    let mut dir: miopenRNNDirectionMode_t = mem::zeroed();
    let mut modo: miopenRNNMode_t = mem::zeroed();
    let mut bias: miopenRNNBiasMode_t = mem::zeroed();
    let mut algo: miopenRNNAlgo_t = mem::zeroed();
    let mut tipo: miopenDataType_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetRNNDescriptor_V2(
        rnn, &mut hs, &mut strati, &mut drop, &mut inm, &mut dir, &mut modo, &mut bias,
        &mut algo, &mut tipo,
    ))?;

    let porte = porte_per_modo(modo.0);
    let pseudo = strati * if dir.0 == 1 { 2 } else { 1 };
    let byte_elem = byte_per_elemento(tipo);

    let mut d: miopenTensorDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateTensorDescriptor(&mut d))?;

    let mut blocchi = Vec::with_capacity((pseudo as usize) * porte * 4);
    let mut cur = 0usize;
    for l in 0..pseudo {
        // prima le matrici, poi le polarizzazioni: e' l'ordine di cuDNN
        for k in 0..(2 * porte) {
            let id = indice_porta_modo(modo.0, k as i32)?;
            let mut sz: usize = 0;
            let mut off: usize = 0;
            miopen_to_cudnn_status(miopenGetRNNLayerParamSize(mh, rnn, l, x_desc, id, &mut sz))?;
            miopen_to_cudnn_status(miopenGetRNNLayerParamOffset(rnn, l, x_desc, id, d, &mut off))?;
            let n = sz / byte_elem;
            blocchi.push(BloccoPesi { cudnn: cur, miopen: off, elementi: n });
            cur += n;
        }
        for k in 0..(2 * porte) {
            let id = indice_porta_modo(modo.0, k as i32)?;
            let mut sz: usize = 0;
            let mut off: usize = 0;
            miopen_to_cudnn_status(miopenGetRNNLayerBiasSize(mh, rnn, l, id, &mut sz))?;
            miopen_to_cudnn_status(miopenGetRNNLayerBiasOffset(rnn, l, x_desc, id, d, &mut off))?;
            let n = sz / byte_elem;
            blocchi.push(BloccoPesi { cudnn: cur, miopen: off, elementi: n });
            cur += n;
        }
    }
    miopenDestroyTensorDescriptor(d).ok();
    Ok((blocchi, cur, byte_elem))
}

/// Come `indice_porta`, ma col modo gia' noto: evita di rileggere il
/// descrittore una volta per blocco.
fn indice_porta_modo(modo: u32, id: i32) -> Result<i32, cudnnError_t> {
    const LSTM: [i32; 8] = [0, 1, 3, 2, 4, 5, 7, 6];
    const GRU: [i32; 6] = [1, 0, 2, 4, 3, 5];
    let tabella: &[i32] = match modo {
        2 => &LSTM,
        3 => &GRU,
        _ => &[0, 1],
    };
    tabella.get(id as usize).copied().ok_or(cudnnError_t::BAD_PARAM)
}

/// Ricopia il blocco dei pesi da una disposizione all'altra.
/// `verso_miopen` vero: da cuDNN a MIOpen (prima di chiamare).
/// falso: da MIOpen a cuDNN (per i gradienti che tornano indietro).
unsafe fn permuta_pesi(
    mh: miopenHandle_t,
    lato_cudnn: *mut u8,
    lato_miopen: *mut u8,
    blocchi: &[BloccoPesi],
    byte_elem: usize,
    verso_miopen: bool,
) -> Result<(), cudnnError_t> {
    let mut stream: hipStream_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetStream(mh, &mut stream))?;
    for b in blocchi {
        let quanti = b.elementi * byte_elem;
        let da_cudnn = lato_cudnn.add(b.cudnn * byte_elem);
        let da_miopen = lato_miopen.add(b.miopen * byte_elem);
        let (src, dst) = if verso_miopen {
            (da_cudnn, da_miopen)
        } else {
            (da_miopen, da_cudnn)
        };
        hipMemcpyAsync(
            dst.cast(),
            src.cast(),
            quanti,
            hipMemcpyKind::hipMemcpyDeviceToDevice,
            stream,
        )
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
    }
    Ok(())
}

/// Prepara il blocco nell'ordine di MIOpen e restituisce il puntatore da usare.
/// Il buffer di appoggio vive nel Context e viene riusato.
unsafe fn pesi_per_miopen(
    handle: ct9::cudnnHandle_t,
    rnn: miopenRNNDescriptor_t,
    x_desc: miopenTensorDescriptor_t,
    w: *const ::core::ffi::c_void,
) -> Result<(*mut ::core::ffi::c_void, Vec<BloccoPesi>, usize), cudnnError_t> {
    let c = ctx(handle);
    let mh = c.base;
    let (blocchi, totale, byte_elem) = mappa_pesi_rnn(mh, rnn, x_desc)?;
    let appoggio = c.buffer_pesi_rnn(totale * byte_elem)?;
    permuta_pesi(mh, w as *mut u8, appoggio.cast(), &blocchi, byte_elem, true)?;
    Ok((appoggio, blocchi, byte_elem))
}

pub(crate) unsafe extern "C" fn cudnnCreateRNNDescriptor(
    rnn_desc: *mut ct9::cudnnRNNDescriptor_t,
) -> ct9::cudnnStatus_t {
    if rnn_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreateRNNDescriptor(
        rnn_desc.cast(),
    ))
}

pub(crate) unsafe extern "C" fn cudnnDestroyRNNDescriptor(
    rnn_desc: ct9::cudnnRNNDescriptor_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenDestroyRNNDescriptor(
        mem::transmute(rnn_desc),
    ))
}

pub(crate) unsafe extern "C" fn cudnnCreateRNNDataDescriptor(
    rnn_data_desc: *mut ct9::cudnnRNNDataDescriptor_t,
) -> ct9::cudnnStatus_t {
    if rnn_data_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenCreateSeqTensorDescriptor(
        rnn_data_desc.cast(),
    ))
}

pub(crate) unsafe extern "C" fn cudnnDestroyRNNDataDescriptor(
    rnn_data_desc: ct9::cudnnRNNDataDescriptor_t,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenDestroySeqTensorDescriptor(
        mem::transmute(rnn_data_desc),
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetRNNDataDescriptor(
    rnn_data_desc: ct9::cudnnRNNDataDescriptor_t,
    data_type: ct9::cudnnDataType_t,
    layout: ct9::cudnnRNNDataLayout_t,
    max_seq_length: ::core::ffi::c_int,
    batch_size: ::core::ffi::c_int,
    vector_size: ::core::ffi::c_int,
    seq_length_array: *const ::core::ffi::c_int,
    padding_fill: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    // cuDNN:  0 SEQ_MAJOR_UNPACKED, 1 SEQ_MAJOR_PACKED, 2 BATCH_MAJOR_UNPACKED
    // MIOpen: 1 SeqMajorNotPadded,  2 SeqMajorPadded,   3 BatchMajorPadded
    // "unpacked" di cuDNN vuol dire con riempimento, quindi "padded".
    let l: u32 = match layout.0 {
        0 => 2,
        1 => 1,
        2 => 3,
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    };
    let tipo = match tipo_dato(data_type) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    miopen_to_cudnn_status(miopenSetRNNDataSeqTensorDescriptor(
        mem::transmute(rnn_data_desc),
        tipo,
        mem::transmute(l),
        max_seq_length,
        batch_size,
        vector_size,
        seq_length_array,
        padding_fill,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetRNNDescriptor_v6(
    _handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    hidden_size: ::core::ffi::c_int,
    num_layers: ::core::ffi::c_int,
    dropout_desc: ct9::cudnnDropoutDescriptor_t,
    input_mode: ct9::cudnnRNNInputMode_t,
    direction: ct9::cudnnDirectionMode_t,
    cell_mode: ct9::cudnnRNNMode_t,
    _algo: ct9::cudnnRNNAlgo_t,
    math_prec: ct9::cudnnDataType_t,
) -> ct9::cudnnStatus_t {
    // Modo, direzione e modalita' d'ingresso hanno gli stessi valori numerici
    // nelle due librerie: verificato sulle tabelle degli enum.
    //
    // `biasMode`: cuDNN ha SEMPRE la polarizzazione nelle RNN (due vettori per
    // porta), quindi si passa `miopenRNNwithBias`.
    //
    // `algo`: cuDNN ne ha quattro (standard e tre "persistent"), MIOpen tre
    // diversi. Si passa sempre il predefinito: i "persistent" sono
    // ottimizzazioni per NVIDIA che qui non hanno corrispondenza, e sceglierne
    // uno a caso sarebbe peggio.
    let tipo = match tipo_dato(math_prec) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    miopen_to_cudnn_status(miopenSetRNNDescriptor_V2(
        mem::transmute(rnn_desc),
        hidden_size,
        num_layers,
        mem::transmute(dropout_desc),
        mem::transmute(input_mode.0),
        mem::transmute(direction.0),
        mem::transmute(cell_mode.0),
        miopenRNNBiasMode_t::miopenRNNwithBias,
        mem::transmute(0u32),
        tipo,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetRNNMatrixMathType(
    _rnn_desc: ct9::cudnnRNNDescriptor_t,
    _m_type: ct9::cudnnMathType_t,
) -> ct9::cudnnStatus_t {
    // MIOpen non espone la scelta della precisione interna: la decide lui in
    // base al tipo di dato. Come per `cudnnSetConvolutionMathType`, si accetta
    // e si ignora — dire di no bloccherebbe PyTorch senza motivo.
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnSetRNNProjectionLayers(
    _handle: ct9::cudnnHandle_t,
    _rnn_desc: ct9::cudnnRNNDescriptor_t,
    rec_proj_size: ::core::ffi::c_int,
    _out_proj_size: ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    // La proiezione ricorrente (LSTMP) in MIOpen non c'e'. PyTorch la chiede
    // solo con `proj_size > 0`; se non e' richiesta, non c'e' niente da fare e
    // si accetta. Se e' richiesta si dichiara, invece di ignorarla in silenzio e
    // restituire una rete di forma diversa da quella chiesta.
    if rec_proj_size > 0 {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetRNNParamsSize(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    x_desc: ct9::cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
    data_type: ct9::cudnnDataType_t,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let tipo = match tipo_dato(data_type) {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    miopen_to_cudnn_status(miopenGetRNNParamsSize(
        ctx_handle(handle),
        mem::transmute(rnn_desc),
        mem::transmute(x_desc),
        size_in_bytes,
        tipo,
    ))
}

pub(crate) unsafe extern "C" fn cudnnGetRNNTrainingReserveSize(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: ::core::ffi::c_int,
    x_desc: *const ct9::cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenGetRNNTrainingReserveSize(
        ctx_handle(handle),
        mem::transmute(rnn_desc),
        seq_length,
        x_desc.cast(),
        size_in_bytes,
    ))
}

/// Restituisce descrittore e puntatore alla matrice di una singola porta dentro
/// il blocco unico dei pesi.
///
/// MIOpen da' l'offset in ELEMENTI (cosi' lo usa il suo stesso driver), non in
/// byte: va moltiplicato per la dimensione del tipo. Se un giorno i pesi
/// risultassero traslati, e' la prima cosa da rivedere.
pub(crate) unsafe extern "C" fn cudnnGetRNNLinLayerMatrixParams(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    pseudo_layer: ::core::ffi::c_int,
    x_desc: ct9::cudnnTensorDescriptor_t,
    _w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    lin_layer_id: ::core::ffi::c_int,
    lin_layer_mat_desc: ct9::cudnnFilterDescriptor_t,
    lin_layer_mat: *mut *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    // L'offset che si riferisce e' quello della disposizione cuDNN, non di
    // MIOpen: PyTorch verifica che i parametri siano contigui in quell'ordine.
    // La corrispondenza con le posizioni vere la tiene `mappa_pesi_rnn`, e il
    // riordino avviene al momento della chiamata a MIOpen.
    let rnn: miopenRNNDescriptor_t = mem::transmute(rnn_desc);
    let mh = ctx_handle(handle);
    let (blocchi, _, byte_elem) = mappa_pesi_rnn(mh, rnn, mem::transmute(x_desc))?;
    let (modo, _) = leggi_modo_rnn(rnn)?;
    let porte = porte_per_modo(modo);
    if lin_layer_id < 0 || lin_layer_id as usize >= 2 * porte {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let posto = (pseudo_layer as usize) * 4 * porte + lin_layer_id as usize;
    let b = *blocchi.get(posto).ok_or(cudnnError_t::BAD_PARAM)?;
    // il descrittore lo popola MIOpen, con l'indice VERO della porta
    let id = indice_porta_modo(modo, lin_layer_id)?;
    let mut scarto: usize = 0;
    miopen_to_cudnn_status(miopenGetRNNLayerParamOffset(
        rnn,
        pseudo_layer,
        mem::transmute(x_desc),
        id,
        mem::transmute(lin_layer_mat_desc),
        &mut scarto,
    ))?;
    if !lin_layer_mat.is_null() {
        *lin_layer_mat = (w as *mut u8).add(b.cudnn * byte_elem).cast();
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetRNNLinLayerBiasParams(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    pseudo_layer: ::core::ffi::c_int,
    x_desc: ct9::cudnnTensorDescriptor_t,
    _w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    lin_layer_id: ::core::ffi::c_int,
    lin_layer_bias_desc: ct9::cudnnFilterDescriptor_t,
    lin_layer_bias: *mut *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    // Come sopra: offset nella disposizione cuDNN.
    let rnn: miopenRNNDescriptor_t = mem::transmute(rnn_desc);
    let mh = ctx_handle(handle);
    let (blocchi, _, byte_elem) = mappa_pesi_rnn(mh, rnn, mem::transmute(x_desc))?;
    let (modo, _) = leggi_modo_rnn(rnn)?;
    let porte = porte_per_modo(modo);
    if lin_layer_id < 0 || lin_layer_id as usize >= 2 * porte {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let posto = (pseudo_layer as usize) * 4 * porte + 2 * porte + lin_layer_id as usize;
    let b = *blocchi.get(posto).ok_or(cudnnError_t::BAD_PARAM)?;
    let id = indice_porta_modo(modo, lin_layer_id)?;
    let mut scarto: usize = 0;
    miopen_to_cudnn_status(miopenGetRNNLayerBiasOffset(
        rnn,
        pseudo_layer,
        mem::transmute(x_desc),
        id,
        mem::transmute(lin_layer_bias_desc),
        &mut scarto,
    ))?;
    if !lin_layer_bias.is_null() {
        *lin_layer_bias = (w as *mut u8).add(b.cudnn * byte_elem).cast();
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnRNNForwardTraining(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: ::core::ffi::c_int,
    x_desc: *const ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    hx_desc: ct9::cudnnTensorDescriptor_t,
    hx: *const ::core::ffi::c_void,
    cx_desc: ct9::cudnnTensorDescriptor_t,
    cx: *const ::core::ffi::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    y_desc: *const ct9::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    hy_desc: ct9::cudnnTensorDescriptor_t,
    hy: *mut ::core::ffi::c_void,
    cy_desc: ct9::cudnnTensorDescriptor_t,
    cy: *mut ::core::ffi::c_void,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
    reserve_space: *mut ::core::ffi::c_void,
    reserve_space_size: usize,
) -> ct9::cudnnStatus_t {
    // GRU e RNN semplice non hanno lo stato di cella: PyTorch passa un
    // descrittore NULLO per cx/cy. cuDNN lo tollera, MIOpen lo dereferenzia
    // («Dereferencing nullptr», errors.hpp:128) e la rete non parte.
    // Si sostituisce col descrittore dello stato nascosto, che ha la stessa
    // forma; il puntatore ai dati resta nullo, ed e' quello che conta.
    let cx_desc = if cx_desc.is_null() { hx_desc } else { cx_desc };
    let cy_desc = if cy_desc.is_null() { hy_desc } else { cy_desc };
    // I pesi arrivano nella disposizione di cuDNN: vanno riordinati in quella
    // di MIOpen prima di passarli. Vedi il blocco sulla traduzione, sopra.
    let (pesi_mio, _blocchi, _byte_elem) = pesi_per_miopen(
        handle,
        mem::transmute(rnn_desc),
        mem::transmute(*x_desc),
        w,
    )?;
    // Firma identica a quella di MIOpen, argomento per argomento.
    miopen_to_cudnn_status(miopenRNNForwardTraining(
        ctx_handle(handle),
        mem::transmute(rnn_desc),
        seq_length,
        x_desc.cast(),
        x,
        mem::transmute(hx_desc),
        hx,
        mem::transmute(cx_desc),
        cx,
        mem::transmute(w_desc),
        pesi_mio,
        y_desc.cast(),
        y,
        mem::transmute(hy_desc),
        hy,
        mem::transmute(cy_desc),
        cy,
        workspace,
        workspace_size,
        reserve_space,
        reserve_space_size,
    ))
}

// ===========================================================================
//  Spatial transformer — 2 ago 2026
//
//  MIOpen NON ha nessuna di queste operazioni: non c'e' niente da tradurre, i
//  kernel vanno scritti. Sono quattro, tutti semplici, e vengono compilati a
//  runtime con hipRTC (la stessa via che usa MIOpen per i propri) e caricati
//  una volta sola.
//
//  Convenzioni di cuDNN, rispettate qui:
//   - la griglia ha coordinate in [-1, 1], nell'ordine (x, y)
//   - i vertici corrispondono ai centri dei pixel estremi (`align_corners`)
//   - fuori dal bordo il campionamento vale zero
// ===========================================================================

/// Sorgente dei quattro kernel. `float` soltanto: cuDNN accetta anche fp16 per
/// lo spatial transformer, ma PyTorch qui passa sempre fp32 e non si scrive
/// codice non verificabile.
const SORGENTE_SPAZIALE: &str = r#"
extern "C" __global__ void genera_griglia(
    const float* __restrict__ theta, float* __restrict__ griglia,
    int N, int H, int W)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totale = N * H * W;
    if (i >= totale) return;
    int w = i % W;
    int h = (i / W) % H;
    int n = i / (W * H);
    // coordinate del bersaglio normalizzate in [-1,1], vertici sui centri
    float xt = (W > 1) ? (-1.0f + 2.0f * (float)w / (float)(W - 1)) : 0.0f;
    float yt = (H > 1) ? (-1.0f + 2.0f * (float)h / (float)(H - 1)) : 0.0f;
    const float* t = theta + n * 6;
    griglia[2 * i + 0] = t[0] * xt + t[1] * yt + t[2];
    griglia[2 * i + 1] = t[3] * xt + t[4] * yt + t[5];
}

extern "C" __global__ void genera_griglia_indietro(
    const float* __restrict__ dgriglia, float* __restrict__ dtheta,
    int N, int H, int W)
{
    // un blocco per immagine: si accumulano i sei coefficienti
    int n = blockIdx.x;
    if (n >= N) return;
    __shared__ float acc[6][256];
    int tid = threadIdx.x;
    for (int k = 0; k < 6; k++) acc[k][tid] = 0.0f;
    int punti = H * W;
    for (int p = tid; p < punti; p += blockDim.x) {
        int w = p % W;
        int h = p / W;
        float xt = (W > 1) ? (-1.0f + 2.0f * (float)w / (float)(W - 1)) : 0.0f;
        float yt = (H > 1) ? (-1.0f + 2.0f * (float)h / (float)(H - 1)) : 0.0f;
        float gx = dgriglia[2 * (n * punti + p) + 0];
        float gy = dgriglia[2 * (n * punti + p) + 1];
        acc[0][tid] += gx * xt;  acc[1][tid] += gx * yt;  acc[2][tid] += gx;
        acc[3][tid] += gy * xt;  acc[4][tid] += gy * yt;  acc[5][tid] += gy;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) for (int k = 0; k < 6; k++) acc[k][tid] += acc[k][tid + s];
        __syncthreads();
    }
    if (tid == 0) for (int k = 0; k < 6; k++) dtheta[n * 6 + k] = acc[k][0];
}

extern "C" __global__ void campiona(
    const float* __restrict__ x, const float* __restrict__ griglia,
    float* __restrict__ y, int N, int C, int Hi, int Wi, int Ho, int Wo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totale = N * Ho * Wo;
    if (i >= totale) return;
    int wo = i % Wo;
    int ho = (i / Wo) % Ho;
    int n = i / (Wo * Ho);
    float gx = griglia[2 * i + 0];
    float gy = griglia[2 * i + 1];
    // da [-1,1] a indici di pixel
    float px = (gx + 1.0f) * 0.5f * (float)(Wi - 1);
    float py = (gy + 1.0f) * 0.5f * (float)(Hi - 1);
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    float ax = px - (float)x0, ay = py - (float)y0;
    for (int c = 0; c < C; c++) {
        const float* piano = x + ((size_t)n * C + c) * Hi * Wi;
        float somma = 0.0f;
        for (int dy = 0; dy < 2; dy++) {
            for (int dx = 0; dx < 2; dx++) {
                int xx = x0 + dx, yy = y0 + dy;
                if (xx < 0 || xx >= Wi || yy < 0 || yy >= Hi) continue;
                float peso = (dx ? ax : 1.0f - ax) * (dy ? ay : 1.0f - ay);
                somma += peso * piano[yy * Wi + xx];
            }
        }
        y[((size_t)n * C + c) * Ho * Wo + ho * Wo + wo] = somma;
    }
}

extern "C" __global__ void campiona_indietro(
    const float* __restrict__ x, const float* __restrict__ dy,
    const float* __restrict__ griglia, float* __restrict__ dx,
    float* __restrict__ dgriglia, int N, int C, int Hi, int Wi, int Ho, int Wo)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totale = N * Ho * Wo;
    if (i >= totale) return;
    int wo = i % Wo;
    int ho = (i / Wo) % Ho;
    int n = i / (Wo * Ho);
    float gx = griglia[2 * i + 0];
    float gy = griglia[2 * i + 1];
    float px = (gx + 1.0f) * 0.5f * (float)(Wi - 1);
    float py = (gy + 1.0f) * 0.5f * (float)(Hi - 1);
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    float ax = px - (float)x0, ay = py - (float)y0;
    float sgx = 0.0f, sgy = 0.0f;
    for (int c = 0; c < C; c++) {
        const float* piano = x + ((size_t)n * C + c) * Hi * Wi;
        float* dpiano = dx + ((size_t)n * C + c) * Hi * Wi;
        float g = dy[((size_t)n * C + c) * Ho * Wo + ho * Wo + wo];
        for (int ddy = 0; ddy < 2; ddy++) {
            for (int ddx = 0; ddx < 2; ddx++) {
                int xx = x0 + ddx, yy = y0 + ddy;
                if (xx < 0 || xx >= Wi || yy < 0 || yy >= Hi) continue;
                float px_peso = ddx ? ax : 1.0f - ax;
                float py_peso = ddy ? ay : 1.0f - ay;
                atomicAdd(&dpiano[yy * Wi + xx], g * px_peso * py_peso);
                float v = piano[yy * Wi + xx];
                sgx += g * v * (ddx ? 1.0f : -1.0f) * py_peso;
                sgy += g * v * (ddy ? 1.0f : -1.0f) * px_peso;
            }
        }
    }
    // dalla derivata in pixel a quella in coordinate normalizzate
    dgriglia[2 * i + 0] = sgx * 0.5f * (float)(Wi - 1);
    dgriglia[2 * i + 1] = sgy * 0.5f * (float)(Hi - 1);
}
"#;

/// I quattro kernel, compilati una volta sola per processo.
struct KernelSpaziali {
    genera: hipFunction_t,
    genera_indietro: hipFunction_t,
    campiona: hipFunction_t,
    campiona_indietro: hipFunction_t,
}
unsafe impl Send for KernelSpaziali {}
unsafe impl Sync for KernelSpaziali {}

static KERNEL_SPAZIALI: std::sync::OnceLock<Option<KernelSpaziali>> = std::sync::OnceLock::new();

/// Compila i kernel con hipRTC, caricata dinamicamente. Non si mette una
/// dipendenza a link perche' il pack potrebbe non esserci: in quel caso le
/// funzioni dello spatial transformer riferiscono NOT_SUPPORTED e il resto della
/// libreria continua a funzionare.
#[cfg(windows)]
unsafe fn compila_kernel_spaziali() -> Option<KernelSpaziali> {
    use std::ffi::CString;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut ::core::ffi::c_void;
        fn GetProcAddress(
            module: *mut ::core::ffi::c_void,
            name: *const ::core::ffi::c_char,
        ) -> *mut ::core::ffi::c_void;
    }
    fn utf16z(s: &str) -> Vec<u16> {
        let mut v: Vec<u16> = s.encode_utf16().collect();
        v.push(0);
        v
    }
    // Stesso ordine di ZLUDA: la 5.7 per prima, poi lo shim del pack.
    let mut h = ptr::null_mut();
    for nome in ["hiprtc0507.dll", "hiprtc07013.dll", "hiprtc0701.dll", "hiprtc.dll"] {
        h = LoadLibraryW(utf16z(nome).as_ptr());
        if !h.is_null() {
            break;
        }
    }
    if h.is_null() {
        return None;
    }
    macro_rules! sim {
        ($nome:literal) => {{
            let c = concat!($nome, "\0");
            let p = GetProcAddress(h, c.as_ptr() as *const ::core::ffi::c_char);
            if p.is_null() {
                return None;
            }
            p
        }};
    }
    type Crea = unsafe extern "C" fn(
        *mut *mut ::core::ffi::c_void,
        *const ::core::ffi::c_char,
        *const ::core::ffi::c_char,
        ::core::ffi::c_int,
        *const *const ::core::ffi::c_char,
        *const *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    type Comp = unsafe extern "C" fn(
        *mut ::core::ffi::c_void,
        ::core::ffi::c_int,
        *const *const ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;
    type Dim = unsafe extern "C" fn(*mut ::core::ffi::c_void, *mut usize) -> ::core::ffi::c_int;
    type Cod = unsafe extern "C" fn(
        *mut ::core::ffi::c_void,
        *mut ::core::ffi::c_char,
    ) -> ::core::ffi::c_int;

    let crea: Crea = mem::transmute(sim!("hiprtcCreateProgram"));
    let comp: Comp = mem::transmute(sim!("hiprtcCompileProgram"));
    let dim: Dim = mem::transmute(sim!("hiprtcGetCodeSize"));
    let cod: Cod = mem::transmute(sim!("hiprtcGetCode"));

    let sorgente = CString::new(SORGENTE_SPAZIALE).ok()?;
    let nome = CString::new("spaziale.hip").ok()?;
    let mut prog: *mut ::core::ffi::c_void = ptr::null_mut();
    if crea(&mut prog, sorgente.as_ptr(), nome.as_ptr(), 0, ptr::null(), ptr::null()) != 0 {
        return None;
    }
    let arch = CString::new("--offload-arch=gfx906").ok()?;
    let opzioni = [arch.as_ptr()];
    if comp(prog, 1, opzioni.as_ptr()) != 0 {
        return None;
    }
    let mut quanti: usize = 0;
    if dim(prog, &mut quanti) != 0 || quanti == 0 {
        return None;
    }
    let mut binario = vec![0u8; quanti];
    if cod(prog, binario.as_mut_ptr() as *mut ::core::ffi::c_char) != 0 {
        return None;
    }
    let mut modulo: hipModule_t = mem::zeroed();
    hipModuleLoadData(&mut modulo, binario.as_ptr().cast()).ok()?;
    let mut prendi = |nome: &str| -> Option<hipFunction_t> {
        let c = CString::new(nome).ok()?;
        let mut f: hipFunction_t = unsafe { mem::zeroed() };
        unsafe { hipModuleGetFunction(&mut f, modulo, c.as_ptr()) }.ok()?;
        Some(f)
    };
    Some(KernelSpaziali {
        genera: prendi("genera_griglia")?,
        genera_indietro: prendi("genera_griglia_indietro")?,
        campiona: prendi("campiona")?,
        campiona_indietro: prendi("campiona_indietro")?,
    })
}

#[cfg(not(windows))]
unsafe fn compila_kernel_spaziali() -> Option<KernelSpaziali> {
    None
}

unsafe fn kernel_spaziali() -> Result<&'static KernelSpaziali, cudnnError_t> {
    KERNEL_SPAZIALI
        .get_or_init(|| compila_kernel_spaziali())
        .as_ref()
        .ok_or(cudnnError_t::NOT_SUPPORTED)
}

/// Il descrittore: cuDNN lo tratta come opaco, quindi ce lo teniamo noi.
#[repr(C)]
struct DescrittoreSpaziale {
    n: ::core::ffi::c_int,
    c: ::core::ffi::c_int,
    h: ::core::ffi::c_int,
    w: ::core::ffi::c_int,
}

pub(crate) unsafe extern "C" fn cudnnCreateSpatialTransformerDescriptor(
    st_desc: *mut ct9::cudnnSpatialTransformerDescriptor_t,
) -> ct9::cudnnStatus_t {
    if st_desc.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let d = Box::new(DescrittoreSpaziale { n: 0, c: 0, h: 0, w: 0 });
    *st_desc = mem::transmute(Box::into_raw(d));
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnDestroySpatialTransformerDescriptor(
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
) -> ct9::cudnnStatus_t {
    let p: *mut DescrittoreSpaziale = mem::transmute(st_desc);
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnSetSpatialTransformerNdDescriptor(
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    sampler_type: ct9::cudnnSamplerType_t,
    data_type: ct9::cudnnDataType_t,
    nb_dims: ::core::ffi::c_int,
    dim_a: *const ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    // cuDNN ha un solo tipo di campionatore, il bilineare (valore 0).
    if sampler_type.0 != 0 {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    if data_type != ct9::cudnnDataType_t::CUDNN_DATA_FLOAT {
        // I kernel qui sotto sono scritti per fp32 soltanto.
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    if nb_dims != 4 || dim_a.is_null() {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    let p: *mut DescrittoreSpaziale = mem::transmute(st_desc);
    if p.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    (*p).n = *dim_a.add(0);
    (*p).c = *dim_a.add(1);
    (*p).h = *dim_a.add(2);
    (*p).w = *dim_a.add(3);
    Ok(())
}

unsafe fn lancia(
    f: hipFunction_t,
    quanti: usize,
    blocco: u32,
    stream: hipStream_t,
    argomenti: &mut [*mut ::core::ffi::c_void],
) -> Result<(), cudnnError_t> {
    let griglia = ((quanti as u32) + blocco - 1) / blocco;
    hipModuleLaunchKernel(
        f,
        griglia.max(1),
        1,
        1,
        blocco,
        1,
        1,
        0,
        stream,
        argomenti.as_mut_ptr(),
        ptr::null_mut(),
    )
    .map_err(|_| cudnnError_t::INTERNAL_ERROR)
}

pub(crate) unsafe extern "C" fn cudnnSpatialTfGridGeneratorForward(
    handle: ct9::cudnnHandle_t,
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    theta: *const ::core::ffi::c_void,
    grid: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    let k = kernel_spaziali()?;
    let d: *const DescrittoreSpaziale = mem::transmute(st_desc);
    if d.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mut stream: hipStream_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetStream(ctx_handle(handle), &mut stream))?;
    let (mut t, mut g) = (theta, grid);
    let (mut n, mut h, mut w) = ((*d).n, (*d).h, (*d).w);
    let mut arg: [*mut ::core::ffi::c_void; 5] = [
        &mut t as *mut _ as *mut _,
        &mut g as *mut _ as *mut _,
        &mut n as *mut _ as *mut _,
        &mut h as *mut _ as *mut _,
        &mut w as *mut _ as *mut _,
    ];
    lancia(k.genera, (n * h * w) as usize, 256, stream, &mut arg)
}

pub(crate) unsafe extern "C" fn cudnnSpatialTfGridGeneratorBackward(
    handle: ct9::cudnnHandle_t,
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    dgrid: *const ::core::ffi::c_void,
    dtheta: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    let k = kernel_spaziali()?;
    let d: *const DescrittoreSpaziale = mem::transmute(st_desc);
    if d.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mut stream: hipStream_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetStream(ctx_handle(handle), &mut stream))?;
    let (mut gd, mut dt) = (dgrid, dtheta);
    let (mut n, mut h, mut w) = ((*d).n, (*d).h, (*d).w);
    let mut arg: [*mut ::core::ffi::c_void; 5] = [
        &mut gd as *mut _ as *mut _,
        &mut dt as *mut _ as *mut _,
        &mut n as *mut _ as *mut _,
        &mut h as *mut _ as *mut _,
        &mut w as *mut _ as *mut _,
    ];
    // un blocco per immagine, 256 fili che si dividono i punti
    hipModuleLaunchKernel(
        k.genera_indietro,
        n.max(1) as u32,
        1,
        1,
        256,
        1,
        1,
        0,
        stream,
        arg.as_mut_ptr(),
        ptr::null_mut(),
    )
    .map_err(|_| cudnnError_t::INTERNAL_ERROR)
}

/// Legge N, C, H, W da un descrittore MIOpen a quattro dimensioni.
unsafe fn quattro_dim(
    desc: miopenTensorDescriptor_t,
) -> Result<(i32, i32, i32, i32), cudnnError_t> {
    let mut dims = [0i32; 8];
    let mut passi = [0i32; 8];
    let mut tipo: miopenDataType_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetTensorDescriptor(
        desc,
        &mut tipo,
        dims.as_mut_ptr(),
        passi.as_mut_ptr(),
    ))?;
    Ok((dims[0], dims[1], dims[2], dims[3]))
}

pub(crate) unsafe extern "C" fn cudnnSpatialTfSamplerForward(
    handle: ct9::cudnnHandle_t,
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    _alpha: *const ::core::ffi::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    _beta: *const ::core::ffi::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    // alpha/beta: PyTorch passa sempre 1 e 0. Non si finge di supportare la
    // combinazione lineare: se servisse, andrebbe aggiunta nel kernel.
    let k = kernel_spaziali()?;
    let _ = st_desc;
    let mut stream: hipStream_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetStream(ctx_handle(handle), &mut stream))?;
    let (n, c, hi, wi) = quattro_dim(mem::transmute(x_desc))?;
    let (_, _, ho, wo) = quattro_dim(mem::transmute(y_desc))?;
    let (mut px, mut pg, mut py) = (x, grid, y);
    let (mut vn, mut vc, mut vhi, mut vwi, mut vho, mut vwo) = (n, c, hi, wi, ho, wo);
    let mut arg: [*mut ::core::ffi::c_void; 9] = [
        &mut px as *mut _ as *mut _,
        &mut pg as *mut _ as *mut _,
        &mut py as *mut _ as *mut _,
        &mut vn as *mut _ as *mut _,
        &mut vc as *mut _ as *mut _,
        &mut vhi as *mut _ as *mut _,
        &mut vwi as *mut _ as *mut _,
        &mut vho as *mut _ as *mut _,
        &mut vwo as *mut _ as *mut _,
    ];
    lancia(k.campiona, (n * ho * wo) as usize, 256, stream, &mut arg)
}

pub(crate) unsafe extern "C" fn cudnnSpatialTfSamplerBackward(
    handle: ct9::cudnnHandle_t,
    st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    _alpha: *const ::core::ffi::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    _beta: *const ::core::ffi::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    _alpha_dgrid: *const ::core::ffi::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    grid: *const ::core::ffi::c_void,
    _beta_dgrid: *const ::core::ffi::c_void,
    dgrid: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    let k = kernel_spaziali()?;
    let _ = (st_desc, dx_desc);
    let mut stream: hipStream_t = mem::zeroed();
    miopen_to_cudnn_status(miopenGetStream(ctx_handle(handle), &mut stream))?;
    let (n, c, hi, wi) = quattro_dim(mem::transmute(x_desc))?;
    let (_, _, ho, wo) = quattro_dim(mem::transmute(dy_desc))?;
    // il kernel accumula con atomicAdd: dx va azzerato prima
    hipMemsetAsync(dx, 0, (n * c * hi * wi) as usize * 4, stream)
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?;
    let (mut px, mut pdy, mut pg, mut pdx, mut pdg) = (x, dy, grid, dx, dgrid);
    let (mut vn, mut vc, mut vhi, mut vwi, mut vho, mut vwo) = (n, c, hi, wi, ho, wo);
    let mut arg: [*mut ::core::ffi::c_void; 11] = [
        &mut px as *mut _ as *mut _,
        &mut pdy as *mut _ as *mut _,
        &mut pg as *mut _ as *mut _,
        &mut pdx as *mut _ as *mut _,
        &mut pdg as *mut _ as *mut _,
        &mut vn as *mut _ as *mut _,
        &mut vc as *mut _ as *mut _,
        &mut vhi as *mut _ as *mut _,
        &mut vwi as *mut _ as *mut _,
        &mut vho as *mut _ as *mut _,
        &mut vwo as *mut _ as *mut _,
    ];
    lancia(k.campiona_indietro, (n * ho * wo) as usize, 256, stream, &mut arg)
}

// ===========================================================================
//  API "backend" v8 di cuDNN — 2 ago 2026
//
//  E' un'API a grafo, molto diversa dalle altre: si creano descrittori opachi
//  di un TIPO, ci si mettono dentro degli ATTRIBUTI, si finalizzano, e alla
//  fine si esegue un piano. cudnn_frontend (che PyTorch usa) costruisce cosi':
//
//    tensori -> descrittore di convoluzione -> operazione -> grafo
//      -> euristica (che restituisce configurazioni di motore)
//      -> piano di esecuzione -> pacchetto di varianti -> esecuzione
//
//  MIOpen non ha niente di simile, ma non serve: il grafo che PyTorch costruisce
//  per una convoluzione contiene UNA sola operazione, e a quel punto si sa
//  esattamente quale chiamata MIOpen fare. Quello che segue e' quindi un
//  magazzino di attributi piu' un traduttore che, al momento dell'esecuzione,
//  ricostruisce la convoluzione e la manda per la stessa strada della v7 —
//  quella gia' verificata.
//
//  Cosa NON copre: i grafi con piu' operazioni (le fusioni), i pointwise, le
//  matmul, i normalization. Su quelli si riferisce NOT_SUPPORTED, che e' il
//  segnale che cudnn_frontend si aspetta per passare oltre.
// ===========================================================================

/// Segna a che punto siamo dentro l'esecuzione, per capire dove cede.
unsafe fn passo(nome: &str) {
    if let Ok(perc) = std::env::var("ZLUDA_DNN_TRACCIA") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(perc) {
            let _ = writeln!(f, "  passo: {nome}");
        }
    }
}

/// Come `ctx_handle`, ma senza panico: `ctx_handle` fa `unwrap()` e un panico
/// che attraversa un confine extern "C" ABORTA il processo. Nell'API backend
/// l'handle arriva da un attributo memorizzato dal chiamante e non e' detto che
/// sia valido: qui un handle sbagliato deve diventare un errore, non un crollo.
unsafe fn handle_sicuro(h: ct9::cudnnHandle_t) -> Result<miopenHandle_t, cudnnError_t> {
    let vivo = zluda_common::as_ref::<Context>(&h).ok_or(cudnnError_t::BAD_PARAM)?;
    let c = vivo.as_result().map_err(|_| cudnnError_t::BAD_PARAM)?;
    Ok(c.base)
}

mod attributi {
    // Tipi di descrittore
    pub const CONVOLUZIONE: u32 = 1;
    pub const MOTORE: u32 = 2;
    pub const CONFIG_MOTORE: u32 = 3;
    pub const EURISTICA: u32 = 4;
    pub const PIANO: u32 = 5;
    pub const OP_CONV_AVANTI: u32 = 10;
    pub const OP_CONV_PESI: u32 = 11;
    pub const OP_CONV_DATI: u32 = 12;
    pub const GRAFO: u32 = 15;
    pub const VARIANTI: u32 = 16;
    pub const TENSORE: u32 = 17;

    // Attributi dei tensori
    pub const T_TIPO: u32 = 901;
    pub const T_DIM: u32 = 902;
    pub const T_PASSI: u32 = 903;
    pub const T_ID: u32 = 906;

    // Attributi della convoluzione
    pub const C_DILATAZIONI: u32 = 102;
    pub const C_PASSI: u32 = 103;
    pub const C_BORDO_PRIMA: u32 = 105;
    pub const C_DIM_SPAZIALI: u32 = 106;

    // Attributi delle operazioni (avanti / indietro dati / indietro pesi)
    pub const OP_AV_CONV: u32 = 702;
    pub const OP_AV_W: u32 = 703;
    pub const OP_AV_X: u32 = 704;
    pub const OP_AV_Y: u32 = 705;
    pub const OP_DA_CONV: u32 = 708;
    pub const OP_DA_W: u32 = 709;
    pub const OP_DA_DX: u32 = 710;
    pub const OP_DA_DY: u32 = 711;
    pub const OP_PE_CONV: u32 = 714;
    pub const OP_PE_DW: u32 = 715;
    pub const OP_PE_X: u32 = 716;
    pub const OP_PE_DY: u32 = 717;

    // Grafo, motore, configurazione, euristica, piano, varianti
    pub const G_OPS: u32 = 801;
    pub const G_NUMERO_MOTORI: u32 = 802;
    pub const M_GRAFO: u32 = 1300;
    pub const M_INDICE: u32 = 1301;
    pub const M_NOTE_NUMERICHE: u32 = 1303;
    pub const M_NOTE_COMPORTAMENTO: u32 = 1305;
    pub const M_INFO_MANOPOLE: u32 = 1302;
    pub const CFG_MOTORE: u32 = 300;
    pub const CFG_MANOPOLE: u32 = 302;
    pub const CFG_SPAZIO: u32 = 303;
    pub const EU_GRAFO: u32 = 201;
    pub const EU_RISULTATI: u32 = 202;
    pub const P_CONFIG: u32 = 401;
    pub const P_SPAZIO: u32 = 402;
    pub const V_ID: u32 = 1000;
    pub const V_PUNTATORI: u32 = 1001;
    pub const V_SPAZIO: u32 = 1003;

    // Tipi degli attributi
    pub const TIPO_INT64: u32 = 3;
    pub const TIPO_PUNTATORE: u32 = 6;
    pub const TIPO_DESCRITTORE: u32 = 15;

    /// Chiave nostra, fuori dallo spazio di cuDNN: nelle configurazioni di
    /// motore che fabbrichiamo noi ci teniamo il grafo, cosi non serve
    /// tenere in vita descrittori nostri.
    pub const GRAFO_PRIVATO: u32 = 90001;
    /// Dove il piano tiene la propria copia risolta.
    pub const PIANO_PRIVATO: u32 = 90002;
}

/// Un attributo memorizzato: i byte grezzi cosi' come sono arrivati, piu' il
/// tipo e quanti elementi. Non si interpreta finche' non serve.
struct Attributo {
    tipo: u32,
    quanti: i64,
    byte: Vec<u8>,
}

struct DescrittoreBackend {
    tipo: u32,
    finalizzato: bool,
    attributi: FxHashMap<u32, Attributo>,
    /// I descrittori che questo ha creato da solo (euristica -> motore e
    /// configurazione) e che deve liberare.
    posseduti: Vec<*mut DescrittoreBackend>,
}

impl DescrittoreBackend {
    fn nuovo(tipo: u32) -> Self {
        Self { tipo, finalizzato: false, attributi: FxHashMap::default(), posseduti: Vec::new() }
    }
    fn interi(&self, attr: u32) -> Option<Vec<i64>> {
        let a = self.attributi.get(&attr)?;
        let n = a.quanti as usize;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let mut x = 0i64;
            unsafe {
                ptr::copy_nonoverlapping(a.byte.as_ptr().add(i * 8), &mut x as *mut i64 as *mut u8, 8)
            };
            v.push(x);
        }
        Some(v)
    }
    fn puntatori(&self, attr: u32) -> Option<Vec<*mut ::core::ffi::c_void>> {
        let a = self.attributi.get(&attr)?;
        let n = a.quanti as usize;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let mut x: *mut ::core::ffi::c_void = ptr::null_mut();
            unsafe {
                ptr::copy_nonoverlapping(
                    a.byte.as_ptr().add(i * 8),
                    &mut x as *mut *mut ::core::ffi::c_void as *mut u8,
                    8,
                )
            };
            v.push(x);
        }
        Some(v)
    }
    fn primo_descrittore(&self, attr: u32) -> Option<*mut DescrittoreBackend> {
        self.puntatori(attr)?.first().map(|p| (*p) as *mut DescrittoreBackend)
    }
}

/// Un tensore copiato per valore: niente puntatori al chiamante.
struct TensoreRisolto {
    dim: Vec<i32>,
    passi: Vec<i32>,
    tipo: miopenDataType_t,
    id: i64,
}

/// Tutto quello che serve a eseguire, copiato dentro il piano.
///
/// PERCHE' UNA COPIA. Il piano teneva puntatori ai descrittori del chiamante
/// (configurazione -> motore -> grafo -> operazione -> tensori). cudnn_frontend
/// pero' distrugge configurazioni e grafo appena costruito il piano, e
/// all'esecuzione quei puntatori sono memoria liberata: si leggeva spazzatura e
/// il processo cadeva con access violation. Misurato il 2 ago 2026.
/// Il piano dev'essere autosufficiente.
struct PianoRisolto {
    tipo_op: u32,
    x: TensoreRisolto,
    w: TensoreRisolto,
    y: TensoreRisolto,
    bordi: Vec<i32>,
    passi_conv: Vec<i32>,
    dilatazioni: Vec<i32>,
    dim_spaziali: i32,
    spazio: usize,
}

unsafe fn back<'a>(d: ct9::cudnnBackendDescriptor_t) -> Result<&'a mut DescrittoreBackend, cudnnError_t> {
    let p = d as *mut DescrittoreBackend;
    if p.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    Ok(&mut *p)
}

/// Traccia dei soli FALLIMENTI dell'API backend, per capire quale attributo
/// manca senza sommergere di righe. Si accende con ZLUDA_DNN_TRACCIA=<file>.
unsafe fn traccia_backend(dove: &str, tipo: u32, attr: u32, esito: &ct9::cudnnStatus_t) {
    let tutte = std::env::var("ZLUDA_DNN_TRACCIA_TUTTE").is_ok();
    if esito.is_ok() && !tutte {
        return;
    }
    if let Ok(perc) = std::env::var("ZLUDA_DNN_TRACCIA") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(perc) {
            let e = if esito.is_ok() { "ok" } else { "ERRORE" };
            let _ = writeln!(f, "{dove}: tipo {tipo}, attributo {attr} -> {e}");
        }
    }
}

pub(crate) unsafe extern "C" fn cudnnBackendCreateDescriptor(
    descriptor_type: ct9::cudnnBackendDescriptorType_t,
    descriptor: *mut ct9::cudnnBackendDescriptor_t,
) -> ct9::cudnnStatus_t {
    if descriptor.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let d = Box::new(DescrittoreBackend::nuovo(descriptor_type.0));
    *descriptor = Box::into_raw(d) as ct9::cudnnBackendDescriptor_t;
    Ok(())
}

/// Libera le copie possedute, scendendo lungo la catena.
unsafe fn libera_posseduti(d: &DescrittoreBackend) {
    for q in &d.posseduti {
        if !q.is_null() {
            let figlio = Box::from_raw(*q);
            libera_posseduti(&figlio);
        }
    }
}

pub(crate) unsafe extern "C" fn cudnnBackendDestroyDescriptor(
    descriptor: ct9::cudnnBackendDescriptor_t,
) -> ct9::cudnnStatus_t {
    let p = descriptor as *mut DescrittoreBackend;
    if p.is_null() {
        return Ok(());
    }
    let d = Box::from_raw(p);
    if let Some(v) = d.interi(attributi::PIANO_PRIVATO).and_then(|v| v.first().copied()) {
        if v != 0 {
            drop(Box::from_raw(v as *mut PianoRisolto));
        }
    }
    libera_posseduti(&d);
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnBackendSetAttribute(
    descriptor: ct9::cudnnBackendDescriptor_t,
    attribute_name: ct9::cudnnBackendAttributeName_t,
    attribute_type: ct9::cudnnBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *const ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    let esito = (|| -> ct9::cudnnStatus_t {
    let d = back(descriptor)?;
    if d.finalizzato {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    // Ogni tipo di attributo ha una dimensione fissa; quelli che non conosciamo
    // si memorizzano comunque come otto byte, che copre handle e descrittori.
    let dim = match attribute_type.0 {
        2 => 1,           // BOOLEAN
        4 => 4,           // FLOAT
        23 | 1 | 7 | 8 | 11 | 19 => 4, // INT32 e i vari enum
        5 => 8,           // DOUBLE
        _ => 8,           // INT64, VOID_PTR, HANDLE, BACKEND_DESCRIPTOR
    };
    let quanti = element_count.max(0) as usize;
    let mut byte = vec![0u8; quanti * dim];
    if quanti > 0 {
        if array_of_elements.is_null() {
            return Err(cudnnError_t::BAD_PARAM);
        }
        ptr::copy_nonoverlapping(array_of_elements as *const u8, byte.as_mut_ptr(), quanti * dim);
    }
    d.attributi.insert(
        attribute_name.0,
        Attributo { tipo: attribute_type.0, quanti: element_count, byte },
    );
    Ok(())
})();
    traccia_backend("imposta", back(descriptor).map(|d| d.tipo).unwrap_or(9999), attribute_name.0, &esito);
    esito
}

pub(crate) unsafe extern "C" fn cudnnBackendGetAttribute(
    descriptor: ct9::cudnnBackendDescriptor_t,
    attribute_name: ct9::cudnnBackendAttributeName_t,
    attribute_type: ct9::cudnnBackendAttributeType_t,
    requested_element_count: i64,
    element_count: *mut i64,
    array_of_elements: *mut ::core::ffi::c_void,
) -> ct9::cudnnStatus_t {
    let esito = (|| -> ct9::cudnnStatus_t {
    use attributi::*;
    let d = back(descriptor)?;

    // --- una convenzione dell'API che sbagliarla costa caro ----------------
    // Quando l'attributo e' di tipo "descrittore", il chiamante ha GIA' creato
    // i descrittori di destinazione e passa il loro elenco: cuDNN deve
    // COPIARCI DENTRO il contenuto, non restituire i propri puntatori.
    // cudnn_frontend riempie un vettore temporaneo e poi tiene i SUOI
    // descrittori: scrivendoci dentro un puntatore, quelli che tiene restano
    // vuoti e ogni lettura successiva fallisce. Misurato il 2 ago 2026: si
    // vedeva come «unable to find an engine», con la lettura dell'attributo
    // 300 (il motore di una configurazione) che tornava errore quattro volte.
    if attribute_type.0 == TIPO_DESCRITTORE {
        // I risultati dell'euristica li fabbrichiamo qui, riempiendo le
        // configurazioni del chiamante con il grafo da eseguire.
        if d.tipo == EURISTICA && attribute_name.0 == EU_RISULTATI {
            let grafo = match d.attributi.get(&EU_RISULTATI) {
                Some(a) if a.quanti > 0 => {
                    let mut g = 0u64;
                    ptr::copy_nonoverlapping(a.byte.as_ptr(), &mut g as *mut u64 as *mut u8, 8);
                    g
                }
                _ => return Err(cudnnError_t::NOT_SUPPORTED),
            };
            if !element_count.is_null() {
                *element_count = 1;
            }
            if requested_element_count >= 1 && !array_of_elements.is_null() {
                let dest = *(array_of_elements as *const ct9::cudnnBackendDescriptor_t);
                // La configurazione riceve una copia PROPRIA del grafo: chi
                // l'ha costruito puo' liberarlo quando vuole.
                let copia = clona_profondo(grafo as *const DescrittoreBackend);
                let cfg = back(dest)?;
                cfg.posseduti.push(copia);
                cfg.attributi.insert(
                    GRAFO_PRIVATO,
                    Attributo {
                        tipo: TIPO_INT64,
                        quanti: 1,
                        byte: (copia as u64).to_ne_bytes().to_vec(),
                    },
                );
                cfg.finalizzato = true;
            }
            return Ok(());
        }
        // Il motore di una configurazione fabbricata da noi: si riempie il
        // descrittore del chiamante col grafo, che e' l'unica cosa che serve.
        if d.tipo == CONFIG_MOTORE
            && attribute_name.0 == CFG_MOTORE
            && d.attributi.contains_key(&GRAFO_PRIVATO)
        {
            let g = d.attributi.get(&GRAFO_PRIVATO).unwrap().byte.clone();
            if !element_count.is_null() {
                *element_count = 1;
            }
            if requested_element_count >= 1 && !array_of_elements.is_null() {
                let dest = *(array_of_elements as *const ct9::cudnnBackendDescriptor_t);
                let motore = back(dest)?;
                motore
                    .attributi
                    .insert(M_GRAFO, Attributo { tipo: TIPO_DESCRITTORE, quanti: 1, byte: g });
                motore.finalizzato = true;
            }
            return Ok(());
        }
        // Caso generale: si copia il contenuto dei descrittori memorizzati
        // dentro quelli che il chiamante ha preparato.
        if let Some(a) = d.attributi.get(&attribute_name.0) {
            let quanti = a.quanti;
            if !element_count.is_null() {
                *element_count = quanti;
            }
            let da_fare = requested_element_count.min(quanti).max(0) as usize;
            if da_fare > 0 && !array_of_elements.is_null() {
                let sorgenti: Vec<u64> = (0..da_fare)
                    .map(|i| {
                        let mut x = 0u64;
                        ptr::copy_nonoverlapping(
                            a.byte.as_ptr().add(i * 8),
                            &mut x as *mut u64 as *mut u8,
                            8,
                        );
                        x
                    })
                    .collect();
                for (i, src) in sorgenti.iter().enumerate() {
                    let dest = *(array_of_elements as *const ct9::cudnnBackendDescriptor_t).add(i);
                    let sorgente = &*(*src as *const DescrittoreBackend);
                    let copia: Vec<(u32, Attributo)> = sorgente
                        .attributi
                        .iter()
                        .map(|(k, v)| {
                            (*k, Attributo { tipo: v.tipo, quanti: v.quanti, byte: v.byte.clone() })
                        })
                        .collect();
                    let d2 = back(dest)?;
                    for (k, v) in copia {
                        d2.attributi.insert(k, v);
                    }
                    d2.finalizzato = true;
                }
            }
            return Ok(());
        }
        // Un elenco di sotto-descrittori che non abbiamo (manopole, note, info)
        // e' una lista VUOTA, non un errore: cudnn_frontend le interroga sempre
        // e su un errore scarta la configurazione. Era questo a far fallire
        // tutto con «unable to find an engine»: l'attributo 302 (le manopole di
        // una configurazione) chiesto come descrittore. Misurato il 2 ago 2026.
        if !element_count.is_null() {
            *element_count = 0;
        }
        return Ok(());
    }

    // --- attributi normali -------------------------------------------------
    if let Some(a) = d.attributi.get(&attribute_name.0) {
        if a.tipo != attribute_type.0 {
            return Err(cudnnError_t::BAD_PARAM);
        }
        if !element_count.is_null() {
            *element_count = a.quanti;
        }
        if requested_element_count > 0 && !array_of_elements.is_null() {
            let dim = if a.quanti > 0 { a.byte.len() / a.quanti as usize } else { 0 };
            let quanti = (requested_element_count.min(a.quanti)).max(0) as usize;
            if dim > 0 && quanti > 0 {
                ptr::copy_nonoverlapping(
                    a.byte.as_ptr(),
                    array_of_elements as *mut u8,
                    quanti * dim,
                );
            }
        }
        return Ok(());
    }

    // --- risposte sintetiche, solo se non c'e' niente di memorizzato --------
    let sintetico: Option<i64> = match (d.tipo, attribute_name.0) {
        (GRAFO, G_NUMERO_MOTORI) => Some(1), // un solo motore: il nostro
        (MOTORE, M_INDICE) => Some(0),
        (CONFIG_MOTORE, CFG_SPAZIO) => Some(0),
        _ => None,
    };
    if let Some(v) = sintetico {
        if !element_count.is_null() {
            *element_count = 1;
        }
        if requested_element_count >= 1 && !array_of_elements.is_null() {
            *(array_of_elements as *mut i64) = v;
        }
        return Ok(());
    }
    // Note numeriche, di comportamento e manopole: nessuna. Lista vuota, non un
    // errore — cudnn_frontend le interroga sempre.
    if matches!(
        (d.tipo, attribute_name.0),
        (MOTORE, M_NOTE_NUMERICHE)
            | (MOTORE, M_NOTE_COMPORTAMENTO)
            | (MOTORE, M_INFO_MANOPOLE)
            | (CONFIG_MOTORE, CFG_MANOPOLE)
    ) {
        if !element_count.is_null() {
            *element_count = 0;
        }
        return Ok(());
    }
    Err(cudnnError_t::NOT_SUPPORTED)
})();
    traccia_backend("leggi", back(descriptor).map(|d| d.tipo).unwrap_or(9999), attribute_name.0, &esito);
    esito
}

/// Dal piano risale al grafo, e dal grafo all'unica operazione.
unsafe fn operazione_del_piano(
    piano: &DescrittoreBackend,
) -> Result<*mut DescrittoreBackend, cudnnError_t> {
    use attributi::*;
    let cfg = piano.primo_descrittore(P_CONFIG).ok_or(cudnnError_t::BAD_PARAM)?;
    passo("config del piano");
    // Due strade: la configurazione l'abbiamo fabbricata noi, e allora il
    // grafo sta nella chiave privata; oppure l'ha costruita cudnn_frontend
    // col suo motore (percorso di riserva).
    let grafo = if let Some(g) = (*cfg).interi(GRAFO_PRIVATO).and_then(|v| v.first().copied()) {
        passo("grafo dalla chiave privata");
        g as *mut DescrittoreBackend
    } else {
        passo("grafo dal motore");
        let motore = (*cfg).primo_descrittore(CFG_MOTORE).ok_or(cudnnError_t::BAD_PARAM)?;
        (*motore).primo_descrittore(M_GRAFO).ok_or(cudnnError_t::BAD_PARAM)?
    };
    let ops = (*grafo).puntatori(G_OPS).ok_or(cudnnError_t::BAD_PARAM)?;
    passo("operazioni del grafo");
    if ops.len() != 1 {
        // Un grafo con piu' operazioni e' una fusione: qui non si sa farla, e
        // dirlo e' meglio che eseguirne una a caso.
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    Ok(ops[0] as *mut DescrittoreBackend)
}

/// Costruisce un descrittore MIOpen da un tensore del grafo.
unsafe fn descrittore_da_tensore(
    t: &DescrittoreBackend,
) -> Result<(miopenTensorDescriptor_t, i64), cudnnError_t> {
    use attributi::*;
    let dim = t.interi(T_DIM).ok_or(cudnnError_t::BAD_PARAM)?;
    let passi = t.interi(T_PASSI).ok_or(cudnnError_t::BAD_PARAM)?;
    let id = *t.interi(T_ID).ok_or(cudnnError_t::BAD_PARAM)?.first().unwrap_or(&0);
    let tipo_cudnn = t
        .attributi
        .get(&T_TIPO)
        .map(|a| {
            let mut v = 0i32;
            ptr::copy_nonoverlapping(a.byte.as_ptr(), &mut v as *mut i32 as *mut u8, 4);
            v
        })
        .unwrap_or(0);
    let tipo = tipo_dato(ct9::cudnnDataType_t(tipo_cudnn as u32))?;
    let mut d: miopenTensorDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateTensorDescriptor(&mut d))?;
    let dd: Vec<i32> = dim.iter().map(|x| *x as i32).collect();
    let pp: Vec<i32> = passi.iter().map(|x| *x as i32).collect();
    miopen_to_cudnn_status(miopenSetTensorDescriptor(
        d,
        tipo,
        dd.len() as i32,
        dd.as_ptr(),
        pp.as_ptr(),
    ))?;
    Ok((d, id))
}

/// Costruisce un descrittore di convoluzione MIOpen da quello del grafo.
unsafe fn conv_da_grafo(
    c: &DescrittoreBackend,
) -> Result<miopenConvolutionDescriptor_t, cudnnError_t> {
    use attributi::*;
    let n = *c.interi(C_DIM_SPAZIALI).ok_or(cudnnError_t::BAD_PARAM)?.first().unwrap_or(&2) as i32;
    let bordi = c.interi(C_BORDO_PRIMA).ok_or(cudnnError_t::BAD_PARAM)?;
    let passi = c.interi(C_PASSI).ok_or(cudnnError_t::BAD_PARAM)?;
    let dilat = c.interi(C_DILATAZIONI).ok_or(cudnnError_t::BAD_PARAM)?;
    let b: Vec<i32> = bordi.iter().map(|x| *x as i32).collect();
    let s: Vec<i32> = passi.iter().map(|x| *x as i32).collect();
    let d: Vec<i32> = dilat.iter().map(|x| *x as i32).collect();
    let mut cd: miopenConvolutionDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateConvolutionDescriptor(&mut cd))?;
    miopen_to_cudnn_status(miopenInitConvolutionNdDescriptor(
        cd,
        n,
        b.as_ptr(),
        s.as_ptr(),
        d.as_ptr(),
        miopenConvolutionMode_t::miopenConvolution,
    ))?;
    Ok(cd)
}

/// Copia un tensore del grafo in una struttura autosufficiente.
unsafe fn risolvi_tensore(t: &DescrittoreBackend) -> Result<TensoreRisolto, cudnnError_t> {
    use attributi::*;
    let dim = t.interi(T_DIM).ok_or(cudnnError_t::BAD_PARAM)?;
    let passi = t.interi(T_PASSI).ok_or(cudnnError_t::BAD_PARAM)?;
    let id = *t.interi(T_ID).ok_or(cudnnError_t::BAD_PARAM)?.first().unwrap_or(&0);
    let tipo_cudnn = t
        .attributi
        .get(&T_TIPO)
        .map(|a| {
            let mut v = 0i32;
            ptr::copy_nonoverlapping(a.byte.as_ptr(), &mut v as *mut i32 as *mut u8, 4);
            v
        })
        .unwrap_or(0);
    Ok(TensoreRisolto {
        dim: dim.iter().map(|x| *x as i32).collect(),
        passi: passi.iter().map(|x| *x as i32).collect(),
        tipo: tipo_dato(ct9::cudnnDataType_t(tipo_cudnn as u32))?,
        id,
    })
}

/// Costruisce un descrittore MIOpen da un tensore gia' risolto.
unsafe fn descrittore_risolto(
    t: &TensoreRisolto,
) -> Result<miopenTensorDescriptor_t, cudnnError_t> {
    let mut d: miopenTensorDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateTensorDescriptor(&mut d))?;
    miopen_to_cudnn_status(miopenSetTensorDescriptor(
        d,
        t.tipo,
        t.dim.len() as i32,
        t.dim.as_ptr(),
        t.passi.as_ptr(),
    ))?;
    Ok(d)
}

/// Risolve il piano: percorre una volta sola la catena di descrittori del
/// chiamante e ne copia il contenuto, poi chiede a MIOpen lo spazio di lavoro.
unsafe fn risolvi_piano(piano: &DescrittoreBackend) -> Result<PianoRisolto, cudnnError_t> {
    use attributi::*;
    let op = operazione_del_piano(piano)?;
    passo("r: operazione");
    let (tipo_op, tx, tw, ty, tc) = pezzi_operazione(&*op)?;
    passo("r: pezzi");
    let x = risolvi_tensore(&*tx)?;
    let w = risolvi_tensore(&*tw)?;
    let y = risolvi_tensore(&*ty)?;
    passo("r: tensori");
    let c = &*tc;
    let dim_spaziali =
        *c.interi(C_DIM_SPAZIALI).ok_or(cudnnError_t::BAD_PARAM)?.first().unwrap_or(&2) as i32;
    let bordi: Vec<i32> = c
        .interi(C_BORDO_PRIMA)
        .ok_or(cudnnError_t::BAD_PARAM)?
        .iter()
        .map(|v| *v as i32)
        .collect();
    let passi_conv: Vec<i32> = c
        .interi(C_PASSI)
        .ok_or(cudnnError_t::BAD_PARAM)?
        .iter()
        .map(|v| *v as i32)
        .collect();
    let dilatazioni: Vec<i32> = c
        .interi(C_DILATAZIONI)
        .ok_or(cudnnError_t::BAD_PARAM)?
        .iter()
        .map(|v| *v as i32)
        .collect();

    let handle_p = piano
        .puntatori(400)
        .and_then(|v| v.first().copied())
        .ok_or(cudnnError_t::BAD_PARAM)?;
    let mh = handle_sicuro(handle_p as ct9::cudnnHandle_t)?;
    passo("r: handle");
    let xd = descrittore_risolto(&x)?;
    let wd = descrittore_risolto(&w)?;
    let yd = descrittore_risolto(&y)?;
    passo("r: descrittori");
    let mut cd: miopenConvolutionDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateConvolutionDescriptor(&mut cd))?;
    miopen_to_cudnn_status(miopenInitConvolutionNdDescriptor(
        cd,
        dim_spaziali,
        bordi.as_ptr(),
        passi_conv.as_ptr(),
        dilatazioni.as_ptr(),
        miopenConvolutionMode_t::miopenConvolution,
    ))?;
    let mut spazio: usize = 0;
    let rc = match tipo_op {
        OP_CONV_AVANTI => miopenConvolutionForwardGetWorkSpaceSize(mh, wd, xd, cd, yd, &mut spazio),
        OP_CONV_DATI => miopenConvolutionBackwardDataGetWorkSpaceSize(mh, yd, wd, cd, xd, &mut spazio),
        _ => miopenConvolutionBackwardWeightsGetWorkSpaceSize(mh, yd, xd, cd, wd, &mut spazio),
    };
    miopenDestroyTensorDescriptor(xd).ok();
    miopenDestroyTensorDescriptor(wd).ok();
    miopenDestroyTensorDescriptor(yd).ok();
    miopenDestroyConvolutionDescriptor(cd).ok();
    miopen_to_cudnn_status(rc)?;
    passo("r: spazio ottenuto");
    Ok(PianoRisolto {
        tipo_op,
        x,
        w,
        y,
        bordi,
        passi_conv,
        dilatazioni,
        dim_spaziali,
        spazio,
    })
}

/// Copia profonda di un descrittore, comprese le catene che riferisce.
///
/// PERCHE'. In cuDNN vera, `cudnnBackendFinalize` CATTURA il contenuto dei
/// descrittori riferiti: da quel momento il chiamante puo' distruggerli.
/// cudnn_frontend conta su questo e passa tensori temporanei, che distrugge
/// appena costruita l'operazione. Tenendo solo i puntatori, alla finalizzazione
/// del piano si leggeva memoria gia' liberata. Misurato il 2 ago 2026.
///
/// Il nodo copiato possiede le copie dei figli e le libera con se'.
unsafe fn clona_profondo(src: *const DescrittoreBackend) -> *mut DescrittoreBackend {
    if src.is_null() {
        return ptr::null_mut();
    }
    let s = &*src;
    let mut copia = DescrittoreBackend::nuovo(s.tipo);
    copia.finalizzato = s.finalizzato;
    for (chiave, a) in s.attributi.iter() {
        if a.tipo == attributi::TIPO_DESCRITTORE && a.quanti > 0 {
            let mut byte = Vec::with_capacity(a.byte.len());
            for i in 0..a.quanti as usize {
                let mut punt = 0u64;
                ptr::copy_nonoverlapping(
                    a.byte.as_ptr().add(i * 8),
                    &mut punt as *mut u64 as *mut u8,
                    8,
                );
                let figlio = clona_profondo(punt as *const DescrittoreBackend);
                copia.posseduti.push(figlio);
                byte.extend_from_slice(&(figlio as u64).to_ne_bytes());
            }
            copia.attributi.insert(*chiave, Attributo { tipo: a.tipo, quanti: a.quanti, byte });
        } else {
            copia.attributi.insert(
                *chiave,
                Attributo { tipo: a.tipo, quanti: a.quanti, byte: a.byte.clone() },
            );
        }
    }
    Box::into_raw(Box::new(copia))
}

/// Sostituisce ogni attributo che riferisce descrittori con copie possedute.
unsafe fn cattura_riferimenti(d: &mut DescrittoreBackend) {
    let chiavi: Vec<u32> = d
        .attributi
        .iter()
        .filter(|(_, a)| a.tipo == attributi::TIPO_DESCRITTORE && a.quanti > 0)
        .map(|(k, _)| *k)
        .collect();
    for chiave in chiavi {
        let a = match d.attributi.get(&chiave) {
            Some(a) => Attributo { tipo: a.tipo, quanti: a.quanti, byte: a.byte.clone() },
            None => continue,
        };
        let mut byte = Vec::with_capacity(a.byte.len());
        let mut nuovi = Vec::new();
        for i in 0..a.quanti as usize {
            let mut punt = 0u64;
            ptr::copy_nonoverlapping(a.byte.as_ptr().add(i * 8), &mut punt as *mut u64 as *mut u8, 8);
            let figlio = clona_profondo(punt as *const DescrittoreBackend);
            nuovi.push(figlio);
            byte.extend_from_slice(&(figlio as u64).to_ne_bytes());
        }
        d.posseduti.extend(nuovi);
        d.attributi.insert(chiave, Attributo { tipo: a.tipo, quanti: a.quanti, byte });
    }
}

pub(crate) unsafe extern "C" fn cudnnBackendFinalize(
    descriptor: ct9::cudnnBackendDescriptor_t,
) -> ct9::cudnnStatus_t {
    let esito = (|| -> ct9::cudnnStatus_t {
    use attributi::*;
    let d = back(descriptor)?;
    match d.tipo {
        EURISTICA => {
            // Non si fabbricano descrittori nostri: sarebbero da tenere in
            // vita e cudnn_frontend comunque li scarta (vedi la nota sulla
            // copia dei contenuti in cudnnBackendGetAttribute). Si annota
            // solo il grafo; le configurazioni si riempiono alla lettura.
            let grafo = d.primo_descrittore(EU_GRAFO).ok_or(cudnnError_t::BAD_PARAM)?;
            d.attributi.insert(
                EU_RISULTATI,
                Attributo {
                    tipo: TIPO_DESCRITTORE,
                    quanti: 1,
                    byte: (grafo as u64).to_ne_bytes().to_vec(),
                },
            );
        }
        PIANO => {
            // Qui si risolve TUTTO e se ne tiene una copia: dopo, i descrittori
            // del chiamante possono sparire senza conseguenze.
            let risolto = risolvi_piano(d)?;
            let spazio = risolto.spazio;
            let p = Box::into_raw(Box::new(risolto));
            d.attributi.insert(
                PIANO_PRIVATO,
                Attributo { tipo: TIPO_INT64, quanti: 1, byte: (p as u64).to_ne_bytes().to_vec() },
            );
            d.attributi.insert(
                P_SPAZIO,
                Attributo { tipo: TIPO_INT64, quanti: 1, byte: (spazio as i64).to_ne_bytes().to_vec() },
            );
        }
        OP_CONV_AVANTI | OP_CONV_DATI | OP_CONV_PESI | GRAFO => {
            // Da qui in poi questo descrittore non dipende piu' da nessun altro.
            cattura_riferimenti(d);
        }
        _ => {}
    }
    d.finalizzato = true;
    Ok(())
})();
    traccia_backend("finalizza", back(descriptor).map(|d| d.tipo).unwrap_or(9999), 0, &esito);
    esito
}

unsafe fn spazio_del_piano(piano: &DescrittoreBackend) -> Result<usize, cudnnError_t> {
    use attributi::*;
    let op = operazione_del_piano(piano)?;
    let handle_p = piano
        .puntatori(400) // CUDNN_ATTR_EXECUTION_PLAN_HANDLE
        .and_then(|v| v.first().copied())
        .ok_or(cudnnError_t::BAD_PARAM)?;
    passo("handle del piano");
    let mh = handle_sicuro(handle_p as ct9::cudnnHandle_t)?;
    passo("handle risolto");
    let (tipo, x, w, y, conv) = pezzi_operazione(&*op)?;
    passo("pezzi (spazio)");
    let (xd, _) = descrittore_da_tensore(&*x)?;
    let (wd, _) = descrittore_da_tensore(&*w)?;
    let (yd, _) = descrittore_da_tensore(&*y)?;
    let cd = conv_da_grafo(&*conv)?;
    passo("descrittori (spazio)");
    let mut n: usize = 0;
    let rc = match tipo {
        OP_CONV_AVANTI => miopenConvolutionForwardGetWorkSpaceSize(mh, wd, xd, cd, yd, &mut n),
        OP_CONV_DATI => miopenConvolutionBackwardDataGetWorkSpaceSize(mh, yd, wd, cd, xd, &mut n),
        _ => miopenConvolutionBackwardWeightsGetWorkSpaceSize(mh, yd, xd, cd, wd, &mut n),
    };
    miopen_to_cudnn_status(rc)?;
    miopenDestroyTensorDescriptor(xd).ok();
    miopenDestroyTensorDescriptor(wd).ok();
    miopenDestroyTensorDescriptor(yd).ok();
    miopenDestroyConvolutionDescriptor(cd).ok();
    Ok(n)
}

/// Estrae dall'operazione i tre tensori e la convoluzione, nell'ordine
/// (ingresso, pesi, uscita) qualunque sia il verso.
unsafe fn pezzi_operazione(
    op: &DescrittoreBackend,
) -> Result<
    (
        u32,
        *mut DescrittoreBackend,
        *mut DescrittoreBackend,
        *mut DescrittoreBackend,
        *mut DescrittoreBackend,
    ),
    cudnnError_t,
> {
    use attributi::*;
    let (ax, aw, ay, ac) = match op.tipo {
        OP_CONV_AVANTI => (OP_AV_X, OP_AV_W, OP_AV_Y, OP_AV_CONV),
        OP_CONV_DATI => (OP_DA_DX, OP_DA_W, OP_DA_DY, OP_DA_CONV),
        OP_CONV_PESI => (OP_PE_X, OP_PE_DW, OP_PE_DY, OP_PE_CONV),
        _ => return Err(cudnnError_t::NOT_SUPPORTED),
    };
    Ok((
        op.tipo,
        op.primo_descrittore(ax).ok_or(cudnnError_t::BAD_PARAM)?,
        op.primo_descrittore(aw).ok_or(cudnnError_t::BAD_PARAM)?,
        op.primo_descrittore(ay).ok_or(cudnnError_t::BAD_PARAM)?,
        op.primo_descrittore(ac).ok_or(cudnnError_t::BAD_PARAM)?,
    ))
}


pub(crate) unsafe extern "C" fn cudnnBackendExecute(
    handle: ct9::cudnnHandle_t,
    execution_plan: ct9::cudnnBackendDescriptor_t,
    variant_pack: ct9::cudnnBackendDescriptor_t,
) -> ct9::cudnnStatus_t {
    let esito = (|| -> ct9::cudnnStatus_t {
    use attributi::*;
    let piano = back(execution_plan)?;
    let varianti = back(variant_pack)?;

    // Tutto quello che serve e' gia' stato copiato dentro il piano quando e'
    // stato finalizzato: qui non si tocca piu' nessun descrittore del
    // chiamante, tranne il pacchetto di varianti, che e' vivo per definizione.
    let punt_risolto = piano
        .interi(PIANO_PRIVATO)
        .and_then(|v| v.first().copied())
        .ok_or(cudnnError_t::BAD_PARAM)?;
    if punt_risolto == 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let r = &*(punt_risolto as *const PianoRisolto);

    let xd = descrittore_risolto(&r.x)?;
    let wd = descrittore_risolto(&r.w)?;
    let yd = descrittore_risolto(&r.y)?;
    let mut cd: miopenConvolutionDescriptor_t = mem::zeroed();
    miopen_to_cudnn_status(miopenCreateConvolutionDescriptor(&mut cd))?;
    miopen_to_cudnn_status(miopenInitConvolutionNdDescriptor(
        cd,
        r.dim_spaziali,
        r.bordi.as_ptr(),
        r.passi_conv.as_ptr(),
        r.dilatazioni.as_ptr(),
        miopenConvolutionMode_t::miopenConvolution,
    ))?;

    // Il pacchetto di varianti associa gli identificatori dei tensori ai
    // puntatori veri.
    let ids = varianti.interi(V_ID).ok_or(cudnnError_t::BAD_PARAM)?;
    let punt = varianti.puntatori(V_PUNTATORI).ok_or(cudnnError_t::BAD_PARAM)?;
    let trova = |id: i64| -> Option<*mut ::core::ffi::c_void> {
        ids.iter().position(|x| *x == id).and_then(|i| punt.get(i).copied())
    };
    let px = trova(r.x.id).ok_or(cudnnError_t::BAD_PARAM)?;
    let pw = trova(r.w.id).ok_or(cudnnError_t::BAD_PARAM)?;
    let py = trova(r.y.id).ok_or(cudnnError_t::BAD_PARAM)?;
    let spazio = varianti
        .puntatori(V_SPAZIO)
        .and_then(|v| v.first().copied())
        .unwrap_or(ptr::null_mut());

    let uno = 1.0f32;
    let zero = 0.0f32;
    let a = &uno as *const f32 as *const ::core::ffi::c_void;
    let b = &zero as *const f32 as *const ::core::ffi::c_void;

    // Da qui in poi e' la stessa strada della v7, gia' verificata.
    let esito = match r.tipo_op {
        OP_CONV_AVANTI => miopen_to_cudnn_status(convolution_forward(
            ctx(handle), a, xd, px, wd, pw, cd, mem::transmute(0u32),
            spazio, r.spazio, b, yd, py,
        )),
        OP_CONV_DATI => cudnnConvolutionBackwardData(
            handle, a, mem::transmute(wd), pw, mem::transmute(yd), py,
            mem::transmute(cd), 0, spazio, r.spazio, b, mem::transmute(xd), px,
        ),
        _ => cudnnConvolutionBackwardFilter(
            handle, a, mem::transmute(xd), px, mem::transmute(yd), py,
            mem::transmute(cd), 0, spazio, r.spazio, b, mem::transmute(wd), pw,
        ),
    };
    miopenDestroyTensorDescriptor(xd).ok();
    miopenDestroyTensorDescriptor(wd).ok();
    miopenDestroyTensorDescriptor(yd).ok();
    miopenDestroyConvolutionDescriptor(cd).ok();
    esito
})();
    traccia_backend("esegui", 0, 0, &esito);
    esito
}

// ---------------------------------------------------------------------------
// Le tre ricerche "Ex" — servono a `torch.backends.cudnn.benchmark = True`
//
// Sullo stub PyTorch non riceveva un errore gestibile: leggeva `perfResults[0]`
// mai scritto e il processo cadeva con access violation. Una modalita' comune
// come benchmark=True non puo' essere una mina.
//
// La corrispondenza con MIOpen e' diretta: le stesse Find che il traduttore usa
// gia' internamente, qui con i buffer che il chiamante fornisce. Si riferisce un
// solo algoritmo, per lo stesso motivo delle `_v7`: le impl di convoluzione
// ignorano comunque l'algoritmo richiesto.
// ---------------------------------------------------------------------------

pub(crate) unsafe extern "C" fn cudnnFindConvolutionForwardAlgorithmEx(
    handle: ct9::cudnnHandle_t,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    requested_algo_count: ::core::ffi::c_int,
    returned_algo_count: *mut ::core::ffi::c_int,
    perf_results: *mut ct9::cudnnConvolutionFwdAlgoPerf_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
) -> ct9::cudnnStatus_t {
    if requested_algo_count <= 0 || returned_algo_count.is_null() || perf_results.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    let mut trovati: ::core::ffi::c_int = 0;
    let mut perf: [miopenConvAlgoPerf_t; 5] = mem::zeroed();
    miopenFindConvolutionForwardAlgorithm(
        mh,
        mem::transmute(x_desc),
        x,
        mem::transmute(w_desc),
        w,
        mem::transmute(conv_desc),
        mem::transmute(y_desc),
        y,
        perf.len() as ::core::ffi::c_int,
        &mut trovati,
        perf.as_mut_ptr(),
        workspace,
        workspace_size,
        false,
    )?;
    if trovati <= 0 {
        *returned_algo_count = 0;
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    *returned_algo_count = 1;
    *perf_results = algo_perf_to_cudnn(perf[0]);
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnFindConvolutionBackwardDataAlgorithmEx(
    handle: ct9::cudnnHandle_t,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::core::ffi::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    requested_algo_count: ::core::ffi::c_int,
    returned_algo_count: *mut ::core::ffi::c_int,
    perf_results: *mut ct9::cudnnConvolutionBwdDataAlgoPerf_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
) -> ct9::cudnnStatus_t {
    if requested_algo_count <= 0 || returned_algo_count.is_null() || perf_results.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    let mut trovati: ::core::ffi::c_int = 0;
    let mut perf: [miopenConvAlgoPerf_t; 5] = mem::zeroed();
    miopenFindConvolutionBackwardDataAlgorithm(
        mh,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(w_desc),
        w,
        mem::transmute(conv_desc),
        mem::transmute(dx_desc),
        dx,
        perf.len() as ::core::ffi::c_int,
        &mut trovati,
        perf.as_mut_ptr(),
        workspace,
        workspace_size,
        false,
    )?;
    if trovati <= 0 {
        *returned_algo_count = 0;
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    *returned_algo_count = 1;
    *perf_results = ct9::cudnnConvolutionBwdDataAlgoPerf_t {
        algo: cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        status: Ok(()),
        time: perf[0].time,
        memory: perf[0].memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0, 0, 0],
    };
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnFindConvolutionBackwardFilterAlgorithmEx(
    handle: ct9::cudnnHandle_t,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    dw_desc: ct9::cudnnFilterDescriptor_t,
    dw: *mut ::core::ffi::c_void,
    requested_algo_count: ::core::ffi::c_int,
    returned_algo_count: *mut ::core::ffi::c_int,
    perf_results: *mut ct9::cudnnConvolutionBwdFilterAlgoPerf_t,
    workspace: *mut ::core::ffi::c_void,
    workspace_size: usize,
) -> ct9::cudnnStatus_t {
    if requested_algo_count <= 0 || returned_algo_count.is_null() || perf_results.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    let mh = ctx_handle(handle);
    let mut trovati: ::core::ffi::c_int = 0;
    let mut perf: [miopenConvAlgoPerf_t; 5] = mem::zeroed();
    miopenFindConvolutionBackwardWeightsAlgorithm(
        mh,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(x_desc),
        x,
        mem::transmute(conv_desc),
        mem::transmute(dw_desc),
        dw,
        perf.len() as ::core::ffi::c_int,
        &mut trovati,
        perf.as_mut_ptr(),
        workspace,
        workspace_size,
        false,
    )?;
    if trovati <= 0 {
        *returned_algo_count = 0;
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    *returned_algo_count = 1;
    *perf_results = ct9::cudnnConvolutionBwdFilterAlgoPerf_t {
        algo: cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        status: Ok(()),
        time: perf[0].time,
        memory: perf[0].memory,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0, 0, 0],
    };
    Ok(())
}

// ---------------------------------------------------------------------------
// Batch norm, varianti "Ex"
//
// PyTorch usa SOLO queste, mai le tre non-Ex che erano gia' implementate qui.
// Le Ex aggiungono la possibilita' di fondere un'attivazione (e una somma)
// dentro la normalizzazione: e' quello che dice `bnOps`. Con
// `bnOps = CUDNN_BATCHNORM_OPS_BN` (0), cioe' solo normalizzazione, sono
// esattamente equivalenti alle non-Ex, e qui si delega.
//
// Le altre due modalita' non si fingono: si restituisce NOT_SUPPORTED, cosi'
// PyTorch ripiega sul proprio percorso invece di ricevere numeri sbagliati.
// MIOpen non ha comunque un equivalente fuso.
// ---------------------------------------------------------------------------

const SOLO_NORMALIZZAZIONE: u32 = 0; // CUDNN_BATCHNORM_OPS_BN

pub(crate) unsafe extern "C" fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    _handle: ct9::cudnnHandle_t,
    _mode: ct9::cudnnBatchNormMode_t,
    bn_ops: ct9::cudnnBatchNormOps_t,
    _x_desc: ct9::cudnnTensorDescriptor_t,
    _z_desc: ct9::cudnnTensorDescriptor_t,
    _y_desc: ct9::cudnnTensorDescriptor_t,
    _bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
    _activation_desc: ct9::cudnnActivationDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    if bn_ops.0 != SOLO_NORMALIZZAZIONE {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    // Le non-Ex di MIOpen non vogliono spazio di lavoro.
    *size_in_bytes = 0;
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    _handle: ct9::cudnnHandle_t,
    _mode: ct9::cudnnBatchNormMode_t,
    bn_ops: ct9::cudnnBatchNormOps_t,
    _x_desc: ct9::cudnnTensorDescriptor_t,
    _y_desc: ct9::cudnnTensorDescriptor_t,
    _dy_desc: ct9::cudnnTensorDescriptor_t,
    _dz_desc: ct9::cudnnTensorDescriptor_t,
    _dx_desc: ct9::cudnnTensorDescriptor_t,
    _d_bn_sb_desc: ct9::cudnnTensorDescriptor_t,
    _activation_desc: ct9::cudnnActivationDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    if bn_ops.0 != SOLO_NORMALIZZAZIONE {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    *size_in_bytes = 0;
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    _handle: ct9::cudnnHandle_t,
    _mode: ct9::cudnnBatchNormMode_t,
    bn_ops: ct9::cudnnBatchNormOps_t,
    _activation_desc: ct9::cudnnActivationDescriptor_t,
    _x_desc: ct9::cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    if size_in_bytes.is_null() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    if bn_ops.0 != SOLO_NORMALIZZAZIONE {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    // Lo spazio di riserva serve alla Ex per passare stato dall'avanti
    // all'indietro quando c'e' un'attivazione fusa. Senza fusione non serve.
    *size_in_bytes = 0;
    Ok(())
}

pub(crate) unsafe extern "C" fn cudnnBatchNormalizationForwardTrainingEx(
    handle: ct9::cudnnHandle_t,
    mode: ct9::cudnnBatchNormMode_t,
    bn_ops: ct9::cudnnBatchNormOps_t,
    alpha: *const ::core::ffi::c_void,
    beta: *const ::core::ffi::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    _z_desc: ct9::cudnnTensorDescriptor_t,
    _z: *const ::core::ffi::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::core::ffi::c_void,
    bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *const ::core::ffi::c_void,
    bn_bias: *const ::core::ffi::c_void,
    exp_avg_factor: f64,
    running_mean: *mut ::core::ffi::c_void,
    running_var: *mut ::core::ffi::c_void,
    epsilon: f64,
    save_mean: *mut ::core::ffi::c_void,
    save_inv_var: *mut ::core::ffi::c_void,
    _activation_desc: ct9::cudnnActivationDescriptor_t,
    _workspace: *mut ::core::ffi::c_void,
    _workspace_size: usize,
    _reserve_space: *mut ::core::ffi::c_void,
    _reserve_space_size: usize,
) -> ct9::cudnnStatus_t {
    if bn_ops.0 != SOLO_NORMALIZZAZIONE {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    cudnnBatchNormalizationForwardTraining(
        handle,
        mode.0 as i32,
        alpha as *mut ::std::os::raw::c_void,
        beta as *mut ::std::os::raw::c_void,
        x_desc,
        x,
        y_desc,
        y,
        bn_sbmv_desc,
        bn_scale as *mut ::std::os::raw::c_void,
        bn_bias as *mut ::std::os::raw::c_void,
        exp_avg_factor,
        running_mean,
        running_var,
        epsilon,
        save_mean,
        save_inv_var,
    )
}

pub(crate) unsafe extern "C" fn cudnnBatchNormalizationBackwardEx(
    handle: ct9::cudnnHandle_t,
    mode: ct9::cudnnBatchNormMode_t,
    bn_ops: ct9::cudnnBatchNormOps_t,
    alpha_data_diff: *const ::core::ffi::c_void,
    beta_data_diff: *const ::core::ffi::c_void,
    alpha_param_diff: *const ::core::ffi::c_void,
    beta_param_diff: *const ::core::ffi::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::core::ffi::c_void,
    _y_desc: ct9::cudnnTensorDescriptor_t,
    _y: *const ::core::ffi::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::core::ffi::c_void,
    _dz_desc: ct9::cudnnTensorDescriptor_t,
    _dz: *mut ::core::ffi::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::core::ffi::c_void,
    d_bn_sb_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *const ::core::ffi::c_void,
    _bn_bias: *const ::core::ffi::c_void,
    d_bn_scale: *mut ::core::ffi::c_void,
    d_bn_bias: *mut ::core::ffi::c_void,
    epsilon: f64,
    saved_mean: *const ::core::ffi::c_void,
    saved_inv_var: *const ::core::ffi::c_void,
    _activation_desc: ct9::cudnnActivationDescriptor_t,
    _workspace: *mut ::core::ffi::c_void,
    _workspace_size: usize,
    _reserve_space: *mut ::core::ffi::c_void,
    _reserve_space_size: usize,
) -> ct9::cudnnStatus_t {
    if bn_ops.0 != SOLO_NORMALIZZAZIONE {
        return Err(cudnnError_t::NOT_SUPPORTED);
    }
    cudnnBatchNormalizationBackward(
        handle,
        mode.0 as i32,
        alpha_data_diff,
        beta_data_diff,
        alpha_param_diff,
        beta_param_diff,
        x_desc,
        x,
        dy_desc,
        dy,
        dx_desc,
        dx,
        d_bn_sb_desc,
        bn_scale,
        d_bn_scale,
        d_bn_bias,
        epsilon,
        saved_mean,
        saved_inv_var,
    )
}

pub(crate) unsafe extern "C" fn cudnnSetTensorNdDescriptor(
    tensor_desc: ct9::cudnnTensorDescriptor_t,
    data_type: ct9::cudnnDataType_t,
    nb_dims: ::core::ffi::c_int,
    dim_a: *const ::core::ffi::c_int,
    stride_a: *const ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if dim_a.is_null() || stride_a.is_null() || nb_dims <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenSetTensorDescriptor(
        mem::transmute(tensor_desc),
        tipo_dato(data_type)?,
        nb_dims,
        dim_a,
        stride_a,
    ))
}

/// Imposta un descrittore da un elenco di dimensioni.
///
/// `miopenSetNdTensorDescriptorWithLayout` accetta una disposizione (NCHW,
/// NHWC) ma pretende che il numero di dimensioni corrisponda: con NCHW e 3
/// dimensioni risponde «Unsupported layout» (tensor.cpp:387). PyTorch pero'
/// descrive i pesi delle RNN come un tensore a 3 dimensioni [n, 1, 1] dichiarato
/// NCHW, e cosi' ogni LSTM falliva. Misurato il 2 ago 2026.
///
/// Quando le dimensioni non sono quattro si calcolano i passi a mano — il
/// tensore e' comunque compatto — e si usa il setter generico.
unsafe fn imposta_descrittore_nd(
    desc: miopenTensorDescriptor_t,
    tipo: miopenDataType_t,
    layout: miopenTensorLayout_t,
    dim_a: *const ::core::ffi::c_int,
    n: ::core::ffi::c_int,
) -> miopenStatus_t {
    if n == 4 {
        return miopenSetNdTensorDescriptorWithLayout(desc, tipo, layout, dim_a, n);
    }
    if layout != miopenTensorLayout_t::miopenTensorNCHW {
        // Fuori da quattro dimensioni una disposizione diversa da NCHW non ha
        // un significato univoco: meglio dichiararlo che indovinare.
        return Err(miopenError_t::UnsupportedOp);
    }
    let quante = n as usize;
    if quante == 0 || quante > 8 {
        return Err(miopenError_t::BadParm);
    }
    let mut passi = [0i32; 8];
    let mut acc: i64 = 1;
    for i in (0..quante).rev() {
        passi[i] = acc as i32;
        acc *= *dim_a.add(i) as i64;
    }
    miopenSetTensorDescriptor(desc, tipo, n, dim_a, passi.as_ptr())
}

pub(crate) unsafe extern "C" fn cudnnSetTensorNdDescriptorEx(
    tensor_desc: ct9::cudnnTensorDescriptor_t,
    format: ct9::cudnnTensorFormat_t,
    data_type: ct9::cudnnDataType_t,
    nb_dims: ::core::ffi::c_int,
    dim_a: *const ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if dim_a.is_null() || nb_dims <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    // Come per la 4d: anche con disposizione NHWC, MIOpen vuole le lunghezze in
    // ordine NCHW.
    miopen_to_cudnn_status(imposta_descrittore_nd(
        mem::transmute(tensor_desc),
        tipo_dato(data_type)?,
        disposizione(format)?,
        dim_a,
        nb_dims,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetFilterNdDescriptor(
    filter_desc: ct9::cudnnFilterDescriptor_t,
    data_type: ct9::cudnnDataType_t,
    format: ct9::cudnnTensorFormat_t,
    nb_dims: ::core::ffi::c_int,
    filter_dim_a: *const ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if filter_dim_a.is_null() || nb_dims <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(imposta_descrittore_nd(
        mem::transmute(filter_desc),
        tipo_dato(data_type)?,
        disposizione(format)?,
        filter_dim_a,
        nb_dims,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetConvolutionNdDescriptor(
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    array_length: ::core::ffi::c_int,
    pad_a: *const ::core::ffi::c_int,
    filter_stride_a: *const ::core::ffi::c_int,
    dilation_a: *const ::core::ffi::c_int,
    mode: ct9::cudnnConvolutionMode_t,
    _compute_type: ct9::cudnnDataType_t,
) -> ct9::cudnnStatus_t {
    // `array_length` e' il numero di dimensioni SPAZIALI (2 per una conv 2d),
    // che e' esattamente lo `spatialDim` che vuole MIOpen.
    if pad_a.is_null() || filter_stride_a.is_null() || dilation_a.is_null() || array_length <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    miopen_to_cudnn_status(miopenInitConvolutionNdDescriptor(
        mem::transmute(conv_desc),
        array_length,
        pad_a,
        filter_stride_a,
        dilation_a,
        modo_conv(mode)?,
    ))
}

pub(crate) unsafe extern "C" fn cudnnSetConvolutionGroupCount(
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    group_count: ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    miopen_to_cudnn_status(miopenSetConvolutionGroupCount(
        mem::transmute(conv_desc),
        group_count,
    ))
}

pub(crate) unsafe extern "C" fn cudnnGetConvolutionNdForwardOutputDim(
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    input_tensor_desc: ct9::cudnnTensorDescriptor_t,
    filter_desc: ct9::cudnnFilterDescriptor_t,
    nb_dims: ::core::ffi::c_int,
    tensor_output_dim_a: *mut ::core::ffi::c_int,
) -> ct9::cudnnStatus_t {
    if tensor_output_dim_a.is_null() || nb_dims <= 0 {
        return Err(cudnnError_t::BAD_PARAM);
    }
    // cuDNN passa quante dimensioni ha allocato; MIOpen invece RESTITUISCE
    // quante ne ha scritte. Si scrive in un buffer nostro e si ricopia solo se
    // ci sta, altrimenti si sborderebbe la memoria del chiamante.
    let mut quante: ::core::ffi::c_int = 0;
    let mut buffer = [0i32; 8];
    let rc = miopenGetConvolutionNdForwardOutputDim(
        mem::transmute(conv_desc),
        mem::transmute(input_tensor_desc),
        mem::transmute(filter_desc),
        &mut quante,
        buffer.as_mut_ptr(),
    );
    miopen_to_cudnn_status(rc)?;
    if quante > nb_dims || quante as usize > buffer.len() {
        return Err(cudnnError_t::BAD_PARAM);
    }
    ptr::copy_nonoverlapping(buffer.as_ptr(), tensor_output_dim_a, quante as usize);
    Ok(())
}

pub(crate) unsafe fn get_convolution_forward_algorithm_v7(
    handle: &Context,
    x_desc: miopenTensorDescriptor_t,
    w_desc: miopenTensorDescriptor_t,
    conv_desc: miopenConvolutionDescriptor_t,
    y_desc: miopenTensorDescriptor_t,
    requested_algo_count: ::std::os::raw::c_int,
    returned_algo_count: &mut ::std::os::raw::c_int,
    perf_results: &mut cudnnConvolutionFwdAlgoPerf_t,
) -> Result<(), miopenError_t> {
    if requested_algo_count <= 0 {
        return Err(miopenError_t::BadParm);
    }
    let miopen_result =
        get_or_search_convolution_forward_algorithm(handle, x_desc, w_desc, conv_desc, y_desc)?;
    match miopen_result {
        Some(miopen_result) => {
            *returned_algo_count = 1;
            *perf_results = algo_perf_to_cudnn(miopen_result);
        }
        None => {
            *returned_algo_count = 0;
        }
    }
    Ok(())
}

unsafe fn get_or_search_convolution_forward_algorithm(
    handle: &Context,
    x_desc: miopenTensorDescriptor_t,
    w_desc: miopenTensorDescriptor_t,
    conv_desc: miopenConvolutionDescriptor_t,
    y_desc: miopenTensorDescriptor_t,
) -> Result<Option<miopenConvAlgoPerf_t>, miopenError_t> {
    let miopen_result = {
        let mut search_workspace = handle
            .search_workspace
            .lock()
            .map_err(|_| miopenError_t::UnknownError)?;
        let search_workspace = &mut *search_workspace;
        let cache_key = ConvolutionOpCacheKey::new(x_desc, w_desc, conv_desc, y_desc)?;
        let scratchpad = &mut search_workspace.beta_buffer_cache;
        let cache = &mut search_workspace.algo_cache;
        match cache.entry(cache_key) {
            std::collections::hash_map::Entry::Occupied(occupied_entry) => {
                Some(occupied_entry.get().clone())
            }
            std::collections::hash_map::Entry::Vacant(vacant_entry) => {
                search_convolution_forward_algorithm(
                    handle.base,
                    scratchpad,
                    x_desc,
                    w_desc,
                    conv_desc,
                    y_desc,
                )?
                .map(|x| vacant_entry.insert(x).clone())
            }
        }
    };
    Ok(miopen_result)
}

unsafe fn search_convolution_forward_algorithm(
    handle: miopenHandle_t,
    scratchpad: &mut TemporaryBufferAllocator,
    x_desc: miopenTensorDescriptor_t,
    w_desc: miopenTensorDescriptor_t,
    conv_desc: miopenConvolutionDescriptor_t,
    y_desc: miopenTensorDescriptor_t,
) -> Result<Option<miopenConvAlgoPerf_t>, miopenError_t> {
    fn get_tensor_size(desc: miopenTensorDescriptor_t) -> Result<usize, miopenError_t> {
        let mut size_in_bytes = 0;
        unsafe { miopenGetTensorNumBytes(desc, &mut size_in_bytes)? };
        Ok(size_in_bytes)
    }
    let mut required_search_workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(
        handle,
        w_desc,
        x_desc,
        conv_desc,
        y_desc,
        &mut required_search_workspace_size,
    )?;
    let w_size = get_tensor_size(w_desc)?;
    let x_size = get_tensor_size(x_desc)?;
    let y_size = get_tensor_size(y_desc)?;
    let fake_buffer_size = required_search_workspace_size
        .max(w_size)
        .max(x_size)
        .max(y_size);
    let fake_tensor = scratchpad
        .get_or_allocate(fake_buffer_size)
        .map_err(|_| miopenError_t::AllocFailed)?;
    let mut algo_count = 0;
    let mut perf_result = mem::zeroed();
    miopenFindConvolutionForwardAlgorithm(
        handle,
        x_desc,
        fake_tensor.data,
        w_desc,
        fake_tensor.data,
        conv_desc,
        y_desc,
        fake_tensor.data,
        1,
        &mut algo_count,
        &mut perf_result,
        fake_tensor.data,
        required_search_workspace_size,
        false,
    )?;
    if algo_count == 0 {
        Ok(None)
    } else {
        Ok(Some(perf_result))
    }
}

#[derive(PartialEq, Eq, Hash)]
struct ConvolutionOpCacheKey {
    x: TensorCacheKey,
    w: TensorCacheKey,
    conv: ConvolutionDescriptorCacheKey,
    y: TensorCacheKey,
}

impl ConvolutionOpCacheKey {
    fn new(
        x_desc: miopenTensorDescriptor_t,
        w_desc: miopenTensorDescriptor_t,
        conv_desc: miopenConvolutionDescriptor_t,
        y_desc: miopenTensorDescriptor_t,
    ) -> Result<Self, miopenError_t> {
        Ok(Self {
            x: TensorCacheKey::new(x_desc)?,
            w: TensorCacheKey::new(w_desc)?,
            conv: ConvolutionDescriptorCacheKey::new(conv_desc)?,
            y: TensorCacheKey::new(y_desc)?,
        })
    }
}

#[derive(PartialEq, Eq, Hash)]
struct TensorCacheKey {
    data_type: miopenDataType_t,
    dimensions: [i32; 5],
    strides: [i32; 5],
}

impl TensorCacheKey {
    fn new(desc: miopenTensorDescriptor_t) -> Result<Self, miopenError_t> {
        let mut data_type = unsafe { mem::zeroed() };
        let mut dimensions = [0; 5];
        let mut strides = [0; 5];
        unsafe {
            miopenGetTensorDescriptor(
                desc,
                &mut data_type,
                dimensions.as_mut_ptr(),
                strides.as_mut_ptr(),
            )?;
        }
        Ok(Self {
            data_type,
            dimensions,
            strides,
        })
    }
}

#[derive(PartialEq, Eq, Hash)]
struct ConvolutionDescriptorCacheKey {
    mode: miopenConvolutionMode_t,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dilation_h: i32,
    dilation_w: i32,
}

impl ConvolutionDescriptorCacheKey {
    fn new(desc: miopenConvolutionDescriptor_t) -> Result<Self, miopenError_t> {
        let mut mode = unsafe { mem::zeroed() };
        let mut pad_h = 0;
        let mut pad_w = 0;
        let mut stride_h = 0;
        let mut stride_w = 0;
        let mut dilation_h = 0;
        let mut dilation_w = 0;
        unsafe {
            miopenGetConvolutionDescriptor(
                desc,
                &mut mode,
                &mut pad_h,
                &mut pad_w,
                &mut stride_h,
                &mut stride_w,
                &mut dilation_h,
                &mut dilation_w,
            )?;
        }
        Ok(Self {
            mode,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
        })
    }
}

unsafe fn algo_perf_to_cudnn(result: miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgoPerf_t {
    cudnnConvolutionFwdAlgoPerf_t {
        algo: algo_to_cudnn(&result),
        time: result.time,
        memory: result.memory,
        status: cudnnStatus_t::SUCCESS,
        determinism: cudnnDeterminism_t::CUDNN_NON_DETERMINISTIC,
        mathType: cudnnMathType_t::CUDNN_DEFAULT_MATH,
        reserved: [0, 0, 0],
    }
}

unsafe fn algo_to_cudnn(result: &miopenConvAlgoPerf_t) -> cudnnConvolutionFwdAlgo_t {
    match result.__bindgen_anon_1.fwd_algo {
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoDirect => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoFFT => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoWinograd => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
        }
        miopenConvFwdAlgorithm_t::miopenConvolutionFwdAlgoImplicitGEMM => {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
        }
        _ => cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    }
}

pub(crate) unsafe fn get_convolution_forward_workspace_size(
    handle: &Context,
    x_desc: miopenTensorDescriptor_t,
    w_desc: miopenTensorDescriptor_t,
    conv_desc: miopenConvolutionDescriptor_t,
    y_desc: miopenTensorDescriptor_t,
    _algo: miopenConvFwdAlgorithm_t,
    size_in_bytes: &mut usize,
) -> miopenStatus_t {
    let algo =
        get_or_search_convolution_forward_algorithm(handle, x_desc, w_desc, conv_desc, y_desc)?
            .ok_or(miopenError_t::UnsupportedOp)?;
    *size_in_bytes = algo.memory;
    Ok(())
}

pub(crate) unsafe fn convolution_forward(
    handle: &Context,
    alpha: *const ::std::os::raw::c_void,
    x_desc: miopenTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    w_desc: miopenTensorDescriptor_t,
    w: *const ::std::os::raw::c_void,
    conv_desc: miopenConvolutionDescriptor_t,
    _algo: miopenConvFwdAlgorithm_t,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size_in_bytes: usize,
    beta: *const ::std::os::raw::c_void,
    y_desc: miopenTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
) -> miopenStatus_t {
    let do_convolution = |algo, pre_op_buffer| {
        let zero = 0u64;
        miopenConvolutionForward(
            handle.base,
            alpha,
            x_desc,
            x,
            w_desc,
            w,
            conv_desc,
            algo,
            std::ptr::from_ref(&zero).cast(),
            y_desc,
            y,
            workspace,
            workspace_size_in_bytes,
        )?;
        if let Some(pre_op_buffer) = pre_op_buffer {
            let one = 1.0f32;
            miopenOpTensor(
                handle.base,
                miopenTensorOp_t::miopenTensorOpAdd,
                beta,
                y_desc,
                pre_op_buffer,
                std::ptr::from_ref(&one).cast(),
                y_desc,
                y,
                std::ptr::from_ref(&zero).cast(),
                y_desc,
                y,
            )?;
        }
        Ok(())
    };
    // We don't allow users to select their own algorithm
    // because they might select algorithm that is not supported by miopen for this configuration
    // and there's no way to ask miopen to fall back to another algorithm
    let algo =
        get_or_search_convolution_forward_algorithm(handle, x_desc, w_desc, conv_desc, y_desc)?
            .ok_or(miopenError_t::UnsupportedOp)?;
    if algo.memory > workspace_size_in_bytes {
        return miopenStatus_t::ErrorInvalidValue;
    }
    let mut type_ = mem::zeroed();
    miopenGetTensorDescriptor(y_desc, &mut type_, ptr::null_mut(), ptr::null_mut())?;
    let beta_value = if type_ == miopenDataType_t::miopenDouble {
        *beta.cast::<f64>()
    } else {
        *beta.cast::<f32>() as f64
    };
    if beta_value != 0.0 {
        let mut y_size = 0;
        miopenGetTensorNumBytes(y_desc, &mut y_size)?;
        let mut mutable = handle
            .search_workspace
            .lock()
            .map_err(|_| miopenError_t::UnknownError)?;
        mutable.beta_buffer_cache.with_async_buffer(
            y_size,
            hipStream_t(ptr::null_mut()),
            |temp_buffer| {
                hipMemcpyDtoDAsync(
                    hipDeviceptr_t(temp_buffer),
                    hipDeviceptr_t(y),
                    y_size,
                    hipStream_t(ptr::null_mut()),
                )
                .map_err(|_| miopenError_t::InternalError)?;
                do_convolution(algo.__bindgen_anon_1.fwd_algo, Some(temp_buffer))
            },
        )
    } else {
        do_convolution(algo.__bindgen_anon_1.fwd_algo, None)
    }
}

pub(crate) unsafe fn destroy_convolution_descriptor(
    conv_desc: miopenConvolutionDescriptor_t,
) -> miopenStatus_t {
    miopenDestroyConvolutionDescriptor(conv_desc)
}

pub(crate) unsafe fn destroy_filter_descriptor(desc: miopenTensorDescriptor_t) -> miopenStatus_t {
    miopenDestroyTensorDescriptor(desc)
}

pub(crate) unsafe fn destroy_tensor_descriptor(desc: miopenTensorDescriptor_t) -> miopenStatus_t {
    miopenDestroyTensorDescriptor(desc)
}

// ============================================================================
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-1..10
// 10 forwarders cudnn -> miopen per sbloccare LSTM/GRU, BN, ConvBwd,
// Activation, Pooling. Opaque handle types (cudnn*Descriptor_t = *mut Struct)
// vengono trasmutati a miopen*Descriptor_t (stesso layout *mut Struct).
// NOTA: param semantics (algo enums, mode enums) sono naive — iter-2 affinare.
// ============================================================================

use cuda_types::cudnn9 as ct9;

#[inline]
unsafe fn miopen_to_cudnn_status(r: miopenStatus_t) -> ct9::cudnnStatus_t {
    match r {
        Ok(()) => Ok(()),
        Err(miopenError_t::NotInitialized) => Err(ct9::cudnnError_t::NOT_INITIALIZED),
        Err(miopenError_t::InvalidValue) => Err(ct9::cudnnError_t::INVALID_VALUE),
        Err(miopenError_t::BadParm) => Err(ct9::cudnnError_t::BAD_PARAM),
        Err(miopenError_t::AllocFailed) => Err(ct9::cudnnError_t::ALLOC_FAILED),
        Err(miopenError_t::InternalError) => Err(ct9::cudnnError_t::INTERNAL_ERROR),
        Err(miopenError_t::NotImplemented) => Err(ct9::cudnnError_t::NOT_SUPPORTED),
        Err(miopenError_t::UnsupportedOp) => Err(ct9::cudnnError_t::NOT_SUPPORTED),
        Err(_) => Err(ct9::cudnnError_t::INTERNAL_ERROR),
    }
}

// Trasmute opaque cudnn handle (*mut cudnnContext) a miopen handle
#[inline]
/// Il Context avvolto, per arrivare alle sue cache. `ctx_handle` da' solo
/// l'handle MIOpen nudo e perde il resto.
unsafe fn ctx<'a>(h: ct9::cudnnHandle_t) -> &'a Context {
    let r = zluda_common::as_ref::<Context>(&h).unwrap().as_result().unwrap();
    // `as_ref` lega il prestito al parametro locale `h`, che pero' e' solo un
    // puntatore copiato: il Context vive quanto l'handle del chiamante.
    mem::transmute::<&Context, &'a Context>(r)
}

unsafe fn ctx_handle(h: ct9::cudnnHandle_t) -> miopenHandle_t {
    // cudnnHandle_t e' *mut cudnnContext via ZludaObject wrap.
    // Estrai il miopenHandle_t base dal Context wrappato.
    let live = zluda_common::as_ref::<Context>(&h).unwrap();
    live.as_result().unwrap().base
}

// --- Task 1: cudnnRNNForward + cudnnRNNForwardInference ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-1
pub(crate) unsafe extern "C" fn cudnnRNNForward(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    _fwd_mode: i32, // cudnnForwardMode_t (INFERENCE=0, TRAINING=1)
    _dev_seq_lengths: *const i32,
    x_desc: ct9::cudnnRNNDataDescriptor_t,
    x: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnRNNDataDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    h_desc: ct9::cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    hy: *mut ::std::os::raw::c_void,
    c_desc: ct9::cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    cy: *mut ::std::os::raw::c_void,
    weight_space_size: usize,
    weight_space: *const ::std::os::raw::c_void,
    work_space_size: usize,
    work_space: *mut ::std::os::raw::c_void,
    reserve_space_size: usize,
    reserve_space: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let fwd = if _fwd_mode == 0 {
        miopenRNNFWDMode_t::miopenRNNInference
    } else {
        miopenRNNFWDMode_t::miopenRNNTraining
    };
    miopen_to_cudnn_status(miopenRNNForward(
        mh,
        mem::transmute(rnn_desc),
        fwd,
        mem::transmute(x_desc),
        x,
        mem::transmute(h_desc),
        hx,
        hy,
        mem::transmute(c_desc),
        cx,
        cy,
        mem::transmute(y_desc),
        y,
        weight_space,
        weight_space_size,
        work_space,
        work_space_size,
        reserve_space,
        reserve_space_size,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-1b
pub(crate) unsafe extern "C" fn cudnnRNNForwardInference(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: *const ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    hx_desc: ct9::cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    cx_desc: ct9::cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    y_desc: *const ct9::cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    hy_desc: ct9::cudnnTensorDescriptor_t,
    hy: *mut ::std::os::raw::c_void,
    cy_desc: ct9::cudnnTensorDescriptor_t,
    cy: *mut ::std::os::raw::c_void,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size: usize,
) -> ct9::cudnnStatus_t {
    // GRU e RNN semplice non hanno lo stato di cella: PyTorch passa un
    // descrittore NULLO per cx/cy. cuDNN lo tollera, MIOpen lo dereferenzia
    // («Dereferencing nullptr», errors.hpp:128) e la rete non parte.
    // Si sostituisce col descrittore dello stato nascosto, che ha la stessa
    // forma; il puntatore ai dati resta nullo, ed e' quello che conta.
    let cx_desc = if cx_desc.is_null() { hx_desc } else { cx_desc };
    let cy_desc = if cy_desc.is_null() { hy_desc } else { cy_desc };
    // I pesi arrivano nella disposizione di cuDNN: vanno riordinati in quella
    // di MIOpen prima di passarli. Vedi il blocco sulla traduzione, sopra.
    let (pesi_mio, _blocchi, _byte_elem) = pesi_per_miopen(
        handle,
        mem::transmute(rnn_desc),
        mem::transmute(*x_desc),
        w,
    )?;
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenRNNForwardInference(
        mh,
        mem::transmute(rnn_desc),
        seq_length,
        x_desc as *const miopenTensorDescriptor_t,
        x,
        mem::transmute(hx_desc),
        hx,
        mem::transmute(cx_desc),
        cx,
        mem::transmute(w_desc),
        pesi_mio,
        y_desc as *const miopenTensorDescriptor_t,
        y,
        mem::transmute(hy_desc),
        hy,
        mem::transmute(cy_desc),
        cy,
        workspace,
        workspace_size,
    ))
}

// --- Task 2: cudnnRNNBackwardData ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-2
pub(crate) unsafe extern "C" fn cudnnRNNBackwardData(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: i32,
    y_desc: *const ct9::cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dy_desc: *const ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dhy_desc: ct9::cudnnTensorDescriptor_t,
    dhy: *const ::std::os::raw::c_void,
    dcy_desc: ct9::cudnnTensorDescriptor_t,
    dcy: *const ::std::os::raw::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    hx_desc: ct9::cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    cx_desc: ct9::cudnnTensorDescriptor_t,
    cx: *const ::std::os::raw::c_void,
    dx_desc: *const ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    dhx_desc: ct9::cudnnTensorDescriptor_t,
    dhx: *mut ::std::os::raw::c_void,
    dcx_desc: ct9::cudnnTensorDescriptor_t,
    dcx: *mut ::std::os::raw::c_void,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size: usize,
    reserve: *mut ::std::os::raw::c_void,
    reserve_size: usize,
) -> ct9::cudnnStatus_t {
    // GRU e RNN semplice non hanno lo stato di cella: PyTorch passa un
    // descrittore NULLO per cx/cy. cuDNN lo tollera, MIOpen lo dereferenzia
    // («Dereferencing nullptr», errors.hpp:128) e la rete non parte.
    // Si sostituisce col descrittore dello stato nascosto, che ha la stessa
    // forma; il puntatore ai dati resta nullo, ed e' quello che conta.
    let dcy_desc = if dcy_desc.is_null() { dhy_desc } else { dcy_desc };
    let cx_desc = if cx_desc.is_null() { hx_desc } else { cx_desc };
    let dcx_desc = if dcx_desc.is_null() { dhx_desc } else { dcx_desc };
    // I pesi arrivano nella disposizione di cuDNN: vanno riordinati in quella
    // di MIOpen prima di passarli. Vedi il blocco sulla traduzione, sopra.
    let (pesi_mio, _blocchi, _byte_elem) = pesi_per_miopen(
        handle,
        mem::transmute(rnn_desc),
        mem::transmute(*dx_desc),
        w,
    )?;
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenRNNBackwardData(
        mh,
        mem::transmute(rnn_desc),
        seq_length,
        y_desc as *const miopenTensorDescriptor_t,
        y,
        dy_desc as *const miopenTensorDescriptor_t,
        dy,
        mem::transmute(dhy_desc),
        dhy,
        mem::transmute(dcy_desc),
        dcy,
        mem::transmute(w_desc),
        pesi_mio,
        mem::transmute(hx_desc),
        hx,
        mem::transmute(cx_desc),
        cx,
        dx_desc as *const miopenTensorDescriptor_t,
        dx,
        mem::transmute(dhx_desc),
        dhx,
        mem::transmute(dcx_desc),
        dcx,
        workspace,
        workspace_size,
        reserve,
        reserve_size,
    ))
}

// --- Task 3: cudnnRNNBackwardWeights ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-3
pub(crate) unsafe extern "C" fn cudnnRNNBackwardWeights(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: *const ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    hx_desc: ct9::cudnnTensorDescriptor_t,
    hx: *const ::std::os::raw::c_void,
    y_desc: *const ct9::cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size: usize,
    dw_desc: ct9::cudnnFilterDescriptor_t,
    dw: *mut ::std::os::raw::c_void,
    reserve: *const ::std::os::raw::c_void,
    reserve_size: usize,
) -> ct9::cudnnStatus_t {
    // `dw` e' un'uscita e MIOpen ACCUMULA dentro. Si semina l'appoggio col
    // contenuto attuale di dw riordinato, si lascia accumulare a MIOpen, e alla
    // fine si riporta il risultato nella disposizione di cuDNN.
    let (pesi_mio, blocchi, byte_elem) = pesi_per_miopen(
        handle,
        mem::transmute(rnn_desc),
        mem::transmute(*x_desc),
        dw,
    )?;
    let mh = ctx_handle(handle);
    let esito = miopen_to_cudnn_status(miopenRNNBackwardWeights(
        mh,
        mem::transmute(rnn_desc),
        seq_length,
        x_desc as *const miopenTensorDescriptor_t,
        x,
        mem::transmute(hx_desc),
        hx,
        y_desc as *const miopenTensorDescriptor_t,
        y,
        mem::transmute(dw_desc),
        pesi_mio,
        workspace,
        workspace_size,
        reserve,
        reserve_size,
    ));
    permuta_pesi(ctx_handle(handle), dw as *mut u8, pesi_mio.cast(),
                 &blocchi, byte_elem, false)?;
    esito}

// --- Task 4: cudnnSetRNNDescriptor_v8 + cudnnGetRNNWorkspaceSize + cudnnGetRNNTempSpaceSizes ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-4a
pub(crate) unsafe extern "C" fn cudnnSetRNNDescriptor_v8(
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    _algo: i32,
    _cell_mode: i32, // RELU=0, TANH=1, LSTM=2, GRU=3
    _bias_mode: i32,
    _dir_mode: i32,
    _input_mode: i32,
    _data_type: i32,
    _math_prec: i32,
    _math_type: i32,
    _input_size: i32,
    hidden_size: i32,
    _proj_size: i32,
    num_layers: i32,
    _dropout_desc: *mut ::std::os::raw::c_void,
    _aux_flags: u32,
) -> ct9::cudnnStatus_t {
    let mode = match _cell_mode {
        0 => miopenRNNMode_t::miopenRNNRELU,
        1 => miopenRNNMode_t::miopenRNNTANH,
        2 => miopenRNNMode_t::miopenLSTM,
        3 => miopenRNNMode_t::miopenGRU,
        _ => miopenRNNMode_t::miopenLSTM,
    };
    let dir = if _dir_mode == 0 {
        miopenRNNDirectionMode_t::miopenRNNunidirection
    } else {
        miopenRNNDirectionMode_t::miopenRNNbidirection
    };
    let inp = if _input_mode == 0 {
        miopenRNNInputMode_t::miopenRNNlinear
    } else {
        miopenRNNInputMode_t::miopenRNNskip
    };
    let bias = if _bias_mode == 0 {
        miopenRNNBiasMode_t::miopenRNNNoBias
    } else {
        miopenRNNBiasMode_t::miopenRNNwithBias
    };
    let dtype = match _data_type {
        0 => miopenDataType_t::miopenFloat,
        1 => miopenDataType_t::miopenHalf,
        2 => miopenDataType_t::miopenDouble,
        _ => miopenDataType_t::miopenFloat,
    };
    miopen_to_cudnn_status(miopenSetRNNDescriptor(
        mem::transmute(rnn_desc),
        hidden_size,
        num_layers,
        inp,
        dir,
        mode,
        bias,
        miopenRNNAlgo_t::miopenRNNdefault,
        dtype,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-4b
pub(crate) unsafe extern "C" fn cudnnGetRNNWorkspaceSize(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    seq_length: i32,
    x_desc: *const ct9::cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenGetRNNWorkspaceSize(
        mh,
        mem::transmute(rnn_desc),
        seq_length,
        x_desc as *const miopenTensorDescriptor_t,
        size_in_bytes,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-4c
pub(crate) unsafe extern "C" fn cudnnGetRNNTempSpaceSizes(
    handle: ct9::cudnnHandle_t,
    rnn_desc: ct9::cudnnRNNDescriptor_t,
    _fwd_mode: i32,
    x_desc: ct9::cudnnRNNDataDescriptor_t,
    work_space_size: *mut usize,
    reserve_space_size: *mut usize,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let fwd = if _fwd_mode == 0 {
        miopenRNNFWDMode_t::miopenRNNInference
    } else {
        miopenRNNFWDMode_t::miopenRNNTraining
    };
    miopen_to_cudnn_status(miopenGetRNNTempSpaceSizes(
        mh,
        mem::transmute(rnn_desc),
        mem::transmute(x_desc),
        fwd,
        work_space_size,
        reserve_space_size,
    ))
}

// --- Task 5: cudnnConvolutionBackwardData ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-5
pub(crate) unsafe extern "C" fn cudnnConvolutionBackwardData(
    handle: ct9::cudnnHandle_t,
    alpha: *const ::std::os::raw::c_void,
    w_desc: ct9::cudnnFilterDescriptor_t,
    w: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    _algo: i32,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size: usize,
    beta: *const ::std::os::raw::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    // MIOpen NON permette di scegliere l'algoritmo a mano: pretende che una
    // ricerca sia passata prima, perche' e' la ricerca a registrare l'invoker.
    // Chiamando direttamente con un algoritmo scelto a tavolino si ottiene
    //   «No invoker was registered for convolution backward. Was find executed?»
    // (convolutionocl.cpp:1258). Misurato il 2 ago 2026.
    // La ricerca costa: rifatta a ogni chiamata sono ~230 ms per passo su una
    // rete minuscola. Si fa una volta per forma e poi si riusa l'esito.
    let chiave = ConvolutionOpCacheKey::new(
        mem::transmute(dx_desc),
        mem::transmute(w_desc),
        mem::transmute(conv_desc),
        mem::transmute(dy_desc),
    )?;
    let gia = ctx(handle)
        .search_workspace
        .lock()
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
        .algo_cache_bwd_dati
        .get(&chiave)
        .copied();
    let algo = match gia {
        Some(a) => a,
        None => {
            let mut trovati: ::core::ffi::c_int = 0;
            let mut perf: [miopenConvAlgoPerf_t; 4] = mem::zeroed();
            miopenFindConvolutionBackwardDataAlgorithm(
                mh,
                mem::transmute(dy_desc),
                dy,
                mem::transmute(w_desc),
                w,
                mem::transmute(conv_desc),
                mem::transmute(dx_desc),
                dx,
                perf.len() as ::core::ffi::c_int,
                &mut trovati,
                perf.as_mut_ptr(),
                workspace,
                workspace_size,
                false,
            )?;
            if trovati <= 0 {
                return Err(cudnnError_t::NOT_SUPPORTED);
            }
            let a = perf[0].__bindgen_anon_1.bwd_data_algo;
            ctx(handle)
                .search_workspace
                .lock()
                .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
                .algo_cache_bwd_dati
                .insert(chiave, a);
            a
        }
    };
    miopen_to_cudnn_status(miopenConvolutionBackwardData(
        mh,
        alpha,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(w_desc),
        w,
        mem::transmute(conv_desc),
        algo,
        beta,
        mem::transmute(dx_desc),
        dx,
        workspace,
        workspace_size,
    ))
}

// --- Task 6: cudnnConvolutionBackwardFilter ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-6
pub(crate) unsafe extern "C" fn cudnnConvolutionBackwardFilter(
    handle: ct9::cudnnHandle_t,
    alpha: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    conv_desc: ct9::cudnnConvolutionDescriptor_t,
    _algo: i32,
    workspace: *mut ::std::os::raw::c_void,
    workspace_size: usize,
    beta: *const ::std::os::raw::c_void,
    dw_desc: ct9::cudnnFilterDescriptor_t,
    dw: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    // Stessa storia della backward sui dati: senza ricerca niente invoker, e
    // senza cache la ricerca rigira a ogni chiamata.
    let chiave = ConvolutionOpCacheKey::new(
        mem::transmute(x_desc),
        mem::transmute(dw_desc),
        mem::transmute(conv_desc),
        mem::transmute(dy_desc),
    )?;
    let gia = ctx(handle)
        .search_workspace
        .lock()
        .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
        .algo_cache_bwd_pesi
        .get(&chiave)
        .copied();
    let algo = match gia {
        Some(a) => a,
        None => {
            let mut trovati: ::core::ffi::c_int = 0;
            let mut perf: [miopenConvAlgoPerf_t; 4] = mem::zeroed();
            miopenFindConvolutionBackwardWeightsAlgorithm(
                mh,
                mem::transmute(dy_desc),
                dy,
                mem::transmute(x_desc),
                x,
                mem::transmute(conv_desc),
                mem::transmute(dw_desc),
                dw,
                perf.len() as ::core::ffi::c_int,
                &mut trovati,
                perf.as_mut_ptr(),
                workspace,
                workspace_size,
                false,
            )?;
            if trovati <= 0 {
                return Err(cudnnError_t::NOT_SUPPORTED);
            }
            let a = perf[0].__bindgen_anon_1.bwd_weights_algo;
            ctx(handle)
                .search_workspace
                .lock()
                .map_err(|_| cudnnError_t::INTERNAL_ERROR)?
                .algo_cache_bwd_pesi
                .insert(chiave, a);
            a
        }
    };
    miopen_to_cudnn_status(miopenConvolutionBackwardWeights(
        mh,
        alpha,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(x_desc),
        x,
        mem::transmute(conv_desc),
        algo,
        beta,
        mem::transmute(dw_desc),
        dw,
        workspace,
        workspace_size,
    ))
}

// --- Task 7: cudnnBatchNormalizationForwardTraining + ForwardInference ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-7a
pub(crate) unsafe extern "C" fn cudnnBatchNormalizationForwardTraining(
    handle: ct9::cudnnHandle_t,
    _mode: i32, // 0=PER_ACTIVATION, 1=SPATIAL
    alpha: *mut ::std::os::raw::c_void,
    beta: *mut ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *mut ::std::os::raw::c_void,
    bn_bias: *mut ::std::os::raw::c_void,
    exp_avg_factor: f64,
    running_mean: *mut ::std::os::raw::c_void,
    running_var: *mut ::std::os::raw::c_void,
    epsilon: f64,
    save_mean: *mut ::std::os::raw::c_void,
    save_inv_var: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let bn_mode: miopenBatchNormMode_t = mem::transmute(_mode as u32);
    miopen_to_cudnn_status(miopenBatchNormalizationForwardTraining(
        mh,
        bn_mode,
        alpha,
        beta,
        mem::transmute(x_desc),
        x,
        mem::transmute(y_desc),
        y,
        mem::transmute(bn_sbmv_desc),
        bn_scale,
        bn_bias,
        exp_avg_factor,
        running_mean,
        running_var,
        epsilon,
        save_mean,
        save_inv_var,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-7b
pub(crate) unsafe extern "C" fn cudnnBatchNormalizationForwardInference(
    handle: ct9::cudnnHandle_t,
    _mode: i32,
    alpha: *mut ::std::os::raw::c_void,
    beta: *mut ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
    bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *mut ::std::os::raw::c_void,
    bn_bias: *mut ::std::os::raw::c_void,
    est_mean: *mut ::std::os::raw::c_void,
    est_var: *mut ::std::os::raw::c_void,
    epsilon: f64,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let bn_mode: miopenBatchNormMode_t = mem::transmute(_mode as u32);
    miopen_to_cudnn_status(miopenBatchNormalizationForwardInference(
        mh,
        bn_mode,
        alpha,
        beta,
        mem::transmute(x_desc),
        x,
        mem::transmute(y_desc),
        y,
        mem::transmute(bn_sbmv_desc),
        bn_scale,
        bn_bias,
        est_mean,
        est_var,
        epsilon,
    ))
}

// --- Task 8: cudnnBatchNormalizationBackward ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-8
pub(crate) unsafe extern "C" fn cudnnBatchNormalizationBackward(
    handle: ct9::cudnnHandle_t,
    _mode: i32,
    alpha_data_diff: *const ::std::os::raw::c_void,
    beta_data_diff: *const ::std::os::raw::c_void,
    alpha_param_diff: *const ::std::os::raw::c_void,
    beta_param_diff: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    bn_sbd_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *const ::std::os::raw::c_void,
    result_bn_scale_diff: *mut ::std::os::raw::c_void,
    result_bn_bias_diff: *mut ::std::os::raw::c_void,
    epsilon: f64,
    saved_mean: *const ::std::os::raw::c_void,
    saved_inv_var: *const ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let bn_mode: miopenBatchNormMode_t = mem::transmute(_mode as u32);
    miopen_to_cudnn_status(miopenBatchNormalizationBackward(
        mh,
        bn_mode,
        alpha_data_diff,
        beta_data_diff,
        alpha_param_diff,
        beta_param_diff,
        mem::transmute(x_desc),
        x,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(dx_desc),
        dx,
        mem::transmute(bn_sbd_desc),
        bn_scale,
        result_bn_scale_diff,
        result_bn_bias_diff,
        epsilon,
        saved_mean,
        saved_inv_var,
    ))
}

// --- Task 9: cudnnActivationForward + cudnnActivationBackward ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-9a
pub(crate) unsafe extern "C" fn cudnnActivationForward(
    handle: ct9::cudnnHandle_t,
    activ_desc: ct9::cudnnActivationDescriptor_t,
    alpha: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenActivationForward(
        mh,
        mem::transmute(activ_desc),
        alpha,
        mem::transmute(x_desc),
        x,
        beta,
        mem::transmute(y_desc),
        y,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-9b
pub(crate) unsafe extern "C" fn cudnnActivationBackward(
    handle: ct9::cudnnHandle_t,
    activ_desc: ct9::cudnnActivationDescriptor_t,
    alpha: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenActivationBackward(
        mh,
        mem::transmute(activ_desc),
        alpha,
        mem::transmute(y_desc),
        y,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(x_desc),
        x,
        beta,
        mem::transmute(dx_desc),
        dx,
    ))
}

// --- Task 10: cudnnPoolingForward + cudnnPoolingBackward ---
// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-10a
pub(crate) unsafe extern "C" fn cudnnPoolingForward(
    handle: ct9::cudnnHandle_t,
    pool_desc: ct9::cudnnPoolingDescriptor_t,
    alpha: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenPoolingForward(
        mh,
        mem::transmute(pool_desc),
        alpha,
        mem::transmute(x_desc),
        x,
        beta,
        mem::transmute(y_desc),
        y,
        false, // do_backward
        std::ptr::null_mut(),
        0,
    ))
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 cudnn-10b
pub(crate) unsafe extern "C" fn cudnnPoolingBackward(
    handle: ct9::cudnnHandle_t,
    pool_desc: ct9::cudnnPoolingDescriptor_t,
    alpha: *const ::std::os::raw::c_void,
    y_desc: ct9::cudnnTensorDescriptor_t,
    y: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    beta: *const ::std::os::raw::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    miopen_to_cudnn_status(miopenPoolingBackward(
        mh,
        mem::transmute(pool_desc),
        alpha,
        mem::transmute(y_desc),
        y,
        mem::transmute(dy_desc),
        dy,
        mem::transmute(x_desc),
        x,
        beta,
        mem::transmute(dx_desc),
        dx,
        std::ptr::null_mut(),
    ))
}

// --- Task 11: cudnnLayerNormBackward (Safe Contiguous Guard) ---
// VEGA-PATCH 2026-06-30 sprint Rust E LITE worker 2/3 TASK 1
// Root cause T17:11: aten::native_layer_norm_backward su tensor non-contig
// (post transpose multi-head Q/K/V) -> V7 stride reduction wave64 BUG 50-700%.
// Strategia A: guardia .contiguous() pre-call via miopenTransformTensor
// (pattern Task 6 W2 = elementwise SAFE wave64, NO reduction stride). Mappa LN
// -> miopenBatchNormalizationBackward mode=PER_ACTIVATION (LN equivalente).
// Regola 4-bis ROCm 5.7 esclusivo. Regola 17 cargo-buildable.
//
// NOTA: PyTorch native_layer_norm_backward dispatch ATEN diretto a kernel CUDA
// custom (vectorized_layer_norm_kernel) -> NON passa per cudnn.dll. Per
// intercezione end-to-end servono DUE pezzi: (1) questo wrapper Rust cudnn-canonical
// per call-site LA RUOTA torch_hip MIOpen bridge, (2) monkey-patch safety_v7.py
// su torch.nn.functional.layer_norm.backward che forza .contiguous() su grad_out
// + input prima della chiamata aten. Il pattern miopenTransformTensor in questo
// helper e' identico semanticamente al .contiguous() Python (memcpy stride-aware
// SAFE), e puo' essere riusato via simbolo cudnn exported per dispatch alternativo.

// VEGA-PATCH 2026-06-30 worker 1/3 TASK 1 — fix 3 bug critici worker 2:
//   (a) Mutex deadlock: nested with_contiguous_packed lockava search_workspace
//       2 volte sullo stesso ctx (x_packed closure + dy_packed closure) ->
//       freeze training. Sostituito con hipMalloc/hipFree diretto.
//   (b) Descrittore dst sbagliato: passava tensor_desc (strided) come dst
//       a miopenTransformTensor -> MIOpen scriveva con stride originali in
//       buffer packed = layout sbagliato. Ora creo dst_packed_desc nuovo
//       con strides naturali row-major.
//   (c) BN-bwd riceveva descrittori originali strided ma con ptr packed ->
//       garbage. with_contiguous_packed ora ritorna (ptr_packed, desc_packed)
//       via callback a 2 argomenti.
//   (d) Loop dims [0..5] assumeva sempre 5D: ora leggo ndim reale via
//       miopenGetTensorDescriptorSize (max 5).
unsafe fn with_contiguous_packed(
    handle: miopenHandle_t,
    tensor_desc: miopenTensorDescriptor_t,
    src: *const ::std::os::raw::c_void,
    fn_: &mut dyn FnMut(
        miopenTensorDescriptor_t,
        *const ::std::os::raw::c_void,
    ) -> Result<(), miopenError_t>,
) -> Result<(), miopenError_t> {
    let mut data_type: miopenDataType_t = mem::zeroed();
    let mut dims = [0i32; 5];
    let mut strides = [0i32; 5];
    miopenGetTensorDescriptor(
        tensor_desc,
        &mut data_type,
        dims.as_mut_ptr(),
        strides.as_mut_ptr(),
    )?;
    let mut n_dims: i32 = 0;
    miopenGetTensorDescriptorSize(tensor_desc, &mut n_dims)?;
    let n = n_dims.max(0).min(5) as usize;
    if n == 0 {
        return fn_(tensor_desc, src);
    }
    // Packed contig check: stride[n-1] == 1, stride[i] == stride[i+1]*dims[i+1].
    let mut is_contig = strides[n - 1] == 1;
    if is_contig && n > 1 {
        for i in (0..n - 1).rev() {
            if strides[i] != strides[i + 1].saturating_mul(dims[i + 1]) {
                is_contig = false;
                break;
            }
        }
    }
    if is_contig {
        return fn_(tensor_desc, src);
    }
    // --- Non-contig: hipMalloc scratch + descrittore packed nuovo + transform ---
    let mut nbytes: usize = 0;
    miopenGetTensorNumBytes(tensor_desc, &mut nbytes)?;
    let mut scratch: *mut ::std::os::raw::c_void = std::ptr::null_mut();
    if hipMalloc(&mut scratch, nbytes).is_err() {
        return Err(miopenError_t::AllocFailed);
    }
    // Calcola stride packed row-major naturale.
    let mut packed_strides = [1i32; 5];
    if n > 1 {
        for i in (0..n - 1).rev() {
            packed_strides[i] = packed_strides[i + 1].saturating_mul(dims[i + 1]);
        }
    }
    let mut packed_desc: miopenTensorDescriptor_t = miopenTensorDescriptor_t(std::ptr::null_mut());
    if let Err(e) = miopenCreateTensorDescriptor(&mut packed_desc) {
        let _ = hipFree(scratch);
        return Err(e);
    }
    if let Err(e) = miopenSetTensorDescriptor(
        packed_desc,
        data_type,
        n_dims,
        dims.as_mut_ptr(),
        packed_strides.as_mut_ptr(),
    ) {
        let _ = miopenDestroyTensorDescriptor(packed_desc);
        let _ = hipFree(scratch);
        return Err(e);
    }
    let one = 1.0f32;
    let zero = 0.0f32;
    // miopenTransformTensor src(strided) -> dst(packed): elementwise SAFE wave64,
    // NO cross-lane reduction -> NON triggera BUG V7 stride.
    let xform = miopenTransformTensor(
        handle,
        std::ptr::from_ref(&one).cast(),
        tensor_desc,
        src,
        std::ptr::from_ref(&zero).cast(),
        packed_desc,
        scratch,
    );
    let result = match xform {
        Ok(()) => fn_(packed_desc, scratch as *const ::std::os::raw::c_void),
        Err(e) => Err(e),
    };
    let _ = miopenDestroyTensorDescriptor(packed_desc);
    let _ = hipFree(scratch);
    result
}

pub(crate) unsafe extern "C" fn cudnnLayerNormBackward(
    handle: ct9::cudnnHandle_t,
    x_desc: ct9::cudnnTensorDescriptor_t,
    x: *const ::std::os::raw::c_void,
    dy_desc: ct9::cudnnTensorDescriptor_t,
    dy: *const ::std::os::raw::c_void,
    dx_desc: ct9::cudnnTensorDescriptor_t,
    dx: *mut ::std::os::raw::c_void,
    bn_param_desc: ct9::cudnnTensorDescriptor_t,
    bn_scale: *const ::std::os::raw::c_void,
    result_bn_scale_diff: *mut ::std::os::raw::c_void,
    result_bn_bias_diff: *mut ::std::os::raw::c_void,
    epsilon: f64,
    saved_mean: *const ::std::os::raw::c_void,
    saved_inv_var: *const ::std::os::raw::c_void,
) -> ct9::cudnnStatus_t {
    let mh = ctx_handle(handle);
    let alpha_data = 1.0f32;
    let beta_data = 0.0f32;
    let alpha_param = 1.0f32;
    let beta_param = 0.0f32;
    let x_desc_m: miopenTensorDescriptor_t = mem::transmute(x_desc);
    let dy_desc_m: miopenTensorDescriptor_t = mem::transmute(dy_desc);
    let dx_desc_m: miopenTensorDescriptor_t = mem::transmute(dx_desc);
    let bn_param_desc_m: miopenTensorDescriptor_t = mem::transmute(bn_param_desc);
    // Nested guard SENZA mutex (fix bug worker 2): la callback riceve
    // (desc_packed, ptr_packed) cosi' BN-bwd vede metadata coerenti col buffer.
    let r: Result<(), miopenError_t> =
        with_contiguous_packed(mh, x_desc_m, x, &mut |x_safe_desc, x_packed| {
            with_contiguous_packed(mh, dy_desc_m, dy, &mut |dy_safe_desc, dy_packed| {
                miopenBatchNormalizationBackward(
                    mh,
                    miopenBatchNormMode_t::miopenBNPerActivation,
                    std::ptr::from_ref(&alpha_data).cast(),
                    std::ptr::from_ref(&beta_data).cast(),
                    std::ptr::from_ref(&alpha_param).cast(),
                    std::ptr::from_ref(&beta_param).cast(),
                    x_safe_desc,
                    x_packed,
                    dy_safe_desc,
                    dy_packed,
                    dx_desc_m,
                    dx,
                    bn_param_desc_m,
                    bn_scale,
                    result_bn_scale_diff,
                    result_bn_bias_diff,
                    epsilon,
                    saved_mean,
                    saved_inv_var,
                )
            })
        });
    miopen_to_cudnn_status(r)
}

pub mod dnn8 {
    use cuda_types::cudnn8::*;
    use std::mem;

    pub(crate) fn get_error_string(
        status: cuda_types::cudnn8::cudnnStatus_t,
    ) -> *const ::core::ffi::c_char {
        super::dnn9::get_error_string(status8_to_9(status))
    }

    fn status8_to_9(
        status: cuda_types::cudnn8::cudnnStatus_t,
    ) -> cuda_types::cudnn9::cudnnStatus_t {
        match status {
            Ok(()) => Ok(()),
            Err(err) => Err(match err {
                cuda_types::cudnn8::cudnnError_t::NOT_INITIALIZED => {
                    cuda_types::cudnn9::cudnnError_t::NOT_INITIALIZED
                }
                cuda_types::cudnn8::cudnnError_t::ALLOC_FAILED => {
                    cuda_types::cudnn9::cudnnError_t::ALLOC_FAILED
                }
                cuda_types::cudnn8::cudnnError_t::BAD_PARAM => {
                    cuda_types::cudnn9::cudnnError_t::BAD_PARAM
                }
                cuda_types::cudnn8::cudnnError_t::INTERNAL_ERROR => {
                    cuda_types::cudnn9::cudnnError_t::INTERNAL_ERROR
                }
                cuda_types::cudnn8::cudnnError_t::INVALID_VALUE => {
                    cuda_types::cudnn9::cudnnError_t::INVALID_VALUE
                }
                cuda_types::cudnn8::cudnnError_t::ARCH_MISMATCH => {
                    cuda_types::cudnn9::cudnnError_t::ARCH_MISMATCH
                }
                cuda_types::cudnn8::cudnnError_t::MAPPING_ERROR => {
                    cuda_types::cudnn9::cudnnError_t::MAPPING_ERROR
                }
                cuda_types::cudnn8::cudnnError_t::EXECUTION_FAILED => {
                    cuda_types::cudnn9::cudnnError_t::EXECUTION_FAILED
                }
                cuda_types::cudnn8::cudnnError_t::NOT_SUPPORTED => {
                    cuda_types::cudnn9::cudnnError_t::NOT_SUPPORTED
                }
                cuda_types::cudnn8::cudnnError_t::LICENSE_ERROR => {
                    cuda_types::cudnn9::cudnnError_t::LICENSE_ERROR
                }
                cuda_types::cudnn8::cudnnError_t::RUNTIME_PREREQUISITE_MISSING => {
                    cuda_types::cudnn9::cudnnError_t::RUNTIME_PREREQUISITE_MISSING
                }
                cuda_types::cudnn8::cudnnError_t::RUNTIME_IN_PROGRESS => {
                    cuda_types::cudnn9::cudnnError_t::RUNTIME_IN_PROGRESS
                }
                cuda_types::cudnn8::cudnnError_t::RUNTIME_FP_OVERFLOW => {
                    cuda_types::cudnn9::cudnnError_t::RUNTIME_FP_OVERFLOW
                }
                cuda_types::cudnn8::cudnnError_t::VERSION_MISMATCH => {
                    cuda_types::cudnn9::cudnnError_t::VERSION_MISMATCH
                }
                _ => cuda_types::cudnn9::cudnnError_t::INTERNAL_ERROR,
            }),
        }
    }

    pub(crate) unsafe fn get_convolution_forward_algorithm_v7(
        handle: cudnnHandle_t,
        x_desc: cudnnTensorDescriptor_t,
        w_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        requested_algo_count: ::std::os::raw::c_int,
        returned_algo_count: *mut ::std::os::raw::c_int,
        perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
    ) -> Result<(), cudnnError_t> {
        if requested_algo_count <= 0 {
            return Err(cudnnError_t::BAD_PARAM);
        }
        let mut perf_results_dnn9 = mem::zeroed();
        super::dnn9::get_convolution_forward_algorithm_v7(
            handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            1,
            returned_algo_count,
            &mut perf_results_dnn9,
        )?;
        *perf_results = cudnnConvolutionFwdAlgoPerf_t {
            algo: perf_results_dnn9.algo,
            time: perf_results_dnn9.time,
            memory: perf_results_dnn9.memory,
            status: status9_to_8(perf_results_dnn9.status),
            determinism: perf_results_dnn9.determinism,
            mathType: perf_results_dnn9.mathType,
            reserved: [0, 0, 0],
        };
        Ok(())
    }

    fn status9_to_8(
        status: cuda_types::cudnn9::cudnnStatus_t,
    ) -> cuda_types::cudnn8::cudnnStatus_t {
        match status {
            Ok(()) => Ok(()),
            Err(err) => Err(match err {
                cuda_types::cudnn9::cudnnError_t::NOT_INITIALIZED => {
                    cuda_types::cudnn8::cudnnError_t::NOT_INITIALIZED
                }
                cuda_types::cudnn9::cudnnError_t::ALLOC_FAILED => {
                    cuda_types::cudnn8::cudnnError_t::ALLOC_FAILED
                }
                cuda_types::cudnn9::cudnnError_t::BAD_PARAM => {
                    cuda_types::cudnn8::cudnnError_t::BAD_PARAM
                }
                cuda_types::cudnn9::cudnnError_t::INTERNAL_ERROR => {
                    cuda_types::cudnn8::cudnnError_t::INTERNAL_ERROR
                }
                cuda_types::cudnn9::cudnnError_t::INVALID_VALUE => {
                    cuda_types::cudnn8::cudnnError_t::INVALID_VALUE
                }
                cuda_types::cudnn9::cudnnError_t::ARCH_MISMATCH => {
                    cuda_types::cudnn8::cudnnError_t::ARCH_MISMATCH
                }
                cuda_types::cudnn9::cudnnError_t::MAPPING_ERROR => {
                    cuda_types::cudnn8::cudnnError_t::MAPPING_ERROR
                }
                cuda_types::cudnn9::cudnnError_t::EXECUTION_FAILED => {
                    cuda_types::cudnn8::cudnnError_t::EXECUTION_FAILED
                }
                cuda_types::cudnn9::cudnnError_t::NOT_SUPPORTED => {
                    cuda_types::cudnn8::cudnnError_t::NOT_SUPPORTED
                }
                cuda_types::cudnn9::cudnnError_t::LICENSE_ERROR => {
                    cuda_types::cudnn8::cudnnError_t::LICENSE_ERROR
                }
                cuda_types::cudnn9::cudnnError_t::RUNTIME_PREREQUISITE_MISSING => {
                    cuda_types::cudnn8::cudnnError_t::RUNTIME_PREREQUISITE_MISSING
                }
                cuda_types::cudnn9::cudnnError_t::RUNTIME_IN_PROGRESS => {
                    cuda_types::cudnn8::cudnnError_t::RUNTIME_IN_PROGRESS
                }
                cuda_types::cudnn9::cudnnError_t::RUNTIME_FP_OVERFLOW => {
                    cuda_types::cudnn8::cudnnError_t::RUNTIME_FP_OVERFLOW
                }
                cuda_types::cudnn9::cudnnError_t::VERSION_MISMATCH => {
                    cuda_types::cudnn8::cudnnError_t::VERSION_MISMATCH
                }
                _ => cuda_types::cudnn8::cudnnError_t::INTERNAL_ERROR,
            }),
        }
    }
}

// ============================================================================
// VEGA-PATCH 2026-06-30 sprint Rust E LITE Task1 worker3/3
// CONTIGUOUS GUARD per aten::native_layer_norm_backward (V7 stride reduction bug).
// ============================================================================
// CONTESTO: il bug primario composite 709 MiniBlock+AdamW e' isolato T17:11
// in aten::native_layer_norm_backward su tensor non-contig (post transpose
// multi-head Q/K/V) -> V7 stride reduction wave64 esplode (Fix #17 lookalike).
//
// STRADA SCELTA: A) helper extern "C" callable via ctypes da safety_v7.py.
//   MIOpen 5.7 NON espone miopenLayerNormBackward (solo Forward + T5/Add bwd
//   con normalized_dim != "ultima axis classic LN PyTorch") -> wrapper diretto
//   non possibile. La via pulita e' pre-pack a contig PRIMA che ATen lanci
//   la reduction stride-aware difettosa. Pattern: miopenTransformTensor
//   (elementwise SAFE wave64, Task 6 W2 confermato) sa rimaterializzare
//   un tensor arbitrary-stride in un tensor contig dello stesso shape.
//
// FIRMA: dst e src devono avere lo STESSO SHAPE. src_strides arbitrari (in
// elementi), dst contig in row-major (computati internamente). type=fp16/bf16/fp32.
// Fast-path: se src e' gia' contig, hipMemcpyDtoD diretto (no kernel).
//
// USO PYTHON (safety_v7.py):
//   import ctypes
//   cudnn = ctypes.CDLL("cudnn64_9.dll")
//   cudnn.zludaContiguousGuardLayerNormBwd.restype = ctypes.c_int
//   # patch torch.nn.functional.layer_norm.backward hook: pre-call su grad_out
//   # e su input (X) se non-contig, poi lascia che ATen lanci reduction su buf contig.
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn zludaContiguousGuardLayerNormBwd(
    handle: ct9::cudnnHandle_t,
    // Shape: ndim <= 5 (LN tipico [B,S,H] o [B,H,S,D] transposto = 4D, max 5).
    ndim: i32,
    dims: *const i32,        // ndim elementi
    src_strides: *const i32, // ndim elementi (in numero di elementi, NON byte)
    data_type: i32,          // miopenDataType_t: 0=fp32, 1=fp16, 5=bf16
    src: *const ::std::os::raw::c_void,
    dst: *mut ::std::os::raw::c_void, // buffer contig gia' allocato da Python
) -> i32 {
    // --- Validation ---
    if handle.is_null() || dims.is_null() || src_strides.is_null()
        || src.is_null() || dst.is_null() {
        return -1; // INVALID_VALUE
    }
    if ndim < 1 || ndim > 5 {
        return -2; // BAD_NDIM
    }
    let ndim_u = ndim as usize;
    let dims_slice = std::slice::from_raw_parts(dims, ndim_u);
    let stride_slice = std::slice::from_raw_parts(src_strides, ndim_u);

    // Numel + check positivi
    let mut numel: i64 = 1;
    for &d in dims_slice {
        if d <= 0 { return -3; } // BAD_DIM
        numel = numel.saturating_mul(d as i64);
    }

    // Element size
    let dt: miopenDataType_t = match data_type {
        0 => miopenDataType_t::miopenFloat,
        1 => miopenDataType_t::miopenHalf,
        5 => miopenDataType_t::miopenBFloat16,
        _ => return -4, // UNSUPPORTED_DTYPE
    };
    let elem_bytes: usize = match dt {
        miopenDataType_t::miopenFloat => 4,
        miopenDataType_t::miopenHalf => 2,
        miopenDataType_t::miopenBFloat16 => 2,
        _ => return -4,
    };
    let total_bytes = (numel as usize).saturating_mul(elem_bytes);

    // --- Compute contig strides row-major ---
    let mut contig_strides = [1i32; 5];
    for i in (0..ndim_u.saturating_sub(1)).rev() {
        contig_strides[i] = contig_strides[i + 1] * dims_slice[i + 1];
    }

    // --- Fast path: gia' contig? memcpy diretto, no kernel reduction ---
    let mut already_contig = true;
    for i in 0..ndim_u {
        if stride_slice[i] != contig_strides[i] {
            already_contig = false;
            break;
        }
    }
    if already_contig {
        let r = hipMemcpyDtoDAsync(
            hipDeviceptr_t(dst),
            hipDeviceptr_t(src as *mut _),
            total_bytes,
            hipStream_t(ptr::null_mut()),
        );
        return match r {
            Ok(()) => 0,
            Err(_) => -10, // HIP_MEMCPY_FAIL
        };
    }

    // --- Slow path: miopenTransformTensor (SAFE wave64 elementwise) ---
    // Crea due tensor descriptor: src con strides arbitrari, dst contig.
    let ctx = match zluda_common::as_ref::<Context>(&handle) {
        Some(v) => v,
        None => return -5, // BAD_HANDLE
    };
    let live = match ctx.as_result() {
        Ok(v) => v,
        Err(_) => return -5,
    };
    let mh = live.base;

    let mut src_desc: miopenTensorDescriptor_t = mem::zeroed();
    let mut dst_desc: miopenTensorDescriptor_t = mem::zeroed();
    if miopenCreateTensorDescriptor(&mut src_desc).is_err() { return -6; }
    if miopenCreateTensorDescriptor(&mut dst_desc).is_err() {
        let _ = miopenDestroyTensorDescriptor(src_desc);
        return -6;
    }

    // miopenSetTensorDescriptor (NdTensorDescriptor) accetta strides arbitrari
    let set_src = miopenSetTensorDescriptor(
        src_desc,
        dt,
        ndim,
        dims,
        src_strides,
    );
    let set_dst = miopenSetTensorDescriptor(
        dst_desc,
        dt,
        ndim,
        dims,
        contig_strides.as_ptr(),
    );
    if set_src.is_err() || set_dst.is_err() {
        let _ = miopenDestroyTensorDescriptor(src_desc);
        let _ = miopenDestroyTensorDescriptor(dst_desc);
        return -7;
    }

    // alpha=1.0, beta=0.0 (overwrite dst). Type host scalar = fp32 sempre.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let r = miopenTransformTensor(
        mh,
        std::ptr::from_ref(&alpha).cast(),
        src_desc,
        src,
        std::ptr::from_ref(&beta).cast(),
        dst_desc,
        dst,
    );

    let _ = miopenDestroyTensorDescriptor(src_desc);
    let _ = miopenDestroyTensorDescriptor(dst_desc);

    match r {
        Ok(()) => 0,
        Err(_) => -8, // MIOPEN_TRANSFORM_FAIL
    }
}

pub mod dnn9 {
    use cuda_types::cudnn9::*;
    use zluda_common::FromCuda;

    /// Anche questa era `todo!()`: viene chiamata proprio quando qualcosa e'
    /// gia' andato storto, e far abortire il processo mentre si compone un
    /// messaggio d'errore nasconde l'errore vero.
    pub(crate) fn get_error_string(status: cudnnStatus_t) -> *const ::core::ffi::c_char {
        // Non si distingue fra i codici: qui `cudnnStatus_t` e' un `Result`, non
        // un enum piatto, e il valore di questa funzione sta tutto nel non far
        // abortire il processo mentre si compone un messaggio d'errore.
        b"cudnn (ZLUDA): consultare il codice di stato restituito\0".as_ptr()
            as *const ::core::ffi::c_char
    }

    pub(crate) unsafe fn get_convolution_forward_algorithm_v7(
        handle: cudnnHandle_t,
        x_desc: cudnnTensorDescriptor_t,
        w_desc: cudnnFilterDescriptor_t,
        conv_desc: cudnnConvolutionDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        requested_algo_count: ::std::os::raw::c_int,
        returned_algo_count: *mut ::std::os::raw::c_int,
        perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
    ) -> Result<(), cudnnError_t> {
        super::get_convolution_forward_algorithm_v7(
            FromCuda::<_, cudnnError_t>::from_cuda(&handle)?,
            FromCuda::<_, cudnnError_t>::from_cuda(&x_desc)?,
            FromCuda::<_, cudnnError_t>::from_cuda(&w_desc)?,
            FromCuda::<_, cudnnError_t>::from_cuda(&conv_desc)?,
            FromCuda::<_, cudnnError_t>::from_cuda(&y_desc)?,
            requested_algo_count,
            FromCuda::<_, cudnnError_t>::from_cuda(&returned_algo_count)?,
            FromCuda::<_, cudnnError_t>::from_cuda(&perf_results)?,
        )?;
        Ok(())
    }
}

// VEGA-PATCH 2026-06-29 T07:55 sprint-orch T2 cabling Family B — SPIKE 4 funzioni "facili"
// Strategia: 4 funzioni senza parametri enum (Activation Forward/Backward, Pooling Forward/Backward)
// = solo descriptor opachi (alias type identici cudnn8==cudnn9) + return type conversion.
// Se spike passa, scalo a 16. Se fail E0308, rollback onesto.
// Conformita' regola 4-bis: shim Marconi v4.0 sopra amdhip64.dll v5.7, zero ROCm 6.x/7.x.

pub mod dnn9_b {
    use cuda_types::cudnn9 as ct9;
    // wrapper esplicito perche' super::cudnnXxx e' pub(crate), non re-exportabile.
    // VEGA-PATCH 2026-06-29 WORKER GAP2 — signature ALLINEATE al cudnn9 macro
    // (enum tipati passati .0 as i32 alle impl super::* che vogliono i32 raw;
    // BatchNorm alpha/beta/scale/bias *const c_void cast as *mut alle impl super::*).

    pub unsafe extern "C" fn cudnnGetProperty(
        type_: ct9::libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetProperty(type_, value)
    }

    pub unsafe extern "C" fn cudnnBackendCreateDescriptor(
        descriptor_type: ct9::cudnnBackendDescriptorType_t,
        descriptor: *mut ct9::cudnnBackendDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendCreateDescriptor(descriptor_type, descriptor)
    }

    pub unsafe extern "C" fn cudnnBackendDestroyDescriptor(
        descriptor: ct9::cudnnBackendDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendDestroyDescriptor(descriptor)
    }

    pub unsafe extern "C" fn cudnnBackendSetAttribute(
        descriptor: ct9::cudnnBackendDescriptor_t,
        attribute_name: ct9::cudnnBackendAttributeName_t,
        attribute_type: ct9::cudnnBackendAttributeType_t,
        element_count: i64,
        array_of_elements: *const ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendSetAttribute(descriptor, attribute_name, attribute_type, element_count, array_of_elements)
    }

    pub unsafe extern "C" fn cudnnBackendGetAttribute(
        descriptor: ct9::cudnnBackendDescriptor_t,
        attribute_name: ct9::cudnnBackendAttributeName_t,
        attribute_type: ct9::cudnnBackendAttributeType_t,
        requested_element_count: i64,
        element_count: *mut i64,
        array_of_elements: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendGetAttribute(descriptor, attribute_name, attribute_type, requested_element_count, element_count, array_of_elements)
    }

    pub unsafe extern "C" fn cudnnBackendFinalize(
        descriptor: ct9::cudnnBackendDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendFinalize(descriptor)
    }

    pub unsafe extern "C" fn cudnnBackendExecute(
        handle: ct9::cudnnHandle_t,
        execution_plan: ct9::cudnnBackendDescriptor_t,
        variant_pack: ct9::cudnnBackendDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBackendExecute(handle, execution_plan, variant_pack)
    }

    pub unsafe extern "C" fn cudnnCreateSpatialTransformerDescriptor(
        st_desc: *mut ct9::cudnnSpatialTransformerDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateSpatialTransformerDescriptor(st_desc)
    }

    pub unsafe extern "C" fn cudnnDestroySpatialTransformerDescriptor(
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroySpatialTransformerDescriptor(st_desc)
    }

    pub unsafe extern "C" fn cudnnSetSpatialTransformerNdDescriptor(
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
        sampler_type: ct9::cudnnSamplerType_t,
        data_type: ct9::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetSpatialTransformerNdDescriptor(st_desc, sampler_type, data_type, nb_dims, dim_a)
    }

    pub unsafe extern "C" fn cudnnSpatialTfGridGeneratorForward(
        handle: ct9::cudnnHandle_t,
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
        theta: *const ::core::ffi::c_void,
        grid: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSpatialTfGridGeneratorForward(handle, st_desc, theta, grid)
    }

    pub unsafe extern "C" fn cudnnSpatialTfGridGeneratorBackward(
        handle: ct9::cudnnHandle_t,
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
        dgrid: *const ::core::ffi::c_void,
        dtheta: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSpatialTfGridGeneratorBackward(handle, st_desc, dgrid, dtheta)
    }

    pub unsafe extern "C" fn cudnnSpatialTfSamplerForward(
        handle: ct9::cudnnHandle_t,
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        grid: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        y_desc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSpatialTfSamplerForward(handle, st_desc, alpha, x_desc, x, grid, beta, y_desc, y)
    }

    pub unsafe extern "C" fn cudnnSpatialTfSamplerBackward(
        handle: ct9::cudnnHandle_t,
        st_desc: ct9::cudnnSpatialTransformerDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dx_desc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        alpha_dgrid: *const ::core::ffi::c_void,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        grid: *const ::core::ffi::c_void,
        beta_dgrid: *const ::core::ffi::c_void,
        dgrid: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSpatialTfSamplerBackward(handle, st_desc, alpha, x_desc, x, beta, dx_desc, dx, alpha_dgrid, dy_desc, dy, grid, beta_dgrid, dgrid)
    }

    pub unsafe extern "C" fn cudnnCreateRNNDescriptor(
        rnn_desc: *mut ct9::cudnnRNNDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateRNNDescriptor(rnn_desc)
    }

    pub unsafe extern "C" fn cudnnDestroyRNNDescriptor(
        rnn_desc: ct9::cudnnRNNDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroyRNNDescriptor(rnn_desc)
    }

    pub unsafe extern "C" fn cudnnCreateRNNDataDescriptor(
        rnn_data_desc: *mut ct9::cudnnRNNDataDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateRNNDataDescriptor(rnn_data_desc)
    }

    pub unsafe extern "C" fn cudnnDestroyRNNDataDescriptor(
        rnn_data_desc: ct9::cudnnRNNDataDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroyRNNDataDescriptor(rnn_data_desc)
    }

    pub unsafe extern "C" fn cudnnSetRNNDataDescriptor(
        rnn_data_desc: ct9::cudnnRNNDataDescriptor_t,
        data_type: ct9::cudnnDataType_t,
        layout: ct9::cudnnRNNDataLayout_t,
        max_seq_length: ::core::ffi::c_int,
        batch_size: ::core::ffi::c_int,
        vector_size: ::core::ffi::c_int,
        seq_length_array: *const ::core::ffi::c_int,
        padding_fill: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetRNNDataDescriptor(rnn_data_desc, data_type, layout, max_seq_length, batch_size, vector_size, seq_length_array, padding_fill)
    }

    pub unsafe extern "C" fn cudnnSetRNNDescriptor_v6(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        hidden_size: ::core::ffi::c_int,
        num_layers: ::core::ffi::c_int,
        dropout_desc: ct9::cudnnDropoutDescriptor_t,
        input_mode: ct9::cudnnRNNInputMode_t,
        direction: ct9::cudnnDirectionMode_t,
        cell_mode: ct9::cudnnRNNMode_t,
        algo: ct9::cudnnRNNAlgo_t,
        math_prec: ct9::cudnnDataType_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetRNNDescriptor_v6(handle, rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, direction, cell_mode, algo, math_prec)
    }

    pub unsafe extern "C" fn cudnnSetRNNMatrixMathType(
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        m_type: ct9::cudnnMathType_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetRNNMatrixMathType(rnn_desc, m_type)
    }

    pub unsafe extern "C" fn cudnnSetRNNProjectionLayers(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        rec_proj_size: ::core::ffi::c_int,
        out_proj_size: ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetRNNProjectionLayers(handle, rnn_desc, rec_proj_size, out_proj_size)
    }

    pub unsafe extern "C" fn cudnnGetRNNParamsSize(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
        data_type: ct9::cudnnDataType_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNParamsSize(handle, rnn_desc, x_desc, size_in_bytes, data_type)
    }

    pub unsafe extern "C" fn cudnnGetRNNTrainingReserveSize(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        seq_length: ::core::ffi::c_int,
        x_desc: *const ct9::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNTrainingReserveSize(handle, rnn_desc, seq_length, x_desc, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnGetRNNLinLayerMatrixParams(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        pseudo_layer: ::core::ffi::c_int,
        x_desc: ct9::cudnnTensorDescriptor_t,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        lin_layer_id: ::core::ffi::c_int,
        lin_layer_mat_desc: ct9::cudnnFilterDescriptor_t,
        lin_layer_mat: *mut *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNLinLayerMatrixParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w, lin_layer_id, lin_layer_mat_desc, lin_layer_mat)
    }

    pub unsafe extern "C" fn cudnnGetRNNLinLayerBiasParams(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        pseudo_layer: ::core::ffi::c_int,
        x_desc: ct9::cudnnTensorDescriptor_t,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        lin_layer_id: ::core::ffi::c_int,
        lin_layer_bias_desc: ct9::cudnnFilterDescriptor_t,
        lin_layer_bias: *mut *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNLinLayerBiasParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w, lin_layer_id, lin_layer_bias_desc, lin_layer_bias)
    }

    pub unsafe extern "C" fn cudnnRNNForwardTraining(
        handle: ct9::cudnnHandle_t,
        rnn_desc: ct9::cudnnRNNDescriptor_t,
        seq_length: ::core::ffi::c_int,
        x_desc: *const ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hx_desc: ct9::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cx_desc: ct9::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        y_desc: *const ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hy_desc: ct9::cudnnTensorDescriptor_t,
        hy: *mut ::core::ffi::c_void,
        cy_desc: ct9::cudnnTensorDescriptor_t,
        cy: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRNNForwardTraining(handle, rnn_desc, seq_length, x_desc, x, hx_desc, hx, cx_desc, cx, w_desc, w, y_desc, y, hy_desc, hy, cy_desc, cy, workspace, workspace_size, reserve_space, reserve_space_size)
    }

    pub unsafe extern "C" fn cudnnCreateDropoutDescriptor(
        dropout_desc: *mut ct9::cudnnDropoutDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateDropoutDescriptor(dropout_desc)
    }

    pub unsafe extern "C" fn cudnnDestroyDropoutDescriptor(
        dropout_desc: ct9::cudnnDropoutDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroyDropoutDescriptor(dropout_desc)
    }

    pub unsafe extern "C" fn cudnnDropoutGetStatesSize(
        handle: ct9::cudnnHandle_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDropoutGetStatesSize(handle, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnSetDropoutDescriptor(
        dropout_desc: ct9::cudnnDropoutDescriptor_t,
        handle: ct9::cudnnHandle_t,
        dropout: f32,
        states: *mut ::core::ffi::c_void,
        state_size_in_bytes: usize,
        seed: ::core::ffi::c_ulonglong,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)
    }

    pub unsafe extern "C" fn cudnnRestoreDropoutDescriptor(
        dropout_desc: ct9::cudnnDropoutDescriptor_t,
        handle: ct9::cudnnHandle_t,
        dropout: f32,
        states: *mut ::core::ffi::c_void,
        state_size_in_bytes: usize,
        seed: ::core::ffi::c_ulonglong,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRestoreDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed)
    }

    pub unsafe extern "C" fn cudnnCreateCTCLossDescriptor(
        ctc_loss_desc: *mut ct9::cudnnCTCLossDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateCTCLossDescriptor(ctc_loss_desc)
    }

    pub unsafe extern "C" fn cudnnDestroyCTCLossDescriptor(
        ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroyCTCLossDescriptor(ctc_loss_desc)
    }

    pub unsafe extern "C" fn cudnnSetCTCLossDescriptor(
        ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
        comp_type: ct9::cudnnDataType_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetCTCLossDescriptor(ctc_loss_desc, comp_type)
    }

    pub unsafe extern "C" fn cudnnSetCTCLossDescriptorEx(
        ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
        comp_type: ct9::cudnnDataType_t,
        norm_mode: ct9::cudnnLossNormalizationMode_t,
        grad_mode: ct9::cudnnNanPropagation_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetCTCLossDescriptorEx(ctc_loss_desc, comp_type, norm_mode, grad_mode)
    }

    pub unsafe extern "C" fn cudnnGetCTCLossWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        probs_desc: ct9::cudnnTensorDescriptor_t,
        gradients_desc: ct9::cudnnTensorDescriptor_t,
        labels: *const ::core::ffi::c_int,
        label_lengths: *const ::core::ffi::c_int,
        input_lengths: *const ::core::ffi::c_int,
        algo: ct9::cudnnCTCLossAlgo_t,
        ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetCTCLossWorkspaceSize(handle, probs_desc, gradients_desc, labels, label_lengths, input_lengths, algo, ctc_loss_desc, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnCTCLoss(
        handle: ct9::cudnnHandle_t,
        probs_desc: ct9::cudnnTensorDescriptor_t,
        probs: *const ::core::ffi::c_void,
        host_labels: *const ::core::ffi::c_int,
        host_label_lengths: *const ::core::ffi::c_int,
        host_input_lengths: *const ::core::ffi::c_int,
        costs: *mut ::core::ffi::c_void,
        gradients_desc: ct9::cudnnTensorDescriptor_t,
        gradients: *mut ::core::ffi::c_void,
        algo: ct9::cudnnCTCLossAlgo_t,
        ctc_loss_desc: ct9::cudnnCTCLossDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCTCLoss(handle, probs_desc, probs, host_labels, host_label_lengths, host_input_lengths, costs, gradients_desc, gradients, algo, ctc_loss_desc, workspace, workspace_size)
    }

    pub unsafe extern "C" fn cudnnGetTensorNdDescriptor(
        tensor_desc: ct9::cudnnTensorDescriptor_t,
        nb_dims_requested: ::core::ffi::c_int,
        data_type: *mut ct9::cudnnDataType_t,
        nb_dims: *mut ::core::ffi::c_int,
        dim_a: *mut ::core::ffi::c_int,
        stride_a: *mut ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, data_type, nb_dims, dim_a, stride_a)
    }

    pub unsafe extern "C" fn cudnnGetFilterNdDescriptor(
        filter_desc: ct9::cudnnFilterDescriptor_t,
        nb_dims_requested: ::core::ffi::c_int,
        data_type: *mut ct9::cudnnDataType_t,
        format: *mut ct9::cudnnTensorFormat_t,
        nb_dims: *mut ::core::ffi::c_int,
        filter_dim_a: *mut ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetFilterNdDescriptor(filter_desc, nb_dims_requested, data_type, format, nb_dims, filter_dim_a)
    }

    pub unsafe extern "C" fn cudnnCreateActivationDescriptor(
        activation_desc: *mut ct9::cudnnActivationDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreateActivationDescriptor(activation_desc)
    }

    pub unsafe extern "C" fn cudnnDestroyActivationDescriptor(
        activation_desc: ct9::cudnnActivationDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnDestroyActivationDescriptor(activation_desc)
    }

    pub unsafe extern "C" fn cudnnSetActivationDescriptor(
        activation_desc: ct9::cudnnActivationDescriptor_t,
        mode: ct9::cudnnActivationMode_t,
        relu_nan_opt: ct9::cudnnNanPropagation_t,
        coef: f64,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetActivationDescriptor(activation_desc, mode, relu_nan_opt, coef)
    }

    pub unsafe extern "C" fn cudnnCreatePoolingDescriptor(
        pooling_desc: *mut ct9::cudnnPoolingDescriptor_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnCreatePoolingDescriptor(pooling_desc)
    }

    pub unsafe extern "C" fn cudnnSetPooling2dDescriptor(
        pooling_desc: ct9::cudnnPoolingDescriptor_t,
        mode: ct9::cudnnPoolingMode_t,
        maxpooling_nan_opt: ct9::cudnnNanPropagation_t,
        window_height: ::core::ffi::c_int,
        window_width: ::core::ffi::c_int,
        vertical_padding: ::core::ffi::c_int,
        horizontal_padding: ::core::ffi::c_int,
        vertical_stride: ::core::ffi::c_int,
        horizontal_stride: ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetPooling2dDescriptor(pooling_desc, mode, maxpooling_nan_opt, window_height, window_width, vertical_padding, horizontal_padding, vertical_stride, horizontal_stride)
    }

    pub unsafe extern "C" fn cudnnConvolutionBiasActivationForward(
        handle: ct9::cudnnHandle_t,
        alpha1: *const ::core::ffi::c_void,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        algo: ct9::cudnnConvolutionFwdAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        alpha2: *const ::core::ffi::c_void,
        z_desc: ct9::cudnnTensorDescriptor_t,
        z: *const ::core::ffi::c_void,
        bias_desc: ct9::cudnnTensorDescriptor_t,
        bias: *const ::core::ffi::c_void,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        y_desc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnConvolutionBiasActivationForward(handle, alpha1, x_desc, x, w_desc, w, conv_desc, algo, workspace, workspace_size, alpha2, z_desc, z, bias_desc, bias, activation_desc, y_desc, y)
    }

    pub unsafe extern "C" fn cudnnFindConvolutionForwardAlgorithmEx(
        handle: ct9::cudnnHandle_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        y_desc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct9::cudnnConvolutionFwdAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnFindConvolutionForwardAlgorithmEx(handle, x_desc, x, w_desc, w, conv_desc, y_desc, y, requested_algo_count, returned_algo_count, perf_results, workspace, workspace_size)
    }

    pub unsafe extern "C" fn cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle: ct9::cudnnHandle_t,
        w_desc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        dx_desc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct9::cudnnConvolutionBwdDataAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnFindConvolutionBackwardDataAlgorithmEx(handle, w_desc, w, dy_desc, dy, conv_desc, dx_desc, dx, requested_algo_count, returned_algo_count, perf_results, workspace, workspace_size)
    }

    pub unsafe extern "C" fn cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle: ct9::cudnnHandle_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        dw_desc: ct9::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct9::cudnnConvolutionBwdFilterAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, x_desc, x, dy_desc, dy, conv_desc, dw_desc, dw, requested_algo_count, returned_algo_count, perf_results, workspace, workspace_size)
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        bn_ops: ct9::cudnnBatchNormOps_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        z_desc: ct9::cudnnTensorDescriptor_t,
        y_desc: ct9::cudnnTensorDescriptor_t,
        bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mode, bn_ops, x_desc, z_desc, y_desc, bn_sbmv_desc, activation_desc, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        bn_ops: ct9::cudnnBatchNormOps_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        y_desc: ct9::cudnnTensorDescriptor_t,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        dz_desc: ct9::cudnnTensorDescriptor_t,
        dx_desc: ct9::cudnnTensorDescriptor_t,
        d_bn_sb_desc: ct9::cudnnTensorDescriptor_t,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mode, bn_ops, x_desc, y_desc, dy_desc, dz_desc, dx_desc, d_bn_sb_desc, activation_desc, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        bn_ops: ct9::cudnnBatchNormOps_t,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mode, bn_ops, activation_desc, x_desc, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardTrainingEx(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        bn_ops: ct9::cudnnBatchNormOps_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        z_desc: ct9::cudnnTensorDescriptor_t,
        z: *const ::core::ffi::c_void,
        y_desc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bn_sbmv_desc: ct9::cudnnTensorDescriptor_t,
        bn_scale: *const ::core::ffi::c_void,
        bn_bias: *const ::core::ffi::c_void,
        exp_avg_factor: f64,
        running_mean: *mut ::core::ffi::c_void,
        running_var: *mut ::core::ffi::c_void,
        epsilon: f64,
        save_mean: *mut ::core::ffi::c_void,
        save_inv_var: *mut ::core::ffi::c_void,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBatchNormalizationForwardTrainingEx(handle, mode, bn_ops, alpha, beta, x_desc, x, z_desc, z, y_desc, y, bn_sbmv_desc, bn_scale, bn_bias, exp_avg_factor, running_mean, running_var, epsilon, save_mean, save_inv_var, activation_desc, workspace, workspace_size, reserve_space, reserve_space_size)
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationBackwardEx(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        bn_ops: ct9::cudnnBatchNormOps_t,
        alpha_data_diff: *const ::core::ffi::c_void,
        beta_data_diff: *const ::core::ffi::c_void,
        alpha_param_diff: *const ::core::ffi::c_void,
        beta_param_diff: *const ::core::ffi::c_void,
        x_desc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        y_desc: ct9::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dz_desc: ct9::cudnnTensorDescriptor_t,
        dz: *mut ::core::ffi::c_void,
        dx_desc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        d_bn_sb_desc: ct9::cudnnTensorDescriptor_t,
        bn_scale: *const ::core::ffi::c_void,
        bn_bias: *const ::core::ffi::c_void,
        d_bn_scale: *mut ::core::ffi::c_void,
        d_bn_bias: *mut ::core::ffi::c_void,
        epsilon: f64,
        saved_mean: *const ::core::ffi::c_void,
        saved_inv_var: *const ::core::ffi::c_void,
        activation_desc: ct9::cudnnActivationDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBatchNormalizationBackwardEx(handle, mode, bn_ops, alpha_data_diff, beta_data_diff, alpha_param_diff, beta_param_diff, x_desc, x, y_desc, y, dy_desc, dy, dz_desc, dz, dx_desc, dx, d_bn_sb_desc, bn_scale, bn_bias, d_bn_scale, d_bn_bias, epsilon, saved_mean, saved_inv_var, activation_desc, workspace, workspace_size, reserve_space, reserve_space_size)
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        w_desc: ct9::cudnnFilterDescriptor_t,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        dx_desc: ct9::cudnnTensorDescriptor_t,
        algo: ct9::cudnnConvolutionBwdDataAlgo_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, dy_desc, conv_desc, dx_desc, algo, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle: ct9::cudnnHandle_t,
        filter_desc: ct9::cudnnFilterDescriptor_t,
        diff_desc: ct9::cudnnTensorDescriptor_t,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        grad_desc: ct9::cudnnTensorDescriptor_t,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct9::cudnnConvolutionBwdDataAlgoPerf_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc, diff_desc, conv_desc, grad_desc, requested_algo_count, returned_algo_count, perf_results)
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        x_desc: ct9::cudnnTensorDescriptor_t,
        dy_desc: ct9::cudnnTensorDescriptor_t,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        grad_desc: ct9::cudnnFilterDescriptor_t,
        algo: ct9::cudnnConvolutionBwdFilterAlgo_t,
        size_in_bytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, dy_desc, conv_desc, grad_desc, algo, size_in_bytes)
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle: ct9::cudnnHandle_t,
        src_desc: ct9::cudnnTensorDescriptor_t,
        diff_desc: ct9::cudnnTensorDescriptor_t,
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        grad_desc: ct9::cudnnFilterDescriptor_t,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct9::cudnnConvolutionBwdFilterAlgoPerf_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, src_desc, diff_desc, conv_desc, grad_desc, requested_algo_count, returned_algo_count, perf_results)
    }

    pub unsafe extern "C" fn cudnnSetStream(
        handle: ct9::cudnnHandle_t,
        stream_id: ct9::cudaStream_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetStream(handle, stream_id)
    }

    pub unsafe extern "C" fn cudnnGetStream(
        handle: ct9::cudnnHandle_t,
        stream_id: *mut ct9::cudaStream_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetStream(handle, stream_id)
    }

    pub unsafe extern "C" fn cudnnSetTensorNdDescriptor(
        tensor_desc: ct9::cudnnTensorDescriptor_t,
        data_type: ct9::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
        stride_a: *const ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetTensorNdDescriptor(tensor_desc, data_type, nb_dims, dim_a, stride_a)
    }

    pub unsafe extern "C" fn cudnnSetTensorNdDescriptorEx(
        tensor_desc: ct9::cudnnTensorDescriptor_t,
        format: ct9::cudnnTensorFormat_t,
        data_type: ct9::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetTensorNdDescriptorEx(tensor_desc, format, data_type, nb_dims, dim_a)
    }

    pub unsafe extern "C" fn cudnnSetFilterNdDescriptor(
        filter_desc: ct9::cudnnFilterDescriptor_t,
        data_type: ct9::cudnnDataType_t,
        format: ct9::cudnnTensorFormat_t,
        nb_dims: ::core::ffi::c_int,
        filter_dim_a: *const ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetFilterNdDescriptor(filter_desc, data_type, format, nb_dims, filter_dim_a)
    }

    pub unsafe extern "C" fn cudnnSetConvolutionNdDescriptor(
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        array_length: ::core::ffi::c_int,
        pad_a: *const ::core::ffi::c_int,
        filter_stride_a: *const ::core::ffi::c_int,
        dilation_a: *const ::core::ffi::c_int,
        mode: ct9::cudnnConvolutionMode_t,
        compute_type: ct9::cudnnDataType_t,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetConvolutionNdDescriptor(conv_desc, array_length, pad_a, filter_stride_a, dilation_a, mode, compute_type)
    }

    pub unsafe extern "C" fn cudnnSetConvolutionGroupCount(
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        group_count: ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetConvolutionGroupCount(conv_desc, group_count)
    }

    pub unsafe extern "C" fn cudnnGetConvolutionNdForwardOutputDim(
        conv_desc: ct9::cudnnConvolutionDescriptor_t,
        input_tensor_desc: ct9::cudnnTensorDescriptor_t,
        filter_desc: ct9::cudnnFilterDescriptor_t,
        nb_dims: ::core::ffi::c_int,
        tensor_output_dim_a: *mut ::core::ffi::c_int,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_tensor_desc, filter_desc, nb_dims, tensor_output_dim_a)
    }

    pub unsafe extern "C" fn cudnnActivationForward(
        handle: ct9::cudnnHandle_t,
        activationDesc: ct9::cudnnActivationDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y)
    }

    pub unsafe extern "C" fn cudnnActivationBackward(
        handle: ct9::cudnnHandle_t,
        activationDesc: ct9::cudnnActivationDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    }

    pub unsafe extern "C" fn cudnnPoolingForward(
        handle: ct9::cudnnHandle_t,
        poolingDesc: ct9::cudnnPoolingDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y)
    }

    pub unsafe extern "C" fn cudnnPoolingBackward(
        handle: ct9::cudnnHandle_t,
        poolingDesc: ct9::cudnnPoolingDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    }

    // VEGA-PATCH 2026-06-29 sprint-orch GAP2 worker3 — 12 wrapper Family B restanti.
    // Le impl super::* prendono i32 raw per i parametri enum (cudnnForwardMode_t,
    // cudnnBatchNormMode_t, cudnnConvolutionBwdDataAlgo_t, ecc.) quindi NON serve
    // cast enum cudnn8/9 — passthrough diretto identico al pattern Activation/Pooling.
    // Sblocca RNN/LSTM + BN/Conv-bwd in un addestramento QAT-LoRA completo.

    pub unsafe extern "C" fn cudnnRNNForward(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        fwdMode: ct9::cudnnForwardMode_t,
        devSeqLengths: *const i32,
        xDesc: ct9::cudnnRNNDataDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnRNNDataDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hDesc: ct9::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        hy: *mut ::core::ffi::c_void,
        cDesc: ct9::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        cy: *mut ::core::ffi::c_void,
        weightSpaceSize: usize,
        weightSpace: *const ::core::ffi::c_void,
        workSpaceSize: usize,
        workSpace: *mut ::core::ffi::c_void,
        reserveSpaceSize: usize,
        reserveSpace: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRNNForward(handle, rnnDesc, fwdMode.0 as i32, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace)
    }

    pub unsafe extern "C" fn cudnnRNNForwardInference(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hxDesc: ct9::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cxDesc: ct9::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        wDesc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        yDesc: *const ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hyDesc: ct9::cudnnTensorDescriptor_t,
        hy: *mut ::core::ffi::c_void,
        cyDesc: ct9::cudnnTensorDescriptor_t,
        cy: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workspaceSize)
    }

    pub unsafe extern "C" fn cudnnRNNBackwardData(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        seqLength: i32,
        yDesc: *const ct9::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: *const ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dhyDesc: ct9::cudnnTensorDescriptor_t,
        dhy: *const ::core::ffi::c_void,
        dcyDesc: ct9::cudnnTensorDescriptor_t,
        dcy: *const ::core::ffi::c_void,
        wDesc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        hxDesc: ct9::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cxDesc: ct9::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        dxDesc: *const ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        dhxDesc: ct9::cudnnTensorDescriptor_t,
        dhx: *mut ::core::ffi::c_void,
        dcxDesc: ct9::cudnnTensorDescriptor_t,
        dcx: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        reserve: *mut ::core::ffi::c_void,
        reserveSize: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workspaceSize, reserve, reserveSize)
    }

    pub unsafe extern "C" fn cudnnRNNBackwardWeights(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hxDesc: ct9::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        yDesc: *const ct9::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        dwDesc: ct9::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
        reserve: *const ::core::ffi::c_void,
        reserveSize: usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace, workspaceSize, dwDesc, dw, reserve, reserveSize)
    }

    pub unsafe extern "C" fn cudnnSetRNNDescriptor_v8(
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        algo: ct9::cudnnRNNAlgo_t,
        cellMode: ct9::cudnnRNNMode_t,
        biasMode: ct9::cudnnRNNBiasMode_t,
        dirMode: ct9::cudnnDirectionMode_t,
        inputMode: ct9::cudnnRNNInputMode_t,
        dataType: ct9::cudnnDataType_t,
        mathPrec: ct9::cudnnDataType_t,
        mathType: ct9::cudnnMathType_t,
        inputSize: i32,
        hiddenSize: i32,
        projSize: i32,
        numLayers: i32,
        dropoutDesc: ct9::cudnnDropoutDescriptor_t,
        auxFlags: u32,
    ) -> ct9::cudnnStatus_t {
        super::cudnnSetRNNDescriptor_v8(rnnDesc, algo.0 as i32, cellMode.0 as i32, biasMode.0 as i32, dirMode.0 as i32, inputMode.0 as i32, dataType.0 as i32, mathPrec.0 as i32, mathType.0 as i32, inputSize, hiddenSize, projSize, numLayers, dropoutDesc as *mut ::core::ffi::c_void, auxFlags)
    }

    pub unsafe extern "C" fn cudnnGetRNNWorkspaceSize(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct9::cudnnTensorDescriptor_t,
        sizeInBytes: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes)
    }

    pub unsafe extern "C" fn cudnnGetRNNTempSpaceSizes(
        handle: ct9::cudnnHandle_t,
        rnnDesc: ct9::cudnnRNNDescriptor_t,
        fwdMode: ct9::cudnnForwardMode_t,
        xDesc: ct9::cudnnRNNDataDescriptor_t,
        workSpaceSize: *mut usize,
        reserveSpaceSize: *mut usize,
    ) -> ct9::cudnnStatus_t {
        super::cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fwdMode.0 as i32, xDesc, workSpaceSize, reserveSpaceSize)
    }

    pub unsafe extern "C" fn cudnnConvolutionBackwardData(
        handle: ct9::cudnnHandle_t,
        alpha: *const ::core::ffi::c_void,
        wDesc: ct9::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        convDesc: ct9::cudnnConvolutionDescriptor_t,
        algo: ct9::cudnnConvolutionBwdDataAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo.0 as i32, workspace, workspaceSize, beta, dxDesc, dx)
    }

    pub unsafe extern "C" fn cudnnConvolutionBackwardFilter(
        handle: ct9::cudnnHandle_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        convDesc: ct9::cudnnConvolutionDescriptor_t,
        algo: ct9::cudnnConvolutionBwdFilterAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        beta: *const ::core::ffi::c_void,
        dwDesc: ct9::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo.0 as i32, workspace, workspaceSize, beta, dwDesc, dw)
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardTraining(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bnSbmvDesc: ct9::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        bnBias: *const ::core::ffi::c_void,
        expAvgFactor: f64,
        runningMean: *mut ::core::ffi::c_void,
        runningVar: *mut ::core::ffi::c_void,
        epsilon: f64,
        saveMean: *mut ::core::ffi::c_void,
        saveInvVar: *mut ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBatchNormalizationForwardTraining(handle, mode.0 as i32, alpha as *mut _, beta as *mut _, xDesc, x, yDesc, y, bnSbmvDesc, bnScale as *mut _, bnBias as *mut _, expAvgFactor, runningMean, runningVar, epsilon, saveMean, saveInvVar)
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardInference(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct9::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bnSbmvDesc: ct9::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        bnBias: *const ::core::ffi::c_void,
        estMean: *const ::core::ffi::c_void,
        estVar: *const ::core::ffi::c_void,
        epsilon: f64,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBatchNormalizationForwardInference(handle, mode.0 as i32, alpha as *mut _, beta as *mut _, xDesc, x, yDesc, y, bnSbmvDesc, bnScale as *mut _, bnBias as *mut _, estMean as *mut _, estVar as *mut _, epsilon)
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationBackward(
        handle: ct9::cudnnHandle_t,
        mode: ct9::cudnnBatchNormMode_t,
        alphaDataDiff: *const ::core::ffi::c_void,
        betaDataDiff: *const ::core::ffi::c_void,
        alphaParamDiff: *const ::core::ffi::c_void,
        betaParamDiff: *const ::core::ffi::c_void,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dxDesc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        bnSbdDesc: ct9::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        resultBnScaleDiff: *mut ::core::ffi::c_void,
        resultBnBiasDiff: *mut ::core::ffi::c_void,
        epsilon: f64,
        savedMean: *const ::core::ffi::c_void,
        savedInvVar: *const ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnBatchNormalizationBackward(handle, mode.0 as i32, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, bnSbdDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVar)
    }

    // VEGA-PATCH 2026-06-30 sprint Rust E LITE worker 2/3 TASK 1 — LayerNorm bwd safe
    pub unsafe extern "C" fn cudnnLayerNormBackward(
        handle: ct9::cudnnHandle_t,
        xDesc: ct9::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct9::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dxDesc: ct9::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        bnParamDesc: ct9::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        resultBnScaleDiff: *mut ::core::ffi::c_void,
        resultBnBiasDiff: *mut ::core::ffi::c_void,
        epsilon: f64,
        savedMean: *const ::core::ffi::c_void,
        savedInvVar: *const ::core::ffi::c_void,
    ) -> ct9::cudnnStatus_t {
        super::cudnnLayerNormBackward(handle, xDesc, x, dyDesc, dy, dxDesc, dx, bnParamDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVar)
    }
}

pub mod dnn8_b {
    use cuda_types::cudnn8 as ct8;
    use std::mem;

    fn s9_to_s8(s: cuda_types::cudnn9::cudnnStatus_t) -> ct8::cudnnStatus_t {
        // cudnnError_t e' #[repr(transparent)] NonZeroU32 in entrambi: transmute SAFE.
        match s {
            Ok(()) => Ok(()),
            Err(e) => Err(unsafe { mem::transmute::<cuda_types::cudnn9::cudnnError_t, ct8::cudnnError_t>(e) }),
        }
    }

    pub unsafe extern "C" fn cudnnGetProperty(
        type_: ct8::libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetProperty(type_, value))
    }

    pub unsafe extern "C" fn cudnnBackendCreateDescriptor(
        descriptor_type: ct8::cudnnBackendDescriptorType_t,
        descriptor: *mut ct8::cudnnBackendDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendCreateDescriptor(mem::transmute(descriptor_type), mem::transmute(descriptor)))
    }

    pub unsafe extern "C" fn cudnnBackendDestroyDescriptor(
        descriptor: ct8::cudnnBackendDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendDestroyDescriptor(mem::transmute(descriptor)))
    }

    pub unsafe extern "C" fn cudnnBackendSetAttribute(
        descriptor: ct8::cudnnBackendDescriptor_t,
        attribute_name: ct8::cudnnBackendAttributeName_t,
        attribute_type: ct8::cudnnBackendAttributeType_t,
        element_count: i64,
        array_of_elements: *const ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendSetAttribute(mem::transmute(descriptor), mem::transmute(attribute_name), mem::transmute(attribute_type), element_count, array_of_elements))
    }

    pub unsafe extern "C" fn cudnnBackendGetAttribute(
        descriptor: ct8::cudnnBackendDescriptor_t,
        attribute_name: ct8::cudnnBackendAttributeName_t,
        attribute_type: ct8::cudnnBackendAttributeType_t,
        requested_element_count: i64,
        element_count: *mut i64,
        array_of_elements: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendGetAttribute(mem::transmute(descriptor), mem::transmute(attribute_name), mem::transmute(attribute_type), requested_element_count, element_count, array_of_elements))
    }

    pub unsafe extern "C" fn cudnnBackendFinalize(
        descriptor: ct8::cudnnBackendDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendFinalize(mem::transmute(descriptor)))
    }

    pub unsafe extern "C" fn cudnnBackendExecute(
        handle: ct8::cudnnHandle_t,
        execution_plan: ct8::cudnnBackendDescriptor_t,
        variant_pack: ct8::cudnnBackendDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBackendExecute(handle, mem::transmute(execution_plan), mem::transmute(variant_pack)))
    }

    pub unsafe extern "C" fn cudnnCreateSpatialTransformerDescriptor(
        st_desc: *mut ct8::cudnnSpatialTransformerDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateSpatialTransformerDescriptor(st_desc))
    }

    pub unsafe extern "C" fn cudnnDestroySpatialTransformerDescriptor(
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroySpatialTransformerDescriptor(st_desc))
    }

    pub unsafe extern "C" fn cudnnSetSpatialTransformerNdDescriptor(
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
        sampler_type: ct8::cudnnSamplerType_t,
        data_type: ct8::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetSpatialTransformerNdDescriptor(st_desc, mem::transmute(sampler_type), mem::transmute(data_type), nb_dims, dim_a))
    }

    pub unsafe extern "C" fn cudnnSpatialTfGridGeneratorForward(
        handle: ct8::cudnnHandle_t,
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
        theta: *const ::core::ffi::c_void,
        grid: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSpatialTfGridGeneratorForward(handle, st_desc, theta, grid))
    }

    pub unsafe extern "C" fn cudnnSpatialTfGridGeneratorBackward(
        handle: ct8::cudnnHandle_t,
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
        dgrid: *const ::core::ffi::c_void,
        dtheta: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSpatialTfGridGeneratorBackward(handle, st_desc, dgrid, dtheta))
    }

    pub unsafe extern "C" fn cudnnSpatialTfSamplerForward(
        handle: ct8::cudnnHandle_t,
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        grid: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        y_desc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSpatialTfSamplerForward(handle, st_desc, alpha, x_desc, x, grid, beta, y_desc, y))
    }

    pub unsafe extern "C" fn cudnnSpatialTfSamplerBackward(
        handle: ct8::cudnnHandle_t,
        st_desc: ct8::cudnnSpatialTransformerDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dx_desc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        alpha_dgrid: *const ::core::ffi::c_void,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        grid: *const ::core::ffi::c_void,
        beta_dgrid: *const ::core::ffi::c_void,
        dgrid: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSpatialTfSamplerBackward(handle, st_desc, alpha, x_desc, x, beta, dx_desc, dx, alpha_dgrid, dy_desc, dy, grid, beta_dgrid, dgrid))
    }

    pub unsafe extern "C" fn cudnnCreateRNNDescriptor(
        rnn_desc: *mut ct8::cudnnRNNDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateRNNDescriptor(rnn_desc))
    }

    pub unsafe extern "C" fn cudnnDestroyRNNDescriptor(
        rnn_desc: ct8::cudnnRNNDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroyRNNDescriptor(rnn_desc))
    }

    pub unsafe extern "C" fn cudnnCreateRNNDataDescriptor(
        rnn_data_desc: *mut ct8::cudnnRNNDataDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateRNNDataDescriptor(rnn_data_desc))
    }

    pub unsafe extern "C" fn cudnnDestroyRNNDataDescriptor(
        rnn_data_desc: ct8::cudnnRNNDataDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroyRNNDataDescriptor(rnn_data_desc))
    }

    pub unsafe extern "C" fn cudnnSetRNNDataDescriptor(
        rnn_data_desc: ct8::cudnnRNNDataDescriptor_t,
        data_type: ct8::cudnnDataType_t,
        layout: ct8::cudnnRNNDataLayout_t,
        max_seq_length: ::core::ffi::c_int,
        batch_size: ::core::ffi::c_int,
        vector_size: ::core::ffi::c_int,
        seq_length_array: *const ::core::ffi::c_int,
        padding_fill: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetRNNDataDescriptor(rnn_data_desc, mem::transmute(data_type), mem::transmute(layout), max_seq_length, batch_size, vector_size, seq_length_array, padding_fill))
    }

    pub unsafe extern "C" fn cudnnSetRNNDescriptor_v6(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        hidden_size: ::core::ffi::c_int,
        num_layers: ::core::ffi::c_int,
        dropout_desc: ct8::cudnnDropoutDescriptor_t,
        input_mode: ct8::cudnnRNNInputMode_t,
        direction: ct8::cudnnDirectionMode_t,
        cell_mode: ct8::cudnnRNNMode_t,
        algo: ct8::cudnnRNNAlgo_t,
        math_prec: ct8::cudnnDataType_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetRNNDescriptor_v6(handle, rnn_desc, hidden_size, num_layers, dropout_desc, mem::transmute(input_mode), mem::transmute(direction), mem::transmute(cell_mode), mem::transmute(algo), mem::transmute(math_prec)))
    }

    pub unsafe extern "C" fn cudnnSetRNNMatrixMathType(
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        m_type: ct8::cudnnMathType_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetRNNMatrixMathType(rnn_desc, mem::transmute(m_type)))
    }

    pub unsafe extern "C" fn cudnnSetRNNProjectionLayers(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        rec_proj_size: ::core::ffi::c_int,
        out_proj_size: ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetRNNProjectionLayers(handle, rnn_desc, rec_proj_size, out_proj_size))
    }

    pub unsafe extern "C" fn cudnnGetRNNParamsSize(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
        data_type: ct8::cudnnDataType_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNParamsSize(handle, rnn_desc, x_desc, size_in_bytes, mem::transmute(data_type)))
    }

    pub unsafe extern "C" fn cudnnGetRNNTrainingReserveSize(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        seq_length: ::core::ffi::c_int,
        x_desc: *const ct8::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNTrainingReserveSize(handle, rnn_desc, seq_length, mem::transmute(x_desc), size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnGetRNNLinLayerMatrixParams(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        pseudo_layer: ::core::ffi::c_int,
        x_desc: ct8::cudnnTensorDescriptor_t,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        lin_layer_id: ::core::ffi::c_int,
        lin_layer_mat_desc: ct8::cudnnFilterDescriptor_t,
        lin_layer_mat: *mut *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNLinLayerMatrixParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w, lin_layer_id, lin_layer_mat_desc, lin_layer_mat))
    }

    pub unsafe extern "C" fn cudnnGetRNNLinLayerBiasParams(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        pseudo_layer: ::core::ffi::c_int,
        x_desc: ct8::cudnnTensorDescriptor_t,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        lin_layer_id: ::core::ffi::c_int,
        lin_layer_bias_desc: ct8::cudnnFilterDescriptor_t,
        lin_layer_bias: *mut *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNLinLayerBiasParams(handle, rnn_desc, pseudo_layer, x_desc, w_desc, w, lin_layer_id, lin_layer_bias_desc, lin_layer_bias))
    }

    pub unsafe extern "C" fn cudnnRNNForwardTraining(
        handle: ct8::cudnnHandle_t,
        rnn_desc: ct8::cudnnRNNDescriptor_t,
        seq_length: ::core::ffi::c_int,
        x_desc: *const ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hx_desc: ct8::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cx_desc: ct8::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        y_desc: *const ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hy_desc: ct8::cudnnTensorDescriptor_t,
        hy: *mut ::core::ffi::c_void,
        cy_desc: ct8::cudnnTensorDescriptor_t,
        cy: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRNNForwardTraining(handle, rnn_desc, seq_length, mem::transmute(x_desc), x, hx_desc, hx, cx_desc, cx, w_desc, w, mem::transmute(y_desc), y, hy_desc, hy, cy_desc, cy, workspace, workspace_size, reserve_space, reserve_space_size))
    }

    pub unsafe extern "C" fn cudnnCreateDropoutDescriptor(
        dropout_desc: *mut ct8::cudnnDropoutDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateDropoutDescriptor(dropout_desc))
    }

    pub unsafe extern "C" fn cudnnDestroyDropoutDescriptor(
        dropout_desc: ct8::cudnnDropoutDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroyDropoutDescriptor(dropout_desc))
    }

    pub unsafe extern "C" fn cudnnDropoutGetStatesSize(
        handle: ct8::cudnnHandle_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDropoutGetStatesSize(handle, size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnSetDropoutDescriptor(
        dropout_desc: ct8::cudnnDropoutDescriptor_t,
        handle: ct8::cudnnHandle_t,
        dropout: f32,
        states: *mut ::core::ffi::c_void,
        state_size_in_bytes: usize,
        seed: ::core::ffi::c_ulonglong,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed))
    }

    pub unsafe extern "C" fn cudnnRestoreDropoutDescriptor(
        dropout_desc: ct8::cudnnDropoutDescriptor_t,
        handle: ct8::cudnnHandle_t,
        dropout: f32,
        states: *mut ::core::ffi::c_void,
        state_size_in_bytes: usize,
        seed: ::core::ffi::c_ulonglong,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRestoreDropoutDescriptor(dropout_desc, handle, dropout, states, state_size_in_bytes, seed))
    }

    pub unsafe extern "C" fn cudnnCreateCTCLossDescriptor(
        ctc_loss_desc: *mut ct8::cudnnCTCLossDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateCTCLossDescriptor(ctc_loss_desc))
    }

    pub unsafe extern "C" fn cudnnDestroyCTCLossDescriptor(
        ctc_loss_desc: ct8::cudnnCTCLossDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroyCTCLossDescriptor(ctc_loss_desc))
    }

    pub unsafe extern "C" fn cudnnSetCTCLossDescriptor(
        ctc_loss_desc: ct8::cudnnCTCLossDescriptor_t,
        comp_type: ct8::cudnnDataType_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetCTCLossDescriptor(ctc_loss_desc, mem::transmute(comp_type)))
    }

    pub unsafe extern "C" fn cudnnSetCTCLossDescriptorEx(
        ctc_loss_desc: ct8::cudnnCTCLossDescriptor_t,
        comp_type: ct8::cudnnDataType_t,
        norm_mode: ct8::cudnnLossNormalizationMode_t,
        grad_mode: ct8::cudnnNanPropagation_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetCTCLossDescriptorEx(ctc_loss_desc, mem::transmute(comp_type), mem::transmute(norm_mode), mem::transmute(grad_mode)))
    }

    pub unsafe extern "C" fn cudnnGetCTCLossWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        probs_desc: ct8::cudnnTensorDescriptor_t,
        gradients_desc: ct8::cudnnTensorDescriptor_t,
        labels: *const ::core::ffi::c_int,
        label_lengths: *const ::core::ffi::c_int,
        input_lengths: *const ::core::ffi::c_int,
        algo: ct8::cudnnCTCLossAlgo_t,
        ctc_loss_desc: ct8::cudnnCTCLossDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetCTCLossWorkspaceSize(handle, probs_desc, gradients_desc, labels, label_lengths, input_lengths, mem::transmute(algo), ctc_loss_desc, size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnCTCLoss(
        handle: ct8::cudnnHandle_t,
        probs_desc: ct8::cudnnTensorDescriptor_t,
        probs: *const ::core::ffi::c_void,
        host_labels: *const ::core::ffi::c_int,
        host_label_lengths: *const ::core::ffi::c_int,
        host_input_lengths: *const ::core::ffi::c_int,
        costs: *mut ::core::ffi::c_void,
        gradients_desc: ct8::cudnnTensorDescriptor_t,
        gradients: *mut ::core::ffi::c_void,
        algo: ct8::cudnnCTCLossAlgo_t,
        ctc_loss_desc: ct8::cudnnCTCLossDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCTCLoss(handle, probs_desc, probs, host_labels, host_label_lengths, host_input_lengths, costs, gradients_desc, gradients, mem::transmute(algo), ctc_loss_desc, workspace, workspace_size))
    }

    pub unsafe extern "C" fn cudnnGetTensorNdDescriptor(
        tensor_desc: ct8::cudnnTensorDescriptor_t,
        nb_dims_requested: ::core::ffi::c_int,
        data_type: *mut ct8::cudnnDataType_t,
        nb_dims: *mut ::core::ffi::c_int,
        dim_a: *mut ::core::ffi::c_int,
        stride_a: *mut ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetTensorNdDescriptor(tensor_desc, nb_dims_requested, mem::transmute(data_type), nb_dims, dim_a, stride_a))
    }

    pub unsafe extern "C" fn cudnnGetFilterNdDescriptor(
        filter_desc: ct8::cudnnFilterDescriptor_t,
        nb_dims_requested: ::core::ffi::c_int,
        data_type: *mut ct8::cudnnDataType_t,
        format: *mut ct8::cudnnTensorFormat_t,
        nb_dims: *mut ::core::ffi::c_int,
        filter_dim_a: *mut ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetFilterNdDescriptor(filter_desc, nb_dims_requested, mem::transmute(data_type), mem::transmute(format), nb_dims, filter_dim_a))
    }

    pub unsafe extern "C" fn cudnnCreateActivationDescriptor(
        activation_desc: *mut ct8::cudnnActivationDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreateActivationDescriptor(activation_desc))
    }

    pub unsafe extern "C" fn cudnnDestroyActivationDescriptor(
        activation_desc: ct8::cudnnActivationDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnDestroyActivationDescriptor(activation_desc))
    }

    pub unsafe extern "C" fn cudnnSetActivationDescriptor(
        activation_desc: ct8::cudnnActivationDescriptor_t,
        mode: ct8::cudnnActivationMode_t,
        relu_nan_opt: ct8::cudnnNanPropagation_t,
        coef: f64,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetActivationDescriptor(activation_desc, mem::transmute(mode), mem::transmute(relu_nan_opt), coef))
    }

    pub unsafe extern "C" fn cudnnCreatePoolingDescriptor(
        pooling_desc: *mut ct8::cudnnPoolingDescriptor_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnCreatePoolingDescriptor(pooling_desc))
    }

    pub unsafe extern "C" fn cudnnSetPooling2dDescriptor(
        pooling_desc: ct8::cudnnPoolingDescriptor_t,
        mode: ct8::cudnnPoolingMode_t,
        maxpooling_nan_opt: ct8::cudnnNanPropagation_t,
        window_height: ::core::ffi::c_int,
        window_width: ::core::ffi::c_int,
        vertical_padding: ::core::ffi::c_int,
        horizontal_padding: ::core::ffi::c_int,
        vertical_stride: ::core::ffi::c_int,
        horizontal_stride: ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetPooling2dDescriptor(pooling_desc, mem::transmute(mode), mem::transmute(maxpooling_nan_opt), window_height, window_width, vertical_padding, horizontal_padding, vertical_stride, horizontal_stride))
    }

    pub unsafe extern "C" fn cudnnConvolutionBiasActivationForward(
        handle: ct8::cudnnHandle_t,
        alpha1: *const ::core::ffi::c_void,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        algo: ct8::cudnnConvolutionFwdAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        alpha2: *const ::core::ffi::c_void,
        z_desc: ct8::cudnnTensorDescriptor_t,
        z: *const ::core::ffi::c_void,
        bias_desc: ct8::cudnnTensorDescriptor_t,
        bias: *const ::core::ffi::c_void,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        y_desc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnConvolutionBiasActivationForward(handle, alpha1, x_desc, x, w_desc, w, conv_desc, mem::transmute(algo), workspace, workspace_size, alpha2, z_desc, z, bias_desc, bias, activation_desc, y_desc, y))
    }

    pub unsafe extern "C" fn cudnnFindConvolutionForwardAlgorithmEx(
        handle: ct8::cudnnHandle_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        y_desc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct8::cudnnConvolutionFwdAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnFindConvolutionForwardAlgorithmEx(handle, x_desc, x, w_desc, w, conv_desc, y_desc, y, requested_algo_count, returned_algo_count, mem::transmute(perf_results), workspace, workspace_size))
    }

    pub unsafe extern "C" fn cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle: ct8::cudnnHandle_t,
        w_desc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        dx_desc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct8::cudnnConvolutionBwdDataAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnFindConvolutionBackwardDataAlgorithmEx(handle, w_desc, w, dy_desc, dy, conv_desc, dx_desc, dx, requested_algo_count, returned_algo_count, mem::transmute(perf_results), workspace, workspace_size))
    }

    pub unsafe extern "C" fn cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle: ct8::cudnnHandle_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        dw_desc: ct8::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct8::cudnnConvolutionBwdFilterAlgoPerf_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnFindConvolutionBackwardFilterAlgorithmEx(handle, x_desc, x, dy_desc, dy, conv_desc, dw_desc, dw, requested_algo_count, returned_algo_count, mem::transmute(perf_results), workspace, workspace_size))
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        bn_ops: ct8::cudnnBatchNormOps_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        z_desc: ct8::cudnnTensorDescriptor_t,
        y_desc: ct8::cudnnTensorDescriptor_t,
        bn_sbmv_desc: ct8::cudnnTensorDescriptor_t,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle, mem::transmute(mode), mem::transmute(bn_ops), x_desc, z_desc, y_desc, bn_sbmv_desc, activation_desc, size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        bn_ops: ct8::cudnnBatchNormOps_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        y_desc: ct8::cudnnTensorDescriptor_t,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        dz_desc: ct8::cudnnTensorDescriptor_t,
        dx_desc: ct8::cudnnTensorDescriptor_t,
        d_bn_sb_desc: ct8::cudnnTensorDescriptor_t,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetBatchNormalizationBackwardExWorkspaceSize(handle, mem::transmute(mode), mem::transmute(bn_ops), x_desc, y_desc, dy_desc, dz_desc, dx_desc, d_bn_sb_desc, activation_desc, size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        bn_ops: ct8::cudnnBatchNormOps_t,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetBatchNormalizationTrainingExReserveSpaceSize(handle, mem::transmute(mode), mem::transmute(bn_ops), activation_desc, x_desc, size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardTrainingEx(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        bn_ops: ct8::cudnnBatchNormOps_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        z_desc: ct8::cudnnTensorDescriptor_t,
        z: *const ::core::ffi::c_void,
        y_desc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bn_sbmv_desc: ct8::cudnnTensorDescriptor_t,
        bn_scale: *const ::core::ffi::c_void,
        bn_bias: *const ::core::ffi::c_void,
        exp_avg_factor: f64,
        running_mean: *mut ::core::ffi::c_void,
        running_var: *mut ::core::ffi::c_void,
        epsilon: f64,
        save_mean: *mut ::core::ffi::c_void,
        save_inv_var: *mut ::core::ffi::c_void,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBatchNormalizationForwardTrainingEx(handle, mem::transmute(mode), mem::transmute(bn_ops), alpha, beta, x_desc, x, z_desc, z, y_desc, y, bn_sbmv_desc, bn_scale, bn_bias, exp_avg_factor, running_mean, running_var, epsilon, save_mean, save_inv_var, activation_desc, workspace, workspace_size, reserve_space, reserve_space_size))
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationBackwardEx(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        bn_ops: ct8::cudnnBatchNormOps_t,
        alpha_data_diff: *const ::core::ffi::c_void,
        beta_data_diff: *const ::core::ffi::c_void,
        alpha_param_diff: *const ::core::ffi::c_void,
        beta_param_diff: *const ::core::ffi::c_void,
        x_desc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        y_desc: ct8::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dz_desc: ct8::cudnnTensorDescriptor_t,
        dz: *mut ::core::ffi::c_void,
        dx_desc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        d_bn_sb_desc: ct8::cudnnTensorDescriptor_t,
        bn_scale: *const ::core::ffi::c_void,
        bn_bias: *const ::core::ffi::c_void,
        d_bn_scale: *mut ::core::ffi::c_void,
        d_bn_bias: *mut ::core::ffi::c_void,
        epsilon: f64,
        saved_mean: *const ::core::ffi::c_void,
        saved_inv_var: *const ::core::ffi::c_void,
        activation_desc: ct8::cudnnActivationDescriptor_t,
        workspace: *mut ::core::ffi::c_void,
        workspace_size: usize,
        reserve_space: *mut ::core::ffi::c_void,
        reserve_space_size: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBatchNormalizationBackwardEx(handle, mem::transmute(mode), mem::transmute(bn_ops), alpha_data_diff, beta_data_diff, alpha_param_diff, beta_param_diff, x_desc, x, y_desc, y, dy_desc, dy, dz_desc, dz, dx_desc, dx, d_bn_sb_desc, bn_scale, bn_bias, d_bn_scale, d_bn_bias, epsilon, saved_mean, saved_inv_var, activation_desc, workspace, workspace_size, reserve_space, reserve_space_size))
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        w_desc: ct8::cudnnFilterDescriptor_t,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        dx_desc: ct8::cudnnTensorDescriptor_t,
        algo: ct8::cudnnConvolutionBwdDataAlgo_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetConvolutionBackwardDataWorkspaceSize(handle, w_desc, dy_desc, conv_desc, dx_desc, mem::transmute(algo), size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle: ct8::cudnnHandle_t,
        filter_desc: ct8::cudnnFilterDescriptor_t,
        diff_desc: ct8::cudnnTensorDescriptor_t,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        grad_desc: ct8::cudnnTensorDescriptor_t,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct8::cudnnConvolutionBwdDataAlgoPerf_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, filter_desc, diff_desc, conv_desc, grad_desc, requested_algo_count, returned_algo_count, mem::transmute(perf_results)))
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        x_desc: ct8::cudnnTensorDescriptor_t,
        dy_desc: ct8::cudnnTensorDescriptor_t,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        grad_desc: ct8::cudnnFilterDescriptor_t,
        algo: ct8::cudnnConvolutionBwdFilterAlgo_t,
        size_in_bytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, x_desc, dy_desc, conv_desc, grad_desc, mem::transmute(algo), size_in_bytes))
    }

    pub unsafe extern "C" fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle: ct8::cudnnHandle_t,
        src_desc: ct8::cudnnTensorDescriptor_t,
        diff_desc: ct8::cudnnTensorDescriptor_t,
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        grad_desc: ct8::cudnnFilterDescriptor_t,
        requested_algo_count: ::core::ffi::c_int,
        returned_algo_count: *mut ::core::ffi::c_int,
        perf_results: *mut ct8::cudnnConvolutionBwdFilterAlgoPerf_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, src_desc, diff_desc, conv_desc, grad_desc, requested_algo_count, returned_algo_count, mem::transmute(perf_results)))
    }

    pub unsafe extern "C" fn cudnnSetStream(
        handle: ct8::cudnnHandle_t,
        stream_id: ct8::cudaStream_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetStream(handle, mem::transmute(stream_id)))
    }

    pub unsafe extern "C" fn cudnnGetStream(
        handle: ct8::cudnnHandle_t,
        stream_id: *mut ct8::cudaStream_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetStream(handle, mem::transmute(stream_id)))
    }

    pub unsafe extern "C" fn cudnnSetTensorNdDescriptor(
        tensor_desc: ct8::cudnnTensorDescriptor_t,
        data_type: ct8::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
        stride_a: *const ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetTensorNdDescriptor(tensor_desc, mem::transmute(data_type), nb_dims, dim_a, stride_a))
    }

    pub unsafe extern "C" fn cudnnSetTensorNdDescriptorEx(
        tensor_desc: ct8::cudnnTensorDescriptor_t,
        format: ct8::cudnnTensorFormat_t,
        data_type: ct8::cudnnDataType_t,
        nb_dims: ::core::ffi::c_int,
        dim_a: *const ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetTensorNdDescriptorEx(tensor_desc, mem::transmute(format), mem::transmute(data_type), nb_dims, dim_a))
    }

    pub unsafe extern "C" fn cudnnSetFilterNdDescriptor(
        filter_desc: ct8::cudnnFilterDescriptor_t,
        data_type: ct8::cudnnDataType_t,
        format: ct8::cudnnTensorFormat_t,
        nb_dims: ::core::ffi::c_int,
        filter_dim_a: *const ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetFilterNdDescriptor(filter_desc, mem::transmute(data_type), mem::transmute(format), nb_dims, filter_dim_a))
    }

    pub unsafe extern "C" fn cudnnSetConvolutionNdDescriptor(
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        array_length: ::core::ffi::c_int,
        pad_a: *const ::core::ffi::c_int,
        filter_stride_a: *const ::core::ffi::c_int,
        dilation_a: *const ::core::ffi::c_int,
        mode: ct8::cudnnConvolutionMode_t,
        compute_type: ct8::cudnnDataType_t,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetConvolutionNdDescriptor(conv_desc, array_length, pad_a, filter_stride_a, dilation_a, mem::transmute(mode), mem::transmute(compute_type)))
    }

    pub unsafe extern "C" fn cudnnSetConvolutionGroupCount(
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        group_count: ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetConvolutionGroupCount(conv_desc, group_count))
    }

    pub unsafe extern "C" fn cudnnGetConvolutionNdForwardOutputDim(
        conv_desc: ct8::cudnnConvolutionDescriptor_t,
        input_tensor_desc: ct8::cudnnTensorDescriptor_t,
        filter_desc: ct8::cudnnFilterDescriptor_t,
        nb_dims: ::core::ffi::c_int,
        tensor_output_dim_a: *mut ::core::ffi::c_int,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetConvolutionNdForwardOutputDim(conv_desc, input_tensor_desc, filter_desc, nb_dims, tensor_output_dim_a))
    }

    pub unsafe extern "C" fn cudnnActivationForward(
        handle: ct8::cudnnHandle_t,
        activationDesc: ct8::cudnnActivationDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        // I tipi opachi (*mut cudnnContext etc) sono alias super::cudnn::xxx identici tra cudnn8 e cudnn9.
        s9_to_s8(super::cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y))
    }

    pub unsafe extern "C" fn cudnnActivationBackward(
        handle: ct8::cudnnHandle_t,
        activationDesc: ct8::cudnnActivationDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx))
    }

    pub unsafe extern "C" fn cudnnPoolingForward(
        handle: ct8::cudnnHandle_t,
        poolingDesc: ct8::cudnnPoolingDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y))
    }

    pub unsafe extern "C" fn cudnnPoolingBackward(
        handle: ct8::cudnnHandle_t,
        poolingDesc: ct8::cudnnPoolingDescriptor_t,
        alpha: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnPoolingBackward(handle, poolingDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx))
    }

    // VEGA-PATCH 2026-06-29 sprint-orch GAP2 worker3 — 12 wrapper Family B restanti.
    // Tipi opachi (Handle/Descriptor) sono alias *mut cudnn::cudnnXxxStruct identici
    // tra ct8 e ct9 (entrambi pub use super::cudnn::*). Enum params sono i32 raw nelle
    // impl super::* (cudnnForwardMode_t, cudnnBatchNormMode_t, cudnnConv*Algo_t):
    // passthrough diretto + s9_to_s8 conversione status finale. ZERO cast enum richiesto.

    pub unsafe extern "C" fn cudnnRNNForward(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        fwdMode: ct8::cudnnForwardMode_t,
        devSeqLengths: *const i32,
        xDesc: ct8::cudnnRNNDataDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnRNNDataDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hDesc: ct8::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        hy: *mut ::core::ffi::c_void,
        cDesc: ct8::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        cy: *mut ::core::ffi::c_void,
        weightSpaceSize: usize,
        weightSpace: *const ::core::ffi::c_void,
        workSpaceSize: usize,
        workSpace: *mut ::core::ffi::c_void,
        reserveSpaceSize: usize,
        reserveSpace: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRNNForward(handle, rnnDesc, fwdMode.0 as i32, devSeqLengths, xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize, weightSpace, workSpaceSize, workSpace, reserveSpaceSize, reserveSpace))
    }

    pub unsafe extern "C" fn cudnnRNNForwardInference(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hxDesc: ct8::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cxDesc: ct8::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        wDesc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        yDesc: *const ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        hyDesc: ct8::cudnnTensorDescriptor_t,
        hy: *mut ::core::ffi::c_void,
        cyDesc: ct8::cudnnTensorDescriptor_t,
        cy: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRNNForwardInference(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, cxDesc, cx, wDesc, w, yDesc, y, hyDesc, hy, cyDesc, cy, workspace, workspaceSize))
    }

    pub unsafe extern "C" fn cudnnRNNBackwardData(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        seqLength: i32,
        yDesc: *const ct8::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        dyDesc: *const ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dhyDesc: ct8::cudnnTensorDescriptor_t,
        dhy: *const ::core::ffi::c_void,
        dcyDesc: ct8::cudnnTensorDescriptor_t,
        dcy: *const ::core::ffi::c_void,
        wDesc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        hxDesc: ct8::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        cxDesc: ct8::cudnnTensorDescriptor_t,
        cx: *const ::core::ffi::c_void,
        dxDesc: *const ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        dhxDesc: ct8::cudnnTensorDescriptor_t,
        dhx: *mut ::core::ffi::c_void,
        dcxDesc: ct8::cudnnTensorDescriptor_t,
        dcx: *mut ::core::ffi::c_void,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        reserve: *mut ::core::ffi::c_void,
        reserveSize: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRNNBackwardData(handle, rnnDesc, seqLength, yDesc, y, dyDesc, dy, dhyDesc, dhy, dcyDesc, dcy, wDesc, w, hxDesc, hx, cxDesc, cx, dxDesc, dx, dhxDesc, dhx, dcxDesc, dcx, workspace, workspaceSize, reserve, reserveSize))
    }

    pub unsafe extern "C" fn cudnnRNNBackwardWeights(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        hxDesc: ct8::cudnnTensorDescriptor_t,
        hx: *const ::core::ffi::c_void,
        yDesc: *const ct8::cudnnTensorDescriptor_t,
        y: *const ::core::ffi::c_void,
        workspace: *const ::core::ffi::c_void,
        workspaceSize: usize,
        dwDesc: ct8::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
        reserve: *const ::core::ffi::c_void,
        reserveSize: usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnRNNBackwardWeights(handle, rnnDesc, seqLength, xDesc, x, hxDesc, hx, yDesc, y, workspace as *mut _, workspaceSize, dwDesc, dw, reserve, reserveSize))
    }

    pub unsafe extern "C" fn cudnnSetRNNDescriptor_v8(
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        algo: ct8::cudnnRNNAlgo_t,
        cellMode: ct8::cudnnRNNMode_t,
        biasMode: ct8::cudnnRNNBiasMode_t,
        dirMode: ct8::cudnnDirectionMode_t,
        inputMode: ct8::cudnnRNNInputMode_t,
        dataType: ct8::cudnnDataType_t,
        mathPrec: ct8::cudnnDataType_t,
        mathType: ct8::cudnnMathType_t,
        inputSize: i32,
        hiddenSize: i32,
        projSize: i32,
        numLayers: i32,
        dropoutDesc: ct8::cudnnDropoutDescriptor_t,
        auxFlags: u32,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnSetRNNDescriptor_v8(rnnDesc, algo.0 as i32, cellMode.0 as i32, biasMode.0 as i32, dirMode.0 as i32, inputMode.0 as i32, dataType.0 as i32, mathPrec.0 as i32, mathType.0 as i32, inputSize, hiddenSize, projSize, numLayers, dropoutDesc as *mut ::core::ffi::c_void, auxFlags))
    }

    pub unsafe extern "C" fn cudnnGetRNNWorkspaceSize(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        seqLength: i32,
        xDesc: *const ct8::cudnnTensorDescriptor_t,
        sizeInBytes: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNWorkspaceSize(handle, rnnDesc, seqLength, xDesc, sizeInBytes))
    }

    pub unsafe extern "C" fn cudnnGetRNNTempSpaceSizes(
        handle: ct8::cudnnHandle_t,
        rnnDesc: ct8::cudnnRNNDescriptor_t,
        fwdMode: ct8::cudnnForwardMode_t,
        xDesc: ct8::cudnnRNNDataDescriptor_t,
        workSpaceSize: *mut usize,
        reserveSpaceSize: *mut usize,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnGetRNNTempSpaceSizes(handle, rnnDesc, fwdMode.0 as i32, xDesc, workSpaceSize, reserveSpaceSize))
    }

    pub unsafe extern "C" fn cudnnConvolutionBackwardData(
        handle: ct8::cudnnHandle_t,
        alpha: *const ::core::ffi::c_void,
        wDesc: ct8::cudnnFilterDescriptor_t,
        w: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        convDesc: ct8::cudnnConvolutionDescriptor_t,
        algo: ct8::cudnnConvolutionBwdDataAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        beta: *const ::core::ffi::c_void,
        dxDesc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo.0 as i32, workspace, workspaceSize, beta, dxDesc, dx))
    }

    pub unsafe extern "C" fn cudnnConvolutionBackwardFilter(
        handle: ct8::cudnnHandle_t,
        alpha: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        convDesc: ct8::cudnnConvolutionDescriptor_t,
        algo: ct8::cudnnConvolutionBwdFilterAlgo_t,
        workspace: *mut ::core::ffi::c_void,
        workspaceSize: usize,
        beta: *const ::core::ffi::c_void,
        dwDesc: ct8::cudnnFilterDescriptor_t,
        dw: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnConvolutionBackwardFilter(handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo.0 as i32, workspace, workspaceSize, beta, dwDesc, dw))
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardTraining(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bnSbmvDesc: ct8::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        bnBias: *const ::core::ffi::c_void,
        expAvgFactor: f64,
        runningMean: *mut ::core::ffi::c_void,
        runningVar: *mut ::core::ffi::c_void,
        epsilon: f64,
        saveMean: *mut ::core::ffi::c_void,
        saveInvVar: *mut ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBatchNormalizationForwardTraining(handle, mode.0 as i32, alpha as *mut _, beta as *mut _, xDesc, x, yDesc, y, bnSbmvDesc, bnScale as *mut _, bnBias as *mut _, expAvgFactor, runningMean, runningVar, epsilon, saveMean, saveInvVar))
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationForwardInference(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        alpha: *const ::core::ffi::c_void,
        beta: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        yDesc: ct8::cudnnTensorDescriptor_t,
        y: *mut ::core::ffi::c_void,
        bnSbmvDesc: ct8::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        bnBias: *const ::core::ffi::c_void,
        estMean: *const ::core::ffi::c_void,
        estVar: *const ::core::ffi::c_void,
        epsilon: f64,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBatchNormalizationForwardInference(handle, mode.0 as i32, alpha as *mut _, beta as *mut _, xDesc, x, yDesc, y, bnSbmvDesc, bnScale as *mut _, bnBias as *mut _, estMean as *mut _, estVar as *mut _, epsilon))
    }

    pub unsafe extern "C" fn cudnnBatchNormalizationBackward(
        handle: ct8::cudnnHandle_t,
        mode: ct8::cudnnBatchNormMode_t,
        alphaDataDiff: *const ::core::ffi::c_void,
        betaDataDiff: *const ::core::ffi::c_void,
        alphaParamDiff: *const ::core::ffi::c_void,
        betaParamDiff: *const ::core::ffi::c_void,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dxDesc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        bnSbdDesc: ct8::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        resultBnScaleDiff: *mut ::core::ffi::c_void,
        resultBnBiasDiff: *mut ::core::ffi::c_void,
        epsilon: f64,
        savedMean: *const ::core::ffi::c_void,
        savedInvVar: *const ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnBatchNormalizationBackward(handle, mode.0 as i32, alphaDataDiff, betaDataDiff, alphaParamDiff, betaParamDiff, xDesc, x, dyDesc, dy, dxDesc, dx, bnSbdDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVar))
    }

    // VEGA-PATCH 2026-06-30 sprint Rust E LITE worker 2/3 TASK 1 — LayerNorm bwd safe
    pub unsafe extern "C" fn cudnnLayerNormBackward(
        handle: ct8::cudnnHandle_t,
        xDesc: ct8::cudnnTensorDescriptor_t,
        x: *const ::core::ffi::c_void,
        dyDesc: ct8::cudnnTensorDescriptor_t,
        dy: *const ::core::ffi::c_void,
        dxDesc: ct8::cudnnTensorDescriptor_t,
        dx: *mut ::core::ffi::c_void,
        bnParamDesc: ct8::cudnnTensorDescriptor_t,
        bnScale: *const ::core::ffi::c_void,
        resultBnScaleDiff: *mut ::core::ffi::c_void,
        resultBnBiasDiff: *mut ::core::ffi::c_void,
        epsilon: f64,
        savedMean: *const ::core::ffi::c_void,
        savedInvVar: *const ::core::ffi::c_void,
    ) -> ct8::cudnnStatus_t {
        s9_to_s8(super::cudnnLayerNormBackward(handle, xDesc, x, dyDesc, dy, dxDesc, dx, bnParamDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff, epsilon, savedMean, savedInvVar))
    }
}
