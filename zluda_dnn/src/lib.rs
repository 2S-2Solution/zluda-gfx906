mod r#impl;

macro_rules! unimplemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::unimplemented()?;
                Ok(())
            }
        )*
    };
}

macro_rules! implemented_no_conversion {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cudnn_normalize_fn!( crate::r#impl::$fn_name ) ( $($arg_id),*)
            }
        )*
    };
}

macro_rules! implemented_dnn8 {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cudnn_normalize_fn!( crate::r#impl::dnn8::$fn_name ) ( $($arg_id),*)
            }
        )*
    };
}

macro_rules! implemented_dnn9 {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cudnn_normalize_fn!( crate::r#impl::dnn9::$fn_name ) ( $($arg_id),*)
            }
        )*
    };
}

// VEGA-PATCH 2026-06-29 T08:00 sprint-orch T2 — wrapper Family B passthrough.
// dnn8_b/dnn9_b moduli sono raw extern "C" (NO snake_case conversion via cudnn_normalize_fn),
// chiamati direttamente. Necessario perche' le impl Family B in r#impl::* sono extern "C" raw,
// non snake_case Rust-conventional come quelle gestite da implemented! /implemented_dnn8/9.
macro_rules! implemented_dnn8_b {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::dnn8_b::$fn_name ( $( $arg_id ),* )
            }
        )*
    };
}

macro_rules! implemented_dnn9_b {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::dnn9_b::$fn_name ( $( $arg_id ),* )
            }
        )*
    };
}

macro_rules! implemented {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                cuda_macros::cudnn_normalize_fn!( crate::r#impl::$fn_name )  ($(zluda_common::FromCuda::<_, cuda_types::cudnn9::cudnnError_t>::from_cuda(&$arg_id)?),*)?;
                Ok(())
            }
        )*
    };
}

// VEGA-PATCH 2026-06-29 sprint-orch T2 cudnn-bridge consenso UNANIME 3/3 supervisori 5+3
// Macro per Family B (16 bridge MIOpen impl.rs 705-1325): chiama crate::r#impl::cudnnXxx
// DIRETTAMENTE col nome cudnn completo (no CamelCase->snake_case via cudnn_normalize_fn!),
// no FromCuda conversion, no Ok wrapping. Le impl ritornano gia' cudnnStatus_t e gestiscono
// internamente ctx_handle + miopen_to_cudnn_status. Sblocca LSTM +
// BatchNorm/Activation/Pooling/Conv-bwd (addestramento QAT-LoRA). Shim Marconi v4.0 sopra
// amdhip64.dll v5.7 (regola 4-bis ROCm 5.7 esclusivo, zero ROCm 6.x/7.x).
macro_rules! implemented_direct {
    ($($abi:literal fn $fn_name:ident( $($arg_id:ident : $arg_type:ty),* ) -> $ret_type:ty;)*) => {
        $(
            pub unsafe extern $abi fn $fn_name ( $( $arg_id : $arg_type),* ) -> $ret_type {
                crate::r#impl::$fn_name ( $($arg_id),* )
            }
        )*
    };
}

// VEGA-PATCH 2026-06-29 T08:00 sprint-orch T2 — dual_function_declarations duale
// (versioned dnn8/dnn9) + (versioned Family B dnn8_b/dnn9_b) per cabling 4 wrapper
// signature-perfect costruiti in impl.rs::dnn8_b / dnn9_b.
macro_rules! dual_function_declarations {
    ($default:ident,
     ($dnn8:ident | $dnn9:ident) <= [$($fn_versioned:ident),*],
     ($dnn8_b:ident | $dnn9_b:ident) <= [$($fn_b:ident),*],
     $($class:ident <= [$($fn_:ident),*]),*) => {
        pub mod dnn8 {
            cuda_macros::cudnn8_function_declarations!(
                $default,
                $dnn8 <= [$($fn_versioned),*],
                $dnn8_b <= [$($fn_b),*],
                $(
                    $class <= [$($fn_),*]
                ),*
            );
        }

        pub mod dnn9 {
            cuda_macros::cudnn9_function_declarations!(
                $default,
                $dnn9 <= [$($fn_versioned),*],
                $dnn9_b <= [$($fn_b),*],
                $(
                    $class <= [$($fn_),*]
                ),*
            );
        }
    };
}

dual_function_declarations! {
    unimplemented,
    (implemented_dnn8|implemented_dnn9) <= [
        cudnnGetErrorString,
        cudnnGetConvolutionForwardAlgorithm_v7
    ],
    // VEGA-PATCH 2026-06-29 T08:00 sprint-orch T2 — 4 Family B cablate (spike validato)
    // Activation Forward/Backward + Pooling Forward/Backward via MIOpen
    // (shim Marconi v4.0 sopra amdhip64.dll v5.7, regola 4-bis ROCm 5.7 esclusivo).
    // VEGA-PATCH 2026-06-29 sprint-orch GAP2 worker3 — +12 Family B (RNN/BN/Conv-bwd).
    // Verifica empirica: tipi opachi ct8/ct9 sono alias identici *mut cudnn::cudnnXxxStruct
    // (super::cudnn::*), e impl super::* prendono i32 raw per enum -> passthrough diretto
    // senza cast enum cudnn8/9. Sblocca LSTM + BN/Conv-bwd end-to-end.
    (implemented_dnn8_b|implemented_dnn9_b) <= [
        // 2 ago 2026 — senza questa PyTorch non arriva nemmeno a chiamare una
        // convoluzione: at::native::getRuntimeVersion() legge major/minor/patch
        // da cudnnGetProperty, non da cudnnGetVersion, e sullo stub vuoto
        // leggeva memoria non inizializzata.
        cudnnGetProperty,
        // Fase 6: API backend v8. Magazzino di attributi + traduttore che al
        // momento dell'esecuzione ricostruisce la convoluzione e la manda sulla v7.
        cudnnBackendCreateDescriptor,
        cudnnBackendDestroyDescriptor,
        cudnnBackendSetAttribute,
        cudnnBackendGetAttribute,
        cudnnBackendFinalize,
        cudnnBackendExecute,
        // Fase 5: spatial transformer. MIOpen non ce l'ha: kernel HIP nostri,
        // compilati a runtime con hipRTC. Vedi impl.rs.
        cudnnCreateSpatialTransformerDescriptor,
        cudnnDestroySpatialTransformerDescriptor,
        cudnnSetSpatialTransformerNdDescriptor,
        cudnnSpatialTfGridGeneratorForward,
        cudnnSpatialTfGridGeneratorBackward,
        cudnnSpatialTfSamplerForward,
        cudnnSpatialTfSamplerBackward,
        // Fase 4: RNN/LSTM/GRU. Attenzione all'ordine delle porte, vedi impl.rs.
        cudnnCreateRNNDescriptor,
        cudnnDestroyRNNDescriptor,
        cudnnCreateRNNDataDescriptor,
        cudnnDestroyRNNDataDescriptor,
        cudnnSetRNNDataDescriptor,
        cudnnSetRNNDescriptor_v6,
        cudnnSetRNNMatrixMathType,
        cudnnSetRNNProjectionLayers,
        cudnnGetRNNParamsSize,
        cudnnGetRNNTrainingReserveSize,
        cudnnGetRNNLinLayerMatrixParams,
        cudnnGetRNNLinLayerBiasParams,
        cudnnRNNForwardTraining,
        // Fase 2-3: dropout e CTC loss, entrambe con controparte MIOpen completa.
        cudnnCreateDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
        cudnnDropoutGetStatesSize,
        cudnnSetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnCreateCTCLossDescriptor,
        cudnnDestroyCTCLossDescriptor,
        cudnnSetCTCLossDescriptor,
        cudnnSetCTCLossDescriptorEx,
        cudnnGetCTCLossWorkspaceSize,
        cudnnCTCLoss,
        // Fase 1: lettura descrittori, descrittori di attivazione e pooling,
        // conv+bias+attivazione fusa. Tutte con controparte MIOpen diretta.
        cudnnGetTensorNdDescriptor,
        cudnnGetFilterNdDescriptor,
        cudnnCreateActivationDescriptor,
        cudnnDestroyActivationDescriptor,
        cudnnSetActivationDescriptor,
        cudnnCreatePoolingDescriptor,
        cudnnSetPooling2dDescriptor,
        cudnnConvolutionBiasActivationForward,
        // PyTorch lega l'handle allo stream corrente prima di ogni cosa:
        // senza queste due, nessuna chiamata cuDNN arriva a MIOpen.
        cudnnSetStream,
        cudnnGetStream,
        // Senza queste quattro il backward faceva cadere il processo: PyTorch
        // usava una dimensione di spazio di lavoro mai scritta.
        cudnnGetConvolutionBackwardDataAlgorithm_v7,
        cudnnGetConvolutionBackwardDataWorkspaceSize,
        cudnnGetConvolutionBackwardFilterAlgorithm_v7,
        cudnnGetConvolutionBackwardFilterWorkspaceSize,
        // Le ricerche Ex: servono a torch.backends.cudnn.benchmark = True,
        // che sullo stub faceva cadere il processo.
        cudnnFindConvolutionForwardAlgorithmEx,
        cudnnFindConvolutionBackwardDataAlgorithmEx,
        cudnnFindConvolutionBackwardFilterAlgorithmEx,
        // BatchNorm: PyTorch usa solo le varianti Ex, mai le tre gia' presenti.
        cudnnBatchNormalizationForwardTrainingEx,
        cudnnBatchNormalizationBackwardEx,
        cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize,
        cudnnGetBatchNormalizationBackwardExWorkspaceSize,
        cudnnGetBatchNormalizationTrainingExReserveSpaceSize,
        // 2 ago 2026 — la famiglia Nd. PyTorch non chiama mai le varianti
        // 2d/4d: descrittori e convoluzione li imposta sempre da qui. Finche'
        // erano stub, nessuna conv arrivava a MIOpen.
        cudnnSetTensorNdDescriptor,
        cudnnSetTensorNdDescriptorEx,
        cudnnSetFilterNdDescriptor,
        cudnnSetConvolutionNdDescriptor,
        cudnnSetConvolutionGroupCount,
        cudnnGetConvolutionNdForwardOutputDim,
        cudnnActivationForward,
        cudnnActivationBackward,
        cudnnPoolingForward,
        cudnnPoolingBackward,
        cudnnRNNForward,
        cudnnRNNForwardInference,
        cudnnRNNBackwardData,
        cudnnRNNBackwardWeights,
        cudnnSetRNNDescriptor_v8,
        cudnnGetRNNWorkspaceSize,
        cudnnGetRNNTempSpaceSizes,
        cudnnConvolutionBackwardData,
        cudnnConvolutionBackwardFilter,
        cudnnBatchNormalizationForwardTraining,
        cudnnBatchNormalizationForwardInference,
        cudnnBatchNormalizationBackward,
        // VEGA-PATCH 2026-06-30 sprint Rust E LITE worker 2/3 TASK 1 — LayerNorm bwd safe
        // (contiguous guard via miopenTransformTensor, mappa LN -> BN PER_ACTIVATION)
        cudnnLayerNormBackward
    ],
    implemented_no_conversion <= [
        cudnnGetVersion,
        cudnnGetMaxDeviceVersion,
        cudnnGetCudartVersion,
        cudnnGetLastErrorString
    ],
    implemented <= [
        cudnnConvolutionForward,
        cudnnCreate,
        cudnnCreateConvolutionDescriptor,
        cudnnCreateFilterDescriptor,
        cudnnCreateTensorDescriptor,
        cudnnDestroy,
        cudnnDestroyConvolutionDescriptor,
        cudnnDestroyFilterDescriptor,
        cudnnDestroyTensorDescriptor,
        cudnnGetConvolutionForwardWorkspaceSize,
        cudnnSetConvolution2dDescriptor,
        cudnnSetConvolutionMathType,
        cudnnSetFilter4dDescriptor,
        cudnnSetTensor4dDescriptor
    ]
    // VEGA-NOTE 2026-06-29 T07:20 sprint-orch T2 cabling ROLLBACK:
    // Le 16 impl Family B in impl.rs righe 705-1325 sono SCRITTE SOLO per cudnn9 types
    // (cudnn9::cudnnError_t + enum tipati cudnnRNNAlgo_t/cudnnBatchNormMode_t ecc).
    // Macro implemented_direct + versioned (unimplemented|implemented_direct) FALLISCONO build
    // perche' dual_function_declarations espande sia in dnn8 (i32 raw) sia in dnn9 (enum tipati).
    // 3/3 supervisori UNANIMI 8.67/10 confidence avevano sottostimato divergenza profonda.
    // Per chiusura vera serve scrivere 16 impl PARALLELE per cudnn8 + macro implemented_direct
    // duale dnn8/dnn9 separato. Sprint dedicato 2-3h. Per ora cudnn rimane senza 16 Family B
    // — LSTM rimane bloccato MA forward matmul OK via cublas.
}
