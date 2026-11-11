use cuda_types::cuda::{CUerror, CUgraphExecUpdateResult, CUresult, CUresultConsts};
use hip_runtime_sys::*;
use zluda_common::FromCuda;

pub(crate) unsafe fn destroy(graph: hipGraph_t) -> hipError_t {
    hipGraphDestroy(graph)
}

pub(crate) unsafe fn exec_destroy(graph_exec: hipGraphExec_t) -> hipError_t {
    hipGraphExecDestroy(graph_exec)
}

pub(crate) fn exec_update_v2(
    h_graph_exec: hipGraphExec_t,
    h_graph: hipGraph_t,
    result_info: &mut cuda_types::cuda::CUgraphExecUpdateResultInfo,
) -> CUresult {
    let mut h_error_node: hipGraphNode_t = unsafe { std::mem::zeroed() };
    let mut update_result: hipGraphExecUpdateResult = hipGraphExecUpdateResult(0);
    unsafe { hipGraphExecUpdate(h_graph_exec, h_graph, &mut h_error_node, &mut update_result) }?;

    // We use FromCuda here instead of transmute in case our hipGraphNode_t representation changes
    // in the future.
    let error_node: *mut hipGraphNode_t =
        FromCuda::<_, CUerror>::from_cuda(&std::ptr::from_mut(&mut result_info.errorNode))?;
    let error_from_node: *mut hipGraphNode_t =
        FromCuda::<_, CUerror>::from_cuda(&std::ptr::from_mut(&mut result_info.errorFromNode))?;
    unsafe { *error_node = h_error_node };
    unsafe { *error_from_node = h_error_node };

    result_info.errorFromNode = result_info.errorNode;
    result_info.result = match update_result {
        hipGraphExecUpdateResult::hipGraphExecUpdateSuccess => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_SUCCESS
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateError => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorTopologyChanged => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorNodeTypeChanged => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorFunctionChanged => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorParametersChanged => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorNotSupported => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED
        }
        hipGraphExecUpdateResult::hipGraphExecUpdateErrorUnsupportedFunctionChange => {
            CUgraphExecUpdateResult::CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE
        }
        _ => return CUresult::ERROR_NOT_SUPPORTED,
    };

    Ok(())
}

pub(crate) unsafe fn get_nodes(
    graph: hipGraph_t,
    nodes: *mut hipGraphNode_t,
    num_nodes: *mut usize,
) -> hipError_t {
    hipGraphGetNodes(graph, nodes, num_nodes)
}

pub(crate) unsafe fn instantiate_with_flags(
    graph_exec: *mut hipGraphExec_t,
    graph: hipGraph_t,
    flags: u64,
) -> hipError_t {
    hipGraphInstantiateWithFlags(graph_exec, graph, flags)
}

pub(crate) unsafe fn launch(graph_exec: hipGraphExec_t, stream: hipStream_t) -> hipError_t {
    hipGraphLaunch(graph_exec, stream)
}

// VEGA-PATCH 2026-06-28 sprint-orch v2 nvcuda-3: forwarder cuGraphAddKernelNode -> hipGraphAddKernelNode
// Conversione field-by-field di CUDA_KERNEL_NODE_PARAMS_st verso hipKernelNodeParams.
// Le due struct hanno layout di campi differente (ordine + dim3 contro 3 uint scalari),
// quindi serve copia manuale invece di un transmute.
pub(crate) unsafe fn add_kernel_node(
    ph_graph_node: *mut hipGraphNode_t,
    h_graph: hipGraph_t,
    dependencies: *const hipGraphNode_t,
    num_dependencies: usize,
    node_params: *const cuda_types::cuda::CUDA_KERNEL_NODE_PARAMS_st,
) -> hipError_t {
    // VEGA-FIX 2026-06-28 T21:00 post-sprint-orch v2: hipError_t e' Result<(), hipErrorCode_t>. Variant si chiama InvalidValue (no prefix Error)
    if node_params.is_null() {
        return Err(hipErrorCode_t::InvalidValue);
    }
    let cu_params = &*node_params;
    let hip_params = hipKernelNodeParams {
        blockDim: dim3 {
            x: cu_params.blockDimX,
            y: cu_params.blockDimY,
            z: cu_params.blockDimZ,
        },
        extra: cu_params.extra,
        // VEGA-FIX 2026-06-28 T21:00: CUfunction e' tuple-struct newtype, serve .0 accessor
        func: cu_params.func.0 as *mut ::core::ffi::c_void,
        gridDim: dim3 {
            x: cu_params.gridDimX,
            y: cu_params.gridDimY,
            z: cu_params.gridDimZ,
        },
        kernelParams: cu_params.kernelParams,
        sharedMemBytes: cu_params.sharedMemBytes,
    };
    hipGraphAddKernelNode(
        ph_graph_node,
        h_graph,
        dependencies,
        num_dependencies,
        &hip_params,
    )
}
