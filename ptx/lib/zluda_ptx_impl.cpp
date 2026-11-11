/*
Every time this file changes it must be rebuilt.
You must use LLVM from ZLUDA submodule dir ext/llvm-project:

cd ext/llvm-project && \
mkdir -p build && \
cd build && \
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" \
    -GNinja \
    ../llvm && \
ninja clang llvm-dis llvm-as

then cd to the directory with this file and run this simple command:

../../ext/llvm-project/build/bin/clang \
    -std=c++20 \
    -Xclang -fdenormal-fp-math=dynamic \
    -Wall -Wextra -Wsign-compare -Wconversion \
    -x hip \
    zluda_ptx_impl.cpp \
    -nogpulib \
    -O3 \
    -o zluda_ptx_impl.bc \
    -emit-llvm \
    -c \
    --offload-device-only --offload-arch=gfx906 && \
../../ext/llvm-project/build/bin/llvm-dis zluda_ptx_impl.bc -o - \
    | sed '/@llvm.used/d' \
    | sed '/wchar_size/d' \
    | sed '/llvm.module.flags/d' \
    | sed '/__hip_cuid/d' \
    | sed 's/optnone//g' \
    | sed 's/define hidden/define linkonce_odr/g' \
    | sed 's/\"target-cpu\"=\"gfx906\"//g' \
    | sed -E 's/\"target-features\"=\"[^\"]+\"//g'| \
../../ext/llvm-project/build/bin/llvm-as - -o  zluda_ptx_impl.bc && \
../../ext/llvm-project/build/bin/llvm-dis zluda_ptx_impl.bc

RISOLTO FINO ALLA FINE (Vega, 2026-07-02) — target gfx906 NON gfx1030, e NIENTE
-mno-wavefrontsize64. Motivo: la reduction wave64 (sum/mean dim=None di PyTorch)
dava garbage. Root cause: compilando per gfx1030 wave32, clang a -O3 folda via
__builtin_amdgcn_mbcnt_hi dentro __lane_id() (su wave32 le lane 32-63 non
esistono), lasciando solo mbcnt.lo. Sul codegen finale gfx906 wave64 quel
__lane_id() ritorna 32 per TUTTE le hw-lane 32-63 -> il confinamento 32-subsection
di SHFL_SYNC_IMPL collassa (self = self_abs - twarp_base = 0 per le lane alte) e le
warp dispari non riducono. Compilando per gfx906 wave64 mbcnt.hi sopravvive,
__lane_id() ritorna 0-63 veri, il confinamento funziona. Il sed strippa comunque
target-cpu/features -> IR target-neutral, ZLUDA rimette gfx906 al link (le funzioni
elementwise senza cross-lane sono insensibili alla wave size). Verificato empirico:
kernel HIP che chiama __zluda_ptx_impl_shfl_sync_down_b32 reale del .bc, reduction
64-lane -> lane0=528 lane32=1552 CORRETTI (asm-only pipeline, no hang).
*/

#include <cstddef>
#include <cstdint>
#include <bit>
#include <cmath>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>

#define SHARED_SPACE __attribute__((address_space(3)))
#define CONSTANT_SPACE __attribute__((address_space(4)))

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16;
typedef _Float16 f16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef __bf16 bf16;
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef char s8x4 __attribute__((ext_vector_type(4)));

#define FUNC(NAME) __device__ __attribute__((retain)) __zluda_ptx_impl_##NAME
#define FUNC_CALL(NAME) __zluda_ptx_impl_##NAME
#define ATTR(NAME) __ZLUDA_PTX_IMPL_ATTRIBUTE_##NAME
#define DECLARE_ATTR(TYPE, NAME)                                        \
    extern "C" __attribute__((constant)) CONSTANT_SPACE TYPE ATTR(NAME) \
    __device__

extern "C"
{
    extern "C" __attribute__((constant)) CONSTANT_SPACE uint32_t __oclc_ISA_version __device__;

    // [VEGA-FIX WAVE64-WARP 31 lug 2026] activemask confinato al torch-warp da 32 lane.
    // Era `read_exec_lo()`, cioe' SEMPRE i 32 bit bassi della wavefront: per il
    // torch-warp alto [32-63] restituiva la maschera del warp basso — quello
    // sbagliato. Le altre primitive (laneid, lanemask_*, ballot, vote) erano gia'
    // state confinate il 4 lug; questa era rimasta indietro.
    // Conseguenza misurata: il decoupled look-back di cub (ScanTileState) sceglieva
    // tile predecessori sbagliati, e `unique` restituiva un conteggio negativo
    // (-319 su 64 elementi). Colpiti unique / randperm / nonzero / masked_select /
    // topk / scatter_add / cumsum su n grande.
    // Stesso calcolo di vote_sync_all_pred: si prende la propria meta' di exec.
    uint32_t FUNC(activemask)()
    {
        uint32_t base = (uint32_t)__lane_id() & 0x20U;
        return (uint32_t)(__builtin_amdgcn_read_exec() >> base);
    }

    size_t __ockl_get_local_id(uint32_t) __device__;
    uint32_t FUNC(sreg_tid)(uint8_t member)
    {
        return (uint32_t)__ockl_get_local_id(member);
    }

    // [VEGA-FIX 2 lug 2026 (Fable 5)] ROOT CAUSE reduction rotta: __ockl_get_local_size /
    // __ockl_get_num_groups ritornano 0 su ZLUDA gfx906 (l'implicit-kernarg che gli serve
    // non e' impostato dal launch). => blockDim=gridDim=0 => ogni loop di block_reduce gira
    // 0 volte => sopravvive solo il thread 0 (coverage=ceil(n/nthreads)). Bisect PTX confermato:
    // %ntid.x=0, %nctaid.x=0, mentre %tid.x/%ctaid.x (register hw) OK.
    // FIX: leggo blockDim/gridDim DIRETTAMENTE dal dispatch packet AQL via dispatch_ptr,
    // self-contained (no __ockl). Layout hsa_kernel_dispatch_packet_t:
    //   off 4/6/8  = uint16 workgroup_size_x/y/z ; off 12/16/20 = uint32 grid_size_x/y/z (TOT thread).
    //   gridDim(CUDA) = ceil(grid_size / workgroup_size).
    uint32_t FUNC(sreg_ntid)(uint8_t member)
    {
        const CONSTANT_SPACE uint8_t *dp = (const CONSTANT_SPACE uint8_t *)__builtin_amdgcn_dispatch_ptr();
        const CONSTANT_SPACE uint16_t *wgs = (const CONSTANT_SPACE uint16_t *)(dp + 4);
        return (uint32_t)wgs[member & 3u];
    }

    size_t __ockl_get_group_id(uint32_t) __device__;
    uint32_t FUNC(sreg_ctaid)(uint8_t member)
    {
        return (uint32_t)__ockl_get_group_id(member);
    }

    uint32_t FUNC(sreg_nctaid)(uint8_t member)
    {
        const CONSTANT_SPACE uint8_t *dp = (const CONSTANT_SPACE uint8_t *)__builtin_amdgcn_dispatch_ptr();
        const CONSTANT_SPACE uint16_t *wgs = (const CONSTANT_SPACE uint16_t *)(dp + 4);
        const CONSTANT_SPACE uint32_t *gs = (const CONSTANT_SPACE uint32_t *)(dp + 12);
        uint32_t w = (uint32_t)wgs[member & 3u];
        uint32_t g = gs[member & 3u];
        return w ? (g + w - 1u) / w : 0u;
    }

    // [VEGA-FIX WAVE64-WARP 4 lug 2026] Confinamento subsection-32 coerente con SHFL_SYNC_IMPL.
    // CUDA compila con warpSize=32: %laneid e %lanemask_* sono definiti su un warp da 32 lane.
    // Su gfx906 wave64 la wavefront fisica ha 64 lane = DUE torch-warp logici [0-31] e [32-63].
    // __lane_id() ritorna 0-63 (assoluto); la lane LOGICA nel torch-warp e' (laneid & 0x1f).
    // Tutte le primitive di warp (laneid, lanemask_*, ballot, vote) DEVONO vivere in questo
    // spazio 0-31 — come gia' fa shfl (self_abs & 0x20) — altrimenti il look-back cooperativo
    // di cub (radix sort Onesweep: ballot + lanemask + shfl insieme) sceglie tile predecessori
    // sbagliati -> sort/argsort/topk n>=2000 silenziosamente errati.
    // Bug precedenti chiusi da qui: (a) laneid 0-63 usato come indice a 32-lane; (b) lanemask
    // con shift u32 per lane>=32 = UB (era il crash/wedge di topk %lanemask_le); (c) %lanemask_le/
    // gt/eq non implementati -> modulo Onesweep UnknownSymbol -> kernel non gira -> buffer garbage.
    // Calcolo in 64-bit poi cast u32 per evitare ogni shift >= 32.
    uint32_t FUNC(sreg_laneid)()
    {
        return __lane_id() & 0x1fU;
    }

    uint32_t FUNC(sreg_lanemask_lt)()
    {
        uint32_t lane = __lane_id() & 0x1fU;
        return (uint32_t)((((uint64_t)1U) << lane) - 1U);
    }

    uint32_t FUNC(sreg_lanemask_le)()
    {
        uint32_t lane = __lane_id() & 0x1fU;
        return (uint32_t)((((uint64_t)1U) << (lane + 1U)) - 1U);
    }

    uint32_t FUNC(sreg_lanemask_ge)()
    {
        uint32_t lane = __lane_id() & 0x1fU;
        return (uint32_t)(~((((uint64_t)1U) << lane) - 1U));
    }

    uint32_t FUNC(sreg_lanemask_gt)()
    {
        uint32_t lane = __lane_id() & 0x1fU;
        return (uint32_t)(~((((uint64_t)1U) << (lane + 1U)) - 1U));
    }

    uint32_t FUNC(sreg_lanemask_eq)()
    {
        uint32_t lane = __lane_id() & 0x1fU;
        return (uint32_t)(((uint64_t)1U) << lane);
    }

    uint32_t __ockl_bfe_u32(uint32_t, uint32_t, uint32_t) __device__;
    uint32_t FUNC(bfe_u32)(uint32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (pos >= 32)
            return 0;
        // V_BFE_U32 only uses bits [4:0] for len (max value is 31)
        if (len >= 32)
            return base >> pos;
        len = std::min(len, 31U);
        return __ockl_bfe_u32(base, pos, len);
    }

    // LLVM contains mentions of llvm.amdgcn.ubfe.i64 and llvm.amdgcn.sbfe.i64,
    // but using it only leads to LLVM crashes on RDNA2
    uint64_t FUNC(bfe_u64)(uint64_t base, uint32_t pos, uint32_t len)
    {
        // NVIDIA docs are incorrect. In 64 bit `bfe` both `pos` and `len`
        // parameters use whole 32 bit number and not just bottom 8 bits
        if (pos >= 64)
            return 0;
        if (len >= 64)
            return base >> pos;
        len = std::min(len, 63U);
        return (base >> pos) & ((1UL << len) - 1UL);
    }

    int32_t __ockl_bfe_i32(int32_t, uint32_t, uint32_t) __device__;
    int32_t FUNC(bfe_s32)(int32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (len == 0)
            return 0;
        if (pos >= 32)
            return (base >> 31);
        // V_BFE_I32 only uses bits [4:0] for len (max value is 31)
        if (len >= 32)
            return base >> pos;
        len = std::min(len, 31U);
        return __ockl_bfe_i32(base, pos, len);
    }

    static __device__ uint32_t add_sat(uint32_t x, uint32_t y)
    {
        uint32_t result;
        if (__builtin_add_overflow(x, y, &result))
        {
            return UINT32_MAX;
        }
        else
        {
            return result;
        }
    }

    int64_t FUNC(bfe_s64)(int64_t base, uint32_t pos, uint32_t len)
    {
        // NVIDIA docs are incorrect. In 64 bit `bfe` both `pos` and `len`
        // parameters use whole 32 bit number and not just bottom 8 bits
        if (len == 0)
            return 0;
        if (pos >= 64)
            return (base >> 63U);
        if (add_sat(pos, len) >= 64)
            len = 64 - pos;
        return (base << (64U - pos - len)) >> (64U - len);
    }

    uint32_t __ockl_bfm_u32(uint32_t count, uint32_t offset) __device__;
    uint32_t FUNC(bfi_b32)(uint32_t insert, uint32_t base, uint32_t pos_32, uint32_t len_32)
    {
        uint32_t pos = pos_32 & 0xFFU;
        uint32_t len = len_32 & 0xFFU;
        if (pos >= 32)
            return base;
        uint32_t mask;
        if (len >= 32)
            mask = UINT32_MAX << pos;
        else
            mask = __ockl_bfm_u32(len, pos);
        return (~mask & base) | (mask & (insert << pos));
    }

    uint64_t FUNC(bfi_b64)(uint64_t insert, uint64_t base, uint32_t pos, uint32_t len)
    {
        // NVIDIA docs are incorrect. In 64 bit `bfe` both `pos` and `len`
        // parameters use whole 32 bit number and not just bottom 8 bits
        if (pos >= 64)
            return base;
        uint64_t mask;
        if (len >= 64)
            mask = UINT64_MAX << pos;
        else
            mask = ((1UL << len) - 1UL) << (pos);
        return (~mask & base) | (mask & (insert << pos));
    }

    void FUNC(bar_sync)(uint32_t)
    {
        __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
        __builtin_amdgcn_s_barrier();
    }

    int32_t __ockl_wgred_and_i32(int32_t) __device__;
    int32_t __ockl_wgred_or_i32(int32_t) __device__;

#define BAR_RED_IMPL(reducer)                                                                                            \
    bool FUNC(bar_red_##reducer##_pred)(uint32_t barrier __attribute__((unused)), bool predicate, bool invert_predicate) \
    {                                                                                                                    \
        /* TODO: handle barrier */                                                                                       \
        return __ockl_wgred_##reducer##_i32(predicate ^ invert_predicate);                                               \
    }

    BAR_RED_IMPL(and);
    BAR_RED_IMPL(or);

    typedef uint32_t ShflSyncResult __attribute__((ext_vector_type(2)));

    // shfl.sync opts consists of two values, the warp end ID and the subsection mask.
    //
    // [VEGA-FIX F.1 22 giu 2026 T15:00] Wave64 patch — bit-width mask expanded from 5-bit (0b11111=31)
    // to 6-bit (0b111111=63) per supportare lane id 0-63 di wave64 gfx906.
    // Root cause BUG F.1: PyTorch ATen reduction last-axis tramite __shfl_*_sync produceva off-by-half
    // (14000%+ rel diff) perchè il subsection era clippato a 5-bit (max lane 31) ignorando lanes 32-63.
    // CUDA __shfl_sync è documentato per width 0-32, ma su HIP wave64 il subsection deve estendersi a 0-64.
    // Vedi sediment ZLUDA_BUG_TRACKER cat F.1 + REGOLE_TASSATIVE.md regola 17.
    //
    // The current warp is partitioned into some number of subsections with a width of w. The
    // subsection mask is 32 - w (or 64 - w on wave64), and indicates which bits of the lane id are
    // part of the subsection address. For example, if each subsection is 8 lanes wide on wave64,
    // the subsection mask will be 56 – 111000 in binary. This indicates that the three most
    // significant bits in the 6-bit lane ID are the subsection address.
    //
    // The warp end ID is the max lane ID for a specific mode. For the CUDA __shfl_sync
    // intrinsics, it is always 31 (wave32) or 63 (wave64) for idx, bfly, and down, and 0 for up.
    // This is used for the bounds check.
// [VEGA-FIX WAVE64-REDUCE 2 lug 2026 (Fable 5)] Root cause bug reduction dim=None.
// torch wheel e' CUDA-compiled con warpSize=32: le shfl.sync codificano opts a
// 5-bit (warp_end=31 per down/bfly/idx, 0 per up; section_mask 0-31). Su gfx906
// wave64 la wavefront fisica ha 64 lane. Col vecchio codice le lane 32-63 avevano
// subsection_end=31 -> sempre out-of-bounds -> non partecipavano allo shuffle ->
// la loro META' di dati era persa. torch legge lane 0 (torch-warp 0) e lane 32
// (torch-warp 1) via shared mem, ma la lane 32 non aveva ridotto nulla -> sum/mean
// dimezzate/garbage (GPU 10.1 vs CPU 13.5 su vettore da 5).
// FIX: se opts e' wave32-encoded (section_mask<=31 && warp_end<=31), confino lo
// shuffle al torch-warp di appartenenza [0-31] o [32-63] (base = self_abs & 0x20),
// cosi' ogni torch-warp riduce indipendente e torch ricombina i 2 leader via LDS.
// La patch F.1 5->6 bit era un no-op (torch manda comunque 5-bit). Vero fix = questo.
#define SHFL_SYNC_IMPL(mode, calculate_index, CMP)                                                                                              \
    ShflSyncResult FUNC(shfl_sync_##mode##_b32_pred)(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask __attribute__((unused))) \
    {                                                                                                                                           \
        int32_t section_mask = (opts >> 8) & 0b111111;                                                                                          \
        int32_t warp_end = opts & 0b111111;                                                                                                     \
        int32_t self_abs = (int32_t)__lane_id();                                                                                                \
        int32_t twarp_base = ((section_mask <= 0x1f) && (warp_end <= 0x1f)) ? (self_abs & 0x20) : 0;                                            \
        int32_t self = self_abs - twarp_base;                                                                                                   \
        int32_t subsection = section_mask & self;                                                                                               \
        int32_t subsection_end = subsection | (~section_mask & warp_end);                                                                       \
        int32_t idx = calculate_index;                                                                                                          \
        bool out_of_bounds = idx CMP subsection_end;                                                                                            \
        if (out_of_bounds)                                                                                                                      \
        {                                                                                                                                       \
            idx = self;                                                                                                                         \
        }                                                                                                                                       \
        int32_t abs_idx = twarp_base + (idx & 0x1f);                                                                                            \
        int32_t output = __builtin_amdgcn_ds_bpermute(abs_idx << 2, (int32_t)input);                                                            \
        return {(uint32_t)output, uint32_t(!out_of_bounds)};                                                                                    \
    }                                                                                                                                           \
                                                                                                                                                \
    uint32_t FUNC(shfl_sync_##mode##_b32)(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask)                                    \
    {                                                                                                                                           \
        return __zluda_ptx_impl_shfl_sync_##mode##_b32_pred(input, delta, opts, membermask).x;                                                  \
    }

    // We are using the HIP __shfl intrinsics to implement these, rather than the __shfl_sync
    // intrinsics, as those only add an assertion checking that the membermask is used correctly.
    // They do not return the result of the range check, so we must replicate that logic here.

    SHFL_SYNC_IMPL(up, self - delta, <);
    SHFL_SYNC_IMPL(down, self + delta, >);
    SHFL_SYNC_IMPL(bfly, self ^ delta, >);
    SHFL_SYNC_IMPL(idx, (delta & ~section_mask) | subsection, >);

    // ============================================================================
    // VEGA TASK D 30 giu 2026 T07:55 — LDS scratchpad shuffle (opt-in)
    // ============================================================================
    //
    // Implementazione alternativa che NON usa __builtin_amdgcn_ds_bpermute.
    // Pattern: ogni thread del workgroup scrive il proprio input in LDS shared
    // memory all'offset corrispondente al proprio lane id, esegue __syncthreads
    // (s_barrier + s_waitcnt lgkmcnt(0)), poi rilegge dall'offset target.
    //
    // Costo: ~3-5x latenza rispetto a ds_bpermute (2 LDS op + 1 barriera vs
    // 1 LDS permute hw). Vantaggio: la barriera __syncthreads e' un safepoint
    // che il driver Windows WDDM tratta come progresso evitando TDR durante
    // sequenze butterfly log2(64) consecutive in backward LN2.
    //
    // IMPORTANTE: questo helper assume blockDim.x % 64 == 0 (multipli di una
    // wavefront wave64). Il caller PyTorch ATen reduction respecta sempre questa
    // condizione. Usiamo HIP_DYNAMIC_SHARED per non allocare LDS staticamente
    // (i kernel chiamanti gestiscono gia' lo shared memory pool).
    //
    // Riservazione LDS: 64 * sizeof(uint32_t) = 256 byte per wavefront.
    // Indicizziamo nello scratchpad con offset wave_id * 64 + lane_id.
    //
    // [VEGA TASK D worker 2/5 fix 30 giu T08:10] Switch da `extern __shared__`
    // (dynamic shared, richiede 3rd arg in <<<grid,block,smem>>> che PyTorch
    // ATen NON imposta — alias silenzioso con dynamic smem altrui) a
    // statically-sized `__attribute__((shared))`. 4 KB = 16 wave × 64 lane ×
    // 4 byte, trascurabile su gfx906 64 KB LDS/CU. Garantisce isolation totale
    // dal pool dynamic shared.
    //
    // Vedi pass rewrite_shfl_to_lds.rs + feature wave64_lds_scratchpad in Cargo.toml.

    // Max 16 wave per block (1024 thread max ÷ 64 wave64 = 16 wave) × 64 slot
    // = 1024 uint32 = 4 KB LDS statically reserved per block.
    static constexpr int SHFL_LDS_MAX_WAVES_PER_BLOCK = 16;
    static constexpr int SHFL_LDS_WAVE_LANES = 64;
    static constexpr int SHFL_LDS_TOTAL_SLOTS =
        SHFL_LDS_MAX_WAVES_PER_BLOCK * SHFL_LDS_WAVE_LANES;

    __attribute__((shared)) uint32_t
        __zluda_shfl_lds_scratch[SHFL_LDS_TOTAL_SLOTS];

    static __device__ __forceinline__ uint32_t shfl_lds_core(uint32_t input, int32_t target_lane)
    {
        const uint32_t lane = __lane_id();                          // 0..63
        // wave_id locale al block: flat thread id (3D) >> 6 (=÷64).
        // workitem_id_x da solo non basta per kernel 2D/3D PyTorch.
        const uint32_t flat_tid =
            (uint32_t)__builtin_amdgcn_workitem_id_x()
            + (uint32_t)__builtin_amdgcn_workitem_id_y()
                * (uint32_t)__builtin_amdgcn_workgroup_size_x()
            + (uint32_t)__builtin_amdgcn_workitem_id_z()
                * (uint32_t)__builtin_amdgcn_workgroup_size_x()
                * (uint32_t)__builtin_amdgcn_workgroup_size_y();
        const uint32_t wave_id = flat_tid >> 6;
        uint32_t* slot = __zluda_shfl_lds_scratch + (wave_id << 6); // 64 slot per wave
        slot[lane] = input;
        __builtin_amdgcn_s_waitcnt(0);                              // attendi LDS write
        __builtin_amdgcn_s_barrier();                               // sync workgroup
        uint32_t result = slot[target_lane & 0x3f];                 // mask 6-bit wave64
        __builtin_amdgcn_s_waitcnt(0);                              // attendi LDS read
        __builtin_amdgcn_s_barrier();                               // chiudi finestra lettura
        return result;
    }

#define SHFL_LDS_IMPL(mode, calculate_index, CMP)                                                                                              \
    ShflSyncResult FUNC(shfl_lds_##mode##_b32_pred)(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask __attribute__((unused))) \
    {                                                                                                                                          \
        int32_t section_mask = (opts >> 8) & 0b111111;                                                                                         \
        int32_t warp_end = opts & 0b111111;                                                                                                    \
        int32_t self = (int32_t)__lane_id();                                                                                                   \
        int32_t subsection = section_mask & self;                                                                                              \
        int32_t subsection_end = subsection | (~section_mask & warp_end);                                                                      \
        int32_t idx = calculate_index;                                                                                                         \
        bool out_of_bounds = idx CMP subsection_end;                                                                                           \
        if (out_of_bounds)                                                                                                                     \
        {                                                                                                                                      \
            idx = self;                                                                                                                        \
        }                                                                                                                                      \
        uint32_t output = shfl_lds_core(input, idx);                                                                                           \
        return {output, uint32_t(!out_of_bounds)};                                                                                             \
    }                                                                                                                                          \
                                                                                                                                               \
    uint32_t FUNC(shfl_lds_##mode##_b32)(uint32_t input, int32_t delta, uint32_t opts, uint32_t membermask)                                    \
    {                                                                                                                                          \
        return __zluda_ptx_impl_shfl_lds_##mode##_b32_pred(input, delta, opts, membermask).x;                                                  \
    }

    SHFL_LDS_IMPL(up, self - delta, <);
    SHFL_LDS_IMPL(down, self + delta, >);
    SHFL_LDS_IMPL(bfly, self ^ delta, >);
    SHFL_LDS_IMPL(idx, (delta & ~section_mask) | subsection, >);
    // ============================================================================
    // END VEGA TASK D LDS scratchpad block
    // ============================================================================

    DECLARE_ATTR(uint32_t, CLOCK_RATE);
    void FUNC(nanosleep_u32)(uint32_t nanoseconds)
    {
        // clock_rate is in kHz
        uint64_t cycles_per_ns = ATTR(CLOCK_RATE) / 1000000;
        uint64_t cycles = nanoseconds * cycles_per_ns;
        // Avoid small sleep values resulting in s_sleep 0
        cycles += 63;
        // s_sleep N sleeps for 64 * N cycles
        uint64_t sleep_amount = cycles / 64;

        // The argument to s_sleep must be a constant
        for (size_t i = 0; i < sleep_amount >> 4; i++)
            __builtin_amdgcn_s_sleep(16);
        if (sleep_amount & 8U)
            __builtin_amdgcn_s_sleep(8);
        if (sleep_amount & 4U)
            __builtin_amdgcn_s_sleep(4);
        if (sleep_amount & 2U)
            __builtin_amdgcn_s_sleep(2);
        if (sleep_amount & 1U)
            __builtin_amdgcn_s_sleep(1);
    }

    // VEGA-FIX T14:35: __assert_fail non disponibile in HIP gfx906 device libs.
    // Stub abort() per preservare ABI senza dipendenza glibc-style assert.
    extern "C" __device__ void abort(void);
    void FUNC(__assertfail)(uint64_t message __attribute__((unused)),
                            uint64_t file __attribute__((unused)),
                            uint32_t line __attribute__((unused)),
                            uint64_t function __attribute__((unused)),
                            uint64_t char_size __attribute__((unused)))
    {
        abort();
    }

    // * Smallest denormal is 1.4 × 10^-45
    // * Smallest normal is ~1.175494351 × 10^(-38)
    // * Now, 1.175494351×10^-38 / 1.4 × 10^-45  = 8396388 + 31/140
    // * Next power of 2 is 16777216
    const float DENORMAL_TO_NORMAL_FACTOR_F32 = 16777216.0f;
    // * Largest subnormal is ~1.175494210692441e × 10^(-38)
    // * Then any value equal or larger than following will produce subnormals: 8.50706018714406320806444272332455743547934627837873057975602739772164... × 10^37
    const float RCP_DENORMAL_OUTPUT = 8.50706018714406320806444272332455743547934627837873057975602739772164e37f;
    const float REVERSE_DENORMAL_TO_NORMAL_FACTOR_F32 = 0.029387360490963111877208252592662410455594571842846914442095471744599661631813495980086003637902577995683214210345151992265999035207077609582844f;

    float FUNC(sqrt_approx_f32)(float x)
    {
        bool is_subnormal = __builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL);
        float input = x;
        if (is_subnormal)
            input = x * DENORMAL_TO_NORMAL_FACTOR_F32;
        float value = __builtin_amdgcn_sqrtf(input);
        if (is_subnormal)
            return value * 0.000244140625f;
        else
            return value;
    }

    float FUNC(rsqrt_approx_f32)(float x)
    {
        bool is_subnormal = __builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL);
        float input = x;
        if (is_subnormal)
            input = x * DENORMAL_TO_NORMAL_FACTOR_F32;
        float value = __builtin_amdgcn_rsqf(input);
        if (is_subnormal)
            return value * 4096.0f;
        else
            return value;
    }

    float FUNC(rcp_approx_f32)(float x)
    {
        float factor = 1.0f;
        if (__builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL))
        {
            factor = DENORMAL_TO_NORMAL_FACTOR_F32;
        }
        if (std::fabs(x) >= RCP_DENORMAL_OUTPUT)
        {
            factor = REVERSE_DENORMAL_TO_NORMAL_FACTOR_F32;
        }
        return __builtin_amdgcn_rcpf(x * factor) * factor;
    }

    // When x = -126, exp2(x) = 2^(-126) ≈ 1.175494351 × 10^(-38),
    // which is the smallest normalized number in FP32
    float FUNC(ex2_approx_f32)(float x)
    {
        bool special_handling = x < -126.0f;
        float input = x;
        if (special_handling)
            input *= 0.5f;
        float result = __builtin_amdgcn_exp2f(input);
        if (special_handling)
            return result * result;
        else
            return result;
    }

    float FUNC(lg2_approx_f32)(float x)
    {
        bool is_subnormal = __builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL);
        float input = x;
        if (is_subnormal)
            input = x * DENORMAL_TO_NORMAL_FACTOR_F32;
        float value = __builtin_amdgcn_logf(input);
        if (is_subnormal)
            return value - 24.0f;
        else
            return value;
    }

    // Logic taken from legalizeFSQRTF32/lowerFSQRTF32 in LLVM AMDGPU target
    __device__ static float precise_square_root(float x, bool needs_denorm_handling)
    {

        // Constants for denormal handling
        const float scale_threshold = 0x1.0p-96f;   // Very small value threshold
        const float scale_up_factor = 0x1.0p+32f;   // 2^32
        const float scale_down_factor = 0x1.0p-16f; // 2^-16

        // Check if input needs scaling (for very small values)
        bool need_scale = scale_threshold > x;
        auto scaled = scale_up_factor * x;

        // Scale up input if needed
        float sqrt_x = need_scale ? scaled : x;

        float sqrt_s;

        // Check if we need special denormal handling

        if (needs_denorm_handling)
        {
            // Use hardware sqrt as initial approximation
            sqrt_s = __builtin_sqrtf(sqrt_x); // Or equivalent hardware instruction

            // Bit manipulations to get next values up/down
            uint32_t sqrt_s_bits = std::bit_cast<uint32_t>(sqrt_s);

            // Next value down (subtract 1 from bit pattern)
            uint32_t sqrt_s_next_down_bits = sqrt_s_bits - 1;
            float sqrt_s_next_down = std::bit_cast<float>(sqrt_s_next_down_bits);

            // Calculate residual: x - sqrt_next_down * sqrt
            float neg_sqrt_s_next_down = -sqrt_s_next_down;
            float sqrt_vp = std::fma(neg_sqrt_s_next_down, sqrt_s, sqrt_x);

            // Next value up (add 1 to bit pattern)
            uint32_t sqrt_s_next_up_bits = sqrt_s_bits + 1;
            float sqrt_s_next_up = std::bit_cast<float>(sqrt_s_next_up_bits);

            // Calculate residual: x - sqrt_next_up * sqrt
            float neg_sqrt_s_next_up = -sqrt_s_next_up;
            float sqrt_vs = std::fma(neg_sqrt_s_next_up, sqrt_s, sqrt_x);

            // Select correctly rounded result
            if (sqrt_vp <= 0.0f)
            {
                sqrt_s = sqrt_s_next_down;
            }

            if (sqrt_vs > 0.0f)
            {
                sqrt_s = sqrt_s_next_up;
            }
        }
        else
        {
            // Use Newton-Raphson method with reciprocal square root

            // Initial approximation
            float sqrt_r = __builtin_amdgcn_rsqf(sqrt_x); // Or equivalent hardware 1/sqrt instruction
            sqrt_s = sqrt_x * sqrt_r;

            // Refine approximation
            float half = 0.5f;
            float sqrt_h = sqrt_r * half;
            float neg_sqrt_h = -sqrt_h;

            // Calculate error term
            float sqrt_e = std::fma(neg_sqrt_h, sqrt_s, half);

            // First refinement
            sqrt_h = std::fma(sqrt_h, sqrt_e, sqrt_h);
            sqrt_s = std::fma(sqrt_s, sqrt_e, sqrt_s);

            // Second refinement
            float neg_sqrt_s = -sqrt_s;
            float sqrt_d = std::fma(neg_sqrt_s, sqrt_s, sqrt_x);
            sqrt_s = std::fma(sqrt_d, sqrt_h, sqrt_s);
        }

        // Scale back if input was scaled
        if (need_scale)
        {
            sqrt_s *= scale_down_factor;
        }

        // Special case handling for zero and infinity
        bool is_zero_or_inf = __builtin_isfpclass(sqrt_x, __FPCLASS_POSINF | __FPCLASS_POSZERO | __FPCLASS_NEGZERO);

        return is_zero_or_inf ? sqrt_x : sqrt_s;
    }

    float FUNC(sqrt_rn_f32)(float x)
    {
        return precise_square_root(x, true);
    }

    float FUNC(sqrt_rn_ftz_f32)(float x)
    {
        return precise_square_root(x, false);
    }

    struct DivRnFtzF32Part1Result
    {
        float fma_4;
        float fma_1;
        float fma_3;
        uint8_t numerator_scaled_flag;
    };

    DivRnFtzF32Part1Result FUNC(div_f32_part1)(float lhs, float rhs)
    {
        float one = 1.0f;

        // Division scale operations
        bool denominator_scaled_flag;
        float denominator_scaled = __builtin_amdgcn_div_scalef(lhs, rhs, false, &denominator_scaled_flag);

        bool numerator_scaled_flag;
        float numerator_scaled = __builtin_amdgcn_div_scalef(lhs, rhs, true, &numerator_scaled_flag);

        // Reciprocal approximation
        float approx_rcp = __builtin_amdgcn_rcpf(denominator_scaled);
        float neg_div_scale0 = -denominator_scaled;

        // Perform division approximation steps
        float fma_0 = fmaf(neg_div_scale0, approx_rcp, one);
        float fma_1 = fmaf(fma_0, approx_rcp, approx_rcp);
        float mul = numerator_scaled * fma_1;
        float fma_2 = fmaf(neg_div_scale0, mul, numerator_scaled);
        float fma_3 = fmaf(fma_2, fma_1, mul);
        float fma_4 = fmaf(neg_div_scale0, fma_3, numerator_scaled);
        return {fma_4, fma_1, fma_3, numerator_scaled_flag};
    }

    __device__ static float div_f32_part2(float x, float y, DivRnFtzF32Part1Result part1)
    {
        float fmas = __builtin_amdgcn_div_fmasf(part1.fma_4, part1.fma_1, part1.fma_3, part1.numerator_scaled_flag);
        float result = __builtin_amdgcn_div_fixupf(fmas, y, x);

        return result;
    }

    float FUNC(div_f32_part2)(float x, float y,
                              float fma_4,
                              float fma_1,
                              float fma_3,
                              uint8_t numerator_scaled_flag)
    {
        return div_f32_part2(x, y, {fma_4, fma_1, fma_3, numerator_scaled_flag});
    }

    // VEGA-FIX T14:30 23 giu: fp8 NON supportato gfx906 hardware. Stub functions ritornano 0
    // (preserva ABI per dispatcher PTX, ma kernel torch normali NON chiamano fp8 su Vega 20).
    // Headers ROCm 6.2 non hanno __hip_fp8x2_e4m3 (solo _fnuz). Skippiamo definizioni reali.
    struct Fp8x2
    {
        uint8_t b;
        uint8_t a;
    };

    Fp8x2 FUNC(cvt_rn_satfinite_e4m3x2_f32)(float a __attribute__((unused)), float b __attribute__((unused)))
    {
        return {0, 0};
    }

    Fp8x2 FUNC(cvt_rn_satfinite_e5m2x2_f32)(float a __attribute__((unused)), float b __attribute__((unused)))
    {
        return {0, 0};
    }

    // return type uint32_t (raw 2x f16 packed bits) per evitare dipendenza __half2 da hip_fp8.h
    uint32_t FUNC(cvt_rn_f16x2_e4m3x2)(uint16_t in __attribute__((unused)))
    {
        return 0;
    }

    uint32_t FUNC(cvt_rn_f16x2_e5m2x2)(uint16_t in __attribute__((unused)))
    {
        return 0;
    }

    // VEGA-FIX T14:30 23 giu: ballot per gfx906 wave64 (era _w32 i32 RDNA, fail clang Cannot select i32 SETCC)
    __device__ static inline uint64_t ballot(bool value, bool negate)
    {
        __builtin_amdgcn_wave_barrier();
        return __builtin_amdgcn_ballot_w64(negate ? !value : value);
    }

    // [VEGA-FIX WAVE64-WARP 4 lug 2026] ballot confinato al torch-warp da 32 lane.
    // ballot_w64 ballotta tutte le 64 lane della wavefront e IGNORA membermask; il vecchio
    // (uint32_t)ballot teneva sempre le hw-lane 0-31 -> per il torch-warp alto [32-63] restituiva
    // i bit del torch-warp basso -> __clz/__ffs del look-back cub sceglievano il tile sbagliato.
    // Estraggo i 32 bit della MIA subsection: base = laneid & 0x20 (0 o 32), poi >> base.
    __device__ static inline uint32_t ballot32(bool value, bool negate)
    {
        uint64_t full = ballot(value, negate);
        uint32_t base = (uint32_t)__lane_id() & 0x20U;
        return (uint32_t)(full >> base);
    }

    bool FUNC(vote_sync_any_pred)(bool value, uint32_t membermask __attribute__((unused)))
    {
        return ballot32(value, false) != 0U;
    }

    bool FUNC(vote_sync_any_pred_negate)(bool value, uint32_t membermask __attribute__((unused)))
    {
        return ballot32(value, true) != 0U;
    }

    // IMPORTANT: exec mask must be a subset of membermask, the behavior is undefined otherwise
    // VEGA-FIX T14:30: __builtin_amdgcn_read_exec() full i64 (era _lo i32 wave32)
    // WAVE64-WARP 4 lug: confronto confinato alla subsection (exec dei MIEI 32 lane).
    bool FUNC(vote_sync_all_pred)(bool value, uint32_t membermask __attribute__((unused)))
    {
        uint32_t base = (uint32_t)__lane_id() & 0x20U;
        uint32_t sub_exec = (uint32_t)(__builtin_amdgcn_read_exec() >> base);
        return ballot32(value, false) == sub_exec;
    }

    // also known as "none"
    bool FUNC(vote_sync_all_pred_negate)(bool value, uint32_t membermask __attribute__((unused)))
    {
        return ballot32(value, false) == 0U;
    }

    // PTX ballot.b32 ritorna 32-bit = la MIA subsection di 32 lane (WAVE64-WARP 4 lug).
    uint32_t FUNC(vote_sync_ballot_b32)(bool value, uint32_t membermask __attribute__((unused)))
    {
        return ballot32(value, false);
    }

    uint32_t FUNC(vote_sync_ballot_b32_negate)(bool value, uint32_t membermask __attribute__((unused)))
    {
        return ballot32(value, true);
    }

#define REDUX_SYNC_TYPE_IMPL(reducer, ptx_type, amd_type, cpp_type)                                             \
    cpp_type __ockl_wfred_##reducer##_##amd_type(cpp_type) __device__;                                          \
    cpp_type FUNC(redux_sync_##reducer##_##ptx_type)(cpp_type src, uint32_t membermask __attribute__((unused))) \
    {                                                                                                           \
        return __ockl_wfred_##reducer##_##amd_type(src);                                                        \
    }

#define REDUX_SYNC_IMPL(reducer)                      \
    REDUX_SYNC_TYPE_IMPL(reducer, u32, u32, uint32_t) \
    REDUX_SYNC_TYPE_IMPL(reducer, s32, i32, int32_t)

    REDUX_SYNC_IMPL(add);
    REDUX_SYNC_IMPL(min);
    REDUX_SYNC_IMPL(max);

    __device__ inline static uint32_t load_single_matrix(void SHARED_SPACE *lds_address, uint32_t warp_offset)
    {
        uint32_t laneid = __zluda_ptx_impl_sreg_laneid();
        int32_t row_address = __builtin_amdgcn_ds_bpermute((int32_t)(warp_offset + (laneid / 4U)) << 2U, (int32_t)lds_address);
        uint32_t matrix_cell_address = (uint32_t)row_address + ((laneid % 4) * 4);
        return *((uint32_t SHARED_SPACE *)matrix_cell_address);
    }

    __device__ inline static uint32_t load_single_matrix_trans(void SHARED_SPACE *lds_address, uint32_t warp_offset)
    {
        uint32_t laneid = __zluda_ptx_impl_sreg_laneid();
        int32_t row_address_lo = __builtin_amdgcn_ds_bpermute((int32_t)(warp_offset + ((laneid % 4U) * 2)) << 2U, (int32_t)lds_address);
        uint32_t address_lo = (uint32_t)row_address_lo + ((laneid / 4) * 2);
        uint16_t lo = *((uint16_t SHARED_SPACE *)address_lo);
        int32_t row_address_hi = __builtin_amdgcn_ds_bpermute((int32_t)(warp_offset + ((laneid % 4U) * 2) + 1) << 2U, (int32_t)lds_address);
        uint32_t address_hi = (uint32_t)row_address_hi + ((laneid / 4) * 2);
        uint16_t hi = *((uint16_t SHARED_SPACE *)address_hi);
        return std::bit_cast<uint32_t>(ushort2::Native_vec_{lo, hi});
    }

    uint2::Native_vec_ FUNC(ldmatrix_m8n8_x2_b16)(void SHARED_SPACE *address)
    {
        uint32_t x0 = load_single_matrix(address, 0);
        uint32_t x1 = load_single_matrix(address, 8);
        return uint2::Native_vec_{x0, x1};
    }

    uint4::Native_vec_ FUNC(ldmatrix_m8n8_x4_b16)(void SHARED_SPACE *address)
    {
        uint32_t x0 = load_single_matrix(address, 0);
        uint32_t x1 = load_single_matrix(address, 8);
        uint32_t x2 = load_single_matrix(address, 16);
        uint32_t x3 = load_single_matrix(address, 24);
        return uint4::Native_vec_{x0, x1, x2, x3};
    }

    uint4::Native_vec_ FUNC(ldmatrix_m8n8_x4_trans_b16)(void SHARED_SPACE *address)
    {
        uint32_t x0 = load_single_matrix_trans(address, 0);
        uint32_t x1 = load_single_matrix_trans(address, 8);
        uint32_t x2 = load_single_matrix_trans(address, 16);
        uint32_t x3 = load_single_matrix_trans(address, 24);
        return uint4::Native_vec_{x0, x1, x2, x3};
    }

    struct byte4
    {
        union
        {
            uint32_t u32;
            uint8_t u8x4[4];
        };
    } __attribute__((aligned(4)));

    struct byte8
    {
        union
        {
            uint32_t u32x2[2];
            uint8_t u8x8[8];
        };
    } __attribute__((aligned(8)));

    uint32_t FUNC(prmt_b32)(uint32_t x, uint32_t y, uint32_t s)
    {
        byte4 v_perm_selector;
        v_perm_selector.u32 = 0;

        byte8 input;
        input.u32x2[0] = x;
        input.u32x2[1] = y;

        for (size_t i = 0; i < 4; i++)
        {
            uint8_t sel = static_cast<uint8_t>(s >> (i * 4));
            uint8_t addr = sel & 0x7;
            if (sel & 0x8)
            {
                if (addr % 2 == 1)
                {
                    v_perm_selector.u8x4[i] = 0x8 + addr / 2;
                    continue;
                }
            }
            v_perm_selector.u8x4[i] = addr;
        }

        byte4 output;
        output.u32 = __builtin_amdgcn_perm(input.u32x2[1], input.u32x2[0], v_perm_selector.u32);

        for (size_t i = 0; i < 4; i++)
        {
            uint8_t sel = static_cast<uint8_t>(s >> (i * 4));
            uint8_t addr = sel & 0x7;
            if (sel & 0x8)
            {
                if (addr % 2 != 1)
                {
                    output.u8x4[i] = (output.u8x4[i] & 0x80) * 0xff;
                    continue;
                }
            }
        }

        return output.u32;
    }

    int FUNC(vprintf)(const char *format __attribute__((unused)), void *vlist __attribute__((unused)))
    {
        // TODO: replace calls to vprintf with a raising pass to printf when we have a mechanism
        // to write SSA passes
        // Use https://github.com/ROCm/llvm-project/blob/99a81d16b9d811cadd420190bed16981a0a57bc6/llvm/lib/Transforms/Utils/AMDGPUEmitPrintf.cpp#L426
        return -1;
    }
}

template <typename Acc, typename T>
__device__ static Acc dot_product(Acc initial_value, T row[8], T column[8]);

template <>
__device__ int32_t dot_product<int32_t, s8x4>(int32_t initial_value, s8x4 row[8], s8x4 column[8])
{
    int32_t result = initial_value;
    for (int i = 0; i < 8; i++)
    {
        // ockl bug
        if (__oclc_ISA_version == 10103)
        {
            result += int32_t(row[i].x) * int32_t(column[i].x);
            result += int32_t(row[i].y) * int32_t(column[i].y);
            result += int32_t(row[i].z) * int32_t(column[i].z);
            result += int32_t(row[i].w) * int32_t(column[i].w);
        }
        else
        {
            result = __ockl_sdot4(row[i], column[i], result, false);
        }
    }
    return result;
}

template <>
__device__ float dot_product<float, bf16x2>(float initial_value, bf16x2 row[8], bf16x2 column[8])
{
    float result = initial_value;
    for (int i = 0; i < 8; i++)
    {
        result = std::fma(float(row[i].x), float(column[i].x), result);
        result = std::fma(float(row[i].y), float(column[i].y), result);
    }
    return result;
}

template <>
__device__ float dot_product<float, f16x2>(float initial_value, f16x2 row[8], f16x2 column[8])
{
    float result = initial_value;
    for (int i = 0; i < 8; i++)
    {
        // ockl bug
        if (__oclc_ISA_version == 10103)
        {
            result = std::fma(float(row[i].x), float(column[i].x), result);
            result = std::fma(float(row[i].y), float(column[i].y), result);
        }
        else
        {
            result = __ockl_fdot2(row[i], column[i], result, false);
        }
    }
    return result;
}

// Template function because DPP mask must be a compile time constant
template <typename T, const int DPP_MASK>
__device__ static void mma_load_rowcol(T upper_row[8], T lower_row[8], T left_column[8], T right_column[8],
                                       int index, int left_column_start,
                                       uint8_t quad_index,
                                       uint32_t a0a1, uint32_t a2a3,
                                       uint32_t b0b1)
{
    uint8_t laneid = uint8_t(__lane_id());
    uint8_t quad_source = (laneid + quad_index) % 4;
    upper_row[index] = std::bit_cast<T>(__builtin_amdgcn_mov_dpp(std::bit_cast<int32_t>(a0a1), DPP_MASK, 0xf, 0xf, 1));
    lower_row[index] = std::bit_cast<T>(__builtin_amdgcn_mov_dpp(std::bit_cast<int32_t>(a2a3), DPP_MASK, 0xf, 0xf, 1));
    left_column[index] = std::bit_cast<T>(__builtin_amdgcn_ds_bpermute((left_column_start + quad_source) << 2, std::bit_cast<int32_t>(b0b1)));
    right_column[index] = std::bit_cast<T>(__builtin_amdgcn_ds_bpermute((left_column_start + 4 + quad_source) << 2, std::bit_cast<int32_t>(b0b1)));
}

template <typename T>
__device__ static void mma_load_col(T upper_row[16], T lower_row[16], T left_column[16], T right_column[16],
                                    int index, int left_column_start,
                                    uint32_t a0a1, uint32_t a2a3,
                                    uint32_t b0b1)
{
    uint8_t laneid = uint8_t(__lane_id());
    uint8_t quad_source = laneid % 4;
    upper_row[index] = std::bit_cast<T>(a0a1);
    lower_row[index] = std::bit_cast<T>(a2a3);
    left_column[index] = std::bit_cast<T>(__builtin_amdgcn_ds_bpermute((left_column_start + quad_source) << 2, std::bit_cast<int32_t>(b0b1)));
    right_column[index] = std::bit_cast<T>(__builtin_amdgcn_ds_bpermute((left_column_start + 4 + quad_source) << 2, std::bit_cast<int32_t>(b0b1)));
}

template <typename Acc, typename T>
__device__ HIP_vector_base<Acc, 4> fallback_mma_sync_aligned(uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, HIP_vector_base<Acc, 4> c_reg)
{
    uint8_t laneid = uint8_t(FUNC_CALL(sreg_laneid)());
    uint8_t quad_index = laneid % 4;
    const Acc c0 = c_reg.x;
    const Acc c1 = c_reg.y;
    const Acc c2 = c_reg.z;
    const Acc c3 = c_reg.w;
    uint8_t left_column_start = quad_index * 8;
    T upper_row[8];
    T lower_row[8];
    T left_column[8];
    T right_column[8];
    mma_load_col<T>(upper_row, lower_row, left_column, right_column,
                    0, left_column_start,
                    a_reg[0], a_reg[1],
                    b_reg[0]);
    mma_load_rowcol<T, 0b00'11'10'01>(upper_row, lower_row, left_column, right_column,
                                      1, left_column_start,
                                      1,
                                      a_reg[0], a_reg[1],
                                      b_reg[0]);
    mma_load_rowcol<T, 0b01'00'11'10>(upper_row, lower_row, left_column, right_column,
                                      2, left_column_start,
                                      2,
                                      a_reg[0], a_reg[1],
                                      b_reg[0]);
    mma_load_rowcol<T, 0b10'01'00'11>(upper_row, lower_row, left_column, right_column,
                                      3, left_column_start,
                                      3,
                                      a_reg[0], a_reg[1],
                                      b_reg[0]);
    mma_load_col<T>(upper_row, lower_row, left_column, right_column,
                    4, left_column_start,
                    a_reg[2], a_reg[3],
                    b_reg[1]);
    mma_load_rowcol<T, 0b00'11'10'01>(upper_row, lower_row, left_column, right_column,
                                      5, left_column_start,
                                      1,
                                      a_reg[2], a_reg[3],
                                      b_reg[1]);
    mma_load_rowcol<T, 0b01'00'11'10>(upper_row, lower_row, left_column, right_column,
                                      6, left_column_start,
                                      2,
                                      a_reg[2], a_reg[3],
                                      b_reg[1]);
    mma_load_rowcol<T, 0b10'01'00'11>(upper_row, lower_row, left_column, right_column,
                                      7, left_column_start,
                                      3,
                                      a_reg[2], a_reg[3],
                                      b_reg[1]);
    Acc d0 = dot_product<Acc, T>(c0, upper_row, left_column);
    Acc d1 = dot_product<Acc, T>(c1, upper_row, right_column);
    Acc d2 = dot_product<Acc, T>(c2, lower_row, left_column);
    Acc d3 = dot_product<Acc, T>(c3, lower_row, right_column);
    return {d0, d1, d2, d3};
}

extern "C"
{
    float4::Native_vec_ FUNC(mma_sync_aligned_m16n8k16_row_col_f32_f16_f16_f32)(uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, float4::Native_vec_ c_reg)
    {
        return fallback_mma_sync_aligned<float, f16x2>(a_reg, b_reg, HIP_vector_base<float, 4>(c_reg.x, c_reg.y, c_reg.z, c_reg.w)).data;
    }

    // We wrap the intrinsic in an optnone function to prevent ZLUDA-specific
    // passes from optimizing away the intrinsic call
    static __device__ float4::Native_vec_ __llvm_zluda_mma_m16n8k16_f32_bf16_bf16_f32_optnone [[clang::optnone]] (uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, float4::Native_vec_ c_reg)
    {
        __device__ uint4::Native_vec_  __llvm_zluda_mma_m16n8k16_f32_bf16_bf16_f32(uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, uint4::Native_vec_ c_reg)  __asm("llvm.zluda.mma.m16n8k16.f32.bf16.bf16.f32");
        return std::bit_cast<float4::Native_vec_>(__llvm_zluda_mma_m16n8k16_f32_bf16_bf16_f32(a_reg, b_reg, std::bit_cast<uint4::Native_vec_>(c_reg)));
    }

    float4::Native_vec_ FUNC(mma_sync_aligned_m16n8k16_row_col_f32_bf16_bf16_f32) (uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, float4::Native_vec_ c_reg)
    {
        if (__oclc_ISA_version >= 11000 && __oclc_ISA_version < 12000)
        {
            [[clang::always_inline]] return __llvm_zluda_mma_m16n8k16_f32_bf16_bf16_f32_optnone(a_reg, b_reg, c_reg);
        }
        else 
        {
            return fallback_mma_sync_aligned<float, bf16x2>(a_reg, b_reg, HIP_vector_base<float, 4>(c_reg.x, c_reg.y, c_reg.z, c_reg.w)).data;
        }
    }

    // We wrap the intrinsic in an optnone function to prevent ZLUDA-specific
    // passes from optimizing away the intrinsic call
    static __device__ uint4::Native_vec_ __llvm_zluda_mma_m16n8k32_s32_s8_s8_fs32_optnone [[clang::optnone]] (uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, uint4::Native_vec_ c_reg)
    {
        __device__ uint4::Native_vec_  __llvm_zluda_mma_m16n8k32_s32_s8_s8_fs32(uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, uint4::Native_vec_ c_reg)  __asm("llvm.zluda.mma.m16n8k32.s32.s8.s8.s32");
        return __llvm_zluda_mma_m16n8k32_s32_s8_s8_fs32(a_reg, b_reg, c_reg);
    }

    uint4::Native_vec_ FUNC(mma_sync_aligned_m16n8k32_row_col_s32_s8_s8_s32)(uint4::Native_vec_ a_reg, uint2::Native_vec_ b_reg, uint4::Native_vec_ c_reg)
    {
        if (__oclc_ISA_version >= 11000 && __oclc_ISA_version < 12000)
        {
            [[clang::always_inline]] return __llvm_zluda_mma_m16n8k32_s32_s8_s8_fs32_optnone(a_reg, b_reg, c_reg);
        }
        return std::bit_cast<uint4::Native_vec_>(fallback_mma_sync_aligned<int32_t, s8x4>(a_reg, b_reg, std::bit_cast<HIP_vector_base<int32_t, 4>>(c_reg)));
    }
}
