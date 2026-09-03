// Intel UHD D=64 prefill variant. The mask is prescanned once per query group
// to bound the KV range and classify each tile. Fully masked tiles skip K/V
// staging, while finite all-zero tiles skip per-score mask loads. QK and PV
// accumulators may reassociate and contract.
//
// FA_PF_BR must match fa_tile_br in ggml-dx12.cpp; a mismatch under-dispatches
// query groups and silently drops rows.
#define HEAD_DIM 64
#define FA_PF_BR 32
#define FA_PF_BC 32
#define FA_PF_PRESCAN 1
#define FA_PF_RELAXED_ACC 1
#define FA_PF_MASK_CLASS 1
#include "flash_attn_pf.hlsli"
