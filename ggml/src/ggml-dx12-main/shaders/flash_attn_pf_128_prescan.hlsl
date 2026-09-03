// Intel D=128 prefill variant of flash_attn_pf_128.hlsl. Same FA_PF_BR/BC, so
// the two blobs are interchangeable at the same pf_var.br; this one prescans
// the mask once per query group to bound the KV range and classify each tile.
//
// FA_PF_BR must match fa_tile_br in ggml-dx12.cpp; a mismatch under-dispatches
// query groups and silently drops rows.
#define HEAD_DIM 128
#define FA_PF_BR 16
#define FA_PF_PRESCAN 1
#define FA_PF_RELAXED_ACC 1
#define FA_PF_MASK_CLASS 1
#include "flash_attn_pf.hlsli"
