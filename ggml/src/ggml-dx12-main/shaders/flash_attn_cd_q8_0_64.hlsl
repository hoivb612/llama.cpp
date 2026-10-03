// flash_attn_cd_q8_0_64.hlsl - cooperative decode FA, Q8_0 K/V cache, head_dim 64
#define HEAD_DIM 64
#define CD_KV_Q8_0
#include "flash_attn_cd.hlsli"
