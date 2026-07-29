// flash_attn_cd_q8_0_96.hlsl - cooperative decode FA, Q8_0 K/V cache, head_dim 96 (Phi-3)
#define HEAD_DIM 96
#define CD_KV_Q8_0
#include "flash_attn_cd.hlsli"
