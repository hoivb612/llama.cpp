// flash_attn_cd_gqa_q8_0_128.hlsl - GQA-shared cooperative decode FA, Q8_0 K/V cache, head_dim 128
#define HEAD_DIM 128
#define CD_KV_Q8_0
#include "flash_attn_cd_gqa.hlsli"
