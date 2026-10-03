// RMS_NORM+MUL absorbed into the portable 256-thread combined Q8_0 Q/K/V
// projection variant (fl=90). See mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl for
// the op-param map and the single-pass algebra.
#define RMS_FUSED 1
#define GROUP_SIZE 256
#include "mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl"
