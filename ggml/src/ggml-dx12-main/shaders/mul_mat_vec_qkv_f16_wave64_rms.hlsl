// RMS_NORM+MUL absorbed into the combined Q/K/V projection matvec (fl=88).
// See mul_mat_vec_qkv_f16_wave64.hlsl for the op-param map and the algebra
// that lets a single pass produce both the projection and the RMS scale.
#define RMS_FUSED 1
#include "mul_mat_vec_qkv_f16_wave64.hlsl"
