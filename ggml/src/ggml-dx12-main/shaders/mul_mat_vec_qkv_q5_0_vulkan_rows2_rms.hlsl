// RMS_NORM+MUL absorbed into the combined Q5_0 Q/K/V projection variant
// (fl=94). g rides src6/t6, which this pure-Q5_0 path leaves free. See
// mul_mat_vec_qkv_q5_0_vulkan_rows2.hlsl for the op-param map and the
// single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_qkv_q5_0_vulkan_rows2.hlsl"
