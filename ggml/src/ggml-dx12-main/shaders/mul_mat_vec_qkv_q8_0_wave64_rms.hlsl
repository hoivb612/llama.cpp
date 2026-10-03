// RMS_NORM+MUL absorbed into the wave64 combined Q8_0 Q/K/V projection variant
// (fl=93). Same base as fl=90 but without the 256-thread group override. See
// mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl for the op-param map and the
// single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl"
