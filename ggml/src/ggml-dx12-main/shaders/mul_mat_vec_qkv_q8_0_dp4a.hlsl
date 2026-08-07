// Combined Q/K/V projection using dp4a over a pre-quantized Q8_1 activation.
// Same region/post-op structure as the wave64 rows2 variant; only the inner
// accumulation differs. 32 threads matches the Q8_0 dp4a matvec group size
// that fl=17 already uses on Intel Xe-HPG+.
#define QKV_DP4A 1
#define GROUP_SIZE 32
#include "mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl"
