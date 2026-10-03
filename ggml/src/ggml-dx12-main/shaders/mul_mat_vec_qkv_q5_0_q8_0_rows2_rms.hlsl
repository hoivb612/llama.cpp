// RMS_NORM+MUL absorbed into the mixed Q5_0 Q/K + Q8_0 V combined projection
// variant (fl=97). src6/t6 already carries the Q8_0 Wv weights here, so g is
// read from src4/t4 instead; the host only takes this fold when the model has
// no freq_factors, which is what otherwise occupies t4.
#define RMS_FUSED 1
#define QKV_V_Q8_0
#define MMV_G_BUF src4
#include "mul_mat_vec_qkv_q5_0_vulkan_rows2.hlsl"
