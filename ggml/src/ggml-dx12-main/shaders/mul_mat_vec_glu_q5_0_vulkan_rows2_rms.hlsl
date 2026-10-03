// RMS_NORM+MUL absorbed into the wave64 Q5_0 gate/up + SwiGLU matvec (fl=95).
// See mul_mat_vec_glu_q5_0_vulkan_rows2.hlsl for the single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_glu_q5_0_vulkan_rows2.hlsl"
