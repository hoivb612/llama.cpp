// RMS_NORM+MUL absorbed into the wave64 Q8_0 gate/up + SwiGLU matvec (fl=96).
// See mul_mat_vec_glu_q8_0_wave64_rows2.hlsl for the single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_glu_q8_0_wave64_rows2.hlsl"
