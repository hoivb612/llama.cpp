// RMS_NORM+MUL absorbed into the Q8_0 gate/up + SwiGLU matvec (fl=98).
// See mul_mat_vec_glu_q8_0.hlsl for the single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_glu_q8_0.hlsl"
