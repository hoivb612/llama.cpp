// RMS_NORM+MUL absorbed into the fused Q4_K FFN gate/up + SwiGLU matvec
// (fl=91). See mul_mat_vec_glu_q4_k.hlsl for the op-param map and the
// single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_glu_q4_k.hlsl"
