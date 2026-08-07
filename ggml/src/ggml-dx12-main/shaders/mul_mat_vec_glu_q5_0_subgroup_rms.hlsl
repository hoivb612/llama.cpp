// RMS_NORM+MUL absorbed into the fused Q5_0 subgroup FFN gate/up + SwiGLU
// matvec (fl=92). See mul_mat_vec_glu_q5_0_subgroup.hlsl for the op-param map
// and the single-pass algebra.
#define RMS_FUSED 1
#include "mul_mat_vec_glu_q5_0_subgroup.hlsl"
