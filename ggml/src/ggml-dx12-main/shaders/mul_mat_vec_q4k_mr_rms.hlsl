// Q4_K multi-row matvec (M=1) with the preceding RMS_NORM+MUL folded in.
// See mul_mat_vec_q4k_mr.hlsl for the op-param map and the algebra.

#define RMS_FUSED 1
#include "mul_mat_vec_q4k_mr.hlsl"
