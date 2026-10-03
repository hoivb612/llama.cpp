// RMS_NORM+MUL absorbed into the fused FFN gate/up + SwiGLU matvec (fl=89).
// See mul_mat_vec_glu.hlsl for the op-param map and the algebra that lets a
// single pass produce both the projections and the RMS scale.
#define RMS_FUSED 1
#include "mul_mat_vec_glu.hlsl"
