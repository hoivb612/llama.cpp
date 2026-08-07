// Merged Q+K projection matvec for Q4_K weights (M=1 decode), fl=103.
//
// Wq and Wk are contiguous in one buffer and Qcur/Kcur live in the same
// destination buffer at disjoint offsets, so the two projections collapse into
// a single dispatch over q_rows + k_rows output rows. Per-row results are
// bit-identical to the two separate mul_mat_vec_q4k_mr dispatches.
//
//   op2  q_rows                op3  K destination base byte offset
//   dst_offset  Q destination base byte offset

#define QK_SPLIT 1
#include "mul_mat_vec_q4k_mr.hlsl"
