#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// One-time per-process initialization of the RyzenAI runtime. Eagerly
// triggers the xclbin / AIE program load by constructing a sentinel
// qlinear_2 instance. Aborts on failure when the SDK is built in but the
// NPU is not available (fail-fast policy).
void ggml_ryzenai_impl_init(void);

// Pre-construct + upload the per-weight qlinear_2 instance for one src0
// tensor so the next mul_mat on it skips the unpack/transpose/DMA path.
// No-op if the tensor is already cached or does not pass can_mul_mat
// against the supplied template src1 / dst.
void ggml_ryzenai_impl_preload_weight(const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1,
                                      const struct ggml_tensor * dst);

// Gate: returns true if the given MUL_MAT op shape/types are eligible for
// NPU offload. Matches the historical b612_llama.dc heuristic:
// Q4_0 weights, F32 activations, contiguous src1, ne0 >= 4096, ne2=ne3=1.
bool ggml_ryzenai_impl_can_mul_mat(const struct ggml_tensor * src0,
                                   const struct ggml_tensor * src1,
                                   const struct ggml_tensor * dst);

// Execute MUL_MAT on the NPU (or software emulation when RYZENAI_EMULATION
// is defined). Caller must have verified ggml_ryzenai_impl_can_mul_mat().
void ggml_ryzenai_impl_mul_mat(const struct ggml_tensor * src0,
                               const struct ggml_tensor * src1,
                               struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
