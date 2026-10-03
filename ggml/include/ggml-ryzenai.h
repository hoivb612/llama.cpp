#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_RYZENAI_NAME "RyzenAI"

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_ryzenai_init(void);

GGML_BACKEND_API bool ggml_backend_is_ryzenai(ggml_backend_t backend);

// number of threads used for software fallback / pre/post-processing
GGML_BACKEND_API void ggml_backend_ryzenai_set_n_threads(ggml_backend_t backend_ryzenai, int n_threads);

// Eagerly pre-construct and upload all NPU-eligible weight tensors referenced
// by `cgraph`. After this call, the next graph_compute will not pay the
// per-tensor unpack / transpose / DMA cost on first matmul.
//
// Safe to call multiple times; tensors already cached are skipped.
GGML_BACKEND_API void ggml_backend_ryzenai_preload_weights(ggml_backend_t backend_ryzenai, struct ggml_cgraph * cgraph);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_ryzenai_reg(void);

#ifdef __cplusplus
}
#endif
