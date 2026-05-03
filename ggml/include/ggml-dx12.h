#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_DX12_NAME "DX12"
#define GGML_DX12_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_dx12_init(size_t dev_num);

GGML_BACKEND_API bool ggml_backend_is_dx12(ggml_backend_t backend);
GGML_BACKEND_API int  ggml_backend_dx12_get_device_count(void);
GGML_BACKEND_API void ggml_backend_dx12_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_dx12_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_dx12_buffer_type(size_t dev_num);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_dx12_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_dx12_reg(void);

#ifdef  __cplusplus
}
#endif
