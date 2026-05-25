// ggml-ryzenai: AMD Ryzen AI NPU (XDNA) backend.
//
// Backend wrapper around the qlinear_2-based MUL_MAT implementation in
// ggml-ryzenai-impl.{h,cpp}. Modeled on ggml-blas: we are an "extras"-style
// op-level accelerator that accepts CPU-allocated buffers and only claims
// the MUL_MAT shapes/types our NPU kernels can handle. Everything else
// falls back to the CPU backend through the scheduler.

#include "ggml-ryzenai.h"
#include "ggml-ryzenai-impl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstring>

struct ggml_backend_ryzenai_context {
    int n_threads = GGML_DEFAULT_N_THREADS;
};

// ----- backend interface -----

static const char * ggml_backend_ryzenai_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_RYZENAI_NAME;
}

static void ggml_backend_ryzenai_free(ggml_backend_t backend) {
    ggml_backend_ryzenai_context * ctx = (ggml_backend_ryzenai_context *) backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_ryzenai_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_ryzenai_impl_mul_mat(node->src[0], node->src[1], node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("ggml-ryzenai: unsupported op %s\n", ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ryzenai_backend_i = {
    /* .get_name                = */ ggml_backend_ryzenai_get_name,
    /* .free                    = */ ggml_backend_ryzenai_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ryzenai_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ryzenai_guid(void) {
    static ggml_guid guid = { 0x52, 0x59, 0x5a, 0x4e, 0x41, 0x49, 0x4e, 0x50,
                              0x55, 0x41, 0x4d, 0x44, 0x58, 0x44, 0x4e, 0x41 };
    return &guid;
}

ggml_backend_t ggml_backend_ryzenai_init(void) {
    ggml_ryzenai_impl_init();

    ggml_backend_ryzenai_context * ctx = new ggml_backend_ryzenai_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_ryzenai_guid(),
        /* .iface   = */ ryzenai_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_ryzenai_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_ryzenai(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_ryzenai_guid());
}

void ggml_backend_ryzenai_set_n_threads(ggml_backend_t backend_ryzenai, int n_threads) {
    GGML_ASSERT(ggml_backend_is_ryzenai(backend_ryzenai));
    auto * ctx = (ggml_backend_ryzenai_context *) backend_ryzenai->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_ryzenai_preload_weights(ggml_backend_t backend_ryzenai, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(ggml_backend_is_ryzenai(backend_ryzenai));
    if (cgraph == NULL) {
        return;
    }
    int preloaded = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_MUL_MAT) {
            continue;
        }
        const struct ggml_tensor * src0 = node->src[0];
        const struct ggml_tensor * src1 = node->src[1];
        if (!ggml_ryzenai_impl_can_mul_mat(src0, src1, node)) {
            continue;
        }
        ggml_ryzenai_impl_preload_weight(src0, src1, node);
        preloaded++;
    }
    GGML_LOG_INFO("ggml-ryzenai: preloaded %d weight tensor(s)\n", preloaded);
}

// ----- device interface -----

static const char * ggml_backend_ryzenai_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_RYZENAI_NAME;
}

static const char * ggml_backend_ryzenai_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
#ifdef RYZENAI_EMULATION
    return "AMD Ryzen AI NPU (emulated)";
#else
    return "AMD Ryzen AI NPU (XDNA)";
#endif
}

static void ggml_backend_ryzenai_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_ryzenai_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_ryzenai_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_ryzenai_device_get_name(dev);
    props->description = ggml_backend_ryzenai_device_get_description(dev);
    props->type        = ggml_backend_ryzenai_device_get_type(dev);
    ggml_backend_ryzenai_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_ryzenai_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_ryzenai_init();
}

static ggml_backend_buffer_type_t ggml_backend_ryzenai_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_ryzenai_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_ryzenai_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return ggml_ryzenai_impl_can_mul_mat(op->src[0], op->src[1], op);

        default:
            return false;
    }
}

static bool ggml_backend_ryzenai_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_ryzenai_device_i = {
    /* .get_name             = */ ggml_backend_ryzenai_device_get_name,
    /* .get_description      = */ ggml_backend_ryzenai_device_get_description,
    /* .get_memory           = */ ggml_backend_ryzenai_device_get_memory,
    /* .get_type             = */ ggml_backend_ryzenai_device_get_type,
    /* .get_props            = */ ggml_backend_ryzenai_device_get_props,
    /* .init_backend         = */ ggml_backend_ryzenai_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ryzenai_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_ryzenai_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_ryzenai_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ryzenai_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ----- backend reg interface -----

static const char * ggml_backend_ryzenai_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_RYZENAI_NAME;
}

static size_t ggml_backend_ryzenai_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_ryzenai_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(index);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_ryzenai_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &device;
}

static void * ggml_backend_ryzenai_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_ryzenai_set_n_threads;
    }
    if (std::strcmp(name, "ggml_backend_ryzenai_preload_weights") == 0) {
        return (void *) ggml_backend_ryzenai_preload_weights;
    }
    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_ryzenai_reg_i = {
    /* .get_name         = */ ggml_backend_ryzenai_reg_get_name,
    /* .get_device_count = */ ggml_backend_ryzenai_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ryzenai_reg_get_device,
    /* .get_proc_address = */ ggml_backend_ryzenai_get_proc_address,
};

ggml_backend_reg_t ggml_backend_ryzenai_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ryzenai_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ryzenai_reg)
