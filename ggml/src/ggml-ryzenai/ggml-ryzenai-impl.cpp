// RyzenAI MUL_MAT implementation.
//
// Lifted with minimal modification from b612_llama.dc/ggml-ryzenai.cpp
// (https://github.com/hoivb612/llama.dc), which itself was derived from
// AMD's transformers/ext/llama.cpp drop. The qlinear_2-based path
// targets AMD XDNA NPUs (Phoenix / Strix Point / Strix Halo) for Q4_0
// weight x F32 activation matmul.
//
// When built with -DRYZENAI_EMULATION (set automatically by CMake when the
// AMD Ryzen AI SDK is not found), all NPU calls are stubbed out and matmul
// is computed in software for development on non-NPU hosts.

#include "ggml-ryzenai-impl.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef RYZENAI_EMULATION
#include <ryzenai/ops/qlinear_2/qlinear_2.hpp>
#include <ryzenai/utils/dtype_utils.h>
#endif

#define RYZ_TRY_CATCH(expression)                                              \
    try {                                                                      \
        expression                                                             \
    } catch (const std::exception & e) {                                       \
        std::cerr << "ggml-ryzenai exception: " << e.what() << std::endl;      \
        throw;                                                                 \
    } catch (...) {                                                            \
        std::cerr << "ggml-ryzenai unknown exception" << std::endl;            \
        throw;                                                                 \
    }

namespace {

// Transpose the two innermost dimensions of a [ne00, ne01, ne02, ne03] tensor.
template <typename T>
std::vector<T> transpose_inner(const std::vector<T> & tensor,
                               const std::tuple<int, int, int, int> & shapes) {
    const auto & [ne00, ne01, ne02, ne03] = shapes;

    std::vector<T> out(static_cast<size_t>(ne00) * ne01 * ne02 * ne03);

    const int64_t s3 = static_cast<int64_t>(ne00) * ne01 * ne02;
    const int64_t s2 = static_cast<int64_t>(ne00) * ne01;

    for (int64_t b3 = 0; b3 < ne03; ++b3) {
        for (int64_t b2 = 0; b2 < ne02; ++b2) {
            for (int64_t i = 0; i < ne01; ++i) {
                for (int64_t j = 0; j < ne00; ++j) {
                    out[b3 * s3 + b2 * s2 + j * ne01 + i] =
                        tensor[b3 * s3 + b2 * s2 + i * ne00 + j];
                }
            }
        }
    }
    return out;
}

// Q4_0 row unpack: two int4 weights per byte. Q4_0 zero-point is always 8.
void unpack_row_q4_0(const char * xx, int k,
                     std::vector<int8_t> & weights,
                     std::vector<int8_t> & /*zeros*/,
                     std::vector<float> & scales) {
    const auto * x = reinterpret_cast<const block_q4_0 *>(xx);
    static const int qk = QK4_0;
    assert(k % qk == 0);

    const int nb = k / qk;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        scales.push_back(d);
        for (int j = 0; j < qk / 2; ++j) {
            weights.push_back(x[i].qs[j] & 0xF);
        }
        for (int j = 0; j < qk / 2; ++j) {
            weights.push_back(x[i].qs[j] >> 4);
        }
    }
}

} // namespace

#ifndef RYZENAI_EMULATION

using op_t = ryzenai::qlinear_2<int16_t, int8_t, float, float>;

// Singleton holding all per-weight qlinear_2 instances, keyed by tensor name.
class RyzenAIContext {
    RyzenAIContext() = default;
    RyzenAIContext(const RyzenAIContext &) = delete;
    RyzenAIContext & operator=(const RyzenAIContext &) = delete;
    std::mutex mtx_;

public:
    static RyzenAIContext & get() {
        static RyzenAIContext instance;
        return instance;
    }

    void lock()   { mtx_.lock(); }
    void unlock() { mtx_.unlock(); }

    std::unordered_map<std::string_view, op_t> map;
};

#endif // !RYZENAI_EMULATION

#ifndef RYZENAI_EMULATION

// Internal: ensure a qlinear_2 instance exists in the singleton map for this
// weight tensor; create + initialize it from the Q4_0 data on first call.
// Returns true if a new instance was created, false if already cached.
// Caller must hold the context lock.
static bool ensure_op_for_weight_locked(RyzenAIContext & ctx,
                                        const struct ggml_tensor * src0) {
    auto key = std::string_view(src0->name);
    if (ctx.map.count(key) != 0) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1];
    const size_t  nb02 = src0->nb[2];
    const size_t  nb03 = src0->nb[3];

    RYZ_TRY_CATCH(ctx.map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(key),
        std::forward_as_tuple("bfloat16", "uint4", "float32"));)

    std::vector<int8_t> weights;
    weights.reserve(ggml_nelements(src0));
    std::vector<int8_t> zeros(ggml_nelements(src0) / 32, 8);
    std::vector<float> scales;
    scales.reserve(ggml_nelements(src0) / 32);
    std::vector<float> bias(ne01, 0.0f);

    void * w = src0->data;
    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                unpack_row_q4_0((const char *) w + i01 * nb01 + i02 * nb02 + i03 * nb03,
                                ne00, weights, zeros, scales);
            }
        }
    }

    // See note in ggml_ryzenai_impl_mul_mat_npu about the transpose trick.
    auto transposed_weights = transpose_inner(weights,
        std::make_tuple((int) ne00, (int) ne01, (int) ne02, (int) ne03));
    auto transposed_scales  = transpose_inner(scales,
        std::make_tuple((int) (ne00 / 32), (int) ne01, (int) ne02, (int) ne03));

    auto w_shape = std::make_tuple((int) ne00, (int) ne01);

    RYZ_TRY_CATCH(ctx.map.at(key).initialize_weights_int4(
        transposed_weights.data(), zeros.data(), transposed_scales.data(),
        bias.data(), w_shape);)
    return true;
}

#endif // !RYZENAI_EMULATION

void ggml_ryzenai_impl_init(void) {
    static bool initialized = false;
    if (initialized) {
        return;
    }
#ifdef RYZENAI_EMULATION
    GGML_LOG_WARN("==================================================================\n");
    GGML_LOG_WARN("  ggml-ryzenai: running in EMULATION mode (no NPU).\n");
    GGML_LOG_WARN("  Eligible matmuls are computed by a SCALAR SOFTWARE GEMM.\n");
    GGML_LOG_WARN("  This is a correctness baseline, NOT a performance path.\n");
    GGML_LOG_WARN("  Rebuild with the AMD Ryzen AI SDK installed for real NPU.\n");
    GGML_LOG_WARN("==================================================================\n");
#else
    // Eager xclbin / AIE program load. Constructing a qlinear_2 instance
    // triggers xrt_context::get_instance(XCLBIN_FNAME), which loads the
    // .xclbin onto the NPU and prepares the kernel handles. We discard the
    // sentinel; the singleton xrt_context stays alive for subsequent ops.
    //
    // Fail-fast: if the NPU driver isn't loaded or the device is missing,
    // the qlinear_2 ctor throws. We let it propagate as an abort so the
    // user sees the failure immediately at backend init time rather than
    // on the first decode step.
    try {
        op_t sentinel("bfloat16", "uint4", "float32");
        (void) sentinel;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("ggml-ryzenai: NPU initialization failed: %s\n", e.what());
        GGML_ABORT("ggml-ryzenai: NPU initialization failed (driver loaded? device present?)");
    } catch (...) {
        GGML_ABORT("ggml-ryzenai: NPU initialization failed (unknown exception)");
    }
    GGML_LOG_INFO("ggml-ryzenai: NPU initialized (xclbin loaded)\n");
#endif
    initialized = true;
}

bool ggml_ryzenai_impl_can_mul_mat(const struct ggml_tensor * src0,
                                   const struct ggml_tensor * src1,
                                   const struct ggml_tensor * dst) {
    const int64_t ne3 = dst->ne[3];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne0 = dst->ne[0];

    return src0->type == GGML_TYPE_Q4_0 &&
           ggml_is_contiguous(src1) && src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 &&
           ne3 == 1 && ne2 == 1 &&
           ne0 >= 4096;
}

// Software emulation path: dequantize Q4_0 and do a plain GEMM. Only used
// when the SDK is not available at build time.
static void ggml_ryzenai_impl_mul_mat_emu(const struct ggml_tensor * src0,
                                          const struct ggml_tensor * src1,
                                          struct ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    void * w = src0->data;

    std::vector<int8_t> weights;
    std::vector<int8_t> zeros;
    std::vector<float>  scales;

    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                unpack_row_q4_0((const char *) w + i01 * nb01 + i02 * nb02 + i03 * nb03,
                                ne00, weights, zeros, scales);
            }
        }
    }

    std::vector<float> A(ggml_nelements(src0));
    int64_t sidx = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        // Q4_0 zero point is 8 (symmetric int4 range).
        A[i] = scales[sidx] * (weights[i] - 8);
        if ((i + 1) % 32 == 0) {
            ++sidx;
        }
    }

    const float * aptr = A.data();
    const float * bptr = (const float *) src1->data;
    float * cptr       = (float *) dst->data;

    for (int i3 = 0; i3 < ne3; ++i3) {
        for (int i2 = 0; i2 < ne2; ++i2) {
            for (int i1 = 0; i1 < ne1; ++i1) {
                for (int i0 = 0; i0 < ne0; ++i0) {
                    float acc = 0.0f;
                    for (int k = 0; k < ne10; ++k) {
                        acc += aptr[i3 * ne02 * ne01 * ne00 + i2 * ne01 * ne00 + i0 * ne00 + k] *
                               bptr[i3 * ne12 * ne11 * ne10 + i2 * ne11 * ne10 + i1 * ne10 + k];
                    }
                    cptr[i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0] = acc;
                }
            }
        }
    }
}

#ifndef RYZENAI_EMULATION

static void ggml_ryzenai_impl_mul_mat_npu(const struct ggml_tensor * src0,
                                          const struct ggml_tensor * src1,
                                          struct ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    auto & ctx = RyzenAIContext::get();

    // Lazy fallback in case preload_weights wasn't called for this tensor.
    ctx.lock();
    ensure_op_for_weight_locked(ctx, src0);
    ctx.unlock();

    auto key = std::string_view(src0->name);

    // Convert F32 activations to bfloat16 for the NPU kernel.
    const static bool use_avx = ryzenai::check_avx512_and_bf16_support();
    std::vector<int16_t> bf16_inputs(ggml_nelements(src1));
    ryzenai::float_buffer_to_bfloat16((float *) src1->data,
                                      ggml_nelements(src1),
                                      (uint16_t *) bf16_inputs.data(),
                                      use_avx);

    RYZ_TRY_CATCH(ctx.map.at(key).execute(
        bf16_inputs.data(),
        std::make_tuple((int) ne11, (int) ne10),
        (float *) dst->data);)

    GGML_UNUSED(nb01); GGML_UNUSED(nb02); GGML_UNUSED(nb03);
    GGML_UNUSED(ne00); GGML_UNUSED(ne01); GGML_UNUSED(ne02); GGML_UNUSED(ne03);
}

#endif // !RYZENAI_EMULATION

bool ggml_ryzenai_impl_preload_weight(const struct ggml_tensor * src0,
                                      const struct ggml_tensor * src1,
                                      const struct ggml_tensor * dst) {
    if (!ggml_ryzenai_impl_can_mul_mat(src0, src1, dst)) {
        return false;
    }
#ifndef RYZENAI_EMULATION
    auto & ctx = RyzenAIContext::get();
    ctx.lock();
    bool created = ensure_op_for_weight_locked(ctx, src0);
    ctx.unlock();
    return created;
#else
    return false;
#endif
}

void ggml_ryzenai_impl_mul_mat(const struct ggml_tensor * src0,
                               const struct ggml_tensor * src1,
                               struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_ryzenai_impl_can_mul_mat(src0, src1, dst));
#ifdef RYZENAI_EMULATION
    ggml_ryzenai_impl_mul_mat_emu(src0, src1, dst);
#else
    ggml_ryzenai_impl_mul_mat_npu(src0, src1, dst);
#endif
}
