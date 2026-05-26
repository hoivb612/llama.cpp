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

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Q4_0 row unpack: two int4 weights per byte. Q4_0 zero-point is always 8.
// Used by the EMU path only; the real NPU path uses the fused unpack+transpose
// below to avoid materializing the row-major intermediate.
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

// Fused Q4_0 unpack + transpose for the NPU upload path.
//
// qlinear_2::initialize_weights_int4 wants int4 weights laid out so the K
// dimension (ne00) is contiguous in the outer index, i.e. column-major in
// (ne00, ne01) terms. ggml's Q4_0 is row-major. The previous implementation
// did: unpack_row_q4_0 → row-major int8 vector → transpose_inner → transposed
// int8 vector, materializing two ~1-byte/weight intermediates plus a separate
// scales/transposed-scales pair.
//
// This fused walker writes int4 weights (as int8, range 0..15) and fp32 scales
// directly into the final transposed layout in one pass. Memory cost is
// exactly one output for each, no intermediates. The qlinear_2 BO consumed
// from these buffers, after which the caller frees both (see comment in
// ensure_op_for_weight_locked).
static void unpack_q4_0_transposed(const struct ggml_tensor * src0,
                                   int8_t * weights_T,
                                   float  * scales_T) {
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];
    const size_t  nb01 = src0->nb[1];
    const size_t  nb02 = src0->nb[2];
    const size_t  nb03 = src0->nb[3];

    GGML_ASSERT(ne00 % QK4_0 == 0);
    const int64_t nblocks_per_row = ne00 / QK4_0;

    // Strides for the transposed output (outer two dims swapped).
    const int64_t w_s2 = ne00 * ne01;
    const int64_t w_s3 = w_s2 * ne02;
    const int64_t s_s2 = nblocks_per_row * ne01;
    const int64_t s_s3 = s_s2 * ne02;

    const char * base = (const char *) src0->data;
    for (int64_t i03 = 0; i03 < ne03; ++i03) {
        for (int64_t i02 = 0; i02 < ne02; ++i02) {
            for (int64_t i01 = 0; i01 < ne01; ++i01) {
                const block_q4_0 * row = (const block_q4_0 *)
                    (base + i01 * nb01 + i02 * nb02 + i03 * nb03);
                int8_t * w_batch = weights_T + i03 * w_s3 + i02 * w_s2;
                float  * s_batch = scales_T  + i03 * s_s3 + i02 * s_s2;
                for (int64_t g = 0; g < nblocks_per_row; ++g) {
                    s_batch[g * ne01 + i01] = GGML_FP16_TO_FP32(row[g].d);
                    const int64_t col_lo = g * QK4_0;
                    const int64_t col_hi = col_lo + QK4_0 / 2;
                    for (int j = 0; j < QK4_0 / 2; ++j) {
                        const uint8_t b = (uint8_t) row[g].qs[j];
                        w_batch[(col_lo + j) * ne01 + i01] = (int8_t) (b & 0xF);
                        w_batch[(col_hi + j) * ne01 + i01] = (int8_t) (b >> 4);
                    }
                }
            }
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

// Per-call timing counters for the NPU mul_mat hot path. Enabled at runtime
// via env var GGML_RYZENAI_STATS=1. Prints a summary at process exit.
struct RyzenAIStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> calls_m_eq_1{0};
    std::atomic<uint64_t> calls_m_gt_1{0};
    std::atomic<uint64_t> us_convert{0};
    std::atomic<uint64_t> us_execute{0};
    std::atomic<uint64_t> us_alloc{0};
    std::atomic<uint64_t> bytes_activations{0};
    bool enabled = false;

    RyzenAIStats() {
        const char * env = std::getenv("GGML_RYZENAI_STATS");
        enabled = env && env[0] != '0' && env[0] != '\0';
    }

    ~RyzenAIStats() {
        if (!enabled || calls.load() == 0) return;
        uint64_t n = calls.load();
        uint64_t conv = us_convert.load();
        uint64_t exe  = us_execute.load();
        uint64_t alc  = us_alloc.load();
        uint64_t total = conv + exe + alc;
        std::fprintf(stderr,
            "\n[ggml-ryzenai stats]\n"
            "  calls           = %llu  (M=1: %llu, M>1: %llu)\n"
            "  alloc bf16 buf  = %8.2f ms  (%5.2f%%, avg %6.2f us/call)\n"
            "  F32->BF16 conv  = %8.2f ms  (%5.2f%%, avg %6.2f us/call)\n"
            "  NPU execute()   = %8.2f ms  (%5.2f%%, avg %6.2f us/call)\n"
            "  total measured  = %8.2f ms\n"
            "  activations DMA = %8.2f MB\n",
            (unsigned long long) n,
            (unsigned long long) calls_m_eq_1.load(),
            (unsigned long long) calls_m_gt_1.load(),
            alc  / 1000.0, total ? 100.0 * alc  / total : 0.0, n ? (double) alc  / n : 0.0,
            conv / 1000.0, total ? 100.0 * conv / total : 0.0, n ? (double) conv / n : 0.0,
            exe  / 1000.0, total ? 100.0 * exe  / total : 0.0, n ? (double) exe  / n : 0.0,
            total / 1000.0,
            bytes_activations.load() / (1024.0 * 1024.0));
    }
};

static RyzenAIStats & ryzenai_stats() {
    static RyzenAIStats s;
    return s;
}

#ifndef RYZENAI_EMULATION

// Internal: ensure a qlinear_2 instance exists in the singleton map for this
// weight tensor; create + initialize it from the Q4_0 data on first call.
// Returns true if a new instance was created, false if already cached.
// Caller must hold the context lock.
//
// Memory note: qlinear_2::initialize_weights_int4 copies/formats/tiles the
// input buffers into NPU-side xrt::bo objects. After that call returns,
// the host-side scratch (weights, scales, zeros, bias) is no longer
// referenced by qlinear_2 (verified against qlinear_2.hpp v1.6 — the only
// weight storage is `std::vector<xrt::bo> weights_bo_`; execute() reads
// from BO references and never re-touches the input pointers). We therefore
// scope the scratch tightly so it is destroyed before the function returns.
static bool ensure_op_for_weight_locked(RyzenAIContext & ctx,
                                        const struct ggml_tensor * src0) {
    auto key = std::string_view(src0->name);
    if (ctx.map.count(key) != 0) {
        return false;
    }

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    RYZ_TRY_CATCH(ctx.map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(key),
        std::forward_as_tuple("bfloat16", "uint4", "float32"));)

    {
        // Single fused pass: unpack Q4_0 directly into the transposed layout
        // qlinear_2 expects. Eliminates the row-major-int8 + transposed-int8
        // + row-major-fp32 + transposed-fp32 intermediates of the legacy
        // unpack_row_q4_0 + transpose_inner pipeline.
        const int64_t N = ggml_nelements(src0);
        std::vector<int8_t> weights_T((size_t) N);
        std::vector<float>  scales_T((size_t) N / QK4_0);
        std::vector<int8_t> zeros((size_t) N / QK4_0, 8);
        std::vector<float>  bias((size_t) ne01, 0.0f);

        unpack_q4_0_transposed(src0, weights_T.data(), scales_T.data());

        // See note in ggml_ryzenai_impl_mul_mat_npu about the transpose trick.
        auto w_shape = std::make_tuple((int) ne00, (int) ne01);

        RYZ_TRY_CATCH(ctx.map.at(key).initialize_weights_int4(
            weights_T.data(), zeros.data(), scales_T.data(),
            bias.data(), w_shape);)

        // weights_T, scales_T, zeros, bias destroyed here; their backing
        // pages are released before the next tensor is uploaded.
    }
    return true;
}

#endif // !RYZENAI_EMULATION

void ggml_ryzenai_impl_init(void) {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    // Force-construct the stats singleton early so the GGML_RYZENAI_STATS
    // env var is read before any matmul calls and the dtor prints the
    // summary at process exit.
    auto & stats = ryzenai_stats();
    if (stats.enabled) {
        GGML_LOG_INFO("ggml-ryzenai: per-call stats enabled (GGML_RYZENAI_STATS=1)\n");
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

// Minimum M dimension (number of activation tokens) required to claim
// MUL_MAT for the NPU. Read once from env var GGML_RYZENAI_MIN_M.
//
// Rationale: NPU execute() has ~2 ms of fixed per-call overhead
// (kernel launch + DMA descriptor setup + AIE program start + completion
// signal) regardless of M. At M=1 (decode), arithmetic intensity is so
// low that CPU AVX-512 + Q4_0 repack consistently beats the NPU by 2-4x.
// At M>=16 the NPU starts to amortize this overhead and pulls ahead.
//
// Default = 2 (skip M=1 decode, run everything else on NPU). Set to 1
// to claim everything (legacy behavior), or a higher value to be more
// conservative.
static int ryzenai_min_m() {
    static int cached = []() {
        const char * env = std::getenv("GGML_RYZENAI_MIN_M");
        if (env && env[0] != '\0') {
            int v = std::atoi(env);
            if (v >= 1) return v;
        }
        return 2; // default: skip decode (M=1), keep prefill / batched
    }();
    return cached;
}

bool ggml_ryzenai_impl_can_mul_mat(const struct ggml_tensor * src0,
                                   const struct ggml_tensor * src1,
                                   const struct ggml_tensor * dst) {
    const int64_t ne3 = dst->ne[3];
    const int64_t ne2 = dst->ne[2];
    const int64_t ne0 = dst->ne[0];
    const int64_t M   = src1->ne[1];

    return src0->type == GGML_TYPE_Q4_0 &&
           ggml_is_contiguous(src1) && src1->type == GGML_TYPE_F32 &&
           dst->type == GGML_TYPE_F32 &&
           ne3 == 1 && ne2 == 1 &&
           ne0 >= 4096 &&
           M >= ryzenai_min_m();
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
    auto & stats = ryzenai_stats();

    // Lazy fallback in case preload_weights wasn't called for this tensor.
    ctx.lock();
    ensure_op_for_weight_locked(ctx, src0);
    ctx.unlock();

    auto key = std::string_view(src0->name);

    const int64_t n_elem = ggml_nelements(src1);

    // ---- segment 1: allocate BF16 activation buffer ----
    int64_t t0 = stats.enabled ? ggml_time_us() : 0;
    std::vector<int16_t> bf16_inputs(n_elem);
    int64_t t1 = stats.enabled ? ggml_time_us() : 0;

    // ---- segment 2: F32 -> BF16 conversion (CPU AVX-512) ----
    const static bool use_avx = ryzenai::check_avx512_and_bf16_support();
    ryzenai::float_buffer_to_bfloat16((float *) src1->data,
                                      n_elem,
                                      (uint16_t *) bf16_inputs.data(),
                                      use_avx);
    int64_t t2 = stats.enabled ? ggml_time_us() : 0;

    // ---- segment 3: NPU execute (DMA + compute + writeback) ----
    RYZ_TRY_CATCH(ctx.map.at(key).execute(
        bf16_inputs.data(),
        std::make_tuple((int) ne11, (int) ne10),
        (float *) dst->data);)
    int64_t t3 = stats.enabled ? ggml_time_us() : 0;

    if (stats.enabled) {
        stats.calls.fetch_add(1, std::memory_order_relaxed);
        if (ne11 == 1) {
            stats.calls_m_eq_1.fetch_add(1, std::memory_order_relaxed);
        } else {
            stats.calls_m_gt_1.fetch_add(1, std::memory_order_relaxed);
        }
        stats.us_alloc.fetch_add((uint64_t)(t1 - t0), std::memory_order_relaxed);
        stats.us_convert.fetch_add((uint64_t)(t2 - t1), std::memory_order_relaxed);
        stats.us_execute.fetch_add((uint64_t)(t3 - t2), std::memory_order_relaxed);
        stats.bytes_activations.fetch_add((uint64_t)(n_elem * sizeof(int16_t)), std::memory_order_relaxed);
    }

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
