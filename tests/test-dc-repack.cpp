#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#undef NDEBUG
#include <cassert>

static std::vector<float> compute(
        ggml_backend_t backend, ggml_type type, ggml_type repacked_type,
        ggml_tensor_repack_mode_t mode, int threads, bool no_repack) {
    constexpr int k = 256;
    constexpr int m = 64;
    constexpr int n = 3;
    ggml_cpu_set_tensor_repack_mode(mode);
    ggml_backend_cpu_set_n_threads(backend, threads);

    ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    assert(ctx);
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * inputs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    if (no_repack) {
        ggml_set_no_repack(weights);
    }
    ggml_tensor * output = ggml_mul_mat(ctx, weights, inputs);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);

    std::vector<float> data(k * m);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(float(i) * 0.37f);
    }
    std::vector<uint8_t> packed(ggml_nbytes(weights));
    assert(ggml_quantize_chunk(type, data.data(), packed.data(), 0, m, k, nullptr) == packed.size());
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    for (int i = 0; i < k * n; ++i) {
        data[i] = std::cos(float(i) * 0.19f);
    }
    ggml_backend_tensor_set(inputs, data.data(), 0, ggml_nbytes(inputs));

    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto repack = reinterpret_cast<void (*)(ggml_cgraph *)>(
        ggml_backend_reg_get_proc_address(reg, "ggml_cpu_repack_tensor_callgraph"));
    assert(repack);
    repack(graph);

    const bool should_repack = !no_repack &&
        (mode == GGML_TENSOR_REPACK_MODE_XBOX ||
         mode == GGML_TENSOR_REPACK_MODE_XBCG ||
         mode == GGML_TENSOR_REPACK_MODE_XBOX_SINGLE_THREAD);
    if (type == GGML_TYPE_Q8_0 &&
        (mode == GGML_TENSOR_REPACK_MODE_XBCG || mode == GGML_TENSOR_REPACK_MODE_XBOX_SINGLE_THREAD)) {
        // DC's single-thread producer uses this tag for the same Q8_0 layout.
        repacked_type = GGML_TYPE_Q8_0_Q8_0_x8;
    }
    assert(weights->type == (should_repack && mode == GGML_TENSOR_REPACK_MODE_XBCG ? repacked_type : type));
    std::vector<float> result(m * n);
    for (int repeat = 0; repeat < 2; ++repeat) {
        assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        assert(weights->type == (should_repack ? repacked_type : type));
        std::vector<float> current(m * n);
        ggml_backend_tensor_get(output, current.data(), 0, ggml_nbytes(output));
        if (repeat != 0) {
            assert(current == result);
        }
        result = current;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

int main() {
    ggml_backend_t backend = ggml_backend_cpu_init();
    assert(backend);
    const ggml_type types[][2] = {
        { GGML_TYPE_Q4_0, GGML_TYPE_Q4_0_x8 },
        { GGML_TYPE_Q8_0, GGML_TYPE_Q8_0_x8 },
        { GGML_TYPE_Q2_K, GGML_TYPE_Q2_K_x8 },
        { GGML_TYPE_Q3_K, GGML_TYPE_Q3_K_x8 },
        { GGML_TYPE_Q4_K, GGML_TYPE_Q4_K_x8 },
        { GGML_TYPE_Q6_K, GGML_TYPE_Q6_K_x8 },
        { GGML_TYPE_Q2_0, GGML_TYPE_Q2_0 },
    };
    const ggml_tensor_repack_mode_t modes[] = {
        GGML_TENSOR_REPACK_MODE_XBOX,
        GGML_TENSOR_REPACK_MODE_XBCG,
        GGML_TENSOR_REPACK_MODE_XBOX_SINGLE_THREAD,
        GGML_TENSOR_MULMAT_MODE_XBOX,
    };
    for (const auto & types_pair : types) {
        const auto reference = compute(backend, types_pair[0], types_pair[1], GGML_TENSOR_REPACK_MODE_NONE, 1, false);
        for (int threads : { 1, 4 }) {
            for (bool no_repack : { false, true }) {
                for (auto mode : modes) {
                    const auto actual = compute(backend, types_pair[0], types_pair[1], mode, threads, no_repack);
                    float max_error = 0.0f;
                    for (size_t i = 0; i < actual.size(); ++i) {
                        const float error = std::abs(actual[i] - reference[i]);
                        max_error = std::max(max_error, error);
                        assert(std::isfinite(actual[i]));
                        assert(error <= 5e-4f * (1.0f + std::abs(reference[i])));
                    }
                    std::printf("%s mode=%d threads=%d no_repack=%d error=%g: PASS\n",
                                ggml_type_name(types_pair[0]), int(mode), threads, no_repack, max_error);
                }
            }
        }
    }
    ggml_cpu_set_tensor_repack_mode(GGML_TENSOR_REPACK_MODE_NONE);
    ggml_backend_free(backend);
    return 0;
}
