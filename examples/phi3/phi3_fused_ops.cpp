#include "phi3_fused_ops.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama_b612.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

// Pull weight bytes into a host-visible mirror so we can call the CPU vec_dot
// kernel directly. For non-host (e.g. CUDA) backends this is a one-time copy
// at init; for host backends we just borrow the live data pointer.
bool resolve_weight_data(Phi3FusedLmHead & lm, std::string & error) {
    const ggml_tensor * w = lm.weight;
    if (!w) {
        error = "phi3 fused lm_head: weight tensor not set";
        return false;
    }

    ggml_backend_buffer_t buf = w->buffer;
    if (buf && ggml_backend_buffer_is_host(buf) && w->data) {
        lm.weight_data = static_cast<const uint8_t *>(w->data);
        lm.host_weight.clear();
        return true;
    }

    const size_t total = ggml_nbytes(w);
    lm.host_weight.assign(total, 0);
    ggml_backend_tensor_get(w, lm.host_weight.data(), 0, total);
    lm.weight_data = lm.host_weight.data();
    return true;
}

} // namespace

bool phi3_fused_lmhead_init(const llama_model * model, Phi3FusedLmHead & out, std::string & error) {
    out = {};

    if (!model) {
        error = "phi3 fused lm_head: model is null";
        return false;
    }

    // Prefer explicit lm_head weight; fall back to tied token-embedding for
    // small Phi3 variants that share the matrix.
    const ggml_tensor * w = llama_model_get_tensor_by_name(model, "output.weight");
    if (!w) {
        w = llama_model_get_tensor_by_name(model, "token_embd.weight");
    }
    if (!w) {
        error = "phi3 fused lm_head: cannot find output.weight or token_embd.weight";
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu(w->type);
    if (!w_traits || !w_traits->vec_dot) {
        error = "phi3 fused lm_head: weight type has no CPU vec_dot kernel";
        return false;
    }

    const ggml_type q_type   = w_traits->vec_dot_type;
    const auto *    q_traits = ggml_get_type_traits_cpu(q_type);
    if (!q_traits || !q_traits->from_float) {
        error = "phi3 fused lm_head: vec_dot_type has no from_float kernel";
        return false;
    }

    out.weight       = w;
    out.n_embd       = w->ne[0];
    out.n_vocab      = w->ne[1];
    out.weight_type  = (int) w->type;
    out.quant_type   = (int) q_type;
    out.weight_row_bytes = ggml_row_size(w->type, out.n_embd);

    out.quant_hidden.assign(ggml_row_size(q_type, out.n_embd), 0);

    if (!resolve_weight_data(out, error)) {
        return false;
    }

    error.clear();
    return true;
}

bool phi3_fused_lmhead_argmax(
        Phi3FusedLmHead & lm,
        const float *     hidden,
        int               n_threads,
        llama_token &     out_token,
        std::string &     error) {
    if (!hidden) {
        error = "phi3 fused lm_head: hidden state is null";
        return false;
    }
    if (lm.n_vocab <= 0 || lm.n_embd <= 0 || !lm.weight_data) {
        error = "phi3 fused lm_head: not initialized";
        return false;
    }

    const auto * w_traits = ggml_get_type_traits_cpu((ggml_type) lm.weight_type);
    const auto * q_traits = ggml_get_type_traits_cpu((ggml_type) lm.quant_type);
    if (!w_traits || !w_traits->vec_dot || !q_traits || !q_traits->from_float) {
        error = "phi3 fused lm_head: missing CPU traits";
        return false;
    }

    // Once-per-step: quantize hidden state into the format expected by vec_dot.
    q_traits->from_float(hidden, lm.quant_hidden.data(), lm.n_embd);

    const uint8_t * w_base       = lm.weight_data;
    const size_t    row_bytes    = lm.weight_row_bytes;
    const int64_t   n_vocab      = lm.n_vocab;
    const int       n_embd_int   = (int) lm.n_embd;
    const void *    q_hidden_ptr = lm.quant_hidden.data();

    int n_workers = n_threads > 0 ? n_threads : 1;
    if (n_workers > n_vocab) {
        n_workers = (int) n_vocab;
    }

    if (n_workers <= 1) {
        float   best_val = -std::numeric_limits<float>::infinity();
        int64_t best_id  = 0;
        for (int64_t v = 0; v < n_vocab; ++v) {
            float s = 0.0f;
            w_traits->vec_dot(n_embd_int, &s, 0, w_base + (size_t) v * row_bytes, 0, q_hidden_ptr, 0, 1);
            if (s > best_val) {
                best_val = s;
                best_id  = v;
            }
        }
        out_token = (llama_token) best_id;
        error.clear();
        return true;
    }

    struct Slot { float val; int64_t id; };
    std::vector<Slot> partials(n_workers, Slot{-std::numeric_limits<float>::infinity(), 0});
    std::vector<std::thread> workers;
    workers.reserve(n_workers);

    const int64_t chunk = (n_vocab + n_workers - 1) / n_workers;
    for (int wid = 0; wid < n_workers; ++wid) {
        const int64_t lo = (int64_t) wid * chunk;
        const int64_t hi = std::min<int64_t>(lo + chunk, n_vocab);
        if (lo >= hi) {
            continue;
        }
        workers.emplace_back([&, wid, lo, hi]() {
            float   best_val = -std::numeric_limits<float>::infinity();
            int64_t best_id  = lo;
            for (int64_t v = lo; v < hi; ++v) {
                float s = 0.0f;
                w_traits->vec_dot(n_embd_int, &s, 0, w_base + (size_t) v * row_bytes, 0, q_hidden_ptr, 0, 1);
                if (s > best_val) {
                    best_val = s;
                    best_id  = v;
                }
            }
            partials[wid].val = best_val;
            partials[wid].id  = best_id;
        });
    }
    for (auto & t : workers) {
        t.join();
    }

    float   best_val = -std::numeric_limits<float>::infinity();
    int64_t best_id  = 0;
    for (const auto & p : partials) {
        if (p.val > best_val) {
            best_val = p.val;
            best_id  = p.id;
        }
    }
    out_token = (llama_token) best_id;
    error.clear();
    return true;
}
