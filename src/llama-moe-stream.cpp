#include "llama-moe-stream.h"
#include "llama-layer-window.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <atomic>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

bool llama_moe_stream_enabled() {
    // On by default under an active weight budget (--weight-budget). The
    // GGML_LW_MOE_STREAM env var is an explicit override (0 = off, 1 = on).
    static const int env = []() {
        const char * e = getenv("GGML_LW_MOE_STREAM");
        if (!e || !e[0]) return -1;           // unset -> follow the weight budget
        return (e[0] != '0') ? 1 : 0;         // explicit override
    }();
    if (env >= 0) return env != 0;
    const layer_window_manager * m = llama_get_layer_window_manager();
    return m && m->budget_bytes > 0;
}

namespace {

// Phase A2: persistent expert pool (delta streaming). On by default under an
// active weight budget (--weight-budget); GGML_LW_MOE_POOL is an explicit
// override (0 = off -> fall back to the M2 per-token compact gather, 1 = on).
// When off, the M2 per-token compact gather is used (correct, but re-copies every
// routed expert every token). When on, decode keeps gathered experts resident in
// a per-(layer,role) host-coherent Vulkan buffer and only copies the per-token
// *misses* (the 74.9%-hit telemetry says ~2 of 8 change per token), cutting the
// per-token expert copy ~4x.
bool moe_pool_enabled() {
    static const int env = []() {
        const char * e = getenv("GGML_LW_MOE_POOL");
        if (!e || !e[0]) return -1;           // unset -> follow the weight budget
        return (e[0] != '0') ? 1 : 0;         // explicit override
    }();
    if (env >= 0) return env != 0;
    const layer_window_manager * m = llama_get_layer_window_manager();
    return m && m->budget_bytes > 0;
}

// Explicit slot override (GGML_LW_MOE_POOL_SLOTS). Returns 0 when unset, meaning
// "derive the slot count from the weight budget".
int moe_pool_slots_override() {
    static const int slots = []() {
        const char * e = getenv("GGML_LW_MOE_POOL_SLOTS");
        int v = (e && e[0]) ? atoi(e) : 0;
        if (v < 0) v = 0;
        return v;
    }();
    return slots;
}

constexpr int MOE_STREAM_MAX_ROLES = 4; // gate_up, up, gate, down

// Everything the CPU gather callback needs, resolved at graph-build time and
// kept alive for the lifetime of the process (decode rebuilds the graph every
// token but reuses the same (il, role) slot; decode is sequential so build for
// token t+1 never overlaps compute for token t).
struct gather_ud {
    layer_window_manager * mgr        = nullptr; // for file-read fallback
    const uint8_t *        mmap_src   = nullptr; // mmap base + file_offset (or null)
    uint16_t               file_idx   = 0;
    size_t                 file_offset = 0;       // start of the full expert tensor
    size_t                 slab_bytes = 0;        // bytes per expert (nb[2])
    long long              n_expert_used = 0;
    long long              n_expert      = 0;     // total experts (full-copy count)
    bool                   full_copy  = false;    // prefill: gather ALL n_expert slabs
    bool                   warned     = false;
};

std::mutex               g_mtx;
std::vector<gather_ud *> g_uds; // indexed by il*MOE_STREAM_MAX_ROLES + role

gather_ud * get_ud(int il, int role) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const size_t idx = (size_t) il * MOE_STREAM_MAX_ROLES + role;
    if (idx >= g_uds.size()) {
        g_uds.resize(idx + 1, nullptr);
    }
    if (!g_uds[idx]) {
        g_uds[idx] = new gather_ud();
    }
    return g_uds[idx];
}

// ---- Phase A2: persistent per-(layer,role) expert pool ----
//
// A pool holds up to `pool_slots` expert slabs in a single host-coherent Vulkan
// buffer imported once (buffer_from_host_ptr). Across decode tokens, experts that
// are already resident (hit) are reused with zero copy; only misses are memcpy'd
// from mmap into their LRU-chosen slot (delta streaming). The pool memory is
// HOST_COHERENT, so these CPU writes are visible to the GPU mul_mat_id without an
// explicit flush. The patch custom-op emits per-token slot ids used to index the
// pool; mul_mat_id output position i is expert selected_experts[i], so the
// existing per-expert scale path (get_rows(scale, selected_experts)) is unchanged.
struct expert_pool {
    bool     inited     = false;
    bool     failed     = false;           // init failed -> caller falls back
    void *   host       = nullptr;          // committed page-aligned host memory
    size_t   total      = 0;                // bytes of host/imported buffer
    ggml_backend_buffer_t buf = nullptr;    // imported Vulkan buffer (host-coherent)
    size_t   slab_bytes = 0;                // bytes per expert (nb[2])
    int      pool_slots = 0;
    int64_t  n_expert   = 0;
    int64_t  ne0 = 0, ne1 = 0;
    ggml_type type = GGML_TYPE_F32;

    std::vector<int32_t>  slot_expert;      // slot -> expert id (-1 = empty)
    std::vector<int32_t>  expert_slot;      // expert id -> slot (-1 = not resident)
    std::vector<uint64_t> slot_tick;        // slot -> last-use tick (LRU)
    uint64_t tick = 0;

    // source for slab fills
    layer_window_manager * mgr = nullptr;
    const uint8_t * mmap_src = nullptr;
    uint16_t file_idx = 0;
    size_t   file_offset = 0;

    uint64_t hits = 0, misses = 0;
    bool warned = false;
    int  il = -1, role = -1;
};

std::vector<expert_pool *> g_pools; // indexed like g_uds

expert_pool * get_pool(int il, int role) {
    std::lock_guard<std::mutex> lk(g_mtx);
    const size_t idx = (size_t) il * MOE_STREAM_MAX_ROLES + role;
    if (idx >= g_pools.size()) {
        g_pools.resize(idx + 1, nullptr);
    }
    if (!g_pools[idx]) {
        g_pools[idx] = new expert_pool();
    }
    return g_pools[idx];
}

// Total bytes of all streamed expert weight matrices (cached). Used to map the
// weight budget onto a per-pool slot count: slots = budget / per_slot_bytes,
// where per_slot_bytes = total_expert_bytes / n_expert (sum of one slab across
// every pool). Scans the manager's recorded tensor locations once.
size_t total_expert_bytes(layer_window_manager * mgr) {
    static size_t cached = 0;
    static bool   done   = false;
    if (done) return cached;
    size_t sum = 0;
    for (const auto & kv : mgr->layer_tensors) {
        for (const auto & loc : kv.second) {
            if (loc.name.find("_exps.weight") != std::string::npos) {
                sum += loc.n_bytes;
            }
        }
    }
    cached = sum;
    done   = true;
    return cached;
}

// Resolve the pool slot count for a (layer, role): explicit override if set,
// otherwise derived from the weight budget. Clamped to [n_expert_used, n_expert]
// so a pool always holds at least one full token's routed set. Returns 0 when
// the pool should not be used (e.g. no budget info and no override).
int resolve_pool_slots(layer_window_manager * mgr, int64_t n_expert, long long n_expert_used) {
    long long slots;
    const int override_slots = moe_pool_slots_override();
    if (override_slots > 0) {
        slots = override_slots;
    } else if (mgr->budget_bytes == 0) {
        slots = n_expert;                       // unlimited budget -> all resident
    } else {
        const size_t total = total_expert_bytes(mgr);
        if (total == 0) return 0;               // unknown -> caller falls back
        slots = (long long) ((double) mgr->budget_bytes * (double) n_expert / (double) total);
    }
    if (slots < n_expert_used) slots = n_expert_used;
    if (slots > n_expert)      slots = n_expert;
    return (int) slots;
}

// Lazily create the pool's host memory + imported Vulkan buffer. dev is derived
// from the (dummy) buffer that full_exps sits on under the residency split.
void init_pool(expert_pool * p, ggml_tensor * full_exps, int64_t n_expert, int slots, int il, int role) {
    p->inited = true;
    p->il = il; p->role = role;
    p->type       = full_exps->type;
    p->ne0        = full_exps->ne[0];
    p->ne1        = full_exps->ne[1];
    p->n_expert   = n_expert;
    p->slab_bytes = full_exps->nb[2];

    if (slots > (int) n_expert) slots = (int) n_expert;   // no more than all experts
    if (slots < 1) slots = 1;
    p->pool_slots = slots;

    if (!full_exps->buffer) { p->failed = true; return; }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(full_exps->buffer);
    ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
    if (!dev) { p->failed = true; return; }

    size_t align = ggml_backend_buft_get_alignment(buft);
    if (align == 0) align = 256;
    p->total = ((size_t) p->pool_slots * p->slab_bytes + align - 1) & ~(align - 1);

#ifdef _WIN32
    const unsigned long MEM_COMMIT_  = 0x00001000;
    const unsigned long MEM_RESERVE_ = 0x00002000;
    const unsigned long PAGE_RW_     = 0x04;
    p->host = VirtualAlloc(nullptr, p->total, MEM_COMMIT_ | MEM_RESERVE_, PAGE_RW_);
#else
    p->host = malloc(p->total);
#endif
    if (!p->host) { p->failed = true; return; }

    p->buf = ggml_backend_dev_buffer_from_host_ptr(dev, p->host, p->total, p->slab_bytes);
    if (!p->buf) {
        // Not an integrated/host-ptr-importing device (e.g. dGPU) — fall back.
#ifdef _WIN32
        VirtualFree(p->host, 0, 0x8000 /*MEM_RELEASE*/);
#else
        free(p->host);
#endif
        p->host = nullptr;
        p->failed = true;
        return;
    }

    p->slot_expert.assign(p->pool_slots, -1);
    p->expert_slot.assign((size_t) n_expert, -1);
    p->slot_tick.assign(p->pool_slots, 0);

    if (il == 0 && role == 0 && p->mgr) {
        const size_t total_exp = total_expert_bytes(p->mgr);
        const double per_slot  = n_expert ? (double) total_exp / (double) n_expert : 0.0;
        fprintf(stderr,
            "LW-MOE-POOL: budget=%.0f MiB total_experts=%.1f GiB n_expert=%lld => %d slots/pool "
            "(~%.2f GiB pool RAM, %s)\n",
            p->mgr->budget_bytes / (1024.0 * 1024.0),
            total_exp / (1024.0 * 1024.0 * 1024.0), (long long) n_expert, p->pool_slots,
            (per_slot * p->pool_slots) / (1024.0 * 1024.0 * 1024.0),
            moe_pool_slots_override() > 0 ? "override" : "budget-derived");
    }

    fprintf(stderr, "LW-MOE-POOL: init il=%d role=%d slots=%d slab=%zu KiB total=%.1f MiB\n",
            il, role, p->pool_slots, p->slab_bytes / 1024,
            p->total / (1024.0 * 1024.0));
}

// Copy one expert's slab from mmap (preferred) or file into pool slot.
void pool_fill_slot(expert_pool * p, int slot, int32_t e) {
    uint8_t * dst = (uint8_t *) p->host + (size_t) slot * p->slab_bytes;
    if (p->mmap_src) {
        memcpy(dst, p->mmap_src + (size_t) e * p->slab_bytes, p->slab_bytes);
    } else if (p->mgr && p->mgr->read_tensor_bytes(
                   p->file_idx, p->file_offset + (size_t) e * p->slab_bytes, p->slab_bytes, dst)) {
        // read from file
    } else if (!p->warned) {
        p->warned = true;
        fprintf(stderr, "LW-MOE-POOL: no source for expert slab (file_idx=%u) — output will be wrong\n",
                (unsigned) p->file_idx);
        memset(dst, 0, p->slab_bytes);
    } else {
        memset(dst, 0, p->slab_bytes);
    }
}

// CPU custom-op: for each routed expert ensure it is resident in a slot (LRU
// evict + delta memcpy on miss), then emit that slot index. dst->src[0] =
// selected_experts (I32); dst->data receives the slot ids [n_expert_used].
void patch_pool_cb(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return; // single-task
    }
    expert_pool * p = (expert_pool *) userdata;

    const ggml_tensor * ids = dst->src[0];
    const int32_t *     sel = (const int32_t *) ids->data;
    int32_t *           out = (int32_t *) dst->data;
    const int64_t       n_used = dst->ne[0];

    for (int64_t i = 0; i < n_used; ++i) {
        const int32_t e = sel[i];
        int32_t slot = (e >= 0 && e < (int32_t) p->n_expert) ? p->expert_slot[e] : -1;
        if (slot < 0) {
            // miss: pick the least-recently-used slot (guaranteed not one used
            // this token because pool_slots >= n_expert_used and ticks bump each i)
            int32_t victim = 0;
            uint64_t oldest = p->slot_tick[0];
            for (int s = 1; s < p->pool_slots; ++s) {
                if (p->slot_tick[s] < oldest) { oldest = p->slot_tick[s]; victim = s; }
            }
            const int32_t prev = p->slot_expert[victim];
            if (prev >= 0 && prev < (int32_t) p->n_expert) p->expert_slot[prev] = -1;
            if (e >= 0 && e < (int32_t) p->n_expert) {
                p->slot_expert[victim] = e;
                p->expert_slot[e]      = victim;
                pool_fill_slot(p, victim, e);
            }
            slot = victim;
            p->misses++;
        } else {
            p->hits++;
        }
        p->slot_tick[slot] = ++p->tick;
        out[i] = slot;
    }

    // Optional cumulative hit-rate diagnostic (GGML_LW_MOE_POOL_DIAG=1). Confirms
    // the delta mechanism: a high hit-rate means most experts stay resident and
    // only misses are copied per token.
    static const bool diag = []() {
        const char * e = getenv("GGML_LW_MOE_POOL_DIAG");
        return e && e[0] && e[0] != '0';
    }();
    if (diag) {
        static std::atomic<uint64_t> g_calls{0};
        const uint64_t c = ++g_calls;
        if ((c % 3000) == 0) { // ~ every 50 decode tokens (60 pools/token)
            uint64_t h = 0, m = 0;
            for (auto * pp : g_pools) { if (pp) { h += pp->hits; m += pp->misses; } }
            const double rate = (h + m) ? 100.0 * (double) h / (double) (h + m) : 0.0;
            fprintf(stderr, "LW-MOE-POOL: cumulative hit-rate %.1f%% (hits=%llu miss=%llu)\n",
                    rate, (unsigned long long) h, (unsigned long long) m);
        }
    }
}

// Locate the file location for a specific expert tensor within a layer.
const layer_window_manager::tensor_location * find_location(
        layer_window_manager * mgr, int il, const ggml_tensor * t) {
    auto it = mgr->layer_tensors.find(il);
    if (it == mgr->layer_tensors.end()) {
        return nullptr;
    }
    for (const auto & loc : it->second) {
        if (loc.tensor == t) {
            return &loc;
        }
    }
    return nullptr;
}

// CPU custom-op: compact the routed experts into dst.
//   decode (full_copy == false): slot i = expert selected_experts[i], for
//     i in [0, n_expert_used). dst->src[0] = selected_experts (I32).
//   prefill (full_copy == true): slot e = expert e, for e in [0, n_expert).
//     The final mul_mat_id then indexes the compact tensor with the real
//     selected_experts ids, matching the full-expert path exactly.
void fill_compact_cb(ggml_tensor * dst, int ith, int /*nth*/, void * userdata) {
    if (ith != 0) {
        return; // single-task gather
    }
    gather_ud * ud = (gather_ud *) userdata;

    const size_t  slab = ud->slab_bytes;
    uint8_t *     out  = (uint8_t *) dst->data;

    auto copy_slab = [&](long long slot, int32_t e) {
        uint8_t * dstp = out + (size_t) slot * dst->nb[2];
        if (ud->mmap_src) {
            memcpy(dstp, ud->mmap_src + (size_t) e * slab, slab);
        } else if (ud->mgr && ud->mgr->read_tensor_bytes(
                       ud->file_idx, ud->file_offset + (size_t) e * slab, slab, dstp)) {
            // read from file
        } else if (!ud->warned) {
            ud->warned = true;
            fprintf(stderr, "LW-MOE: no source for expert slab (file_idx=%u) — output will be wrong\n",
                    (unsigned) ud->file_idx);
            memset(dstp, 0, slab);
        } else {
            memset(dstp, 0, slab);
        }
    };

    if (ud->full_copy) {
        for (long long e = 0; e < ud->n_expert; ++e) {
            copy_slab(e, (int32_t) e);
        }
    } else {
        const ggml_tensor * ids = dst->src[0];
        const int32_t *     sel = (const int32_t *) ids->data;
        for (long long i = 0; i < ud->n_expert_used; ++i) {
            copy_slab(i, sel[i]);
        }
    }
}

} // namespace

ggml_tensor * llama_moe_streamed_mul_mat_id(
        ggml_context * ctx0,
        ggml_tensor  * full_exps,
        ggml_tensor  * cur,
        ggml_tensor  * selected_experts,
        long long      n_expert_used,
        long long      n_expert,
        long long      n_tokens,
        int            il,
        int            role) {
    layer_window_manager * mgr = llama_get_layer_window_manager();
    if (!mgr || il < 0 || role < 0 || role >= MOE_STREAM_MAX_ROLES) {
        return nullptr;
    }

    const layer_window_manager::tensor_location * loc = find_location(mgr, il, full_exps);
    if (!loc) {
        return nullptr; // no file location — caller falls back to normal path
    }

    // Decode (n_tokens == 1): gather only the n_expert_used routed experts in
    // selection order, index with iota. Prefill (n_tokens > 1): gather ALL
    // n_expert experts (static shape) and index with the real routed ids.
    const bool full_copy = n_tokens > 1;

    // Phase A2: persistent expert pool with delta streaming (decode only). Keeps
    // routed experts resident across tokens and copies only per-token misses. The
    // slot count is derived from --weight-budget (or GGML_LW_MOE_POOL_SLOTS).
    if (moe_pool_enabled() && !full_copy) {
        const int slots = resolve_pool_slots(mgr, n_expert, n_expert_used);
        if (slots >= (int) n_expert_used) {
            expert_pool * p = get_pool(il, role);
            p->mgr = mgr;
            if (!p->inited) {
                init_pool(p, full_exps, n_expert, slots, il, role);
            }
            if (!p->failed) {
                p->file_idx    = loc->file_idx;
                p->file_offset = loc->file_offset;
                p->mmap_src    = (loc->file_idx < mgr->mmap_bases.size() && mgr->mmap_bases[loc->file_idx])
                                     ? mgr->mmap_bases[loc->file_idx] + loc->file_offset
                                     : nullptr;

                // patch op: ensure routed experts resident, emit slot ids [n_used, 1]
                ggml_tensor * pargs[1] = { selected_experts };
                ggml_tensor * slot_ids = ggml_custom_4d(ctx0, GGML_TYPE_I32,
                        n_expert_used, 1, 1, 1, pargs, 1, patch_pool_cb, 1, p);
                slot_ids = ggml_reshape_2d(ctx0, slot_ids, n_expert_used, 1);

                // persistent pool tensor viewing the imported host-coherent buffer
                ggml_tensor * poolt = ggml_new_tensor_3d(ctx0, p->type, p->ne0, p->ne1, p->pool_slots);
                poolt->data   = p->host;
                poolt->buffer = p->buf;

                return ggml_mul_mat_id(ctx0, poolt, cur, slot_ids);
            }
            // pool init failed (e.g. dGPU) — fall through to the M2 compact path.
        }
    }

    gather_ud * ud = get_ud(il, role);
    ud->mgr           = mgr;
    ud->file_idx      = loc->file_idx;
    ud->file_offset   = loc->file_offset;
    ud->slab_bytes    = full_exps->nb[2];
    ud->n_expert_used = n_expert_used;
    ud->n_expert      = n_expert;
    ud->full_copy     = full_copy;
    ud->mmap_src      = (loc->file_idx < mgr->mmap_bases.size() && mgr->mmap_bases[loc->file_idx])
                            ? mgr->mmap_bases[loc->file_idx] + loc->file_offset
                            : nullptr;

    const long long n_slabs = full_copy ? n_expert : n_expert_used;

    // compact expert tensor [ne0, ne1, n_slabs, 1]
    ggml_tensor * args[1] = { selected_experts };
    ggml_tensor * compact = ggml_custom_4d(ctx0, full_exps->type,
            full_exps->ne[0], full_exps->ne[1], n_slabs, 1,
            args, 1, fill_compact_cb, 1, ud);

    if (full_copy) {
        // compact[e] == full expert e -> index with the real routed ids.
        return ggml_mul_mat_id(ctx0, compact, cur, selected_experts);
    }

    // identity ids [n_expert_used, 1] = {0, 1, ..., n_expert_used-1}
    ggml_tensor * iota = ggml_cast(ctx0,
            ggml_arange(ctx0, 0.0f, (float) n_expert_used, 1.0f), GGML_TYPE_I32);
    iota = ggml_reshape_2d(ctx0, iota, n_expert_used, 1);

    return ggml_mul_mat_id(ctx0, compact, cur, iota);
}

size_t llama_moe_stream_pool_bytes(int * n_pools, long long * total_slots, double * hit_rate) {
    std::lock_guard<std::mutex> lk(g_mtx);
    size_t    bytes = 0;
    int       pools = 0;
    long long slots = 0;
    uint64_t  hits  = 0, miss = 0;
    for (const expert_pool * p : g_pools) {
        if (!p || !p->inited || p->failed) continue;
        pools += 1;
        bytes += p->total;
        slots += p->pool_slots;
        hits  += p->hits;
        miss  += p->misses;
    }
    if (n_pools)     *n_pools     = pools;
    if (total_slots) *total_slots = slots;
    if (hit_rate)    *hit_rate    = (hits + miss) ? 100.0 * (double) hits / (double) (hits + miss) : 0.0;
    return bytes;
}
