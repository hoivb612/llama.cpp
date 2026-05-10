#include "llama-layer-window.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <set>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

// Global instance pointer
static layer_window_manager * g_layer_window_mgr = nullptr;

layer_window_manager * llama_get_layer_window_manager() {
    return g_layer_window_mgr;
}

void llama_set_layer_window_manager(layer_window_manager * mgr) {
    g_layer_window_mgr = mgr;
}

void layer_window_manager::init(int n_layers, size_t budget_mb) {
    total_layers = n_layers;
    budget_bytes = budget_mb * 1024ULL * 1024ULL;  // 0 = unlimited
    resident_bytes = 0;
    access_counter = 0;
    all_resident = false;
    current_layer = -1;

    entries.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        entries[i].layer_idx   = i;
        entries[i].resident    = false;
        entries[i].repacked    = false;
        entries[i].memory_size = 0;
        entries[i].last_access = 0;
    }
}

void layer_window_manager::set_layer_size(int layer_idx, size_t bytes) {
    if (layer_idx >= 0 && layer_idx < total_layers) {
        entries[layer_idx].memory_size = bytes;
    }
}

bool layer_window_manager::should_load_layer(int layer_idx) const {
    if (budget_bytes == 0) {
        return true;  // unlimited — load everything
    }
    if (layer_idx < 0 || layer_idx >= total_layers) {
        return true;  // safety: always load unknown
    }

    // Strategy: load the first N layers that fit within the budget.
    size_t accum = 0;
    for (int i = 0; i < total_layers; i++) {
        if (accum + entries[i].memory_size > budget_bytes) {
            return layer_idx < i;
        }
        accum += entries[i].memory_size;
    }
    return true;  // all fit
}

void layer_window_manager::record_tensor_location(int layer_idx, const std::string & name,
                                                   uint16_t file_idx, size_t offset, size_t n_bytes,
                                                   ggml_tensor * tensor) {
    layer_tensors[layer_idx].push_back({name, file_idx, offset, n_bytes, tensor, 0});
}

// ---- Layer Windowing Diagnostic Environment Variables ----
//
// These env vars aid debugging of the DX12 reserved-resource layer windowing
// system. Each isolates a different stage of the evict/reload pipeline so
// failures can be attributed to a single mechanism. All are OFF by default
// (zero cost — checked once via getenv on first call).
//
//   GGML_LW_DIAG       Enable diagnostic logging + mmap checksum verification.
//                       Computes XOR-rotate checksums (first+last 4 KB) of every
//                       tensor after mmap and re-verifies before each GPU upload.
//                       Use to rule out host-side data corruption.
//
//   GGML_LW_KEEP_MMAP  Skip DiscardVirtualMemory on mmap pages after GPU upload.
//                       Normally pages are released to reduce working-set pressure.
//                       Use to rule out OS page reclaim corrupting source data.
//
//   GGML_LW_NO_EVICT   Skip ALL eviction (tile decommit AND bookkeeping).
//                       Layers loaded once stay resident forever. Useful to confirm
//                       whether a bug is in the eviction/reload path vs. elsewhere.
//                       Will over-commit VRAM if budget < model size.
//
//   GGML_LW_SOFT_EVICT Update bookkeeping (mark non-resident, free byte budget)
//                       but skip the actual GPU tile unmap (UpdateTileMappings).
//                       Subgraph boundaries are still created.  Use to isolate
//                       tile-unmap faults from subgraph-boundary faults.
//
//   GGML_LW_SOFT_NOUP  (Requires SOFT_EVICT) Skip CopyBufferRegion re-uploads
//                       for layers that have already been loaded once. First load
//                       still happens.  Combined with SOFT_EVICT this means the
//                       ONLY effect of eviction is the subgraph boundary itself —
//                       no GPU memory operations at all.  Use to isolate whether
//                       graph splitting alone causes incorrect output.
//
static bool lw_diag_enabled() {
    static int enabled = -1;
    if (enabled < 0) enabled = (getenv("GGML_LW_DIAG") != nullptr) ? 1 : 0;
    return enabled != 0;
}

// Compute reference checksums for all layer tensors (call after mmap_bases is set)
static uint64_t lw_checksum(const uint8_t * data, size_t n_bytes) {
    uint64_t cksum = 0;
    size_t check_len = (n_bytes < 4096) ? n_bytes : 4096;
    for (size_t i = 0; i < check_len; i++) {
        cksum = (cksum << 1) | (cksum >> 63);
        cksum ^= data[i];
    }
    if (n_bytes > 4096) {
        const uint8_t * tail = data + n_bytes - 4096;
        for (size_t i = 0; i < 4096; i++) {
            cksum = (cksum << 1) | (cksum >> 63);
            cksum ^= tail[i];
        }
    }
    return cksum;
}

void layer_window_manager::compute_reference_checksums() {
    if (mmap_bases.empty()) return;
    for (auto & [layer_idx, locs] : layer_tensors) {
        for (auto & loc : locs) {
            if (loc.file_idx >= mmap_bases.size() || !mmap_bases[loc.file_idx]) continue;
            const uint8_t * data = mmap_bases[loc.file_idx] + loc.file_offset;
            loc.checksum = lw_checksum(data, loc.n_bytes);
        }
    }
}

static bool lw_keep_mmap() {
    static int keep = -1;
    if (keep < 0) keep = (getenv("GGML_LW_KEEP_MMAP") != nullptr) ? 1 : 0;
    return keep != 0;
}

static bool lw_no_evict() {
    static int no = -1;
    if (no < 0) no = (getenv("GGML_LW_NO_EVICT") != nullptr) ? 1 : 0;
    return no != 0;
}

static bool lw_soft_evict() {
    static int soft = -1;
    if (soft < 0) soft = (getenv("GGML_LW_SOFT_EVICT") != nullptr) ? 1 : 0;
    return soft != 0;
}

int layer_window_manager::get_initial_resident_count() const {
    if (budget_bytes == 0) {
        return total_layers;
    }
    size_t accum = 0;
    int count = 0;
    for (int i = 0; i < total_layers; i++) {
        if (accum + entries[i].memory_size > budget_bytes) {
            break;
        }
        accum += entries[i].memory_size;
        count++;
    }
    return count;
}

bool layer_window_manager::ensure_layer_resident(int layer_idx, bool allow_evict) {
    if (layer_idx < 0 || layer_idx >= total_layers) return false;
    if (entries[layer_idx].resident) {
        entries[layer_idx].last_access = ++access_counter;
        return false;  // already resident
    }

    auto it = layer_tensors.find(layer_idx);
    if (it == layer_tensors.end()) {
        // No recorded tensors — layer was loaded initially, mark resident
        entries[layer_idx].resident = true;
        entries[layer_idx].last_access = ++access_counter;
        return false;
    }

    // Evict BEFORE committing tiles — make room for the incoming layer
    if (allow_evict) {
        evict_to_budget(layer_idx);
    }

    // For mmap: the data pointer is already valid (points into mmap).
    // We just need to "touch" the pages to ensure they're paged in.
    // On Windows: VirtualLock pins pages in physical RAM.
    // On Linux: madvise(WILLNEED) + mlock.
    if (use_mmap) {
        for (const auto & loc : it->second) {
            if (!loc.tensor || !loc.tensor->data) continue;
#ifdef _WIN32
            VirtualLock(loc.tensor->data, loc.n_bytes);
#else
            madvise(loc.tensor->data, loc.n_bytes, MADV_WILLNEED);
            mlock(loc.tensor->data, loc.n_bytes);
#endif
        }
    } else {
        // SOFT_EVICT + SOFT_NOUP: skip RE-upload — tiles still have correct data.
        // Only skip when layer was previously loaded (first load must always happen).
        // This isolates whether the bug is in CopyBufferRegion or subgraph boundaries.
        static std::set<int> layers_loaded_once;
        bool skip_upload = lw_soft_evict() && (getenv("GGML_LW_SOFT_NOUP") != nullptr)
                           && layers_loaded_once.count(layer_idx);
        if (skip_upload) {
            if (lw_diag_enabled()) {
                fprintf(stderr, "LW: SOFT_NOUP layer=%d — skipping upload (tiles retained)\n", layer_idx);
            }
        } else {
            // Non-mmap: batch upload all tensors for this layer to GPU in one submission
            std::vector<ggml_tensor *> batch_tensors;
            std::vector<const void *>  batch_data;
            std::vector<size_t>        batch_sizes;
            for (const auto & loc : it->second) {
                if (!loc.tensor) continue;
                if (loc.file_idx >= mmap_bases.size() || !mmap_bases[loc.file_idx]) continue;

                // Verify mmap data integrity before upload (GGML_LW_DIAG)
                if (lw_diag_enabled() && loc.checksum != 0) {
                    const uint8_t * data = mmap_bases[loc.file_idx] + loc.file_offset;
                    uint64_t cur_cksum = lw_checksum(data, loc.n_bytes);
                    if (cur_cksum != loc.checksum) {
                        fprintf(stderr, "LW: CHECKSUM MISMATCH layer=%d tensor=%s "
                                "expected=%016llx got=%016llx (mmap data corrupted!)\n",
                                layer_idx, loc.name.c_str(),
                                (unsigned long long)loc.checksum, (unsigned long long)cur_cksum);
                    } else if (total_passes < 2) {
                        fprintf(stderr, "LW: checksum OK layer=%d tensor=%s (%016llx)\n",
                                layer_idx, loc.name.c_str(), (unsigned long long)cur_cksum);
                    }
                }

                batch_tensors.push_back(loc.tensor);
                batch_data.push_back(mmap_bases[loc.file_idx] + loc.file_offset);
                batch_sizes.push_back(loc.n_bytes);
            }
            if (!batch_tensors.empty()) {
                ggml_backend_batch_tensor_set(
                    batch_tensors.data(), batch_data.data(), batch_sizes.data(), (int)batch_tensors.size());
            }
#ifdef _WIN32
            // Release mmap pages after copy (unless GGML_LW_KEEP_MMAP is set)
            if (!lw_keep_mmap()) {
                for (const auto & loc : it->second) {
                    if (!loc.tensor) continue;
                    if (loc.file_idx >= mmap_bases.size() || !mmap_bases[loc.file_idx]) continue;
                    const uint8_t * data = mmap_bases[loc.file_idx] + loc.file_offset;
                    DiscardVirtualMemory((void *)data, loc.n_bytes);
                }
            }
#endif
        }
        layers_loaded_once.insert(layer_idx);
    }

    entries[layer_idx].resident = true;
    entries[layer_idx].last_access = ++access_counter;
    resident_bytes += entries[layer_idx].memory_size;
    return true;
}

void layer_window_manager::evict_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= total_layers) return;
    if (!entries[layer_idx].resident) return;
    if (lw_no_evict()) return;  // skip everything — tiles stay committed

    auto it = layer_tensors.find(layer_idx);
    if (it == layer_tensors.end()) return;  // no recorded tensors — can't reload

    // For mmap: release physical pages back to OS
    if (use_mmap) {
        for (const auto & loc : it->second) {
            if (!loc.tensor || !loc.tensor->data) continue;
#ifdef _WIN32
            VirtualUnlock(loc.tensor->data, loc.n_bytes);
            DiscardVirtualMemory(loc.tensor->data, loc.n_bytes);
#else
            munlock(loc.tensor->data, loc.n_bytes);
            madvise(loc.tensor->data, loc.n_bytes, MADV_DONTNEED);
#endif
        }
    }
    // For non-mmap: decommit GPU tiles (reserved resource) to free physical memory
    // SOFT_EVICT: skip the GPU unmap but still do bookkeeping — isolates tile unmap bugs
    if (!use_mmap && !lw_soft_evict()) {
        for (const auto & loc : it->second) {
            if (!loc.tensor) continue;
            ggml_backend_tensor_decommit(loc.tensor);
        }
    }

    entries[layer_idx].resident = false;
    if (resident_bytes >= entries[layer_idx].memory_size) {
        resident_bytes -= entries[layer_idx].memory_size;
    }
}

void layer_window_manager::evict_to_budget(int protected_layer) {
    if (budget_bytes == 0) return;
    if (lw_no_evict()) return;  // diagnostic: skip all eviction

    // Target: ensure room for the incoming layer (protected_layer).
    // resident_bytes tracks only LAYER memory, but the backing heap also holds
    // non-layer data (embed/output). We must keep resident_bytes low enough that
    // resident + incoming_layer + non_layer fits in the heap.
    size_t incoming = (protected_layer >= 0 && protected_layer < total_layers &&
                       !entries[protected_layer].resident)
                    ? entries[protected_layer].memory_size : 0;
    size_t target = (budget_bytes > incoming) ? budget_bytes - incoming : 0;

    while (resident_bytes > target) {
        // MRU eviction: for sequential access (0→N), evict the most recently used
        // layer (excluding current). This is provably optimal for sequential scan —
        // the just-finished layer won't be needed until the next full pass.
        int victim = -1;
        uint64_t newest = 0;
        for (int i = 0; i < total_layers; i++) {
            if (!entries[i].resident) continue;
            if (i == protected_layer) continue;
            if (layer_tensors.find(i) == layer_tensors.end()) continue;
            if (entries[i].last_access > newest) {
                newest = entries[i].last_access;
                victim = i;
            }
        }
        if (victim < 0) break;  // nothing evictable
        if (lw_diag_enabled()) {
            fprintf(stderr, "LW: evict %d (%.1f MiB, access=%llu) protect=%d resident=%.1f MiB target=%.1f MiB\n",
                    victim, entries[victim].memory_size / (1024.0 * 1024.0),
                    (unsigned long long)entries[victim].last_access,
                    protected_layer, resident_bytes / (1024.0 * 1024.0), target / (1024.0 * 1024.0));
        }
        evict_layer(victim);
        evicts_this_pass++;
        bytes_evicted += entries[victim].memory_size;
    }
}

// Eval callback: called by the scheduler for each graph node
// When ask==true: we detect layer transitions, ensure the new layer is loaded,
//   and return true to create a subgraph boundary (causes sync after compute).
// When ask==false: runs AFTER compute+sync — safe to evict old layers (GPU is done).
//
// CRITICAL: eviction must ONLY happen in ask=false (post-compute).
// The scheduler batches all ask=false nodes into one subgraph with the ask=true node.
// Evicting in ask=true would decommit tiles for layers still in the pending subgraph.
bool layer_window_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    layer_window_manager * lwm = (layer_window_manager *)user_data;
    if (!lwm || lwm->budget_bytes == 0) return false;

    int layer = layer_window_manager::get_layer_from_tensor(t);

    if (ask) {
        if (layer >= 0 && layer != lwm->current_layer) {
            int prev = lwm->current_layer;
            lwm->current_layer = layer;
            // Load WITHOUT eviction — heap headroom accommodates one extra layer.
            // Eviction is deferred to ask=false after compute completes.
            if (lwm->ensure_layer_resident(layer, /*allow_evict=*/false)) {
                lwm->loads_this_pass++;
                lwm->bytes_loaded += lwm->entries[layer].memory_size;
                if (lw_diag_enabled()) {
                    fprintf(stderr, "LW: ask=1 %d->%d LOAD (%.1f MiB, %zu tensors) trigger=%s\n",
                            prev, layer, lwm->entries[layer].memory_size / (1024.0 * 1024.0),
                            lwm->layer_tensors.count(layer) ? lwm->layer_tensors[layer].size() : 0,
                            t->name);
                }
                return true;  // data was loaded — sync after compute
            }
            if (lw_diag_enabled() && lwm->total_passes < 2) {
                // Log transitions in first pass only (avoid flooding)
                fprintf(stderr, "LW: ask=1 %d->%d resident trigger=%s\n", prev, layer, t->name);
            }
            return false;  // already resident — no sync needed, batch together
        }
        return false;  // same layer or non-layer node: batch together
    }

    // ask == false: subgraph fully computed + synced — safe to evict
    if (layer >= 0) {
        lwm->evict_to_budget(layer);
    }

    return true;  // continue computation
}

void layer_window_manager::ensure_all_layers_resident() {
    if (all_resident) return;
    if (budget_bytes == 0) { all_resident = true; return; }

    int loaded = 0;
    size_t loaded_bytes = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < total_layers; i++) {
        if (ensure_layer_resident(i)) {
            loaded++;
            loaded_bytes += entries[i].memory_size;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (loaded > 0) {
        printf("layer_window: loaded %d deferred layers (%.1f MiB) in %.1f ms (%.1f GB/s)\n",
               loaded, loaded_bytes / (1024.0 * 1024.0), ms,
               (loaded_bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0));
        fflush(stdout);
    }

    all_resident = true;
}

void layer_window_manager::begin_pass() {
    current_layer = -1;
    loads_this_pass = 0;
    evicts_this_pass = 0;
    bytes_loaded = 0;
    bytes_evicted = 0;
    if (lw_diag_enabled()) {
        fprintf(stderr, "LW: === begin pass %d ===\n", total_passes + 1);
    }
}

void layer_window_manager::end_pass() {
    if (loads_this_pass > 0 || evicts_this_pass > 0) {
        total_loads  += loads_this_pass;
        total_evicts += evicts_this_pass;
        total_bytes_loaded  += bytes_loaded;
        total_bytes_evicted += bytes_evicted;
        total_passes++;
    }
}

void layer_window_manager::print_stats() const {
    if (budget_bytes > 0) {
        if (total_passes == 0) {
            printf("layer_window: no windowing passes (all layers fit in budget)\n");
        } else {
            printf("layer_window: %d passes, %d loads (%.1f MiB), %d evictions (%.1f MiB)",
                   total_passes, total_loads, total_bytes_loaded / (1024.0 * 1024.0),
                   total_evicts, total_bytes_evicted / (1024.0 * 1024.0));
            uint32_t overflow = ggml_backend_get_heap_overflow_count();
            if (overflow > 0) {
                printf(", %u heap overflows", overflow);
            }
            printf("\n");
        }
    }
    fflush(stdout);
}

int layer_window_manager::get_layer_from_tensor(const ggml_tensor * t) {
    if (!t) return -1;

    // Check all source tensors for a blk.N pattern
    for (int s = 0; s < GGML_MAX_SRC; s++) {
        const ggml_tensor * src = t->src[s];
        if (!src) continue;
        const char * name = src->name;
        if (strncmp(name, "blk.", 4) == 0) {
            return atoi(name + 4);
        }
    }

    // Also check the tensor itself
    if (strncmp(t->name, "blk.", 4) == 0) {
        return atoi(t->name + 4);
    }

    return -1;
}

void layer_window_manager::mark_initially_resident() {
    if (budget_bytes == 0) return;
    for (int i = 0; i < total_layers; i++) {
        if (should_load_layer(i)) {
            entries[i].resident    = true;
            entries[i].last_access = ++access_counter;
            resident_bytes += entries[i].memory_size;
        }
    }
}

void layer_window_manager::release_mmap_pages() {
    if (budget_bytes == 0 || use_mmap) return;
    if (mmap_bases.empty()) return;
    if (lw_keep_mmap()) {
        printf("layer_window: GGML_LW_KEEP_MMAP set — skipping mmap page release\n");
        fflush(stdout);
        return;
    }

    size_t released = 0;
    for (const auto & [layer_idx, locs] : layer_tensors) {
        for (const auto & loc : locs) {
            if (loc.file_idx >= mmap_bases.size() || !mmap_bases[loc.file_idx]) continue;
            const uint8_t * data = mmap_bases[loc.file_idx] + loc.file_offset;
#ifdef _WIN32
            DiscardVirtualMemory((void *)data, loc.n_bytes);
#else
            madvise((void *)data, loc.n_bytes, MADV_DONTNEED);
#endif
            released += loc.n_bytes;
        }
    }
    if (released > 0) {
        printf("layer_window: released %.1f MiB of mmap pages after initial load\n",
               released / (1024.0 * 1024.0));
        fflush(stdout);
    }
}

void layer_window_manager::print_config() const {
    if (budget_bytes == 0) {
        printf("layer_window: disabled (all %d layers resident)\n", total_layers);
        fflush(stdout);
        return;
    }

    int resident = get_initial_resident_count();
    size_t resident_total = 0;
    for (int i = 0; i < resident; i++) {
        resident_total += entries[i].memory_size;
    }

    printf("\nlayer_window: budget = %zu MiB, initial resident = %d of %d layers (%.1f MiB)\n",
           budget_bytes / (1024 * 1024), resident, total_layers,
           resident_total / (1024.0 * 1024.0));
    printf("layer_window: non-layer (always resident) = %.1f MiB\n",
           non_layer_bytes / (1024.0 * 1024.0));
    printf("layer_window: total initial memory = %.1f MiB (non-layer + %d layers)\n",
           (non_layer_bytes + resident_total) / (1024.0 * 1024.0), resident);

    if (resident < total_layers) {
        printf("layer_window: %d layers deferred (will stream on demand during compute)\n",
               total_layers - resident);
    }

    // Diagnostic: dump per-layer tensor counts and sizes
    if (lw_diag_enabled()) {
        fprintf(stderr, "LW: recorded tensors per layer:\n");
        for (int i = 0; i < total_layers; i++) {
            auto it = layer_tensors.find(i);
            size_t cnt = it != layer_tensors.end() ? it->second.size() : 0;
            fprintf(stderr, "  blk.%d: %zu tensors, size=%.1f MiB, resident=%d",
                    i, cnt, entries[i].memory_size / (1024.0 * 1024.0), entries[i].resident);
            if (it != layer_tensors.end() && !it->second.empty()) {
                size_t recorded_total = 0;
                for (const auto & loc : it->second) recorded_total += loc.n_bytes;
                if (recorded_total != entries[i].memory_size) {
                    fprintf(stderr, " MISMATCH(recorded=%.1f MiB)", recorded_total / (1024.0 * 1024.0));
                }
            }
            fprintf(stderr, "\n");
        }
    }

    printf("\n");
    fflush(stdout);
}
