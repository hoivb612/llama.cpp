#include "llama-layer-window.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>

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
    layer_tensors[layer_idx].push_back({name, file_idx, offset, n_bytes, tensor});
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

bool layer_window_manager::ensure_layer_resident(int layer_idx) {
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
        // Non-mmap: copy data from file mapping into GPU buffer
        for (const auto & loc : it->second) {
            if (!loc.tensor) continue;
            if (loc.file_idx >= mmap_bases.size() || !mmap_bases[loc.file_idx]) continue;
            const uint8_t * data = mmap_bases[loc.file_idx] + loc.file_offset;
            ggml_backend_tensor_set(loc.tensor, data, 0, loc.n_bytes);
        }
    }

    entries[layer_idx].resident = true;
    entries[layer_idx].last_access = ++access_counter;
    resident_bytes += entries[layer_idx].memory_size;
    return true;
}

void layer_window_manager::evict_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= total_layers) return;
    if (!entries[layer_idx].resident) return;

    auto it = layer_tensors.find(layer_idx);
    if (it == layer_tensors.end()) return;  // initially loaded layer — don't evict

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
    // For non-mmap: we'd free the GPU buffer (Phase 5)

    entries[layer_idx].resident = false;
    if (resident_bytes >= entries[layer_idx].memory_size) {
        resident_bytes -= entries[layer_idx].memory_size;
    }
}

void layer_window_manager::evict_to_budget(int protected_layer) {
    if (budget_bytes == 0) return;

    while (resident_bytes > budget_bytes) {
        // Find LRU layer (oldest access, not protected)
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < total_layers; i++) {
            if (!entries[i].resident) continue;
            if (i == protected_layer) continue;
            // Don't evict layers that were loaded initially (no recorded tensors = no way to reload)
            if (layer_tensors.find(i) == layer_tensors.end()) continue;
            if (entries[i].last_access < oldest) {
                oldest = entries[i].last_access;
                victim = i;
            }
        }
        if (victim < 0) break;  // nothing evictable
        evict_layer(victim);
        evicts_this_pass++;
        bytes_evicted += entries[victim].memory_size;
    }
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
}

void layer_window_manager::end_pass() {
    if (loads_this_pass > 0 || evicts_this_pass > 0) {
        printf("layer_window: pass done — loaded %d layers (%.1f MiB), evicted %d layers (%.1f MiB)\n",
               loads_this_pass, bytes_loaded / (1024.0 * 1024.0),
               evicts_this_pass, bytes_evicted / (1024.0 * 1024.0));
        fflush(stdout);
    }
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

// Eval callback: called by the scheduler for each graph node
// When ask==true: we check if this node touches a new layer and ensure it's resident
// When ask==false: we observe (no action needed for windowing)
// Returns true to "observe" the node (causes sync), false to let it batch
bool layer_window_eval_callback(ggml_tensor * t, bool ask, void * user_data) {
    layer_window_manager * lwm = (layer_window_manager *)user_data;
    if (!lwm || lwm->budget_bytes == 0) return false;

    if (ask) {
        int layer = layer_window_manager::get_layer_from_tensor(t);
        if (layer >= 0 && layer != lwm->current_layer) {
            // New layer boundary — we need to intercept
            return true;
        }
        return false;  // same layer or non-layer node: batch together
    }

    // ask == false: the node just computed (or is about to in a synced sub-graph)
    int layer = layer_window_manager::get_layer_from_tensor(t);
    if (layer >= 0 && layer != lwm->current_layer) {
        // Transition to a new layer
        lwm->current_layer = layer;

        // Ensure this layer is resident
        if (lwm->ensure_layer_resident(layer)) {
            lwm->loads_this_pass++;
            lwm->bytes_loaded += lwm->entries[layer].memory_size;
        }

        // Evict layers to stay within budget
        lwm->evict_to_budget(layer);
    }

    return true;  // continue computation
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
    printf("\n");
    fflush(stdout);
}
