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

void layer_window_manager::set_source_file(uint16_t file_idx, const std::string & path) {
    if (src_file_paths.size() <= file_idx) {
        src_file_paths.resize(file_idx + 1);
        src_file_handles.resize(file_idx + 1, nullptr);
    }
    src_file_paths[file_idx] = path;
}

bool layer_window_manager::read_tensor_bytes(uint16_t file_idx, size_t file_offset, size_t n_bytes, void * dst) {
    if (file_idx >= src_file_paths.size() || src_file_paths[file_idx].empty()) {
        return false;
    }
    FILE * f = (FILE *) src_file_handles[file_idx];
    if (!f) {
        f = fopen(src_file_paths[file_idx].c_str(), "rb");
        if (!f) {
            fprintf(stderr, "LW: failed to open source file '%s' for streaming\n",
                    src_file_paths[file_idx].c_str());
            return false;
        }
        src_file_handles[file_idx] = f;
    }
#ifdef _WIN32
    if (_fseeki64(f, (long long) file_offset, SEEK_SET) != 0) return false;
#else
    if (fseeko(f, (off_t) file_offset, SEEK_SET) != 0) return false;
#endif
    return fread(dst, 1, n_bytes, f) == n_bytes;
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

    // Stage 2b-2: aliased-stream mode — build this layer's imported anonymous
    // buffer on demand (re-pointing its tensors off the dummy buffer). Eviction
    // is handled separately (deferred to ask=false) so allow_evict mirrors the
    // non-aliased path: build here, free evicted layers post-compute.
    if (aliased_streaming) {
        if (allow_evict) {
            evict_to_budget(layer_idx);
        }
        if (!build_layer_aliased(layer_idx)) {
            fprintf(stderr, "LW-ALIAS: failed to build layer %d on demand\n", layer_idx);
            return false;
        }
        entries[layer_idx].resident = true;
        entries[layer_idx].last_access = ++access_counter;
        resident_bytes += entries[layer_idx].memory_size;
        return true;
    }

    // Evict BEFORE committing tiles — make room for the incoming layer
    if (allow_evict) {
        evict_to_budget(layer_idx);
    }

    // Determine where this layer's weights physically live.
    // A deferred layer must be re-populated on residency UNLESS its tensors are
    // aliased directly onto the mmap (buffer_from_host_ptr: CPU, DX12-UMA) -- in
    // which case paging the mmap back in is enough. is_host() is NOT a reliable
    // test: a Vulkan/CUDA UMA buffer can be DEVICE_LOCAL|HOST_VISIBLE (reports
    // host) yet still be a SEPARATE allocation the loader copied weights into,
    // which the windowing skip-load left unfilled -> garbage. Detect the true
    // alias by comparing the tensor data pointer to its mmap source address.
    bool layer_needs_upload = false;
    for (const auto & loc : it->second) {
        if (!loc.tensor) continue;
        const uint8_t * mmap_src =
            (loc.file_idx < mmap_bases.size() && mmap_bases[loc.file_idx])
            ? mmap_bases[loc.file_idx] + loc.file_offset : nullptr;
        if ((const void *) loc.tensor->data != (const void *) mmap_src) {
            layer_needs_upload = true;
            break;
        }
    }
    {
        // One-shot decisive diagnostic: dump the actual residency state for the
        // first deferred layer we're asked to make resident. This disambiguates
        // the Vulkan garbage-output bug (upload vs alias vs not-recorded).
        static bool dumped = false;
        if (!dumped && lw_diag_enabled()) {
            dumped = true;
            const tensor_location * l0 = nullptr;
            for (const auto & loc : it->second) { if (loc.tensor) { l0 = &loc; break; } }
            const uint8_t * mmap_src0 = (l0 && l0->file_idx < mmap_bases.size() && mmap_bases[l0->file_idx])
                ? mmap_bases[l0->file_idx] + l0->file_offset : nullptr;
            ggml_backend_buffer_t buf = l0 && l0->tensor ? l0->tensor->buffer : nullptr;
            fprintf(stderr,
                "LW-DIAG: first deferred layer=%d n_tensors=%zu use_mmap=%d mmap_bases=%zu\n"
                "         needs_upload=%d tensor0=%s data=%p mmap_src=%p equal=%d\n"
                "         buffer=%p is_host=%d buft=%s\n",
                layer_idx, it->second.size(), (int)use_mmap, mmap_bases.size(),
                (int)layer_needs_upload,
                l0 ? l0->name.c_str() : "(none)",
                l0 && l0->tensor ? l0->tensor->data : nullptr, (const void *)mmap_src0,
                (int)(l0 && l0->tensor && (const void*)l0->tensor->data == (const void*)mmap_src0),
                (void *)buf,
                buf ? (int)ggml_backend_buffer_is_host(buf) : -1,
                buf ? ggml_backend_buffer_name(buf) : "(none)");
            fflush(stderr);
        }
    }
    if (use_mmap && layer_needs_upload) {
        static bool announced = false;
        if (!announced) {
            announced = true;
            printf("layer_window: weights are NOT mmap-aliased (separate device/host-visible "
                   "buffer) — streaming deferred layers via upload (mmap = byte source)\n");
        }
    }
    // For mmap-aliased weights: the data pointer already points into the mmap.
    // We just "touch" the pages to ensure they're paged in.
    // On Windows: VirtualLock pins pages in physical RAM.
    // On Linux: madvise(WILLNEED) + mlock.
    if (use_mmap && !layer_needs_upload) {
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
            // Non-mmap: batch upload all tensors for this layer to GPU in one submission.
            // Byte source is the mmap (when aliased/available) OR an on-demand file read
            // (direct_io / GPU offload disables mmap — the layer bytes are streamed from
            // the GGUF at the recorded file_offset). File-read tensors are uploaded
            // immediately (their scratch buffer is transient); mmap tensors are batched.
            std::vector<ggml_tensor *> batch_tensors;
            std::vector<const void *>  batch_data;
            std::vector<size_t>        batch_sizes;
            for (const auto & loc : it->second) {
                if (!loc.tensor) continue;
                const bool have_mmap =
                    loc.file_idx < mmap_bases.size() && mmap_bases[loc.file_idx];
                if (!have_mmap) {
                    // Stream this tensor's bytes from the backing GGUF file.
                    if (reload_scratch.size() < loc.n_bytes) reload_scratch.resize(loc.n_bytes);
                    if (!read_tensor_bytes(loc.file_idx, loc.file_offset, loc.n_bytes, reload_scratch.data())) {
                        fprintf(stderr, "LW: FAILED to stream layer=%d tensor=%s (%zu bytes @ off=%zu file=%u)\n",
                                layer_idx, loc.name.c_str(), loc.n_bytes, loc.file_offset, (unsigned)loc.file_idx);
                        continue;
                    }
                    ggml_backend_tensor_set(loc.tensor, reload_scratch.data(), 0, loc.n_bytes);
                    continue;
                }

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

    // Stage 2b-2: aliased-stream mode — free this layer's imported anonymous
    // buffer to release its RAM. Re-point its tensors at the dummy 0-size buffer
    // first so they keep a valid buft (graph_reserve/supports_buft read buft, not
    // data); they are never dereferenced while the layer is non-resident.
    if (aliased_streaming) {
        for (auto & loc : it->second) {
            if (!loc.tensor) continue;
            loc.tensor->buffer = (ggml_backend_buffer_t) alias_dummy_buf;
            loc.tensor->data   = nullptr;
        }
        free_layer_aliased(layer_idx);
        entries[layer_idx].resident = false;
        if (resident_bytes >= entries[layer_idx].memory_size) {
            resident_bytes -= entries[layer_idx].memory_size;
        }
        return;
    }

    // Mirror ensure_layer_resident: mmap-aliased weights freed their pages via
    // the mmap release below; separate device/host-visible buffers (Vulkan/CUDA,
    // incl. under mmap) need their tiles decommitted to free physical memory.
    bool layer_needs_upload = false;
    for (const auto & loc : it->second) {
        if (!loc.tensor) continue;
        const uint8_t * mmap_src =
            (loc.file_idx < mmap_bases.size() && mmap_bases[loc.file_idx])
            ? mmap_bases[loc.file_idx] + loc.file_offset : nullptr;
        if ((const void *) loc.tensor->data != (const void *) mmap_src) {
            layer_needs_upload = true;
            break;
        }
    }

    // For mmap-aliased weights: release physical pages back to OS.
    // (Non-aliased/device-buffer layers already discarded their mmap source
    // pages right after the upload in ensure_layer_resident, so skip them here.)
    if (use_mmap && !layer_needs_upload) {
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
    // For separate device/host-visible buffers (Vulkan/CUDA, including under
    // mmap): decommit tiles to free physical memory (no-op if the backend has
    // no decommit fn registered).
    // SOFT_EVICT: skip the GPU unmap but still do bookkeeping — isolates tile unmap bugs
    if (layer_needs_upload && !lw_soft_evict()) {
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
    // Stage 1: aliased cache keeps every layer resident in imported anonymous
    // buffers, so compute reads them directly — no per-token load/evict. In
    // Stage 2b-2 streaming mode the cache is partial, so DON'T early-return:
    // fall through to on-demand build/evict.
    if (lwm->aliased_cache && !lwm->aliased_streaming) return false;

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

// Stage 1: convert each layer's weights into a per-layer ANONYMOUS host buffer
// that is imported (zero-copy) into the backend, then re-point the layer's
// tensors at the imported buffer. Used when the backend cannot alias the
// file-backed mmap directly and has no sparse residency (AMD Vulkan/Windows).
bool layer_window_manager::convert_layers_to_aliased_cache() {
#ifdef _WIN32
    if (budget_bytes == 0 || layer_tensors.empty()) return false;

    // Stage 2b-2: in aliased-stream load mode with a real budget that doesn't fit
    // every layer, build ONLY the initially-resident (should_load) layers and leave
    // the rest on the dummy buffer to be streamed on demand. Capture the dummy
    // buffer so evicted layers can be re-pointed at it (valid buft for graph_reserve).
    bool streaming = false;
    if (aliased_load_mode) {
        for (auto & [layer_idx, locs] : layer_tensors) {
            for (auto & loc : locs) {
                if (loc.tensor && loc.tensor->buffer) { alias_dummy_buf = (void *) loc.tensor->buffer; break; }
            }
            if (alias_dummy_buf) break;
        }
        int fit = get_initial_resident_count();
        streaming = (fit < total_layers);
    }

    size_t converted_layers = 0;
    for (auto & [layer_idx, locs] : layer_tensors) {
        if (locs.empty()) continue;
        if (streaming && !should_load_layer(layer_idx)) continue;  // deferred: stream later
        if (build_layer_aliased(layer_idx)) converted_layers++;
    }

    if (converted_layers > 0) {
        aliased_cache     = true;
        aliased_streaming = streaming;
        if (streaming) {
            printf("layer_window: aliased-cache active (streaming) — %zu of %d layers built into "
                   "imported anonymous buffers; %d deferred (stream on demand)\n",
                   converted_layers, total_layers, total_layers - (int)converted_layers);
        } else {
            printf("layer_window: aliased-cache active — %zu layers converted to imported "
                   "anonymous buffers, zero-copy compute\n", converted_layers);
        }
        fflush(stdout);
    }
    return aliased_cache;
#else
    return false;
#endif
}

// Build (VirtualAlloc + fill from mmap/file + import + re-point) one layer's
// aliased buffer. Idempotent-safe: if the layer already has an aliased buffer,
// returns true without rebuilding.
bool layer_window_manager::build_layer_aliased(int layer_idx) {
#ifdef _WIN32
    if (layer_alias_buf.count(layer_idx)) return true;  // already built

    auto it = layer_tensors.find(layer_idx);
    if (it == layer_tensors.end() || it->second.empty()) return false;
    auto & locs = it->second;

    const unsigned long MEM_COMMIT_  = 0x00001000;
    const unsigned long MEM_RESERVE_ = 0x00002000;
    const unsigned long PAGE_RW_     = 0x04;
    const unsigned long MEM_RELEASE_ = 0x8000;

    // Derive backend device + alignment from the first tensor's buffer.
    ggml_tensor * t0 = nullptr;
    for (auto & loc : locs) { if (loc.tensor && loc.tensor->buffer) { t0 = loc.tensor; break; } }
    if (!t0) return false;

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t0->buffer);
    ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
    if (!dev) return false;
    size_t align = ggml_backend_buft_get_alignment(buft);
    if (align == 0) align = 256;

    // Pack tensors densely with buft alignment; compute packed offsets.
    std::vector<size_t> packed_off(locs.size(), 0);
    size_t packed = 0;
    size_t max_tsize = 0;
    for (size_t i = 0; i < locs.size(); i++) {
        if (!locs[i].tensor) continue;
        packed = (packed + align - 1) & ~(align - 1);
        packed_off[i] = packed;
        packed += locs[i].n_bytes;
        if (locs[i].n_bytes > max_tsize) max_tsize = locs[i].n_bytes;
    }
    const size_t total = (packed + align - 1) & ~(align - 1);
    if (total == 0) return false;

    // Anonymous committed host memory (64K-aligned base — safe to import).
    void * anon = VirtualAlloc(nullptr, total, MEM_COMMIT_ | MEM_RESERVE_, PAGE_RW_);
    if (!anon) {
        fprintf(stderr, "LW-ALIAS: VirtualAlloc(%zu) failed for layer %d\n", total, layer_idx);
        return false;
    }

    // Fill from mmap (preferred) or by streaming from the GGUF file.
    bool fill_ok = true;
    for (size_t i = 0; i < locs.size(); i++) {
        auto & loc = locs[i];
        if (!loc.tensor) continue;
        void * dst = (uint8_t *) anon + packed_off[i];
        const bool have_mmap = loc.file_idx < mmap_bases.size() && mmap_bases[loc.file_idx];
        if (have_mmap) {
            memcpy(dst, mmap_bases[loc.file_idx] + loc.file_offset, loc.n_bytes);
        } else if (!read_tensor_bytes(loc.file_idx, loc.file_offset, loc.n_bytes, dst)) {
            fprintf(stderr, "LW-ALIAS: failed to fill layer %d tensor %s\n", layer_idx, loc.name.c_str());
            fill_ok = false;
            break;
        }
    }
    if (!fill_ok) { VirtualFree(anon, 0, MEM_RELEASE_); return false; }

    // Import the anonymous buffer into the backend (zero-copy alias).
    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, anon, total, max_tsize);
    if (!buf) {
        fprintf(stderr, "LW-ALIAS: buffer_from_host_ptr failed for layer %d (%zu bytes)\n", layer_idx, total);
        VirtualFree(anon, 0, MEM_RELEASE_);
        return false;
    }

    // Re-point the layer's weight tensors at the imported buffer.
    for (size_t i = 0; i < locs.size(); i++) {
        if (!locs[i].tensor) continue;
        locs[i].tensor->buffer = buf;
        locs[i].tensor->data   = (uint8_t *) anon + packed_off[i];
    }

    layer_alias_anon[layer_idx] = anon;
    layer_alias_buf[layer_idx]  = (void *) buf;
    return true;
#else
    (void) layer_idx;
    return false;
#endif
}

// Free one layer's aliased buffer: destroy the imported backend buffer, then
// release its anonymous host memory. The layer's tensors keep stale pointers —
// they must not be accessed until the layer is rebuilt (windowing guarantees a
// layer's ops only run while it is resident).
void layer_window_manager::free_layer_aliased(int layer_idx) {
#ifdef _WIN32
    auto bit = layer_alias_buf.find(layer_idx);
    if (bit != layer_alias_buf.end()) {
        ggml_backend_buffer_free((ggml_backend_buffer_t) bit->second);
        layer_alias_buf.erase(bit);
    }
    auto ait = layer_alias_anon.find(layer_idx);
    if (ait != layer_alias_anon.end()) {
        VirtualFree(ait->second, 0, 0x8000 /*MEM_RELEASE*/);
        layer_alias_anon.erase(ait);
    }
#else
    (void) layer_idx;
#endif
}


// Stage 2a: migrate one residual GPU-resident weight (non-blk.* tensor sharing the
// big layer buffer, e.g. output.weight) into its own imported anonymous buffer.
bool layer_window_manager::migrate_residual_tensor(ggml_tensor * t) {
#ifdef _WIN32
    if (!t || !t->buffer) return false;

    const unsigned long MEM_COMMIT_  = 0x00001000;
    const unsigned long MEM_RESERVE_ = 0x00002000;
    const unsigned long PAGE_RW_     = 0x04;
    const unsigned long MEM_RELEASE_ = 0x8000;

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
    if (!dev) return false;
    size_t align = ggml_backend_buft_get_alignment(buft);
    if (align == 0) align = 256;

    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) return false;
    const size_t total = (nbytes + align - 1) & ~(align - 1);

    void * anon = VirtualAlloc(nullptr, total, MEM_COMMIT_ | MEM_RESERVE_, PAGE_RW_);
    if (!anon) {
        fprintf(stderr, "LW-ALIAS: VirtualAlloc(%zu) failed migrating residual %s\n", total, t->name);
        return false;
    }

    // Pull the tensor's current bytes back from the device into host memory.
    ggml_backend_tensor_get(t, anon, 0, nbytes);

    ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, anon, total, nbytes);
    if (!buf) {
        fprintf(stderr, "LW-ALIAS: buffer_from_host_ptr failed migrating residual %s (%zu bytes)\n",
                t->name, total);
        VirtualFree(anon, 0, MEM_RELEASE_);
        return false;
    }

    t->buffer = buf;
    t->data   = anon;

    alias_anon.push_back(anon);
    alias_bufs.push_back((void *) buf);
    return true;
#else
    (void) t;
    return false;
#endif
}

// ---- Stage 2b: load-time aliased streaming ----

bool layer_window_manager::alias_stream_enabled() {
    static const int v = (getenv("GGML_LW_ALIAS_STREAM") != nullptr) ? 1 : 0;
    return v != 0;
}

bool layer_window_manager::aliased_load_pending = false;

void layer_window_manager::record_non_layer_location(const std::string & name, uint16_t file_idx,
                                                     size_t offset, size_t n_bytes, ggml_tensor * tensor) {
    non_layer_locs.push_back({name, file_idx, offset, n_bytes, tensor, 0});
}

// Build imported anonymous buffers for every recorded non-layer GPU weight
// (e.g. output.weight), filling from mmap/file. Each tensor gets its own buffer.
// Returns the number successfully built.
int layer_window_manager::build_non_layer_aliased() {
#ifdef _WIN32
    const unsigned long MEM_COMMIT_  = 0x00001000;
    const unsigned long MEM_RESERVE_ = 0x00002000;
    const unsigned long PAGE_RW_     = 0x04;
    const unsigned long MEM_RELEASE_ = 0x8000;

    int    built = 0;
    size_t built_bytes = 0;
    for (auto & loc : non_layer_locs) {
        ggml_tensor * t = loc.tensor;
        if (!t || !t->buffer) continue;

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
        ggml_backend_dev_t         dev  = ggml_backend_buft_get_device(buft);
        if (!dev) continue;
        size_t align = ggml_backend_buft_get_alignment(buft);
        if (align == 0) align = 256;

        const size_t total = (loc.n_bytes + align - 1) & ~(align - 1);
        if (total == 0) continue;

        void * anon = VirtualAlloc(nullptr, total, MEM_COMMIT_ | MEM_RESERVE_, PAGE_RW_);
        if (!anon) {
            fprintf(stderr, "LW-ALIAS: VirtualAlloc(%zu) failed for non-layer %s\n", total, loc.name.c_str());
            continue;
        }

        const bool have_mmap = loc.file_idx < mmap_bases.size() && mmap_bases[loc.file_idx];
        if (have_mmap) {
            memcpy(anon, mmap_bases[loc.file_idx] + loc.file_offset, loc.n_bytes);
        } else if (!read_tensor_bytes(loc.file_idx, loc.file_offset, loc.n_bytes, anon)) {
            fprintf(stderr, "LW-ALIAS: failed to fill non-layer %s\n", loc.name.c_str());
            VirtualFree(anon, 0, MEM_RELEASE_);
            continue;
        }

        ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, anon, total, loc.n_bytes);
        if (!buf) {
            fprintf(stderr, "LW-ALIAS: buffer_from_host_ptr failed for non-layer %s (%zu bytes)\n",
                    loc.name.c_str(), total);
            VirtualFree(anon, 0, MEM_RELEASE_);
            continue;
        }

        t->buffer = buf;
        t->data   = anon;
        alias_anon.push_back(anon);
        alias_bufs.push_back((void *) buf);
        built++;
        built_bytes += total;
    }
    if (built > 0) {
        printf("layer_window: built %d non-layer weight(s) (%.1f MiB) into imported anonymous buffers\n",
               built, built_bytes / (1024.0 * 1024.0));
        fflush(stdout);
    }
    return built;
#else
    return 0;
#endif
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
