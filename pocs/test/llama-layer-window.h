#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <map>

struct ggml_tensor;

// Layer Window Manager — controls which model layers are resident in memory.
// When a memory budget is set, only a subset of layers are loaded initially;
// remaining layers can be streamed from storage on demand.
//
// Env var: GGML_WEIGHT_BUDGET_MB overrides programmatic setting.
// Value 0 = unlimited (all layers resident, default behavior).
//
// Phase 4: uses eval callback to load/evict layers during graph compute.
// For mmap systems, "evict" releases physical pages back to the OS.

struct layer_window_entry {
    int      layer_idx;
    bool     resident;       // weights currently in physical memory
    bool     repacked;       // CPU repack done (deferred in windowed mode)
    size_t   memory_size;    // total bytes for this layer's weight tensors
    uint64_t last_access;    // monotonic counter for LRU eviction
};

struct layer_window_manager {
    int      total_layers    = 0;
    size_t   budget_bytes    = 0;   // 0 = unlimited (all resident)
    size_t   resident_bytes  = 0;   // current resident layer memory
    size_t   non_layer_bytes = 0;   // always-resident embed/output (informational)
    uint64_t access_counter  = 0;   // monotonic for LRU
    bool     all_resident    = false; // set true after ensure_all called
    bool     use_mmap        = false; // true if weight data is memory-mapped
    int      current_layer   = -1;   // layer currently being computed (Phase 4)

    // Stats for Phase 4 per-pass tracking
    int      loads_this_pass  = 0;
    int      evicts_this_pass = 0;
    size_t   bytes_loaded     = 0;
    size_t   bytes_evicted    = 0;

    // Lifetime counters (accumulated across all passes)
    int      total_loads      = 0;
    int      total_evicts     = 0;
    int      total_passes     = 0;
    size_t   total_bytes_loaded  = 0;
    size_t   total_bytes_evicted = 0;

    std::vector<layer_window_entry> entries;

    // Per-layer tensor info for reload from file
    struct tensor_location {
        std::string     name;
        uint16_t        file_idx;
        size_t          file_offset;
        size_t          n_bytes;
        ggml_tensor *   tensor;     // pointer to the tensor (for reload)
        uint64_t        checksum;   // reference checksum (set during initial load)
    };
    std::map<int, std::vector<tensor_location>> layer_tensors;  // layer_idx -> tensors

    // mmap base addresses (set after load, indexed by file_idx)
    std::vector<const uint8_t *> mmap_bases;

    // On-demand file-read source (used when mmap is disabled, e.g. direct_io or
    // GPU offload). Indexed by file_idx; paths registered from the loader. When
    // mmap_bases is unavailable for a deferred layer, its weight bytes are read
    // from these files at the recorded file_offset and uploaded via tensor_set.
    std::vector<std::string> src_file_paths;
    std::vector<void *>      src_file_handles;   // lazily opened FILE* per file_idx
    std::vector<uint8_t>     reload_scratch;     // reused per-tensor read buffer

    // Register the backing GGUF file path for a given file_idx (from the loader).
    void set_source_file(uint16_t file_idx, const std::string & path);
    // Read n_bytes at file_offset from file_idx into dst. Returns false on error.
    bool read_tensor_bytes(uint16_t file_idx, size_t file_offset, size_t n_bytes, void * dst);

    // Initialize from measured layer sizes
    void init(int n_layers, size_t budget_mb);

    // Set layer size (called during load_all_data enumeration)
    void set_layer_size(int layer_idx, size_t bytes);

    // Determine which layers should be initially loaded given the budget
    // Returns true if layer should be loaded (resident)
    bool should_load_layer(int layer_idx) const;

    // Record a tensor's file location for later reload
    void record_tensor_location(int layer_idx, const std::string & name,
                                uint16_t file_idx, size_t offset, size_t n_bytes,
                                ggml_tensor * tensor);

    // Get number of initially resident layers based on budget
    int get_initial_resident_count() const;

    // Ensure a specific layer is resident (load from mmap if needed)
    // Returns true if layer was loaded (false if already resident)
    // When allow_evict=false, loads without evicting (caller handles eviction later)
    bool ensure_layer_resident(int layer_idx, bool allow_evict = true);

    // Evict a layer — release physical pages for mmap (OS reclaims memory)
    void evict_layer(int layer_idx);

    // Evict layers to bring resident_bytes under budget, skipping protected layer
    void evict_to_budget(int protected_layer);

    // Ensure all deferred layers are loaded (Phase 3 convenience)
    void ensure_all_layers_resident();

    // Reset per-pass stats (call at start of each forward pass)
    void begin_pass();

    // Accumulate per-pass stats into lifetime counters (call at end of each forward pass)
    void end_pass();

    // Print lifetime windowing stats summary
    void print_stats() const;

    // Determine layer index from a graph node's source tensors
    // Returns -1 if no layer weight is referenced
    static int get_layer_from_tensor(const ggml_tensor * t);

    // Print summary of windowing configuration
    void print_config() const;

    // Mark initially loaded layers as resident (call after model loading completes)
    void mark_initially_resident();

    // Release mmap pages to reclaim physical RAM (call after mmap_bases is set)
    void release_mmap_pages();

    // ---- Stage 1: per-layer zero-copy aliased cache (Vulkan UMA fallback) ----
    // When the backend cannot alias the file-backed mmap directly (e.g. AMD
    // Vulkan on Windows: no file-backed host-ptr import, no sparse residency),
    // convert each layer's weights into a per-layer ANONYMOUS host buffer that
    // IS imported into the backend (zero-copy). Tensors are re-pointed at the
    // imported buffer so compute reads host memory directly — no per-token
    // upload copy. Returns true if at least one layer was converted.
    bool aliased_cache = false;              // set true after successful conversion
    std::vector<void *> alias_anon;          // non-layer anonymous bases (for free)
    std::vector<void *> alias_bufs;          // non-layer ggml_backend_buffer_t (kept alive)
    std::map<int, void *> layer_alias_anon;  // layer_idx -> anonymous base (per-layer free)
    std::map<int, void *> layer_alias_buf;   // layer_idx -> ggml_backend_buffer_t
    bool convert_layers_to_aliased_cache();

    // Build (VirtualAlloc + fill from mmap/file + import + re-point) the aliased
    // buffer for a single layer. Stores anon/buf in the per-layer maps. Returns
    // true on success. Used by convert_layers_to_aliased_cache and Stage 2b
    // on-demand streaming.
    bool build_layer_aliased(int layer_idx);
    // Free a single layer's aliased buffer (destroy imported buffer + VirtualFree).
    void free_layer_aliased(int layer_idx);

    // Stage 2a: migrate a single residual GPU-resident weight (e.g. output.weight)
    // that shares the monolithic layer buffer but is NOT a blk.* layer tensor into
    // its own imported ANONYMOUS buffer. Reads the tensor's current bytes back from
    // the device, imports host memory zero-copy, and re-points the tensor. This
    // vacates the last referents off the big layer buffer so it can be freed.
    // Returns true on success (tensor re-pointed); false leaves the tensor as-is.
    bool migrate_residual_tensor(ggml_tensor * t);

    // ---- Stage 2b: load-time aliased streaming (never allocate the full buffer) ----
    // When GGML_LW_ALIAS_STREAM is set, the loader gives the windowed weight ctx a
    // dummy 0-size buffer instead of the monolithic device buffer (which OOMs on
    // constrained UMA systems). The manager then OWNS every windowed weight: it
    // builds per-layer imported anonymous buffers from mmap/file post-load and (in
    // 2b) streams deferred layers on demand. Non-blk.* GPU weights (output.weight)
    // are recorded separately so they too can be built from disk (no device copy
    // exists to read back).
    bool aliased_load_mode = false;                 // set by loader when dummy buffer chosen
    std::vector<tensor_location> non_layer_locs;    // recorded non-blk.* GPU weights
    void record_non_layer_location(const std::string & name, uint16_t file_idx,
                                   size_t offset, size_t n_bytes, ggml_tensor * tensor);
    // Build imported anonymous buffers for all recorded non-layer GPU weights,
    // filling from mmap/file. Returns the number built.
    int build_non_layer_aliased();
    // True when the aliased-stream load path should be used (env-gated).
    static bool alias_stream_enabled();

    // Compute reference checksums for mmap data integrity verification
    // Must be called AFTER mmap_bases is populated and BEFORE release_mmap_pages
    void compute_reference_checksums();
};

// Eval callback for Phase 4 layer windowing
// Intercepts graph compute to load/evict layers on demand
bool layer_window_eval_callback(ggml_tensor * t, bool ask, void * user_data);

// Global accessor (set during model load)
layer_window_manager * llama_get_layer_window_manager();
void llama_set_layer_window_manager(layer_window_manager * mgr);
