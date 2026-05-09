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

    // Batch windowing: process layers in batches to minimize GPU syncs
    int      window_size          = 0;   // max layers per batch (= initial_resident_count)

    // Stats for Phase 4 per-pass tracking
    int      loads_this_pass  = 0;
    int      evicts_this_pass = 0;
    size_t   bytes_loaded     = 0;
    size_t   bytes_evicted    = 0;

    std::vector<layer_window_entry> entries;

    // Per-layer tensor info for reload from file
    struct tensor_location {
        std::string     name;
        uint16_t        file_idx;
        size_t          file_offset;
        size_t          n_bytes;
        ggml_tensor *   tensor;     // pointer to the tensor (for reload)
    };
    std::map<int, std::vector<tensor_location>> layer_tensors;  // layer_idx -> tensors

    // mmap base addresses (set after load, indexed by file_idx)
    std::vector<const uint8_t *> mmap_bases;

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
    bool ensure_layer_resident(int layer_idx);

    // Evict a layer — release physical pages for mmap (OS reclaims memory)
    void evict_layer(int layer_idx);

    // Evict layers to bring resident_bytes under budget, skipping protected layer
    void evict_to_budget(int protected_layer);

    // Look-ahead batch load: starting from layer_idx, find contiguous non-resident
    // layers (capped at window_size), MRU-evict to make room, batch-load all.
    // Returns true if any data was loaded (caller should sync).
    bool batch_load_ahead(int layer_idx);

    // Ensure all deferred layers are loaded (Phase 3 convenience)
    void ensure_all_layers_resident();

    // Reset per-pass stats (call at start of each forward pass)
    void begin_pass();

    // Print per-pass stats (call at end of each forward pass)
    void end_pass();

    // Determine layer index from a graph node's source tensors
    // Returns -1 if no layer weight is referenced
    static int get_layer_from_tensor(const ggml_tensor * t);

    // Print summary of windowing configuration
    void print_config() const;

    // Mark initially loaded layers as resident (call after model loading completes)
    void mark_initially_resident();
};

// Eval callback for Phase 4 layer windowing
// Intercepts graph compute to load/evict layers on demand
bool layer_window_eval_callback(ggml_tensor * t, bool ask, void * user_data);

// Global accessor (set during model load)
layer_window_manager * llama_get_layer_window_manager();
void llama_set_layer_window_manager(layer_window_manager * mgr);
