#pragma once

// Gemma-4 MoE P1 -- pread hard-cap ExpertStore.
//
// Colibri-style expert streaming for the standalone gemma4 hand-forward.
// The model is loaded by llama with mmap (default), so the expert bank
// tensors' ->data point into the mmap'd GGUF file. During the all-resident
// P0 forward, matmul_expert_qf32 dereferences those pointers, faulting the
// expert pages into the process working set on demand -- unbounded RSS.
//
// P1 replaces that with a HARD memory cap:
//   * We never dereference the mmap'd expert pages (so they stay unfaulted
//     and cost ~0 RSS).
//   * Instead we pread each expert's contiguous quantized sub-block from a
//     SEPARATE file handle into our own fixed-size buffer pool, keyed by
//     (tensor, expert) with LRU eviction. The pool byte budget is the hard
//     cap on resident expert memory: process RSS = dense weights + pool.
//
// Because the stacked expert tensors store the expert dimension outermost
// (ne[2] = n_expert), expert e's block begins at file_offset + e*nb[2] and
// is exactly nb[2] bytes laid out identically to a standalone 2D quant
// tensor [n_in, n_out] with the native row stride -- so the streamed bytes
// can be fed straight to ggml_mul_mat (see matmul_qblock_qf32).

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct ggml_tensor;
struct llama_model;

namespace gemma4 {

struct Weights;

// File-offset + geometry record for one stacked expert tensor.
struct ExpertTensorRec {
    uint64_t file_offset = 0;   // absolute byte offset of expert 0's block
    uint64_t nb2         = 0;   // stride between experts (== block bytes)
    uint64_t block_bytes = 0;   // bytes per expert block (== nb2)
    int      n_expert    = 0;
    int      type        = 0;   // ggml_type of the block
    int64_t  ne0         = 0;   // per-expert n_in
    int64_t  ne1         = 0;   // per-expert n_out
};

struct ExpertStoreStats {
    uint64_t hits        = 0;
    uint64_t misses      = 0;   // == synchronous (fetch-path) pread count
    uint64_t evictions   = 0;
    uint64_t bytes_read  = 0;   // total bytes pread from disk (fetch + worker)
    uint64_t peak_bytes  = 0;   // peak resident pool bytes
    uint64_t fetches      = 0;  // hits + misses
    uint64_t prefetch_reads = 0; // blocks read by the background worker
    uint64_t waits          = 0; // fetches that blocked on an in-flight worker read
};

// Hard-capped, LRU expert block cache backed by positioned file reads.
//
// fetch() is called from a single consumer thread (the moe_ffn / MatmulCtx
// invariant). An optional background I/O worker (enabled via set_prefetch)
// services prefetch() requests concurrently; a mutex serializes all pool
// mutations between the consumer and the worker, and the block returned by
// the most recent fetch() is "pinned" so neither thread evicts it while its
// matmul runs. With prefetch disabled the worker never starts and behavior
// is identical to the pure-synchronous path.
class ExpertStore {
public:
    // (tensor, expert) identity for a single expert block.
    using KeyPair = std::pair<const ggml_tensor *, int>;

    ExpertStore() = default;
    ~ExpertStore();

    ExpertStore(const ExpertStore &) = delete;
    ExpertStore & operator=(const ExpertStore &) = delete;

    // Build the file-offset index for every expert bank tensor in `w`,
    // open a private read handle on the GGUF, and set the pool budget.
    // budget_bytes is the hard cap on resident expert bytes; it is raised
    // to hold at least one full expert working set (the largest single
    // block) so a single expert's matmul can always be served.
    bool init(const llama_model * model, const std::string & gguf_path,
              const Weights & w, size_t budget_bytes, std::string & error);

    bool ready() const { return handle_valid_; }

    // Return a pointer to a resident copy of expert e's block for tensor
    // W3d, reading it from disk on a miss and evicting LRU blocks to stay
    // within budget. The pointer is stable until the next fetch() that
    // triggers eviction; callers consume it (run the matmul) before the
    // next fetch, so it is always valid at the point of use.
    const void * fetch(const ggml_tensor * W3d, int expert, std::string & error);

    // Geometry lookup (for the streaming matmul: type + dims).
    const ExpertTensorRec * rec(const ggml_tensor * W3d) const;

    // Enable/disable the background prefetch worker. Turning it on starts a
    // single I/O thread; turning it off (or destruction) stops and joins it.
    // Safe to call once after init(); no-op if the state is unchanged.
    void set_prefetch(bool on);
    bool prefetch_enabled() const { return prefetch_on_; }

    // Asynchronously warm the given (tensor, expert) blocks into the pool in
    // list order (which should be usage order for best overlap). No-op when
    // prefetch is disabled. Blocks already resident or already queued are
    // skipped. Requests are advisory: a later fetch() of an un-serviced key
    // simply reads it synchronously.
    void prefetch(const std::vector<KeyPair> & keys);

    size_t budget_bytes() const { return budget_; }
    size_t resident_bytes() const { return cur_bytes_; }
    const ExpertStoreStats & stats() const { return stats_; }

    // Zero the telemetry counters WITHOUT clearing the cache -- used to
    // separate prefill (warmup) from steady-state decode measurements.
    void reset_stats() { stats_ = ExpertStoreStats{}; }

    // Block until the prefetch queue is fully serviced (no queued or in-flight
    // reads). Used before reading telemetry so counters are stable/accurate.
    // No-op when prefetch is disabled.
    void drain();

    // Emit a human-readable telemetry line (hit rate, bytes, peak) to stderr.
    void log_stats(const char * tag) const;

private:
    struct Node {
        const ggml_tensor * t = nullptr;
        int                 e = 0;
        void *              buf = nullptr;
        size_t              bytes = 0;
    };

    // (tensor, expert) key -> position in the LRU list (front = most recent).
    struct Key {
        const ggml_tensor * t;
        int                 e;
        bool operator==(const Key & o) const noexcept { return t == o.t && e == o.e; }
    };
    struct KeyHash {
        size_t operator()(const Key & k) const noexcept {
            const size_t p = reinterpret_cast<size_t>(k.t);
            return (p >> 4) ^ ((size_t) (uint32_t) k.e * 0x9E3779B1u);
        }
    };

    bool read_at(uint64_t offset, void * dst, size_t bytes, std::string & error);

    // Evict LRU blocks until within budget. Must be called with mtx_ held.
    // Never evicts the pinned block (the one the consumer is using now).
    void evict_to_budget();

    // Insert a freshly-read block. Must be called with mtx_ held. Takes
    // ownership of buf. Returns a pointer to the resident buffer.
    void * insert_locked(const ggml_tensor * t, int e, void * buf, size_t bytes);

    // Background worker entry point (present only while prefetch is on).
    void worker_loop();

    std::unordered_map<const ggml_tensor *, ExpertTensorRec> recs_;
    std::list<Node>                                          lru_;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> map_;

    size_t budget_    = 0;
    size_t cur_bytes_ = 0;

    // Platform file handle (void* to keep this header free of <windows.h>).
    void * handle_       = nullptr;
    bool   handle_valid_ = false;

    // The buffer returned by the most recent fetch(): protected from eviction
    // by both the consumer and the worker until the next fetch() replaces it.
    const void * pinned_ = nullptr;

    // Concurrency: serializes all mutations of the pool state above.
    std::mutex               mtx_;
    std::condition_variable  cv_worker_; // wakes the worker when work arrives
    std::condition_variable  cv_done_;   // wakes fetch() when a read completes
    std::thread              worker_;
    std::deque<Key>          queue_;      // pending prefetch requests (FIFO)
    std::unordered_set<Key, KeyHash> claimed_; // queued or in-flight keys
    bool                     prefetch_on_ = false;
    bool                     stop_        = false;

    ExpertStoreStats stats_;
};

// Global accessor (matches the set_matmul_cache / set_attn_parallel
// convention already used by this example). When set, moe_ffn streams
// expert blocks through the store instead of the resident mmap path.
void          set_expert_store(ExpertStore * s);
ExpertStore * get_expert_store();

} // namespace gemma4
