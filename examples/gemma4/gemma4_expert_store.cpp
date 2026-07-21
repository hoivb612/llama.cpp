#include "gemma4_expert_store.h"

#include "gemma4_weights.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace gemma4 {

namespace {

void * aligned_alloc_bytes(size_t bytes) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, 32);
#else
    void * p = nullptr;
    if (posix_memalign(&p, 32, bytes) != 0) return nullptr;
    return p;
#endif
}

void aligned_free_bytes(void * p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// Collect the distinct stacked-expert tensors referenced by the resolved
// weights (merged gate_up or separate gate/up, plus down).
void collect_expert_tensors(const Weights & w, std::vector<const ggml_tensor *> & out) {
    for (const LayerWeights & L : w.layers) {
        if (!L.is_moe_layer) continue;
        if (L.ffn_gate_up_exps) out.push_back(L.ffn_gate_up_exps);
        if (L.ffn_gate_exps)    out.push_back(L.ffn_gate_exps);
        if (L.ffn_up_exps)      out.push_back(L.ffn_up_exps);
        if (L.ffn_down_exps)    out.push_back(L.ffn_down_exps);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Global accessor.
// ---------------------------------------------------------------------------
namespace { ExpertStore * g_expert_store = nullptr; }

void          set_expert_store(ExpertStore * s) { g_expert_store = s; }
ExpertStore * get_expert_store()                { return g_expert_store; }

// ---------------------------------------------------------------------------
// ExpertStore.
// ---------------------------------------------------------------------------

ExpertStore::~ExpertStore() {
    set_prefetch(false); // stop + join the worker before freeing anything
    for (Node & n : lru_) aligned_free_bytes(n.buf);
    lru_.clear();
    map_.clear();
#if defined(_WIN32)
    if (handle_valid_ && handle_) CloseHandle((HANDLE) handle_);
#else
    if (handle_valid_) close((int) (intptr_t) handle_);
#endif
    handle_valid_ = false;
}

bool ExpertStore::init(const llama_model * model, const std::string & gguf_path,
                       const Weights & w, size_t budget_bytes, std::string & error) {
    (void) model;

    // 1) Parse the GGUF header/tensor-infos (no tensor data) to obtain the
    //    absolute file offset of every tensor's data.
    gguf_init_params gp{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) {
        error = "ExpertStore::init: gguf_init_from_file failed for '" + gguf_path + "'";
        return false;
    }
    const uint64_t data_off = (uint64_t) gguf_get_data_offset(g);

    std::vector<const ggml_tensor *> tensors;
    collect_expert_tensors(w, tensors);

    size_t max_block = 0;
    for (const ggml_tensor * t : tensors) {
        if (recs_.count(t)) continue; // dedup (tensors are per-layer distinct anyway)
        const char * name = t->name;
        const int64_t id = gguf_find_tensor(g, name);
        if (id < 0) {
            std::ostringstream ss;
            ss << "ExpertStore::init: expert tensor '" << name << "' not found in GGUF";
            error = ss.str();
            gguf_free(g);
            return false;
        }
        const uint64_t nb2 = (uint64_t) t->nb[2];
        ExpertTensorRec r;
        r.file_offset = data_off + (uint64_t) gguf_get_tensor_offset(g, id);
        r.nb2         = nb2;
        r.block_bytes = nb2;
        r.n_expert    = (int) t->ne[2];
        r.type        = (int) t->type;
        r.ne0         = t->ne[0];
        r.ne1         = t->ne[1];

        // Sanity: the whole tensor must be n_expert contiguous blocks.
        const uint64_t tsize = (uint64_t) gguf_get_tensor_size(g, id);
        if (tsize != nb2 * (uint64_t) r.n_expert) {
            std::ostringstream ss;
            ss << "ExpertStore::init: tensor '" << name << "' size " << tsize
               << " != nb2(" << nb2 << ")*n_expert(" << r.n_expert << ")";
            error = ss.str();
            gguf_free(g);
            return false;
        }
        recs_[t] = r;
        max_block = std::max(max_block, (size_t) nb2);
    }
    gguf_free(g);

    if (recs_.empty()) {
        error = "ExpertStore::init: no expert tensors found (not a MoE model?)";
        return false;
    }

    // 2) Budget: never smaller than one expert working set. For the merged
    //    form a single expert needs 1 gate_up block + 1 down block resident
    //    across the two matmuls; for the separate form gate+up+down. We
    //    bound below by 3 max-blocks to guarantee forward progress.
    const size_t floor_bytes = max_block * 3;
    budget_ = std::max(budget_bytes, floor_bytes);

    // 3) Open a private read handle on the GGUF (positioned reads only; we
    //    never touch the mmap'd expert pages so RSS stays capped).
#if defined(_WIN32)
    HANDLE h = CreateFileA(gguf_path.c_str(), GENERIC_READ,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        std::ostringstream ss;
        ss << "ExpertStore::init: CreateFileA failed (err=" << GetLastError()
           << ") for '" << gguf_path << "'";
        error = ss.str();
        return false;
    }
    handle_ = (void *) h;
#else
    int fd = open(gguf_path.c_str(), O_RDONLY);
    if (fd < 0) {
        error = "ExpertStore::init: open failed for '" + gguf_path + "'";
        return false;
    }
    handle_ = (void *) (intptr_t) fd;
#endif
    handle_valid_ = true;

    std::fprintf(stderr,
        "gemma4 ExpertStore: %zu expert tensors indexed, max block = %.2f MiB, "
        "budget = %.1f MiB (>= floor %.1f MiB)\n",
        recs_.size(), max_block / (1024.0 * 1024.0),
        budget_ / (1024.0 * 1024.0), floor_bytes / (1024.0 * 1024.0));
    return true;
}

bool ExpertStore::read_at(uint64_t offset, void * dst, size_t bytes, std::string & error) {
#if defined(_WIN32)
    HANDLE h = (HANDLE) handle_;
    size_t done = 0;
    while (done < bytes) {
        OVERLAPPED ov{};
        const uint64_t off = offset + done;
        ov.Offset     = (DWORD) (off & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD want = (DWORD) std::min<size_t>(bytes - done, 0x40000000ull); // 1 GiB cap/call
        DWORD got = 0;
        if (!ReadFile(h, (char *) dst + done, want, &got, &ov) || got == 0) {
            std::ostringstream ss;
            ss << "ExpertStore::read_at: ReadFile failed (err=" << GetLastError()
               << ") off=" << off << " want=" << want;
            error = ss.str();
            return false;
        }
        done += got;
    }
    return true;
#else
    int fd = (int) (intptr_t) handle_;
    size_t done = 0;
    while (done < bytes) {
        ssize_t got = pread(fd, (char *) dst + done, bytes - done,
                            (off_t) (offset + done));
        if (got <= 0) {
            error = "ExpertStore::read_at: pread failed";
            return false;
        }
        done += (size_t) got;
    }
    return true;
#endif
}

void ExpertStore::evict_to_budget() {
    // Evict least-recently-used blocks (list back) until within budget,
    // never evicting the pinned block (the one currently feeding a matmul).
    // Caller holds mtx_.
    while (cur_bytes_ > budget_ && lru_.size() > 1) {
        Node & back = lru_.back();
        if (back.buf == pinned_) {
            // Pinned block sits at the LRU tail: rotate it to the front so the
            // next-oldest becomes the eviction candidate, then re-examine.
            lru_.splice(lru_.begin(), lru_, std::prev(lru_.end()));
            map_[Key{ back.t, back.e }] = lru_.begin();
            continue;
        }
        map_.erase(Key{ back.t, back.e });
        cur_bytes_ -= back.bytes;
        aligned_free_bytes(back.buf);
        lru_.pop_back();
        ++stats_.evictions;
    }
}

void * ExpertStore::insert_locked(const ggml_tensor * t, int e, void * buf, size_t bytes) {
    // Caller holds mtx_. Publishes a freshly-read block at the MRU front.
    lru_.push_front(Node{ t, e, buf, bytes });
    map_[Key{ t, e }] = lru_.begin();
    cur_bytes_ += bytes;
    evict_to_budget();
    stats_.peak_bytes = std::max(stats_.peak_bytes, (uint64_t) cur_bytes_);
    return buf;
}

const void * ExpertStore::fetch(const ggml_tensor * W3d, int expert, std::string & error) {
    const Key key{ W3d, expert };

    std::unique_lock<std::mutex> lock(mtx_);
    ++stats_.fetches;

    // Fast path / wait-for-worker: loop because a wait may wake to find the
    // block still not resident (worker dropped it on read error) in which
    // case we fall through and read it ourselves.
    for (;;) {
        auto mit = map_.find(key);
        if (mit != map_.end()) {
            ++stats_.hits;
            lru_.splice(lru_.begin(), lru_, mit->second); // promote to MRU
            pinned_ = mit->second->buf;
            return mit->second->buf;
        }
        if (claimed_.count(key)) {
            // A prefetch for this key is queued or in flight -- wait for the
            // worker to finish it rather than issue a duplicate read.
            ++stats_.waits;
            cv_done_.wait(lock, [&] {
                return map_.count(key) || !claimed_.count(key);
            });
            continue; // re-check map_/claimed_
        }
        break; // not resident, not claimed -> synchronous read below
    }

    ++stats_.misses;
    auto rit = recs_.find(W3d);
    if (rit == recs_.end()) {
        error = "ExpertStore::fetch: tensor not indexed";
        return nullptr;
    }
    const ExpertTensorRec r = rit->second; // copy (read outside the lock)
    if (expert < 0 || expert >= r.n_expert) {
        std::ostringstream ss;
        ss << "ExpertStore::fetch: expert " << expert << " out of range [0,"
           << r.n_expert << ")";
        error = ss.str();
        return nullptr;
    }

    const size_t   bytes = (size_t) r.block_bytes;
    const uint64_t off   = r.file_offset + (uint64_t) expert * r.nb2;

    lock.unlock();
    void * buf = aligned_alloc_bytes(bytes);
    if (!buf) { error = "ExpertStore::fetch: allocation failed"; return nullptr; }
    if (!read_at(off, buf, bytes, error)) {
        aligned_free_bytes(buf);
        return nullptr;
    }
    lock.lock();

    // Another thread (worker) may have raced this key in while we read; if so,
    // drop our copy and use the resident one.
    auto mit2 = map_.find(key);
    if (mit2 != map_.end()) {
        aligned_free_bytes(buf);
        lru_.splice(lru_.begin(), lru_, mit2->second);
        pinned_ = mit2->second->buf;
        return mit2->second->buf;
    }
    stats_.bytes_read += bytes;
    pinned_ = buf; // protect before eviction can run
    insert_locked(W3d, expert, buf, bytes);
    return buf;
}

void ExpertStore::set_prefetch(bool on, int n_workers) {
    if (on == prefetch_on_) return;
    if (on) {
        if (!handle_valid_) return; // nothing to prefetch from
        const int nw = std::max(1, n_workers);
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = false;
            prefetch_on_ = true;
            n_workers_   = nw;
        }
        workers_.reserve((size_t) nw);
        for (int i = 0; i < nw; ++i) {
            workers_.emplace_back(&ExpertStore::worker_loop, this);
        }
    } else {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            stop_ = true;
            prefetch_on_ = false;
            queue_.clear();
        }
        cv_worker_.notify_all();
        for (std::thread & t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
        n_workers_ = 1;
        // Any keys still marked claimed but never serviced are now free.
        std::lock_guard<std::mutex> lock(mtx_);
        claimed_.clear();
        cv_done_.notify_all();
    }
}

void ExpertStore::prefetch(const std::vector<KeyPair> & keys) {
    if (!prefetch_on_) return;
    bool queued = false;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (const KeyPair & kp : keys) {
            const Key k{ kp.first, kp.second };
            if (map_.count(k) || claimed_.count(k)) continue; // resident/pending
            if (!recs_.count(kp.first)) continue;             // not an expert tensor
            queue_.push_back(k);
            claimed_.insert(k);
            queued = true;
        }
    }
    if (queued) cv_worker_.notify_all();
}

void ExpertStore::worker_loop() {
    std::string werr;
    std::unique_lock<std::mutex> lock(mtx_);
    for (;;) {
        cv_worker_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (stop_) break;

        const Key k = queue_.front();
        queue_.pop_front();

        // Skip if it became resident since being queued.
        if (map_.count(k)) {
            claimed_.erase(k);
            cv_done_.notify_all();
            continue;
        }
        auto rit = recs_.find(k.t);
        if (rit == recs_.end()) { claimed_.erase(k); cv_done_.notify_all(); continue; }
        const ExpertTensorRec r = rit->second;
        if (k.e < 0 || k.e >= r.n_expert) { claimed_.erase(k); cv_done_.notify_all(); continue; }

        const size_t   bytes = (size_t) r.block_bytes;
        const uint64_t off   = r.file_offset + (uint64_t) k.e * r.nb2;

        lock.unlock();
        void * buf = aligned_alloc_bytes(bytes);
        const bool ok = buf && read_at(off, buf, bytes, werr);
        lock.lock();

        if (!ok) {
            // Drop the request; a later fetch() will read it synchronously and
            // surface the real error. Free any partial allocation.
            if (buf) aligned_free_bytes(buf);
            claimed_.erase(k);
            cv_done_.notify_all();
            continue;
        }
        // A racing synchronous fetch() may have inserted it while we read.
        if (map_.count(k)) {
            aligned_free_bytes(buf);
        } else {
            stats_.bytes_read     += bytes;
            stats_.prefetch_reads += 1;
            insert_locked(k.t, k.e, buf, bytes);
        }
        claimed_.erase(k);
        cv_done_.notify_all();
    }
}

void ExpertStore::drain() {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_done_.wait(lock, [&] { return queue_.empty() && claimed_.empty(); });
}

const ExpertTensorRec * ExpertStore::rec(const ggml_tensor * W3d) const {
    auto it = recs_.find(W3d);
    return it == recs_.end() ? nullptr : &it->second;
}

void ExpertStore::log_stats(const char * tag) const {
    const double hr = stats_.fetches
        ? (100.0 * (double) stats_.hits / (double) stats_.fetches) : 0.0;
    std::fprintf(stderr,
        "gemma4 ExpertStore[%s]: fetches=%llu hits=%llu misses=%llu (hit rate %.1f%%) "
        "prefetch_reads=%llu waits=%llu evictions=%llu bytes_read=%.1f MiB "
        "peak_resident=%.1f MiB budget=%.1f MiB prefetch=%s(%dw)\n",
        tag ? tag : "",
        (unsigned long long) stats_.fetches,
        (unsigned long long) stats_.hits,
        (unsigned long long) stats_.misses,
        hr,
        (unsigned long long) stats_.prefetch_reads,
        (unsigned long long) stats_.waits,
        (unsigned long long) stats_.evictions,
        stats_.bytes_read / (1024.0 * 1024.0),
        stats_.peak_bytes / (1024.0 * 1024.0),
        budget_ / (1024.0 * 1024.0),
        prefetch_on_ ? "on" : "off",
        prefetch_on_ ? n_workers_ : 0);
}

} // namespace gemma4
