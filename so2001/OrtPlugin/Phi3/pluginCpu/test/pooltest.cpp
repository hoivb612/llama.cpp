#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <nt.h>
#include <ntrtl.h>
#include <nturtl.h>

#include <windows.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <random>

//
// Parallel for callback function: (id, nways, ctx) - runs specified function in
// parallel for thread id on nways threads.
//

typedef void (*BodyFn)(int id, int nways, void* ctx);

#define __NEW_CODE__ 1

#if defined(__NEW_CODE__)

typedef struct FusionThreadPool FusionThreadPool;

typedef struct WorkerArg {
    HANDLE thread_handle;
    int id;
    DWORD thread_id;
} WorkerArg;

struct DECLSPEC_CACHEALIGN FusionThreadPool {
    int nthreads;
    int nworkers;
    WorkerArg* worker_args;
    BodyFn body_fn;
    void* body_ctx;
    int nways;
    volatile LONG stop;
    DECLSPEC_CACHEALIGN volatile LONG remaining;
};

static FusionThreadPool* g_fusion_pool = NULL;

static DWORD FusionPoolWorkerMain(void* arg);

void fusion_pool_create(int nthreads) {

    FusionThreadPool* pool;

    //
    // Allocate thread pool structure.
    //

    pool = (FusionThreadPool*)calloc(1, sizeof(*pool));
    if (!pool) {
        goto fail;
    }

    //
    // Set master thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    //
    // Set thread pool address and check if there is only one thread.
    //

    g_fusion_pool = pool;
    pool->nthreads = nthreads;

    if (nthreads == 1) {
        return;
    }

    //
    // Allocate worker structure.
    //

    pool->nworkers = pool->nthreads - 1;
    pool->worker_args = (WorkerArg*)calloc(pool->nworkers, sizeof(WorkerArg));
    if (!pool->worker_args) {
        goto fail;
    }

    //
    // Create worker threads.
    //

    for (int i = 0; i < pool->nworkers; i += 1) {
        pool->worker_args[i].id = i + 1;
        pool->worker_args[i].thread_handle = CreateThread(NULL,
                                                          0,
                                                          FusionPoolWorkerMain,
                                                          &pool->worker_args[i],
                                                          0,
                                                          &pool->worker_args[i].thread_id);

        if (!pool->worker_args[i].thread_handle) {
            goto fail;
        }
    }

    return;

fail:

    if (pool) {
        if (pool->worker_args) {
            InterlockedExchange(&pool->stop, 1);

            for (int i = 0; i < pool->nworkers; i += 1) {
                if (pool->worker_args[i].thread_handle) {
                    NtAlertThreadByThreadId(UlongToHandle(pool->worker_args[i].thread_id));
                    WaitForSingleObject(pool->worker_args[i].thread_handle, INFINITE);
                    CloseHandle(pool->worker_args[i].thread_handle);
                }
            }

            free(pool->worker_args);
        }

        free(pool);
    }

    fprintf(stderr, "Failed to allocate memory for thread pool - OOM\n");
    fflush(stderr);

    exit(3);
}

extern inline int fusion_pool_size(void) {
    return g_fusion_pool->nthreads;
}

void
fusion_pool_parallel_for(
    int nways,
    BodyFn fn,
    void* ctx)
{
    FusionThreadPool* pool = g_fusion_pool;

    //
    // If the number of ways is negative or the thread count is one, then
    // execute the body using the master thread.
    //

    if ((nways <= 1) || (pool->nthreads == 1)) {
        fn(0, 1, ctx);
        return;
    }

    if (nways > pool->nthreads) {
        nways = pool->nthreads;
    }

    pool->body_fn = fn;
    pool->body_ctx = ctx;
    pool->nways = nways;
    InterlockedExchange(&pool->remaining, nways);

    for (uint64_t i = 0; i < pool->nworkers; i += 1) {
        NtAlertThreadByThreadIdEx(UlongToHandle(pool->worker_args[i].thread_id),
                                  (ULONG64)pool->worker_args[i].thread_handle);
    }

    //
    // Execute the body using the master thread and wait for all pool threads
    // to complete.
    //

    fn(0, nways, ctx);
    if (InterlockedExchangeAdd(&pool->remaining, -1) != 1) {
        do {
        } while (pool->remaining != 0);
    }
}

static DWORD FusionPoolWorkerMain(void* arg) {
    WorkerArg* worker_args = (WorkerArg*)arg;

    const int id = worker_args->id;
    FusionThreadPool* pool = g_fusion_pool;

    //
    // Set worker thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    for (;;) {

        //
        // Wait for work to arrive.
        //

        NTSTATUS status = NtWaitForAlertByThreadId(worker_args->thread_handle, NULL);
        if (status != STATUS_ALERTED) {
            printf("alert status 0x%08lx\n", status);
            continue;
        }

        //
        // Check if stop has been signalled.
        //

        if (pool->stop) {
            return 0;
        }

        //
        // If this worker is included in the set of threads that are specified to
        // execute the work, then call the worker function.
        //

        if (id < pool->nways) {
            pool->body_fn(id, pool->nways, pool->body_ctx);
        }

        InterlockedExchangeAdd(&pool->remaining, -1);
    }
}

#else // defined(__NEW_CODE__)

class FusionThreadPool {
 public:
    explicit FusionThreadPool(int num_ways) : nthreads_(std::max(1, num_ways)) {

        //
        // Initialize the SRW lock and the condition variable.
        //

        mu_srw.Ptr = NULL;
        cv_mu.Ptr = NULL;

        if (nthreads_ != 1) {

            //
            // Pin worker threads including the master thread.
            //
    
            // PinWorkers(nthreads_);

            //
            // Start worker threads. 
            //
    
            for (int id = 1; id < nthreads_; ++id) {
                workers_.emplace_back([this, id] { WorkerLoop(id); });
            }
        }
    }

    ~FusionThreadPool() noexcept {

        //
        // Set predicate value stop_ under lock.
        //

        AcquireSRWLockExclusive(&mu_srw);
        stop_ = true;
        ReleaseSRWLockExclusive(&mu_srw);

        //
        // Wake all threads waiting on the associated condition variable.
        //

        WakeAllConditionVariable(&cv_mu);

        //
        // Wait for all threads to terminate.
        //

        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    int size() const { return nthreads_; }

    //
    // Runs body(id, nways, ctx) for thread id (0) plus all other threads in the
    // thread pool with an id less than nways, blocking until all tasks complete.
    // nways is clamped to the thread pool size. body must be safe to call
    // concurrently.
    //

    void ParallelFor(int nways, BodyFn fn, void* ctx) {

        if (nways <= 1 || nthreads_ == 1) { 
            fn(0, 1, ctx); 
            return; 
        }

        if (nways > nthreads_) {
            nways = nthreads_;
        }

        //
        // Set the execution function and the number of ways the body is to be
        // executed in parallel and the predicate value gen_ under lock.
        // 

        AcquireSRWLockExclusive(&mu_srw);
        body_fn_ = fn;
        body_ctx_ = ctx;
        nways_ = nways;
        remaining_.store(nways, std::memory_order_release);
        gen_ += 1;
        ReleaseSRWLockExclusive(&mu_srw);

        //
        // Notify all thread pool threads there is work to be performed.
        //

        WakeAllConditionVariable(&cv_mu);

        //
        // Execute the function body using the master thread.
        //

        fn(0, nways, ctx);

        //
        // Interlocked exchange decrement the remaining count. If the previous value
        // was not one, then this is not the last thread to finish the body execution
        // and the master thread needs to wait for all worker threads to complete.
        //

        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) != 1) {
            do {
            } while (remaining_.load(std::memory_order_acquire) != 0);
        }
    }

private:
    void WorkerLoop(int id) {

        uint64_t my_gen = 0;

        for (;;) {

            void* body_ctx;
            BodyFn body_fn;
            int nways;

            //
            // Wait for work to be signalled.
            //

            AcquireSRWLockExclusive(&mu_srw);
            do {
                if (stop_ || (gen_ != my_gen)) {
                    body_ctx = body_ctx_;
                    body_fn = body_fn_;
                    nways = nways_;
                    my_gen = gen_;
                    break;
                }

                SleepConditionVariableSRW(&cv_mu, &mu_srw, NULL, 0);

            } while (true);

            ReleaseSRWLockExclusive(&mu_srw);
            if (stop_) {
                return;
            }

            //
            // If this thread is within the execution group, then execute the function
            // body.
            //

            if (id < nways) {
                body_fn(id, nways, body_ctx);

            }

            //
            // Interlocked exchange decrement the remaining count and loop back to
            // wait for more work.
            //

            remaining_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    const int nthreads_;
    std::vector<std::thread> workers_;
    SRWLOCK mu_srw;
    CONDITION_VARIABLE cv_mu;

    BodyFn body_fn_ = NULL;
    void* body_ctx_ = NULL;
    int nways_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::atomic<int> remaining_{0};
};

#endif // defined(__NEW_CODE__)

// ---- Workload: write out[i] = tag over computed range

struct FillCtx {
    int count;
    int* out;
    int tag;
};

static void FillRange(int id, int nways, void* p) {
    FillCtx* c = static_cast<FillCtx*>(p);

    //
    // Compute the begining and ending bounds.
    //

    const int64_t begin = (int64_t)id * c->count / nways;
    const int64_t end = (int64_t)(id + 1) * c->count / nways;

//    printf("thread %d, count %d, begin %d, end %d\n", id, c->count, begin, end);

    //
    // Fill the specified range.
    //

    for (int64_t i = begin; i < end; i += 1) {

        //
        // If the current tag array entry is zero, then store the tag value in the
        // array entry. Otherwise, force a detectable mismatch.
        //

        if (!c->out[i]) {
            c->out[i] = c->tag;

        } else {
            c->out[i] = ~c->tag;
            printf("attempting to overwrite nonzero entry at index %d\n", i);
        }
    }

    volatile spin = 0;

    for (int64_t i = 0; i < 50000; i += 1) {
        spin += 1;
    }
}

#define TAG_ARRAY_SIZE (64 * 64)

DECLSPEC_ALIGN(4096) int tag_array[TAG_ARRAY_SIZE];

int main(int argc, char** argv) {

    //
    // Set default values.
    //

    int nthreads = 8;
    int rounds = 1000000;
    int N = TAG_ARRAY_SIZE;
    int mode = 0;

    //
    // Parse command arguments.
    //

    for (int i = 1; i < argc; i += 1) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cout << "Missing arg value for " + k << std::endl;
                std::exit(1);
            };

            return std::string(argv[++i]);
        };

        if (k == "--threads") {
            nthreads = std::max(1, std::stoi(next()));
            nthreads = std::min(nthreads, 64);

        } else if (k == "--rounds") {
            rounds = std::max(1, std::stoi(next()));

        } else if (k == "--tag_array_size") {
            N = std::max(1, std::stoi(next()));

        } else if (k == "--mode") {
            mode = std::stoi(next());
            if ((mode != 0) && (mode != 1)) {
                std::cout << "Unsupported mode value " << mode << std::endl;
                std::exit(2);
            }

        } else if ((k == "--help") || k == "--h") {
            std::cout
                << "Usage: pooltest [options]\n"
                << "  --threads <number-of-threads\n"
                << "  --rounds <number-of-iterations>\n"
                << "  --tag_array_size <tag-array-size>\n"
                << "  --mode <tag-array-sizing-mode>\n"
                << "    mode(0) - use tag array size\n"
                << "    mode(1) use rand() % tag_array_size\n";
           std::exit(0);

        } else {
            std::cout << "Unknown switch " + k << std::endl;
            std::exit(2);
        }
    }

    //
    // Create thread pool.
    //

#if defined(__NEW_CODE__)

    fusion_pool_create(nthreads);

#else // defined(__NEW_CODE__)

    FusionThreadPool pool(nthreads);

#endif // defined(__NEW_CODE__)


    printf("pooltest: threads = %d rounds = %d tag_array_size = %d mode = %d\n",
           nthreads,
           rounds,
           N,
           mode);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

//    std::vector<int> out(N, 0);

    int* out = tag_array;

    int64_t corruptions = 0;
    int64_t first_bad = -1;

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    LARGE_INTEGER start_time;
    QueryPerformanceCounter(&start_time);

    for (int current_round = 0; current_round < rounds; current_round += 1) {
        int count;
        FillCtx fc;

        if (!mode) {
            count = N;

        } else {
            count = 1 + (int)(std::rand() % (unsigned)N);
        }

        memset(out, 0, sizeof(int) * count);

        fc.count = count;
        fc.out = out;
        fc.tag = (int)std::max(1, std::rand());

#if defined(__NEW_CODE__)

        fusion_pool_parallel_for(nthreads, FillRange, &fc);

#else // __NEW_CODE__

        pool.ParallelFor(nthreads, FillRange, &fc);

#endif // defined(__NEW_CODE__)

        for (int i = 0; i < count; i += 1) {
            if (out[i] != fc.tag) {

                printf("tag mismatch, tag 0x%08x, expected 0x%08x\n",
                       static_cast<unsigned>(out[i]),
                       static_cast<unsigned>(fc.tag));

                corruptions += 1;

                if (first_bad < 0) {
                    first_bad = current_round;
                }

                break;
            }
        }

//        printf("round %d completed\n", current_round);
    }

    LARGE_INTEGER end_time;
    QueryPerformanceCounter(&end_time);

    float elapsed = (float)(end_time.QuadPart - start_time.QuadPart) / (float)freq.QuadPart;

    printf("Elapsed time %6.2fsec\n", elapsed);

    if (corruptions) {
        printf("pooltest: corruptions = %zd first_bad_round = %zd\n",
               corruptions,
               first_bad);

        return 2;

    } else {
        printf("pooltest: no corruptions were detected\n");
        return 0;
    }
}
