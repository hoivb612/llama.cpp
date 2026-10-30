#include "fusion_cpu_ep_sync.h"

#include <process.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#if defined(_MSC_VER)
#include <intrin.h>
#define FUSION_CPU_PAUSE() _mm_pause()
#else
#define FUSION_CPU_PAUSE() __builtin_ia32_pause()
#endif

#ifndef FUSIONCPU_FUTEX_POOL
#define FUSIONCPU_FUTEX_POOL 0
#endif

static int64_t g_ticks_per_ms = 1;

static int64_t FusionPoolNowTicks(void) {
    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);
    return ticks.QuadPart;
}

enum { FUSION_POOL_SPIN_BUDGET_MS = 2 };

typedef struct FusionPoolWorkerArg {
    FusionThreadPool* pool;
    int id;
} FusionPoolWorkerArg;

struct FusionThreadPool {
    int nthreads;
    HANDLE* workers;
    FusionPoolWorkerArg* worker_args;
    int nworkers;
    CRITICAL_SECTION run_cs;
    fusion_body_fn body_fn;
    void* body_ctx;
    int nways;
    _Atomic uint64_t generation;
    _Atomic int remaining;
    _Atomic int stop;
#if FUSIONCPU_FUTEX_POOL
    _Atomic int work_waiters;
    _Atomic int done_waiter;
#else
    CRITICAL_SECTION wake_cs;
    CONDITION_VARIABLE wake_cv;
    CRITICAL_SECTION done_cs;
    CONDITION_VARIABLE done_cv;
#endif
};

static unsigned __stdcall FusionPoolWorkerMain(void* arg);

static bool FusionPoolWaitToStart(FusionThreadPool* pool, uint64_t generation) {
    const int64_t deadline =
        FusionPoolNowTicks() + g_ticks_per_ms * FUSION_POOL_SPIN_BUDGET_MS;
    int spin = 0;
    for (;;) {
        if (atomic_load_explicit(&pool->stop, memory_order_acquire)) return false;
        if (atomic_load_explicit(&pool->generation, memory_order_acquire) != generation) {
            return true;
        }
        if (((++spin) & 1023) == 0 && FusionPoolNowTicks() >= deadline) {
#if FUSIONCPU_FUTEX_POOL
            uint64_t compare = generation;
            atomic_fetch_add_explicit(&pool->work_waiters, 1, memory_order_seq_cst);
            while (atomic_load_explicit(&pool->generation, memory_order_acquire) == generation &&
                   !atomic_load_explicit(&pool->stop, memory_order_acquire)) {
                WaitOnAddress((volatile VOID*)&pool->generation, &compare,
                              sizeof(compare), INFINITE);
            }
            atomic_fetch_sub_explicit(&pool->work_waiters, 1, memory_order_seq_cst);
            return !atomic_load_explicit(&pool->stop, memory_order_acquire);
#else
            EnterCriticalSection(&pool->wake_cs);
            while (!atomic_load_explicit(&pool->stop, memory_order_acquire) &&
                   atomic_load_explicit(&pool->generation, memory_order_acquire) == generation) {
                SleepConditionVariableCS(&pool->wake_cv, &pool->wake_cs, INFINITE);
            }
            LeaveCriticalSection(&pool->wake_cs);
            return !atomic_load_explicit(&pool->stop, memory_order_acquire);
#endif
        }
        FUSION_CPU_PAUSE();
    }
}

static void FusionPoolWaitForDone(FusionThreadPool* pool) {
    const int64_t deadline =
        FusionPoolNowTicks() + g_ticks_per_ms * FUSION_POOL_SPIN_BUDGET_MS;
    int spin = 0;
    for (;;) {
        if (atomic_load_explicit(&pool->remaining, memory_order_acquire) == 0) return;
        if (((++spin) & 1023) == 0 && FusionPoolNowTicks() >= deadline) {
#if FUSIONCPU_FUTEX_POOL
            int remaining;
            atomic_store_explicit(&pool->done_waiter, 1, memory_order_seq_cst);
            while ((remaining = atomic_load_explicit(
                        &pool->remaining, memory_order_acquire)) != 0) {
                WaitOnAddress((volatile VOID*)&pool->remaining, &remaining,
                              sizeof(remaining), INFINITE);
            }
            atomic_store_explicit(&pool->done_waiter, 0, memory_order_seq_cst);
#else
            EnterCriticalSection(&pool->done_cs);
            while (atomic_load_explicit(&pool->remaining, memory_order_acquire) != 0) {
                SleepConditionVariableCS(&pool->done_cv, &pool->done_cs, INFINITE);
            }
            LeaveCriticalSection(&pool->done_cs);
#endif
            return;
        }
        FUSION_CPU_PAUSE();
    }
}

FusionThreadPool* fusion_pool_create(int num_ways) {
    LARGE_INTEGER frequency;
    FusionThreadPool* pool;
    int index;
    QueryPerformanceFrequency(&frequency);
    g_ticks_per_ms = frequency.QuadPart / 1000;
    if (g_ticks_per_ms <= 0) g_ticks_per_ms = 1;

    pool = (FusionThreadPool*)calloc(1, sizeof(*pool));
    if (pool == NULL) return NULL;
    pool->nthreads = num_ways < 1 ? 1 : num_ways;
    InitializeCriticalSection(&pool->run_cs);
#if !FUSIONCPU_FUTEX_POOL
    InitializeCriticalSection(&pool->wake_cs);
    InitializeConditionVariable(&pool->wake_cv);
    InitializeCriticalSection(&pool->done_cs);
    InitializeConditionVariable(&pool->done_cv);
#endif
    if (pool->nthreads == 1) return pool;

    if (fusion_ccx_pin_enabled()) fusion_pin_worker_to_core(0, pool->nthreads);
    pool->nworkers = pool->nthreads - 1;
    pool->workers = (HANDLE*)calloc((size_t)pool->nworkers, sizeof(*pool->workers));
    pool->worker_args =
        (FusionPoolWorkerArg*)calloc((size_t)pool->nworkers, sizeof(*pool->worker_args));
    if (pool->workers == NULL || pool->worker_args == NULL) {
        fusion_pool_destroy(pool);
        return NULL;
    }
    for (index = 0; index < pool->nworkers; ++index) {
        pool->worker_args[index].pool = pool;
        pool->worker_args[index].id = index + 1;
        pool->workers[index] = (HANDLE)_beginthreadex(
            NULL, 0, FusionPoolWorkerMain, &pool->worker_args[index], 0, NULL);
        if (pool->workers[index] == NULL) {
            fusion_pool_destroy(pool);
            return NULL;
        }
    }
    return pool;
}

int fusion_pool_size(const FusionThreadPool* pool) {
    return pool != NULL ? pool->nthreads : 0;
}

void fusion_pool_parallel_for(FusionThreadPool* pool, int nways,
                              fusion_body_fn fn, void* ctx) {
    if (pool == NULL || nways <= 1 || pool->nthreads == 1) {
        fn(ctx, 0, 1);
        return;
    }
    if (nways > pool->nthreads) nways = pool->nthreads;
    EnterCriticalSection(&pool->run_cs);
#if FUSIONCPU_FUTEX_POOL
    pool->body_fn = fn;
    pool->body_ctx = ctx;
    pool->nways = nways;
    atomic_store_explicit(&pool->remaining, nways, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_seq_cst);
    if (atomic_load_explicit(&pool->work_waiters, memory_order_seq_cst) != 0) {
        WakeByAddressAll((PVOID)&pool->generation);
    }
#else
    EnterCriticalSection(&pool->wake_cs);
    pool->body_fn = fn;
    pool->body_ctx = ctx;
    pool->nways = nways;
    atomic_store_explicit(&pool->remaining, nways, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->generation, 1, memory_order_release);
    LeaveCriticalSection(&pool->wake_cs);
    WakeAllConditionVariable(&pool->wake_cv);
#endif
    fn(ctx, 0, nways);
    if (atomic_fetch_sub_explicit(&pool->remaining, 1, memory_order_acq_rel) != 1) {
        FusionPoolWaitForDone(pool);
    }
    LeaveCriticalSection(&pool->run_cs);
}

static unsigned __stdcall FusionPoolWorkerMain(void* arg) {
    FusionPoolWorkerArg* worker = (FusionPoolWorkerArg*)arg;
    FusionThreadPool* pool = worker->pool;
    const int id = worker->id;
    uint64_t generation = 0;
    if (fusion_ccx_pin_enabled()) fusion_pin_worker_to_core(id, pool->nthreads);
    for (;;) {
        fusion_body_fn fn;
        void* ctx;
        int nways;
        if (!FusionPoolWaitToStart(pool, generation)) return 0;
        generation = atomic_load_explicit(&pool->generation, memory_order_acquire);
        fn = pool->body_fn;
        ctx = pool->body_ctx;
        nways = pool->nways;
        if (id < nways) {
            fn(ctx, id, nways);
            if (atomic_fetch_sub_explicit(
                    &pool->remaining, 1, memory_order_acq_rel) == 1) {
#if FUSIONCPU_FUTEX_POOL
                if (atomic_load_explicit(&pool->done_waiter, memory_order_seq_cst) != 0) {
                    WakeByAddressSingle((PVOID)&pool->remaining);
                }
#else
                EnterCriticalSection(&pool->done_cs);
                WakeConditionVariable(&pool->done_cv);
                LeaveCriticalSection(&pool->done_cs);
#endif
            }
        }
    }
}

void fusion_pool_destroy(FusionThreadPool* pool) {
    int index;
    if (pool == NULL) return;
    atomic_store_explicit(&pool->stop, 1, memory_order_release);
#if FUSIONCPU_FUTEX_POOL
    WakeByAddressAll((PVOID)&pool->generation);
#else
    WakeAllConditionVariable(&pool->wake_cv);
#endif
    for (index = 0; index < pool->nworkers; ++index) {
        if (pool->workers != NULL && pool->workers[index] != NULL) {
            WaitForSingleObject(pool->workers[index], INFINITE);
            CloseHandle(pool->workers[index]);
        }
    }
    free(pool->workers);
    free(pool->worker_args);
    DeleteCriticalSection(&pool->run_cs);
#if !FUSIONCPU_FUTEX_POOL
    DeleteCriticalSection(&pool->wake_cs);
    DeleteCriticalSection(&pool->done_cs);
#endif
    free(pool);
}
