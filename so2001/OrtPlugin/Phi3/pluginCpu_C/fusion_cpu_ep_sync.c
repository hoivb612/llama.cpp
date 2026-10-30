// #define __NEW_CODE__ 1

#if defined(__NEW_CODE__)

#include "fusion_cpu_ep.h"

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
#include <intrin.h>

#include <process.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    fusion_body_fn body_fn;
    void* body_ctx;
    int nways;
    volatile LONG stop;
    volatile LONG64 generation;
    DECLSPEC_CACHEALIGN volatile LONG remaining;
};

static FusionThreadPool* g_fusion_pool = NULL;

//
// Establishes the process affinity of the current process dependent on the number
// of ccx's and threads requested. The process affinity is computed such that the
// threads are as sparsely spread as possible across the ccx's in the system. This
// has a marked effect on performance.
//

static void PinWorkers(uint32_t nthreads) {

    typedef struct {
        WORD group;
        ULONG base_lp;
        uint32_t cores;
        KAFFINITY mask;
    } ccx_t;

    char *buf = NULL;
    ccx_t ccx[64];

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    uint32_t i;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info;
    DWORD len = 0;
    uint32_t n_ccx = 0;
    char *p = NULL;

    //
    // Check if the process group affinity has already been set.
    //

    USHORT group_count = 0;

    GetProcessDefaultCpuSetMasks(GetCurrentProcess(), NULL, 0, &group_count);
    if (group_count != 0) {
        return;
    }

    //
    // Check for AMD processor.
    //

    // ****** add cpuid code

    //
    // Check if SMT is enabled.
    //

    __cpuid((int *)&cpu_info, 0x8000001e);
    if ((((cpu_info.ebx & 0xf00) >> 8) + 1) != 2) {
        printf("SMT is not enabled\n");
        goto skip;
    }

    //
    // Query the processor relationship information.
    //

    GetLogicalProcessorInformationEx(RelationCache, NULL, &len);
    if (!len) {
        printf("failed to get processor information length\n");
        goto skip;
    }

    buf = (char *)malloc(len);
    if (!buf) {
        printf("failed tp allocate processor information buffer\n");
        goto skip;
    }

    info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf;
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &len)) {
        printf("failed to get processor information\n");
        free(buf);
        goto skip;
    }

    //
    // Parse the processor relationship information.
    //

    p = buf;
    while (p < (buf + len)) {
        info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;

        if ((info->Relationship == RelationCache) &&
            (info->Cache.Level == 3) &&
            (n_ccx < ARRAYSIZE(ccx))) {

            ccx[n_ccx].group = info->Cache.GroupMask.Group;
            ccx[n_ccx].mask = info->Cache.GroupMask.Mask;
            ccx[n_ccx].base_lp = 0;
            ccx[n_ccx].cores = (uint32_t)PopulationCount64(ccx[n_ccx].mask);

            BitScanForward64(&ccx[n_ccx].base_lp, info->Cache.GroupMask.Mask);
            n_ccx += 1;
        }

        p += info->Size;
    }

    free(buf);

    //
    // Check if any ccx's were discovered.
    //

    if (n_ccx) {
        GROUP_AFFINITY ga[ARRAYSIZE(ccx) / 16] = {0};

        if (FusionCpuMatMulVerbose()) {
            printf("Populated CCX table\n");

            for (uint32_t i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Compute the number of SMT pairs.
        //

        uint32_t smt_pairs = 0;

        for (i = 0; i < n_ccx; i += 1) {
            ccx[i].cores /= 2;
            smt_pairs += ccx[i].cores;
        }

        if (smt_pairs < nthreads) {
            printf("insufficient SMT pairs %u to allocate by ccx\n", smt_pairs);
            goto skip;
        }

        //
        // Depopulate the SMT paired logical processor ccx masks.
        //
        // N.B. It is possible that a ccx does not have the full complement of cores.
        //

        for (i = 0; i < n_ccx; i += 1) {
            for (uint32_t j = 0; j < 8; j += 1) {
                ccx[i].mask &= ~(1ull << ((j * 2) + ccx[i].base_lp + 1));
            }
        }

        if (FusionCpuMatMulVerbose()) {
            printf("Depopulated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Construct a set of group affinity structures which includes the specified
        // number of threads. These group affinity structures are used to set the
        // process affinity.
        //

        srand((unsigned)time(NULL));

        //
        // Start from a random ccx.
        //

        uint32_t k = (uint32_t)(rand() % (int)n_ccx);

        for (i = 0; i < nthreads; i += 1) {

            //
            // N.B. The ccx mask may not have any bits set if it does not have the
            //      full complement of cores.
            //

            uint32_t count = (uint32_t)PopulationCount64(ccx[k].mask);

            if (count) {
                WORD group = ccx[k].group;
                uint32_t select = ccx[k].base_lp;
                uint32_t position = (uint32_t)(rand() % (int)count);
    
                //
                // Assign the affinity randomly within the remaining bits.
                //

#if 0
                printf("count %u, select %u, position %u, mask 0x%016llx\n",
                        count,
                        select,
                        position,
                        ccx[k].mask);
#endif // #if 0

    
                do {
                    if (ccx[k].mask & (1ull << select)) {
                        if (!position) {
                            break;
                        }
    
                        position -= 1;
                    }
    
                    select += 1;
                } while (true);
    
                ccx[k].mask ^= 1ull << select;
    
                ga[group].Group = group;
                ga[group].Mask |= 1ull << select;
            }

            //
            // Round robin the ccx's.
            //

            k += 1;
            if (k >= n_ccx) {
                k = 0;
            }
        }

        //
        // Compact the group affinity array.
        //

        uint32_t n_groups = 0;

        for (i = 0; i < ARRAYSIZE(ga); i += 1) {
            if (ga[i].Mask) {
                ga[n_groups] = ga[i];
                n_groups += 1;
            }
        }

        if (FusionCpuMatMulVerbose()) {
            printf("Allocated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }

            printf("Process group affinity\n");

            for (i = 0; i < n_groups; i += 1) {
                printf("  group= %2u mask=0x%016llx\n", ga[i].Group, ga[i].Mask);
            }
        }

        //
        // Set process affinity.
        //

        if (!SetProcessDefaultCpuSetMasks(GetCurrentProcess(), ga, (WORD)n_groups)) {
            printf("Set process affinity group set failed\n");

        } else {
            printf("set process affinity group set succeed\n");
        }

        return;
    }

skip:

    //
    // The number of threads must be greater than zero and less than or equal to
    // one affinity group (64).
    //

    if (!nthreads || (nthreads > 64)) {
        printf("number of threads must be non-zero and less than or equal to 64\n");
        return;
    }

    //
    // The hardware concurrency must not be odd and greater than 1 - assume 2-way SMT.
    //

    uint32_t logical = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (logical == 0) logical = 1;

    if ((logical <= 1) || (logical & 1)) {
        printf("hardware concurency is odd or zero\n");
        return;
    }

    //
    // The number of physical cores must be greater than or equal to the number of
    // threads.
    //

    uint32_t phys = logical / 2u;

    if (nthreads > phys) {
        printf("number of threads is greater than the number of physical cores\n");
        return;
    }

    //
    // Attempt to spread the workers across multiple ccx's.
    //

    if (!(nthreads % 8)) {
        phys -= nthreads / 2;
        printf("phys = %d\n", phys);
    }

    //
    // Evenly spread workers across the middle/high physical cores.
    //

    uint32_t core;
    GROUP_AFFINITY ga = {0};
    uint32_t lp;

    for (uint32_t i = 0; i < nthreads; i += 1) {
        core = phys - nthreads + i;
        lp = (core * 2u) % logical;

        printf("core %d, lp %d\n", core, lp);

        ga.Group = (WORD)(lp / 64u);
        ga.Mask |= (KAFFINITY)(1ull << (lp % 64u));
    }

    //
    // Set the process group affinity.
    //

    if (SetProcessAffinityMask(GetCurrentProcess(), ga.Mask)) {
        printf("process set group affinity group %u, mask 0x%08I64x\n", ga.Group, ga.Mask);

    } else {
        printf("process set group affinity failed\n");
    }
}

static DWORD FusionPoolWorkerMain(void* arg);

void fusion_pool_create(void) {

    //
    // N.B. If thread pool creation fails, there is no return from this function
    //      and the execution provider is terminated. This is treated as an
    //      out-of-memory error from which there is no recovery. If a thread pool
    //      cannot be created, there is no way to support an execution environment.
    //

    FusionThreadPool* pool;
    int nways;

    pool = calloc(1, sizeof(*pool));
    if (!pool) {
        goto fail;
    }

    //
    // Compute the thread count.
    //
    // N.B. The thread count is clamped to the range 1-64.
    //

    nways = 0;
    const char* value = getenv("ORT_FUSIONCPU_MATMULNBITS_THREADS");

    if (value != NULL && value[0] != '\0') {
        char* end = NULL;
        int parsed = strtol(value, &end, 10);
        if ((end != value) && (parsed >= 1)) {
            nways = parsed > 64 ? 64 : parsed;
        }
    }

    if (!nways) {
        DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        if (count == 0) {
            count = 1;
        }

        nways = (int)(count > 64 ? 64 : count);
    }

    if (FusionCpuTraceEnabled()) {
        fprintf(stderr,
                "[FusionCpuEP][trace] thread pool create ways=%d backend=%s ccx_pin=%d\n",
                nways,
                "cpu",
                fusion_ccx_pin_enabled());

        fflush(stderr);
    }

    //
    // Set master thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    //
    // Set thread pool address and check if there is only one thread.
    //

    g_fusion_pool = pool;
    pool->nthreads = nways;

    if (nways == 1) {
        return;
    }

    //
    // If specified, pin worker threads including the master thread.
    //

    if (fusion_ccx_pin_enabled()) {
        PinWorkers((uint32_t)pool->nthreads);
    }

    //
    // Allocate worker structure.
    //

    pool->nworkers = pool->nthreads - 1;
    pool->worker_args = calloc(pool->nworkers, sizeof(WorkerArg));
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

    RaiseFailFastException(NULL, NULL, 0);
    return;
}

extern inline int fusion_pool_size(void) {
    return g_fusion_pool->nthreads;
}

void
fusion_pool_parallel_for(
    int nways,
    fusion_body_fn fn,
    void* ctx)
{
    FusionThreadPool* pool = g_fusion_pool;

    //
    // If the number of ways is negative or the thread count is one, then
    // execute the body using the master thread.
    //

    if ((nways <= 1) || (pool->nthreads == 1)) {
        fn(ctx, 0, 1);
        return;
    }

    if (nways > pool->nthreads) {
        nways = pool->nthreads;
    }

    pool->body_fn = fn;
    pool->body_ctx = ctx;
    pool->nways = nways;
    InterlockedExchange(&pool->remaining, nways);
    InterlockedIncrement64(&pool->generation);

    for (uint64_t i = 0; i < pool->nworkers; i += 1) {
        NtAlertThreadByThreadIdEx(UlongToHandle(pool->worker_args[i].thread_id),
                                  (ULONG64)pool->worker_args[i].thread_handle);
    }

    //
    // Execute the body using the master thread and wait for all pool threads
    // to complete.
    //

    fn(ctx, 0, nways);
    if (InterlockedExchangeAdd(&pool->remaining, -1) != 1) {
        do {
        } while (pool->remaining != 0);
    }
}

static DWORD FusionPoolWorkerMain(void* arg) {
    WorkerArg* worker_args = arg;

    const int id = worker_args->id;
    FusionThreadPool* pool = g_fusion_pool;
    LONG64 generation = 0;

    //
    // Spin budget (in QPC ticks) before falling back to a blocking wait. Keeps
    // workers hot across decode's many tiny parallel_for barriers (near-zero
    // start latency) while still sleeping during long idle gaps so CPU isn't
    // burned. Configured once at pool creation from ORT_FUSIONCPU_SPIN_BUDGET_MS
    // (default 2 ms; 0 disables the spin). Tune down if prefill/power regress.
    //

    const int64_t spin_ticks = FusionCpuSpinTicks();

    //
    // Set worker thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    for (;;) {
        int spin_hit = 0;

        //
        // Adaptive spin: poll the lock-free dispatch generation before entering
        // NtWaitForAlertByThreadId. If work arrives while spinning, execute it
        // immediately and defer consuming the queued alert until after the body.
        // On a spin miss, block in the NT wait as before.
        //

        if (spin_ticks > 0) {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            const int64_t deadline = now.QuadPart + spin_ticks;
            int spin = 0;
            for (;;) {
                if (pool->generation != generation) {
                    spin_hit = 1;
                    break;
                }
                if (pool->stop) break;
                if (((++spin) & 1023) == 0) {
                    QueryPerformanceCounter(&now);
                    if (now.QuadPart >= deadline) break;
                }
                _mm_pause();
            }
        }

        //
        // Wait for work to arrive.
        //

        if (!spin_hit) {
            NTSTATUS status = NtWaitForAlertByThreadId(worker_args->thread_handle, NULL);
            if (status != STATUS_ALERTED) {
                printf("alert status 0x%08lx\n", status);
            }
        }

        generation = pool->generation;

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
            pool->body_fn(pool->body_ctx, id, pool->nways);
        }

        if (spin_hit) {
            //
            // The dispatch still queued one alert for this worker. Consume it
            // after useful work but before publishing completion. The master
            // cannot issue the next dispatch until remaining reaches zero, so
            // this cannot accidentally consume the next dispatch's alert.
            //

            NTSTATUS status = NtWaitForAlertByThreadId(worker_args->thread_handle, NULL);
            if (status != STATUS_ALERTED) {
                printf("alert drain status 0x%08lx\n", status);
            }
        }

        InterlockedExchangeAdd(&pool->remaining, -1);
    }
}

#else // defined(__NEW_CODE__)

#include "fusion_cpu_ep.h"

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
#include <intrin.h>

#include <process.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct FusionThreadPool FusionThreadPool;

typedef struct WorkerArg {
    HANDLE handle;
    int id;
} WorkerArg;

struct DECLSPEC_CACHEALIGN FusionThreadPool {
    int nthreads;
    int nworkers;
    WorkerArg* worker_args;
    fusion_body_fn body_fn;
    void* body_ctx;
    uint64_t generation;
    int nways;
    int stop;
    DECLSPEC_CACHEALIGN volatile LONG remaining;
    DECLSPEC_CACHEALIGN SRWLOCK wake_srw;
    DECLSPEC_CACHEALIGN CONDITION_VARIABLE wake_cv;
};

static FusionThreadPool* g_fusion_pool = NULL;

//
// Establishes the process affinity of the current process dependent on the number
// of ccx's and threads requested. The process affinity is computed such that the
// threads are as sparsely spread as possible across the ccx's in the system. This
// has a marked effect on performance.
//

static void PinWorkers(uint32_t nthreads) {

    typedef struct {
        WORD group;
        ULONG base_lp;
        uint32_t cores;
        KAFFINITY mask;
    } ccx_t;

    char *buf = NULL;
    ccx_t ccx[64];

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    uint32_t i;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *info;
    DWORD len = 0;
    uint32_t n_ccx = 0;
    char *p = NULL;

    //
    // Check if the process group affinity has already been set.
    //

    USHORT group_count = 0;

    GetProcessDefaultCpuSetMasks(GetCurrentProcess(), NULL, 0, &group_count);
    if (group_count != 0) {
        return;
    }

    //
    // Check for AMD processor.
    //

    // ****** add cpuid code

    //
    // Check if SMT is enabled.
    //

    __cpuid((int *)&cpu_info, 0x8000001e);
    if ((((cpu_info.ebx & 0xf00) >> 8) + 1) != 2) {
        printf("SMT is not enabled\n");
        goto skip;
    }

    //
    // Query the processor relationship information.
    //

    GetLogicalProcessorInformationEx(RelationCache, NULL, &len);
    if (!len) {
        printf("failed to get processor information length\n");
        goto skip;
    }

    buf = (char *)malloc(len);
    if (!buf) {
        printf("failed tp allocate processor information buffer\n");
        goto skip;
    }

    info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf;
    if (!GetLogicalProcessorInformationEx(RelationCache, info, &len)) {
        printf("failed to get processor information\n");
        free(buf);
        goto skip;
    }

    //
    // Parse the processor relationship information.
    //

    p = buf;
    while (p < (buf + len)) {
        info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)p;

        if ((info->Relationship == RelationCache) &&
            (info->Cache.Level == 3) &&
            (n_ccx < ARRAYSIZE(ccx))) {

            ccx[n_ccx].group = info->Cache.GroupMask.Group;
            ccx[n_ccx].mask = info->Cache.GroupMask.Mask;
            ccx[n_ccx].base_lp = 0;
            ccx[n_ccx].cores = (uint32_t)PopulationCount64(ccx[n_ccx].mask);

            BitScanForward64(&ccx[n_ccx].base_lp, info->Cache.GroupMask.Mask);
            n_ccx += 1;
        }

        p += info->Size;
    }

    free(buf);

    //
    // Check if any ccx's were discovered.
    //

    if (n_ccx) {
        GROUP_AFFINITY ga[ARRAYSIZE(ccx) / 16] = {0};

        if (FusionCpuMatMulVerbose()) {
            printf("Populated CCX table\n");

            for (uint32_t i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Compute the number of SMT pairs.
        //

        uint32_t smt_pairs = 0;

        for (i = 0; i < n_ccx; i += 1) {
            ccx[i].cores /= 2;
            smt_pairs += ccx[i].cores;
        }

        if (smt_pairs < nthreads) {
            printf("insufficient SMT pairs %u to allocate by ccx\n", smt_pairs);
            goto skip;
        }

        //
        // Depopulate the SMT paired logical processor ccx masks.
        //
        // N.B. It is possible that a ccx does not have the full complement of cores.
        //

        for (i = 0; i < n_ccx; i += 1) {
            for (uint32_t j = 0; j < 8; j += 1) {
                ccx[i].mask &= ~(1ull << ((j * 2) + ccx[i].base_lp + 1));
            }
        }

        if (FusionCpuMatMulVerbose()) {
            printf("Depopulated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }
        }

        //
        // Construct a set of group affinity structures which includes the specified
        // number of threads. These group affinity structures are used to set the
        // process affinity.
        //

        srand((unsigned)time(NULL));

        //
        // Start from a random ccx.
        //

        uint32_t k = (uint32_t)(rand() % (int)n_ccx);

        for (i = 0; i < nthreads; i += 1) {

            //
            // N.B. The ccx mask may not have any bits set if it does not have the
            //      full complement of cores.
            //

            uint32_t count = (uint32_t)PopulationCount64(ccx[k].mask);

            if (count) {
                WORD group = ccx[k].group;
                uint32_t select = ccx[k].base_lp;
                uint32_t position = (uint32_t)(rand() % (int)count);
    
                //
                // Assign the affinity randomly within the remaining bits.
                //

#if 0
                printf("count %u, select %u, position %u, mask 0x%016llx\n",
                        count,
                        select,
                        position,
                        ccx[k].mask);
#endif // #if 0

    
                do {
                    if (ccx[k].mask & (1ull << select)) {
                        if (!position) {
                            break;
                        }
    
                        position -= 1;
                    }
    
                    select += 1;
                } while (true);
    
                ccx[k].mask ^= 1ull << select;
    
                ga[group].Group = group;
                ga[group].Mask |= 1ull << select;
            }

            //
            // Round robin the ccx's.
            //

            k += 1;
            if (k >= n_ccx) {
                k = 0;
            }
        }

        //
        // Compact the group affinity array.
        //

        uint32_t n_groups = 0;

        for (i = 0; i < ARRAYSIZE(ga); i += 1) {
            if (ga[i].Mask) {
                ga[n_groups] = ga[i];
                n_groups += 1;
            }
        }

        if (FusionCpuMatMulVerbose()) {
            printf("Allocated CCX table\n");

            for (i = 0; i < n_ccx; i += 1) {
                printf("  CCX %2d: group=%u base_lp=%2d cores=%2d mask=0x%016llx\n",
                       i,
                       ccx[i].group,
                       ccx[i].base_lp,
                       ccx[i].cores,
                       ccx[i].mask);
            }

            printf("Process group affinity\n");

            for (i = 0; i < n_groups; i += 1) {
                printf("  group= %2u mask=0x%016llx\n", ga[i].Group, ga[i].Mask);
            }
        }

        //
        // Set process affinity.
        //

        if (!SetProcessDefaultCpuSetMasks(GetCurrentProcess(), ga, (WORD)n_groups)) {
            printf("Set process affinity group set failed\n");

        } else {
            printf("set process affinity group set succeed\n");
        }

        return;
    }

skip:

    //
    // The number of threads must be greater than zero and less than or equal to
    // one affinity group (64).
    //

    if (!nthreads || (nthreads > 64)) {
        printf("number of threads must be non-zero and less than or equal to 64\n");
        return;
    }

    //
    // The hardware concurrency must not be odd and greater than 1 - assume 2-way SMT.
    //

    uint32_t logical = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (logical == 0) logical = 1;

    if ((logical <= 1) || (logical & 1)) {
        printf("hardware concurency is odd or zero\n");
        return;
    }

    //
    // The number of physical cores must be greater than or equal to the number of
    // threads.
    //

    uint32_t phys = logical / 2u;

    if (nthreads > phys) {
        printf("number of threads is greater than the number of physical cores\n");
        return;
    }

    //
    // Attempt to spread the workers across multiple ccx's.
    //

    if (!(nthreads % 8)) {
        phys -= nthreads / 2;
        printf("phys = %d\n", phys);
    }

    //
    // Evenly spread workers across the middle/high physical cores.
    //

    uint32_t core;
    GROUP_AFFINITY ga = {0};
    uint32_t lp;

    for (uint32_t i = 0; i < nthreads; i += 1) {
        core = phys - nthreads + i;
        lp = (core * 2u) % logical;

        printf("core %d, lp %d\n", core, lp);

        ga.Group = (WORD)(lp / 64u);
        ga.Mask |= (KAFFINITY)(1ull << (lp % 64u));
    }

    //
    // Set the process group affinity.
    //

    if (SetProcessAffinityMask(GetCurrentProcess(), ga.Mask)) {
        printf("process set group affinity group %u, mask 0x%08I64x\n", ga.Group, ga.Mask);

    } else {
        printf("process set group affinity failed\n");
    }
}

static DWORD FusionPoolWorkerMain(void* arg);

void fusion_pool_create(void) {

    //
    // N.B. If thread pool creation fails, there is no return from this function
    //      and the execution provider is terminated. This is treated as an
    //      out-of-memory error from which there is no recovery. If a thread pool
    //      cannot be created, there is no way to support an execution environment.
    //

    FusionThreadPool* pool;
    int nways;

    pool = calloc(1, sizeof(*pool));
    if (!pool) {
        goto fail;
    }

    //
    // Compute the thread count.
    //
    // N.B. The thread count is clamped to the range 1-64.
    //

    nways = 0;
    const char* value = getenv("ORT_FUSIONCPU_MATMULNBITS_THREADS");

    if (value != NULL && value[0] != '\0') {
        char* end = NULL;
        int parsed = strtol(value, &end, 10);
        if ((end != value) && (parsed >= 1)) {
            nways = parsed > 64 ? 64 : parsed;
        }
    }

    if (!nways) {
        DWORD count = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        if (count == 0) {
            count = 1;
        }

        nways = (int)(count > 64 ? 64 : count);
    }

    if (FusionCpuTraceEnabled()) {
        fprintf(stderr,
                "[FusionCpuEP][trace] thread pool create ways=%d backend=%s ccx_pin=%d\n",
                nways,
                "cpu",
                fusion_ccx_pin_enabled());

        fflush(stderr);
    }

    InitializeSRWLock(&pool->wake_srw);
    InitializeConditionVariable(&pool->wake_cv);

    //
    // Set master thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    //
    // Set thread pool address and check if there is only one thread.
    //

    g_fusion_pool = pool;
    pool->nthreads = nways;

    if (nways == 1) {
        return;
    }

    //
    // If specified, pin worker threads including the master thread.
    //

    if (fusion_ccx_pin_enabled()) {
        PinWorkers((uint32_t)pool->nthreads);
    }

    //
    // Allocate worker structure and creat thread pool.
    //

    pool->nworkers = pool->nthreads - 1;
    pool->worker_args = calloc(pool->nworkers, sizeof(WorkerArg));
    if (!pool->worker_args) {
        goto fail;
    }

    for (int i = 0; i < pool->nworkers; i += 1) {
        pool->worker_args[i].id = i + 1;
        pool->worker_args[i].handle = CreateThread(NULL,
                                                   0,
                                                   FusionPoolWorkerMain,
                                                   &pool->worker_args[i],
                                                   0,
                                                   NULL);

        if (!pool->worker_args[i].handle) {
            goto fail;
        }
    }

    return;

fail:

    if (pool) {
        if (pool->worker_args) {
            RtlAcquireSRWLockExclusive(&pool->wake_srw);
            pool->stop = 1;
            RtlReleaseSRWLockExclusive(&pool->wake_srw);
        
            RtlWakeAllConditionVariable(&pool->wake_cv);
        
            for (int i = 0; i < pool->nworkers; i += 1) {
                if (pool->worker_args[i].handle) {
                    WaitForSingleObject(pool->worker_args[i].handle, INFINITE);
                    CloseHandle(pool->worker_args[i].handle);
                }
            }

            free(pool->worker_args);
        }

        free(pool);
    }

    fprintf(stderr, "Failed to allocate memory for thread pool - OOM\n");
    fflush(stderr);

    RaiseFailFastException(NULL, NULL, 0);
    return;
}

extern inline int fusion_pool_size(void) {
    return g_fusion_pool->nthreads;
}

void
fusion_pool_parallel_for(
    int nways,
    fusion_body_fn fn,
    void* ctx)
{
    FusionThreadPool* pool = g_fusion_pool;

    if ((nways <= 1) || (pool->nthreads == 1)) {
        fn(ctx, 0, 1);
        return;
    }

    if (nways > pool->nthreads) {
        nways = pool->nthreads;
    }

    RtlAcquireSRWLockExclusive(&pool->wake_srw);
    pool->body_fn = fn;
    pool->body_ctx = ctx;
    pool->nways = nways;
    pool->remaining = nways;
    pool->generation += 1;
    RtlReleaseSRWLockExclusive(&pool->wake_srw);

    RtlWakeAllConditionVariable(&pool->wake_cv);

    fn(ctx, 0, nways);
    if (InterlockedExchangeAdd(&pool->remaining, -1) != 1) {
        do {
        } while (pool->remaining != 0);
    }
}

static DWORD FusionPoolWorkerMain(void* arg) {
    WorkerArg* worker_args = arg;

    const int id = worker_args->id;
    uint64_t generation = 0;
    FusionThreadPool* pool = g_fusion_pool;

    //
    // Spin budget (in QPC ticks) before falling back to a blocking wait. Keeps
    // workers hot across decode's many tiny parallel_for barriers (near-zero
    // start latency) while still sleeping during long idle gaps so CPU isn't
    // burned. Configured once at pool creation from ORT_FUSIONCPU_SPIN_BUDGET_MS
    // (default 2 ms; 0 disables the spin). Tune down if prefill/power regress.
    //

    const int64_t spin_ticks = FusionCpuSpinTicks();

    //
    // Set worker thread priority.
    //

    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

    for (;;) {
        fusion_body_fn fn;
        void* ctx;
        int nways;

        //
        // Adaptive spin: poll the (lock-free) generation for new work before
        // taking the lock and blocking on the condition variable. This is only
        // a latency hint -- the authoritative check happens under wake_srw
        // below, so a stale/racy read here cannot cause incorrect execution.
        //

        if (spin_ticks > 0) {
            LARGE_INTEGER now;
            QueryPerformanceCounter(&now);
            const int64_t deadline = now.QuadPart + spin_ticks;
            int spin = 0;
            for (;;) {
                if (pool->stop || (pool->generation != generation)) break;
                if (((++spin) & 1023) == 0) {
                    QueryPerformanceCounter(&now);
                    if (now.QuadPart >= deadline) break;
                }
                _mm_pause();
            }
        }

        RtlAcquireSRWLockExclusive(&pool->wake_srw);
        do {
            if (pool->stop || (pool->generation != generation)) {
                fn = pool->body_fn;
                ctx = pool->body_ctx;
                nways = pool->nways;
                generation = pool->generation;
                break;
            }

            RtlSleepConditionVariableSRW(&pool->wake_cv, &pool->wake_srw, NULL, 0);

        } while (TRUE);

        RtlReleaseSRWLockExclusive(&pool->wake_srw);

        if (pool->stop) {
            return 0;
        }

        if (id < nways) {
            fn(ctx, id, nways);
        }

        InterlockedExchangeAdd(&pool->remaining, -1);
    }
}

#endif // defined(__NEW_CODE__)
