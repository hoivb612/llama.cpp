#define _CRT_SECURE_NO_DEPRECATE // Disables "unsafe" warnings on Windows
#define _USE_MATH_DEFINES // For M_PI on MSVC

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu-traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "ggml-cpu-quants.h"
#include "ggml-threading.h"

#include "unary-ops.h"
#include "ops.h"
#include "binary-ops.h"
#include "vec-b612.h"

#include "ggml.h"

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h> // using malloc.h with MSC/MINGW
#elif !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
#include <alloca.h>
#endif

#include <assert.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <signal.h>
#if defined(__gnu_linux__)
#include <syscall.h>
#endif

// #if defined(GGML_USE_OPENMP)
#include <omp.h>
// #endif

#if defined(__ARM_FEATURE_SVE) || defined(__ARM_FEATURE_MATMUL_INT8)
#undef GGML_USE_LLAMAFILE
#endif

#ifdef GGML_USE_LLAMAFILE
#include "llamafile/sgemm.h"
#endif

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)

// disable POSIX deprecation warnings
// these functions are never going away, anyway
#pragma warning(disable: 4996)

// unreachable code because of multiple instances of code after GGML_ABORT
#pragma warning(disable: 4702)
#endif

// Note: once we move threading into a separate C++ file
// will use std::hardware_destructive_interference_size instead of hardcoding it here
// and we'll use C++ attribute syntax.
#define GGML_CACHE_LINE  64

#if defined(__clang__) || defined(__GNUC__)
#define GGML_CACHE_ALIGN __attribute__((aligned(GGML_CACHE_LINE)))
#endif

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define GGML_TSAN_ENABLED 1
#endif
#else  // __has_feature
#if defined(__SANITIZE_THREAD__)
#define GGML_TSAN_ENABLED 1
#endif
#endif // __has_feature

#define UNUSED GGML_UNUSED
#define SWAP(x, y, T) do { T SWAP = x; (x) = y; (y) = SWAP; } while (0)

#if defined(__ARM_ARCH)
struct ggml_arm_arch_features_type {
    int has_neon;
    int has_dotprod;
    int has_i8mm;
    int has_sve;
    int sve_cnt;
    int has_sme;
} ggml_arm_arch_features = {-1, -1, -1, -1, 0, -1};
#endif


#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#include <windows.h>

#if defined(_MSC_VER) && !defined(__clang__)
#define GGML_CACHE_ALIGN __declspec(align(GGML_CACHE_LINE))

typedef volatile LONG atomic_int;
typedef atomic_int atomic_bool;
typedef atomic_int atomic_flag;

#define ATOMIC_FLAG_INIT 0

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

static void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
static void atomic_store_explicit(atomic_int * ptr, LONG val, memory_order mo) {
    // TODO: add support for explicit memory order
    InterlockedExchange(ptr, val);
}
static LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_load_explicit(atomic_int * ptr, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedCompareExchange(ptr, 0, 0);
}
static LONG atomic_fetch_add(atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
static LONG atomic_fetch_add_explicit(atomic_int * ptr, LONG inc, memory_order mo) {
    // TODO: add support for explicit memory order
    return InterlockedExchangeAdd(ptr, inc);
}
static atomic_bool atomic_flag_test_and_set(atomic_flag * ptr) {
    return InterlockedExchange(ptr, 1);
}
static void atomic_flag_clear(atomic_flag * ptr) {
    InterlockedExchange(ptr, 0);
}
static void atomic_thread_fence(memory_order mo) {
    MemoryBarrier();
}
#else // clang
#include <stdatomic.h>
#endif

typedef HANDLE pthread_t;

typedef DWORD thread_ret_t;
static int pthread_create(pthread_t * out, void * unused, thread_ret_t(*func)(void *), void * arg) {
    (void) unused;
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL)
    {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int pthread_join(pthread_t thread, void * unused) {
    (void) unused;
    int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}

static int sched_yield (void) {
    Sleep (0);
    return 0;
}
#else

#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

typedef void * thread_ret_t;

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif // _WIN32

typedef pthread_t ggml_thread_t;

#if defined(__APPLE__)
#include <unistd.h>
#include <mach/mach.h>
#include <TargetConditionals.h>
#endif

void ggml_vec_dot_bf16_f32(const int n, float * restrict s, size_t bs, const ggml_bf16_t * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);
void ggml_vec_dot_f16_f32(const int64_t n, float * restrict s, size_t bs, const ggml_fp16_t * restrict x, size_t bx, const float * restrict y, size_t by, int nrc);
void ggml_bf16_to_fp32_row_cpu(const ggml_bf16_t * x, float * y, int64_t n);
void ggml_fp32_to_bf16_row_cpu(const float * x, ggml_bf16_t * y, int64_t n);
void ggml_fp16_to_fp32_row_cpu(const ggml_fp16_t * x, float * y, int64_t n);
void ggml_fp32_to_fp16_row_cpu(const float * x, ggml_fp16_t * y, int64_t n);
void dequantize_row_q2_K_cpu(const block_q2_K * restrict x, float * restrict y, int64_t k);
void dequantize_row_q3_K_cpu(const block_q3_K * restrict x, float * restrict y, int64_t k);
void dequantize_row_q4_0_cpu(const block_q4_0 * restrict x, float * restrict y, int64_t k);
void dequantize_row_q4_K_cpu(const block_q4_K * restrict x, float * restrict y, int64_t k);
void dequantize_row_q6_K_cpu(const block_q6_K * restrict x, float * restrict y, int64_t k);
void dequantize_row_q8_0_cpu(const block_q8_0 * restrict x, float * restrict y, int64_t k);
void dequantize_row_q8_K_cpu(const block_q8_K * restrict x, float * restrict y, int64_t k);

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
    [GGML_TYPE_F32] = {
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
#if !defined(GGML_B612)
        .from_float               = (ggml_from_float_t) ggml_fp32_to_fp16_row,
#else        
        .from_float               = (ggml_from_float_t) ggml_fp32_to_fp16_row_cpu,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row_cpu,
#endif
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .from_float               = quantize_row_q4_0,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .from_float               = quantize_row_q4_1,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q5_0] = {
        .from_float               = quantize_row_q5_0,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .from_float               = quantize_row_q5_1,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .from_float               = quantize_row_q8_0,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .from_float               = quantize_row_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .from_float               = quantize_row_q2_K,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .from_float               = quantize_row_q3_K,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .from_float               = quantize_row_q4_K,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_K] = {
        .from_float               = quantize_row_q5_K,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .from_float               = quantize_row_q6_K,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K_cpu,
#endif
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
        //.from_float               = quantize_row_iq3_xxs,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        //.from_float               = quantize_row_iq3_s,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        //.from_float               = quantize_row_iq2_s,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_M] = {
        .from_float               = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .from_float               = quantize_row_iq4_nl,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .from_float               = quantize_row_iq4_xs,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .from_float               = quantize_row_q8_K,
#if defined(GGML_B612)
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_K_cpu,
#endif
    },
    [GGML_TYPE_BF16] = {
#if !defined(GGML_B612)
        .from_float               = (ggml_from_float_t) ggml_fp32_to_bf16_row,
#else
        .from_float               = (ggml_from_float_t) ggml_fp32_to_bf16_row_cpu,
        .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row_cpu,
#endif
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_bf16,
        .vec_dot_type             = GGML_TYPE_BF16,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ1_0] = {
        .from_float               = quantize_row_tq1_0,
        .vec_dot                  = ggml_vec_dot_tq1_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0] = {
        .from_float               = quantize_row_tq2_0,
        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
};

const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

//
// Threading defs
//

typedef pthread_t          ggml_thread_t;

#if defined(_WIN32)

typedef CONDITION_VARIABLE ggml_cond_t;
typedef SRWLOCK            ggml_mutex_t;

#define ggml_mutex_init(m)   InitializeSRWLock(m)
#define ggml_mutex_destroy(m)
#define ggml_mutex_lock(m)   AcquireSRWLockExclusive(m)
#define ggml_mutex_unlock(m) ReleaseSRWLockExclusive(m)
#define ggml_mutex_lock_shared(m)   AcquireSRWLockShared(m)
#define ggml_mutex_unlock_shared(m) ReleaseSRWLockShared(m)

#define ggml_cond_init(c)    InitializeConditionVariable(c)
#define ggml_cond_destroy(c)
#define ggml_cond_wait(c, m) SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_SHARED)
#define ggml_cond_broadcast(c) WakeAllConditionVariable(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

typedef pthread_cond_t     ggml_cond_t;
typedef pthread_mutex_t    ggml_mutex_t;

#define ggml_mutex_init(m)          pthread_mutex_init(m, NULL)
#define ggml_mutex_destroy(m)       pthread_mutex_destroy(m)
#define ggml_mutex_lock(m)          pthread_mutex_lock(m)
#define ggml_mutex_unlock(m)        pthread_mutex_unlock(m)
#define ggml_mutex_lock_shared(m)   pthread_mutex_lock(m)
#define ggml_mutex_unlock_shared(m) pthread_mutex_unlock(m)

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))
#define ggml_lock_lock(x)    _mm_pause()
#else
#define ggml_lock_lock(x)    UNUSED(x)
#endif
#define ggml_lock_unlock(x)  UNUSED(x)

#define GGML_LOCK_INITIALIZER 0
#define ggml_cond_init(c)      pthread_cond_init(c, NULL)
#define ggml_cond_destroy(c)   pthread_cond_destroy(c)
#define ggml_cond_wait(c, m)   pthread_cond_wait(c, m)
#define ggml_cond_broadcast(c) pthread_cond_broadcast(c)

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

// Threadpool def
struct ggml_threadpool {
    ggml_mutex_t mutex;       // mutex for cond.var
    ggml_cond_t  cond;        // cond.var for waiting for new work

    struct ggml_cgraph * cgraph;
    struct ggml_cplan  * cplan;

    // synchronization primitives
    atomic_int n_graph;       // incremented when there is work to be done (i.e each graph)
    atomic_int GGML_CACHE_ALIGN n_barrier;
    atomic_int GGML_CACHE_ALIGN n_barrier_passed;
    atomic_int GGML_CACHE_ALIGN current_chunk; // currently processing chunk during Mat_Mul, shared between all the threads.

    // these are atomic as an annotation for thread-sanitizer
    atomic_bool stop;         // Used for stopping the threadpool altogether
    atomic_bool pause;        // Used for pausing the threadpool or individual threads
    atomic_int abort;         // Used for aborting processing of a graph

    struct ggml_compute_state * workers;   // per thread state
    int          n_threads_max; // number of threads in the pool
    atomic_int   n_threads_cur; // number of threads used in the current graph

    int32_t      prio;        // Scheduling priority
    uint32_t     poll;        // Polling level (0 - no polling)

    enum ggml_status ec;
};

// Per-thread state
struct ggml_compute_state {
// #if !defined(GGML_USE_OPENMP)
    ggml_thread_t thrd;
    bool cpumask[GGML_MAX_N_THREADS];
    int  last_graph;
    bool pending;
// #endif // GGML_USE_OPENMP
    struct ggml_threadpool * threadpool;
    int ith;
};

//
// fundamental operations -> vec.h / vec.cpp
//

// ggml_vec_norm_f32                        -> vec.cpp
// ggml_vec_sqr_f32                         -> vec.cpp
// ggml_vec_sqr_f16                         -> vec.cpp
// ggml_vec_sqrt_f32                        -> vec.cpp
// ggml_vec_sqrt_f16                        -> vec.cpp
// ggml_vec_log_f32                         -> vec.cpp
// ggml_vec_log_f16                         -> vec.cpp
// ggml_vec_sin_f32                         -> vec.cpp
// ggml_vec_sin_f16                         -> vec.cpp
// ggml_vec_cos_f32                         -> vec.cpp
// ggml_vec_cos_f16                         -> vec.cpp
// ggml_vec_abs_f32                         -> vec.cpp
// ggml_vec_abs_f16                         -> vec.cpp
// ggml_vec_sgn_f32                         -> vec.cpp
// ggml_vec_sgn_f16                         -> vec.cpp
// ggml_vec_step_f32                        -> vec.cpp
// ggml_vec_step_f16                        -> vec.cpp
// ggml_vec_tanh_f32                        -> vec.cpp
// ggml_vec_tanh_f16                        -> vec.cpp
// ggml_vec_elu_f32                         -> vec.cpp
// ggml_vec_elu_f16                         -> vec.cpp
// ggml_vec_relu_f32                        -> vec.cpp
// ggml_vec_relu_f16                        -> vec.cpp
// ggml_vec_leaky_relu_f32                  -> vec.cpp
// ggml_vec_leaky_relu_f16                  -> vec.cpp
// ggml_vec_sigmoid_f32                     -> vec.cpp
// ggml_vec_sigmoid_f16                     -> vec.cpp

// TODO: optimize performance

// ggml_vec_hardswish_f32                   -> vec.h
// ggml_vec_hardswish_f16                   -> vec.h

// ggml_vec_hardsigmoid_f32                 -> vec.h
// ggml_vec_hardsigmoid_f16                 -> vec.h

// ggml_vec_exp_f32                         -> vec.h
// ggml_vec_exp_f16                         -> vec.h

// ggml_gelu_f32                            -> vec.h

// ggml_vec_gelu_f16                        -> vec.h

// ggml_gelu_quick_f32                      -> vec.h

//inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

// ggml_vec_gelu_quick_f32                  -> vec.h

// ggml_vec_gelu_quick_f16                  -> vec.h

// ggml_silu_f32                            -> vec.h
// ggml_silu_f16                            -> vec.h

// ggml_v_expf                              -> vec.h
// ggml_v_silu                              -> vec.h

// ggml_vec_silu_f32                        -> vec.cpp

// ggml_vec_silu_f16                        -> vec.h

// ggml_vec_soft_max_f32                    -> vec.cpp

// ggml_vec_log_soft_max_f32                -> vec.cpp

// ggml_silu_backward_f32                   -> vec.h
// ggml_silu_backward_f16                   -> vec.h

// ggml_vec_silu_backward_f32               -> vec.h
// ggml_vec_silu_backward_f16               -> vec.h

// ggml_vec_sum_f32                         -> vec.cpp

// ggml_vec_max_f32                         -> vec.cpp

// ggml_vec_norm_inv_f32                    -> vec.h

// ggml_vec_argmax_f32                      -> vec.h

void ggml_bf16_to_fp32_row_cpu(const ggml_bf16_t * x, float * y, int64_t n) {

    const uint64_t nc = n;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_bf16_to_fp32_row_cpu")
    __m256i ax[GGML_F16_ARR];
    __m512i ay[GGML_F16_ARR];

    const uint64_t np = (nc & ~(GGML_F16_STEP16 - 1));

    for (; i < np; i += GGML_F16_STEP16) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = _mm256_loadu_si256((__m256i *)(x + i + j * GGML_F16_EPR16));
            ay[j] = _mm512_cvtepu16_epi32(ax[j]);
            ay[j] = _mm512_slli_epi32(ay[j], 16);
            _mm512_storeu_si512((y + i + j * GGML_F16_EPR16), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F16_EPR16 - 1));

    for (; i < xn; i += GGML_F16_EPR16) {
        ax[0] = _mm256_loadu_si256((__m256i *)(x + i));
        ay[0] = _mm512_cvtepu16_epi32(ax[0]);
        ay[0] = _mm512_slli_epi32(ay[0], 16);
        _mm512_storeu_si512((y + i), ay[0]); 
    }

    // leftovers

    if (nc & (GGML_F16_EPR16 - 1)) {
        do {
            y[i] = GGML_BF16_TO_FP32(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)
#pragma message("Building AVX2 ggml_bf16_to_fp32_row_cpu")

    __m128i ax[GGML_F16_ARR];
    __m256i ay[GGML_F16_ARR];

    const uint64_t np = (nc & ~(GGML_F16_STEP - 1));

    for (; i < np; i += GGML_F16_STEP) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = _mm_loadu_si128((__m128i *)(x + i + j * GGML_F16_EPR));
            ay[j] = _mm256_cvtepu16_epi32(ax[j]);
            ay[j] = _mm256_slli_epi32(ay[j], 16);
            _mm256_storeu_si256((__m256i *)(y + i + j * GGML_F16_EPR), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F16_EPR - 1));

    for (; i < xn; i += GGML_F16_EPR) {
        ax[0] = _mm_loadu_si128((__m128i *)(x + i));
        ay[0] = _mm256_cvtepu16_epi32(ax[0]);
        ay[0] = _mm256_slli_epi32(ay[0], 16);
        _mm256_storeu_si256((__m256i *)(y + i), ay[0]); 
    }

    // leftovers

    if (nc & (GGML_F16_EPR - 1)) {
        do {
            y[i] = GGML_BF16_TO_FP32(x[i]);
            i += 1;
        } while (i < nc);
    }

#else
#pragma message("Building Scalar ggml_bf16_to_fp32_row_cpu")

    for (; i < nc; i++) {
        y[i] = GGML_BF16_TO_FP32(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_fp32_to_bf16_row_cpu(const float * x, ggml_bf16_t * y, int64_t n) {

    const uint64_t nc = n;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_fp32_to_bf16_row_cpu")

    __m512 ax;
    __m512 bx;
    __m512bh ay;
    __m256bh by;

    const uint64_t np = (nc & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        ax = _mm512_loadu_ps(x + i + 16);
        bx = _mm512_loadu_ps(x + i + 0);
        ay = _mm512_cvtne2ps_pbh(ax, bx);
        _mm512_storeu_si512((__m512i *)(y + i + 0), ay);

        ax = _mm512_loadu_ps(x + i + 48);
        bx = _mm512_loadu_ps(x + i + 32);
        ay = _mm512_cvtne2ps_pbh(ax, bx);
        _mm512_storeu_si512((__m512i *)(y + i + 32), ay); 
    }

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax = _mm512_loadu_ps(x + i);
        by = _mm512_cvtneps_pbh(ax);
        _mm256_storeu_si256((__m256i *)(y + i), by); 
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] = GGML_FP32_TO_BF16(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__) && !defined(__gnu_linux__)
#pragma message("Building AVX2 ggml_fp32_to_bf16_row_cpu")

    __m256 ax[GGML_F32_ARR];
    __m128bh ay[GGML_F32_ARR];

    const uint64_t np = (nc & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ay[j] = _mm256_cvtneps_pbh(ax[j]);
            _mm_storeu_si128((__m128i *)(y + i + j * GGML_F32_EPR), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ay[0] = _mm256_cvtneps_pbh(ax[0]);
        _mm_storeu_si128((__m128i *)(y + i), ay[0]); 
    }

    // leftovers

    if (nc& (GGML_F32_EPR - 1)) {
        do {
            y[i] = GGML_FP32_TO_BF16(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__gnu_linux__)
#pragma message("Building Linux ggml_fp32_to_bf16_row_cpu")

#if defined(__AVX512BF16__)
    // subnormals are flushed to zero on this platform
    for (; i + 32 <= n; i += 32) {
        _mm512_storeu_si512(
            (__m512i *)(y + i),
            m512i(_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16),
                                _mm512_loadu_ps(x + i))));
    }
#endif
    for (; i < n; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }

#else

    for (; i < nc; i++) {
        y[i] = GGML_FP32_TO_BF16(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_fp16_to_fp32_row_cpu(const ggml_fp16_t * x, float * y, int64_t n) {

    const uint64_t nc = n;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_fp16_to_fp32_row_cpu")

    __m256i ax[GGML_F16_ARR];
    __m512 ay[GGML_F16_ARR];

    const uint64_t np = (nc & ~(GGML_F16_STEP16 - 1));

    for (; i < np; i += GGML_F16_STEP16) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = _mm256_loadu_si256((__m256i *)(x + i + j * GGML_F16_EPR16));
            ay[j] = _mm512_cvtph_ps(ax[j]);
            _mm512_storeu_ps((y + i + j * GGML_F16_EPR16), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F16_EPR16 - 1));

    for (; i < xn; i += GGML_F16_EPR16) {
        ax[0] = _mm256_loadu_si256((__m256i *)(x + i));
        ay[0] = _mm512_cvtph_ps(ax[0]);
        _mm512_storeu_ps((y + i), ay[0]); 
    }

    // leftovers

    if (nc & (GGML_F16_EPR16 - 1)) {
        do {
            y[i] = GGML_FP16_TO_FP32(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)

    __m128i ax[GGML_F16_ARR];
    __m256 ay[GGML_F16_ARR];

    const uint64_t np = (nc & ~(GGML_F16_STEP - 1));

    for (; i < np; i += GGML_F16_STEP) {
        for (uint64_t j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = _mm_loadu_si128((__m128i *)(x + i + j * GGML_F16_EPR));
            ay[j] = _mm256_cvtph_ps(ax[j]);
            _mm256_storeu_ps((y + i + j * GGML_F16_EPR), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F16_EPR - 1));

    for (; i < xn; i += GGML_F16_EPR) {
        ax[0] = _mm_loadu_si128((__m128i *)(x + i));
        ay[0] = _mm256_cvtph_ps(ax[0]);
        _mm256_storeu_ps((y + i), ay[0]); 
    }

    // leftovers

    if (nc & (GGML_F16_EPR - 1)) {
        do {
            y[i] = GGML_FP16_TO_FP32(x[i]);
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; i++) {
        y[i] = GGML_FP16_TO_FP32(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void ggml_fp32_to_fp16_row_cpu(const float * x, ggml_fp16_t * y, int64_t n) {

    const uint64_t nc = n;
    uint64_t i = 0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F ggml_fp32_to_fp16_row_cpu")

    __m512 ax[GGML_F32_ARR];
    __m256i ay[GGML_F32_ARR];

    const uint64_t np = (nc & ~(GGML_F32_STEP16 - 1));

    for (; i < np; i += GGML_F32_STEP16) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm512_loadu_ps(x + i + j * GGML_F32_EPR16);
            ay[j] = _mm512_cvtps_ph(ax[j], _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i *)(y + i + j * GGML_F32_EPR16), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F32_EPR16 - 1));

    for (; i < xn; i += GGML_F32_EPR16) {
        ax[0] = _mm512_loadu_ps(x + i);
        ay[0] = _mm512_cvtps_ph(ax[0], _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), ay[0]); 
    }

    // leftovers

    if (nc & (GGML_F32_EPR16 - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(x[i]);
            i += 1;
        } while (i < nc);
    }

#elif defined(__AVX2__)

    __m256 ax[GGML_F32_ARR];
    __m128i ay[GGML_F32_ARR];

    const uint64_t np = (nc & ~(GGML_F32_STEP - 1));

    for (; i < np; i += GGML_F32_STEP) {
        for (uint64_t j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = _mm256_loadu_ps(x + i + j * GGML_F32_EPR);
            ay[j] = _mm256_cvtps_ph(ax[j], _MM_FROUND_TO_NEAREST_INT);
            _mm_storeu_si128((__m128i *)(y + i + j * GGML_F32_EPR), ay[j]); 
        }
    }

    const uint64_t xn = (nc & ~(GGML_F32_EPR - 1));

    for (; i < xn; i += GGML_F32_EPR) {
        ax[0] = _mm256_loadu_ps(x + i);
        ay[0] = _mm256_cvtps_ph(ax[0], _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), ay[0]); 
    }

    // leftovers

    if (nc& (GGML_F32_EPR - 1)) {
        do {
            y[i] = GGML_FP32_TO_FP16(x[i]);
            i += 1;
        } while (i < nc);
    }

#else

    for (; i < nc; i++) {
        y[i] = GGML_FP32_TO_FP16(x[i]);
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

// New versions replacing the C versions in ggml/src/ggml-quants.c

void dequantize_row_q2_K_cpu(const block_q2_K * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK_K;
    assert(k % qk == 0);

    const uint64_t nb = k / qk;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q2_K_cpu")

    const __m128 zero128 = _mm_setzero_ps();

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin);

        const __m512 dv = _mm512_broadcastss_ps(_mm_load_ss(&d));
        const __m128 dmin_ss = _mm_load_ss(&dmin);

        const uint64_t * q = (const uint64_t *)x[i].qs;
    
        for (int64_t l = 0; l < QK_K / 128; l++) {

            for (int64_t j = 0; j < 4; j++) {
                __m128i qli;
                __m128 ml;
                __m512 mv;
                __m512 yv;

                uint32_t sc = x[i].scales[l * 8 + j * 2];
                uint64_t scale = sc & 0xF;

                ml = _mm_mul_ss(dmin_ss, _mm_cvti32_ss(zero128, sc >> 4));
                mv = _mm512_broadcastss_ps(ml);
        
                qli = _mm_cvtsi64_si128(((q[0] >> (j * 2)) & 0x0303030303030303) * scale);
                qli = _mm_insert_epi64(qli,
                                       ((q[1] >> (j * 2)) & 0x0303030303030303) * scale,
                                       1);

                yv = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qli));
                _mm512_storeu_ps(y, _mm512_fmsub_ps(dv, yv, mv));

                sc = x[i].scales[l * 8 + j * 2 + 1];
                scale = sc & 0xF;

                ml = _mm_mul_ss(dmin_ss, _mm_cvti32_ss(zero128, sc >> 4));
                mv = _mm512_broadcastss_ps(ml);

                qli = _mm_cvtsi64_si128(((q[2] >> (j * 2)) & 0x0303030303030303) * scale);
                qli = _mm_insert_epi64(qli,
                                       ((q[3] >> (j * 2)) & 0x0303030303030303) * scale,
                                       1);

                yv = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qli));
                _mm512_storeu_ps(y + 16, _mm512_fmsub_ps(dv, yv, mv));

                y += 32;
            }

            q += 4;
        }
    }

#elif defined(__AVX2__) 
// #elif defined(__AVX2__) && !defined(__clang__) // on AXV2 systems clang generates errors for _mm_cvti32_ss()

    const __m128 zero128 = _mm_setzero_ps();

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin);

        const __m256 dlv = _mm256_broadcast_ss(&d);
        const __m128 dmin_ss = _mm_load_ss(&dmin);

        const uint64_t * q = (const uint64_t *)x[i].qs;
    
        for (int64_t l = 0; l < QK_K / 128; l++) {

            for (int64_t j = 0; j < 4; j++) {
                __m128 ml;
                __m256 mlv;
                __m128i q1i;
                __m256 q1v;
                __m256 y1;

                uint32_t sc = x[i].scales[l * 8 + j * 2];
                uint64_t scale = sc & 0xF;
                ml = _mm_mul_ss(dmin_ss, _mm_cvti32_ss(zero128, sc >> 4));
                mlv = _mm256_broadcastss_ps(ml);
        
                for (int64_t k = 0; k < 2; k++) {
                    q1i = _mm_cvtsi64_si128(((q[k] >> (j * 2)) & 0x0303030303030303) * scale);
                    q1v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1i));
                    y1 = _mm256_fmsub_ps(dlv, q1v, mlv);
                    _mm256_storeu_ps(y + (k * 8), y1);
                }

                sc = x[i].scales[l * 8 + j * 2 + 1];
                scale = sc & 0xF;
                ml = _mm_mul_ss(dmin_ss, _mm_cvti32_ss(zero128, sc >> 4));
                mlv = _mm256_broadcastss_ps(ml);

                for (int64_t k = 0; k < 2; k++) {
                    q1i = _mm_cvtsi64_si128(((q[k + 2] >> (j * 2)) & 0x0303030303030303) * scale);
                    q1v = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1i));
                    y1 = _mm256_fmsub_ps(dlv, q1v, mlv);
                    _mm256_storeu_ps(y + (k * 8) + 16, y1); 
                }

                y += 32;
            }
    
            q += 4;
        }
    }

#else

    for (int64_t i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void dequantize_row_q3_K_cpu(const block_q3_K * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK_K;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q3_K_cpu")

    const __m128 zero128 = _mm_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(4); 

    for (uint64_t i = 0; i < nb; i++) {
        __m128 dl;
        __m512 dlv;
        __m128i q1i;
        __m128i q2i;
        __m128i q3i;
        __m512 qv;
        __m512 y1;

        const float dall = GGML_FP16_TO_FP32(x[i].d);
        const __m128 dall_ss = _mm_load_ss(&dall);

        const uint64_t * q = (const uint64_t *)x[i].qs;
        const uint64_t * hmk = (const uint64_t *)x[i].hmask;
        __m128i hm[2];

        //
        // Complement the upper bits and preshift into place.
        //

        for (uint64_t ii = 0; ii < 2; ii++) {
            hm[ii] = _mm_cvtsi64_si128(~hmk[ii * 2 + 0]);
            hm[ii] = _mm_insert_epi64(hm[ii], ~hmk[ii * 2 + 1], 1);
            hm[ii] = _mm_rol_epi32(hm[ii], 2);
        }

        //
        // Expand the scale values into a 16-byte array.
        //

        union {
            uint32_t aux[4];
            int8_t scales[16];
        } as;

        memcpy(as.aux, x[i].scales, 12);
        uint32_t tmp = as.aux[2];
        as.aux[2] = ((as.aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        as.aux[3] = ((as.aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        as.aux[0] = (as.aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        as.aux[1] = (as.aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        //
        // Process all the q3_k quant blocks.
        //

        for (int64_t n = 0; n < QK_K / 128; n++) {
            for (int j = 0; j < 4; ++j) {

                dl = _mm_mul_ss(dall_ss,
                                _mm_cvti32_ss(zero128, as.scales[n * 8 + j * 2] - 32));

                dlv = _mm512_broadcastss_ps(dl);

                //
                // Isolate the low 2-bits for 16 quant values.
                //

                q1i = _mm_cvtsi64_si128((q[0] >> (j * 2)) & 0x0303030303030303);
                q1i = _mm_insert_epi64(q1i,
                                       (q[1] >> (j * 2)) & 0x0303030303030303,
                                       1);

                //
                // Isolate the hm bits for 16 quant values and shift next bit into
                // position.
                //

                q2i = _mm_and_si128(hm[0], m4);
                hm[0] = _mm_ror_epi32(hm[0], 1);

                //
                // Sub low hm values (either 0 or 4) from the low two bits for 16 quants.
                //

                q3i = _mm_sub_epi8(q1i, q2i);

                //
                // Convert the 16 quant bytes to float and multiply by scales * d_all.
                //

                qv = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3i)); 
                y1 = _mm512_mul_ps(dlv, qv);

                //
                // Store result.
                //

                _mm512_storeu_ps(y, y1);

                dl = _mm_mul_ss(dall_ss,
                                _mm_cvti32_ss(zero128, as.scales[n * 8 + j * 2 + 1] - 32));

                dlv = _mm512_broadcastss_ps(dl);

                //
                // Isolate the low 2-bits for 16 quant values.
                //

                q1i = _mm_cvtsi64_si128((q[2] >> (j * 2)) & 0x0303030303030303);
                q1i = _mm_insert_epi64(q1i,
                                       (q[3] >> (j * 2)) & 0x0303030303030303,
                                       1);

                //
                // Isolate the hm bits for 16 quant values and shift next bit into
                // position.
                //

                q2i = _mm_and_si128(hm[1], m4);
                hm[1] = _mm_ror_epi32(hm[1], 1);

                //
                // Sub low hm values (either 0 or 4) from the low two bits for 16 quants.
                //

                q3i = _mm_sub_epi8(q1i, q2i);

                //
                // Convert the 16 quant bytes to float and multiply by scales * d_all.
                //

                qv = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3i)); 
                y1 = _mm512_mul_ps(dlv, qv);

                //
                // Store result.
                //

                _mm512_storeu_ps(y + 16, y1);

                y += 32;
            }

            q += 4;
        }
    }

#elif defined(__AVX2__) 
// #elif defined(__AVX2__) && !defined(__clang__) // on AVX2 systems clang generates error for _mm_ror_epi32()

    const __m128 zero128 = _mm_setzero_ps();
    const __m128i m4 = _mm_set1_epi8(4); 

    for (uint64_t i = 0; i < nb; i++) {
        __m128 dl;
        __m256 dlv;
        __m128i q1i;
        __m128i q2i;
        __m128i q3i;
        __m256 qv;
        __m256 y1;

        const float dall = GGML_FP16_TO_FP32(x[i].d);
        const __m128 dall_ss = _mm_load_ss(&dall);

        const uint64_t * q = (const uint64_t *)x[i].qs;
        const uint64_t * hmk = (const uint64_t *)x[i].hmask;
        __m128i hm[4];

        //
        // Complement the upper bits and preshift into place.
        //

        for (uint64_t i = 0; i < 4; i++) {
            hm[i] = _mm_cvtsi64_si128(~hmk[i]);
            hm[i] = _mm_rol_epi32(hm[i], 2);
        }

        //
        // Expand the scale values into a 16-byte array.
        //

        union {
            uint32_t aux[4];
            int8_t scales[16];
        } as;

        memcpy(as.aux, x[i].scales, 12);
        uint32_t tmp = as.aux[2];
        as.aux[2] = ((as.aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        as.aux[3] = ((as.aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        as.aux[0] = (as.aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        as.aux[1] = (as.aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        //
        // Process all the q3_k quant blocks.
        //

        for (int64_t n = 0; n < QK_K / 128; n++) {
            for (int j = 0; j < 4; ++j) {

                dl = _mm_mul_ss(dall_ss,
                                _mm_cvti32_ss(zero128, as.scales[n * 8 + j * 2] - 32));

                dlv = _mm256_broadcastss_ps(dl);

                for (int l = 0; l < 2; l++) {

                    //
                    // Isolate the low 2-bits for 8 quant values.
                    //

                    q1i = _mm_cvtsi64_si128((q[l] >> (j * 2)) & 0x0303030303030303);

                    //
                    // Isolate the hm bits for 8 quant values and shift next bit into
                    // position.
                    //

                    q2i = _mm_and_si128(hm[l], m4);
                    hm[l] = _mm_ror_epi32(hm[l], 1);

                    //
                    // Sub low hm values (either 0 or 4) from the low two bits for 8 quants.
                    //

                    q3i = _mm_sub_epi8(q1i, q2i);

                    //
                    // Convert the 8 quant bytes to float and multiply by scales * d_all.
                    //

                    qv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3i)); 
                    y1 = _mm256_mul_ps(dlv, qv);

                    //
                    // Store result.
                    //

                    _mm256_storeu_ps(y + (l * 8), y1);
                }

                dl = _mm_mul_ss(dall_ss,
                                _mm_cvti32_ss(zero128, as.scales[n * 8 + j * 2 + 1] - 32));

                dlv = _mm256_broadcastss_ps(dl);

                for (int l = 0; l < 2; l++) {

                    //
                    // Isolate the low 2-bits for 8 quant values.
                    //

                    q1i = _mm_cvtsi64_si128((q[l + 2] >> (j * 2)) & 0x0303030303030303);

                    //
                    // Isolate the hm bits for 8 quant values and shift next bit into
                    // position.
                    //

                    q2i = _mm_and_si128(hm[l + 2], m4);
                    hm[l + 2] = _mm_ror_epi32(hm[l + 2], 1);

                    //
                    // Sub low hm values (either 0 or 4) from the low two bits for 8 quants.
                    //

                    q3i = _mm_sub_epi8(q1i, q2i);

                    //
                    // Convert the 8 quant bytes to float and multiply by scales * d_all.
                    //

                    qv = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3i)); 
                    y1 = _mm256_mul_ps(dlv, qv);

                    //
                    // Store result.
                    //

                    _mm256_storeu_ps(y + (l * 8) + 16, y1);
                }

                y += 32;
            }

            q += 4;
        }
    }

#else

    uint32_t aux[4];

    const int8_t * scales = (const int8_t*)aux;

    for (uint64_t i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        float dl;
        for (int64_t n = 0; n < QK_K / 128; n++) {
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[n * 8 + j * 2] - 32);
                for (int l = 0; l < 16; ++l) {
                    y[l] = dl * ((int8_t)((q[l+ 0] >> (j * 2)) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[n * 8 + j * 2 + 1] - 32);
                for (int l = 0; l < 16; ++l) {
                    y[l + 16] = dl * ((int8_t)((q[l+16] >> (j * 2)) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                m <<= 1;
                y += 32;
            }

            q += 32;
        }
    }

#endif

}

void dequantize_row_q4_0_cpu(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK4_0;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q4_0_cpu")

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m8 = _mm_set1_epi8(8);

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const __m512 dv = _mm512_broadcastss_ps(_mm_load_ss(&d));

        //
        // Load all the q4_0 quant nibble values.
        //

        __m128i nibbles = _mm_loadu_si128((__m128i *)x[i].qs);

        for (uint64_t j = 0; j < (qk / 16); j += 1) {

            //
            // Extract the next set of nibbles and subtract out the quant bias.
            //

            __m128i nibs_8 = _mm_and_si128(nibbles, m4);
            nibs_8 = _mm_sub_epi8(nibs_8, m8);

            //
            // Convert to 16-bit nibs, to 32-bit nibs, and then to float.
            //

            const __m256i nibs_16 = _mm256_cvtepi8_epi16(nibs_8);
            const __m512i nibs_32 = _mm512_cvtepi16_epi32(nibs_16);
            const __m512 nibs_ps = _mm512_cvtepi32_ps(nibs_32);

            //
            // Multiply by quant block scale factor and store the result.
            //

            const __m512 result = _mm512_mul_ps(dv, nibs_ps);
            _mm512_storeu_ps(y + (j * 16), result);

            //
            // Shift next nibble into position.
            //

            nibbles = _mm_srli_epi16(nibbles, 4);
        }

        y += 32;
    }

#else

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (uint64_t j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void dequantize_row_q4_K_cpu(const block_q4_K * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK_K;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

    for (uint64_t i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__GEN_AVX512__))
#pragma message("Building AVX2/AVX512F dequantize_row_q4_K_cpu")

        const uint32_t kmask1 = 0x3f3f3f3f;
        const uint32_t kmask2 = 0x0f0f0f0f;
        const uint32_t kmask4 = 0xc0c0c0c0;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)

        const __m128i kmaskq = _mm_set1_epi8(0xf);

#else

        const uint64_t kmaskq = 0x0f0f0f0f0f0f0f0full;
        const __m128i qiz = _mm_setzero_si128();

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

        union {
            uint32_t utmp[4];
            struct {
                uint8_t scale[8];
                uint8_t mins[8];
            };
        } sm;

        //
        // Unpack 8 scale and 8 minimum values.
        //
        // The scale values are the first 8 bytes and the min values are the second 8 bytes.
        //

        const uint32_t * vscales = (uint32_t *)x[i].scales;
        sm.utmp[3] = ((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2);
        sm.utmp[2] = vscales[1] & kmask1;
        sm.utmp[1] = (vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2);
        sm.utmp[0] = vscales[0] & kmask1;

        //
        // Initialize pointer to q array.
        //

        const uint64_t * q = (uint64_t *)x[i].qs;

        for (int64_t j = 0; j < QK_K / 32; j += 2) {

            //
            // Compute scale and minimum values for first set of 32 quant values.
            //

            const float d1 = (float)sm.scale[j] * d;
            const float m1 = (float)sm.mins[j] * min;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)

            const __m512 d1v = _mm512_broadcastss_ps(_mm_load_ss(&d1));
            const __m512 m1v = _mm512_broadcastss_ps(_mm_load_ss(&m1));

            //
            // Dequantize first 32 quant values from the low quant nibble.
            //

            for (int64_t kk = 0; kk < 2; ++kk) {
                const __m128i q1iq = _mm_loadu_epi8(q + (kk * 2));
                const __m128i q1i = _mm_and_si128(q1iq, kmaskq);
                const __m512i q1vi = _mm512_cvtepi8_epi32(q1i);
                const __m512 q1v = _mm512_cvtepi32_ps(q1vi);
                const __m512 y1 = _mm512_fmsub_ps(d1v, q1v, m1v);
                _mm512_storeu_ps(y + (kk * 16), y1);
            }

#else

            const __m256 d1v = _mm256_broadcast_ss(&d1);
            const __m256 m1v = _mm256_broadcast_ss(&m1);

            //
            // Dequantize first 32 quant values from the low quant nibble.
            //

            for (int64_t kk = 0; kk < 4; ++kk) {
                const __m128i q1i = _mm_insert_epi64(qiz, q[kk] & kmaskq, 0);
                const __m256i q1vi = _mm256_cvtepi8_epi32(q1i);
                const __m256 q1v = _mm256_cvtepi32_ps(q1vi);
                const __m256 y1 = _mm256_fmsub_ps(d1v, q1v, m1v);
                _mm256_storeu_ps(y + (kk * 8), y1);
            }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

            y += 32;

            //
            // Compute scale and minimum values for second set of 32 quant values.
            //

            const float d2 = (float)sm.scale[j + 1] * d;
            const float m2 = (float)sm.mins[j + 1] * min;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)

            const __m512 d2v = _mm512_broadcastss_ps(_mm_load_ss(&d2));
            const __m512 m2v = _mm512_broadcastss_ps(_mm_load_ss(&m2));

            //
            // Dequantize first 32 quant values from the high quant nibble.
            //

            for (int64_t kk = 0; kk < 2; ++kk) {
                const __m128i q2iq = _mm_loadu_epi8(q + (kk * 2));
                const __m128i q2is = _mm_srli_epi16(q2iq, 4);
                const __m128i q2i = _mm_and_si128(q2is, kmaskq);
                const __m512i q2vi = _mm512_cvtepi8_epi32(q2i);
                const __m512 q2v = _mm512_cvtepi32_ps(q2vi);
                const __m512 y2 = _mm512_fmsub_ps(d2v, q2v, m2v);
                _mm512_storeu_ps(y + (kk * 16), y2);
            }

#else

            const __m256 d2v = _mm256_broadcast_ss(&d2);
            const __m256 m2v = _mm256_broadcast_ss(&m2);

            //
            // Dequantize second 32 quant values from the high quant nibble.
            //

            for (int64_t kk = 0; kk < 4; ++kk) {
                const __m128i q2i = _mm_insert_epi64(qiz, (q[kk] >> 4) & kmaskq, 0);
                const __m256i q2vi = _mm256_cvtepi8_epi32(q2i);
                const __m256 q2v = _mm256_cvtepi32_ps(q2vi);
                const __m256 y2 = _mm256_fmsub_ps(d2v, q2v, m2v);
                _mm256_storeu_ps(y + (kk * 8), y2);
            }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

            q += 4;
            y += 32;
        }

#else

        const uint8_t * q = x[i].qs;

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }

#endif // defined(__AVX2__) || (defined(__AVX512F__) && defined(__GEN_AVX512__))

    }
}

void dequantize_row_q6_K_cpu(const block_q6_K * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK_K;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q6_K_cpu")

    static const uint16_t k_perm[8][32] = {
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    };

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x30);
    const __m256i m32s = _mm256_set1_epi8(32);
    const __m512i zero512i = _mm512_setzero_si512();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
	    const __m512 dv = _mm512_broadcastss_ps(_mm_load_ss(&d));

     	// 
     	// Set up pointers to the low 4-bit values and the high 2-bit values.
     	//

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;

    	//
    	// Load the 8-bit scale values and convert to 16-bits scale values.
    	//

        const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
	    const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);
	    const __m512i scales_all = _mm512_inserti32x8(zero512i, scales16, 0);

        for (uint64_t j = 0; j < QK_K / 128; ++j) {

    	    //
            // Load the low 4-bit values and the high 2-bit values.
    	    //
    
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)(ql + (j * 64) + 0));
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)(ql + (j * 64) + 32));
            const __m256i q2bitsH = _mm256_loadu_si256((const __m256i*)(qh + (j * 32)));
    
    	    //
    	    // Unpack the high (<5:4>) 2-bit values.
    	    //
    
            const __m256i q6h_0 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 4), m2);
            const __m256i q6h_1 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 2), m2);
            const __m256i q6h_2 = _mm256_and_si256(q2bitsH, m2);
            const __m256i q6h_3 = _mm256_and_si256(_mm256_ror_epi64(q2bitsH, 2), m2);
    	
    	    //
            // Unpack the low (<3:0>) 4-bit values and or in the high 2-bit values.
    	    //
    
            __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q6h_0);
            __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q6h_1);
            __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q6h_2);
            __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q6h_3);
    
    	    //
    	    // Subtract out the bias values from the quant values.
    	    //
    	
    	    q6_0 = _mm256_sub_epi8(q6_0, m32s);
    	    q6_1 = _mm256_sub_epi8(q6_1, m32s);
    	    q6_2 = _mm256_sub_epi8(q6_2, m32s);
    	    q6_3 = _mm256_sub_epi8(q6_3, m32s);
    
    	    //
    	    // Convert the 8-bit quant values to 16-bit quant values.
    	    //
    
    	    __m512i q6_0_16 = _mm512_cvtepi8_epi16(q6_0);
    	    __m512i q6_1_16 = _mm512_cvtepi8_epi16(q6_1);
    	    __m512i q6_2_16 = _mm512_cvtepi8_epi16(q6_2);
    	    __m512i q6_3_16 = _mm512_cvtepi8_epi16(q6_3);	    
    
    	    //
    	    // Load the scale selection index values.
    	    //
    
    	    const __m512i idx0 = _mm512_loadu_si512((__m512i *)&k_perm[(j * 4) + 0][0]);
    	    const __m512i idx1 = _mm512_loadu_si512((__m512i *)&k_perm[(j * 4) + 1][0]);
    	    const __m512i idx2 = _mm512_loadu_si512((__m512i *)&k_perm[(j * 4) + 2][0]);
    	    const __m512i idx3 = _mm512_loadu_si512((__m512i *)&k_perm[(j * 4) + 3][0]);
    
    	    //
    	    // Select the scale values.
    	    //
     
    	    const __m512i v_scale0 = _mm512_permutexvar_epi16(idx0, scales_all);
    	    const __m512i v_scale1 = _mm512_permutexvar_epi16(idx1, scales_all);
    	    const __m512i v_scale2 = _mm512_permutexvar_epi16(idx2, scales_all);
    	    const __m512i v_scale3 = _mm512_permutexvar_epi16(idx3, scales_all);
    
    	    //
    	    // Multiply the 16-bit quant values by the 16-bit scale values.
    	    //
    	    // N.B. This product cannot overflow 16-bits, and thus, only the low 16-bits
            //      are retained.
    	    //
    
    	    q6_0_16 = _mm512_mullo_epi16(q6_0_16, v_scale0);
    	    q6_1_16 = _mm512_mullo_epi16(q6_1_16, v_scale1);
    	    q6_2_16 = _mm512_mullo_epi16(q6_2_16, v_scale2);
    	    q6_3_16 = _mm512_mullo_epi16(q6_3_16, v_scale3);
    
    	    //
    	    // Split the 64-byte quant values into two 32-byte quant values so they can be
    	    // converted to floating. 
    	    // 
                
           	const __m256i q6_0_16l = _mm512_castsi512_si256(q6_0_16);
    	    const __m256i q6_0_16h = _mm512_extracti32x8_epi32(q6_0_16, 1);
    
    	    const __m256i q6_1_16l = _mm512_castsi512_si256(q6_1_16);
    	    const __m256i q6_1_16h = _mm512_extracti32x8_epi32(q6_1_16, 1);
    	    
    	    const __m256i q6_2_16l = _mm512_castsi512_si256(q6_2_16);
            const __m256i q6_2_16h = _mm512_extracti32x8_epi32(q6_2_16, 1);
    
    	    const __m256i q6_3_16l = _mm512_castsi512_si256(q6_3_16);
     	    const __m256i q6_3_16h = _mm512_extracti32x8_epi32(q6_3_16, 1);
    
    	    //
    	    // Convert low and high parts to floating.
            //
    
    	    __m512 q6_0_psl = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_0_16l));
    	    __m512 q6_0_psh = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_0_16h));
    
    	    __m512 q6_1_psl = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_1_16l));
    	    __m512 q6_1_psh = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_1_16h)); 
    
    	    __m512 q6_2_psl = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_2_16l));
    	    __m512 q6_2_psh = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_2_16h));
    
    	    __m512 q6_3_psl = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_3_16l));
    	    __m512 q6_3_psh = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q6_3_16h));
    
    	    //
    	    // Scale floating values.
    	    //
    
    	    q6_0_psl = _mm512_mul_ps(dv, q6_0_psl);
            q6_0_psh = _mm512_mul_ps(dv, q6_0_psh);	    
    	    
    	    q6_1_psl = _mm512_mul_ps(dv, q6_1_psl);
            q6_1_psh = _mm512_mul_ps(dv, q6_1_psh);
    
    	    q6_2_psl = _mm512_mul_ps(dv, q6_2_psl);
            q6_2_psh = _mm512_mul_ps(dv, q6_2_psh);
    
    	    q6_3_psl = _mm512_mul_ps(dv, q6_3_psl);
            q6_3_psh = _mm512_mul_ps(dv, q6_3_psh);
    
    	    //
    	    // Store the resultant 8x16 floating values.
    	    //
    
    	    _mm512_storeu_ps(y + 0, q6_0_psl);
    	    _mm512_storeu_ps(y + 16, q6_0_psh);
    
            _mm512_storeu_ps(y + 32, q6_1_psl);
    	    _mm512_storeu_ps(y + 48, q6_1_psh);
    
    	    _mm512_storeu_ps(y + 64, q6_2_psl);
    	    _mm512_storeu_ps(y + 80, q6_2_psh);
    	    
    	    _mm512_storeu_ps(y + 96, q6_3_psl);
    	    _mm512_storeu_ps(y + 112, q6_3_psh);
    
    	    y += 128;
        }
    }

#else

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict ql = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void dequantize_row_q8_0_cpu(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK8_0;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q8_0_cpu")

    for (uint64_t i = 0; i < nb; i++) {
        __m512 d = _mm512_set1_ps(GGML_FP16_TO_FP32(x[i].d));
        __m128i qs;
        __m512 qp;

         qs = _mm_loadu_si128((__m128i *)&x[i].qs[0]);
         qp = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qs));
         qp = _mm512_mul_ps(qp, d);
         _mm512_storeu_ps(y, qp);

         qs = _mm_loadu_si128((__m128i *)&x[i].qs[16]);
         qp = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qs));
         qp = _mm512_mul_ps(qp, d);
         _mm512_storeu_ps(y + 16, qp);

        y += qk;
    }

#elif defined(__AVX2__)

    for (uint64_t i = 0; i < nb; i++) {
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[i].d));
        __m128i qs[4];
        __m256 qp[4];

        for (uint64_t j = 0; j < (qk / 8); j++) {
            qs[j] = _mm_loadu_si64((__m128i *)&x[i].qs[j * 8]);
            qp[j] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qs[j]));
            qp[j] = _mm256_mul_ps(qp[j], d);
            _mm256_storeu_ps(y + j * 8, qp[j]);
        }

        y += qk;
    }

#else

    for (uint64_t i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

void dequantize_row_q8_K_cpu(const block_q8_K * restrict x, float * restrict y, int64_t k) {
    const uint64_t qk = QK_K;

    assert(k % qk == 0);

    const uint64_t nb = k / qk;

#if defined(__AVX512F__) && defined(__GEN_AVX512__)
#pragma message("Building AVX512F dequantize_row_q8_K_cpu")

    for (uint64_t i = 0; i < nb; i++) {
        const __m512 d = _mm512_set1_ps(x[i].d);
        __m128i qs[4];
        __m512 qp[4];

        for (uint64_t l = 0; l < qk / 64; l += 1) {
            for (uint64_t j = 0; j < 4; j++) {
                qs[j] = _mm_loadu_si128((__m128i *)&x[i].qs[j * 16 + l * 64]);
                qp[j] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(qs[j]));
                qp[j] = _mm512_mul_ps(qp[j], d);
                _mm512_storeu_ps(y + j * 16 + l * 64, qp[j]);
            }
        }

        y += qk;
    }

#elif defined(__AVX2__)
#pragma message("Building AVX2 dequantize_row_q8_K")

    for (uint64_t i = 0; i < nb; i++) {
        const __m256 d = _mm256_set1_ps(x[i].d);
        __m128i qs[4];
        __m256 qp[4];

        for (uint64_t l = 0; l < qk / 32; l += 1) {
            for (uint64_t j = 0; j < 4; j++) {
                qs[j] = _mm_loadu_si64((__m128i *)&x[i].qs[j * 8 + l * 32]);
                qp[j] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(qs[j]));
                qp[j] = _mm256_mul_ps(qp[j], d);
                _mm256_storeu_ps(y + j * 8 + l * 32, qp[j]);
            }
        }

        y += qk;
    }

#else
#pragma message("Building Scalar dequantize_row_q8_K")

    for (uint64_t i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }

#endif // defined(__AVX512F__) && defined(__GEN_AVX512__)

}

#if defined(GGML_XBOX_PERF)

int32_t vec_dot_type_counts[GGML_TYPE_COUNT] = {0};
int64_t vec_dot_type_times[GGML_TYPE_COUNT] = {0};
int64_t vec_dot_type_conversion_time[GGML_TYPE_COUNT] = {0};
int32_t vec_dot_src0_counts[GGML_TYPE_COUNT] = {0};
int64_t vec_dot_src0_time[GGML_TYPE_COUNT] = {0};
int compute_op_counts[GGML_OP_COUNT] = {0};
int64_t compute_op_time[GGML_OP_COUNT] = {0};
int openMP_compute_runs = 0;

#define GGML_TENSOR_NODE_COUNT 4096
atomic_int graph_tensor_counts[GGML_TENSOR_NODE_COUNT] = {0};

#define ROW_SIZE_BUCKETS 16385

typedef struct {
    int32_t total_count;
    int32_t counts[ROW_SIZE_BUCKETS];
    int64_t times[ROW_SIZE_BUCKETS];
    int64_t times_max[ROW_SIZE_BUCKETS];
    int64_t conversion_from_float_times[ROW_SIZE_BUCKETS];
    int32_t max_ne00;
    int32_t max_ne01;
    int32_t max_ne10;
    int32_t max_ne11;
    int64_t max_time;
} quant_type_info;

#if defined(__gnu_linux__)
#define DECLSPEC__CACHEALIGN
#define ARRAYSIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#else
#define DECLSPEC__CACHEALIGN DECLSPEC_CACHEALIGN 
#endif // __gnu_linux

DECLSPEC__CACHEALIGN quant_type_info quant_type_row_size[GGML_TYPE_COUNT] = {0};

void ggml_cpu_print_tensor_op_perf() {

    int32_t total_count = 0;
    int32_t total_op_count = 0;
    double total_percent = 0.;
    int64_t total_time = 0;
    double percent;

    printf("\n\n OpenMP runs = %d\n\n", openMP_compute_runs);
    printf("          Total     Total  Tensor\n");
    printf("   Count Time(sec)   %%     Time(us) Tensor Op\n");

    for (uint64_t i = 0; i < ARRAYSIZE(compute_op_counts); i += 1) {
        total_count += compute_op_counts[i];
        total_time += compute_op_time[i];
    }

    total_op_count = total_count;
    total_percent = 0.;
    for (uint64_t i = 0; i < ARRAYSIZE(compute_op_counts); i += 1) {
        if (compute_op_counts[i]) {
            percent = (double)compute_op_time[i] * 100.f / (double)total_time;
            total_percent += percent;
            printf("%8d %8.2f  %5.2f   %8.2f GGML_OP_%s\n",
                   compute_op_counts[i],
                   (double)(compute_op_time[i]) / (1000. * 1000.),
                   percent,
                   (double)(compute_op_time[i]) / (double)compute_op_counts[i],
                   ggml_op_name(i));
        }
    }

    // Number of total tensors processed and times
    printf("\n%8d %8.2f %4.2f\n\n",
           total_count,
           (double)(total_time) / (1000. * 1000.),
           total_percent);


    total_count = 0;
    int32_t total_tensors = 0;

    printf("Graph Size  #_Nodes  #_Tensors\n");
    for (uint32_t i = 0; i < ARRAYSIZE(graph_tensor_counts); i += 1) {
        if (graph_tensor_counts[i]) {
            total_count += graph_tensor_counts[i];
            total_tensors += graph_tensor_counts[i] * i;
            printf("%5d       %5d    %8d\n",
                   i,
                   graph_tensor_counts[i],
                   graph_tensor_counts[i] * i);
        }
    }

    printf("\nTotal       %5d    %8d\n", total_count, total_tensors);
    printf("Total OPs Tensors    %8d\n", total_op_count);
    printf("Total NOP Tensors    %8d (skipped)\n\n", total_tensors - total_op_count);

    printf("vector dot matrix multiply type frequency\n");
    printf("   Count     %%    Time(ms)       %%   init_mat(ms) vec_dot_type\n");

    total_count = 0;
    total_percent = 0.;
    total_time = 0;
    for (uint64_t i = 0; i < ARRAYSIZE(vec_dot_type_counts); i += 1) {
        total_count += vec_dot_type_counts[i];
        total_time += vec_dot_type_times[i];
    }

    for (uint64_t i = 0; i < ARRAYSIZE(vec_dot_type_counts); i += 1) {
        if (vec_dot_type_counts[i]) {
            percent = (double)vec_dot_type_counts[i] * 100.f / (double)total_count;
            total_percent += percent;
            printf("%8d   %5.2f  %9.2f %8.2f  %8.2f    GGML_TYPE_%s\n",
                   vec_dot_type_counts[i],
                   percent,
                   (double)(vec_dot_type_times[i]) / 1000.0f,
                   (vec_dot_type_times[i] * 100.0) / total_time,
                   (double)(vec_dot_type_conversion_time[i]) / 1000.0f,
                   ggml_type_name(i));
        }
    }

    printf("\n%8d  %5.2f\n\n", total_count, total_percent);

    printf("Vector Dot Matrix Multiply Src0 Type Frequency\n\n");
    printf("          Total    Total  Tensor\n");
    printf("   Count Time(sec)   %%   Time(ms) Src0 Type\n\n");

    total_count = 0;
    total_time = 0;
    for (uint64_t i = 0; i < ARRAYSIZE(vec_dot_src0_counts); i += 1) {
        total_count += vec_dot_src0_counts[i];
        total_time += vec_dot_src0_time[i];
    }

    total_percent = 0.;
    for (uint64_t i = 0; i < ARRAYSIZE(vec_dot_src0_counts); i += 1) {
        if (vec_dot_src0_counts[i]) {
            percent = (float)vec_dot_src0_time[i] * 100.f / (float)total_time;
            total_percent += percent;
            printf("%8d %8.2f  %5.2f %8.2f ggml_type_%s\n",
                   vec_dot_src0_counts[i],
                   (double)(vec_dot_src0_time[i]) / (1000. * 1000.),
                   percent,
                   (double)(vec_dot_src0_time[i]) / (1000. * (float)vec_dot_src0_counts[i]),
                   ggml_type_name(i));
        }
    }

    printf("\n%8d %8.2f %4.2f\n\n",
           total_count,
           (double)(total_time) / (1000. * 1000.),
           total_percent);

    //
    // Scan through all the quant types looking for types that have a non-zero
    // total count.
    //

    for (uint64_t i = 0; i < ARRAYSIZE(quant_type_row_size); i += 1) {
        if (quant_type_row_size[i].total_count) {
            printf("vector row size count histogram for quant type: %s\n\n",
                   ggml_type_name(i));

            printf("  Size   Count    %%     Time(ms)    Max(ms) From_Float(ms)\n");

            total_count = quant_type_row_size[i].total_count;
            total_percent = 0;
            total_time = 0;
            int64_t weighted_rowsize = 0;

            for (uint64_t j = 0; j < ARRAYSIZE(quant_type_row_size[i].counts); j += 1) {
                if (quant_type_row_size[i].counts[j]) {
                    percent = (double)quant_type_row_size[i].counts[j] * 100.f / (double)total_count;
                    total_percent += percent;
                    weighted_rowsize += (j + 1) * quant_type_row_size[i].counts[j];
                    total_time += quant_type_row_size[i].times[j];
                    printf("%6zd  %6d  %5.2f  %9.2f %9.2f %9.2f\n",
                           j + 1,
                           quant_type_row_size[i].counts[j],
                           percent,
                           quant_type_row_size[i].times[j] / 1000.0,
                           (double) quant_type_row_size[i].times_max[j] / 1000.0,
                           quant_type_row_size[i].conversion_from_float_times[j] / 1000.0);
                }
            }

            printf("\n      %8d %5.2f  %8.2f (avg row size %zd)\n\n", 
                total_count, total_percent,
                total_time / 1000.0,
                weighted_rowsize / total_count);
            printf("  Max entry: ne00 ne01 ne10 ne11  Time(ms)\n");
            printf("             %4d %4d %4d %4d %9.2f\n\n",
                quant_type_row_size[i].max_ne00,
                quant_type_row_size[i].max_ne01,
                quant_type_row_size[i].max_ne10,
                quant_type_row_size[i].max_ne01,
                (double)quant_type_row_size[i].max_time/ 1000.0);
        }
    }
}

// default behavior
bool allow_tensor_repacking = true;

bool ggml_cpu_allow_tensor_repacking() {
    return(allow_tensor_repacking);
}

void ggml_cpu_set_tensor_repacking_flag(bool allow_repacking) {
    printf("%s: set tensor repacking to %s\n", __func__, allow_repacking ? "true" : "false");
    allow_tensor_repacking = allow_repacking;
}

#endif // GGML_XBOX_PERF

// Helpers for polling loops
#if defined(__aarch64__) && ( defined(__clang__) || defined(__GNUC__) )
static inline void ggml_thread_cpu_relax(void) {
    __asm__ volatile("yield" ::: "memory");
}
#elif defined(__x86_64__)
static inline void ggml_thread_cpu_relax(void) {
    _mm_pause();
}
#else
static inline void ggml_thread_cpu_relax(void) {;}
#endif

//
// NUMA support
//

#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512

struct ggml_numa_node {
    uint32_t cpus[GGML_NUMA_MAX_CPUS]; // hardware threads on this node
    uint32_t n_cpus;
};

struct ggml_numa_nodes {
    enum ggml_numa_strategy numa_strategy;
    struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
    uint32_t n_nodes;
    uint32_t total_cpus; // hardware threads on system
    uint32_t current_node; // node on which main process is execting
#if defined(__gnu_linux__)
    cpu_set_t cpuset; // cpuset from numactl
#else
    uint32_t cpuset; // no NUMA support outside of Linux at this time. Use a portable datatype
#endif
};

//
// ggml state
//

struct ggml_state {
    struct ggml_numa_nodes numa;
};

static struct ggml_state g_state = {0};

bool use_OpenMP = false;

void ggml_cpu_select_OpenMP() {
    #ifdef GGML_USE_OPENMP
        printf("%s: GGML switching to OpenMP runtime\n", __func__);
        use_OpenMP = true;
    #else
        printf("%s: OpenMP runtime not available...\n", __func__);
        use_OpenMP = false;
    #endif // GGML_USE_OPENMP
}

void ggml_barrier(struct ggml_threadpool * tp) {
    int n_threads = atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed);
    if (n_threads == 1) {
        return;
    }

    if (use_OpenMP) {
        #ifdef GGML_USE_OPENMP
        #pragma omp barrier
        #endif // GGML_USE_OPENMP
    }
    else {
        int n_passed = atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed);

        // enter barrier (full seq-cst fence)
        int n_barrier = atomic_fetch_add_explicit(&tp->n_barrier, 1, memory_order_seq_cst);

        if (n_barrier == (n_threads - 1)) {
            // last thread
            atomic_store_explicit(&tp->n_barrier, 0, memory_order_relaxed);

            // exit barrier (fill seq-cst fence)
            atomic_fetch_add_explicit(&tp->n_barrier_passed, 1, memory_order_seq_cst);
            return;
        }

        // wait for other threads
        while (atomic_load_explicit(&tp->n_barrier_passed, memory_order_relaxed) == n_passed) {
            ggml_thread_cpu_relax();
        }

        // exit barrier (full seq-cst fence)
        // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
        #ifdef GGML_TSAN_ENABLED
        atomic_fetch_add_explicit(&tp->n_barrier_passed, 0, memory_order_seq_cst);
        #else
        atomic_thread_fence(memory_order_seq_cst);
        #endif
    }
}

#if defined(__gnu_linux__)
static cpu_set_t ggml_get_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread;
    thread = pthread_self();
    CPU_ZERO(&cpuset);
    pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    return cpuset;
}
#else
static uint32_t ggml_get_numa_affinity(void) {
    return 0; // no NUMA support
}
#endif

void ggml_numa_init(enum ggml_numa_strategy numa_flag) {
    if (g_state.numa.n_nodes > 0) {
        fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");

        return;
    }

#if defined(__gnu_linux__)
    struct stat st;
    char path[256];
    int rv;

    // set numa scheme
    g_state.numa.numa_strategy = numa_flag;

    GGML_PRINT_DEBUG("numa strategy %u\n",g_state.numa.numa_strategy);

    g_state.numa.cpuset = ggml_get_numa_affinity();

    // enumerate nodes
    while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.n_nodes;
    }

    // enumerate CPUs
    while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
        rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
        GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
        if (stat(path, &st) != 0) { break; }
        ++g_state.numa.total_cpus;
    }

    GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

    // figure out which node we're on
    uint current_cpu;
    int getcpu_ret = 0;
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
    getcpu_ret = getcpu(&current_cpu, &g_state.numa.current_node);
#else
    // old glibc doesn't have a wrapper for this call. Fall back on direct syscall
#   if !defined(SYS_getcpu) && defined(SYS_get_cpu)
#       define SYS_getcpu SYS_get_cpu // some older glibc versions use this name
#   endif
    getcpu_ret = syscall(SYS_getcpu, &current_cpu, &g_state.numa.current_node);
#endif

    if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1 || getcpu_ret != 0) {
        g_state.numa.n_nodes = 0;
        return;
    }

    GGML_PRINT_DEBUG("found our process on numa node %u, CPU %u\n", g_state.numa.current_node, current_cpu);

    for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
        struct ggml_numa_node * node = &g_state.numa.nodes[n];
        GGML_PRINT_DEBUG("CPUs on node %u:", n);
        node->n_cpus = 0;
        for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
            rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
            GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
            if (stat(path, &st) == 0) {
                node->cpus[node->n_cpus++] = c;
                GGML_PRINT_DEBUG(" %u", c);
            }
        }
        GGML_PRINT_DEBUG("\n");
    }

    if (ggml_is_numa()) {
        FILE *fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (fptr != NULL) {
            char buf[42];
            if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
                GGML_LOG_WARN("/proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
            }
            fclose(fptr);
        }
    }
#else
    UNUSED(numa_flag);
    // TODO
#endif
}

bool ggml_is_numa(void) {
    return g_state.numa.n_nodes > 1;
}

#if defined(__ARM_ARCH)

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#if !defined(HWCAP2_I8MM)
#define HWCAP2_I8MM (1 << 13)
#endif

#if !defined(HWCAP2_SME)
#define HWCAP2_SME (1 << 23)
#endif

static void ggml_init_arm_arch_features(void) {
#if defined(__linux__) && defined(__aarch64__)
    uint32_t hwcap = getauxval(AT_HWCAP);
    uint32_t hwcap2 = getauxval(AT_HWCAP2);

    ggml_arm_arch_features.has_neon    = !!(hwcap & HWCAP_ASIMD);
    ggml_arm_arch_features.has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
    ggml_arm_arch_features.has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
    ggml_arm_arch_features.has_sve     = !!(hwcap & HWCAP_SVE);
    ggml_arm_arch_features.has_sme     = !!(hwcap2 & HWCAP2_SME);

#if defined(__ARM_FEATURE_SVE)
    ggml_arm_arch_features.sve_cnt = PR_SVE_VL_LEN_MASK & prctl(PR_SVE_GET_VL);
#endif
#elif defined(__APPLE__)
    int oldp = 0;
    size_t size = sizeof(oldp);
    if (sysctlbyname("hw.optional.AdvSIMD", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    ggml_arm_arch_features.has_neon = oldp;

    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    ggml_arm_arch_features.has_dotprod = oldp;

    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    ggml_arm_arch_features.has_i8mm = oldp;

    if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) != 0) {
        oldp = 0;
    }
    ggml_arm_arch_features.has_sme = oldp;

    ggml_arm_arch_features.has_sve = 0;
    ggml_arm_arch_features.sve_cnt = 0;
#else
// Run-time CPU feature detection not implemented for this platform, fallback to compile time
#if defined(__ARM_NEON)
    ggml_arm_arch_features.has_neon = 1;
#else
    ggml_arm_arch_features.has_neon = 0;
#endif

#if defined(__ARM_FEATURE_MATMUL_INT8)
    ggml_arm_arch_features.has_i8mm = 1;
#else
    ggml_arm_arch_features.has_i8mm = 0;
#endif

#if defined(__ARM_FEATURE_SVE)
    ggml_arm_arch_features.has_sve = 1;
    ggml_arm_arch_features.sve_cnt = 16;
#else
    ggml_arm_arch_features.has_sve = 0;
    ggml_arm_arch_features.sve_cnt = 0;
#endif

#if defined(__ARM_FEATURE_SME) || defined(__ARM_FEATURE_SME2)
    ggml_arm_arch_features.has_sme = 1;
#else
    ggml_arm_arch_features.has_sme = 0;
#endif
#endif
}
#endif

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    GGML_ASSERT(!ggml_get_no_alloc(ctx));

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case GGML_TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), GGML_FP32_TO_FP16(value));
                }
            } break;
        case GGML_TYPE_BF16:
            {
                assert(tensor->nb[0] == sizeof(ggml_bf16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_bf16(nc, (ggml_bf16_t *)(data + i*n1), GGML_FP32_TO_BF16(value));
                }
            } break;
        case GGML_TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                GGML_ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        return ggml_get_f32_nd(tensor, id[0], id[1], id[2], id[3]);
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                return ((int8_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I16:
            {
                return ((int16_t *)(tensor->data))[i];
            }
        case GGML_TYPE_I32:
            {
                return ((int32_t *)(tensor->data))[i];
            }
        case GGML_TYPE_F16:
            {
                return GGML_FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_BF16:
            {
                return GGML_BF16_TO_FP32(((ggml_bf16_t *)(tensor->data))[i]);
            }
        case GGML_TYPE_F32:
            {
                return ((float *)(tensor->data))[i];
            }
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    if (!ggml_is_contiguous(tensor)) {
        int64_t id[4] = { 0, 0, 0, 0 };
        ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
        ggml_set_f32_nd(tensor, id[0], id[1], id[2], id[3], value);
        return;
    }
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(tensor->data))[i] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(tensor->data))[i] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(tensor->data))[i] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            return ((int8_t *) data)[0];
        case GGML_TYPE_I16:
            return ((int16_t *) data)[0];
        case GGML_TYPE_I32:
            return ((int32_t *) data)[0];
        case GGML_TYPE_F16:
            return GGML_FP16_TO_FP32(((ggml_fp16_t *) data)[0]);
        case GGML_TYPE_BF16:
            return GGML_BF16_TO_FP32(((ggml_bf16_t *) data)[0]);
        case GGML_TYPE_F32:
            return ((float *) data)[0];
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value) {
    void * data   = (char *) tensor->data + i0*tensor->nb[0] + i1*tensor->nb[1] + i2*tensor->nb[2] + i3*tensor->nb[3];
    switch (tensor->type) {
        case GGML_TYPE_I8:
            {
                ((int8_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I16:
            {
                ((int16_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_I32:
            {
                ((int32_t *)(data))[0] = value;
            } break;
        case GGML_TYPE_F16:
            {
                ((ggml_fp16_t *)(data))[0] = GGML_FP32_TO_FP16(value);
            } break;
        case GGML_TYPE_BF16:
            {
                ((ggml_bf16_t *)(data))[0] = GGML_FP32_TO_BF16(value);
            } break;
        case GGML_TYPE_F32:
            {
                ((float *)(data))[0] = value;
            } break;
        default:
            {
                GGML_ABORT("fatal error");
            }
    }
}

////////////////////////////////////////////////////////////////////////////////

// ggml_compute_forward_dup                 -> ops.cpp

// ggml_compute_forward_add                 -> ops.cpp

// ggml_compute_forward_add1                -> ops.cpp

// ggml_compute_forward_acc                 -> ops.cpp

// ggml_compute_forward_sqr                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_sqrt                -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_sin                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_cos                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_log                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_sub                 -> ops.cpp (zo -> binary_ops.cpp)

// ggml_compute_forward_mul                 -> ops.cpp (zo -> binary_ops.cpp)

// ggml_compute_forward_div                 -> ops.cpp (zo -> binary_ops.cpp)

// ggml_compute_forward_sum                 -> ops.cpp

// ggml_compute_forward_sum_rows            -> ops.cpp

// ggml_compute_forward_mean                -> ops.cpp

// ggml_compute_forward_argmax              -> ops.cpp

// ggml_compute_forward_count_equal         -> ops.cpp

// ggml_compute_forward_repeat              -> ops.cpp

// ggml_compute_forward_repeat_back         -> ops.cpp

// ggml_compute_forward_concat              -> ops.cpp

// ggml_compute_forward_abs                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_sgn                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_neg                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_step                -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_tanh                -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_elu                 -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_relu                -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_sigmoid             -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_gelu()              -> ops.cpp

// ggml_compute_forward_gelu_quick()        -> ops.cpp

// ggml_compute_forward_silu()              -> ops.cpp

// ggml_compute_forward_leaky_relu()        -> ops.cpp

// ggml_compute_forward_silu_back()         -> ops.cpp

// ggml_compute_forward_hardswish()         -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_hardsigmoid()       -> ops.cpp (zo -> unary_ops.cpp)

// ggml_compute_forward_exp()               -> unary_ops.cpp

// ggml_compute_forward_norm                -> opscpp

// ggml_compute_forward_group_rms_norm      -> ops.cpp

// ggml_compute_forward_group_rms_norm_back -> ops.cpp 

// ggml_compute_forward_group_norm          -> ops.cpp

// ggml_compute_forward_l2_norm             -> ops.cpp

// ggml_compute_forward_mul_mat

static void ggml_compute_forward_mul_mat_one_chunk(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst,
    const enum ggml_type type,
    const int64_t num_rows_per_vec_dot,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t       vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    //printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

    // threads with no work simply yield (not sure if it helps)
    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

#if !defined(GGML_XBOX_PERF)
    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];
#endif // GGML_XBOX_PERF

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int64_t i13 = (ir1 / (ne12 * ne1));
                const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int64_t i03 = i13 / r3;
                const int64_t i02 = i12 / r2;

                const int64_t i1 = i11;
                const int64_t i2 = i12;
                const int64_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                    (src1_cont || src1->type != vec_dot_type
                        ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                        : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

#if !defined(GGML_XBOX_PERF)
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
#else
                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &dst_col[ir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }
#endif // GGML_XBOX_PERF
            }
        }
    }
}

static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type type = src0->type;

#ifdef GGML_XBOX_PERF
    int64_t vec_dot_src0_t0 = 0;
    if (!ith) {
        vec_dot_src0_t0 = ggml_time_us();
        vec_dot_type_counts[type_traits_cpu[type].vec_dot_type] += 1;
    }
#endif // GGML_XBOX_PERF

    enum ggml_type           const vec_dot_type         = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int64_t                  const vec_dot_num_rows     = type_traits_cpu[type].nrows;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows

    // TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
    // broadcast factors
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    const bool src1_cont = ggml_is_contiguous(src1);

    if (src1_cont) {
        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)src1->data + i12*nb12 + i13*nb13,
                                     nb11/ggml_type_size(src1->type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     src1->type,
                                     dst->type))
                    goto UseGgmlGemm1;
        return;
    }
UseGgmlGemm1:;
#endif

    const bool init_mat = (vec_dot_type != src1->type);
#if defined(GGML_XBOX_PERF)
    // const bool init_mat = ((vec_dot_type != src1->type) && (vec_dot_type != GGML_TYPE_F16));
    // for ggml_vec_dot_f16_f32() - currently not compatible
    int64_t time_for_float_conversion = 0;
#endif // GGML_XBOX_PERF
    
    if (init_mat) {
        char * wdata = params->wdata;

#ifdef GGML_XBOX_PERF
        time_for_float_conversion = ggml_time_us();
#endif // GGML_XBOX_PERF

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

    #if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                                ne10);
                }
            }
        }
    #else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    #endif

#ifdef GGML_XBOX_PERF
        time_for_float_conversion = ggml_time_us() - time_for_float_conversion;
        vec_dot_type_conversion_time[type_traits_cpu[type].vec_dot_type] += time_for_float_conversion;
#endif // GGML_XBOX_PERF

    }

    if (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
        atomic_store_explicit(&params->threadpool->current_chunk, nth, memory_order_relaxed);
    }

    ggml_barrier(params->threadpool);

#if GGML_USE_LLAMAFILE
    if (src1->type != vec_dot_type) {
        const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        for (int64_t i13 = 0; i13 < ne13; i13++)
            for (int64_t i12 = 0; i12 < ne12; i12++)
                if (!llamafile_sgemm(params,
                                     ne01, ne11, ne00/ggml_blck_size(src0->type),
                                     (const char *)src0->data + i12/r2*nb02 + i13/r3*nb03,
                                     nb01/ggml_type_size(src0->type),
                                     (const char *)wdata + (i12*ne11 + i13*ne12*ne11)*row_size,
                                     row_size/ggml_type_size(vec_dot_type),
                                     (char *)dst->data + i12*nb2 + i13*nb3,
                                     nb1/ggml_type_size(dst->type),
                                     src0->type,
                                     vec_dot_type,
                                     dst->type))
                    goto UseGgmlGemm2;
        return;
    }
UseGgmlGemm2:;
#endif

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int64_t nr1 = ne1 * ne2 * ne3;

#if defined(GGML_XBOX_PERF)
    uint64_t bucket_index = ggml_row_size(type, ne00);
    if (!ith) {
        // type == src0->type
        if (bucket_index > ARRAYSIZE(quant_type_row_size[type].counts)) {
            bucket_index = ARRAYSIZE(quant_type_row_size[type].counts);
        }

        quant_type_row_size[type].total_count += 1;
        quant_type_row_size[type].counts[bucket_index - 1] += 1;
    }

    quant_type_row_size[type].conversion_from_float_times[bucket_index - 1] += time_for_float_conversion;
#endif // GGML_XBOX_PERF

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 < nth * 4 || ggml_is_numa()) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
        nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = ith;

    while (current_chunk < nchunk0 * nchunk1) {
        const int64_t ith0 = current_chunk % nchunk0;
        const int64_t ith1 = current_chunk / nchunk0;

        const int64_t ir0_start = dr0 * ith0;
        const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int64_t ir1_start = dr1 * ith1;
        const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int64_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(params, dst, type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (nth >= nchunk0 * nchunk1) {
            break;
        }

        current_chunk = atomic_fetch_add_explicit(&params->threadpool->current_chunk, 1, memory_order_relaxed);
    }

#ifdef GGML_XBOX_PERF
    if (!ith) {
        vec_dot_src0_counts[type] += 1;
        vec_dot_src0_time[type] += ggml_time_us() - vec_dot_src0_t0;
    }
#endif // GGML_XBOX_PERF
}

// ggml_compute_forward_mul_mat_id

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ids->ne[0]*ids->ne[1] + (i1)]

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

static void ggml_compute_forward_mul_mat_id_one_chunk(
    struct ggml_tensor * dst,
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    const struct ggml_tensor * ids,
    const int64_t cur_a,
    const int64_t ir0_start,
    const int64_t ir0_end,
    const int64_t ir1_start,
    const int64_t ir1_end,
    const char * src0_cur,
    const struct mmid_row_mapping * matrix_rows,
    const size_t row_size,
    const bool src1_cont,
    const void * wdata) {

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type type = src0->type;

    ggml_vec_dot_t    const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    float tmp[16];

    for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
                const int64_t _i12 = ir1; // logical row index for this expert

                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char *) wdata +
                    (src1_cont || src1->type != vec_dot_type
                    ? (i11      + i12*ne11)*row_size
                    : (i11*nb11 + i12*nb12));

                float * dst_col = (float *) ((char *) dst->data + (i1*nb1 + i2*nb2));

                for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                    vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
                }

                memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir0_end) - iir0)*sizeof(float));
            }
        }
    }
}

static void * incr_ptr_aligned(void ** p, size_t size, size_t align) {

    void * ptr = *p;
    ptr = (void *) GGML_PAD((uintptr_t) ptr, align);
    *p = (void *) ((char *) ptr + size);
    return ptr;
}

static void ggml_compute_forward_mul_mat_id(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    enum ggml_type    const vec_dot_type    = type_traits_cpu[type].vec_dot_type;
    ggml_from_float_t const from_float      = type_traits_cpu[vec_dot_type].from_float;

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    // row groups
    const int n_ids = ids->ne[0]; // n_expert_used
    const int n_as  = ne02;       // n_expert

    void * wdata_cur = params->wdata;

    if (src1->type != vec_dot_type) {
        incr_ptr_aligned(&wdata_cur, ggml_row_size(vec_dot_type, ggml_nelements(src1)), sizeof(int64_t));
    }

    int64_t * matrix_row_counts = // [n_as]
        incr_ptr_aligned(&wdata_cur, n_as*sizeof(int64_t), sizeof(int64_t));

    struct mmid_row_mapping * matrix_rows = // [n_as][ids->ne[0]*ids->ne[1]]
        incr_ptr_aligned(&wdata_cur, n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping), sizeof(int64_t));

    char (*atomic_current_chunk)[CACHE_LINE_SIZE] = // [n_as]
        incr_ptr_aligned(&wdata_cur, CACHE_LINE_SIZE * n_as, CACHE_LINE_SIZE);

    GGML_ASSERT(params->wsize >= (size_t)((char *) wdata_cur - (char *) params->wdata));

    if (src1->type != vec_dot_type) {
        char * wdata = params->wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

#if 0
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = ith; i12 < ne12; i12 += nth) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1),
                               ne10);
                }
            }
        }
#else
        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
#endif
    }

    if (ith == 0) {
        // initialize matrix_row_counts
        memset(matrix_row_counts, 0, n_as*sizeof(int64_t));

        // group rows by src0 matrix
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);

                assert(i02 >= 0 && i02 < n_as);

                MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = (struct mmid_row_mapping) {id, iid1};
                matrix_row_counts[i02] += 1;
            }
        }
    }

    // reset current_chunk
    for (int cur_a = ith; cur_a < n_as; cur_a += nth) {
        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);
        *current_chunk_ctr = nth;
    }

    ggml_barrier(params->threadpool);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t cne1 = matrix_row_counts[cur_a];

        if (cne1 == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a * nb02;
        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
        const size_t row_size = ggml_row_size(vec_dot_type, ne10);

        const int64_t nr0 = ne01;
        const int64_t nr1 = cne1;

        int chunk_size = 16;
        if (nr0 == 1 || nr1 == 1) {
            chunk_size = 64;
        }

#if defined(__aarch64__)
        // disable for ARM
        const bool disable_chunking = true;
#else
        // disable for NUMA
        const bool disable_chunking = ggml_is_numa();
#endif // defined(__aarch64__)

        int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

        if (nchunk0 * nchunk1 < nth * 4 || disable_chunking) {
            nchunk0 = nr0 > nr1 ? nth : 1;
            nchunk1 = nr0 > nr1 ? 1 : nth;
        }

        const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

        int current_chunk = ith;

        atomic_int * current_chunk_ctr = (atomic_int *)(atomic_current_chunk + cur_a);

        while (current_chunk < nchunk0 * nchunk1) {
            const int64_t ith0 = current_chunk % nchunk0;
            const int64_t ith1 = current_chunk / nchunk0;

            const int64_t ir0_start = dr0 * ith0;
            const int64_t ir0_end = MIN(ir0_start + dr0, nr0);

            const int64_t ir1_start = dr1 * ith1;
            const int64_t ir1_end = MIN(ir1_start + dr1, nr1);

            ggml_compute_forward_mul_mat_id_one_chunk(
                dst, src0, src1, ids, cur_a,
                ir0_start, ir0_end, ir1_start, ir1_end,
                src0_cur, matrix_rows, row_size, src1_cont, wdata
            );

            if (nth >= nchunk0 * nchunk1) {
                break;
            }

            current_chunk = atomic_fetch_add_explicit(current_chunk_ctr, 1, memory_order_relaxed);
        }
    }
}

// ggml_compute_forward_out_prod            -> ops.cpp

// ggml_compute_forward_scale               -> ops.cpp

// ggml_compute_forward_set                 -> ops.cpp

// ggml_compute_forward_cpy (forward_dup)   -> ops.cpp

// ggml_compute_forward_cont (forward_dup   -> ops.cpp

// ggml_compute_forward_reshape (NOP)       -> ops.cpp

// ggml_compute_forward_view (NOP)          -> ops.cpp

// ggml_compute_forward_permute (NOP)       -> ops.cpp

// ggml_compute_forward_transpose (NOP)     -> ops.cpp

// ggml_compute_forward_get_rows            -> ops.cpp

// ggml_compute_forward_get_rows_back       -> ops.cpp

// ggml_compute_forward_diag                -> ops.cpp

// ggml_compute_forward_diag_mask_inf       -> ops.cpp

// ggml_compute_forward_soft_max            -> ops.cpp

// ggml_compute_forward_soft_max_ext_back   -> ops.cpp

// ggml_compute_forward_clamp               -> ops.cpp

// ggml_compute_forward_rope                -> ops.cpp

// ggml_compute_forward_rope_back           -> ops.cpp

// ggml_compute_forward_conv_transpose_1d   -> ops.cpp

// ggml_compute_forward_im2col_f32          -> ops.cpp

// ggml_compute_forward_im2col_back_f32     -> ops.cpp

// ggml_compute_forward_pool_1d_sk_p0       -> opscpp

// ggml_compute_forward_pool_1d             -> ops.cpp

// ggml_compute_forward_pool_2d             -> ops.cpp

// ggml_compute_forward_pool_2d_back        -> ops.cpp

// ggml_compute_forward_upscale             -> ops.cpp

// ggml_compute_forward_pad                 -> ops.cpp

// ggml_compute_forward_pad_reflect_1d      -> ops.cpp

// ggml_compute_forward_arange              -> ops.cpp

// ggml_compute_forward_timestep_embedding  -> ops.cpp

// ggml_compute_forward_argsort             -> ops.cpp

// ggml_compute_forward_flash_attn_ext      -> ops.cpp

// ggml_compute_forward_flash_attn_back     -> ops.cpp

// ggml_compute_forward_ssm_conv            -> ops.cpp

// ggml_compute_forward_ssm_scan            -> ops.cpp

// ggml_compute_forward_win_part            -> ops.cpp

// ggml_compute_forward_win_unpart          -> ops.cpp

//gmml_compute_forward_unary                -> ops.cpp

// ggml_compute_forward_get_rel_pos         -> ops.cpp

// ggml_compute_forward_add_rel_pos         -> ops.cpp

// ggml_compute_forward_rwkv_wkv6           -> ops.cpp

// ggml_compute_forward_gla                 -> ops.cpp

// ggml_compute_forward_rwkv_wkv7           -> ops.cpp

// ggml_compute_forward_map_unary           <-GGML_OP_MAP_UNARY Removed->

// ggml_compute_forward_map_binary          <-GGML_OP_MAP_BINARY Removed->

// ggml_compute_forward_map_custom1         -> ops.cpp

// ggml_compute_forward_map_custom2         -> ops.cpp

// ggml_compute_forward_map_custom3         -> ops.cpp

// ggml_compute_forward_cross_entropy_loss  -> ops.cpp

// ggml_compute_forward_opt_step_adamw      -> ops.cpp

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);

    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }

    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }

    switch (tensor->op) {
        case GGML_OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor);
            } break;
        case GGML_OP_ADD:
            {
                ggml_compute_forward_add(params, tensor);
            } break;
        case GGML_OP_ADD1:
            {
                ggml_compute_forward_add1(params, tensor);
            } break;
        case GGML_OP_ACC:
            {
                ggml_compute_forward_acc(params, tensor);
            } break;
        case GGML_OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor);
            } break;
        case GGML_OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor);
            } break;
        case GGML_OP_DIV:
            {
                ggml_compute_forward_div(params, tensor);
            } break;
        case GGML_OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor);
            } break;
        case GGML_OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor);
            } break;
        case GGML_OP_LOG:
            {
                ggml_compute_forward_log(params, tensor);
            } break;
        case GGML_OP_SIN:
            {
                ggml_compute_forward_sin(params, tensor);
            } break;
        case GGML_OP_COS:
            {
                ggml_compute_forward_cos(params, tensor);
            } break;
        case GGML_OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor);
            } break;
        case GGML_OP_SUM_ROWS:
            {
                ggml_compute_forward_sum_rows(params, tensor);
            } break;
        case GGML_OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor);
            } break;
        case GGML_OP_ARGMAX:
            {
                ggml_compute_forward_argmax(params, tensor);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                ggml_compute_forward_count_equal(params, tensor);
            } break;
        case GGML_OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                ggml_compute_forward_repeat_back(params, tensor);
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_compute_forward_concat(params, tensor);
            } break;
        case GGML_OP_SILU_BACK:
            {
                ggml_compute_forward_silu_back(params, tensor);
            } break;
        case GGML_OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                ggml_compute_forward_rms_norm_back(params, tensor);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                ggml_compute_forward_group_norm(params, tensor);
            } break;
        case GGML_OP_L2_NORM:
            {
                ggml_compute_forward_l2_norm(params, tensor);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                ggml_compute_forward_mul_mat_id(params, tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                ggml_compute_forward_out_prod(params, tensor);
            } break;
        case GGML_OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor);
            } break;
        case GGML_OP_SET:
            {
                ggml_compute_forward_set(params, tensor);
            } break;
        case GGML_OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor);
            } break;
        case GGML_OP_CONT:
            {
                ggml_compute_forward_cont(params, tensor);
            } break;
        case GGML_OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor);
            } break;
        case GGML_OP_VIEW:
            {
                ggml_compute_forward_view(params, tensor);
            } break;
        case GGML_OP_PERMUTE:
            {
                ggml_compute_forward_permute(params, tensor);
            } break;
        case GGML_OP_TRANSPOSE:
            {
                ggml_compute_forward_transpose(params, tensor);
            } break;
        case GGML_OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor);
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                ggml_compute_forward_get_rows_back(params, tensor);
            } break;
        case GGML_OP_DIAG:
            {
                ggml_compute_forward_diag(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor);
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
            {
                ggml_compute_forward_diag_mask_zero(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                ggml_compute_forward_soft_max_ext_back(params, tensor);
            } break;
        case GGML_OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor);
            } break;
        case GGML_OP_ROPE_BACK:
            {
                ggml_compute_forward_rope_back(params, tensor);
            } break;
        case GGML_OP_CLAMP:
            {
                ggml_compute_forward_clamp(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_compute_forward_conv_transpose_1d(params, tensor);
            } break;
        case GGML_OP_IM2COL:
            {
                ggml_compute_forward_im2col(params, tensor);
            } break;
        case GGML_OP_IM2COL_BACK:
            {
                ggml_compute_forward_im2col_back_f32(params, tensor);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                ggml_compute_forward_conv_transpose_2d(params, tensor);
            } break;
        case GGML_OP_POOL_1D:
            {
                ggml_compute_forward_pool_1d(params, tensor);
            } break;
        case GGML_OP_POOL_2D:
            {
                ggml_compute_forward_pool_2d(params, tensor);
            } break;
        case GGML_OP_POOL_2D_BACK:
            {
                ggml_compute_forward_pool_2d_back(params, tensor);
            } break;
        case GGML_OP_UPSCALE:
            {
                ggml_compute_forward_upscale(params, tensor);
            } break;
        case GGML_OP_PAD:
            {
                ggml_compute_forward_pad(params, tensor);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                ggml_compute_forward_pad_reflect_1d(params, tensor);
            } break;
        case GGML_OP_ARANGE:
            {
                ggml_compute_forward_arange(params, tensor);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                ggml_compute_forward_timestep_embedding(params, tensor);
            } break;
        case GGML_OP_ARGSORT:
            {
                ggml_compute_forward_argsort(params, tensor);
            } break;
        case GGML_OP_LEAKY_RELU:
            {
                ggml_compute_forward_leaky_relu(params, tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
            } break;
        case GGML_OP_FLASH_ATTN_BACK:
            {
                int32_t t = ggml_get_op_params_i32(tensor, 0);
                GGML_ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn_back(params, masked, tensor);
            } break;
        case GGML_OP_SSM_CONV:
            {
                ggml_compute_forward_ssm_conv(params, tensor);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                ggml_compute_forward_ssm_scan(params, tensor);
            } break;
        case GGML_OP_WIN_PART:
            {
                ggml_compute_forward_win_part(params, tensor);
            } break;
        case GGML_OP_WIN_UNPART:
            {
                ggml_compute_forward_win_unpart(params, tensor);
            } break;
        case GGML_OP_UNARY:
            {
                ggml_compute_forward_unary(params, tensor);
            } break;
        case GGML_OP_GET_REL_POS:
            {
                ggml_compute_forward_get_rel_pos(params, tensor);
            } break;
        case GGML_OP_ADD_REL_POS:
            {
                ggml_compute_forward_add_rel_pos(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV6:
            {
                ggml_compute_forward_rwkv_wkv6(params, tensor);
            } break;
        case GGML_OP_GATED_LINEAR_ATTN:
            {
                ggml_compute_forward_gla(params, tensor);
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                ggml_compute_forward_rwkv_wkv7(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                ggml_compute_forward_map_custom1(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM2:
            {
                ggml_compute_forward_map_custom2(params, tensor);
            }
            break;
        case GGML_OP_MAP_CUSTOM3:
            {
                ggml_compute_forward_map_custom3(params, tensor);
            }
            break;
        case GGML_OP_CUSTOM:
            {
                ggml_compute_forward_custom(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            {
                ggml_compute_forward_cross_entropy_loss(params, tensor);
            }
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            {
                ggml_compute_forward_cross_entropy_loss_back(params, tensor);
            }
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                ggml_compute_forward_opt_step_adamw(params, tensor);
            }
            break;
        case GGML_OP_NONE:
            {
                // nop
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
    }
}

// Android's libc implementation "bionic" does not support setting affinity
#if defined(__gnu_linux__)
static void set_numa_thread_affinity(int thread_n) {
    if (!ggml_is_numa()) {
        return;
    }

    int node_num;
    int rv;
    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    switch(g_state.numa.numa_strategy) {
        case GGML_NUMA_STRATEGY_DISTRIBUTE:
            // run thread on node_num thread_n / (threads per node)
            node_num = thread_n % g_state.numa.n_nodes;
            break;
        case GGML_NUMA_STRATEGY_ISOLATE:
            // run thread on current_node
            node_num = g_state.numa.current_node;
            break;
        case GGML_NUMA_STRATEGY_NUMACTL:
            // use the cpuset that numactl gave us
            rv = pthread_setaffinity_np(pthread_self(), setsize, &g_state.numa.cpuset);
            if (rv) {
                fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n",strerror(rv));
            }
            return;
        default:
            return;
    }

    struct ggml_numa_node * node = &g_state.numa.nodes[node_num];

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (size_t i = 0; i < node->n_cpus; ++i) {
        CPU_SET_S(node->cpus[i], setsize, cpus);
    }

    rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
            fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
    if (!ggml_is_numa()) {
        return;
    }

    size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

    cpu_set_t * cpus = CPU_ALLOC(g_state.numa.total_cpus);
    CPU_ZERO_S(setsize, cpus);
    for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
        CPU_SET_S(i, setsize, cpus);
    }

    int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
    if (rv) {
        fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
    }

    CPU_FREE(cpus);
}
#else
// TODO: Windows etc.
// (the linux implementation may also work on BSD, someone should test)
static void set_numa_thread_affinity(int thread_n) { UNUSED(thread_n);  }
static void clear_numa_thread_affinity(void) {}
#endif

static int ggml_get_n_tasks(struct ggml_tensor * node, int n_threads) {
    int n_tasks = 0;

    if (ggml_is_empty(node)) {
        // no need to multi-thread a no-op
        n_tasks = 1;
        return n_tasks;
    }

    switch (node->op) {
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_ADD:
        case GGML_OP_ADD1:
        case GGML_OP_ACC:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_SUB:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_SUM:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
        case GGML_OP_ARGMAX:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:
        case GGML_OP_REPEAT_BACK:
        case GGML_OP_LEAKY_RELU:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(node)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                    {
                        n_tasks = 1;
                    } break;

                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                    {
                        n_tasks = n_threads;
                    } break;
                default:
                    GGML_ABORT("fatal error");
            }
            break;
        case GGML_OP_SILU_BACK:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK:
        case GGML_OP_L2_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_CONCAT:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_OUT_PROD:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_GET_ROWS:
            {
                // FIXME: get_rows can use additional threads, but the cost of launching additional threads
                // decreases performance with GPU offloading
                //n_tasks = n_threads;
                n_tasks = 1;
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_SET:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_GET_ROWS_BACK:
        case GGML_OP_DIAG:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_DIAG_MASK_ZERO:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX_BACK:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ADD_REL_POS:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_CLAMP:
            {
                n_tasks = 1; //TODO
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_tasks = MIN(n_threads, ggml_nrows(node->src[0]));
            } break;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_BACK:
        case GGML_OP_CONV_TRANSPOSE_1D:
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_2D_BACK:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_ARGSORT:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_FLASH_ATTN_BACK:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
        case GGML_OP_GET_REL_POS:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_MAP_CUSTOM1:
            {
                struct ggml_map_custom1_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM2:
            {
                struct ggml_map_custom2_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_MAP_CUSTOM3:
            {
                struct ggml_map_custom3_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CUSTOM:
            {
                struct ggml_custom_op_params p;
                memcpy(&p, node->op_params, sizeof(p));
                if (p.n_tasks == GGML_N_TASKS_MAX) {
                    n_tasks = n_threads;
                } else {
                    n_tasks = MIN(p.n_tasks, n_threads);
                }
            } break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_NONE:
            {
                n_tasks = 1;
            } break;
        case GGML_OP_COUNT:
            {
                GGML_ABORT("fatal error");
            }
        default:
            {
                fprintf(stderr, "%s: op not implemented: ", __func__);
                if (node->op < GGML_OP_COUNT) {
                    fprintf(stderr, "%s\n", ggml_op_name(node->op));
                } else {
                    fprintf(stderr, "%d\n", node->op);
                }
                GGML_ABORT("fatal error");
            }
    }

    assert(n_tasks > 0);

    return n_tasks;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data);

#if defined(_WIN32)
#include "windows.h"

// TODO: support > 64 CPUs
static bool ggml_thread_apply_affinity(bool * mask) {
    HANDLE    h = GetCurrentThread();
    uint64_t  bitmask = 0ULL;

    assert(GGML_MAX_N_THREADS >= 64);

    for (int32_t i = 0; i < 8; i++) {
        int32_t idx = i * 8;
        uint8_t val = 0;
        val |= mask[idx + 0] << 0;
        val |= mask[idx + 1] << 1;
        val |= mask[idx + 2] << 2;
        val |= mask[idx + 3] << 3;
        val |= mask[idx + 4] << 4;
        val |= mask[idx + 5] << 5;
        val |= mask[idx + 6] << 6;
        val |= mask[idx + 7] << 7;
        bitmask |= (uint64_t)val << idx;
    }

    for (int32_t i = 64; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            fprintf(stderr, "warn: setting thread-affinity for > 64 CPUs isn't supported on windows!\n");
            break;
        }
    }

    DWORD_PTR m = (DWORD_PTR)bitmask;

    m = SetThreadAffinityMask(h, m);

    return m != 0;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    // Note that on Windows the Process Priority Class must be updated in order to set Thread priority.
    // This is up to the applications.
    DWORD p = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   p = THREAD_PRIORITY_NORMAL;        break;
        case GGML_SCHED_PRIO_MEDIUM:   p = THREAD_PRIORITY_ABOVE_NORMAL;  break;
        case GGML_SCHED_PRIO_HIGH:     p = THREAD_PRIORITY_HIGHEST;       break;
        case GGML_SCHED_PRIO_REALTIME: p = THREAD_PRIORITY_TIME_CRITICAL; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    if (!SetThreadPriority(GetCurrentThread(), p)) {
        fprintf(stderr, "warn: failed to set thread priority %d : (%d)\n", prio, (int) GetLastError());
        return false;
    }

    return true;
}

#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/resource.h>

static bool ggml_thread_apply_affinity(const bool * mask) {
    // Not supported on Apple platforms
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#elif defined(__gnu_linux__)
// TODO: this may not work on BSD, to be verified

static bool ggml_thread_apply_affinity(const bool * mask) {
    cpu_set_t cpuset;
    int err;

    CPU_ZERO(&cpuset);

    for (uint32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) {
            GGML_PRINT_DEBUG("Thread %lx: adding %d to cpuset\n", pthread_self(), i);
            CPU_SET(i, &cpuset);
        }
    }

#ifdef __ANDROID__
    err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (err < 0) {
        err = errno;
    }
#else
    err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
    if (err != 0) {
        fprintf(stderr, "warn: failed to set affinity mask 0x%llx : %s (%d)\n", (unsigned long long)mask, strerror(err), err);
        return false;
    }

    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    struct sched_param p;
    int32_t policy = SCHED_OTHER;
    switch (prio) {
        case GGML_SCHED_PRIO_NORMAL:   policy = SCHED_OTHER; p.sched_priority = 0;  break;
        case GGML_SCHED_PRIO_MEDIUM:   policy = SCHED_FIFO;  p.sched_priority = 40; break;
        case GGML_SCHED_PRIO_HIGH:     policy = SCHED_FIFO;  p.sched_priority = 80; break;
        case GGML_SCHED_PRIO_REALTIME: policy = SCHED_FIFO;  p.sched_priority = 90; break;
    }

    if (prio == GGML_SCHED_PRIO_NORMAL) {
        // Keep inherited policy/priority
        return true;
    }

    int32_t err = pthread_setschedparam(pthread_self(), policy, &p);
    if (err != 0) {
        fprintf(stderr, "warn: failed to set thread priority %d : %s (%d)\n", prio, strerror(err), err);
        return false;
    }

    return true;
}

#else // unsupported platforms

static bool ggml_thread_apply_affinity(const bool * mask) {
    UNUSED(mask);
    return true;
}

static bool ggml_thread_apply_priority(int32_t prio) {
    UNUSED(prio);
    return true;
}

#endif

static bool ggml_thread_cpumask_is_valid(const bool * mask) {
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        if (mask[i]) { return true; }
    }
    return false;
}

static void ggml_thread_cpumask_next(const bool * global_mask, bool * local_mask, bool strict, int32_t* iter) {
    if (!strict) {
        memcpy(local_mask, global_mask, GGML_MAX_N_THREADS);
        return;
    } else {
        memset(local_mask, 0, GGML_MAX_N_THREADS);
        int32_t base_idx = *iter;
        for (int32_t i = 0; i < GGML_MAX_N_THREADS; i++) {
            int32_t idx = base_idx + i;
            if (idx >= GGML_MAX_N_THREADS) {
                // Just a cheaper modulo
                idx -= GGML_MAX_N_THREADS;
            }
            if (global_mask[idx]) {
                local_mask[idx] = 1;
                *iter = idx + 1;
                return;
            }
        }
    }
}

void ggml_threadpool_free(struct ggml_threadpool* threadpool) {
    if (!threadpool) return;

    const int n_threads = threadpool->n_threads_max;

    if (!use_OpenMP) {
        struct ggml_compute_state* workers = threadpool->workers;
        
        ggml_mutex_lock(&threadpool->mutex);
        
        threadpool->stop = true;
        threadpool->pause = false;
        
        ggml_cond_broadcast(&threadpool->cond);
        ggml_mutex_unlock(&threadpool->mutex);
        
        for (int j = 1; j < n_threads; j++) {
            int32_t rc = ggml_thread_join(workers[j].thrd, NULL);
            GGML_ASSERT(rc == GGML_EXIT_SUCCESS || rc == GGML_EXIT_ABORTED);
            UNUSED(rc);
        }
    
        ggml_mutex_destroy(&threadpool->mutex);
        ggml_cond_destroy(&threadpool->cond);
    }

    const size_t workers_size = sizeof(struct ggml_compute_state) * n_threads;
    ggml_aligned_free(threadpool->workers, workers_size);
    ggml_aligned_free(threadpool, sizeof(struct ggml_threadpool));
}

// #ifndef GGML_USE_OPENMP
// pause/resume must be called under mutex
static void ggml_threadpool_pause_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Pausing threadpool\n");
    threadpool->pause = true;
    ggml_cond_broadcast(&threadpool->cond);
}

static void ggml_threadpool_resume_locked(struct ggml_threadpool * threadpool) {
    GGML_PRINT_DEBUG("Resuming threadpool\n");
    threadpool->pause = false;
    ggml_cond_broadcast(&threadpool->cond);
}
// #endif // GGML_USE_OPENMP

void ggml_threadpool_pause(struct ggml_threadpool * threadpool) {
    if (!use_OpenMP) {
        ggml_mutex_lock(&threadpool->mutex);
        if (!threadpool->pause) {
            ggml_threadpool_pause_locked(threadpool);
        }
        ggml_mutex_unlock(&threadpool->mutex);
    } else {
        UNUSED(threadpool);
    }
}

void ggml_threadpool_resume(struct ggml_threadpool * threadpool) {
    if (!use_OpenMP) {
        ggml_mutex_lock(&threadpool->mutex);
        if (threadpool->pause) {
           ggml_threadpool_resume_locked(threadpool);
        }
        ggml_mutex_unlock(&threadpool->mutex);
    } else {
        UNUSED(threadpool);
    }
}

struct ggml_cplan ggml_graph_plan(
          const struct ggml_cgraph * cgraph,
                               int   n_threads,
            struct ggml_threadpool * threadpool) {

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
    }
    if (n_threads <= 0) {
        n_threads = threadpool ? threadpool->n_threads_max : GGML_DEFAULT_N_THREADS;
    }

    size_t work_size = 0;

    struct ggml_cplan cplan;
    memset(&cplan, 0, sizeof(struct ggml_cplan));

    int max_tasks = 1;

    // thread scheduling for the different operations + work buffer size estimation
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        const int n_tasks = ggml_get_n_tasks(node, n_threads);

        max_tasks = MAX(max_tasks, n_tasks);

        size_t cur = 0;

        if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
            switch (node->op) {
                case GGML_OP_CPY:
                case GGML_OP_DUP:
                    {
                        if (ggml_is_quantized(node->type) ||
                            // F16 -> BF16 and BF16 -> F16 copies go through intermediate F32
                            (node->src[0]->type == GGML_TYPE_F16  && node->src[1] && node->src[1]->type == GGML_TYPE_BF16) ||
                            (node->src[0]->type == GGML_TYPE_BF16 && node->src[1] && node->src[1]->type == GGML_TYPE_F16)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ADD:
                case GGML_OP_ADD1:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_ACC:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_COUNT_EQUAL:
                    {
                        cur = ggml_type_size(node->type)*n_tasks;
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

                        if (node->src[1]->type != vec_dot_type) {
                            cur = ggml_row_size(vec_dot_type, ggml_nelements(node->src[1]));
                        }
                    } break;
                case GGML_OP_MUL_MAT_ID:
                    {
                        cur = 0;
                        const struct ggml_tensor * src0 = node->src[0];
                        const struct ggml_tensor * src1 = node->src[1];
                        const struct ggml_tensor * ids = node->src[2];
                        const enum ggml_type vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
                        const int n_as = src0->ne[2];
                        // src1
                        if (src1->type != vec_dot_type) {
                            cur += ggml_row_size(vec_dot_type, ggml_nelements(src1)) + sizeof(int64_t);
                        }
                        // matrix_row_counts
                        cur += n_as * sizeof(int64_t) + sizeof(int64_t);
                        // matrix_rows
                        cur += n_as*ids->ne[0]*ids->ne[1]*sizeof(struct mmid_row_mapping) + sizeof(int64_t);
                        // atomic_current_chunk
                        cur += CACHE_LINE_SIZE*n_as + CACHE_LINE_SIZE;
                    } break;
                case GGML_OP_OUT_PROD:
                    {
                        if (ggml_is_quantized(node->src[0]->type)) {
                            cur = ggml_type_size(GGML_TYPE_F32) * node->src[0]->ne[0] * n_tasks;
                        }
                    } break;
                case GGML_OP_SOFT_MAX:
                case GGML_OP_ROPE:
                case GGML_OP_ROPE_BACK:
                    {
                        cur = ggml_type_size(GGML_TYPE_F32) * node->ne[0] * n_tasks;
                    } break;
                case GGML_OP_CONV_TRANSPOSE_1D:
                    {
                        GGML_ASSERT(node->src[0]->ne[3] == 1);
                        GGML_ASSERT(node->src[1]->ne[2] == 1);
                        GGML_ASSERT(node->src[1]->ne[3] == 1);

                        const int64_t ne00 = node->src[0]->ne[0];  // K
                        const int64_t ne01 = node->src[0]->ne[1];  // Cout
                        const int64_t ne02 = node->src[0]->ne[2];  // Cin
                        const int64_t ne10 = node->src[1]->ne[0];  // L
                        const int64_t ne11 = node->src[1]->ne[1];  // Cin

                        if ((node->src[0]->type == GGML_TYPE_F16 ||
                             node->src[0]->type == GGML_TYPE_BF16) &&
                            node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02;
                            cur += sizeof(ggml_fp16_t)*ne10*ne11;
                        } else if (node->src[0]->type == GGML_TYPE_F32 &&
                                   node->src[1]->type == GGML_TYPE_F32) {
                            cur += sizeof(float)*ne00*ne01*ne02;
                            cur += sizeof(float)*ne10*ne11;
                        } else {
                            GGML_ABORT("fatal error");
                        }
                    } break;
                case GGML_OP_CONV_TRANSPOSE_2D:
                    {
                        const int64_t ne00 = node->src[0]->ne[0]; // W
                        const int64_t ne01 = node->src[0]->ne[1]; // H
                        const int64_t ne02 = node->src[0]->ne[2]; // Channels Out
                        const int64_t ne03 = node->src[0]->ne[3]; // Channels In

                        const int64_t ne10 = node->src[1]->ne[0]; // W
                        const int64_t ne11 = node->src[1]->ne[1]; // H
                        const int64_t ne12 = node->src[1]->ne[2]; // Channels In

                        cur += sizeof(ggml_fp16_t)*ne00*ne01*ne02*ne03;
                        cur += sizeof(ggml_fp16_t)*ne10*ne11*ne12;
                    } break;
                case GGML_OP_FLASH_ATTN_EXT:
                    {
                        const int64_t ne10 = node->src[1]->ne[0]; // DK
                        const int64_t ne20 = node->src[2]->ne[0]; // DV

                        cur = sizeof(float)*(1*ne10 + 2*ne20)*n_tasks; // 1x head size K + 2x head size V (per thread)
                    } break;
                case GGML_OP_FLASH_ATTN_BACK:
                    {
                        const int64_t    D = node->src[0]->ne[0];
                        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
                        const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
                        if (node->src[1]->type == GGML_TYPE_F32) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_F16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        } else if (node->src[1]->type == GGML_TYPE_BF16) {
                            cur  = sizeof(float)*mxDn*n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*mxDn*n_tasks; // this is overestimated by x2
                        }
                    } break;

                case GGML_OP_CROSS_ENTROPY_LOSS:
                    {
                        cur = ggml_type_size(node->type)*(n_tasks + node->src[0]->ne[0]*n_tasks);
                    } break;
                case GGML_OP_COUNT:
                    {
                        GGML_ABORT("fatal error");
                    }
                default:
                    break;
            }
        }

        work_size = MAX(work_size, cur);
    }

    if (work_size > 0) {
        work_size += CACHE_LINE_SIZE*(n_threads);
    }

    cplan.threadpool = threadpool;
    cplan.n_threads  = MIN(max_tasks, n_threads);
    cplan.work_size  = work_size;
    cplan.work_data  = NULL;

    return cplan;
}

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool    * tp    = state->threadpool;

    const struct ggml_cgraph * cgraph = tp->cgraph;
    const struct ggml_cplan  * cplan  = tp->cplan;

    set_numa_thread_affinity(state->ith);

    struct ggml_compute_params params = {
        /*.ith       =*/ state->ith,
        /*.nth       =*/ atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
        /*.wsize     =*/ cplan->work_size,
        /*.wdata     =*/ cplan->work_data,
        /*.threadpool=*/ tp,
    };

    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];

#if defined(GGML_XBOX_PERF)
        if (node->is_skipped) {
            continue;
        }

        int64_t tensor_t0 = ggml_time_us();
#endif // GGML_XBOX_PERF

        ggml_compute_forward(&params, node);

        if (!state->ith) {
#if defined(GGML_XBOX_PERF)
            // update tensor op count
            compute_op_counts[node->op] += 1;    
            // update tensor op time
            tensor_t0 = ggml_time_us() - tensor_t0;
            compute_op_time[node->op] += tensor_t0;
            // update time per vec_dot_type and per src0_row_size for mul_mat
            if (node->op == GGML_OP_MUL_MAT) {
                // printf("=================================================\n");
                const struct ggml_tensor * src0 = node->src[0];
                const enum ggml_type src0_type = src0->type;
                vec_dot_type_times[type_traits_cpu[src0_type].vec_dot_type] += tensor_t0;
                quant_type_info *quant_type_info_data = &(quant_type_row_size[src0_type]);
                uint64_t bucket_index = ggml_row_size(src0_type, src0->ne[0] /* ne00 */);
                if (bucket_index > ARRAYSIZE(quant_type_info_data->counts)) {
                    bucket_index = ARRAYSIZE(quant_type_info_data->counts);
                }
                quant_type_info_data->times[bucket_index - 1] += tensor_t0;
                if (tensor_t0 > quant_type_info_data->times_max[bucket_index - 1]) {
                    quant_type_info_data->times_max[bucket_index - 1] = tensor_t0;
                }
                if (tensor_t0 > quant_type_info_data->max_time) {
                    quant_type_info_data->max_time = tensor_t0;
                    quant_type_info_data->max_ne00 = (int32_t) src0->ne[0];
                    quant_type_info_data->max_ne01 = (int32_t) src0->ne[1];
                    const struct ggml_tensor * src1 = node->src[1];
                    quant_type_info_data->max_ne10 = (int32_t) src1->ne[0];
                    quant_type_info_data->max_ne11 = (int32_t) src1->ne[1];
                }
            }
#endif // GGML_XBOX_PERF

            if (cplan->abort_callback &&
                cplan->abort_callback(cplan->abort_callback_data)) {
                atomic_store_explicit(&tp->abort, node_n + 1, memory_order_relaxed);
                tp->ec    = GGML_STATUS_ABORTED;
            }
        }
        
        if (node_n + 1 < cgraph->n_nodes) {
            ggml_barrier(state->threadpool);
        }
    }

    ggml_barrier(state->threadpool);

    return 0;
}

// #ifndef GGML_USE_OPENMP

// check if thread is active
static inline bool ggml_graph_compute_thread_active(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;
    int n_threads = atomic_load_explicit(&threadpool->n_threads_cur, memory_order_relaxed);
    return (state->ith < n_threads);
}

// check if thread is ready to proceed (exit from polling or sleeping)
static inline bool ggml_graph_compute_thread_ready(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (state->pending || threadpool->stop || threadpool->pause) { return true; }

    // check for new graph/work
    int new_graph = atomic_load_explicit(&threadpool->n_graph, memory_order_relaxed);
    if (new_graph != state->last_graph) {
        state->pending    = ggml_graph_compute_thread_active(state);
        state->last_graph = new_graph;
    }

    return state->pending;
}

// sync thread state after polling
static inline void ggml_graph_compute_thread_sync(struct ggml_compute_state * state) {
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
    #ifdef GGML_TSAN_ENABLED
    atomic_fetch_add_explicit(&state->threadpool->n_graph, 0, memory_order_seq_cst);
    #else
    atomic_thread_fence(memory_order_seq_cst);
    #endif
    UNUSED(state);
}

static inline bool ggml_graph_compute_poll_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    // Skip polling for unused threads
    if (!ggml_graph_compute_thread_active(state)) {
        return state->pending;
    }

    // This seems to make 0 ... 100 a decent range for polling level across modern processors.
    // Perhaps, we can adjust it dynamically based on load and things.
    const uint64_t n_rounds = 1024UL * 128 * threadpool->poll;

    for (uint64_t i=0; !ggml_graph_compute_thread_ready(state) && i < n_rounds; i++) {
        // No new work. Keep polling.
        ggml_thread_cpu_relax();
    }

    return state->pending;
}

static inline bool ggml_graph_compute_check_for_work(struct ggml_compute_state * state) {
    struct ggml_threadpool * threadpool = state->threadpool;

    if (ggml_graph_compute_poll_for_work(state)) {
        ggml_graph_compute_thread_sync(state);
        return state->pending;
    }

    ggml_mutex_lock_shared(&threadpool->mutex);
    while (!ggml_graph_compute_thread_ready(state)) {
        // No new work. Wait for the signal.
        GGML_PRINT_DEBUG("thread #%d waiting for work (sleeping)\n", state->ith);
        ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
    }
    ggml_mutex_unlock_shared(&threadpool->mutex);

    return state->pending;
}

static thread_ret_t ggml_graph_compute_secondary_thread(void* data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;
    struct ggml_threadpool * threadpool = state->threadpool;

    ggml_thread_apply_priority(threadpool->prio);
    if (ggml_thread_cpumask_is_valid(state->cpumask)) {
        ggml_thread_apply_affinity(state->cpumask);
    }

    while (true) {
        // Check if we need to sleep
        while (threadpool->pause) {
            GGML_PRINT_DEBUG("thread #%d inside pause loop\n", state->ith);
            ggml_mutex_lock_shared(&threadpool->mutex);
            if (threadpool->pause) {
                ggml_cond_wait(&threadpool->cond, &threadpool->mutex);
            }
            GGML_PRINT_DEBUG("thread #%d resuming after wait\n", state->ith);
            ggml_mutex_unlock_shared(&threadpool->mutex);
        }

        // This needs to be checked for after the cond_wait
        if (threadpool->stop) break;

        // Check if there is new work
        // The main thread is the only one that can dispatch new work

        ggml_graph_compute_check_for_work(state);
        if (state->pending) {
            state->pending = false;

            ggml_graph_compute_thread(state);
        }
    }

    return (thread_ret_t) 0;
}

// Start processing new graph
static void ggml_graph_compute_kickoff(struct ggml_threadpool * threadpool, int n_threads)
{
    // Always take the mutex here because the worker threads are doing hybrid poll/wait

    ggml_mutex_lock(&threadpool->mutex);

    GGML_PRINT_DEBUG("threadpool: n_threads_cur %d n_threads %d\n", threadpool->n_threads_cur, n_threads);

    // Update the number of active threads
    atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);

    // Indicate the graph is ready to be processed
    // We need the full seq-cst fence here because of the polling threads (used in thread_sync)
    atomic_fetch_add_explicit(&threadpool->n_graph, 1, memory_order_seq_cst);

    if (threadpool->pause) {
       // Update main thread prio and affinity to match the threadpool settings
       ggml_thread_apply_priority(threadpool->prio);
       if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
           ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
       }

       // resume does cond broadcast
       ggml_threadpool_resume_locked(threadpool);
    } else {
       ggml_cond_broadcast(&threadpool->cond);
    }

    ggml_mutex_unlock(&threadpool->mutex);
}

// #endif // GGML_USE_OPENMP

static struct ggml_threadpool * ggml_threadpool_new_impl(
    struct ggml_threadpool_params * tpp,
               struct ggml_cgraph * cgraph,
                struct ggml_cplan * cplan) {

    struct ggml_threadpool * threadpool =
        ggml_aligned_malloc(sizeof(struct ggml_threadpool));
    {
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->n_graph          = 0;
        threadpool->n_barrier        = 0;
        threadpool->n_barrier_passed = 0;
        threadpool->current_chunk    = 0;
        threadpool->stop             = false;
        threadpool->pause            = tpp->paused;
        threadpool->abort            = -1;
        threadpool->workers          = NULL;
        threadpool->n_threads_max    = tpp->n_threads;
        threadpool->n_threads_cur    = tpp->n_threads;
        threadpool->poll             = tpp->poll;
        threadpool->prio             = tpp->prio;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    // Allocate and init workers state
    const size_t workers_size = sizeof(struct ggml_compute_state) * tpp->n_threads;
    struct ggml_compute_state * workers = ggml_aligned_malloc(workers_size);

    memset(workers, 0, workers_size);
    for (int j = 0; j < tpp->n_threads; j++) {
        workers[j].threadpool = threadpool;
        workers[j].ith        = j;
    }

    threadpool->workers = workers;

    if (!use_OpenMP) {
        ggml_mutex_init(&threadpool->mutex);
        ggml_cond_init(&threadpool->cond);

        // Spin the threads for all workers, and update CPU placements.
        // Place the main thread last (towards the higher numbered CPU cores).

        int32_t cpumask_iter = 0;

        for (int j = 1; j < tpp->n_threads; j++) {
            ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, &cpumask_iter);

            int32_t rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_secondary_thread, &workers[j]);
            GGML_ASSERT(rc == 0);
        }

        ggml_thread_cpumask_next(tpp->cpumask, workers[0].cpumask, tpp->strict_cpu, &cpumask_iter);

        if (!threadpool->pause) {
            // Update main thread prio and affinity at the start, otherwise we'll do it in resume
            ggml_thread_apply_priority(threadpool->prio);
            if (ggml_thread_cpumask_is_valid(threadpool->workers[0].cpumask)) {
                ggml_thread_apply_affinity(threadpool->workers[0].cpumask);
            }
        }
    }

    return threadpool;
}

struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * tpp) {
    return ggml_threadpool_new_impl(tpp, NULL, NULL);
}

enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan) {
    ggml_cpu_init();

#ifdef GGML_XBOX_PERF
    uint32_t tensor_index = cgraph->n_nodes;
    if (tensor_index >= ARRAYSIZE(graph_tensor_counts)) {
        printf("****** overflow nodes per graph %d\n", tensor_index);
        printf("****** this graph entered in the 0th tensor size bucket\n");
        tensor_index = 0;
    }

    atomic_fetch_add(&graph_tensor_counts[tensor_index], 1);
#endif // GGML_XBOX_PERF

    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);
    GGML_ASSERT(cplan->work_size == 0 || cplan->work_data != NULL);

    int n_threads                               = cplan->n_threads;
    struct ggml_threadpool * threadpool = cplan->threadpool;

    bool disposable_threadpool = false;

    if (threadpool == NULL) {
        //GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
        disposable_threadpool = true;

        struct ggml_threadpool_params ttp = ggml_threadpool_params_default(n_threads);
        threadpool = ggml_threadpool_new_impl(&ttp, cgraph, cplan);
    } else {
        // Reset some of the parameters that need resetting
        // No worker threads should be accessing the parameters below at this stage
        threadpool->cgraph           = cgraph;
        threadpool->cplan            = cplan;
        threadpool->current_chunk    = 0;
        threadpool->abort            = -1;
        threadpool->ec               = GGML_STATUS_SUCCESS;
    }

    if (use_OpenMP) {
        if (n_threads > 1) {
            #ifdef GGML_XBOX_PERF
                openMP_compute_runs += 1;
            #endif // GGML_XBOX_PERF
            #ifdef GGML_USE_OPENMP
                #pragma omp parallel num_threads(n_threads)
                {
                    #pragma omp single
                    {
                        // update the number of threads from the actual number of threads that we got from OpenMP
                        n_threads = omp_get_num_threads();
                        atomic_store_explicit(&threadpool->n_threads_cur, n_threads, memory_order_relaxed);
                    }

                    ggml_graph_compute_thread(&threadpool->workers[omp_get_thread_num()]);
                }
            #endif // GGML_USE_OPENMP
        } else {
            atomic_store_explicit(&threadpool->n_threads_cur, 1, memory_order_relaxed);
            ggml_graph_compute_thread(&threadpool->workers[0]);
        }

    } else {
        if (n_threads > threadpool->n_threads_max) {
            GGML_LOG_WARN("cplan requested more threads (%d) than available (%d)\n", n_threads, threadpool->n_threads_max);
            n_threads = threadpool->n_threads_max;
        }

        // Kick all threads to start the new graph
        ggml_graph_compute_kickoff(threadpool, n_threads);

        // This is a work thread too
        ggml_graph_compute_thread(&threadpool->workers[0]);
    }

    // don't leave affinity set on the main thread
    clear_numa_thread_affinity();

    enum ggml_status ret = threadpool->ec;

    if (disposable_threadpool) {
        ggml_threadpool_free(threadpool);
    }

    return ret;
}

enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads) {
    struct ggml_cplan cplan = ggml_graph_plan(cgraph, n_threads, NULL);

    cplan.work_data = (uint8_t *)ggml_new_buffer(ctx, cplan.work_size);

    return ggml_graph_compute(cgraph, &cplan);
}


int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx_vnni(void) {
#if defined(__AVXVNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vbmi(void) {
#if defined(__AVX512VBMI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_vnni(void) {
#if defined(__AVX512VNNI__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512_bf16(void) {
#if defined(__AVX512BF16__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_amx_int8(void) {
#if defined(__AMX_INT8__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_bmi2(void) {
#if defined(__BMI2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_riscv_v(void) {
#if defined(__riscv_v_intrinsic)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_llamafile(void) {
#if defined(GGML_USE_LLAMAFILE)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_ssse3(void) {
#if defined(__SSSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vxe(void) {
#if defined(__VXE__) || defined(__VXE2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_ARCH) && defined(__ARM_NEON)
    return ggml_arm_arch_features.has_neon;
#else
    return 0;
#endif
}

int ggml_cpu_has_dotprod(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_DOTPROD)
    return ggml_arm_arch_features.has_dotprod;
#else
    return 0;
#endif
}

int ggml_cpu_has_sve(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.has_sve;
#else
    return 0;
#endif
}

int ggml_cpu_has_matmul_int8(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_MATMUL_INT8)
    return ggml_arm_arch_features.has_i8mm;
#else
    return 0;
#endif
}

int ggml_cpu_get_sve_cnt(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SVE)
    return ggml_arm_arch_features.sve_cnt;
#else
    return 0;
#endif
}

int ggml_cpu_has_sme(void) {
#if defined(__ARM_ARCH) && defined(__ARM_FEATURE_SME)
    return ggml_arm_arch_features.has_sme;
#else
    return 0;
#endif
}

#if defined(GGML_B612) && defined(NOT_NOW)
// static functions to forward to ggml-base.dll for perfavx
typedef void (*PFN_ggml_bf16_to_fp32_row)(const ggml_bf16_t *, float *, const int64_t);
static PFN_ggml_bf16_to_fp32_row pfn_ggml_bf16_to_fp32_row;

typedef void (*PFN_ggml_fp32_to_bf16_row)(const float *, ggml_bf16_t *, const int64_t);
static PFN_ggml_fp32_to_bf16_row pfn_ggml_fp32_to_bf16_row;

typedef void (*PFN_ggml_fp16_to_fp32_row)(const ggml_fp16_t *, float *, const int64_t);
static PFN_ggml_fp16_to_fp32_row pfn_ggml_fp16_to_fp32_row;

typedef void (*PFN_ggml_fp32_to_fp16_row)(const float *, ggml_fp16_t *, const int64_t);
static PFN_ggml_fp32_to_fp16_row pfn_ggml_fp32_to_fp16_row;

typedef void (*PFN_dequantize_row_q2_K)(const block_q2_K *, float *, int64_t);
static PFN_dequantize_row_q2_K pfn_dequantize_row_q2_K;

typedef void (*PFN_dequantize_row_q3_K)(const block_q3_K *, float *, int64_t);
static PFN_dequantize_row_q3_K pfn_dequantize_row_q3_K;

typedef void (*PFN_dequantize_row_q4_K)(const block_q4_K *, float *, int64_t);
static PFN_dequantize_row_q4_K pfn_dequantize_row_q4_K;

typedef void (*PFN_dequantize_row_q4_0)(const block_q4_0 *, float *, int64_t);
static PFN_dequantize_row_q4_0 pfn_dequantize_row_q4_0;

typedef void (*PFN_dequantize_row_q6_K)(const block_q6_K *, float *, int64_t);
static PFN_dequantize_row_q6_K pfn_dequantize_row_q6_K;

typedef void (*PFN_dequantize_row_q8_K)(const block_q8_K *, float *, int64_t);
static PFN_dequantize_row_q8_K pfn_dequantize_row_q8_K;

typedef void (*PFN_dequantize_row_q8_0)(const block_q8_0 *, float *, int64_t);
static PFN_dequantize_row_q8_0 pfn_dequantize_row_q8_0;
#endif // GGML_B612

void ggml_cpu_init(void) {
    static bool is_first_call = true;

#if defined(GGML_B612)
    if (is_first_call)
    // avoid being called repeatedly from here (ggml_cpu_init() is popular)
#endif
    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        printf("%s: from ggml-cpu-b612.c - calling ggml!ggml_init() to initialize fp16 tables\n", __func__);
        // must be done prior to initializing the F32 tables below
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }

    ggml_critical_section_start();

    if (is_first_call) {
        // initialize GELU, Quick GELU, SILU and EXP F32 tables
        {
            const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

            printf("%s: from ggml-cpu-b612.c - initializing GELU, SILU and EXP fp32 tables\n", __func__);
            for (int i = 0; i < (1 << 16); ++i) {
                union {
                    uint16_t u16;
                    ggml_fp16_t fp16;
                } u = {i};
                float f = GGML_FP16_TO_FP32(u.fp16);
                ggml_table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
                ggml_table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
            }

            const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

            GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0);
        }

#if defined(__ARM_ARCH)
        ggml_init_arm_arch_features();
#endif

#if defined(GGML_B612) && defined(NOT_NOW)
        pfn_ggml_bf16_to_fp32_row = (PFN_ggml_bf16_to_fp32_row)ggml_bf16_to_fp32_row_cpu;
        pfn_ggml_fp32_to_bf16_row = (PFN_ggml_fp32_to_bf16_row)ggml_fp32_to_bf16_row_cpu;
        pfn_ggml_fp16_to_fp32_row = (PFN_ggml_fp16_to_fp32_row)ggml_fp16_to_fp32_row_cpu;
        pfn_ggml_fp32_to_fp16_row = (PFN_ggml_fp32_to_fp16_row)ggml_fp32_to_fp16_row_cpu;

        pfn_dequantize_row_q2_K = (PFN_dequantize_row_q2_K)dequantize_row_q2_K_cpu;
        pfn_dequantize_row_q3_K = (PFN_dequantize_row_q3_K)dequantize_row_q3_K_cpu;
        pfn_dequantize_row_q4_K = (PFN_dequantize_row_q4_K)dequantize_row_q4_K_cpu;
        pfn_dequantize_row_q4_0 = (PFN_dequantize_row_q4_0)dequantize_row_q4_0_cpu;
        pfn_dequantize_row_q6_K = (PFN_dequantize_row_q6_K)dequantize_row_q6_K_cpu;
        pfn_dequantize_row_q8_K = (PFN_dequantize_row_q8_K)dequantize_row_q8_K_cpu;
        pfn_dequantize_row_q8_0 = (PFN_dequantize_row_q8_0)dequantize_row_q8_0_cpu;
#endif // GGML_B612

        is_first_call = false;
    }

    ggml_critical_section_end();
}

#if defined(GGML_B612) && !defined(__GGML_B612_STATIC__) && defined(NOT_NOW)
#pragma message("Building B612 proxies for ggml-cpu.dll")
// for perfavx
void ggml_init_tables(void) {
    ggml_cpu_init();
}

void ggml_time_init(void) {
    // this is just for show and satisfies perfavx. The real call is 
    // done via ggml_cpu_init() -> ggml_init() -> ggml_time_init().
}

// the following are just proxies for the real funcs in ggml-base.dll

void ggml_bf16_to_fp32_row(const ggml_bf16_t * x, float * y, int64_t n) {
    pfn_ggml_bf16_to_fp32_row(x, y, n);
}

void ggml_fp32_to_bf16_row(const float * x, ggml_bf16_t * y, int64_t n) {
    pfn_ggml_fp32_to_bf16_row(x, y, n);
}

void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    pfn_ggml_fp16_to_fp32_row(x, y, n);
}

void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    pfn_ggml_fp32_to_fp16_row(x, y, n);
}

void dequantize_row_q2_K(const block_q2_K * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q2_K(x, y, k);
}

void dequantize_row_q3_K(const block_q3_K * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q3_K(x, y, k);
}

void dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q4_K(x, y, k);
}

void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q4_0(x, y, k);
}

void dequantize_row_q6_K(const block_q6_K * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q6_K(x, y, k);
}

void dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q8_K(x, y, k);
}

void dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    pfn_dequantize_row_q8_0(x, y, k);
}

#endif //GGML_B612
