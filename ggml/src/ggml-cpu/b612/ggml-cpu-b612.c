#define _CRT_SECURE_NO_DEPRECATE
#define _USE_MATH_DEFINES

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu-repack.h"
#include "ggml-quants.h"
#include "quants.h"
#include "ops.h"

#include <string.h>
#include <stdint.h>
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <malloc.h>
#else
#include <alloca.h>
#endif

// compat globals consumed by ggml-cpu-repack.c
int mul_mat_repack_duplicate_tensor_count = 0;
int64_t mul_mat_repack_duplicate_tensor_total_size = 0;
int mul_mat_repack_failed_count = 0;
int mul_mat_repack_count = 0;
int64_t mul_mat_repack_time_us = 0;
int64_t mul_mat_repack_time_current_op_us = 0;
int mul_mat_repack_shared = 0;

extern int32_t vec_dot_type_counts[GGML_TYPE_COUNT];
extern int64_t vec_dot_type_conversion_time[GGML_TYPE_COUNT];
extern int32_t vec_dot_src0_counts[GGML_TYPE_COUNT];
extern int64_t vec_dot_src0_time[GGML_TYPE_COUNT];

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
extern quant_type_info quant_type_row_size[GGML_TYPE_COUNT];

static const struct ggml_type_traits_cpu ggml_b612_q8_0_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q8_0_q8_0_x8,
    .vec_dot_type             = GGML_TYPE_Q8_0_Q8_0_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q8_0_q8_0_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q8_0_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q8_0_q8_0_x8,
    .vec_dot_type             = GGML_TYPE_Q8_0_Q8_0_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu ggml_b612_q2_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q2_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q2_K_Q8_K_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q2_k_q8_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q236_k_q8_k_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q2_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q2_K_Q8_K_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu ggml_b612_q3_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q3_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q3_K_Q8_K_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q3_k_q8_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q236_k_q8_k_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q3_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q3_K_Q8_K_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu ggml_b612_q4_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q4_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q4_K_Q8_K_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q4_k_q8_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q4_k_q8_k_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q4_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q4_K_Q8_K_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu ggml_b612_q6_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q6_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q6_K_Q8_K_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q6_k_q8_k_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q236_k_q8_k_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q6_k_q8_k_x8,
    .vec_dot_type             = GGML_TYPE_Q6_K_Q8_K_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu ggml_b612_q4_0_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = NULL,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q4_0_q8_0_x8,
    .vec_dot_type             = GGML_TYPE_Q4_0_Q8_0_x8,
    .nrows                    = 1,
};

static const struct ggml_type_traits_cpu ggml_b612_q4_0_q8_0_x8_type_traits = {
    .to_float                 = NULL,
    .from_float               = (ggml_from_float_t) quantize_row_q8_0_x8,
    .vec_dot                  = (ggml_vec_dot_t) xx_vec_dot_q4_0_q8_0_x8,
    .vec_dot_type             = GGML_TYPE_Q4_0_Q8_0_x8,
    .nrows                    = -1,
};

static const struct ggml_type_traits_cpu * ggml_b612_get_type_traits_cpu(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q8_0_x8:         return &ggml_b612_q8_0_x8_type_traits;
        case GGML_TYPE_Q8_0_Q8_0_x8:    return &ggml_b612_q8_0_q8_0_x8_type_traits;
        case GGML_TYPE_Q2_K_x8:         return &ggml_b612_q2_k_x8_type_traits;
        case GGML_TYPE_Q2_K_Q8_K_x8:    return &ggml_b612_q2_k_q8_k_x8_type_traits;
        case GGML_TYPE_Q3_K_x8:         return &ggml_b612_q3_k_x8_type_traits;
        case GGML_TYPE_Q3_K_Q8_K_x8:    return &ggml_b612_q3_k_q8_k_x8_type_traits;
        case GGML_TYPE_Q4_K_x8:         return &ggml_b612_q4_k_x8_type_traits;
        case GGML_TYPE_Q4_K_Q8_K_x8:    return &ggml_b612_q4_k_q8_k_x8_type_traits;
        case GGML_TYPE_Q6_K_x8:         return &ggml_b612_q6_k_x8_type_traits;
        case GGML_TYPE_Q6_K_Q8_K_x8:    return &ggml_b612_q6_k_q8_k_x8_type_traits;
        case GGML_TYPE_Q4_0_x8:         return &ggml_b612_q4_0_x8_type_traits;
        case GGML_TYPE_Q4_0_Q8_0_x8:    return &ggml_b612_q4_0_q8_0_x8_type_traits;
        default:                        return ggml_get_type_traits_cpu(type);
    }
}

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
typedef volatile LONG xb_atomic_int;
static inline LONG xb_atomic_fetch_add(xb_atomic_int * ptr, LONG inc) {
    return InterlockedExchangeAdd(ptr, inc);
}
#else
#include <sched.h>
typedef int xb_atomic_int;
static inline int xb_atomic_fetch_add(xb_atomic_int * ptr, int inc) {
    return __sync_fetch_and_add(ptr, inc);
}
#endif

static inline bool ggml_b612_repackable_type(enum ggml_type t) {
    return t == GGML_TYPE_Q4_0 ||
           t == GGML_TYPE_Q8_0 ||
           t == GGML_TYPE_Q2_K ||
           t == GGML_TYPE_Q3_K ||
           t == GGML_TYPE_Q4_K ||
           t == GGML_TYPE_Q6_K;
}

// compat export used by ggml-cpu.cpp proc-address table
void ggml_cpu_select_OpenMP(void) {
}

// compat export used by ops.cpp b612 FA path
void ggml_fp16_to_fp32_row_cpu(const ggml_fp16_t * x, float * y, int64_t n) {
    ggml_cpu_fp16_to_fp32(x, y, n);
}

void ggml_wait_for_done_xbox(const struct ggml_compute_params * params) {
    const int n_tasks = params->nth;
    if (n_tasks == 1 || params->barrier == NULL || params->generation == NULL) {
        return;
    }

    xb_atomic_int * barrier = (xb_atomic_int *) params->barrier;
    xb_atomic_int * generation = (xb_atomic_int *) params->generation;

    const int generation_old = (int) *generation;
    if (xb_atomic_fetch_add(barrier, 1) == (n_tasks - 1)) {
        *barrier = 0;
        xb_atomic_fetch_add(generation, 1);
    } else {
        while ((int) *generation == generation_old) {
#if defined(_WIN32)
            YieldProcessor();
#else
            sched_yield();
#endif
        }
    }
}

void ggml_wait_to_finalize_xbox(const struct ggml_compute_params * params) {
    const int n_tasks = params->nth;
    if (n_tasks == 1 || params->barrier == NULL) {
        return;
    }

    xb_atomic_int * barrier = (xb_atomic_int *) params->barrier;
    while ((int) *barrier != (n_tasks - 1)) {
#if defined(_WIN32)
        YieldProcessor();
#else
        sched_yield();
#endif
    }
}

bool ggml_cpu_tensor_repack_mode_xbox_callgraph(void) {
    return g_tensor_repack_mode == GGML_TENSOR_REPACK_MODE_XBCG;
}

bool ggml_cpu_tensor_repack_mode_xbox_single_thread(void) {
    return g_tensor_repack_mode == GGML_TENSOR_REPACK_MODE_XBOX_SINGLE_THREAD;
}

bool ggml_cpu_tensor_mulmat_mode_xbox(void) {
    return g_tensor_repack_mode == GGML_TENSOR_MULMAT_MODE_XBOX;
}

typedef struct {
    void * data;
    uintptr_t begin;
    uintptr_t end;
    struct ggml_tensor * tensor;
} repack_candidate_t;

static inline bool ggml_b612_tensor_data_range(const struct ggml_tensor * tensor, uintptr_t * begin, uintptr_t * end) {
    if (tensor == NULL || tensor->data == NULL) {
        return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) {
        return false;
    }

    const uintptr_t start = (uintptr_t) tensor->data;
    const uintptr_t stop = start + nbytes;
    if (stop <= start) {
        return false;
    }

    *begin = start;
    *end = stop;
    return true;
}

static inline bool ggml_b612_ranges_overlap(uintptr_t a_begin, uintptr_t a_end, uintptr_t b_begin, uintptr_t b_end) {
    return a_begin < b_end && b_begin < a_end;
}

static void ggml_repack_scan_aliased_data_pointers(struct ggml_cgraph * cgraph) {
    #define MAX_REPACK_CANDIDATES 512
    repack_candidate_t candidates[MAX_REPACK_CANDIDATES];
    uint32_t n_candidates = 0;

    struct ggml_tensor * const * tensors = cgraph->nodes;
    const uint32_t n_nodes = cgraph->n_nodes;

    for (uint32_t node_n = 0; node_n < n_nodes; node_n++) {
        struct ggml_tensor * tensor = tensors[node_n];
        if (tensor->is_skipped || tensor->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * src0 = tensor->src[0];
        if (src0 == NULL || src0->data == NULL || !ggml_b612_repackable_type(src0->type)) {
            continue;
        }

        uintptr_t src0_begin = 0;
        uintptr_t src0_end = 0;
        if (!ggml_b612_tensor_data_range(src0, &src0_begin, &src0_end)) {
            continue;
        }

        bool found = false;
        for (uint32_t c = 0; c < n_candidates; c++) {
            if (candidates[c].data == src0->data) {
                found = true;
                break;
            }
        }
        if (!found && n_candidates < MAX_REPACK_CANDIDATES) {
            candidates[n_candidates].data = src0->data;
            candidates[n_candidates].begin = src0_begin;
            candidates[n_candidates].end = src0_end;
            candidates[n_candidates].tensor = src0;
            n_candidates++;
        }
    }

    if (n_candidates == 0) {
        return;
    }

    for (uint32_t node_n = 0; node_n < n_nodes; node_n++) {
        struct ggml_tensor * tensor = tensors[node_n];
        if (tensor->is_skipped || tensor->op == GGML_OP_MUL_MAT) {
            continue;
        }

        for (int s = 0; s < GGML_MAX_SRC; s++) {
            struct ggml_tensor * src = tensor->src[s];
            if (src == NULL) {
                break;
            }
            if (src->data == NULL) {
                continue;
            }

            uintptr_t src_begin = 0;
            uintptr_t src_end = 0;
            if (!ggml_b612_tensor_data_range(src, &src_begin, &src_end)) {
                continue;
            }

            for (uint32_t c = 0; c < n_candidates; c++) {
                if (ggml_b612_ranges_overlap(candidates[c].begin, candidates[c].end, src_begin, src_end)) {
                    candidates[c].tensor->flags |= GGML_TENSOR_FLAG_DUP;
                    candidates[c] = candidates[n_candidates - 1];
                    n_candidates--;
                    break;
                }
            }
        }
        if (n_candidates == 0) {
            break;
        }
    }
    #undef MAX_REPACK_CANDIDATES
}

void ggml_cpu_repack_tensor_callgraph(struct ggml_cgraph * cgraph) {
    if ((g_tensor_repack_mode == GGML_TENSOR_REPACK_MODE_NONE) ||
        (g_tensor_repack_mode == GGML_TENSOR_REPACK_MODE_GGML)) {
        return;
    }

    ggml_repack_scan_aliased_data_pointers(cgraph);

    struct ggml_tensor * const * tensors = cgraph->nodes;
    for (uint32_t node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        struct ggml_tensor * tensor = tensors[node_n];
        if (tensor->is_skipped || tensor->op != GGML_OP_MUL_MAT) {
            continue;
        }

        struct ggml_tensor * src0 = tensor->src[0];
        const struct ggml_tensor * src1 = tensor->src[1];
        if (src0 == NULL || src1 == NULL || src1->type != GGML_TYPE_F32 ||
            !ggml_b612_repackable_type(src0->type) || (src0->flags & GGML_TENSOR_FLAG_NO_REPACK)) {
            continue;
        }

        const enum ggml_type src0_type = src0->type;
        const enum ggml_type new_type = ggml_repack_tensor_single_thread(NULL, src0);
        if (new_type != src0_type) {
            src0->type = new_type;
        }
    }
}

static void ggml_compute_forward_mul_mat_xbox(
    const struct ggml_compute_params * params,
    struct ggml_tensor * dst) {

    struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ith = params->ith;
    const int nth = params->nth;

    enum ggml_type src0_type = src0->type;
    const enum ggml_type src1_type = src1->type;

    if (ggml_cpu_tensor_repack_mode_xbox_single_thread()) {
        if (src1_type == GGML_TYPE_F32 && ggml_b612_repackable_type(src0_type) && !(src0->flags & GGML_TENSOR_FLAG_NO_REPACK)) {
            if (ith == 0) {
                const int64_t repack_t0 = ggml_time_us();
                src0_type = ggml_repack_tensor_single_thread(params, src0);
                ggml_wait_to_finalize_xbox(params);
                src0->type = src0_type;

                mul_mat_repack_count += 1;
                mul_mat_repack_time_current_op_us = ggml_time_us() - repack_t0;
                mul_mat_repack_time_us += mul_mat_repack_time_current_op_us;
            }

            ggml_wait_for_done_xbox(params);
        }

        src0_type = src0->type;
    } else if (ggml_cpu_tensor_repack_mode_xbox()) {
        if (src1_type == GGML_TYPE_F32 && ggml_b612_repackable_type(src0_type) && !(src0->flags & GGML_TENSOR_FLAG_NO_REPACK)) {
            int64_t repack_t0 = 0;
            if (ith == 0) {
                repack_t0 = ggml_time_us();
            }

            ggml_repack_tensor(params, src0);

            if (ith == 0) {
                if (src0_type != src0->type) {
                    mul_mat_repack_count += 1;
                    mul_mat_repack_time_current_op_us = ggml_time_us() - repack_t0;
                    mul_mat_repack_time_us += mul_mat_repack_time_current_op_us;
                }
            } else if (src0_type != src0->type) {
                mul_mat_repack_shared += 1;
            }

            src0_type = src0->type;
        }
    } else {
        GGML_ASSERT(ggml_cpu_tensor_mulmat_mode_xbox() || ggml_cpu_tensor_repack_mode_xbox_callgraph());
    }

    GGML_TENSOR_BINARY_OP_LOCALS
    const bool src1_cont = ggml_is_contiguous(src1);

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;
    const int64_t nr0 = ne01;
    const int64_t nr1 = ne1 * ne12 * ne13;

    const struct ggml_type_traits_cpu * const src0_traits = ggml_b612_get_type_traits_cpu(src0_type);
    GGML_ASSERT(src0_traits != NULL);
    GGML_ASSERT(src0_traits->vec_dot != NULL);

    ggml_vec_dot_t const vec_dot = src0_traits->vec_dot;
    enum ggml_type const vec_dot_type = src0_traits->vec_dot_type;
    GGML_ASSERT(vec_dot_type >= 0 && vec_dot_type < GGML_TYPE_COUNT);

    const struct ggml_type_traits_cpu * const vec_dot_traits = ggml_b612_get_type_traits_cpu(vec_dot_type);
    GGML_ASSERT(vec_dot_traits != NULL);

    const bool init_mat = (vec_dot_type != src1_type);
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);
    char * wdata = src1->data;

    int64_t vec_dot_src0_t0 = 0;
    if (ith == 0) {
        vec_dot_src0_t0 = ggml_time_us();
        vec_dot_type_counts[vec_dot_type] += 1;
    }

    int64_t time_for_float_conversion = 0;
    if (init_mat) {
        GGML_ASSERT(vec_dot_traits->from_float != NULL);
        GGML_ASSERT(src1_type == GGML_TYPE_F32);
        wdata = params->wdata;
        GGML_ASSERT(params->wsize >= ne11 * ne12 * ne13 * row_size);

        if (ith == 0) {
            time_for_float_conversion = ggml_time_us();
        }

        ggml_from_float_t const from_float_to_vec_dot = vec_dot_traits->from_float;
        const int64_t rows_per_thread = (ne11 + nth - 1) / nth;
        const int64_t start_row = rows_per_thread * ith;
        const int64_t end_row = MIN(start_row + rows_per_thread, ne11);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            char * row_data = wdata + (i13 * ne12 * ne11 * row_size);
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                char * row_base = row_data + (((i12 * ne11) + start_row) * row_size);
                for (int64_t i11 = start_row; i11 < end_row; ++i11) {
                    from_float_to_vec_dot((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), row_base, ne10);
                    row_base += row_size;
                }
            }
        }

        ggml_wait_for_done_xbox(params);

        if (ith == 0) {
            time_for_float_conversion = ggml_time_us() - time_for_float_conversion;
            vec_dot_type_conversion_time[vec_dot_type] += time_for_float_conversion;
        }
    }

    int64_t ir010;
    int64_t ir011;
    int64_t ir110;
    int64_t ir111;
    int64_t src0_rpc = 0;

    if ((nr0 >= nr1) && ((nr0 >= nth) || (nr1 < nth))) {
        src0_rpc = (nr0 + nth - 1) / nth;
        ir010 = src0_rpc * ith;
        ir011 = MIN(ir010 + src0_rpc, nr0);
        ir110 = 0;
        ir111 = nr1;
    } else {
        ir010 = 0;
        ir011 = nr0;
        src0_rpc = nr0;

        const int64_t rpc = (nr1 + nth - 1) / nth;
        ir110 = rpc * ith;
        ir111 = MIN(ir110 + rpc, nr1);
    }

    if (ir010 >= ir011 || ir110 >= ir111) {
        return;
    }

    size_t src0_row_size = ggml_row_size(src0_type, ne00);
    int64_t blck0_factor = (int64_t) ((l1d_cache_size + (src0_row_size / 2) - row_size) / src0_row_size);
    blck0_factor = MAX((int64_t) 1, blck0_factor);
    blck0_factor = MAX(blck0_factor, (int64_t) nth * 2);
    blck0_factor = MIN(blck0_factor, src0_rpc);

    if (ith == 0) {
        uint64_t bucket_index = src0_row_size;
        if (bucket_index > ARRAYSIZE(quant_type_row_size[src0_type].counts)) {
            bucket_index = ARRAYSIZE(quant_type_row_size[src0_type].counts);
        }

        quant_type_row_size[src0_type].total_count += 1;
        quant_type_row_size[src0_type].counts[bucket_index - 1] += 1;
        quant_type_row_size[src0_type].conversion_from_float_times[bucket_index - 1] += time_for_float_conversion;
    }

    char * dst_data = dst->data;

    if (vec_dot_traits->nrows == -1) {
        const char * src0_row = src0->data;
        const char * src1_col = wdata + (row_size * ir110);
        float * dst_col = (float *)(dst_data + (ir110 * nb1));

        for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck0_factor) {
            const int64_t limit0 = MIN(iir0 + blck0_factor, ir011);
            vec_dot(ne00,
                    &dst_col[iir0],
                    nb1,
                    src0_row + (iir0 * nb01),
                    0,
                    src1_col,
                    ir111 - ir110,
                    limit0 - iir0);
        }
    } else {
        for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck0_factor) {
            for (int64_t ir1 = ir110; ir1 < ir111; ++ir1) {
                const int64_t i13 = (ir1/(ne12*ne1));
                const int64_t i12 = (ir1 - i13*ne12*ne1)/ne1;
                const int64_t i11 = (ir1 - i13*ne12*ne1 - i12*ne1);

                const int64_t i03 = i13/r3;
                const int64_t i02 = i12/r2;
                const char * src0_row = (const char *) src0->data + (0 + i02*nb02 + i03*nb03);

                const char * src1_col = wdata +
                    (src1_cont || init_mat
                     ? (i11      + i12*ne11 + i13*ne12*ne11)*row_size
                     : (i11*nb11 + i12*nb12 + i13*nb13));

                float * dst_col = (float *)(dst_data + (i11*nb1 + i12*nb2 + i13*nb3));
                const int64_t limit0 = MIN(iir0 + blck0_factor, ir011);
                for (int64_t ir0 = iir0; ir0 < limit0; ++ir0) {
                    vec_dot(ne00, &dst_col[ir0], 0, src0_row + ir0*nb01, 0, src1_col, 0, 1);
                }
            }
        }
    }

    if (ith == 0) {
        vec_dot_src0_counts[src0_type] += 1;
        vec_dot_src0_time[src0_type] += ggml_time_us() - vec_dot_src0_t0;
    }
}

bool ggml_cpu_mul_mat_override(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    if (dst->op != GGML_OP_MUL_MAT) {
        return false;
    }

    if (ggml_cpu_tensor_mulmat_mode_xbox() || ggml_cpu_tensor_repack_mode_xbox_callgraph()) {
        ggml_compute_forward_mul_mat_xbox(params, dst);
        return true;
    }

    if (ggml_cpu_tensor_repack_mode_xbox() || ggml_cpu_tensor_repack_mode_xbox_single_thread()) {
        ggml_compute_forward_mul_mat_xbox(params, dst);
        return true;
    }

    return false;
}

// minimal compatibility for upstream dispatch tables not present in quants-b612.c yet
void quantize_row_q1_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q1_0_ref(x, (block_q1_0 *) y, k);
}

void quantize_row_nvfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_nvfp4_ref(x, (block_nvfp4 *) y, k);
}

void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void) bs; (void) bx; (void) by; (void) nrc;
    const int nb = n / QK8_0;
    float * tx = (float *) alloca((size_t) n * sizeof(float));
    float * ty = (float *) alloca((size_t) n * sizeof(float));
    dequantize_row_q1_0((const block_q1_0 *) vx, tx, n);
    const block_q8_0 * q8 = (const block_q8_0 *) vy;
    for (int b = 0; b < nb; ++b) {
        const float d = ggml_fp16_to_fp32(q8[b].d);
        for (int j = 0; j < QK8_0; ++j) {
            ty[b*QK8_0 + j] = d * q8[b].qs[j];
        }
    }
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += tx[i] * ty[i];
    }
    *s = acc;
}

void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    (void) bs; (void) bx; (void) by; (void) nrc;
    const int nb = n / QK8_0;
    float * tx = (float *) alloca((size_t) n * sizeof(float));
    float * ty = (float *) alloca((size_t) n * sizeof(float));
    dequantize_row_nvfp4((const block_nvfp4 *) vx, tx, n);
    const block_q8_0 * q8 = (const block_q8_0 *) vy;
    for (int b = 0; b < nb; ++b) {
        const float d = ggml_fp16_to_fp32(q8[b].d);
        for (int j = 0; j < QK8_0; ++j) {
            ty[b*QK8_0 + j] = d * q8[b].qs[j];
        }
    }
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += tx[i] * ty[i];
    }
    *s = acc;
}
