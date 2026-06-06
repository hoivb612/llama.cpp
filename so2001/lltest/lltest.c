#pragma warning (disable:4201) // nameless struct/union
#pragma warning (disable:4242) // possible loss of data
#pragma warning (disable:4244) // possible loss of data

#include <windows.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <intrin.h>
#include <immintrin.h>
#include "ggml.h"

#define UNUSED(x) (void)(x)

static int64_t timer_freq = 0, timer_start = 0;
void local_timer_init(void) {
    if (!timer_freq) {
        LARGE_INTEGER t;
        QueryPerformanceFrequency(&t);
        timer_freq = t.QuadPart;

        // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
        // and the uptime is high enough.
        // We subtract the program start time to reduce the likelihood of that happening.
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

int64_t local_timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}

#define VECTOR_SIZE_4096u 4096u         // use this setting with logging off for perf runs

//
// Define the vector length extension.
//
// The vector length extension is used for all tests except those that use quants which
// must remain at 0 mod QK_k (256). The extension is to force the middle and left over
// loop to execute on vector operations. The value 16 is chosen because it is one AVX512
// middle loop iteration and two AVX2 middle loop iterations. The value 6 is chosen
// because it causes the left over loop to execute on both AVX512 and AVX2. The goal is
// to exercise all the paths in the various vector operations.
//
// N.B. The vector length extension must be even to accomodate conversion between
//      floating types.
//

UINT32 vector_size = VECTOR_SIZE_4096u; // default vector size

//
// Define random number filter to limit the range of generated values.
//
// N.B. If compiling with clang, narrow the range of q2 quants. The code generated with
//      clang produces different results from msvc for quantize_row_q2_k in some cases.
//      It is unclear why this is happening since it is the same code.
//

#define FLOAT_FILTER_Q2 (1u << 14)
#define FLOAT_FILTER_Q3 (1u << 14)
#define FLOAT_FILTER_Q4 (1u << 14)
#define FLOAT_FILTER_Q6 (1u << 14)
#define FLOAT_FILTER_Q8 (1u << 14)

//
// Fraction multiplier for quantize/dequantize tests.
//

#define FLOAT_FRACTION_X 0.893f
#define FLOAT_FRACTION_Y 0.917f

#define QK_K 256u
#define K_SCALE_SIZE 12u

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
} block_q2_K;
C_ASSERT(!(sizeof(block_q2_K) % sizeof(uint32_t)));

typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;
C_ASSERT(sizeof(block_q3_K) == (sizeof(ggml_half) + QK_K / 8 + QK_K / 4 + 12));

typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
C_ASSERT(!(sizeof(block_q4_K) % sizeof(uint32_t)));

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale 
} block_q6_K;
C_ASSERT(sizeof(block_q6_K) == 210); 

typedef struct {
    float d;                            // delta
    int8_t qs[QK_K];                    // quants
    int16_t bsums[QK_K/16];             // sum of quants in groups of 16
} block_q8_K;
C_ASSERT(!(sizeof(block_q8_K) % sizeof(uint32_t)));

uint32_t iter_repeat = 10;
const uint32_t iter_vec_dot_q2_K_q8_K = 500000;
const uint32_t iter_vec_dot_q3_K_q8_K = 500000;
const uint32_t iter_vec_dot_q4_K_q8_K = 500000;
const uint32_t iter_vec_dot_q6_K_q8_K = 500000;

typedef struct _improve_desc {
    char * text;
    int64_t best_time_Dll;
    int64_t worse_time_Dll;
    int64_t best_time_Local;
    int64_t worse_time_Local;
    int64_t iter;
} improve_desc;

uint32_t improve_table_size = 0;

improve_desc improve_table[64] = {0};

void
log_increase (
    char * text,
    uint64_t best_time_Dll,
    uint64_t worse_time_Dll,
    uint64_t best_time_Local,
    uint64_t worse_time_Local,
    uint64_t iter
    )
{

    if (improve_table_size < ARRAYSIZE(improve_table)) {
        improve_table[improve_table_size].text = text;
        improve_table[improve_table_size].best_time_Dll = best_time_Dll;
        improve_table[improve_table_size].best_time_Local = best_time_Local;
        improve_table[improve_table_size].worse_time_Dll = worse_time_Dll;
        improve_table[improve_table_size].worse_time_Local = worse_time_Local;
        improve_table[improve_table_size].iter = iter;
        improve_table_size += 1;

    } else {
        printf("improvement table overflow\n");
    }

    return;
}

uint64_t seed = 7561; // 4139; // 1439; // 1019; // 997; // 37;
uint32_t xrand_count = 0;

uint32_t
xrand (
    void
    )
{
    xrand_count += 1;
    seed = (seed * 214013ull) + 2531011ull;
    return (uint32_t)((seed >> 16) & 0x7fff);
}

float
genx_float_value (
    uint32_t filter
    )
{
    static uint32_t iterator = 0;

    iterator += 1;
    float value = (float)(xrand() % filter);
    if (!(iterator % 7) || !(iterator % 13)) {
        value = -value;
    }

    return value;
}

float
geny_float_value (
    uint32_t filter
    )

{
    static uint32_t iterator = 7;

    iterator += 1;
    float value = (float)(xrand() % filter);
    if (!(iterator % 11) || !(iterator % 17)) {
        value = -value;
    }

    return value;
}

inline
void *zalloc (size_t size) {
    return VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); 
}

inline
void zfree (void * base) {
    VirtualFree(base, 0, MEM_DECOMMIT | MEM_RELEASE);
    return;
}

typedef void (*PFN_ggml_init_tables)(void);
PFN_ggml_init_tables pfn_ggml_init_tables;

typedef void (*PFN_ggml_time_init)(void);
PFN_ggml_time_init pfn_ggml_time_init;

typedef void (*PFN_quantize_row_q2_K)(const float * restrict, block_q2_K * restrict, int64_t);
PFN_quantize_row_q2_K pfn_quantize_row_q2_K;

typedef void (*PFN_quantize_row_q3_K)(const float * restrict, block_q3_K * restrict, int64_t);
PFN_quantize_row_q3_K pfn_quantize_row_q3_K;

typedef void (*PFN_quantize_row_q4_K)(const float * restrict, block_q4_K * restrict, int64_t);
PFN_quantize_row_q4_K pfn_quantize_row_q4_K;

typedef void (*PFN_quantize_row_q6_K)(const float * restrict, block_q6_K * restrict, int64_t);
PFN_quantize_row_q6_K pfn_quantize_row_q6_K;

typedef void (*PFN_quantize_row_q8_K)(const float * restrict, block_q8_K * restrict, int64_t);
PFN_quantize_row_q8_K pfn_quantize_row_q8_K;

typedef void (*PFN_ggml_vec_dot_q2_K_q8_K)(const int, float *, size_t, const block_q2_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q2_K_q8_K pfn_ggml_vec_dot_q2_K_q8_K;

typedef void (*PFN_ggml_vec_dot_q3_K_q8_K)(const int, float *, size_t, const block_q3_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q3_K_q8_K pfn_ggml_vec_dot_q3_K_q8_K;

typedef void (*PFN_ggml_vec_dot_q4_K_q8_K)(const int, float *, size_t, const block_q4_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q4_K_q8_K pfn_ggml_vec_dot_q4_K_q8_K;

typedef void (*PFN_ggml_vec_dot_q6_K_q8_K)(const int, float *, size_t, const block_q6_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q6_K_q8_K pfn_ggml_vec_dot_q6_K_q8_K;

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x),
                                 _mm256_extractf128_ps(x, 1));

    const __m128 t1 = _mm_hadd_ps(t0, t0);
    return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

// horizontally add 16 floats
static inline float hsum_float_16(const __m512 x) {
    const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(x),
                                     _mm512_extractf32x8_ps(x, 1));

    return hsum_float_8(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i t0 = _mm_add_epi32(_mm256_castsi256_si128(a),
                                     _mm256_extractf128_si256(a, 1));

    const __m128i t1 = _mm_hadd_epi32(t0, t0);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(t1, t1));
}

#define GGML_COMPUTE_FP16_TO_FP32(x) _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))

#endif // __AVX__ || __AVX2__ || __AVX512F__

DECLSPEC_CACHEALIGN float ggml_table_f32_f16[1 << 16];

void local_init_tables(void)
{
    for (int i = 0; i < (1 << 16); ++i) {
        union {
            uint16_t u16;
            ggml_fp16_t fp16;
        } u = {i};
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }
}

inline float local_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    s = *(uint16_t *)&f;
    return ggml_table_f32_f16[s];
}

#define GROUP_MAX_EPS 1e-15f

static inline int nearest_int(float fval) {
    if (fval > 4194303.f) { 
        printf("Math ERROR in %s\n", __func__);
        exit(1);
    }
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type,
        const float * restrict qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + max(-nmax, min(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = max(-nmax, min(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = max(-nmax, min(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + max(-nmax, min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = max(0, min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = max(0, min(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

#define GGML_FP32_TO_FP16(x) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)

void local_quantize_row_q2_K(const float * restrict x, void * restrict vy, int64_t k) {
    const uint64_t qk = QK_K;

    const uint64_t nb = k / qk;

    block_q2_K * restrict y = vy;

    uint8_t L[QK_K];
    uint8_t Laux[16];
    float   weights[16];
    float mins[QK_K/16];
    float scales[QK_K/16];

    const float q4scale = 15.f;

    for (uint64_t i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16*j + l]);
            scales[j] = make_qkx2_quants(16, 3, x + 16*j, weights, L + 16*j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        if (max_scale > 0) {
            float iscale = q4scale/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                y[i].scales[j] = l;
            }
            y[i].d = GGML_FP32_TO_FP16(max_scale/q4scale);
        } else {
            for (int j = 0; j < QK_K/16; ++j) y[i].scales[j] = 0;
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }
        if (max_min > 0) {
            float iscale = q4scale/max_min;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*mins[j]);
                y[i].scales[j] |= (l << 4);
            }
            y[i].dmin = GGML_FP32_TO_FP16(max_min/q4scale);
        } else {
            y[i].dmin = GGML_FP32_TO_FP16(0.f);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            const float d = local_fp16_to_fp32(y[i].d) * (y[i].scales[j] & 0xF);
            if (!d) continue;
            const float dm = local_fp16_to_fp32(y[i].dmin) * (y[i].scales[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int((x[16*j + ii] + dm)/d);
                l = max(0, min(3, l));
                L[16*j + ii] = l;
            }
        }

        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void local_vec_dot_q2_K_q8_K(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    nrc;
    by;
    bx;
    bs;

    const uint64_t nb = n / QK_K;

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

#if defined(__AVX512F__)

    static __declspec(align(64)) const uint16_t perm0[32] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        8, 8, 8, 8, 8, 8, 8, 8,
        9, 9, 9, 9, 9, 9, 9, 9
    };

    static __declspec(align(64)) const uint16_t perm1[32] = {
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3,
        10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11
    };


    static __declspec(align(64)) const uint16_t perm2[32] = {
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        12, 12, 12, 12, 12, 12, 12, 12,
        13, 13, 13, 13, 13, 13, 13, 13
    };

    static __declspec(align(64)) const uint16_t perm3[32] = {
        6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7,
        14, 14, 14, 14, 14, 14, 14, 14,
        15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m512 acc = _mm512_setzero_ps();
    __m256 mins_acc = _mm256_setzero_ps();

    const __m512i idx0 = _mm512_loadu_si512((__m512i *)perm0);
    const __m512i idx1 = _mm512_loadu_si512((__m512i *)perm1);
    const __m512i idx2 = _mm512_loadu_si512((__m512i *)perm2);
    const __m512i idx3 = _mm512_loadu_si512((__m512i *)perm3);

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * local_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);

        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);

        const __m512i scales_all = _mm512_castsi256_si512(_mm256_cvtepi8_epi16(scales8));

        const __m256i q8_0_low = _mm256_loadu_si256((const __m256i*)(q8 + 0));
        const __m256i q8_1_low = _mm256_loadu_si256((const __m256i*)(q8 + 32));
        const __m256i q8_2_low = _mm256_loadu_si256((const __m256i*)(q8 + 64));
        const __m256i q8_3_low = _mm256_loadu_si256((const __m256i*)(q8 + 96));

        const __m512i q2_0 = _mm512_and_si512(q2bits, m3);
        const __m512i q2_1 = _mm512_and_si512(_mm512_srli_epi16(q2bits, 2), m3);
        const __m512i q2_2 = _mm512_and_si512(_mm512_srli_epi16(q2bits, 4), m3);
        const __m512i q2_3 = _mm512_and_si512(_mm512_srli_epi16(q2bits, 6), m3);

        const __m256i q8_0_high = _mm256_loadu_si256((const __m256i*)(q8 + 128));
        const __m256i q8_1_high = _mm256_loadu_si256((const __m256i*)(q8 + 160));
        const __m256i q8_2_high = _mm256_loadu_si256((const __m256i*)(q8 + 192));
        const __m256i q8_3_high = _mm256_loadu_si256((const __m256i*)(q8 + 224));

        __m512i q8_0 = _mm512_castsi256_si512(q8_0_low);
        __m512i q8_1 = _mm512_castsi256_si512(q8_1_low);
        __m512i q8_2 = _mm512_castsi256_si512(q8_2_low);
        __m512i q8_3 = _mm512_castsi256_si512(q8_3_low);

        q8_0 = _mm512_inserti64x4(q8_0, q8_0_high, 1);
        q8_1 = _mm512_inserti64x4(q8_1, q8_1_high, 1);
        q8_2 = _mm512_inserti64x4(q8_2, q8_2_high, 1);
        q8_3 = _mm512_inserti64x4(q8_3, q8_3_high, 1);

        __m512i p0 = _mm512_maddubs_epi16(q2_0, q8_0);
        __m512i p1 = _mm512_maddubs_epi16(q2_1, q8_1);
        __m512i p2 = _mm512_maddubs_epi16(q2_2, q8_2);
        __m512i p3 = _mm512_maddubs_epi16(q2_3, q8_3);

        const __m512i v_scale0 = _mm512_permutexvar_epi16(idx0, scales_all);
        const __m512i v_scale1 = _mm512_permutexvar_epi16(idx1, scales_all);
        const __m512i v_scale2 = _mm512_permutexvar_epi16(idx2, scales_all);
        const __m512i v_scale3 = _mm512_permutexvar_epi16(idx3, scales_all);

        p0 = _mm512_madd_epi16(v_scale0, p0);
        p1 = _mm512_madd_epi16(v_scale1, p1);
        p2 = _mm512_madd_epi16(v_scale2, p2);
        p3 = _mm512_madd_epi16(v_scale3, p3);

        p0 = _mm512_add_epi32(p0, p1);
        p2 = _mm512_add_epi32(p2, p3);
        p0 = _mm512_add_epi32(p0, p2);

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(p0), acc);

        //
        // Load 16 q8 bsums values, multiply by mins values, convert to float,
        // and accumulate the floating results.
        //

        const __m256i bsums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, bsums);
        mins_acc = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), mins_acc);
    }

    __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                               _mm512_extractf32x8_ps(acc, 1));

    res = _mm256_sub_ps(res, mins_acc);

    __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                           _mm256_extractf128_ps(res, 1));

    const __m128 t1 = _mm_hadd_ps(t0, t0);
    *s = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));;

#elif defined(__AVX2__)

    static const uint16_t k_perm[8][16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * local_fp16_to_fp32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m256i scales_all = _mm256_cvtepi8_epi16(scales8);

        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)(q2 + (j * 32)));

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 96));

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            const __m256i idx0 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 0][0]));
            const __m256i idx1 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 1][0]));
            const __m256i idx2 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 2][0]));
            const __m256i idx3 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 3][0]));

            const __m256i v_scale0 = _mm256_permutexvar_epi16(idx0, scales_all);
            const __m256i v_scale1 = _mm256_permutexvar_epi16(idx1, scales_all);
            const __m256i v_scale2 = _mm256_permutexvar_epi16(idx2, scales_all);
            const __m256i v_scale3 = _mm256_permutexvar_epi16(idx3, scales_all);

            p0 = _mm256_madd_epi16(v_scale0, p0);
            p1 = _mm256_madd_epi16(v_scale1, p1);
            p2 = _mm256_madd_epi16(v_scale2, p2);
            p3 = _mm256_madd_epi16(v_scale3, p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    float sumf = 0;

    for (uint64_t i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * local_fp16_to_fp32(x[i].d);
        const float dmin = y[i].d * local_fp16_to_fp32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }

    *s = sumf;

#endif // defined(__AVX512F__)

}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = max(-nmax, min(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = max(-nmax, min(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = max(-nmax, min(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

void local_quantize_row_q3_K(const float * restrict x, void * restrict vy, int64_t k) {
    const uint64_t qk = QK_K;

    const uint64_t nb = k / qk;

    block_q3_K * restrict y = vy;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    for (uint64_t i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int8_t l = nearest_int(iscale*scales[j]);
                l = max(-32, min(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(1/iscale);
        } else {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = local_fp16_to_fp32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = max(-4, min(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}

void local_vec_dot_q3_K_q8_K(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const uint64_t nb = n / QK_K;

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

#if defined(__AVX512F__)

    static const uint16_t k_perm[4][32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m4 = _mm256_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    __m512 acc = _mm512_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        //
        // Set up scales
        //

        uint32_t * aux = (uint32_t *)x[i].scales;
        __m128i scales8 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));

        scales8 = _mm_sub_epi8(scales8, m32);
        const __m512i scales_all = _mm512_castsi256_si512(_mm256_cvtepi8_epi16(scales8));

        //
        // high bits.
        //

        __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        //
        // integer accumulator
        //

        __m512i sumi = _mm512_setzero_si512();

        for (uint64_t j = 0; j < QK_K/128; ++j) {

            //
            // load low 2 bits for 128 values (4 per byte).
            //

            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)(q3 + (j * 32)));

            //
            // prepare low and high bits
            //

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_andnot_si256(_mm256_rol_epi32(hbits, 2), m4);

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_andnot_si256(_mm256_rol_epi32(hbits, 1), m4);

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_andnot_si256(hbits, m4);

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_andnot_si256(_mm256_ror_epi32(hbits, 1), m4);

            hbits = _mm256_ror_epi32(hbits, 4);

            //
            // Combine low 2-bit values.
            //

            __m512i q3lc_0 = _mm512_castsi256_si512(q3l_0);
            q3lc_0 = _mm512_inserti32x8(q3lc_0, q3l_1, 1);

            __m512i q3lc_1 = _mm512_castsi256_si512(q3l_2);
            q3lc_1 = _mm512_inserti32x8(q3lc_1, q3l_3, 1);

            //
            // Combine high 1-bit values.
            //

            __m512i q3hc_0 = _mm512_castsi256_si512(q3h_0);
            q3hc_0 = _mm512_inserti32x8(q3hc_0, q3h_1, 1);

            __m512i q3hc_1 = _mm512_castsi256_si512(q3h_2);
            q3hc_1 = _mm512_inserti32x8(q3hc_1, q3h_3, 1);

            //
            // load Q8 quants
            //

            const __m512i q8_0 = _mm512_loadu_si512(q8 + (j * 128) + 0);
            const __m512i q8_1 = _mm512_loadu_si512(q8 + (j * 128) + 64);

            //
            // Dot product: multiply the 2 low bits and 1 high bit part separately, so
            // _mm256_maddubs_epi16 can be used, and then subtract. The high bit part has
            // the 4 already subtracted (and so, it is zero if the high bit was not set,
            // and 4 if the high bit was set)
            //

            __m512i q8s_0 = _mm512_maddubs_epi16(q3hc_0, q8_0);
            __m512i q8s_1 = _mm512_maddubs_epi16(q3hc_1, q8_1);

            __m512i p16_0 = _mm512_maddubs_epi16(q3lc_0, q8_0);
            __m512i p16_1 = _mm512_maddubs_epi16(q3lc_1, q8_1);

            p16_0 = _mm512_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm512_sub_epi16(p16_1, q8s_1);

            //
            // multiply with scales
            //

            __m512i idx0 = _mm512_loadu_si512((__m256i *)(&k_perm[(j * 2) + 0][0]));
            __m512i idx1 = _mm512_loadu_si512((__m256i *)(&k_perm[(j * 2) + 1][0]));

            __m512i v_scale0 = _mm512_permutexvar_epi16(idx0, scales_all);
            __m512i v_scale1 = _mm512_permutexvar_epi16(idx1, scales_all);

            p16_0 = _mm512_madd_epi16(v_scale0, p16_0);
            p16_1 = _mm512_madd_epi16(v_scale1, p16_1);

            //
            // accumulate
            //

            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p16_0, p16_1));
        }

        //
        // multiply with block scale and accumulate
        //

        acc = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&d)),
                              _mm512_cvtepi32_ps(sumi),
                              acc);
    }

    *s = hsum_float_16(acc);

#elif defined(__AVX2__)

    static const uint16_t k_perm[8][16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m4 = _mm256_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);
        const __m256 dlv = _mm256_broadcast_ss(&d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        //
        // Set up scales
        //

        uint32_t * aux = (uint32_t *)x[i].scales;
        __m128i scales8 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));

        scales8 = _mm_sub_epi8(scales8, m32);
        const __m256i scales_all = _mm256_cvtepi8_epi16(scales8);

        //
        // high bits.
        //

        __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        //
        // integer accumulator
        //

        __m256i sumi = _mm256_setzero_si256();

        for (uint64_t j = 0; j < QK_K/128; ++j) {

            //
            // load low 2 bits for 128 values (4 per byte).
            //

            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)(q3 + (j * 32)));

            //
            // prepare low and high bits
            //

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_andnot_si256(_mm256_rol_epi32(hbits, 2), m4);

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_andnot_si256(_mm256_rol_epi32(hbits, 1), m4);

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_andnot_si256(hbits, m4);

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_andnot_si256(_mm256_ror_epi32(hbits, 1), m4);

            hbits = _mm256_ror_epi32(hbits, 4);

            //
            // load Q8 quants
            //

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 96));

            //
            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so
            // can use _mm256_maddubs_epi16, and then subtract. The high bit part has the
            // 4 already subtracted (and so, it is zero if the high bit was not set, and
            // 4 if the high bit was set)
            //

            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            //
            // multiply with scales
            //

            __m256i idx0 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 0][0]));
            __m256i idx1 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 1][0]));
            __m256i idx2 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 2][0]));
            __m256i idx3 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 3][0]));

            __m256i v_scale0 = _mm256_permutexvar_epi16(idx0, scales_all);
            __m256i v_scale1 = _mm256_permutexvar_epi16(idx1, scales_all);
            __m256i v_scale2 = _mm256_permutexvar_epi16(idx2, scales_all);
            __m256i v_scale3 = _mm256_permutexvar_epi16(idx3, scales_all);

            p16_0 = _mm256_madd_epi16(v_scale0, p16_0);
            p16_1 = _mm256_madd_epi16(v_scale1, p16_1);
            p16_2 = _mm256_madd_epi16(v_scale2, p16_2);
            p16_3 = _mm256_madd_epi16(v_scale3, p16_3);

            //
            // accumulate
            //

            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
        }

        //
        // multiply with block scale and accumulate
        //

        acc = _mm256_fmadd_ps(dlv, _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    //
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, etc.
    //

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = local_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif // defined(__AVX2__) || (defined(__AVX512F__)) 

}

void local_quantize_row_q4_K(const float * restrict x, void * restrict vy, int64_t k) {
    const uint64_t qk = QK_K;

    const uint64_t nb = k / qk;

    block_q4_K * restrict y = vy;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (uint64_t i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = min(63, ls);
            lm = min(63, lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = local_fp16_to_fp32(y[i].d) * sc;
            if (!d) continue;
            const float dm = local_fp16_to_fp32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = max(0, min(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}

void local_vec_dot_q4_K_q8_K(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint64_t nb = n / QK_K;

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t utmp[4];

#if defined(__AVX512F__)

    static const uint16_t k_perm[4][32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    };

    static const uint32_t kmask4 = 0xc0c0c0c0;

    __m512 acc = _mm512_setzero_ps();
    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512 zero512 = _mm512_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * local_fp16_to_fp32(x[i].dmin);

        const uint32_t * vscales = (uint32_t *)x[i].scales;
        utmp[3] = ((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2);
        utmp[2] = vscales[1] & kmask1;
        utmp[1] = (vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2);
        utmp[0] = vscales[0] & kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m512i scales = _mm512_castsi256_si512(mins_and_scales);

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                           _mm256_extracti128_si256(q8sums, 1));

        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);

        const __m128 prod_m = _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod));
        acc = _mm512_add_ps(acc, _mm512_insertf32x4(zero512, prod_m, 0));

        __m512i sumi = _mm512_setzero_si512();

        for (uint64_t j = 0; j < QK_K/64; ++j) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
            const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
            const __m512i idx = _mm512_loadu_si512(&k_perm[j][0]);

            __m512i q4v = _mm512_castsi256_si512(q4bits);
            q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
            q4v = _mm512_and_si512(q4v, m4);

            __m512i p32v = _mm512_maddubs_epi16(q4v, q8v);

            const __m512i scale = _mm512_permutexvar_epi16(idx, scales);
            p32v = _mm512_madd_epi16(scale, p32v);
            sumi = _mm512_add_epi32(sumi, p32v);
        }

        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_16(acc);

#elif defined(__AVX2__)

    static const uint16_t k_perm[8][16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    };

    static const uint32_t kmask4 = 0xc0c0c0c0;

    __m256 acc = _mm256_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256 zero256 = _mm256_setzero_ps();

   for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * local_fp16_to_fp32(x[i].dmin);

        const uint32_t * vscales = (uint32_t *)x[i].scales;
        utmp[3] = ((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2);
        utmp[2] = vscales[1] & kmask1;
        utmp[1] = (vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2);
        utmp[0] = vscales[0] & kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                           _mm256_extracti128_si256(q8sums, 1));

        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);

        const __m128 prod_m = _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod));
        acc = _mm256_add_ps(acc, _mm256_insertf128_ps(zero256, prod_m, 0));

        __m256i sumi = _mm256_setzero_si256();

        for (uint64_t j = 0; j < QK_K/64; ++j) {
            const __m256i idx0 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 2) + 0][0]));
            const __m256i idx1 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 2) + 1][0]));

            const __m256i scale_l = _mm256_permutexvar_epi16(idx0, mins_and_scales);
            const __m256i scale_h = _mm256_permutexvar_epi16(idx1, mins_and_scales);

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)(q8 + (j * 64) + 0));
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)(q8 + (j * 64) + 32));
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }

        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    static const uint32_t kmask3 = 0x03030303;

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (uint64_t i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = local_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = local_fp16_to_fp32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }

    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif // defined(__AVX512F__)
}

void local_quantize_row_q6_K(const float * restrict x, void * restrict vy, int64_t k) {
    const uint64_t qk = QK_K;

    const uint64_t nb = k / qk;

    block_q6_K * restrict y = vy;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (uint64_t i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = min(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = local_fp16_to_fp32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = max(-32, min(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * restrict ql = y[i].ql;
        uint8_t * restrict qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

void local_vec_dot_q6_K_q8_K(const int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const uint64_t nb = n / QK_K;

#if defined(__AVX512F__)

    static const uint16_t k_perm[4][32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    __m512 acc = _mm512_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m512i scales_all = _mm512_castsi256_si512(_mm256_cvtepi8_epi16(scales8));

        __m512i sumi = _mm512_setzero_si512();

        for (uint64_t j = 0; j < QK_K / 128; ++j) {

            //
            // Load the low 4-bit values and the high 2-bit values.
            //

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)(q4 + (j * 64) + 0));
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)(q4 + (j * 64) + 32));
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)(qh + (j * 32)));

            //
            // Unpack the high (<5:4>) 2-bit values.
            //

            const __m256i q4h_0 = _mm256_and_si256(_mm256_rol_epi64(q4bitsH, 4), m2);
            const __m256i q4h_1 = _mm256_and_si256(_mm256_rol_epi64(q4bitsH, 2), m2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m2);
            const __m256i q4h_3 = _mm256_and_si256(_mm256_ror_epi64(q4bitsH, 2), m2);

            //
            // Unpack the low (<3:0>) 4-bit values and or in the high 2-bit values.
            //

            const __m256i q4lh_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4lh_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4lh_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4lh_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            __m512i q4_0 = _mm512_castsi256_si512(q4lh_0);
            q4_0 = _mm512_inserti32x8(q4_0, q4lh_1, 1);

            __m512i q4_1 = _mm512_castsi256_si512(q4lh_2);
            q4_1 = _mm512_inserti32x8(q4_1, q4lh_3, 1);

            const __m512i q8_0 = _mm512_loadu_si512(q8 + (j * 128) + 0);
            const __m512i q8_1 = _mm512_loadu_si512(q8 + (j * 128) + 64);

            const __m512i q8s_0 = _mm512_maddubs_epi16(m32s, q8_0);
            const __m512i q8s_1 = _mm512_maddubs_epi16(m32s, q8_1);

            __m512i p16_0 = _mm512_maddubs_epi16(q4_0, q8_0);
            __m512i p16_1 = _mm512_maddubs_epi16(q4_1, q8_1);

            p16_0 = _mm512_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm512_sub_epi16(p16_1, q8s_1);

            const __m512i idx0 = _mm512_loadu_si512((__m512i *)(&k_perm[(j * 2) + 0][0]));
            const __m512i idx1 = _mm512_loadu_si512((__m512i *)(&k_perm[(j * 2) + 1][0]));

            const __m512i v_scale0 = _mm512_permutexvar_epi16(idx0, scales_all);
            const __m512i v_scale1 = _mm512_permutexvar_epi16(idx1, scales_all);

            p16_0 = _mm512_madd_epi16(v_scale0, p16_0);
            p16_1 = _mm512_madd_epi16(v_scale1, p16_1);

            sumi = _mm512_add_epi32(sumi, _mm512_add_epi32(p16_0, p16_1));
        }

        acc = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&d)),
                                                    _mm512_cvtepi32_ps(sumi),
                                                    acc);
    }

    *s = hsum_float_16(acc);

#elif defined __AVX2__

    static const uint16_t k_perm[8][16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x30);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * local_fp16_to_fp32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_all = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi = _mm256_setzero_si256();

        for (uint64_t j = 0; j < QK_K / 128; ++j) {

            //
            // Load the low 4-bit values and the high 2-bit values.
            //

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)(q4 + (j * 64) + 0));
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)(q4 + (j * 64) + 32));
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)(qh + (j * 32)));

            //
            // Unpack the high (<5:4>) 2-bit values.
            //

            const __m256i q4h_0 = _mm256_and_si256(_mm256_rol_epi64(q4bitsH, 4), m2);
            const __m256i q4h_1 = _mm256_and_si256(_mm256_rol_epi64(q4bitsH, 2), m2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m2);
            const __m256i q4h_3 = _mm256_and_si256(_mm256_ror_epi64(q4bitsH, 2), m2);

            //
            // Unpack the low (<3:0>) 4-bit values and or in the high 2-bit values.
            //

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 0));
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 32));
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 64));
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + (j * 128) + 96));

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            const __m256i idx0 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 0][0]));
            const __m256i idx1 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 1][0]));
            const __m256i idx2 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 2][0]));
            const __m256i idx3 = _mm256_loadu_si256((__m256i *)(&k_perm[(j * 4) + 3][0]));

            const __m256i v_scale0 = _mm256_permutexvar_epi16(idx0, scales_all);
            const __m256i v_scale1 = _mm256_permutexvar_epi16(idx1, scales_all);
            const __m256i v_scale2 = _mm256_permutexvar_epi16(idx2, scales_all);
            const __m256i v_scale3 = _mm256_permutexvar_epi16(idx3, scales_all);

            p16_0 = _mm256_madd_epi16(v_scale0, p16_0);
            p16_1 = _mm256_madd_epi16(v_scale1, p16_1);
            p16_2 = _mm256_madd_epi16(v_scale2, p16_2);
            p16_3 = _mm256_madd_epi16(v_scale3, p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = local_fp16_to_fp32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void local_quantize_row_q8_K(const float * restrict x, block_q8_K * restrict y, int64_t k) {
    const uint64_t qk = QK_K;

    const uint64_t nb = k / qk;

#if defined(__AVX512F__)

    const __m128i one = _mm_set1_epi8(1);
    const __m128i zero128i = _mm_setzero_si128();

    for (uint64_t i = 0; i < nb; i++) {

        __m512 maxvx = _mm512_setzero_ps();

        for (uint64_t j = 0; j < QK_K; j += 64) {
            __m512 ax0 = _mm512_abs_ps(_mm512_loadu_ps(x + j + 0));
            __m512 ax1 = _mm512_abs_ps(_mm512_loadu_ps(x + j + 16));
            maxvx = _mm512_max_ps(maxvx, _mm512_max_ps(ax0, ax1));

            __m512 ax2 = _mm512_abs_ps(_mm512_loadu_ps(x + j + 32));
            __m512 ax3 = _mm512_abs_ps(_mm512_loadu_ps(x + j + 48));
            maxvx = _mm512_max_ps(maxvx, _mm512_max_ps(ax2, ax3));
        }

        const float amax = _mm512_reduce_max_ps(maxvx);

        //
        // Initialize loop values.
        //
        // N.B. These values will either cause the entire quant block to be zeroed,
        //      or the actual quant values will be computed.
        //

        float iscale = 0.0f;

        if (amax != 0.0f) {
            iscale = 127.f / amax;
        }

        y[i].d = amax / 127.f;

        const __m512 xscale = _mm512_set1_ps(iscale);

        int8_t * q8 = y[i].qs;

        for (int l = 0; l < QK_K / 64; l += 1) {
            __m512 xv0 = _mm512_loadu_ps(x + (l * 64) + 0);
            xv0 = _mm512_mul_ps(xscale, xv0);
//              xv0 = _mm512_round_ps(xv0, _MM_ROUND_NEAREST);
            const __m128i v0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(xv0));
            _mm_storeu_si128((__m128i *)(q8 + (l * 64) + 0), v0);

            __m128i v0sum = _mm_dpbusd_epi32(zero128i, one, v0);
            v0sum = _mm_hadd_epi32(v0sum, v0sum);
            y[i].bsums[(l * 4) + 0] = _mm_cvtsi128_si32(_mm_hadd_epi32(v0sum, v0sum));

            __m512 xv1 = _mm512_loadu_ps(x + (l * 64) + 16);
            xv1 = _mm512_mul_ps(xscale, xv1);
//              xv1 = _mm512_round_ps(xv1, _MM_ROUND_NEAREST);
            const __m128i v1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(xv1));
            _mm_storeu_si128((__m128i *)(q8 + (l * 64) + 16), v1);

            __m128i v1sum = _mm_dpbusd_epi32(zero128i, one, v1);
            v1sum = _mm_hadd_epi32(v1sum, v1sum);
            y[i].bsums[(l * 4) + 1] = _mm_cvtsi128_si32(_mm_hadd_epi32(v1sum, v1sum));

            __m512 xv2 = _mm512_loadu_ps(x + (l * 64) + 32);
            xv2 = _mm512_mul_ps(xscale, xv2);
//              xv2 = _mm512_round_ps(xv2, _MM_ROUND_NEAREST);
            const __m128i v2 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(xv2));
            _mm_storeu_si128((__m128i *)(q8 + (l * 64) + 32), v2);

            __m128i v2sum = _mm_dpbusd_epi32(zero128i, one, v2);
            v2sum = _mm_hadd_epi32(v2sum, v2sum);
            y[i].bsums[(l * 4) + 2] = _mm_cvtsi128_si32(_mm_hadd_epi32(v2sum, v2sum));

            __m512 xv3 = _mm512_loadu_ps(x + (l * 64) + 48);
            xv3 = _mm512_mul_ps(xscale, xv3);
//              xv3 = _mm512_round_ps(xv3, _MM_ROUND_NEAREST);
            const __m128i v3 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(xv3));
            _mm_storeu_si128((__m128i *)(q8 + (l * 64) + 48), v3);

            __m128i v3sum = _mm_dpbusd_epi32(zero128i, one, v3);
            v3sum = _mm_hadd_epi32(v3sum, v3sum);
            y[i].bsums[(l * 4) + 3] = _mm_cvtsi128_si32(_mm_hadd_epi32(v3sum, v3sum));
        }

        x += 256;
    }

#elif defined(__AVX2__)

    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for (uint64_t i = 0; i < nb; i += 1) {
        __m256 ax;
        __m256 maxvx = _mm256_setzero_ps();

        float amax;

        for (uint64_t j = 0; j < QK_K / 32; j += 1) {
            for (uint64_t l = 0; l < 4; l += 1) {
                ax = _mm256_loadu_ps(x + (j * 32) + (l * 8));
                maxvx = _mm256_max_ps(maxvx, _mm256_andnot_ps(sign_bit, ax));
            }

            __m128 t0 = _mm_max_ps(_mm256_castps256_ps128(maxvx),
                                   _mm256_extractf128_ps(maxvx, 1));

            t0 = _mm_max_ps(t0, _mm_movehl_ps(t0, t0));
            t0 = _mm_max_ss(t0, _mm_movehdup_ps(t0));
            amax = _mm_cvtss_f32(t0);
        }

        //
        // Initialize loop values.
        //
        // N.B. These values will either cause the entire quant block to be zeroed,
        //      or the actual quant values will be computed.
        //

        const float iscale = (amax == 0.0f) ? 0.0f : 127.f / amax;

        y[i].d = amax / 127.f;

        const __m256 xscale = _mm256_broadcast_ss(&iscale);

        int8_t * q8 = y[i].qs;

        for (int l = 0; l < QK_K / 32; l += 1) {
            __m256 xv[4];
            __m256i xvi[4];

            for (uint64_t j = 0; j < 4; j += 1) {
                xv[j] = _mm256_loadu_ps(x + (j * 8) + (l * 32));
                xv[j] = _mm256_mul_ps(xscale, xv[j]);
//                xv[j] = _mm256_round_ps(xv[j], _MM_ROUND_NEAREST);
                xvi[j] = _mm256_cvtps_epi32(xv[j]);
            }

            y[i].bsums[(l * 2) + 0] = hsum_i32_8(_mm256_add_epi32(xvi[0], xvi[1]));
            y[i].bsums[(l * 2) + 1] = hsum_i32_8(_mm256_add_epi32(xvi[2], xvi[3]));

            xvi[0] = _mm256_packs_epi32(xvi[0], xvi[1]);
            xvi[2] = _mm256_packs_epi32(xvi[2], xvi[3]);
            xvi[0] = _mm256_packs_epi16(xvi[0], xvi[2]);
            xvi[0] = _mm256_permutevar8x32_epi32(xvi[0], perm);

            _mm256_storeu_si256((__m256i *)(q8 + (l * 32)), xvi[0]);
        }

        x += 256;
    }

#else

    for (int i = 0; i < nb; i++) {

        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
            }
        }
        if (amax == 0.0f) {
            memset(&y[i], 0, sizeof(block_q8_K));
            goto next_block;
        }
        //const float iscale = 128.f / amax;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward

        const float iscale = 127.f / amax;
        y[i].d = amax / 127.f;

        const __m128 iscale_ss = _mm_load_ss(&iscale);
//        const __m128 zero = _mm_setzero_ps();
        for (int j = 0; j < QK_K; ++j) {
//        int v = nearest_int(iscale*x[j]);
//        y[i].qs[j] = min(127, v);

            const __m128 xj_ss = _mm_load_ss(&x[j]);
            __m128 prod = _mm_mul_ss(iscale_ss, xj_ss);
//            prod = _mm_round_ss(zero, prod, _MM_ROUND_NEAREST);
            y[i].qs[j] = _mm_cvtss_i32(prod);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }

next_block:
        x += QK_K;
    }

#endif // __AVX512__
}

void
vec_dot_q2_K_q8_K (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of q2/q8 vector elements.
//
//  sum += q2x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q2_K * q2x;
    block_q8_K * q8y;

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q2_K_q8_K performance test for Dll vs. Local\n\n");

    //
    // Allocate  vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q2x = zalloc(vec_size / QK_K * sizeof(block_q2_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q2x || !q8y) {
        printf("  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q2) * FLOAT_FRACTION_X;
        y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
    }

    //
    // Log generated x and y data.
    //

    // log_float_data(vec_size, (void *)x, "generated x vector:");
    // log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q2_K(x, q2x, vec_size);
    pfn_quantize_row_q8_K(y, q8y, vec_size);

    //
    // log q2_k and q8_k quant data.
    //

    // log_q2_k_quant_data(vec_size, (void *)q2x, false);
    // log_q8_k_quant_data(vec_size, (void *)q8y, false);

    int64_t worse_time_Dll = 0;
    int64_t best_time_Dll = MAXLONG64;

    if (pfn_ggml_vec_dot_q2_K_q8_K != NULL) {

        //
        // Announce perf test.
        //

        printf("Running vec_dot_q2_K_q8_K performance test for DLL\n\n");

        //
        // Run the test multiple times to get rid of outliers.
        //


        for (j = 0; j < iter_repeat; j += 1) {

            //
            // Compute the time to do the summation of the product of vector elements.
            //
        
            const int64_t start_time = local_timer_us();
            for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
                pfn_ggml_vec_dot_q2_K_q8_K(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
            }
        
            const int64_t total_time = local_timer_us() - start_time;

            if (total_time < best_time_Dll) {
                best_time_Dll = total_time;
            }

            if (total_time > worse_time_Dll) {
                worse_time_Dll = total_time;
            }
        }
    }

    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x \n\n", *(uint32_t *)&sum);

#if 1 

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q2_K_q8_K performance test for Local\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t worse_time_Local = 0;
    int64_t best_time_Local = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = local_timer_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            local_vec_dot_q2_K_q8_K(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = local_timer_us() - start_time;

        if (total_time < best_time_Local) {
            best_time_Local = total_time;
        }

        if (total_time > worse_time_Local) {
            worse_time_Local = total_time;
        }
    }

    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

#endif

    //
    // Report percent increase from Dll vs. Local
    //

    log_increase("vec_dot_q2_K_q8_K:",
                 best_time_Dll,
                 worse_time_Dll,
                 best_time_Local,
                 worse_time_Local,
                 iter_vec_dot_q2_K_q8_K);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q2x) {
        zfree(q2x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void
vec_dot_q3_K_q8_K (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of q3/q8 vector elements.
//
//  sum += q3x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q3_K * q3x;
    block_q8_K * q8y;

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q3_K_q8_K performance test for Dll vs. Local\n\n");

    //
    // Allocate  vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q3x = zalloc(vec_size / QK_K * sizeof(block_q3_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q3x || !q8y) {
        printf("  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q3) * FLOAT_FRACTION_X;
        y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
    }

    //
    // Log generate x and y data.
    //

    // log_float_data(vec_size, (void *)x, "generated x vector:");
    // log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q3_K(x, q3x, vec_size);
    pfn_quantize_row_q8_K(y, q8y, vec_size);

    //
    // log q3 and q8 quant data.
    //

    // log_q3_k_quant_data(vec_size, (void *)q3x, true);
    // log_q8_k_quant_data(vec_size, (void *)q8y, true);

    int64_t worse_time_Dll = 0;
    int64_t best_time_Dll = MAXLONG64;

    if (pfn_ggml_vec_dot_q3_K_q8_K != NULL) {

        //
        // Announce perf test.
        //

        printf("Running vec_dot_q3_K_q8_K performance test for DLL\n\n");

        //
        // Run the test multiple times to get rid of outliers.
        //

        for (j = 0; j < iter_repeat; j += 1) {

            //
            // Compute the time to do the summation of the product of vector elements.
            //

            const int64_t start_time = local_timer_us();
            for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
                pfn_ggml_vec_dot_q3_K_q8_K(vec_size, &sum, 0, q3x, 0, q8y, 0, 1);
            }
        
            const int64_t total_time = local_timer_us() - start_time;

            if (total_time < best_time_Dll) {
                best_time_Dll = total_time;
            }

            if (total_time > worse_time_Dll) {
                worse_time_Dll = total_time;
            }
        }
    }

    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q3_K_q8_K performance test for Local\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t worse_time_Local = 0;
    int64_t best_time_Local = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do summation of the product of vector elements.
        //
    
        const int64_t start_time = local_timer_us();
        for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
            local_vec_dot_q3_K_q8_K(vec_size, &sum, 0, q3x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = local_timer_us() - start_time;

        if (total_time < best_time_Local) {
            best_time_Local = total_time;
        }

        if (total_time > worse_time_Local) {
            worse_time_Local = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from Dll vs Local
    //

    log_increase("vec_dot_q3_K_q8_K:",
                 best_time_Dll,
                 worse_time_Dll,
                 best_time_Local,
                 worse_time_Local,
                 iter_vec_dot_q3_K_q8_K);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q3x) {
        zfree(q3x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void
vec_dot_q4_K_q8_K (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of q4/q8 vector elements.
//
//  sum += q4x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q4_K * q4x;
    block_q8_K * q8y;

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q4_K_q8_K performance test for Dll vs. Local\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q4x = zalloc(vec_size / QK_K * sizeof(block_q4_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q4x || !q8y) {
        printf("  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q4) * FLOAT_FRACTION_X;
        y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
    }

    //
    // Log generated x and y data.
    //

    // log_float_data(vec_size, (void *)x, "generated x vector:");
    // log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_K(x, q4x, vec_size);
    pfn_quantize_row_q8_K(y, q8y, vec_size);

    //
    // log q4 and q8_k quant data.
    //

    // log_q4_k_quant_data(vec_size, (void *)q4x, true);
    // log_q8_k_quant_data(vec_size, (void *)q8y, true);

    int64_t worse_time_Dll = 0;
    int64_t best_time_Dll = MAXLONG64;

    if (pfn_ggml_vec_dot_q4_K_q8_K != NULL) {

        //
        // Announce perf test.
        //

        printf("Running vec_dot_q4_K_q8_K performance test for DLL\n\n");

        //
        // Run the test multiple times to get rid of outliers.
        //

        for (j = 0; j < iter_repeat; j += 1) {

            //
            // Compute the time to do the summation of the product of vector elements.
            //
        
            const int64_t start_time = local_timer_us();
            for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
                pfn_ggml_vec_dot_q4_K_q8_K(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
            }
        
            const int64_t total_time = local_timer_us() - start_time;

            if (total_time < best_time_Dll) {
                best_time_Dll = total_time;
            }

            if (total_time > worse_time_Dll) {
                worse_time_Dll = total_time;
            }
        }
    }
 
    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q4_K_q8_K performance test for Local\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t worse_time_Local = 0;
    int64_t best_time_Local = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = local_timer_us();
        for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
            local_vec_dot_q4_K_q8_K(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = local_timer_us() - start_time;

        if (total_time < best_time_Local) {
            best_time_Local = total_time;
        }

        if (total_time > worse_time_Local) {
            worse_time_Local = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q4_K_q8_K:",
                 best_time_Dll,
                 worse_time_Dll,
                 best_time_Local,
                 worse_time_Local,
                 iter_vec_dot_q4_K_q8_K);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q4x) {
        zfree(q4x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void
vec_dot_q6_K_q8_K (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of q6/q8 vector elements.
//
//  sum += q6x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q6_K * q6x;
    block_q8_K * q8y;

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q6_K_q8_K performance test for Dll vs. Local\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q6x = zalloc(vec_size / QK_K * sizeof(block_q6_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q6x || !q8y) {
        printf("  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q6) * FLOAT_FRACTION_X;
        y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
    }

    //
    // Log generated x and y data.
    //

    // log_float_data(vec_size, (void *)x, "generated x vector:");
    // log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the
    //      same.
    //

    pfn_quantize_row_q6_K(x, q6x, vec_size);
    pfn_quantize_row_q8_K(y, q8y, vec_size);

    //
    // log q6 and q8_k quant data.
    //

    // log_q6_k_quant_data(vec_size, (void *)q6x, true);
    // log_q8_k_quant_data(vec_size, (void *)q8y, true);

    //
    // Announce perf test.
    //

    printf("Running vec_dot_q6_K_q8_K performance test for DLL\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t worse_time_Dll = 0;
    int64_t best_time_Dll = MAXLONG64;

    if (pfn_ggml_vec_dot_q6_K_q8_K != NULL) {

        for (j = 0; j < iter_repeat; j += 1) {

            //
            // Compute the time to do the summation of the product of vector elements.
            //
        
            const int64_t start_time = local_timer_us();
            for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
                pfn_ggml_vec_dot_q6_K_q8_K(vec_size, &sum, 0, q6x, 0, q8y, 0, 1);
            }
        
            const int64_t total_time = local_timer_us() - start_time;

            if (total_time < best_time_Dll) {
                best_time_Dll = total_time;
            }

            if (total_time > worse_time_Dll) {
                worse_time_Dll = total_time;
            }
        }

        //
        // Log vector sum of products.
        //

        printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);
    }
 
    //
    // Announce perf test.
    //

    printf("Running vec_dot_q6_K_q8_K performance test for Local\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t worse_time_Local = 0;
    int64_t best_time_Local = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = local_timer_us();
        for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
            local_vec_dot_q6_K_q8_K(vec_size, &sum, 0, q6x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = local_timer_us() - start_time;

        if (total_time < best_time_Local) {
            best_time_Local = total_time;
        }

        if (total_time > worse_time_Local) {
            worse_time_Local = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    printf("vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q6_K_q8_K:",
                 best_time_Dll,
                 worse_time_Dll,
                 best_time_Local,
                 worse_time_Local,
                 iter_vec_dot_q6_K_q8_K);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q6x) {
        zfree(q6x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void * get_proc(HMODULE hMod, const char *func_name) {
    void* func = (void*)GetProcAddress(hMod, func_name);
    if (func == NULL) {
        printf("failed to get ProcAddr for '%s'\n", func_name);
        exit(1);
    }

    if (!FlushInstructionCache(GetCurrentProcess(), func, 4 * 1024)) {
        printf("ERROR in FlushInstructionCache() on '%s'\n", func_name);
    }

    return func;
}

int main(int argc, char *argv[]) {

    const char* szDll = NULL;

    // Quick simple command line args: lltest [dll] [iter_repeat_count]
    if (argc >= 2) {
        szDll = argv[1];
        printf("Overriding default DLL with '%s'\n", szDll);
    }
    if (argc >= 3) {
        printf("Overriding default iter_repeat count of %d", iter_repeat); 
        iter_repeat = atoi(argv[2]);
        printf(" with %d\n", iter_repeat);
    }

    if (szDll != NULL) {
        HMODULE za_dll = LoadLibraryA(szDll);
        if (za_dll == NULL) {
            printf("failed to load library for '%s'\n", szDll);
            exit(1);
        }
 
        pfn_ggml_init_tables = (PFN_ggml_init_tables)(get_proc(za_dll, "ggml_init_tables"));
        pfn_ggml_time_init = (PFN_ggml_time_init)(get_proc(za_dll, "ggml_time_init"));

        pfn_ggml_vec_dot_q2_K_q8_K = (PFN_ggml_vec_dot_q2_K_q8_K)(get_proc(za_dll, "ggml_vec_dot_q2_K_q8_K"));
        pfn_ggml_vec_dot_q3_K_q8_K = (PFN_ggml_vec_dot_q3_K_q8_K)(get_proc(za_dll, "ggml_vec_dot_q3_K_q8_K"));
        pfn_ggml_vec_dot_q4_K_q8_K = (PFN_ggml_vec_dot_q4_K_q8_K)(get_proc(za_dll, "ggml_vec_dot_q4_K_q8_K"));
        pfn_ggml_vec_dot_q6_K_q8_K = (PFN_ggml_vec_dot_q6_K_q8_K)(get_proc(za_dll, "ggml_vec_dot_q6_K_q8_K"));

        pfn_quantize_row_q2_K = (PFN_quantize_row_q2_K)(get_proc(za_dll, "quantize_row_q2_K"));
        pfn_quantize_row_q3_K = (PFN_quantize_row_q3_K)(get_proc(za_dll, "quantize_row_q3_K"));
        pfn_quantize_row_q4_K = (PFN_quantize_row_q4_K)(get_proc(za_dll, "quantize_row_q4_K"));
        pfn_quantize_row_q6_K = (PFN_quantize_row_q6_K)(get_proc(za_dll, "quantize_row_q6_K"));
        pfn_quantize_row_q8_K = (PFN_quantize_row_q8_K)(get_proc(za_dll, "quantize_row_q8_K"));

    } else {
        pfn_ggml_init_tables = NULL;
        pfn_ggml_time_init = NULL;

        pfn_ggml_vec_dot_q2_K_q8_K = NULL;
        pfn_ggml_vec_dot_q3_K_q8_K = NULL;
        pfn_ggml_vec_dot_q4_K_q8_K = NULL;
        pfn_ggml_vec_dot_q6_K_q8_K = NULL;

        pfn_quantize_row_q2_K = local_quantize_row_q2_K;
        pfn_quantize_row_q3_K = local_quantize_row_q3_K;
        pfn_quantize_row_q4_K = local_quantize_row_q4_K;
        pfn_quantize_row_q6_K = local_quantize_row_q6_K;
        pfn_quantize_row_q8_K = local_quantize_row_q8_K;
    }

    printf("Running test for Local vs. Dll ('%s')\n", szDll);

    // initialization

    if (szDll != NULL) {
        pfn_ggml_time_init(); // is needed for zo if invoked
        pfn_ggml_init_tables();
    }

    local_init_tables();
    local_timer_init();

    // first one to warm up

    vec_dot_q6_K_q8_K(vector_size);

    // real data

    vec_dot_q2_K_q8_K(vector_size);

    vec_dot_q3_K_q8_K(vector_size);

    vec_dot_q4_K_q8_K(vector_size);

    vec_dot_q6_K_q8_K(vector_size);

    // report results

    printf("Performance test improvement report - Vector Size %d\n\n", vector_size);

    for (uint32_t j = 0; j < improve_table_size; j += 1) {
        char * text = improve_table[j].text;
        int64_t best_time_Dll = improve_table[j].best_time_Dll;
        int64_t best_time_Local = improve_table[j].best_time_Local;
        int64_t worse_time_Dll = improve_table[j].worse_time_Dll;
        int64_t worse_time_Local = improve_table[j].worse_time_Local;
        int64_t iter = improve_table[j].iter;
        
        if (szDll == NULL) {
            //
            // no DLL - just local stats
            //
            best_time_Local = max(1, best_time_Local);
            printf("  loop iterations %zd - repeat count %zd\n", iter, iter_repeat);
            printf("  best Local time %zdns - %zdns\n", best_time_Local, worse_time_Local);
            printf("    time per iteration %6.2fus - %6.2fus\n\n", 
                (float)best_time_Local * 1000. / (float)iter,
                (float)worse_time_Local * 1000. / (float)iter);

        } else {
            best_time_Local = max(1, best_time_Local);
            float multiple = (float)best_time_Dll / (float)best_time_Local;
            printf("%s Local is %5.2f x speed of DLL\n", text, multiple);
            printf("  loop iterations %zd - repeat count %zd\n", iter, iter_repeat);
            printf("  best Dll time %zdns - %zdns\n", best_time_Dll, worse_time_Dll);
            printf("    time per iteration %6.2fus - %6.2fus\t\t%s\n", 
                (float)best_time_Dll * 1000. / (float)(iter),
                (float)worse_time_Dll * 1000. / (float)(iter),
                (best_time_Dll < best_time_Local) ? "<<< Dll is faster!" : "");
            printf("  best Local time %zdns - %zdns\n", best_time_Local, worse_time_Local);
            printf("    time per iteration %6.2fus - %6.2fus\t\t%s\n\n", 
                (float)best_time_Local * 1000. / (float)(iter),
                (float)worse_time_Local * 1000. / (float)(iter),
                (best_time_Local < best_time_Dll) ? "<<< Local is faster!" : "");
        }
    }

    return 0;
}
