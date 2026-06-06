#pragma warning (disable:4201) // nameless struct/union

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <intrin.h>
#include <immintrin.h>
#include "ggml.h"

//
// Define unused parameter macro for compatility with ggml.
//

#define GGML_UNUSED(x) (void)(x)

//
// Control for turning data logging and show improvement on.
//

BOOLEAN log_data = FALSE;               // used to turn on data logging
BOOLEAN show_improve = FALSE;           // used to show improvement data

//
// Define vector size.
//

#define VECTOR_SIZE_4096u 4096u         // use this setting with logging off for perf runs
#define VECTOR_SIZE_1024u 1024u         // use this setting with logging on for test runs
#define VECTOR_SIZE_256u 256u           // alternate setting for testing

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

#define VECTOR_SIZE_EXTENSION 16 + 6    // extension of vector length, but not quants 

UINT32 vector_size = VECTOR_SIZE_4096u; // default vector size

//
// Define random number filter to limit the range of generated values.
//
// N.B. If compiling with clang, narrow the range of q2 quants. The code generated with
//      clang produces different results from msvc for quantize_row_q2_k in some cases.
//      It is unclear why this is happening since it is the same code.
//

#define FLOAT_FILTER (1u << 14)
#define FLOAT_FILTER_SILU (1u << 10)
#define FLOAT_FILTER_SOFT_MAX (1u << 10)
//#define FLOAT_FILTER_Q2 (1u << 10) // if compiling for clang
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

//
// Define ggml prototypes.
//

typedef void (*PFN_ggml_disable_core_parking)(void);
PFN_ggml_disable_core_parking pfn_ggml_disable_core_parking_AVX2;

typedef void (*PFN_ggml_init_tables)(void);
PFN_ggml_init_tables pfn_ggml_init_tables_AVX2;
PFN_ggml_init_tables pfn_ggml_init_tables_AVX512;

typedef void (*PFN_ggml_time_init)();
PFN_ggml_time_init pfn_ggml_time_init_AVX2;
PFN_ggml_time_init pfn_ggml_time_init_AVX512;

typedef int64_t (*PFN_ggml_time_us)();
PFN_ggml_time_us pfn_ggml_time_us;

#define QK_K 256u
#define QK4_0 32u
#define QK8_0 32u

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

typedef block_q2_K block_q2_K_repack;

C_ASSERT(!(sizeof(block_q2_K) % sizeof(uint32_t)));

typedef void (*PFN_quantize_row_q2_K)(const float *, block_q2_K *, int64_t);
PFN_quantize_row_q2_K pfn_quantize_row_q2_K_AVX2;
PFN_quantize_row_q2_K pfn_quantize_row_q2_K_AVX512;

void
quantize_row_q2_K (
    const float * x,
    block_q2_K * y,
    int64_t k);

typedef void (*PFN_dequantize_row_q2_K)(const block_q2_K *, float *, int64_t);
PFN_dequantize_row_q2_K pfn_dequantize_row_q2_K_AVX2;
PFN_dequantize_row_q2_K pfn_dequantize_row_q2_K_AVX512;

void
dequantize_row_q2_K (
    const block_q2_K * x,
    float * y,
    int64_t k);

typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

typedef block_q3_K block_q3_K_repack;

C_ASSERT(sizeof(block_q3_K) == (sizeof(ggml_half) + QK_K / 8 + QK_K / 4 + 12));

typedef void (*PFN_quantize_row_q3_K)(const float *, block_q3_K *, int64_t);
PFN_quantize_row_q3_K pfn_quantize_row_q3_K_AVX2;
PFN_quantize_row_q3_K pfn_quantize_row_q3_K_AVX512;

void
quantize_row_q3_K (
    const float * x,
    block_q3_K * y,
    int64_t k);

typedef void (*PFN_dequantize_row_q3_K)(const block_q3_K *, float *, int64_t);
PFN_dequantize_row_q3_K pfn_dequantize_row_q3_K_AVX2;
PFN_dequantize_row_q3_K pfn_dequantize_row_q3_K_AVX512;

void
dequantize_row_q3_K (
    const block_q3_K * x,
    float * y,
    int64_t k);

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

typedef block_q4_K block_q4_K_repack;

C_ASSERT(!(sizeof(block_q4_K) % sizeof(uint32_t)));

void
quantize_row_q4_K (
    const float * x,
    block_q4_K * y,
    int64_t k);

typedef void (*PFN_quantize_row_q4_K)(const float *, block_q4_K *, int64_t);
PFN_quantize_row_q4_K pfn_quantize_row_q4_K_AVX2;
PFN_quantize_row_q4_K pfn_quantize_row_q4_K_AVX512;

typedef void (*PFN_dequantize_row_q4_K)(const block_q4_K *, float *, int64_t);
PFN_dequantize_row_q4_K pfn_dequantize_row_q4_K_AVX2;
PFN_dequantize_row_q4_K pfn_dequantize_row_q4_K_AVX512;

void
dequantize_row_q4_K (
    const block_q4_K * x,
    float * y,
    int64_t k);

typedef struct {
    ggml_half d;         // delta
    uint8_t qs[QK4_0/2]; // nibbles - two 4-bit quants
} block_q4_0;

C_ASSERT(sizeof(block_q4_0) == 18);

void
quantize_row_q4_0 (
    const float * x,
    block_q4_0 * y,
    int64_t k);

typedef void (*PFN_quantize_row_q4_0)(const float *, block_q4_0 *, int64_t);
PFN_quantize_row_q4_0 pfn_quantize_row_q4_0_AVX2;
PFN_quantize_row_q4_0 pfn_quantize_row_q4_0_AVX512;

typedef void (*PFN_dequantize_row_q4_0)(const block_q4_0 *, float *, int64_t);
PFN_dequantize_row_q4_0 pfn_dequantize_row_q4_0_AVX2;
PFN_dequantize_row_q4_0 pfn_dequantize_row_q4_0_AVX512;

void
dequantize_row_q4_0 (
    const block_q4_0 * x,
    float * y,
    int64_t k);

typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale 
} block_q6_K;

typedef block_q6_K block_q6_K_repack;

C_ASSERT(sizeof(block_q6_K) == 210); 

void
quantize_row_q6_K (
    const float * x,
    block_q6_K * y,
    int64_t k);

typedef void (*PFN_quantize_row_q6_K)(const float *, block_q6_K *, int64_t);
PFN_quantize_row_q6_K pfn_quantize_row_q6_K_AVX2;
PFN_quantize_row_q6_K pfn_quantize_row_q6_K_AVX512;

typedef void (*PFN_dequantize_row_q6_K)(const block_q6_K *, float *, int64_t);
PFN_dequantize_row_q6_K pfn_dequantize_row_q6_K_AVX2;
PFN_dequantize_row_q6_K pfn_dequantize_row_q6_K_AVX512;

void
dequantize_row_q6_K (
    const block_q6_K * x,
    float * y,
    int64_t k);

typedef struct {
    float d;                            // delta
    int8_t qs[QK_K];                    // quants
    int16_t bsums[QK_K/16];             // sum of quants in groups of 16
} block_q8_K;

typedef block_q8_K block_q8_K_repack;

C_ASSERT(!(sizeof(block_q8_K) % sizeof(uint32_t)));

typedef void (*PFN_quantize_row_q8_K)(const float *, block_q8_K *, int64_t);
PFN_quantize_row_q8_K pfn_quantize_row_q8_K_AVX2;
PFN_quantize_row_q8_K pfn_quantize_row_q8_K_AVX512;

void
quantize_row_q8_K (
    const float * x,
    block_q8_K * y,
    int64_t k);

typedef void (*PFN_dequantize_row_q8_K)(const block_q8_K *, float *, int64_t);
PFN_dequantize_row_q8_K pfn_dequantize_row_q8_K_AVX2;
PFN_dequantize_row_q8_K pfn_dequantize_row_q8_K_AVX512;

void
dequantize_row_q8_K (
    const block_q8_K * x,
    float * y,
    int64_t k);

typedef struct {
    ggml_half d;                        // delta
    int8_t qs[QK8_0];                   // quants
} block_q8_0;

C_ASSERT(!(sizeof(block_q8_0) % sizeof(uint16_t)));

typedef void (*PFN_quantize_row_q8_0)(const float *, block_q8_0 *, int64_t);
PFN_quantize_row_q8_0 pfn_quantize_row_q8_0_AVX2;
PFN_quantize_row_q8_0 pfn_quantize_row_q8_0_AVX512;

void
quantize_row_q8_0 (
    const float * x,
    block_q8_0 * y,
    int64_t k);

typedef void (*PFN_dequantize_row_q8_0)(const block_q8_0 *, float *, int64_t);
PFN_dequantize_row_q8_0 pfn_dequantize_row_q8_0_AVX2;
PFN_dequantize_row_q8_0 pfn_dequantize_row_q8_0_AVX512;

void
dequantize_row_q8_0 (
    const block_q8_0 * x,
    const float * y,
    int64_t k);

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K / 2];                // quants interleaved packed two per byte
} block_q4_0_repack;

C_ASSERT((sizeof(block_q4_0) * 8) == sizeof(block_q4_0_repack));

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K];                    // quants interleaved
} block_q8_0_repack;

C_ASSERT((sizeof(block_q8_0) * 8) == sizeof(block_q8_0_repack));

typedef void (*PFN_make_q4_0_repack_quant)(uint64_t, block_q4_0_repack *, block_q4_0 *);
PFN_make_q4_0_repack_quant pfn_make_q4_0_repack_quant_AVX512;

void
make_q4_0_repack_quant (
    uint64_t ne,
    block_q4_0_repack * out,
    block_q4_0 * in);

typedef void (*PFN_make_q2_k_repack_quant)(uint64_t, block_q2_K_repack *, block_q2_K *);
PFN_make_q2_k_repack_quant pfn_make_q2_k_repack_quant_AVX512;

void
make_q2_k_repack_quant (
    uint64_t ne,
    block_q2_K_repack * out,
    block_q2_K * in);

typedef void (*PFN_make_q3_k_repack_quant)(uint64_t, block_q3_K_repack *, block_q3_K *);
PFN_make_q3_k_repack_quant pfn_make_q3_k_repack_quant_AVX512;

void
make_q3_k_repack_quant (
    uint64_t ne,
    block_q3_K_repack * out,
    block_q3_K * in);

typedef void (*PFN_make_q4_k_repack_quant)(uint64_t, block_q4_K_repack *, block_q4_K *);
PFN_make_q4_k_repack_quant pfn_make_q4_k_repack_quant_AVX512;

void
make_q4_k_repack_quant (
    uint64_t ne,
    block_q4_K_repack * out,
    block_q4_K * in);

typedef void (*PFN_make_q6_k_repack_quant)(uint64_t, block_q6_K_repack *, block_q6_K *);
PFN_make_q6_k_repack_quant pfn_make_q6_k_repack_quant_AVX512;

void
make_q6_k_repack_quant (
    uint64_t ne,
    block_q6_K_repack * out,
    block_q6_K * in);

typedef void (*PFN_make_q8_0_repack_quant)(uint64_t, block_q8_0_repack *, block_q8_0 *);
PFN_make_q8_0_repack_quant pfn_make_q8_0_repack_quant_AVX512;

void
make_q8_0_repack_quant (
    uint64_t ne,
    block_q8_0_repack * out,
    block_q8_0 * in);

typedef void (*PFN_make_q236_k_q8_k_repack_quant)(uint64_t, block_q8_K_repack *, block_q8_K *);
PFN_make_q236_k_q8_k_repack_quant pfn_make_q236_k_q8_k_repack_quant_AVX512;

void
make_q236_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in);

typedef void (*PFN_make_q4_k_q8_k_repack_quant)(uint64_t, block_q8_K_repack *, block_q8_K *);
PFN_make_q4_k_q8_k_repack_quant pfn_make_q4_k_q8_k_repack_quant_AVX512;

void
make_q4_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in);

typedef void (*PFN_xx_vec_dot_q4_0_q8_0_x8)(uint32_t, float *, size_t, const block_q4_0_repack *, size_t, const block_q8_0_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q4_0_q8_0_x8 pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512;

void
xx_vec_dot_q4_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_xx_vec_dot_q2_k_q8_k_x8)(uint32_t, float *, size_t, const block_q2_K_repack *, size_t, const block_q8_K_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q2_k_q8_k_x8 pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512;

void
xx_vec_dot_q2_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_xx_vec_dot_q3_k_q8_k_x8)(uint32_t, float *, size_t, const block_q3_K_repack *, size_t, const block_q8_K_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q3_k_q8_k_x8 pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512;

void
xx_vec_dot_q3_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q3_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_xx_vec_dot_q4_k_q8_k_x8)(uint32_t, float *, size_t, const block_q4_K_repack *, size_t, const block_q8_K_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q4_k_q8_k_x8 pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512;

void
xx_vec_dot_q4_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t rn_nb1,
    const block_q4_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_xx_vec_dot_q6_k_q8_k_x8)(uint32_t, float *, size_t, const block_q6_K_repack *, size_t, const block_q8_K_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q6_k_q8_k_x8 pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512;

void
xx_vec_dot_q6_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t rn_nb1,
    const block_q6_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_xx_vec_dot_q8_0_q8_0_x8)(uint32_t, float *, size_t, const block_q8_0_repack *, size_t, const block_q8_0_repack *, size_t, uint32_t);
PFN_xx_vec_dot_q8_0_q8_0_x8 pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512;

void
xx_vec_dot_q8_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t ncols,
    uint32_t nrows);

typedef void (*PFN_quantize_row_q4_0_x8)(const float *, block_q4_0 *, uint32_t);
PFN_quantize_row_q4_0_x8 pfn_quantize_row_q4_0_x8_AVX512;

void
quantize_row_q4_0_x8 (
    const float * x,
    block_q4_0 * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q2_k_x8)(const float *, block_q2_K *, uint32_t);
PFN_quantize_row_q2_k_x8 pfn_quantize_row_q2_k_x8_AVX512;

void                   
quantize_row_q2_k_x8 (
    const float * x,
    block_q2_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q3_k_x8)(const float *, block_q3_K *, uint32_t);
PFN_quantize_row_q3_k_x8 pfn_quantize_row_q3_k_x8_AVX512;

void                   
quantize_row_q3_k_x8 (
    const float * x,
    block_q3_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q4_k_x8)(const float *, block_q4_K *, uint32_t);
PFN_quantize_row_q4_k_x8 pfn_quantize_row_q4_k_x8_AVX512;

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q6_k_x8)(const float *, block_q6_K *, uint32_t);
PFN_quantize_row_q6_k_x8 pfn_quantize_row_q6_k_x8_AVX512;

void                   
quantize_row_q6_k_x8 (
    const float * x,
    block_q6_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q236_k_q8_k_x8)(const float *, block_q8_K *, uint32_t);
PFN_quantize_row_q236_k_q8_k_x8 pfn_quantize_row_q236_k_q8_k_x8_AVX512;

void                   
quantize_row_q236_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q4_k_q8_k_x8)(const float *, block_q8_K *, uint32_t);
PFN_quantize_row_q4_k_q8_k_x8 pfn_quantize_row_q4_k_q8_k_x8_AVX512;

void                   
quantize_row_q4_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size);

typedef void (*PFN_quantize_row_q8_0_x8)(const float *, block_q8_0 *, uint32_t);
PFN_quantize_row_q8_0_x8 pfn_quantize_row_q8_0_x8_AVX512;

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint32_t vec_size);

typedef void (*PFN_ggml_vec_silu_f32)(const int, float *, const float *);
PFN_ggml_vec_silu_f32 pfn_ggml_vec_silu_f32_AVX2;
PFN_ggml_vec_silu_f32 pfn_ggml_vec_silu_f32_AVX512;

void
ggml_vec_silu_f32 (
    const int n,
    float * y,
    const float * x);

//typedef double ggml_float;
typedef float ggml_float; // ****** consider changing to float

typedef ggml_float (*PFN_ggml_vec_soft_max_f32)(const int, float *, const float *, float);
PFN_ggml_vec_soft_max_f32 pfn_ggml_vec_soft_max_f32_AVX2;
PFN_ggml_vec_soft_max_f32 pfn_ggml_vec_soft_max_f32_AVX512;

ggml_float
ggml_vec_soft_max_f32 (
    const int n,
    float * y,
    const float * x,
    float max);

typedef void (*PFN_ggml_bf16_to_fp32_row)(const ggml_bf16_t *, float *, const int64_t);
PFN_ggml_bf16_to_fp32_row pfn_ggml_bf16_to_fp32_row_AVX2;
PFN_ggml_bf16_to_fp32_row pfn_ggml_bf16_to_fp32_row_AVX512;

void
ggml_bf16_to_fp32_row (
    const ggml_bf16_t * x,
    float * y,
    const int64_t n);

typedef void (*PFN_ggml_fp32_to_bf16_row)(const float *, ggml_bf16_t *, const int64_t);
PFN_ggml_fp32_to_bf16_row pfn_ggml_fp32_to_bf16_row_AVX2;
PFN_ggml_fp32_to_bf16_row pfn_ggml_fp32_to_bf16_row_AVX512;

void
ggml_fp32_to_bf16_row (
    const float * x,
    ggml_bf16_t * y,
    const int64_t n);

typedef void (*PFN_ggml_fp16_to_fp32_row)(const ggml_fp16_t *, float *, const int64_t);
PFN_ggml_fp16_to_fp32_row pfn_ggml_fp16_to_fp32_row_AVX2;
PFN_ggml_fp16_to_fp32_row pfn_ggml_fp16_to_fp32_row_AVX512;

void
ggml_fp16_to_fp32_row (
    const ggml_fp16_t * x,
    float * y,
    const int64_t n);

typedef void (*PFN_ggml_fp32_to_fp16_row)(const float *, ggml_fp16_t *, const int64_t);
PFN_ggml_fp32_to_fp16_row pfn_ggml_fp32_to_fp16_row_AVX2;
PFN_ggml_fp32_to_fp16_row pfn_ggml_fp32_to_fp16_row_AVX512;

void
ggml_fp32_to_fp16_row (
    const float * x,
    ggml_fp16_t * y,
    const int64_t n);

typedef void (*PFN_ggml_vec_abs_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_abs_f32 pfn_ggml_vec_abs_f32_AVX2;
PFN_ggml_vec_abs_f32 pfn_ggml_vec_abs_f32_AVX512;

void
ggml_vec_abs_f32 (
    const uint64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_add_f32)(const uint64_t, float *, const float *, const float *);
PFN_ggml_vec_add_f32 pfn_ggml_vec_add_f32_AVX2;
PFN_ggml_vec_add_f32 pfn_ggml_vec_add_f32_AVX512;

void
ggml_vec_add_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_add1_f32)(const uint64_t, float *, const float *, const float);
PFN_ggml_vec_add1_f32 pfn_ggml_vec_add1_f32_AVX2;
PFN_ggml_vec_add1_f32 pfn_ggml_vec_add1_f32_AVX512;

void
ggml_vec_add1_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float v);

typedef void (*PFN_ggml_vec_acc_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_acc_f32 pfn_ggml_vec_acc_f32_AVX2;
PFN_ggml_vec_acc_f32 pfn_ggml_vec_acc_f32_AVX512;

void
ggml_vec_acc_f32 (
    const uint64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_acc1_f32)(const uint64_t, float *, const float);
PFN_ggml_vec_acc1_f32 pfn_ggml_vec_acc1_f32_AVX2;
PFN_ggml_vec_acc1_f32 pfn_ggml_vec_acc1_f32_AVX512;

void
ggml_vec_acc1_f32 (
    const uint64_t n,
    float * y,
    const float v);

typedef void (*PFN_ggml_vec_sub_f32)(const uint64_t, float *, const float *, const float *);
PFN_ggml_vec_sub_f32 pfn_ggml_vec_sub_f32_AVX2;
PFN_ggml_vec_sub_f32 pfn_ggml_vec_sub_f32_AVX512;

void
ggml_vec_sub_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_set_f32)(const uint64_t, float *, const float);
PFN_ggml_vec_set_f32 pfn_ggml_vec_set_f32_AVX2;
PFN_ggml_vec_set_f32 pfn_ggml_vec_set_f32_AVX512;

void
ggml_vec_set_f32 (
    const uint64_t n,
    float * x,
    const float v);

typedef void (*PFN_ggml_vec_cpy_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_cpy_f32 pfn_ggml_vec_cpy_f32_AVX2;
PFN_ggml_vec_cpy_f32 pfn_ggml_vec_cpy_f32_AVX512;

void
ggml_vec_cpy_f32 (
    const uint64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_neg_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_neg_f32 pfn_ggml_vec_neg_f32_AVX2;
PFN_ggml_vec_neg_f32 pfn_ggml_vec_neg_f32_AVX512;

void
ggml_vec_neg_f32 (
    const uint64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_mul_f32)(const uint64_t, float *, const float *, const float *);
PFN_ggml_vec_mul_f32 pfn_ggml_vec_mul_f32_AVX2;
PFN_ggml_vec_mul_f32 pfn_ggml_vec_mul_f32_AVX512;

void
ggml_vec_mul_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_mul1_f32)(const uint64_t, float *, const float *, const float);
PFN_ggml_vec_mul1_f32 pfn_ggml_vec_mul1_f32_AVX2;
PFN_ggml_vec_mul1_f32 pfn_ggml_vec_mul1_f32_AVX512;

void
ggml_vec_mul1_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float v);

typedef void (*PFN_ggml_vec_div_f32)(const uint64_t, float *, const float *, const float *);
PFN_ggml_vec_div_f32 pfn_ggml_vec_div_f32_AVX2;
PFN_ggml_vec_div_f32 pfn_ggml_vec_div_f32_AVX512;

void
ggml_vec_div_f32 (
    const uint64_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_sum_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_sum_f32 pfn_ggml_vec_sum_f32_AVX2;
PFN_ggml_vec_sum_f32 pfn_ggml_vec_sum_f32_AVX512;

void
ggml_vec_sum_f32 (
    const uint64_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_sumsq_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_sumsq_f32 pfn_ggml_vec_sumsq_f32_AVX2;
PFN_ggml_vec_sumsq_f32 pfn_ggml_vec_sumsq_f32_AVX512;

void
ggml_vec_sumsq_f32 (
    const uint64_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_sumsq_bf16)(const uint64_t, float *, const ggml_bf16_t *);
PFN_ggml_vec_sumsq_bf16 pfn_ggml_vec_sumsq_bf16_AVX2;
PFN_ggml_vec_sumsq_bf16 pfn_ggml_vec_sumsq_bf16_AVX512;

void
ggml_vec_sumsq_bf16 (
    const uint64_t n,
    float * s,
    const ggml_bf16_t * x);

typedef void (*PFN_ggml_vec_max_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_max_f32 pfn_ggml_vec_max_f32_AVX2;
PFN_ggml_vec_max_f32 pfn_ggml_vec_max_f32_AVX512;

void
ggml_vec_max_f32 (
    const int32_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_scale_f16)(const uint64_t, ggml_fp16_t *, const float);
PFN_ggml_vec_scale_f16 pfn_ggml_vec_scale_f16_AVX2;
PFN_ggml_vec_scale_f16 pfn_ggml_vec_scale_f16_AVX512;

void
ggml_vec_scale_f16 (
    const uint64_t n,
    ggml_fp16_t * y,
    const float v);

typedef void (*PFN_ggml_vec_scale_f32)(const uint64_t, float *, const float);
PFN_ggml_vec_scale_f32 pfn_ggml_vec_scale_f32_AVX2;
PFN_ggml_vec_scale_f32 pfn_ggml_vec_scale_f32_AVX512;

void
ggml_vec_scale_f32 (
    const uint64_t n,
    float * y,
    const float v);

typedef void (*PFN_ggml_vec_sqrt_f32)(const uint64_t, float *, const float *);
PFN_ggml_vec_sqrt_f32 pfn_ggml_vec_sqrt_f32_AVX2;
PFN_ggml_vec_sqrt_f32 pfn_ggml_vec_sqrt_f32_AVX512;

void
ggml_vec_sqrt_f32 (
    const uint64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_mad_f16)(const uint64_t, ggml_fp16_t *, const ggml_fp16_t *, const float);
PFN_ggml_vec_mad_f16 pfn_ggml_vec_mad_f16_AVX2;
PFN_ggml_vec_mad_f16 pfn_ggml_vec_mad_f16_AVX512;

void
ggml_vec_mad_f16 (
    const uint64_t n,
    ggml_fp16_t * y,
    const ggml_fp16_t * x,
    const float v);

typedef void (*PFN_ggml_vec_mad_f32)(const uint64_t, float *, const float *, const float);
PFN_ggml_vec_mad_f32 pfn_ggml_vec_mad_f32_AVX2;
PFN_ggml_vec_mad_f32 pfn_ggml_vec_mad_f32_AVX512;

void
ggml_vec_mad_f32 (
    const uint64_t n,
    float * y,
    const float * x,
    const float v);

typedef void (*PFN_ggml_vec_normsq_f32)(const uint64_t, float *, float, float *, const float *);
PFN_ggml_vec_normsq_f32 pfn_ggml_vec_normsq_f32_AVX2;
PFN_ggml_vec_normsq_f32 pfn_ggml_vec_normsq_f32_AVX512;

void
ggml_vec_normsq_f32 (
    const uint64_t n,
    float * s,
    const float mean,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_dot_bf16)(const int, float *, size_t, const ggml_bf16_t *, size_t, const ggml_bf16_t *, size_t, int);
PFN_ggml_vec_dot_bf16 pfn_ggml_vec_dot_bf16_AVX2;
PFN_ggml_vec_dot_bf16 pfn_ggml_vec_dot_bf16_AVX512;

void
ggml_vec_dot_bf16 (
    const int n,
    float * s,
    size_t bs,
    const ggml_bf16_t * x,
    size_t bx,
    const ggml_bf16_t * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_f16)(const int, float *, size_t, const ggml_fp16_t *, size_t, const ggml_fp16_t *, size_t, int);
PFN_ggml_vec_dot_f16 pfn_ggml_vec_dot_f16_AVX2;
PFN_ggml_vec_dot_f16 pfn_ggml_vec_dot_f16_AVX512;

void
ggml_vec_dot_f16 (
    const int n,
    float * s,
    size_t bs,
    const ggml_fp16_t * x,
    size_t bx,
    const ggml_fp16_t * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_f32)(const int, float *, size_t, const float *, size_t, const float *, size_t, int);
PFN_ggml_vec_dot_f32 pfn_ggml_vec_dot_f32_AVX2;
PFN_ggml_vec_dot_f32 pfn_ggml_vec_dot_f32_AVX512;

void
ggml_vec_dot_f32 (
    const int n,
    float * s,
    size_t bs,
    const float * x,
    size_t bx,
    const float * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_bf16_f32)(const int, float *, size_t, const ggml_bf16_t *, size_t, const float *, size_t, int);
PFN_ggml_vec_dot_bf16_f32 pfn_ggml_vec_dot_bf16_f32_AVX2;
PFN_ggml_vec_dot_bf16_f32 pfn_ggml_vec_dot_bf16_f32_AVX512;

void
ggml_vec_dot_bf16_f32 (
    const int n,
    float * s,
    size_t bs,
    const ggml_bf16_t * x,
    size_t bx,
    const float * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_f16_f32)(const int, float *, size_t, const ggml_fp16_t *, size_t, const float *, size_t, int);
PFN_ggml_vec_dot_f16_f32 pfn_ggml_vec_dot_f16_f32_AVX2;
PFN_ggml_vec_dot_f16_f32 pfn_ggml_vec_dot_f16_f32_AVX512;

void
ggml_vec_dot_f16_f32 (
    const int n,
    float * s,
    size_t bs,
    const ggml_fp16_t * x,
    size_t bx,
    const float * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q2_K_q8_K)(const int, float *, size_t, const block_q2_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q2_K_q8_K pfn_ggml_vec_dot_q2_K_q8_K_AVX2;
PFN_ggml_vec_dot_q2_K_q8_K pfn_ggml_vec_dot_q2_K_q8_K_AVX512;

void
ggml_vec_dot_q2_K_q8_K (
    const int n,
    float * s,
    size_t bs,
    const block_q2_K * x,
    size_t bx,
    const block_q8_K * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q3_K_q8_K)(const int, float *, size_t, const block_q3_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q3_K_q8_K pfn_ggml_vec_dot_q3_K_q8_K_AVX2;
PFN_ggml_vec_dot_q3_K_q8_K pfn_ggml_vec_dot_q3_K_q8_K_AVX512;

void
ggml_vec_dot_q3_K_q8_K (
    const int n,
    float * s,
    size_t bs,
    const block_q3_K * x,
    size_t bx,
    const block_q8_K * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q4_K_q8_K)(const int, float *, size_t, const block_q4_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q4_K_q8_K pfn_ggml_vec_dot_q4_K_q8_K_AVX2;
PFN_ggml_vec_dot_q4_K_q8_K pfn_ggml_vec_dot_q4_K_q8_K_AVX512;

void
ggml_vec_dot_q4_K_q8_K (
    const int n,
    float * s,
    size_t bs,
    const block_q4_K * x,
    size_t bx,
    const block_q8_K * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q6_K_q8_K)(const int, float *, size_t, const block_q6_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q6_K_q8_K pfn_ggml_vec_dot_q6_K_q8_K_AVX2;
PFN_ggml_vec_dot_q6_K_q8_K pfn_ggml_vec_dot_q6_K_q8_K_AVX512;

void
ggml_vec_dot_q6_K_q8_K (
    const int n,
    float * s,
    size_t bs,
    const block_q6_K * x,
    size_t bx,
    const block_q8_K * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q4_0_q8_0)(const int, float *, size_t, const void *, size_t, const void *, size_t, int);
PFN_ggml_vec_dot_q4_0_q8_0 pfn_ggml_vec_dot_q4_0_q8_0_AVX2;
PFN_ggml_vec_dot_q4_0_q8_0 pfn_ggml_vec_dot_q4_0_q8_0_AVX512;

void
ggml_vec_dot_q4_0_q8_0 (
    const int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q8_0_q8_0)(const int, float *, size_t, const void *, size_t, const void *, size_t, int);
PFN_ggml_vec_dot_q8_0_q8_0 pfn_ggml_vec_dot_q8_0_q8_0_AVX2;
PFN_ggml_vec_dot_q8_0_q8_0 pfn_ggml_vec_dot_q8_0_q8_0_AVX512;

void
ggml_vec_dot_q8_0_q8_0 (
    const int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc);

typedef float (*PFN_ggml_cosine_similarity_f32)(const int, const float *, const float *);
PFN_ggml_cosine_similarity_f32 pfn_ggml_cosine_similarity_f32_AVX2;
PFN_ggml_cosine_similarity_f32 pfn_ggml_cosine_similarity_f32_AVX512;

float
ggml_cosine_similarity_f32 (
    const int n,
    const float *x,
    const float *y);

typedef float (*PFN_ggml_cosine_similarity_bf16)(const int, const ggml_bf16_t *, const ggml_bf16_t *);
PFN_ggml_cosine_similarity_bf16 pfn_ggml_cosine_similarity_bf16_AVX2;
PFN_ggml_cosine_similarity_bf16 pfn_ggml_cosine_similarity_bf16_AVX512;

float
ggml_cosine_similarity_bf16 (
    const int n,
    const ggml_bf16_t *x,
    const ggml_bf16_t *y);

//
// Declare logfile descriptor.
//

FILE * logfile = NULL;

//
// Declare test improvement report table.
//

typedef struct _improve_desc {
    char * text;
    int64_t best_time_first;
    char * type_first;
    int64_t best_time_second;
    char * type_second;
    int64_t iter;
} improve_desc;

uint32_t improve_table_size = 0;

improve_desc improve_table[128] = {0};

//
// Declare iteration values.
//
// N.B. These counts should be such that the respecive test runs for about 10ms.
//
// N.B. Set iter_vec_mad_f16 to 1 and iter_repeat to 1 in order to verify the calculation.
//      Most other values will result in exponent saturation.
//

const uint32_t iter_bf16_to_fp32 = 10000; 
const uint32_t iter_fp32_to_bf16 = 10000; 
const uint32_t iter_fp16_to_fp32 = 10000; 
const uint32_t iter_fp32_to_fp16 = 10000; 
const uint32_t iter_vec_abs_f32 = 5000;
const uint32_t iter_vec_add_f32 = 5000;
const uint32_t iter_vec_add1_f32 = 20000;
const uint32_t iter_vec_acc_f32 = 20000;
const uint32_t iter_vec_acc1_f32 = 20000;
const uint32_t iter_vec_sub_f32 = 5000;
const uint32_t iter_vec_set_f32 = 20000;
const uint32_t iter_vec_cpy_f32 = 20000;
const uint32_t iter_vec_neg_f32 = 20000;
const uint32_t iter_vec_mul_f32 = 10000;
const uint32_t iter_vec_mul1_f32 = 10000;
const uint32_t iter_vec_div_f32 = 10000;
const uint32_t iter_vec_sum_f32 = 20000;
const uint32_t iter_vec_sumsq_f32 = 20000;
const uint32_t iter_vec_sumsq_bf16 = 20000;
const uint32_t iter_vec_max_f32 = 20000;
const uint32_t iter_vec_scale_f16 = 10000;
const uint32_t iter_vec_scale_f32 = 20000;
const uint32_t iter_vec_sqrt_f32 = 20000;
//const uint32_t iter_vec_mad_f16 = 1;
const uint32_t iter_vec_mad_f16 = 10000;
const uint32_t iter_vec_mad_f32 = 20000;
const uint32_t iter_vec_normsq_f32 = 20000;
const uint32_t iter_vec_silu_f32 = 20000;
const uint32_t iter_vec_soft_max_f32 = 20000;
const uint32_t iter_vec_dot_bf16 = 15000;
const uint32_t iter_vec_dot_f16 = 15000;
const uint32_t iter_vec_dot_f32 = 15000;
const uint32_t iter_vec_dot_bf16_f32 = 15000;
const uint32_t iter_vec_dot_f16_f32 = 15000;
const uint32_t iter_vec_dot_q2_K_q8_K = 15000;
const uint32_t iter_vec_dot_q3_K_q8_K = 15000;
//const uint32_t iter_vec_dot_q3_K_q8_K = 1;
const uint32_t iter_vec_dot_q4_K_q8_K = 15000;
//const uint32_t iter_vec_dot_q4_K_q8_K = 1;
const uint32_t iter_vec_dot_q6_K_q8_K = 15000;
const uint32_t iter_vec_dot_q4_0_q8_0 = 15000;
const uint32_t iter_vec_dot_q8_0_q8_0 = 15000;
const uint32_t iter_vec_dot_q8_0_q8_0_repack = 15000;
const uint32_t iter_dequantize_q2_k = 5000;
const uint32_t iter_dequantize_q3_k = 5000;
const uint32_t iter_dequantize_q4_k = 5000;
const uint32_t iter_dequantize_q4_0 = 5000;
const uint32_t iter_dequantize_q6_k = 5000;
const uint32_t iter_dequantize_q8_k = 5000;
const uint32_t iter_dequantize_q8_0 = 5000;
const uint32_t iter_quantize_q8_k = 1250;
const uint32_t iter_quantize_q8_0 = 1250;
const uint32_t iter_quantize_q8_k_dot_q4_k_q8_k = 10000;
const uint32_t iter_quantize_q8_0_dot_q4_0_q8_0 = 10000;
const uint32_t iter_vec_similarity_f32 = 4000;
const uint32_t iter_vec_similarity_bf16 = 4000;

//
// Declare repeat iteration count. This the number of times a test will be repeated to get the
// best answer.
//

const uint32_t iter_repeat = 10;
//const uint32_t iter_repeat = 1;

//
// Local pseudo random number generator.
//

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

//
// Allocate and free page aligned memory.
//

inline
void *
zalloc (
    size_t size
    )

{

    return VirtualAlloc(NULL,
                        size,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE); 
}

inline
void
zfree (
    void * base
    )
{

    VirtualFree(base, 0, MEM_RELEASE);
    return;
}

inline
float
convert_bf16_to_fp32 (
    ggml_bf16_t h
    )
{
    union {
        float f;
        uint32_t i;
    } u;

    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

inline
ggml_bf16_t
convert_fp32_to_bf16 (
    float x
    )
{
    ggml_bf16_t h;

#if 0
    union {
        float x;
        uint32_t i;
    } u;

    u.x = x;
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
#endif // #if 0

    h.bits = (uint16_t)_mm_extract_epi16(_mm_cvtneps_pbh(_mm_set_ss(x)), 0);
    return h;
}

inline
ggml_fp16_t
convert_fp32_to_fp16 (
    float x
    )

{
    return (ggml_fp16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0); 
}

#define GGML_FP32_TO_FP16(x) convert_fp32_to_fp16(x)

inline
float
convert_fp16_to_fp32 (
    ggml_fp16_t x
    )

{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));  
}

#define GGML_FP16_TO_FP32(x) convert_fp16_to_fp32(x)

void
log_increase_x (
    char * text,
    uint64_t best_time_first,
    char * type_first,
    uint64_t best_time_second,
    char * type_second,
    uint64_t iter
    )
{

    if (improve_table_size < ARRAYSIZE(improve_table)) {
        improve_table[improve_table_size].text = text;
        improve_table[improve_table_size].best_time_first = best_time_first;
        improve_table[improve_table_size].type_first = type_first,
        improve_table[improve_table_size].best_time_second = best_time_second;
        improve_table[improve_table_size].type_second = type_second,
        improve_table[improve_table_size].iter = iter;
        improve_table_size += 1;

    } else {
        printf("improvement table overflow\n");
    }

    return;
}

void
log_increase (
    char * text,
    uint64_t best_time_AVX2,
    uint64_t best_time_AVX512,
    uint64_t iter
    )
{

    log_increase_x(text,
                   best_time_AVX2,
                   "AVX2",
                   best_time_AVX512,
                   "AVX512",
                   iter);

    return;
}

void
log_q2_k_quant_data (
    uint32_t vec_size,
    uint32_t * z,
    bool q2_k
    )

{

    if (log_data) {

        if (q2_k) {
            fprintf(logfile, "quantize row q2_k data:\n\n");

        } else {
            fprintf(logfile, "quantize row q2_k_repack data:\n\n");
        }

        const uint32_t count = vec_size / QK_K;
    
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t j;
    
            for (j = 0; j < 4; j += 1) {
                fprintf(logfile, "%08x ", z[j]); 
            }
    
            fprintf(logfile, "\n");
    
            uint32_t l = 0;
            for (; j < 20; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "%04x %04x - ", (uint16_t)z[j], (uint16_t)(z[j] >> 16));
            fprintf(logfile,
                    "(d - %5.2f, dmin - %5.2f)\n\n",
                    convert_fp16_to_fp32((uint16_t)z[j]),
                    convert_fp16_to_fp32((uint16_t)(z[j] >> 16)));
    
            z += (sizeof(block_q2_K) / sizeof(uint32_t));
        }
    }

    return;
}

void
log_q3_k_quant_data (
    uint32_t vec_size,
    uint32_t * z,
    bool q3_k
    )

{

    if (log_data) {

        if (q3_k) {
            fprintf(logfile, "quantize row q3_k data:\n\n");

        } else {
            fprintf(logfile, "quantize row q3_k_repack data:\n\n");
        }
    
        const uint32_t count = vec_size / QK_K;
    
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t j;

            //
            // Log hmask bits - QK_K / 8 = 32 bytes - 8 dwords.
            //

            for (j = 0; j < 8; j += 1) {
                fprintf(logfile, "%08x ", z[j]); 
            }
    
            fprintf(logfile, "\n");

            //
            // log qs bytes - QK_K / 4 = 64 bytes - 16 dwords
            //

            uint32_t l = 0;
            for (; j < 24; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }

            //
            // Log scales bytes - 12 bytes - 3 dwords
            //

            for (; j < 27; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
            }

            fprintf(logfile, "\n");

            //
            // Log d value - ggml_half
            //

            fprintf(logfile, "%04x - ", (uint16_t)z[j]);
            fprintf(logfile, "(d - %5.2f)\n\n",  convert_fp16_to_fp32((uint16_t)z[j]));
    
            z = (void *)((uint8_t *)z + sizeof(block_q3_K));
        }
    }

    return;
}

void
log_q4_k_quant_data (
    uint32_t vec_size,
    uint32_t * z,
    bool q4_k
    )

{

    if (log_data) {

        if (q4_k) {
            fprintf(logfile, "quantize row q4_k data:\n\n");

        } else {
            fprintf(logfile, "quantize row q4_k_repack data:\n\n");
        }
    
        const uint32_t count = vec_size / QK_K;
    
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t j;
    
            fprintf(logfile, "%04x %04x - ", (uint16_t)z[0], (uint16_t)(z[0] >> 16));
            fprintf(logfile,
                    "(d - %5.2f, dmin - %5.2f)\n\n",
                    convert_fp16_to_fp32((uint16_t)z[0]),
                    convert_fp16_to_fp32((uint16_t)(z[0] >> 16)));
    
            for (j = 1; j < 4; j += 1) {
                fprintf(logfile, "%08x ", z[j]); 
            }
    
            fprintf(logfile, "\n\n");
    
            uint32_t l = 0;
            for (; j < 36; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "\n");
    
            z += (sizeof(block_q4_K) / sizeof(uint32_t));
        }
    }

    return;
}

void
log_q4_0_quant_data (
    uint32_t vec_size,
    uint16_t * z
    )

{

    if (log_data) {

        fprintf(logfile, "quantize row q4_0 data:\n\n");
    
        const uint32_t count = vec_size / QK4_0;
    
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t j;
    
            fprintf(logfile, "%04x - ", z[0]);
            fprintf(logfile, "(d - %5.2f)\n", convert_fp16_to_fp32(z[0]));
    
            for (j = 1; j < 9; j += 1) {
                fprintf(logfile, "%04x ", z[j]); 
            }
    
            fprintf(logfile, "\n\n");
    
            z = (uint16_t *)((char *)z + sizeof(block_q4_0));
        }
    }

    return;
}

void
log_q4_0_repack_quant_data (
    uint32_t vec_size,
    uint32_t * z
    )

{

    if (log_data) {

        fprintf(logfile, "quantize row q4_0_repack data:\n\n");

        const uint32_t count = vec_size / QK_K;

        for (uint32_t i = 0; i < count; i += 1) {

            //
            // Log float multipliers.
            //

            uint16_t * d = (uint16_t *)z;
            for (uint32_t j = 0; j < QK_K / QK4_0; j += 1) {
                fprintf(logfile, "%04x ", d[j]);
            }
    
            fprintf(logfile, "\n\n");

            z += (QK_K / QK4_0) / 2;

            //
            // Log rows of quant values.
            //

            for (uint32_t j = 0; j < QK_K / QK4_0; j += 1) {
                for (uint32_t k = 0; k < QK_K / (QK4_0 * 2); k += 1) {
                    fprintf(logfile, "%08x ", z[k + j * (QK_K / (QK4_0 * 2))]);
                }

                fprintf(logfile, "\n");
            }
    
            fprintf(logfile, "\n");
    
            z += QK_K / (2 * sizeof(uint32_t));
        }
    }

    return;
}

void
log_q6_k_quant_data (
    uint32_t vec_size,
    uint32_t * z,
    bool q6_k
    )

{

    if (log_data) {

        if (q6_k) {
            fprintf(logfile, "quantize row q6_k data:\n\n");

        } else {
            fprintf(logfile, "quantize row q6_k_repack data:\n\n");
        }
    
        const uint32_t count = vec_size / QK_K;
    
        for (uint32_t i = 0; i < count; i += 1) {
            uint32_t j;
            uint32_t l;

            //
            // Log the ql 4-bit values.
            //

            l = 0;
            for (j = 0; j < 32; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "\n");

            //
            // Log the qh 2-bit values.
            //

            l = 0;
            for (; j < 48; j += 1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "\n");

            //
            // Log the quant 8-bit scale values.
            //

            for (; j < 52; j +=1 ) {
                fprintf(logfile, "%08x ", z[j]);
            }

            fprintf(logfile, "\n");

            //
            // Log the f16 floating scale value.
            //

            fprintf(logfile, "%04x ", (uint16_t)z[j]);
            fprintf(logfile,
                    "(d - %5.2f)\n",
                    convert_fp16_to_fp32((uint16_t)z[j]));
    
            fprintf(logfile, "\n");
    
            z = (uint32_t *)((char *)z + sizeof(block_q6_K));
        }
    }

    return;
}

void
log_q8_k_quant_data (
    uint32_t vec_size,
    uint32_t * z,
    bool q8_k
    )

{

    if (log_data) {

        if (q8_k) {
            fprintf(logfile, "quantize row q8_k data:\n\n");

        } else {
            fprintf(logfile, "quantize row q8_k_repack data:\n\n");
        }
    
        const uint32_t count = vec_size / QK_K;
    
        for (uint32_t i = 0; i < count; i += 1) {
    
            fprintf(logfile, "%08x ", z[0]);
            fprintf(logfile, "(d - %7.2f)\n\n", *((float  *)(&z[0])));  
    
            uint32_t j;
            uint32_t l = 0;
    
            for (j = 1; j < 65; j +=1) {
                fprintf(logfile, "%08x ", z[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }

            fprintf(logfile, "\n");

            l = 0;
            for (; j < 73; j += 1) {
                fprintf(logfile, "%04x %04x ", (uint16_t)z[j], (uint16_t)(z[j] >> 16));
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "\n");
    
            z += (sizeof(block_q8_K) / sizeof(uint32_t)); 
        }
    }

    return;
}

void
log_q8_0_quant_data (
    uint32_t vec_size,
    uint16_t * z
    )

{

    if (log_data) {

        fprintf(logfile, "quantize row q8_0 data:\n\n");
    
        const uint32_t count = vec_size / QK8_0;
    
        for (uint32_t i = 0; i < count; i += 1) {
            fprintf(logfile, "%04x\n", z[0]);
    
            uint32_t j;
            uint32_t l = 0;
            uint32_t * zd = (void *)(z + 1);
    
            for (j = 0; j < QK8_0 / 4; j += 1) {
                fprintf(logfile, "%08x ", zd[j]);
                l += 1;
                if (!(l & 7)) {
                    fprintf(logfile, "\n");
                }
            }
    
            fprintf(logfile, "\n");
    
            z += (sizeof(block_q8_0) / sizeof(uint16_t));
        }
    }

    return;
}

void
log_q8_0_repack_quant_data (
    uint32_t vec_size,
    uint32_t * z
    )

{

    if (log_data) {

        fprintf(logfile, "quantize row q8_0_repack data:\n\n");

        const uint32_t count = vec_size / QK_K;

        for (uint32_t i = 0; i < count; i += 1) {

            //
            // Log float multipliers.
            //

            uint16_t * d = (uint16_t *)z;
            for (uint32_t j = 0; j < QK_K / QK8_0; j += 1) {
                fprintf(logfile, "%04x ", d[j]);
            }
    
            fprintf(logfile, "\n\n");
            z += (QK_K / QK8_0) / 2;

            //
            // Log rows of quant values.
            //

            for (uint32_t j = 0; j < QK_K / QK8_0; j += 1) {
                for (uint32_t k = 0; k < QK_K / QK8_0; k += 1) {
                    fprintf(logfile, "%08x ", z[k + j * (QK_K / QK8_0)]);
                }

                fprintf(logfile, "\n");
            }
    
            fprintf(logfile, "\n");
    
            z += QK_K / sizeof(uint32_t);
        }
    }

    return;
}

void
log_float_data (
    uint32_t count,
    uint32_t * z,
    char * msg
    )

{

    if (log_data) {

        fprintf(logfile, "%s\n\n", msg);
    
        for (uint32_t i = 0; i < count; i += 1) {

            //
            // Convert minus zero to zero - arithmetically they are equivalent.
            //

            uint32_t v = (z[i] == 0x80000000) ? 0 : z[i];
            
            fprintf(logfile, "%08x ", v);
            if (!((i + 1) & 7)) {
                fprintf(logfile, "\n");
            }
        }
     
        fprintf(logfile, "\n");
    }

    if (count % 8) {
        fprintf(logfile, "\n");
    }

    return;
}

void
log_raw_data (
    uint32_t count,
    uint32_t * z,
    char * msg
    )

{

    if (log_data) {

        fprintf(logfile, "%s\n\n", msg);
    
        for (uint32_t i = 0; i < count; i += 1) {
            fprintf(logfile, "%08x ", z[i]);
            if (!((i + 1) & 7)) {
                fprintf(logfile, "\n");
            }
        }
     
        fprintf(logfile, "\n");
    }

    if (count % 8) {
        fprintf(logfile, "\n");
    }

    return;
}

void
vec_test_quantize (
    uint32_t vec_size
    )

//
// Test the quantization/ dequantization for:
//
//  quantize_row_q2_k/dequantize_row_q2_k
//  quantize_row_q3_k/dequantize_row_q3_k
//  quantize_row_q4_k/dequantization_q4_k
//  quantize_row_q4_0/dequantization_q4_0
//  quantize_row_q6_k/dequantization_q6_k
//  quantize_row_q8_k/dequantization_q8_k
//  quantize_row_q8_0/dequantization_q8_0
//

{

    if (log_data) {
    
        uint32_t i;
        float * x;
        float * y;
        block_q2_K * q2x;
        block_q3_K * q3x;
        block_q4_K * q4x;
        block_q4_0 * q4_0;
        block_q6_K * q6x;
        block_q8_K * q8x;
        block_q8_0 * q8x_0;
    
        //
        // Announce perf test.
        //
    
        fprintf(logfile, "Running quantization/dequantization tests for AVX2/AVX512.\n\n");
    
        //
        // Allocate vectors of the required size.
        //
    
        x = zalloc(vec_size * sizeof(float));
        y = zalloc(vec_size * sizeof(float));
        q2x = zalloc(vec_size / QK_K * sizeof(block_q2_K));
        q3x = zalloc(vec_size / QK_K * sizeof(block_q3_K));
        q4x = zalloc(vec_size / QK_K * sizeof(block_q4_K));
        q4_0 = zalloc(vec_size / QK4_0 * sizeof(block_q4_0));
        q6x = zalloc(vec_size / QK_K * sizeof(block_q6_K));
        q8x = zalloc(vec_size / QK_K * sizeof(block_q8_K));
        q8x_0 = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
        if (!x || !y || !q2x || !q3x || !q4x || !q4_0 || !q6x || !q8x || !q8x_0) {
            fprintf(logfile, "  failed to allocate vectors \n");
            goto exit;
        }
    
        //
        // Fill the vector x with random filtered values that are positive and negative.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER) * FLOAT_FRACTION_X;
        }
    
        //
        // Log float data.
        //
    
        log_float_data(vec_size, (void *)x, "x vector input data");
    
        //
        // Run quantize_row_qx_k AVX2 tests.
        //
    
        fprintf(logfile, "Running quantize_q2_k/dequantize_q2_k) AVX2\n\n");
    
        pfn_quantize_row_q2_K_AVX2(x, q2x, vec_size);
        log_q2_k_quant_data(vec_size, (void *)q2x, false);
    
        pfn_dequantize_row_q2_K_AVX2(q2x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q2_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q3_k/dequantize_q3_k) AVX2\n\n");
    
        pfn_quantize_row_q3_K_AVX2(x, q3x, vec_size);
        log_q3_k_quant_data(vec_size, (void *)q3x, true);
    
        pfn_dequantize_row_q3_K_AVX2(q3x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q3_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q4_k/dequantize_q4_k) AVX2\n\n");
    
        pfn_quantize_row_q4_K_AVX2(x, q4x, vec_size);
        log_q4_k_quant_data(vec_size, (void *)q4x, true);
    
        pfn_dequantize_row_q4_K_AVX2(q4x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q4_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q4_0/dequantize_q4_0) AVX2\n\n");
    
        pfn_quantize_row_q4_0_AVX2(x, q4_0, vec_size);
        log_q4_0_quant_data(vec_size, (void *)q4_0);
    
        pfn_dequantize_row_q4_0_AVX2(q4_0, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q4_0 y vector output data:");
    
        fprintf(logfile, "Running quantize_q6_k/dequantize_q6_k) AVX2\n\n");
    
        pfn_quantize_row_q6_K_AVX2(x, q6x, vec_size);
        log_q6_k_quant_data(vec_size, (void *)q6x, true);
    
        pfn_dequantize_row_q6_K_AVX2(q6x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q6_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q8_k/dequantize_q8_k) AVX2\n\n");
    
        pfn_quantize_row_q8_K_AVX2(x, q8x, vec_size);
        log_q8_k_quant_data(vec_size, (void *)q8x, true);
    
        pfn_dequantize_row_q8_K_AVX2(q8x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q8_k y output data");
    
        fprintf(logfile, "Running quantize_q8_0/dequantize_q8_0) AVX2\n\n");
    
        pfn_quantize_row_q8_0_AVX2(x, q8x_0, vec_size);
        log_q8_0_quant_data(vec_size, (void *)q8x_0);
    
        pfn_dequantize_row_q8_0_AVX2(q8x_0, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q8_0 y output data");
    
        //
        // Run quantize_row_qx_k AVX512 tests.
        //
    
        fprintf(logfile, "Running quantize_q2_k/dequantize_q2_k) AVX512\n\n");
    
        pfn_quantize_row_q2_K_AVX512(x, q2x, vec_size);
        log_q2_k_quant_data(vec_size, (void *)q2x, false);
    
        pfn_dequantize_row_q2_K_AVX512(q2x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q2_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q3_k/dequantize_q3_k) AVX512\n\n");
    
        pfn_quantize_row_q3_K_AVX512(x, q3x, vec_size);
        log_q3_k_quant_data(vec_size, (void *)q3x, true);
    
        pfn_dequantize_row_q3_K_AVX512(q3x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q3_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q4_k/dequantize_q4_k) AVX512\n\n");
    
        pfn_quantize_row_q4_K_AVX512(x, q4x, vec_size);
        log_q4_k_quant_data(vec_size, (void *)q4x, true);
    
        pfn_dequantize_row_q4_K_AVX512(q4x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q4_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q4_0/dequantize_q4_0) AVX512\n\n");
    
        pfn_quantize_row_q4_0_AVX512(x, q4_0, vec_size);
        log_q4_0_quant_data(vec_size, (void *)q4_0);
    
        pfn_dequantize_row_q4_0_AVX512(q4_0, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q4_0 y vector output data:");
    
        fprintf(logfile, "Running quantize_q6_k/dequantize_q6_k) AVX512\n\n");
    
        pfn_quantize_row_q6_K_AVX512(x, q6x, vec_size);
        log_q6_k_quant_data(vec_size, (void *)q6x, true);
    
        pfn_dequantize_row_q6_K_AVX512(q6x, y, vec_size);
        log_float_data(vec_size, (void *)y, "dequantize q6_k y vector output data:");
    
        fprintf(logfile, "Running quantize_q8_k/dequantize_q8_k) AVX512\n\n");
    
        pfn_quantize_row_q8_K_AVX512(x, q8x, vec_size);
        log_q8_k_quant_data(vec_size, (void *)q8x, true);
    
        pfn_dequantize_row_q8_K_AVX512(q8x, y, vec_size);
        fprintf(logfile, "dequantize row q8_k data\n\n");
        log_float_data(vec_size, (void *)y, "dequantize q8_k y output data");
    
        fprintf(logfile, "Running quantize_q8_0/dequantize_q8_0) AVX512\n\n");
    
        pfn_quantize_row_q8_0_AVX512(x, q8x_0, vec_size);
        log_q8_0_quant_data(vec_size, (void *)q8x_0);
    
        pfn_dequantize_row_q8_0_AVX512(q8x_0, y, vec_size);
        fprintf(logfile, "dequantize row q8_0 data\n\n");
        log_float_data(vec_size, (void *)y, "dequantize q8_0 y output data");
    
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
    
        if (q3x) {
            zfree(q3x);
        }
    
        if (q4x) {
            zfree(q4x);
        }

        if (q4_0) {
            zfree(q4_0);
        }

        if (q6x) {
            zfree(q6x);
        }
    
        if (q8x) {
            zfree(q8x);
        }
    
        if (q8x_0) {
            zfree(q8x_0);
        }
    }

    return;
}

void
vec_bf16_to_fp32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector bf16 converted to float.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_bf16_t * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_bf16_to_fp32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the required size.
    //

    x = zalloc(vec_size * sizeof(ggml_bf16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors\n");
        goto exit;
    }

    //
    // Fill the vector x with random filtered ggml_bf16_t values.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_bf16(genx_float_value(FLOAT_FILTER));
    }

    //
    // log raw (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (bf16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_bf16_to_fp32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_bf16_to_fp32; i += 1) {
            pfn_ggml_bf16_to_fp32_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // log float data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_bf16_to_fp32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_bf16_to_fp32; i += 1) {
            pfn_ggml_bf16_to_fp32_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // log float data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_bf16_to_fp32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_bf16_to_fp32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_fp32_to_bf16 (
    uint32_t vec_size
    )

//
// Compute the performance of vector fp32 converted to bf16.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    ggml_bf16_t * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_bf16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the required size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(ggml_bf16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the vector x with random filtered float values.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // log float data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_bf16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_bf16; i += 1) {
            pfn_ggml_fp32_to_bf16_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // log raw (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (bf16) vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_bf16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_bf16; i += 1) {
            pfn_ggml_fp32_to_bf16_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // log raw (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (bf16) vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_fp32_to_bf16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_fp32_to_bf16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_fp16_to_fp32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector fp16 converted to float.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the required size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the vector x with random filtered ggml_fp16_t values.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_fp16(genx_float_value(FLOAT_FILTER));
    }

    //
    // log raw (fp16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (fp16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp16_to_fp32; i += 1) {
            pfn_ggml_fp16_to_fp32_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // log float data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp16_to_fp32; i += 1) {
            pfn_ggml_fp16_to_fp32_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // log float data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_fp16_to_fp32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_fp16_to_fp32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_fp32_to_fp16 (
    uint32_t vec_size
    )

//
// Compute the performance of vector fp32 converted to fp16.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    ggml_fp16_t * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the required size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the vector x with random filtered float values.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // log float data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_fp16; i += 1) {
            pfn_ggml_fp32_to_fp16_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // log raw (f16) data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_fp16; i += 1) {
            pfn_ggml_fp32_to_fp16_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // log raw (fp16) data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_fp32_to_fp16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_fp32_to_fp16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_abs_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector sqrt f32.
//
//  y[i] = fabsf(x[i])
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_abs_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_abs_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector abs.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_abs_f32; i += 1) {
            pfn_ggml_vec_abs_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_abs_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector sqrt.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_abs_f32; i += 1) {
            pfn_ggml_vec_abs_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_abs_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_abs_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_add_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector add f32.
//
//  z[i] = x[i] + y[i]
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    // 

    log_increase("vec_add_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_add_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_add1_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector add1 f32.
//
//  z[i] = x[i] + v
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float v;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add1_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v and the x vector with random filtered values converted to float.
    //

    v = genx_float_value(FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated v and x data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add1_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add1_f32; i += 1) {
            pfn_ggml_vec_add1_f32_AVX2(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add1_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add1_f32; i += 1) {
            pfn_ggml_vec_add1_f32_AVX512(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_add1_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_add1_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_acc_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector acc f32.
//
//  y[i] += x[i]
//
// N.B. This test repeatedly adds x[i] to y[i] which causes a modified y vector to be used on
//      the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc_f32; i += 1) {
            pfn_ggml_vec_acc_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc_f32; i += 1) {
            pfn_ggml_vec_acc_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_acc_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_acc_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_acc1_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector add1 f32.
//
//  y[i] += v
//
// N.B. This test repeatedly adds v to y[i] which causes a modified y vector to be used on
//      the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can still be compared correctly.
//

{
    uint32_t i;
    uint32_t j;
    float v;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc1_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    y = zalloc(vec_size * sizeof(float));
    if (!y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v and the y vector with random filtered values converted to float.
    //

    v = genx_float_value(FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        y[i] = geny_float_value(FLOAT_FILTER);
    }

    //
    // Log generated v and y data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc1_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc1_f32; i += 1) {
            pfn_ggml_vec_acc1_f32_AVX2(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc1_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector add.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc1_f32; i += 1) {
            pfn_ggml_vec_acc1_f32_AVX512(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_acc1_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_acc1_f32);

exit:
    if (y) {
        zfree(y);
    }

    return;
}

void
vec_sub_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector add f32.
//
//  z[i] = x[i] - y[i]
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sub_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sub_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector sub.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sub_f32; i += 1) {
            pfn_ggml_vec_sub_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sub_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector sub.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sub_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_sub_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_sub_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_set_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector set f32.
//
//  x[i] = v
//

{

    uint32_t i;
    uint32_t j;
    float v;
    float * x;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_set_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v with a random filtered values converted to float.
    //

    v = genx_float_value(FLOAT_FILTER);

    //
    // Log generated v data.
    //

    log_float_data(1, (void *)&v, "generated v value:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_set_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector set.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_set_f32; i += 1) {
            pfn_ggml_vec_set_f32_AVX2(vec_size, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)x, "x vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_set_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector set.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_set_f32; i += 1) {
            pfn_ggml_vec_set_f32_AVX512(vec_size, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)x, "x vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_set_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_set_f32);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_cpy_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector copy f32.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cpy_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cpy_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector cpy.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_cpy_f32; i += 1) {
            pfn_ggml_vec_cpy_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cpy_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector cpy.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_cpy_f32; i += 1) {
            pfn_ggml_vec_cpy_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_cpy_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_cpy_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_neg_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector negate f32.
//
//  y[i] = -x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_neg_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_neg_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector negate.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_neg_f32; i += 1) {
            pfn_ggml_vec_neg_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_neg_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector negate.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_neg_f32; i += 1) {
            pfn_ggml_vec_neg_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_neg_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_neg_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mul_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector multiply f32.
//
//  z[i] = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul_f32; i += 1) {
            pfn_ggml_vec_mul_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul_f32; i += 1) {
            pfn_ggml_vec_mul_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_mul_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_mul_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_mul1_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector multiply f32.
//
//  z[i] = x[i] * v
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float v;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul1_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v and the x vector with random filtered values converted to float.
    //

    v = genx_float_value(FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated generated v and x data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul1_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul1_f32; i += 1) {
            pfn_ggml_vec_mul1_f32_AVX2(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul1_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul1_f32; i += 1) {
            pfn_ggml_vec_mul1_f32_AVX512(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_mul1_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_mul1_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_div_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector divide f32.
//
//  z[i] = x[i] / y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_div_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_div_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector divides.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_div_f32; i += 1) {
            pfn_ggml_vec_div_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_div_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector divides.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_div_f32; i += 1) {
            pfn_ggml_vec_div_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_div_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_div_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_sum_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of vector elements.
//
//  sum += x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sum_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sum_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sum_f32; i += 1) {
            pfn_ggml_vec_sum_f32_AVX2(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum.
    //

    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sum_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sum_f32; i += 1) {
            pfn_ggml_vec_sum_f32_AVX512(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log vector sum.
    //

    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_sum_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_sum_f32);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_sumsq_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the square of vector elements.
//
//  sum += x[i] * x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float sum = 0.f; 

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }


    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the square of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_f32; i += 1) {
            pfn_ggml_vec_sumsq_f32_AVX2(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of squares.
    //

    fprintf(logfile, "  vector sum of squares %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the square of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_f32; i += 1) {
            pfn_ggml_vec_sumsq_f32_AVX512(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of squares.
    //

    fprintf(logfile, "  vector sum of squares %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_sumsq_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_sumsq_f32);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_sumsq_bf16 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the of the square of vector elements.
//
//  sum += x[i] * x[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_bf16_t * x;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_bf16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate bf16 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_bf16_t));
    if (!x) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to bf16.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_bf16(genx_float_value(FLOAT_FILTER));
    }

    //
    // Log generated x (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (bf16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_bf16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the square of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_bf16; i += 1) {
            pfn_ggml_vec_sumsq_bf16_AVX2(vec_size, &sum, x); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of squares.
    //

    fprintf(logfile, "  vector sum of squares %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_bf16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the square of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_bf16; i += 1) {
            pfn_ggml_vec_sumsq_bf16_AVX512(vec_size, &sum, x); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of squares.
    //

    fprintf(logfile, "  vector sum of squares %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_sumsq_bf16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_sumsq_bf16);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_max_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of maximum value of vector elements.
//
//  fmax = max(x[i], fmax)
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float max = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_max_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_max_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the maximization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_max_f32; i += 1) {
            pfn_ggml_vec_max_f32_AVX2(vec_size, &max, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector max.
    //

    fprintf(logfile, "  vector max %08x\n\n", *(uint32_t *)&max);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_max_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the maximization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_max_f32; i += 1) {
            pfn_ggml_vec_max_f32_AVX512(vec_size, &max, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector max.
    //

    fprintf(logfile, "  vector max %08x\n\n", *(uint32_t *)&max);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_max_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_max_f32);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_scale_f16 (
    uint32_t vec_size
    )

//
// Compute the performance of the product of a vector elements and a scale value.
//
//  y[i] *= v
//
// N.B. This test repeatedly adds y[i] * v to y[i] which causes a modified y vector to be used
//      on the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vector of the specified size.
    //

    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v and the y vector with random filtered values converted to float.
    //
    // N.B. the value of v is set to one to avoid overlowing/saturating individual values
    //      of y[i].
    //

    v = 1.0f;
    for (i = 0; i < vec_size; i += 1) {
        y[i] = convert_fp32_to_fp16(geny_float_value(FLOAT_FILTER));
    }

    //
    // Log generated v and y (f16) data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_raw_data(vec_size / 2, (void *)y, "generated y (f16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and a scale factor.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f16; i += 1) {
            pfn_ggml_vec_scale_f16_AVX2(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log raw (f16) output data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f16 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and a scale factor.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f16; i += 1) {
            pfn_ggml_vec_scale_f16_AVX512(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log raw (f16) output data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_scale_f16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_scale_f16);

exit:
    if (y) {
        zfree(y);
    }

    return;
}

void
vec_scale_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of the product of a vector elements and a scale value.
//
//  y[i] *= v
//
// N.B. This test repeatedly adds y[i] * v to y[i] which causes a modified y vector to be used
//      on the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{

    uint32_t i;
    uint32_t j;
    float * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vector of the specified size.
    //

    y = zalloc(vec_size * sizeof(float));
    if (!y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v and the y vector with random filtered values converted to float.
    //
    // N.B. the value of v is set to one to avoid overlowing/saturating individual values
    //      of y[i].
    //

    v = 1.0f;
    for (i = 0; i < vec_size; i += 1) {
        y[i] = geny_float_value(FLOAT_FILTER);
    }

    //
    // Log generated v and y data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and a scale factor.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f32; i += 1) {
            pfn_ggml_vec_scale_f32_AVX2(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and a scale factor.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f32; i += 1) {
            pfn_ggml_vec_scale_f32_AVX512(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_scale_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_scale_f32);

exit:
    if (y) {
        zfree(y);
    }

    return;
}

void
vec_sqrt_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of vector sqrt f32.
//
//  y[i] = sqrtf(x[i])
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sqrt_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sqrt_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector sqrt.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sqrt_f32; i += 1) {
            pfn_ggml_vec_sqrt_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //
 
    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sqrt_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the vector sqrt.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sqrt_f32; i += 1) {
            pfn_ggml_vec_sqrt_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_sqrt_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_sqrt_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mad_f16 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of vector elements with v.
//
//  y[i] += x[i]*v
//
// N.B. This test repeatedly adds y[i] + x[i] * v to y[i] which causes a modified y
//      vector to be used on the next iteration where the y vector is again used as
//      input. Although the value of the y vector is changing, the resultant number
//      across za and zo ggml implementations can still be compared correctly.
//
                                             
{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    ggml_fp16_t * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v, x and the y vector with random filtered values converted to float.
    //
    // N.B. the value of v is set to one to avoid overlowing/saturating individual values
    //      of y[i].
    //

    v = 1.0;
    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_fp16(genx_float_value(FLOAT_FILTER));
        y[i] = convert_fp32_to_fp16(geny_float_value(FLOAT_FILTER));
    }

    //
    // Log generated v, x (f16), and y (f16) data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_raw_data(vec_size / 2, (void *)x, "generated x (f16) vector:");
    log_raw_data(vec_size / 2, (void *)y, "generated y (f16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and v.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f16; i += 1) {
            pfn_ggml_vec_mad_f16_AVX2(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log raw (f16) output data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f16 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the product of vector elements and v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f16; i += 1) {
            pfn_ggml_vec_mad_f16_AVX512(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log raw (f16) output data.
    //

    log_raw_data(vec_size / 2, (void *)y, "y (f16) vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_mad_f16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_mad_f16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mad_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of vector elements with v.
//
//  y[i] += x[i]*v
//
// N.B. This test repeatedly adds y[i] + x[i] * v to y[i] which causes a modified y
//      vector to be used on the next iteration where the y vector is again used as
//      input. Although the value of the y vector is changing, the resultant number
//      across za and zo ggml implementations can still be compared correctly.
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v, x and the y vector with random filtered values converted to float.
    //

    v = genx_float_value(FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER);
    }

    //
    // Log generated v, x, and y data.
    //

    log_float_data(1, (void *)&v, "generated v value:");
    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements
        // with v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f32; i += 1) {
            pfn_ggml_vec_mad_f32_AVX2(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements
        // with v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f32; i += 1) {
            pfn_ggml_vec_mad_f32_AVX512(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_mad_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_mad_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_normsq_f32 (
    uint32_t vec_size
    )

//
// Compute the summation of the product of vector elements with mean.
//
//  bx = x[i] - mean
//  y[i] = bx
//  sum += bx * bx
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float mean;
    float sum = 0.f;

    //
    // Announce functional test.
    //

    fprintf(logfile, "Running vec_normsq_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill in x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
    }

    //
    // Compute the mean of the x values.
    //

    pfn_ggml_vec_sum_f32_AVX2(vec_size, &sum, x);
    mean = sum / (float)vec_size;

    //
    // Log generated mean and x data.
    //

    log_float_data(1, (void *)&sum, "generated x vector mean:");
    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_normsq_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements
        //
        //  bx = x[i] - mean
        //  y[i] = bx
        //  sum += bx * bx
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_normsq_f32; i += 1) {
            pfn_ggml_vec_normsq_f32_AVX2(vec_size, &sum, mean, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Log sum of square of x - mean.
    //

    fprintf(logfile, "  vector sum of square of x - mean %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_normsq_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements
        //
        //  bx = x[i] - mean
        //  y[i] = bx
        //  sum += bx * bx
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_normsq_f32; i += 1) {
            pfn_ggml_vec_normsq_f32_AVX512(vec_size, &sum, mean, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Log sum of square of x - mean.
    //

    fprintf(logfile, "  vector sum of square of x - mean %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_normsq_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_normsq_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_silu_f32 (
    uint32_t vec_size
    )

//
// Compute the silu of a vector of elements.
//
//  y[i] = x[i] / (1 + exp(-x[i]))
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce test.
    //

    fprintf(logfile, "Running vec_silu_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill in the vector x with random filtered values that are positive and negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_SILU) * FLOAT_FRACTION_X;
    }

    //
    // Log float data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_silu_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the silu of the x vector.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_silu_f32; i += 1) {
            pfn_ggml_vec_silu_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_silu_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the silu of the x vector.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_silu_f32; i += 1) {
            pfn_ggml_vec_silu_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_silu_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_silu_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_soft_max_f32 (
    uint32_t vec_size
    )

//
// Compute the soft max of a vector of elements.
//
//  val = expf(x[i] - max)
//  y[i] = val
//  sum += val
//

{

    uint32_t i;
    uint32_t j;
    float max;
    float sum = 0.f;
    float * x;
    float * y;

    //
    // Announce test.
    //

    fprintf(logfile, "Running vec_soft_max_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill in the vector x with random filtered values that are positive and negative..
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_SOFT_MAX) * FLOAT_FRACTION_X;
    }

    //
    // Log float data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_soft_max_f32 performance test for AVX2\n\n");

    //
    // Compute AVX2 x vector maximum value.
    //

    pfn_ggml_vec_max_f32_AVX2(vec_size, &max, x);

    //
    // Log AVX2 x vector maximum value.
    //

    fprintf(logfile, "x vector maximum %08x\n\n", *(uint32_t *)&max);

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the soft max of the x vector.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_soft_max_f32; i += 1) {
            sum = (float)pfn_ggml_vec_soft_max_f32_AVX2(vec_size, y, x, max);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Log sum of soft max sum.
    //

    fprintf(logfile, "y vector sum %08x / %8.4f\n\n",
            *(uint32_t *)&sum,
            sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_soft_max_f32 performance test for AVX512\n\n");

    //
    // Compute AVX512 x vector maximum value.
    //

    pfn_ggml_vec_max_f32_AVX512(vec_size, &max, x);

    //
    // Log AVX512 x vector maximum value.
    //

    fprintf(logfile, "x vector maximum %08x\n\n", *(uint32_t *)&max);

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the soft max of the x vector.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_soft_max_f32; i += 1) {
            sum = (float)pfn_ggml_vec_soft_max_f32_AVX512(vec_size, y, x, max);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)y, "y vector output data:");

    //
    // Log sum of soft max sum.
    //

    fprintf(logfile, "y vector sum %08x / %8.4f\n\n",
            *(uint32_t *)&sum,
            sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_soft_max_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_soft_max_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
dequantize_q2_k (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q2 by generating a quantized set of q2 blocks,
// then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q2_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q2_k performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q2_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q2_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q2);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q2_K_AVX2(x, y, vec_size);

    //
    // log q2 quant data.
    //

    log_q2_k_quant_data(vec_size, (void *)y, false);

    //
    // dequantize generated q2 quant blocks.
    //

    pfn_dequantize_row_q2_K_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q2 float z data:");

    pfn_dequantize_row_q2_K_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q2 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q2_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q2_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q2_k; i += 1) {
            pfn_dequantize_row_q2_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q2 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q2_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q2_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q2_k; i += 1) {
            pfn_dequantize_row_q2_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q2 float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q2_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q2_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q3_k (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q3 by generating a quantized set of q3 blocks,
// then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q3_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q3_k performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q3_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q3_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q3);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q3_K_AVX2(x, y, vec_size);

    //
    // log q3 quant data.
    //

    log_q3_k_quant_data(vec_size, (void *)y, true);

    //
    // dequantize generated q3 quant blocks.
    //

    pfn_dequantize_row_q3_K_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q3 float z data:");

    pfn_dequantize_row_q3_K_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q3 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q3_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q3_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q3_k; i += 1) {
            pfn_dequantize_row_q3_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q3 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q3_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q3_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q3_k; i += 1) {
            pfn_dequantize_row_q3_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q3 float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q3_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q3_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q4_k (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q4_k by generating a quantized set of
// q4_k blocks, then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q4_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_k performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q4_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q4_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q4);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_K_AVX2(x, y, vec_size);

    //
    // log q4_k quant data.
    //

    log_q4_k_quant_data(vec_size, (void *)y, true);

    //
    // dequantize generated q4_k quant blocks.
    //

    pfn_dequantize_row_q4_K_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q4_k float z data:");

    pfn_dequantize_row_q4_K_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q4_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q4_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_k; i += 1) {
            pfn_dequantize_row_q4_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q4_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q4_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_k; i += 1) {
            pfn_dequantize_row_q4_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q4_k float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q4_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q4_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q4_0 (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q4_0 by generating a quantized set of
// q4_0 blocks, then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q4_0 * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_0 performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q4_0 vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK4_0) * sizeof(block_q4_0));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q4);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_0_AVX2(x, y, vec_size);

    //
    // log q4_0 quant data.
    //

    log_q4_0_quant_data(vec_size, (void *)y);

    //
    // dequantize generated q4_0 quant blocks.
    //

    pfn_dequantize_row_q4_0_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q4_0 float z data:");

    pfn_dequantize_row_q4_0_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q4_0 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_0 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q4_0 dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_0; i += 1) {
            pfn_dequantize_row_q4_0_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q4_0 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q4_0 dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_0; i += 1) {
            pfn_dequantize_row_q4_0_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q4_0 float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q4_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q4_0);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q6_k (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q6 by generating a quantized set of q6
// blocks, then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q6_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q6_k performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q6_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q6_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q6);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q6_K_AVX2(x, y, vec_size);

    //
    // log q6 quant data.
    //

    log_q6_k_quant_data(vec_size, (void *)y, true);

    //
    // dequantize generated q6 quant blocks.
    //

    pfn_dequantize_row_q6_K_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q6_k float z data:");

    pfn_dequantize_row_q6_K_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q6_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q6_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q6_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q6_k; i += 1) {
            pfn_dequantize_row_q6_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float data output.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q6_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q6_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do q6_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q6_k; i += 1) {
            pfn_dequantize_row_q6_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float data output.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q6_k float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q6_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q6_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q8_k (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q8_k by generating a quantized set of q8_k blocks,
// then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q8_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_k performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q8_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q8_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q8);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats for AVX2.
    //

    pfn_quantize_row_q8_K_AVX2(x, y, vec_size);

    //
    // log q8_k quant data.
    //

    log_q8_k_quant_data(vec_size, (void *)y, true);

    //
    // quantize the x vector of floats for AVX512.
    //

    pfn_quantize_row_q8_K_AVX512(x, y, vec_size);

    //
    // log q8_k quant data.
    //

    log_q8_k_quant_data(vec_size, (void *)y, true);

    //
    // dequantize generated q8_k quant blocks.
    //

    pfn_dequantize_row_q8_K_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q8_k float z data:");

    pfn_dequantize_row_q8_K_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q8_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q8_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q8_k; i += 1) {
            pfn_dequantize_row_q8_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q8_k float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q8_k dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q8_k; i += 1) {
            pfn_dequantize_row_q8_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q8_k float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q8_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q8_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
dequantize_q8_0 (
    uint32_t vec_size
    )

//
// Compute the performance of dequantize q8_0 by generating a quantized set of q8_0 blocks,
// then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q8_0 * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_0 performance tests for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q8_0 vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK8_0) * sizeof(block_q8_0));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q8);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // quantize the x vector of floats for AVX2.
    //

    pfn_quantize_row_q8_0_AVX2(x, y, vec_size);

    //
    // log q8_0 quant data.
    //

    log_q8_0_quant_data(vec_size, (void *)y);

    //
    // quantize the x vector of floats for AVX512.
    //

    pfn_quantize_row_q8_0_AVX512(x, y, vec_size);

    //
    // log q8_0 quant data.
    //

    log_q8_0_quant_data(vec_size, (void *)y);

    //
    // dequantize generated q8_0 quant blocks.
    //

    pfn_dequantize_row_q8_0_AVX2(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX2 dequantized q8_0 float z data:");

    pfn_dequantize_row_q8_0_AVX512(y, z, vec_size);
    log_float_data(vec_size, (void *)z, "AVX512 dequantized q8_0 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_0 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q8_0 dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q8_0; i += 1) {
            pfn_dequantize_row_q8_0_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX2 dequantized q8_0 float z data:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q8_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the q8_0 dequantize.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q8_0; i += 1) {
            pfn_dequantize_row_q8_0_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log float output data.
    //

    log_float_data(vec_size, (void *)z, "AVX512 dequantized q8_0 float z data:");

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("dequantize_q8_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_dequantize_q8_0);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
quantize_q8_k (
    uint32_t vec_size
    )

//
// Compute the performance of the quantization of vector elements to q8_k.
//
//  y = quantize_q8_k(x);
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    block_q8_K * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q8_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q8_K));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q8);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the quantization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX2(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log quant q8_k output data.
    //

    log_q8_k_quant_data(vec_size, (void *)y, true);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the quantization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX512(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log quant q8_k output data.
    //
 
    log_q8_k_quant_data(vec_size, (void *)y, true);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("quantize_q8_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_quantize_q8_k);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
quantize_q8_0 (
    uint32_t vec_size
    )

//
// Compute the performance of the quantization of vector elements to q8_0.
//
//  y = quantize_q0_k(x);
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    block_q8_0 * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0 performance test for AVX2/AVX512\n\n");

    //
    // Allocate float vector and block_q8_0 vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK8_0) * sizeof(block_q8_0));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q8);
    }

    //
    // Log generated x data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the quantization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_0; i += 1) {
            pfn_quantize_row_q8_0_AVX2(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log quant q8_0 output data.
    //

    log_q8_0_quant_data(vec_size, (void *)y);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the quantization of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_0; i += 1) {
            pfn_quantize_row_q8_0_AVX512(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log quant q8_0 output output.
    //
 
    log_q8_0_quant_data(vec_size, (void *)y);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("quantize_q8_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_quantize_q8_0);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_bf16 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of bf16/bf16 vector elements.
//
//  sum = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_bf16_t * x;
    ggml_bf16_t * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate bf16 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_bf16_t));
    y = zalloc(vec_size * sizeof(ggml_bf16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_bf16(genx_float_value(FLOAT_FILTER));
        y[i] = convert_fp32_to_bf16(geny_float_value(FLOAT_FILTER)); 
    }

    //
    // Log generated x (bf16) and y (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (bf16) vector:");
    log_raw_data(vec_size / 2, (void *)y, "generated y (bf16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_bf16; i += 1) {
            pfn_ggml_vec_dot_bf16_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_bf16; i += 1) {
            pfn_ggml_vec_dot_bf16_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_bf16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_bf16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f16 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of f16/f16 vector elements.
//
//  sum = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    ggml_fp16_t * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f16 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_fp16(genx_float_value(FLOAT_FILTER));
        y[i] = convert_fp32_to_fp16(geny_float_value(FLOAT_FILTER)); 
    }

    //
    // Log generated x (f16) and y (f16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (f16) vector:");
    log_raw_data(vec_size / 2, (void *)y, "generated y (f16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_f16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_f16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of f32/f32 vector elements.
//
//  sum = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f32; i += 1) {
            pfn_ggml_vec_dot_f32_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f32; i += 1) {
            pfn_ggml_vec_dot_f32_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_bf16_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of bf16/f32 vector elements.
//
//  sum = x[i] * y[i]
//

{
    uint32_t i;
    uint32_t j;
    ggml_bf16_t * x;
    float * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate bf16 and f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_bf16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_bf16(genx_float_value(FLOAT_FILTER));
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x (bf16) and y data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (bf16) vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_bf16_f32; i += 1) {
            pfn_ggml_vec_dot_bf16_f32_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_bf16_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_bf16_f32; i += 1) {
            pfn_ggml_vec_dot_bf16_f32_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_bf16_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_bf16_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f16_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of summation of the product of f16/f32 vector elements.
//
//  sum = x[i] * y[i]
//

{
    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    float * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f16 and f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_fp16(genx_float_value(FLOAT_FILTER));
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x (f6) and y data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (f16) vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16_f32; i += 1) {
            pfn_ggml_vec_dot_f16_f32_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16_f32; i += 1) {
            pfn_ggml_vec_dot_f16_f32_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_f16_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_f16_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
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

    fprintf(logfile, "Running vec_dot_q2_K_q8_K performance test for AVX2/AVX512\n\n");

    //
    // Allocate  vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q2x = zalloc(vec_size / QK_K * sizeof(block_q2_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q2x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
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

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q2_K_AVX512(x, q2x, vec_size);
    pfn_quantize_row_q8_K_AVX512(y, q8y, vec_size);

    //
    // log q2_k and q8_k quant data.
    //

    log_q2_k_quant_data(vec_size, (void *)q2x, false);
    log_q8_k_quant_data(vec_size, (void *)q8y, false);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_K_q8_K performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q2_K_q8_K_AVX2(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_K_q8_K performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q2_K_q8_K_AVX512(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q2_K_q8_K:",
                 best_time_AVX2,
                 best_time_AVX512,
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

    fprintf(logfile, "Running vec_dot_q3_K_q8_K performance test for AVX2/AVX512\n\n");

    //
    // Allocate  vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q3x = zalloc(vec_size / QK_K * sizeof(block_q3_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q3x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
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

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q3_K_AVX512(x, q3x, vec_size);
    pfn_quantize_row_q8_K_AVX512(y, q8y, vec_size);

    //
    // log q3 and q8 quant data.
    //

    log_q3_k_quant_data(vec_size, (void *)q3x, true);
    log_q8_k_quant_data(vec_size, (void *)q8y, true);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q3_K_q8_K performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q3_K_q8_K_AVX2(vec_size, &sum, 0, q3x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q3_K_q8_K performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q3_K_q8_K_AVX512(vec_size, &sum, 0, q3x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q3_K_q8_K:",
                 best_time_AVX2,
                 best_time_AVX512,
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

    fprintf(logfile, "Running vec_dot_q4_K_q8_K performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q4x = zalloc(vec_size / QK_K * sizeof(block_q4_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q4x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
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

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_K_AVX512(x, q4x, vec_size);
    pfn_quantize_row_q8_K_AVX512(y, q8y, vec_size);

    //
    // log q4 and q8_k quant data.
    //

    log_q4_k_quant_data(vec_size, (void *)q4x, true);
    log_q8_k_quant_data(vec_size, (void *)q8y, true);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_K_q8_K performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q4_K_q8_K_AVX2(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_K_q8_K performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q4_K_q8_K_AVX512(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q4_K_q8_K:",
                 best_time_AVX2,
                 best_time_AVX512,
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

    fprintf(logfile, "Running vec_dot_q6_K_q8_K performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q6x = zalloc(vec_size / QK_K * sizeof(block_q6_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q6x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
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

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the
    //      same.
    //

    pfn_quantize_row_q6_K_AVX512(x, q6x, vec_size);
    pfn_quantize_row_q8_K_AVX512(y, q8y, vec_size);

    //
    // log q6 and q8_k quant data.
    //

    log_q6_k_quant_data(vec_size, (void *)q6x, true);
    log_q8_k_quant_data(vec_size, (void *)q8y, true);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q6_K_q8_K performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q6_K_q8_K_AVX2(vec_size, &sum, 0, q6x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q6_K_q8_K performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q6_K_q8_K_AVX512(vec_size, &sum, 0, q6x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q6_K_q8_K:",
                 best_time_AVX2,
                 best_time_AVX512,
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

void
vec_dot_q4_0_q8_0 (
    uint32_t vec_size
    )

//
// Compute the performance of the summation of the product of q4_0/q8_0 vector
// elements.
//
//  sum = q4x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q4_0 * q4x;
    block_q8_0 * q8y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q4x = zalloc(vec_size / QK4_0 * sizeof(block_q4_0));
    q8y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    if (!x || !y || !q4x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
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

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_0_AVX512(x, q4x, vec_size);
    pfn_quantize_row_q8_0_AVX512(y, q8y, vec_size);

    //
    // log q4/q8 quant data.
    //

    log_q4_0_quant_data(vec_size, (void *)q4x);
    log_q8_0_quant_data(vec_size, (void *)q8y);

    //
    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0 performance test for AVX2\n\n");

    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q4_0_q8_0_AVX2(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q4_0_q8_0_AVX512(vec_size, &sum, 0, q4x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q4_0_q8_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_q4_0_q8_0);

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
vec_dot_q8_0_q8_0 (
    uint32_t vec_size
    )

//
// Compute the performance of the summation of the product of q8_0/q8_0 vector
// elements.
//
//  sum = q8x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;
    block_q8_0 * q8x;
    block_q8_0 * q8y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test for AVX2/AVX512\n\n");

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q8x = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    q8y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    if (!x || !y || !q8x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
        y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q8_0_AVX512(x, q8x, vec_size);
    pfn_quantize_row_q8_0_AVX512(y, q8y, vec_size);

    //
    // log q8 quant data.
    //

    log_q8_0_quant_data(vec_size, (void *)q8x);
    log_q8_0_quant_data(vec_size, (void *)q8y);

    //
    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test for AVX2\n\n");

    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q8_0_q8_0_AVX2(vec_size, &sum, 0, q8x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q8_0_q8_0_AVX512(vec_size, &sum, 0, q8x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_dot_q8_0_q8_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_dot_q8_0_q8_0);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q8x) {
        zfree(q8x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void
make_q4_0_repack_quant (
    uint64_t ne,
    block_q4_0_repack * out,
    block_q4_0 * in
    )

//
// Convert groups of eight q4_0 quant blocks to one q4_0_repack quant block. The q4_0
// quant blocks are interleaved in the q4_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    block_q8_0 qs_in[8];
    block_q8_0_repack qs_out;

    const __m256i m4 = _mm256_set1_epi8(0xf);

    //
    // Convert groups of eight q4_0 quant blocks into one q4_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {
        for (i = 0; i < 8; i += 1) {

            //
            // Unpack 8 q4_0 quant blocks into 8 temporary q8_0 quant blocks.
            //
            // N.B. The multiplier value from the q4_0 quant blocks is moved
            //      directly to the temporary q8_0_repack quant block.
            //

            qs_out.d[i] = in[i].d;
            const __m128i tmp1 = _mm_loadu_si128((const __m128i *)in[i].qs);
            const __m128i tmp2 = _mm_srli_epi16(tmp1, 4);
            __m256i qx = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp2, 1);
            qx = _mm256_and_si256(m4, qx);
            _mm256_storeu_si256((__m256i *)qs_in[i].qs, qx);
        }
    
        //
        // Rearrange the quant bytes into lanes of four bytes interleaved into the
        // temporary q8_0_repack quant block.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset + 0];
                uint32_t * qs_src = (uint32_t *)&qs_in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }
    
        //
        // Repack the temporary q8_0_repack quant block into the q4_0_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_0_repack format is performed
        //      64 quant values rather than the 32 quant values of the q4_0 format. This
        //      makes unpacking of the nibble values in the eventual vector dot function
        //      more efficient.
        //
    
        for (i = 0; i < 4; i += 1) {
            const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 0]);
            __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 32]);
            tmp2 = _mm256_slli_epi16(tmp2, 4);
            tmp2 = _mm256_or_epi32(tmp1, tmp2);
            _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
        }

        //
        // Copy the multiplier values to the q4_0_repack quant block.
        //

        const __m128i d = _mm_loadu_si128((__m128i *)qs_out.d);
        _mm_storeu_si128((__m128i *)out->d, d);

        out += 1;
        in += 8;
    }
}

void
make_q2_k_repack_quant (
    uint64_t ne,
    block_q2_K_repack * out,
    block_q2_K * in
    )

//
// Convert one q2_K quant block to one q2_K_repack quant block. The q2_K quant block
// values are interleaved in the q2_K_repack quant block such that they can be loaded
// and acted on directly by the q2_k_q8_k_x8 vector dot code.
//
// N.B. The q2_k quant block and q2_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q2_k quant block to one q2_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multipliers/(d-dmin) and scales/(mins-scales) values directly
        // from the input quant block to the output quant block.
        //
    
        out->dm = in->dm;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 2-bit q2_k quant values into an array of 8-bit quant values.
        //

        const __m256i m3 = _mm256_set1_epi8(0x3);

        for (i = 0; i < (QK_K / 128); i += 1) {
            __m256i q2bits = _mm256_loadu_si256((const __m256i*)&in->qs[i * 32]);

            for (j = 0; j < (QK_K / 64); j += 1) {
                __m256i q2x = _mm256_and_si256(q2bits, m3);
                _mm256_storeu_si256((__m256i *)&qs_in[i * 128 + j * 32], q2x);

                q2bits = _mm256_srli_epi16(q2bits, 2);
            }
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q2_k, and thus, 16 lanes are required.
        //
    
        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q2_k_repack quant block.
        //

        __m512i q2bytes = _mm512_loadu_si512((const __m512i*)&qs_out[0]);
        const __m512i q2bytes_0 = _mm512_loadu_si512((const __m512i*)&qs_out[64]);
        const __m512i q2bytes_1 = _mm512_loadu_si512((const __m512i*)&qs_out[128]);
        const __m512i q2bytes_2 = _mm512_loadu_si512((const __m512i*)&qs_out[192]);

        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_0, 2));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_1, 4));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q2bytes_2, 6));

        _mm512_storeu_si512((__m512i *)out->qs, q2bytes);

        out += 1;
        in += 1;
    }
}
                      
void
make_q3_k_repack_quant (
    uint64_t ne,
    block_q3_K_repack * out,
    block_q3_K * in
    )

//
// Convert one q3_K quant block to one q3_K_repack quant block. The q3_K quant block
// values are interleaved in the q3_K_repack quant block such that they can be loaded
// and acted on directly by the q3_k_q8_k_x8 vector dot code.
//
// N.B. The q3_k quant block and q3_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q3_k quant block to one q3_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    const __m256i m3_256 = _mm256_set1_epi8(3);
    const __m512i m3_512 = _mm512_set1_epi8(3);
    const __m256i m4_256 = _mm256_set1_epi8(4);

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier and scales values directly from the input quant block
        // to the output quant block.
        //
    
        out->d = in->d;
        memcpy(out->scales, in->scales, sizeof(in->scales));

        //
        // Load the high bits and preshift to first set of bits.
        //

        __m256i hbits = _mm256_loadu_si256((const __m256i*)in->hmask);
        hbits = _mm256_rol_epi32(hbits, 2);

        //
        // Unpack the 3-bit q3_k quant values into an array of 8-bit quant values.
        //

        for (i = 0; i < (QK_K / 128); i += 1) {
            __m256i q2bits = _mm256_loadu_si256((const __m256i*)&in->qs[i * 32]);

            for (j = 0; j < (QK_K / 64); j += 1) {

                //
                // Isolate 2-bit values, merge with high bits, and store the value.
                //

                const __m256i q3l = _mm256_and_si256(q2bits, m3_256);
                const __m256i q3h = _mm256_and_si256(hbits, m4_256);
                const __m256i q3 = _mm256_or_si256(q3l, q3h);

                _mm256_storeu_si256((__m256i *)&qs_in[i * 128 + j * 32], q3);

                hbits = _mm256_ror_epi32(hbits, 1);
                q2bits = _mm256_srli_epi16(q2bits, 2);
            }
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q3_k, and thus, 16 lanes are required.
        //
    
        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q3_k_repack quant block.
        //
        // N.B. The repacking is on 64-byte boundaries instead of the normal 32-byte
        //      boundaries. This is to facilitate more efficent computation of the
        //      dot product.
        //

        const __m512i q3bytes_0 = _mm512_loadu_si512((const __m512i*)&qs_out[0]);
        const __m512i q3bytes_1 = _mm512_loadu_si512((const __m512i*)&qs_out[64]);
        const __m512i q3bytes_2 = _mm512_loadu_si512((const __m512i*)&qs_out[128]);
        const __m512i q3bytes_3 = _mm512_loadu_si512((const __m512i*)&qs_out[192]);

        //
        // Isolate low bits.
        //

        const __m512i q3lbits_0 = _mm512_and_si512(q3bytes_0, m3_512);
        const __m512i q3lbits_1 = _mm512_and_si512(q3bytes_1, m3_512);
        const __m512i q3lbits_2 = _mm512_and_si512(q3bytes_2, m3_512);
        const __m512i q3lbits_3 = _mm512_and_si512(q3bytes_3, m3_512);

        //
        // Pack low bits into 64 bytes.
        //

        __m512i q2bytes = _mm512_or_si512(q3lbits_0, _mm512_slli_epi16(q3lbits_1, 2));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q3lbits_2, 4));
        q2bytes = _mm512_or_si512(q2bytes, _mm512_slli_epi16(q3lbits_3, 6));

        _mm512_storeu_si512((__m512i *)out->qs, q2bytes);

        //
        // Shift the high bits into the sign bit.
        //

        const __m512i q3hbits_0 = _mm512_slli_epi16(q3bytes_0, 5);
        const __m512i q3hbits_1 = _mm512_slli_epi16(q3bytes_1, 5);
        const __m512i q3hbits_2 = _mm512_slli_epi16(q3bytes_2, 5);
        const __m512i q3hbits_3 = _mm512_slli_epi16(q3bytes_3, 5);

        //
        // Collect the high bits into 64-bit masks.
        //
        // N.B. The high bits are stored as the mask of sign bits.
        //

        const __mmask64 high_bits_mask_0 = _mm512_movepi8_mask(q3hbits_0);
        const __mmask64 high_bits_mask_1 = _mm512_movepi8_mask(q3hbits_1);
        const __mmask64 high_bits_mask_2 = _mm512_movepi8_mask(q3hbits_2);
        const __mmask64 high_bits_mask_3 = _mm512_movepi8_mask(q3hbits_3);

        *((uint64_t *)out->hmask + 0) = _cvtmask64_u64(high_bits_mask_0);
        *((uint64_t *)out->hmask + 1) = _cvtmask64_u64(high_bits_mask_1);
        *((uint64_t *)out->hmask + 2) = _cvtmask64_u64(high_bits_mask_2);
        *((uint64_t *)out->hmask + 3) = _cvtmask64_u64(high_bits_mask_3);

        out += 1;
        in += 1;
    }
}
                      
void
make_q4_k_repack_quant (
    uint64_t ne,
    block_q4_K_repack * out,
    block_q4_K * in
    )

//
// Convert one q4_K quant block to one q4_K_repack quant block. The q4_K quant block
// values are interleaved in the q4_K_repack quant block such that they can be loaded
// and acted on directly by the vector q4_k_q8_k_x8 code.
//
// N.B. The q4_k quant block and q4_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q4_k quant block to one q4_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multipliers/(d-dmin) and scales/(mins-scales) values directly
        // from the input quant block to the output quant block.
        //
    
        out->dm = in->dm;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 4-bit q4_k quant values into an array of 8-bit quant values.
        //
    
        const __m512i m4 = _mm512_set1_epi8(0xf);
    
        for (i = 0; i < (QK_K / 64); i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)(in->qs + i * 32));
            __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
    
            __m512i qx = _mm512_castsi256_si512(tmp1); 
            qx = _mm512_inserti32x8(qx, tmp2, 1);
    
            qx = _mm512_and_si512(m4, qx);
            _mm512_storeu_si512((__m512i *)(&qs_in[i * 64]), qx);
        }
    
        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Repack the q8_k quant block into the q4_k_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_k_repack format is
        //      performed as 64 quant values rather than the 32 quant values of the
        //      q4_k format. This makes unpacking of the nibble values in the eventual
        //      vector dot function more efficient.
        //
    
        for (i = 0; i < 4; i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 0]);
            __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 32]);
    
            tmp2 = _mm256_slli_epi16(tmp2, 4);
            tmp2 = _mm256_or_epi32(tmp1, tmp2);
            _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
        }

        out += 1;
        in += 1;
    }
}

void
make_q6_k_repack_quant (
    uint64_t ne,
    block_q6_K_repack * out,
    block_q6_K * in
    )

//
// Convert one q6_K quant block to one q6_K_repack quant block. The q6_K quant block
// values are interleaved in the q6_K_repack quant block such that they can be loaded
// and acted on directly by the q6_k_q8_k_x8 vector dot code.
//
// N.B. The q6_k quant block and q6_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q6_k quant block to one q6_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (uint64_t k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier and scales values directly from the input quant block
        // to the output quant block.
        //
    
        out->d = in->d;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 6-bit q6_k quant values into an array of 8-bit quant values.
        //

        const __m256i m4_256 = _mm256_set1_epi8(0xf);
        const __m512i m4_512 = _mm512_set1_epi8(0xf);
        const __m256i m2_256 = _mm256_set1_epi8(0x30);
        const __m512i m2_512 = _mm512_set1_epi8(0x30);

        for (uint64_t j = 0; j < (QK_K / 128); j += 1) {

            //
            // Load the low 4-bit values and the high 2-bit values.
            //

            __m256i q4bits1 = _mm256_loadu_si256((__m256i *)(in->ql + (j * 64) + 0));
            __m256i q4bits2 = _mm256_loadu_si256((__m256i *)(in->ql + (j * 64) + 32));
            const __m256i q2bitsH = _mm256_loadu_si256((__m256i *)(in->qh + (j * 32)));

            //
            // Unpack the high (<5:4>) 2-bit values.
            //

            const __m256i q2_0 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 4), m2_256);
            const __m256i q2_1 = _mm256_and_si256(_mm256_rol_epi64(q2bitsH, 2), m2_256);
            const __m256i q2_2 = _mm256_and_si256(q2bitsH, m2_256);
            const __m256i q2_3 = _mm256_and_si256(_mm256_ror_epi64(q2bitsH, 2), m2_256);

            //
            // Unpack the low (<3:0>) 4-bit values and or in the high 2-bit values.
            //

            const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4_256), q2_0);
            const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4_256), q2_1);

            q4bits1 = _mm256_srli_epi16(q4bits1, 4);
            const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4_256), q2_2);

            q4bits2 = _mm256_srli_epi16(q4bits2, 4);
            const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4_256), q2_3);

            //
            // Store the 8-bit quant values.
            //

            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 0], q6_0);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 32], q6_1);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 64], q6_2);
            _mm256_storeu_si256((__m256i *)&qs_in[(j * 128) + 96], q6_3);
        }

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q6_k, and thus, 16 lanes are required.
        //
    
        for (uint64_t i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (uint64_t j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }

        //
        // Repack the 8-bit quant values into the q6_k_repack quant block.
        //

        const __m512i qs_0 = _mm512_loadu_si512(&qs_out[0]);
        const __m512i qs_1 = _mm512_loadu_si512(&qs_out[64]);
        const __m512i qs_2 = _mm512_loadu_si512(&qs_out[128]);
        const __m512i qs_3 = _mm512_loadu_si512(&qs_out[192]);

        __m512i qs4_l = _mm512_and_si512(qs_0, m4_512);
        __m512i qs4_h = _mm512_and_si512(qs_1, m4_512);
        __m512i qs4 = _mm512_or_si512(qs4_l, _mm512_slli_epi16(qs4_h, 4));
        _mm512_storeu_si512((__m512i *)&out->ql[0], qs4);

        qs4_l = _mm512_and_si512(qs_2, m4_512);
        qs4_h = _mm512_and_si512(qs_3, m4_512);
        qs4 = _mm512_or_si512(qs4_l, _mm512_slli_epi16(qs4_h, 4));
        _mm512_storeu_si512((__m512i *)&out->ql[64], qs4);

        const __m512i qs2_0 = _mm512_srli_epi16(_mm512_and_si512(qs_0, m2_512), 4);
        const __m512i qs2_1 = _mm512_srli_epi16(_mm512_and_si512(qs_1, m2_512), 2);
        const __m512i qs2_2 = _mm512_and_si512(qs_2, m2_512);
        const __m512i qs2_3 = _mm512_slli_epi16(_mm512_and_si512(qs_3, m2_512), 2);
        __m512i qs2 = _mm512_or_si512(qs2_0, qs2_1);
        qs2 = _mm512_or_si512(qs2, qs2_2);
        qs2 = _mm512_or_si512(qs2, qs2_3);
        _mm512_storeu_si512((__m512i *)&out->qh[0], qs2);

        out += 1;
        in += 1;
    }
}
                      
void
make_q8_0_repack_quant (
    uint64_t ne,
    block_q8_0_repack * out,
    block_q8_0 * in
    )

//
// Convert groups of eight q8_0 quant blocks to one q8_0_repack quant block. The q8_0
// quant blocks are interleaved in the q8_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    block_q8_0_repack qs_out;

    //
    // Convert groups of eight q8_0 quant blocks into one q8_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {
        for (i = 0; i < 8; i += 1) {
            qs_out.d[i] = in[i].d;
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset + 0];
                uint32_t * qs_src = (uint32_t *)&in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }

        //
        // Copy the temporary q8_0_repack quant block to the output quant block.
        //

        memcpy(out, &qs_out, sizeof(*out));

        out += 1;
        in += 8;
    }
}
    
void
make_q236_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in
    )

//
// Convert one q8_K quant block to one q8_K_repack quant block. The q8_K quant block
// values are interleaved in the block_q8_K_repack quant block such that they can be
// loaded and acted on directly in the q2_k_q8_k_x8/q3_k_q8_k_x8 vector dot code.
//
// N.B. The block_q8_K quant block and block_q8_K_repack quant block have the same
//      layout and storage requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_out[QK_K];

    //
    // Convert one q8_k quant block to one q8_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier/(d) value from the input quant block to the output
        // quant block.
        //
    
        out->d = in->d;

        //
        // Copy the bsum values from the input quant block to the output quant block.
        //

        memcpy(out->bsums, in->bsums, sizeof(out->bsums));

        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        //
        // N.B. There are 16 scale values for q2_k/q3_k, and thus, 16 lanes are required.
        //      

        for (i = 0; i < 16; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 4; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in->qs[i * 16 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 64;
            }
        }
    
        //
        // Copy the rearranged q8_k quant block into the q8_k_repack quant block.
        //
    
        memcpy(out->qs, qs_out, sizeof(out->qs));

        out += 1;
        in += 1;
    }
}

void
make_q4_k_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in
    )

//
// Convert one q8_K quant block to one q8_K_repack quant block. The q8_K quant block
// values are interleaved in the block_q8_K_repack quant block such that they can be
// loaded and acted on directly in the q4_k_q8_k_x8 vector dot code.
//
// N.B. The block_q8_K quant block and block_q8_K_repack quant block have the same
//      layout and storage requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_out[QK_K];

    //
    // Convert one q8_k quant block to one q8_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier/(d) value from the input quant block to the output
        // quant block.
        //
    
        out->d = in->d;

        //
        // Precompute bsums half add that is required in the q4_k_q8_k vector dot
        // function.
        //

        __m128i bsums0 = _mm_loadu_si128((__m128i *)&in->bsums[0]);
        __m128i bsums1 = _mm_loadu_si128((__m128i *)&in->bsums[8]);
        bsums0 = _mm_hadd_epi16(bsums0, bsums1);
        _mm_storeu_si128((__m128i *)&out->bsums[0], bsums0);
        _mm_storeu_si128((__m128i *)&out->bsums[8], bsums0);
    
        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in->qs[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Copy the rearranged q8_k quant block into the q8_k_repack quant block.
        //
    
        memcpy(out->qs, qs_out, sizeof(out->qs));

        out += 1;
        in += 1;
    }
}

void
xx_vec_dot_q4_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_0_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q4_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute combined multiplier for the quant block.
                //
        
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
        
                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&x[i].qs[j * QK4_0 + 0]);
                    __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
        
                    const __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
                    __m512i qx = _mm512_inserti32x8(_mm512_castsi256_si512(tmp1), tmp2, 1);
                    qx = _mm512_and_si512(m4, qx);
        
                    //
                    // Multiply unsigned scaled q4 quant values by signed q8 quant
                    // values and acculate the results.
                    //
        
                    sumi = _mm512_dpbusd_epi32(sumi, qx, qy);
        
                    //
                    // Multiply the unsigned bias quant values by the signed q8 quant
                    // values and acculate the results.
                    //
        
                    bias = _mm512_dpbusd_epi32(bias, offset, qy);
                }
        
                //
                // Subtract the bias values from the sum, convert to float, multiply
                // by the block multiplier, and accumulate the results.
                //
        
                sumi = _mm512_sub_epi32(sumi, bias);
                acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                         _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
xx_vec_dot_q2_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q2_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q2_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m256 mins_acc = _mm256_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        
                const uint8_t * q2 = x[i].qs;
                const int8_t * q8 = y[i].qs;
        
                const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
        
                const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
                const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        
                const __m512i scale = _mm512_cvtepi8_epi32(scales8);
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q2 and q8 quants and accumulate the
                // integer results.
                //
        
                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
                    const __m512i q2v = _mm512_and_si512(q2bits, m3);
                    q2bits = _mm512_srli_epi16(q2bits, 2);
        
                    sumi = _mm512_dpbusd_epi32(sumi, q2v, q8v);
                }
        
                //
                // Multiply the accumulated integer result by the q2 scale, convert to float,
                // multiply by the q8 multiplier, and accumulate the results.
                //
        
                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
        
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
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
xx_vec_dot_q3_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q3_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t kmask1 = 0x0000000030303030ull;
    const uint64_t kmask2 = 0x0f0f0f0f0f0f0f0full;

    const uint64_t nb = n / QK_K;

    const __m512i m3 = _mm512_set1_epi8(3);
    const __m512i m4 = _mm512_set1_epi8(4);
    const __m128i m32 = _mm_set1_epi8(32);

    const __m128i zero128 = _mm_setzero_si128();
    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q3_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                const uint8_t * q3 = x[i].qs;
                const int8_t * q8 = y[i].qs;
        
                //
                // Set up scales. There are 16 scales packed in 6-bit format.
                //

                uint64_t scales_h = *((uint64_t *)(x[i].scales));
                uint64_t scales_l = scales_h & kmask2;
                scales_h = (scales_h >> 4) & kmask2;

                uint64_t high_2bits = *((uint32_t *)(x[i].scales + 8));

                scales_h |= ((high_2bits >> 2) & kmask1) << 32;
                scales_h |= (high_2bits & kmask1);

                scales_l |= ((high_2bits << 2) & kmask1) << 32;
                scales_l |= ((high_2bits << 4) & kmask1);

                __m128i scales8 = _mm_insert_epi64(zero128, scales_l, 0);
                scales8 = _mm_insert_epi64(scales8, scales_h, 1);

#if 0
                uint32_t * aux = (uint32_t *)x[i].scales;
                __m128i scales8 = _mm_set_epi32(
                        ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                        ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                        (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                        (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
#endif // #if 0
        
                scales8 = _mm_sub_epi8(scales8, m32);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                //
                // Compute the integer product of the q3 and q8 quants and accumulate the
                // integer results.
                //
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();
        
                __m512i q3bytes = _mm512_loadu_si512((const __m512i*)q3);
        
                for (uint64_t j = 0; j < QK_K / 64; j += 1) {
        
                    //
                    // Load the q8 values.
                    //
        
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
        
                    //
                    // Isolate the low 2-bits of the q3 values and move to the next value.
                    //
        
                    const __m512i q3lv = _mm512_and_si512(q3bytes, m3);
                    q3bytes = _mm512_srli_epi16(q3bytes, 2);
        
                    //
                    // Multiply the low 2-bit unsigned values by the signed q8 quant
                    // values and acculate the results.
                    //
        
                    sumi = _mm512_dpbusd_epi32(sumi, q3lv, q8v);
        
                    //
                    // Compute the high bits from the q3 mask.
                    //
                    // N.B. The high bit is flipped to conform with the unpacked
                    //      implementation, i.e., and_not.
                    //
        
                    uint64_t high_bits = *((uint64_t *)x[i].hmask + j);
                    const __mmask64 hmask = _cvtu64_mask64(high_bits);
                    const __m512i q3hv = _mm512_mask_blend_epi8(hmask, m4, zero512);
        
                    //
                    // Multiply the unsigned high values by the signed q8 quant
                    // values and accumulate the results.
                    //
        
                    bias = _mm512_dpbusd_epi32(bias, q3hv, q8v);
                }
        
                //
                // Subtract the bias values from the sum, multiply the accumulated sum
                // by the q3 scale, convert to float, multiply by the block multiplier,
                // and accumulate the results.
                //
        
                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }
        
            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
xx_vec_dot_q4_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q4_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask4 = 0xc0c0c0c0u;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m128i zero128 = _mm_setzero_si128();

    uint64_t utmp[2];

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows
        //

        const block_q4_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();
            __m128 mins_acc = _mm_setzero_ps();
        
            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
                const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);
        
                const uint32_t * vscales = (uint32_t *)x[i].scales;
                utmp[1] = (uint64_t)(((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2)) << 32;
                utmp[1] |= (uint64_t)(vscales[1] & kmask1);
                utmp[0] = (uint64_t)((vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2)) << 32;
                utmp[0] |= (uint64_t)(vscales[0] & kmask1);
        
                const uint8_t * q4 = x[i].qs;
                const int8_t  * q8 = y[i].qs;
        
                //
                // Insert 8 q4 mins and 8 q4 scales.
                //
                // N.B. Both mins and scales are 6-bit unsigned values.
                //
        
                const __m128i scales8 = _mm_insert_epi64(zero128, utmp[0], 0);
                const __m128i mins8 = _mm_insert_epi64(zero128, utmp[1], 0);
        
                //
                // Compute the scale vector.
                //
                // N.B. The 8 scale values are replicated to 16 scale values.
                //
        
                __m512i scale = _mm512_cvtepi8_epi32(scales8);
                scale = _mm512_inserti64x4(scale, _mm512_castsi512_si256(scale), 1);
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute the integer product of the q4 and q8 quants and accumulate the
                // integer results.
                //
        
                for (uint64_t j = 0; j < QK_K / 64; ++j) {
                    const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
                    const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));
        
                    __m512i q4v = _mm512_castsi256_si512(q4bits);
                    q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
                    q4v = _mm512_and_si512(q4v, m4);
        
                    sumi = _mm512_dpbusd_epi32(sumi, q4v, q8v);
                }
        
                //
                // Multiply the accumulated integer result by the q4 scale, convert to float,
                // multiply by the q8 multiplier, and accumulate the results.
                //
        
                sumi = _mm512_mullo_epi32(sumi, scale);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
        
                //
                // Load 8 q8 bsums values, multiply by the mins values, and accumulate
                // the floating results.
                //
                // N.B. The half add to fold the 16 bsum values into 8 values is performed
                //      in the make q8_k quant code.
                //
        
                const __m128i q8s = _mm_loadu_si128((const __m128i *)y[i].bsums);
                const __m128i mins = _mm_cvtepi8_epi16(mins8);
                const __m128i prod = _mm_madd_epi16(mins, q8s);
                mins_acc = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), mins_acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));
        
            t0 = _mm_sub_ps(t0, mins_acc);
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
xx_vec_dot_q6_k_q8_k_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q6_K_repack * vx,
    size_t bx,
    const block_q8_K_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512i m2 = _mm512_set1_epi8(0x30);
    const __m512i m32s = _mm512_set1_epi8(32);

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_K_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate through the specified number of rows.
        //

        const block_q6_K_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        
                const uint8_t * q4 = x[i].ql;
                const uint8_t * q2 = x[i].qh;
                const int8_t * q8 = y[i].qs;
        
                const __m128i scales8 = _mm_loadu_si128((const __m128i*)x[i].scales);
                const __m512i scales = _mm512_cvtepi8_epi32(scales8);
        
                __m512i bias = _mm512_setzero_si512();
                __m512i sumi = _mm512_setzero_si512();

                //
                // Load the high 2-bits and rotate the bits left by 4 - 256 2-bit values.
                //
                // N.B. This lines up the first set of 2-bits in the proper position to
                //      merge with the first set of 4-bits.
                //

                __m512i q2bits = _mm512_loadu_si512((const __m512i*)q2);
                q2bits = _mm512_rol_epi32(q2bits, 4);

                //
                // Unpack the 6-bit quant values into unsigned 8-bit quant values
                // and perform dot product.
                //

                for (uint64_t j = 0; j < QK_K / 128; ++j) {

                    //
                    // Load 64 nibble packed 4-bit quant values - 128 4-bit values.
                    //

                    __m512i q4bits = _mm512_loadu_si512((const __m512i*)(q4 + (j * 64)));

                    //
                    // Merge the low 4-bits and current high 2-bits of the 6-bit quant
                    // values into unsigned 6-bit quant values.
                    //
        
                    const __m512i q4bits1 = _mm512_and_si512(q4bits, m4);
                    const __m512i q2bits1 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_0 = _mm512_or_si512(q2bits1, q4bits1);

                    //
                    // Shift next set of high 2-bits into place.
                    //

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Load 8-bit quant values - 64 8-bit values.
                    //

                    const __m512i q8_0 = _mm512_loadu_si512(q8 + (j * 128) + 0);

                    //
                    // Multiply the unsigned q6 quant values by the signed q8 quant
                    // values and accumulate the results.
                    // 

                    sumi = _mm512_dpbusd_epi32(sumi, q6_0, q8_0);

                    //
                    // Multiply the unsigned bias values by the signed q8 quant
                    // values, and accumulate the results.
                    //

                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_0);

                    //
                    // Shift the next nibble into place.
                    //

                    __m512i q4bits2 = _mm512_srli_epi16(q4bits, 4);

                    //
                    // Merge the high 4-bits and current high 2-bits of the 6-bit
                    // quant value into unsigned 6-bit quant values.
                    //

                    q4bits2 = _mm512_and_si512(q4bits2, m4);
                    const __m512i q2bits2 = _mm512_and_si512(q2bits, m2);
                    const __m512i q6_1 = _mm512_or_si512(q2bits2, q4bits2);

                    //
                    // Shift next set of high 2-bits into place.
                    //

                    q2bits = _mm512_ror_epi32(q2bits, 2);

                    //
                    // Load 8-bit quant values - 64 8-bit values.
                    //

                    const __m512i q8_1 = _mm512_loadu_si512(q8 + (j * 128) + 64);

                    //
                    // Multiply the unsigned q6 quant values by the signed q8 quant
                    // values and accumulate the results.
                    // 

                    sumi = _mm512_dpbusd_epi32(sumi, q6_1, q8_1);

                    //
                    // Multiply the unsigned bias values by the signed q8 quant
                    // values and accumulate the results.
                    //

                    bias = _mm512_dpbusd_epi32(bias, m32s, q8_1);
                }

                //
                // Subtract the bias values from the sum, multiply the accumulated
                // sum by the q6 scale, convert to float, multiply by the block
                // multiplier, and accumulate the results.
                //

                sumi = _mm512_sub_epi32(sumi, bias);
                sumi = _mm512_mullo_epi32(sumi, scales);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
            }

            __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                       _mm512_extractf32x8_ps(acc, 1));

            __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                   _mm256_extractf128_ps(res, 1));

            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));

            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
xx_vec_dot_q8_0_q8_0_x8 (
    uint32_t n,
    float * s,
    size_t nr_nb1,
    const block_q8_0_repack * vx,
    size_t bx,
    const block_q8_0_repack * vy,
    size_t ncols,
    uint32_t nrows
    )
{
    GGML_UNUSED(bx);

    const uint64_t nb = n / QK_K;

    const __m512i zero512 = _mm512_setzero_si512();

    //
    // Iterate through the specified number of columns.
    //

    const block_q8_0_repack * y = vy;

    for (uint64_t l = 0; l < ncols; l += 1) {

        //
        // Iterate throught the specified number of rows.
        //

        const block_q8_0_repack * x = vx;

        for (uint64_t k = 0; k < nrows; k += 1) {
            __m512 acc = _mm512_setzero_ps();

            for (uint64_t i = 0; i < nb; ++i) {
        
                __m512i sumi = _mm512_setzero_si512();
        
                //
                // Compute combined scale for the an entire quant block.
                //
        
                const __m128h xd = _mm_loadu_ph(x[i].d);
                const __m128h yd = _mm_loadu_ph(y[i].d);
                const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                                   _mm256_cvtph_ps(yd));
        
                __m512 d = _mm512_castps256_ps512(scale);
                d = _mm512_insertf32x8(d, scale, 1);
        
                //
                // Compute the dot product and accumulate.
                //
        
                for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
                    const __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2 + 0]);
                    const __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
    
                    //
                    // Compute the absolute value of qx and generate a mask of the corresponding
                    // values of qx that are negative.
                    //
    
                    const __m512i ax = _mm512_abs_epi8(qx);
                    const __mmask64 is_negative_qx = _mm512_movepi8_mask(qx);
    
                    //
                    // Compute the signed value of qy taking into account the negative values
                    // in the corresponding byte of qx.
                    //
    
                    const __m512i sy = _mm512_mask_sub_epi8(qy, is_negative_qx, zero512, qy);
    
                    //
                    // mul (ax * sy) + sumi directly to epi32
                    //
                    // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
                    //
            
                    sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
                }
        
                //
                // Multiply q with scale and accumulate.
                //
        
                const __m512 q = _mm512_cvtepi32_ps(sumi);
                acc = _mm512_fmadd_ps(d, q, acc);
            }
        
            const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                             _mm512_extractf32x8_ps(acc, 1));
        
            const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                         _mm256_extractf128_ps(res, 1));
        
            const __m128 t1 = _mm_hadd_ps(t0, t0);
            s[k] = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
    
            x += nb;
        }

        s = (float *)((char *)s + nr_nb1);
        y += nb;
    }
}

void
quantize_row_q4_0_x8 (
    const float * x,
    block_q4_0 * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q4_0 quants.
    //

    pfn_quantize_row_q4_0_AVX512(x, y, vec_size);

    //
    // Make q4_0_repack quant blocks
    //

    pfn_make_q4_0_repack_quant_AVX512(vec_size, (block_q4_0_repack *)y, y);
}

void                   
quantize_row_q2_k_x8 (
    const float * x,
    block_q2_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q2_K quants.
    //

    pfn_quantize_row_q2_K_AVX512(x, y, vec_size);


    //
    // Make q2_k_repack quant blocks.
    //

    pfn_make_q2_k_repack_quant_AVX512(vec_size, (block_q2_K_repack *)y, y);
}

void                   
quantize_row_q3_k_x8 (
    const float * x,
    block_q3_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q3_K quants.
    //

    pfn_quantize_row_q3_K_AVX512(x, y, vec_size);

    //
    // Make q3_k_repack quant blocks.
    //

    pfn_make_q3_k_repack_quant_AVX512(vec_size, (block_q3_K_repack *)y, y);
}

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q4_K quants.
    //

    pfn_quantize_row_q4_K_AVX512(x, y, vec_size);

    //
    // Make q4_k_repack quant blocks
    //

    pfn_make_q4_k_repack_quant_AVX512(vec_size, (block_q4_K_repack *)y, y);
}

void                   
quantize_row_q6_k_x8 (
    const float * x,
    block_q6_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q6_K quants.
    //

    pfn_quantize_row_q6_K_AVX512(x, y, vec_size);

    //
    // Make q6_k_repack quant blocks
    //

    pfn_make_q6_k_repack_quant_AVX512(vec_size, (block_q6_K_repack *)y, y);
}

void                   
quantize_row_q236_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q8_K quants.
    //

    pfn_quantize_row_q8_K_AVX512(x, y, vec_size);

    //
    // Make q8_k_repack quant blocks
    //

    pfn_make_q236_k_q8_k_repack_quant_AVX512(vec_size, (block_q8_K_repack *)y, y);
}

void                   
quantize_row_q4_k_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint32_t vec_size
    )
{

    //
    // Quantize the x vector into q8_K quants.
    //

    pfn_quantize_row_q8_K_AVX512(x, y, vec_size);

    //
    // Make q8_k_repack quant blocks
    //

    pfn_make_q4_k_q8_k_repack_quant_AVX512(vec_size, (block_q8_K_repack *)y, y);
}

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint32_t vec_size
    )
{

    //
    // quantize the x vector into q8_0 guants.
    //

    pfn_quantize_row_q8_0_AVX512(x, y, vec_size);

    //
    // Make q8_0_repack quant blocks
    //

    pfn_make_q8_0_repack_quant_AVX512(vec_size, (block_q8_0_repack *)y, y);
}

void
vec_dot_q4_0_q8_0_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q4_0_repack/block_q8_0_repack quant formats.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t m;
    uint64_t nbr;
    uint64_t nb0;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q4_0 * q40x;
    block_q8_0 * q80y;
    block_q4_0_repack * q4kx;
    block_q8_0_repack * q8ky;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0/q4_0_q8_0_repack performance tests\n\n");

    //
    // Check the number of q4_0_repack/q8_0_repack quant blocks that will be generated.
    //
    // N.B. This must be 0 mod QK_K.
    //

    nbr = vec_size / QK_K;
    nb0 = vec_size / QK4_0;
    if ((vec_size % QK4_0) || (vec_size % QK_K) || (QK4_0 != QK8_0)) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K/QK4_0)\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q40x = zalloc((vec_size / QK4_0 * sizeof(block_q4_0)) * mat_size);
    q80y = zalloc((vec_size / QK8_0 * sizeof(block_q8_0)) * mat_size);
    q4kx = zalloc((vec_size / QK_K * sizeof(block_q4_0_repack)) * mat_size);
    q8ky = zalloc((vec_size / QK_K * sizeof(block_q8_0_repack)) * mat_size);
    if (!sum || !x || !y || !q40x || !q80y || !q4kx || !q8ky) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;

    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q4_0 and q8_0 quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 AND avx512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q4_0_AVX512(&x[i * x_stride], &q40x[i * nb0], vec_size);
        pfn_quantize_row_q8_0_AVX512(&y[i * y_stride], &q80y[i * nb0], vec_size);
    }

    //
    // log the q4_0 and q8_0 quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q4_0_quant_data(vec_size, (void *)&q40x[i * nb0]);
        log_q8_0_quant_data(vec_size, (void *)&q80y[i * nb0]);
    }

    //
    // Quantize the x and y vectors for q4_0_repack and q8_0_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q4_0_x8_AVX512(&x[i * x_stride],
                                        (block_q4_0 *)&q4kx[i * nbr],
                                        vec_size);

        pfn_quantize_row_q8_0_x8_AVX512(&y[i * y_stride],
                                        (block_q8_0 *)&q8ky[i * nbr],
                                        vec_size);
    }

    //
    // log the q4_0 and q8_0 repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q4_0_repack_quant_data(vec_size, (void *)&q4kx[i * nbr]);
        log_q8_0_repack_quant_data(vec_size, (void *)&q8ky[i * nbr]);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0 performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q4_0_q8_0 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_0_q8_0; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q4_0_q8_0_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q40x[m * nb0],
                                                      0,
                                                      &q80y[k * nb0],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q4_0_q8_0) {
            best_time_q4_0_q8_0 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q4_0/q8_0 vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_0_q8_0_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q4_0_q8_0_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0_repack; i += 1) {
            pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q4kx,
                                               0,
                                               q8ky,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q4_0_q8_0_repack) {
            best_time_q4_0_q8_0_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q4_0_repack/q8_0_repack vector dot sum of products");

    //
    // Report percent increase from q4_0_q8_0 (AVX512) to q4_0_q8_0_repack (AVX512).
    //

    log_increase_x("vec_dot_q4_0_q8_0_repack:",
                   best_time_q4_0_q8_0,
                   "AVX512",
                   best_time_q4_0_q8_0_repack,
                   "repack AVX512",
                   iter_vec_dot_q4_0_q8_0);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q40x) {
        zfree(q40x);
    }

    if (q80y) {
        zfree(q80y);
    }

    if (q4kx) {
        zfree(q4kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }
}

void
test_dot_q4_0_q8_0_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q4_0_repack/block_q8_0_repack quant formats
// versus vec_dot_q4_0_q8_0.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q4_0 * q40x;
    block_q8_0 * q80y;
    block_q4_0_repack * q4rx;
    block_q8_0_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q4_0_q8_0/q4_0_q8_0_repack accuracy test\n\n");

    //
    // Check the number of q4_0_repack/q8_0_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Check if block sizes are correct.
    //

    if ((vec_size / QK4_0 * sizeof(block_q4_0)) != (vec_size / QK_K * sizeof(block_q4_0_repack)) ||
        (vec_size / QK8_0 * sizeof(block_q8_0)) != (vec_size / QK_K * sizeof(block_q8_0_repack))) {

        fprintf(logfile, "quant block sizes do not match\n");
        return;
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q40x = zalloc(vec_size / QK4_0 * sizeof(block_q4_0));
    q80y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    q4rx = zalloc(vec_size / QK_K * sizeof(block_q4_0_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_0_repack));
    if (!x || !y || !q40x || !q80y || !q4rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q4_0/q8_0 and q4_0_repack/q8_0_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q4_0/q8_0 and q4_0_repack/q8_0_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q4_0 and q8_0 quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q4_0_AVX512(x, q40x, vec_size);
        pfn_quantize_row_q8_0_AVX512(y, q80y, vec_size);

        //
        // Quantize the x and y vectors for q4_0_repack and q8_0_repack quants.
        //
    
        pfn_quantize_row_q4_0_x8_AVX512(x, (block_q4_0 *)q4rx, vec_size);
        pfn_quantize_row_q8_0_x8_AVX512(y, (block_q8_0 *)q8ry, vec_size);

        //
        // Compute the dot products for q4_0/q8_0 and q4_0_repack/q8_0_repack.
        //

        pfn_ggml_vec_dot_q4_0_q8_0_AVX512(vec_size, &sum0, 0, q40x, 0, q80y, 0, 1);
        pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512(vec_size, &sum1, 0, q4rx, 0, q8ry, 1, 1);

        //
        // Log vector q4_0/q8_0 sum of products.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q40x) {
        zfree(q40x);
    }

    if (q80y) {
        zfree(q80y);
    }

    if (q4rx) {
        zfree(q4rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
test_dot_q2_k_q8_k_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q2_K_repack/block_q8_K_repack quant formats
// versus vec_dot_q2_K_q8_k.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q2_K * q2kx;
    block_q8_K * q8ky;
    block_q2_K_repack * q2rx;
    block_q8_K_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q2_k_q8_k/q2_k_q8_k_repack accuracy test\n\n");

    //
    // Check the number of q2_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q2kx = zalloc(vec_size / QK_K * sizeof(block_q2_K));
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    q2rx = zalloc(vec_size / QK_K * sizeof(block_q2_K_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack));
    if (!x || !y || !q2kx || !q8ky || !q2rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q2_K/q8_K and q2_K_repack/q8_K_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q2_k/q8_k and q2_K_repack/q8_K_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q2_K and q8_K quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q2_K_AVX512(x, q2kx, vec_size);
        pfn_quantize_row_q8_K_AVX512(y, q8ky, vec_size);

        //
        // Quantize the x and y vectors for q2_K_x8 and q2_k_q8_K_x8 quants.
        //
    
        pfn_quantize_row_q2_k_x8_AVX512(x, (block_q2_K *)q2rx, vec_size);
        pfn_quantize_row_q236_k_q8_k_x8_AVX512(y, (block_q8_K *)q8ry, vec_size);

        //
        // Compute the dot products for q2_K/q8_K and q2_K_x8/q2_k_q8_k_x8 quants.
        //

        pfn_ggml_vec_dot_q2_K_q8_K_AVX512(vec_size, &sum0, 0, q2kx, 0, q8ky, 0, 1);
        pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512(vec_size, &sum1, 0, q2rx, 0, q8ry, 1, 1);

        //
        // Log vector q2_K/q8_K sum of products if there is a mismatch.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q2kx) {
        zfree(q2kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q2rx) {
        zfree(q2rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
test_dot_q3_k_q8_k_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q3_K_repack/block_q8_K_repack quant formats
// versus vec_dot_q3_K_q8_k.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q3_K * q3kx;
    block_q8_K * q8ky;
    block_q3_K_repack * q3rx;
    block_q8_K_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q3_k_q8_k/q3_k_q8_k_repack accuracy test\n\n");

    //
    // Check the number of q3_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q3kx = zalloc(vec_size / QK_K * sizeof(block_q3_K));
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    q3rx = zalloc(vec_size / QK_K * sizeof(block_q3_K_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack));
    if (!x || !y || !q3kx || !q8ky || !q3rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q3_K/q8_K and q3_K_repack/q8_K_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q3_k/q8_k and q3_K_repack/q8_K_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q2_K and q8_K quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q3_K_AVX512(x, q3kx, vec_size);
        pfn_quantize_row_q8_K_AVX512(y, q8ky, vec_size);

        //
        // Quantize the x and y vectors for q3_K_x8 and q3_k_q8_K_x8 quants.
        //
    
        pfn_quantize_row_q3_k_x8_AVX512(x, (block_q3_K *)q3rx, vec_size);
        pfn_quantize_row_q236_k_q8_k_x8_AVX512(y, (block_q8_K *)q8ry, vec_size);

        //
        // Compute the dot products for q2_K/q8_K and q2_K_x8/q2_k_q8_k_x8 quants.
        //

        pfn_ggml_vec_dot_q3_K_q8_K_AVX512(vec_size, &sum0, 0, q3kx, 0, q8ky, 0, 1);
        pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512(vec_size, &sum1, 0, q3rx, 0, q8ry, 1, 1);

        //
        // Log vector q3_K/q8_K sum of products if there is a mismatch.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q3kx) {
        zfree(q3kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q3rx) {
        zfree(q3rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
test_dot_q4_k_q8_k_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q4_K_repack/block_q8_K_repack quant formats
// versus vec_dot_q4_K_q8_k.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q4_K * q4kx;
    block_q8_K * q8ky;
    block_q4_K_repack * q4rx;
    block_q8_K_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q4_k_q8_k/q4_k_q8_k_repack accuracy test\n\n");

    //
    // Check the number of q4_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q4kx = zalloc(vec_size / QK_K * sizeof(block_q4_K));
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    q4rx = zalloc(vec_size / QK_K * sizeof(block_q4_K_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack));
    if (!x || !y || !q4kx || !q8ky || !q4rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q4_K/q8_K and q4_K_repack/q8_K_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q4_k/q8_k and q4_K_repack/q8_K_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q4_K and q8_K quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q4_K_AVX512(x, q4kx, vec_size);
        pfn_quantize_row_q8_K_AVX512(y, q8ky, vec_size);

        //
        // Quantize the x and y vectors for q4_K_repack and q8_K_repack quants.
        //
    
        pfn_quantize_row_q4_k_x8_AVX512(x, (block_q4_K *)q4rx, vec_size);
        pfn_quantize_row_q4_k_q8_k_x8_AVX512(y, (block_q8_K *)q8ry, vec_size);

        //
        // Compute the dot products for q4_K/q8_K and q4_K_repack/q8_k_repack.
        //

        pfn_ggml_vec_dot_q4_K_q8_K_AVX512(vec_size, &sum0, 0, q4kx, 0, q8ky, 0, 1);
        pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512(vec_size, &sum1, 0, q4rx, 0, q8ry, 1, 1);

        //
        // Log vector q4_K/q8_K sum of products.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q4kx) {
        zfree(q4kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q4rx) {
        zfree(q4rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
test_dot_q6_k_q8_k_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q6_K_repack/block_q6_K_repack quant formats
// versus vec_dot_q6_K_q8_k.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q6_K * q6kx;
    block_q8_K * q8ky;
    block_q6_K_repack * q6rx;
    block_q8_K_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q6_k_q8_k/q6_k_q8_k_repack accuracy test\n\n");

    //
    // Check the number of q6_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q6kx = zalloc(vec_size / QK_K * sizeof(block_q6_K));
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    q6rx = zalloc(vec_size / QK_K * sizeof(block_q6_K_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack));
    if (!x || !y || !q6kx || !q8ky || !q6rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q6_K/q8_K and q6_K_repack/q8_K_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q6_k/q8_k and q6_K_repack/q8_K_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q6_K and q8_K quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q6_K_AVX512(x, q6kx, vec_size);
        pfn_quantize_row_q8_K_AVX512(y, q8ky, vec_size);

        //
        // Quantize the x and y vectors for q6_K_repack and q8_K_repack quants.
        //
    
        pfn_quantize_row_q6_k_x8_AVX512(x, (block_q6_K *)q6rx, vec_size);
        pfn_quantize_row_q236_k_q8_k_x8_AVX512(y, (block_q8_K *)q8ry, vec_size);

        //
        // Compute the dot products for q6_K/q8_K and q6_K_repack/q8_k_repack.
        //

        pfn_ggml_vec_dot_q6_K_q8_K_AVX512(vec_size, &sum0, 0, q6kx, 0, q8ky, 0, 1);
        pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512(vec_size, &sum1, 0, q6rx, 0, q8ry, 1, 1);

        //
        // Log vector q6_K/q8_K sum of products.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q6kx) {
        zfree(q6kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q6rx) {
        zfree(q6rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
test_dot_q8_0_q8_0_repack (
    uint32_t vec_size
    )

//
// Compute the accuracy of vec_dot_q8_0_repack/block_q8_0_repack quant formats
// versus vec_dot_q8_0_q8_0.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t nb;
    float sum0;
    float sum1;
    float * x;
    float * y;
    block_q8_0 * q80x;
    block_q8_0 * q80y;
    block_q8_0_repack * q8rx;
    block_q8_0_repack * q8ry;

    //
    // Announce accuracy test.
    //

    fprintf(logfile, "Running test_dot_q8_0_q8_0/q8_0_q8_0_repack accuracy test\n\n");

    //
    // Check the number of q8_0_repack/q8_0_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
        return;
    }

    //
    // Check if block sizes are correct.
    //

    if ((vec_size / QK8_0 * sizeof(block_q8_0)) != (vec_size / QK_K * sizeof(block_q8_0_repack))) {
        fprintf(logfile, "quant block sizes do not match\n");
        return;
    }

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q80x = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    q80y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    q8rx = zalloc(vec_size / QK_K * sizeof(block_q8_0_repack));
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_0_repack));
    if (!x || !y || !q80x || !q80y || !q8rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Compute q8_0/q8_0 and q8_0_repack/q8_0_repack dot products on the same input
    // and compare the results.
    //

    uint32_t iter = 4096;
    int32_t accuracy = 8;
    uint32_t mismatch = 0;

    fprintf(logfile, "q8_0/q8_0 and q8_0_repack/q8_0_repack comparision\n");
    fprintf(logfile,
            "vector size %u, iterations %u, accuracy %d\n\n",
             vec_size,
             iter,
             accuracy);

    for (j = 0; j < iter; j += 1) {

        //
        // Fill the x and y vectors with random filtered values converted to float.
        //
    
        for (i = 0; i < vec_size; i += 1) {
            x[i] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }

        //
        // Quantize the x and y vectors for q8_0 and q8_0 quants. These are used for the
        // comparison values.
        //
        // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
        //
    
        pfn_quantize_row_q8_0_AVX512(x, q80x, vec_size);
        pfn_quantize_row_q8_0_AVX512(y, q80y, vec_size);

        //
        // Quantize the x and y vectors for q4_0_repack and q8_0_repack quants.
        //

        pfn_quantize_row_q8_0_AVX512(x, (block_q8_0 *)q8rx, vec_size);
        pfn_quantize_row_q8_0_AVX512(y, (block_q8_0 *)q8ry, vec_size);

        pfn_make_q8_0_repack_quant_AVX512(vec_size, q8rx, (block_q8_0 *)q8rx);
        pfn_make_q8_0_repack_quant_AVX512(vec_size, q8ry, (block_q8_0 *)q8ry);

        //
        // Compute the dot products for q8_0/q8_0 and q8_0_repack/q8_0_repack.
        //

        pfn_ggml_vec_dot_q8_0_q8_0_AVX512(vec_size, &sum0, 0, q80x, 0, q80y, 0, 1);
        pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512(vec_size, &sum1, 0, q8rx, 0, q8ry, 1, 1);

        //
        // Log vector q8_0/q8_0 sum of products.
        //

        int32_t isum0 = *(int32_t *)&sum0;
        int32_t isum1 = *(int32_t *)&sum1;

        //
        // Check if the resultant values are within the specified accuracy.
        //

        if (abs(isum0 - isum1) > accuracy) {
            mismatch += 1;
            fprintf(logfile, "mismatch sum0 %08x sum1 %08x ", isum0, isum1);
            fprintf(logfile, "%8.2f %8.2f\n\n", sum0, sum1);
        }
    }

    fprintf(logfile, "number of mismatches %u\n\n", mismatch);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q80x) {
        zfree(q80x);
    }

    if (q80y) {
        zfree(q80y);
    }

    if (q8rx) {
        zfree(q8rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
vec_dot_q2_k_q8_k_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q2_K_repack/block_q8_K_repack quant formats.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t k;
    uint32_t m;
    uint32_t nb;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q2_K * q2kx;
    block_q8_K * q8ky;
    block_q2_K_repack * q2rx;
    block_q8_K_repack * q8ry;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_k_q8_k/q2_k_q8_k_repack performance tests\n\n");

    //
    // Check the number of q2_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q2kx = zalloc(vec_size / QK_K * sizeof(block_q2_K) * mat_size);
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K) * mat_size);
    q2rx = zalloc(vec_size / QK_K * sizeof(block_q2_K_repack) * mat_size);
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack) * mat_size);
    if (!sum || !x || !y || !q2kx || !q8ky || !q2rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;
    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q2_0 and q8_0 quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q2_K_AVX512(&x[i * x_stride], &q2kx[i * nb], vec_size);
        pfn_quantize_row_q8_K_AVX512(&y[i * y_stride], &q8ky[i * nb], vec_size);
    }

    //
    // log q2_k/q8_k quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q2_k_quant_data(vec_size, (void *)&q2kx[i * nb], true);
        log_q8_k_quant_data(vec_size, (void *)&q8ky[i * nb], true);
    }

    //
    // Quantize the x and y vectors for q2_K_repack and q8_K_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q2_k_x8_AVX512(&x[i * x_stride],
                                        (block_q2_K *)&q2rx[i * nb],
                                        vec_size);

        pfn_quantize_row_q236_k_q8_k_x8_AVX512(&y[i * y_stride],
                                               (block_q8_K *)&q8ry[i * nb],
                                               vec_size);
    }

    //
    // log the q2_k and q8_k repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q2_k_quant_data(vec_size, (void *)&q2rx[i * nb], false);
        log_q8_k_quant_data(vec_size, (void *)&q8ry[i * nb], false);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_k_q8_k performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q2_k_q8_k = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q2_K_q8_K_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q2kx[m * nb],
                                                      0,
                                                      &q8ky[k * nb],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q2_k_q8_k) {
            best_time_q2_k_q8_k = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q2_k/q8_k vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_k_q8_k_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q2_k_q8_k_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q2rx,
                                               0,
                                               q8ry,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q2_k_q8_k_repack) {
            best_time_q2_k_q8_k_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q2_k_repack/q8_K_repack vector dot sum of products");

    //
    // Report percent increase from q2_k_q8_k (AVX512) to q2_k_q8_K_repack (AVX512).
    //

    log_increase_x("vec_dot_q2_k_q8_k_repack:",
                   best_time_q2_k_q8_k,
                   "AVX512",
                   best_time_q2_k_q8_k_repack,
                   "repack AVX512",
                   iter_vec_dot_q2_K_q8_K);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q2kx) {
        zfree(q2kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q2rx) {
        zfree(q2rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
vec_dot_q3_k_q8_k_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q3_K_repack/block_q8_K_repack quant formats.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint32_t i;
    uint64_t j;
    uint64_t k;
    uint64_t m;
    uint32_t nb;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q3_K * q3kx;
    block_q8_K * q8ky;
    block_q3_K_repack * q3rx;
    block_q8_K_repack * q8ry;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q3_k_q8_k/q3_k_q8_k_repack performance tests\n\n");

    //
    // Check the number of q3_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q3kx = zalloc(vec_size / QK_K * sizeof(block_q3_K) * mat_size);
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K) * mat_size);
    q3rx = zalloc(vec_size / QK_K * sizeof(block_q3_K_repack) * mat_size);
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack) * mat_size);
    if (!sum || !x || !y || !q3kx || !q8ky || !q3rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;
    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q3_0 and q8_0 quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q3_K_AVX512(&x[i * x_stride], &q3kx[i * nb], vec_size);
        pfn_quantize_row_q8_K_AVX512(&y[i * y_stride], &q8ky[i * nb], vec_size);
    }

    //
    // log q3_k/q8_k quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q3_k_quant_data(vec_size, (void *)&q3kx[i * nb], true);
        log_q8_k_quant_data(vec_size, (void *)&q8ky[i * nb], true);
    }

    //
    // Quantize the x and y vectors for q3_K_repack and q8_K_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q3_k_x8_AVX512(&x[i * x_stride],
                                        (block_q3_K *)&q3rx[i * nb],
                                        vec_size);

        pfn_quantize_row_q236_k_q8_k_x8_AVX512(&y[i * y_stride],
                                               (block_q8_K *)&q8ry[i * nb],
                                               vec_size);
    }

    //
    // log the q3_k and q8_k repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q3_k_quant_data(vec_size, (void *)&q3rx[i * nb], false);
        log_q8_k_quant_data(vec_size, (void *)&q8ry[i * nb], false);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q3_k_q8_k performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q3_k_q8_k = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q3_K_q8_K_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q3kx[m * nb],
                                                      0,
                                                      &q8ky[k * nb],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q3_k_q8_k) {
            best_time_q3_k_q8_k = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q3_k/q8_k vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q3_k_q8_k_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q3_k_q8_k_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q3_K_q8_K; i += 1) {
            pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q3rx,
                                               0,
                                               q8ry,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q3_k_q8_k_repack) {
            best_time_q3_k_q8_k_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q3_k_repack/q8_K_repack vector dot sum of products");

    //
    // Report percent increase from q3_k_q8_k (AVX2) to q3_k_q8_K_repack (AVX512).
    //

    log_increase_x("vec_dot_q3_k_q8_k_repack:",
                   best_time_q3_k_q8_k,
                   "AVX512",
                   best_time_q3_k_q8_k_repack,
                   "repack AVX512",
                   iter_vec_dot_q3_K_q8_K);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q3kx) {
        zfree(q3kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q3rx) {
        zfree(q3rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
vec_dot_q4_k_q8_k_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q4_K_repack/block_q8_K_repack quant formats.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t m;
    uint64_t nb;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q4_K * q4kx;
    block_q8_K * q8ky;
    block_q4_K_repack * q4rx;
    block_q8_K_repack * q8ry;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_k_q8_k/q4_k_q8_k_repack performance tests\n\n");

    //
    // Check the number of q4_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);  
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q4kx = zalloc(vec_size / QK_K * sizeof(block_q4_K) * mat_size);
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K) * mat_size);
    q4rx = zalloc(vec_size / QK_K * sizeof(block_q4_K_repack) * mat_size);
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack) * mat_size);
    if (!sum || !x || !y || !q4kx || !q8ky || !q4rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;

    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q4_k and q8_k quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q4_K_AVX512(&x[i * x_stride], &q4kx[i * nb], vec_size);
        pfn_quantize_row_q8_K_AVX512(&y[i * y_stride], &q8ky[i * nb], vec_size);
    }

    //
    // log q4_k/q8_k quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q4_k_quant_data(vec_size, (void *)&q4kx[i * nb], true);
        log_q8_k_quant_data(vec_size, (void *)&q8ky[i * nb], true);
    }

    //
    // Quantize the x and y vectors for q4_K_repack and q8_K_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q4_k_x8_AVX512(&x[i * x_stride],
                                        (block_q4_K *)&q4rx[i * nb],
                                        vec_size);

        pfn_quantize_row_q4_k_q8_k_x8_AVX512(&y[i * y_stride],
                                             (block_q8_K *)&q8ry[i * nb],
                                             vec_size);
    }

    //
    // log the q4_k and q8_k repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q4_k_quant_data(vec_size, (void *)&q4rx[i * nb], false);
        log_q8_k_quant_data(vec_size, (void *)&q8ry[i * nb], false);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_k_q8_k performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q4_k_q8_k = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q4_K_q8_K_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q4kx[m * nb],
                                                      0,
                                                      &q8ky[k * nb],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q4_k_q8_k) {
            best_time_q4_k_q8_k = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q4_k/q8_k vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q4_k_q8_k_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q4_k_q8_k_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q4_K_q8_K; i += 1) {
            pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q4rx,
                                               0,
                                               q8ry,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q4_k_q8_k_repack) {
            best_time_q4_k_q8_k_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q4_k_repack/q8_K_repack vector dot sum of products");

    //
    // Report percent increase from q4_k_q8_k (AVX512) to q4_k_q8_K_repack (AVX512).
    //

    log_increase_x("vec_dot_q4_k_q8_k_repack:",
                   best_time_q4_k_q8_k,
                   "AVX512",
                   best_time_q4_k_q8_k_repack,
                   "repack AVX512",
                   iter_vec_dot_q4_K_q8_K);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q4kx) {
        zfree(q4kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q4rx) {
        zfree(q4rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
vec_dot_q6_k_q8_k_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q6_K_repack/block_q8_K_repack quant formats.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint32_t i;
    uint64_t j;
    uint64_t k;
    uint64_t m;
    uint32_t nb;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q6_K * q6kx;
    block_q8_K * q8ky;
    block_q6_K_repack * q6rx;
    block_q8_K_repack * q8ry;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q6_k_q8_k/q6_k_q8_k_repack performance tests\n\n");

    //
    // Check the number of q6_k_repack/q8_k_repack quant blocks that will be generated.
    //

    nb = vec_size / QK_K;
    if (vec_size % QK_K) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q6kx = zalloc(vec_size / QK_K * sizeof(block_q6_K) * mat_size);
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_K) * mat_size);
    q6rx = zalloc(vec_size / QK_K * sizeof(block_q6_K_repack) * mat_size);
    q8ry = zalloc(vec_size / QK_K * sizeof(block_q8_K_repack) * mat_size);
    if (!sum || !x || !y || !q6kx || !q8ky || !q6rx || !q8ry) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;

    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q6_k and q8_k quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q6_K_AVX512(&x[i * x_stride], &q6kx[i * nb], vec_size);
        pfn_quantize_row_q8_K_AVX512(&y[i * y_stride], &q8ky[i * nb], vec_size);
    }

    //
    // log q6_k/q8_k quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q6_k_quant_data(vec_size, (void *)&q6kx[i * nb], true);
        log_q8_k_quant_data(vec_size, (void *)&q8ky[i * nb], true);
    }

    //
    // Quantize the x and y vectors for q6_K_repack and q8_K_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q6_k_x8_AVX512(&x[i * x_stride],
                                        (block_q6_K *)&q6rx[i * nb],
                                        vec_size);

        pfn_quantize_row_q236_k_q8_k_x8_AVX512(&y[i * y_stride],
                                               (block_q8_K *)&q8ry[i * nb],
                                               vec_size);
    }

    //
    // log the q6_k and q8_k repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q6_k_quant_data(vec_size, (void *)&q6rx[i * nb], false);
        log_q8_k_quant_data(vec_size, (void *)&q8ry[i * nb], false);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q6_k_q8_k performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q6_k_q8_k = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q6_K_q8_K_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q6kx[m * nb],
                                                      0,
                                                      &q8ky[k * nb],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q6_k_q8_k) {
            best_time_q6_k_q8_k = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q6_k/q8_k vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q6_k_q8_k_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q6_k_q8_k_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q6_K_q8_K; i += 1) {
            pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q6rx,
                                               0,
                                               q8ry,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q6_k_q8_k_repack) {
            best_time_q6_k_q8_k_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q6_k_repack/q8_K_repack vector dot sum of products");

    //
    // Report percent increase from q6_k_q8_k (AVX512) to q6_k_q8_K_repack (AVX512).
    //

    log_increase_x("vec_dot_q6_k_q8_k_repack:",
                   best_time_q6_k_q8_k,
                   "AVX512",
                   best_time_q6_k_q8_k_repack,
                   "repack AVX512",
                   iter_vec_dot_q6_K_q8_K);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q6kx) {
        zfree(q6kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }

    if (q6rx) {
        zfree(q6rx);
    }

    if (q8ry) {
        zfree(q8ry);
    }
}

void
vec_dot_q8_0_q8_0_repack (
    uint32_t vec_size,
    uint32_t mat_size
    )

//
// Compute the performance of the block_q8_0_repack quant format.
//
// This test operates on a square matrix that is mat_size by mat_size.
//

{

    uint32_t i;
    uint64_t j;
    uint32_t k;
    uint32_t m;
    uint32_t nbr;
    uint32_t nb0;
    float * sum;
    float * x;
    uint64_t x_stride;
    float * y;
    uint64_t y_stride;
    block_q8_0 * q80x;
    block_q8_0 * q80y;
    block_q8_0_repack * q8kx;
    block_q8_0_repack * q8ky;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0/q8_0_q8_0_repack performance tests\n\n");

    //
    // Check the number of q8_0_repack quant blocks that will be generated.
    //
    // N.B. This must be 0 mod QK_K.
    //

    nbr = vec_size / QK_K;
    nb0 = vec_size / QK8_0;
    if ((vec_size % QK8_0) || (vec_size % QK_K)) {
        fprintf(logfile, "  the number of vector elements is not 0 mod QK_K/QK8_0\n");
    }

    //
    // Allocate vectors of the specified size.
    //

    sum = zalloc(mat_size * sizeof(float) * mat_size);
    x = zalloc(vec_size * sizeof(float) * mat_size);
    y = zalloc(vec_size * sizeof(float) * mat_size);
    q80x = zalloc(vec_size / QK8_0 * sizeof(block_q8_0) * mat_size);
    q80y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0) * mat_size);
    q8kx = zalloc(vec_size / QK_K * sizeof(block_q8_0_repack) * mat_size);
    q8ky = zalloc(vec_size / QK_K * sizeof(block_q8_0_repack) * mat_size);
    if (!sum || !x || !y || !q80x || !q80y || !q8kx || !q8ky) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //

    x_stride = vec_size;
    y_stride = vec_size;

    for (i = 0; i < mat_size; i += 1) {
        for (j = 0; j < vec_size; j += 1) {
            x[i * x_stride + j] = genx_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_X;
            y[i * y_stride + j] = geny_float_value(FLOAT_FILTER_Q8) * FLOAT_FRACTION_Y;
        }
    }

    //
    // Log generated x and y data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_float_data(vec_size, (void *)&x[i * x_stride], "generated x vector:");
        log_float_data(vec_size, (void *)&y[i * y_stride], "generated y vector:");
    }

    //
    // Quantize the x and y vectors for q8_0 and q8_0 quants. These are used for the
    // comparison values.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q8_0_AVX512(&x[i * x_stride], &q80x[i * nb0], vec_size);
        pfn_quantize_row_q8_0_AVX512(&y[i * y_stride], &q80y[i * nb0], vec_size);
    }

    //
    // log q8_0 quant data.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q8_0_quant_data(vec_size, (void *)&q80x[i * nb0]);
        log_q8_0_quant_data(vec_size, (void *)&q80y[i * nb0]);
    }

    //
    // Quantize the x and y vectors for the q8_0_repack quants.
    //

    for (i = 0; i < mat_size; i += 1) {
        pfn_quantize_row_q8_0_x8_AVX512(&x[i * x_stride],
                                        (block_q8_0 *)&q8kx[i * nbr],
                                        vec_size);

        pfn_quantize_row_q8_0_x8_AVX512(&y[i * y_stride],
                                        (block_q8_0 *)&q8ky[i * nbr],
                                        vec_size);
    }

    //
    // log the q8_0 repack quant blocks.
    //

    for (i = 0; i < mat_size; i += 1) {
        log_q8_0_repack_quant_data(vec_size, (void *)&q8kx[i * nbr]);
        log_q8_0_repack_quant_data(vec_size, (void *)&q8ky[i * nbr]);
    }

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q8_0 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0; i += 1) {
            for (k = 0; k < mat_size; k += 1) {
                for (m = 0; m < mat_size; m += 1) {
                    pfn_ggml_vec_dot_q8_0_q8_0_AVX512(vec_size,
                                                      &sum[k * mat_size + m],
                                                      sizeof(float),
                                                      &q80x[m * nb0],
                                                      0,
                                                      &q80y[k * nb0],
                                                      1,
                                                      1);
                }
            }
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q8_0) {
            best_time_q8_0 = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q8_0/q8_0 vector dot sum of products");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0_repack performance test\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_q8_0_repack = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0_repack; i += 1) {
            pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512(vec_size,
                                               sum,
                                               sizeof(float) * mat_size,
                                               q8kx,
                                               0,
                                               q8ky,
                                               mat_size,
                                               mat_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_q8_0_repack) {
            best_time_q8_0_repack = total_time;
        }
    }
 
    //
    // Log vector sum of products.
    //

    log_float_data(mat_size * mat_size,
                   (uint32_t *)sum,
                   "q8_0_repack/q8_0_repack vector dot sum of products");

    //
    // Report percent increase from q8_0 (AVX512) to q8_0_repack (AVX512).
    //

    log_increase_x("vec_dot_q8_0_q8_0_repack:",
                   best_time_q8_0,
                   "AVX512",
                   best_time_q8_0_repack,
                   "repack AVX512",
                   iter_vec_dot_q8_0_q8_0);

exit:
    if (sum) {
        zfree(sum);
    }

    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q80x) {
        zfree(q80x);
    }

    if (q80y) {
        zfree(q80y);
    }

    if (q8kx) {
        zfree(q8kx);
    }

    if (q8ky) {
        zfree(q8ky);
    }
}

void
vec_quantize_q8_k_dot_q4_k_q8_k (
    uint32_t vec_size
    )

//
// Compute the combined performance of quantize_q8_k and vec_dot_q4_k_q8_k.
//

{

    uint32_t i;
    uint32_t j;
    float * q4x;
    float * q8x;
    block_q4_K * q4b;
    block_q8_K * q8b;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k_dot_q4_k_q8_k performance test for AVX2/AVX512.\n\n");

    //
    // Allocate vectors of the required size.
    //

    q4x = zalloc(vec_size * sizeof(float));
    q8x = zalloc(vec_size * sizeof(float));
    q4b = zalloc(vec_size / QK_K * sizeof(block_q4_K));
    q8b = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!q4x || !q8x || !q4b || !q8b) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the q4x and q8x vectors with random filtered values that are positive and
    // negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        q4x[i] = genx_float_value(FLOAT_FILTER) * FLOAT_FRACTION_X;
        q8x[i] = genx_float_value(FLOAT_FILTER) * FLOAT_FRACTION_X;
    }

    //
    // Log float data.
    //

    log_float_data(vec_size, (void *)q4x, "q4x vector input data");
    log_float_data(vec_size, (void *)q8x, "q8x vector input data");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k_dot_q4_k_q8_k performance test for AVX2\n\n");

    //
    // Quantize and log q4x and q8x data.
    //

    pfn_quantize_row_q4_K_AVX2(q4x, q4b, vec_size);
    log_q4_k_quant_data(vec_size, (void *)q4b, true);

    pfn_quantize_row_q8_K_AVX2(q8x, q8b, vec_size);
    log_q8_k_quant_data(vec_size, (void *)q8b, true);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do quantize q8_k followed by vector dot q4_k_q8_k.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k_dot_q4_k_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX2(q8x, q8b, vec_size);
            pfn_ggml_vec_dot_q4_K_q8_K_AVX2(vec_size, &sum, 0, q4b, 0, q8b, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k_dot_q4_k_q8_k performance test for AVX512\n\n");

    //
    // Quantize and log q4x and q8x data.
    //

    pfn_quantize_row_q4_K_AVX512(q4x, q4b, vec_size);
    log_q4_k_quant_data(vec_size, (void *)q4b, true);

    pfn_quantize_row_q8_K_AVX512(q8x, q8b, vec_size);
    log_q8_k_quant_data(vec_size, (void *)q8b, true);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do quantize q8_k followed by vector dot q4_k_q8_k.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k_dot_q4_k_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX512(q8x, q8b, vec_size);
            pfn_ggml_vec_dot_q4_K_q8_K_AVX512(vec_size, &sum, 0, q4b, 0, q8b, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("quantize_q8_k_dot_q4_k_q8_k:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_quantize_q8_k_dot_q4_k_q8_k);

exit:
    if (q4x) {
        zfree(q4x);
    }

    if (q8x) {
        zfree(q8x);
    }

    if (q4b) {
        zfree(q4b);
    }

    if (q8b) {
        zfree(q8b);
    }

    return;
}

void
vec_quantize_q8_0_dot_q4_0_q8_0 (
    uint32_t vec_size
    )

//
// Compute the combined performance of quantize_q8_0 and vec_dot_q4_0_q8_0.
//

{

    uint32_t i;
    uint32_t j;
    float * q4x;
    float * q8x;
    block_q4_0 * q4b;
    block_q8_0 * q8b;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0_dot_q4_0_q8_0 performance test for AVX2/AVX512.\n\n");

    //
    // Allocate vectors of the required size.
    //

    q4x = zalloc(vec_size * sizeof(float));
    q8x = zalloc(vec_size * sizeof(float));
    q4b = zalloc(vec_size / QK4_0 * sizeof(block_q4_0));
    q8b = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    if (!q4x || !q8x || !q4b || !q8b) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the q4x and q8x vectors with random filtered values that are positive and
    // negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        q4x[i] = genx_float_value(FLOAT_FILTER) * FLOAT_FRACTION_X;
        q8x[i] = genx_float_value(FLOAT_FILTER) * FLOAT_FRACTION_X;
    }

    //
    // Log float data.
    //

    log_float_data(vec_size, (void *)q4x, "q4x vector input data");
    log_float_data(vec_size, (void *)q8x, "q8x vector input data");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0_dot_q4_0_q8_0 performance test for AVX2\n\n");

    //
    // Quantize and log q4x and q8x data.
    //

    pfn_quantize_row_q4_0_AVX2(q4x, q4b, vec_size);
    log_q4_0_quant_data(vec_size, (void *)q4b);

    pfn_quantize_row_q8_0_AVX2(q8x, q8b, vec_size);
    log_q8_0_quant_data(vec_size, (void *)q8b);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do quantize q8_0 followed by vector dot q0_k_q8_0.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_0_dot_q4_0_q8_0; i += 1) {
            pfn_quantize_row_q8_0_AVX2(q8x, q8b, vec_size);
            pfn_ggml_vec_dot_q4_0_q8_0_AVX2(vec_size, &sum, 0, q4b, 0, q8b, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_0_dot_q4_0_q8_0 performance test for AVX512\n\n");

    //
    // Quantize and log q4x and q8x data.
    //

    pfn_quantize_row_q4_0_AVX512(q4x, q4b, vec_size);
    log_q4_0_quant_data(vec_size, (void *)q4b);

    pfn_quantize_row_q8_0_AVX512(q8x, q8b, vec_size);
    log_q8_0_quant_data(vec_size, (void *)q8b);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do quantize q8_0 followed by vector dot q0_k_q8_0.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_0_dot_q4_0_q8_0; i += 1) {
            pfn_quantize_row_q8_0_AVX512(q8x, q8b, vec_size);
            pfn_ggml_vec_dot_q4_0_q8_0_AVX512(vec_size, &sum, 0, q4b, 0, q8b, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
    //
    // Log vector sum of products.
    //

    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("quantize_q8_0_dot_q4_0_q8_0:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_quantize_q8_0_dot_q4_0_q8_0);

exit:
    if (q4x) {
        zfree(q4x);
    }

    if (q8x) {
        zfree(q8x);
    }

    if (q4b) {
        zfree(q4b);
    }

    if (q8b) {
        zfree(q8b);
    }

    return;
}

void
vec_cosine_similarity_f32 (
    uint32_t vec_size
    )

//
// Compute the performance of the cosine similarity of two f32 vectors.
//
//  dot += x[i] * y[i]
//  denom_x += x[i] * x[i]
//  denom_y += y[i] * y[i]
//  result = dot / (float)((sqrt(denom_a) * sqrt(denom_b)))
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_f32 performance test for AVX2/AVX512\n\n");

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = genx_float_value(FLOAT_FILTER);
        y[i] = geny_float_value(FLOAT_FILTER); 
    }

    //
    // Log generated x and y data.
    //

    log_float_data(vec_size, (void *)x, "generated x vector:");
    log_float_data(vec_size, (void *)y, "generated y vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_f32 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the cosine similarity computation.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_similarity_f32; i += 1) {
            sum = pfn_ggml_cosine_similarity_f32_AVX2(vec_size, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log cosine similarity.
    //

    fprintf(logfile, "  cosine similarity %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the summation of the product of vector
        // elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_similarity_f32; i += 1) {
            sum = pfn_ggml_cosine_similarity_f32_AVX512(vec_size, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log cosine similarity.
    //

    fprintf(logfile, "  cosine similarity %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_cosine_similarity_f32:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_similarity_f32);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_cosine_similarity_bf16 (
    uint32_t vec_size
    )

//
// Compute the performance of the cosine similarity of two bf16 vectors.
//
//  dot += x[i] * y[i]
//  denom_x += x[i] * x[i]
//  denom_y += y[i] * y[i]
//  result = dot / (float)((sqrt(denom_a) * sqrt(denom_b)))
//

{

    uint32_t i;
    uint32_t j;
    ggml_bf16_t * x;
    ggml_bf16_t * y;
    float sum = 0.f;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_bf16 performance test for AVX2/AVX512\n\n");

    //
    // Allocate bf16 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_bf16_t));
    y = zalloc(vec_size * sizeof(ggml_bf16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_fp32_to_bf16(genx_float_value(FLOAT_FILTER));
        y[i] = convert_fp32_to_bf16(geny_float_value(FLOAT_FILTER));
    }

    //
    // Log generated x (bf16) and y (bf16) data.
    //

    log_raw_data(vec_size / 2, (void *)x, "generated x (bf16) vector:");
    log_raw_data(vec_size / 2, (void *)y, "generated y (bf16) vector:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_bf16 performance test for AVX2\n\n");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the cosine similarity computation.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_similarity_bf16; i += 1) {
            sum = pfn_ggml_cosine_similarity_bf16_AVX2(vec_size, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Log cosine similarity.
    //

    fprintf(logfile, "  cosine similarity %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cosine_similarity_bf16 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do the cosine similarity computation.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_similarity_bf16; i += 1) {
            sum = pfn_ggml_cosine_similarity_bf16_AVX512(vec_size, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Log cosine similarity.
    //

    fprintf(logfile, "  cosine similarity %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //

    log_increase("vec_cosine_similarity_bf16:",
                 best_time_AVX2,
                 best_time_AVX512,
                 iter_vec_similarity_bf16);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

int
main (
    int argc,
    char *argv[]
    )
{

    int rc = -1;

    //
    // Parse command switches.
    //

    while (argc > 1) {
        if (!_stricmp(argv[1], "-l")) {
            log_data = TRUE;
            vector_size = VECTOR_SIZE_1024u;

        } else if (!_stricmp(argv[1], "-s")) {
            show_improve = TRUE;

        } else {
            printf("invalid switch %s ignored, the only valid switches are -l and -s\n", argv[1]);
        }
        
        argc -= 1;
        argv += 1;
    }

    //
    // Initialize function pointers
    //

    HMODULE dll_AVX2 = NULL;
    HMODULE dll_AVX512 = NULL;

#ifdef __GEN_ZA_VERSION__

    dll_AVX2 = LoadLibraryA("za-ggml-avx2.dll");
    if (dll_AVX2 == NULL) {
        printf("failed to load library for 'za-ggml-avx2.dll'\n");
        goto exit;
    }

    dll_AVX512 = LoadLibraryA("za-ggml-avx512.dll");
    if (dll_AVX512 == NULL) {
        printf("failed to load library for 'za-ggml-avx512.dll'\n");
        goto exit;
    }

#else

    dll_AVX2 = LoadLibraryA("zo-ggml-avx2.dll");
    if (dll_AVX2 == NULL) {
        printf("failed to load library for 'zo-ggml-avx2.dll'\n");
        goto exit;
    }

    dll_AVX512 = LoadLibraryA("zo-ggml-avx512.dll");
    if (dll_AVX512 == NULL) {
        printf("failed to load library for 'zo-ggml-avx512.dll'\n");
        goto exit;
    }

#endif // __GEN_ZA_VERSION__

    pfn_ggml_disable_core_parking_AVX2 = (PFN_ggml_disable_core_parking)(GetProcAddress(dll_AVX2, "ggml_disable_core_parking"));
    if (!pfn_ggml_disable_core_parking_AVX2) {
        printf("failed to get proc address of pfn_ggml_disable_core_parking_AVX2\n");
        goto exit;
    }

    pfn_ggml_init_tables_AVX2 = (PFN_ggml_init_tables)(GetProcAddress(dll_AVX2, "ggml_init_tables"));
    if (!pfn_ggml_init_tables_AVX2) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX2\n");
        goto exit;
    }

    pfn_ggml_init_tables_AVX512 = (PFN_ggml_init_tables)(GetProcAddress(dll_AVX512, "ggml_init_tables"));
    if (!pfn_ggml_init_tables_AVX512) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX512\n");
        goto exit;
    }

    pfn_ggml_time_init_AVX2 = (PFN_ggml_time_init)(GetProcAddress(dll_AVX2, "ggml_time_init"));
    if (!pfn_ggml_time_init_AVX2) {
        printf("failed to get proc address of pfn_ggml_init_time_AVX2\n");
        goto exit;
    }

    pfn_ggml_time_init_AVX512 = (PFN_ggml_time_init)(GetProcAddress(dll_AVX512, "ggml_time_init"));
    if (!pfn_ggml_time_init_AVX512) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX512\n");
        goto exit;
    }

    pfn_ggml_time_us = (PFN_ggml_time_us)(GetProcAddress(dll_AVX2, "ggml_time_us"));
    if (!pfn_ggml_time_us) {
        printf("failed to get proc address of pfn_ggml_time_us\n");
        goto exit;
    }

    pfn_ggml_bf16_to_fp32_row_AVX2 = (PFN_ggml_bf16_to_fp32_row)(GetProcAddress(dll_AVX2, "ggml_bf16_to_fp32_row"));
    pfn_ggml_fp32_to_bf16_row_AVX2 = (PFN_ggml_fp32_to_bf16_row)(GetProcAddress(dll_AVX2, "ggml_fp32_to_bf16_row"));
    pfn_ggml_fp16_to_fp32_row_AVX2 = (PFN_ggml_fp16_to_fp32_row)(GetProcAddress(dll_AVX2, "ggml_fp16_to_fp32_row"));
    pfn_ggml_fp32_to_fp16_row_AVX2 = (PFN_ggml_fp32_to_fp16_row)(GetProcAddress(dll_AVX2, "ggml_fp32_to_fp16_row"));
    pfn_ggml_vec_abs_f32_AVX2 = (PFN_ggml_vec_abs_f32)(GetProcAddress(dll_AVX2, "ggml_vec_abs_f32"));
    pfn_ggml_vec_add_f32_AVX2 = (PFN_ggml_vec_add_f32)(GetProcAddress(dll_AVX2, "ggml_vec_add_f32"));
    pfn_ggml_vec_add1_f32_AVX2 = (PFN_ggml_vec_add1_f32)(GetProcAddress(dll_AVX2, "ggml_vec_add1_f32"));
    pfn_ggml_vec_acc_f32_AVX2 = (PFN_ggml_vec_acc_f32)(GetProcAddress(dll_AVX2, "ggml_vec_acc_f32"));
    pfn_ggml_vec_acc1_f32_AVX2 = (PFN_ggml_vec_acc1_f32)(GetProcAddress(dll_AVX2, "ggml_vec_acc1_f32"));
    pfn_ggml_vec_sub_f32_AVX2 = (PFN_ggml_vec_sub_f32)(GetProcAddress(dll_AVX2, "ggml_vec_sub_f32"));
    pfn_ggml_vec_set_f32_AVX2 = (PFN_ggml_vec_set_f32)(GetProcAddress(dll_AVX2, "ggml_vec_set_f32"));
    pfn_ggml_vec_cpy_f32_AVX2 = (PFN_ggml_vec_cpy_f32)(GetProcAddress(dll_AVX2, "ggml_vec_cpy_f32"));
    pfn_ggml_vec_neg_f32_AVX2 = (PFN_ggml_vec_neg_f32)(GetProcAddress(dll_AVX2, "ggml_vec_neg_f32"));
    pfn_ggml_vec_mul_f32_AVX2 = (PFN_ggml_vec_mul_f32)(GetProcAddress(dll_AVX2, "ggml_vec_mul_f32"));
    pfn_ggml_vec_mul1_f32_AVX2 = (PFN_ggml_vec_mul1_f32)(GetProcAddress(dll_AVX2, "ggml_vec_mul1_f32"));
    pfn_ggml_vec_div_f32_AVX2 = (PFN_ggml_vec_div_f32)(GetProcAddress(dll_AVX2, "ggml_vec_div_f32"));
    pfn_ggml_vec_sum_f32_AVX2 = (PFN_ggml_vec_sum_f32)(GetProcAddress(dll_AVX2, "ggml_vec_sum_f32"));
    pfn_ggml_vec_sumsq_f32_AVX2 = (PFN_ggml_vec_sumsq_f32)(GetProcAddress(dll_AVX2, "ggml_vec_sumsq_f32"));
    pfn_ggml_vec_sumsq_bf16_AVX2 = (PFN_ggml_vec_sumsq_bf16)(GetProcAddress(dll_AVX2, "ggml_vec_sumsq_bf16"));
    pfn_ggml_vec_max_f32_AVX2 = (PFN_ggml_vec_max_f32)(GetProcAddress(dll_AVX2, "ggml_vec_max_f32"));
    pfn_ggml_vec_scale_f16_AVX2 = (PFN_ggml_vec_scale_f16)(GetProcAddress(dll_AVX2, "ggml_vec_scale_f16"));
    pfn_ggml_vec_scale_f32_AVX2 = (PFN_ggml_vec_scale_f32)(GetProcAddress(dll_AVX2, "ggml_vec_scale_f32"));
    pfn_ggml_vec_sqrt_f32_AVX2 = (PFN_ggml_vec_sqrt_f32)(GetProcAddress(dll_AVX2, "ggml_vec_sqrt_f32"));
    pfn_ggml_vec_mad_f16_AVX2 = (PFN_ggml_vec_mad_f16)(GetProcAddress(dll_AVX2, "ggml_vec_mad_f16"));
    pfn_ggml_vec_mad_f32_AVX2 = (PFN_ggml_vec_mad_f32)(GetProcAddress(dll_AVX2, "ggml_vec_mad_f32"));
    pfn_ggml_vec_normsq_f32_AVX2 = (PFN_ggml_vec_normsq_f32)(GetProcAddress(dll_AVX2, "ggml_vec_normsq_f32"));
    pfn_ggml_vec_dot_bf16_AVX2 = (PFN_ggml_vec_dot_bf16)(GetProcAddress(dll_AVX2, "ggml_vec_dot_bf16"));
    pfn_ggml_vec_dot_f16_AVX2 = (PFN_ggml_vec_dot_f16)(GetProcAddress(dll_AVX2, "ggml_vec_dot_f16"));
    pfn_ggml_vec_dot_f32_AVX2 = (PFN_ggml_vec_dot_f32)(GetProcAddress(dll_AVX2, "ggml_vec_dot_f32"));
    pfn_ggml_vec_dot_bf16_f32_AVX2 = (PFN_ggml_vec_dot_bf16_f32)(GetProcAddress(dll_AVX2, "ggml_vec_dot_bf16_f32"));
    pfn_ggml_vec_dot_f16_f32_AVX2 = (PFN_ggml_vec_dot_f16_f32)(GetProcAddress(dll_AVX2, "ggml_vec_dot_f16_f32"));
    pfn_ggml_vec_dot_q2_K_q8_K_AVX2 = (PFN_ggml_vec_dot_q2_K_q8_K)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q2_K_q8_K"));
    pfn_ggml_vec_dot_q3_K_q8_K_AVX2 = (PFN_ggml_vec_dot_q3_K_q8_K)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q3_K_q8_K"));
    pfn_ggml_vec_dot_q4_K_q8_K_AVX2 = (PFN_ggml_vec_dot_q4_K_q8_K)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q4_K_q8_K"));
    pfn_ggml_vec_dot_q6_K_q8_K_AVX2 = (PFN_ggml_vec_dot_q6_K_q8_K)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q6_K_q8_K"));
    pfn_ggml_vec_dot_q4_0_q8_0_AVX2 = (PFN_ggml_vec_dot_q4_0_q8_0)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q4_0_q8_0"));
    pfn_ggml_vec_dot_q8_0_q8_0_AVX2 = (PFN_ggml_vec_dot_q8_0_q8_0)(GetProcAddress(dll_AVX2, "ggml_vec_dot_q8_0_q8_0"));
    pfn_ggml_cosine_similarity_f32_AVX2 = (PFN_ggml_cosine_similarity_f32)(GetProcAddress(dll_AVX2, "ggml_cosine_similarity_f32"));
    pfn_ggml_cosine_similarity_bf16_AVX2 = (PFN_ggml_cosine_similarity_bf16)(GetProcAddress(dll_AVX2, "ggml_cosine_similarity_bf16"));
    pfn_quantize_row_q2_K_AVX2 = (PFN_quantize_row_q2_K)(GetProcAddress(dll_AVX2, "quantize_row_q2_K"));
    pfn_dequantize_row_q2_K_AVX2 = (PFN_dequantize_row_q2_K)(GetProcAddress(dll_AVX2, "dequantize_row_q2_K"));
    pfn_quantize_row_q3_K_AVX2 = (PFN_quantize_row_q3_K)(GetProcAddress(dll_AVX2, "quantize_row_q3_K"));
    pfn_dequantize_row_q3_K_AVX2 = (PFN_dequantize_row_q3_K)(GetProcAddress(dll_AVX2, "dequantize_row_q3_K"));
    pfn_quantize_row_q4_K_AVX2 = (PFN_quantize_row_q4_K)(GetProcAddress(dll_AVX2, "quantize_row_q4_K"));
    pfn_dequantize_row_q4_K_AVX2 = (PFN_dequantize_row_q4_K)(GetProcAddress(dll_AVX2, "dequantize_row_q4_K"));
    pfn_quantize_row_q4_0_AVX2 = (PFN_quantize_row_q4_0)(GetProcAddress(dll_AVX2, "quantize_row_q4_0"));
    pfn_dequantize_row_q4_0_AVX2 = (PFN_dequantize_row_q4_0)(GetProcAddress(dll_AVX2, "dequantize_row_q4_0"));
    pfn_quantize_row_q6_K_AVX2 = (PFN_quantize_row_q6_K)(GetProcAddress(dll_AVX2, "quantize_row_q6_K"));
    pfn_dequantize_row_q6_K_AVX2 = (PFN_dequantize_row_q6_K)(GetProcAddress(dll_AVX2, "dequantize_row_q6_K"));
    pfn_quantize_row_q8_K_AVX2 = (PFN_quantize_row_q8_K)(GetProcAddress(dll_AVX2, "quantize_row_q8_K"));
    pfn_dequantize_row_q8_K_AVX2 = (PFN_dequantize_row_q8_K)(GetProcAddress(dll_AVX2, "dequantize_row_q8_K"));
    pfn_quantize_row_q8_0_AVX2 = (PFN_quantize_row_q8_0)(GetProcAddress(dll_AVX2, "quantize_row_q8_0"));
    pfn_dequantize_row_q8_0_AVX2 = (PFN_dequantize_row_q8_0)(GetProcAddress(dll_AVX2, "dequantize_row_q8_0"));
    pfn_ggml_vec_silu_f32_AVX2 = (PFN_ggml_vec_silu_f32)(GetProcAddress(dll_AVX2, "ggml_vec_silu_f32"));
    pfn_ggml_vec_soft_max_f32_AVX2 = (PFN_ggml_vec_soft_max_f32)(GetProcAddress(dll_AVX2, "ggml_vec_soft_max_f32"));

    if ((pfn_ggml_bf16_to_fp32_row_AVX2 == NULL) || 
        (pfn_ggml_fp32_to_bf16_row_AVX2 == NULL) ||
        (pfn_ggml_fp16_to_fp32_row_AVX2 == NULL) ||
        (pfn_ggml_fp32_to_fp16_row_AVX2 == NULL) ||
        (pfn_ggml_vec_abs_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_add_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_add1_f32_AVX2 == NULL) || 
        (pfn_ggml_vec_acc_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_acc1_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_sub_f32_AVX2 == NULL) || 
        (pfn_ggml_vec_set_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_cpy_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_neg_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_mul_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_mul1_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_div_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_sum_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_sumsq_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_sumsq_bf16_AVX2 == NULL) ||
        (pfn_ggml_vec_max_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_scale_f16_AVX2 == NULL) ||
        (pfn_ggml_vec_scale_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_sqrt_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_mad_f16_AVX2 == NULL) ||
        (pfn_ggml_vec_mad_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_normsq_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_bf16_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_f16_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_bf16_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_f16_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q2_K_q8_K_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q3_K_q8_K_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q4_K_q8_K_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q6_K_q8_K_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q4_0_q8_0_AVX2 == NULL) ||
        (pfn_ggml_vec_dot_q8_0_q8_0_AVX2 == NULL) ||
        (pfn_ggml_cosine_similarity_f32_AVX2 == NULL) ||
        (pfn_ggml_cosine_similarity_bf16_AVX2 == NULL) ||
        (pfn_quantize_row_q2_K_AVX2 == NULL) ||
        (pfn_dequantize_row_q2_K_AVX2 == NULL) ||
        (pfn_quantize_row_q3_K_AVX2 == NULL) ||
        (pfn_dequantize_row_q3_K_AVX2 == NULL) ||
        (pfn_quantize_row_q4_K_AVX2 == NULL) ||
        (pfn_dequantize_row_q4_K_AVX2 == NULL) ||
        (pfn_quantize_row_q4_0_AVX2 == NULL) ||
        (pfn_dequantize_row_q4_0_AVX2 == NULL) ||
        (pfn_quantize_row_q6_K_AVX2 == NULL) ||
        (pfn_dequantize_row_q6_K_AVX2 == NULL) ||
        (pfn_quantize_row_q8_K_AVX2 == NULL) ||
        (pfn_dequantize_row_q8_K_AVX2 == NULL) ||
        (pfn_quantize_row_q8_0_AVX2 == NULL) ||
        (pfn_dequantize_row_q8_0_AVX2 == NULL) ||
        (pfn_ggml_vec_silu_f32_AVX2 == NULL) ||
        (pfn_ggml_vec_soft_max_f32_AVX2 == NULL)) {

        printf("failed to get address for one of more AVX2 vector functions\n");
        goto exit;
    }

    pfn_ggml_bf16_to_fp32_row_AVX512 = (PFN_ggml_bf16_to_fp32_row)(GetProcAddress(dll_AVX512, "ggml_bf16_to_fp32_row"));
    pfn_ggml_fp32_to_bf16_row_AVX512 = (PFN_ggml_fp32_to_bf16_row)(GetProcAddress(dll_AVX512, "ggml_fp32_to_bf16_row"));
    pfn_ggml_fp16_to_fp32_row_AVX512 = (PFN_ggml_fp16_to_fp32_row)(GetProcAddress(dll_AVX512, "ggml_fp16_to_fp32_row"));
    pfn_ggml_fp32_to_fp16_row_AVX512 = (PFN_ggml_fp32_to_fp16_row)(GetProcAddress(dll_AVX512, "ggml_fp32_to_fp16_row"));
    pfn_ggml_vec_abs_f32_AVX512 = (PFN_ggml_vec_abs_f32)(GetProcAddress(dll_AVX512, "ggml_vec_abs_f32"));
    pfn_ggml_vec_add_f32_AVX512 = (PFN_ggml_vec_add_f32)(GetProcAddress(dll_AVX512, "ggml_vec_add_f32"));
    pfn_ggml_vec_add1_f32_AVX512 = (PFN_ggml_vec_add1_f32)(GetProcAddress(dll_AVX512, "ggml_vec_add1_f32"));
    pfn_ggml_vec_acc_f32_AVX512 = (PFN_ggml_vec_acc_f32)(GetProcAddress(dll_AVX512, "ggml_vec_acc_f32"));
    pfn_ggml_vec_acc1_f32_AVX512 = (PFN_ggml_vec_acc1_f32)(GetProcAddress(dll_AVX512, "ggml_vec_acc1_f32"));
    pfn_ggml_vec_sub_f32_AVX512 = (PFN_ggml_vec_sub_f32)(GetProcAddress(dll_AVX512, "ggml_vec_sub_f32"));
    pfn_ggml_vec_set_f32_AVX512 = (PFN_ggml_vec_set_f32)(GetProcAddress(dll_AVX512, "ggml_vec_set_f32"));
    pfn_ggml_vec_cpy_f32_AVX512 = (PFN_ggml_vec_cpy_f32)(GetProcAddress(dll_AVX512, "ggml_vec_cpy_f32"));
    pfn_ggml_vec_neg_f32_AVX512 = (PFN_ggml_vec_neg_f32)(GetProcAddress(dll_AVX512, "ggml_vec_neg_f32"));
    pfn_ggml_vec_mul_f32_AVX512 = (PFN_ggml_vec_mul_f32)(GetProcAddress(dll_AVX512, "ggml_vec_mul_f32"));
    pfn_ggml_vec_mul1_f32_AVX512 = (PFN_ggml_vec_mul1_f32)(GetProcAddress(dll_AVX512, "ggml_vec_mul1_f32"));
    pfn_ggml_vec_div_f32_AVX512 = (PFN_ggml_vec_div_f32)(GetProcAddress(dll_AVX512, "ggml_vec_div_f32"));
    pfn_ggml_vec_sum_f32_AVX512 = (PFN_ggml_vec_sum_f32)(GetProcAddress(dll_AVX512, "ggml_vec_sum_f32"));
    pfn_ggml_vec_sumsq_f32_AVX512 = (PFN_ggml_vec_sumsq_f32)(GetProcAddress(dll_AVX512, "ggml_vec_sumsq_f32"));
    pfn_ggml_vec_sumsq_bf16_AVX512 = (PFN_ggml_vec_sumsq_bf16)(GetProcAddress(dll_AVX512, "ggml_vec_sumsq_bf16"));
    pfn_ggml_vec_max_f32_AVX512 = (PFN_ggml_vec_max_f32)(GetProcAddress(dll_AVX512, "ggml_vec_max_f32"));
    pfn_ggml_vec_scale_f16_AVX512 = (PFN_ggml_vec_scale_f16)(GetProcAddress(dll_AVX512, "ggml_vec_scale_f16"));
    pfn_ggml_vec_scale_f32_AVX512 = (PFN_ggml_vec_scale_f32)(GetProcAddress(dll_AVX512, "ggml_vec_scale_f32"));
    pfn_ggml_vec_sqrt_f32_AVX512 = (PFN_ggml_vec_sqrt_f32)(GetProcAddress(dll_AVX512, "ggml_vec_sqrt_f32"));
    pfn_ggml_vec_mad_f16_AVX512 = (PFN_ggml_vec_mad_f16)(GetProcAddress(dll_AVX512, "ggml_vec_mad_f16"));
    pfn_ggml_vec_mad_f32_AVX512 = (PFN_ggml_vec_mad_f32)(GetProcAddress(dll_AVX512, "ggml_vec_mad_f32"));
    pfn_ggml_vec_normsq_f32_AVX512 = (PFN_ggml_vec_normsq_f32)(GetProcAddress(dll_AVX512, "ggml_vec_normsq_f32"));
    pfn_ggml_vec_dot_bf16_AVX512 = (PFN_ggml_vec_dot_bf16)(GetProcAddress(dll_AVX512, "ggml_vec_dot_bf16"));
    pfn_ggml_vec_dot_f16_AVX512 = (PFN_ggml_vec_dot_f16)(GetProcAddress(dll_AVX512, "ggml_vec_dot_f16"));
    pfn_ggml_vec_dot_f32_AVX512 = (PFN_ggml_vec_dot_f32)(GetProcAddress(dll_AVX512, "ggml_vec_dot_f32"));
    pfn_ggml_vec_dot_bf16_f32_AVX512 = (PFN_ggml_vec_dot_bf16_f32)(GetProcAddress(dll_AVX512, "ggml_vec_dot_bf16_f32"));
    pfn_ggml_vec_dot_f16_f32_AVX512 = (PFN_ggml_vec_dot_f16_f32)(GetProcAddress(dll_AVX512, "ggml_vec_dot_f16_f32"));
    pfn_ggml_vec_dot_q2_K_q8_K_AVX512 = (PFN_ggml_vec_dot_q2_K_q8_K)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q2_K_q8_K"));
    pfn_ggml_vec_dot_q3_K_q8_K_AVX512 = (PFN_ggml_vec_dot_q3_K_q8_K)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q3_K_q8_K"));
    pfn_ggml_vec_dot_q4_K_q8_K_AVX512 = (PFN_ggml_vec_dot_q4_K_q8_K)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q4_K_q8_K"));
    pfn_ggml_vec_dot_q6_K_q8_K_AVX512 = (PFN_ggml_vec_dot_q6_K_q8_K)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q6_K_q8_K"));
    pfn_ggml_vec_dot_q4_0_q8_0_AVX512 = (PFN_ggml_vec_dot_q4_0_q8_0)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q4_0_q8_0"));
    pfn_ggml_vec_dot_q8_0_q8_0_AVX512 = (PFN_ggml_vec_dot_q8_0_q8_0)(GetProcAddress(dll_AVX512, "ggml_vec_dot_q8_0_q8_0"));

#ifdef __GEN_ZA_VERSION__

    pfn_make_q4_0_repack_quant_AVX512 = (PFN_make_q4_0_repack_quant)(GetProcAddress(dll_AVX512, "make_q4_0_repack_quant"));
    pfn_make_q2_k_repack_quant_AVX512 = (PFN_make_q2_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q2_k_repack_quant"));
    pfn_make_q3_k_repack_quant_AVX512 = (PFN_make_q3_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q3_k_repack_quant"));
    pfn_make_q4_k_repack_quant_AVX512 = (PFN_make_q4_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q4_k_repack_quant"));
    pfn_make_q6_k_repack_quant_AVX512 = (PFN_make_q6_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q6_k_repack_quant"));
    pfn_make_q8_0_repack_quant_AVX512 = (PFN_make_q8_0_repack_quant)(GetProcAddress(dll_AVX512, "make_q8_0_repack_quant"));
    pfn_make_q236_k_q8_k_repack_quant_AVX512 = (PFN_make_q236_k_q8_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q236_k_q8_k_repack_quant"));
    pfn_make_q4_k_q8_k_repack_quant_AVX512 = (PFN_make_q4_k_q8_k_repack_quant)(GetProcAddress(dll_AVX512, "make_q4_k_q8_k_repack_quant"));
    pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512 = (PFN_xx_vec_dot_q4_0_q8_0_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q4_0_q8_0_x8"));
    pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512 = (PFN_xx_vec_dot_q2_k_q8_k_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q2_k_q8_k_x8"));
    pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512 = (PFN_xx_vec_dot_q3_k_q8_k_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q3_k_q8_k_x8"));
    pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512 = (PFN_xx_vec_dot_q4_k_q8_k_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q4_k_q8_k_x8"));
    pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512 = (PFN_xx_vec_dot_q6_k_q8_k_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q6_k_q8_k_x8"));
    pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512 = (PFN_xx_vec_dot_q8_0_q8_0_x8)(GetProcAddress(dll_AVX512, "xx_vec_dot_q8_0_q8_0_x8"));
    pfn_quantize_row_q4_0_x8_AVX512 = (PFN_quantize_row_q4_0_x8)(GetProcAddress(dll_AVX512, "quantize_row_q4_0_x8"));
    pfn_quantize_row_q2_k_x8_AVX512 = (PFN_quantize_row_q2_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q2_k_x8"));
    pfn_quantize_row_q3_k_x8_AVX512 = (PFN_quantize_row_q3_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q3_k_x8"));
    pfn_quantize_row_q4_k_x8_AVX512 = (PFN_quantize_row_q4_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q4_k_x8"));
    pfn_quantize_row_q6_k_x8_AVX512 = (PFN_quantize_row_q6_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q6_k_x8"));
    pfn_quantize_row_q236_k_q8_k_x8_AVX512 = (PFN_quantize_row_q236_k_q8_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q236_k_q8_k_x8"));
    pfn_quantize_row_q4_k_q8_k_x8_AVX512 = (PFN_quantize_row_q4_k_q8_k_x8)(GetProcAddress(dll_AVX512, "quantize_row_q4_k_q8_k_x8"));
    pfn_quantize_row_q8_0_x8_AVX512 = (PFN_quantize_row_q8_0_x8)(GetProcAddress(dll_AVX512, "quantize_row_q8_0_x8"));
    
#else

    pfn_make_q4_0_repack_quant_AVX512 = &make_q4_0_repack_quant;
    pfn_make_q2_k_repack_quant_AVX512 = &make_q2_k_repack_quant;
    pfn_make_q3_k_repack_quant_AVX512 = &make_q3_k_repack_quant;
    pfn_make_q4_k_repack_quant_AVX512 = &make_q4_k_repack_quant;
    pfn_make_q6_k_repack_quant_AVX512 = &make_q6_k_repack_quant;
    pfn_make_q8_0_repack_quant_AVX512 = &make_q8_0_repack_quant;
    pfn_make_q236_k_q8_k_repack_quant_AVX512 = &make_q236_k_q8_k_repack_quant;
    pfn_make_q4_k_q8_k_repack_quant_AVX512 = &make_q4_k_q8_k_repack_quant;
    pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512 = &xx_vec_dot_q4_0_q8_0_x8;
    pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512 = &xx_vec_dot_q2_k_q8_k_x8;
    pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512 = &xx_vec_dot_q3_k_q8_k_x8;
    pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512 = &xx_vec_dot_q4_k_q8_k_x8;
    pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512 = &xx_vec_dot_q6_k_q8_k_x8;
    pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512 = &xx_vec_dot_q8_0_q8_0_x8;
    pfn_quantize_row_q4_0_x8_AVX512 = &quantize_row_q4_0_x8;
    pfn_quantize_row_q2_k_x8_AVX512 = &quantize_row_q2_k_x8;
    pfn_quantize_row_q3_k_x8_AVX512 = &quantize_row_q3_k_x8;
    pfn_quantize_row_q4_k_x8_AVX512 = &quantize_row_q4_k_x8;
    pfn_quantize_row_q6_k_x8_AVX512 = &quantize_row_q6_k_x8;
    pfn_quantize_row_q236_k_q8_k_x8_AVX512 = &quantize_row_q236_k_q8_k_x8;
    pfn_quantize_row_q4_k_q8_k_x8_AVX512 = &quantize_row_q4_k_q8_k_x8;
    pfn_quantize_row_q8_0_x8_AVX512 = &quantize_row_q8_0_x8;

#endif // __GEN_ZA_VERSION__

    pfn_ggml_cosine_similarity_f32_AVX512 = (PFN_ggml_cosine_similarity_f32)(GetProcAddress(dll_AVX512, "ggml_cosine_similarity_f32"));
    pfn_ggml_cosine_similarity_bf16_AVX512 = (PFN_ggml_cosine_similarity_bf16)(GetProcAddress(dll_AVX512, "ggml_cosine_similarity_bf16"));
    pfn_quantize_row_q2_K_AVX512 = (PFN_quantize_row_q2_K)(GetProcAddress(dll_AVX512, "quantize_row_q2_K"));
    pfn_dequantize_row_q2_K_AVX512 = (PFN_dequantize_row_q2_K)(GetProcAddress(dll_AVX512, "dequantize_row_q2_K"));
    pfn_quantize_row_q3_K_AVX512 = (PFN_quantize_row_q3_K)(GetProcAddress(dll_AVX512, "quantize_row_q3_K"));
    pfn_dequantize_row_q3_K_AVX512 = (PFN_dequantize_row_q3_K)(GetProcAddress(dll_AVX512, "dequantize_row_q3_K"));
    pfn_quantize_row_q4_K_AVX512 = (PFN_quantize_row_q4_K)(GetProcAddress(dll_AVX512, "quantize_row_q4_K"));
    pfn_dequantize_row_q4_K_AVX512 = (PFN_dequantize_row_q4_K)(GetProcAddress(dll_AVX512, "dequantize_row_q4_K"));
    pfn_quantize_row_q4_0_AVX512 = (PFN_quantize_row_q4_0)(GetProcAddress(dll_AVX512, "quantize_row_q4_0"));
    pfn_dequantize_row_q4_0_AVX512 = (PFN_dequantize_row_q4_0)(GetProcAddress(dll_AVX512, "dequantize_row_q4_0"));
    pfn_quantize_row_q6_K_AVX512 = (PFN_quantize_row_q6_K)(GetProcAddress(dll_AVX512, "quantize_row_q6_K"));
    pfn_dequantize_row_q6_K_AVX512 = (PFN_dequantize_row_q6_K)(GetProcAddress(dll_AVX512, "dequantize_row_q6_K"));
    pfn_quantize_row_q8_K_AVX512 = (PFN_quantize_row_q8_K)(GetProcAddress(dll_AVX512, "quantize_row_q8_K"));
    pfn_dequantize_row_q8_K_AVX512 = (PFN_dequantize_row_q8_K)(GetProcAddress(dll_AVX512, "dequantize_row_q8_K"));
    pfn_quantize_row_q8_0_AVX512 = (PFN_quantize_row_q8_0)(GetProcAddress(dll_AVX512, "quantize_row_q8_0"));
    pfn_dequantize_row_q8_0_AVX512 = (PFN_dequantize_row_q8_0)(GetProcAddress(dll_AVX512, "dequantize_row_q8_0"));
    pfn_ggml_vec_silu_f32_AVX512 = (PFN_ggml_vec_silu_f32)(GetProcAddress(dll_AVX512, "ggml_vec_silu_f32"));
    pfn_ggml_vec_soft_max_f32_AVX512 = (PFN_ggml_vec_soft_max_f32)(GetProcAddress(dll_AVX512, "ggml_vec_soft_max_f32"));

    if ((pfn_ggml_bf16_to_fp32_row_AVX512 == NULL) ||
        (pfn_ggml_fp32_to_bf16_row_AVX512 == NULL) ||
        (pfn_ggml_fp16_to_fp32_row_AVX512 == NULL) ||
        (pfn_ggml_fp32_to_fp16_row_AVX512 == NULL) ||
        (pfn_ggml_vec_abs_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_add_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_add1_f32_AVX512 == NULL) || 
        (pfn_ggml_vec_acc_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_acc1_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_sub_f32_AVX512 == NULL) || 
        (pfn_ggml_vec_set_f32_AVX512 == NULL) || 
        (pfn_ggml_vec_cpy_f32_AVX512 == NULL) || 
        (pfn_ggml_vec_neg_f32_AVX512 == NULL) || 
        (pfn_ggml_vec_mul_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_mul1_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_div_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_sum_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_sumsq_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_sumsq_bf16_AVX512 == NULL) ||
        (pfn_ggml_vec_max_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_scale_f16_AVX512 == NULL) ||
        (pfn_ggml_vec_scale_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_sqrt_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_mad_f16_AVX512 == NULL) ||
        (pfn_ggml_vec_mad_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_normsq_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_bf16_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_f16_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_bf16_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_f16_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q2_K_q8_K_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q3_K_q8_K_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q4_K_q8_K_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q6_K_q8_K_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q4_0_q8_0_AVX512 == NULL) ||
        (pfn_ggml_vec_dot_q8_0_q8_0_AVX512 == NULL) ||
        (pfn_make_q4_0_repack_quant_AVX512 == NULL) ||
        (pfn_make_q2_k_repack_quant_AVX512 == NULL) ||
        (pfn_make_q3_k_repack_quant_AVX512 == NULL) ||
        (pfn_make_q4_k_repack_quant_AVX512 == NULL) ||
        (pfn_make_q6_k_repack_quant_AVX512 == NULL) ||
        (pfn_make_q8_0_repack_quant_AVX512 == NULL) ||
        (pfn_make_q236_k_q8_k_repack_quant_AVX512 == NULL) ||
        (pfn_make_q4_k_q8_k_repack_quant_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q4_0_q8_0_x8_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q2_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q3_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q4_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q6_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_xx_vec_dot_q8_0_q8_0_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q4_0_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q2_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q3_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q4_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q6_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q236_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q4_k_q8_k_x8_AVX512 == NULL) ||
        (pfn_quantize_row_q8_0_x8_AVX512 == NULL) ||
        (pfn_ggml_cosine_similarity_f32_AVX512 == NULL) ||
        (pfn_ggml_cosine_similarity_bf16_AVX512 == NULL) ||
        (pfn_quantize_row_q2_K_AVX512 == NULL) ||
        (pfn_dequantize_row_q2_K_AVX512 == NULL) ||
        (pfn_quantize_row_q3_K_AVX512 == NULL) ||
        (pfn_dequantize_row_q3_K_AVX512 == NULL) ||
        (pfn_quantize_row_q4_K_AVX512 == NULL) ||
        (pfn_dequantize_row_q4_K_AVX512 == NULL) ||
        (pfn_quantize_row_q4_0_AVX512 == NULL) ||
        (pfn_dequantize_row_q4_0_AVX512 == NULL) ||
        (pfn_quantize_row_q6_K_AVX512 == NULL) ||
        (pfn_dequantize_row_q6_K_AVX512 == NULL) ||
        (pfn_quantize_row_q8_K_AVX512 == NULL) ||
        (pfn_dequantize_row_q8_K_AVX512 == NULL) ||
        (pfn_quantize_row_q8_0_AVX512 == NULL) ||
        (pfn_dequantize_row_q8_0_AVX512 == NULL) ||
        (pfn_ggml_vec_silu_f32_AVX512 == NULL) ||
        (pfn_ggml_vec_soft_max_f32_AVX512 == NULL)) {

        printf("failed to get address for one or more AVX512 vector functions\n");
        goto exit;
    }

    //
    // Set default log file name.
    //

#ifdef __GEN_ZA_VERSION__

    char * filename = "perfavxza.log";

#else

    char * filename = "perfavxzo.log";

#endif // __GEN_ZA_VERSION__

    //
    // Open log file for write.
    //

    printf("log filename is %s\n", filename);

    logfile = fopen(filename, "w");
    if (!logfile) {
        printf("failed to open log file %s\n", filename);
        goto exit;
    }

    //
    // Set process affinity - ignore if failure.
    //

    if (!SetProcessAffinityMask(GetCurrentProcess(), 1ull << 8)) {
        printf("failed to set thread affinity ignored\n");
    }

#if 0
    uint64_t affinity;

    if (SetThreadAffinityMask(GetCurrentThread(), 1ull << 8)) {
        affinity = SetThreadAffinityMask(GetCurrentThread(), 1ull << 8);
        fprintf(logfile, "running with group afffinity %p\n", (void *)affinity);

    } else {
        printf("failed to set thread affinity ignored\n");
    }
#endif // #if 0

    //
    // Set thread priority - ignore if failure.
    //

    if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
        fprintf(logfile, "running at time critical priority\n");

    } else {
        printf("failed to set thread priority ignored\n");
    }

    fprintf(logfile, "\n");

    //
    // Disable core parking.
    //

    pfn_ggml_disable_core_parking_AVX2();

    //
    // Initialize ggml floating conversion tables.
    //
    // N.B. The timing code must be explicitly initialized for both libraries. The initialization
    //      of the ggml tables depends on this. 
    //

    pfn_ggml_time_init_AVX2();
    pfn_ggml_init_tables_AVX2();

    pfn_ggml_time_init_AVX512();
    pfn_ggml_init_tables_AVX512(); 

    //
    // Test quantization.
    //

    vec_test_quantize(vector_size);

    //
    // Test repack vector dot operation accuracy.
    //

    test_dot_q4_0_q8_0_repack(vector_size);

    test_dot_q2_k_q8_k_repack(vector_size);

    test_dot_q3_k_q8_k_repack(vector_size);

    test_dot_q4_k_q8_k_repack(vector_size);

    test_dot_q6_k_q8_k_repack(vector_size);

    test_dot_q8_0_q8_0_repack(vector_size);

    //
    // Compute the performance of various avx code paths.
    //

    vec_bf16_to_fp32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_fp32_to_bf16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_fp16_to_fp32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_fp32_to_fp16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_abs_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_add_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_add1_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_acc_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_acc1_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_sub_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_set_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_cpy_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_neg_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_mul_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_mul1_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_div_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_sum_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_sumsq_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_sumsq_bf16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_max_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_scale_f16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_scale_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_sqrt_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_mad_f16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_mad_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_normsq_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_silu_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_soft_max_f32(vector_size + VECTOR_SIZE_EXTENSION);

    dequantize_q2_k(vector_size);

    dequantize_q3_k(vector_size);

    dequantize_q4_k(vector_size);

    dequantize_q4_0(vector_size);

    dequantize_q6_k(vector_size);

    dequantize_q8_k(vector_size);

    dequantize_q8_0(vector_size);

    quantize_q8_k(vector_size);

    quantize_q8_0(vector_size);

    vec_dot_bf16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_dot_f16(vector_size + VECTOR_SIZE_EXTENSION);

    vec_dot_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_dot_bf16_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_dot_f16_f32(vector_size + VECTOR_SIZE_EXTENSION);

    //
    // Compute the performance of vector dot operations.
    //

    vec_dot_q2_K_q8_K(vector_size);

    vec_dot_q3_K_q8_K(vector_size);

    vec_dot_q4_K_q8_K(vector_size);

    vec_dot_q6_K_q8_K(vector_size);

    vec_dot_q4_0_q8_0(vector_size);

    vec_dot_q8_0_q8_0(vector_size);

    //
    // Compute the performance of repack vector dot operations.
    //

    vec_dot_q2_k_q8_k_repack(vector_size, 5);

    vec_dot_q3_k_q8_k_repack(vector_size, 5);

    vec_dot_q4_k_q8_k_repack(vector_size, 5);

    vec_dot_q6_k_q8_k_repack(vector_size, 5);

    vec_dot_q4_0_q8_0_repack(vector_size, 5);

    vec_dot_q8_0_q8_0_repack(vector_size, 5);

    //
    // Compute the performance of combinied quantize/vector dot operations.
    //

    vec_quantize_q8_k_dot_q4_k_q8_k(vector_size);

    vec_quantize_q8_0_dot_q4_0_q8_0(vector_size);

    //
    // Compute the performance of cosine similarity.
    //

    vec_cosine_similarity_f32(vector_size + VECTOR_SIZE_EXTENSION);

    vec_cosine_similarity_bf16(vector_size + VECTOR_SIZE_EXTENSION);

    rc = 0;

//
// Output the test improvement report.
//

    if (!log_data || show_improve) {

        fprintf(logfile, "Performance test improvement report - Vector Size %d\n\n", vector_size);
    
        for (uint32_t j = 0; j < improve_table_size; j += 1) {
            char * text = improve_table[j].text;
            int64_t best_time_first = improve_table[j].best_time_first;
            char * type_first = improve_table[j].type_first;
            int64_t best_time_second = improve_table[j].best_time_second;
            char * type_second = improve_table[j].type_second;
            int64_t iter = improve_table[j].iter;
    
            best_time_second = max(1, best_time_second);
            float multiple = (float)best_time_first / (float)best_time_second;
            fprintf(logfile, "%s %s is %5.2f x speed of %s\n",
                    text,
                    type_second,
                    multiple,
                    type_first);

            if (show_improve) {
                printf("%s %s is %5.2f x speed of %s\n",
                       text,
                       type_second,
                       multiple,
                       type_first);
            }
        
            fprintf(logfile, "  loop iterations %zd\n", iter);
            fprintf(logfile, "  best %s time %zdns\n", type_first, best_time_first);
            fprintf(logfile, "    time per iteration %6.2fus\n", (float)best_time_first * 1000. / (float)(iter));

            fprintf(logfile, "  best %s time %zdns\n", type_second, best_time_second);
            fprintf(logfile, "    time per iteration %6.2fus\n\n", (float)best_time_second * 1000. / (float)(iter));
    
        }
    }

    fprintf(logfile, "xrand count %d\n", xrand_count);

exit:
    if (dll_AVX2) {
        FreeLibrary(dll_AVX2);
    }

    if (dll_AVX512) {
        FreeLibrary(dll_AVX512);
    }

    if (logfile) {
        fclose(logfile);
    }

    return rc;
}
