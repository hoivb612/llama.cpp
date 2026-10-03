// mul_mat_linalg_f16.hlsl - MUL_MAT via SM 6.10 dx::linalg wave matrices.
//
// Requires the LinAlg preview toolchain (cs_6_10) and a driver advertising
// D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT. Uses wave-scope 16x16x16 matrices
// with F16 inputs and F32 accumulation.
//
// Both operands are staged through LDS as float16_t rather than loaded from
// the buffers directly. That costs a round trip but buys three things:
//   - src1 may be F32 or F16; the conversion happens during staging via
//     load_auto(), so this shader is not restricted to F16 activations.
//   - out-of-range rows/columns are zero-filled in LDS, so edge tiles need no
//     special case (Matrix::Load has no bounds checking).
//   - quantized weights dequantize into the same LDS tile.
//
// Tiling: the group's waves form a 2D grid over the output. Each wave owns
// LA_MT row tiles by LA_NT column tiles and reuses those staged fragments
// across LA_MT*LA_NT matrix accumulators. The best balance is vendor-specific:
// AMD favors fewer waves with more accumulators each, while NVIDIA needs lower
// per-wave register pressure.
//
// The tile shape is a compile-time parameter because throughput here is set by
// how much of the GPU the dispatch fills, not by per-group efficiency: a
// 576x512 output is only 72 groups at 64x64, which idles most of the CUs. The
// host picks the largest shape that still yields enough groups.
#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#error "mul_mat_linalg_f16 requires -D WAVE_SIZE"
#endif
#ifndef LA_NWAVE
#define LA_NWAVE 4
#endif
#ifndef LA_NT
#define LA_NT 4
#endif
// Row strips a single wave owns. Raising it keeps the group tile the same
// while giving each wave LA_MT*LA_NT accumulators for LA_MT+LA_NT staging
// loads, so the matrix pipe sees more work per LDS read.
#ifndef LA_MT
#define LA_MT 1
#endif
// Waves along N. The group's LA_NWAVE waves form an (LA_NWAVE/LA_WN) x LA_WN
// grid over the output tile, so a wave owns LA_MT*LA_NT accumulators for only
// LA_MT+LA_NT staging loads. Stacking waves along M alone (LA_WN=1) caps that
// ratio: reaching 16 accumulators per wave would need LA_MT=16 and a 256-row
// group tile, which no real shape has the rows for. Splitting both ways gets
// there at 128x128, which is what the Vulkan coopmat path uses.
#ifndef LA_WN
#define LA_WN 1
#endif
#define LA_WM (LA_NWAVE / LA_WN)
// 0 = both operands are float; 8 = the weight side is Q8_0 and is dequantised
// into the same staging tile, leaving the matrix path unchanged; 4 = the same
// for Q4_K.
#ifndef LA_QUANT
#define LA_QUANT 0
#endif
#ifndef LA_ALIGNED
#define LA_ALIGNED 0
#endif
// LA_FULL_TILE requires exact M/N/K tiles, K-contiguous inputs, aligned outer strides, and F32 output. Q8_0 requires F32 activations.
#ifndef LA_FULL_TILE
#define LA_FULL_TILE 0
#endif
// LA_DENSE_FIXED selects F16 x F32 inputs.
#ifndef LA_DENSE_FIXED
#define LA_DENSE_FIXED 0
#endif
#ifndef LA_SINGLE_BUFFER
#define LA_SINGLE_BUFFER 0
#endif
#ifndef LA_NATIVE16_LOADS
#define LA_NATIVE16_LOADS LA_ALIGNED
#endif
#ifndef LA_Q50_PACKED
#define LA_Q50_PACKED 0
#endif
#if LA_NATIVE16_LOADS && ((LA_QUANT != 8 && LA_QUANT != 50) || WAVE_SIZE != 64)
#error "LA_NATIVE16_LOADS requires wave64 Q8_0 or Q5_0 with even weight offsets and strides"
#endif
#if LA_Q50_PACKED && (LA_QUANT != 50 || !LA_NATIVE16_LOADS)
#error "LA_Q50_PACKED requires Q5_0 with native16 loads"
#endif

#if LA_QUANT != 0
#define QK8_0      32
#define Q8_0_BSIZE 34
#define QK5_0      32
#define Q5_0_BSIZE 22
#define QK4_0      32
#define Q4_0_BSIZE 18
#define QK4_1      32
#define Q4_1_BSIZE 20
#define QK5_1      32
#define Q5_1_BSIZE 24
#define QK4_NL      32
#define Q4_NL_BSIZE 18
#define QK_MXFP4    32
#define MXFP4_BSIZE 17
#define QK_K       256
#define Q4_K_BSIZE 144
#define Q5_K_BSIZE 176
#define Q6_K_BSIZE 210

// Native loads require even weight offsets and strides, not full output tiles.
float la_q8_scale(ByteAddressBuffer buf, uint byte_off) {
#if LA_NATIVE16_LOADS
    return f16_to_f32((uint)buf.Load<uint16_t>(byte_off));
#else
    const uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
#endif
}

uint la_q8_byte(ByteAddressBuffer buf, uint byte_off) {
    return (buf.Load(byte_off & ~3u) >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

uint la_q8_quads(ByteAddressBuffer buf, uint byte_off) {
#if LA_NATIVE16_LOADS
    const uint16_t2 words = buf.Load<uint16_t2>(byte_off);
    return (uint)words.x | ((uint)words.y << 16u);
#else
    const uint aligned = byte_off & ~3u;
    const uint shift   = (byte_off & 3u) * 8u;
    const uint lo = buf.Load(aligned);
    // the trailing word is addressed unconditionally: root SRVs are not
    // bounds checked and DXC speculates a guarded load. See GOTCHAS.md.
    const uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
#endif
}
#endif

#if LA_QUANT == 4 || LA_QUANT == 5
// The 6-bit packed per-sub-block scale and min, shared by Q4_K and Q5_K.
float2 la_k_scale_min(ByteAddressBuffer buf, uint sc_off, uint sb) {
    const bool lt4 = sb < 4u;
    const uint sj  = la_q8_byte(buf, sc_off + sb);
    const uint sj4 = la_q8_byte(buf, sc_off + sb + 4u);
    const uint sjm = la_q8_byte(buf, sc_off + (lt4 ? sb : sb - 4u));
    const uint sc  = lt4 ? (sj  & 0x3Fu) : ((sj4 & 0x0Fu) | ((sjm & 0xC0u) >> 2));
    const uint mb  = lt4 ? (sj4 & 0x3Fu) : ((sj4 >> 4)    | ((sj  & 0xC0u) >> 2));
    return float2((float)sc, (float)mb);
}
#endif

#ifndef LA_MMID
#define LA_MMID 0
#endif
#ifndef LA_MMID_BUCKET
#define LA_MMID_BUCKET 0
#endif
#if LA_FULL_TILE < 0 || LA_FULL_TILE > 1
#error "LA_FULL_TILE must be 0 or 1"
#endif
#if LA_SINGLE_BUFFER < 0 || LA_SINGLE_BUFFER > 1
#error "LA_SINGLE_BUFFER must be 0 or 1"
#endif
#if LA_FULL_TILE && LA_MMID
#error "LA_FULL_TILE does not support MUL_MAT_ID"
#endif
#if LA_FULL_TILE && LA_QUANT != 0 && LA_QUANT != 8
#error "LA_FULL_TILE supports dense float and Q8_0 weights"
#endif
#if LA_DENSE_FIXED < 0 || LA_DENSE_FIXED > 1
#error "LA_DENSE_FIXED must be 0 or 1"
#endif
#if LA_DENSE_FIXED && (LA_QUANT != 0 || !LA_FULL_TILE)
#error "LA_DENSE_FIXED requires dense float weights and LA_FULL_TILE"
#endif
#if LA_MMID_BUCKET && (!LA_MMID || (WAVE_SIZE != 64 && WAVE_SIZE != 32))
#error "LA_MMID_BUCKET requires the wave32 or wave64 MUL_MAT_ID kernel"
#endif
#if LA_ALIGNED && (LA_QUANT != 8 || LA_MMID || WAVE_SIZE != 64)
#error "LA_ALIGNED requires the wave64 Q8_0 MUL_MAT kernel"
#endif
// Experts the MoE path can route to. Sized to a groupshared counter each.
#ifndef LA_MMID_MAX_EXPERT
#define LA_MMID_MAX_EXPERT 256
#endif

#define TILE 16
// K depth of one staging step. A multiple of TILE; the multiply loop below
// walks BK/TILE fragments out of each staged step. Beyond TILE it amortises
// the barrier and the per-block quant setup - a Q8_0/Q4_K sub-block is 32
// elements, so BK 32 decodes each scale once instead of twice - at the cost
// of staging LDS, which is why the wide tiles keep 16.
#ifndef LA_BK
#define LA_BK 16
#endif
#define BK   LA_BK
#define BM   (LA_WM * LA_MT * TILE)
#define BN   (LA_WN * LA_NT * TILE)

#if (BK % TILE) != 0
#error "LA_BK must be a multiple of TILE"
#endif

// One wave per 16-row strip, so the group is always LA_NWAVE full waves.
// MatrixScope::Wave fragments are per-wave, so the lane-to-wave mapping
// below is only valid if the dispatch really runs at WAVE_SIZE - hence the
// [WaveSize] pin rather than trusting the driver's choice.
#define THREADS (LA_NWAVE * WAVE_SIZE)

#define LA_STAGE_SLOTS (LA_SINGLE_BUFFER ? 1 : 2)
groupshared float16_t tile_a[LA_STAGE_SLOTS * BM * BK]; // rows = tokens (M), stride BK
groupshared float16_t tile_b[LA_STAGE_SLOTS * BK * BN]; // rows = N,          stride BK
// One 16x16 slot per wave, reused per output tile. A slot per output tile
// would be LA_NT times larger for no benefit - the epilogue runs once.
groupshared float     tile_c[LA_NWAVE * TILE * TILE];

#if LA_MMID
#define LA_MMID_NONE 0xFFFFFFFFu
// MoE rows are gathered, so each tile row carries its own activation and
// destination base offset instead of deriving them from a linear row index.
// Computed once per group; 0xFFFFFFFF marks a row past the expert's count.
groupshared uint sh_a_off[BM];
groupshared uint sh_d_off[BM];
groupshared uint sh_expert;
groupshared uint sh_tile_local;
#if !LA_MMID_BUCKET
// The group derives its own tile assignment rather than reading a row map a
// prepare pass wrote: a histogram over the ids array is a couple of thousand
// loads against a tile that is millions of MACs, and keeping it in the same
// dispatch removes a scratch buffer and a cross-dispatch dependency.
groupshared uint sh_cnt[LA_MMID_MAX_EXPERT];
groupshared uint sh_wave_cnt[LA_NWAVE];

// Row ids run linearly over (slot, token, batch); ne1 is n_expert_used and
// ne2 the token count, matching the dst layout.
uint la_mmid_id(uint r, uint n_expert) {
    const uint slot  = r % ne1;
    const uint token = (r / ne1) % ne2;
    const uint id    = (uint)asint(src2.Load(op0 + slot * op1 + token * op2));
    return id < n_expert ? id : LA_MMID_NONE;
}
#endif
#endif

typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>           MatA;
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>           MatB;
typedef Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

// The weight side of the staging fetch. B is staged column-major ([n][k],
// stride BK), so a thread's B_PER_THREAD elements are consecutive along K in
// one row - contiguous in a float tensor, and inside a single block in a
// quantised one, since BK is 16 and a Q8_0 block is 32.
#if LA_QUANT == 8 || LA_QUANT == 4 || LA_QUANT == 5 || LA_QUANT == 6 || LA_QUANT == 50 || LA_QUANT == 40 || LA_QUANT == 41 || LA_QUANT == 51 || LA_QUANT == 49 || LA_QUANT == 30
// Elements per thread on the B side, as a preprocessor constant: the fetch
// below reads four quants from one dword, which only divides evenly for some
// tile-shape and wave-size combinations.
#define LA_B_PT   ((BK * BN) / THREADS)
#define LA_B_STEP (((LA_B_PT) % 4) == 0 ? 4 : (LA_B_PT))
#endif

#if LA_QUANT == 8
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (LA_ALIGNED || LA_FULL_TILE || (b_gn < ne01 && b_gk < K)) {          \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK8_0) * Q8_0_BSIZE;            \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const uint b_qs  = b_blk + 2u + (b_gk % QK8_0);                    \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_qs + e);                   \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint raw = (q4 >> (j * 8u)) & 0xFFu;                 \
                    const int  q   = (int)(raw ^ 0x80u) - 128;                 \
                    rb[e + j] = (float16_t)(b_d * (float)q);                   \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 50
#if LA_Q50_PACKED
#define LA_Q50_DECODE                                                          \
    [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {                \
        const uint q4 = la_q8_quads(src0, b_qs + ((b_e0 + e) & 12u));           \
        [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                          \
            const uint ee  = b_e0 + e + j;                                     \
            const uint by  = (q4 >> ((ee & 3u) * 8u)) & 0xFFu;                 \
            const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);            \
            const uint hb  = (b_qh >> ee) & 1u;                                \
            const int  q   = (int)(nib | (hb << 4)) - 16;                       \
            rb[e + j] = (float16_t)(b_d * (float)q);                            \
        }                                                                      \
    }
#else
#define LA_Q50_DECODE                                                          \
    [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                          \
        const uint ee  = b_e0 + e;                                             \
        const uint by  = la_q8_byte(src0, b_qs + (ee & 15u));                   \
        const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);                \
        const uint hb  = (b_qh >> ee) & 1u;                                    \
        const int  q   = (int)(nib | (hb << 4)) - 16;                           \
        rb[e] = (float16_t)(b_d * (float)q);                                    \
    }
#endif
// Q5_0 is what a K-quant file falls back to on rows that are not a multiple
// of 256, which is every 576-wide tensor in SmolLM2/SmolVLM2. BK is 16 and a
// block is 32, so a tile sits entirely in one nibble half of one block: the
// scale and the high-bit word are read once and the quants come four to a
// dword, exactly as in the Q8_0 path above.
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK5_0) * Q5_0_BSIZE;            \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const uint b_qh  = la_q8_quads(src0, b_blk + 2u);                  \
            const uint b_e0  = b_gk % QK5_0;                                   \
            const uint b_qs  = b_blk + 6u;                                     \
            LA_Q50_DECODE                                                      \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 40
// Q4_0 is Q5_0 without the high-bit word: 18 bytes, scale then 16 nibble
// bytes, quants biased by -8.
// Round the quad address down within the nibble half, also for two-element runs.
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK4_0) * Q4_0_BSIZE;            \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const uint b_e0  = b_gk % QK4_0;                                   \
            const uint b_qs  = b_blk + 2u;                                     \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_qs + ((b_e0 + e) & 12u));  \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint ee  = b_e0 + e + j;                             \
                    const uint by  = (q4 >> ((ee & 3u) * 8u)) & 0xFFu;        \
                    const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);   \
                    rb[e + j] = (float16_t)(b_d * (float)((int)nib - 8));      \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 41
// Q4_1 carries a min alongside the scale: 20 bytes, d then m then 16 nibble
// bytes, and the quant is unbiased.
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK4_1) * Q4_1_BSIZE;            \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const float b_m  = la_q8_scale(src0, b_blk + 2u);                  \
            const uint b_e0  = b_gk % QK4_1;                                   \
            const uint b_qs  = b_blk + 4u;                                     \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_qs + ((b_e0 + e) & 12u));  \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint ee  = b_e0 + e + j;                             \
                    const uint by  = (q4 >> ((ee & 3u) * 8u)) & 0xFFu;        \
                    const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);   \
                    rb[e + j] = (float16_t)(b_d * (float)nib + b_m);           \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 51
// Q5_1 is Q4_1 plus the Q5_0 high-bit word: 24 bytes, d, m, qh, then qs.
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK5_1) * Q5_1_BSIZE;            \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const float b_m  = la_q8_scale(src0, b_blk + 2u);                  \
            const uint b_qh  = la_q8_quads(src0, b_blk + 4u);                  \
            const uint b_e0  = b_gk % QK5_1;                                   \
            const uint b_qs  = b_blk + 8u;                                     \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                const uint ee  = b_e0 + e;                                     \
                const uint by  = la_q8_byte(src0, b_qs + (ee & 15u));          \
                const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);       \
                const uint hb  = (b_qh >> ee) & 1u;                            \
                rb[e] = (float16_t)(b_d * (float)(nib | (hb << 4)) + b_m);     \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 49
// IQ4_NL has the Q4_0 layout (18 bytes, d then 16 nibble bytes) but the
// nibble indexes a non-linear codebook instead of being a biased integer.
// Codebook matches kvalues_iq4nl in ggml-common.h.
int la_kvalues_iq4nl(uint idx) {
    const uint packed[4] = {
        0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
    };
    const uint w = packed[idx >> 2];
    const uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK4_NL) * Q4_NL_BSIZE;          \
            const float b_d  = la_q8_scale(src0, b_blk);                       \
            const uint b_e0  = b_gk % QK4_NL;                                  \
            const uint b_qs  = b_blk + 2u;                                     \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                const uint ee  = b_e0 + e;                                     \
                const uint by  = la_q8_byte(src0, b_qs + (ee & 15u));          \
                const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);       \
                rb[e] = (float16_t)(b_d * (float)la_kvalues_iq4nl(nib));       \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 30
// MXFP4: 17-byte block, a one-byte E8M0 exponent then 16 nibble bytes with
// the same low/high split as Q4_0. kvalues_fp4 stores 2x the E2M1 values, so
// the 0.5 is folded into the scale (GGML_E8M0_TO_FP32_HALF).
int la_kvalues_fp4(uint idx) {
    const uint packed[4] = {
        0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu
    };
    const uint w = packed[idx >> 2];
    const uint b = (w >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b << 24) >> 24;
}
float la_e8m0_half(uint e) {
    return asfloat((e < 2u) ? (0x00200000u << e) : ((e - 1u) << 23));
}
// The fetch below reads four nibble bytes from one dword, which is only
// contiguous within a nibble half if a thread's run starts on a multiple of
// its own length inside BK.
#if (16 % LA_B_PT) != 0
#error "MXFP4 LA_FETCH_B needs B_PER_THREAD to divide BK"
#endif
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK_MXFP4) * MXFP4_BSIZE;        \
            const float b_d  = la_e8m0_half(la_q8_byte(src0, b_blk));          \
            const uint b_e0  = b_gk % QK_MXFP4;                                \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint ee0 = b_e0 + e;                                     \
                const uint q4  = la_q8_quads(src0, b_blk + 1u + (ee0 & 15u));  \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint ee = ee0 + j;                                   \
                    /* the dword only covers the run while it stays inside  */ \
                    /* the 16-byte nibble half it started in                */ \
                    const uint by = ((ee0 & 15u) + j < 16u)                    \
                        ? ((q4 >> (j * 8u)) & 0xFFu)                           \
                        : la_q8_byte(src0, b_blk + 1u + (ee & 15u));           \
                    const uint nib = (ee >= 16u) ? (by >> 4) : (by & 0x0Fu);   \
                    rb[e + j] = (float16_t)(b_d * (float)la_kvalues_fp4(nib)); \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 4
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK_K) * Q4_K_BSIZE;             \
            /* the 32-element sub-block this K tile sits in; BK is 16, so   */ \
            /* a whole tile shares one scale/min pair and one nibble half   */ \
            const uint  b_sb = (b_gk % QK_K) / 32u;                            \
            const uint b_skn = b_gk >> 5;                                      \
            if (b_skn != b_skey) {                                             \
                b_skey = b_skn;                                                \
                const float2 b_sm = la_k_scale_min(src0, b_blk + 4u, b_sb);    \
                b_dc = la_q8_scale(src0, b_blk)      * b_sm.x;                 \
                b_mc = la_q8_scale(src0, b_blk + 2u) * b_sm.y;                 \
            }                                                                  \
            const float b_de = b_dc;                                           \
            const float b_me = b_mc;                                           \
            const uint b_qs  = b_blk + 16u + (b_sb / 2u) * 32u + (b_gk % 32u); \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_qs + e);                   \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint by = (q4 >> (j * 8u)) & 0xFFu;                  \
                    const uint q  = (b_sb & 1u) != 0u ? (by >> 4) : (by & 0x0Fu); \
                    rb[e + j] = (float16_t)(b_de * (float)q - b_me);           \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 5
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK_K) * Q5_K_BSIZE;             \
            const uint  b_sb = (b_gk % QK_K) / 32u;                            \
            const uint b_skn = b_gk >> 5;                                      \
            if (b_skn != b_skey) {                                             \
                b_skey = b_skn;                                                \
                const float2 b_sm = la_k_scale_min(src0, b_blk + 4u, b_sb);    \
                b_dc = la_q8_scale(src0, b_blk)      * b_sm.x;                 \
                b_mc = la_q8_scale(src0, b_blk + 2u) * b_sm.y;                 \
            }                                                                  \
            const float b_de = b_dc;                                           \
            const float b_me = b_mc;                                           \
            const uint b_j   = b_gk % 32u;                                     \
            const uint b_qh  = b_blk + 16u + b_j;                              \
            const uint b_qs  = b_blk + 48u + (b_sb / 2u) * 32u + b_j;          \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_qs + e);                   \
                const uint h4 = la_q8_quads(src0, b_qh + e);                   \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint by = (q4 >> (j * 8u)) & 0xFFu;                  \
                    const uint hb = (h4 >> (j * 8u)) & 0xFFu;                  \
                    const uint q  = ((b_sb & 1u) != 0u ? (by >> 4) : (by & 0x0Fu)) \
                                  | (((hb >> b_sb) & 1u) << 4);                \
                    rb[e + j] = (float16_t)(b_de * (float)q - b_me);           \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#elif LA_QUANT == 6
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        if (b_gn < ne01 && b_gk < K) {                                         \
            const uint b_row = src0_offset + b_gn * nb01                       \
                             + i2_src0 * nb02 + i3_src0 * nb03;                \
            const uint b_blk = b_row + (b_gk / QK_K) * Q6_K_BSIZE;             \
            /* Q6_K interleaves each 256-block as two 128 halves of four   */ \
            /* 32-element runs; a 16-wide K tile stays inside one run.     */ \
            const uint b_r   = b_gk % QK_K;                                    \
            const uint b_hf  = b_r / 128u;                                     \
            const uint b_sub = (b_r % 128u) / 32u;                             \
            const uint b_l   = b_r % 32u;                                      \
            const uint b_ql  = b_blk + b_hf * 64u + (b_sub & 1u) * 32u + b_l;  \
            const uint b_qh  = b_blk + 128u + b_hf * 32u + b_l;                \
            const uint b_lsh = b_sub >= 2u ? 4u : 0u;                          \
            const uint b_hsh = b_sub * 2u;                                     \
            const uint b_si  = b_hf * 8u + b_sub * 2u + b_l / 16u;             \
            const uint b_sr  = la_q8_byte(src0, b_blk + 192u + b_si);          \
            const float b_de = la_q8_scale(src0, b_blk + 208u)                 \
                             * (float)((int)(b_sr ^ 0x80u) - 128);             \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e += LA_B_STEP) {      \
                const uint q4 = la_q8_quads(src0, b_ql + e);                   \
                const uint h4 = la_q8_quads(src0, b_qh + e);                   \
                [unroll] for (uint j = 0; j < LA_B_STEP; j++) {                \
                    const uint by = (q4 >> (j * 8u)) & 0xFFu;                  \
                    const uint hb = (h4 >> (j * 8u)) & 0xFFu;                  \
                    const int  q  = (int)(((by >> b_lsh) & 0x0Fu)              \
                                        | (((hb >> b_hsh) & 3u) << 4)) - 32;   \
                    rb[e + j] = (float16_t)(b_de * (float)q);                  \
                }                                                              \
            }                                                                  \
        } else {                                                               \
            [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                 \
                rb[e] = (float16_t)0;                                          \
            }                                                                  \
        }                                                                      \
    }
#else
#define LA_FETCH_B(k_start_)                                                   \
    {                                                                          \
        const uint b_idx0 = tid * B_PER_THREAD;                                \
        const uint b_gn   = col_block * BN + b_idx0 / BK;                      \
        const uint b_gk   = (k_start_) + b_idx0 % BK;                          \
        const bool b_wide = LA_FULL_TILE || ((la_b_f32 || la_b_f16 || la_b_bf16) && \
                         (b_gn < ne01)                                          \
                         && (b_gk + B_PER_THREAD <= K));                       \
        if (b_wide) {                                                          \
            const uint b_off = offset_4d(b_gk, b_gn, i2_src0, i3_src0,         \
                                         nb00, nb01, nb02, nb03, src0_offset); \
            if (la_b_f32) {                                                    \
                [unroll] for (uint e = 0; e < B_PER_THREAD; e += 4) {          \
                    const uint4 w = src0.Load4(b_off + e * 4u);                \
                    rb[e]      = (float16_t)asfloat(w.x);                      \
                    rb[e + 1u] = (float16_t)asfloat(w.y);                      \
                    rb[e + 2u] = (float16_t)asfloat(w.z);                      \
                    rb[e + 3u] = (float16_t)asfloat(w.w);                      \
                }                                                              \
            } else if (la_b_bf16) {                                            \
                [unroll] for (uint e = 0; e < B_PER_THREAD; e += 4) {          \
                    const uint2 w = src0.Load2(b_off + e * 2u);                \
                    rb[e]      = (float16_t)asfloat((w.x & 0xFFFFu) << 16);    \
                    rb[e + 1u] = (float16_t)asfloat(w.x & 0xFFFF0000u);        \
                    rb[e + 2u] = (float16_t)asfloat((w.y & 0xFFFFu) << 16);    \
                    rb[e + 3u] = (float16_t)asfloat(w.y & 0xFFFF0000u);        \
                }                                                              \
            } else {                                                           \
                [unroll] for (uint e = 0; e < B_PER_THREAD; e += 4) {          \
                    const uint2 w = src0.Load2(b_off + e * 2u);                \
                    rb[e]      = asfloat16((uint16_t)(w.x & 0xFFFFu));         \
                    rb[e + 1u] = asfloat16((uint16_t)(w.x >> 16));             \
                    rb[e + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));         \
                    rb[e + 3u] = asfloat16((uint16_t)(w.y >> 16));             \
                }                                                              \
            }                                                                  \
        } else                                                                 \
        [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {                     \
            const uint idx = tid * B_PER_THREAD + e;                           \
            const uint n   = idx / BK;                                         \
            const uint k   = idx % BK;                                         \
            const uint gk  = (k_start_) + k;                                   \
            const uint gn  = col_block * BN + n;                               \
            float16_t val = (float16_t)0;                                      \
            if (gk < K && gn < ne01) {                                         \
                const uint off = offset_4d(gk, gn, i2_src0, i3_src0,           \
                                           nb00, nb01, nb02, nb03, src0_offset); \
                val = (float16_t)load_auto(src0, off, src0_esize);             \
            }                                                                  \
            rb[e] = val;                                                       \
        }                                                                      \
    }
#endif

WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tid   = gtid.x;
    const uint wave  = tid / WAVE_SIZE;
    const uint lane  = tid % WAVE_SIZE;

    // Position of this wave in the LA_WM x LA_WN grid over the group tile.
    // warp_r varies fastest so neighbouring waves share a B strip.
    const uint warp_r = wave % LA_WM;
    const uint warp_c = wave / LA_WM;

    // gid.x is the token block and gid.y the output block, not the other way
    // round: groups are dispatched with x varying fastest, so this keeps the
    // groups that share a weight tile resident together and lets them hit in
    // L2 instead of each streaming its own slice of the weights.
    const uint row_block = gid.x;
    const uint col_block = gid.y;
    const uint batch     = gid.z;

#if LA_MMID
    // gid.x indexes a flat list of BM-row tiles: experts are laid out in
    // order and each expert's run is padded up to a tile boundary, so one
    // tile always belongs to exactly one expert and the host can size the
    // dispatch without knowing the per-expert counts.
    const uint n_expert = ne02;
    const uint n_rows   = ne1 * ne2 * ne3;

#if LA_MMID_BUCKET
    if (tid == 0) {
        sh_expert = LA_MMID_NONE;
        uint tile_base = 0;
        uint start = temp.Load(0);
        for (uint e = 0; e < n_expert; ++e) {
            const uint end = temp.Load((e + 1u) * 4u);
            const uint tiles = (end - start + BM - 1u) / BM;
            if (row_block >= tile_base && row_block < tile_base + tiles) {
                sh_expert = e;
                sh_tile_local = row_block - tile_base;
                break;
            }
            tile_base += tiles;
            start = end;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    const uint expert = sh_expert;
    if (expert == LA_MMID_NONE) {
        return;
    }
    const uint start = temp.Load(expert * 4u);
    const uint end = temp.Load((expert + 1u) * 4u);
    for (uint m = tid; m < BM; m += THREADS) {
        const uint r = start + sh_tile_local * BM + m;
        sh_a_off[m] = LA_MMID_NONE;
        sh_d_off[m] = LA_MMID_NONE;
        if (r < end) {
#if LA_MMID_EXPERT_MAJOR
            const uint pair = temp.Load(op6 + r * 4u);
            if (op7 == 2u) {
                sh_a_off[m] = src1_offset + r * nb11;
                sh_d_off[m] = dst_offset + pair * nb1;
            } else {
                sh_a_off[m] = src1_offset + (pair % ne1 % ne11) * nb11 + (pair / ne1) * nb12;
                sh_d_off[m] = dst_offset + r * nb1;
            }
#else
            const uint pair = temp.Load(op6 + r * 4u);
            const uint slot = pair % ne1;
            const uint token = pair / ne1;
            sh_a_off[m] = src1_offset + (slot % ne11) * nb11 + token * nb12;
            sh_d_off[m] = dst_offset + slot * nb1 + token * nb2;
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
#else
    for (uint e0 = tid; e0 < n_expert; e0 += THREADS) {
        sh_cnt[e0] = 0;
    }
    if (tid == 0) {
        sh_expert = LA_MMID_NONE;
    }
    for (uint m0 = tid; m0 < BM; m0 += THREADS) {
        sh_a_off[m0] = LA_MMID_NONE;
        sh_d_off[m0] = LA_MMID_NONE;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint r0 = tid; r0 < n_rows; r0 += THREADS) {
        const uint id = la_mmid_id(r0, n_expert);
        if (id != LA_MMID_NONE) {
            uint prev;
            InterlockedAdd(sh_cnt[id], 1u, prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // n_expert is a few hundred at most, so walking the padded runs serially
    // costs less than the barriers a parallel scan would need.
    if (tid == 0) {
        uint t0 = 0;
        for (uint e1 = 0; e1 < n_expert; e1++) {
            const uint nt = (sh_cnt[e1] + BM - 1u) / BM;
            if (row_block >= t0 && row_block < t0 + nt) {
                sh_expert     = e1;
                sh_tile_local = row_block - t0;
            }
            t0 += nt;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint expert = sh_expert;
    if (expert == LA_MMID_NONE) {
        return;
    }

    // Rank within the expert's run decides the slot, and every tile of the
    // same expert has to agree on it, so the scan is a deterministic prefix
    // count over the rows rather than an atomic hand-out.
    const uint slot_base = sh_tile_local * BM;
    uint scanned = 0;
    for (uint c = 0; c < n_rows; c += THREADS) {
        const uint r1    = c + tid;
        const bool match = (r1 < n_rows) && (la_mmid_id(r1, n_expert) == expert);
        const uint wpre  = WavePrefixCountBits(match);
        const uint wtot  = WaveActiveCountBits(match);
        if (lane == 0) {
            sh_wave_cnt[wave] = wtot;
        }
        GroupMemoryBarrierWithGroupSync();

        uint pre   = 0;
        uint total = 0;
        [unroll] for (uint w = 0; w < LA_NWAVE; w++) {
            const uint n = sh_wave_cnt[w];
            if (w < wave) {
                pre += n;
            }
            total += n;
        }
        if (match) {
            const uint rank = scanned + pre + wpre;
            if (rank >= slot_base && rank < slot_base + BM) {
                const uint slot  = r1 % ne1;
                const uint token = (r1 / ne1) % ne2;
                const uint bat   = r1 / (ne1 * ne2);
                sh_a_off[rank - slot_base] =
                    src1_offset + (slot % ne11) * nb11 + token * nb12 + bat * nb13;
                sh_d_off[rank - slot_base] =
                    dst_offset + slot * nb1 + token * nb2 + bat * nb3;
            }
        }
        scanned += total;
        GroupMemoryBarrierWithGroupSync();
    }
#endif

    const uint i2 = 0;
    const uint i3 = 0;
    const uint i2_src0 = expert;
    const uint i3_src0 = 0;
#else
    const uint i2 = batch % ne2;
    const uint i3 = batch / ne2;
    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;
#endif

    const uint K = ne00;
    const uint num_k_tiles = (K + BK - 1) / BK;

    MatAcc acc[LA_MT][LA_NT];
    [unroll] for (uint m = 0; m < LA_MT; m++) {
        [unroll] for (uint t = 0; t < LA_NT; t++) {
            acc[m][t] = MatAcc::Splat(0.0f);
        }
    }

    const uint A_PER_THREAD = (BM * BK) / THREADS;
    const uint B_PER_THREAD = (BK * BN) / THREADS;

    // Global loads for the next K step are issued before the current step's
    // MultiplyAccumulate chain and only land in LDS at the top of the next
    // iteration, so their latency overlaps the matrix work instead of
    // stalling in front of it. Alternating the LDS buffer also drops the
    // per-iteration barrier count from two to one.
    float16_t ra[A_PER_THREAD];
    float16_t rb[B_PER_THREAD];

    // A thread's B elements stay in one row, and BK is 16 while the smallest
    // quant block spans 32 K values, so the dequant scales only change every
    // second K step. Carry them across steps instead of reloading each time.
    uint  b_skey = 0xFFFFFFFFu;
    float b_dc   = 0.0f;
    float b_mc   = 0.0f;

    // Each thread's A elements are A_PER_THREAD consecutive K values in one
    // row, so a contiguous source can be pulled with wide loads.
#if LA_DENSE_FIXED || LA_ALIGNED
    const bool la_a_f32 = true;
    const bool la_a_f16 = false;
    const bool la_a_bf16 = false;
#else
    const bool la_a_f32 = (src1_esize == 4u) && (nb10 == 4u)
                       && (((src1_offset | nb11 | nb12 | nb13) & 15u) == 0u);
    const bool la_a_f16 = (src1_esize == 2u) && (nb10 == 2u)
                       && (((src1_offset | nb11 | nb12 | nb13) & 7u) == 0u);
    const bool la_a_bf16 = (src1_esize == 3u) && (nb10 == 2u)
                        && (((src1_offset | nb11 | nb12 | nb13) & 7u) == 0u);
#endif
    // B is staged column-major ([n][k], stride BK) so the same trick applies
    // to the weight side, whose rows run along K.
#if LA_DENSE_FIXED == 1
    const bool la_b_f32 = false;
    const bool la_b_f16 = true;
    const bool la_b_bf16 = false;
#else
    const bool la_b_f32 = (src0_esize == 4u) && (nb00 == 4u)
                       && (((src0_offset | nb01 | nb02 | nb03) & 15u) == 0u);
    const bool la_b_f16 = (src0_esize == 2u) && (nb00 == 2u)
                       && (((src0_offset | nb01 | nb02 | nb03) & 7u) == 0u);
    const bool la_b_bf16 = (src0_esize == 3u) && (nb00 == 2u)
                        && (((src0_offset | nb01 | nb02 | nb03) & 7u) == 0u);
#endif

#define LA_FETCH(kt_)                                                          \
    {                                                                          \
        const uint k_start = (kt_) * BK;                                       \
        const uint a_idx0  = tid * A_PER_THREAD;                               \
        const uint a_gm    = row_block * BM + a_idx0 / BK;                     \
        const uint a_gk    = k_start + a_idx0 % BK;                            \
        LA_FETCH_A(k_start, a_idx0, a_gm, a_gk)                                \
        LA_FETCH_B(k_start)                                                    \
    }

#if LA_MMID
// The gathered row's activation base is already resolved in sh_a_off, so the
// wide path only needs the row to exist and the run to stay inside K.
#define LA_FETCH_A(k_start, a_idx0, a_gm, a_gk)                                \
    {                                                                          \
        const uint a_row  = a_idx0 / BK;                                       \
        const uint a_base_off = sh_a_off[a_row];                               \
        const bool a_wide = (la_a_f32 || la_a_f16 || la_a_bf16)                \
                         && (a_base_off != 0xFFFFFFFFu)                        \
                         && (a_gk + A_PER_THREAD <= K);                        \
        if (a_wide) {                                                          \
            const uint a_off = a_base_off + a_gk * nb10;                       \
            if (la_a_f32) {                                                    \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint4 w = src1.Load4(a_off + e * 4u);                \
                    ra[e]      = (float16_t)asfloat(w.x);                      \
                    ra[e + 1u] = (float16_t)asfloat(w.y);                      \
                    ra[e + 2u] = (float16_t)asfloat(w.z);                      \
                    ra[e + 3u] = (float16_t)asfloat(w.w);                      \
                }                                                              \
            } else if (la_a_bf16) {                                            \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint2 w = src1.Load2(a_off + e * 2u);                \
                    ra[e]      = (float16_t)asfloat((w.x & 0xFFFFu) << 16);    \
                    ra[e + 1u] = (float16_t)asfloat(w.x & 0xFFFF0000u);        \
                    ra[e + 2u] = (float16_t)asfloat((w.y & 0xFFFFu) << 16);    \
                    ra[e + 3u] = (float16_t)asfloat(w.y & 0xFFFF0000u);        \
                }                                                              \
            } else {                                                           \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint2 w = src1.Load2(a_off + e * 2u);                \
                    ra[e]      = asfloat16((uint16_t)(w.x & 0xFFFFu));         \
                    ra[e + 1u] = asfloat16((uint16_t)(w.x >> 16));             \
                    ra[e + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));         \
                    ra[e + 3u] = asfloat16((uint16_t)(w.y >> 16));             \
                }                                                              \
            }                                                                  \
        } else                                                                 \
        [unroll] for (uint e = 0; e < A_PER_THREAD; e++) {                     \
            const uint idx = tid * A_PER_THREAD + e;                           \
            const uint m   = idx / BK;                                         \
            const uint k   = idx % BK;                                         \
            const uint gk  = k_start + k;                                      \
            const uint bo  = sh_a_off[m];                                      \
            float16_t val = (float16_t)0;                                      \
            if (bo != 0xFFFFFFFFu && gk < K) {                                 \
                val = (float16_t)load_auto(src1, bo + gk * nb10, src1_esize);  \
            }                                                                  \
            ra[e] = val;                                                       \
        }                                                                      \
    }
#else
#define LA_FETCH_A(k_start, a_idx0, a_gm, a_gk)                                \
    {                                                                          \
        const bool a_wide  = LA_ALIGNED || LA_FULL_TILE || ((la_a_f32 || la_a_f16 || la_a_bf16) && \
                          (a_gm < ne11)                                         \
                          && (a_gk + A_PER_THREAD <= K));                      \
        if (a_wide) {                                                          \
            const uint a_off = offset_4d(a_gk, a_gm, i2, i3,                   \
                                         nb10, nb11, nb12, nb13, src1_offset); \
            if (la_a_f32) {                                                    \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint4 w = src1.Load4(a_off + e * 4u);                \
                    ra[e]      = (float16_t)asfloat(w.x);                      \
                    ra[e + 1u] = (float16_t)asfloat(w.y);                      \
                    ra[e + 2u] = (float16_t)asfloat(w.z);                      \
                    ra[e + 3u] = (float16_t)asfloat(w.w);                      \
                }                                                              \
            } else if (la_a_bf16) {                                            \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint2 w = src1.Load2(a_off + e * 2u);                \
                    ra[e]      = (float16_t)asfloat((w.x & 0xFFFFu) << 16);    \
                    ra[e + 1u] = (float16_t)asfloat(w.x & 0xFFFF0000u);        \
                    ra[e + 2u] = (float16_t)asfloat((w.y & 0xFFFFu) << 16);    \
                    ra[e + 3u] = (float16_t)asfloat(w.y & 0xFFFF0000u);        \
                }                                                              \
            } else {                                                           \
                [unroll] for (uint e = 0; e < A_PER_THREAD; e += 4) {          \
                    const uint2 w = src1.Load2(a_off + e * 2u);                \
                    ra[e]      = asfloat16((uint16_t)(w.x & 0xFFFFu));         \
                    ra[e + 1u] = asfloat16((uint16_t)(w.x >> 16));             \
                    ra[e + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));         \
                    ra[e + 3u] = asfloat16((uint16_t)(w.y >> 16));             \
                }                                                              \
            }                                                                  \
        } else                                                                 \
        [unroll] for (uint e = 0; e < A_PER_THREAD; e++) {                     \
            const uint idx = tid * A_PER_THREAD + e;                           \
            const uint m   = idx / BK;                                         \
            const uint k   = idx % BK;                                         \
            const uint gm  = row_block * BM + m;                               \
            const uint gk  = k_start + k;                                      \
            float16_t val = (float16_t)0;                                      \
            if (gm < ne11 && gk < K) {                                         \
                const uint off = offset_4d(gk, gm, i2, i3,                     \
                                           nb10, nb11, nb12, nb13, src1_offset); \
                val = (float16_t)load_auto(src1, off, src1_esize);             \
            }                                                                  \
            ra[e] = val;                                                       \
        }                                                                      \
    }
#endif

    LA_FETCH(0)

    uint buf = 0;
    for (uint kt = 0; kt < num_k_tiles; kt++) {
#if LA_SINGLE_BUFFER
        const uint a_base = 0;
        const uint b_base = 0;
#else
        const uint a_base = buf * (BM * BK);
        const uint b_base = buf * (BK * BN);
#endif

        // Publish the prefetched step. A_PER_THREAD elements are contiguous
        // in the staged layout, so this is a straight run of LDS writes.
        [unroll] for (uint e = 0; e < A_PER_THREAD; e++) {
            tile_a[a_base + tid * A_PER_THREAD + e] = ra[e];
        }
        [unroll] for (uint e = 0; e < B_PER_THREAD; e++) {
            tile_b[b_base + tid * B_PER_THREAD + e] = rb[e];
        }

        // The next step's global loads are issued before the barrier, so the
        // time the group spends waiting for its slowest wave overlaps their
        // latency instead of following it.
        if (!LA_SINGLE_BUFFER && kt + 1 < num_k_tiles) {
            LA_FETCH(kt + 1)
        }

        GroupMemoryBarrierWithGroupSync();

        // One staged step holds BK/TILE fragments along K. Both staged tiles
        // are strided by BK, so stepping a fragment is just an offset of TILE
        // into the same row.
        [unroll] for (uint kk = 0; kk < BK / TILE; kk++) {
            MatB b[LA_NT];
            [unroll] for (uint t = 0; t < LA_NT; t++) {
                b[t] = MatB::Load(tile_b,
                                  b_base + (warp_c * LA_NT + t) * TILE * BK + kk * TILE,
                                  BK, MatrixLayout::ColMajor);
            }
            [unroll] for (uint m = 0; m < LA_MT; m++) {
                MatA a = MatA::Load(tile_a,
                                    a_base + (warp_r * LA_MT + m) * TILE * BK + kk * TILE,
                                    BK, MatrixLayout::RowMajor);
                [unroll] for (uint t = 0; t < LA_NT; t++) {
                    acc[m][t].MultiplyAccumulate(a, b[t]);
                }
            }
        }

#if LA_SINGLE_BUFFER
        if (kt + 1 < num_k_tiles) {
            LA_FETCH(kt + 1)
            GroupMemoryBarrierWithGroupSync();
        }
#else
        buf ^= 1;
#endif
    }
#undef LA_FETCH

    // Accumulators land in LDS so the global store can be bounds-checked;
    // Matrix::Store writes a whole 16x16 tile with no masking. Each wave owns
    // a private slice and its lanes run in lockstep, so one tile at a time
    // needs only a barrier strong enough to order the wave's own LDS accesses.
    const uint c_base = wave * TILE * TILE;
#if LA_REG_EPILOGUE
    // NVIDIA exposes a correct accumulator element mapping, so its variant can
    // drain directly to the output without the LDS store/reload round trip.
    [unroll] for (uint m = 0; m < LA_MT; m++) {
    [unroll] for (uint t = 0; t < LA_NT; t++) {
        [unroll] for (uint e = 0; e < (TILE * TILE) / WAVE_SIZE; e++) {
            const uint2 rc = acc[m][t].GetCoordinate(e);
            const uint r  = rc.x;
            const uint c  = rc.y;
            const uint gm = row_block * BM + (warp_r * LA_MT + m) * TILE + r;
            const uint gn = col_block * BN + (warp_c * LA_NT + t) * TILE + c;
#if LA_MMID
            const uint d_base = sh_d_off[(warp_r * LA_MT + m) * TILE + r];
            if (d_base != LA_MMID_NONE && gn < ne0) {
                store_auto(dst, d_base + gn * nb0, acc[m][t].Get(e), dst_esize);
            }
#else
            if (gm < ne1 && gn < ne0) {
                const uint off = offset_4d(gn, gm, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                float value = acc[m][t].Get(e);
#if !LA_MMID
                if (op0 == 1u) {
                    value += asfloat(src2.Load(op1 + gn * op2));
                }
#endif
                store_auto(dst, off, value, dst_esize);
            }
#endif
        }
    }
    }
#else
    [unroll] for (uint m = 0; m < LA_MT; m++) {
    [unroll] for (uint t = 0; t < LA_NT; t++) {
        acc[m][t].Store(tile_c, c_base, TILE, MatrixLayout::RowMajor);
        GroupMemoryBarrier();

        // WAVE_SIZE lanes cover a 16x16 tile in TILE*TILE/WAVE_SIZE passes.
        [unroll] for (uint p = 0; p < (TILE * TILE) / WAVE_SIZE; p++) {
            const uint elem = p * WAVE_SIZE + lane;
            const uint r    = elem / TILE;
            const uint c    = elem % TILE;
            const uint gm   = row_block * BM + (warp_r * LA_MT + m) * TILE + r;
            const uint gn   = col_block * BN + (warp_c * LA_NT + t) * TILE + c;
#if LA_MMID
            const uint d_base = sh_d_off[(warp_r * LA_MT + m) * TILE + r];
            if (d_base != 0xFFFFFFFFu && gn < ne0) {
#if LA_MMID_EXPERT_MAJOR
                float value = tile_c[c_base + elem];
                if (op7 == 2u) {
                    value *= asfloat(src2.Load(op8 + ((d_base - dst_offset) / nb1) * 4u));
                }
                store_auto(dst, d_base + gn * nb0, value, dst_esize);
#else
                store_auto(dst, d_base + gn * nb0, tile_c[c_base + elem], dst_esize);
#endif
            }
#else
            if (LA_ALIGNED || LA_FULL_TILE || (gm < ne1 && gn < ne0)) {
                const uint off = offset_4d(gn, gm, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                float value = tile_c[c_base + elem];
#if !LA_MMID
                if (op0 == 1u) {
                    value += asfloat(src2.Load(op1 + gn * op2));
                }
#endif
#if LA_ALIGNED || LA_FULL_TILE
                dst.Store(off, asuint(value));
#else
                store_auto(dst, off, value, dst_esize);
#endif
            }
#endif
        }
        GroupMemoryBarrier();
    }
    }
#endif
}
