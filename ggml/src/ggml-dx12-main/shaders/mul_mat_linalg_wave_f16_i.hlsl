// MUL_MAT on the wave matrix shape Intel Xe3 actually implements.
//
// The existing LinAlg GEMM (mul_mat_linalg_f16.hlsl) cannot run here: it asks
// for a 16x16x16 tile with an F32 accumulator, and this part offers 8x16x16
// with an F16 one. See TUNING.md section 21.
//
// So the structure is inverted relative to the other GEMMs. The operands:
//
//   A  activations, 8(M) x 16(K), row-major, row pitch ne00*2
//      pre-converted to F16 by cvt_f32_f16_flat.hlsl, contiguous
//   B  weights, 16(K) x 16(N), column-major, column pitch nb01
//      F16 in the model, already contiguous along K
//
// B is read straight from the descriptor: each wave owns a different column
// strip, so there is nothing to share.
//
// A is different. Its address does not depend on the wave, so every wave in
// the group reads the same rows, and the descriptor path fetches them NWAVE
// times over. LDS_STAGE_A stages that strip once per fold window instead.
// The array is float16_t so the type matches the matrix component type - with
// a mismatch the load silently converts (TUNING.md section 35).
//
// Both pitches are multiples of 4 whenever ne00 is even, which the host gate
// enforces - the load silently drops leading bytes below 4-byte alignment.
//
// The accumulator is F16, which is not enough on its own to carry a full K
// reduction. It is drained into F32 registers every FOLD_BLOCKS tiles, so F16
// only ever holds a bounded partial sum while the long reduction happens in
// F32. GetCoordinate is correct on this hardware, so the drain reads straight
// out of registers with no LDS round trip.
//
// A Matrix is never reassigned or passed by value here; a fresh accumulator is
// declared per fold window. Doing otherwise produced wrong results during
// bring-up.

#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#define WAVE_SIZE 16
#endif
#ifndef NWAVE
#define NWAVE 4
#endif
#ifndef FOLD_BLOCKS
#define FOLD_BLOCKS 4
#endif

#ifndef MTILE
#define MTILE 8
#endif
#ifndef NTILE
#define NTILE 1
#endif

#ifndef SWIZZLE_Y
#define SWIZZLE_Y 1
#endif

#ifndef LDS_STAGE_A
#define LDS_STAGE_A 0
#endif

// Weight source: 0 = F16, 8 = Q8_0, 4/5/6 = Q4_K/Q5_K/Q6_K,
// 40/41/50/51 = Q4_0/Q4_1/Q5_0/Q5_1, 14 = IQ4_NL, 17 = MXFP4.
// Quantized weights are dequantized into dense F16 LDS tiles before matrix loads.
#ifndef QUANT_B
#define QUANT_B 0
#endif

#ifndef IW_FUSED_BIAS
#define IW_FUSED_BIAS 0
#endif

// Probe hook, off in every shipped variant. See TUNING.md section 38.
#ifndef DRAIN_PROBE
#define DRAIN_PROBE 0
#endif

// True when the weights are staged through LDS, whatever the block type.
#define IW_QUANT_B ((QUANT_B == 8) || (QUANT_B == 4) || (QUANT_B == 5) || (QUANT_B == 6) || (QUANT_B == 50) || (QUANT_B == 40) || (QUANT_B == 41) || (QUANT_B == 51) || (QUANT_B == 14) || (QUANT_B == 17))

// True when anything is staged, and therefore when the group-wide barriers
// exist and no wave may leave early.
#define IW_STAGED (LDS_STAGE_A || IW_QUANT_B)

#define LA_M 8
#define LA_K 16
#define LA_N 16

// Waves are arranged as a WAVE_M x WAVE_N grid over the group tile. Splitting
// M across waves too is the only way to grow the group tile: MTILE and NTILE
// set the per-wave accumulator count, and growing those spills (TUNING.md
// section 26). Growing the wave grid leaves the accumulator set alone.
// F16 uses WAVE_M=1. Quantized variants use WAVE_M=2 to reduce the per-thread
// accumulator set without changing the group tile.
#ifndef WAVE_M
#define WAVE_M 1
#endif
#define WAVE_N (NWAVE / WAVE_M)

#define BM      (LA_M * MTILE * WAVE_M)
#define BN      (LA_N * NTILE * WAVE_N)
#define THREADS (WAVE_SIZE * NWAVE)
#define ACC_E   ((LA_M * LA_N) / WAVE_SIZE)
#define FOLD_K  (LA_K * FOLD_BLOCKS)

#if IW_QUANT_B && (FOLD_K % 32) != 0
#error "Quantized staging requires FOLD_K to be a multiple of 32"
#endif
#if QUANT_B == 17 && (FOLD_K % 64) != 0
#error "MXFP4 staging requires FOLD_K to be a multiple of 64"
#endif

typedef Matrix<ComponentType::F16, LA_M, LA_K, MatrixUse::A,           MatrixScope::Wave> MatAf;
typedef Matrix<ComponentType::F16, LA_K, LA_N, MatrixUse::B,           MatrixScope::Wave> MatBf;
typedef Matrix<ComponentType::F16, LA_M, LA_N, MatrixUse::Accumulator, MatrixScope::Wave> MatAccf;

#if LDS_STAGE_A
// One fold window of the activation strip, row-major, row pitch FOLD_K.
// FOLD_K is a multiple of 8 halves, so the row pitch is 16-byte aligned - the
// alignment rule hlsl-specs PR 879 adds (TUNING.md section 35).
groupshared float16_t lds_a[BM * FOLD_K];

// Bytes are moved as uint4 and unpacked, the same way the other staged
// kernels do it. One row of the window is FOLD_K*2 bytes.
#define A_U4_ROW (FOLD_K / 8)
#define A_U4_TOT (BM * A_U4_ROW)
#endif

#if QUANT_B == 8
#define QK8_0      32
#define Q8_0_BSIZE 34
#endif

#if QUANT_B == 50
#define QK5_0      32
#define Q5_0_BSIZE 22
#endif

#if QUANT_B == 4 || QUANT_B == 5
#define QK_K       256
#define Q4_K_BSIZE 144
#define Q5_K_BSIZE 176
#define Q4_K_SUB   32

// Byte i of the 12 packed scale bytes.
uint q4k_sbyte(uint3 w, uint i) {
    const uint d = (i < 4u) ? w.x : ((i < 8u) ? w.y : w.z);
    return (d >> ((i & 3u) * 8u)) & 0xFFu;
}
#endif

#if QUANT_B == 6 || QUANT_B == 40 || QUANT_B == 41 || QUANT_B == 51 || QUANT_B == 14
uint iw_load_u32(uint off) {
    const uint16_t2 w = src0.Load<uint16_t2>(off);
    return (uint)w.x | ((uint)w.y << 16u);
}
#endif

#if QUANT_B == 40 || QUANT_B == 14
#define IW_BLOCK_BYTES 18
#elif QUANT_B == 41
#define IW_BLOCK_BYTES 20
#elif QUANT_B == 51
#define IW_BLOCK_BYTES 24
#endif

#if QUANT_B == 14 || QUANT_B == 17
int iw_quant_value(uint idx) {
#if QUANT_B == 14
    const uint packed[4] = { 0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u };
#else
    const uint packed[4] = { 0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu };
#endif
    const uint b = (packed[idx >> 2] >> ((idx & 3u) * 8u)) & 0xFFu;
    return (int)(b ^ 0x80u) - 128;
}
#endif

#if IW_QUANT_B
// One fold window of the weight tile, column-major: column n starts at
// n*FOLD_K, which is what MatrixLayout::ColMajor with stride FOLD_K wants.
groupshared float16_t lds_b[BN * FOLD_K];
#endif

// The Matrix objects below are MatrixScope::Wave, and both ACC_E and the
// tid/WAVE_SIZE wave index assume the dispatch really runs at WAVE_SIZE
// lanes. Pin it rather than trusting the driver's choice - Intel Xe drivers
// have been seen picking a wider wave for shaders tuned to a narrower one.
WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint wave   = tid / WAVE_SIZE;
    const uint wave_m = wave / WAVE_N;
    const uint wave_n = wave % WAVE_N;

    // Group swizzle. The dispatch walks x (tokens) fastest, so without this an
    // x-sweep re-reads every activation row for each weight column block, and
    // the activations get read once per column block over the whole dispatch.
    // Banding SWIZZLE_Y column blocks together makes SWIZZLE_Y consecutive
    // groups share one activation strip, cutting activation traffic by the same
    // factor. The weights still stream once. This matters because the GEMM is
    // bound by bytes fetched, not by MACs (TUNING.md section 30).
    const uint n_x = (ne1 + BM - 1) / BM;
    const uint n_y = (ne0 + BN - 1) / BN;
#if SWIZZLE_Y > 1
    uint tile_x;
    uint tile_y;
    // Banding only pays when there are enough column blocks to band. With
    // fewer than SWIZZLE_Y the remap degenerates to column-major over the whole
    // grid, which re-reads the weights instead of the activations - measured
    // -24% on SmolLM2-135M, whose largest n_y is 12. The test is uniform across
    // the dispatch, so it costs no divergence.
    if (n_y >= SWIZZLE_Y) {
        const uint lin    = gid.y * n_x + gid.x;
        const uint band   = SWIZZLE_Y * n_x;
        const uint bfirst = (lin / band) * SWIZZLE_Y;
        // The last band is short when n_y is not a multiple of SWIZZLE_Y. Using
        // its real height keeps the remap a bijection.
        const uint bh     = min(n_y - bfirst, (uint)SWIZZLE_Y);
        tile_x = (lin % band) / bh;
        tile_y = bfirst + (lin % bh);
    } else {
        tile_x = gid.x;
        tile_y = gid.y;
    }
#else
    const uint tile_x = gid.x;
    const uint tile_y = gid.y;
#endif

    const uint rowg = tile_x * BM;                 // group's first token
    const uint row0 = rowg + wave_m * (LA_M * MTILE);        // tokens   (M, along ne1)
    const uint colg = tile_y * BN;                 // group's first channel
    const uint col0 = colg + wave_n * (LA_N * NTILE);        // channels (N, along ne0)

    // ne0 is a multiple of LA_N, so a wave that starts inside the tensor has
    // all of its columns inside it. A wave past the end has none.
#if IW_STAGED
    // It cannot leave, though: the staging barriers below are group-wide, and
    // a wave that returned early would never arrive at them.
    const bool active = (col0 < ne0);
#else
    if (col0 >= ne0) {
        return;
    }
    const bool active = true;
#endif
    const uint batch = gid.z;
    const uint i2 = batch % ne2;
    const uint i3 = batch / ne2;
    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;

    // Activations are flat F16 in the scratch: one contiguous ne00-long row per
    // token, tokens ordered by (i3, i2, i1).
    const uint act_pitch = ne00 * 2u;
    const uint act_row   = ((i3 * ne12) + i2) * ne11 + rowg;
    // Staging fills the whole group tile, so it starts at the group's row.
    // The descriptor path reads only this wave's rows.
    const uint a_grp     = src1_offset + act_row * act_pitch;
    const uint a_base    = a_grp + wave_m * (LA_M * MTILE) * act_pitch;

#if IW_QUANT_B
    // Quantized rows are addressed per column during staging, so only the
    // batch part of the base is common.
    const uint b_grp = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
#else
    const uint b_base = offset_4d(0, col0, i2_src0, i3_src0,
                                  nb00, nb01, nb02, nb03, src0_offset);
#endif

    float sum[MTILE][NTILE][ACC_E];
    [unroll] for (uint m = 0; m < MTILE; m++) {
        [unroll] for (uint n = 0; n < NTILE; n++) {
            [unroll] for (uint z = 0; z < ACC_E; z++) {
                sum[m][n][z] = 0.0f;
            }
        }
    }

    for (uint k0 = 0; k0 < ne00; k0 += FOLD_K) {
#if IW_STAGED
        // Wait for every wave to finish reading the previous window before
        // overwriting it. Redundant on the first pass, uniform, and cheap.
        GroupMemoryBarrierWithGroupSync();
#endif
#if LDS_STAGE_A
        // The M tail is not gated: the scratch carries BM spare rows, so the
        // read stays in bounds. K is whole windows by the host gate.
        for (uint u = tid; u < A_U4_TOT; u += THREADS) {
            const uint  r  = u / A_U4_ROW;
            const uint  h  = (u % A_U4_ROW) * 8u;   // half index within the row
            const uint4 w  = src1.Load4(a_grp + r * act_pitch + (k0 + h) * 2u);
            const uint  o  = r * FOLD_K + h;
            lds_a[o + 0u] = asfloat16((uint16_t)(w.x & 0xFFFFu));
            lds_a[o + 1u] = asfloat16((uint16_t)(w.x >> 16));
            lds_a[o + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));
            lds_a[o + 3u] = asfloat16((uint16_t)(w.y >> 16));
            lds_a[o + 4u] = asfloat16((uint16_t)(w.z & 0xFFFFu));
            lds_a[o + 5u] = asfloat16((uint16_t)(w.z >> 16));
            lds_a[o + 6u] = asfloat16((uint16_t)(w.w & 0xFFFFu));
            lds_a[o + 7u] = asfloat16((uint16_t)(w.w >> 16));
        }
#endif
#if QUANT_B == 8
        // Dequantize this window of the weight tile into LDS. One thread owns a
        // whole Q8_0 block, so the scale is read once and the 34 bytes come out
        // of nine aligned dwords - reading per group of four quants instead
        // fetched three dwords for every four bytes used.
        // Columns past ne0 are zero filled, which keeps the matrix load in
        // bounds without a per-tile edge case.
        for (uint bq = tid; bq < (BN * FOLD_K) / QK8_0; bq += THREADS) {
            const uint n   = bq / (FOLD_K / QK8_0);
            const uint kb0 = (bq % (FOLD_K / QK8_0)) * QK8_0;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < QK8_0; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint base = b_grp + gn * nb01 +
                              ((k0 + kb0) / QK8_0) * Q8_0_BSIZE;
            // K is 64-aligned, so only non-final blocks read the next scale.
            const uint a0  = base & ~3u;
            const uint off = base - a0;
            const uint4 w0 = src0.Load4(a0);
            const uint4 w1 = src0.Load4(a0 + 16u);
            const uint  w2 = src0.Load(a0 + 32u);
            uint d[9] = { w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2 };
            const float sc = f16_to_f32((d[0] >> (off * 8u)) & 0xFFFFu);
            [unroll] for (uint j = 0; j < QK8_0 / 4; j++) {
                const uint q4 = (off == 0u)
                    ? ((d[j] >> 16) | (d[j + 1] << 16))
                    : d[j + 1];
                const uint ob = o + j * 4u;
                lds_b[ob + 0u] = (float16_t)(sc * (float)((int)(((q4      ) & 0xFFu) ^ 0x80u) - 128));
                lds_b[ob + 1u] = (float16_t)(sc * (float)((int)(((q4 >>  8) & 0xFFu) ^ 0x80u) - 128));
                lds_b[ob + 2u] = (float16_t)(sc * (float)((int)(((q4 >> 16) & 0xFFu) ^ 0x80u) - 128));
                lds_b[ob + 3u] = (float16_t)(sc * (float)((int)(((q4 >> 24) & 0xFFu) ^ 0x80u) - 128));
            }
        }
#endif
#if QUANT_B == 50
        for (uint bq = tid; bq < (BN * FOLD_K) / QK5_0; bq += THREADS) {
            const uint n   = bq / (FOLD_K / QK5_0);
            const uint kb0 = (bq % (FOLD_K / QK5_0)) * QK5_0;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < QK5_0; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint base = b_grp + gn * nb01 +
                              ((k0 + kb0) / QK5_0) * Q5_0_BSIZE;
            // Exact halfword loads keep the final 22-byte block in bounds.
            const float sc = f16_to_f32(src0.Load<uint16_t>(base));
            const uint qh = (uint)src0.Load<uint16_t>(base + 2u) |
                           ((uint)src0.Load<uint16_t>(base + 4u) << 16u);
            [unroll] for (uint j = 0; j < 4u; j++) {
                const uint qs = (uint)src0.Load<uint16_t>(base + 6u + j * 4u) |
                                ((uint)src0.Load<uint16_t>(base + 8u + j * 4u) << 16u);
                [unroll] for (uint e = 0; e < 4u; e++) {
                    const uint i = j * 4u + e;
                    const uint lo = ((qs >> (e * 8u)) & 15u) | (((qh >> i) & 1u) << 4u);
                    const uint hi = ((qs >> (e * 8u + 4u)) & 15u) | (((qh >> (i + 16u)) & 1u) << 4u);
                    lds_b[o + i]       = (float16_t)(sc * (float)((int)lo - 16));
                    lds_b[o + i + 16u] = (float16_t)(sc * (float)((int)hi - 16));
                }
            }
        }
#endif
#if QUANT_B == 4 || QUANT_B == 5
        // Dequantize this window of the weight tile into LDS. One thread owns
        // one 32-element sub-block, so its 6-bit scale pair is unpacked once
        // and then reused for all 32 values. That is the point of staging a
        // K-quant: MMQ redoes the nibble and scale unpack on every dp4a step.
        for (uint bq = tid; bq < (BN * FOLD_K) / Q4_K_SUB; bq += THREADS) {
            const uint n   = bq / (FOLD_K / Q4_K_SUB);
            const uint kb0 = (bq % (FOLD_K / Q4_K_SUB)) * Q4_K_SUB;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < Q4_K_SUB; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint kk = k0 + kb0;
            // Q4_K and Q5_K blocks and their row pitches are dword aligned.
#if QUANT_B == 4
            const uint bb = b_grp + gn * nb01 + (kk / QK_K) * Q4_K_BSIZE;
#else
            const uint bb = b_grp + gn * nb01 + (kk / QK_K) * Q5_K_BSIZE;
#endif
            const uint s  = (kk % QK_K) / Q4_K_SUB;

            const uint  dm       = src0.Load(bb);
            const float d_all    = f16_to_f32(dm & 0xFFFFu);
            const float dmin_all = f16_to_f32(dm >> 16);

            // 12 bytes hold eight 6-bit (scale, min) pairs.
            const uint3 sw = uint3(src0.Load(bb + 4u), src0.Load(bb + 8u),
                                   src0.Load(bb + 12u));
            uint sc6, m6;
            if (s < 4u) {
                sc6 = q4k_sbyte(sw, s)      & 63u;
                m6  = q4k_sbyte(sw, s + 4u) & 63u;
            } else {
                const uint hi = q4k_sbyte(sw, s + 4u);
                sc6 = (hi & 0xFu) | ((q4k_sbyte(sw, s - 4u) >> 6) << 4);
                m6  = (hi >> 4)   | ((q4k_sbyte(sw, s)      >> 6) << 4);
            }
            const float ds = d_all * (float)sc6;
            const float ms = dmin_all * (float)m6;

            // Two sub-blocks share 32 quant bytes: low nibbles are the even
            // one, high nibbles the odd one.
#if QUANT_B == 4
            const uint qbase = bb + 16u + (s >> 1) * 32u;
#else
            const uint qbase = bb + 48u + (s >> 1) * 32u;
#endif
            const uint shift = (s & 1u) * 4u;
            [unroll] for (uint j = 0; j < Q4_K_SUB / 4; j++) {
                const uint q4 = src0.Load(qbase + j * 4u);
                const uint ob = o + j * 4u;
#if QUANT_B == 4
                lds_b[ob + 0u] = (float16_t)(ds * (float)((q4 >> (shift       )) & 0xFu) - ms);
                lds_b[ob + 1u] = (float16_t)(ds * (float)((q4 >> (shift +  8u)) & 0xFu) - ms);
                lds_b[ob + 2u] = (float16_t)(ds * (float)((q4 >> (shift + 16u)) & 0xFu) - ms);
                lds_b[ob + 3u] = (float16_t)(ds * (float)((q4 >> (shift + 24u)) & 0xFu) - ms);
#else
                const uint h4 = src0.Load(bb + 16u + j * 4u);
                [unroll] for (uint e = 0; e < 4u; e++) {
                    const uint q = ((q4 >> (shift + e * 8u)) & 15u) | (((h4 >> (s + e * 8u)) & 1u) << 4u);
                    lds_b[ob + e] = (float16_t)(ds * (float)q - ms);
                }
#endif
            }
        }
#endif
#if QUANT_B == 6
        for (uint bq = tid; bq < (BN * FOLD_K) / 32u; bq += THREADS) {
            const uint n   = bq / (FOLD_K / 32u);
            const uint kb0 = (bq % (FOLD_K / 32u)) * 32u;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < 32u; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint kk = k0 + kb0;
            const uint bb = b_grp + gn * nb01 + (kk / 256u) * 210u;
            const uint hf = (kk % 256u) / 128u;
            const uint s  = (kk % 128u) / 32u;
            const uint ql = bb + hf * 64u + (s & 1u) * 32u;
            const uint qh = bb + 128u + hf * 32u;
            const uint scales = src0.Load<uint16_t>(bb + 192u + hf * 8u + s * 2u);
            const float d = f16_to_f32(src0.Load<uint16_t>(bb + 208u));
            const float2 ds = d * float2((int)((scales & 255u) ^ 128u) - 128,
                                        (int)((scales >> 8u) ^ 128u) - 128);
            [unroll] for (uint j = 0; j < 8u; j++) {
                const uint l4 = iw_load_u32(ql + j * 4u);
                const uint h4 = iw_load_u32(qh + j * 4u);
                const float sc = j < 4u ? ds.x : ds.y;
                [unroll] for (uint e = 0; e < 4u; e++) {
                    const uint lo = (l4 >> (e * 8u + (s / 2u) * 4u)) & 15u;
                    const uint hi = (h4 >> (e * 8u + s * 2u)) & 3u;
                    lds_b[o + j * 4u + e] = (float16_t)(sc * (float)((int)(lo | (hi << 4u)) - 32));
                }
            }
        }
#endif
#if QUANT_B == 40 || QUANT_B == 41 || QUANT_B == 51 || QUANT_B == 14
        for (uint bq = tid; bq < (BN * FOLD_K) / 32u; bq += THREADS) {
            const uint n   = bq / (FOLD_K / 32u);
            const uint kb0 = (bq % (FOLD_K / 32u)) * 32u;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < 32u; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint bb = b_grp + gn * nb01 + ((k0 + kb0) / 32u) * IW_BLOCK_BYTES;
            const float d = f16_to_f32(src0.Load<uint16_t>(bb));
#if QUANT_B == 41 || QUANT_B == 51
            const float m = f16_to_f32(src0.Load<uint16_t>(bb + 2u));
#endif
#if QUANT_B == 51
            const uint qh = iw_load_u32(bb + 4u);
            const uint qs = bb + 8u;
#elif QUANT_B == 41
            const uint qs = bb + 4u;
#else
            const uint qs = bb + 2u;
#endif
            [unroll] for (uint j = 0; j < 4u; j++) {
                const uint q4 = iw_load_u32(qs + j * 4u);
                [unroll] for (uint e = 0; e < 4u; e++) {
                    const uint i = j * 4u + e;
                    const uint lo = (q4 >> (e * 8u)) & 15u;
                    const uint hi = (q4 >> (e * 8u + 4u)) & 15u;
#if QUANT_B == 40
                    lds_b[o + i]       = (float16_t)(d * (float)((int)lo - 8));
                    lds_b[o + i + 16u] = (float16_t)(d * (float)((int)hi - 8));
#elif QUANT_B == 14
                    lds_b[o + i]       = (float16_t)(d * (float)iw_quant_value(lo));
                    lds_b[o + i + 16u] = (float16_t)(d * (float)iw_quant_value(hi));
#elif QUANT_B == 51
                    lds_b[o + i]       = (float16_t)(d * (float)(lo | (((qh >> i) & 1u) << 4u)) + m);
                    lds_b[o + i + 16u] = (float16_t)(d * (float)(hi | (((qh >> (i + 16u)) & 1u) << 4u)) + m);
#else
                    lds_b[o + i]       = (float16_t)(d * (float)lo + m);
                    lds_b[o + i + 16u] = (float16_t)(d * (float)hi + m);
#endif
                }
            }
        }
#endif
#if QUANT_B == 17
        // Pair 17-byte blocks so exact halfword loads need no tensor padding.
        for (uint bq = tid; bq < (BN * FOLD_K) / 64u; bq += THREADS) {
            const uint n   = bq / (FOLD_K / 64u);
            const uint kb0 = (bq % (FOLD_K / 64u)) * 64u;
            const uint gn  = colg + n;
            const uint o   = n * FOLD_K + kb0;
            if (gn >= ne0) {
                [unroll] for (uint z = 0; z < 64u; z++) {
                    lds_b[o + z] = (float16_t)0;
                }
                continue;
            }
            const uint bb = b_grp + gn * nb01 + ((k0 + kb0) / 32u) * 17u;
            uint words[17];
            [unroll] for (uint j = 0; j < 17u; j++) {
                words[j] = src0.Load<uint16_t>(bb + j * 2u);
            }
            [unroll] for (uint b = 0; b < 2u; b++) {
                const uint exponent = b == 0u ? (words[0] & 255u) : (words[8] >> 8u);
                const float d = asfloat(exponent < 2u ? (0x00200000u << exponent) : ((exponent - 1u) << 23u));
                [unroll] for (uint j = 0; j < 16u; j++) {
                    const uint off = b * 17u + 1u + j;
                    const uint q = (words[off / 2u] >> ((off & 1u) * 8u)) & 255u;
                    lds_b[o + b * 32u + j]       = (float16_t)(d * (float)iw_quant_value(q & 15u));
                    lds_b[o + b * 32u + j + 16u] = (float16_t)(d * (float)iw_quant_value(q >> 4u));
                }
            }
        }
#endif
#if IW_STAGED
        GroupMemoryBarrierWithGroupSync();
        if (!active) {
            continue;
        }
#endif
        MatAccf acc[MTILE][NTILE];
        [unroll] for (uint mi = 0; mi < MTILE; mi++) {
            [unroll] for (uint ni = 0; ni < NTILE; ni++) {
                acc[mi][ni] = MatAccf::Splat((float16_t)0.0);
            }
        }
        [unroll] for (uint f = 0; f < FOLD_BLOCKS; f++) {
#if (!IW_QUANT_B) || (LDS_STAGE_A == 0)
            const uint kb = (k0 + f * LA_K) * 2u;
#endif
            // Every A tile feeds NTILE weight tiles and every weight tile
            // feeds MTILE row tiles, so MTILE and NTILE set the reuse.
            //
            // F16 weights come straight from the descriptor: each wave owns
            // its own columns, so there is no redundancy to remove. Quantized
            // weights have to be staged because the matrix load only speaks
            // F16.
            //
            // The weight tiles are named locals, not an array. Holding them in
            // a Matrix array and reading it back in the mj loop miscompiles on
            // this driver: whole output rows come back unwritten once the tile
            // count is odd. Same class of failure the accumulator hit during
            // bring-up (see the note at the top of this file).
#if IW_QUANT_B
            const uint bw = wave_n * (LA_N * NTILE) * FOLD_K + f * LA_K;
            MatBf b0 = MatBf::Load(lds_b, bw, FOLD_K, MatrixLayout::ColMajor);
#if NTILE > 1
            MatBf b1 = MatBf::Load(lds_b, bw + LA_N * FOLD_K,
                                   FOLD_K, MatrixLayout::ColMajor);
#endif
#else
            MatBf b0 = MatBf::Load(src0, b_base + kb,
                                   nb01, MatrixLayout::ColMajor, 4u);
#if NTILE > 1
            MatBf b1 = MatBf::Load(src0, b_base + LA_N * nb01 + kb,
                                   nb01, MatrixLayout::ColMajor, 4u);
#endif
#endif
            [unroll] for (uint mj = 0; mj < MTILE; mj++) {
#if LDS_STAGE_A
                MatAf a = MatAf::Load(lds_a,
                                      (wave_m * MTILE + mj) * LA_M * FOLD_K + f * LA_K,
                                      FOLD_K, MatrixLayout::RowMajor);
#else
                MatAf a = MatAf::Load(src1, a_base + mj * LA_M * act_pitch + kb,
                                      act_pitch, MatrixLayout::RowMajor, 4u);
#endif
                acc[mj][0].MultiplyAccumulate(a, b0);
#if NTILE > 1
                acc[mj][1].MultiplyAccumulate(a, b1);
#endif
            }
        }
        [unroll] for (uint mk = 0; mk < MTILE; mk++) {
            [unroll] for (uint nk = 0; nk < NTILE; nk++) {
                [unroll] for (uint e = 0; e < ACC_E; e++) {
#if DRAIN_PROBE
                    // Probe only: prices the f32 drain. D3D12 LinAlg has no
                    // f16xf16->f32 shape on this part, so the f16 accumulator
                    // must be drained to f32. Draining on the last window only
                    // keeps every load and MAC live but gives wrong results.
                    if (k0 + FOLD_K >= ne00)
#endif
                    sum[mk][nk][e] += (float)acc[mk][nk].Get(e);
                }
            }
        }
    }

    MatAccf coords = MatAccf::Splat((float16_t)0.0);
    [unroll] for (uint mo = 0; mo < MTILE; mo++) {
        [unroll] for (uint no = 0; no < NTILE; no++) {
            [unroll] for (uint o = 0; o < ACC_E; o++) {
                const uint2 rc = coords.GetCoordinate(o);
                const uint row = row0 + mo * LA_M + rc.x;
                // The M tail is read, not gated: the scratch carries BM spare
                // rows so the load stays in bounds. Only the store has to be
                // held back, and a wave with no columns stores nothing.
                if (active && row < ne1) {
                    const uint d_off = offset_4d(col0 + no * LA_N + rc.y, row,
                                                 i2, i3, nb0, nb1, nb2, nb3,
                                                 dst_offset);
#if IW_FUSED_BIAS
                    float value = sum[mo][no][o];
                    if (op0 == 1u) {
                        value += asfloat(src2.Load(op1 + (col0 + no * LA_N + rc.y) * op2));
                    }
                    store_auto(dst, d_off, value, nb0);
#else
                    store_auto(dst, d_off, sum[mo][no][o], nb0);
#endif
                }
            }
        }
    }
}
