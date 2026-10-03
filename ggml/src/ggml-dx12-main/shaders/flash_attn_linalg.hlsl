// flash_attn_linalg.hlsl - prefill Flash Attention on SM 6.10 wave matrices.
//
// The scalar tiled kernel computes QK^T one dot() at a time and carries only
// 8-16 query rows per group, which measures ~1.9 TFLOP/s on RDNA4 - a few
// percent of the card. Both products here are matrix ops instead:
//
//   S = Q  @ K^T   (FA_BR x FA_BC), or K @ Q^T stored transposed
//   O = P  @ V     (FA_BR x FA_D), accumulated over KV tiles
//
// Each wave owns 16x16x16 matrix tiles with F16 operands and F32 accumulation.
//
// S lands in LDS as f32 straight out of the accumulator. Everything that is
// not a matrix multiply - scale, softcap, mask, ALiBi slope, online softmax,
// sinks - then runs over that array with the row and column known, which is
// how this keeps feature parity with the scalar kernel. P is written back as
// f16 and re-loaded as an A fragment, the same round trip Vulkan's
// flash_attn_cm1.comp makes through Psh, because neither API lets an
// accumulator be fed back in as a multiplicand.
//
// PV results pass through wave-private LDS slots. Persistent output stays in F32 registers for per-row softmax correction.
#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef WAVE_SIZE
#error "flash_attn_linalg requires -D WAVE_SIZE"
#endif
#ifndef FA_D
#error "flash_attn_linalg requires -D FA_D (head dim, multiple of 16)"
#endif
#ifndef FA_O_SPLITS
#define FA_O_SPLITS 1
#endif
#ifndef FA_ROW_WAVE
#define FA_ROW_WAVE 0
#endif
#ifndef FA_MASK_WAVE
#define FA_MASK_WAVE 0
#endif
#ifndef FA_PV_F16
#define FA_PV_F16 0
#endif
#ifndef FA_COMPACT_KV
#define FA_COMPACT_KV 0
#endif
#define FA_ROW_OWNED (FA_COMPACT_KV && !FA_PV_F16)

#define TILE 16

// Quantised K/V cache. Only the LDS staging step differs from the F16 path:
// the tiles land in LDS as float16_t either way, so the wave-matrix core below
// is shared. FA_KV_Q8_0 / FA_KV_Q4_0 select the block layout at compile time,
// matching how flash_attn_cd specialises its decode kernel.
#if defined(FA_KV_Q8_0) || defined(FA_KV_Q4_0)
#define FA_KV_QUANT 1
#if defined(FA_KV_Q8_0)
#define FA_KV_QK    32u
#define FA_KV_BLK   34u
#else
#define FA_KV_QK    32u
#define FA_KV_BLK   18u
#endif
// row_base is the byte offset of the row start; d indexes head-dim elements.
float fa_kv_load(ByteAddressBuffer buf, uint row_base, uint d) {
    const uint blk  = row_base + (d / FA_KV_QK) * FA_KV_BLK;
    const uint elem = d % FA_KV_QK;
    const float dscale = f16_to_f32((buf.Load(blk & ~3u) >> ((blk & 2u) * 8u)) & 0xFFFFu);
#if defined(FA_KV_Q8_0)
    const uint qo = blk + 2u + elem;
    const uint q  = (buf.Load(qo & ~3u) >> ((qo & 3u) * 8u)) & 0xFFu;
    return dscale * (float)((int)(q << 24) >> 24);
#else
    const uint qo = blk + 2u + (elem % 16u);
    const uint b  = (buf.Load(qo & ~3u) >> ((qo & 3u) * 8u)) & 0xFFu;
    const int  q  = (elem < 16u) ? ((int)(b & 0x0Fu) - 8) : ((int)(b >> 4) - 8);
    return dscale * (float)q;
#endif
}
#endif

#if defined(FA_INTEL_WAVE)
#include "flash_attn_linalg_wave_i.hlsli"
#else

// Tile shape. Every per-KV-tile cost - 11 group barriers, restaging K and V,
// and the FA_BR*FA_D read-modify-write of the output accumulator - is
// amortised over FA_BC keys, so FA_BC wants to be as wide as the 32 KB D3D12
// allows for groupshared. Trading query rows for keys is the way to buy that
// at narrow head dims.
#ifndef FA_BC
#define FA_BC 32
#endif

#ifndef FA_BR
#if FA_D <= 96
#define FA_BR 64
#else
#define FA_BR 32
#endif
#endif

// Tiles per wave. One 16x16 tile per wave leaves each wave with only FA_DBLK
// back-to-back MMAs and no way to hide their latency; taking two doubles the
// work per wave, halves the wave count (so every group barrier is cheaper) and
// lets one K fragment feed both accumulators. D=64 also has a 32-row,
// one-tile NVIDIA variant because its lower register pressure is required for
// PSO creation.
#ifndef FA_TPW
#if FA_D <= 96
#define FA_TPW 2
#else
// D=128 already runs at FA_BR=32, so halving the wave count again leaves too
// few waves per group to cover the memory latency - it measures 12% slower.
#define FA_TPW 1
#endif
#endif

#define FA_RBLK (FA_BR / TILE)
#define FA_CBLK (FA_BC / TILE)
#define FA_DBLK (FA_D / TILE)
#define FA_ODBLK (FA_DBLK / FA_O_SPLITS)
#define FA_STILES (FA_RBLK * FA_CBLK)
#define FA_QK_WAVES (FA_STILES / FA_TPW)
// Extra waves duplicate complete QK tiles into unused staging slots. This
// keeps matrix operations uniform while allowing more waves to split PV.
#ifndef FA_NWAVE
#define FA_NWAVE FA_QK_WAVES
#endif
#define NWAVE FA_NWAVE
#define FA_OTILES (FA_RBLK * FA_ODBLK)
#define THREADS (NWAVE * WAVE_SIZE)

#if (FA_STILES % FA_TPW) != 0 || (FA_QK_WAVES % FA_CBLK) != 0
#error "flash_attn_linalg: FA_TPW must divide the S tiles and keep cblk fixed"
#endif
#if (NWAVE % FA_QK_WAVES) != 0
#error "flash_attn_linalg: extra waves must duplicate complete score tiles"
#endif
#if (FA_DBLK % FA_O_SPLITS) != 0
#error "flash_attn_linalg: FA_O_SPLITS must divide the output D tiles"
#endif

// The output accumulator lives in registers, not groupshared. It is the
// largest thing a flash-attention group carries (FA_BR*FA_D floats), and in
// LDS it both halved the number of groups resident per CU and cost a
// read-modify-write of the whole block per KV tile. The PV result still has to
// land in LDS - dx::linalg exposes no mapping from an accumulator element to
// its (row, col) - but only through the small s_tmp staging tile, which is
// read straight into the registers below. Each (r, d) is owned by exactly one
// lane of one wave, so the per-row softmax correction stays exact.
#define O_TPW  (FA_OTILES / NWAVE)         // o-tiles handled per wave
#define O_EPT  ((TILE * TILE) / WAVE_SIZE) // elements per lane per o-tile
#define O_REGS (O_TPW * O_EPT)
#if FA_COMPACT_KV && !FA_PV_F16
// Keep each lane's output values in one row to share correction and normalization.
#define O_ELEM(e) (lane * O_EPT + (e))
#else
#define O_ELEM(e) ((e) * WAVE_SIZE + lane)
#endif

// Score elements per thread. The mask scan and elementwise pass use the same
// stride, so each thread reuses the mask values it loaded.
#define MCOUNT ((FA_BR * FA_BC) / THREADS)

// Row-scan partitioning: NPART threads cooperate on one query row, each
// covering FA_CPT contiguous score columns. D=96 is nearly out of LDS, so its
// partial buffer is capped and the scans there run at half group width.
#if FA_D == 96
#define NPART_MAX 4
#else
#define NPART_MAX 8
#endif
#define NPART  ((THREADS / FA_BR) < NPART_MAX ? (THREADS / FA_BR) : NPART_MAX)
#define FA_CPT (FA_BC / NPART)

// Full group width for the exponential pass. When the partial buffer cannot
// span that (D=96 is nearly out of LDS) the original one-thread-per-row scans
// are used instead, which measured faster there than a half-width partition.
#define EPART  (THREADS / FA_BR)

#if NPART == EPART
#define SPART_LEN (FA_BR * NPART)
#else
#define SPART_LEN 1
#endif

#if (THREADS % FA_BR) != 0 || (FA_BC % NPART) != 0
#error "flash_attn_linalg: row scan partitioning does not divide evenly"
#endif

// Every wave must take the same number of o-tiles: the PV loop carries a group
// barrier, and an uneven split both deadlocks and drops tiles on the floor.
#if (FA_OTILES % NWAVE) != 0
#error "FA_OTILES must be a multiple of NWAVE - pick a different FA_BR/FA_BC"
#endif

// A wave takes a contiguous run of o-tiles so its row block stays fixed; that
// only holds if the run cannot straddle a row-block boundary.
#if (FA_DBLK % O_TPW) != 0
#error "FA_DBLK must be a multiple of O_TPW - pick a different FA_BR/FA_BC"
#endif

// Scores live in s_tmp, in the wave-tiled layout the matrix Store produces,
// rather than in a separate row-major array. Three FA_BR*FA_BC buffers do not
// fit alongside a wide tile, and the elementwise pass reads and writes the
// same element, so no repacking is needed - only the two row scans have to
// address through SIDX.
#define SIDX(r, c) ((((r) / TILE) * FA_CBLK + ((c) / TILE)) * (TILE * TILE) \
                    + ((r) % TILE) * TILE + ((c) % TILE))

// V is fetched here rather than after the softmax: the loads have nothing to
// do with the QK product or the row scans, so issuing them early lets the
// whole of that work hide their latency. Only the wide path prefetches - the
// scalar fallback would need four times the registers.
#define FA_VQ_ITERS ((FA_BC * (FA_D / 4)) / THREADS)
#if ((FA_BC * (FA_D / 4)) % THREADS) != 0
#error "flash_attn_linalg: V prefetch does not divide evenly across threads"
#endif

// The transposed-QK variant keeps K in its natural [c][d] layout. The
// baseline stages K^T and pads its stride to break up the transposing write's
// LDS bank pattern.
#if defined(FA_QK_TRANSPOSED)
#define FA_KSTRIDE FA_D
#define FA_KVLEN   (FA_BC * FA_D)
#else
#define FA_KSTRIDE (FA_BC + 1)
#define FA_KVLEN   ((FA_D * FA_KSTRIDE) > (FA_BC * FA_D) \
                    ? (FA_D * FA_KSTRIDE) : (FA_BC * FA_D))
#endif

groupshared float16_t s_kv[FA_KVLEN];

// Q is only read to build the A fragments. Those are hoisted into registers,
// so Q is dead for the rest of the kernel and the P tile shares its storage -
// worth FA_BR * min(FA_D, FA_BC) halves of groupshared.

// Only one TILE-wide column block of Q is live at a time, so the staging area
// never has to hold the whole FA_BR x FA_D tile.
#define FA_QPLEN ((FA_BR * TILE) > (FA_BR * FA_BC) ? (FA_BR * TILE) : (FA_BR * FA_BC))
#if FA_COMPACT_KV
#if FA_D != 128 || FA_BR != 32 || FA_BC != 64 || FA_QPLEN > FA_KVLEN || defined(FA_KV_QUANT)
#error "Compact staging requires aligned D128 F16 tiles"
#endif
// Direct V leaves the K tile free for Q and P staging.
#define s_q s_kv
#define s_p s_kv
#else
groupshared float16_t s_qp[FA_QPLEN];
#define s_q s_qp                            // [r][d], preamble only
#define s_p s_qp                            // [r][c]
#endif

// PV staging is wave-private. Selected variants use one slot to reduce LDS and improve residency.
#ifndef FA_PV_SLOTS
#define FA_PV_SLOTS 2
#endif
#define PV_SLOTS FA_PV_SLOTS
#if FA_ROW_OWNED && (!FA_ROW_WAVE || WAVE_SIZE != 64 || NWAVE != 4 || FA_O_SPLITS != 1 || (O_TPW % PV_SLOTS) != 0)
#error "Register row ownership requires complete compact wave64 PV phases"
#endif
#define S_TMP_TILES (FA_STILES > (NWAVE * PV_SLOTS) ? FA_STILES : (NWAVE * PV_SLOTS))

groupshared float     s_tmp[S_TMP_TILES * TILE * TILE];  // scores, then PV staging
#if !FA_ROW_OWNED
groupshared float     s_gmax[FA_BR];
groupshared float     s_gsum[FA_BR];
groupshared float     s_corr[FA_BR];
#endif
groupshared float     s_part[SPART_LEN];    // per-thread row-scan partials
groupshared uint      s_any;
#if FA_MASK_WAVE
groupshared uint s_mask_wave[NWAVE];
#endif
#if FA_ROW_WAVE && (NPART != EPART || (WAVE_SIZE % NPART) != 0)
#error "Wave row reductions need complete row partitions within each wave"
#endif

typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A, MatrixScope::Wave>           MatA;
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B, MatrixScope::Wave>           MatB;
typedef Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;
#if FA_PV_F16
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatPV;
#else
typedef MatAcc MatPV;
#endif
#if defined(FA_QK_TRANSPOSED)
typedef MatB MatQ;
#else
typedef MatA MatQ;
#endif

#if FA_PV_F16
MatB fa_load_v(uint tile_start, uint dblk, uint kc, uint kv_head, uint batch_idx, bool direct_v) {
    if (direct_v) {
        const uint base = op0 + (tile_start + kc * TILE) * op2 + kv_head * op3 + batch_idx * op4 + dblk * TILE * 2u;
        return MatB::Load(src2, base, op2, MatrixLayout::RowMajor, 32u);
    }
    return MatB::Load(s_kv, kc * TILE * FA_D + dblk * TILE, FA_D, MatrixLayout::RowMajor);
}
#endif

WAVE_SIZE_ATTR
[numthreads(THREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid  = gtid.x;
    const uint wave = tid / WAVE_SIZE;
    const uint lane = tid % WAVE_SIZE;

#if defined(FA_Q_FORWARD)
    const uint q_start   = gid.x * FA_BR;
#else
    const uint n_qgroups = (ne01 + FA_BR - 1u) / FA_BR;
    const uint q_start   = (n_qgroups - 1u - gid.x) * FA_BR;
#endif
    const uint head_idx  = gid.y;
    const uint n_splits  = max(op15 & 0xFFFFu, 1u);
    const uint output_split = gid.z % FA_O_SPLITS;
    const uint logical_z    = gid.z / FA_O_SPLITS;
    const uint batch_idx = logical_z / n_splits;
    const uint split_idx = logical_z % n_splits;

    const uint D          = ne00;
    const uint D_v        = op5 >> 8;
    const uint N_queries  = ne01;
    const uint N_kv       = ne11;
    const uint n_heads    = ne02;
    const uint n_kv_heads = ne12;
    const uint kv_head    = head_idx * n_kv_heads / n_heads;

    if (q_start >= N_queries) {
        return;
    }

    const uint src2_off = op0;
    const uint src2_nb0 = op1;
    const uint src2_nb1 = op2;
    const uint src2_nb2 = op3;
    const uint src2_nb3 = op4;
    const uint src2_es  = op5 & 0xFFu;

    const uint mask_info = op8;
    const uint has_mask  = mask_info & 1u;
    const uint has_sinks = (mask_info >> 24) & 1u;
    const uint mask_nb0  = (mask_info >> 8) & 0xFFu;
    const uint mask_es   = (mask_info >> 16) & 0xFFu;
    const uint mask_off  = op9;
    const uint mask_nb1  = op10;
    const uint mask_nb2  = op11;
    const uint mask_nb3  = op12;
    const uint mask_ne2  = op13 & 0xFFFFu;
    const uint mask_ne3  = (op13 >> 16) & 0xFFFFu;

    const float scale         = asfloat(op6);
    const float logit_softcap = asfloat(op7);
    const float max_bias      = asfloat(op14);
    const float neg_max       = -3.402823466e+38f;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint n_head_log2 = (n_heads > 0u) ? (1u << firstbithigh(n_heads)) : 1u;
        float n_head_log2_f = (float)n_head_log2;
        float m0 = exp2(-max_bias * 0.5f / n_head_log2_f * 2.0f);
        float m1 = exp2(-max_bias * 0.5f / n_head_log2_f);
        if (head_idx < n_head_log2) {
            slope = pow(m0, (float)(head_idx + 1u));
        } else {
            slope = pow(m1, (float)(2u * (head_idx - n_head_log2) + 1u));
        }
    }

    float o_reg[O_REGS];
    [unroll] for (uint oi = 0; oi < O_REGS; ++oi) {
        o_reg[oi] = 0.0f;
    }
#if FA_ROW_OWNED
    const uint owned_row = tid / NPART;
    const uint owned_col = tid % NPART;
    float owned_max = neg_max;
    float owned_sum = 0.0f;
#else
    if (tid < FA_BR) {
        s_gmax[tid] = neg_max;
        s_gsum[tid] = 0.0f;
    }
#endif

    // Q never changes, so its A fragments are loaded once instead of once per
    // KV tile. Staging one TILE-wide column block at a time keeps the shared
    // buffer small enough to share with the P tile, which is what lets the
    // group fit three-to-a-CU instead of two.
    MatQ q_frag[FA_TPW][FA_DBLK];
    {
        [unroll] for (uint k = 0; k < FA_DBLK; ++k) {
            GroupMemoryBarrierWithGroupSync();
            for (uint idx = tid; idx < FA_BR * TILE; idx += THREADS) {
                const uint r = idx / TILE;
                const uint d = k * TILE + (idx % TILE);
                const uint query_idx = q_start + r;
                float qv = 0.0f;
                if (query_idx < N_queries && d < D) {
                    const uint q_base = src0_offset + query_idx * nb01
                                      + head_idx * nb02 + batch_idx * nb03;
                    qv = asfloat(src0.Load(q_base + d * 4u));
                }
                s_q[idx] = (float16_t)qv;
            }
            GroupMemoryBarrierWithGroupSync();
            [unroll] for (uint j = 0; j < FA_TPW; ++j) {
                const uint score_tile = (wave % FA_QK_WAVES) + j * FA_QK_WAVES;
                const uint rblk = score_tile / FA_CBLK;
#if defined(FA_QK_TRANSPOSED)
                q_frag[j][k] = MatB::Load(s_q, rblk * TILE * TILE, TILE,
                                          MatrixLayout::ColMajor);
#else
                q_frag[j][k] = MatA::Load(s_q, rblk * TILE * TILE, TILE,
                                          MatrixLayout::RowMajor);
#endif
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Split-KV. With 64 query rows a prefill dispatch can fall to well under
    // one group per CU (SmolLM2 at ubatch 512 gives 8 x 9 = 72 on 64 CUs,
    // which runs as two rounds with the second nearly empty). Splitting the
    // KV range multiplies the group count; each split emits an unnormalised
    // partial plus its running max and sum, and flash_attn_reduce combines
    // them with the same online-softmax update used here.
    uint kv_begin = 0u;
    uint kv_end   = N_kv;
    if (n_splits > 1u) {
        // Round the split boundary to the KV tile so no group starts mid-tile.
        const uint tiles       = (N_kv + FA_BC - 1u) / FA_BC;
        const uint tiles_split = (tiles + n_splits - 1u) / n_splits;
        kv_begin = min(split_idx * tiles_split * FA_BC, N_kv);
        kv_end   = min(kv_begin + tiles_split * FA_BC, N_kv);
    }

    const bool kv_quad =
#if FA_COMPACT_KV
                         true;
#elif defined(FA_KV_QUANT)
                         false;
#else
                         (src1_esize == 2u) && (nb10 == 2u)
                      && (src2_es == 2u) && (src2_nb0 == 2u)
                      && (FA_D == D) && (FA_D == D_v)
                      && (((nb11 | src2_nb1 | src1_offset | src2_off) & 7u) == 0u);
#endif
    const bool pipeline_k = kv_quad;
    uint2 k_pre[FA_VQ_ITERS];
    bool k_prefetched = false;
    if (pipeline_k && kv_begin < kv_end) {
        [unroll] for (uint ki = 0; ki < FA_VQ_ITERS; ++ki) {
            const uint kidx = ki * THREADS + tid;
            const uint c    = kidx / (FA_D / 4u);
            const uint d4   = (kidx % (FA_D / 4u)) * 4u;
            uint2 w = uint2(0u, 0u);
            if (c < min((uint)FA_BC, kv_end - kv_begin)) {
                const uint k_base = src1_offset + (kv_begin + c) * nb11
                                  + kv_head * nb12 + batch_idx * nb13;
                w = src1.Load2(k_base + d4 * 2u);
            }
            k_pre[ki] = w;
        }
        k_prefetched = true;
    }

    for (uint tile_start = kv_begin; tile_start < kv_end; tile_start += FA_BC) {
        const uint tile_size = min((uint)FA_BC, N_kv - tile_start);

        // Cache the mask values for softmax and skip fully masked tiles.
        const uint mhb = (head_idx % mask_ne2) * mask_nb2
                       + (batch_idx % mask_ne3) * mask_nb3;
        float mcache[MCOUNT];
        bool apply_mask = has_mask != 0u;
        if (has_mask != 0u) {
#if !FA_MASK_WAVE
            if (tid == 0) {
                s_any = 0u;
            }
            GroupMemoryBarrierWithGroupSync();
#endif

            uint any_local = 0u;
#if NPART == EPART
            const uint mpr = tid / NPART;
            const uint mpc = tid % NPART;
            const uint mq0 = q_start + mpr;
            // Each thread owns a contiguous column span, so an F16 mask row
            // can be pulled four elements at a time.
            const bool mask_quad = (mask_es == 2u) && (mask_nb0 == 2u)
                                && ((MCOUNT % 4u) == 0u)
                                && (((mask_off | mask_nb1 | mhb) & 7u) == 0u)
                                && (mq0 < N_queries)
                                && ((mpc * FA_CPT + MCOUNT) <= tile_size);
            if (mask_quad) {
                const uint mrow = mask_off + mq0 * mask_nb1 + mhb
                                + (tile_start + mpc * FA_CPT) * 2u;
                [unroll] for (uint mj = 0; mj < MCOUNT; mj += 4u) {
                    const uint2 w = src3.Load2(mrow + mj * 2u);
                    const float m0 = (float)asfloat16((uint16_t)(w.x & 0xFFFFu));
                    const float m1 = (float)asfloat16((uint16_t)(w.x >> 16));
                    const float m2 = (float)asfloat16((uint16_t)(w.y & 0xFFFFu));
                    const float m3 = (float)asfloat16((uint16_t)(w.y >> 16));
                    mcache[mj]      = m0;
                    mcache[mj + 1u] = m1;
                    mcache[mj + 2u] = m2;
                    mcache[mj + 3u] = m3;
                    if (!isinf(m0) || !isinf(m1) || !isinf(m2) || !isinf(m3)) {
                        any_local |= 1u;
                    }
#if FA_MASK_WAVE
                    if (m0 != 0.0f || m1 != 0.0f || m2 != 0.0f || m3 != 0.0f) {
                        any_local |= 2u;
                    }
#endif
                }
            } else
#endif
            {
#if NPART == EPART
            [unroll] for (uint mi = 0; mi < MCOUNT; ++mi) {
                const uint mc = mpc * FA_CPT + mi;
                const uint mq = q_start + mpr;
#else
            [unroll] for (uint mi = 0; mi < MCOUNT; ++mi) {
                const uint midx = mi * THREADS + tid;
                const uint mr = midx / FA_BC;
                const uint mc = midx % FA_BC;
                const uint mq = q_start + mr;
#endif
                float mv = 0.0f;
                if (mq < N_queries && mc < tile_size) {
                    mv = load_auto(src3,
                                   mask_off + mq * mask_nb1 + mhb
                                   + (tile_start + mc) * mask_nb0,
                                   mask_es);
                    if (!isinf(mv)) {
                        any_local |= 1u;
                    }
#if FA_MASK_WAVE
                    if (mv != 0.0f) {
                        any_local |= 2u;
                    }
#endif
                }
                mcache[mi] = mv;
            }
            }
#if FA_MASK_WAVE
            const uint wave_mask = WaveActiveBitOr(any_local);
            if (lane == 0u) {
                s_mask_wave[wave] = wave_mask;
            }
            GroupMemoryBarrierWithGroupSync();
            uint tile_mask = 0u;
            [unroll] for (uint w = 0u; w < NWAVE; ++w) {
                tile_mask |= s_mask_wave[w];
            }
            apply_mask = (tile_mask & 2u) != 0u;
            if ((tile_mask & 1u) == 0u) {
                GroupMemoryBarrierWithGroupSync();
#else
            if (any_local != 0u) {
                InterlockedOr(s_any, 1u);
            }
            GroupMemoryBarrierWithGroupSync();

            if (s_any == 0u) {
#endif
                k_prefetched = false;
                continue;
            }
        }

        // The baseline stages K^T as [d][c]. The transposed-QK variant keeps
        // K in its natural [c][d] layout. Threads walk d fastest in both cases.
        // A contiguous F16 cache lets one thread pull four head-dim elements
        // out of a single 8-byte load, cutting the load and address-math count
        // by 4x.
        // Aligned D=128 F16 K can be loaded row-major and transposed with a
        // matrix cast. D=96 is faster through the staged path. Aligned F16 V
        // can be consumed by the matrix unit directly. Partial tiles and
        // quantised caches retain the staged path.
        const bool direct_k =
#if defined(FA_KV_QUANT)
                              false;
#elif defined(FA_QK_TRANSPOSED)
                              kv_quad && (tile_size == FA_BC)
                           && (((src1_offset | nb11 | nb12 | nb13) & 31u) == 0u);
#elif defined(FA_DIRECT_K)
                              (FA_D == 128u) && kv_quad && (tile_size == FA_BC)
                           && (((src1_offset | nb11 | nb12 | nb13) & 127u) == 0u);
#else
                              false;
#endif
        const bool direct_v =
#if FA_COMPACT_KV
                              true;
#elif defined(FA_KV_QUANT)
                              false;
#else
                              kv_quad && (tile_size == FA_BC)
                           && (((src2_off | src2_nb1 | src2_nb2 | src2_nb3) & 31u) == 0u);
#endif
        uint2 v_pre[FA_VQ_ITERS];
        if (!direct_k && kv_quad) {
            [unroll] for (uint ki = 0; ki < FA_VQ_ITERS; ++ki) {
                const uint kidx = ki * THREADS + tid;
                const uint c  = kidx / (FA_D / 4u);
                const uint d4 = (kidx % (FA_D / 4u)) * 4u;
                uint2 w = uint2(0u, 0u);
                if (k_prefetched) {
                    w = k_pre[ki];
                } else if (c < tile_size) {
                    const uint k_base = src1_offset + (tile_start + c) * nb11
                                      + kv_head * nb12 + batch_idx * nb13;
                    w = src1.Load2(k_base + d4 * 2u);
                }
#if defined(FA_QK_TRANSPOSED)
                s_kv[c * FA_D + d4]      = asfloat16((uint16_t)(w.x & 0xFFFFu));
                s_kv[c * FA_D + d4 + 1u] = asfloat16((uint16_t)(w.x >> 16));
                s_kv[c * FA_D + d4 + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));
                s_kv[c * FA_D + d4 + 3u] = asfloat16((uint16_t)(w.y >> 16));
#else
                s_kv[ d4       * FA_KSTRIDE + c] = asfloat16((uint16_t)(w.x & 0xFFFFu));
                s_kv[(d4 + 1u) * FA_KSTRIDE + c] = asfloat16((uint16_t)(w.x >> 16));
                s_kv[(d4 + 2u) * FA_KSTRIDE + c] = asfloat16((uint16_t)(w.y & 0xFFFFu));
                s_kv[(d4 + 3u) * FA_KSTRIDE + c] = asfloat16((uint16_t)(w.y >> 16));
#endif
            }
        } else if (!direct_k)
        for (uint kidx = tid; kidx < FA_BC * FA_D; kidx += THREADS) {
            const uint c = kidx / FA_D;
            const uint d = kidx % FA_D;
            float kval = 0.0f;
            if (c < tile_size && d < D) {
                const uint k_base = src1_offset + (tile_start + c) * nb11
                                  + kv_head * nb12 + batch_idx * nb13;
                kval =
#if defined(FA_KV_QUANT)
                       fa_kv_load(src1, k_base, d);
#else
                       load_auto(src1, k_base + d * nb10, src1_esize);
#endif
            }
#if defined(FA_QK_TRANSPOSED)
            s_kv[c * FA_D + d] = (float16_t)kval;
#else
            s_kv[d * FA_KSTRIDE + c] = (float16_t)kval;
#endif
        }
        if (!direct_v) {
            [unroll] for (uint vi = 0; vi < FA_VQ_ITERS; ++vi) {
                const uint vidx = vi * THREADS + tid;
                const uint c    = vidx / (FA_D / 4u);
                const uint d4   = (vidx % (FA_D / 4u)) * 4u;
                uint2 w = uint2(0u, 0u);
                if (c < tile_size) {
                    const uint v_base = src2_off + (tile_start + c) * src2_nb1
                                      + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    w = src2.Load2(v_base + d4 * 2u);
                }
                v_pre[vi] = w;
            }
        }
        if (!direct_k) {
            GroupMemoryBarrierWithGroupSync();
        }
        k_prefetched = false;

        {
            const uint score_tile0 = wave % FA_QK_WAVES;
            const uint cblk = score_tile0 % FA_CBLK;

            MatAcc acc[FA_TPW];
            [unroll] for (uint j = 0; j < FA_TPW; ++j) {
                acc[j] = MatAcc::Splat(0.0f);
            }
            [unroll] for (uint k = 0; k < FA_DBLK; ++k) {
#if defined(FA_QK_TRANSPOSED)
                MatA a;
                if (direct_k) {
                    const uint k_base = src1_offset
                                      + (tile_start + cblk * TILE) * nb11
                                      + kv_head * nb12 + batch_idx * nb13
                                      + k * TILE * 2u;
                    a = MatA::Load(src1, k_base, nb11, MatrixLayout::RowMajor, 32u);
                } else {
                    a = MatA::Load(s_kv, cblk * TILE * FA_D + k * TILE, FA_D,
                                   MatrixLayout::RowMajor);
                }
                [unroll] for (uint j = 0; j < FA_TPW; ++j) {
                    acc[j].MultiplyAccumulate(a, q_frag[j][k]);
                }
#else
                MatB b;
                if (direct_k) {
                    const uint k_base = src1_offset
                                      + (tile_start + cblk * TILE) * nb11
                                      + kv_head * nb12 + batch_idx * nb13
                                      + k * TILE * 2u;
                    MatA k_row = MatA::Load(src1, k_base, nb11, MatrixLayout::RowMajor, 128u);
                    b = k_row.Cast<ComponentType::F16, MatrixUse::B, true>();
                } else {
                    b = MatB::Load(s_kv, k * TILE * FA_KSTRIDE + cblk * TILE, FA_KSTRIDE,
                                   MatrixLayout::RowMajor);
                }
                [unroll] for (uint j = 0; j < FA_TPW; ++j) {
                    acc[j].MultiplyAccumulate(q_frag[j][k], b);
                }
#endif
            }
            // The scores stay in s_tmp; the elementwise pass below reads and
            // writes each element in place.
            [unroll] for (uint j = 0; j < FA_TPW; ++j) {
                acc[j].Store(s_tmp, (wave + j * NWAVE) * TILE * TILE, TILE,
#if defined(FA_QK_TRANSPOSED)
                             MatrixLayout::ColMajor);
#else
                             MatrixLayout::RowMajor);
#endif
            }
        }
        GroupMemoryBarrierWithGroupSync();

        const uint next_tile = tile_start + FA_BC;
        if (!direct_k && pipeline_k && next_tile < kv_end) {
            const uint next_size = min((uint)FA_BC, kv_end - next_tile);
            [unroll] for (uint ki = 0; ki < FA_VQ_ITERS; ++ki) {
                const uint kidx = ki * THREADS + tid;
                const uint c    = kidx / (FA_D / 4u);
                const uint d4   = (kidx % (FA_D / 4u)) * 4u;
                uint2 w = uint2(0u, 0u);
                if (c < next_size) {
                    const uint k_base = src1_offset + (next_tile + c) * nb11
                                      + kv_head * nb12 + batch_idx * nb13;
                    w = src1.Load2(k_base + d4 * 2u);
                }
                k_pre[ki] = w;
            }
            k_prefetched = true;
        }

        // Everything that is not a matrix multiply happens here, where the
        // row and column of each score are known. In the partitioned form this
        // is fused into the max scan, so the scores are read and written once.
#if NPART != EPART
        [unroll] for (uint si = 0; si < MCOUNT; ++si) {
            const uint sidx = si * THREADS + tid;
            const uint r = sidx / FA_BC;
            const uint c = sidx % FA_BC;
            const uint slot = SIDX(r, c);

            float sv = neg_max;
            if (q_start + r < N_queries && c < tile_size) {
                float scaled = s_tmp[slot] * scale;
                if (logit_softcap != 0.0f) {
                    scaled = logit_softcap * tanh(scaled);
                }
                if (apply_mask) {
                    const float mv = mcache[si] * slope;
                    scaled = isinf(mv) ? neg_max : (scaled + mv);
                }
                sv = scaled;
            }
            s_tmp[slot] = sv;
        }
        GroupMemoryBarrierWithGroupSync();
#endif

#if FA_ROW_WAVE
        const uint pr = tid / NPART;
        const uint pc = tid % NPART;
        float scache[FA_CPT];
        float tile_max = neg_max;
        [unroll] for (uint i = 0u; i < FA_CPT; ++i) {
            const uint c = pc * FA_CPT + i;
            float sv = neg_max;
            if (q_start + pr < N_queries && c < tile_size) {
                sv = s_tmp[SIDX(pr, c)] * scale;
                if (logit_softcap != 0.0f) {
                    sv = logit_softcap * tanh(sv);
                }
                if (apply_mask) {
                    const float mv = mcache[i] * slope;
                    sv = isinf(mv) ? neg_max : sv + mv;
                }
            }
            scache[i] = sv;
            tile_max = max(tile_max, sv);
        }
        [unroll] for (uint delta = NPART / 2u; delta > 0u; delta /= 2u) {
            tile_max = max(tile_max, WaveReadLaneAt(tile_max, lane ^ delta));
        }
#if FA_ROW_OWNED
        float gm = owned_max;
        float row_sum = owned_sum;
#else
        float gm = s_gmax[pr];
        float row_sum = s_gsum[pr];
#endif
        float corr = 1.0f;
        if (q_start + pr < N_queries && tile_max != neg_max) {
            const float new_max = max(gm, tile_max);
            corr = row_sum > 0.0f ? exp(gm - new_max) : 0.0f;
            gm = new_max;
            row_sum *= corr;
        }
        float psum = 0.0f;
        [unroll] for (uint i = 0u; i < FA_CPT; ++i) {
            const uint c = pc * FA_CPT + i;
            const float p = scache[i] == neg_max ? 0.0f : exp(scache[i] - gm);
#if FA_COMPACT_KV
            // Round subnormal probabilities explicitly; native packed conversion truncates them.
            const uint rounded = (uint)round(min(p, 6.103515625e-5f) * 16777216.0f);
            s_p[pr * FA_BC + c] = asfloat16((uint16_t)(p < 6.103515625e-5f ? rounded : f32tof16(p)));
#else
            s_p[pr * FA_BC + c] = (float16_t)p;
#endif
            psum += p;
        }
        [unroll] for (uint delta = NPART / 2u; delta > 0u; delta /= 2u) {
            psum += WaveReadLaneAt(psum, lane ^ delta);
        }
#if FA_ROW_OWNED
        owned_max = gm;
        owned_sum = row_sum + psum;
#else
        if (pc == 0u) {
            s_gmax[pr] = gm;
            s_gsum[pr] = row_sum + psum;
            s_corr[pr] = corr;
        }
#endif
        GroupMemoryBarrierWithGroupSync();
#else
        // Online softmax. NPART threads scan each query row in parallel and
        // reduce through s_part; one thread per row then updates the running
        // max, which keeps the whole group busy instead of only FA_BR lanes.
#if NPART == EPART
        // The scaled scores stay in registers between the two passes, so the
        // score tile is read out of LDS once and never written back.
        float scache[FA_CPT];
        if (tid < FA_BR * NPART) {
            const uint pr = tid / NPART;
            const uint pc = tid % NPART;
            float pm = neg_max;
            [unroll] for (uint i = 0; i < FA_CPT; ++i) {
                const uint c    = pc * FA_CPT + i;
                const uint slot = SIDX(pr, c);

                float sv = neg_max;
                if (q_start + pr < N_queries && c < tile_size) {
                    float scaled = s_tmp[slot] * scale;
                    if (logit_softcap != 0.0f) {
                        scaled = logit_softcap * tanh(scaled);
                    }
                    if (apply_mask) {
                        const float mv = mcache[i] * slope;
                        scaled = isinf(mv) ? neg_max : (scaled + mv);
                    }
                    sv = scaled;
                }
                scache[i] = sv;
                pm = max(pm, sv);
            }
            s_part[tid] = pm;
        }
        GroupMemoryBarrierWithGroupSync();
#endif

        if (tid < FA_BR) {
#if NPART == EPART
            float tile_max = neg_max;
            [unroll] for (uint k = 0; k < NPART; ++k) {
                tile_max = max(tile_max, s_part[tid * NPART + k]);
            }
#else
            float tile_max = neg_max;
            for (uint c = 0; c < tile_size; ++c) {
                tile_max = max(tile_max, s_tmp[SIDX(tid, c)]);
            }
#endif
            if (q_start + tid < N_queries && tile_max != neg_max) {
                const float old_max = s_gmax[tid];
                const float new_max = max(old_max, tile_max);
                const float corr = (s_gsum[tid] > 0.0f) ? exp(old_max - new_max) : 0.0f;
                s_gmax[tid] = new_max;
                s_gsum[tid] *= corr;
                s_corr[tid] = corr;
            } else {
                s_corr[tid] = 1.0f;
            }
        }
        GroupMemoryBarrierWithGroupSync();

#if NPART == EPART
        // Exponentiate with the same row partitioning, so each thread can also
        // accumulate its slice of the row sum and no second scan is needed.
        if (tid < FA_BR * NPART) {
            const uint pr = tid / NPART;
            const uint pc = tid % NPART;
            const float gm = s_gmax[pr];
            float psum = 0.0f;
            [unroll] for (uint i = 0; i < FA_CPT; ++i) {
                const uint c    = pc * FA_CPT + i;
                const float sv  = scache[i];
                const float p   = (sv == neg_max) ? 0.0f : exp(sv - gm);
                s_p[pr * FA_BC + c] = (float16_t)p;
                psum += p;
            }
            s_part[tid] = psum;
        }
#else
        for (uint pidx = tid; pidx < FA_BR * FA_BC; pidx += THREADS) {
            const uint r    = pidx / FA_BC;
            const uint slot = SIDX(r, pidx % FA_BC);
            const float sv = s_tmp[slot];
            const float p = (sv == neg_max) ? 0.0f : exp(sv - s_gmax[r]);
            s_p[pidx]   = (float16_t)p;
            s_tmp[slot] = p;
        }
#endif
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_BR) {
#if NPART == EPART
            float row_sum = 0.0f;
            [unroll] for (uint k = 0; k < NPART; ++k) {
                row_sum += s_part[tid * NPART + k];
            }
#else
            float row_sum = 0.0f;
            for (uint c = 0; c < tile_size; ++c) {
                row_sum += s_tmp[SIDX(tid, c)];
            }
#endif
            s_gsum[tid] += row_sum;
        }
#endif

        // The fallback reuses the K tile for V in its natural [c][d] layout.
        if (!direct_v && kv_quad) {
            [unroll] for (uint vi = 0; vi < FA_VQ_ITERS; ++vi) {
                const uint vidx = vi * THREADS + tid;
                const uint c    = vidx / (FA_D / 4u);
                const uint d4   = (vidx % (FA_D / 4u)) * 4u;
                const uint2 w   = v_pre[vi];
                s_kv[c * FA_D + d4]      = asfloat16((uint16_t)(w.x & 0xFFFFu));
                s_kv[c * FA_D + d4 + 1u] = asfloat16((uint16_t)(w.x >> 16));
                s_kv[c * FA_D + d4 + 2u] = asfloat16((uint16_t)(w.y & 0xFFFFu));
                s_kv[c * FA_D + d4 + 3u] = asfloat16((uint16_t)(w.y >> 16));
            }
        } else if (!direct_v) {
            for (uint vidx = tid; vidx < FA_BC * FA_D; vidx += THREADS) {
                const uint c = vidx / FA_D;
                const uint d = vidx % FA_D;
                float vval = 0.0f;
                if (c < tile_size && d < D_v) {
                    const uint v_base = src2_off + (tile_start + c) * src2_nb1
                                      + kv_head * src2_nb2 + batch_idx * src2_nb3;
                    vval =
#if defined(FA_KV_QUANT)
                           fa_kv_load(src2, v_base, d);
#else
                           load_auto(src2, v_base + d * src2_nb0, src2_es);
#endif
                }
                s_kv[vidx] = (float16_t)vval;
            }
        }
        if (!direct_v) {
            GroupMemoryBarrierWithGroupSync();
        }

        // O += P @ V, with the previous tile's contribution rescaled by the
        // online-softmax correction. Each (r, d) element is touched by exactly
        // one o-tile, so the correction lands once per row per KV tile.
        // Tiles are handed out in contiguous blocks so that a wave's rblk is
        // fixed, letting the P fragments be loaded once and reused across the
        // wave's D tiles instead of being re-read for every one.
        const uint rblk = (wave * O_TPW) / FA_ODBLK;

        MatA p_frag[FA_CBLK];
        [unroll] for (uint kc = 0; kc < FA_CBLK; ++kc) {
            p_frag[kc] = MatA::Load(s_p, rblk * TILE * FA_BC + kc * TILE, FA_BC,
                                    MatrixLayout::RowMajor);
        }

#if FA_ROW_OWNED
        // Exchange canonical PV tiles so each lane keeps its softmax row.
        [unroll] for (uint phase = 0; phase < O_TPW / PV_SLOTS; ++phase) {
        [unroll] for (uint pi = 0; pi < PV_SLOTS; ++pi) {
            const uint ti = phase * PV_SLOTS + pi;
#else
        [unroll] for (uint ti = 0; ti < O_TPW; ++ti) {
#endif
            const uint dblk = output_split * FA_ODBLK
                            + (wave * O_TPW + ti) % FA_ODBLK;
            const uint slot = (wave * PV_SLOTS + (ti % PV_SLOTS)) * TILE * TILE;

            MatPV pv = MatPV::Splat(0.0f);
            for (uint kc = 0; kc < FA_CBLK; ++kc) {
                MatB b;
                if (direct_v) {
                    const uint v_base = src2_off
                                      + (tile_start + kc * TILE) * src2_nb1
                                      + kv_head * src2_nb2 + batch_idx * src2_nb3
                                      + dblk * TILE * 2u;
                    b = MatB::Load(src2, v_base, src2_nb1, MatrixLayout::RowMajor, 32u);
                } else {
                    b = MatB::Load(s_kv, kc * TILE * FA_D + dblk * TILE, FA_D,
                                   MatrixLayout::RowMajor);
                }
                pv.MultiplyAccumulate(p_frag[kc], b);
            }
#if FA_PV_F16
            MatAcc pv_wide = pv.Cast<ComponentType::F32>();
            pv_wide.Store(s_tmp, slot, TILE, MatrixLayout::RowMajor);
#else
            pv.Store(s_tmp, slot, TILE, MatrixLayout::RowMajor);
#endif
            // Each wave writes private slots. Register-owned rows read them after the phase barrier.
            GroupMemoryBarrier();

#if FA_PV_F16
            bool invalid = false;
            float largest = 0.0f;
            float pv_values[O_EPT];
            [unroll] for (uint e = 0; e < O_EPT; ++e) {
                const float value = s_tmp[slot + O_ELEM(e)];
                pv_values[e] = value;
                invalid = invalid || !isfinite(value);
                largest = max(largest, abs(value));
            }
            bool repair = WaveActiveAnyTrue(invalid);
            // Recover whole-tile underflow only when P contains a nonzero input.
            if (!repair && WaveActiveMax(largest) < 6.103515625e-5f) {
                uint has_probability = 0u;
                [unroll] for (uint kc = 0; kc < FA_CBLK; ++kc) {
                    const uint count = WaveActiveMax(p_frag[kc].Length());
                    [loop] for (uint e = 0; e < count; ++e) {
                        has_probability |= (uint)(p_frag[kc].Get(e) != (float16_t)0.0f);
                    }
                }
                repair = WaveActiveAnyTrue(has_probability != 0u);
            }
            if (repair) {
                MatAcc repaired = MatAcc::Splat(0.0f);
                for (uint kc = 0; kc < FA_CBLK; ++kc) {
                    repaired.MultiplyAccumulate(p_frag[kc], fa_load_v(tile_start, dblk, kc, kv_head, batch_idx, direct_v));
                }
                repaired.Store(s_tmp, slot, TILE, MatrixLayout::RowMajor);
                GroupMemoryBarrier();
                [unroll] for (uint e = 0; e < O_EPT; ++e) {
                    pv_values[e] = s_tmp[slot + O_ELEM(e)];
                }
            }
#endif

#if FA_ROW_OWNED
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint wi = 0; wi < NWAVE / FA_RBLK; ++wi) {
            const uint owner = (owned_row / TILE) * (NWAVE / FA_RBLK) + wi;
            [unroll] for (uint pi = 0; pi < PV_SLOTS; ++pi) {
                const uint dt = wi * O_TPW + phase * PV_SLOTS + pi;
                const uint read_slot = (owner * PV_SLOTS + pi) * TILE * TILE;
                [unroll] for (uint e = 0; e < TILE / NPART; ++e) {
                    const uint col = owned_col * (TILE / NPART) + e;
                    const uint oi = dt * (TILE / NPART) + e;
                    o_reg[oi] = o_reg[oi] * corr + s_tmp[read_slot + (owned_row % TILE) * TILE + col];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
#else
            [unroll] for (uint e = 0; e < O_EPT; ++e) {
                const uint elem = O_ELEM(e);
                const uint r    = elem / TILE;
                const uint c    = elem % TILE;
                const uint gr   = rblk * TILE + r;
                o_reg[ti * O_EPT + e] = o_reg[ti * O_EPT + e] * s_corr[gr]
#if FA_PV_F16
                                      + pv_values[e];
#else
                                      + s_tmp[slot + r * TILE + c];
#endif
            }
#endif
        }
        // The score tile is written before the next KV tile's PV staging, so
        // the last read of a slot still has to be fenced off from it.
#if !FA_ROW_OWNED
        GroupMemoryBarrierWithGroupSync();
#endif
    }

    // Sinks carry no V contribution, so only the first split may apply them -
    // otherwise the reduce would fold one in per split.
    if (has_sinks != 0u && split_idx == 0u) {
#if FA_ROW_OWNED
        const float sink_score = asfloat(src4.Load(head_idx * 4u));
        const float new_max = max(owned_max, sink_score);
        const float corr = owned_sum > 0.0f ? exp(owned_max - new_max) : 0.0f;
        owned_sum = owned_sum * corr + exp(sink_score - new_max);
        owned_max = new_max;
        [unroll] for (uint oi = 0; oi < O_REGS; ++oi) {
            o_reg[oi] *= corr;
        }
#else
        if (tid < FA_BR) {
            const float sink_score = asfloat(src4.Load(head_idx * 4u));
            const float old_max = s_gmax[tid];
            const float new_max = max(old_max, sink_score);
            const float corr = (s_gsum[tid] > 0.0f) ? exp(old_max - new_max) : 0.0f;
            s_gmax[tid] = new_max;
            s_gsum[tid] = s_gsum[tid] * corr + exp(sink_score - new_max);
            s_corr[tid] = corr;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint ti = 0; ti < O_TPW; ++ti) {
            const uint rblk = (wave * O_TPW + ti) / FA_ODBLK;
            [unroll] for (uint e = 0; e < O_EPT; ++e) {
                const uint gr = rblk * TILE + O_ELEM(e) / TILE;
                o_reg[ti * O_EPT + e] *= s_corr[gr];
            }
        }
#endif
    }

    // Partial layout matches flash_attn.hlsl and flash_attn_reduce.hlsl:
    // [batch][head][query][split] x (max + sum + D_v floats).
    const uint partial_stride = (D_v + 2u) * 4u;
    const uint partial_base   = ((batch_idx * n_heads + head_idx) * N_queries + q_start)
                              * n_splits + split_idx;

#if FA_ROW_OWNED
    if (n_splits > 1u && owned_col == 0u && q_start + owned_row < N_queries) {
        const uint p_off = (partial_base + owned_row * n_splits) * partial_stride;
        temp.Store(p_off, asuint(owned_max));
        temp.Store(p_off + 4u, asuint(owned_sum));
    }
    [unroll] for (uint dt = 0; dt < FA_DBLK; ++dt) {
        [unroll] for (uint e = 0; e < TILE / NPART; ++e) {
            const uint d = dt * TILE + owned_col * (TILE / NPART) + e;
            const uint oi = dt * (TILE / NPART) + e;
            if (q_start + owned_row < N_queries) {
                if (n_splits > 1u) {
                    const uint p_off = (partial_base + owned_row * n_splits) * partial_stride;
                    temp.Store(p_off + 8u + d * 4u, asuint(o_reg[oi]));
                } else {
                    const float inv_sum = owned_sum > 0.0f ? 1.0f / owned_sum : 0.0f;
                    const uint out_off = dst_offset + d * nb0 + head_idx * nb1
                                       + (q_start + owned_row) * nb2 + batch_idx * nb3;
                    store_auto(dst, out_off, o_reg[oi] * inv_sum, dst_esize);
                }
            }
        }
    }
#else
    if (n_splits > 1u && tid < FA_BR && q_start + tid < N_queries) {
        const uint p_off = (partial_base + tid * n_splits) * partial_stride;
        temp.Store(p_off,      asuint(s_gmax[tid]));
        temp.Store(p_off + 4u, asuint(s_gsum[tid]));
    }

    [unroll] for (uint ti = 0; ti < O_TPW; ++ti) {
        const uint t    = wave * O_TPW + ti;
        const uint rblk = t / FA_ODBLK;
        const uint dblk = output_split * FA_ODBLK + t % FA_ODBLK;
        [unroll] for (uint e = 0; e < O_EPT; ++e) {
            const uint elem = O_ELEM(e);
            const uint r    = rblk * TILE + elem / TILE;
            const uint d    = dblk * TILE + elem % TILE;
            const uint query_idx = q_start + r;
            if (query_idx < N_queries && d < D_v) {
                if (n_splits > 1u) {
                    const uint p_off = (partial_base + r * n_splits) * partial_stride;
                    temp.Store(p_off + 8u + d * 4u, asuint(o_reg[ti * O_EPT + e]));
                } else {
                    const float inv_sum = s_gsum[r] > 0.0f ? (1.0f / s_gsum[r]) : 0.0f;
                    const uint out_off = dst_offset + d * nb0 + head_idx * nb1
                                       + query_idx * nb2 + batch_idx * nb3;
                    store_auto(dst, out_off, o_reg[ti * O_EPT + e] * inv_sum, dst_esize);
                }
            }
        }
    }
#endif
}
#endif
