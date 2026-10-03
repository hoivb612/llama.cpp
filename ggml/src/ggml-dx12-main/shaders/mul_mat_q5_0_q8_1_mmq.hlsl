// Register-blocked Q5_0 x Q8_1 MUL_MAT using packed int8 dot products.
//
// Dispatch: groups_x = ceil(N/MMQ_BN), groups_y = ceil(M/MMQ_BM),
//           groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK5_0       32
#define Q5_0_BSIZE  22
#define Q8_1_BSIZE  36

#define MMQ_QUADS 8

#ifndef MMQ_TM
#define MMQ_TM 8
#endif
#ifndef MMQ_TN
#define MMQ_TN 4
#endif

#define MMQ_TX      16
#define MMQ_TY      16
#define MMQ_THREADS (MMQ_TX * MMQ_TY)
#define MMQ_BM      (MMQ_TY * MMQ_TM)
#define MMQ_BN      (MMQ_TX * MMQ_TN)

#define MMQ_LDROWS (MMQ_THREADS / 2)
#define MMQ_WITER  ((MMQ_BN + MMQ_LDROWS - 1) / MMQ_LDROWS)
#define MMQ_AITER  ((MMQ_BM + MMQ_LDROWS - 1) / MMQ_LDROWS)

groupshared uint  tile_w_qs[MMQ_QUADS][MMQ_BN];
groupshared float tile_w_d [MMQ_BN];
groupshared uint  tile_a_qs[MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_BM];

uint q5_0_load_u32(ByteAddressBuffer buf, uint byte_off) {
    const uint lo = buf.Load<uint16_t>(byte_off);
    const uint hi = buf.Load<uint16_t>(byte_off + 2u);
    return lo | (hi << 16u);
}

uint q5_0_unpack4(uint qs_word, uint qh, uint packed_idx) {
    const uint nib = packed_idx < 4u
        ? (qs_word & 0x0F0F0F0Fu)
        : ((qs_word >> 4u) & 0x0F0F0F0Fu);
    const uint hb = (qh >> (packed_idx * 4u)) & 0xFu;
    const uint spread = ((hb & 1u) << 4u) | (((hb >> 1u) & 1u) << 12u) |
                        (((hb >> 2u) & 1u) << 20u) | (((hb >> 3u) & 1u) << 28u);
    const uint v = nib | spread;
    const uint neg = (v & 0x10101010u) ^ 0x10101010u;
    return (v & 0x0F0F0F0Fu) | (neg * 0x0Fu);
}

#ifdef Q50_MMQ_WAVE_SHARE
[numthreads(MMQ_THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    const uint tx = tid % MMQ_TX;
    const uint ty = tid / MMQ_TX;
#else
[numthreads(MMQ_TX, MMQ_TY, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tx  = gtid.x;
    const uint ty  = gtid.y;
    const uint tid = ty * MMQ_TX + tx;
#endif

    const uint i2 = gid.z % ne2;
    const uint i3 = gid.z / ne2;
    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;

    const uint num_blocks = ne00 / QK5_0;
    const uint ld_row  = tid / 2u;
    const uint ld_part = (tid % 2u) * 4u;

    precise float acc[MMQ_TM][MMQ_TN];
    [unroll] for (uint ia = 0; ia < MMQ_TM; ia++) {
        [unroll] for (uint ja = 0; ja < MMQ_TN; ja++) {
            acc[ia][ja] = 0.0f;
        }
    }

#define MMQ_MIDX(i) (ty * MMQ_TM + (i))
#define MMQ_NIDX(j) (tx * MMQ_TN + (j))

    for (uint block = 0; block < num_blocks; block++) {
        [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
            const uint gn = gid.x * MMQ_BN + r;
            uint4 wv = uint4(0u, 0u, 0u, 0u);
            float wd = 0.0f;
            if (gn < ne01) {
                const uint row_off = src0_offset + gn * nb01
                                   + i2_src0 * nb02 + i3_src0 * nb03;
                const uint blk_off = row_off + block * Q5_0_BSIZE;
#ifdef Q50_MMQ_WAVE_SHARE
                uint4 raw = uint4(0u, 0u, 0u, 0u);
                uint qh = 0u;
                if (ld_part == 0u) {
                    raw.x = q5_0_load_u32(src0, blk_off + 6u);
                    raw.y = q5_0_load_u32(src0, blk_off + 10u);
                    raw.z = q5_0_load_u32(src0, blk_off + 14u);
                    raw.w = q5_0_load_u32(src0, blk_off + 18u);
                    qh = q5_0_load_u32(src0, blk_off + 2u);
                    wd = f16_to_f32(src0.Load<uint16_t>(blk_off));
                }
                const uint src_lane = WaveGetLaneIndex() & ~1u;
                raw.x = WaveReadLaneAt(raw.x, src_lane);
                raw.y = WaveReadLaneAt(raw.y, src_lane);
                raw.z = WaveReadLaneAt(raw.z, src_lane);
                raw.w = WaveReadLaneAt(raw.w, src_lane);
                qh = WaveReadLaneAt(qh, src_lane);
                [unroll] for (uint q = 0; q < 4u; q++) {
                    const uint packed_idx = ld_part + q;
                    wv[q] = q5_0_unpack4(raw[q], qh, packed_idx);
                }
#else
                const uint qh = q5_0_load_u32(src0, blk_off + 2u);
                [unroll] for (uint q = 0; q < 4u; q++) {
                    const uint packed_idx = ld_part + q;
                    const uint qs_off = blk_off + 6u + (packed_idx & 3u) * 4u;
                    wv[q] = q5_0_unpack4(q5_0_load_u32(src0, qs_off), qh, packed_idx);
                }
                if (ld_part == 0u) {
                    wd = f16_to_f32(src0.Load<uint16_t>(blk_off));
                }
#endif
            }
            tile_w_qs[ld_part + 0u][r] = wv.x;
            tile_w_qs[ld_part + 1u][r] = wv.y;
            tile_w_qs[ld_part + 2u][r] = wv.z;
            tile_w_qs[ld_part + 3u][r] = wv.w;
            if (ld_part == 0u) {
                tile_w_d[r] = wd;
            }
        }

        [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
            const uint gm = gid.y * MMQ_BM + r;
            uint4 av = uint4(0u, 0u, 0u, 0u);
            float ad = 0.0f;
            if (gm < ne11) {
                const uint flat_row = (i3 * ne12 + i2) * ne11 + gm;
                const uint blk_off  = src1_offset
                                    + (flat_row * num_blocks + block) * Q8_1_BSIZE;
                av = src1.Load4(blk_off + 4u + ld_part * 4u);
                if (ld_part == 0u) {
                    ad = f16_to_f32(src1.Load(blk_off) & 0xFFFFu);
                }
            }
            tile_a_qs[ld_part + 0u][r] = av.x;
            tile_a_qs[ld_part + 1u][r] = av.y;
            tile_a_qs[ld_part + 2u][r] = av.z;
            tile_a_qs[ld_part + 3u][r] = av.w;
            if (ld_part == 0u) {
                tile_a_d[r] = ad;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        int dots[MMQ_TM][MMQ_TN];
        [unroll] for (uint i0 = 0; i0 < MMQ_TM; i0++) {
            [unroll] for (uint j0 = 0; j0 < MMQ_TN; j0++) {
                dots[i0][j0] = 0;
            }
        }

        [unroll] for (uint q = 0; q < MMQ_QUADS; q++) {
            uint av[MMQ_TM];
            [unroll] for (uint i1 = 0; i1 < MMQ_TM; i1++) {
                av[i1] = tile_a_qs[q][MMQ_MIDX(i1)];
            }
            uint bv[MMQ_TN];
            [unroll] for (uint j1 = 0; j1 < MMQ_TN; j1++) {
                bv[j1] = tile_w_qs[q][MMQ_NIDX(j1)];
            }
            [unroll] for (uint i2b = 0; i2b < MMQ_TM; i2b++) {
                [unroll] for (uint j2 = 0; j2 < MMQ_TN; j2++) {
                    dots[i2b][j2] = dot4add_i8packed(bv[j2], av[i2b], dots[i2b][j2]);
                }
            }
        }

        [unroll] for (uint i3b = 0; i3b < MMQ_TM; i3b++) {
            const float da = tile_a_d[MMQ_MIDX(i3b)];
            [unroll] for (uint j3 = 0; j3 < MMQ_TN; j3++) {
                acc[i3b][j3] += tile_w_d[MMQ_NIDX(j3)] * da * (float)dots[i3b][j3];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < MMQ_TM; i++) {
        const uint gm = gid.y * MMQ_BM + MMQ_MIDX(i);
        if (gm >= ne1) {
            continue;
        }
        [unroll] for (uint j = 0; j < MMQ_TN; j++) {
            const uint gn = gid.x * MMQ_BN + MMQ_NIDX(j);
            if (gn < ne0) {
                store_auto(dst,
                           offset_4d(gn, gm, i2, i3, nb0, nb1, nb2, nb3, dst_offset),
                           acc[i][j], dst_esize);
            }
        }
    }
}
