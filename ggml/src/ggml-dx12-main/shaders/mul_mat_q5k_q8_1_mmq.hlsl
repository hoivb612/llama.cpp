// Register-blocked Q5_K x Q8_1 MUL_MAT using packed int8 dot products.
//
// Identical to mul_mat_q4k_q8_1_mmq.hlsl apart from the weight fetch, which
// pulls the fifth bit out of the qh plane before the dot product. See that
// file for the tile shape and the quad-major groupshared rationale.
//
// Dispatch: groups_x = ceil(N/MMQ_BN), groups_y = ceil(M/MMQ_BM),
//           groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K        256
#define Q5K_BSIZE   176
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
groupshared float tile_w_m [MMQ_BN];
groupshared uint  tile_a_qs[MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_BM];
groupshared float tile_a_s [MMQ_BM];

uint q5k_byte(ByteAddressBuffer buf, uint byte_off) {
    const uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(MMQ_TX, MMQ_TY, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tx  = gtid.x;
    const uint ty  = gtid.y;
    const uint tid = ty * MMQ_TX + tx;

    const uint i2 = gid.z % ne2;
    const uint i3 = gid.z / ne2;
    const uint i2_src0 = i2 * ne02 / ne2;
    const uint i3_src0 = i3 * ne03 / ne3;

    const uint K = ne00;
    const uint num_q8_blocks = K / 32u;

    const uint ld_row  = tid / 2u;
    const uint ld_part = (tid % 2u) * 4u;

    float acc[MMQ_TM][MMQ_TN];
    [unroll] for (uint ia = 0; ia < MMQ_TM; ia++) {
        [unroll] for (uint ja = 0; ja < MMQ_TN; ja++) {
            acc[ia][ja] = 0.0f;
        }
    }

#define MMQ_MIDX(i) (ty * MMQ_TM + (i))
#define MMQ_NIDX(j) (tx * MMQ_TN + (j))

    for (uint block = 0; block < num_q8_blocks; block++) {
        const uint q5_block      = block / 8u;
        const uint tile_in_block = block & 7u;
        const uint il            = tile_in_block / 2u;
        const bool high_nibble   = (tile_in_block & 1u) != 0u;

        [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
            const uint gn = gid.x * MMQ_BN + r;
            uint4 wv = uint4(0u, 0u, 0u, 0u);
            float wd = 0.0f;
            float wm = 0.0f;
            if (gn < ne01) {
                const uint row_off = src0_offset + gn * nb01
                                   + i2_src0 * nb02 + i3_src0 * nb03;
                const uint blk_off = row_off + q5_block * Q5K_BSIZE;
                const uint4 raw = src0.Load4(blk_off + 48u + il * 32u
                                             + ld_part * 4u);
                const uint4 qh  = src0.Load4(blk_off + 16u + ld_part * 4u);
                const uint4 lo  = high_nibble ? ((raw >> 4u) & 0x0F0F0F0Fu)
                                              : (raw & 0x0F0F0F0Fu);
                const uint4 hi  = ((qh >> tile_in_block) & 0x01010101u) << 4u;
                wv = lo | hi;

                if (ld_part == 0u) {
                    const uint  dm_raw = src0.Load(blk_off);
                    const float dall   = f16_to_f32(dm_raw & 0xFFFFu);
                    const float dmin   = f16_to_f32(dm_raw >> 16);
                    const uint  sc_off = blk_off + 4u;
                    const uint  is_eff = tile_in_block;
                    const bool  low    = il < 2u;

                    const uint scidx0  = low ? is_eff : (is_eff + 4u);
                    const uint scidx1  = low ? is_eff : (is_eff - 4u);
                    const uint scmask1 = low ? 0x30u : 0xC0u;
                    const uint scshft1 = low ? 0u : 2u;
                    const uint mbidx0  = is_eff + 4u;
                    const uint mbidx1  = low ? (is_eff + 4u) : is_eff;
                    const uint mbmask0 = low ? 0x0Fu : 0xF0u;
                    const uint mbshft0 = low ? 0u : 4u;
                    const uint mbmask1 = low ? 0x30u : 0xC0u;
                    const uint mbshft1 = low ? 0u : 2u;

                    const uint sc = (q5k_byte(src0, sc_off + scidx0) & 0x0Fu)
                        | ((q5k_byte(src0, sc_off + scidx1) & scmask1) >> scshft1);
                    const uint m = ((q5k_byte(src0, sc_off + mbidx0) & mbmask0) >> mbshft0)
                        | ((q5k_byte(src0, sc_off + mbidx1) & mbmask1) >> mbshft1);

                    wd = dall * float(sc);
                    wm = dmin * float(m);
                }
            }
            tile_w_qs[ld_part + 0u][r] = wv.x;
            tile_w_qs[ld_part + 1u][r] = wv.y;
            tile_w_qs[ld_part + 2u][r] = wv.z;
            tile_w_qs[ld_part + 3u][r] = wv.w;
            if (ld_part == 0u) {
                tile_w_d[r] = wd;
                tile_w_m[r] = wm;
            }
        }

        [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
            const uint gm = gid.y * MMQ_BM + r;
            uint4 av = uint4(0u, 0u, 0u, 0u);
            float ad = 0.0f;
            float as = 0.0f;
            if (gm < ne11) {
                const uint flat_row = (i3 * ne12 + i2) * ne11 + gm;
                const uint blk_off  = src1_offset
                                    + (flat_row * num_q8_blocks + block) * Q8_1_BSIZE;
                av = src1.Load4(blk_off + 4u + ld_part * 4u);
                if (ld_part == 0u) {
                    const uint ds = src1.Load(blk_off);
                    ad = f16_to_f32(ds & 0xFFFFu);
                    as = f16_to_f32(ds >> 16);
                }
            }
            tile_a_qs[ld_part + 0u][r] = av.x;
            tile_a_qs[ld_part + 1u][r] = av.y;
            tile_a_qs[ld_part + 2u][r] = av.z;
            tile_a_qs[ld_part + 3u][r] = av.w;
            if (ld_part == 0u) {
                tile_a_d[r] = ad;
                tile_a_s[r] = as;
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
                    dots[i2b][j2] = dot4add_i8packed(bv[j2], av[i2b],
                                                     dots[i2b][j2]);
                }
            }
        }

        [unroll] for (uint i3b = 0; i3b < MMQ_TM; i3b++) {
            const float da = tile_a_d[MMQ_MIDX(i3b)];
            const float sa = tile_a_s[MMQ_MIDX(i3b)];
            [unroll] for (uint j3 = 0; j3 < MMQ_TN; j3++) {
                acc[i3b][j3] += tile_w_d[MMQ_NIDX(j3)] * da * (float)dots[i3b][j3]
                              - tile_w_m[MMQ_NIDX(j3)] * sa;
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
