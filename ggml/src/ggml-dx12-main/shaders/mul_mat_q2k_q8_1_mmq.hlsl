// Register-blocked Q2_K x Q8_1 MUL_MAT using packed int8 dot products.
//
// Each 32-element Q8_1 block maps to two 16-element Q2_K sub-blocks with
// independent scale/min pairs. The activation loader therefore keeps the two
// half-block sums needed by the Q2_K minimum terms.
//
// Dispatch: groups_x = ceil(N/MMQ_BN), groups_y = ceil(M/MMQ_BM),
//           groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K        256
#define Q2K_BSIZE   84
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
groupshared float tile_w_d0[MMQ_BN];
groupshared float tile_w_d1[MMQ_BN];
groupshared float tile_w_m0[MMQ_BN];
groupshared float tile_w_m1[MMQ_BN];
groupshared uint  tile_a_qs[MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_BM];
groupshared float tile_a_s0[MMQ_BM];
groupshared float tile_a_s1[MMQ_BM];

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
        const uint q2_block      = block / 8u;
        const uint tile_in_block = block & 7u;

        [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
            const uint gn = gid.x * MMQ_BN + r;
            uint4 wv = uint4(0u, 0u, 0u, 0u);
            float wd0 = 0.0f;
            float wd1 = 0.0f;
            float wm0 = 0.0f;
            float wm1 = 0.0f;
            if (gn < ne01) {
                const uint row_off = src0_offset + gn * nb01
                                   + i2_src0 * nb02 + i3_src0 * nb03;
                const uint blk_off = row_off + q2_block * Q2K_BSIZE;
                const uint shift = (tile_in_block & 3u) * 2u;
                const uint qs_off = blk_off + 16u
                                  + (tile_in_block >> 2u) * 32u
                                  + ld_part * 4u;
                wv = (src0.Load4(qs_off) >> shift) & 0x03030303u;

                if (ld_part == 0u) {
                    const uint dm = src0.Load(blk_off + 80u);
                    const float d    = f16_to_f32(dm & 0xFFFFu);
                    const float dmin = f16_to_f32(dm >> 16);
                    const uint sc0 = (src0.Load(blk_off + (2u * tile_in_block & ~3u))
                                    >> ((2u * tile_in_block & 3u) * 8u)) & 0xFFu;
                    const uint sc1 = (src0.Load(blk_off + ((2u * tile_in_block + 1u) & ~3u))
                                    >> (((2u * tile_in_block + 1u) & 3u) * 8u)) & 0xFFu;
                    wd0 = d    * float(sc0 & 0x0Fu);
                    wd1 = d    * float(sc1 & 0x0Fu);
                    wm0 = dmin * float(sc0 >> 4u);
                    wm1 = dmin * float(sc1 >> 4u);
                }
            }
            tile_w_qs[ld_part + 0u][r] = wv.x;
            tile_w_qs[ld_part + 1u][r] = wv.y;
            tile_w_qs[ld_part + 2u][r] = wv.z;
            tile_w_qs[ld_part + 3u][r] = wv.w;
            if (ld_part == 0u) {
                tile_w_d0[r] = wd0;
                tile_w_d1[r] = wd1;
                tile_w_m0[r] = wm0;
                tile_w_m1[r] = wm1;
            }
        }

        [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
            const uint gm = gid.y * MMQ_BM + r;
            uint4 av = uint4(0u, 0u, 0u, 0u);
            float ad = 0.0f;
            int asum = 0;
            if (gm < ne11) {
                const uint flat_row = (i3 * ne12 + i2) * ne11 + gm;
                const uint blk_off  = src1_offset
                                    + (flat_row * num_q8_blocks + block) * Q8_1_BSIZE;
                av = src1.Load4(blk_off + 4u + ld_part * 4u);
                // A chained constant operand is miscompiled on some AMD
                // drivers. Keep each packed sum independent, as in the
                // established Q3_K/Q4_K DP4A paths.
                int p0 = 0; p0 = dot4add_i8packed(0x01010101u, av.x, p0);
                int p1 = 0; p1 = dot4add_i8packed(0x01010101u, av.y, p1);
                int p2 = 0; p2 = dot4add_i8packed(0x01010101u, av.z, p2);
                int p3 = 0; p3 = dot4add_i8packed(0x01010101u, av.w, p3);
                asum = p0 + p1 + p2 + p3;
                if (ld_part == 0u) {
                    ad = f16_to_f32(src1.Load(blk_off) & 0xFFFFu);
                }
            }
            tile_a_qs[ld_part + 0u][r] = av.x;
            tile_a_qs[ld_part + 1u][r] = av.y;
            tile_a_qs[ld_part + 2u][r] = av.z;
            tile_a_qs[ld_part + 3u][r] = av.w;
            if (ld_part == 0u) {
                tile_a_d[r]  = ad;
                tile_a_s0[r] = float(asum);
            } else {
                tile_a_s1[r] = float(asum);
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll(2)] for (uint half_i = 0; half_i < 2u; half_i++) {
            int dots[MMQ_TM][MMQ_TN];
            [unroll] for (uint i0 = 0; i0 < MMQ_TM; i0++) {
                [unroll] for (uint j0 = 0; j0 < MMQ_TN; j0++) {
                    dots[i0][j0] = 0;
                }
            }

            [unroll(4)] for (uint qq = 0; qq < 4u; qq++) {
                const uint q = half_i * 4u + qq;
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
                const uint mi = MMQ_MIDX(i3b);
                const float da = tile_a_d[mi];
                const float sa = half_i == 0u ? tile_a_s0[mi] : tile_a_s1[mi];
                [unroll] for (uint j3 = 0; j3 < MMQ_TN; j3++) {
                    const uint nj = MMQ_NIDX(j3);
                    const float wd = half_i == 0u ? tile_w_d0[nj] : tile_w_d1[nj];
                    const float wm = half_i == 0u ? tile_w_m0[nj] : tile_w_m1[nj];
                    acc[i3b][j3] += wd * da * float(dots[i3b][j3]) - wm * da * sa;
                }
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
