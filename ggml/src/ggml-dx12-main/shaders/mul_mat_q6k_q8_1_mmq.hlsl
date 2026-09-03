// Register-blocked Q6_K x Q8_1 MUL_MAT using packed int8 dot products.
//
// Same tile shape and quad-major groupshared layout as
// mul_mat_q4k_q8_1_mmq.hlsl. Two differences: Q6_K superblocks are 210 bytes,
// so nothing is 4-byte aligned and every weight load goes through the
// unaligned funnel path, and each 32-element tile spans two of the 16 scales,
// so the K loop folds quads 0-3 and 4-7 with different scales. Folding each
// half separately keeps the accumulator count at MMQ_TM*MMQ_TN rather than
// doubling it.
//
// Dispatch: groups_x = ceil(N/MMQ_BN), groups_y = ceil(M/MMQ_BM),
//           groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K        256
#define Q6K_BSIZE   210
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
groupshared uint  tile_a_qs[MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_BM];

uint q6k_byte(ByteAddressBuffer buf, uint byte_off) {
    const uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int q6k_i8(ByteAddressBuffer buf, uint byte_off) {
    return int(q6k_byte(buf, byte_off) ^ 0x80u) - 128;
}

float q6k_f16(ByteAddressBuffer buf, uint byte_off) {
    const uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint q6k_u32(ByteAddressBuffer buf, uint byte_off) {
    const uint aligned = byte_off & ~3u;
    const uint shift   = (byte_off & 3u) * 8u;
    const uint lo = buf.Load(aligned);
    const uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
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
        const uint q6_block      = block / 8u;
        const uint tile_in_block = block & 7u;
        const uint sub = tile_in_block & 3u;
        const uint h   = tile_in_block >> 2u;

        [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
            const uint gn = gid.x * MMQ_BN + r;
            uint wv[4] = { 0u, 0u, 0u, 0u };
            float wd0 = 0.0f;
            float wd1 = 0.0f;
            if (gn < ne01) {
                const uint row_off = src0_offset + gn * nb01
                                   + i2_src0 * nb02 + i3_src0 * nb03;
                const uint blk_off = row_off + q6_block * Q6K_BSIZE;
                [unroll(4)] for (uint e = 0; e < 4u; e++) {
                    const uint idx = ld_part + e;
                    const uint ql = q6k_u32(src0, blk_off + h * 64u
                                                  + (sub & 1u) * 32u + idx * 4u);
                    const uint lo = (sub < 2u) ? (ql & 0x0F0F0F0Fu)
                                               : ((ql >> 4u) & 0x0F0F0F0Fu);
                    const uint qh = q6k_u32(src0, blk_off + 128u + h * 32u
                                                  + idx * 4u);
                    const uint hi = ((qh >> (2u * sub)) & 0x03030303u) << 4u;
                    const uint v   = lo | hi;
                    const uint neg = (v & 0x20202020u) ^ 0x20202020u;
                    wv[e] = (v & 0x1F1F1F1Fu) | (neg * 0x07u);
                }
                if (ld_part == 0u) {
                    const float d = q6k_f16(src0, blk_off + 208u);
                    const uint sidx0 = h * 8u + 2u * sub;
                    wd0 = d * float(q6k_i8(src0, blk_off + 192u + sidx0));
                    wd1 = d * float(q6k_i8(src0, blk_off + 192u + sidx0 + 1u));
                }
            }
            tile_w_qs[ld_part + 0u][r] = wv[0];
            tile_w_qs[ld_part + 1u][r] = wv[1];
            tile_w_qs[ld_part + 2u][r] = wv[2];
            tile_w_qs[ld_part + 3u][r] = wv[3];
            if (ld_part == 0u) {
                tile_w_d0[r] = wd0;
                tile_w_d1[r] = wd1;
            }
        }

        [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
            const uint gm = gid.y * MMQ_BM + r;
            uint4 av = uint4(0u, 0u, 0u, 0u);
            float ad = 0.0f;
            if (gm < ne11) {
                const uint flat_row = (i3 * ne12 + i2) * ne11 + gm;
                const uint blk_off  = src1_offset
                                    + (flat_row * num_q8_blocks + block) * Q8_1_BSIZE;
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

        // half 0 = quads 0..3 (scale d0), half 1 = quads 4..7 (scale d1)
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
                const float da = tile_a_d[MMQ_MIDX(i3b)];
                [unroll] for (uint j3 = 0; j3 < MMQ_TN; j3++) {
                    const float wd = (half_i == 0u) ? tile_w_d0[MMQ_NIDX(j3)]
                                                    : tile_w_d1[MMQ_NIDX(j3)];
                    acc[i3b][j3] += da * wd * (float)dots[i3b][j3];
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
