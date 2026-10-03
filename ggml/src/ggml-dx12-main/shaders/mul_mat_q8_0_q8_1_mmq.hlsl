// Register-blocked Q8_0 x Q8_1 MUL_MAT using packed int8 dot products.
//
// The older tiled integer-dot kernel keeps a 32x32 tile with four
// accumulators per thread, so every dot product needs two fresh groupshared
// reads. This one blocks MMQ_TM x MMQ_TN outputs into registers, which turns
// MMQ_TM + MMQ_TN groupshared reads into MMQ_TM * MMQ_TN dot products.
//
// Dispatch: groups_x = ceil(N/MMQ_BN), groups_y = ceil(M/MMQ_BM),
//           groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK8_0       32
#define Q8_0_BSIZE  34
#define Q8_1_BSIZE  36

// Quants per block, packed four to a dword.
#define MMQ_QUADS 8

#ifndef MMQ_TM
#define MMQ_TM 8
#endif
#ifndef MMQ_TN
#define MMQ_TN 4
#endif
#ifndef MMQ_KSTEP
#define MMQ_KSTEP 1
#endif

#define MMQ_TX      16
#define MMQ_TY      16
#define MMQ_THREADS (MMQ_TX * MMQ_TY)
#define MMQ_BM      (MMQ_TY * MMQ_TM)
#define MMQ_BN      (MMQ_TX * MMQ_TN)

// Two threads cover a row's eight quad-dwords, four each.
#define MMQ_LDROWS (MMQ_THREADS / 2)
#define MMQ_WITER  ((MMQ_BN + MMQ_LDROWS - 1) / MMQ_LDROWS)
#define MMQ_AITER  ((MMQ_BM + MMQ_LDROWS - 1) / MMQ_LDROWS)

// Quant tiles are stored quad-major so that the MMQ_TN consecutive rows a
// thread reads land on consecutive dwords; row-major puts every thread of a
// quarter-wave on the same LDS bank.
groupshared uint  tile_w_qs[MMQ_KSTEP][MMQ_QUADS][MMQ_BN];
groupshared float tile_w_d [MMQ_KSTEP][MMQ_BN];
groupshared uint  tile_a_qs[MMQ_KSTEP][MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_KSTEP][MMQ_BM];

// Q8_0 blocks are 34 bytes, so a block's quants start two bytes into a dword.
// Pull the covering dwords and funnel-shift rather than doing byte loads.
void q8_0_quads4(ByteAddressBuffer buf, uint byte_off, out uint4 v) {
    const uint  aligned = byte_off & ~3u;
    const uint  shift   = (byte_off & 3u) * 8u;
    const uint4 lo      = buf.Load4(aligned);
    // The fifth word is only meaningful for a misaligned start, but it is
    // addressed unconditionally: a load guarded by `shift != 0` can still be
    // speculated by the compiler, and these buffers are bound as root SRVs,
    // which D3D12 does not bounds check. Reading aligned+16 for an already
    // aligned offset walks off the end of the last tensor in an allocation
    // and faults the device.
    const uint hi = buf.Load(aligned + (shift == 0u ? 12u : 16u));
    if (shift == 0u) {
        v = lo;
        return;
    }
    v.x = (lo.x >> shift) | (lo.y << (32u - shift));
    v.y = (lo.y >> shift) | (lo.z << (32u - shift));
    v.z = (lo.z >> shift) | (lo.w << (32u - shift));
    v.w = (lo.w >> shift) | (hi   << (32u - shift));
}

float q8_0_scale(ByteAddressBuffer buf, uint byte_off) {
    const uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
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
    const uint num_blocks = K / QK8_0;

    // Each thread stages four consecutive quad-dwords of one row, so a row's
    // 32 quants are covered by two threads and the loads stay contiguous.
    const uint ld_row  = tid / 2u;
    const uint ld_part = (tid % 2u) * 4u;

    float acc[MMQ_TM][MMQ_TN];
    [unroll] for (uint ia = 0; ia < MMQ_TM; ia++) {
        [unroll] for (uint ja = 0; ja < MMQ_TN; ja++) {
            acc[ia][ja] = 0.0f;
        }
    }
    // Contiguous per-thread outputs: the epilogue stores coalesce, which is
    // worth more than the residual LDS bank conflict of a strided mapping.
#define MMQ_MIDX(i) (ty * MMQ_TM + (i))
#define MMQ_NIDX(j) (tx * MMQ_TN + (j))

    for (uint block = 0; block < num_blocks; block += MMQ_KSTEP) {
        [unroll] for (uint s = 0; s < MMQ_KSTEP; s++) {
            const uint blk = block + s;
            const bool live = blk < num_blocks;

            // Weight rows: MMQ_BN of them, two threads each.
            [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
                const uint gn = gid.x * MMQ_BN + r;
                uint4 wv = uint4(0u, 0u, 0u, 0u);
                float wd = 0.0f;
                if (live && gn < ne01) {
                    const uint row_off = src0_offset + gn * nb01
                                       + i2_src0 * nb02 + i3_src0 * nb03;
                    const uint blk_off = row_off + blk * Q8_0_BSIZE;
                    q8_0_quads4(src0, blk_off + 2u + ld_part * 4u, wv);
                    wd = q8_0_scale(src0, blk_off);
                }
                tile_w_qs[s][ld_part + 0u][r] = wv.x;
                tile_w_qs[s][ld_part + 1u][r] = wv.y;
                tile_w_qs[s][ld_part + 2u][r] = wv.z;
                tile_w_qs[s][ld_part + 3u][r] = wv.w;
                if (ld_part == 0u) {
                    tile_w_d[s][r] = wd;
                }
            }

            // Activation rows: MMQ_BM of them. Q8_1 blocks are 36 bytes, so
            // the quants are dword aligned and load directly.
            [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
                const uint gm = gid.y * MMQ_BM + r;
                uint4 av = uint4(0u, 0u, 0u, 0u);
                float ad = 0.0f;
                if (live && gm < ne11) {
                    const uint flat_row = (i3 * ne12 + i2) * ne11 + gm;
                    const uint blk_off  = src1_offset
                                        + (flat_row * num_blocks + blk) * Q8_1_BSIZE;
                    av = src1.Load4(blk_off + 4u + ld_part * 4u);
                    ad = f16_to_f32(src1.Load(blk_off) & 0xFFFFu);
                }
                tile_a_qs[s][ld_part + 0u][r] = av.x;
                tile_a_qs[s][ld_part + 1u][r] = av.y;
                tile_a_qs[s][ld_part + 2u][r] = av.z;
                tile_a_qs[s][ld_part + 3u][r] = av.w;
                if (ld_part == 0u) {
                    tile_a_d[s][r] = ad;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint s = 0; s < MMQ_KSTEP; s++) {
            // The scales are per block, so the integer sums have to be closed
            // out and folded into the float accumulators every block.
            int dots[MMQ_TM][MMQ_TN];
            [unroll] for (uint i0 = 0; i0 < MMQ_TM; i0++) {
                [unroll] for (uint j0 = 0; j0 < MMQ_TN; j0++) {
                    dots[i0][j0] = 0;
                }
            }

            [unroll] for (uint q = 0; q < MMQ_QUADS; q++) {
                uint av[MMQ_TM];
                [unroll] for (uint i1 = 0; i1 < MMQ_TM; i1++) {
                    av[i1] = tile_a_qs[s][q][MMQ_MIDX(i1)];
                }
                uint bv[MMQ_TN];
                [unroll] for (uint j1 = 0; j1 < MMQ_TN; j1++) {
                    bv[j1] = tile_w_qs[s][q][MMQ_NIDX(j1)];
                }
                [unroll] for (uint i2b = 0; i2b < MMQ_TM; i2b++) {
                    [unroll] for (uint j2 = 0; j2 < MMQ_TN; j2++) {
                        dots[i2b][j2] = dot4add_i8packed(bv[j2], av[i2b],
                                                         dots[i2b][j2]);
                    }
                }
            }

            [unroll] for (uint i3b = 0; i3b < MMQ_TM; i3b++) {
                const float da = tile_a_d[s][MMQ_MIDX(i3b)];
                [unroll] for (uint j3 = 0; j3 < MMQ_TN; j3++) {
                    acc[i3b][j3] += da * tile_w_d[s][MMQ_NIDX(j3)]
                                  * (float)dots[i3b][j3];
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
