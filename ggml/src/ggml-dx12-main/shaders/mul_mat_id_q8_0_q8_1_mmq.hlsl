// Grouped-expert Q8_0 x Q8_1 GEMM for MUL_MAT_ID.
//
// The group builds a 128-row tile containing only rows routed to one expert,
// then reuses that expert's 64-row weight tile across all selected rows.
#include "ggml_common.hlsli"

#define QK8_0          32
#define Q8_0_BSIZE     34
#define Q8_1_BSIZE     36
#define MMQ_QUADS      8
#define MMQ_TM         8
#define MMQ_TN         4
#define MMQ_KSTEP      1
#define MMQ_TX         16
#define MMQ_TY         16
#define MMQ_THREADS    (MMQ_TX * MMQ_TY)
#define MMQ_BM         (MMQ_TY * MMQ_TM)
#define MMQ_BN         (MMQ_TX * MMQ_TN)
#define MMQ_LDROWS     (MMQ_THREADS / 2)
#define MMQ_WITER      ((MMQ_BN + MMQ_LDROWS - 1) / MMQ_LDROWS)
#define MMQ_AITER      ((MMQ_BM + MMQ_LDROWS - 1) / MMQ_LDROWS)
#define MMID_MAX_EXPERT 256
#define MMID_MAX_WAVES  16
#define MMID_NONE       0xFFFFFFFFu

groupshared uint  tile_w_qs[MMQ_KSTEP][MMQ_QUADS][MMQ_BN];
groupshared float tile_w_d [MMQ_KSTEP][MMQ_BN];
groupshared uint  tile_a_qs[MMQ_KSTEP][MMQ_QUADS][MMQ_BM];
groupshared float tile_a_d [MMQ_KSTEP][MMQ_BM];

groupshared uint sh_a_off[MMQ_BM];
groupshared uint sh_d_off[MMQ_BM];
groupshared uint sh_cnt[MMID_MAX_EXPERT];
groupshared uint sh_expert;
groupshared uint sh_tile_local;
groupshared uint sh_wave_cnt[MMID_MAX_WAVES];

void q8_0_quads4(ByteAddressBuffer buf, uint byte_off, out uint4 v) {
    const uint aligned = byte_off & ~3u;
    const uint shift = (byte_off & 3u) * 8u;
    const uint4 lo = buf.Load4(aligned);
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

uint mmid_id(uint r, uint n_expert) {
    const uint slot = r % ne1;
    const uint token = (r / ne1) % ne2;
    const uint id = (uint)asint(src2.Load(op0 + slot * op1 + token * op2));
    return id < n_expert ? id : MMID_NONE;
}

[numthreads(MMQ_TX, MMQ_TY, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    const uint tx = gtid.x;
    const uint ty = gtid.y;
    const uint tid = ty * MMQ_TX + tx;
    const uint n_expert = ne02;
    const uint n_rows = ne1 * ne2 * ne3;
    const uint row_block = gid.x;
    const uint col_block = gid.y;

    for (uint e = tid; e < n_expert; e += MMQ_THREADS) {
        sh_cnt[e] = 0;
    }
    if (tid == 0) {
        sh_expert = MMID_NONE;
        sh_tile_local = 0;
    }
    for (uint m = tid; m < MMQ_BM; m += MMQ_THREADS) {
        sh_a_off[m] = MMID_NONE;
        sh_d_off[m] = MMID_NONE;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint r = tid; r < n_rows; r += MMQ_THREADS) {
        const uint id = mmid_id(r, n_expert);
        if (id != MMID_NONE) {
            uint prev;
            InterlockedAdd(sh_cnt[id], 1u, prev);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        uint tile = 0;
        for (uint e = 0; e < n_expert; e++) {
            const uint nt = (sh_cnt[e] + MMQ_BM - 1u) / MMQ_BM;
            if (row_block >= tile && row_block < tile + nt) {
                sh_expert = e;
                sh_tile_local = row_block - tile;
            }
            tile += nt;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const uint expert = sh_expert;
    if (expert == MMID_NONE) {
        return;
    }

    const uint wave_size = WaveGetLaneCount();
    const uint wave = tid / wave_size;
    const uint lane = WaveGetLaneIndex();
    const uint num_waves = MMQ_THREADS / wave_size;
    const uint slot_base = sh_tile_local * MMQ_BM;
    const uint num_blocks = ne00 / QK8_0;
    uint scanned = 0;

    for (uint base = 0; base < n_rows; base += MMQ_THREADS) {
        const uint r = base + tid;
        const bool match = r < n_rows && mmid_id(r, n_expert) == expert;
        const uint wave_prefix = WavePrefixCountBits(match);
        const uint wave_total = WaveActiveCountBits(match);
        if (lane == 0) {
            sh_wave_cnt[wave] = wave_total;
        }
        GroupMemoryBarrierWithGroupSync();

        uint group_prefix = 0;
        uint group_total = 0;
        for (uint w = 0; w < num_waves; w++) {
            const uint count = sh_wave_cnt[w];
            if (w < wave) {
                group_prefix += count;
            }
            group_total += count;
        }
        if (match) {
            const uint rank = scanned + group_prefix + wave_prefix;
            if (rank >= slot_base && rank < slot_base + MMQ_BM) {
                const uint slot = r % ne1;
                const uint token = (r / ne1) % ne2;
                const uint batch = r / (ne1 * ne2);
                const uint tile_row = rank - slot_base;
                const uint q8_row = (batch * ne12 + token) * ne11 + slot % ne11;
                sh_a_off[tile_row] =
                    src1_offset + q8_row * num_blocks * Q8_1_BSIZE;
                sh_d_off[tile_row] =
                    dst_offset + slot * nb1 + token * nb2 + batch * nb3;
            }
        }
        scanned += group_total;
        GroupMemoryBarrierWithGroupSync();
    }

    const uint ld_row = tid / 2u;
    const uint ld_part = (tid % 2u) * 4u;
    float acc[MMQ_TM][MMQ_TN];
    [unroll] for (uint i = 0; i < MMQ_TM; i++) {
        [unroll] for (uint j = 0; j < MMQ_TN; j++) {
            acc[i][j] = 0.0f;
        }
    }

#define MMQ_MIDX(i) (ty * MMQ_TM + (i))
#define MMQ_NIDX(j) (tx * MMQ_TN + (j))

    for (uint block = 0; block < num_blocks; block += MMQ_KSTEP) {
        [unroll] for (uint s = 0; s < MMQ_KSTEP; s++) {
            const uint blk = block + s;
            const bool live = blk < num_blocks;

            [unroll(MMQ_WITER)] for (uint r = ld_row; r < MMQ_BN; r += MMQ_LDROWS) {
                const uint gn = col_block * MMQ_BN + r;
                uint4 wv = uint4(0u, 0u, 0u, 0u);
                float wd = 0.0f;
                if (live && gn < ne0) {
                    const uint blk_off = src0_offset + gn * nb01 + expert * nb02
                                       + blk * Q8_0_BSIZE;
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

            [unroll(MMQ_AITER)] for (uint r = ld_row; r < MMQ_BM; r += MMQ_LDROWS) {
                uint4 av = uint4(0u, 0u, 0u, 0u);
                float ad = 0.0f;
                const uint row_off = sh_a_off[r];
                if (live && row_off != MMID_NONE) {
                    const uint blk_off = row_off + blk * Q8_1_BSIZE;
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
            int dots[MMQ_TM][MMQ_TN];
            [unroll] for (uint i = 0; i < MMQ_TM; i++) {
                [unroll] for (uint j = 0; j < MMQ_TN; j++) {
                    dots[i][j] = 0;
                }
            }

            [unroll] for (uint q = 0; q < MMQ_QUADS; q++) {
                uint av[MMQ_TM];
                uint wv[MMQ_TN];
                [unroll] for (uint i = 0; i < MMQ_TM; i++) {
                    av[i] = tile_a_qs[s][q][MMQ_MIDX(i)];
                }
                [unroll] for (uint j = 0; j < MMQ_TN; j++) {
                    wv[j] = tile_w_qs[s][q][MMQ_NIDX(j)];
                }
                [unroll] for (uint i = 0; i < MMQ_TM; i++) {
                    [unroll] for (uint j = 0; j < MMQ_TN; j++) {
                        dots[i][j] = dot4add_i8packed(wv[j], av[i], dots[i][j]);
                    }
                }
            }

            [unroll] for (uint i = 0; i < MMQ_TM; i++) {
                const float ad = tile_a_d[s][MMQ_MIDX(i)];
                [unroll] for (uint j = 0; j < MMQ_TN; j++) {
                    acc[i][j] += ad * tile_w_d[s][MMQ_NIDX(j)] * (float)dots[i][j];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint i = 0; i < MMQ_TM; i++) {
        const uint row = MMQ_MIDX(i);
        const uint dst_row = sh_d_off[row];
        if (dst_row == MMID_NONE) {
            continue;
        }
        [unroll] for (uint j = 0; j < MMQ_TN; j++) {
            const uint gn = col_block * MMQ_BN + MMQ_NIDX(j);
            if (gn < ne0) {
                store_auto(dst, dst_row + gn * nb0, acc[i][j], dst_esize);
            }
        }
    }
}
