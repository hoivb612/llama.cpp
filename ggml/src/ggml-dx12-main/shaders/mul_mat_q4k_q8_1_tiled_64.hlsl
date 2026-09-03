// Tiled Q4_K x Q8_1 batch MUL_MAT using packed int8 dot products, 64x64 tile.
//
// Same K blocking and staging as mul_mat_q4k_q8_1_tiled.hlsl (one Q8_1 block
// per pair of barriers), widened from a 32x32 / 2x2 tile to 64x64 / 4x4.
// The 2x2 inner loop ran 4 dot chains over 4 LDS reads; 4x4 runs 16 chains over
// 8 reads and covers 4x the output per group.
//
// Dispatch: groups_x = ceil(N/64), groups_y = ceil(M/64), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K          256
#define Q4K_BSIZE     144
#define Q8_1_BSIZE     36
#define BM             64
#define BN             64
#define TM              4
#define TN              4

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d[BN];
groupshared float tile_w_m[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];
groupshared float tile_a_s[BM];

uint read_byte_q4k_intdot_64(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16u + tx;

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_q8_blocks = K / 32u;

    // 256 threads stage 64 rows x 8 words per side, so each thread owns one
    // word column in two rows 32 apart; the word column is rep-invariant.
    uint packed_idx = flat_id & 7u;
    uint local_row0 = flat_id >> 3u;
    uint qs_lane_off = 16u + packed_idx * 4u;

    bool w_row_ok[2];
    uint w_row_off[2];
    bool a_row_ok[2];
    uint a_row_off[2];
    [unroll]
    for (uint r = 0; r < 2u; ++r) {
        uint local_row = local_row0 + r * 32u;

        uint global_n = gid.x * BN + local_row;
        w_row_ok[r]  = global_n < ne01;
        w_row_off[r] = w_row_ok[r]
            ? src0_offset + global_n * nb01 + i2_src0 * nb02 + i3_src0 * nb03
            : 0u;

        uint global_m = gid.y * BM + local_row;
        uint flat_row = (i3 * ne12 + i2) * ne11 + global_m;
        a_row_ok[r]  = global_m < ne11;
        a_row_off[r] = a_row_ok[r]
            ? src1_offset + flat_row * num_q8_blocks * Q8_1_BSIZE
            : 0u;
    }

    uint m_base = ty * TM;
    uint n_base = tx * TN;

    precise float acc[TM][TN];
    [unroll]
    for (uint im = 0; im < TM; ++im) {
        [unroll]
        for (uint in_ = 0; in_ < TN; ++in_) {
            acc[im][in_] = 0.0f;
        }
    }

    for (uint block = 0; block < num_q8_blocks; ++block) {
        uint q4_block = block / 8u;
        uint tile_in_block = block & 7u;
        uint il = tile_in_block / 2u;
        bool high_nibble = (tile_in_block & 1u) != 0u;
        uint qs_off = qs_lane_off + il * 32u;

        // Scale/min sub-block indexing is row-invariant, so it is resolved
        // once per K-block and reused by both staging reps.
        uint is_eff = tile_in_block;
        bool low_half = il < 2u;
        uint scidx0 = low_half ? is_eff : (is_eff + 4u);
        uint scidx1 = low_half ? is_eff : (is_eff - 4u);
        uint scmask1 = low_half ? 0x30u : 0xC0u;
        uint scshift1 = low_half ? 0u : 2u;
        uint mbidx0 = is_eff + 4u;
        uint mbidx1 = low_half ? (is_eff + 4u) : is_eff;
        uint mbmask0 = low_half ? 0x0Fu : 0xF0u;
        uint mbshift0 = low_half ? 0u : 4u;
        uint mbmask1 = low_half ? 0x30u : 0xC0u;
        uint mbshift1 = low_half ? 0u : 2u;

        [unroll]
        for (uint rep = 0; rep < 2u; ++rep) {
            uint local_row = local_row0 + rep * 32u;

            uint w_qs = 0u;
            if (w_row_ok[rep]) {
                uint block_off = w_row_off[rep] + q4_block * Q4K_BSIZE;
                uint raw_qs = src0.Load(block_off + qs_off);
                w_qs = high_nibble ? ((raw_qs >> 4u) & 0x0F0F0F0Fu)
                                   : (raw_qs & 0x0F0F0F0Fu);

                if (packed_idx == 0u) {
                    uint dm_raw = src0.Load(block_off);
                    float dall = f16_to_f32(dm_raw & 0xFFFFu);
                    float dmin = f16_to_f32(dm_raw >> 16);
                    uint scales_off = block_off + 4u;

                    uint sc = (read_byte_q4k_intdot_64(src0, scales_off + scidx0) & 0x0Fu)
                            | ((read_byte_q4k_intdot_64(src0, scales_off + scidx1) & scmask1) >> scshift1);
                    uint m = ((read_byte_q4k_intdot_64(src0, scales_off + mbidx0) & mbmask0) >> mbshift0)
                           | ((read_byte_q4k_intdot_64(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);

                    tile_w_d[local_row] = dall * float(sc);
                    tile_w_m[local_row] = dmin * float(m);
                }
            } else if (packed_idx == 0u) {
                tile_w_d[local_row] = 0.0f;
                tile_w_m[local_row] = 0.0f;
            }
            tile_w_qs[local_row][packed_idx] = w_qs;

            uint a_qs = 0u;
            if (a_row_ok[rep]) {
                uint a_off = a_row_off[rep] + block * Q8_1_BSIZE;
                uint ds = src1.Load(a_off);
                a_qs = src1.Load(a_off + 4u + packed_idx * 4u);
                if (packed_idx == 0u) {
                    tile_a_d[local_row] = f16_to_f32(ds & 0xFFFFu);
                    tile_a_s[local_row] = f16_to_f32(ds >> 16);
                }
            } else if (packed_idx == 0u) {
                tile_a_d[local_row] = 0.0f;
                tile_a_s[local_row] = 0.0f;
            }
            tile_a_qs[local_row][packed_idx] = a_qs;
        }

        GroupMemoryBarrierWithGroupSync();

        int dots[TM][TN];
        [unroll]
        for (uint im = 0; im < TM; ++im) {
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                dots[im][in_] = 0;
            }
        }

        [unroll]
        for (uint q = 0; q < 8u; ++q) {
            uint a[TM];
            uint w[TN];
            [unroll]
            for (uint im = 0; im < TM; ++im) {
                a[im] = tile_a_qs[m_base + im][q];
            }
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                w[in_] = tile_w_qs[n_base + in_][q];
            }
            [unroll]
            for (uint im = 0; im < TM; ++im) {
                [unroll]
                for (uint in_ = 0; in_ < TN; ++in_) {
                    dots[im][in_] = dot4add_i8packed(w[in_], a[im], dots[im][in_]);
                }
            }
        }

        [unroll]
        for (uint im = 0; im < TM; ++im) {
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                acc[im][in_] += tile_w_d[n_base + in_] * tile_a_d[m_base + im] *
                                float(dots[im][in_])
                              - tile_w_m[n_base + in_] * tile_a_s[m_base + im];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (uint im = 0; im < TM; ++im) {
        uint global_m = gid.y * BM + m_base + im;
        if (global_m >= ne1) {
            continue;
        }
        [unroll]
        for (uint in_ = 0; in_ < TN; ++in_) {
            uint global_n = gid.x * BN + n_base + in_;
            if (global_n >= ne0) {
                continue;
            }
            store_auto(dst,
                       offset_4d(global_n, global_m, i2, i3, nb0, nb1, nb2, nb3, dst_offset),
                       acc[im][in_], dst_esize);
        }
    }
}
