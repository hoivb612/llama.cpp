// Tiled Q6_K x Q8_1 batch MUL_MAT using packed int8 dot products, 64x64 tile.
//
// Same K blocking and staging as mul_mat_q6k_q8_1_tiled.hlsl (one Q8_1 block
// per pair of barriers), widened from a 32x32 / 2x2 tile to 64x64 / 4x4.
// The 2x2 inner loop ran 8 dot chains over 4 LDS reads; 4x4 runs 32 chains over
// 8 reads and covers 4x the output per group.
//
// Dispatch: groups_x = ceil(N/64), groups_y = ceil(M/64), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK_K          256
#define Q6K_BSIZE     210
#define Q8_1_BSIZE     36
#define BM             64
#define BN             64
#define TM              4
#define TN              4

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d0[BN];
groupshared float tile_w_d1[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];

// Q6_K superblocks are 210 bytes, so neither the block base nor the row base
// is 4-byte aligned; every load has to go through the unaligned path.
uint read_byte_q6k_64(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int read_i8_q6k_64(ByteAddressBuffer buf, uint byte_off) {
    return int(read_byte_q6k_64(buf, byte_off) ^ 0x80u) - 128;
}

float read_f16_q6k_64(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_u32_q6k_64(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
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
    uint lane_off = packed_idx * 4u;

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
        uint q6_block = block / 8u;
        uint tile_in_block = block & 7u;
        // Each 32-element tile sits in one 128-element half (h) and one of the
        // four 32-element quarters (sub) that ggml's dequant walks in lockstep.
        uint sub = tile_in_block & 3u;
        uint h = tile_in_block >> 2u;

        // Superblock-relative offsets are row-invariant, so they are resolved
        // once per K-block and reused by both staging reps.
        uint sb_off    = q6_block * Q6K_BSIZE;
        uint ql_off    = sb_off + h * 64u + (sub & 1u) * 32u + lane_off;
        uint qh_off    = sb_off + 128u + h * 32u + lane_off;
        uint qh_shift  = 2u * sub;
        bool ql_high   = sub >= 2u;
        // 16 scales per superblock, so one tile spans two of them: elements
        // 0-15 take sc0 and 16-31 take sc0 + 1.
        uint sc_off    = sb_off + 192u + h * 8u + 2u * sub;
        uint d_off     = sb_off + 208u;

        [unroll]
        for (uint rep = 0; rep < 2u; ++rep) {
            uint local_row = local_row0 + rep * 32u;

            uint w_qs = 0u;
            if (w_row_ok[rep]) {
                uint row_off = w_row_off[rep];

                uint ql_word = read_u32_q6k_64(src0, row_off + ql_off);
                uint low = ql_high ? ((ql_word >> 4u) & 0x0F0F0F0Fu)
                                   : (ql_word & 0x0F0F0F0Fu);
                uint qh_word = read_u32_q6k_64(src0, row_off + qh_off);
                uint high = ((qh_word >> qh_shift) & 0x03030303u) << 4u;

                // v holds 0..63 per byte; subtract 32 by keeping the low 5 bits
                // and sign-filling the top 3 when bit 5 is clear
                // (0x20 * 0x07 == 0xE0).
                uint v = low | high;
                uint neg = (v & 0x20202020u) ^ 0x20202020u;
                w_qs = (v & 0x1F1F1F1Fu) | (neg * 0x07u);

                if (packed_idx == 0u) {
                    float d = read_f16_q6k_64(src0, row_off + d_off);
                    tile_w_d0[local_row] = d * float(read_i8_q6k_64(src0, row_off + sc_off));
                    tile_w_d1[local_row] = d * float(read_i8_q6k_64(src0, row_off + sc_off + 1u));
                }
            } else if (packed_idx == 0u) {
                tile_w_d0[local_row] = 0.0f;
                tile_w_d1[local_row] = 0.0f;
            }
            tile_w_qs[local_row][packed_idx] = w_qs;

            uint a_qs = 0u;
            if (a_row_ok[rep]) {
                uint a_off = a_row_off[rep] + block * Q8_1_BSIZE;
                a_qs = src1.Load(a_off + 4u + lane_off);
                if (packed_idx == 0u) {
                    tile_a_d[local_row] = f16_to_f32(src1.Load(a_off) & 0xFFFFu);
                }
            } else if (packed_idx == 0u) {
                tile_a_d[local_row] = 0.0f;
            }
            tile_a_qs[local_row][packed_idx] = a_qs;
        }

        GroupMemoryBarrierWithGroupSync();

        int dots_a[TM][TN];
        int dots_b[TM][TN];
        [unroll]
        for (uint im = 0; im < TM; ++im) {
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                dots_a[im][in_] = 0;
                dots_b[im][in_] = 0;
            }
        }

        [unroll]
        for (uint q = 0; q < 4u; ++q) {
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
                    dots_a[im][in_] = dot4add_i8packed(w[in_], a[im], dots_a[im][in_]);
                }
            }
        }

        [unroll]
        for (uint q2 = 4u; q2 < 8u; ++q2) {
            uint a[TM];
            uint w[TN];
            [unroll]
            for (uint im = 0; im < TM; ++im) {
                a[im] = tile_a_qs[m_base + im][q2];
            }
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                w[in_] = tile_w_qs[n_base + in_][q2];
            }
            [unroll]
            for (uint im = 0; im < TM; ++im) {
                [unroll]
                for (uint in_ = 0; in_ < TN; ++in_) {
                    dots_b[im][in_] = dot4add_i8packed(w[in_], a[im], dots_b[im][in_]);
                }
            }
        }

        [unroll]
        for (uint im = 0; im < TM; ++im) {
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                acc[im][in_] += tile_a_d[m_base + im] *
                                (tile_w_d0[n_base + in_] * float(dots_a[im][in_]) +
                                 tile_w_d1[n_base + in_] * float(dots_b[im][in_]));
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
