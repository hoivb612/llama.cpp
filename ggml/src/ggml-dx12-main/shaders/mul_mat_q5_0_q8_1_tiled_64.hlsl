// Tiled Q5_0 x Q8_1 batch MUL_MAT using packed int8 dot products, 64x64 tile.
//
// Same K blocking and staging as mul_mat_q5_0_q8_1_tiled.hlsl (one Q5_0/Q8_1
// block per pair of barriers), widened from a 32x32 / 2x2 tile to 64x64 / 4x4.
// The 2x2 inner loop ran 4 dot chains over 4 LDS reads; 4x4 runs 16 chains over
// 8 reads and covers 4x the output per group.
//
// Dispatch: groups_x = ceil(N/64), groups_y = ceil(M/64), groups_z = ne2*ne3

#include "ggml_common.hlsli"

#define QK5_0       32
#define Q5_0_BSIZE  22
#define Q8_1_BSIZE  36
#define BM          64
#define BN          64
#define TM          4
#define TN          4

groupshared uint  tile_w_qs[BN][8];
groupshared float tile_w_d[BN];
groupshared uint  tile_a_qs[BM][8];
groupshared float tile_a_d[BM];

float read_f16_q50_64(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xFFFFu);
}

uint read_u32_q50_64(ByteAddressBuffer buf, uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = buf.Load(aligned);
    uint hi = buf.Load(aligned + (shift == 0u ? 0u : 4u));
    if (shift == 0u) {
        return lo;
    }
    return (lo >> shift) | (hi << (32u - shift));
}

// Rebuild 4 consecutive Q5_0 weights as packed int8. Element i of the block
// takes its low 4 bits from nibble i of qs and its 5th bit from qh bit i,
// biased by -16. Group p covers elements 4p..4p+3, which live in the low
// nibbles of qs word p for p < 4 and the high nibbles of qs word p-4 above.
uint unpack4_q5_0_64(uint qs_word, uint qh, uint packed_idx) {
    uint nib = packed_idx < 4u
        ? (qs_word & 0x0F0F0F0Fu)
        : ((qs_word >> 4u) & 0x0F0F0F0Fu);

    uint hb = (qh >> (packed_idx * 4u)) & 0xFu;
    uint spread = ((hb & 1u) << 4u) | (((hb >> 1u) & 1u) << 12u) |
                  (((hb >> 2u) & 1u) << 20u) | (((hb >> 3u) & 1u) << 28u);

    // v holds 0..31 per byte; subtracting 16 without cross-byte borrow means
    // keeping the low nibble and sign-filling the high nibble when bit 4 is
    // clear (0x10 * 0x0F == 0xF0 stays inside its own byte).
    uint v = nib | spread;
    uint neg = (v & 0x10101010u) ^ 0x10101010u;
    return (v & 0x0F0F0F0Fu) | (neg * 0x0Fu);
}

int dot4_i8_q50_64(uint a, uint b, int acc) {
    return dot4add_i8packed(a, b, acc);
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
    uint num_blocks = K / QK5_0;

    // 256 threads stage 64 rows x 8 words per side, so each thread owns one
    // word column in two rows 32 apart; the word column is rep-invariant.
    uint packed_idx = flat_id & 7u;
    uint local_row0 = flat_id >> 3u;
    uint qs_delta   = 6u + (packed_idx < 4u ? packed_idx : packed_idx - 4u) * 4u;

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
            ? src1_offset + flat_row * num_blocks * Q8_1_BSIZE
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

    for (uint block = 0; block < num_blocks; ++block) {
        [unroll]
        for (uint rep = 0; rep < 2u; ++rep) {
            uint local_row = local_row0 + rep * 32u;

            uint w_qs = 0u;
            if (w_row_ok[rep]) {
                uint w_off = w_row_off[rep] + block * Q5_0_BSIZE;
                uint qh = read_u32_q50_64(src0, w_off + 2u);
                w_qs = unpack4_q5_0_64(read_u32_q50_64(src0, w_off + qs_delta),
                                       qh, packed_idx);
                if (packed_idx == 0u) {
                    tile_w_d[local_row] = read_f16_q50_64(src0, w_off);
                }
            } else if (packed_idx == 0u) {
                tile_w_d[local_row] = 0.0f;
            }
            tile_w_qs[local_row][packed_idx] = w_qs;

            uint a_qs = 0u;
            if (a_row_ok[rep]) {
                uint a_off = a_row_off[rep] + block * Q8_1_BSIZE;
                a_qs = src1.Load(a_off + 4u + packed_idx * 4u);
                if (packed_idx == 0u) {
                    tile_a_d[local_row] = f16_to_f32(src1.Load(a_off) & 0xFFFFu);
                }
            } else if (packed_idx == 0u) {
                tile_a_d[local_row] = 0.0f;
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
                    dots[im][in_] = dot4_i8_q50_64(w[in_], a[im], dots[im][in_]);
                }
            }
        }

        [unroll]
        for (uint im = 0; im < TM; ++im) {
            [unroll]
            for (uint in_ = 0; in_ < TN; ++in_) {
                acc[im][in_] += tile_w_d[n_base + in_] * tile_a_d[m_base + im] *
                                float(dots[im][in_]);
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
