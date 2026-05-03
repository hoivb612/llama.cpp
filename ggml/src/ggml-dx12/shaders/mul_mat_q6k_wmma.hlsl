// mul_mat_q6k_wmma.hlsl - Register-blocked tiled MUL_MAT with Q6_K weights
// Q6_K block layout: ql[128] + qh[64] + scales[16] + d(f16) = 210 bytes
// BK=16 aligns with Q6_K scale granularity (16 elements per scale).
//
// Dispatch: groups_x = ceil(N/32), groups_y = ceil(M/32), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK_K 256
#define Q6K_BLOCK_SIZE 210

#define BM 32
#define BN 32
#define BK 16

uint read_byte_q6k(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

int read_sbyte_q6k(ByteAddressBuffer buf, uint byte_off) {
    uint b = read_byte_q6k(buf, byte_off);
    return (b < 128) ? (int)b : (int)b - 256;
}

float dequant_q6k(ByteAddressBuffer buf, uint block_off, uint elem_in_block) {
    uint ql_off = block_off;
    uint qh_off = block_off + 128;
    uint scales_off = block_off + 192;
    uint d_off = block_off + 208;

    uint d_word = buf.Load(d_off & ~3u);
    float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

    uint ip = elem_in_block / 128;
    uint il = elem_in_block % 128;

    uint is = 8 * ip + il / 16;
    int scale = read_sbyte_q6k(buf, scales_off + is);

    uint ql_idx = 64 * ip + (il % 64);
    uint ql_val = read_byte_q6k(buf, ql_off + ql_idx);

    uint qh_val = read_byte_q6k(buf, qh_off + 32 * ip + (il % 32));

    int q;
    if (il < 32) {
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 0) & 3u) << 4)) - 32;
    } else if (il < 64) {
        q = (int)((ql_val & 0x0Fu) | (((qh_val >> 2) & 3u) << 4)) - 32;
    } else if (il < 96) {
        q = (int)((ql_val >> 4) | (((qh_val >> 4) & 3u) << 4)) - 32;
    } else {
        q = (int)((ql_val >> 4) | (((qh_val >> 6) & 3u) << 4)) - 32;
    }

    return d * (float)scale * (float)q;
}

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // Load tile_a from src1 (512 elements, 256 threads, 2 each)
        {
            uint base = flat_id * 2;
            [unroll] for (uint e = 0; e < 2; e++) {
                uint idx = base + e;
                uint m_local = idx / BK;
                uint k_local = idx % BK;
                uint global_m = row_block * BM + m_local;
                uint global_k = k_start + k_local;
                float val = 0.0f;
                if (global_m < ne11 && global_k < K) {
                    uint off = offset_4d(global_k, global_m, i2, i3,
                                         nb10, nb11, nb12, nb13, src1_offset);
                    val = load_auto(src1, off, src1_esize);
                }
                tile_a[m_local][k_local] = val;
            }
        }

        // Load tile_b: dequantize Q6_K (512 elements, 256 threads, 2 each)
        {
            uint base = flat_id * 2;
            [unroll] for (uint e = 0; e < 2; e++) {
                uint idx = base + e;
                uint k_local = idx / BN;
                uint n_local = idx % BN;
                uint global_k = k_start + k_local;
                uint global_n = col_block * BN + n_local;
                float val = 0.0f;
                if (global_k < K && global_n < ne01) {
                    uint block_idx = global_k / QK_K;
                    uint elem_in_block = global_k % QK_K;
                    uint row_off = src0_offset + global_n * nb01
                                 + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q6K_BLOCK_SIZE;
                    val = dequant_q6k(src0, block_off, elem_in_block);
                }
                tile_b[k_local][n_local] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < BK; k++) {
            float a0 = tile_a[ty * 2    ][k];
            float a1 = tile_a[ty * 2 + 1][k];
            float b0 = tile_b[k][tx * 2    ];
            float b1 = tile_b[k][tx * 2 + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    uint m0 = row_block * BM + ty * 2;
    uint m1 = m0 + 1;
    uint n0 = col_block * BN + tx * 2;
    uint n1 = n0 + 1;

    if (m0 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc00, dst_esize);
    }
    if (m0 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc01, dst_esize);
    }
    if (m1 < ne1 && n0 < ne0) {
        uint off = offset_4d(n0, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc10, dst_esize);
    }
    if (m1 < ne1 && n1 < ne0) {
        uint off = offset_4d(n1, m1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off, acc11, dst_esize);
    }
}
