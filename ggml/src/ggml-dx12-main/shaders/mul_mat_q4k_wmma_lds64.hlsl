// mul_mat_q4k_wmma_lds64.hlsl - 64x64 cooperative-LDS Q4_K batch MUL_MAT
//
// Same cooperative Q4_K decode as mul_mat_q4k_wmma_lds.hlsl (scale/min and qs
// staged in LDS once per K-tile per column), but with a 64x64 output tile and
// 4x4 register blocking instead of 32x32 / 2x2.
//
// The 2x2 inner loop did 4 FMAs per 4 LDS reads (1:1), which caps the shader
// well below the FP32 roofline. 4x4 does 16 FMAs per 8 reads (2:1) and covers
// 4x the output per group, halving global traffic per output element.
//
// Q4_K block = 144 bytes: [d:f16][dmin:f16][scales:12B][qs:128B]
// BK=32 keeps every K-tile inside a single Q4_K block so one (sc_idx, mb_val)
// pair governs the whole tile for a given column.
//
// Dispatch: groups_x = ceil(N/64), groups_y = ceil(M/64), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define QK_K           256
#define Q4K_BLOCK_SIZE 144

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4
#define THREADS 256

groupshared float tile_a[BM][BK];
groupshared float tile_b[BK][BN];
groupshared float d_eff_lds[BN];
groupshared float m_eff_lds[BN];
groupshared uint  qs_dw_lds[BN][BK / 4];

uint read_byte_q4k_lds64(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;
    uint flat_id = ty * 16 + tx;

    uint col_block = gid.x;
    uint row_block = gid.y;
    uint batch     = gid.z;

    uint i2 = batch % ne2;
    uint i3 = batch / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_k_tiles = (K + BK - 1) / BK;

    precise float acc[TM][TN];
    [unroll] for (uint im = 0; im < TM; im++) {
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            acc[im][in_] = 0.0f;
        }
    }

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        // tile_a: BM x BK activations (2048 floats, 8 per thread).
        {
            uint base = flat_id * 8;
            [unroll] for (uint e = 0; e < 8; e++) {
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

        uint block_idx        = k_start / QK_K;
        uint tile_idx_in_blk  = (k_start % QK_K) / BK;     // 0..7
        uint il               = tile_idx_in_blk / 2u;       // 0..3
        bool is_high          = (tile_idx_in_blk & 1u) != 0u;
        uint is_eff           = tile_idx_in_blk;
        bool is_lt4           = (il < 2u);

        // Phase A: decode (d_eff, m_eff) once per N column (64 columns).
        if (flat_id < BN) {
            uint n_local  = flat_id;
            uint global_n = col_block * BN + n_local;
            float d_eff = 0.0f;
            float m_eff = 0.0f;
            if (global_n < ne01) {
                uint row_off   = src0_offset + global_n * nb01
                               + i2_src0 * nb02 + i3_src0 * nb03;
                uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;

                uint dm_raw = src0.Load(block_off);
                float dall    = f16_to_f32(dm_raw & 0xFFFFu);
                float dmin_v  = f16_to_f32(dm_raw >> 16);

                uint scales_off = block_off + 4;

                uint scidx0 = is_lt4 ? is_eff : (is_eff + 4u);
                uint scidx1 = is_lt4 ? is_eff : (is_eff - 4u);
                uint scmask1 = is_lt4 ? 0x30u : 0xC0u;
                uint scshift1 = is_lt4 ? 0u : 2u;
                uint mbidx0 = is_eff + 4u;
                uint mbidx1 = is_lt4 ? (is_eff + 4u) : is_eff;
                uint mbmask0 = is_lt4 ? 0x0Fu : 0xF0u;
                uint mbshift0 = is_lt4 ? 0u : 4u;
                uint mbmask1 = is_lt4 ? 0x30u : 0xC0u;
                uint mbshift1 = is_lt4 ? 0u : 2u;

                uint sc_idx = (read_byte_q4k_lds64(src0, scales_off + scidx0) & 0x0Fu)
                            | ((read_byte_q4k_lds64(src0, scales_off + scidx1) & scmask1) >> scshift1);
                uint mb_val = ((read_byte_q4k_lds64(src0, scales_off + mbidx0) & mbmask0) >> mbshift0)
                            | ((read_byte_q4k_lds64(src0, scales_off + mbidx1) & mbmask1) >> mbshift1);

                d_eff = dall   * float(sc_idx);
                m_eff = dmin_v * float(mb_val);
            }
            d_eff_lds[n_local] = d_eff;
            m_eff_lds[n_local] = m_eff;
        }

        // Phase B: 64 cols x 8 dwords = 512 loads, 2 per thread.
        {
            [unroll] for (uint rep = 0; rep < 2u; rep++) {
                uint t = flat_id + rep * THREADS;
                uint n_local   = t / 8u;
                uint dw_in_col = t % 8u;
                uint global_n  = col_block * BN + n_local;
                uint qs_dw = 0u;
                if (global_n < ne01) {
                    uint row_off   = src0_offset + global_n * nb01
                                   + i2_src0 * nb02 + i3_src0 * nb03;
                    uint block_off = row_off + block_idx * Q4K_BLOCK_SIZE;
                    uint qs_off    = block_off + 16u + il * 32u + dw_in_col * 4u;
                    qs_dw = src0.Load(qs_off);
                }
                qs_dw_lds[n_local][dw_in_col] = qs_dw;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Phase C: dequant tile_b[k][n] (2048 floats, 8 per thread).
        {
            uint base = flat_id * 8;
            [unroll] for (uint e = 0; e < 8; e++) {
                uint idx = base + e;
                uint k_local = idx / BN;
                uint n_local = idx % BN;
                uint global_k = k_start + k_local;
                uint global_n = col_block * BN + n_local;
                float val = 0.0f;
                if (global_k < K && global_n < ne01) {
                    uint qs_dw = qs_dw_lds[n_local][k_local / 4u];
                    uint qs_byte = (qs_dw >> ((k_local & 3u) * 8u)) & 0xFFu;
                    uint q = is_high ? (qs_byte >> 4) : (qs_byte & 0x0Fu);
                    val = d_eff_lds[n_local] * float(q) - m_eff_lds[n_local];
                }
                tile_b[k_local][n_local] = val;
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Thread (tx, ty) owns output rows [ty*TM .. +3], cols [tx*TN .. +3].
        [unroll]
        for (uint k = 0; k < BK; k++) {
            float a[TM];
            float b[TN];
            [unroll] for (uint im = 0; im < TM; im++) {
                a[im] = tile_a[ty * TM + im][k];
            }
            [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                b[in_] = tile_b[k][tx * TN + in_];
            }
            [unroll] for (uint im = 0; im < TM; im++) {
                [unroll] for (uint in_ = 0; in_ < TN; in_++) {
                    acc[im][in_] += a[im] * b[in_];
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    [unroll] for (uint im = 0; im < TM; im++) {
        uint global_m = row_block * BM + ty * TM + im;
        if (global_m >= ne1) continue;
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            uint global_n = col_block * BN + tx * TN + in_;
            if (global_n >= ne0) continue;
            uint off = offset_4d(global_n, global_m, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
            store_auto(dst, off, acc[im][in_], dst_esize);
        }
    }
}
