// mul_mat_wmma64.hlsl - 64x64 register-blocked tiled matrix multiplication (FP32)
//
// FP32 counterpart of mul_mat_wmma_fp16.hlsl. Same 64x64 tile with 4x4
// register blocking, but without the -enable-16bit-types requirement so it
// can serve every src0/src1 type the generic loader handles.
//
// Versus the 32x32 / 2x2 mul_mat_wmma.hlsl this replaces: the inner loop
// does 16 FMAs per 8 LDS reads (2:1) instead of 4 FMAs per 4 reads (1:1),
// and covers 4x the output per thread group, so global traffic per output
// drops 2x as well.
//
// ggml MUL_MAT: dst[i1, i0] = sum_k(src0[i0, k] * src1[i1, k])
//
// Dispatch: groups_x = ceil(N/64), groups_y = ceil(M/64), groups_z = ne2*ne3
#include "ggml_common.hlsli"

// Fetch 4 consecutive unit-stride elements as float. Caller must guarantee the
// run is in range, contiguous and 4-byte aligned; BF16 (esize 3) is excluded.
void load_run4(ByteAddressBuffer buf, uint off0, uint esize, out float v[4]) {
    if (esize == 2) {
        uint2 w = buf.Load2(off0);
        v[0] = f16tof32(w.x & 0xFFFFu);
        v[1] = f16tof32(w.x >> 16);
        v[2] = f16tof32(w.y & 0xFFFFu);
        v[3] = f16tof32(w.y >> 16);
    } else {
        uint4 w = buf.Load4(off0);
        v[0] = asfloat(w.x);
        v[1] = asfloat(w.y);
        v[2] = asfloat(w.z);
        v[3] = asfloat(w.w);
    }
}

bool run4_ok(uint esize, uint nb0, uint off0) {
    return ((esize == 4 && nb0 == 4) || (esize == 2 && nb0 == 2)) && (off0 & 3u) == 0u;
}

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
#define THREADS 256

groupshared float tile_a[BM][BK]; // src1: batch x K
groupshared float tile_b[BK][BN]; // src0: K x output features

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x; // 0..15 -> col tile index
    uint ty = gtid.y; // 0..15 -> row tile index
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

    precise float acc[TM][TN];
    [unroll] for (uint im = 0; im < TM; im++) {
        [unroll] for (uint in_ = 0; in_ < TN; in_++) {
            acc[im][in_] = 0.0f;
        }
    }

    const uint LDS_PER_THREAD = (BM * BK) / THREADS; // == 4

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k_start = kt * BK;

        {
            uint base = flat_id * LDS_PER_THREAD;
            uint m = base / BK;
            uint k0 = base % BK;
            uint global_m = row_block * BM + m;
            uint global_k0 = k_start + k0;
            uint off0 = offset_4d(global_k0, global_m, i2, i3,
                                  nb10, nb11, nb12, nb13, src1_offset);
            if (global_m < ne11 && (global_k0 + LDS_PER_THREAD - 1) < K &&
                run4_ok(src1_esize, nb10, off0)) {
                float v[4];
                load_run4(src1, off0, src1_esize, v);
                [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                    tile_a[m][k0 + e] = v[e];
                }
            } else {
                [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                    uint k = k0 + e;
                    uint global_k = k_start + k;
                    float val = 0.0f;
                    if (global_m < ne11 && global_k < K) {
                        uint off = offset_4d(global_k, global_m, i2, i3,
                                             nb10, nb11, nb12, nb13, src1_offset);
                        val = load_auto(src1, off, src1_esize);
                    }
                    tile_a[m][k] = val;
                }
            }
        }

        // Walk K (the contiguous src0 axis) within a thread; mapping the fast
        // axis to N instead strides every load by nb01.
        {
            uint base = flat_id * LDS_PER_THREAD;
            uint n = base / BK;
            uint k0 = base % BK;
            uint global_n = col_block * BN + n;
            uint global_k0 = k_start + k0;
            uint off0 = offset_4d(global_k0, global_n, i2_src0, i3_src0,
                                  nb00, nb01, nb02, nb03, src0_offset);
            if (global_n < ne01 && (global_k0 + LDS_PER_THREAD - 1) < K &&
                run4_ok(src0_esize, nb00, off0)) {
                float v[4];
                load_run4(src0, off0, src0_esize, v);
                [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                    tile_b[k0 + e][n] = v[e];
                }
            } else {
                [unroll] for (uint e = 0; e < LDS_PER_THREAD; e++) {
                    uint k = k0 + e;
                    uint global_k = k_start + k;
                    float val = 0.0f;
                    if (global_n < ne01 && global_k < K) {
                        uint off = offset_4d(global_k, global_n, i2_src0, i3_src0,
                                             nb00, nb01, nb02, nb03, src0_offset);
                        val = load_auto(src0, off, src0_esize);
                    }
                    tile_b[k][n] = val;
                }
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
