// mul_mat.hlsl - Tiled matrix multiplication using shared memory
// ggml MUL_MAT: dst[i1, i0] = sum_k(src0[i0, k] * src1[i1, k])
//
// src0: weights, ne00 = K (inner), ne01 = N (output features) — F32 or F16
// src1: input,   ne10 = K (inner), ne11 = M (batch)            — F32
// dst:  output,  ne0  = N,         ne1  = M                    — F32
//
// Dispatch: groups_x = ceil(N/16), groups_y = ceil(M/16), groups_z = ne2*ne3
#include "ggml_common.hlsli"

#define TILE_SIZE 16

groupshared float tile_a[TILE_SIZE][TILE_SIZE];
groupshared float tile_b[TILE_SIZE][TILE_SIZE];

[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint tx = gtid.x;
    uint ty = gtid.y;

    uint col_n = gid.x * TILE_SIZE + tx; // output feature index (i0)
    uint row_m = gid.y * TILE_SIZE + ty; // batch index (i1)

    // Decompose batch dimension
    uint batch = gid.z;
    uint i2 = batch % ne2;
    uint i3 = batch / ne2;

    // Broadcast: src0 may have fewer batch dims than dst (GQA attention)
    // Use integer division to map contiguous groups: i2_src0 = i2 / (ne2/ne02)
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    precise float acc = 0.0f;

    for (uint t = 0; t < num_tiles; t++) {
        uint k_a = t * TILE_SIZE + tx; // k index for tile_a load
        uint n_a = gid.x * TILE_SIZE + ty; // n index for tile_a load

        // Load tile_a[ty][tx] = src0[k=k_a, n=n_a] with broadcast
        if (k_a < K && n_a < ne01) {
            uint off0 = offset_4d(k_a, n_a, i2_src0, i3_src0, nb00, nb01, nb02, nb03, src0_offset);
            tile_a[ty][tx] = load_auto(src0, off0, src0_esize);
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        uint k_b = t * TILE_SIZE + tx; // k index for tile_b load
        uint m_b = gid.y * TILE_SIZE + ty; // m index for tile_b load

        // Load tile_b[ty][tx] = src1[k=k_b, m=m_b]
        if (k_b < K && m_b < ne11) {
            uint off1 = offset_4d(k_b, m_b, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
            tile_b[ty][tx] = load_auto(src1, off1, src1_esize);
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_SIZE; k++) {
            acc += tile_a[tx][k] * tile_b[ty][k];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (col_n < ne0 && row_m < ne1) {
        uint off_d = offset_4d(col_n, row_m, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, acc, dst_esize);
    }
}
