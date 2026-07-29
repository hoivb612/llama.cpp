// top_k.hlsl - per-row indices of the K largest src0 values.
//
// Mirrors argsort.hlsl DESC but only writes the first K indices. The order of
// the K written indices is not required to match the CPU reference (the test
// harness either compares values or only validates the final result via
// run_whole_graph() when ties=true).
//
// op_params layout: none required (k is dst->ne[0]).
// Supports src0->ne[0] <= 1024.
#include "ggml_common.hlsli"

#define BLOCK_SIZE 1024

groupshared int2 dst_row[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i1 = gid.x;
    uint i2 = gid.y;
    uint i3 = gid.z;
    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) return;

    uint col   = gtid.x;
    uint ncols = ne00;
    uint k     = ne0;  // dst innermost dim is K

    int val_bits = 0;
    if (col < ncols) {
        uint off = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01 + col * nb00;
        val_bits = asint(src0.Load(off));
    }
    dst_row[col] = int2((int)col, val_bits);
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint kk = 2; kk <= BLOCK_SIZE; kk *= 2) {
        [unroll] for (uint j = kk / 2; j > 0; j /= 2) {
            int icol = (int)col;
            int ixj  = icol ^ (int)j;
            int idx_0 = (icol & (int)kk) == 0 ? icol : ixj;
            int idx_1 = (icol & (int)kk) == 0 ? ixj  : icol;

            int2 sh_0 = dst_row[idx_0];
            int2 sh_1 = dst_row[idx_1];
            bool oob_0 = (uint)sh_0.x >= ncols;
            bool oob_1 = (uint)sh_1.x >= ncols;

            bool swap = (oob_0 || (!oob_1 && asfloat(sh_0.y) > asfloat(sh_1.y))) && (ixj > icol);
            if (swap) {
                dst_row[idx_0] = sh_1;
                dst_row[idx_1] = sh_0;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Result is ascending; we want the K largest, which are at the tail.
    // Write dst[w] = sorted_index from slot (ncols - 1 - w) for w in [0, k).
    if (col < k && col < ncols) {
        uint src_slot = ncols - 1u - col;
        uint dst_row_off = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
        dst.Store(dst_row_off + col * nb0, asuint(dst_row[src_slot].x));
    }
}
