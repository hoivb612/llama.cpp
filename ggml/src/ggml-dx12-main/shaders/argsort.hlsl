// argsort.hlsl - bitonic sort of indices per row.
//
// One thread group per row across (ne01, ne02, ne03). 1024 threads per group:
// each thread owns one slot in the padded sort array (BLOCK_SIZE = 1024).
// Cols >= ne00 are flagged as out-of-bounds so they sort to the tail; cols
// 0..ne00-1 end up sorted in ascending order of their src0 value. For DESC,
// the result is written in reverse: dst[ne00-1-col] = idx.
//
// op_params layout (filled by ggml-dx12.cpp):
//   op0 = ggml_sort_order (0=ASC, 1=DESC)
//
// Supports ne00 <= 1024 only; supports_op gates larger sizes.
#include "ggml_common.hlsli"

#define BLOCK_SIZE 1024

groupshared int2 dst_row[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint i1 = gid.x;
    uint i2 = gid.y;
    uint i3 = gid.z;
    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) return;

    uint col = gtid.x;
    uint ncols = ne00;
    uint order = op0;

    // Load value bits for cols < ncols; OOB slots get value 0 but rely on the
    // bounds check below to keep them at the tail of the sort.
    int val_bits = 0;
    if (col < ncols) {
        uint off = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01 + col * nb00;
        val_bits = asint(src0.Load(off));
    }
    dst_row[col] = int2((int)col, val_bits);
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort over the full BLOCK_SIZE.
    [unroll] for (uint k = 2; k <= BLOCK_SIZE; k *= 2) {
        [unroll] for (uint j = k / 2; j > 0; j /= 2) {
            int icol = (int)col;
            int ixj  = icol ^ (int)j;
            int idx_0 = (icol & (int)k) == 0 ? icol : ixj;
            int idx_1 = (icol & (int)k) == 0 ? ixj  : icol;

            int2 sh_0 = dst_row[idx_0];
            int2 sh_1 = dst_row[idx_1];
            bool oob_0 = (uint)sh_0.x >= ncols;
            bool oob_1 = (uint)sh_1.x >= ncols;

            // OOB slot is treated as +infinity (sorts to the high index side).
            bool swap = (oob_0 || (!oob_1 && asfloat(sh_0.y) > asfloat(sh_1.y))) && (ixj > icol);
            if (swap) {
                dst_row[idx_0] = sh_1;
                dst_row[idx_1] = sh_0;
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    if (col < ncols) {
        uint dst_row_off = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
        uint write_col = (order == 0u) ? col : (ncols - 1u - col);
        dst.Store(dst_row_off + write_col * nb0, asuint(dst_row[col].x));
    }
}
