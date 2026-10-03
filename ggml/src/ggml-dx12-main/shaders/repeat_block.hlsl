// repeat_block.hlsl - byte-level REPEAT for types load_auto cannot express
// (I16, I64 and quants). One thread per block; block byte size is nb0 and
// blocks-per-row is derived from the row stride. Requires an even block size.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_id : SV_GroupThreadID) {
    uint group_lin = gid.y * 65535u + gid.x;
    uint block_idx = group_lin * 256u + local_id;

    uint bpr_d = nb1 / nb0;
    uint blck  = ne0 / bpr_d;
    uint bpr_0 = ne00 / blck;

    uint total_blocks = bpr_d * ne1 * ne2 * ne3;
    if (block_idx >= total_blocks) return;

    uint b0 = block_idx % bpr_d;
    uint rem = block_idx / bpr_d;
    uint i1 = rem % ne1;
    rem = rem / ne1;
    uint i2 = rem % ne2;
    uint i3 = rem / ne2;

    uint src_off = src0_offset + (b0 % bpr_0) * nb00 + (i1 % ne01) * nb01 +
                   (i2 % ne02) * nb02 + (i3 % ne03) * nb03;
    uint dst_off = dst_offset + b0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3;

    uint halves = nb0 >> 1;
    for (uint i = 0; i < halves; ++i) {
        dst.Store<uint16_t>(dst_off + i * 2u, src0.Load<uint16_t>(src_off + i * 2u));
    }
}
