// cpy_quant_block.hlsl - same-type quantized CPY/DUP/CONT.
// Byte-level block copy. One thread per block. Block byte size lives in
// src0_esize / dst_esize (== ggml_type_size). For all supported quants this
// is even (mxfp4 with its 17-byte block is filtered out by supports_op).
//
// op_params[2] = blocks_per_row (= ne0 / blck_size; same for src and dst
//                since dim 0 is preserved by the supported permutations and
//                both tensors are the same quant type).
//
// Mapping mirrors ggml_compute_forward_dup_bytes' non-contiguous path:
// the i-th block in src's canonical iteration order is written at the
// i-th block in dst's canonical iteration order. Because src and dst can
// have different post-permute shapes (e.g. src ne=[32,4,2,3], dst
// ne=[32,2,4,3]), src indices are derived from ne01/02/03 while dst
// indices are derived from ne1/2/3. Block offsets use src nb00..nb03 and
// dst nb0..nb3 respectively.
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_id : SV_GroupThreadID) {
    uint group_lin = gid.y * 65535u + gid.x;
    uint block_idx = group_lin * 256u + local_id;

    uint blocks_per_row = op2;
    uint total_blocks = blocks_per_row * ne01 * ne02 * ne03;
    if (block_idx >= total_blocks) return;

    uint sb0 = block_idx % blocks_per_row;
    uint rem = block_idx / blocks_per_row;
    uint sb1 = rem % ne01;
    rem = rem / ne01;
    uint sb2 = rem % ne02;
    uint sb3 = rem / ne02;

    uint db0 = block_idx % blocks_per_row;
    uint drem = block_idx / blocks_per_row;
    uint db1 = drem % ne1;
    drem = drem / ne1;
    uint db2 = drem % ne2;
    uint db3 = drem / ne2;

    uint src_off = src0_offset + sb0 * nb00 + sb1 * nb01 + sb2 * nb02 + sb3 * nb03;
    uint dst_off = dst_offset  + db0 * nb0  + db1 * nb1  + db2 * nb2  + db3 * nb3;

    uint block_bytes = src0_esize;
    uint halves = block_bytes >> 1;
    for (uint i = 0; i < halves; ++i) {
        uint16_t v = src0.Load<uint16_t>(src_off + i * 2u);
        dst.Store<uint16_t>(dst_off + i * 2u, v);
    }
}

