// count_equal.hlsl - dst[0] (I64) = count of indices where src0[i] == src1[i].
// Both src tensors are I32 with identical shape. dst is scalar I64.
// Dispatched as 1 group of 256 threads, accumulating strided counts.
#include "ggml_common.hlsli"

groupshared uint shared_count[256];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint local_id = gtid.x;
    uint total = ne00 * ne01 * ne02 * ne03;

    uint local_count = 0;
    for (uint flat = local_id; flat < total; flat += 256) {
        uint i0 = flat % ne00; uint rem = flat / ne00;
        uint i1 = rem % ne01; rem = rem / ne01;
        uint i2 = rem % ne02; uint i3 = rem / ne02;
        uint off0 = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off1 = offset_4d(i0, i1, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
        uint a = src0.Load(off0);
        uint b = src1.Load(off1);
        if (a == b) local_count++;
    }

    shared_count[local_id] = local_count;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared_count[local_id] += shared_count[local_id + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        // I64 little-endian: low 32 bits then high 32 bits.
        dst.Store(dst_offset,     shared_count[0]);
        dst.Store(dst_offset + 4, 0u);
    }
}
