// sum.hlsl - dst[0] = sum of all elements in src0 (full 4D reduction)
// dst is scalar; dispatched as 1 group of 256 threads, each accumulating a strided portion.
#include "ggml_common.hlsli"

groupshared float shared_sum[256];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint local_id = gtid.x;
    uint total = ne00 * ne01 * ne02 * ne03;

    precise float local_sum = 0.0f;
    for (uint flat = local_id; flat < total; flat += 256) {
        uint i0 = flat % ne00; uint rem = flat / ne00;
        uint i1 = rem % ne01; rem = rem / ne01;
        uint i2 = rem % ne02; uint i3 = rem / ne02;
        uint off = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        local_sum += load_auto(src0, off, src0_esize);
    }

    shared_sum[local_id] = local_sum;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            precise float a = shared_sum[local_id];
            precise float b = shared_sum[local_id + s];
            shared_sum[local_id] = a + b;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        store_auto(dst, dst_offset, shared_sum[0], dst_esize);
    }
}
