#include "ggml_common.hlsli"

groupshared float selected[256];

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint token = gid.x % ne2;
    uint batch = gid.x / ne2;
    uint n_used = ne1;

    float value = 0.0f;
    if (tid < n_used) {
        uint id_off = src1_offset + tid * nb10 + token * nb11 + batch * nb12;
        uint expert = (uint)asint(src1.Load(id_off));
        uint prob_off = src0_offset + expert * nb01 + token * nb02 + batch * nb03;
        value = asfloat(src0.Load(prob_off));
    }
    selected[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            precise float a = selected[tid];
            precise float b = selected[tid + s];
            selected[tid] = a + b;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid < n_used) {
        float denom = max(selected[0], asfloat(op0));
        uint dst_off = dst_offset + tid * nb0 + token * nb1 + batch * nb2;
        dst.Store(dst_off, asuint(value / denom));
    }
}
