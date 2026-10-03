// argmax.hlsl - per-row argmax of an F32 matrix; dst is I32.
// src0 is a matrix (ne02 = ne03 = 1); dst is 1D with ne01 elements.
// One thread group per row, 256-way parallel reduction picking the max value
// (ties broken by lowest index).
#include "ggml_common.hlsli"

groupshared float shared_val[256];
groupshared uint  shared_idx[256];

[numthreads(256, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint row = gid.x;
    if (row >= ne01) return;

    uint local_id = gtid.x;
    float best_val = -3.4028235e38f;
    uint  best_idx = 0;

    for (uint i = local_id; i < ne00; i += 256) {
        uint off = offset_4d(i, row, 0, 0, nb00, nb01, nb02, nb03, src0_offset);
        float v = asfloat(src0.Load(off));
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    shared_val[local_id] = best_val;
    shared_idx[local_id] = best_idx;
    GroupMemoryBarrierWithGroupSync();

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            float a = shared_val[local_id];
            float b = shared_val[local_id + s];
            uint  ai = shared_idx[local_id];
            uint  bi = shared_idx[local_id + s];
            // Strict > so ties favour the lower index already held in slot `a`.
            if (b > a) {
                shared_val[local_id] = b;
                shared_idx[local_id] = bi;
            } else {
                shared_val[local_id] = a;
                shared_idx[local_id] = ai;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local_id == 0) {
        uint off_dst = dst_offset + row * nb0;
        dst.Store(off_dst, shared_idx[0]);
    }
}
