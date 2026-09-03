#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define MAX_WAVES 32

groupshared float values[GROUP_SIZE];
groupshared float wave_max[MAX_WAVES];
groupshared uint wave_idx[MAX_WAVES];

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i1 = gid.x;
    uint i2 = gid.y;
    uint i3 = gid.z;
    if (i1 >= ne01 || i2 >= ne02 || i3 >= ne03) return;

    uint ncols = ne00;
    float value = -3.402823466e+38f;
    if (tid < ncols) {
        uint off = src0_offset + i3 * nb03 + i2 * nb02 + i1 * nb01 + tid * nb00;
        value = asfloat(src0.Load(off));
    }
    values[tid] = value;
    GroupMemoryBarrierWithGroupSync();

    uint lane = WaveGetLaneIndex();
    uint wave = tid / WaveGetLaneCount();
    uint num_waves = (GROUP_SIZE + WaveGetLaneCount() - 1) / WaveGetLaneCount();

    for (uint rank = 0; rank < op1; ++rank) {
        float local = values[tid];
        float wmax = WaveActiveMax(local);
        uint candidate = local == wmax ? tid : 0xFFFFFFFFu;
        uint widx = WaveActiveMin(candidate);
        if (lane == 0) {
            wave_max[wave] = wmax;
            wave_idx[wave] = widx;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid == 0) {
            float best = wave_max[0];
            uint best_idx = wave_idx[0];
            for (uint w = 1; w < num_waves; ++w) {
                float v = wave_max[w];
                uint idx = wave_idx[w];
                if (v > best || (v == best && idx < best_idx)) {
                    best = v;
                    best_idx = idx;
                }
            }
            uint dst_row = dst_offset + i3 * nb3 + i2 * nb2 + i1 * nb1;
            dst.Store(dst_row + rank * nb0, best_idx);
            values[best_idx] = -3.402823466e+38f;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
