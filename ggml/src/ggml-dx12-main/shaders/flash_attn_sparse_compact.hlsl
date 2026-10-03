#include "ggml_common.hlsli"

groupshared uint s_offsets[16];
groupshared uint s_count;
groupshared uint s_total;

WAVE_SIZE_ATTR
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint wave = tid.x / WARP_SIZE;
    const uint mask_ne2 = op13 & 0xFFFFu;
    const uint row = (gid.z * mask_ne2 + gid.y) * ne01 + gid.x;
    const uint out_base = row * (ne11 + 1u) * 4u;
    const uint mask_base = op9 + gid.x * op10 + gid.y * op11 + gid.z * op12;
    if (tid.x == 0u) {
        s_count = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint start = 0u; start < ne11; start += 256u) {
        const uint kv = start + tid.x;
        bool valid = false;
        if (kv < ne11) {
            valid = load_auto(src3, mask_base + kv * ((op8 >> 8) & 0xFFu),
                              (op8 >> 16) & 0xFFu) != asfloat(0xFF800000u);
        }
        const uint prefix = WavePrefixCountBits(valid);
        const uint count = WaveActiveCountBits(valid);
        if (WaveIsFirstLane()) {
            s_offsets[wave] = count;
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid.x == 0u) {
            uint total = 0u;
            for (uint w = 0u; w < 256u / WARP_SIZE; ++w) {
                const uint n = s_offsets[w];
                s_offsets[w] = total;
                total += n;
            }
            s_total = total;
        }
        GroupMemoryBarrierWithGroupSync();
        if (valid) {
            temp.Store(out_base + (1u + s_count + s_offsets[wave] + prefix) * 4u, kv);
        }
        GroupMemoryBarrierWithGroupSync();
        if (tid.x == 0u) {
            s_count += s_total;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0u) {
        temp.Store(out_base, s_count);
    }
}
