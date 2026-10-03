#include "ggml_common.hlsli"

#ifndef FA_BR
#define FA_BR 16
#endif
#ifndef FA_BC
#define FA_BC 64
#endif
#if (WAVE_SIZE != 64 && WAVE_SIZE != 32 && WAVE_SIZE != 16) || (FA_BC != 64 && FA_BC != 32)
#error "Unsupported packed attention mask geometry"
#endif
#define MASK_ROW_LANES (FA_BC / 4)
#if (WAVE_SIZE % MASK_ROW_LANES) != 0 || (FA_BR % (WAVE_SIZE / MASK_ROW_LANES)) != 0
#error "Packed attention mask rows must divide the wave"
#endif

groupshared uint s_codes[4];

uint mask_class_half(uint bits) {
#if defined(FA_MASK_SCALAR)
    // Scalar prescan skips both infinity signs, but keeps NaNs.
    return (bits & 0x7FFFu) == 0u ? 2u : (bits & 0x7FFFu) == 0x7C00u ? 1u : 3u;
#else
    return (bits & 0x7FFFu) == 0u ? 2u : bits == 0xFC00u ? 1u : 3u;
#endif
}

uint mask_class_float(uint bits) {
#if defined(FA_MASK_SCALAR)
    return (bits & 0x7FFFFFFFu) == 0u ? 2u : (bits & 0x7FFFFFFFu) == 0x7F800000u ? 1u : 3u;
#else
    return (bits & 0x7FFFFFFFu) == 0u ? 2u : bits == 0xFF800000u ? 1u : 3u;
#endif
}

WAVE_SIZE_ATTR
[numthreads(4 * WAVE_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint lane = gtid.x % WAVE_SIZE;
    const uint wave = gtid.x / WAVE_SIZE;
    const uint heads = op13 & 0xFFFFu;
    const uint head = gid.z % heads;
    const uint batch = gid.z / heads;
    const uint element_size = (op8 >> 16) & 0xFFu;
    const uint base = op9 + head * op11 + batch * op12;
    uint packed = 0u;

    // Each wave owns a tile. Only the final packed word needs a group rendezvous.
    [unroll] for (uint round = 0u; round < 4u; ++round) {
        const uint tile = wave + round * 4u;
        const uint key = (gid.x * 16u + tile) * FA_BC + (lane % MASK_ROW_LANES) * 4u;
        uint cls = 0u;
        [unroll] for (uint r = 0u; r < FA_BR; r += WAVE_SIZE / MASK_ROW_LANES) {
            const uint row = gid.y * FA_BR + lane / MASK_ROW_LANES + r;
            if (key < ne11 && row < ne01) {
                const uint offset = base + row * op10 + key * element_size;
                if (element_size == 2u) {
                    const uint2 value = src3.Load2(offset);
                    cls |= mask_class_half(value.x & 0xFFFFu) | mask_class_half(value.x >> 16);
                    cls |= mask_class_half(value.y & 0xFFFFu) | mask_class_half(value.y >> 16);
                } else {
                    const uint4 value = src3.Load4(offset);
                    cls |= mask_class_float(value.x) | mask_class_float(value.y);
                    cls |= mask_class_float(value.z) | mask_class_float(value.w);
                }
            }
        }
        cls = WaveActiveBitOr(cls);
        const uint code = cls == 3u ? 0u : cls == 0u ? 1u : cls;
        packed |= code << (2u * tile);
    }
    if (lane == 0u) {
        s_codes[wave] = packed;
    }
    GroupMemoryBarrierWithGroupSync();
    if (gtid.x == 0u) {
        const uint key_words = (ne11 + 16u * FA_BC - 1u) / (16u * FA_BC);
        const uint query_tiles = (ne01 + FA_BR - 1u) / FA_BR;
        const uint word = (gid.z * query_tiles + gid.y) * key_words + gid.x;
        temp.Store(word * 4u, s_codes[0] | s_codes[1] | s_codes[2] | s_codes[3]);
    }
}
