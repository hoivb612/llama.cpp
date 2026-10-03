// Fused Q8_0 gate/up matvec and SwiGLU with one Q8 block per lane.
// Wide loads replace the scalar packed-word stream used by the mr64 path.

#include "ggml_common.hlsli"

#define GROUP_SIZE 64
#define QK8_0 32
#define Q8_0_BSIZE 34
#define Q8_1_BSIZE 36

#if !(defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE))
groupshared float glu_wave_gate[GROUP_SIZE / 16];
groupshared float glu_wave_up[GROUP_SIZE / 16];
#endif

void load_q8_0_block(
        ByteAddressBuffer buf,
        uint block_off,
        out float d,
        out uint4 q0,
        out uint4 q1) {
    uint data_off = block_off + 2u;
    uint aligned = data_off & ~3u;
    uint shift = (data_off & 3u) * 8u;
    uint4 lo = buf.Load4(aligned);
    uint4 hi = buf.Load4(aligned + 16u);
    uint tail = buf.Load(aligned + (shift == 0u ? 28u : 32u));

    if (shift == 0u) {
        q0 = lo;
        q1 = hi;
        uint scale_word = buf.Load(block_off & ~3u);
        d = f16_to_f32((scale_word >> 16u) & 0xffffu);
    } else {
        uint rshift = 32u - shift;
        q0 = uint4(
            (lo.x >> shift) | (lo.y << rshift),
            (lo.y >> shift) | (lo.z << rshift),
            (lo.z >> shift) | (lo.w << rshift),
            (lo.w >> shift) | (hi.x << rshift));
        q1 = uint4(
            (hi.x >> shift) | (hi.y << rshift),
            (hi.y >> shift) | (hi.z << rshift),
            (hi.z >> shift) | (hi.w << rshift),
            (hi.w >> shift) | (tail << rshift));
        d = f16_to_f32(lo.x & 0xffffu);
    }
}

#if defined(WAVE_SIZE) && (GROUP_SIZE >= WAVE_SIZE)
[WaveSize(WAVE_SIZE)]
#endif
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint row0 = group_x_2d(gid);
    if (row0 >= ne0) {
        return;
    }

    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;
    uint num_blocks = ne00 / QK8_0;

    uint gate_base = src0_offset + i2_src0 * nb02 + i3_src0 * nb03;
    uint up_base = op1 + i2_src0 * nb02 + i3_src0 * nb03;
    uint gate_row0 = gate_base + row0 * nb01;
    uint up_row0 = up_base + row0 * nb01;

    uint i2_q8 = i2 * ne12 / ne2;
    uint i3_q8 = i3 * ne13 / ne3;
    uint q8_base = src1_offset + (i3_q8 * ne12 + i2_q8) * num_blocks * Q8_1_BSIZE;

    precise float gate0 = 0.0f;
    precise float up0 = 0.0f;

    for (uint block = tid; block < num_blocks; block += GROUP_SIZE) {
        uint q8_off = q8_base + block * Q8_1_BSIZE;
        float activation_d = f16_to_f32(src1.Load(q8_off) & 0xffffu);
        uint4 activation0 = src1.Load4(q8_off + 4u);
        uint4 activation1 = src1.Load4(q8_off + 20u);

        float gate_d;
        uint4 gate_q0;
        uint4 gate_q1;
        load_q8_0_block(src0, gate_row0 + block * Q8_0_BSIZE,
                        gate_d, gate_q0, gate_q1);

        int dot_gate = 0;
        dot_gate = dot4add_i8packed(gate_q0.x, activation0.x, dot_gate);
        dot_gate = dot4add_i8packed(gate_q0.y, activation0.y, dot_gate);
        dot_gate = dot4add_i8packed(gate_q0.z, activation0.z, dot_gate);
        dot_gate = dot4add_i8packed(gate_q0.w, activation0.w, dot_gate);
        dot_gate = dot4add_i8packed(gate_q1.x, activation1.x, dot_gate);
        dot_gate = dot4add_i8packed(gate_q1.y, activation1.y, dot_gate);
        dot_gate = dot4add_i8packed(gate_q1.z, activation1.z, dot_gate);
        dot_gate = dot4add_i8packed(gate_q1.w, activation1.w, dot_gate);
        gate0 += (gate_d * activation_d) * float(dot_gate);

        float up_d;
        uint4 up_q0;
        uint4 up_q1;
        load_q8_0_block(src2, up_row0 + block * Q8_0_BSIZE,
                        up_d, up_q0, up_q1);

        int dot_up = 0;
        dot_up = dot4add_i8packed(up_q0.x, activation0.x, dot_up);
        dot_up = dot4add_i8packed(up_q0.y, activation0.y, dot_up);
        dot_up = dot4add_i8packed(up_q0.z, activation0.z, dot_up);
        dot_up = dot4add_i8packed(up_q0.w, activation0.w, dot_up);
        dot_up = dot4add_i8packed(up_q1.x, activation1.x, dot_up);
        dot_up = dot4add_i8packed(up_q1.y, activation1.y, dot_up);
        dot_up = dot4add_i8packed(up_q1.z, activation1.z, dot_up);
        dot_up = dot4add_i8packed(up_q1.w, activation1.w, dot_up);
        up0 += (up_d * activation_d) * float(dot_up);
    }

#if defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE)
    gate0 = WaveActiveSum(gate0);
    up0 = WaveActiveSum(up0);

    if (tid == 0u) {
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);
    }
#else
    float wave_gate = WaveActiveSum(gate0);
    float wave_up = WaveActiveSum(up0);
    uint wave_id = tid / WARP_SIZE;
    uint num_waves = GROUP_SIZE / WARP_SIZE;
    if (WaveIsFirstLane()) {
        glu_wave_gate[wave_id] = wave_gate;
        glu_wave_up[wave_id] = wave_up;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u) {
        float g = 0.0f;
        float u = 0.0f;
        for (uint w = 0u; w < num_waves; ++w) {
            g += glu_wave_gate[w];
            u += glu_wave_up[w];
        }
        float result0 = (g / (1.0f + exp(-g))) * u;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);
    }
#endif
}
