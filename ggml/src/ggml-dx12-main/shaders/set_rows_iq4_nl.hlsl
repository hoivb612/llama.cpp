// set_rows_iq4_nl.hlsl - SET_ROWS with F32 src0 -> IQ4_NL dst.
//
// Vulkan parity: copy_to_quant.comp DATA_A_IQ4_NL quantize.
// One thread per IQ4_NL block (32 F32 elements). 32 threads per workgroup.
// Block layout: { uint16_t d; uint8_t qs[16]; } = 18 bytes.
//
// IQ4_NL uses a non-uniform 16-entry LUT (kvalues_iq4nl).  Each F32 input is
// mapped to the closest LUT index via a 4-step binary search (best_index).
// After the first pass, d is refined from the sumqx/sumq2 weighted ratio.
#include "ggml_common.hlsli"

#define QK_IQ4_NL 32

static const float kvalues_iq4nl[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f
};

uint best_index(float x) {
    if (x <= kvalues_iq4nl[0])  return 0;
    if (x >= kvalues_iq4nl[15]) return 15;
    int ml = 0, mu = 15;
    [unroll] for (int s = 0; s < 4; ++s) {
        if (mu - ml <= 1) break;
        int mav = (ml + mu) / 2;
        if (x < kvalues_iq4nl[mav]) mu = mav; else ml = mav;
    }
    return (x - kvalues_iq4nl[mu - 1]) < (kvalues_iq4nl[mu] - x) ? (uint)(mu - 1) : (uint)mu;
}

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_id : SV_GroupThreadID) {
    uint global_thread = (gid.z * 262144u + gid.y * 512u + gid.x) * 32u + local_id;
    uint block_idx = global_thread;

    uint nb_per_row   = ne00 / QK_IQ4_NL;
    uint total_blocks = nb_per_row * ne01 * ne02 * ne03;
    if (block_idx >= total_blocks) return;

    uint bi0 = block_idx % nb_per_row;
    uint rem = block_idx / nb_per_row;
    uint i1  = rem % ne01;
    rem      = rem / ne01;
    uint i2  = rem % ne02;
    uint i3  = rem / ne02;

    uint i2_idx  = ne11 > 0 ? (i2 % ne11) : 0;
    uint i3_idx  = ne12 > 0 ? (i3 % ne12) : 0;
    uint idx_off = src1_offset + i1 * nb10 + i2_idx * nb11 + i3_idx * nb12;
    int  row_idx = asint(src1.Load(idx_off));

    uint base_i0 = bi0 * QK_IQ4_NL;
    uint off0    = src0_offset + base_i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;

    float amax = 0.0f;
    float vmax = 0.0f;
    [unroll] for (uint j = 0; j < QK_IQ4_NL; ++j) {
        float v = asfloat(src0.Load(off0 + j * nb00));
        if (abs(v) > amax) { amax = abs(v); vmax = v; }
    }

    float d  = vmax / kvalues_iq4nl[0];   // kvalues_iq4nl[0] == -127
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    uint dst_block_off = dst_offset + bi0 * 18u + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;

    float sumqx = 0.0f, sumq2 = 0.0f;
    [unroll] for (uint k = 0; k < 8; ++k) {
        uint  b0  = 2u * k;
        uint  b1  = b0 + 1u;
        float v0_lo = asfloat(src0.Load(off0 + (b0)                  * nb00));
        float v0_hi = asfloat(src0.Load(off0 + (b0 + (QK_IQ4_NL/2u)) * nb00));
        float v1_lo = asfloat(src0.Load(off0 + (b1)                  * nb00));
        float v1_hi = asfloat(src0.Load(off0 + (b1 + (QK_IQ4_NL/2u)) * nb00));
        uint  xi0_lo = best_index(v0_lo * id);
        uint  xi0_hi = best_index(v0_hi * id);
        uint  xi1_lo = best_index(v1_lo * id);
        uint  xi1_hi = best_index(v1_hi * id);
        uint  byte0  = (xi0_lo & 0xFu) | ((xi0_hi & 0xFu) << 4);
        uint  byte1  = (xi1_lo & 0xFu) | ((xi1_hi & 0xFu) << 4);
        uint16_t packed = (uint16_t)(byte0 | (byte1 << 8));
        dst.Store<uint16_t>(dst_block_off + 2u + k * 2u, packed);

        float lut0_lo = kvalues_iq4nl[xi0_lo];
        float lut0_hi = kvalues_iq4nl[xi0_hi];
        float lut1_lo = kvalues_iq4nl[xi1_lo];
        float lut1_hi = kvalues_iq4nl[xi1_hi];
        float w0_lo = v0_lo * v0_lo;
        float w0_hi = v0_hi * v0_hi;
        float w1_lo = v1_lo * v1_lo;
        float w1_hi = v1_hi * v1_hi;
        sumqx += w0_lo * lut0_lo * v0_lo + w0_hi * lut0_hi * v0_hi
              +  w1_lo * lut1_lo * v1_lo + w1_hi * lut1_hi * v1_hi;
        sumq2 += w0_lo * lut0_lo * lut0_lo + w0_hi * lut0_hi * lut0_hi
              +  w1_lo * lut1_lo * lut1_lo + w1_hi * lut1_hi * lut1_hi;
    }
    float d_refined = (sumq2 > 0.0f) ? (sumqx / sumq2) : d;
    dst.Store<uint16_t>(dst_block_off, asuint16((float16_t)d_refined));
}
