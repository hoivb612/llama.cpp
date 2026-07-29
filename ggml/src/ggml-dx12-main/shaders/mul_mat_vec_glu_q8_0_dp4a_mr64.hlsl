// Fused Q8_0 gate/up matvec and SwiGLU using Q8_1 activations.
// One AMD wave64 computes one output row and reuses each activation dot
// across the gate and up weight matrices.

#include "ggml_common.hlsli"

#ifndef GROUP_SIZE
#define GROUP_SIZE       64
#endif
#define QK8_0            32
#define Q8_0_BSIZE       34
#define Q8_1_BSIZE       36
// One block group == GROUP_SIZE/8 threads (8 lanes per block), so each
// iteration must advance by exactly that many blocks to avoid re-processing.
#define BLOCKS_PER_ITER  (GROUP_SIZE / 8)

// Cross-wave reduction scratch (multi-wave devices only). Sized for the
// smallest compiled wave (16) => GROUP_SIZE/16 partials per accumulator.
// Unused on the single-wave (GROUP_SIZE == WAVE_SIZE) fast path.
#if !(defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE))
groupshared float glu_wave_gate[GROUP_SIZE / 16];
groupshared float glu_wave_up[GROUP_SIZE / 16];
#endif

uint read_u32_src0(uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = src0.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = src0.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

uint read_u32_src2(uint byte_off) {
    uint aligned = byte_off & ~3u;
    uint shift = (byte_off & 3u) * 8u;
    uint lo = src2.Load(aligned);
    if (shift == 0u) {
        return lo;
    }
    uint hi = src2.Load(aligned + 4u);
    return (lo >> shift) | (hi << (32u - shift));
}

float read_f16_src0(uint byte_off) {
    uint word = src0.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xffffu);
}

float read_f16_src2(uint byte_off) {
    uint word = src2.Load(byte_off & ~3u);
    return f16_to_f32((word >> ((byte_off & 2u) * 8u)) & 0xffffu);
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

    uint sub = tid / 8;
    uint lane = tid & 7u;
    uint q_offset = lane * 4;

    precise float gate0 = 0.0f;
    precise float up0 = 0.0f;

    for (uint block_iter = 0; block_iter < num_blocks; block_iter += BLOCKS_PER_ITER) {
        uint block = block_iter + sub;
        if (block < num_blocks) {
            uint q8_off = q8_base + block * Q8_1_BSIZE;
            uint ds = src1.Load(q8_off);
            float activation_d = f16_to_f32(ds & 0xffffu);
            uint activation = src1.Load(q8_off + 4 + q_offset);

            uint gate0_off = gate_row0 + block * Q8_0_BSIZE;
            uint up0_off = up_row0 + block * Q8_0_BSIZE;

            uint weight_gate0 = read_u32_src0(gate0_off + 2 + q_offset);
            uint weight_up0 = read_u32_src2(up0_off + 2 + q_offset);
            int dot_gate0 = 0;
            dot_gate0 = dot4add_i8packed(weight_gate0, activation, dot_gate0);
            int dot_up0 = 0;
            dot_up0 = dot4add_i8packed(weight_up0, activation, dot_up0);

            float scale_gate0 = read_f16_src0(gate0_off) * activation_d;
            float scale_up0 = read_f16_src2(up0_off) * activation_d;
            gate0 += scale_gate0 * float(dot_gate0);
            up0 += scale_up0 * float(dot_up0);
        }
    }

#if defined(WAVE_SIZE) && (GROUP_SIZE == WAVE_SIZE)
    gate0 = WaveActiveSum(gate0);
    up0 = WaveActiveSum(up0);

    if (tid == 0) {
        float result0 = (gate0 / (1.0f + exp(-gate0))) * up0;
        uint off_d0 = offset_4d(row0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d0, result0, dst_esize);
    }
#else
    float wave_gate = WaveActiveSum(gate0);
    float wave_up   = WaveActiveSum(up0);
    uint  wave_id   = tid / WARP_SIZE;
    uint  num_waves = GROUP_SIZE / WARP_SIZE;
    if (WaveIsFirstLane()) {
        glu_wave_gate[wave_id] = wave_gate;
        glu_wave_up[wave_id]   = wave_up;
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0) {
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
