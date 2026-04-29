// mul_mat_vec_q6k.hlsl - Matrix-vector multiply for Q6_K weights (M=1)
//
// 256 threads cooperate per output row. Each thread processes K/256 elements
// using strided access, then wave + shared memory reduction.
//
// Q6_K block layout (210 bytes): ql[128] + qh[64] + scales[16] + d(f16)
//
// Dispatch: groups_x = N, groups_y = 1, groups_z = batch*ne2*ne3

#include "ggml_common.hlsli"

#define GROUP_SIZE  256
#define QK_K        256
#define Q6K_BSIZE   210

groupshared float shared_acc[GROUP_SIZE];

uint read_byte_q6(ByteAddressBuffer buf, uint byte_off) {
    uint word = buf.Load(byte_off & ~3u);
    return (word >> ((byte_off & 3u) * 8u)) & 0xFFu;
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i0 = group_id.x;
    uint flat_batch = group_id.z;
    uint i2 = flat_batch % ne2;
    uint i3 = flat_batch / ne2;

    if (i0 >= ne0) return;

    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint K = ne00;
    uint src0_row = src0_offset + i0 * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_base = src1_offset + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;

    for (uint k = tid; k < K; k += GROUP_SIZE) {
        uint block_idx = k / QK_K;
        uint elem = k % QK_K;
        uint block_off = src0_row + block_idx * Q6K_BSIZE;

        // Load d (f16) - alignment-safe
        uint d_off = block_off + 208;
        uint d_word = src0.Load(d_off & ~3u);
        float d = f16_to_f32((d_word >> ((d_off & 2u) * 8u)) & 0xFFFFu);

        uint ip = elem / 128;
        uint il = elem % 128;

        // Scale (int8)
        uint sc_byte = read_byte_q6(src0, block_off + 192 + 8 * ip + il / 16);
        int scale = (sc_byte < 128) ? (int)sc_byte : (int)sc_byte - 256;

        // ql + qh -> 6-bit quantized value
        uint ql_val = read_byte_q6(src0, block_off + 64 * ip + (il % 64));
        uint qh_val = read_byte_q6(src0, block_off + 128 + 32 * ip + (il % 32));

        uint sub32 = il / 32;
        uint ql_bits = (il < 64) ? (ql_val & 0x0Fu) : (ql_val >> 4);
        int q = (int)(ql_bits | (((qh_val >> (sub32 * 2)) & 3u) << 4)) - 32;

        float w = d * (float)scale * (float)q;
        float x = asfloat(src1.Load(src1_base + k * 4));
        acc += w * x;
    }

    // Wave-intrinsic reduction then cross-wave via shared memory
    float wave_sum = WaveActiveSum(acc);
    uint wave_id = tid / WaveGetLaneCount();
    if (WaveIsFirstLane()) shared_acc[wave_id] = wave_sum;
    GroupMemoryBarrierWithGroupSync();

    uint num_waves = GROUP_SIZE / WaveGetLaneCount();
    if (tid < num_waves) {
        float v = shared_acc[tid];
        v = WaveActiveSum(v);
        if (tid == 0) shared_acc[0] = v;
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0) {
        float result = shared_acc[0];
        if (op0 == 1u) result += asfloat(src2.Load(op1 + i0 * 4));
        uint off_d = offset_4d(i0, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_auto(dst, off_d, result, dst_esize);
    }
}