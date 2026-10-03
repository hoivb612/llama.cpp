// One output row per wave64 with vectorized F16 weight and F32 activation loads.

#include "ggml_common.hlsli"
#include "rope_yarn.hlsli"

#define GROUP_SIZE 64

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
groupshared float wave_sums[2][GROUP_SIZE / WAVE_SIZE];
#endif

void accumulate4(uint weight_offset, uint activation_offset, inout float acc) {
#if NATIVE_FP16
    vector<float16_t, 4> w = src0.Load<vector<float16_t, 4> >(weight_offset);
    float4 x = asfloat(src1.Load4(activation_offset));
    acc = mad((float)w.x, x.x,
          mad((float)w.y, x.y,
          mad((float)w.z, x.z,
          mad((float)w.w, x.w, acc))));
#else
    uint2 packed_w = src0.Load2(weight_offset);
    float4 x = asfloat(src1.Load4(activation_offset));
    float w0 = f16_to_f32(packed_w.x & 0xffffu);
    float w1 = f16_to_f32(packed_w.x >> 16);
    float w2 = f16_to_f32(packed_w.y & 0xffffu);
    float w3 = f16_to_f32(packed_w.y >> 16);
    acc = mad(w0, x.x, mad(w1, x.y, mad(w2, x.z, mad(w3, x.w, acc))));
#endif
}

float load_f16_scalar(uint byte_offset) {
    uint word = src0.Load(byte_offset & ~3u);
    return f16_to_f32((word >> ((byte_offset & 2u) * 8u)) & 0xffffu);
}

WAVE_SIZE_ATTR
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint row = group_x_2d(group_id);
    // rope rows2: each group owns a full rotation pair (row, row+1); the host
    // halved the dispatch, so map the group index to the even pair base.
    if (mmv_rope_rows2()) {
        row *= 2u;
    }
    if (row >= ne0) {
        return;
    }

    uint i2 = group_id.z % ne2;
    uint i3 = group_id.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;

    uint src0_row = src0_offset + row * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint src1_row = src1_offset + i2 * nb12 + i3 * nb13;

    // ROPE fusion: to rotate, this group also needs the dot of its NORMAL-mode
    // pair partner (row ^ 1); compute both dots, then write only `row`.
    bool rope = mmv_rope_active();
    uint row_p = row ^ 1u;
    uint src0_row_p = src0_offset + row_p * nb01 + i2_src0 * nb02 + i3_src0 * nb03;

    float acc = 0.0f;
    float acc_p = 0.0f;
    uint k = local_id * 4u;
    const uint stride = GROUP_SIZE * 4u;

    for (; k + 3u < ne00; k += stride) {
        accumulate4(src0_row + k * 2u, src1_row + k * 4u, acc);
        if (rope) {
            accumulate4(src0_row_p + k * 2u, src1_row + k * 4u, acc_p);
        }
    }
    for (; k < ne00; ++k) {
        float x = asfloat(src1.Load(src1_row + k * 4u));
        acc = mad(load_f16_scalar(src0_row + k * 2u), x, acc);
        if (rope) {
            acc_p = mad(load_f16_scalar(src0_row_p + k * 2u), x, acc_p);
        }
    }

    float sum = WaveActiveSum(acc);
    float sum_p = 0.0f;
    if (rope) {
        sum_p = WaveActiveSum(acc_p);
    }

#if defined(WAVE_SIZE) && WAVE_SIZE < GROUP_SIZE
    if (WaveIsFirstLane()) {
        wave_sums[0][local_id / WAVE_SIZE] = sum;
        wave_sums[1][local_id / WAVE_SIZE] = sum_p;
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_id == 0u) {
        sum = 0.0f;
        sum_p = 0.0f;
        [unroll]
        for (uint wave = 0; wave < GROUP_SIZE / WAVE_SIZE; ++wave) {
            sum   += wave_sums[0][wave];
            sum_p += wave_sums[1][wave];
        }
    }
#endif

    if (local_id == 0u) {
        if (rope) {
            uint row0 = row & ~1u;
            float sum0 = (row == row0) ? sum   : sum_p;
            float sum1 = (row == row0) ? sum_p : sum;
            uint pair_in_head = (row0 % op10) / 2u;
            float out0, out1;
            mmv_rope_pair(pair_in_head, sum0, sum1, out0, out1);
            if (mmv_rope_rows2()) {
                // row is the even pair base; write both rotated outputs so the
                // partner row is not recomputed by a second group.
                mmv_rope_store(row0, out0);
                if (row0 + 1u < ne0) {
                    mmv_rope_store(row0 + 1u, out1);
                }
            } else {
                mmv_rope_store(row, (row == row0) ? out0 : out1);
            }
        } else {
            sum += load_fused_bias(row, i2, i3);
            if (mmv_scatter_active()) {
                mmv_store_scatter(row, 0u, sum);
            } else {
                uint dst_row = offset_4d(row, 0, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                store_auto(dst, dst_row, sum, dst_esize);
            }
        }
    }
}
