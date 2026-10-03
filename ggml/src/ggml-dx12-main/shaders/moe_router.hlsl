// MoE router: SOFT_MAX -> ARGSORT top-k -> GET_ROWS weights [-> SUM_ROWS -> CLAMP -> DIV].
// One wave per row. Writes the softmax row, the top-k ids and the (normalized) weights.
// op0 ids offset, op1 ids row stride, op2 k, op3 weights offset,
// op4 weights rank stride, op5 weights row stride, op6 norm, op7/op8 clamp min/max
#include "ggml_common.hlsli"

#define MAX_PER_LANE 16
#define NEG_INF -3.402823466e+38f

WAVE_SIZE_ATTR
[numthreads(WARP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint row = gid.x;
    const uint n = ne00;
    const uint src_row = src0_offset + row * nb01;

    float v[MAX_PER_LANE];
    float row_max = NEG_INF;
    [unroll] for (uint j = 0; j < MAX_PER_LANE; ++j) {
        const uint e = lane + j * WARP_SIZE;
        v[j] = e < n ? asfloat(src0.Load(src_row + e * 4)) : NEG_INF;
        row_max = max(row_max, v[j]);
    }
    row_max = WaveActiveMax(row_max);

    precise float local_sum = 0.0f;
    [unroll] for (uint j = 0; j < MAX_PER_LANE; ++j) {
        const uint e = lane + j * WARP_SIZE;
        v[j] = e < n ? exp(v[j] - row_max) : 0.0f;
        local_sum += v[j];
    }
    const float inv_sum = 1.0f / WaveActiveSum(local_sum);

    const uint dst_row = dst_offset + row * nb1;
    [unroll] for (uint j = 0; j < MAX_PER_LANE; ++j) {
        const uint e = lane + j * WARP_SIZE;
        if (e < n) {
            v[j] *= inv_sum;
            dst.Store(dst_row + e * 4, asuint(v[j]));
        } else {
            v[j] = NEG_INF;
        }
    }

    // Iterative argmax; ties go to the lowest expert id.
    const uint k = op2;
    float my_w = 0.0f;
    uint my_id = 0;
    float w_sum = 0.0f;
    for (uint r = 0; r < k; ++r) {
        float best = NEG_INF;
        uint best_id = 0xFFFFFFFFu;
        [unroll] for (uint j = 0; j < MAX_PER_LANE; ++j) {
            if (v[j] > best) {
                best = v[j];
                best_id = lane + j * WARP_SIZE;
            }
        }
        const float w = WaveActiveMax(best);
        uint id = WaveActiveMin(best == w ? best_id : 0xFFFFFFFFu);
        // NaN rows find no winner; keep the id in range for MUL_MAT_ID.
        if (id >= n) {
            id = r;
        }
        [unroll] for (uint j = 0; j < MAX_PER_LANE; ++j) {
            if (lane + j * WARP_SIZE == id) {
                v[j] = NEG_INF;
            }
        }
        if (lane == r) {
            my_w = w;
            my_id = id;
        }
        w_sum += w;
    }

    if (lane < k) {
        if (op6 != 0) {
            my_w /= clamp(w_sum, asfloat(op7), asfloat(op8));
        }
        dst.Store(op0 + row * op1 + lane * 4, my_id);
        dst.Store(op3 + row * op5 + lane * op4, asuint(my_w));
    }
}
