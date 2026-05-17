// mul_mat_id.hlsl - Indirect matrix multiplication for MoE (Mixture of Experts)
// src0: expert weight bank [K, N, n_experts] (quantized or F16/F32)
// src1: input activations  [K, n_tokens, 1, 1] (F32)
// src2: expert routing ids [n_used_experts, n_tokens] (I32)
// dst:  output [N, n_used_experts, n_tokens, 1]
//
// For each token (i2) and expert slot (i1):
//   expert_id = src2[i1, i2]
//   dst[i0, i1, i2] = sum_k(src0[k, i0, expert_id] * src1[k, i2])

#include "ggml_common.hlsli"

// src1 broadcasts along i1 when ne11 == 1, otherwise it has one slice per
// selected expert slot.
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint local_id : SV_GroupIndex) {
    uint idx = flat_idx_2d(group_id, local_id);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    // dst layout: [N, n_used_experts, n_tokens, 1]
    uint i0 = idx % ne0; uint rem = idx / ne0;
    uint i1 = rem % ne1; rem = rem / ne1;
    uint i2 = rem % ne2; uint i3 = rem / ne2;

    // Look up expert index from src2 (ids)
    // ids layout: [n_used_experts, n_tokens], type I32
    // op_params[0] contains src2 offset (passed via params)
    uint ids_off = op0 + i1 * op1 + i2 * op2;  // op0=src2_offset, op1=src2_nb0, op2=src2_nb1
    int expert_id = asint(src2.Load(ids_off));

    uint K = ne00;

    // Weight row for this expert: src0[*, i0, expert_id]
    uint src0_row = src0_offset + i0 * nb01 + (uint)expert_id * nb02;

    // Input row: src1[*, i2]
    uint i1_src1 = i1 * ne11 / ne1;
    uint src1_row = src1_offset + i1_src1 * nb11 + i2 * nb12 + i3 * nb13;

    precise float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        float w = load_auto(src0, src0_row + k * nb00, src0_esize);
        float x = load_auto(src1, src1_row + k * nb10, src1_esize);
        acc += w * x;
    }

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    store_auto(dst, off_d, acc, dst_esize);
}
