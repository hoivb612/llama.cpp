// flash_attn_reduce.hlsl - Reduce split-KV partial results
//
// Combines partial (max, sum, O[D]) from multiple splits using online softmax.
// Each thread group handles one (query, head, batch) output.
//
// Dispatch: groups_x = N_queries, groups_y = n_heads, groups_z = batch
// op15 = n_splits
//
// Reads from temp buffer (u1), writes final output to dst (u0).

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define D_MAX 256

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint local_id = gtid.x;
    uint query_idx = gid.x;
    uint head_idx  = gid.y;
    uint batch_idx = gid.z;

    if (query_idx >= ne01) return;

    uint D = ne00;
    uint n_splits = op15 & 0xFFFFu;  // GQA-folded FA packs gqa_ratio in high 16 bits
    uint n_heads = ne02;

    // Partial layout: [batch][head][query][split] × (max + sum + D floats)
    uint partial_stride = (D + 2) * 4;  // bytes per partial
    uint base_off = ((batch_idx * n_heads + head_idx) * (uint)ne01 + query_idx) * n_splits;

    // Online softmax reduction across splits
    float global_max = -3.402823466e+38f;
    float global_sum = 0.0f;
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint s = 0; s < n_splits; s++) {
        uint p_off = (base_off + s) * partial_stride;
        float p_max = asfloat(temp.Load(p_off));
        float p_sum = asfloat(temp.Load(p_off + 4));

        if (p_sum == 0.0f) continue;  // empty split

        float new_max = max(global_max, p_max);
        float old_correction = (global_sum > 0.0f) ? exp(global_max - new_max) : 0.0f;
        float new_correction = exp(p_max - new_max);

        // Correct existing accumulators and add new partial
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D) {
                float p_o = asfloat(temp.Load(p_off + 8 + d_out * 4));
                acc[ai] = acc[ai] * old_correction + p_o * new_correction;
            }
        }
        global_sum = global_sum * old_correction + p_sum * new_correction;
        global_max = new_max;
    }

    // Write final normalized output
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint ai = 0; ai < 4; ai++) {
        uint d_out = local_id + ai * GROUP_SIZE;
        if (d_out < D) {
            uint out_off = dst_offset + d_out * nb0 + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
            store_auto(dst, out_off, acc[ai] * inv_sum, dst_esize);
        }
    }
}
