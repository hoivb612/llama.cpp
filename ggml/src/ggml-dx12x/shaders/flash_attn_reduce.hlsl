// flash_attn_reduce.hlsl - Merge Split-KV partial flash attention results
//
// Each split produced partial: O (output), M (max score), L (sum of exp)
// This shader merges them using the log-sum-exp trick for numerical stability.
//
// Temp buffer layout (in src0):
//   Offset 0: O matrices [D × ne1 × split_k × ne2 × ne3] as F32
//   After O:  M values   [ne1 × split_k × ne2 × ne3] as F32
//   After M:  L values   [ne1 × split_k × ne2 × ne3] as F32
//
// op_params:
//   op0: D (head dimension)
//   op1: ne1 (number of query rows)
//   op2: split_k (number of splits)
//   op3: ne2 (n_heads)
//   op4: ne3 (batch)
//   op5: dst_esize (2=F16, 4=F32)

#include "ggml_common.hlsli"

#define REDUCE_GROUP_SIZE 256

[numthreads(REDUCE_GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint local_id = gtid.x;
    uint query_idx = gid.x;   // which query row
    uint head_idx  = gid.y;   // which head
    uint batch_idx = gid.z;   // which batch element

    uint D       = op0;
    uint ne1_val = op1;
    uint split_k = op2;
    uint ne2_val = op3;
    uint ne3_val = op4;
    uint out_es  = op5;

    // Compute offsets into temp buffer
    uint o_stride_per_split = D * ne1_val;                     // floats per split's O block
    uint total_o_floats     = o_stride_per_split * split_k * ne2_val * ne3_val;
    uint ml_stride          = ne1_val;                         // M/L values per split per head/batch
    uint hb_index           = head_idx + ne2_val * batch_idx;  // combined head+batch index

    // Base byte offsets in src0 (temp buffer)
    uint o_base = (hb_index * split_k * o_stride_per_split) * 4;  // bytes
    uint m_base = (total_o_floats + hb_index * split_k * ml_stride) * 4;
    uint l_base = (total_o_floats + (ne2_val * ne3_val + hb_index) * split_k * ml_stride) * 4;

    // Step 1: Find global max across all splits for this query row
    float m_max = -3.402823466e+38f;
    for (uint k = 0; k < split_k; k++) {
        uint m_off = m_base + (k * ml_stride + query_idx) * 4;
        float m_k = asfloat(src0.Load(m_off));
        m_max = max(m_max, m_k);
    }

    // Step 2: Compute normalized L sum
    float L_total = 0.0f;
    for (uint k = 0; k < split_k; k++) {
        uint m_off = m_base + (k * ml_stride + query_idx) * 4;
        uint l_off = l_base + (k * ml_stride + query_idx) * 4;
        float m_k = asfloat(src0.Load(m_off));
        float l_k = asfloat(src0.Load(l_off));
        L_total += exp(m_k - m_max) * l_k;
    }

    float inv_L = (L_total > 0.0f) ? (1.0f / L_total) : 0.0f;

    // Step 3: Merge output vectors — each thread handles multiple D dimensions
    for (uint d = local_id; d < D; d += REDUCE_GROUP_SIZE) {
        float O_merged = 0.0f;
        for (uint k = 0; k < split_k; k++) {
            uint o_off = o_base + (k * o_stride_per_split + query_idx * D + d) * 4;
            uint m_off = m_base + (k * ml_stride + query_idx) * 4;
            float o_k = asfloat(src0.Load(o_off));
            float m_k = asfloat(src0.Load(m_off));
            O_merged += exp(m_k - m_max) * o_k;
        }
        O_merged *= inv_L;

        // Write to final output
        uint out_off = dst_offset + d * nb0 + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
        store_auto(dst, out_off, O_merged, out_es);
    }
}
