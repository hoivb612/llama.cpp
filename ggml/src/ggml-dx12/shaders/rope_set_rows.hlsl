// rope_set_rows.hlsl - Fused ROPE + VIEW + SET_ROWS
// Applies rotary position embedding and writes directly to KV cache
// Eliminates 2 dispatches (VIEW + SET_ROWS) per KV write
//
// src0: input tensor (Q or K projection output) — F32
// src1: position indices for ROPE (int32)
// src2: (unused, or freq_factors if present)
// src3: row indices for SET_ROWS (int32)
// dst:  KV cache tensor (F32 or F16)
//
// op_params layout:
//   [0]=n_past(0), [1]=n_dims, [2]=mode, [3]=n_ctx(0), [4]=n_ctx_orig
//   [5]=freq_base(float), [6]=freq_scale(float), [7]=ext_factor(float)
//   [8]=set_rows_stride (elements per row in KV cache)
//   [9]=set_rows_nb1 (byte stride between rows in KV cache dst)

#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint n_pairs = ne00 / 2;
    uint total_pairs = n_pairs * ne01 * ne02 * ne03;
    if (idx >= total_pairs) return;

    uint i3 = idx / (n_pairs * ne01 * ne02);
    uint rem = idx % (n_pairs * ne01 * ne02);
    uint i2 = rem / (n_pairs * ne01);
    rem = rem % (n_pairs * ne01);
    uint i1 = rem / n_pairs;
    uint pair = rem % n_pairs;

    uint n_dims = op_param_uint(1);
    uint mode   = op_param_uint(2);
    float freq_base  = op_param_f32(5);
    float freq_scale = op_param_f32(6);
    uint set_rows_stride = op_param_uint(8);  // elements per KV cache row
    uint set_rows_nb1    = op_param_uint(9);   // byte stride between KV rows

    bool is_neox = (mode & 2u) != 0;
    uint half_dims = n_dims / 2;

    // Compute ROPE element indices
    uint idx_a, idx_b;
    if (pair >= half_dims) {
        // Passthrough: elements beyond n_dims are copied
        uint pass_idx = n_dims + 2 * (pair - half_dims);
        uint pass_a = pass_idx;
        uint pass_b = pass_idx + 1;

        // Read from src0
        float va = 0.0f, vb = 0.0f;
        if (pass_a < ne00) {
            uint off_a = offset_4d(pass_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            va = asfloat(src0.Load(off_a));
        }
        if (pass_b < ne00) {
            uint off_b = offset_4d(pass_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
            vb = asfloat(src0.Load(off_b));
        }

        // Write directly to KV cache via SET_ROWS indexing
        int row_idx = asint(src3.Load(src1_offset + i2 * 4));  // row index from src3 (I32)
        // Flatten: view merges ne[0]*ne[1], then SET_ROWS scatters by row
        uint flat_a = i1 * ne00 + pass_a;
        uint flat_b = i1 * ne00 + pass_b;
        uint dst_off_a = dst_offset + flat_a * nb0 + (uint)row_idx * set_rows_nb1 + i3 * nb3;
        uint dst_off_b = dst_offset + flat_b * nb0 + (uint)row_idx * set_rows_nb1 + i3 * nb3;

        if (pass_a < ne00) store_auto(dst, dst_off_a, va, dst_esize);
        if (pass_b < ne00) store_auto(dst, dst_off_b, vb, dst_esize);
        return;
    }

    if (is_neox) {
        idx_a = pair;
        idx_b = pair + half_dims;
    } else {
        idx_a = pair * 2;
        idx_b = pair * 2 + 1;
    }

    // Position from src1 (int32)
    uint pos_off = src1_offset + i2 * nb10;
    int pos = asint(src1.Load(pos_off));

    float theta = (float)pos * exp2(-(float)(pair * 2) / (float)n_dims * log2(freq_base));
    theta *= freq_scale;

    float cos_theta, sin_theta;
    sincos(theta, sin_theta, cos_theta);

    // Read from src0
    uint off_a = offset_4d(idx_a, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    uint off_b = offset_4d(idx_b, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
    float x0 = asfloat(src0.Load(off_a));
    float x1 = asfloat(src0.Load(off_b));

    float rot_a = x0 * cos_theta - x1 * sin_theta;
    float rot_b = x0 * sin_theta + x1 * cos_theta;

    // Write directly to KV cache via SET_ROWS indexing
    // src3 contains the row indices (I32), one per i2
    int row_idx = asint(src3.Load(src1_offset + i2 * 4));  // reuse src1 layout for indices
    // Flatten: view merges ne[0]*ne[1] into one dim, then SET_ROWS scatters
    uint flat_a = i1 * ne00 + idx_a;
    uint flat_b = i1 * ne00 + idx_b;
    uint dst_off_a = dst_offset + flat_a * nb0 + (uint)row_idx * set_rows_nb1 + i3 * nb3;
    uint dst_off_b = dst_offset + flat_b * nb0 + (uint)row_idx * set_rows_nb1 + i3 * nb3;

    store_auto(dst, dst_off_a, rot_a, dst_esize);
    store_auto(dst, dst_off_b, rot_b, dst_esize);
}
