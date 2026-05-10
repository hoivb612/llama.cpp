// flash_attn.hlsl - Flash Attention for DX12
//
// Computes: output = softmax(Q @ K^T * scale + mask) @ V
//
// op_params layout:
//   op0: src2_offset    op1: src2_nb0    op2: src2_nb1    op3: src2_nb2
//   op4: src2_nb3       op5: src2_esize  op6: scale(f32)  op7: n_kv_heads
//   op8: has_mask       op9: mask_offset op10: mask_nb1   op11: mask_nb2
//   op12: mask_nb3      op13: mask_ne2   op14: mask_ne3
//   op15: split_k (0 or 1 = no split, >1 = number of KV splits)

#include "ggml_common.hlsli"

#define TILE_KV 256
#define GROUP_SIZE 256

groupshared float s_scores[TILE_KV];
groupshared float s_reduce[GROUP_SIZE];

float load_f16_mask(uint byte_offset) {
    uint word = src3.Load(byte_offset & ~3u);
    uint shift = (byte_offset & 2u) * 8u;
    return f16_to_f32((word >> shift) & 0xFFFFu);
}

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint local_id = gtid.x;

    uint  split_k = op15;
    uint  split_k_index = 0;
    uint  query_idx;

    if (split_k > 1) {
        // X dimension encodes: query_idx * split_k + split_k_index
        query_idx     = gid.x / split_k;
        split_k_index = gid.x % split_k;
    } else {
        query_idx = gid.x;
    }

    uint head_idx  = gid.y;
    uint batch_idx = gid.z;

    if (query_idx >= ne01) return;

    float scale      = asfloat(op6);
    uint  n_kv_heads = op7;
    uint  src2_off   = op0;
    uint  src2_nb0   = op1;
    uint  src2_nb1   = op2;
    uint  src2_nb2   = op3;
    uint  src2_nb3   = op4;
    uint  src2_es    = op5;

    uint  has_mask    = op8;
    uint  mask_off    = op9;
    uint  mask_nb1    = op10;
    uint  mask_nb2    = op11;
    uint  mask_nb3    = op12;
    uint  mask_ne2    = op13;
    uint  mask_ne3    = op14;

    uint D    = ne00;
    uint N_kv = ne11;
    uint kv_head = head_idx * n_kv_heads / ne02;

    // Split-KV: compute this split's KV range
    uint kv_start = 0;
    uint kv_end   = N_kv;
    if (split_k > 1) {
        uint split_kv = (N_kv + split_k - 1) / split_k;
        // Align to TILE_KV for efficient tiling
        split_kv = ((split_kv + TILE_KV - 1) / TILE_KV) * TILE_KV;
        kv_start = split_k_index * split_kv;
        kv_end   = min(kv_start + split_kv, N_kv);
        if (kv_start >= N_kv) {
            // Empty split: write sentinel values so reduce shader sees valid data
            // (exp(-FLT_MAX - m_max) ≈ 0, so this split contributes nothing)
            uint hb_index = head_idx + ne02 * batch_idx;
            uint o_stride = D * ne01;
            uint total_o  = o_stride * split_k * ne02 * ne03;
            uint ml_stride_val = ne01;

            for (uint ai = 0; ai < 4; ai++) {
                uint d_out = local_id + ai * GROUP_SIZE;
                if (d_out < D) {
                    uint o_off = (hb_index * split_k * o_stride + split_k_index * o_stride + query_idx * D + d_out) * 4;
                    dst.Store(o_off, asuint(0.0f));
                }
            }
            if (local_id == 0) {
                uint m_off = (total_o + hb_index * split_k * ml_stride_val + split_k_index * ml_stride_val + query_idx) * 4;
                uint l_off = (total_o + (ne02 * ne03 + hb_index) * split_k * ml_stride_val + split_k_index * ml_stride_val + query_idx) * 4;
                dst.Store(m_off, asuint(-3.402823466e+38f)); // M = -FLT_MAX
                dst.Store(l_off, asuint(0.0f));              // L = 0
            }
            return;
        }
    }

    uint mask_base = 0;
    if (has_mask) {
        mask_base = mask_off
                  + query_idx * mask_nb1
                  + (head_idx % mask_ne2) * mask_nb2
                  + (batch_idx % mask_ne3) * mask_nb3;
    }

    // Precompute Q base address
    uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;

    float global_max = -3.402823466e+38f;
    float global_sum = 0.0f;

    // Per-thread output accumulators — each thread owns D/GROUP_SIZE dimensions
    // Use array for flexibility with different D values
    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint tile_start = kv_start; tile_start < kv_end; tile_start += TILE_KV) {
        uint tile_end = min(tile_start + TILE_KV, kv_end);
        uint tile_size = tile_end - tile_start;

        // Pass 1: Each thread computes one Q·K dot product
        float my_score = -3.402823466e+38f;
        if (local_id < tile_size) {
            uint kv = tile_start + local_id;

            float mv = 0.0f;
            if (has_mask) {
                mv = load_f16_mask(mask_base + kv * 2);
            }

            if (!isinf(mv)) {
                precise float dot = 0.0f;
                uint k_base = src1_offset + kv * nb11 + kv_head * nb12 + batch_idx * nb13;

                // Vectorized QK dot product with Load2
                if (src0_esize == 4 && nb00 == 4 && nb10 == 4) {
                    // Both Q and K are contiguous F32
                    uint d = 0;
                    for (; d + 1 < D; d += 2) {
                        uint2 qp = src0.Load2(q_base + d * 4);
                        uint2 kp = src1.Load2(k_base + d * 4);
                        dot += asfloat(qp.x) * asfloat(kp.x) + asfloat(qp.y) * asfloat(kp.y);
                    }
                    if (d < D) {
                        dot += asfloat(src0.Load(q_base + d * 4)) * asfloat(src1.Load(k_base + d * 4));
                    }
                } else {
                    for (uint d = 0; d < D; d++) {
                        uint q_off = q_base + d * nb00;
                        uint k_off = k_base + d * nb10;
                        dot += load_auto(src0, q_off, src0_esize) * load_auto(src1, k_off, src1_esize);
                    }
                }
                my_score = dot * scale + mv;
            }
        }
        s_scores[local_id] = my_score;
        GroupMemoryBarrierWithGroupSync();

        // Pass 2a: Find tile max using wave reduction
        float local_max = (local_id < tile_size) ? s_scores[local_id] : -3.402823466e+38f;
        float wave_max = WaveActiveMax(local_max);
        uint wave_id = local_id / WaveGetLaneCount();
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_max;
        GroupMemoryBarrierWithGroupSync();
        uint num_waves = GROUP_SIZE / WaveGetLaneCount();
        if (local_id < num_waves) {
            float v = s_reduce[local_id];
            v = WaveActiveMax(v);
            if (local_id == 0) s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        float tile_max = s_reduce[0];

        // Online softmax correction
        float new_max = max(global_max, tile_max);
        float correction = (global_sum > 0.0f) ? exp(global_max - new_max) : 0.0f;
        [unroll] for (uint a = 0; a < 4; a++) acc[a] *= correction;
        global_sum *= correction;
        global_max = new_max;

        // Exponentiate scores
        if (local_id < tile_size) {
            s_scores[local_id] = exp(s_scores[local_id] - global_max);
        } else {
            s_scores[local_id] = 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();

        // Pass 2b: Sum exponentiated scores using wave reduction
        float local_score = s_scores[local_id];
        float wave_sum = WaveActiveSum(local_score);
        if (WaveIsFirstLane()) s_reduce[wave_id] = wave_sum;
        GroupMemoryBarrierWithGroupSync();
        if (local_id < num_waves) {
            float v = s_reduce[local_id];
            v = WaveActiveSum(v);
            if (local_id == 0) s_reduce[0] = v;
        }
        GroupMemoryBarrierWithGroupSync();
        global_sum += s_reduce[0];

        // Pass 3: Cooperative V accumulation
        // For D <= GROUP_SIZE/2: remap threads to (d_out, kv_worker) pairs.
        // Multiple workers split the KV tile, each processing a chunk of KV
        // tokens. This halves the serial FMA dependency chain per thread.
        // After the loop, reduce partial sums across workers via s_reduce[].
        //
        //   D=96  (Phi-3):  2 workers, 128 KV tokens each (was 256 serial)
        //   D=128 (Qwen):   2 workers, 128 KV tokens each
        //   D=64  (Gemma):  4 workers, 64 KV tokens each
        {
            uint v_head_base = src2_off + kv_head * src2_nb2 + batch_idx * src2_nb3;
            uint num_v_workers = D > 0 ? min(GROUP_SIZE / D, 4u) : 1u;

            if (num_v_workers > 1 && src2_es == 2 && src2_nb0 == 2) {
                // Cooperative F16 fast path
                uint d_out = local_id % D;
                uint v_wk  = local_id / D;

                float partial = 0.0f;
                if (v_wk < num_v_workers) {
                    uint chunk = (tile_size + num_v_workers - 1) / num_v_workers;
                    uint kv_lo = v_wk * chunk;
                    uint kv_hi = min(kv_lo + chunk, tile_size);

                    for (uint t = kv_lo; t < kv_hi; t++) {
                        float w = s_scores[t];
                        uint v_row = v_head_base + (tile_start + t) * src2_nb1;
                        uint v_addr = v_row + d_out * 2;
                        uint raw = src2.Load(v_addr & ~3u);
                        partial += w * f16_to_f32((v_addr & 2u) ? (raw >> 16) : (raw & 0xFFFFu));
                    }
                }

                // Reduce across workers -- reuse s_reduce[] (GROUP_SIZE entries,
                // sufficient for up to 4 workers x D<=128 = 512 <= 256... wait,
                // 4x64=256 is the max that fits. For D>64 num_workers<=2 so 2x128=256.)
                if (v_wk < num_v_workers) {
                    s_reduce[v_wk * D + d_out] = partial;
                }
                GroupMemoryBarrierWithGroupSync();

                if (local_id < D) {
                    float total = 0.0f;
                    for (uint w = 0; w < num_v_workers; w++) {
                        total += s_reduce[w * D + local_id];
                    }
                    acc[0] += total;
                }
            } else if (src2_es == 2 && src2_nb0 == 2) {
                // Non-cooperative F16 fast path (D > GROUP_SIZE/2)
                uint d0 = local_id;
                if (d0 < D) {
                    float tile_acc0 = 0.0f;
                    for (uint t = 0; t < tile_size; t++) {
                        float w = s_scores[t];
                        uint v_row = v_head_base + (tile_start + t) * src2_nb1;
                        uint v_addr = v_row + d0 * 2;
                        uint raw = src2.Load(v_addr & ~3u);
                        float v = f16_to_f32((v_addr & 2u) ? (raw >> 16) : (raw & 0xFFFFu));
                        tile_acc0 += w * v;
                    }
                    acc[0] += tile_acc0;
                }
                // Handle d_out >= GROUP_SIZE (only for D > 256, rare)
                [unroll] for (uint ai = 1; ai < 4; ai++) {
                    uint d_out = local_id + ai * GROUP_SIZE;
                    if (d_out < D) {
                        float tile_acc = 0.0f;
                        for (uint t = 0; t < tile_size; t++) {
                            float w = s_scores[t];
                            uint v_row = v_head_base + (tile_start + t) * src2_nb1;
                            uint v_addr = v_row + d_out * 2;
                            uint raw = src2.Load(v_addr & ~3u);
                            float v = f16_to_f32((v_addr & 2u) ? (raw >> 16) : (raw & 0xFFFFu));
                            tile_acc += w * v;
                        }
                        acc[ai] += tile_acc;
                    }
                }
            } else {
                // Generic path: F32, BF16, or non-contiguous V
                for (uint t = 0; t < tile_size; t++) {
                    float w = s_scores[t];
                    uint v_row = v_head_base + (tile_start + t) * src2_nb1;
                    [unroll] for (uint ai = 0; ai < 4; ai++) {
                        uint d_out = local_id + ai * GROUP_SIZE;
                        if (d_out < D) {
                            acc[ai] += w * load_auto(src2, v_row + d_out * src2_nb0, src2_es);
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (split_k > 1) {
        // Split-KV: write partial results to temp buffer (dst)
        // Layout: [O: D × ne01 × split_k × ne02 × ne03] [M: ne01 × split_k × ne02 × ne03] [L: same]
        uint hb_index = head_idx + ne02 * batch_idx;
        uint o_stride = D * ne01;
        uint total_o  = o_stride * split_k * ne02 * ne03;
        uint ml_stride_val = ne01;

        // Write O partial (un-normalized: acc contains sum of exp(s-m)*v)
        for (uint ai = 0; ai < 4; ai++) {
            uint d_out = local_id + ai * GROUP_SIZE;
            if (d_out < D) {
                uint o_off = (hb_index * split_k * o_stride + split_k_index * o_stride + query_idx * D + d_out) * 4;
                dst.Store(o_off, asuint(acc[ai]));
            }
        }

        // Write M and L (one per query row per split) — only thread 0
        if (local_id == 0) {
            uint m_off = (total_o + hb_index * split_k * ml_stride_val + split_k_index * ml_stride_val + query_idx) * 4;
            uint l_off = (total_o + (ne02 * ne03 + hb_index) * split_k * ml_stride_val + split_k_index * ml_stride_val + query_idx) * 4;
            dst.Store(m_off, asuint(global_max));
            dst.Store(l_off, asuint(global_sum));
        }
        return;
    }

    // Write final output (no split)
    float inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint ai = 0; ai < 4; ai++) {
        uint d_out = local_id + ai * GROUP_SIZE;
        if (d_out < D) {
            uint out_off = dst_offset + d_out * nb0 + head_idx * nb1 + query_idx * nb2 + batch_idx * nb3;
            store_auto(dst, out_off, acc[ai] * inv_sum, dst_esize);
        }
    }
}
