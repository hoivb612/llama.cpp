// cumsum.hlsl - Per-row inclusive prefix sum (cumulative sum).
//
// CPU reference: ggml_compute_forward_cumsum_f32 / ggml_vec_cumsum_f32
// Output ne == input ne; per (i01, i02, i03) row, scans along ne00.
//
// Dispatch: one thread group per row. groups.x = ne01 * ne02 * ne03.
//
// Algorithm (matches Vulkan's cumsum.comp):
//   1. Each thread processes ELEM_PER_THREAD = 4 contiguous columns,
//      computing local prefix sums in v[j] and total in thread_sum.
//   2. WavePrefixSum (exclusive) on thread_sum across the wave.
//   3. Cross-wave merge: last lane of each wave writes wave-total to
//      partial[wave_id]; threads add partials[0..wave_id-1] to v[].
//   4. Add last_sum carry from previous iteration; the last thread of
//      the group then updates last_sum for the next iteration.
//   5. Write results.
//
// MAX_WAVES = 32 accommodates Intel UHD wave=8 (256/8 = 32 waves).

#include "ggml_common.hlsli"

#define GROUP_SIZE 256
#define ELEM_PER_THREAD 4
#define MAX_WAVES 32

groupshared float partial[MAX_WAVES];
groupshared float last_sum;

[numthreads(GROUP_SIZE, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid_idx : SV_GroupIndex) {
    uint row = gid.x;
    uint i01 = row % ne01;
    uint rem = row / ne01;
    uint i02 = rem % ne02;
    uint i03 = rem / ne02;

    uint off_src_row = src0_offset + i01 * nb01 + i02 * nb02 + i03 * nb03;
    uint off_dst_row = dst_offset  + i01 * nb1  + i02 * nb2  + i03 * nb3;

    uint n_cols    = ne00;
    uint wave_size = WaveGetLaneCount();
    uint wave_id   = tid_idx / wave_size;
    uint lane      = WaveGetLaneIndex();

    if (tid_idx == 0) {
        last_sum = 0.0f;
    }
    GroupMemoryBarrierWithGroupSync();

    uint chunk_size = GROUP_SIZE * ELEM_PER_THREAD;
    uint num_iter   = (n_cols + chunk_size - 1) / chunk_size;
    uint col        = tid_idx * ELEM_PER_THREAD;

    for (uint iter = 0; iter < num_iter; ++iter) {
        float v[ELEM_PER_THREAD];
        float thread_sum = 0.0f;

        [unroll]
        for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            if (col + j < n_cols) {
                thread_sum += asfloat(src0.Load(off_src_row + (col + j) * 4u));
            }
            v[j] = thread_sum;
        }

        float wave_prefix = WavePrefixSum(thread_sum);
        [unroll]
        for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            v[j] += wave_prefix;
        }

        if (lane == wave_size - 1) {
            partial[wave_id] = v[ELEM_PER_THREAD - 1];
        }
        GroupMemoryBarrierWithGroupSync();

        float wave_offset = 0.0f;
        for (uint s = 0; s < wave_id; ++s) {
            wave_offset += partial[s];
        }
        [unroll]
        for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            v[j] += wave_offset + last_sum;
        }

        GroupMemoryBarrierWithGroupSync();
        if (tid_idx == GROUP_SIZE - 1) {
            last_sum = v[ELEM_PER_THREAD - 1];
        }

        [unroll]
        for (uint j = 0; j < ELEM_PER_THREAD; ++j) {
            if (col + j < n_cols) {
                dst.Store(off_dst_row + (col + j) * 4u, asuint(v[j]));
            }
        }
        GroupMemoryBarrierWithGroupSync();

        col += chunk_size;
    }
}
