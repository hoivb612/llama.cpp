// set_rows_q8_0.hlsl - SET_ROWS with F32 src0 -> Q8_0 dst.
//
// Vulkan parity: copy_to_quant.comp (SET_ROWS variant, DATA_A_Q8_0 quantize).
// One thread per Q8_0 block (32 F32 elements). 32 threads per workgroup.
//
// Requires native 16-bit shader ops (-enable-16bit-types). The Q8_0 block
// stride is 34 bytes (uint16_t d + int8 qs[32]), so per-row strides are
// generally NOT a multiple of 4. Using Store<uint16_t> sidesteps the
// 4-byte alignment requirement of uint32 byte-address stores.
//
// Dispatch geometry (matches Vulkan's CEIL_DIV(nelements, 32 * QUANT_K)):
//   groups_x = ceil(total_blocks / 32), 32 threads/wg, 1 block/thread.
//
// 8-bit types (uint8_t/int8_t) are NOT available in -enable-16bit-types,
// so qs[0..31] (32 int8 bytes) are written as 16 packed uint16_t stores
// at 2-byte-aligned offsets within each block.
#include "ggml_common.hlsli"

#define QK8_0 32

[numthreads(32, 1, 1)]
void main(uint3 gid : SV_GroupID, uint local_id : SV_GroupThreadID) {
    // Vulkan-style 3D group_id flattening so we can grow dispatch beyond
    // the per-axis DispatchThreadID limit without changing the shader.
    uint global_thread = (gid.z * 262144u + gid.y * 512u + gid.x) * 32u + local_id;
    uint block_idx = global_thread;

    // Total Q8_0 blocks across the entire src0 tensor.
    uint nb_per_row   = ne00 / QK8_0;
    uint total_blocks = nb_per_row * ne01 * ne02 * ne03;
    if (block_idx >= total_blocks) return;

    // Map flat block index back to (bi0, i1, i2, i3) src0 coords.
    uint bi0 = block_idx % nb_per_row;
    uint rem = block_idx / nb_per_row;
    uint i1  = rem % ne01;
    rem      = rem / ne01;
    uint i2  = rem % ne02;
    uint i3  = rem / ne02;

    // Row index lookup from src1.
    // src1 (indices) is 3D [ne10, ne11, ne12] mapping to dst (i1, i2, i3).
    // Match CPU semantics: broadcast dim2/3 with modulo (matches set_rows.hlsl).
    uint i2_idx  = ne11 > 0 ? (i2 % ne11) : 0;
    uint i3_idx  = ne12 > 0 ? (i3 % ne12) : 0;
    uint idx_off = src1_offset + i1 * nb10 + i2_idx * nb11 + i3_idx * nb12;
    int  row_idx = asint(src1.Load(idx_off));

    // src0 byte offset of the first F32 element in this block.
    uint base_i0 = bi0 * QK8_0;
    uint off0    = src0_offset + base_i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03;

    // Pass 1: find amax across the 32 elements of this block.
    float amax = 0.0f;
    [unroll] for (uint j = 0; j < QK8_0; ++j) {
        float v = asfloat(src0.Load(off0 + j * nb00));
        amax = max(amax, abs(v));
    }

    float d  = amax / 127.0f;
    // Match Vulkan's two-step formulation: amax/127, then 1/d.
    float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

    // dst block layout: { uint16_t d; int8 qs[32]; } = 34 bytes total.
    // dst_block_off is the byte offset of THIS block within the dst buffer.
    // All addresses below are at least 2-byte aligned because nb1/nb2/nb3
    // and 34 are all even (host gates ne00 % 32 == 0).
    uint dst_block_off = dst_offset + bi0 * 34u + (uint)row_idx * nb1 + i2 * nb2 + i3 * nb3;

    // Write d as a native 16-bit value (2-aligned). The native cast is an
    // IEEE fptrunc (round-to-nearest-even), matching GGML_FP32_TO_FP16 on the
    // CPU; the legacy f32tof16 intrinsic truncates instead and leaves a
    // systematic ~1 f16 ulp bias in every block scale.
    dst.Store<uint16_t>(dst_block_off, asuint16((float16_t)d));

    // Pass 2: quantize the 32 elements and pack 2 int8s per uint16_t store
    // (16 stores at offsets dst_block_off + 2 + 0,2,4,...,30).
    //
    // HLSL `round()` is round-to-nearest-EVEN on DXIL, matching both
    // CPU AVX `_MM_ROUND_NEAREST` and Vulkan `round()`.
    [unroll] for (uint k = 0; k < 16; ++k) {
        float v0 = asfloat(src0.Load(off0 + (2u * k)      * nb00));
        float v1 = asfloat(src0.Load(off0 + (2u * k + 1u) * nb00));
        int q0 = (int)round(v0 * id);
        int q1 = (int)round(v1 * id);
        q0 = clamp(q0, -128, 127);
        q1 = clamp(q1, -128, 127);
        uint16_t packed = (uint16_t)((uint(q0) & 0xFFu) | ((uint(q1) & 0xFFu) << 8));
        dst.Store<uint16_t>(dst_block_off + 2u + k * 2u, packed);
    }
}
