// cpy.hlsl - Copy tensor data: dst = src0
// Handles different strides/layouts, reshape, F32↔F16↔BF16 conversion, and
// I32→F32 / F32→I32 cast.
//
// op_params[0] = 1 if src0 type is I32 (else 0; F32/F16/BF16 inferred from src0_esize)
// op_params[1] = 1 if dst type is I32 (else 0)
//
// Type encoding via element size sentinels (set in dx12_fill_params):
//   esize=2 → F16
//   esize=3 → BF16 (sentinel; actual element size is 2)
//   esize=4 → F32 (or I32 if the corresponding op_params flag is set)
#include "ggml_common.hlsli"

// Load one element as float, type-aware.
// is_i32: pass true to interpret a 4-byte word as int32 and convert to float
//         (for I32 source); false to use load_auto_bf16 which handles F16/BF16/F32.
float load_typed(ByteAddressBuffer buf, uint byte_offset, uint elem_stride, bool is_i32) {
    if (is_i32) {
        return (float)asint(buf.Load(byte_offset));
    }
    return load_auto_bf16(buf, byte_offset, elem_stride);
}

// Store one float element, type-aware.
void store_typed(RWByteAddressBuffer buf, uint byte_offset, float val, uint elem_stride, bool is_i32) {
    if (is_i32) {
        buf.Store(byte_offset, asuint((int)val));
        return;
    }
    store_auto_bf16(buf, byte_offset, val, elem_stride);
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    bool src_is_i32 = (op0 != 0u);
    bool dst_is_i32 = (op1 != 0u);

    // Paired F16 store optimization: only safe when both src0 and dst are F16,
    // src0 is contiguous in dim 0, ne0 is even, and dst aligns.
    // I32 and BF16 paths cannot use the paired F16 helper.
    bool paired = !src_is_i32 && !dst_is_i32 &&
                  can_pair_f16() && nb00 == src0_esize && (ne00 & 1) == 0 &&
                  src0_esize == 2;

    uint idx = tid.x * (paired ? 2u : 1u);
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    uint i0, i1, i2, i3;
    flat_to_4d(idx, ne0, ne1, ne2, i0, i1, i2, i3);

    uint j0, j1, j2, j3;
    flat_to_4d(idx, ne00, ne01, ne02, j0, j1, j2, j3);
    uint off0 = offset_4d(j0, j1, j2, j3, nb00, nb01, nb02, nb03, src0_offset);

    float v0 = load_typed(src0, off0, src0_esize, src_is_i32);

    if (paired) {
        float v1 = load_typed(src0, off0 + nb00, src0_esize, src_is_i32);
        uint off_d = dst_offset + i0 * 2u + i1 * nb1 + i2 * nb2 + i3 * nb3;
        store_f16_pair(dst, off_d, v0, v1);
    } else {
        uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
        store_typed(dst, off_d, v0, dst_esize, dst_is_i32);
    }
}
