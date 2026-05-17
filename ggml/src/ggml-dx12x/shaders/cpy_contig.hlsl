// cpy_contig.hlsl - Fast-path tensor copy for the contiguous-same-dtype case.
//
// Activated by the C++ dispatch when src and dst are both fully contiguous
// (nb0 = esize, nb1 = ne0 * esize, ...) AND share the same dtype, AND total
// byte count is a multiple of 16. This is the common KV-cache / residual /
// reshape case.
//
// Compared to cpy.hlsl this shader:
//   - Copies 16 bytes per thread via uint4 loads/stores instead of 1 element
//     per thread (8 F16 or 4 F32 per thread).
//   - Has no flat_to_4d -> offset_4d index math.
//   - Has no atomic compare-exchange loop (the generic shader's F16/BF16
//     write path uses InterlockedCompareExchange to avoid corrupting the
//     other half of a 32-bit word; for contig-same-dtype copies the writes
//     are naturally aligned in groups of 8 elements per uint4 so straight
//     stores are safe).
//
// Root constants: only ne0/ne1/ne2/ne3 + offsets are used. The C++ side
// passes the standard dx12_shader_params block; we read total elements from
// dst dimensions (which equals src element count for this fast path).
#include "ggml_common.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint quad_idx = tid.x;
    // Total bytes = total elements * dst_esize. Caller guarantees src and
    // dst esize are equal and total_bytes is a multiple of 16.
    uint total_elems = ne0 * ne1 * ne2 * ne3;
    uint total_bytes = total_elems * dst_esize;
    uint total_quads = total_bytes >> 4;  // /16
    if (quad_idx >= total_quads) return;

    uint byte_off  = quad_idx << 4;       // *16
    uint src_off   = src0_offset + byte_off;
    uint dst_off   = dst_offset  + byte_off;
    uint4 v        = src0.Load4(src_off);
    dst.Store4(dst_off, v);
}
