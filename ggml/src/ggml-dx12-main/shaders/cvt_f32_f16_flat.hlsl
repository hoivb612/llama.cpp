// F32 -> F16 flat conversion of the MUL_MAT activation tensor.
//
// Pre-pass for mul_mat_linalg_wave_f16_i.hlsl. That shader hands its A operand
// to a matrix load, so the activations must exist as F16 in a buffer before the
// GEMM starts.
//
// Written when the groupshared matrix load looked broken. It is not: it
// converts when the array type does not match the component type
// (TUNING.md section 35). Staging in LDS instead is untried.
//
// This is the same shape of pre-pass as the Q8_1 quantize that feeds the dp4a
// tiles, and it reuses that scratch buffer. src1 is required contiguous, so the
// copy is flat and the destination row pitch is just ne00*2.
//
// src0 here is the activation tensor (the host binds it into t0), dst is the
// scratch. ne0 carries the element count.

#include "ggml_common.hlsli"

#ifndef THREADS
#define THREADS 64
#endif

// 8 elements per thread: two 16-byte loads feed one 16-byte store. The old
// 2-element version issued 4-byte loads and ran at ~86 GB/s (TUNING.md sec 40).
#define CVT_EPT 8

[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 ltid : SV_GroupThreadID) {
    // Group ids are linearised the same way the host splits them, so that a
    // conversion wider than the 65535 group limit still addresses correctly.
    const uint g = gid.x + gid.y * 65535u;
    const uint i = (g * THREADS + ltid.x) * CVT_EPT;
    if (i >= ne0) {
        return;
    }

    // ne0 is the destination extent, ne1 the source extent. The destination is
    // padded so the GEMM can read whole tiles; the pad is zeroed, not read.
    if (i + CVT_EPT <= ne1 && i + CVT_EPT <= ne0) {
        const uint4 a = src0.Load4(src0_offset + i * 4u);
        const uint4 b = src0.Load4(src0_offset + i * 4u + 16u);
        uint4 o;
        o.x = f32tof16(asfloat(a.x)) | (f32tof16(asfloat(a.y)) << 16);
        o.y = f32tof16(asfloat(a.z)) | (f32tof16(asfloat(a.w)) << 16);
        o.z = f32tof16(asfloat(b.x)) | (f32tof16(asfloat(b.y)) << 16);
        o.w = f32tof16(asfloat(b.z)) | (f32tof16(asfloat(b.w)) << 16);
        dst.Store4(i * 2u, o);
        return;
    }

    [unroll]
    for (uint e = 0; e < CVT_EPT; e += 2u) {
        const uint j = i + e;
        if (j >= ne0) {
            return;
        }
        const float v0 = (j      < ne1) ? asfloat(src0.Load(src0_offset + j * 4u))        : 0.0f;
        const float v1 = (j + 1u < ne1) ? asfloat(src0.Load(src0_offset + (j + 1u) * 4u)) : 0.0f;
        dst.Store(j * 2u, f32tof16(v0) | (f32tof16(v1) << 16));
    }
}
