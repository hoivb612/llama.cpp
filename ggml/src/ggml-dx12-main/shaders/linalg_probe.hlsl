// linalg_probe.hlsl — Phase 0 toolchain probe for the D3D12 LinAlg Matrix
// preview (Shader Model 6.10).
//
// Purpose:
//   Verify that the LinAlg-aware build path (DXC 1.10.2605.2 preview + Agility
//   SDK 1.720+ preview + cs_6_10 target + dx/linalg.h header) compiles end-to-
//   end. This shader is intentionally minimal: it loads one F16 matrix at
//   MatrixScope::Thread and one F16 vector, computes a single matrix-vector
//   multiply via the new linalg::Multiply intrinsic, and stores the result.
//
//   It is NOT yet wired into runtime dispatch; the goal of Phase 0 is purely
//   to confirm the toolchain works on the developer's machine. Runtime
//   dispatch and performance benchmarking arrive in Phase 1.
//
// Build:
//   Compiled only when -DGGML_DX12_LINALG_PREVIEW=ON is set on the CMake
//   configure line AND both GGML_DX12_AGILITY_SDK_PATH and GGML_DX12_DXC_PATH
//   are provided. See ggml/src/ggml-dx12/CMakeLists.txt for the build wiring.
//
// References:
//   - https://devblogs.microsoft.com/directx/d3d12-linalg-preview/
//   - HLSL spec: https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md

#include <dx/linalg.h>

using namespace dx::linalg;

ByteAddressBuffer  InBuff  : register(t0);
RWByteAddressBuffer OutBuff : register(u0);

// 16x16 F16 matrix at MatrixScope::Thread — directly maps to our matvec
// (MUL_MAT M=1) hot kernels. F16 chosen because it's the most commonly
// accelerated component type on shipping hardware in this preview window.
using MatrixATy = Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Thread>;

[numthreads(8, 1, 1)]
[shader("compute")]
void main() {
    MatrixATy mat = MatrixATy::Load<MatrixLayout::RowMajor>(
        InBuff, /*offset*/ 0, /*row_stride_bytes*/ 16 * 2);

    vector<float16_t, 16> v = (vector<float16_t, 16>)0;
    vector<float16_t, 16> outv =
        Multiply<float16_t, float16_t, 16, 16, ComponentType::F16>(mat, v);

    OutBuff.Store4(0,  asuint(float4(outv[0], outv[1], outv[2], outv[3])));
    OutBuff.Store4(16, asuint(float4(outv[4], outv[5], outv[6], outv[7])));
}
