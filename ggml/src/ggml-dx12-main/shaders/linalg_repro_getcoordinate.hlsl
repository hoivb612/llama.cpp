// linalg_repro_getcoordinate.hlsl - self-checking repro for a suspected driver
// bug in the SM 6.10 LinAlg preview: Accumulator::GetCoordinate() does not
// agree with Accumulator::Store().
//
// Both describe the same thing -- where a lane's accumulator elements sit in
// the MxN tile -- so writing the tile out through each of them must produce
// identical results. On an RX 9070 XT (driver 32.0.23041.2023) they differ.
//
// The shader needs no host-side checking. It fills a 16x16 tile with a
// multiply, writes it out twice (once via Store, once via GetCoordinate/Get)
// and reports the disagreement:
//
//   OutBuff[0]              number of cells where the two paths disagree
//                           (expected 0; nonzero on the affected driver)
//   OutBuff[1..256]         the tile as written by Store         (reference)
//   OutBuff[257..512]       the tile as written by GetCoordinate (suspect)
//   OutBuff[513..768]       per (lane,element) packed coordinate reported by
//                           GetCoordinate, as (row << 16) | col, so the
//                           reported mapping can be inspected directly
//
// Build:
//   dxc -T cs_6_10 -E main -I <dxc>/inc/hlsl linalg_repro_getcoordinate.hlsl
//
// For the F16 x F16 -> F32 shape the GEMM and flash-attention kernels use:
//   dxc -T cs_6_10 -E main -I <dxc>/inc/hlsl -enable-16bit-types \
//       -DREPRO_F32=1 linalg_repro_getcoordinate.hlsl
//
// OutBuff[769] additionally reports how many cells disagree when the mapping
// is hardcoded rather than queried, and OutBuff[770..] dumps the raw Get(e)
// values, which under the REPRO_F32 data pattern name the cell each lane
// really owns. Both showed the layout cannot be reconstructed by hand.
//
// Run with D3D12_FEATURE_D3D12_OPTIONS_EXPERIMENTAL / experimental shader
// models enabled, as the LinAlg preview requires.

#include <dx/linalg.h>
using namespace dx::linalg;

RWByteAddressBuffer OutBuff : register(u0);

#ifndef DUMP_E
#define DUMP_E 4
#endif

// Build with -DREPRO_F32 to run the same check on the F16 x F16 -> F32 shape
// the GEMM and flash-attention kernels actually use, rather than the integer
// one. The two shapes are separate driver paths and need not fail alike.

#define TILE 16
#define WAVE 64
#ifndef ACC_E
#define ACC_E (TILE * TILE / WAVE)   // accumulator elements owned per lane
#endif

#ifdef REPRO_F32
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::F16, TILE, TILE, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::F32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared float16_t a_tile[TILE * TILE];
groupshared float16_t b_tile[TILE * TILE];
groupshared float     c_store[TILE * TILE];
groupshared float     c_coord[TILE * TILE];
groupshared float     c_assume[TILE * TILE];
#else
typedef Matrix<ComponentType::I8,  TILE, TILE, MatrixUse::A,           MatrixScope::Wave> MatA8;
typedef Matrix<ComponentType::I8,  TILE, TILE, MatrixUse::B,           MatrixScope::Wave> MatB8;
typedef Matrix<ComponentType::U32, TILE, TILE, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared int a_tile[TILE * TILE];
groupshared int b_tile[TILE * TILE];
groupshared int c_store[TILE * TILE];
groupshared int c_coord[TILE * TILE];
groupshared int c_assume[TILE * TILE];
#endif

[numthreads(WAVE, 1, 1)]
[shader("compute")]
void main(uint tid : SV_GroupIndex) {
    // I8 matrices pack 4 elements per uint, so a 16x16 tile is 64 uints.
    // Any non-symmetric data works; this makes every cell of the product
    // distinct so a wrong coordinate cannot alias onto a right one.
#ifdef REPRO_F32
    // A is the identity and B[k][c] = k*16+c+1, so C[r][c] == r*16+c+1: every
    // accumulator cell encodes its own coordinates. Whatever Get(e) returns
    // therefore names the cell that lane actually holds, which derives the
    // true layout instead of guessing it. All values are <= 256 and so are
    // exact in F16.
    for (uint i = tid; i < TILE * TILE; i += WAVE) {
        const uint r = i / TILE, c = i % TILE;
        a_tile[i] = (float16_t)(int)(r == c ? 1 : 0);
        b_tile[i] = (float16_t)(int)(i + 1);
    }
    for (uint z = tid; z < TILE * TILE; z += WAVE) {
        c_store[z] = 0.0f;
        c_coord[z] = 0.0f;
        c_assume[z] = 0.0f;
    }
#else
    for (uint i = tid; i < TILE * TILE / 4; i += WAVE) {
        a_tile[i] = (int)(0x01020304u + i);
        b_tile[i] = (int)(0x04030201u + i * 7u);
    }
    for (uint z = tid; z < TILE * TILE; z += WAVE) {
        c_store[z] = 0;
        c_coord[z] = 0;
        c_assume[z] = 0;
    }
#endif
    GroupMemoryBarrierWithGroupSync();

    MatA8  a   = MatA8::Load(a_tile, 0, TILE, MatrixLayout::RowMajor);
    MatB8  b   = MatB8::Load(b_tile, 0, TILE, MatrixLayout::ColMajor);
#ifdef REPRO_F32
    MatAcc acc = MatAcc::Splat(0.0f);
#else
    MatAcc acc = MatAcc::Splat(0u);
#endif
    acc.MultiplyAccumulate(a, b);

    // Path 1: let the API place the elements.
    acc.Store(c_store, 0, TILE, MatrixLayout::RowMajor);

    // Path 2: place them by hand using the coordinates the API reports.
    for (uint e = 0; e < ACC_E; e++) {
        const uint2 rc = acc.GetCoordinate(e);
        OutBuff.Store((513u + tid * ACC_E + e) * 4u, (rc.x << 16) | rc.y);
        if (rc.x < TILE && rc.y < TILE) {
#ifdef REPRO_F32
            c_coord[rc.x * TILE + rc.y] = acc.Get(e);
#else
            c_coord[rc.x * TILE + rc.y] = asint(acc.Get(e));
#endif
        }
    }
    // Path 3: place them by hand using the mapping the hardware actually
    // appears to use, ignoring GetCoordinate entirely. If this agrees with
    // Store() then the layout is knowable and a kernel can keep accumulators
    // in registers despite the broken coordinate query.
    for (uint e2 = 0; e2 < ACC_E; e2++) {
        const uint lane = tid % WAVE;
        const uint r    = lane % TILE;
        const uint c    = (lane / TILE) * ACC_E + e2;
        if (r < TILE && c < TILE) {
#ifdef REPRO_F32
            c_assume[r * TILE + c] = acc.Get(e2);
#else
            c_assume[r * TILE + c] = asint(acc.Get(e2));
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // The two tiles must be identical.
    uint local_bad = 0;
    uint local_bad2 = 0;
    for (uint j = tid; j < TILE * TILE; j += WAVE) {
        OutBuff.Store((1u   + j) * 4u, asuint(c_store[j]));
        OutBuff.Store((257u + j) * 4u, asuint(c_coord[j]));
        if (c_store[j] != c_coord[j]) {
            local_bad++;
        }
        if (c_store[j] != c_assume[j]) {
            local_bad2++;
        }
    }
    const uint total_bad  = WaveActiveSum(local_bad);
    const uint total_bad2 = WaveActiveSum(local_bad2);
    if (WaveIsFirstLane()) {
        OutBuff.Store(0, total_bad);
        OutBuff.Store(769u * 4u, total_bad2);
    }

#ifdef REPRO_F32
    // Ground truth: with the identity-A setup above, Get(e) returns
    // r*16+c+1 for the cell this lane really owns.
    for (uint e3 = 0; e3 < ACC_E; e3++) {
        OutBuff.Store((770u + tid * ACC_E + e3) * 4u, (uint)(int)acc.Get(e3));
    }
#endif
}
