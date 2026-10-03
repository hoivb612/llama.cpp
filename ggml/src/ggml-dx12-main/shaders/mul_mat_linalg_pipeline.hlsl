#include "ggml_common.hlsli"
#include <dx/linalg.h>

using namespace dx::linalg;

#ifndef LA_NWAVE
#define LA_NWAVE 4
#endif
#ifndef LA_MT
#define LA_MT 2
#endif
#ifndef LA_NT
#define LA_NT 4
#endif
#ifndef BK
#define BK 16
#endif
#define BM (LA_NWAVE * LA_MT * 16)
#define BN (LA_NT * 16)
#define THREADS (LA_NWAVE * 64)
#define B_PER_THREAD (BN * BK / THREADS)

#if (BK != 16 && BK != 32) || BM * BK != THREADS * 8 || (B_PER_THREAD != 4 && B_PER_THREAD != 8)
#error Unsupported pipeline load geometry
#endif
#if LA_QUANT != 0 && (LA_QUANT != 8 || B_PER_THREAD != 4)
#error Unsupported pipeline weight format
#endif

typedef Matrix<ComponentType::F16, 16, 16, MatrixUse::A, MatrixScope::Wave> MatA;
typedef Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave> MatB;
typedef Matrix<ComponentType::F32, 16, 16, MatrixUse::Accumulator, MatrixScope::Wave> MatAcc;

groupshared float16_t tile_a[2 * BM * BK];
groupshared float16_t tile_b[2 * BN * BK];
groupshared float tile_c[LA_NWAVE * 16 * 16];

struct RawTile {
    uint4 a0;
    uint4 a1;
    uint scale;
    uint4 qs;
};

RawTile fetch_raw(uint k0, uint a_row, uint b_row, uint tid) {
    RawTile raw;
    uint a_off = a_row + (k0 + (tid % (BK / 8u)) * 8u) * 4u;
    raw.a0 = src1.Load4(a_off);
    raw.a1 = src1.Load4(a_off + 16u);
    uint b_k = k0 + (tid % (BK / B_PER_THREAD)) * B_PER_THREAD;
#if LA_QUANT == 8
    uint b_block = b_row + (b_k / 32u) * 34u;
    raw.scale = src0.Load(b_block & ~3u);
    uint q_off = b_block + 2u + (b_k & 31u);
    raw.qs.x = src0.Load(q_off & ~3u);
    raw.qs.y = (q_off & 2u) != 0u ? src0.Load((q_off & ~3u) + 4u) : 0u;
    raw.qs.zw = 0u;
#else
    raw.scale = 0u;
#if B_PER_THREAD == 8
    raw.qs = src0.Load4(b_row + b_k * 2u);
#else
    raw.qs.xy = src0.Load2(b_row + b_k * 2u);
    raw.qs.zw = 0u;
#endif
#endif
    return raw;
}

void publish_raw(RawTile raw, uint k0, uint b_row, uint tid, uint slot) {
    uint a_base = slot * BM * BK;
    uint b_base = slot * BN * BK;
    [unroll] for (uint e = 0u; e < 4u; ++e) {
        tile_a[a_base + tid * 8u + e] = (float16_t)asfloat(raw.a0[e]);
        tile_a[a_base + tid * 8u + e + 4u] = (float16_t)asfloat(raw.a1[e]);
    }
#if LA_QUANT == 8
    uint block = b_row + (k0 / 32u) * 34u;
    float d = f16_to_f32((raw.scale >> ((block & 2u) * 8u)) & 0xFFFFu);
    uint qs = (block & 2u) != 0u ? raw.qs.x : (raw.qs.x >> 16u) | (raw.qs.y << 16u);
    [unroll] for (uint e = 0u; e < 4u; ++e) {
        int q = (int)(qs << (24u - e * 8u)) >> 24;
        tile_b[b_base + tid * 4u + e] = (float16_t)(d * (float)q);
    }
#else
    [unroll] for (uint e = 0u; e < B_PER_THREAD; ++e) {
        tile_b[b_base + tid * B_PER_THREAD + e] = asfloat16((uint16_t)(raw.qs[e / 2u] >> ((e & 1u) * 16u)));
    }
#endif
}

void accumulate_tile(uint slot, uint wave, inout MatAcc acc[LA_MT][LA_NT]) {
    [unroll] for (uint kk = 0u; kk < BK; kk += 16u) {
        MatB b[LA_NT];
        [unroll] for (uint n = 0u; n < LA_NT; ++n) {
            b[n] = MatB::Load(tile_b, slot * BN * BK + n * 16u * BK + kk, BK, MatrixLayout::ColMajor);
        }
        [unroll] for (uint m = 0u; m < LA_MT; ++m) {
            MatA a = MatA::Load(tile_a, slot * BM * BK + (wave * LA_MT + m) * 16u * BK + kk, BK, MatrixLayout::RowMajor);
            [unroll] for (uint n = 0u; n < LA_NT; ++n) {
                acc[m][n].MultiplyAccumulate(a, b[n]);
            }
        }
    }
}

[WaveSize(64)]
[numthreads(THREADS, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    uint i2 = gid.z % ne2;
    uint i3 = gid.z / ne2;
    uint i2_src0 = i2 * ne02 / ne2;
    uint i3_src0 = i3 * ne03 / ne3;
    uint row0 = gid.x * BM;
    uint col0 = gid.y * BN;
    uint a_row = offset_4d(0u, row0 + tid / (BK / 8u), i2, i3, nb10, nb11, nb12, nb13, src1_offset);
    uint b_row = src0_offset + (col0 + tid / (BK / B_PER_THREAD)) * nb01 + i2_src0 * nb02 + i3_src0 * nb03;
    uint wave = tid / 64u;
    uint lane = tid & 63u;

    MatAcc acc[LA_MT][LA_NT];
    [unroll] for (uint m = 0u; m < LA_MT; ++m) {
        [unroll] for (uint n = 0u; n < LA_NT; ++n) {
            acc[m][n] = MatAcc::Splat(0.0f);
        }
    }

    publish_raw(fetch_raw(0u, a_row, b_row, tid), 0u, b_row, tid, 0u);
    // Keep distinct raw payloads and constant LDS slots across each K pair.
    [loop] for (uint k0 = 0u; k0 + 2u * BK < ne00; k0 += 2u * BK) {
        RawTile next1 = fetch_raw(k0 + BK, a_row, b_row, tid);
        GroupMemoryBarrierWithGroupSync();
        accumulate_tile(0u, wave, acc);
        publish_raw(next1, k0 + BK, b_row, tid, 1u);

        RawTile next0 = fetch_raw(k0 + 2u * BK, a_row, b_row, tid);
        GroupMemoryBarrierWithGroupSync();
        accumulate_tile(1u, wave, acc);
        publish_raw(next0, k0 + 2u * BK, b_row, tid, 0u);
    }
    RawTile tail = fetch_raw(ne00 - BK, a_row, b_row, tid);
    GroupMemoryBarrierWithGroupSync();
    accumulate_tile(0u, wave, acc);
    publish_raw(tail, ne00 - BK, b_row, tid, 1u);
    GroupMemoryBarrierWithGroupSync();
    accumulate_tile(1u, wave, acc);

    uint c_base = wave * 16u * 16u;
    [unroll] for (uint m = 0u; m < LA_MT; ++m) {
        [unroll] for (uint n = 0u; n < LA_NT; ++n) {
            acc[m][n].Store(tile_c, c_base, 16u, MatrixLayout::RowMajor);
            GroupMemoryBarrier();
            [unroll] for (uint p = 0u; p < 4u; ++p) {
                uint elem = p * 64u + lane;
                uint row = row0 + (wave * LA_MT + m) * 16u + elem / 16u;
                uint col = col0 + n * 16u + elem % 16u;
                float value = tile_c[c_base + elem];
                if (op0 == 1u) {
                    value += asfloat(src2.Load(op1 + col * op2));
                }
                uint off = offset_4d(col, row, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
                dst.Store(off, asuint(value));
            }
            GroupMemoryBarrier();
        }
    }
}
