// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// matmulnbits_avx512.h
//
// AVX-512F-vectorized int4 dequant dot-product for the FusionCPU MatMulNBits
// kernel (M3++). The implementation lives in matmulnbits_avx512.cpp, which is
// the ONLY translation unit compiled with /arch:AVX512. The rest of the DLL is
// built at x64 baseline so it still loads on non-AVX-512 CPUs; callers MUST
// gate every call on HasAvx512f() (defined in the baseline TU) so an AVX-512
// instruction is never executed on hardware that lacks the feature.

#pragma once

#include <cstdint>

namespace fusion_cpu_ep {

// Dequantizes one int4 weight column into a dense fp32 buffer (M3+++):
//   out[b*block_size + t] = (q[b][t] - zp[b]) * HalfToFloat(scales_row[b])
// for every block b in [0, k_blocks) and element t in [0, block_size). This is
// the "dequant once, reuse across rows" step that hoists the expensive nibble
// unpack out of the per-output-row loop.
//
//   out         : destination fp32 buffer, length k_blocks * block_size
//   b_row       : packed int4 weights for this column, k_blocks * (block_size/2) bytes
//   scales_row  : fp32 scales, one per block (k_blocks entries). Callers normalize
//                 fp16 scales to fp32 once at the kernel boundary so this TU is
//                 dtype-agnostic (serves both fp16-IO and fp32-IO MatMulNBits).
//   zp_row      : packed int4 zero points (k_blocks nibbles), or nullptr
//   has_zp      : whether zp_row is valid; when false, default_zp is used
//   k_blocks    : number of quant blocks along K
//   block_size  : elements per block (a multiple of 16)
//   default_zp  : zero point used when has_zp is false (typically 8)
//
// Precondition: block_size % 16 == 0 and HasAvx512f().
void DequantColumnAvx512(float* out, const uint8_t* b_row,
                         const float* scales_row, const uint8_t* zp_row,
                         bool has_zp, int64_t k_blocks, int64_t block_size,
                         int default_zp);

// fp32 dot product of two length-k vectors using AVX-512F FMA, returning the
// fp32 accumulator (caller applies the final fp16 round).
//
// Precondition: k % 16 == 0 and HasAvx512f().
float DotFp32Avx512(const float* a, const float* w, int64_t k);

// Per-column VNNI int8 dot (decode-optimal, M4 Path B). Computes one output
// element y[m,n] = sum_b ascale[b] * wscale[b] * (dpbusd(au8_block, wq8_block)
// - 128*wsum[b]). It streams the int8 weight column directly (no prepack /
// interleave), so it wins when M is small -- notably M=1 decode -- where the
// register-blocked tile kernel's prepack cost is not amortized across rows.
//
//   au8       : activation row, uint8 (= int8 + 128), length k_blocks*block_size
//   ascale    : per-block activation scale (fp32), length k_blocks
//   wq8       : weight column re-centered to int8, length k_blocks*block_size
//   wscale    : per-block weight scale (fp32), length k_blocks
//   wsum      : per-block sum of (q - zp) (int32), length k_blocks
//
// Precondition: block_size % 64 == 0 and HasAvx512Vnni().
float DotInt8VnniColumn(const uint8_t* au8, const float* ascale,
                        const int8_t* wq8, const float* wscale,
                        const int32_t* wsum, int64_t k_blocks, int64_t block_size);

// Contiguous int4 -> int8 weight re-pack for the per-column VNNI path. Writes
// wq8_out[b*block_size + t] = (q - zp) as signed int8, the per-block fp32 scale,
// and wsum_out[b] = sum_t (q - zp). Same packed-weight layout as DequantColumnAvx512.
//
// Precondition: block_size % 16 == 0 and HasAvx512f().
void RepackWeightColumnInt8Avx512(int8_t* wq8_out, float* wscale_out,
                                  int32_t* wsum_out, const uint8_t* b_row,
                                  const float* scales_row, const uint8_t* zp_row,
                                  bool has_zp, int64_t k_blocks, int64_t block_size,
                                  int default_zp);

// AVX-512-VNNI register-blocked int8 dot for a tile of 16 output columns (M4
// Path B, MLAS-style). Computes out16[i] for i in [0,16):
//   out16[i] = sum_b ascale[b] * wscale_packed[b*16+i]
//                    * ( dpbusd_lane_i - 128 * wsum_packed[b*16+i] )
// where the 16 int32 dpbusd lanes ARE the 16 columns' block dot products, so the
// per-block dequant is a single 16-wide vector op (no horizontal reduction) and
// the activation row is streamed once (broadcast 4 bytes per dpbusd) rather than
// re-read per column.
//
//   out16         : 16 fp32 outputs (caller rounds/stores the valid columns)
//   au8_row       : activation row, uint8 (= int8 + 128), length k_blocks*block_size
//   ascale        : per-block activation scale (fp32), length k_blocks
//   wq8p          : prepacked weights, layout [block][k-strip][16 cols][4 k],
//                   length k_blocks * block_size * 16 (see RepackWeightTile16Int8Avx512)
//   wscale_packed : per-(block,col) weight scale (fp32), layout [block][16], length k_blocks*16
//   wsum_packed   : per-(block,col) sum of (q - zp) (int32), layout [block][16], length k_blocks*16
//
// Precondition: block_size % 4 == 0 and HasAvx512Vnni().
void DotInt8VnniTile16Avx512(float* out16, const uint8_t* au8_row, const float* ascale,
                             const int8_t* wq8p, const float* wscale_packed,
                             const int32_t* wsum_packed, int64_t k_blocks, int64_t block_size);

// Prepacks up to 16 weight columns into the interleaved [block][k-strip][16 cols][4 k]
// int8 layout consumed by DotInt8VnniTile16Avx512, plus per-(block,col) fp32 scale
// and int32 sum of (q - zp). Columns beyond cnt are zero-filled (zero weights
// contribute nothing to the tile dot). The source columns are addressed by base
// pointer + per-column stride so the caller's full packed-B layout stays here.
//
//   wq8p          : destination int8 tile, length k_blocks * block_size * 16
//   wscale_packed : destination fp32 scales, layout [block][16], length k_blocks*16
//   wsum_packed   : destination int32 weight sums, layout [block][16], length k_blocks*16
//   b_base/b_stride   : packed int4 weights of column 0 and the byte stride between columns
//   sc_base/sc_stride : fp32 scales of column 0 and the element stride between columns
//   zp_base/zp_stride : packed int4 zero points of column 0 and the byte stride (ignored if !has_zp)
//   has_zp        : whether zero points are present; otherwise default_zp is used
//   cnt           : number of valid columns in this tile (1..16)
//   k_blocks, block_size, default_zp : as in DequantColumnAvx512
//
// Precondition: block_size % 16 == 0 and HasAvx512f().
void RepackWeightTile16Int8Avx512(int8_t* wq8p, float* wscale_packed, int32_t* wsum_packed,
                                  const uint8_t* b_base, int64_t b_stride,
                                  const float* sc_base, int64_t sc_stride,
                                  const uint8_t* zp_base, int64_t zp_stride, bool has_zp,
                                  int cnt, int64_t k_blocks, int64_t block_size, int default_zp);

// COMPACT int4 variant of the tile prepack/dot pair (decode bandwidth fix). The
// resident weights stay nibble-packed -- HALF the bytes of the int8 tile, matching
// the model's native int4 footprint -- and are expanded to int8 in-register inside
// the dpbusd strip loop. The math (per-block dpbusd + 128*wsum bias undo + per-col
// scale) and the wscale/wsum layout are IDENTICAL to the int8 tile; only the weight
// storage differs. Used only when has_zp is false (default_zp=8 => (q-zp) in [-8,7]
// fits a signed nibble), so no zero_points are taken.
//
// Compact tile layout (per 16-col tile): [block][strip][16 cols][2 bytes], where
// each 2-byte unit holds a column's 4 k-weights as 4 signed nibbles
//   byte0 = (w_k0 & 0xF) | ((w_k1 & 0xF) << 4),  byte1 = (w_k2 & 0xF) | ((w_k3 & 0xF) << 4)
// with w = (q - default_zp). Total bytes = k_blocks * block_size * 16 / 2.
//
// Precondition: block_size % 16 == 0 and HasAvx512f().
void RepackWeightTile16Int4Avx512(uint8_t* wq4p, float* wscale_packed, int32_t* wsum_packed,
                                  const uint8_t* b_base, int64_t b_stride,
                                  const float* sc_base, int64_t sc_stride,
                                  int cnt, int64_t k_blocks, int64_t block_size, int default_zp);

// VNNI tile dot over the compact int4 tile from RepackWeightTile16Int4Avx512.
// Expands the nibble weights to signed int8 in-register ((n ^ 8) - 8) before each
// dpbusd, so it reads half the weight bytes of DotInt8VnniTile16Avx512 while
// producing the identical result. This is the decode-bandwidth-optimal kernel.
//
// Precondition: block_size % 16 == 0 and HasAvx512Vnni() (VNNI implies AVX512BW).
void DotInt8VnniTile16Int4Avx512(float* out16, const uint8_t* au8_row, const float* ascale,
                                 const uint8_t* wq4p, const float* wscale_packed,
                                 const int32_t* wsum_packed, int64_t k_blocks, int64_t block_size);

// COMPACT-SIDE-DATA variant of the int4 tile dot (decode bandwidth fix, stage 2).
// Identical weight handling and math as DotInt8VnniTile16Int4Avx512, but the two
// per-(block,col) side arrays are HALVED:
//   * scales stored as fp16 (uint16_t), expanded with vcvtph2ps in-register. This
//     matches MLAS's native fp16 scale footprint (1.7 MB vs 3.4 MB for a Phi-3 qkv
//     projection). For an fp16-source model it is lossless; for an fp32-source
//     model it costs one fp16 round on the per-block scale (~5e-4 relative).
//   * weight sums stored as int16, sign-extended with vpmovsxwd in-register. This
//     is LOSSLESS for the no-zp path: (q - default_zp) in [-8, 7], so a block sum
//     is in [-8*block_size, 7*block_size], which fits int16 for block_size <= 4096.
// Together they cut the resident side data from 8 to 4 bytes per (block, col),
// roughly a third of the int4 tile's total decode bandwidth.
//
//   wscale_h  : per-(block,col) fp16 scale, layout [block][16], length k_blocks*16
//   wsum16    : per-(block,col) int16 sum of (q - zp), layout [block][16], length k_blocks*16
//
// Precondition: block_size % 16 == 0, block_size <= 4096, and HasAvx512Vnni()
// (VNNI implies AVX512F for vcvtph2ps / vpmovsxwd).
void DotInt8VnniTile16Int4CompactAvx512(float* out16, const uint8_t* au8_row, const float* ascale,
                                        const uint8_t* wq4p, const uint16_t* wscale_h,
                                        const int16_t* wsum16, int64_t k_blocks, int64_t block_size);

// Largest M-row block the M-blocked compact tile kernel handles per call. Chosen
// to keep MB accumulators + MB running fp32 outputs in AVX-512 registers without
// spilling (MB int32 acc + MB fp32 yacc + weight strip + constants).
constexpr int kCompactMBMax = 4;

// M-BLOCKED compact int4 tile dot (prefill steady-state win). Computes the tile
// outputs for up to kCompactMBMax activation rows at once, expanding each weight
// strip (nibble -> int8) EXACTLY ONCE and reusing it across all rows -- instead of
// re-reading + re-expanding the whole weight tile per row as the single-row kernel
// does. The weight read + nibble-expand is the dominant prefill cost, so amortizing
// it across an M-block closes the steady-state GEMM gap vs MLAS. Per-row math and
// float accumulation order are IDENTICAL to DotInt8VnniTile16Int4CompactAvx512, so
// each row's output is BIT-IDENTICAL to calling the single-row kernel.
//
//   out               : mrows * 16 fp32 outputs; row r's 16 values at out + r*16
//   au8_base          : row 0 activations (uint8 = int8 + 128), length k_blocks*block_size
//   au8_row_stride    : element stride between activation rows (typically K)
//   ascale_base       : row 0 per-block activation scales (fp32), length k_blocks
//   ascale_row_stride : element stride between activation-scale rows (typically k_blocks) 
//   wq4p/wscale_h/wsum16 : the compact tile + fp16 scale + int16 wsum (as the single-row kernel)
//   mrows             : number of valid rows, 1..kCompactMBMax
//
// Precondition: block_size % 16 == 0, block_size <= 4096, and HasAvx512Vnni().
void DotInt8VnniTile16Int4CompactMBAvx512(float* out, const uint8_t* au8_base, int64_t au8_row_stride,
                                          const float* ascale_base, int64_t ascale_row_stride,
                                          const uint8_t* wq4p, const uint16_t* wscale_h,
                                          const int16_t* wsum16, int mrows,
                                          int64_t k_blocks, int64_t block_size);

// fp16 (uint16_t) KV-cache dot / accumulate for the GQA attention kernel. Both
// widen fp16 -> fp32 in-register with F16C (vcvtph2ps, implied by any AVX-512
// machine) so the KV cache is streamed from DRAM at HALF the bytes of an fp32
// cache while the attention math stays fp32 (no accuracy loss vs an fp32 cache
// beyond the one-time fp16 store rounding). fp16 -> fp32 is exact, so the
// scalar tail (HalfToFloat) is bit-identical to the vectorized body.
//
// GqaDotF16Avx512:   returns sum_d q[d] * f16_to_f32(k[d]) for d in [0, hs).
// GqaAccumVF16Avx512: acc[d] += w * f16_to_f32(v[d]) for d in [0, hs).
// Precondition for both: HasAvx512f().
float GqaDotF16Avx512(const float* q, const uint16_t* k, int64_t hs);
void GqaAccumVF16Avx512(float* acc, float w, const uint16_t* v, int64_t hs);

}  // namespace fusion_cpu_ep
