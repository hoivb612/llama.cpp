// flash_attn_pf.hlsli - prefill-oriented tiled Flash Attention.
//
// flash_attn_tiled.hlsl is shaped for the decode-adjacent case: 128 threads, a
// runtime head dim, and a PV pass that only keeps D_v of those threads busy
// while extracting V lanes through a dynamic float4 component index. At
// prefill sizes that leaves most of the machine idle - measured at 1.5 TFLOPS
// on a 91 TFLOPS part, and 94% of prompt time.
//
// This variant specializes on HEAD_DIM so every loop bound is a constant, runs
// 256 threads, and keeps both the QK and PV passes at full occupancy. Without
// NATIVE_FP16 K and V share one f32 LDS tile, so the per-tile cost is one extra
// barrier rather than a third of the LDS budget. Under NATIVE_FP16 K and V get
// separate half tiles, which both stages them in one pass (dropping that extra
// barrier) and costs less LDS than the single f32 tile did; the QK inner loop
// then folds into dot2add. Scores, PV accumulators and the softmax state stay
// f32 throughout.
//
// Thread mapping (D=64, BR=16, BC=64, 256 threads):
//   QK  idx = tid + j*256 -> r = idx/BC, c = idx%BC   (4 scores/thread)
//   PV  d   = tid % D, r_j = tid/D + j*(256/D)        (4 accumulators/thread)
// Both keep the fast-varying index in the low bits, so LDS reads across a wave
// are consecutive and conflict-free.

#include "ggml_common.hlsli"

#ifndef HEAD_DIM
#error "flash_attn_pf.hlsli requires HEAD_DIM"
#endif

#define FA_D HEAD_DIM

#ifndef FA_PF_BR
#define FA_PF_BR 8
#endif

#ifndef FA_PF_BC
#if HEAD_DIM <= 64
#define FA_PF_BC 64
#else
#define FA_PF_BC 32
#endif
#endif

#ifndef FA_PF_THREADS
#define FA_PF_THREADS 256
#endif

// One float of padding rotates the bank across rows, so a wave reading
// s_kv[c][d] with c varying hits FA_D+1 strides instead of a single bank.
#define FA_PF_STRIDE (FA_D + 1)
#define FA_PF_DVEC   (FA_D / 4)

// QK-only fp16: Q and K are staged as half4 and the inner loop folds into
// dot2add, so the dot product runs at half rate with an f32 accumulator. V is
// staged as half too, which is what makes the fp16 path a net LDS *saving*
// rather than an addition: at D=128 the f32 V tile alone was 16512 B of a
// 32768 B budget, capping residency at one group. Scores, the online-softmax
// state and every accumulator stay f32.
#if defined(NATIVE_FP16)
typedef float16_t4 fa_pf_vec4;
// Same padding rationale as FA_PF_STRIDE: an 8-byte element makes the bank
// step 2*(FA_PF_DVEC+1), so one half4 of padding rotates c off a single bank.
#define FA_PF_HSTRIDE (FA_PF_DVEC + 1)
// V is read one scalar per thread in the PV pass, so it is stored flat rather
// than as half4 to keep that a plain index. Two halves of padding rotate the
// bank the same way the f32 tile's single float does.
#define FA_PF_VSTRIDE (FA_D + 2)

float fa_pf_dot4(fa_pf_vec4 q, fa_pf_vec4 k, float acc) {
    float a = dot2add(q.xy, k.xy, acc);
    return dot2add(q.zw, k.zw, a);
}
#endif

// The QK scores and the PV accumulators are the only carriers of the online
// softmax rescale, so they are `precise` by default: reassociation there moves
// the rescale relative to the running max. FA_PF_RELAXED_ACC lifts that on
// those two arrays only, to measure what the qualifier costs.
#if defined(FA_PF_RELAXED_ACC)
#define FA_PF_PRECISE
#else
#define FA_PF_PRECISE precise
#endif

// Mask tile classification, layered on the prescan. The prescan already reads
// this group's whole mask rectangle, so the same reads can also record two bits
// per KV tile and let the tile loop drop work the bounds alone cannot:
//   FA_PF_MC_FINITE  some element of the tile is finite
//   FA_PF_MC_LOAD    some element is inf, or is a finite non-zero bias
// A tile with no finite element is fully masked and is skipped without staging
// K/V. A tile that is finite everywhere with no bias to add (FINITE only) needs
// no mask read at all in the QK pass: the bias is exactly zero for every row,
// and slope * 0 is zero for any ALiBi slope. Everything else keeps the normal
// per-score load, so a tile that mixes zeros with -inf - the causal diagonal -
// is still read element by element. Bits are only ever set, so 0 (a tile no
// thread scanned, unreachable while every tile lies in some scanned row) also
// falls back to the load.
#if defined(FA_PF_MASK_CLASS)
#if !defined(FA_PF_PRESCAN)
#error "FA_PF_MASK_CLASS requires FA_PF_PRESCAN"
#endif
#if (FA_PF_BC % 4) != 0
#error "FA_PF_MASK_CLASS assumes a float4 mask chunk stays inside one KV tile"
#endif
#define FA_PF_MC_FINITE 1u
#define FA_PF_MC_LOAD   2u
// 16 tiles per uint. 128 words = 512 B covers 2048 tiles, i.e. 65536 KV
// elements at FA_PF_BC=32. Larger launches keep the prescan but disable tile
// classification.
#define FA_PF_MCLASS_PER_WORD 16u
#define FA_PF_MCLASS_WORDS    128u
#define FA_PF_MCLASS_TILES    (FA_PF_MCLASS_WORDS * FA_PF_MCLASS_PER_WORD)
#define FA_PF_MCLASS_MAX_KV   (FA_PF_MCLASS_TILES * FA_PF_BC)
#endif

// QK: rows covered per thread, and the row group it starts from.
#define FA_PF_CGROUPS (FA_PF_THREADS / FA_PF_BC)
#define FA_PF_QK_PER  (FA_PF_BR / FA_PF_CGROUPS)

// PV: D-sized groups that fit in the threadgroup. When FA_D does not divide
// FA_PF_THREADS the tail threads sit out the PV pass (D=96: 192 of 256 active).
#define FA_PF_DG      (FA_PF_THREADS / FA_D)
#define FA_PF_ACC     (FA_PF_BR / FA_PF_DG)

// Softmax reductions: FA_PF_RSPLIT threads cooperate on each query row.
#define FA_PF_RSPLIT  (FA_PF_THREADS / FA_PF_BR)

#if (FA_PF_THREADS % FA_PF_BC) != 0
#error "FA_PF_THREADS must be a multiple of FA_PF_BC"
#endif
#if (FA_PF_BR % FA_PF_CGROUPS) != 0
#error "FA_PF_BR must be a multiple of FA_PF_CGROUPS"
#endif
#if (FA_PF_BR % FA_PF_DG) != 0
#error "FA_PF_BR must be a multiple of FA_PF_DG"
#endif
#if (FA_PF_THREADS % FA_PF_BR) != 0
#error "FA_PF_THREADS must be a multiple of FA_PF_BR"
#endif

// LDS budget, bytes (must stay under the 32768 Intel/D3D12 threadgroup limit).
// fp16 stages Q, K and V as half, so it both shrinks Q and replaces the f32 V
// tile. Residency matters as much as the limit: at 32516 B only one group fits
// a 64 KB SLM, and D=128 was sitting there.
//   f32:  s_q + s_kv + s_scores + s_red + scalars
//   fp16: s_qh + s_kh + s_vh + s_scores + s_red + scalars
//   D=64  BR=8  BC=64: 1088 +  8704 +  8448 + 2048 + 1024 + 132 = 21444
//   D=64  BR=32 BC=32: 4352 +  4352 +  4224 + 4096 + 1024 + 516 = 18564
//   D=96  BR=16 BC=32: 3200 +  6400 +  6272 + 2048 + 1024 + 260 = 19204
//   D=128 BR=16 BC=32: 4224 +  8448 +  8320 + 2048 + 1024 + 260 = 24324
// FA_PF_MASK_CLASS adds 512 B of tile classes.
#if defined(NATIVE_FP16)
groupshared fa_pf_vec4 s_qh[FA_PF_BR][FA_PF_HSTRIDE];
groupshared fa_pf_vec4 s_kh[FA_PF_BC][FA_PF_HSTRIDE];
groupshared float16_t  s_vh[FA_PF_BC][FA_PF_VSTRIDE];
#else
groupshared float s_q[FA_PF_BR][FA_PF_STRIDE];
groupshared float s_kv[FA_PF_BC][FA_PF_STRIDE];
#endif
groupshared float s_scores[FA_PF_BR][FA_PF_BC];
groupshared float s_red[FA_PF_BR][FA_PF_RSPLIT];
groupshared float s_max[FA_PF_BR];
groupshared float s_sum[FA_PF_BR];
groupshared float s_corr[FA_PF_BR];
groupshared uint  s_active[FA_PF_BR];
#if defined(FA_PF_PRESCAN)
groupshared uint  s_kv_lo;
groupshared uint  s_kv_hi;
#else
groupshared uint  s_tile_any;
#endif
#if defined(FA_PF_MASK_CLASS)
groupshared uint  s_mclass[FA_PF_MCLASS_WORDS];

uint fa_pf_mclass_bits(float m) {
    if (isinf(m)) {
        return FA_PF_MC_LOAD;
    }
    return (m != 0.0f) ? (FA_PF_MC_FINITE | FA_PF_MC_LOAD) : FA_PF_MC_FINITE;
}

// One atomic per word rather than per element: bits only ever get set, so a
// word already holding them needs no store at all.
void fa_pf_mclass_flush(uint wi, uint word) {
    if (word != 0u && (s_mclass[wi] & word) != word) {
        InterlockedOr(s_mclass[wi], word);
    }
}

// Tiles arrive in non-decreasing order per thread, so keeping one word live in
// a register costs one flush per 16 tiles.
void fa_pf_mclass_add(uint tile, uint bits, inout uint wi, inout uint word) {
    uint w = tile / FA_PF_MCLASS_PER_WORD;
    if (w != wi) {
        fa_pf_mclass_flush(wi, word);
        wi   = w;
        word = 0u;
    }
    word |= bits << ((tile % FA_PF_MCLASS_PER_WORD) * 2u);
}
#endif

float4 fa_pf_load4(ByteAddressBuffer buf, uint byte_offset, uint elem_size) {
    if (elem_size == 4) {
        return asfloat(buf.Load4(byte_offset));
    }

    // Half rows/views may begin at a 2-byte offset. Reconstruct two packed
    // words from aligned loads instead of issuing an undefined misaligned Load2.
    // The third word is only meaningful for a 2-byte-shifted start, but it is
    // addressed unconditionally: a load guarded by `shift != 0` can still be
    // speculated by the compiler, and these buffers are bound as root SRVs,
    // which D3D12 does not bounds check. Reading aligned+8 for an already
    // aligned offset walks off the end of the last tensor in an allocation
    // (the final layer's V cache) and faults the device.
    uint aligned = byte_offset & ~3u;
    uint shift = (byte_offset & 2u) * 8u;
    uint w0 = buf.Load(aligned);
    uint w1 = buf.Load(aligned + 4u);
    uint w2 = buf.Load(aligned + (shift == 0u ? 4u : 8u));
    uint packed0 = shift == 0u ? w0 : ((w0 >> 16) | (w1 << 16));
    uint packed1 = shift == 0u ? w1 : ((w1 >> 16) | (w2 << 16));
    uint v0 =  packed0        & 0xFFFFu;
    uint v1 = (packed0 >> 16) & 0xFFFFu;
    uint v2 =  packed1        & 0xFFFFu;
    uint v3 = (packed1 >> 16) & 0xFFFFu;

    if (elem_size == 3) {
        return float4(
            asfloat(v0 << 16),
            asfloat(v1 << 16),
            asfloat(v2 << 16),
            asfloat(v3 << 16));
    }

    return float4(
        f16_to_f32(v0),
        f16_to_f32(v1),
        f16_to_f32(v2),
        f16_to_f32(v3));
}

#define FA_PF_NEG_MAX (-3.402823466e+38f)

// Online-softmax running-max update for one query row. Called by whichever
// single thread finished that row's max fold.
void fa_pf_fold_max(uint r, float tile_max, uint q_start, uint N_queries) {
    uint active = (q_start + r < N_queries && tile_max != FA_PF_NEG_MAX) ? 1u : 0u;
    s_active[r] = active;
    if (active != 0u) {
        float old_max = s_max[r];
        float new_max = max(old_max, tile_max);
        float correction = (s_sum[r] > 0.0f) ? exp(old_max - new_max) : 0.0f;
        s_max[r] = new_max;
        s_sum[r] *= correction;
        s_corr[r] = correction;
    } else {
        s_corr[r] = 1.0f;
    }
}

WAVE_SIZE_ATTR
[numthreads(FA_PF_THREADS, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    const uint tid       = gtid.x;
    // Under a causal mask the KV range a group covers grows with q_start, so
    // the last group does the most work while being dispatched last, leaving a
    // long tail. Walking the groups backwards launches the heavy ones first.
    const uint n_qgroups = (ne01 + FA_PF_BR - 1u) / FA_PF_BR;
    const uint q_start   = (n_qgroups - 1u - gid.x) * FA_PF_BR;
    const uint head_idx  = gid.y;
    const uint batch_idx = gid.z;

    const uint N_queries  = ne01;
    const uint N_kv       = ne11;
    const uint n_heads    = ne02;
    const uint n_kv_heads = ne12;
    const uint kv_head    = head_idx * n_kv_heads / n_heads;

    if (q_start >= N_queries) {
        return;
    }

    const uint src2_off = op0;
    const uint src2_nb0 = op1;
    const uint src2_nb1 = op2;
    const uint src2_nb2 = op3;
    const uint src2_nb3 = op4;
    const uint src2_es  = op5 & 0xFFu;

    const uint mask_info = op8;
    const uint has_mask  = mask_info & 1u;
    const uint has_sinks = (mask_info >> 24) & 1u;
    const uint mask_nb0  = (mask_info >> 8) & 0xFFu;
    const uint mask_es   = (mask_info >> 16) & 0xFFu;
    const uint mask_off  = op9;
    const uint mask_nb1  = op10;
    const uint mask_nb2  = op11;
    const uint mask_nb3  = op12;
    const uint mask_ne2  = op13 & 0xFFFFu;
    const uint mask_ne3  = (op13 >> 16) & 0xFFFFu;

    const float scale         = asfloat(op6);
    const float logit_softcap = asfloat(op7);
    const float max_bias      = asfloat(op14);
    const float neg_max       = -3.402823466e+38f;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        uint n_head_log2 = (n_heads > 0u) ? (1u << firstbithigh(n_heads)) : 1u;
        float n_head_log2_f = (float)n_head_log2;
        float m0 = exp2(-max_bias * 0.5f / n_head_log2_f * 2.0f);
        float m1 = exp2(-max_bias * 0.5f / n_head_log2_f);
        if (head_idx < n_head_log2) {
            slope = pow(m0, (float)(head_idx + 1u));
        } else {
            slope = pow(m1, (float)(2u * (head_idx - n_head_log2) + 1u));
        }
    }

    // Stage the query rows once for all KV tiles.
    for (uint qidx = tid; qidx < FA_PF_BR * FA_PF_DVEC; qidx += FA_PF_THREADS) {
        uint r  = qidx / FA_PF_DVEC;
        uint dv = qidx % FA_PF_DVEC;
        uint query_idx = q_start + r;
        float4 qv = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (query_idx < N_queries) {
            uint q_base = src0_offset + query_idx * nb01 + head_idx * nb02 + batch_idx * nb03;
            qv = asfloat(src0.Load4(q_base + dv * 16u));
        }
#if defined(NATIVE_FP16)
        s_qh[r][dv] = (fa_pf_vec4)qv;
#else
        s_q[r][dv * 4u + 0u] = qv.x;
        s_q[r][dv * 4u + 1u] = qv.y;
        s_q[r][dv * 4u + 2u] = qv.z;
        s_q[r][dv * 4u + 3u] = qv.w;
#endif
    }

    if (tid < FA_PF_BR) {
        s_max[tid] = neg_max;
        s_sum[tid] = 0.0f;
    }
#if defined(FA_PF_PRESCAN)
    if (tid == 0) {
        s_kv_lo = N_kv;
        s_kv_hi = 0u;
    }
#endif
#if defined(FA_PF_MASK_CLASS)
    // A stale dispatch can outgrow the class table (the replay path rebinds a
    // baked flash-attention dispatch when only N_kv changed), so the table is
    // opt-in per launch and the tile loop keeps the normal mask-load behaviour without
    // it. Zeroing only the words the tile count uses rides the barrier below.
    const uint n_tiles   = (N_kv + FA_PF_BC - 1u) / FA_PF_BC;
    const bool use_class = has_mask != 0u && n_tiles <= FA_PF_MCLASS_TILES;
    if (use_class) {
        for (uint mw = tid; mw < (n_tiles + FA_PF_MCLASS_PER_WORD - 1u) / FA_PF_MCLASS_PER_WORD;
             mw += FA_PF_THREADS) {
            s_mclass[mw] = 0u;
        }
    }
#endif

    // QK ownership: FA_PF_QK_PER (r, c) pairs, c fast-varying.
    const uint qk_c  = tid % FA_PF_BC;
    const uint qk_r0 = tid / FA_PF_BC;

    // PV ownership: one output dim, FA_PF_ACC query rows.
    const uint pv_d    = tid % FA_D;
    const uint pv_r0   = tid / FA_D;
    const bool pv_live = (tid < FA_PF_DG * FA_D);

    // Softmax reduction ownership.
    const uint red_r = tid / FA_PF_RSPLIT;
    const uint red_s = tid % FA_PF_RSPLIT;

    FA_PF_PRECISE float acc[FA_PF_ACC];
    [unroll]
    for (uint ai = 0; ai < FA_PF_ACC; ++ai) {
        acc[ai] = 0.0f;
    }
#if defined(FA_PF_PRESCAN)
    uint mhb = 0u;
    if (has_mask != 0u) {
        mhb = (head_idx % mask_ne2) * mask_nb2
            + (batch_idx % mask_ne3) * mask_nb3;
    }
#endif
    GroupMemoryBarrierWithGroupSync();

#if defined(FA_PF_PRESCAN)
    uint kv_first = 0u;
    uint kv_end   = N_kv;

    // Mask prescan. One pass over this group's whole mask rectangle - the valid
    // query rows by [0, N_kv) - reduced to the union of the first and last KV
    // column that is finite on any of those rows. Every tile outside that range
    // is fully masked for every row this group owns, which is exactly the set
    // the per-tile scan below would have skipped, so the tile loop can start
    // and stop there instead of paying two barriers and an LDS round trip per
    // tile. Interior all-masked tiles stay in range and are simply computed.
    if (has_mask != 0u) {
        uint loc_lo = N_kv;
        uint loc_hi = 0u;
#if defined(FA_PF_MASK_CLASS)
        uint cls_wi   = 0u;
        uint cls_word = 0u;
#endif
        if (q_start + red_r < N_queries) {
            uint mrow = mask_off + (q_start + red_r) * mask_nb1 + mhb;
            // float4 needs the row contiguous at its element size. BF16 reports
            // es=3 against a 2-byte stride, so the equality also excludes it.
            if (mask_nb0 == mask_es && (mask_es == 2u || mask_es == 4u)) {
                uint nvec = N_kv / 4u;
                for (uint v = red_s; v < nvec; v += FA_PF_RSPLIT) {
                    uint c0 = v * 4u;
                    float4 m4 = fa_pf_load4(src3, mrow + c0 * mask_nb0, mask_es);
                    if (!isinf(m4.x)) { loc_lo = min(loc_lo, c0);      loc_hi = max(loc_hi, c0);      }
                    if (!isinf(m4.y)) { loc_lo = min(loc_lo, c0 + 1u); loc_hi = max(loc_hi, c0 + 1u); }
                    if (!isinf(m4.z)) { loc_lo = min(loc_lo, c0 + 2u); loc_hi = max(loc_hi, c0 + 2u); }
                    if (!isinf(m4.w)) { loc_lo = min(loc_lo, c0 + 3u); loc_hi = max(loc_hi, c0 + 3u); }
#if defined(FA_PF_MASK_CLASS)
                    if (use_class) {
                        fa_pf_mclass_add(c0 / FA_PF_BC,
                            fa_pf_mclass_bits(m4.x) | fa_pf_mclass_bits(m4.y) |
                            fa_pf_mclass_bits(m4.z) | fa_pf_mclass_bits(m4.w),
                            cls_wi, cls_word);
                    }
#endif
                }
                for (uint ct = nvec * 4u + red_s; ct < N_kv; ct += FA_PF_RSPLIT) {
                    float mt = load_auto(src3, mrow + ct * mask_nb0, mask_es);
                    if (!isinf(mt)) { loc_lo = min(loc_lo, ct); loc_hi = max(loc_hi, ct); }
#if defined(FA_PF_MASK_CLASS)
                    if (use_class) {
                        fa_pf_mclass_add(ct / FA_PF_BC, fa_pf_mclass_bits(mt), cls_wi, cls_word);
                    }
#endif
                }
            } else {
                for (uint cs = red_s; cs < N_kv; cs += FA_PF_RSPLIT) {
                    float ms = load_auto(src3, mrow + cs * mask_nb0, mask_es);
                    if (!isinf(ms)) { loc_lo = min(loc_lo, cs); loc_hi = max(loc_hi, cs); }
#if defined(FA_PF_MASK_CLASS)
                    if (use_class) {
                        fa_pf_mclass_add(cs / FA_PF_BC, fa_pf_mclass_bits(ms), cls_wi, cls_word);
                    }
#endif
                }
            }
        }
#if defined(FA_PF_MASK_CLASS)
        fa_pf_mclass_flush(cls_wi, cls_word);
#endif
        if (loc_lo < N_kv) {
            InterlockedMin(s_kv_lo, loc_lo);
            InterlockedMax(s_kv_hi, loc_hi);
        }
        GroupMemoryBarrierWithGroupSync();

        if (s_kv_lo > s_kv_hi) {
            kv_end = 0u;
        } else {
            kv_first = (s_kv_lo / FA_PF_BC) * FA_PF_BC;
            kv_end   = min((s_kv_hi / FA_PF_BC + 1u) * FA_PF_BC, N_kv);
        }
    }
#else
    const uint kv_first = 0u;
    const uint kv_end   = N_kv;
#endif

    for (uint tile_start = kv_first; tile_start < kv_end; tile_start += FA_PF_BC) {
        uint tile_size = min((uint)FA_PF_BC, N_kv - tile_start);

#if defined(FA_PF_MASK_CLASS)
        // Group-uniform: tile_start and the class word are the same on every
        // thread, so the skip below leaves the barriers in uniform flow. Only
        // the two positive classes act; anything else keeps the normal path.
        uint tcls = FA_PF_MC_FINITE | FA_PF_MC_LOAD;
        if (use_class) {
            uint ti = tile_start / FA_PF_BC;
            tcls = (s_mclass[ti / FA_PF_MCLASS_PER_WORD] >> ((ti % FA_PF_MCLASS_PER_WORD) * 2u)) & 3u;
            if (tcls == FA_PF_MC_LOAD) {
                continue;
            }
        }
#endif

#if !defined(FA_PF_PRESCAN)
        // A causal or sliding-window mask leaves whole KV tiles fully masked.
        // Scanning the mask first is cheaper than staging K/V and running QK,
        // softmax and PV only to discard them. Skipping is exact: a fully
        // masked tile leaves acc and the online-softmax state untouched.
        if (has_mask != 0u) {
            if (tid == 0) {
                s_tile_any = 0u;
            }
            GroupMemoryBarrierWithGroupSync();

            uint any_local = 0u;
            uint mhb = (head_idx % mask_ne2) * mask_nb2
                     + (batch_idx % mask_ne3) * mask_nb3;
            [unroll]
            for (uint mj = 0; mj < FA_PF_QK_PER; ++mj) {
                uint mr = qk_r0 + mj * FA_PF_CGROUPS;
                uint mq = q_start + mr;
                float mv = 0.0f;
                if (mq < N_queries && qk_c < tile_size) {
                    mv = load_auto(
                        src3, mask_off + mq * mask_nb1 + mhb + (tile_start + qk_c) * mask_nb0,
                        mask_es);
                    if (!isinf(mv)) {
                        any_local = 1u;
                    }
                }
                s_scores[mr][qk_c] = mv;
            }
            if (any_local != 0u) {
                InterlockedOr(s_tile_any, 1u);
            }
            GroupMemoryBarrierWithGroupSync();

            if (s_tile_any == 0u) {
                continue;
            }
        }
#endif

        // Stage K, shared by every query row. The fp16 path has a separate K
        // tile, so stage V at the same time and share one synchronization.
        for (uint kidx = tid; kidx < FA_PF_BC * FA_PF_DVEC; kidx += FA_PF_THREADS) {
            uint c  = kidx / FA_PF_DVEC;
            uint dv = kidx % FA_PF_DVEC;
            float4 kval = float4(0.0f, 0.0f, 0.0f, 0.0f);
#if defined(NATIVE_FP16)
            float4 vval = float4(0.0f, 0.0f, 0.0f, 0.0f);
#endif
            if (c < tile_size) {
                uint k_base = src1_offset + (tile_start + c) * nb11
                            + kv_head * nb12 + batch_idx * nb13;
                kval = fa_pf_load4(src1, k_base + dv * 4u * nb10, src1_esize);
#if defined(NATIVE_FP16)
                uint v_base = src2_off + (tile_start + c) * src2_nb1
                            + kv_head * src2_nb2 + batch_idx * src2_nb3;
                vval = fa_pf_load4(src2, v_base + dv * 4u * src2_nb0, src2_es);
#endif
            }
#if defined(NATIVE_FP16)
            s_kh[c][dv] = (fa_pf_vec4)kval;
            s_vh[c][dv * 4u + 0u] = (float16_t)vval.x;
            s_vh[c][dv * 4u + 1u] = (float16_t)vval.y;
            s_vh[c][dv * 4u + 2u] = (float16_t)vval.z;
            s_vh[c][dv * 4u + 3u] = (float16_t)vval.w;
#else
            s_kv[c][dv * 4u + 0u] = kval.x;
            s_kv[c][dv * 4u + 1u] = kval.y;
            s_kv[c][dv * 4u + 2u] = kval.z;
            s_kv[c][dv * 4u + 3u] = kval.w;
#endif
        }
        GroupMemoryBarrierWithGroupSync();

        // QK. Every thread computes FA_PF_QK_PER full-length dot products.
        // The K element is the loop-carried value so it is fetched once per d
        // and shared across the rows; s_q reads are wave-uniform broadcasts.
        FA_PF_PRECISE float sc[FA_PF_QK_PER];
        float mask_local[FA_PF_QK_PER];
        uint  live_local[FA_PF_QK_PER];
        [unroll]
        for (uint j0 = 0; j0 < FA_PF_QK_PER; ++j0) {
            uint r = qk_r0 + j0 * FA_PF_CGROUPS;
            sc[j0] = 0.0f;
            mask_local[j0] = 0.0f;
            live_local[j0] = (q_start + r < N_queries && qk_c < tile_size) ? 1u : 0u;
            if (has_mask != 0u && live_local[j0] != 0u) {
#if defined(FA_PF_MASK_CLASS)
                // Class FINITE only: every element of this tile is finite and
                // exactly zero, so the bias it would add is zero for every row
                // and the read is dropped. Any other class reads as before.
                if (tcls != FA_PF_MC_FINITE) {
                    mask_local[j0] = load_auto(
                        src3,
                        mask_off + (q_start + r) * mask_nb1 + mhb + (tile_start + qk_c) * mask_nb0,
                        mask_es) * slope;
                }
#elif defined(FA_PF_PRESCAN)
                // The prescan only bounded the tile range, so each QK owner
                // reads its own mask element here instead of via s_scores.
                mask_local[j0] = load_auto(
                    src3,
                    mask_off + (q_start + r) * mask_nb1 + mhb + (tile_start + qk_c) * mask_nb0,
                    mask_es) * slope;
#else
                mask_local[j0] = s_scores[r][qk_c] * slope;
#endif
                if (isinf(mask_local[j0])) {
                    live_local[j0] = 0u;
                }
            }
        }

#if defined(NATIVE_FP16)
        for (uint dv2 = 0; dv2 < FA_PF_DVEC; ++dv2) {
            fa_pf_vec4 kvec = s_kh[qk_c][dv2];
            [unroll]
            for (uint j1 = 0; j1 < FA_PF_QK_PER; ++j1) {
                sc[j1] = fa_pf_dot4(s_qh[qk_r0 + j1 * FA_PF_CGROUPS][dv2], kvec, sc[j1]);
            }
        }
#else
        for (uint d = 0; d < FA_D; ++d) {
            float kval = s_kv[qk_c][d];
            [unroll]
            for (uint j1 = 0; j1 < FA_PF_QK_PER; ++j1) {
                sc[j1] += s_q[qk_r0 + j1 * FA_PF_CGROUPS][d] * kval;
            }
        }
#endif

        [unroll]
        for (uint j = 0; j < FA_PF_QK_PER; ++j) {
            uint r = qk_r0 + j * FA_PF_CGROUPS;
            float dp = neg_max;
            if (live_local[j] != 0u) {
                dp = sc[j] * scale;
                if (logit_softcap != 0.0f) {
                    dp = logit_softcap * tanh(dp);
                }
                dp += mask_local[j];
            }
            sc[j] = dp;
            s_scores[r][qk_c] = dp;
        }
        GroupMemoryBarrierWithGroupSync();

        // Row max: FA_PF_RSPLIT threads per row, then a short serial fold.
        // Collapsing this to one pass (each of FA_PF_BR threads scanning all
        // FA_PF_BC columns) saves a barrier but was measured slower even on
        // Intel UHD, the most barrier-sensitive part here: 494 vs 522 t/s at
        // pp6144. Folding via WaveActiveMax/WaveActiveSum instead, which
        // removes the same two barriers with no redundant work, was a wash on
        // both parts (NVIDIA 13857 vs 13777-14191 run-to-run, Intel 522.7 vs
        // 522.4), so these reductions are not the limiter.
        {
            float m = FA_PF_NEG_MAX;
            for (uint c = red_s; c < FA_PF_BC; c += FA_PF_RSPLIT) {
                m = max(m, s_scores[red_r][c]);
            }
            s_red[red_r][red_s] = m;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_PF_BR) {
            float tile_max = FA_PF_NEG_MAX;
            [unroll]
            for (uint k = 0; k < FA_PF_RSPLIT; ++k) {
                tile_max = max(tile_max, s_red[tid][k]);
            }
            fa_pf_fold_max(tid, tile_max, q_start, N_queries);
        }
        GroupMemoryBarrierWithGroupSync();

        // Exponentiate in place, reusing the scores still held in registers.
        [unroll]
        for (uint j2 = 0; j2 < FA_PF_QK_PER; ++j2) {
            uint r = qk_r0 + j2 * FA_PF_CGROUPS;
            float p = 0.0f;
            if (s_active[r] != 0u && sc[j2] != neg_max && qk_c < tile_size) {
#if defined(NATIVE_FP16)
                p = (float)exp((float16_t)(sc[j2] - s_max[r]));
#else
                p = exp(sc[j2] - s_max[r]);
#endif
            }
            s_scores[r][qk_c] = p;
        }
        if (pv_live) {
            [unroll]
            for (uint ci = 0; ci < FA_PF_ACC; ++ci) {
                acc[ci] *= s_corr[pv_r0 + ci * FA_PF_DG];
            }
        }
        GroupMemoryBarrierWithGroupSync();

        {
            float s = 0.0f;
            for (uint c = red_s; c < FA_PF_BC; c += FA_PF_RSPLIT) {
                s += s_scores[red_r][c];
            }
            s_red[red_r][red_s] = s;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_PF_BR && s_active[tid] != 0u) {
            float tile_sum = 0.0f;
            [unroll]
            for (uint k2 = 0; k2 < FA_PF_RSPLIT; ++k2) {
                tile_sum += s_red[tid][k2];
            }
            s_sum[tid] += tile_sum;
        }

#if !defined(NATIVE_FP16)
        // Reuse the K tile slot for V. The K reads finished at the QK barrier,
        // so no extra sync is needed before overwriting it.
        for (uint vidx = tid; vidx < FA_PF_BC * FA_PF_DVEC; vidx += FA_PF_THREADS) {
            uint c  = vidx / FA_PF_DVEC;
            uint dv = vidx % FA_PF_DVEC;
            float4 vval = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (c < tile_size) {
                uint v_base = src2_off + (tile_start + c) * src2_nb1
                            + kv_head * src2_nb2 + batch_idx * src2_nb3;
                vval = fa_pf_load4(src2, v_base + dv * 4u * src2_nb0, src2_es);
            }
            s_kv[c][dv * 4u + 0u] = vval.x;
            s_kv[c][dv * 4u + 1u] = vval.y;
            s_kv[c][dv * 4u + 2u] = vval.z;
            s_kv[c][dv * 4u + 3u] = vval.w;
        }
        GroupMemoryBarrierWithGroupSync();
#endif

        // PV. One LDS V read feeds FA_PF_ACC accumulators.
        if (pv_live) {
            for (uint vc = 0; vc < tile_size; ++vc) {
#if defined(NATIVE_FP16)
                float vval = (float)s_vh[vc][pv_d];
#else
                float vval = s_kv[vc][pv_d];
#endif
                [unroll]
                for (uint ai2 = 0; ai2 < FA_PF_ACC; ++ai2) {
                    acc[ai2] += s_scores[pv_r0 + ai2 * FA_PF_DG][vc] * vval;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (has_sinks != 0u) {
        if (tid < FA_PF_BR && q_start + tid < N_queries) {
            float sink_score = asfloat(src4.Load(head_idx * 4u));
            float old_max = s_max[tid];
            float new_max = max(old_max, sink_score);
            float correction = (s_sum[tid] > 0.0f) ? exp(old_max - new_max) : 0.0f;
            float sink_weight = exp(sink_score - new_max);
            s_max[tid] = new_max;
            s_sum[tid] = s_sum[tid] * correction + sink_weight;
            s_corr[tid] = correction;
        }
        GroupMemoryBarrierWithGroupSync();
        if (pv_live) {
            [unroll]
            for (uint ai3 = 0; ai3 < FA_PF_ACC; ++ai3) {
                acc[ai3] *= s_corr[pv_r0 + ai3 * FA_PF_DG];
            }
        }
    }

    if (pv_live) {
        [unroll]
        for (uint ai4 = 0; ai4 < FA_PF_ACC; ++ai4) {
            uint r = pv_r0 + ai4 * FA_PF_DG;
            uint query_idx = q_start + r;
            if (query_idx < N_queries) {
                float inv_sum = s_sum[r] > 0.0f ? (1.0f / s_sum[r]) : 0.0f;
                uint out_off = dst_offset + pv_d * nb0 + head_idx * nb1
                             + query_idx * nb2 + batch_idx * nb3;
                store_auto(dst, out_off, acc[ai4] * inv_sum, dst_esize);
            }
        }
    }
}
