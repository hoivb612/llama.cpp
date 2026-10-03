// Intel wave16: QK owns eight rows per wave; pairs of PV waves split the output width.
#ifndef FA_BR
#define FA_BR 32
#endif
#ifndef FA_BC
#define FA_BC 32
#endif
#ifndef FA_NWAVE
#define FA_NWAVE (FA_BR / 8 * 2)
#endif

#define IW_QKV_STRIDE (FA_D + 8)
#define IW_P_STRIDE (FA_BC + 8)

#if WAVE_SIZE != 16 || (FA_D != 64 && FA_D != 96 && FA_D != 128)
#error "FA_INTEL_WAVE requires wave16 and D=64/96/128"
#endif
#if (FA_BR % 8) != 0 || FA_NWAVE != (FA_BR / 8 * 2) || (FA_BC % 16) != 0
#error "FA_INTEL_WAVE requires two PV waves per eight rows and whole 16-column KV tiles"
#endif
#if (FA_D / 16) % (FA_O_SPLITS * 2) != 0
#error "FA_O_SPLITS and paired PV waves must divide the output D tiles"
#endif
#if (2 * (FA_BR + FA_BC) * IW_QKV_STRIDE + 2 * FA_BR * IW_P_STRIDE + 4 * FA_BR * (FA_BC + 1) + 8 * FA_BR + 12) > 32768
#error "FA_INTEL_WAVE tile exceeds 32 KiB of groupshared storage"
#endif

#define IW_THREADS (FA_NWAVE * WAVE_SIZE)
#define IW_ACC_E 8
#define IW_ODBLK (FA_D / 16 / FA_O_SPLITS / 2)
#define IW_MASK_E (FA_BR * FA_BC / IW_THREADS)

typedef Matrix<ComponentType::F16, 8, 16, MatrixUse::A, MatrixScope::Wave> IWMatA;
typedef Matrix<ComponentType::F16, 16, 16, MatrixUse::B, MatrixScope::Wave> IWMatB;
typedef Matrix<ComponentType::F16, 8, 16, MatrixUse::Accumulator, MatrixScope::Wave> IWMatAcc;

groupshared float16_t iw_q[FA_BR * IW_QKV_STRIDE];
groupshared float16_t iw_kv[FA_BC * IW_QKV_STRIDE];
groupshared float16_t iw_p[FA_BR * IW_P_STRIDE];
// Pad row scans without changing matrix operand alignment.
groupshared float iw_s[FA_BR * (FA_BC + 1)];
groupshared float iw_max[FA_BR];
groupshared float iw_sum[FA_BR];
// Reuse score padding so D64 stays below 16 KiB of LDS.
#define IW_CORR(r) iw_s[(r) * (FA_BC + 1) + FA_BC]
groupshared uint iw_q_unsafe;
groupshared uint iw_kv_unsafe;
groupshared uint iw_any;

uint iw_score_index(uint i) {
    return i + i / FA_BC;
}

uint iw_range_flags(float value, float limit, float scaled_min) {
    // Bits: scaling needed, range overflow, raw subnormal, scaled subnormal.
    const float a = abs(value);
    return (a > 32.0f ? 1u : 0u) |
           (!(a <= limit) ? 2u : 0u) |
           (a > 0.0f && a < 0.00006103515625f ? 4u : 0u) |
           (a > 0.0f && a < scaled_min ? 8u : 0u);
}

bool iw_range_unsafe(uint flags) {
    return (flags & 6u) != 0u || ((flags & 1u) != 0u && (flags & 8u) != 0u);
}

float iw_load_k(uint base, uint d) {
#if defined(FA_KV_QUANT)
    return fa_kv_load(src1, base, d);
#else
    return load_auto(src1, base + d * nb10, src1_esize);
#endif
}

float iw_load_v(uint base, uint d) {
#if defined(FA_KV_QUANT)
    return fa_kv_load(src2, base, d);
#else
    return load_auto(src2, base + d * op1, op5 & 0xFFu);
#endif
}

WAVE_SIZE_ATTR
[numthreads(IW_THREADS, 1, 1)]
void main(uint tid : SV_GroupIndex, uint3 gid : SV_GroupID) {
    const uint wave = tid / WAVE_SIZE;
    const uint pv_row = (wave / 2) * 8;
    const uint pv_dblock = (wave % 2) * IW_ODBLK;
#if defined(FA_Q_FORWARD)
    const uint q_start = gid.x * FA_BR;
#else
    const uint q_start = ((ne01 + FA_BR - 1u) / FA_BR - 1u - gid.x) * FA_BR;
#endif
    const uint head = gid.y;
    const uint n_splits = max(op15 & 0xFFFFu, 1u);
    const uint output_split = gid.z % FA_O_SPLITS;
    const uint batch = (gid.z / FA_O_SPLITS) / n_splits;
    const uint split = (gid.z / FA_O_SPLITS) % n_splits;
    const uint kv_head = head * ne12 / ne02;
    const uint dv = op5 >> 8;
    const bool has_mask = (op8 & 1u) != 0u;
    const uint mask_nb0 = (op8 >> 8) & 0xFFu;
    const uint mask_es = (op8 >> 16) & 0xFFu;
    const uint mask_base = op9 + (head % max(op13 & 0xFFFFu, 1u)) * op11
                              + (batch % max(op13 >> 16, 1u)) * op12;
    const float neg_max = -3.402823466e+38f;
    const float scale = asfloat(op6);
    const float softcap = asfloat(op7);
    const float max_bias = asfloat(op14);
    if (q_start >= ne01) {
        return;
    }

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const uint nh = 1u << firstbithigh(ne02);
        const float m0 = exp2(-max_bias / (float)nh);
        const float m1 = exp2(-max_bias * 0.5f / (float)nh);
        slope = head < nh ? pow(m0, (float)(head + 1u)) : pow(m1, (float)(2u * (head - nh) + 1u));
    }

    IWMatAcc coords = IWMatAcc::Splat((float16_t)0);
    float o[IW_ODBLK][IW_ACC_E];
    [unroll] for (uint db = 0; db < IW_ODBLK; ++db) {
        [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
            o[db][e] = 0.0f;
        }
    }
    if (tid < FA_BR) {
        iw_max[tid] = neg_max;
        iw_sum[tid] = 0.0f;
    }
    if (tid == 0) {
        iw_q_unsafe = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint q_flags = 0u;
    for (uint i = tid; i < FA_BR * FA_D; i += IW_THREADS) {
        const uint r = i / FA_D;
        const uint d = i % FA_D;
        float q = 0.0f;
        if (q_start + r < ne01 && d < ne00) {
            q = asfloat(src0.Load(src0_offset + (q_start + r) * nb01 + head * nb02 + batch * nb03 + d * nb00));
        }
        iw_q[r * IW_QKV_STRIDE + d] = (float16_t)q;
        q_flags |= iw_range_flags(q, 128.0f, 0.000244140625f);
    }
    const uint wave_q_flags = WaveActiveBitOr(q_flags);
    if (WaveIsFirstLane()) {
        InterlockedOr(iw_q_unsafe, wave_q_flags);
    }
    GroupMemoryBarrierWithGroupSync();
    const bool q_unsafe = iw_range_unsafe(iw_q_unsafe);
    const float q_restore = (iw_q_unsafe & 1u) != 0u ? 4.0f : 1.0f;
    if (!q_unsafe && q_restore != 1.0f) {
        for (uint i = tid; i < FA_BR * FA_D; i += IW_THREADS) {
            const uint idx = (i / FA_D) * IW_QKV_STRIDE + i % FA_D;
            iw_q[idx] = (float16_t)((float)iw_q[idx] * 0.25f);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    const uint tiles = (ne11 + FA_BC - 1u) / FA_BC;
    const uint tiles_split = (tiles + n_splits - 1u) / n_splits;
    const uint kv_begin = min(split * tiles_split * FA_BC, ne11);
    const uint kv_end = min(kv_begin + tiles_split * FA_BC, ne11);

    for (uint start = kv_begin; start < kv_end; start += FA_BC) {
        const uint size = min((uint)FA_BC, kv_end - start);
        float mask[IW_MASK_E];
        if (tid == 0) {
            iw_any = has_mask ? 0u : 1u;
            iw_kv_unsafe = 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        bool any_visible = false;
        [unroll] for (uint i = 0; i < IW_MASK_E; ++i) {
            const uint idx = i * IW_THREADS + tid;
            const uint r = idx / FA_BC;
            const uint c = idx % FA_BC;
            float m = 0.0f;
            if (has_mask && q_start + r < ne01 && c < size) {
                m = load_auto(src3, mask_base + (q_start + r) * op10 + (start + c) * mask_nb0, mask_es);
                any_visible = any_visible || !isinf(m);
            }
            mask[i] = m;
        }
        const bool wave_visible = WaveActiveAnyTrue(any_visible);
        if (wave_visible && WaveIsFirstLane()) {
            InterlockedOr(iw_any, 1u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (iw_any == 0u) {
            continue;
        }

        uint k_flags = 0u;
        for (uint i = tid; i < FA_BC * FA_D; i += IW_THREADS) {
            const uint c = i / FA_D;
            const uint d = i % FA_D;
            const float k = c < size && d < ne00 ? iw_load_k(src1_offset + (start + c) * nb11 + kv_head * nb12 + batch * nb13, d) : 0.0f;
            iw_kv[c * IW_QKV_STRIDE + d] = (float16_t)k;
            k_flags |= iw_range_flags(k, 512.0f, 0.0009765625f);
        }
        const uint wave_k_flags = WaveActiveBitOr(k_flags);
        if (WaveIsFirstLane()) {
            InterlockedOr(iw_kv_unsafe, wave_k_flags);
        }
        GroupMemoryBarrierWithGroupSync();

        const bool qk_safe = !q_unsafe && !iw_range_unsafe(iw_kv_unsafe);
        const float k_restore = (iw_kv_unsafe & 1u) != 0u ? 16.0f : 1.0f;
        if (qk_safe && k_restore != 1.0f) {
            for (uint i = tid; i < FA_BC * FA_D; i += IW_THREADS) {
                const uint idx = (i / FA_D) * IW_QKV_STRIDE + i % FA_D;
                iw_kv[idx] = (float16_t)((float)iw_kv[idx] * 0.0625f);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (qk_safe && wave < FA_BR / 8) {
            [unroll] for (uint cb = 0; cb < FA_BC / 16; ++cb) {
                float score[IW_ACC_E];
                [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                    score[e] = 0.0f;
                }
                [unroll] for (uint kb = 0; kb < FA_D / 16; ++kb) {
                    IWMatA a = IWMatA::Load(iw_q, wave * 8 * IW_QKV_STRIDE + kb * 16, IW_QKV_STRIDE, MatrixLayout::RowMajor);
                    IWMatB b = IWMatB::Load(iw_kv, cb * 16 * IW_QKV_STRIDE + kb * 16, IW_QKV_STRIDE, MatrixLayout::ColMajor);
                    IWMatAcc partial = IWMatAcc::Splat((float16_t)0);
                    partial.MultiplyAccumulate(a, b);
                    [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                        score[e] += (float)partial.Get(e);
                    }
                }
                [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                    const uint2 rc = coords.GetCoordinate(e);
                    iw_s[iw_score_index((wave * 8 + rc.x) * FA_BC + cb * 16 + rc.y)] = score[e] * (q_restore * k_restore);
                }
            }
        } else if (!qk_safe) {
            // Preserve originals when range or subnormal inputs prevent safe half staging.
            for (uint i = tid; i < FA_BR * FA_BC; i += IW_THREADS) {
                const uint r = i / FA_BC;
                const uint c = i % FA_BC;
                float score = 0.0f;
                if (q_start + r < ne01 && c < size) {
                    const uint qb = src0_offset + (q_start + r) * nb01 + head * nb02 + batch * nb03;
                    const uint kb = src1_offset + (start + c) * nb11 + kv_head * nb12 + batch * nb13;
                    for (uint d = 0; d < ne00; ++d) {
                        score += asfloat(src0.Load(qb + d * nb00)) * iw_load_k(kb, d);
                    }
                }
                iw_s[iw_score_index(i)] = score;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        [unroll] for (uint i = 0; i < IW_MASK_E; ++i) {
            const uint idx = i * IW_THREADS + tid;
            float score = neg_max;
            if (q_start + idx / FA_BC < ne01 && idx % FA_BC < size) {
                score = iw_s[iw_score_index(idx)] * scale;
                if (softcap != 0.0f) {
                    score = softcap * tanh(score);
                }
                if (has_mask) {
                    const float m = mask[i] * slope;
                    score = isinf(m) ? neg_max : score + m;
                }
            }
            iw_s[iw_score_index(idx)] = score;
        }
        GroupMemoryBarrierWithGroupSync();

        if (tid < FA_BR) {
            float tile_max = neg_max;
            [unroll] for (uint c = 0; c < FA_BC; ++c) {
                tile_max = max(tile_max, iw_s[iw_score_index(tid * FA_BC + c)]);
            }
            float corr = 1.0f;
            if (tile_max != neg_max) {
                const float new_max = max(iw_max[tid], tile_max);
                corr = iw_sum[tid] > 0.0f ? exp(iw_max[tid] - new_max) : 0.0f;
                iw_max[tid] = new_max;
            }
            float sum = 0.0f;
            [unroll] for (uint c = 0; c < FA_BC; ++c) {
                const uint idx = tid * FA_BC + c;
                const float score = iw_s[iw_score_index(idx)];
                const float p = score == neg_max ? 0.0f : exp(score - iw_max[tid]);
                iw_s[iw_score_index(idx)] = p;
                iw_p[tid * IW_P_STRIDE + c] = (float16_t)p;
                sum += p;
            }
            iw_sum[tid] = iw_sum[tid] * corr + sum;
            IW_CORR(tid) = corr;
        }
        if (tid == 0) {
            iw_kv_unsafe = 0u;
            iw_any = 0u;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint i = tid; i < FA_BC * FA_D; i += IW_THREADS) {
            const uint c = i / FA_D;
            const uint d = i % FA_D;
            const float v = c < size && d < dv ? iw_load_v(op0 + (start + c) * op2 + kv_head * op3 + batch * op4, d) : 0.0f;
            iw_kv[c * IW_QKV_STRIDE + d] = (float16_t)v;
            if (!(abs(v) <= 65504.0f)) {
                InterlockedOr(iw_kv_unsafe, 1u);
            }
            if (abs(v) > 2048.0f) {
                InterlockedOr(iw_any, 1u);
            }
        }
        GroupMemoryBarrierWithGroupSync();

        // Scaling P by 1/32 bounds each K16 partial by 32752 for finite F16 V.
        const float pv_restore = iw_any != 0u ? 32.0f : 1.0f;
        if (iw_any != 0u) {
            for (uint i = tid; i < FA_BR * FA_BC; i += IW_THREADS) {
                const float p = iw_s[iw_score_index(i)];
                iw_p[(i / FA_BC) * IW_P_STRIDE + i % FA_BC] = (float16_t)(p * 0.03125f);
                // Keep scaled nonzero P normal; otherwise use the original F32 P.
                if (p > 0.0f && p < 0.001953125f) {
                    InterlockedOr(iw_kv_unsafe, 1u);
                }
            }
            // Mixed large and small V can underflow products after scaling.
            for (uint i = tid; i < FA_BC * FA_D; i += IW_THREADS) {
                const float v = (float)iw_kv[(i / FA_D) * IW_QKV_STRIDE + i % FA_D];
                if (v != 0.0f && abs(v) < 1.0f) {
                    InterlockedOr(iw_kv_unsafe, 1u);
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }

        [unroll] for (uint db = 0; db < IW_ODBLK; ++db) {
            const uint dblock = output_split * IW_ODBLK * 2 + pv_dblock + db;
            [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                const uint2 rc = coords.GetCoordinate(e);
                o[db][e] *= IW_CORR(pv_row + rc.x);
            }
            if (iw_kv_unsafe == 0u) {
                [unroll] for (uint cb = 0; cb < FA_BC / 16; ++cb) {
                    IWMatA a = IWMatA::Load(iw_p, pv_row * IW_P_STRIDE + cb * 16, IW_P_STRIDE, MatrixLayout::RowMajor);
                    IWMatB b = IWMatB::Load(iw_kv, cb * 16 * IW_QKV_STRIDE + dblock * 16, IW_QKV_STRIDE, MatrixLayout::RowMajor);
                    IWMatAcc partial = IWMatAcc::Splat((float16_t)0);
                    partial.MultiplyAccumulate(a, b);
                    [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                        o[db][e] += (float)partial.Get(e) * pv_restore;
                    }
                }
            } else {
                // Values outside F16 range and unsafe scaling ranges stay F32.
                [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                    const uint2 rc = coords.GetCoordinate(e);
                    const uint r = pv_row + rc.x;
                    const uint d = dblock * 16 + rc.y;
                    if (q_start + r < ne01 && d < dv) {
                        for (uint c = 0; c < size; ++c) {
                            const float p = iw_s[iw_score_index(r * FA_BC + c)];
                            if (p != 0.0f) {
                                o[db][e] += p * iw_load_v(op0 + (start + c) * op2 + kv_head * op3 + batch * op4, d);
                            }
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (((op8 >> 24) & 1u) != 0u && split == 0u) {
        if (tid < FA_BR) {
            const float sink = asfloat(src4.Load(head * 4u));
            const float new_max = max(iw_max[tid], sink);
            const float corr = iw_sum[tid] > 0.0f ? exp(iw_max[tid] - new_max) : 0.0f;
            iw_sum[tid] = iw_sum[tid] * corr + exp(sink - new_max);
            iw_max[tid] = new_max;
            IW_CORR(tid) = corr;
        }
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint db = 0; db < IW_ODBLK; ++db) {
            [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
                const uint2 rc = coords.GetCoordinate(e);
                o[db][e] *= IW_CORR(pv_row + rc.x);
            }
        }
    }

    const uint partial_stride = (dv + 2u) * 4u;
    const uint partial_base = ((batch * ne02 + head) * ne01 + q_start) * n_splits + split;
    if (n_splits > 1u && output_split == 0u && tid < FA_BR && q_start + tid < ne01) {
        const uint off = (partial_base + tid * n_splits) * partial_stride;
        temp.Store(off, asuint(iw_max[tid]));
        temp.Store(off + 4u, asuint(iw_sum[tid]));
    }
    [unroll] for (uint db = 0; db < IW_ODBLK; ++db) {
        [unroll] for (uint e = 0; e < IW_ACC_E; ++e) {
            const uint2 rc = coords.GetCoordinate(e);
            const uint r = pv_row + rc.x;
            const uint d = (output_split * IW_ODBLK * 2 + pv_dblock + db) * 16 + rc.y;
            if (q_start + r < ne01 && d < dv) {
                if (n_splits > 1u) {
                    temp.Store((partial_base + r * n_splits) * partial_stride + 8u + d * 4u, asuint(o[db][e]));
                } else {
                    const float inv_sum = iw_sum[r] > 0.0f ? 1.0f / iw_sum[r] : 0.0f;
                    store_auto(dst, dst_offset + d * nb0 + head * nb1 + (q_start + r) * nb2 + batch * nb3, o[db][e] * inv_sum, dst_esize);
                }
            }
        }
    }
}
