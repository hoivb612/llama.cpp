// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// fusion_cpu_extra_ops.cpp
//
// The "extra" single-node op kernels claimed by FusionCpuEp so that the whole
// Phi-3 (cpu-int4-rtn) graph runs on the plugin and every parallel region is
// driven by the single FusionThreadPool (via FusionParallelFor) -- no ORT
// intra-op pool participates.
//
// Milestone 1 (this file): the only real unclaimed compute on that graph --
//   * SkipSimplifiedLayerNormalization (com.microsoft): value = input + skip;
//     out = value * rsqrt(mean(value^2) + eps) * gamma; opt out[3] = value.
//   * SimplifiedLayerNormalization (default domain): plain RMSNorm (no skip).
// The trivial integer mask-prep ops (Shape/Gather/ReduceSum/Sub/Cast/Constant)
// are added in a later milestone; until then they fall back to the CPU EP.
//
// See fusion_cpu_extra_ops.h for how this plugs into fusion_cpu_ep.cpp.

#include "fusion_cpu_extra_ops.h"

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fusion_cpu_ep {
namespace {

// --- small type/shape/attr helpers (local mirrors of the ones in
// fusion_cpu_ep.cpp's anonymous namespace, which are not visible across TUs) ---

ONNXTensorElementDataType TensorElementTypeOf(Ort::ConstValueInfo vi) {
    Ort::ConstTypeInfo t = vi.TypeInfo();
    if (t.GetONNXType() != ONNX_TYPE_TENSOR) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    return t.GetTensorTypeAndShapeInfo().GetElementType();
}

std::optional<std::vector<int64_t>> GetShapeAllowSymbolic(Ort::ConstValueInfo vi) {
    Ort::ConstTypeInfo t = vi.TypeInfo();
    if (t.GetONNXType() != ONNX_TYPE_TENSOR) return std::nullopt;
    return t.GetTensorTypeAndShapeInfo().GetShape();
}

bool ReadInt64Attr(const Ort::ConstNode& node, const char* name, int64_t& out) {
    Ort::ConstOpAttr attr;
    Ort::Status st = node.GetAttributeByName(name, attr);
    if (!st.IsOK()) return false;
    return attr.GetValue<int64_t>(out).IsOK();
}

bool ReadFloatAttr(const Ort::ConstNode& node, const char* name, float& out) {
    Ort::ConstOpAttr attr;
    Ort::Status st = node.GetAttributeByName(name, attr);
    if (!st.IsOK()) return false;
    return attr.GetValue<float>(out).IsOK();
}

constexpr ONNXTensorElementDataType kF32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;

// ===== RMSNorm kernel (Simplified / SkipSimplified LayerNormalization) =======
//
// Handles both the plain (no residual) and the Skip (fused residual-add) forms.
// Math (fp32, RMSNorm -- no mean subtraction, no bias/beta):
//   value[h] = input[h] (+ skip[h])
//   ms       = (1/H) * sum_h value[h]^2
//   out[h]   = value[h] * rsqrt(ms + eps) * gamma[h]
// If a residual-sum output is present (Skip out[3]), it receives `value`.
// Parallelised over rows (all leading dims collapsed) on the shared pool.
struct RmsNormKernel : ExtraOpKernel {
    RmsNormKernel(const OrtApi* api, float eps, bool has_skip,
                  int idx_in, int idx_skip, int idx_gamma, int idx_out, int idx_sum)
        : api_(api), eps_(eps), has_skip_(has_skip),
          idx_in_(idx_in), idx_skip_(idx_skip), idx_gamma_(idx_gamma),
          idx_out_(idx_out), idx_sum_(idx_sum) {}

    OrtStatus* Fail(const char* msg) const {
        return api_->CreateStatus(ORT_INVALID_ARGUMENT, msg);
    }

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};

        Ort::ConstValue in = ctx.GetInput(idx_in_);
        auto tin = in.GetTensorTypeAndShapeInfo();
        if (tin.GetElementType() != kF32) return Fail("RmsNorm: input must be float32");
        std::vector<int64_t> shape = tin.GetShape();
        if (shape.empty()) return Fail("RmsNorm: input rank must be >= 1");
        const int64_t H = shape.back();
        int64_t numel = 1;
        for (int64_t d : shape) numel *= d;
        if (H <= 0 || numel <= 0 || numel % H != 0) return Fail("RmsNorm: bad shape");
        const int64_t rows = numel / H;

        const float* in_d = in.GetTensorData<float>();
        const float* skip_d = nullptr;
        if (has_skip_) skip_d = ctx.GetInput(idx_skip_).GetTensorData<float>();
        const float* gamma = ctx.GetInput(idx_gamma_).GetTensorData<float>();

        float* out_d = ctx.GetOutput(idx_out_, shape).GetTensorMutableData<float>();
        float* sum_d = nullptr;
        if (has_skip_ && idx_sum_ >= 0)
            sum_d = ctx.GetOutput(idx_sum_, shape).GetTensorMutableData<float>();

        const float eps = eps_;
        const bool has_skip = has_skip_;
        const int64_t Hc = H;

        FusionParallelFor(static_cast<int>(rows), [&](int begin, int end) {
            std::vector<float> val(static_cast<size_t>(Hc));
            for (int r = begin; r < end; ++r) {
                const float* xin = in_d + static_cast<int64_t>(r) * Hc;
                float ms = 0.0f;
                if (has_skip) {
                    const float* xsk = skip_d + static_cast<int64_t>(r) * Hc;
                    for (int64_t h = 0; h < Hc; ++h) {
                        const float v = xin[h] + xsk[h];
                        val[static_cast<size_t>(h)] = v;
                        ms += v * v;
                    }
                } else {
                    for (int64_t h = 0; h < Hc; ++h) {
                        const float v = xin[h];
                        val[static_cast<size_t>(h)] = v;
                        ms += v * v;
                    }
                }
                const float inv = 1.0f / std::sqrt(ms / static_cast<float>(Hc) + eps);
                float* xout = out_d + static_cast<int64_t>(r) * Hc;
                for (int64_t h = 0; h < Hc; ++h)
                    xout[h] = val[static_cast<size_t>(h)] * inv * gamma[h];
                if (sum_d) {
                    float* xs = sum_d + static_cast<int64_t>(r) * Hc;
                    for (int64_t h = 0; h < Hc; ++h) xs[h] = val[static_cast<size_t>(h)];
                }
            }
        });
        return nullptr;
    }

    const OrtApi* api_;
    float eps_;
    bool has_skip_;
    int idx_in_, idx_skip_, idx_gamma_, idx_out_, idx_sum_;
};

// ===== element-size + dtype helpers ==========================================

size_t ElemSize(ONNXTensorElementDataType dt) {
    switch (dt) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:  return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:  return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return 1;
        default:                                    return 0;
    }
}

// The numeric dtypes handled by the Cast/ReduceSum/Sub compute paths.
bool IsSupportedNumeric(ONNXTensorElementDataType dt) {
    switch (dt) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return true;
        default:                                    return false;
    }
}

// ===== Shape ==================================================================
// out (int64, rank-1) = input dims in [start, end).  Supports the opset-15
// start/end attributes (default 0..rank); negative indices wrap.
struct ShapeKernel : ExtraOpKernel {
    ShapeKernel(const OrtApi* api, int idx_in, int idx_out,
                int64_t start, int64_t end, bool has_end)
        : api_(api), idx_in_(idx_in), idx_out_(idx_out),
          start_(start), end_(end), has_end_(has_end) {}

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};
        Ort::ConstValue in = ctx.GetInput(idx_in_);
        std::vector<int64_t> shape = in.GetTensorTypeAndShapeInfo().GetShape();
        const int64_t r = static_cast<int64_t>(shape.size());
        int64_t s = start_ < 0 ? start_ + r : start_;
        int64_t e = has_end_ ? (end_ < 0 ? end_ + r : end_) : r;
        s = std::clamp<int64_t>(s, 0, r);
        e = std::clamp<int64_t>(e, 0, r);
        const int64_t n = e > s ? e - s : 0;
        int64_t out_shape[1] = {n};
        int64_t* od = ctx.GetOutput(idx_out_, out_shape, 1).GetTensorMutableData<int64_t>();
        for (int64_t i = 0; i < n; ++i) od[i] = shape[static_cast<size_t>(s + i)];
        return nullptr;
    }

    const OrtApi* api_;
    int idx_in_, idx_out_;
    int64_t start_, end_;
    bool has_end_;
};

// ===== Cast ==================================================================
// Elementwise numeric cast to `to_`.  C++ conversion semantics (float->int
// truncates toward zero), matching ONNX Cast for the numeric types we claim.
struct CastKernel : ExtraOpKernel {
    CastKernel(const OrtApi* api, int idx_in, int idx_out, ONNXTensorElementDataType to)
        : api_(api), idx_in_(idx_in), idx_out_(idx_out), to_(to) {}

    OrtStatus* Fail(const char* m) const { return api_->CreateStatus(ORT_INVALID_ARGUMENT, m); }

    template <class S>
    static void WriteAs(const S* s, size_t n, void* dst, ONNXTensorElementDataType to) {
        switch (to) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                { auto d = static_cast<float*>(dst);   for (size_t i=0;i<n;++i) d[i]=static_cast<float>(s[i]); } break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                { auto d = static_cast<double*>(dst);  for (size_t i=0;i<n;++i) d[i]=static_cast<double>(s[i]); } break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                { auto d = static_cast<int64_t*>(dst); for (size_t i=0;i<n;++i) d[i]=static_cast<int64_t>(s[i]); } break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                { auto d = static_cast<int32_t*>(dst); for (size_t i=0;i<n;++i) d[i]=static_cast<int32_t>(s[i]); } break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                { auto d = static_cast<uint8_t*>(dst); for (size_t i=0;i<n;++i) d[i]=static_cast<uint8_t>(s[i]); } break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                { auto d = static_cast<uint8_t*>(dst); for (size_t i=0;i<n;++i) d[i]=static_cast<uint8_t>(s[i]!=S(0)); } break;
            default: break;
        }
    }

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};
        Ort::ConstValue in = ctx.GetInput(idx_in_);
        auto ti = in.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType from = ti.GetElementType();
        std::vector<int64_t> shape = ti.GetShape();
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        void* dst = ctx.GetOutput(idx_out_, shape.data(), shape.size()).GetTensorMutableRawData();
        switch (from) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  WriteAs(in.GetTensorData<float>(),   n, dst, to_); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: WriteAs(in.GetTensorData<double>(),  n, dst, to_); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  WriteAs(in.GetTensorData<int64_t>(), n, dst, to_); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  WriteAs(in.GetTensorData<int32_t>(), n, dst, to_); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:  WriteAs(in.GetTensorData<uint8_t>(), n, dst, to_); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   WriteAs(in.GetTensorData<uint8_t>(), n, dst, to_); break;
            default: return Fail("Cast: unsupported source dtype");
        }
        return nullptr;
    }

    const OrtApi* api_;
    int idx_in_, idx_out_;
    ONNXTensorElementDataType to_;
};

// ===== Gather ================================================================
// out = take(data, indices, axis).  dtype-agnostic (byte-block copy sized from
// the runtime element type); indices may be int64 or int32; negative indices
// wrap.  Parallelised over (outer * n_indices) rows on the shared pool.
struct GatherKernel : ExtraOpKernel {
    GatherKernel(const OrtApi* api, int idx_data, int idx_ind, int idx_out, int64_t axis)
        : api_(api), idx_data_(idx_data), idx_ind_(idx_ind), idx_out_(idx_out), axis_(axis) {}

    OrtStatus* Fail(const char* m) const { return api_->CreateStatus(ORT_INVALID_ARGUMENT, m); }

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};
        Ort::ConstValue data = ctx.GetInput(idx_data_);
        Ort::ConstValue ind  = ctx.GetInput(idx_ind_);
        auto dti = data.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dshape = dti.GetShape();
        const int64_t r = static_cast<int64_t>(dshape.size());
        if (r == 0) return Fail("Gather: data must have rank >= 1");
        int64_t ax = axis_ < 0 ? axis_ + r : axis_;
        if (ax < 0 || ax >= r) return Fail("Gather: axis out of range");
        const size_t esz = ElemSize(dti.GetElementType());
        if (esz == 0) return Fail("Gather: unsupported data dtype");

        const int64_t axis_dim = dshape[static_cast<size_t>(ax)];
        int64_t outer = 1, inner = 1;
        for (int64_t d = 0; d < ax; ++d) outer *= dshape[static_cast<size_t>(d)];
        for (int64_t d = ax + 1; d < r; ++d) inner *= dshape[static_cast<size_t>(d)];

        auto iti = ind.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> ishape = iti.GetShape();
        int64_t nind = 1;
        for (int64_t d : ishape) nind *= d;
        std::vector<int64_t> idxv(static_cast<size_t>(nind));
        const ONNXTensorElementDataType idt = iti.GetElementType();
        if (idt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            const int64_t* ip = ind.GetTensorData<int64_t>();
            for (int64_t i = 0; i < nind; ++i) idxv[static_cast<size_t>(i)] = ip[i];
        } else if (idt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            const int32_t* ip = ind.GetTensorData<int32_t>();
            for (int64_t i = 0; i < nind; ++i) idxv[static_cast<size_t>(i)] = ip[i];
        } else {
            return Fail("Gather: indices must be int64 or int32");
        }
        for (int64_t& v : idxv) { if (v < 0) v += axis_dim; if (v < 0 || v >= axis_dim) return Fail("Gather: index out of range"); }

        // out shape = dshape[:ax] + ishape + dshape[ax+1:]
        std::vector<int64_t> oshape;
        oshape.reserve(static_cast<size_t>(r - 1) + ishape.size());
        for (int64_t d = 0; d < ax; ++d) oshape.push_back(dshape[static_cast<size_t>(d)]);
        for (int64_t d : ishape) oshape.push_back(d);
        for (int64_t d = ax + 1; d < r; ++d) oshape.push_back(dshape[static_cast<size_t>(d)]);

        const uint8_t* dp = data.GetTensorData<uint8_t>();
        uint8_t* op = static_cast<uint8_t*>(
            ctx.GetOutput(idx_out_, oshape.data(), oshape.size()).GetTensorMutableRawData());
        const int64_t block = inner * static_cast<int64_t>(esz);
        const int64_t rows = outer * nind;
        const int64_t inner_c = inner, axis_c = axis_dim, nind_c = nind;
        const int64_t* idxp = idxv.data();

        FusionParallelFor(static_cast<int>(rows), [&](int begin, int end) {
            for (int row = begin; row < end; ++row) {
                const int64_t o = row / nind_c;
                const int64_t k = row % nind_c;
                const int64_t src = ((o * axis_c + idxp[k]) * inner_c) * static_cast<int64_t>(esz);
                std::memcpy(op + static_cast<int64_t>(row) * block, dp + src, static_cast<size_t>(block));
            }
        });
        return nullptr;
    }

    const OrtApi* api_;
    int idx_data_, idx_ind_, idx_out_;
    int64_t axis_;
};

// ===== ReduceSum =============================================================
// Sum over `axes` (opset-13 input tensor; empty => reduce all unless
// noop_with_empty_axes).  int64 or float data.  keepdims honoured.
struct ReduceSumKernel : ExtraOpKernel {
    ReduceSumKernel(const OrtApi* api, int idx_data, int idx_axes, int idx_out,
                    bool keepdims, bool noop_empty)
        : api_(api), idx_data_(idx_data), idx_axes_(idx_axes), idx_out_(idx_out),
          keepdims_(keepdims), noop_empty_(noop_empty) {}

    OrtStatus* Fail(const char* m) const { return api_->CreateStatus(ORT_INVALID_ARGUMENT, m); }

    template <class T>
    void Run(const T* in, const std::vector<int64_t>& dshape,
             const std::vector<char>& reduced, T* out, size_t out_numel) {
        for (size_t i = 0; i < out_numel; ++i) out[i] = T(0);
        const int64_t r = static_cast<int64_t>(dshape.size());
        // strides of the (keepdims) output, where reduced dims are size 1.
        std::vector<int64_t> ostride(static_cast<size_t>(r), 0);
        int64_t acc = 1;
        for (int64_t d = r - 1; d >= 0; --d) {
            const bool red = reduced[static_cast<size_t>(d)] != 0;
            ostride[static_cast<size_t>(d)] = red ? 0 : acc;
            acc *= red ? 1 : dshape[static_cast<size_t>(d)];
        }
        int64_t in_numel = 1;
        for (int64_t d : dshape) in_numel *= d;
        std::vector<int64_t> midx(static_cast<size_t>(r), 0);
        for (int64_t lin = 0; lin < in_numel; ++lin) {
            int64_t ooff = 0;
            for (int64_t d = 0; d < r; ++d) ooff += midx[static_cast<size_t>(d)] * ostride[static_cast<size_t>(d)];
            out[static_cast<size_t>(ooff)] += in[static_cast<size_t>(lin)];
            for (int64_t d = r - 1; d >= 0; --d) {
                if (++midx[static_cast<size_t>(d)] < dshape[static_cast<size_t>(d)]) break;
                midx[static_cast<size_t>(d)] = 0;
            }
        }
    }

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};
        Ort::ConstValue data = ctx.GetInput(idx_data_);
        auto dti = data.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType dt = dti.GetElementType();
        std::vector<int64_t> dshape = dti.GetShape();
        const int64_t r = static_cast<int64_t>(dshape.size());

        std::vector<char> reduced(static_cast<size_t>(r), 0);
        int64_t n_axes = 0;
        if (idx_axes_ >= 0) {
            Ort::ConstValue axv = ctx.GetInput(idx_axes_);
            std::vector<int64_t> ashape = axv.GetTensorTypeAndShapeInfo().GetShape();
            n_axes = 1;
            for (int64_t d : ashape) n_axes *= d;
            const int64_t* ap = axv.GetTensorData<int64_t>();
            for (int64_t i = 0; i < n_axes; ++i) {
                int64_t a = ap[i]; if (a < 0) a += r;
                if (a < 0 || a >= r) return Fail("ReduceSum: axis out of range");
                reduced[static_cast<size_t>(a)] = 1;
            }
        }
        if (n_axes == 0) {
            if (noop_empty_) {  // identity copy
                std::vector<int64_t> osh = dshape;
                size_t nb = ElemSize(dt);
                size_t cnt = 1; for (int64_t d : dshape) cnt *= static_cast<size_t>(d);
                std::memcpy(ctx.GetOutput(idx_out_, osh.data(), osh.size()).GetTensorMutableRawData(),
                            data.GetTensorData<uint8_t>(), cnt * nb);
                return nullptr;
            }
            for (auto& c : reduced) c = 1;  // reduce all
        }

        std::vector<int64_t> oshape;
        for (int64_t d = 0; d < r; ++d) {
            if (reduced[static_cast<size_t>(d)]) { if (keepdims_) oshape.push_back(1); }
            else oshape.push_back(dshape[static_cast<size_t>(d)]);
        }
        size_t out_numel = 1; for (int64_t d : oshape) out_numel *= static_cast<size_t>(d);

        if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_t* od = ctx.GetOutput(idx_out_, oshape.data(), oshape.size()).GetTensorMutableData<int64_t>();
            Run<int64_t>(data.GetTensorData<int64_t>(), dshape, reduced, od, out_numel);
        } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float* od = ctx.GetOutput(idx_out_, oshape.data(), oshape.size()).GetTensorMutableData<float>();
            Run<float>(data.GetTensorData<float>(), dshape, reduced, od, out_numel);
        } else {
            return Fail("ReduceSum: unsupported dtype");
        }
        return nullptr;
    }

    const OrtApi* api_;
    int idx_data_, idx_axes_, idx_out_;
    bool keepdims_, noop_empty_;
};

// ===== Sub ===================================================================
// Elementwise A - B with NumPy broadcasting.  int64 or float.
struct SubKernel : ExtraOpKernel {
    SubKernel(const OrtApi* api, int idx_a, int idx_b, int idx_out)
        : api_(api), idx_a_(idx_a), idx_b_(idx_b), idx_out_(idx_out) {}

    OrtStatus* Fail(const char* m) const { return api_->CreateStatus(ORT_INVALID_ARGUMENT, m); }

    template <class T>
    void Run(const T* a, const std::vector<int64_t>& as,
             const T* b, const std::vector<int64_t>& bs,
             T* out, const std::vector<int64_t>& os) {
        const int64_t r = static_cast<int64_t>(os.size());
        auto strides = [&](const std::vector<int64_t>& s) {
            std::vector<int64_t> st(static_cast<size_t>(r), 0);
            int64_t acc = 1;
            const int64_t off = r - static_cast<int64_t>(s.size());
            for (int64_t d = r - 1; d >= 0; --d) {
                const int64_t sd = (d - off >= 0) ? s[static_cast<size_t>(d - off)] : 1;
                st[static_cast<size_t>(d)] = (sd == 1) ? 0 : acc;
                acc *= sd;
            }
            return st;
        };
        std::vector<int64_t> ast = strides(as), bst = strides(bs);
        int64_t numel = 1; for (int64_t d : os) numel *= d;
        std::vector<int64_t> midx(static_cast<size_t>(r), 0);
        for (int64_t lin = 0; lin < numel; ++lin) {
            int64_t ao = 0, bo = 0;
            for (int64_t d = 0; d < r; ++d) {
                ao += midx[static_cast<size_t>(d)] * ast[static_cast<size_t>(d)];
                bo += midx[static_cast<size_t>(d)] * bst[static_cast<size_t>(d)];
            }
            out[static_cast<size_t>(lin)] = a[static_cast<size_t>(ao)] - b[static_cast<size_t>(bo)];
            for (int64_t d = r - 1; d >= 0; --d) {
                if (++midx[static_cast<size_t>(d)] < os[static_cast<size_t>(d)]) break;
                midx[static_cast<size_t>(d)] = 0;
            }
        }
    }

    OrtStatus* Compute(OrtKernelContext* kctx) override {
        Ort::KernelContext ctx{kctx};
        Ort::ConstValue a = ctx.GetInput(idx_a_);
        Ort::ConstValue b = ctx.GetInput(idx_b_);
        auto ati = a.GetTensorTypeAndShapeInfo();
        auto bti = b.GetTensorTypeAndShapeInfo();
        const ONNXTensorElementDataType dt = ati.GetElementType();
        if (bti.GetElementType() != dt) return Fail("Sub: mixed dtypes");
        std::vector<int64_t> as = ati.GetShape(), bs = bti.GetShape();
        const size_t r = std::max(as.size(), bs.size());
        std::vector<int64_t> os(r, 1);
        for (size_t i = 0; i < r; ++i) {
            const int64_t ad = (i + as.size() >= r) ? as[i - (r - as.size())] : 1;
            const int64_t bd = (i + bs.size() >= r) ? bs[i - (r - bs.size())] : 1;
            os[i] = std::max(ad, bd);
        }
        if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            int64_t* od = ctx.GetOutput(idx_out_, os.data(), os.size()).GetTensorMutableData<int64_t>();
            Run<int64_t>(a.GetTensorData<int64_t>(), as, b.GetTensorData<int64_t>(), bs, od, os);
        } else if (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            float* od = ctx.GetOutput(idx_out_, os.data(), os.size()).GetTensorMutableData<float>();
            Run<float>(a.GetTensorData<float>(), as, b.GetTensorData<float>(), bs, od, os);
        } else {
            return Fail("Sub: unsupported dtype");
        }
        return nullptr;
    }

    const OrtApi* api_;
    int idx_a_, idx_b_, idx_out_;
};

// ===== shared dispatch pair (reused by every extra op) =======================

OrtStatus* LookupExtraKernel(FusionCpuEp& ep, const std::string& node_name, void** out_state) {
    auto it = ep.extra_kernels().find(node_name);
    if (it == ep.extra_kernels().end()) {
        std::string msg = "FusionCpuEp: no extra-op kernel registered for fused node '" + node_name + "'";
        return ep.ort_api()->CreateStatus(ORT_EP_FAIL, msg.c_str());
    }
    *out_state = it->second.get();
    return nullptr;
}

OrtStatus* ComputeExtraKernel(void* state, OrtKernelContext* kctx) {
    return static_cast<ExtraOpKernel*>(state)->Compute(kctx);
}

}  // namespace

// ===== GetCapability hook ====================================================

bool ExtraTryClaim(FusionCpuEp& ep, const OrtNode* node_c, OrtEpGraphSupportInfo* info) {
    Ort::ConstNode node{node_c};
    const std::string op = node.GetOperatorType();
    const std::string dom = node.GetDomain();

    const bool skip  = (op == "SkipSimplifiedLayerNormalization" && dom == "com.microsoft");
    const bool plain = (op == "SimplifiedLayerNormalization" && dom.empty());

    auto ins  = node.GetInputs();
    auto outs = node.GetOutputs();

    auto out_present = [&](size_t i) {
        return i < outs.size() && static_cast<const OrtValueInfo*>(outs[i]) != nullptr;
    };
    auto in_present = [&](size_t i) {
        return i < ins.size() && static_cast<const OrtValueInfo*>(ins[i]) != nullptr;
    };

    // ---- trivial integer mask-prep ops (Shape/Cast/Gather/ReduceSum/Sub) ----
    // These are claimed one-node-at-a-time so the whole graph runs on the
    // FusionThreadPool; each has a dedicated kernel in this TU.
    if (dom.empty() && (op == "Shape" || op == "Cast" || op == "Gather" ||
                        op == "ReduceSum" || op == "Sub")) {
        bool ok = false;
        if (op == "Shape") {
            ok = (ins.size() == 1 && in_present(0) && out_present(0));
        } else if (op == "Cast") {
            int64_t to = 0;
            if (ins.size() == 1 && in_present(0) && out_present(0) &&
                ReadInt64Attr(node, "to", to)) {
                const auto src = TensorElementTypeOf(ins[0]);
                const auto dst = static_cast<ONNXTensorElementDataType>(to);
                ok = IsSupportedNumeric(src) && IsSupportedNumeric(dst);
            }
        } else if (op == "Gather") {
            if (ins.size() == 2 && in_present(0) && in_present(1) && out_present(0)) {
                const auto idt = TensorElementTypeOf(ins[1]);
                ok = ElemSize(TensorElementTypeOf(ins[0])) != 0 &&
                     (idt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                      idt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
            }
        } else if (op == "ReduceSum") {
            if ((ins.size() == 1 || ins.size() == 2) && in_present(0) && out_present(0)) {
                const auto dt = TensorElementTypeOf(ins[0]);
                ok = (dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                      dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            }
        } else {  // Sub
            if (ins.size() == 2 && in_present(0) && in_present(1) && out_present(0)) {
                const auto ad = TensorElementTypeOf(ins[0]);
                const auto bd = TensorElementTypeOf(ins[1]);
                ok = (ad == bd) && (ad == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ||
                                    ad == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
            }
        }
        if (!ok) return false;

        OrtNodeFusionOptions fopts{};
        fopts.ort_version_supported      = ORT_API_VERSION;
        fopts.drop_constant_initializers = false;  // keep const inputs as Compute inputs
        const OrtNode* one = node_c;
        OrtStatus* st = ep.ep_api()->EpGraphSupportInfo_AddNodesToFuse(info, &one, 1, &fopts);
        if (st != nullptr) { ep.ort_api()->ReleaseStatus(st); return false; }
        return true;
    }

    if (!skip && !plain) return false;

    if (skip) {
        // input, skip, gamma (reject the optional-bias 4-input form).
        if (ins.size() != 3) return false;
        if (TensorElementTypeOf(ins[0]) != kF32 ||
            TensorElementTypeOf(ins[1]) != kF32 ||
            TensorElementTypeOf(ins[2]) != kF32) return false;
        if (!ins[2].IsConstantInitializer()) return false;  // gamma must be const
        if (!out_present(0) || TensorElementTypeOf(outs[0]) != kF32) return false;
    } else {
        // input, gamma.
        if (ins.size() != 2) return false;
        if (TensorElementTypeOf(ins[0]) != kF32 ||
            TensorElementTypeOf(ins[1]) != kF32) return false;
        if (!ins[1].IsConstantInitializer()) return false;
        if (!out_present(0) || TensorElementTypeOf(outs[0]) != kF32) return false;
        // Only normalise over the last dim.
        int64_t axis = -1;
        if (ReadInt64Attr(node, "axis", axis) && axis != -1) {
            auto sh = GetShapeAllowSymbolic(ins[0]);
            if (!sh || axis != static_cast<int64_t>(sh->size()) - 1) return false;
        }
    }

    OrtNodeFusionOptions fopts{};
    fopts.ort_version_supported      = ORT_API_VERSION;
    fopts.drop_constant_initializers = false;  // keep gamma as a Compute input
    const OrtNode* one = node_c;
    OrtStatus* st = ep.ep_api()->EpGraphSupportInfo_AddNodesToFuse(info, &one, 1, &fopts);
    if (st != nullptr) {
        ep.ort_api()->ReleaseStatus(st);
        return false;
    }
    return true;
}

// ===== Compile hook (single-node fused subgraphs) ============================

bool ExtraTryCompile(FusionCpuEp& ep, const OrtNode* fused_c, const OrtNode* inner_c,
                     ExtraLookupFn* out_lookup, ExtraComputeFn* out_compute) {
    Ort::ConstNode inner{inner_c};
    Ort::ConstNode fused{fused_c};
    const std::string op = inner.GetOperatorType();
    const std::string dom = inner.GetDomain();

    const bool skip  = (op == "SkipSimplifiedLayerNormalization" && dom == "com.microsoft");
    const bool plain = (op == "SimplifiedLayerNormalization" && dom.empty());

    auto fin  = fused.GetInputs();
    auto fout = fused.GetOutputs();
    auto find_in = [&](const std::string& nm) -> int {
        if (nm.empty()) return -1;
        for (size_t k = 0; k < fin.size(); ++k)
            if (static_cast<const OrtValueInfo*>(fin[k]) != nullptr && fin[k].GetName() == nm)
                return static_cast<int>(k);
        return -1;
    };
    auto find_out = [&](const std::string& nm) -> int {
        if (nm.empty()) return -1;
        for (size_t k = 0; k < fout.size(); ++k)
            if (static_cast<const OrtValueInfo*>(fout[k]) != nullptr && fout[k].GetName() == nm)
                return static_cast<int>(k);
        return -1;
    };

    // ---- trivial integer mask-prep ops (Shape/Cast/Gather/ReduceSum/Sub) ----
    if (dom.empty() && (op == "Shape" || op == "Cast" || op == "Gather" ||
                        op == "ReduceSum" || op == "Sub")) {
        auto xin  = inner.GetInputs();
        auto xout = inner.GetOutputs();
        if (xout.empty()) return false;
        const int io = find_out(xout[0].GetName());
        if (io < 0) return false;
        std::unique_ptr<ExtraOpKernel> k;
        if (op == "Shape") {
            const int ii = find_in(xin[0].GetName());
            if (ii < 0) return false;
            int64_t start = 0, end = 0;
            const bool has_end = ReadInt64Attr(inner, "end", end);
            ReadInt64Attr(inner, "start", start);
            k = std::make_unique<ShapeKernel>(ep.ort_api(), ii, io, start, end, has_end);
        } else if (op == "Cast") {
            const int ii = find_in(xin[0].GetName());
            int64_t to = 0;
            if (ii < 0 || !ReadInt64Attr(inner, "to", to)) return false;
            k = std::make_unique<CastKernel>(ep.ort_api(), ii, io,
                                             static_cast<ONNXTensorElementDataType>(to));
        } else if (op == "Gather") {
            if (xin.size() != 2) return false;
            const int id = find_in(xin[0].GetName());
            const int ik = find_in(xin[1].GetName());
            if (id < 0 || ik < 0) return false;
            int64_t axis = 0;
            ReadInt64Attr(inner, "axis", axis);
            k = std::make_unique<GatherKernel>(ep.ort_api(), id, ik, io, axis);
        } else if (op == "ReduceSum") {
            const int id = find_in(xin[0].GetName());
            if (id < 0) return false;
            int ia = -1;
            if (xin.size() > 1 && static_cast<const OrtValueInfo*>(xin[1]) != nullptr)
                ia = find_in(xin[1].GetName());
            int64_t keep = 1, noop = 0;
            ReadInt64Attr(inner, "keepdims", keep);
            ReadInt64Attr(inner, "noop_with_empty_axes", noop);
            k = std::make_unique<ReduceSumKernel>(ep.ort_api(), id, ia, io, keep != 0, noop != 0);
        } else {  // Sub
            if (xin.size() != 2) return false;
            const int ia = find_in(xin[0].GetName());
            const int ib = find_in(xin[1].GetName());
            if (ia < 0 || ib < 0) return false;
            k = std::make_unique<SubKernel>(ep.ort_api(), ia, ib, io);
        }
        ep.extra_kernels()[fused.GetName()] = std::move(k);
        *out_lookup  = &LookupExtraKernel;
        *out_compute = &ComputeExtraKernel;
        return true;
    }

    if (!skip && !plain) return false;

    auto iin  = inner.GetInputs();
    auto iout = inner.GetOutputs();
    float eps = 1e-5f;
    ReadFloatAttr(inner, "epsilon", eps);

    int idx_in, idx_gamma, idx_out;
    int idx_skip = -1, idx_sum = -1;

    if (skip) {
        if (iin.size() != 3 || iout.empty()) return false;
        idx_in    = find_in(iin[0].GetName());
        idx_skip  = find_in(iin[1].GetName());
        idx_gamma = find_in(iin[2].GetName());
        idx_out   = find_out(iout[0].GetName());
        // The residual-sum output (index 3) is optional; only wire it if present.
        if (iout.size() > 3 && static_cast<const OrtValueInfo*>(iout[3]) != nullptr)
            idx_sum = find_out(iout[3].GetName());
        if (idx_in < 0 || idx_skip < 0 || idx_gamma < 0 || idx_out < 0) return false;
    } else {
        if (iin.size() != 2 || iout.empty()) return false;
        idx_in    = find_in(iin[0].GetName());
        idx_gamma = find_in(iin[1].GetName());
        idx_out   = find_out(iout[0].GetName());
        if (idx_in < 0 || idx_gamma < 0 || idx_out < 0) return false;
    }

    const std::string fname = fused.GetName();
    ep.extra_kernels()[fname] = std::make_unique<RmsNormKernel>(
        ep.ort_api(), eps, skip, idx_in, idx_skip, idx_gamma, idx_out, idx_sum);

    *out_lookup  = &LookupExtraKernel;
    *out_compute = &ComputeExtraKernel;
    return true;
}

}  // namespace fusion_cpu_ep
