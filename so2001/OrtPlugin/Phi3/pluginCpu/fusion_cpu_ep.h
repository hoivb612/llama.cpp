// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.

#pragma once

#include <onnxruntime_c_api.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace fusion_cpu_ep {

struct FusionCpuMulKernel;
struct FusionCpuSigmoidKernel;
struct FusionCpuSwiGluKernel;
struct FusionCpuMatMulNBitsKernel;
struct FusionCpuGqaKernel;

// Polymorphic base for the "extra" ops implemented in fusion_cpu_extra_ops.cpp
// (SkipSimplifiedLayerNormalization, SimplifiedLayerNormalization, Gather, and
// the small integer mask-prep ops). Keeping a single virtual-dtor base lets the
// EP own all extra kernels in one map with correct destruction, while the
// concrete kernel types stay entirely in the extra translation unit. Each
// kernel carries its own OrtApi* (like the built-in kernels), so Compute needs
// only the kernel context.
struct ExtraOpKernel {
    virtual ~ExtraOpKernel() = default;
    virtual OrtStatus* Compute(OrtKernelContext* ctx) = 0;
};

// Per-session OrtEp instance. A pure-CPU plugin EP.
//
// M1 scope (proof-of-life):
//   * GetCapability claims Mul (default domain, fp32, 2 inputs of identical
//     static shape, 1 output) as a single-node fused subgraph.
//   * Compile builds one CPU kernel per fused node, stored by fused-node name.
//   * Sync is a no-op (CPU kernels run synchronously inside Compute).
//
// No allocator / data-transfer / stream / kernel-registry members: tensors
// arrive as plain CPU pointers and outputs land in CPU memory the built-in CPU
// EP can consume directly.
class FusionCpuEp : public OrtEp {
public:
    FusionCpuEp(const OrtApi*    ort_api,
                const OrtEpApi*  ep_api,
                const OrtLogger* logger,
                std::string      name);
    ~FusionCpuEp();

    FusionCpuEp(const FusionCpuEp&) = delete;
    FusionCpuEp& operator=(const FusionCpuEp&) = delete;

    std::unordered_map<std::string, std::unique_ptr<FusionCpuMulKernel>>& mul_kernels() {
        return mul_kernels_;
    }
    std::unordered_map<std::string, std::unique_ptr<FusionCpuSigmoidKernel>>& sigmoid_kernels() {
        return sigmoid_kernels_;
    }
    std::unordered_map<std::string, std::unique_ptr<FusionCpuSwiGluKernel>>& swiglu_kernels() {
        return swiglu_kernels_;
    }
    std::unordered_map<std::string, std::unique_ptr<FusionCpuMatMulNBitsKernel>>& matmulnbits_kernels() {
        return matmulnbits_kernels_;
    }
    std::unordered_map<std::string, std::unique_ptr<FusionCpuGqaKernel>>& gqa_kernels() {
        return gqa_kernels_;
    }
    std::unordered_map<std::string, std::unique_ptr<ExtraOpKernel>>& extra_kernels() {
        return extra_kernels_;
    }

    const OrtApi*    ort_api() const { return ort_api_; }
    const OrtEpApi*  ep_api()  const { return ep_api_; }
    const OrtLogger* logger()  const { return logger_; }
    const std::string& name()  const { return name_; }

private:
    // --- OrtEp v-table trampolines ---
    static const char* ORT_API_CALL GetNameImpl(const OrtEp* this_ptr) noexcept;
    static OrtStatus*  ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr,
                                                      const OrtGraph* graph,
                                                      OrtEpGraphSupportInfo* info) noexcept;
    static OrtStatus*  ORT_API_CALL CompileImpl(OrtEp* this_ptr,
                                                const OrtGraph** graphs,
                                                const OrtNode**  fused_nodes,
                                                size_t           count,
                                                OrtNodeComputeInfo** node_compute_infos,
                                                OrtNode**            ep_context_nodes) noexcept;
    static void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp* this_ptr,
                                                         OrtNodeComputeInfo** infos,
                                                         size_t count) noexcept;
    static OrtStatus* ORT_API_CALL SyncImpl(OrtEp* this_ptr) noexcept;

    const OrtApi*    ort_api_ = nullptr;
    const OrtEpApi*  ep_api_  = nullptr;
    const OrtLogger* logger_  = nullptr;
    std::string      name_;

    std::unordered_map<std::string, std::unique_ptr<FusionCpuMulKernel>> mul_kernels_;
    std::unordered_map<std::string, std::unique_ptr<FusionCpuSigmoidKernel>> sigmoid_kernels_;
    std::unordered_map<std::string, std::unique_ptr<FusionCpuSwiGluKernel>> swiglu_kernels_;
    std::unordered_map<std::string, std::unique_ptr<FusionCpuMatMulNBitsKernel>> matmulnbits_kernels_;
    std::unordered_map<std::string, std::unique_ptr<FusionCpuGqaKernel>> gqa_kernels_;
    std::unordered_map<std::string, std::unique_ptr<ExtraOpKernel>> extra_kernels_;
};

}  // namespace fusion_cpu_ep
