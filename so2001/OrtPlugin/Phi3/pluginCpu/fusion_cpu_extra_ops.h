// Copyright (c) 2026 ONNX Runtime FusionCPU EP contributors.
// Licensed under the MIT License.
//
// fusion_cpu_extra_ops.h
//
// Seam between fusion_cpu_ep.cpp (the EP shell: GetCapability / Compile /
// thread pool) and fusion_cpu_extra_ops.cpp (the extra single-node op kernels:
// SkipSimplifiedLayerNormalization, SimplifiedLayerNormalization, Gather, and
// the small integer attention-mask-prep ops). The goal is to let the EP claim
// *all* remaining model ops so that every intra-op parallel region is driven by
// the one FusionThreadPool (via FusionParallelFor) -- no second (ORT intra-op)
// pool participates.
//
// The seam deliberately uses only C ABI types (OrtNode*, OrtStatus, ...) so this
// header does not need the C++ Ort wrapper (which requires ORT_API_MANUAL_INIT
// ordering). fusion_cpu_extra_ops.cpp uses the C++ wrapper internally.
//
// fusion_cpu_ep.cpp stays intact except for: (1) this include, (2) the
// FusionParallelFor definition, and (3) two one-line dispatch hooks
// (ExtraTryClaim in the GetCapability node loop, ExtraTryCompile in the
// single-node Compile branch). The generic FusionCpuNodeComputeInfo trampoline
// and its lifetime (ReleaseNodeComputeInfos) remain owned by fusion_cpu_ep.cpp:
// the extra ops only supply the kernel (stored in FusionCpuEp::extra_kernels_)
// and the two dispatch function pointers.

#pragma once

#include "fusion_cpu_ep.h"  // FusionCpuEp, ExtraOpKernel, C API types

#include <functional>
#include <string>

namespace fusion_cpu_ep {

// Dispatch function-pointer signatures shared with fusion_cpu_ep.cpp's
// FusionCpuNodeComputeInfo trampoline (same shape as its StateLookupFn /
// ComputeFn). All extra kernels reuse a single pair (LookupExtraKernel /
// ComputeExtraKernel) since they share the polymorphic ExtraOpKernel base.
using ExtraLookupFn  = OrtStatus* (*)(FusionCpuEp& ep, const std::string& node_name, void** out_state);
using ExtraComputeFn = OrtStatus* (*)(void* state, OrtKernelContext* kctx);

// GetCapability hook: if `node` is an extra op this EP can execute, claim it as
// a single-node fused subgraph and return true. Called once per graph node.
bool ExtraTryClaim(FusionCpuEp& ep, const OrtNode* node, OrtEpGraphSupportInfo* info);

// Compile hook for single-node fused subgraphs: if `inner_node` is an extra op,
// build its kernel (stored in ep.extra_kernels()), set *out_lookup/*out_compute
// to the shared dispatch pair, and return true. fusion_cpu_ep.cpp then wraps
// them in its FusionCpuNodeComputeInfo.
bool ExtraTryCompile(FusionCpuEp& ep, const OrtNode* fused_node, const OrtNode* inner_node,
                     ExtraLookupFn* out_lookup, ExtraComputeFn* out_compute);

// Thin wrapper over the single FusionThreadPool (MatMulNBitsThreadPool()),
// defined in fusion_cpu_ep.cpp. `fn(begin, end)` is invoked once per worker
// chunk with a half-open [begin, end) sub-range of [0, count). This is the ONLY
// shared symbol the extra ops need from the EP shell, and it guarantees the
// extra kernels run on the same pool as MatMulNBits/GQA (no second pool).
void FusionParallelFor(int count, const std::function<void(int begin, int end)>& fn);

}  // namespace fusion_cpu_ep
