cmake_minimum_required(VERSION 3.19)

if (NOT WIN32)
    message(WARNING "ggml-dx12: DirectX 12 backend is only supported on Windows")
    return()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/MP)
endif()

ggml_add_backend_library(ggml-dx12
    ggml-dx12.cpp
    ../../include/ggml-dx12.h
)

# Link D3D12 and DXGI system libraries
target_link_libraries(ggml-dx12 PRIVATE
    d3d12
    dxgi
    dxguid
)

target_include_directories(ggml-dx12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Debug options
if (GGML_DX12_DEBUG)
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_DEBUG)
endif()

# ---------------------------------------------------------------------------
# D3D12 LinAlg Matrix preview (SM 6.10) — Phase 0 opt-in scaffolding
# ---------------------------------------------------------------------------
# When GGML_DX12_LINALG_PREVIEW=ON, the build expects the user to supply:
#   - GGML_DX12_AGILITY_SDK_PATH : extracted Agility SDK 1.721.3 preview root
#                                  (must contain build/native/bin/x64/{D3D12Core,
#                                  d3d12SDKLayers}.dll). Get the nupkg from
#                                  https://devblogs.microsoft.com/directx/directx12agility/
#   - GGML_DX12_DXC_PATH         : path to dxc.exe 1.10.2605.24+ (and its
#                                  matching dxcompiler.dll/dxil.dll, plus the
#                                  dx/linalg.h header under inc/ or include/hlsl).
#                                  1.10.2605.24 is the minimum for the LinAlg
#                                  path: earlier validators reject the LinAlg
#                                  matrix types emitted by the FA shader
#                                  (DirectXShaderCompiler issue 8433).
#                                  Get from:
#                                  https://github.com/microsoft/DirectXShaderCompiler/releases
#
# When OFF (default), the build is unchanged. The LinAlg probe shader is
# skipped, no Agility SDK DLLs are copied, and the runtime LinAlg cap query
# is compiled out (returns "no" in the device banner).
option(GGML_DX12_LINALG_PREVIEW "Enable D3D12 LinAlg Matrix preview (SM 6.10, requires Agility SDK 1.721.3 preview + DXC 1.10.2605.24)" OFF)
set(GGML_DX12_AGILITY_SDK_PATH    "" CACHE PATH   "Path to Agility SDK preview root (build/native/bin/x64/D3D12Core.dll)")
set(GGML_DX12_AGILITY_SDK_VERSION "721" CACHE STRING "D3D12SDKVersion exported by the loader (Agility SDK package minor, e.g. 721 for 1.721.3, 720 for 1.720.0)")
set(GGML_DX12_DXC_PATH            "" CACHE FILEPATH "Path to dxc.exe 1.10.2605.24+ (preview compiler with cs_6_10 and dx/linalg.h support)")
set(GGML_DX12_DXC_HLSL_INCLUDE    "" CACHE PATH   "Path to the preview DXC HLSL include dir containing dx/linalg.h (e.g. <dxc>/inc or <dxc>/include/hlsl)")

# Optional: use the preview DXC compiler for *all* production shaders too
# (not just the LinAlg probe). Independent of GGML_DX12_LINALG_PREVIEW —
# this only swaps the compiler, target stays cs_6_6, no runtime changes.
#
# Why opt-in: on Intel Arc B390 May 2026 driver, A/B benchmarks showed
# DXC 1.10.2605.2 vs Windows SDK DXC 1.9.0.5180 gives ~+3.7% on Phi-3 F16
# prompt processing, no regression on quantized models. Driver-agnostic
# pure-codegen gain. Off by default since the Windows SDK DXC is the
# supported/shipping toolchain.
option(GGML_DX12_USE_PREVIEW_DXC "Compile all DX12 shaders with the preview DXC (set via GGML_DX12_DXC_PATH)" OFF)
if (GGML_DX12_USE_PREVIEW_DXC)
    if (NOT GGML_DX12_DXC_PATH OR NOT EXISTS "${GGML_DX12_DXC_PATH}")
        message(FATAL_ERROR "ggml-dx12: GGML_DX12_USE_PREVIEW_DXC=ON requires GGML_DX12_DXC_PATH to point at dxc.exe 1.10.2605.24+")
    endif()
    set(DXC_EXECUTABLE "${GGML_DX12_DXC_PATH}" CACHE FILEPATH "" FORCE)
    message(STATUS "ggml-dx12: USE_PREVIEW_DXC ON - all shaders will be compiled with ${GGML_DX12_DXC_PATH}")
endif()

# HLSL Shader Compilation
# Find DXC in Windows SDK
find_program(DXC_EXECUTABLE dxc
    HINTS
        "$ENV{WindowsSdkDir}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
        "$ENV{VULKAN_SDK}/Bin"
)

if (DXC_EXECUTABLE)
    message(STATUS "ggml-dx12: Found DXC: ${DXC_EXECUTABLE}")

    set(SHADER_INPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

    # List of shaders to compile
    set(DX12_SHADERS
        add
        mul
        scale
        sqr
        sqrt_
        silu
        gelu
        gelu_quick
        gelu_erf
        relu
        tanh_
        sigmoid
        mul_sigmoid_gate
        abs
        neg
        sgn
        step
        elu
        hardsigmoid
        hardswish
        floor
        ceil
        round
        trunc
        xielu
        rms_norm
        norm
        group_norm
        soft_max
        soft_max_cached
        argsort_top_k_small
        get_rows_moe_norm
        moe_router
        mul_mat
        mul_mat_q8_0_f32_small_m
        mul_mat_f16_f32_small_m
        mul_mat_id
        mul_mat_id_coop
        mul_mat_id_f16_vec_rows4
        mul_mat_id_f16_vec_wave64
        mul_mat_id_coop_wide
        moe_expert_bucket
        mul_mat_id_q4_0
        mul_mat_id_q4_0_coop
        mul_mat_id_q4_1
        mul_mat_id_q4_1_coop
        mul_mat_id_q5_0
        mul_mat_id_q5_0_coop
        mul_mat_id_q5_1
        mul_mat_id_q5_1_coop
        mul_mat_id_q8_0
        mul_mat_id_q8_0_coop
        moe_sum
        mul_mat_id_q4k
        mul_mat_id_q4k_coop
        mul_mat_id_q4k_block
        mul_mat_id_q5k
        mul_mat_id_q5k_coop
        mul_mat_id_q6k
        mul_mat_id_q6k_coop
        mul_mat_id_q6k_coop_wide
        mul_mat_id_iq4_nl
        mul_mat_id_iq4_nl_coop
        mul_mat_id_mxfp4
        mul_mat_id_mxfp4_coop
        mul_mat_id_nvfp4
        mul_mat_id_nvfp4_coop
        mul_mat_id_q1_0
        mul_mat_id_q1_0_coop
        mul_mat_id_iq4_xs
        mul_mat_id_iq4_xs_coop
        mul_mat_id_iq2_xxs
        mul_mat_id_q2k
        mul_mat_id_q2k_coop
        mul_mat_id_q3k
        mul_mat_id_q3k_coop
        mul_mat_id_iq2_xs
        mul_mat_id_iq2_s
        mul_mat_id_iq3_xxs
        mul_mat_id_iq3_s
        mul_mat_id_iq1_s
        mul_mat_id_iq1_m
        mul_mat_iq2_xxs_quant
        mul_mat_iq2_xs_quant
        mul_mat_iq2_s_quant
        mul_mat_iq3_xxs_quant
        mul_mat_iq3_s_quant
        mul_mat_iq1_s_quant
        mul_mat_iq1_m_quant
        mul_mat_iq4_xs_quant
        mul_mat_mxfp4_quant
        mul_mat_nvfp4_quant
        mul_mat_q1_0_quant
        mul_mat_q2_0_quant
        mul_mat_tq1_0_quant
        mul_mat_tq2_0_quant
        mul_mat_vec_q2_0
        mul_mat_vec_tq1_0
        mul_mat_vec_tq2_0
        mul_mat_id_q2_0
        mul_mat_id_q2_0_coop
        mul_mat_id_tq1_0
        mul_mat_id_tq1_0_coop
        mul_mat_id_tq2_0
        mul_mat_id_tq2_0_coop
        cpy
        cpy_transpose_02
        get_rows
        diag_mask_inf
        rope
        rope_multi
        clamp
        concat
        repeat
        sum_rows
        sub
        div
        sin
        cos
        exp
        expm1
        pad_reflect_1d
        col2im_1d
    cpy_dq_q4_0
    cpy_dq_q4_1
    cpy_dq_q5_0
    cpy_dq_q5_1
    cpy_dq_q8_0
    cpy_dq_q4_k
    cpy_dq_q5_k
    cpy_dq_q6_k
    cpy_dq_iq4_nl
    cpy_dq_mxfp4
    cpy_dq_nvfp4
    cpy_dq_q1_0
    cpy_dq_q2_0
    cpy_dq_tq1_0
    cpy_dq_tq2_0
    cpy_dq_q2_k
    cpy_dq_q3_k
    cpy_dq_iq4_xs
    cpy_dq_iq2_xxs
    cpy_dq_iq2_xs
    cpy_dq_iq2_s
    cpy_dq_iq3_xxs
    cpy_dq_iq3_s
    cpy_dq_iq1_s
    cpy_dq_iq1_m
        softplus
        log
        leaky_relu
        fill
        tri
        diag
        arange
        timestep_embedding
        sum
        mean
        argmax
        argsort
        top_k_radix
        argsort_large
        top_k
        top_k_large
        add_id
        count_equal
        acc
        set
        cumsum
        solve_tri
        pad
        roll
        ssm_conv
        ssm_conv_silu
        ssm_conv_bias_silu
        ssm_scan
        ssm_scan_d96
        ssm_scan_d256
        dsv4_hc_comb
        dsv4_hc_pre
        dsv4_hc_post
        lightning_indexer
        lightning_indexer_q4_0
        lightning_indexer_q4_1
        lightning_indexer_q5_0
        lightning_indexer_q5_1
        lightning_indexer_q8_0
        lightning_indexer_iq4_nl
        wkv6
        wkv7
        gla
        upscale
        im2col
        im2col_3d
        pool_2d
        dequant_q4_k
        flash_attn
        flash_attn_sparse
        flash_attn_sparse_compact
        flash_attn_64
        flash_attn_128
        flash_attn_pf_64
        flash_attn_pf_64_wide
        flash_attn_pf_64_wide_prescan_relaxed_maskclass
        flash_attn_pf_96
        flash_attn_pf_96_br16
        flash_attn_pf_96_br16_prescan
        flash_attn_pf_128
        flash_attn_pf_96_prescan
        flash_attn_pf_128_prescan
        flash_attn_gqa
        flash_attn_reduce
        flash_attn_q4_0
        flash_attn_q4_1
        flash_attn_q5_0
        flash_attn_q5_1
        flash_attn_q8_0
        flash_attn_iq4_nl
        flash_attn_kvmix
        flash_attn_tiled
        flash_attn_tiled16
        set_rows
        rope_set_rows
        rope_scale_k_set_rows
        conv_2d
        conv_2d_dw
        conv_3d
        conv_transpose_1d
        conv_transpose_2d
        cast
        pool_1d
        get_rows_q4k
        get_rows_q5k
        get_rows_q6k
        glu
        l2_norm
        l2_norm_1024
        gated_delta_net
        gated_delta_net_sv16
        gated_delta_net_sv32
        gated_delta_net_sv64
        gated_delta_net_kda
        gated_delta_net_sv16_kda
        gated_delta_net_sv32_kda
        gated_delta_net_sv64_kda
        rms_norm_mul
        rms_norm_mul_1024
        rms_norm_1024
        norm_1024
        rms_norm_mul_quantize_q8_1
        rms_norm_mul_rope
        rms_norm_mul_rope_set_rows
        rms_norm_mul_rope_set_rows_packed
        rms_norm_mul_rope_qk
        rms_norm_mul_rope_qk_packed
        add_rms_norm_mul
        add_rms_norm_mul_fixed
        rms_norm_mul_fixed
        norm_mul_add
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_f16_wave64
        mul_mat_vec_qkv_f16_wave64
        mul_mat_vec_qkv_f16_wave64_rms
        mul_mat_vec_qkv_f16_wave64_rms_rows2
        mul_mat_vec_q4k_mr
        mul_mat_vec_q4k_mr_rms
        mul_mat_vec_q4k_qk
        mul_mat_vec_q5k_mr
        mul_mat_vec_q5k_subgroup
        mul_mat_vec_q6k_mr
        mul_mat_vec_q6k_mr_blocked
        mul_mat_vec_q6k_subgroup
        mul_mat_vec_q3k_mr_blocked
        mul_mat_vec_q2k_mr_blocked
        mul_mat_vec_q8_0_mr64
        mul_mat_vec_q8_0_wave64
        mul_mat_vec_q8_0_wave64_rows2
        mul_mat_vec_qkv_q8_0_wave64_rows2
        mul_mat_vec_qkv_q8_0_wave64_rms
        mul_mat_vec_qkv_q8_0_mr256
        mul_mat_vec_qkv_q8_0_mr256_rms
        mul_mat_vec_qkv_q8_0_dp4a
        mul_mat_vec_glu_q8_0_wave64_rows2
        mul_mat_vec_glu_q8_0_wave64_rms
        mul_mat_vec_q8_0_dp4a_mr64
        mul_mat_vec_q2_0_dp4a_mr64
        mul_mat_vec_q2_0_dp4a_portable
        mul_mat_vec_q4k_dp4a_mr4
        mul_mat_vec_q6k_dp4a_mr4
        mul_mat_vec_q4k_dp4a_nc2
        mul_mat_vec_q4k_dp4a_nc4
    mul_mat_vec_q4k_dp4a_nc2_r4
    mul_mat_vec_q4k_dp4a_nc4_r4
        mul_mat_vec_q4k_dp4a_nc8
        mul_mat_vec_q4k_dp4a_nc16
        mul_mat_vec_q4k_dp4a_nc32
        mul_mat_vec_q5k_dp4a_nc2
        mul_mat_vec_q5k_dp4a_nc4
        mul_mat_vec_q5k_dp4a_nc8
        mul_mat_vec_q5k_dp4a_nc16
        mul_mat_vec_q5k_dp4a_nc32
        mul_mat_vec_q6k_dp4a_nc2
        mul_mat_vec_q6k_dp4a_nc4
        mul_mat_vec_q6k_dp4a_nc8
        mul_mat_vec_q6k_dp4a_nc16
        mul_mat_vec_q6k_dp4a_nc32
        mul_mat_vec_q8_0_dp4a_nc2
        mul_mat_vec_q8_0_dp4a_nc4
        mul_mat_vec_q8_0_dp4a_nc8
        mul_mat_vec_q8_0_dp4a_nc16
        mul_mat_vec_q8_0_dp4a_nc32
        mul_mat_vec_q5_0_mr64
        mul_mat_vec_q5_0_mr
        mul_mat_vec_q5_0_nc8
        mul_mat_vec_q5_0_mr_lds
        mul_mat_vec_q5_0_subgroup
        mul_mat_vec_q5_0_vulkan_rows2
        mul_mat_vec_qkv_q5_0_vulkan_rows2
        mul_mat_vec_qkv_q5_0_vulkan_rows2_rms
        mul_mat_vec_qkv_q5_0_q8_0_rows2
        mul_mat_vec_qkv_q5_0_q8_0_rows2_rms
        mul_mat_vec_glu_q5_0_subgroup
        mul_mat_vec_glu_q5_0_subgroup_rms
        mul_mat_vec_glu_q5_0_vulkan_rows2
        mul_mat_vec_glu_q5_0_vulkan_rows2_rms
        mul_mat_vec_glu_q5_0
        mul_mat_vec_glu_q4_k
        mul_mat_vec_glu_q4_k_wave64
        mul_mat_vec_glu_q4_k_wave64_rms
        mul_mat_vec_glu_q4_k_rms
        mul_mat_vec_glu_q5_k
        mul_mat_vec_glu_q3_k
        mul_mat_vec_glu_q8_0
        mul_mat_vec_glu_q8_0_rms
        mul_mat_vec_glu_q8_0_dp4a_mr64
        mul_mat_vec_glu_q8_0_dp4a_block64
        mul_mat_vec_q5_1_mr
        mul_mat_vec_q8_0_mr
        mul_mat_vec_q8_0_mr256
        mul_mat_vec_q8_0_mr256v
        mul_mat_vec_glu
        mul_mat_vec_glu_rms
        mul_mat_vec_mr32
        mul_mat_q4_0
        mul_mat_q4_1
        mul_mat_q5_0
        mul_mat_q5_1
        mul_mat_q8_0
        mul_mat_q8_1
        get_rows_q4_0
        get_rows_q4_1
        get_rows_q5_0
        get_rows_q5_1
        get_rows_q8_0
        get_rows_q8_1
        mul_mat_q2k
        mul_mat_q3k
        mul_mat_vec_q2k
        mul_mat_vec_q3k
        mul_mat_vec_q2k_mr
        mul_mat_vec_q3k_mr
        get_rows_q2k
        get_rows_q3k
        mul_mat_iq4_nl
        mul_mat_vec_iq4_nl
        mul_mat_vec_iq4_nl_mr
        mul_mat_vec_iq4_nl_mr256
        get_rows_iq4_nl
        mul_mat_vec_mxfp4
        get_rows_iq1_m
        get_rows_iq1_s
        get_rows_iq2_s
        get_rows_iq2_xs
        get_rows_iq2_xxs
        get_rows_iq3_s
        get_rows_iq3_xxs
        get_rows_iq4_xs
        get_rows_mxfp4
        get_rows_nvfp4
        get_rows_q1_0
        get_rows_q2_0
        get_rows_tq1_0
        get_rows_tq2_0
        mul_mat_vec_iq2_xxs
        mul_mat_vec_iq2_xxs_mr
        mul_mat_vec_iq4_xs
        mul_mat_vec_iq3_xxs
        mul_mat_vec_iq2_xs
        mul_mat_vec_iq2_s
        mul_mat_vec_iq3_s
        mul_mat_vec_iq1_s
        mul_mat_vec_iq1_m
        mul_mat_wmma
        mul_mat_f32_router
        mul_mat_wmma64
        mul_mat_wmma_kfull
        mul_mat_q4k_wmma
        mul_mat_q4k_wmma_lds
        mul_mat_q4k_wmma_lds64
        mul_mat_q5k_wmma
        mul_mat_q6k_wmma
        mul_mat_q8_0_wmma
        mul_mat_q4_0_wmma
        mul_mat_q4_1_wmma
        quantize_q8_1
    )

    # Shaders requiring SM 6.6 (dot4add_i8packed / dp4a — built at 6.6 with rest)
    set(DX12_SHADERS_SM64
        mul_mat_q8_0_q8_1
        mul_mat_q8_0_q8_1_tiled
        mul_mat_q8_0_q8_1_tiled_intel
        mul_mat_q8_0_q8_1_mmq
        mul_mat_q8_0_q8_1_mmq_64
        mul_mat_q4k_q8_1_tiled
        mul_mat_q4k_q8_1_tiled_64
        mul_mat_q4k_q8_1_mmq
        mul_mat_q4k_q8_1_mmq_64
        mul_mat_q2k_q8_1_mmq
        mul_mat_q5k_q8_1_tiled
        mul_mat_q5k_q8_1_mmq
        mul_mat_q5k_q8_1_mmq_64
        mul_mat_q6k_q8_1_mmq
        mul_mat_q6k_q8_1_mmq_64
        mul_mat_q5_0_q8_1_tiled
        mul_mat_q5_0_q8_1_tiled_64
        mul_mat_q6k_q8_1_tiled
        mul_mat_q6k_q8_1_tiled_64
        mul_mat_vec_q4k_dp4a
        mul_mat_vec_q4k_dp4a_32
        mul_mat_vec_q4k_dp4a_64
        mul_mat_vec_q5k_dp4a
        mul_mat_vec_q5k_dp4a_32
        mul_mat_vec_q6k_dp4a
        mul_mat_vec_q6k_dp4a_nvidia
        mul_mat_vec_q6k_dp4a_packed
        mul_mat_vec_q3k_dp4a
        mul_mat_vec_q8_0_dp4a
        mul_mat_id_q8_0_dp4a
        mul_mat_id_q8_0_dp4a_g64
        mul_mat_id_q8_0_dp4a_rows4
        mul_mat_id_q8_0_q8_1_mmq
        mul_mat_id_q4k_dp4a
        mul_mat_id_q4k_dp4a_rows4
        mul_mat_id_q6k_dp4a_rows4
        mul_mat_vec_q4_0_dp4a
        mul_mat_vec_q4_0_dp4a_mr256
        mul_mat_vec_q4_1_dp4a
        mul_mat_vec_q4_1_dp4a_mr256
        mul_mat_vec_q5_0_dp4a
        mul_mat_vec_q5_0_dp4a_mr256
        mul_mat_vec_q5_1_dp4a
    )

    # Shaders compiled twice: once FP32-only and once with native 16-bit types
    # enabled (-enable-16bit-types + -DNATIVE_FP16=1). Runtime selects the FP16
    # variant on devices that report OPTIONS4.Native16BitShaderOpsSupported.
    set(DX12_SHADERS_FP16_VARIANT
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_mr32
        mul_mat_vec_f16_wave64
        mul_mat_vec_qkv_f16_wave64
        mul_mat_vec_qkv_f16_wave64_rms
        mul_mat_vec_qkv_f16_wave64_rms_rows2
        mul_mat_vec_glu
        mul_mat_vec_glu_rms
        flash_attn
        flash_attn_gqa
        flash_attn_tiled
        flash_attn_tiled16
        flash_attn_pf_64
        flash_attn_pf_64_wide
        flash_attn_pf_64_wide_prescan_relaxed_maskclass
        flash_attn_pf_96
        flash_attn_pf_96_br16
        flash_attn_pf_96_br16_prescan
        flash_attn_pf_128
        flash_attn_pf_96_prescan
        flash_attn_pf_128_prescan
    )

    # Shaders that REQUIRE native 16-bit shader ops (compiled with
    # -enable-16bit-types) and produce the standard `g_<name>_w<WS>_dxil`
    # symbol (no _fp16 suffix). Use when a shader has no FP32 fallback —
    # for example, when packing 8-bit values via uint16_t stores at
    # non-4-aligned addresses. Host MUST gate dispatch on
    # OPTIONS4.Native16BitShaderOpsSupported (dx12_device::fp16_supported)
    # so devices without 16-bit support fall back via supports_op.
    set(DX12_SHADERS_FP16_ONLY
        flash_attn_cd_64
        flash_attn_cd_96
        flash_attn_cd_128
        flash_attn_cd_q8_0_64
        flash_attn_cd_q8_0_96
        flash_attn_cd_q8_0_128
        flash_attn_cd_gqa_64
        flash_attn_cd_gqa_96
        flash_attn_cd_gqa_128
        flash_attn_cd_gqa_q8_0_64
        flash_attn_cd_gqa_q8_0_96
        flash_attn_cd_gqa_q8_0_128
        flash_attn_tiled_192
        flash_attn_tiled_256
        flash_attn_tiled_320
        flash_attn_tiled_512
        flash_attn_tiled_576
        set_rows_q8_0
        set_rows_q4_0
        set_rows_q4_1
        set_rows_q5_0
        set_rows_q5_1
        set_rows_iq4_nl
        cpy_quant_block
        concat_block
        repeat_block
        mul_mat_q5_0_q8_1_mmq
        mul_mat_q5_0_q8_1_mmq_wave_share
        mul_mat_wmma_fp16
        mul_mat_id_gemm
        mul_mat_id_gemm_q4_0
        mul_mat_id_gemm_q4_1
        mul_mat_id_gemm_q5_0
        mul_mat_id_gemm_q5_1
        mul_mat_id_gemm_q8_0
        mul_mat_id_gemm_q2k
        mul_mat_id_gemm_q3k
        mul_mat_id_gemm_q4k
        mul_mat_id_gemm_q5k
        mul_mat_id_gemm_q6k
        mul_mat_id_gemm_iq4_nl
        mul_mat_id_gemm_iq4_xs
        mul_mat_id_gemm_mxfp4
        mul_mat_id_gemm_nvfp4
        mul_mat_id_gemm_q1_0
        mul_mat_id_gemm_q2_0
        mul_mat_id_gemm_tq1_0
        mul_mat_id_gemm_tq2_0
        mul_mat_id_gemm_iq2_xxs
        mul_mat_id_gemm_iq2_xs
        mul_mat_id_gemm_iq2_s
        mul_mat_id_gemm_iq3_xxs
        mul_mat_id_gemm_iq3_s
        mul_mat_id_gemm_iq1_s
        mul_mat_id_gemm_iq1_m
        mul_mat_id_gemm_tall_iq1_m
        mul_mat_id_gemm_tall_iq1_s
        mul_mat_id_gemm_tall_iq2_s
        mul_mat_id_gemm_tall_iq2_xs
        mul_mat_id_gemm_tall_iq2_xxs
        mul_mat_id_gemm_tall_iq3_s
        mul_mat_id_gemm_tall_iq3_xxs
        mul_mat_id_gemm_tall_iq4_nl
        mul_mat_id_gemm_tall_iq4_xs
        mul_mat_id_gemm_tall_mxfp4
        mul_mat_id_gemm_tall_nvfp4
        mul_mat_id_gemm_tall_q1_0
        mul_mat_id_gemm_tall_q2_0
        mul_mat_id_gemm_tall_q2k
        mul_mat_id_gemm_tall_q3k
        mul_mat_id_gemm_tall_q4_0
        mul_mat_id_gemm_tall_q4_1
        mul_mat_id_gemm_tall_q4k
        mul_mat_id_gemm_tall_q5_0
        mul_mat_id_gemm_tall_q5_1
        mul_mat_id_gemm_tall_q5k
        mul_mat_id_gemm_tall_q6k
        mul_mat_id_gemm_tall_q8_0
        mul_mat_id_gemm_tall_tq1_0
        mul_mat_id_gemm_tall_tq2_0
        mul_mat_gemm_iq2_xxs
        mul_mat_gemm_iq2_xs
        mul_mat_gemm_iq2_s
        mul_mat_gemm_iq3_xxs
        mul_mat_gemm_iq3_s
        mul_mat_gemm_iq1_s
        mul_mat_gemm_iq1_m
        mul_mat_gemm_iq4_xs
        mul_mat_gemm_q2_0
        mul_mat_gemm_tq1_0
        mul_mat_gemm_tq2_0
        mul_mat_gemm_mxfp4
        mul_mat_gemm_nvfp4
        mul_mat_gemm_q1_0
    )

    set(DX12_SHADER_HEADERS "")

    # Common DXC flags. Shaders that use Inf/NaN-sensitive control flow are
    # listed below and compiled without -ffinite-math-only.
    set(DXC_COMMON_FLAGS -O3 -ffinite-math-only -Qstrip_debug -Qstrip_reflect)

    # Shaders that must NOT use -ffinite-math-only (NaN/Inf sensitivity)
    set(DX12_SHADERS_NO_FINITE_MATH
        flash_attn
        flash_attn_64
        flash_attn_128
        flash_attn_tiled
        flash_attn_tiled16
        flash_attn_tiled_192
        flash_attn_tiled_256
        flash_attn_tiled_320
        flash_attn_tiled_512
        flash_attn_tiled_576
        flash_attn_pf_64
        flash_attn_pf_64_wide
        flash_attn_pf_64_wide_prescan_relaxed_maskclass
        flash_attn_pf_96
        flash_attn_pf_96_br16
        flash_attn_pf_96_br16_prescan
        flash_attn_pf_128
        flash_attn_pf_96_prescan
        flash_attn_pf_128_prescan
        flash_attn_gqa
        flash_attn_reduce
        flash_attn_q4_0
        flash_attn_q4_1
        flash_attn_q5_0
        flash_attn_q5_1
        flash_attn_q8_0
        flash_attn_iq4_nl
        flash_attn_kvmix
        flash_attn_cd_64
        flash_attn_cd_96
        flash_attn_cd_128
        flash_attn_cd_q8_0_64
        flash_attn_cd_q8_0_96
        flash_attn_cd_q8_0_128
        flash_attn_cd_gqa_64
        flash_attn_cd_gqa_96
        flash_attn_cd_gqa_128
        flash_attn_cd_gqa_q8_0_64
        flash_attn_cd_gqa_q8_0_96
        flash_attn_cd_gqa_q8_0_128
    )

    # Compile each shader for wave sizes 16, 32, 64
    # At runtime, the device selects the matching set based on GPU wave size
    set(DX12_WAVE_SIZES 16 32 64)

    # Threadgroup-size tuning variants. The listed matvec shaders treat
    # GROUP_SIZE as a pure threadgroup-size knob (work-split + reduction width),
    # so each extra size is a correct, drop-in variant. The default (64) blob is
    # still built by the main loop; these are the additional sizes a device can
    # be pointed at via DX12_MMV_GROUP_SIZE for vendor sweeps. Symbols follow
    # g_<shader>_g<GS>_w<WS>_dxil so WBLOB_GS() can select them at runtime.
    set(DX12_MMV_TUNE_SHADERS
        mul_mat_vec_q5_0_subgroup
        mul_mat_vec_glu_q5_0_subgroup
        mul_mat_vec_glu_q5_0_subgroup_rms
        mul_mat_vec_glu_q8_0_dp4a_mr64
        mul_mat_vec_q6k_subgroup
        mul_mat_vec_q8_0_dp4a
        mul_mat_vec_q4k_mr
        mul_mat_vec_q4k_mr_rms
        mul_mat_vec_q4k_qk
        mul_mat_vec_q6k_dp4a
        mul_mat_vec_q6k_dp4a_nvidia
        mul_mat_vec_glu_q4_k
        mul_mat_vec_glu_q4_k_rms
    )
    set(DX12_MMV_TUNE_GROUP_SIZES 16 32 64 128 256 512)

    # Shader files that are #included by other shaders (transitive deps).
    # Add to DEPENDS so changes trigger rebuilds of dependent shaders.
    set(SHADER_INCLUDE_DEPS
        ${SHADER_INPUT_DIR}/ggml_common.hlsli
        ${SHADER_INPUT_DIR}/get_rows_quant.hlsli
        ${SHADER_INPUT_DIR}/cpy_quant_f32.hlsli
        ${SHADER_INPUT_DIR}/quant_dequant_kv.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_gemm_quant.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_gemm.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant_coop.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_quant.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_quant.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q8_0_dp4a_nc.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q4k_dp4a_nc.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q5k_dp4a_nc.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q6k_dp4a_nc.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q6k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/quant_dequant.hlsli
        ${SHADER_INPUT_DIR}/rope_yarn.hlsli
        ${SHADER_INPUT_DIR}/flash_attn.hlsl
        ${SHADER_INPUT_DIR}/flash_attn_tiled.hlsl
        ${SHADER_INPUT_DIR}/flash_attn_cd.hlsli
        ${SHADER_INPUT_DIR}/flash_attn_cd_gqa.hlsli
        ${SHADER_INPUT_DIR}/flash_attn_pf.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q4k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q5k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q6k_mr_blocked.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl
        ${SHADER_INPUT_DIR}/gated_delta_net_impl.hlsli
        ${SHADER_INPUT_DIR}/lightning_indexer.hlsli
        ${SHADER_INPUT_DIR}/ssm_conv_impl.hlsli
        ${SHADER_INPUT_DIR}/ssm_scan_body.hlsli
        ${SHADER_INPUT_DIR}/iq2xs_grid.hlsli
        ${SHADER_INPUT_DIR}/iq2s_grid.hlsli
        ${SHADER_INPUT_DIR}/iq3xxs_grid.hlsli
        ${SHADER_INPUT_DIR}/iq3s_grid.hlsli
        ${SHADER_INPUT_DIR}/iq1_grid.hlsli
    )

    foreach(WSIZE ${DX12_WAVE_SIZES})
        foreach(shader_name ${DX12_SHADERS})
            if(shader_name STREQUAL "rms_norm_mul_rope_qk_packed" AND NOT (WSIZE EQUAL 16 OR WSIZE EQUAL 32))
                continue()
            endif()
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")
            set(shader_deps ${shader_src} ${SHADER_INCLUDE_DEPS})
            if(shader_name STREQUAL "rms_norm_mul_rope_set_rows_packed")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/rms_norm_mul_rope_set_rows.hlsl)
            elseif(shader_name STREQUAL "rms_norm_mul_fixed")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/add_rms_norm_mul_fixed.hlsl)
            elseif(shader_name STREQUAL "rms_norm_mul_rope_qk_packed")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/rms_norm_mul_rope_qk.hlsl)
            elseif(shader_name STREQUAL "mul_mat_f16_f32_small_m")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q8_0_f32_small_m.hlsl)
            elseif(shader_name STREQUAL "mul_mat_vec_glu_q4_k_wave64_rms")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_vec_glu_q4_k_wave64.hlsl)
            elseif(shader_name STREQUAL "mul_mat_vec_qkv_f16_wave64_rms" OR
                   shader_name STREQUAL "mul_mat_vec_qkv_f16_wave64_rms_rows2")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_vec_qkv_f16_wave64.hlsl)
            endif()

            # Check if this shader needs safe math
            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -D WAVE_SIZE=${WSIZE}
                    ${shader_src}
                DEPENDS ${shader_deps}
                COMMENT "Compiling HLSL shader: ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()

        foreach(shader_name ${DX12_SHADERS_SM64})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")
            set(shader_deps ${shader_src} ${SHADER_INCLUDE_DEPS})
            if(shader_name STREQUAL "mul_mat_q8_0_q8_1_tiled_intel")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q8_0_q8_1_tiled.hlsl)
            elseif(shader_name STREQUAL "mul_mat_q8_0_q8_1_mmq_64")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q8_0_q8_1_mmq.hlsl)
            elseif(shader_name STREQUAL "mul_mat_q4k_q8_1_mmq_64")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q4k_q8_1_mmq.hlsl)
            elseif(shader_name STREQUAL "mul_mat_q5k_q8_1_mmq_64")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q5k_q8_1_mmq.hlsl)
            elseif(shader_name STREQUAL "mul_mat_q6k_q8_1_mmq_64")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q6k_q8_1_mmq.hlsl)
            elseif(shader_name STREQUAL "mul_mat_id_q8_0_dp4a_rows4")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_id_q8_0_dp4a.hlsl)
            endif()

            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -D WAVE_SIZE=${WSIZE}
                    ${shader_src}
                DEPENDS ${shader_deps}
                COMMENT "Compiling HLSL shader (SM 6.6): ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()

        # Threadgroup-size tuning variants (extra GROUP_SIZE builds).
        foreach(shader_name ${DX12_MMV_TUNE_SHADERS})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")

            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            foreach(GS ${DX12_MMV_TUNE_GROUP_SIZES})
                # A threadgroup can't be smaller than one wave, so clamp the
                # compiled GROUP_SIZE up to the wave size. The blob keeps the
                # requested-size name (g${GS}) so WBLOB_GS() always resolves;
                # on such devices the sub-wave size just aliases the wave-sized
                # group (the sweep script skips these duplicates).
                set(EFF_GS ${GS})
                if(${GS} LESS ${WSIZE})
                    set(EFF_GS ${WSIZE})
                endif()
                set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_g${GS}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${shader_hdr}
                    COMMAND ${DXC_EXECUTABLE}
                        -T cs_6_6
                        -E main
                        -Fh ${shader_hdr}
                        -Vn g_${shader_name}_g${GS}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        ${SHADER_DXC_FLAGS}
                        -D WAVE_SIZE=${WSIZE}
                        -D GROUP_SIZE=${EFF_GS}
                        ${shader_src}
                    DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader: ${shader_name} (wave=${WSIZE} group=${GS})"
                )
                list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
            endforeach()
        endforeach()

        # FP16 variant: same source, compiled with -enable-16bit-types and
        # NATIVE_FP16=1 so the shader can opt into native float16_t paths via
        # `#ifdef NATIVE_FP16`. Runtime picks this blob when the device reports
        # OPTIONS4.Native16BitShaderOpsSupported.
        foreach(shader_name ${DX12_SHADERS_FP16_VARIANT})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_fp16_dxil.h")
            set(shader_fp16_deps "")
            if(shader_name STREQUAL "mul_mat_vec_qkv_f16_wave64_rms" OR
               shader_name STREQUAL "mul_mat_vec_qkv_f16_wave64_rms_rows2")
                list(APPEND shader_fp16_deps ${SHADER_INPUT_DIR}/mul_mat_vec_qkv_f16_wave64.hlsl)
            endif()

            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_fp16_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    ${shader_src}
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS} ${shader_fp16_deps}
                COMMENT "Compiling HLSL shader (FP16): ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()

        # FP16-only shaders: compiled once with -enable-16bit-types, produce
        # the standard `g_<name>_w<WS>_dxil` symbol so they slot into WBLOB()
        # without a special selector. Host must gate dispatch on
        # OPTIONS4.Native16BitShaderOpsSupported via supports_op.
        foreach(shader_name ${DX12_SHADERS_FP16_ONLY})
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")
            set(shader_deps ${shader_src} ${SHADER_INCLUDE_DEPS})
            if(shader_name STREQUAL "mul_mat_q5_0_q8_1_mmq_wave_share")
                list(APPEND shader_deps ${SHADER_INPUT_DIR}/mul_mat_q5_0_q8_1_mmq.hlsl)
            endif()

            set(SHADER_DXC_FLAGS ${DXC_COMMON_FLAGS})
            if(shader_name IN_LIST DX12_SHADERS_NO_FINITE_MATH)
                list(REMOVE_ITEM SHADER_DXC_FLAGS -ffinite-math-only)
            endif()

            add_custom_command(
                OUTPUT ${shader_hdr}
                COMMAND ${DXC_EXECUTABLE}
                    -T cs_6_6
                    -E main
                    -Fh ${shader_hdr}
                    -Vn g_${shader_name}_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    ${SHADER_DXC_FLAGS}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    ${shader_src}
                DEPENDS ${shader_deps}
                COMMENT "Compiling HLSL shader (FP16-only): ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
        endforeach()
    endforeach()

    # Generate a master include header
    set(SHADER_MASTER_HEADER "${SHADER_OUTPUT_DIR}/ggml_dx12_shaders.h")
    set(SHADER_INCLUDES "")
    foreach(WSIZE ${DX12_WAVE_SIZES})
        foreach(shader_name ${DX12_SHADERS})
            if(shader_name STREQUAL "rms_norm_mul_rope_qk_packed" AND NOT (WSIZE EQUAL 16 OR WSIZE EQUAL 32))
                continue()
            endif()
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(shader_name ${DX12_SHADERS_SM64})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(shader_name ${DX12_MMV_TUNE_SHADERS})
            foreach(GS ${DX12_MMV_TUNE_GROUP_SIZES})
                string(APPEND SHADER_INCLUDES "#include \"${shader_name}_g${GS}_w${WSIZE}_dxil.h\"\n")
            endforeach()
        endforeach()
        foreach(shader_name ${DX12_SHADERS_FP16_VARIANT})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_fp16_dxil.h\"\n")
        endforeach()
        foreach(shader_name ${DX12_SHADERS_FP16_ONLY})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_dxil.h\"\n")
        endforeach()
    endforeach()
    file(WRITE "${SHADER_MASTER_HEADER}" "// Auto-generated shader includes\n#pragma once\n${SHADER_INCLUDES}")

    foreach(FA_Q_DIM 64 128)
        if(FA_Q_DIM EQUAL 64)
            set(FA_Q_SOURCE "flash_attn_pf_64_wide_prescan_relaxed_maskclass.hlsl")
        else()
            set(FA_Q_SOURCE "flash_attn_pf_${FA_Q_DIM}_prescan.hlsl")
        endif()
        set(FA_Q_NAME "flash_attn_pf_qregs_${FA_Q_DIM}")
        set(FA_Q_HDR "${SHADER_OUTPUT_DIR}/${FA_Q_NAME}_dxil.h")
        add_custom_command(
            OUTPUT ${FA_Q_HDR}
            COMMAND ${DXC_EXECUTABLE}
                -T cs_6_6 -E main -Fh ${FA_Q_HDR} -Vn g_${FA_Q_NAME}_dxil
                -I ${SHADER_INPUT_DIR} -enable-16bit-types
                -D NATIVE_FP16=1 -D WAVE_SIZE=16 -D FA_PF_Q_REGS=1
                -O3 -Qstrip_debug -Qstrip_reflect
                "${SHADER_INPUT_DIR}/${FA_Q_SOURCE}"
            DEPENDS "${SHADER_INPUT_DIR}/${FA_Q_SOURCE}" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader: ${FA_Q_NAME}"
            VERBATIM)
        target_sources(ggml-dx12 PRIVATE ${FA_Q_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_Q_NAME}_dxil.h\"\n")
    endforeach()

    foreach(FA_MASK_VARIANT RANGE 0 4)
        set(FA_MASK_Q_DEFINES "")
        if(FA_MASK_VARIANT EQUAL 0 OR FA_MASK_VARIANT EQUAL 3)
            set(FA_MASK_SOURCE "flash_attn_pf_64_wide_prescan_relaxed_maskclass.hlsl")
        elseif(FA_MASK_VARIANT EQUAL 1)
            set(FA_MASK_SOURCE "flash_attn_pf_96_prescan.hlsl")
        else()
            set(FA_MASK_SOURCE "flash_attn_pf_128_prescan.hlsl")
        endif()
        if(FA_MASK_VARIANT GREATER_EQUAL 3)
            set(FA_MASK_Q_DEFINES -D FA_PF_Q_REGS=1)
        endif()
        set(FA_MASK_NAME "flash_attn_pf_mask_${FA_MASK_VARIANT}")
        set(FA_MASK_HDR "${SHADER_OUTPUT_DIR}/${FA_MASK_NAME}_dxil.h")
        add_custom_command(
            OUTPUT ${FA_MASK_HDR}
            COMMAND ${DXC_EXECUTABLE}
                -T cs_6_6 -E main -Fh ${FA_MASK_HDR} -Vn g_${FA_MASK_NAME}_dxil
                -I ${SHADER_INPUT_DIR} -enable-16bit-types
                -D NATIVE_FP16=1 -D WAVE_SIZE=16 -D FA_PF_GRAPH_MASK=1 ${FA_MASK_Q_DEFINES}
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/${FA_MASK_SOURCE}"
            DEPENDS "${SHADER_INPUT_DIR}/${FA_MASK_SOURCE}" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader: ${FA_MASK_NAME}"
            VERBATIM)
        target_sources(ggml-dx12 PRIVATE ${FA_MASK_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_MASK_NAME}_dxil.h\"\n")
    endforeach()

    foreach(FA_MASK_BR 16 32)
        set(FA_MASK_NAME "flash_attn_mask_packed_br${FA_MASK_BR}_bc32_w16")
        set(FA_MASK_HDR "${SHADER_OUTPUT_DIR}/${FA_MASK_NAME}_dxil.h")
        add_custom_command(
            OUTPUT ${FA_MASK_HDR}
            COMMAND ${DXC_EXECUTABLE}
                -T cs_6_6 -E main -Fh ${FA_MASK_HDR} -Vn g_${FA_MASK_NAME}_dxil
                -I ${SHADER_INPUT_DIR} -D WAVE_SIZE=16 -D FA_BR=${FA_MASK_BR} -D FA_BC=32 -D FA_MASK_SCALAR=1
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader: ${FA_MASK_NAME}"
            VERBATIM)
        target_sources(ggml-dx12 PRIVATE ${FA_MASK_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_MASK_NAME}_dxil.h\"\n")
    endforeach()

    target_sources(ggml-dx12 PRIVATE ${DX12_SHADER_HEADERS})
    target_include_directories(ggml-dx12 PRIVATE ${SHADER_OUTPUT_DIR})
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_SHADERS_COMPILED)

    # -----------------------------------------------------------------------
    # LinAlg preview: SM 6.10 probe shader compiled with override DXC
    # -----------------------------------------------------------------------
    # We compile the probe to a single wave-size-agnostic blob (LinAlg
    # Matrix is wave-portable by design). The resulting `g_linalg_probe_dxil`
    # symbol is included in the master shader header so the runtime can use
    # its size as a Phase-0 sanity signal even before any dispatch happens.
    if (GGML_DX12_LINALG_PREVIEW)
        if (NOT GGML_DX12_DXC_PATH OR NOT EXISTS "${GGML_DX12_DXC_PATH}")
            message(FATAL_ERROR "ggml-dx12: GGML_DX12_LINALG_PREVIEW=ON but GGML_DX12_DXC_PATH is not set or does not exist. "
                "Download DXC 1.10.2605.24+ from https://github.com/microsoft/DirectXShaderCompiler/releases "
                "and set -DGGML_DX12_DXC_PATH=<path-to>/dxc.exe")
        endif()
        if (NOT GGML_DX12_DXC_HLSL_INCLUDE OR NOT EXISTS "${GGML_DX12_DXC_HLSL_INCLUDE}/dx/linalg.h")
            message(FATAL_ERROR "ggml-dx12: GGML_DX12_LINALG_PREVIEW=ON but dx/linalg.h was not found at "
                "${GGML_DX12_DXC_HLSL_INCLUDE}/dx/linalg.h. "
                "Set -DGGML_DX12_DXC_HLSL_INCLUDE=<path-to>/inc (or include/hlsl) from the DXC preview drop.")
        endif()

        set(LINALG_PROBE_SRC "${SHADER_INPUT_DIR}/linalg_probe.hlsl")
        set(LINALG_PROBE_HDR "${SHADER_OUTPUT_DIR}/linalg_probe_dxil.h")
        add_custom_command(
            OUTPUT ${LINALG_PROBE_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10
                -E main
                -Fh ${LINALG_PROBE_HDR}
                -Vn g_linalg_probe_dxil
                -I ${SHADER_INPUT_DIR}
                -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types
                -O3 -Qstrip_debug -Qstrip_reflect
                ${LINALG_PROBE_SRC}
            DEPENDS ${LINALG_PROBE_SRC}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): linalg_probe"
        )
        target_sources(ggml-dx12 PRIVATE ${LINALG_PROBE_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"linalg_probe_dxil.h\"\n")

        # Integer wave-matrix probe. ComponentType::I8 packs four elements per
        # scalar, so an I8 x I8 -> I32 GEMM would need no more LDS than the
        # dp4a kernel while running on the matrix cores. DXC accepting it says
        # nothing about the driver, so this exists to be turned into a PSO at
        # device init (DX12_LINALG_CAPS=1) before any kernel is written.
        set(LINALG_PROBE_I8_SRC "${SHADER_INPUT_DIR}/linalg_probe_i8.hlsl")
        set(LINALG_PROBE_I8_HDR "${SHADER_OUTPUT_DIR}/linalg_probe_i8_dxil.h")
        add_custom_command(
            OUTPUT ${LINALG_PROBE_I8_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10
                -E main
                -Fh ${LINALG_PROBE_I8_HDR}
                -Vn g_linalg_probe_i8_dxil
                -I ${SHADER_INPUT_DIR}
                -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types
                -O3 -Qstrip_debug -Qstrip_reflect
                ${LINALG_PROBE_I8_SRC}
            DEPENDS ${LINALG_PROBE_I8_SRC}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): linalg_probe_i8"
        )
        target_sources(ggml-dx12 PRIVATE ${LINALG_PROBE_I8_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"linalg_probe_i8_dxil.h\"\n")

        # Intel Xe3 wave GEMM. Separate from the mul_mat_linalg_f16 family
        # below because that family targets a shape this hardware does not
        # have: it wants a 16x16x16 tile with an F32 accumulator, and Xe3
        # offers 8x16x16 with an F16 one. See TUNING.md section 21.
        #
        # FOLD_BLOCKS sets how many 16-deep tiles accumulate in F16 before
        # draining to F32 registers. It bounds the F16 reduction depth, so it
        # is the accuracy knob; ne00 must be a multiple of LA_K*FOLD_BLOCKS,
        # which the host gate enforces.
        #
        # LDS_STAGE_A stages the activation strip in groupshared once per fold
        # window instead of letting every wave re-read it from the descriptor.
        # Worth ~1.4x on Phi-3 F16 pp512 (TUNING.md section 36). Groupshared
        # matrix loads do work; the earlier "returns fill" note was a miscoded
        # probe (TUNING.md section 35).
        if(NOT DEFINED GGML_DX12_IW_LDS_A)
            set(GGML_DX12_IW_LDS_A 1)
        endif()
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LA_IW_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_wave_f16_i_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${LA_IW_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LA_IW_HDR}
                    -Vn g_mul_mat_linalg_wave_f16_i_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D NWAVE=8
                    -D WAVE_M=1
                    -D NTILE=1
                    -D MTILE=4
                    -D FOLD_BLOCKS=4
                    -D SWIZZLE_Y=16
                    -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A}

                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_wave_f16_i (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LA_IW_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_wave_f16_i_w${WSIZE}_dxil.h\"\n")
        endforeach()

        foreach(VP_TYPE F16 Q8_0 Q4_K Q5_K Q6_K Q4_0 Q4_1 Q5_0 Q5_1 IQ4_NL MXFP4)
            string(TOLOWER "${VP_TYPE}" VP_TYPE_NAME)
            if(VP_TYPE STREQUAL "F16")
                set(VP_TYPE_DEFINE IW_TILED_F16)
            else()
                set(VP_TYPE_DEFINE MMID_${VP_TYPE})
            endif()
            foreach(VP_WAVE 16 32)
                foreach(VP_FULL 0 1)
                    set(VP_NAME "mul_mat_linalg_tiled_${VP_TYPE_NAME}_${VP_FULL}_w${VP_WAVE}")
                    set(VP_HEADER "${SHADER_OUTPUT_DIR}/${VP_NAME}_dxil.h")
                    add_custom_command(
                        OUTPUT ${VP_HEADER}
                        COMMAND ${GGML_DX12_DXC_PATH}
                            -T cs_6_10 -E main -Fh ${VP_HEADER} -Vn g_${VP_NAME}_dxil
                            -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                            -enable-16bit-types -D NATIVE_FP16=1
                            -D WAVE_SIZE=${VP_WAVE} -D VP_FULL_F16=${VP_FULL}
                            -D ${VP_TYPE_DEFINE}=1
                            -O3 -Qstrip_debug -Qstrip_reflect
                            "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl"
                        DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl" ${SHADER_INCLUDE_DEPS}
                        COMMENT "Compiling HLSL shader (SM 6.10): ${VP_NAME}"
                        VERBATIM)
                    target_sources(ggml-dx12 PRIVATE ${VP_HEADER})
                    file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${VP_NAME}_dxil.h\"\n")
                endforeach()
            endforeach()
        endforeach()

        # F16 activation variants: src1 comes from the F32->F16 pre-pass (DX12_TILED_B_F16, default on).
        foreach(VP_TYPE F16 Q8_0 Q4_K Q5_K Q6_K Q4_0 Q4_1 Q5_0 Q5_1 IQ4_NL MXFP4)
            string(TOLOWER "${VP_TYPE}" VP_TYPE_NAME)
            if(VP_TYPE STREQUAL "F16")
                set(VP_TYPE_DEFINE IW_TILED_F16)
            else()
                set(VP_TYPE_DEFINE MMID_${VP_TYPE})
            endif()
            foreach(VP_FULL 0 1)
                set(VP_NAME "mul_mat_linalg_tiled_${VP_TYPE_NAME}_${VP_FULL}_w16_bf16")
                set(VP_HEADER "${SHADER_OUTPUT_DIR}/${VP_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${VP_HEADER}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${VP_HEADER} -Vn g_${VP_NAME}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D NATIVE_FP16=1
                        -D WAVE_SIZE=16 -D VP_FULL_F16=${VP_FULL} -D VP_B_F16=1
                        -D ${VP_TYPE_DEFINE}=1
                        -O3 -Qstrip_debug -Qstrip_reflect
                        "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10): ${VP_NAME}"
                    VERBATIM)
                target_sources(ggml-dx12 PRIVATE ${VP_HEADER})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${VP_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()

        foreach(NV_TYPE F16 BF16 Q8_0 Q4_K Q5_K Q6_K Q4_0)
            string(TOLOWER "${NV_TYPE}" NV_TYPE_NAME)
            if(NV_TYPE STREQUAL "F16")
                set(NV_DEFINE IW_TILED_F16)
            elseif(NV_TYPE STREQUAL "BF16")
                set(NV_DEFINE IW_TILED_BF16)
            else()
                set(NV_DEFINE MMID_${NV_TYPE})
            endif()
            set(NV_NAME "mul_mat_composed_${NV_TYPE_NAME}_128x64_w32")
            set(NV_HEADER "${SHADER_OUTPUT_DIR}/${NV_NAME}_dxil.h")
            add_custom_command(
                OUTPUT ${NV_HEADER}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${NV_HEADER} -Vn g_${NV_NAME}_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D NATIVE_FP16=1 -D WAVE_SIZE=32
                    -D VP_NATIVE_F32=1 -D VP_BM=128 -D VP_BN=64
                    -D ${NV_DEFINE}=1 -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling composed F32 GEMM: ${NV_NAME}"
                VERBATIM)
            target_sources(ggml-dx12 PRIVATE ${NV_HEADER})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${NV_NAME}_dxil.h\"\n")
        endforeach()

        foreach(AMD_TYPE Q4_K Q6_K)
            string(TOLOWER "${AMD_TYPE}" AMD_TYPE_NAME)
            set(AMD_NAME "mul_mat_composed_${AMD_TYPE_NAME}_128x64_lds_w32")
            set(AMD_HEADER "${SHADER_OUTPUT_DIR}/${AMD_NAME}_dxil.h")
            add_custom_command(
                OUTPUT ${AMD_HEADER}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${AMD_HEADER} -Vn g_${AMD_NAME}_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D NATIVE_FP16=1 -D WAVE_SIZE=32
                    -D VP_NATIVE_F32=1 -D VP_LDS_OUTPUT=1 -D VP_BM=128 -D VP_BN=64
                    -D MMID_${AMD_TYPE}=1 -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_tiled_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling composed F32 GEMM: ${AMD_NAME}"
                VERBATIM)
            target_sources(ggml-dx12 PRIVATE ${AMD_HEADER})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${AMD_NAME}_dxil.h\"\n")
        endforeach()

        # Same kernel with Q8_0 weights, dequantized into a groupshared F16
        # tile. That is the only way the matrix path can run on a quantized
        # model here: the matrix load speaks F16, and a 34-byte Q8_0 block
        # cannot be handed to a descriptor load (TUNING.md section 21).
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LA_IQ8_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_wave_q8_0_i_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${LA_IQ8_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LA_IQ8_HDR}
                    -Vn g_mul_mat_linalg_wave_q8_0_i_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D NWAVE=8
                    -D WAVE_M=2
                    -D NTILE=1
                    -D MTILE=4
                    -D FOLD_BLOCKS=4
                    -D SWIZZLE_Y=16
                    -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A}
                    -D QUANT_B=8
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_wave_q8_0_i (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LA_IQ8_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_wave_q8_0_i_w${WSIZE}_dxil.h\"\n")
        endforeach()

        # Same kernel again with Q4_K weights. Vulkan runs its K-quants through
        # the cooperative-matrix path this way (dequantize into a shared F16
        # tile, then one matrix GEMM); MMQ instead redoes the nibble and 6-bit
        # scale unpack on every dp4a step (TUNING.md section 43).
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LA_IQ4K_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_wave_q4_k_i_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${LA_IQ4K_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LA_IQ4K_HDR}
                    -Vn g_mul_mat_linalg_wave_q4_k_i_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D NWAVE=8
                    -D WAVE_M=2
                    -D NTILE=1
                    -D MTILE=4
                    -D FOLD_BLOCKS=4
                    -D SWIZZLE_Y=16
                    -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A}
                    -D QUANT_B=4
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_wave_q4_k_i (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LA_IQ4K_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_wave_q4_k_i_w${WSIZE}_dxil.h\"\n")
        endforeach()

        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LA_IQ50_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_wave_q5_0_i_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${LA_IQ50_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LA_IQ50_HDR}
                    -Vn g_mul_mat_linalg_wave_q5_0_i_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D NWAVE=8
                    -D WAVE_M=2
                    -D NTILE=1
                    -D MTILE=4
                    -D FOLD_BLOCKS=4
                    -D SWIZZLE_Y=16
                    -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A}
                    -D QUANT_B=50
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_wave_q5_0_i (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LA_IQ50_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_wave_q5_0_i_w${WSIZE}_dxil.h\"\n")
        endforeach()

        foreach(IW_FORMAT "q6_k;6" "q5_k;5" "q4_0;40" "q4_1;41" "q5_1;51" "iq4_nl;14" "mxfp4;17")
            list(GET IW_FORMAT 0 IW_NAME)
            list(GET IW_FORMAT 1 IW_QUANT)
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(IW_NAME_FULL "mul_mat_linalg_wave_${IW_NAME}_i_w${WSIZE}")
                set(IW_HDR "${SHADER_OUTPUT_DIR}/${IW_NAME_FULL}_dxil.h")
                add_custom_command(
                    OUTPUT ${IW_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${IW_HDR} -Vn g_${IW_NAME_FULL}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=${WSIZE} -D NATIVE_FP16=1
                        -D NWAVE=8 -D WAVE_M=2 -D NTILE=1 -D MTILE=4
                        -D FOLD_BLOCKS=4 -D SWIZZLE_Y=16
                        -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A} -D QUANT_B=${IW_QUANT}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${IW_NAME_FULL}"
                )
                target_sources(ggml-dx12 PRIVATE ${IW_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${IW_NAME_FULL}_dxil.h\"\n")
            endforeach()
        endforeach()

        foreach(IW_FORMAT "f16;0" "q8_0;8" "q4_k;4" "q5_0;50" "q6_k;6" "q5_k;5"
                          "q4_0;40" "q4_1;41" "q5_1;51" "iq4_nl;14" "mxfp4;17")
            list(GET IW_FORMAT 0 IW_NAME)
            list(GET IW_FORMAT 1 IW_QUANT)
            if(IW_QUANT EQUAL 0)
                set(IW_BIAS_WAVE_M 1)
            else()
                set(IW_BIAS_WAVE_M 2)
            endif()
            set(IW_NAME_FULL "mul_mat_linalg_wave_${IW_NAME}_bias_i_w16")
            set(IW_HDR "${SHADER_OUTPUT_DIR}/${IW_NAME_FULL}_dxil.h")
            add_custom_command(
                OUTPUT ${IW_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${IW_HDR} -Vn g_${IW_NAME_FULL}_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D WAVE_SIZE=16 -D NATIVE_FP16=1
                    -D NWAVE=8 -D WAVE_M=${IW_BIAS_WAVE_M} -D NTILE=1 -D MTILE=4
                    -D FOLD_BLOCKS=4 -D SWIZZLE_Y=16 -D IW_FUSED_BIAS=1
                    -D LDS_STAGE_A=${GGML_DX12_IW_LDS_A} -D QUANT_B=${IW_QUANT}
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_wave_f16_i.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${IW_NAME_FULL}"
            )
            target_sources(ggml-dx12 PRIVATE ${IW_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${IW_NAME_FULL}_dxil.h\"\n")
        endforeach()

        foreach(STAGE_TYPE f16 q8_0 q4k q6k q5k q4_0 q4_1 q5_0 q5_1 iq4_nl iq4_xs
                           q2k q3k mxfp4 nvfp4 q1_0 q2_0 tq1_0 tq2_0
                           iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m)
            string(TOUPPER "${STAGE_TYPE}" STAGE_QUANT)
            string(REGEX REPLACE "^Q([23456])K$" "Q\\1_K" STAGE_QUANT "${STAGE_QUANT}")
            if(STAGE_TYPE STREQUAL "f16")
                set(STAGE_DEFINE "-DMM_DENSE_WEIGHT=1")
            else()
                set(STAGE_DEFINE "-DMMID_${STAGE_QUANT}=1")
            endif()
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(STAGE_NAME "mul_mat_stage_wave_${STAGE_TYPE}_i_w${WSIZE}")
                set(STAGE_HDR "${SHADER_OUTPUT_DIR}/${STAGE_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${STAGE_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${STAGE_HDR} -Vn g_${STAGE_NAME}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=${WSIZE} -D NATIVE_FP16=1
                        -D MM_LINALG_WAVE=1 ${STAGE_DEFINE}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        "${SHADER_INPUT_DIR}/mul_mat_gemm_quant.hlsli"
                    DEPENDS "${SHADER_INPUT_DIR}/mul_mat_gemm_quant.hlsli" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${STAGE_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${STAGE_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${STAGE_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()

        foreach(MMID_TYPE f16 q8_0 q4k q6k q5k q4_0 q4_1 q5_0 q5_1 iq4_nl iq4_xs
                          q2k q3k mxfp4 nvfp4 q1_0 q2_0 tq1_0 tq2_0
                          iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq1_s iq1_m)
            if(MMID_TYPE STREQUAL "f16")
                set(MMID_IW_SRC "${SHADER_INPUT_DIR}/mul_mat_id_gemm.hlsl")
            else()
                set(MMID_IW_SRC "${SHADER_INPUT_DIR}/mul_mat_id_gemm_tall_${MMID_TYPE}.hlsl")
            endif()
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(MMID_IW_NAME "mul_mat_id_wave_${MMID_TYPE}_i_w${WSIZE}")
                set(MMID_IW_HDR "${SHADER_OUTPUT_DIR}/${MMID_IW_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${MMID_IW_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main
                        -Fh ${MMID_IW_HDR}
                        -Vn g_${MMID_IW_NAME}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D MMID_BM=128
                        -D MMID_LINALG_WAVE=1
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${MMID_IW_SRC}
                    DEPENDS ${MMID_IW_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${MMID_IW_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${MMID_IW_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${MMID_IW_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()

        # Activation F32 -> F16 pre-pass for the shader above. Plain SM 6.0
        # work, but it only exists to feed the matrix path, so it is built
        # under the same option. Wave variants are identical - they exist so
        # WBLOB() resolves it like every other shader.
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(CVT_F16_HDR "${SHADER_OUTPUT_DIR}/cvt_f32_f16_flat_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${CVT_F16_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_0
                    -E main
                    -Fh ${CVT_F16_HDR}
                    -Vn g_cvt_f32_f16_flat_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -D WAVE_SIZE=${WSIZE}
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/cvt_f32_f16_flat.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/cvt_f32_f16_flat.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader: cvt_f32_f16_flat (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${CVT_F16_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"cvt_f32_f16_flat_w${WSIZE}_dxil.h\"\n")
        endforeach()

        # LinAlg MUL_MAT kernels. Compiled with the preview DXC at cs_6_10 but
        # emitted under the standard g_<name>_w<WS>_dxil symbol so WBLOB()
        # resolves them like any other wave-variant shader. Dispatch is gated
        # on linalg_matrix_supported, so devices without the feature (and
        # builds without this option) never reference these blobs.
        # Tile shapes: <BM>x<BN> as NWAVE:NT:MT:WN:BK. The group's NWAVE waves form
        # an (NWAVE/WN) x WN grid over the tile, so BM = (NWAVE/WN)*MT*16 and
        # BN = WN*NT*16, and a wave owns MT*NT accumulators for MT+NT staging
        # loads. The NVIDIA 128x128 variant uses twice as many waves and half
        # as many accumulators per wave to reduce register pressure.
        # Square per-wave blocking (MT=NT) reuses each staged fragment best,
        # but at 32x32 the geometry forces it to one wave: BM = WM*MT*16 = 32
        # with MT=2 gives WM=1, and BN likewise gives WN=1, so NWAVE=1. That
        # was measured and is a clear loss - pp512 39691 against 50877, and the
        # K=576 N=576 shape goes 26.0 -> 36.8 us/dispatch - because a lone wave
        # also halves the threads available to stage the tile. Reaching MT=NT=2
        # while keeping two waves needs BM=64, i.e. a new tile, not a rebalance
        # of this one.
        # BK is the staging depth. The narrow tiles carry 32 because a Q8_0 or
        # Q4_K sub-block is 32 wide, so one staged step decodes each scale once
        # instead of twice, and because their LDS footprint is small enough to
        # afford it - this mirrors what the Vulkan backend does for its
        # quantised warptiles. The wide tiles stay at 16: 128x128 would need
        # 36 KB of groupshared at BK 32, over the 32 KB limit.
        if(DEFINED ENV{DX12_REG_EPILOGUE})
            set(LA_EPI_FLAG "-D LA_REG_EPILOGUE=1")
        else()
            set(LA_EPI_FLAG "-D LA_REG_EPILOGUE=0")
        endif()
        foreach(PIPE_SHAPE "q8:8:4:2:4:16" "f16:0:4:2:4:16" "f16_small:0:2:1:2:32")
            string(REPLACE ":" ";" PIPE_PARTS ${PIPE_SHAPE})
            list(GET PIPE_PARTS 0 PIPE_TAG)
            list(GET PIPE_PARTS 1 PIPE_QUANT)
            list(GET PIPE_PARTS 2 PIPE_NW)
            list(GET PIPE_PARTS 3 PIPE_MT)
            list(GET PIPE_PARTS 4 PIPE_NT)
            list(GET PIPE_PARTS 5 PIPE_BK)
            set(PIPE_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_pipeline_${PIPE_TAG}_dxil.h")
            add_custom_command(
                OUTPUT ${PIPE_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${PIPE_HDR} -Vn g_mul_mat_linalg_pipeline_${PIPE_TAG}_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                    -D LA_QUANT=${PIPE_QUANT} -D LA_NWAVE=${PIPE_NW} -D LA_MT=${PIPE_MT} -D LA_NT=${PIPE_NT} -D BK=${PIPE_BK}
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_pipeline.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
            )
            target_sources(ggml-dx12 PRIVATE ${PIPE_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_pipeline_${PIPE_TAG}_dxil.h\"\n")
        endforeach()
        foreach(LA_FULL "f16_128x64:4:4:2:1:16:0:1" "q8_32x32:2:2:1:1:32:8:0" "q8_32x16:2:1:1:1:32:8:0")
            string(REPLACE ":" ";" LA_FULL_PARTS ${LA_FULL})
            list(GET LA_FULL_PARTS 0 LA_FULL_TAG)
            list(GET LA_FULL_PARTS 1 LA_FULL_NW)
            list(GET LA_FULL_PARTS 2 LA_FULL_NT)
            list(GET LA_FULL_PARTS 3 LA_FULL_MT)
            list(GET LA_FULL_PARTS 4 LA_FULL_WN)
            list(GET LA_FULL_PARTS 5 LA_FULL_BK)
            list(GET LA_FULL_PARTS 6 LA_FULL_QUANT)
            list(GET LA_FULL_PARTS 7 LA_FULL_DENSE)
            foreach(LA_SINGLE 0 1)
                if(LA_FULL_QUANT EQUAL 8 AND LA_SINGLE EQUAL 1)
                    continue()
                endif()
                set(LA_FULL_NAME "mul_mat_linalg_full_${LA_FULL_TAG}_s${LA_SINGLE}_w64")
                set(LA_FULL_HDR "${SHADER_OUTPUT_DIR}/${LA_FULL_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${LA_FULL_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${LA_FULL_HDR} -Vn g_${LA_FULL_NAME}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                        -D LA_NWAVE=${LA_FULL_NW} -D LA_NT=${LA_FULL_NT} -D LA_MT=${LA_FULL_MT}
                        -D LA_WN=${LA_FULL_WN} -D LA_BK=${LA_FULL_BK} -D LA_QUANT=${LA_FULL_QUANT}
                        -D LA_DENSE_FIXED=${LA_FULL_DENSE} -D LA_FULL_TILE=1 -D LA_SINGLE_BUFFER=${LA_SINGLE}
                        -D LA_REG_EPILOGUE=0 -O3 -Qstrip_debug -Qstrip_reflect
                        "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${LA_FULL_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${LA_FULL_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${LA_FULL_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()
        foreach(LA_SHAPE "128x64:4:4:2:1:16" "128x64_nv:8:4:1:1:16" "128x128:4:4:4:2:16" "128x128_nv:8:4:2:2:16" "32x32:2:2:1:1:32" "32x16:2:1:1:1:32")
            string(REPLACE ":" ";" LA_PARTS ${LA_SHAPE})
            list(GET LA_PARTS 0 LA_TAG)
            list(GET LA_PARTS 1 LA_NW)
            list(GET LA_PARTS 2 LA_NTV)
            list(GET LA_PARTS 3 LA_MTV)
            list(GET LA_PARTS 4 LA_WNV)
            list(GET LA_PARTS 5 LA_BKV)
            set(LA_SHAPE_EPI_FLAG ${LA_EPI_FLAG})
            if(LA_TAG MATCHES "_nv$")
                set(LA_SHAPE_EPI_FLAG "-D LA_REG_EPILOGUE=1")
            endif()
            foreach(WSIZE ${DX12_WAVE_SIZES})
                # A group needs at least BN threads, or B_PER_THREAD exceeds BK
                # and a thread's run no longer fits in one row of the staged
                # tile, which the quantised fetch paths cannot express (MXFP4
                # #errors on it outright). Trade wave count against per-wave
                # row blocking to reach that floor. BM is LA_NWAVE * LA_MT * 16
                # either way, so the host dispatch math is unaffected and the
                # shape list stays wave-size agnostic. None of the shapes below
                # currently trip this at any wave size; it exists so that
                # adding one that does is a one-line change rather than a
                # compile error on the wave16 variant only.
                math(EXPR LA_BNV "${LA_WNV} * ${LA_NTV} * 16")
                set(LA_NW_W ${LA_NW})
                set(LA_MT_W ${LA_MTV})
                math(EXPR LA_THR "${LA_NW_W} * ${WSIZE}")
                while(LA_THR LESS LA_BNV AND LA_MT_W GREATER 1)
                    math(EXPR LA_NW_W "${LA_NW_W} * 2")
                    math(EXPR LA_MT_W "${LA_MT_W} / 2")
                    math(EXPR LA_THR "${LA_NW_W} * ${WSIZE}")
                endwhile()
                # Same rule on the K axis. A thread's B run has to stay inside
                # one scale sub-block, and Q6_K's is 16 wide, so B_PER_THREAD
                # (BK*BN/THREADS) must not exceed 16. With BK 16 the wave floor
                # above already guarantees it; a deeper BK has to give it back
                # when the group is too small, which is what keeps wave16 on the
                # shallow staging it has always used.
                set(LA_BK_W ${LA_BKV})
                math(EXPR LA_BPT "${LA_BK_W} * ${LA_BNV} / ${LA_THR}")
                while(LA_BPT GREATER 16 AND LA_BK_W GREATER 16)
                    math(EXPR LA_BK_W "${LA_BK_W} / 2")
                    math(EXPR LA_BPT "${LA_BK_W} * ${LA_BNV} / ${LA_THR}")
                endwhile()
                set(LINALG_SRC "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl")
                set(LINALG_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_f16_${LA_TAG}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LINALG_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LINALG_HDR}
                        -Vn g_mul_mat_linalg_f16_${LA_TAG}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D LA_NWAVE=${LA_NW_W}
                        -D LA_NT=${LA_NTV}
                        -D LA_MT=${LA_MT_W}
                        -D LA_WN=${LA_WNV}
                        -D LA_BK=${LA_BK_W}
                        ${LA_SHAPE_EPI_FLAG}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LINALG_SRC}
                    DEPENDS ${LINALG_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_f16 (${LA_TAG}, wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LINALG_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_f16_${LA_TAG}_w${WSIZE}_dxil.h\"\n")

                # Same shader with the weight side dequantised from Q8_0 into
                # the staging tile, so the matrix path is byte-for-byte the
                # F16 one. Quantised weights are the bulk of a real prefill.
                set(LINALG_Q8_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_q8_0_${LA_TAG}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LINALG_Q8_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LINALG_Q8_HDR}
                        -Vn g_mul_mat_linalg_q8_0_${LA_TAG}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D LA_NWAVE=${LA_NW_W}
                        -D LA_NT=${LA_NTV}
                        -D LA_MT=${LA_MT_W}
                        -D LA_WN=${LA_WNV}
                        -D LA_BK=${LA_BK_W}
                        ${LA_SHAPE_EPI_FLAG}
                        -D LA_QUANT=8
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LINALG_SRC}
                    DEPENDS ${LINALG_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_q8_0 (${LA_TAG}, wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LINALG_Q8_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_q8_0_${LA_TAG}_w${WSIZE}_dxil.h\"\n")

                # And once more for Q4_K, the format most shipped models use.
                set(LINALG_Q4K_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_q4_k_${LA_TAG}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LINALG_Q4K_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LINALG_Q4K_HDR}
                        -Vn g_mul_mat_linalg_q4_k_${LA_TAG}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D LA_NWAVE=${LA_NW_W}
                        -D LA_NT=${LA_NTV}
                        -D LA_MT=${LA_MT_W}
                        -D LA_WN=${LA_WNV}
                        -D LA_BK=${LA_BK_W}
                        ${LA_SHAPE_EPI_FLAG}
                        -D LA_QUANT=4
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LINALG_SRC}
                    DEPENDS ${LINALG_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_q4_k (${LA_TAG}, wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LINALG_Q4K_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_q4_k_${LA_TAG}_w${WSIZE}_dxil.h\"\n")

                # Q5_K and Q6_K round out the K-quant set a Q4_K_M file uses,
                # and Q5_0 covers the rows those files cannot K-quantise.
                foreach(LA_KQ "q5_k:5" "q6_k:6" "q5_0:50" "q4_0:40" "q4_1:41" "q5_1:51" "iq4_nl:49" "mxfp4:30")
                    string(REPLACE ":" ";" LA_KQP ${LA_KQ})
                    list(GET LA_KQP 0 LA_QNAME)
                    list(GET LA_KQP 1 LA_QVAL)
                    set(LINALG_KQ_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_${LA_QNAME}_${LA_TAG}_w${WSIZE}_dxil.h")
                    add_custom_command(
                        OUTPUT ${LINALG_KQ_HDR}
                        COMMAND ${GGML_DX12_DXC_PATH}
                            -T cs_6_10
                            -E main
                            -Fh ${LINALG_KQ_HDR}
                            -Vn g_mul_mat_linalg_${LA_QNAME}_${LA_TAG}_w${WSIZE}_dxil
                            -I ${SHADER_INPUT_DIR}
                            -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                            -enable-16bit-types
                            -D WAVE_SIZE=${WSIZE}
                            -D NATIVE_FP16=1
                            -D LA_NWAVE=${LA_NW_W}
                            -D LA_NT=${LA_NTV}
                            -D LA_MT=${LA_MT_W}
                            -D LA_WN=${LA_WNV}
                            -D LA_BK=${LA_BK_W}
                            ${LA_SHAPE_EPI_FLAG}
                            -D LA_QUANT=${LA_QVAL}
                            -O3 -Qstrip_debug -Qstrip_reflect
                            ${LINALG_SRC}
                        DEPENDS ${LINALG_SRC} ${SHADER_INCLUDE_DEPS}
                        COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_${LA_QNAME} (${LA_TAG}, wave=${WSIZE})"
                    )
                    target_sources(ggml-dx12 PRIVATE ${LINALG_KQ_HDR})
                    file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_${LA_QNAME}_${LA_TAG}_w${WSIZE}_dxil.h\"\n")
                endforeach()
                if(WSIZE EQUAL 64 AND NOT LA_TAG MATCHES "_nv$")
                    set(LINALG_Q50_PACKED_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_q5_0_packed_${LA_TAG}_w64_dxil.h")
                    add_custom_command(
                        OUTPUT ${LINALG_Q50_PACKED_HDR}
                        COMMAND ${GGML_DX12_DXC_PATH}
                            -T cs_6_10 -E main
                            -Fh ${LINALG_Q50_PACKED_HDR}
                            -Vn g_mul_mat_linalg_q5_0_packed_${LA_TAG}_w64_dxil
                            -I ${SHADER_INPUT_DIR}
                            -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                            -enable-16bit-types
                            -D WAVE_SIZE=64 -D NATIVE_FP16=1
                            -D LA_NWAVE=${LA_NW_W} -D LA_NT=${LA_NTV}
                            -D LA_MT=${LA_MT_W} -D LA_WN=${LA_WNV} -D LA_BK=${LA_BK_W}
                            -D LA_QUANT=50 -D LA_NATIVE16_LOADS=1 -D LA_Q50_PACKED=1
                            ${LA_SHAPE_EPI_FLAG}
                            -O3 -Qstrip_debug -Qstrip_reflect
                            ${LINALG_SRC}
                        DEPENDS ${LINALG_SRC} ${SHADER_INCLUDE_DEPS}
                        COMMENT "Compiling HLSL shader: mul_mat_linalg_q5_0_packed (${LA_TAG}, wave=64)"
                    )
                    target_sources(ggml-dx12 PRIVATE ${LINALG_Q50_PACKED_HDR})
                    file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_q5_0_packed_${LA_TAG}_w64_dxil.h\"\n")
                endif()
            endforeach()

        endforeach()

        # Full Q8_0 tiles with contiguous F32 activations on RDNA4.
        foreach(LA_ALIGNED_SHAPE "128x64:2:1" "128x128:4:2")
            string(REPLACE ":" ";" LA_ALIGNED_PARTS ${LA_ALIGNED_SHAPE})
            list(GET LA_ALIGNED_PARTS 0 LA_ALIGNED_TAG)
            list(GET LA_ALIGNED_PARTS 1 LA_ALIGNED_MT)
            list(GET LA_ALIGNED_PARTS 2 LA_ALIGNED_WN)
            set(LINALG_ALIGNED_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_q8_0_aligned_${LA_ALIGNED_TAG}_w64_dxil.h")
            add_custom_command(
                OUTPUT ${LINALG_ALIGNED_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main
                    -Fh ${LINALG_ALIGNED_HDR}
                    -Vn g_mul_mat_linalg_q8_0_aligned_${LA_ALIGNED_TAG}_w64_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=64 -D NATIVE_FP16=1
                    -D LA_NWAVE=4 -D LA_NT=4 -D LA_MT=${LA_ALIGNED_MT} -D LA_WN=${LA_ALIGNED_WN} -D LA_BK=16
                    -D LA_QUANT=8 -D LA_ALIGNED=1 -D LA_REG_EPILOGUE=0
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader: mul_mat_linalg_q8_0_aligned (${LA_ALIGNED_TAG}, wave=64)"
            )
            target_sources(ggml-dx12 PRIVATE ${LINALG_ALIGNED_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_q8_0_aligned_${LA_ALIGNED_TAG}_w64_dxil.h\"\n")
        endforeach()

        # RDNA4 Q8_0 small-batch tile. The standard narrow tile is 32x16 with
        # two waves; one 16x16 wave avoids padding half or more of the rows for
        # NC batches and wins on large-K models, while the host keeps small-K
        # shapes on the standard tile.
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LINALG_Q8_NC16_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_q8_0_16x16_w${WSIZE}_dxil.h")
            if(WSIZE EQUAL 16)
                set(LA_Q8_NC16_BK 16)
            else()
                set(LA_Q8_NC16_BK 32)
            endif()
            add_custom_command(
                OUTPUT ${LINALG_Q8_NC16_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LINALG_Q8_NC16_HDR}
                    -Vn g_mul_mat_linalg_q8_0_16x16_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D LA_NWAVE=1
                    -D LA_NT=1
                    -D LA_MT=1
                    -D LA_WN=1
                    -D LA_BK=${LA_Q8_NC16_BK}
                    ${LA_EPI_FLAG}
                    -D LA_QUANT=8
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_q8_0 (16x16, wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LINALG_Q8_NC16_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_q8_0_16x16_w${WSIZE}_dxil.h\"\n")
        endforeach()

        foreach(TG_THREADS 64 128 256)
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(LINALG_TG_SRC "${SHADER_INPUT_DIR}/mul_mat_linalg_tg_f16.hlsl")
                set(LINALG_TG_HDR "${SHADER_OUTPUT_DIR}/mul_mat_linalg_tg_f16_64x128_t${TG_THREADS}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LINALG_TG_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LINALG_TG_HDR}
                        -Vn g_mul_mat_linalg_tg_f16_64x128_t${TG_THREADS}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D NATIVE_FP16=1
                        -D TG_BM=64
                        -D TG_BN=128
                        -D THREADS=${TG_THREADS}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LINALG_TG_SRC}
                    DEPENDS ${LINALG_TG_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_linalg_tg_f16 (64x128, threads=${TG_THREADS})"
                )
                target_sources(ggml-dx12 PRIVATE ${LINALG_TG_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_linalg_tg_f16_64x128_t${TG_THREADS}_w${WSIZE}_dxil.h\"\n")
            endforeach()
        endforeach()

        # The same GEMM with the activation rows gathered per expert, for
        # MUL_MAT_ID. Two tile heights: an expert's run is padded up to a
        # whole tile, so the taller tile reads the weights fewer times but
        # wastes more rows when the per-expert count is small.
        foreach(LA_MSHAPE "32x32:2:2:1" "128x64:4:4:2" "bucket_32x32:2:2:1" "bucket_128x64:4:4:2" "bucket_128x128:8:4:2"
                         "expert_32x32:2:2:1" "expert_128x64:4:4:2" "expert_128x128:8:4:2")
            string(REPLACE ":" ";" LA_MSP ${LA_MSHAPE})
            list(GET LA_MSP 0 LA_MMID_TAG)
            list(GET LA_MSP 1 LA_MMID_NW)
            list(GET LA_MSP 2 LA_MMID_NT)
            list(GET LA_MSP 3 LA_MMID_MT)
            foreach(WSIZE ${DX12_WAVE_SIZES})
                if(LA_MMID_TAG STREQUAL "bucket_128x128" AND NOT WSIZE EQUAL 32)
                    continue()
                endif()
                if(LA_MMID_TAG MATCHES "^expert_" AND NOT WSIZE EQUAL 64)
                    continue()
                endif()
                if(LA_MMID_TAG MATCHES "^bucket_" AND NOT (WSIZE EQUAL 64 OR WSIZE EQUAL 32))
                    continue()
                endif()
                foreach(LA_MQ "f16:0" "q8_0:8" "q4_k:4" "q5_k:5" "q6_k:6" "q5_0:50"
                              "q4_0:40" "q4_1:41" "q5_1:51" "iq4_nl:49" "mxfp4:30")
                    string(REPLACE ":" ";" LA_MQP ${LA_MQ})
                    list(GET LA_MQP 0 LA_MNAME)
                    list(GET LA_MQP 1 LA_MVAL)
                    if(LA_MMID_TAG MATCHES "128x128$" AND
                       NOT (LA_MVAL EQUAL 0 OR LA_MVAL EQUAL 4 OR LA_MVAL EQUAL 6 OR LA_MVAL EQUAL 8))
                        continue()
                    endif()
                    set(LA_MMID_DEFS -D LA_MMID=1)
                    if(LA_MMID_TAG MATCHES "128x128$")
                        list(APPEND LA_MMID_DEFS -D LA_WN=2 -D LA_BK=16)
                    endif()
                    if(LA_MMID_TAG MATCHES "^(bucket|expert)_")
                        list(APPEND LA_MMID_DEFS -D LA_MMID_BUCKET=1)
                        if(LA_MVAL EQUAL 8 AND WSIZE EQUAL 64)
                            list(APPEND LA_MMID_DEFS -D LA_NATIVE16_LOADS=1)
                        endif()
                        if(LA_MMID_TAG MATCHES "^expert_")
                            list(APPEND LA_MMID_DEFS -D LA_MMID_EXPERT_MAJOR=1)
                        endif()
                    endif()
                    if (NOT LA_MVAL EQUAL 0)
                        list(APPEND LA_MMID_DEFS -D LA_QUANT=${LA_MVAL})
                    endif()
                    if((LA_MMID_TAG STREQUAL "bucket_128x64" OR LA_MMID_TAG STREQUAL "bucket_128x128") AND WSIZE EQUAL 32 AND
                       (LA_MVAL EQUAL 8 OR LA_MVAL EQUAL 4 OR LA_MVAL EQUAL 6))
                        list(APPEND LA_MMID_DEFS -D LA_REG_EPILOGUE=1)
                    endif()
                    set(LINALG_ID_HDR "${SHADER_OUTPUT_DIR}/mul_mat_id_linalg_${LA_MNAME}_${LA_MMID_TAG}_w${WSIZE}_dxil.h")
                    add_custom_command(
                        OUTPUT ${LINALG_ID_HDR}
                        COMMAND ${GGML_DX12_DXC_PATH}
                            -T cs_6_10
                            -E main
                            -Fh ${LINALG_ID_HDR}
                            -Vn g_mul_mat_id_linalg_${LA_MNAME}_${LA_MMID_TAG}_w${WSIZE}_dxil
                            -I ${SHADER_INPUT_DIR}
                            -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                            -enable-16bit-types
                            -D WAVE_SIZE=${WSIZE}
                            -D NATIVE_FP16=1
                            -D LA_NWAVE=${LA_MMID_NW}
                            -D LA_NT=${LA_MMID_NT}
                            -D LA_MT=${LA_MMID_MT}
                            ${LA_MMID_DEFS}
                            -O3 -Qstrip_debug -Qstrip_reflect
                            "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl"
                        DEPENDS "${SHADER_INPUT_DIR}/mul_mat_linalg_f16.hlsl" ${SHADER_INCLUDE_DEPS}
                        COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): mul_mat_id_linalg_${LA_MNAME} (${LA_MMID_TAG}, wave=${WSIZE})"
                    )
                    target_sources(ggml-dx12 PRIVATE ${LINALG_ID_HDR})
                    file(APPEND "${SHADER_MASTER_HEADER}" "#include \"mul_mat_id_linalg_${LA_MNAME}_${LA_MMID_TAG}_w${WSIZE}_dxil.h\"\n")
                endforeach()
            endforeach()
        endforeach()

        foreach(AUX_KERNEL "conv_2d;2" "conv_transpose_2d;3" "conv_3d;4" "out_prod;0")
            list(GET AUX_KERNEL 0 AUX_TAG)
            list(GET AUX_KERNEL 1 AUX_KIND)
            if(AUX_KIND EQUAL 0)
                set(AUX_SOURCE "${SHADER_INPUT_DIR}/out_prod_linalg.hlsl")
                set(AUX_ROWS 32)
            else()
                set(AUX_SOURCE "${SHADER_INPUT_DIR}/conv_linalg.hlsl")
                set(AUX_ROWS 32 64 128)
            endif()
            foreach(AUX_BM ${AUX_ROWS})
                set(AUX_NAME "${AUX_TAG}_linalg_wave_${AUX_BM}_i")
                set(AUX_HDR "${SHADER_OUTPUT_DIR}/${AUX_NAME}_w16_dxil.h")
                add_custom_command(
                    OUTPUT ${AUX_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${AUX_HDR} -Vn g_${AUX_NAME}_w16_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=16 -D NATIVE_FP16=1
                        -D LA_INTEL_WAVE=1 -D CONV_KIND=${AUX_KIND} -D CONV_BM=${AUX_BM}
                        -O3 -Qstrip_debug -Qstrip_reflect ${AUX_SOURCE}
                    DEPENDS ${AUX_SOURCE} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${AUX_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${AUX_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${AUX_NAME}_w16_dxil.h\"\n")
            endforeach()
        endforeach()

        foreach(FA_KV f16 q8_0 q4_0)
            set(FA_IW_DEFS "")
            if(FA_KV STREQUAL "q8_0")
                set(FA_IW_DEFS "-DFA_KV_Q8_0=1")
            elseif(FA_KV STREQUAL "q4_0")
                set(FA_IW_DEFS "-DFA_KV_Q4_0=1")
            endif()
            foreach(FA_HEAD_DIM 64 96 128)
                set(FA_IW_NAME "flash_attn_linalg_wave_${FA_KV}_${FA_HEAD_DIM}_i")
                set(FA_IW_HDR "${SHADER_OUTPUT_DIR}/${FA_IW_NAME}_w16_dxil.h")
                add_custom_command(
                    OUTPUT ${FA_IW_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${FA_IW_HDR} -Vn g_${FA_IW_NAME}_w16_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=16 -D NATIVE_FP16=1
                        -D FA_INTEL_WAVE=1 -D FA_D=${FA_HEAD_DIM} ${FA_IW_DEFS}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl"
                            "${SHADER_INPUT_DIR}/flash_attn_linalg_wave_i.hlsli" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FA_IW_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${FA_IW_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_IW_NAME}_w16_dxil.h\"\n")
            endforeach()
        endforeach()

        # Flash attention on wave matrices. Compiled per head dim: the LDS
        # budget (32 KB in D3D12) decides the query-row count, so the tile
        # shape is a compile-time property of FA_D.
        set(FAL_RDNA4_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_rdna4_128_w64_dxil.h")
        add_custom_command(
            OUTPUT ${FAL_RDNA4_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main
                -Fh ${FAL_RDNA4_HDR}
                -Vn g_flash_attn_linalg_rdna4_128_w64_dxil
                -I ${SHADER_INPUT_DIR}
                -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types
                -D WAVE_SIZE=64 -D NATIVE_FP16=1 -D FA_D=128
                -D FA_ROW_WAVE=1 -D FA_MASK_WAVE=1
                -D FA_BR=32 -D FA_BC=64 -D FA_TPW=2
                -O3 -Qstrip_debug -Qstrip_reflect
                "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): RDNA4 D128 attention"
        )
        target_sources(ggml-dx12 PRIVATE ${FAL_RDNA4_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_rdna4_128_w64_dxil.h\"\n")
        set(FAL_COMPACT_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_compact_128_w64_dxil.h")
        add_custom_command(
            OUTPUT ${FAL_COMPACT_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FAL_COMPACT_HDR} -Vn g_flash_attn_linalg_compact_128_w64_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1 -D FA_D=128
                -D FA_BR=32 -D FA_BC=64 -D FA_TPW=2
                -D FA_ROW_WAVE=1 -D FA_MASK_WAVE=1 -D FA_COMPACT_KV=1
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): compact D128 attention"
        )
        target_sources(ggml-dx12 PRIVATE ${FAL_COMPACT_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_compact_128_w64_dxil.h\"\n")
        foreach(FA_PIPELINE_NV_SHADER flash_attn_pipeline flash_attn_mask_packed)
            foreach(FA_PIPELINE_NV_STRIDE 256 2048)
                if(FA_PIPELINE_NV_SHADER STREQUAL "flash_attn_mask_packed" AND FA_PIPELINE_NV_STRIDE EQUAL 2048)
                    continue()
                endif()
                set(FA_PIPELINE_NV_NAME "${FA_PIPELINE_NV_SHADER}_128_k${FA_PIPELINE_NV_STRIDE}_w32")
                set(FA_PIPELINE_NV_HDR "${SHADER_OUTPUT_DIR}/${FA_PIPELINE_NV_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${FA_PIPELINE_NV_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${FA_PIPELINE_NV_HDR} -Vn g_${FA_PIPELINE_NV_NAME}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=32 -D NATIVE_FP16=1
                        -D FA_D=128 -D FA_BR=16 -D FA_BC=64 -D FA_PIPE_K_STRIDE=${FA_PIPELINE_NV_STRIDE}
                        -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/${FA_PIPELINE_NV_SHADER}.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/${FA_PIPELINE_NV_SHADER}.hlsl" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FA_PIPELINE_NV_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${FA_PIPELINE_NV_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_PIPELINE_NV_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()
        foreach(FA_PIPELINE_NV_DIM 64 96)
            if(FA_PIPELINE_NV_DIM EQUAL 64)
                set(FA_PIPELINE_NV_ROWS 32)
                set(FA_PIPELINE_NV_STRIDE 1024)
            else()
                set(FA_PIPELINE_NV_ROWS 16)
                set(FA_PIPELINE_NV_STRIDE 6144)
            endif()
            set(FA_PIPELINE_NV_NAME "flash_attn_pipeline_${FA_PIPELINE_NV_DIM}_k${FA_PIPELINE_NV_STRIDE}_w32")
            set(FA_PIPELINE_NV_HDR "${SHADER_OUTPUT_DIR}/${FA_PIPELINE_NV_NAME}_dxil.h")
            add_custom_command(
                OUTPUT ${FA_PIPELINE_NV_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${FA_PIPELINE_NV_HDR} -Vn g_${FA_PIPELINE_NV_NAME}_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D WAVE_SIZE=32 -D NATIVE_FP16=1
                    -D FA_D=${FA_PIPELINE_NV_DIM} -D FA_BR=${FA_PIPELINE_NV_ROWS} -D FA_BC=64 -D FA_PIPE_K_STRIDE=${FA_PIPELINE_NV_STRIDE}
                    -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FA_PIPELINE_NV_NAME}"
            )
            target_sources(ggml-dx12 PRIVATE ${FA_PIPELINE_NV_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_PIPELINE_NV_NAME}_dxil.h\"\n")
        endforeach()
        set(FA_MASK_NV_BR32_HDR "${SHADER_OUTPUT_DIR}/flash_attn_mask_packed_br32_w32_dxil.h")
        set(FA_D96_HDR "${SHADER_OUTPUT_DIR}/flash_attn_pipeline_96_br32_k6144_w32_dxil.h")
        add_custom_command(
            OUTPUT ${FA_D96_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FA_D96_HDR} -Vn g_flash_attn_pipeline_96_br32_k6144_w32_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=32 -D NATIVE_FP16=1
                -D FA_D=96 -D FA_BR=32 -D FA_BC=64 -D FA_PIPE_K_STRIDE=6144
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): wave32 D96 Br32"
        )
        target_sources(ggml-dx12 PRIVATE ${FA_D96_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_pipeline_96_br32_k6144_w32_dxil.h\"\n")
        set(FA_D96_W64_HDR "${SHADER_OUTPUT_DIR}/flash_attn_pipeline_96_br32_k6144_w64_dxil.h")
        add_custom_command(
            OUTPUT ${FA_D96_W64_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FA_D96_W64_HDR} -Vn g_flash_attn_pipeline_96_br32_k6144_w64_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                -D FA_D=96 -D FA_BR=32 -D FA_BC=64 -D FA_PIPE_K_STRIDE=6144 -D FA_D96_W64_BR32=1
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): wave64 D96 Br32"
        )
        target_sources(ggml-dx12 PRIVATE ${FA_D96_W64_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_pipeline_96_br32_k6144_w64_dxil.h\"\n")
        add_custom_command(
            OUTPUT ${FA_MASK_NV_BR32_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FA_MASK_NV_BR32_HDR} -Vn g_flash_attn_mask_packed_br32_w32_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=32 -D NATIVE_FP16=1 -D FA_BR=32 -D FA_BC=64
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): wave32 Br32 packed attention mask"
        )
        target_sources(ggml-dx12 PRIVATE ${FA_MASK_NV_BR32_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_mask_packed_br32_w32_dxil.h\"\n")
        foreach(FA_PIPELINE_SHADER flash_attn_pipeline flash_attn_mask_packed)
            set(FA_PIPELINE_HDR "${SHADER_OUTPUT_DIR}/${FA_PIPELINE_SHADER}_128_w64_dxil.h")
            add_custom_command(
                OUTPUT ${FA_PIPELINE_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${FA_PIPELINE_HDR} -Vn g_${FA_PIPELINE_SHADER}_128_w64_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                    -D FA_D=128 -D FA_BR=16 -D FA_BC=64
                    -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/${FA_PIPELINE_SHADER}.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/${FA_PIPELINE_SHADER}.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FA_PIPELINE_SHADER}"
            )
            target_sources(ggml-dx12 PRIVATE ${FA_PIPELINE_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_PIPELINE_SHADER}_128_w64_dxil.h\"\n")
        endforeach()
        set(FA_PIPELINE_STRIDED_HDR "${SHADER_OUTPUT_DIR}/flash_attn_pipeline_128_k2048_w64_dxil.h")
        add_custom_command(
            OUTPUT ${FA_PIPELINE_STRIDED_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FA_PIPELINE_STRIDED_HDR} -Vn g_flash_attn_pipeline_128_k2048_w64_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                -D FA_D=128 -D FA_BR=16 -D FA_BC=64 -D FA_PIPE_K_STRIDE=2048
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): interleaved D128 attention"
        )
        target_sources(ggml-dx12 PRIVATE ${FA_PIPELINE_STRIDED_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_pipeline_128_k2048_w64_dxil.h\"\n")
        foreach(FA_PIPELINE_DIM 64 96)
            if(FA_PIPELINE_DIM EQUAL 64)
                set(FA_PIPELINE_STRIDES 128 384 1024 1536)
                set(FA_PIPELINE_ROWS 16)
            else()
                set(FA_PIPELINE_STRIDES 192 6144)
                set(FA_PIPELINE_ROWS 16)
            endif()
            foreach(FA_PIPELINE_STRIDE ${FA_PIPELINE_STRIDES})
                if(FA_PIPELINE_DIM EQUAL 64 AND FA_PIPELINE_STRIDE EQUAL 1024)
                    set(FA_PIPELINE_ROWS 32)
                else()
                    set(FA_PIPELINE_ROWS 16)
                endif()
                set(FA_PIPELINE_NAME "flash_attn_pipeline_${FA_PIPELINE_DIM}_k${FA_PIPELINE_STRIDE}_w64")
                set(FA_PIPELINE_HDR "${SHADER_OUTPUT_DIR}/${FA_PIPELINE_NAME}_dxil.h")
                add_custom_command(
                    OUTPUT ${FA_PIPELINE_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10 -E main -Fh ${FA_PIPELINE_HDR} -Vn g_${FA_PIPELINE_NAME}_dxil
                        -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                        -D FA_D=${FA_PIPELINE_DIM} -D FA_BR=${FA_PIPELINE_ROWS} -D FA_BC=64 -D FA_PIPE_K_STRIDE=${FA_PIPELINE_STRIDE}
                        -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl"
                    DEPENDS "${SHADER_INPUT_DIR}/flash_attn_pipeline.hlsl" ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FA_PIPELINE_NAME}"
                )
                target_sources(ggml-dx12 PRIVATE ${FA_PIPELINE_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FA_PIPELINE_NAME}_dxil.h\"\n")
            endforeach()
        endforeach()
        set(FA_MASK_BR32_HDR "${SHADER_OUTPUT_DIR}/flash_attn_mask_packed_br32_w64_dxil.h")
        add_custom_command(
            OUTPUT ${FA_MASK_BR32_HDR}
            COMMAND ${GGML_DX12_DXC_PATH}
                -T cs_6_10 -E main -Fh ${FA_MASK_BR32_HDR} -Vn g_flash_attn_mask_packed_br32_w64_dxil
                -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1 -D FA_BR=32 -D FA_BC=64
                -O3 -Qstrip_debug -Qstrip_reflect "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl"
            DEPENDS "${SHADER_INPUT_DIR}/flash_attn_mask_packed.hlsl" ${SHADER_INCLUDE_DEPS}
            COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): Br32 packed attention mask"
        )
        target_sources(ggml-dx12 PRIVATE ${FA_MASK_BR32_HDR})
        file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_mask_packed_br32_w64_dxil.h\"\n")
        foreach(FAL_PV_VARIANT generic rdna4 compact)
            if(FAL_PV_VARIANT STREQUAL "compact")
                set(FAL_PV_NAME "flash_attn_linalg_pv_f16_compact_128")
                set(FAL_PV_DEFS -DFA_BC=64 -DFA_TPW=2 -DFA_ROW_WAVE=1 -DFA_MASK_WAVE=1 -DFA_COMPACT_KV=1)
            elseif(FAL_PV_VARIANT STREQUAL "rdna4")
                set(FAL_PV_NAME "flash_attn_linalg_pv_f16_rdna4_128")
                set(FAL_PV_DEFS -DFA_BC=64 -DFA_TPW=2 -DFA_ROW_WAVE=1 -DFA_MASK_WAVE=1)
            else()
                set(FAL_PV_NAME "flash_attn_linalg_pv_f16_128")
                set(FAL_PV_DEFS -DFA_BC=32 -DFA_TPW=1)
            endif()
            set(FAL_PV_HDR "${SHADER_OUTPUT_DIR}/${FAL_PV_NAME}_w64_dxil.h")
            add_custom_command(
                OUTPUT ${FAL_PV_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10 -E main -Fh ${FAL_PV_HDR} -Vn g_${FAL_PV_NAME}_w64_dxil
                    -I ${SHADER_INPUT_DIR} -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types -D WAVE_SIZE=64 -D NATIVE_FP16=1
                    -D FA_D=128 -D FA_BR=32 -D FA_PV_F16=1 ${FAL_PV_DEFS}
                    -O3 -Qstrip_debug -Qstrip_reflect
                    "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl"
                DEPENDS "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl" ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${FAL_PV_NAME}"
            )
            target_sources(ggml-dx12 PRIVATE ${FAL_PV_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${FAL_PV_NAME}_w64_dxil.h\"\n")
        endforeach()
        foreach(FA_HEAD_DIM 64 96 128)
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
                set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_${FA_HEAD_DIM}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${FAL_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${FAL_HDR}
                        -Vn g_flash_attn_linalg_${FA_HEAD_DIM}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D FA_D=${FA_HEAD_DIM}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${FAL_SRC}
                    DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg (D=${FA_HEAD_DIM}, wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_${FA_HEAD_DIM}_w${WSIZE}_dxil.h\"\n")
            endforeach()
        endforeach()
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
            set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_64_br32_tpw1_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${FAL_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${FAL_HDR}
                    -Vn g_flash_attn_linalg_64_br32_tpw1_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D FA_D=64
                    -D FA_BR=32
                    -D FA_TPW=1
                    -D FA_Q_FORWARD=1
                    -O3 -Qstrip_debug -Qstrip_reflect
                    ${FAL_SRC}
                DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg (D=64, BR=32, TPW=1, wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_64_br32_tpw1_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
            set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_96_br32_tpw1_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${FAL_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${FAL_HDR}
                    -Vn g_flash_attn_linalg_96_br32_tpw1_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D FA_D=96
                    -D FA_BR=32
                    -D FA_TPW=1
                    -D FA_Q_FORWARD=1
                    -O3 -Qstrip_debug -Qstrip_reflect
                    ${FAL_SRC}
                DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg (D=96, BR=32, TPW=1, wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_96_br32_tpw1_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
            set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_128_w8_pv1_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${FAL_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${FAL_HDR}
                    -Vn g_flash_attn_linalg_128_w8_pv1_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -D NATIVE_FP16=1
                    -D FA_D=128
                    -D FA_DIRECT_K=1
                    -D FA_NWAVE=8
                    -D FA_PV_SLOTS=1
                    -D FA_Q_FORWARD=1
                    -O3 -Qstrip_debug -Qstrip_reflect
                    ${FAL_SRC}
                DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg (D=128, waves=8, PV slots=1, wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_128_w8_pv1_w${WSIZE}_dxil.h\"\n")
        endforeach()
        foreach(FA_QKT_DIM 64 96)
            if (FA_QKT_DIM EQUAL 64)
                set(FA_QKT_NAME "64_br32_tpw1_qkt")
                set(FA_QKT_DEFS -D FA_D=64 -D FA_BR=32 -D FA_TPW=1)
            else()
                set(FA_QKT_NAME "96_br32_tpw1_qkt")
                set(FA_QKT_DEFS -D FA_D=96 -D FA_BR=32 -D FA_TPW=1)
            endif()
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
                set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_${FA_QKT_NAME}_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${FAL_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${FAL_HDR}
                        -Vn g_flash_attn_linalg_${FA_QKT_NAME}_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D NATIVE_FP16=1
                        -D FA_QK_TRANSPOSED=1
                        -D FA_PV_SLOTS=1
                        -D FA_Q_FORWARD=1
                        ${FA_QKT_DEFS}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${FAL_SRC}
                    DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg_${FA_QKT_NAME} (wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_${FA_QKT_NAME}_w${WSIZE}_dxil.h\"\n")
            endforeach()
        endforeach()
        # Quantised K/V cache variants of the same kernel. Only the LDS staging
        # differs, but mmid-style dequant is a compile-time specialisation, so
        # each supported cache type needs its own blob.
        foreach(FA_KVQ q8_0 q4_0)
            if (FA_KVQ STREQUAL "q8_0")
                set(FA_KVQ_DEF "FA_KV_Q8_0=1")
            else()
                set(FA_KVQ_DEF "FA_KV_Q4_0=1")
            endif()
            foreach(FA_HEAD_DIM 64 96 128)
                foreach(WSIZE ${DX12_WAVE_SIZES})
                    set(FAL_SRC "${SHADER_INPUT_DIR}/flash_attn_linalg.hlsl")
                    set(FAL_HDR "${SHADER_OUTPUT_DIR}/flash_attn_linalg_${FA_KVQ}_${FA_HEAD_DIM}_w${WSIZE}_dxil.h")
                    add_custom_command(
                        OUTPUT ${FAL_HDR}
                        COMMAND ${GGML_DX12_DXC_PATH}
                            -T cs_6_10
                            -E main
                            -Fh ${FAL_HDR}
                            -Vn g_flash_attn_linalg_${FA_KVQ}_${FA_HEAD_DIM}_w${WSIZE}_dxil
                            -I ${SHADER_INPUT_DIR}
                            -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                            -enable-16bit-types
                            -D WAVE_SIZE=${WSIZE}
                            -D NATIVE_FP16=1
                            -D FA_D=${FA_HEAD_DIM}
                            -D ${FA_KVQ_DEF}
                            -O3 -Qstrip_debug -Qstrip_reflect
                            ${FAL_SRC}
                        DEPENDS ${FAL_SRC} ${SHADER_INCLUDE_DEPS}
                        COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): flash_attn_linalg_${FA_KVQ} (D=${FA_HEAD_DIM}, wave=${WSIZE})"
                    )
                    target_sources(ggml-dx12 PRIVATE ${FAL_HDR})
                    file(APPEND "${SHADER_MASTER_HEADER}" "#include \"flash_attn_linalg_${FA_KVQ}_${FA_HEAD_DIM}_w${WSIZE}_dxil.h\"\n")
                endforeach()
            endforeach()
        endforeach()

        # Direct convolution through wave matrices. The shader gathers each
        # im2col tile into LDS, so no intermediate tensor or extra dispatch is
        # required. The row tile scales from 32 to 128 while each wave retains
        # the same two 16x16 accumulators.
        foreach(LA_CONV "conv_2d:2" "conv_transpose_2d:3" "conv_3d:4")
            string(REPLACE ":" ";" LA_CONV_PARTS ${LA_CONV})
            list(GET LA_CONV_PARTS 0 LA_CONV_NAME)
            list(GET LA_CONV_PARTS 1 LA_CONV_KIND)
            foreach(WSIZE ${DX12_WAVE_SIZES})
                set(LA_CONV_SRC "${SHADER_INPUT_DIR}/conv_linalg.hlsl")
                set(LA_CONV_HDR "${SHADER_OUTPUT_DIR}/${LA_CONV_NAME}_linalg_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LA_CONV_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LA_CONV_HDR}
                        -Vn g_${LA_CONV_NAME}_linalg_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D CONV_KIND=${LA_CONV_KIND}
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LA_CONV_SRC}
                    DEPENDS ${LA_CONV_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${LA_CONV_NAME}_linalg (wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LA_CONV_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${LA_CONV_NAME}_linalg_w${WSIZE}_dxil.h\"\n")

                set(LA_CONV_WIDE_HDR "${SHADER_OUTPUT_DIR}/${LA_CONV_NAME}_linalg_64x32_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LA_CONV_WIDE_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LA_CONV_WIDE_HDR}
                        -Vn g_${LA_CONV_NAME}_linalg_64x32_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D CONV_KIND=${LA_CONV_KIND}
                        -D CONV_BM=64
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LA_CONV_SRC}
                    DEPENDS ${LA_CONV_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${LA_CONV_NAME}_linalg_64x32 (wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LA_CONV_WIDE_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${LA_CONV_NAME}_linalg_64x32_w${WSIZE}_dxil.h\"\n")

                set(LA_CONV_XWIDE_HDR "${SHADER_OUTPUT_DIR}/${LA_CONV_NAME}_linalg_128x32_w${WSIZE}_dxil.h")
                add_custom_command(
                    OUTPUT ${LA_CONV_XWIDE_HDR}
                    COMMAND ${GGML_DX12_DXC_PATH}
                        -T cs_6_10
                        -E main
                        -Fh ${LA_CONV_XWIDE_HDR}
                        -Vn g_${LA_CONV_NAME}_linalg_128x32_w${WSIZE}_dxil
                        -I ${SHADER_INPUT_DIR}
                        -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                        -enable-16bit-types
                        -D WAVE_SIZE=${WSIZE}
                        -D CONV_KIND=${LA_CONV_KIND}
                        -D CONV_BM=128
                        -O3 -Qstrip_debug -Qstrip_reflect
                        ${LA_CONV_SRC}
                    DEPENDS ${LA_CONV_SRC} ${SHADER_INCLUDE_DEPS}
                    COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): ${LA_CONV_NAME}_linalg_128x32 (wave=${WSIZE})"
                )
                target_sources(ggml-dx12 PRIVATE ${LA_CONV_XWIDE_HDR})
                file(APPEND "${SHADER_MASTER_HEADER}" "#include \"${LA_CONV_NAME}_linalg_128x32_w${WSIZE}_dxil.h\"\n")
            endforeach()
        endforeach()

        foreach(WSIZE ${DX12_WAVE_SIZES})
            set(LA_OUT_PROD_SRC "${SHADER_INPUT_DIR}/out_prod_linalg.hlsl")
            set(LA_OUT_PROD_HDR "${SHADER_OUTPUT_DIR}/out_prod_linalg_w${WSIZE}_dxil.h")
            add_custom_command(
                OUTPUT ${LA_OUT_PROD_HDR}
                COMMAND ${GGML_DX12_DXC_PATH}
                    -T cs_6_10
                    -E main
                    -Fh ${LA_OUT_PROD_HDR}
                    -Vn g_out_prod_linalg_w${WSIZE}_dxil
                    -I ${SHADER_INPUT_DIR}
                    -I ${GGML_DX12_DXC_HLSL_INCLUDE}
                    -enable-16bit-types
                    -D WAVE_SIZE=${WSIZE}
                    -O3 -Qstrip_debug -Qstrip_reflect
                    ${LA_OUT_PROD_SRC}
                DEPENDS ${LA_OUT_PROD_SRC} ${SHADER_INCLUDE_DEPS}
                COMMENT "Compiling HLSL shader (SM 6.10, LinAlg preview): out_prod_linalg (wave=${WSIZE})"
            )
            target_sources(ggml-dx12 PRIVATE ${LA_OUT_PROD_HDR})
            file(APPEND "${SHADER_MASTER_HEADER}" "#include \"out_prod_linalg_w${WSIZE}_dxil.h\"\n")
        endforeach()

        target_compile_definitions(ggml-dx12 PRIVATE
            GGML_DX12_LINALG_PREVIEW=1
            GGML_DX12_AGILITY_SDK_VERSION=${GGML_DX12_AGILITY_SDK_VERSION})

        # Stage the Agility SDK redist DLLs next to every consumer binary in a
        # `D3D12/` subfolder. This matches the Agility loader convention
        # (`D3D12SDKPath` is set to `.\\D3D12\\` by the backend at build time).
        if (NOT GGML_DX12_AGILITY_SDK_PATH OR NOT EXISTS "${GGML_DX12_AGILITY_SDK_PATH}")
            message(FATAL_ERROR "ggml-dx12: GGML_DX12_LINALG_PREVIEW=ON but GGML_DX12_AGILITY_SDK_PATH is not set or does not exist. "
                "Download the Agility SDK 1.721.3 preview nupkg from https://devblogs.microsoft.com/directx/directx12agility/ "
                "extract it, and set -DGGML_DX12_AGILITY_SDK_PATH=<extracted>/build/native (or wherever bin/x64/D3D12Core.dll lives).")
        endif()
        set(AGILITY_DLL_DIR "")
        foreach(_candidate
                "${GGML_DX12_AGILITY_SDK_PATH}/bin/x64"
                "${GGML_DX12_AGILITY_SDK_PATH}/build/native/bin/x64"
                "${GGML_DX12_AGILITY_SDK_PATH}")
            if (EXISTS "${_candidate}/D3D12Core.dll")
                set(AGILITY_DLL_DIR "${_candidate}")
                break()
            endif()
        endforeach()
        if (NOT AGILITY_DLL_DIR)
            message(FATAL_ERROR "ggml-dx12: Could not find D3D12Core.dll under ${GGML_DX12_AGILITY_SDK_PATH} "
                "(searched bin/x64, build/native/bin/x64, root). "
                "Check that you've extracted the Agility SDK nupkg correctly.")
        endif()
        message(STATUS "ggml-dx12: LinAlg preview ENABLED (Agility SDK v${GGML_DX12_AGILITY_SDK_VERSION} from ${AGILITY_DLL_DIR}, DXC ${GGML_DX12_DXC_PATH})")

        # Stage the runtime for explicit consumer targets as well as ALL builds.
        set(_agility_d3d12_dll        "${AGILITY_DLL_DIR}/D3D12Core.dll")
        set(_agility_sdklayers_dll    "${AGILITY_DLL_DIR}/d3d12SDKLayers.dll")
        add_custom_target(ggml-dx12-agility-stage ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:ggml-dx12>/D3D12"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_agility_d3d12_dll}" "$<TARGET_FILE_DIR:ggml-dx12>/D3D12/D3D12Core.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_agility_sdklayers_dll}" "$<TARGET_FILE_DIR:ggml-dx12>/D3D12/d3d12SDKLayers.dll"
            DEPENDS "${_agility_d3d12_dll}" "${_agility_sdklayers_dll}"
            COMMENT "Staging Agility SDK preview DLLs into D3D12/ next to ggml-dx12"
            VERBATIM)
        add_dependencies(ggml-dx12 ggml-dx12-agility-stage)
    endif()
else()
    message(WARNING "ggml-dx12: DXC not found - HLSL shaders will not be compiled, compute operations will be unavailable")
endif()
