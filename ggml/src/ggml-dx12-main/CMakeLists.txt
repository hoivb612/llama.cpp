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
        mul_mat
        mul_mat_id
        mul_mat_id_coop
        mul_mat_id_q4_0
        mul_mat_id_q4_0_coop
        mul_mat_id_q4_1
        mul_mat_id_q4_1_coop
        mul_mat_id_q5_0
        mul_mat_id_q5_0_coop
        mul_mat_id_q5_1
        mul_mat_id_q5_1_coop
        mul_mat_id_q8_0
        mul_mat_id_q4k
        mul_mat_id_q4k_coop
        mul_mat_id_q4k_block
        mul_mat_id_q5k
        mul_mat_id_q5k_coop
        mul_mat_id_q6k
        mul_mat_id_q6k_coop
        mul_mat_id_iq4_nl
        mul_mat_id_iq4_nl_coop
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
        cpy
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
        ssm_scan_d256
        wkv6
        wkv7
        upscale
        im2col
        im2col_3d
        pool_2d
        dequant_q4_k
        flash_attn
        flash_attn_64
        flash_attn_128
        flash_attn_pf_64
        flash_attn_pf_64_wide
        flash_attn_pf_96
        flash_attn_pf_128
        flash_attn_gqa
        flash_attn_reduce
        flash_attn_q4_0
        flash_attn_q4_1
        flash_attn_q5_0
        flash_attn_q5_1
        flash_attn_q8_0
        flash_attn_iq4_nl
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
        rms_norm_mul_quantize_q8_1
        rms_norm_mul_rope
        rms_norm_mul_rope_set_rows
        rms_norm_mul_rope_qk
        add_rms_norm_mul
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_f16_wave64
        mul_mat_vec_qkv_f16_wave64
        mul_mat_vec_qkv_f16_wave64_rms
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
        mul_mat_vec_q4k_dp4a_mr4
        mul_mat_vec_q4k_dp4a_nc2
        mul_mat_vec_q4k_dp4a_nc4
        mul_mat_vec_q4k_dp4a_nc8
        mul_mat_vec_q5k_dp4a_nc2
        mul_mat_vec_q6k_dp4a_nc2
        mul_mat_vec_q8_0_dp4a_nc2
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
        mul_mat_vec_glu_q4_k_rms
        mul_mat_vec_glu_q5_k
        mul_mat_vec_glu_q3_k
        mul_mat_vec_glu_q8_0
        mul_mat_vec_glu_q8_0_rms
        mul_mat_vec_glu_q8_0_dp4a_mr64
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
        mul_mat_q4k_q8_1_tiled
        mul_mat_q5k_q8_1_tiled
        mul_mat_q5_0_q8_1_tiled
        mul_mat_q6k_q8_1_tiled
        mul_mat_vec_q4k_dp4a
        mul_mat_vec_q4k_dp4a_32
        mul_mat_vec_q5k_dp4a
        mul_mat_vec_q5k_dp4a_32
        mul_mat_vec_q6k_dp4a
        mul_mat_vec_q3k_dp4a
        mul_mat_vec_q8_0_dp4a
        mul_mat_id_q8_0_dp4a
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
        mul_mat_vec_glu
        mul_mat_vec_glu_rms
        flash_attn
        flash_attn_gqa
        flash_attn_tiled
        flash_attn_tiled16
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
        set_rows_q8_0
        set_rows_q4_0
        set_rows_q4_1
        set_rows_q5_0
        set_rows_q5_1
        set_rows_iq4_nl
        cpy_quant_block
        mul_mat_wmma_fp16
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
        flash_attn_pf_64
        flash_attn_pf_64_wide
        flash_attn_pf_96
        flash_attn_pf_128
        flash_attn_gqa
        flash_attn_reduce
        flash_attn_q4_0
        flash_attn_q4_1
        flash_attn_q5_0
        flash_attn_q5_1
        flash_attn_q8_0
        flash_attn_iq4_nl
        flash_attn_cd_64
        flash_attn_cd_96
        flash_attn_cd_128
        flash_attn_cd_q8_0_64
        flash_attn_cd_q8_0_96
        flash_attn_cd_q8_0_128
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
        mul_mat_vec_glu_q4_k
        mul_mat_vec_glu_q4_k_rms
    )
    set(DX12_MMV_TUNE_GROUP_SIZES 16 32 64 128 256 512)

    # Shader files that are #included by other shaders (transitive deps).
    # Add to DEPENDS so changes trigger rebuilds of dependent shaders.
    set(SHADER_INCLUDE_DEPS
        ${SHADER_INPUT_DIR}/ggml_common.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant_coop.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_quant.hlsli
        ${SHADER_INPUT_DIR}/quant_dequant.hlsli
        ${SHADER_INPUT_DIR}/rope_yarn.hlsli
        ${SHADER_INPUT_DIR}/flash_attn.hlsl
        ${SHADER_INPUT_DIR}/flash_attn_tiled.hlsl
        ${SHADER_INPUT_DIR}/flash_attn_cd.hlsli
        ${SHADER_INPUT_DIR}/flash_attn_pf.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_vec_q4k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q5k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q6k_mr_blocked.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_qkv_q8_0_wave64_rows2.hlsl
        ${SHADER_INPUT_DIR}/gated_delta_net_impl.hlsli
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
            set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
            set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_w${WSIZE}_dxil.h")

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
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
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
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
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
                DEPENDS ${shader_src} ${SHADER_INCLUDE_DEPS}
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

    target_sources(ggml-dx12 PRIVATE ${DX12_SHADER_HEADERS})
    target_include_directories(ggml-dx12 PRIVATE ${SHADER_OUTPUT_DIR})
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_SHADERS_COMPILED)
else()
    message(WARNING "ggml-dx12: DXC not found - HLSL shaders will not be compiled, compute operations will be unavailable")
endif()
