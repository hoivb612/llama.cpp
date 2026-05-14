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
        rms_norm
        norm
        group_norm
        soft_max
        mul_mat
        mul_mat_id
        mul_mat_id_q4_0
        mul_mat_id_q4_1
        mul_mat_id_q5_0
        mul_mat_id_q5_1
        mul_mat_id_q8_0
        mul_mat_id_q4k
        mul_mat_id_q5k
        mul_mat_id_q6k
        mul_mat_id_iq4_nl
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
        pad
        roll
        ssm_conv
        ssm_scan
        upscale
        im2col
        pool_2d
        dequant_q4_k
        flash_attn
        flash_attn_64
        flash_attn_128
        flash_attn_gqa
        flash_attn_reduce
        set_rows
        rope_set_rows
        conv_2d
        cast
        pool_1d
        get_rows_q4k
        get_rows_q5k
        get_rows_q6k
        glu
        l2_norm
        gated_delta_net
        rms_norm_mul
        rms_norm_mul_rope
        rms_norm_mul_rope_set_rows
        add_rms_norm_mul
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_q4k_mr
        mul_mat_vec_q5k_mr
        mul_mat_vec_q5k_subgroup
        mul_mat_vec_q6k_mr
        mul_mat_vec_q6k_mr_blocked
        mul_mat_vec_q3k_mr_blocked
        mul_mat_vec_q2k_mr_blocked
        mul_mat_vec_q8_0_mr64
        mul_mat_vec_q5_0_mr64
        mul_mat_vec_q5_0_mr
        mul_mat_vec_q5_1_mr
        mul_mat_vec_q8_0_mr
        mul_mat_vec_q8_0_mr256v
        mul_mat_vec_glu
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
        get_rows_iq4_nl
        mul_mat_wmma
        mul_mat_q4k_wmma
        mul_mat_q4k_wmma_lds
        mul_mat_q5k_wmma
        mul_mat_q6k_wmma
        mul_mat_q8_0_wmma
        quantize_q8_1
    )

    # Shaders requiring SM 6.6 (dot4add_i8packed / dp4a — built at 6.6 with rest)
    set(DX12_SHADERS_SM64
        mul_mat_q8_0_q8_1
        mul_mat_vec_q4k_dp4a
        mul_mat_vec_q4k_dp4a_32
        mul_mat_vec_q5k_dp4a
        mul_mat_vec_q5k_dp4a_32
        mul_mat_vec_q6k_dp4a
        mul_mat_vec_q8_0_dp4a
        mul_mat_vec_q5_0_dp4a
        mul_mat_vec_q5_1_dp4a
    )

    # Shaders compiled twice: once FP32-only and once with native 16-bit types
    # enabled (-enable-16bit-types + -DNATIVE_FP16=1). Runtime selects the FP16
    # variant on devices that report OPTIONS4.Native16BitShaderOpsSupported.
    set(DX12_SHADERS_FP16_VARIANT
        mul_mat_vec
        mul_mat_vec_mr
        mul_mat_vec_mr32
        mul_mat_vec_glu
        flash_attn
        flash_attn_gqa
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
        flash_attn_gqa
        flash_attn_reduce
    )

    # Compile each shader for wave sizes 16, 32, 64
    # At runtime, the device selects the matching set based on GPU wave size
    set(DX12_WAVE_SIZES 16 32 64)

    # Shader files that are #included by other shaders (transitive deps).
    # Add to DEPENDS so changes trigger rebuilds of dependent shaders.
    set(SHADER_INCLUDE_DEPS
        ${SHADER_INPUT_DIR}/ggml_common.hlsli
        ${SHADER_INPUT_DIR}/mul_mat_id_quant.hlsli
        ${SHADER_INPUT_DIR}/rope_yarn.hlsli
        ${SHADER_INPUT_DIR}/flash_attn.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q4k_dp4a.hlsl
        ${SHADER_INPUT_DIR}/mul_mat_vec_q5k_dp4a.hlsl
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
                COMMENT "Compiling HLSL shader (SM 6.6): ${shader_name} (wave=${WSIZE})"
            )
            list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
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
        foreach(shader_name ${DX12_SHADERS_FP16_VARIANT})
            string(APPEND SHADER_INCLUDES "#include \"${shader_name}_w${WSIZE}_fp16_dxil.h\"\n")
        endforeach()
    endforeach()
    file(WRITE "${SHADER_MASTER_HEADER}" "// Auto-generated shader includes\n#pragma once\n${SHADER_INCLUDES}")

    target_sources(ggml-dx12 PRIVATE ${DX12_SHADER_HEADERS})
    target_include_directories(ggml-dx12 PRIVATE ${SHADER_OUTPUT_DIR})
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_SHADERS_COMPILED)
else()
    message(WARNING "ggml-dx12: DXC not found - HLSL shaders will not be compiled, compute operations will be unavailable")
endif()
