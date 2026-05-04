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
        relu
        tanh_
        sigmoid
        rms_norm
        norm
        soft_max
        mul_mat
        cpy
        get_rows
        diag_mask_inf
        rope
        clamp
        concat
        repeat
        sum_rows
        dequant_q4_k
        flash_attn
        flash_attn_reduce
        set_rows
        rope_set_rows
        mul_mat_q2k
        mul_mat_q3k
        mul_mat_q4k
        get_rows_q4k
        mul_mat_q5k
        get_rows_q5k
        mul_mat_q6k
        get_rows_q2k
        get_rows_q3k
        get_rows_q6k
        glu
        rms_norm_mul
        rms_norm_mul_rope
        add_rms_norm_mul
        mul_mat_vec
        mul_mat_vec_q4k
        mul_mat_vec_q4k_mr
        mul_mat_vec_q5k
        mul_mat_vec_q2k
        mul_mat_vec_q3k
        mul_mat_vec_q6k
        mul_mat_vec_q6k_32
        mul_mat_vec_q5_0
        mul_mat_vec_q5_0_256
        mul_mat_vec_q8_0
        mul_mat_vec_q8_0_256
        mul_mat_vec_load4
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
        mul_mat_wmma
        mul_mat_q4k_wmma
        mul_mat_q5k_wmma
        mul_mat_q6k_wmma
    )

    set(DX12_SHADER_HEADERS "")

    foreach(shader_name ${DX12_SHADERS})
        set(shader_src "${SHADER_INPUT_DIR}/${shader_name}.hlsl")
        set(shader_hdr "${SHADER_OUTPUT_DIR}/${shader_name}_dxil.h")

        add_custom_command(
            OUTPUT ${shader_hdr}
            COMMAND ${DXC_EXECUTABLE}
                -T cs_6_0
                -E main
                -Fh ${shader_hdr}
                -Vn g_${shader_name}_dxil
                -I ${SHADER_INPUT_DIR}
                -O3
                ${shader_src}
            DEPENDS ${shader_src} ${SHADER_INPUT_DIR}/ggml_common.hlsli
            COMMENT "Compiling HLSL shader: ${shader_name}"
        )
        list(APPEND DX12_SHADER_HEADERS ${shader_hdr})
    endforeach()

    # Generate a master include header at build time (survives partial cleans)
    set(SHADER_MASTER_HEADER "${SHADER_OUTPUT_DIR}/ggml_dx12_shaders.h")

    # Write a cmake script that generates the header (stored outside shaders/ dir)
    set(_gen_script "${CMAKE_CURRENT_BINARY_DIR}/_gen_master_header.cmake")
    file(WRITE "${_gen_script}" "set(_h \"${SHADER_MASTER_HEADER}\")\n")
    file(APPEND "${_gen_script}" "file(WRITE \${_h} \"// Auto-generated shader includes\\n#pragma once\\n\")\n")
    foreach(shader_name ${DX12_SHADERS})
        file(APPEND "${_gen_script}" "file(APPEND \${_h} \"#include \\\"${shader_name}_dxil.h\\\"\\n\")\n")
    endforeach()

    # Run the script at build time when any shader header changes or master is missing
    add_custom_command(
        OUTPUT ${SHADER_MASTER_HEADER}
        COMMAND ${CMAKE_COMMAND} -P "${_gen_script}"
        DEPENDS ${DX12_SHADER_HEADERS}
        COMMENT "Generating ggml_dx12_shaders.h"
    )
    list(APPEND DX12_SHADER_HEADERS ${SHADER_MASTER_HEADER})

    target_sources(ggml-dx12 PRIVATE ${DX12_SHADER_HEADERS})
    target_include_directories(ggml-dx12 PRIVATE ${SHADER_OUTPUT_DIR})
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_SHADERS_COMPILED)
else()
    message(WARNING "ggml-dx12: DXC not found - HLSL shaders will not be compiled, compute operations will be unavailable")
endif()
