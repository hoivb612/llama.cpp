cmake_minimum_required(VERSION 3.19)

if (NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(WARNING "ggml-dx12: DirectX 12 backend is only supported on Windows and WSL2")
    return()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_definitions(/MP)
endif()

ggml_add_backend_library(ggml-dx12
    ggml-dx12.cpp
    ../../include/ggml-dx12.h
)

if (WIN32)
    # Native Windows: link D3D12 and DXGI system libraries
    target_link_libraries(ggml-dx12 PRIVATE
        d3d12
        dxgi
        dxguid
    )
else()
    # WSL2: libraries provided by the Windows host at /usr/lib/wsl/lib/
    find_library(D3D12_LIBRARY d3d12 HINTS /usr/lib/wsl/lib)
    find_library(DXCORE_LIBRARY dxcore HINTS /usr/lib/wsl/lib)

    if (NOT D3D12_LIBRARY)
        message(FATAL_ERROR "ggml-dx12: libd3d12.so not found in /usr/lib/wsl/lib/ - is this a WSL2 environment with GPU support?")
    endif()

    target_link_libraries(ggml-dx12 PRIVATE ${D3D12_LIBRARY})
    if (DXCORE_LIBRARY)
        target_link_libraries(ggml-dx12 PRIVATE ${DXCORE_LIBRARY})
    endif()

    # DirectX GUIDs library (provides DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE etc.)
    find_library(DX_GUIDS_LIBRARY DirectX-Guids HINTS /usr/local/lib /usr/lib)
    if (DX_GUIDS_LIBRARY)
        target_link_libraries(ggml-dx12 PRIVATE ${DX_GUIDS_LIBRARY})
        message(STATUS "ggml-dx12: Found DirectX-Guids at ${DX_GUIDS_LIBRARY}")
    endif()

    # DirectX-Headers for Linux (provides d3d12.h, dxgi1_6.h, wrl/client.h stubs)
    find_package(DirectX-Headers QUIET)
    if (DirectX-Headers_FOUND)
        target_link_libraries(ggml-dx12 PRIVATE Microsoft::DirectX-Headers)
        message(STATUS "ggml-dx12: Using system DirectX-Headers")
    else()
        # Fallback: look for headers in common install locations
        find_path(DX_HEADERS_INCLUDE_DIR "directx/d3d12.h"
            HINTS /usr/include /usr/local/include)
        if (DX_HEADERS_INCLUDE_DIR)
            target_include_directories(ggml-dx12 PRIVATE ${DX_HEADERS_INCLUDE_DIR})
            message(STATUS "ggml-dx12: Found DirectX headers at ${DX_HEADERS_INCLUDE_DIR}")
        else()
            message(WARNING "ggml-dx12: DirectX-Headers not found - install via 'sudo apt install directx-headers-dev' or from https://github.com/microsoft/DirectX-Headers")
        endif()
    endif()

    # WSL stubs for WRL (ComPtr) and winadapter
    find_path(WSL_STUBS_INCLUDE_DIR "wrl/client.h"
        HINTS /usr/include/wsl/stubs /usr/local/include/wsl/stubs)
    if (WSL_STUBS_INCLUDE_DIR)
        target_include_directories(ggml-dx12 PRIVATE ${WSL_STUBS_INCLUDE_DIR})
        message(STATUS "ggml-dx12: Found WSL stubs at ${WSL_STUBS_INCLUDE_DIR}")
    else()
        message(WARNING "ggml-dx12: WSL stubs (wrl/client.h) not found - install directx-headers-dev")
    endif()

    # winadapter.h (provides Windows type definitions for Linux)
    find_path(WSL_WINADAPTER_DIR "winadapter.h"
        HINTS /usr/include/wsl /usr/local/include/wsl
              /usr/include/directx /usr/local/include/directx
              /usr/include /usr/local/include)
    if (WSL_WINADAPTER_DIR)
        target_include_directories(ggml-dx12 PRIVATE ${WSL_WINADAPTER_DIR})
        message(STATUS "ggml-dx12: Found winadapter.h at ${WSL_WINADAPTER_DIR}")
    else()
        message(WARNING "ggml-dx12: winadapter.h not found - install directx-headers-dev or DirectX-Headers from source")
    endif()

    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_WSL2)
endif()

target_include_directories(ggml-dx12 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Debug options
if (GGML_DX12_DEBUG)
    target_compile_definitions(ggml-dx12 PRIVATE GGML_DX12_DEBUG)
endif()

# HLSL Shader Compilation
# Find DXC (DirectX Shader Compiler)
find_program(DXC_EXECUTABLE dxc
    HINTS
        "$ENV{WindowsSdkDir}/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64"
        "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        "/usr/bin"
)

if (NOT DXC_EXECUTABLE)
    message(FATAL_ERROR
        "ggml-dx12: DXC (DirectX Shader Compiler) not found!\n"
        "  GPU compute will produce garbage without compiled shaders.\n"
        "  Install dxc or set -DDXC_EXECUTABLE=/path/to/dxc\n"
        "  On Ubuntu/WSL2: download stable release from\n"
        "  https://github.com/microsoft/DirectXShaderCompiler/releases")
endif()

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
    set
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
    mul_mat_vec_q4k_32
    mul_mat_vec_q4k_mr
    mul_mat_vec_q5k
    mul_mat_vec_q5k_mr
    mul_mat_vec_q2k
    mul_mat_vec_q3k
    mul_mat_vec_q6k
    mul_mat_vec_q6k_32
    mul_mat_vec_q6k_mr
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
