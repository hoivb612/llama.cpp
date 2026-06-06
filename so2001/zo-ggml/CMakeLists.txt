# zo-ggml: Zen-optimized GGML without tensor repacking.
#
# Razzle layout (under zo-ggml\):
#   zo-ggml-avx2\sources       -> static lib  zo-ggml-avx2.lib       (AVX2 codegen)
#   zo-ggml-avx512\sources     -> static lib  zo-ggml-avx512.lib     (AVX512 + __GEN_AVX512__)
#   zo-ggml-avx2-dll\sources   -> DLL         zo-ggml-avx2.dll
#   zo-ggml-avx512-dll\sources -> DLL         zo-ggml-avx512.dll

set(_ZO_GGML_SOURCES
    ggml-alloc.c
    ggml-backend.c
    ggml-quants.c
    ggml.c
    ggml-q4.c)

# ---- zo-ggml-avx2 (static) ------------------------------------------------

add_library(zo-ggml-avx2 STATIC ${_ZO_GGML_SOURCES})
zslm_apply_common_flags(zo-ggml-avx2)
zslm_apply_avx2(zo-ggml-avx2)

target_compile_options(zo-ggml-avx2 PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(zo-ggml-avx2 PUBLIC OpenMP::OpenMP_C powrprof)
target_include_directories(zo-ggml-avx2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})

# ---- zo-ggml-avx512 (static, AVX512 + __GEN_AVX512__) ---------------------

add_library(zo-ggml-avx512 STATIC ${_ZO_GGML_SOURCES})
zslm_apply_common_flags(zo-ggml-avx512)
zslm_apply_avx512(zo-ggml-avx512)

target_compile_definitions(zo-ggml-avx512 PRIVATE __GEN_AVX512__)

target_compile_options(zo-ggml-avx512 PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(zo-ggml-avx512 PUBLIC OpenMP::OpenMP_C powrprof)
target_include_directories(zo-ggml-avx512 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})

# ---- zo-ggml-avx2-dll -----------------------------------------------------
#
# Unlike za-ggml's DLL (which just wraps the static lib), zo-ggml's DLL
# compiles the ggml sources directly. Mirror that.

add_library(zo-ggml-avx2-dll SHARED dllmain.cpp ${_ZO_GGML_SOURCES} zo-ggmlapi.def)
zslm_apply_common_flags(zo-ggml-avx2-dll)
zslm_apply_avx2(zo-ggml-avx2-dll)
set_target_properties(zo-ggml-avx2-dll PROPERTIES
    OUTPUT_NAME zo-ggml-avx2
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dll-import")
target_include_directories(zo-ggml-avx2-dll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})
target_link_libraries(zo-ggml-avx2-dll PRIVATE
    kernel32
    powrprof
    OpenMP::OpenMP_C)

# ---- zo-ggml-avx512-dll ---------------------------------------------------

add_library(zo-ggml-avx512-dll SHARED dllmain.cpp ${_ZO_GGML_SOURCES} zo-ggmlapi.def)
zslm_apply_common_flags(zo-ggml-avx512-dll)
zslm_apply_avx512(zo-ggml-avx512-dll)
target_compile_definitions(zo-ggml-avx512-dll PRIVATE __GEN_AVX512__)
set_target_properties(zo-ggml-avx512-dll PROPERTIES
    OUTPUT_NAME zo-ggml-avx512
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dll-import")
target_include_directories(zo-ggml-avx512-dll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})
target_link_libraries(zo-ggml-avx512-dll PRIVATE
    kernel32
    powrprof
    OpenMP::OpenMP_C)
