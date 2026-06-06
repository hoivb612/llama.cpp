# za-ggml: Zen-optimized GGML with tensor repacking.
#
# Razzle layout (under za-ggml\):
#   za-ggml-avx2\sources       -> static lib  za-ggml-avx2.lib       (AVX2 codegen)
#   za-ggml-avx512\sources     -> static lib  za-ggml-avx512.lib     (AVX512 + __GEN_AVX512__)
#   za-ggml-avx2-dll\sources   -> DLL         za-ggml-avx2.dll
#   za-ggml-avx512-dll\sources -> DLL         za-ggml-avx512.dll
#
# Shared sources.inc adds:
#   /openmp, /arch:AVX2, /arch:AVX512, /fp:fast, /d2jumptablerdata
#   -DWIN32_LEAN_AND_MEAN -D__AVX512VNNI__ -D__AVX512VL__
#
# The static libs build from ..\ggml-*.c (parent dir = za-ggml).
# The DLLs build from dllmain.cpp and link the static lib.

set(_ZA_GGML_SOURCES
    ggml-alloc.c
    ggml-backend.c
    ggml-repack.c
    ggml-quants.c
    ggml-q4.c
    ggml.c)

# ---- za-ggml-avx2 (static) ------------------------------------------------

add_library(za-ggml-avx2 STATIC ${_ZA_GGML_SOURCES})
zslm_apply_common_flags(za-ggml-avx2)
zslm_apply_avx2(za-ggml-avx2)

# Although built with /arch:AVX2 only, sources.inc force-defines these so the
# VNNI/VL fast paths are visible. Mirror that. Real VNNI/VL intrinsics are
# only enabled inside #if defined(__GEN_AVX512__) guards (not set here).
target_compile_definitions(za-ggml-avx2 PRIVATE
    __AVX512VNNI__
    __AVX512VL__)

target_compile_options(za-ggml-avx2 PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(za-ggml-avx2 PUBLIC OpenMP::OpenMP_C powrprof)
target_include_directories(za-ggml-avx2 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})

# ---- za-ggml-avx512 (static, AVX512 codegen + __GEN_AVX512__) -------------

add_library(za-ggml-avx512 STATIC ${_ZA_GGML_SOURCES})
zslm_apply_common_flags(za-ggml-avx512)
zslm_apply_avx512(za-ggml-avx512)

target_compile_definitions(za-ggml-avx512 PRIVATE __GEN_AVX512__)

target_compile_options(za-ggml-avx512 PRIVATE ${OpenMP_C_FLAGS})
target_link_libraries(za-ggml-avx512 PUBLIC OpenMP::OpenMP_C powrprof)
target_include_directories(za-ggml-avx512 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR})

# ---- za-ggml-avx2-dll (DLL with za-ggmlapi.def) ---------------------------

add_library(za-ggml-avx2-dll SHARED dllmain.cpp za-ggmlapi.def)
zslm_apply_common_flags(za-ggml-avx2-dll)
zslm_apply_avx2(za-ggml-avx2-dll)
set_target_properties(za-ggml-avx2-dll PROPERTIES
    OUTPUT_NAME za-ggml-avx2
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dll-import")
target_link_libraries(za-ggml-avx2-dll PRIVATE
    za-ggml-avx2
    kernel32
    OpenMP::OpenMP_C)

# ---- za-ggml-avx512-dll (DLL with za-ggmlapi.def) -------------------------

add_library(za-ggml-avx512-dll SHARED dllmain.cpp za-ggmlapi.def)
zslm_apply_common_flags(za-ggml-avx512-dll)
zslm_apply_avx512(za-ggml-avx512-dll)
target_compile_definitions(za-ggml-avx512-dll PRIVATE __GEN_AVX512__)
set_target_properties(za-ggml-avx512-dll PROPERTIES
    OUTPUT_NAME za-ggml-avx512
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dll-import")
target_link_libraries(za-ggml-avx512-dll PRIVATE
    za-ggml-avx512
    kernel32
    OpenMP::OpenMP_C)
