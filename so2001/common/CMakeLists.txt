# common library (llama.cpp / sampling / unicode / json-schema-to-grammar / etc.)
#
# Razzle layout:
#   common\za-common\sources       -> static lib  za-common.lib  (+ __TENSOR_REPACK__)
#   common\za-common-dll\sources   -> DLL         za-common.dll  (links za-ggml-avx512.lib)
#   common\zo-common\sources       -> static lib  zo-common.lib

set(_COMMON_SOURCES
    common.cpp
    grammar-parser.cpp
    json-schema-to-grammar.cpp
    llama.cpp
    sampling.cpp
    unicode.cpp
    unicode-data.cpp)

# ---- za-common (static, AVX2 codegen + __TENSOR_REPACK__) -----------------

add_library(za-common STATIC ${_COMMON_SOURCES})
zslm_apply_common_flags(za-common)
zslm_apply_avx2(za-common)

target_compile_definitions(za-common PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)

target_include_directories(za-common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/za-ggml)

# ---- zo-common (static, AVX2 codegen) -------------------------------------

add_library(zo-common STATIC ${_COMMON_SOURCES})
zslm_apply_common_flags(zo-common)
zslm_apply_avx2(zo-common)

target_compile_definitions(zo-common PRIVATE
    __SET_PROCESS_AFFINITY__)

target_include_directories(zo-common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/zo-ggml)

# ---- za-common-shared (DLL exporting symbols in za-common.def) ------------
#
# The Razzle DLL is built from the SAME sources as za-common but linked to
# za-ggml-avx512.lib + powrprof.lib + vcomp.lib and exports symbols listed in
# za-common.def.

add_library(za-common-shared SHARED ${_COMMON_SOURCES} za-common.def)
zslm_apply_common_flags(za-common-shared)
zslm_apply_avx512(za-common-shared)

set_target_properties(za-common-shared PROPERTIES
    OUTPUT_NAME za-common
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/dll-import")

target_compile_definitions(za-common-shared PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)

target_include_directories(za-common-shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/za-ggml)

target_link_libraries(za-common-shared PRIVATE
    za-ggml-avx512
    powrprof
    kernel32
    OpenMP::OpenMP_CXX)
