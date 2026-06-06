# perfavx: standalone AVX2/AVX512 microbench.
#
# Razzle:
#   perfavx\za-exe -> perfavxza (defines __GEN_ZA_VERSION__, links only vcomp)
#   perfavx\zo-exe -> perfavxzo (links only vcomp)
#
# perfavx loads the za-ggml / zo-ggml DLLs dynamically (LoadLibrary), so no
# direct link dependency on them.

add_executable(perfavxza perfavx.c)
zslm_apply_common_flags(perfavxza)
zslm_apply_avx512(perfavxza)
target_compile_definitions(perfavxza PRIVATE __GEN_ZA_VERSION__)
target_include_directories(perfavxza PRIVATE
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(perfavxza PRIVATE OpenMP::OpenMP_C)

add_executable(perfavxzo perfavx.c)
zslm_apply_common_flags(perfavxzo)
zslm_apply_avx512(perfavxzo)
target_include_directories(perfavxzo PRIVATE
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(perfavxzo PRIVATE OpenMP::OpenMP_C)
