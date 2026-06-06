# batchslm: za and zo console programs (batchslmza, batchslmzo)
#
# Razzle:
#   batchslm\za-exe\sources -> batchslmza (links za-common DLL import lib)
#   batchslm\zo-exe\sources -> batchslmzo (links zo-ggml-avx512 + zo-common static)

# ---- batchslmza -----------------------------------------------------------

add_executable(batchslmza batchslm.cpp)
zslm_apply_common_flags(batchslmza)
zslm_apply_avx2(batchslmza)

target_compile_definitions(batchslmza PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(batchslmza PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(batchslmza PRIVATE
    za-common-shared
    kernel32)

# ---- batchslmzo -----------------------------------------------------------

add_executable(batchslmzo batchslm.cpp)
zslm_apply_common_flags(batchslmzo)
zslm_apply_avx2(batchslmzo)

target_compile_definitions(batchslmzo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(batchslmzo PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(batchslmzo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
