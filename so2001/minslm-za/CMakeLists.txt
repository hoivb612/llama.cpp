# minslm-za: minimal SLM driver (minslmza, minslmzo)

set(_MINSLM_SOURCES minslm.cpp slminfer.cpp)

# ---- minslmza -------------------------------------------------------------

add_executable(minslmza ${_MINSLM_SOURCES})
zslm_apply_common_flags(minslmza)
zslm_apply_avx2(minslmza)

target_compile_definitions(minslmza PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(minslmza PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(minslmza PRIVATE
    za-common-shared
    kernel32)

# ---- minslmzo -------------------------------------------------------------

add_executable(minslmzo ${_MINSLM_SOURCES})
zslm_apply_common_flags(minslmzo)
zslm_apply_avx2(minslmzo)

target_compile_definitions(minslmzo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(minslmzo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(minslmzo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
