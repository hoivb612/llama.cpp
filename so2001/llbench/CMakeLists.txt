# llbench: per-token timing benchmark.
#
# Razzle variants:
#   llbench\za-exe         -> llbenchza         (links za-common DLL)
#   llbench\za-exe-avx2    -> llbenchza-avx2    (static za-common + za-ggml-avx2)
#   llbench\za-exe-avx512  -> llbenchza-avx512  (static za-common + za-ggml-avx512)
#   llbench\zo-exe         -> llbenchzo         (zo-common + zo-ggml-avx512)
#   llbench\zo-exe-avx2    -> llbenchzo-avx2    (zo-common + zo-ggml-avx2)

# ---- llbenchza (uses za-common DLL) ---------------------------------------

add_executable(llbenchza llbench.cpp)
zslm_apply_common_flags(llbenchza)
zslm_apply_avx2(llbenchza)
target_compile_definitions(llbenchza PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(llbenchza PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llbenchza PRIVATE za-common-shared kernel32)

# ---- llbenchza-avx2 (static link) -----------------------------------------

add_executable(llbenchza-avx2 llbench.cpp)
zslm_apply_common_flags(llbenchza-avx2)
zslm_apply_avx2(llbenchza-avx2)
target_compile_definitions(llbenchza-avx2 PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(llbenchza-avx2 PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llbenchza-avx2 PRIVATE
    za-ggml-avx2
    za-common
    powrprof
    kernel32
    OpenMP::OpenMP_CXX)

# ---- llbenchza-avx512 -----------------------------------------------------

add_executable(llbenchza-avx512 llbench.cpp)
zslm_apply_common_flags(llbenchza-avx512)
zslm_apply_avx512(llbenchza-avx512)
target_compile_definitions(llbenchza-avx512 PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(llbenchza-avx512 PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llbenchza-avx512 PRIVATE
    za-ggml-avx512
    za-common
    powrprof
    kernel32
    OpenMP::OpenMP_CXX)

# ---- llbenchzo (AVX512) ---------------------------------------------------

add_executable(llbenchzo llbench.cpp)
zslm_apply_common_flags(llbenchzo)
zslm_apply_avx512(llbenchzo)
target_compile_definitions(llbenchzo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llbenchzo PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(llbenchzo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)

# ---- llbenchzo-avx2 -------------------------------------------------------

add_executable(llbenchzo-avx2 llbench.cpp)
zslm_apply_common_flags(llbenchzo-avx2)
zslm_apply_avx2(llbenchzo-avx2)
target_compile_definitions(llbenchzo-avx2 PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llbenchzo-avx2 PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(llbenchzo-avx2 PRIVATE
    zo-ggml-avx2
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
