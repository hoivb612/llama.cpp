# llbench-b: variant of llbench (batch?)

add_executable(llbench-b-za llbench-b.cpp)
zslm_apply_common_flags(llbench-b-za)
zslm_apply_avx2(llbench-b-za)
target_compile_definitions(llbench-b-za PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llbench-b-za PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llbench-b-za PRIVATE za-common-shared kernel32)

add_executable(llbench-b-zo llbench-b.cpp)
zslm_apply_common_flags(llbench-b-zo)
zslm_apply_avx512(llbench-b-zo)
target_compile_definitions(llbench-b-zo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llbench-b-zo PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(llbench-b-zo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
