# llindex: index builder/runner

add_executable(llindex-za llindex.cpp)
zslm_apply_common_flags(llindex-za)
zslm_apply_avx2(llindex-za)
target_compile_definitions(llindex-za PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llindex-za PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llindex-za PRIVATE za-common-shared kernel32)

add_executable(llindex-zo llindex.cpp)
zslm_apply_common_flags(llindex-zo)
zslm_apply_avx512(llindex-zo)
target_compile_definitions(llindex-zo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llindex-zo PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(llindex-zo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
