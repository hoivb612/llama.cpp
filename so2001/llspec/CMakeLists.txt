# llspec: speculative decoding driver

add_executable(llspec-za llspec.cpp)
zslm_apply_common_flags(llspec-za)
zslm_apply_avx2(llspec-za)
target_compile_definitions(llspec-za PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llspec-za PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(llspec-za PRIVATE za-common-shared kernel32)

add_executable(llspec-zo llspec.cpp)
zslm_apply_common_flags(llspec-zo)
zslm_apply_avx512(llspec-zo)
target_compile_definitions(llspec-zo PRIVATE __SET_PROCESS_AFFINITY__)
target_include_directories(llspec-zo PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/zo-ggml)
target_link_libraries(llspec-zo PRIVATE
    zo-ggml-avx512
    zo-common
    powrprof
    OpenMP::OpenMP_CXX)
