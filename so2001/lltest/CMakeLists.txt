# lltest: micro tests built two ways (AVX2 and AVX512)

add_executable(lltestza-avx2 lltest.c)
zslm_apply_common_flags(lltestza-avx2)
zslm_apply_avx2(lltestza-avx2)
target_include_directories(lltestza-avx2 PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(lltestza-avx2 PRIVATE kernel32)

add_executable(lltestza-avx512 lltest.c)
zslm_apply_common_flags(lltestza-avx512)
zslm_apply_avx512(lltestza-avx512)
target_include_directories(lltestza-avx512 PRIVATE
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(lltestza-avx512 PRIVATE kernel32)
