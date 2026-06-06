# clip: image embedding index/search built on top of za-ggml-avx512
#
# Razzle:
#   clip\clipindex\sources -> clipindex (clip.cpp + common-clip.cpp + clipindex.cpp)
#   clip\clipsrch\sources  -> clipsrch  (clip.cpp + common-clip.cpp + clipsrch.cpp)

set(_CLIP_BASE clip.cpp common-clip.cpp)

# ---- clipindex ------------------------------------------------------------

add_executable(clipindex ${_CLIP_BASE} clipIndex.cpp)
zslm_apply_common_flags(clipindex)
zslm_apply_avx512(clipindex)
target_compile_definitions(clipindex PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(clipindex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(clipindex PRIVATE
    za-ggml-avx512
    za-common
    powrprof
    kernel32
    OpenMP::OpenMP_CXX)

# ---- clipsrch -------------------------------------------------------------

add_executable(clipsrch ${_CLIP_BASE} clipSrch.cpp)
zslm_apply_common_flags(clipsrch)
zslm_apply_avx512(clipsrch)
target_compile_definitions(clipsrch PRIVATE
    __SET_PROCESS_AFFINITY__
    __TENSOR_REPACK__)
target_include_directories(clipsrch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/za-ggml)
target_link_libraries(clipsrch PRIVATE
    za-ggml-avx512
    za-common
    powrprof
    kernel32
    OpenMP::OpenMP_CXX)
