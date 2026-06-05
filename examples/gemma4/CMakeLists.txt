set(TARGET gemma4)
add_executable(${TARGET}
    Gemma4.cpp
    gemma4_loader.cpp
    gemma4_baseline.cpp
    gemma4_weights.cpp
    gemma4_kernels.cpp
    gemma4_matmul.cpp
    gemma4_forward.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME Gemma4)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
