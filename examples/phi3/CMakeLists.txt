set(TARGET phi3)
add_executable(${TARGET}
    Phi3.cpp
    phi3_loader.cpp
    phi3_transform.cpp
    phi3_kernels.cpp
    phi3_runtime.cpp
)
set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME Phi3)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
