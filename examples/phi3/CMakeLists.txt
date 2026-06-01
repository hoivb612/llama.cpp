set(TARGET Phi3)
add_executable(${TARGET}
    Phi3.cpp
    phi3_loader.cpp
    phi3_transform.cpp
    phi3_kernels.cpp
    phi3_runtime.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
