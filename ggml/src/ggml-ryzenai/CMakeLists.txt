message(STATUS "RyzenAI backend: configuring")

# Locate the AMD Ryzen AI SDK (qlinear_2 etc).
# When the SDK is not installed we fall back to a software emulation path so
# the backend still builds (useful for dev on non-Strix machines).
find_package(RyzenAI QUIET)

set(GGML_RYZENAI_SOURCES
    ggml-ryzenai.cpp
    ggml-ryzenai-impl.cpp
)

set(GGML_RYZENAI_HEADERS
    ggml-ryzenai-impl.h
)

ggml_add_backend_library(ggml-ryzenai
    ${GGML_RYZENAI_SOURCES}
    ${GGML_RYZENAI_HEADERS}
)

if (RyzenAI_FOUND)
    message(STATUS "RyzenAI backend: SDK found - using real NPU path")
    target_link_libraries(ggml-ryzenai PRIVATE ryzenai::qlinear_2)
else()
    message(WARNING
        "==================================================================\n"
        "  RyzenAI backend: SDK NOT FOUND - building in EMULATION mode.\n"
        "  Matmuls that pass can_mul_mat will run as a SCALAR SOFTWARE GEMM\n"
        "  (not the NPU). This is a development / correctness baseline only\n"
        "  and is intentionally slow. Install the AMD Ryzen AI SDK and\n"
        "  reconfigure to use the real NPU path.\n"
        "==================================================================")
    target_compile_definitions(ggml-ryzenai PRIVATE RYZENAI_EMULATION)
endif()
