# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()
if (GGML_B612)
    add_compile_definitions(GGML_XBOX_PERF)
    add_compile_definitions(GGML_B612)
endif()

# examples

if (GGML_B612)
    add_subdirectory(retrieval)
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(xbapp)
    add_subdirectory(chatbot)
    add_subdirectory(llm-infer)
    # add_subdirectory(phi4mm)
    # add_subdirectory(clip)
else()
    add_subdirectory(batched)
    add_subdirectory(embedding)
    add_subdirectory(eval-callback)

    add_subdirectory(gguf-hash)
    add_subdirectory(gguf)
    add_subdirectory(gritlm)
    add_subdirectory(lookahead)
    add_subdirectory(lookup)
    add_subdirectory(parallel)
    add_subdirectory(passkey)
    add_subdirectory(retrieval)
    add_subdirectory(save-load-state)
    add_subdirectory(simple)
    add_subdirectory(simple-chat)
    add_subdirectory(speculative)
    add_subdirectory(speculative-simple)
    add_subdirectory(gen-docs)
    add_subdirectory(training)
    add_subdirectory(diffusion)
    if (NOT GGML_BACKEND_DL)
        add_subdirectory(convert-llama2c-to-ggml)
        if (NOT WIN32)
            # disabled on Windows because it uses internal functions not exported with LLAMA_API
            add_subdirectory(quantize-stats)
        endif()
        add_subdirectory(phi4mm)
        if (GGML_RPC)
            add_subdirectory(rpc)
        endif()
        # these examples use the backends directly and cannot be built with dynamic loading
        if (GGML_SYCL)
            add_subdirectory(sycl)
        endif()
    endif()
endif()
