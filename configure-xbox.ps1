Remove-Item -Recurse -Force build.xbox -ErrorAction SilentlyContinue

& cmake -B build.xbox `
        -G "Visual Studio 17 2022" `
        -A Gaming.Xbox.Scarlett.x64 `
        -DCMAKE_TOOLCHAIN_FILE="cmake/xbox-scarlett.toolchain.cmake" `
        -DGGML_XBOX=ON `
        -DLLAMA_BUILD_TESTS=OFF `
        -DLLAMA_BUILD_TOOLS=OFF `
        -DLLAMA_BUILD_COMMON=OFF `
        -DLLAMA_BUILD_EXAMPLES=ON
