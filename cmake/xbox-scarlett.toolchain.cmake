# cmake/xbox-scarlett.toolchain.cmake
#
# Cross-compile toolchain for the llama.cpp DX12 backend targeting
# Xbox Series X (Scarlett) under the Microsoft GDKX (April 2026 build).
#
# Recommended invocation:
#
#   cmake -B build.xbox -G "Visual Studio 17 2022" -A Gaming.Xbox.Scarlett.x64 ^
#         -DCMAKE_TOOLCHAIN_FILE=cmake/xbox-scarlett.toolchain.cmake ^
#         -DGGML_XBOX=ON ^
#         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_COMMON=OFF
#
# Then:
#
#   cmake --build build.xbox --config Release --target ggml-dx12
#
# The Gaming.Xbox.Scarlett.x64 platform is registered with MSBuild by the
# GDKX installer. If `cmake -A Gaming.Xbox.Scarlett.x64` fails, your
# Visual Studio install does not have the GDKX workload -- install it via
# the GDK setup before continuing.
#
# This toolchain intentionally does *not* set CMAKE_C_COMPILER /
# CMAKE_CXX_COMPILER. The Visual Studio generator + Gaming.Xbox.Scarlett.x64
# platform pick the correct cl.exe from the GDK on its own.

# Do NOT set CMAKE_SYSTEM_NAME=WindowsStore here. That is the UWP knob, and
# CMake will then stamp <ApplicationType>Windows Store</ApplicationType> into
# every generated .vcxproj -- including the MSBuild probe project (VCTargetsPath)
# and every try_compile. MSBuild rejects the combination
#   (ApplicationType=Windows Store, Platform=Gaming.Xbox.Scarlett.x64)
# because GDKX is its own ApplicationType, not UWP, and configure dies with:
#   "The BaseOutputPath/OutputPath property is not set for project
#    'VCTargetsPath.vcxproj' ... Platform='Gaming.Xbox.Scarlett.x64'"
# The -A Gaming.Xbox.Scarlett.x64 platform argument is sufficient on its own to
# get MSBuild to pick the GDKX toolset, includes, libs and SDK.
#
# We still set CMAKE_SYSTEM_NAME=Windows (matching the host) -- not for any
# semantic effect, but because CMake's CMakeDetermineSystem.cmake warns:
#   "CMAKE_CROSSCOMPILING has been set ... CMake is resetting it to false
#    because CMAKE_SYSTEM_NAME was not set. To indicate cross compilation,
#    only CMAKE_SYSTEM_NAME needs to be set."
# any time a toolchain file is loaded without setting CMAKE_SYSTEM_NAME.
# Since host == target from CMake's perspective (both are "Windows"), this
# leaves CMAKE_CROSSCOMPILING=FALSE, which is fine: we never use try_run, and
# everything else flows through MSBuild + the GDKX platform.
set(CMAKE_SYSTEM_NAME       Windows        CACHE STRING "" FORCE)
set(CMAKE_SYSTEM_PROCESSOR  AMD64          CACHE STRING "" FORCE)

# GDKX defines required by the system headers (d3d12_xs.h, xgameruntime.h, etc.)
add_compile_definitions(
    _GAMING_XBOX
    _GAMING_XBOX_SCARLETT
    WINAPI_FAMILY=WINAPI_FAMILY_GAMES
    _USE_DECLSPECS_FOR_SAL=1
    __WRL_NO_DEFAULT_LIB__
)

# GDK headers expect /MD on Release / /MDd on Debug. The Gaming.Xbox.Scarlett.x64
# MSBuild platform usually sets this, but lock it in for safety when generators
# behave inconsistently.
if (POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)

# GDKX rejects /DYNAMICBASE:NO and /NXCOMPAT:NO via Platform.Edition.targets'
# _LinkerValidation target ("Platform Gaming.Xbox.Scarlett.x64 requires
# Address Space Layout Randomization (ASLR)." / "... requires Data Execution
# Prevention (DEP)."). MSBuild's default Debug config emits
#   <RandomizedBaseAddress>false</RandomizedBaseAddress>   -> /DYNAMICBASE:NO
#   <DataExecutionPrevention>false</DataExecutionPrevention> -> /NXCOMPAT:NO
# Append the positive forms for every config so link.exe (which honors the
# last token wins for these switches) always ends up with ASLR + DEP on. This
# must also be in *_INIT so it propagates into try_compile probes during
# configure, otherwise the compiler-detection link step fails before the real
# project is even generated.
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    string(APPEND CMAKE_EXE_LINKER_FLAGS_${_cfg}_INIT    " /DYNAMICBASE /NXCOMPAT")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS_${_cfg}_INIT " /DYNAMICBASE /NXCOMPAT")
    string(APPEND CMAKE_MODULE_LINKER_FLAGS_${_cfg}_INIT " /DYNAMICBASE /NXCOMPAT")
endforeach()

# Xbox Series X|S CPU is AMD "Scarlett" -- a custom 8-core / 16-thread Zen 2.
# Zen 2 features: SSE4.2, AVX, AVX2, FMA3, F16C, BMI1/2, CLMUL, AES, SHA, RDRAND.
# No AVX-512.
#
# Default ggml-cpu's x86 feature switches to that target so the Zen 2 quant /
# repack kernels are picked instead of the GENERIC fallback. These are CACHE
# entries (no FORCE), so a user can still override with -DGGML_AVX2=OFF etc.
#
# Note: the GDKX MSBuild platform already passes /arch:AVX2 /favor:AMD64 to
# cl.exe automatically, but ggml-cpu/CMakeLists.txt only emits the
# GGML_AVX2 / GGML_FMA / GGML_F16C / GGML_BMI2 *defines* when these CMake
# options are ON, so the kernels gate on them at compile time.
set(GGML_AVX2  ON CACHE BOOL "Xbox Scarlett (Zen 2) supports AVX2")
set(GGML_FMA   ON CACHE BOOL "Xbox Scarlett (Zen 2) supports FMA3")
set(GGML_F16C  ON CACHE BOOL "Xbox Scarlett (Zen 2) supports F16C")
set(GGML_BMI2  ON CACHE BOOL "Xbox Scarlett (Zen 2) supports BMI2")
# Explicitly off -- Zen 2 has no AVX-512.
set(GGML_AVX512 OFF CACHE BOOL "Xbox Scarlett (Zen 2) has no AVX-512" FORCE)

# GDKX platform defaults to /fp:fast, which reorders FP reductions in the
# quantized matmul inner loops and produces different rounding vs the PC
# build (/fp:precise). Force /fp:precise for bitwise-reproducible inference.
string(APPEND CMAKE_C_FLAGS_INIT   " /fp:precise")
string(APPEND CMAKE_CXX_FLAGS_INIT " /fp:precise")
# GGML_NATIVE would try to detect host CPU features; we're cross-compiling
# for a known fixed target, so always disable it.
set(GGML_NATIVE OFF CACHE BOOL "Cross-compiling for Xbox Scarlett (no native detect)" FORCE)

# The April 2026 GDKX root, used by ggml/src/ggml-dx12/CMakeLists.txt to
# find the Scarlett HLSL compiler (XboxScarlettShaderCompiler.exe / dxc.exe).
# Override with -DGGML_XBOX_GDK_EDITION=251000 (or similar) to pin a version.
if (DEFINED ENV{GameDK})
    set(GGML_XBOX_GDK_ROOT "$ENV{GameDK}" CACHE PATH "GDKX install root" FORCE)
else()
    message(WARNING
        "GameDK environment variable not set -- the GDKX shader compiler will "
        "not be auto-discovered. Run from a GDKX-enabled command prompt, "
        "or pass -DGGML_XBOX_SHADER_COMPILER=<path> when configuring.")
endif()
