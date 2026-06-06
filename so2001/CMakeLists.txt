#
# z-slmapp standalone build (MSVC toolchain, x64)
#
# Mirrors the Razzle build defined by sources / sources.inc / dirs files under
#   d:\xbc\src\gamecore\so2001\z-slmapp
#
# Targets included (active set from the original 'dirs'):
#
#   Libraries / DLLs:
#     za-common        STATIC LIB (with __TENSOR_REPACK__)
#     za-common-shared DLL    (exports defined in common\za-common.def)
#     zo-common        STATIC LIB
#     za-ggml-avx2     STATIC LIB (AVX2 codegen)
#     za-ggml-avx512   STATIC LIB (AVX512 codegen + __GEN_AVX512__)
#     za-ggml-avx2-dll DLL       (exports za-ggmlapi.def)
#     za-ggml-avx512-dll DLL     (exports za-ggmlapi.def)
#     zo-ggml-avx2     STATIC LIB
#     zo-ggml-avx512   STATIC LIB
#     zo-ggml-avx2-dll DLL
#     zo-ggml-avx512-dll DLL
#
#   Executables:
#     batchslmza, batchslmzo
#     minslmza, minslmzo
#     llbenchza, llbenchza-avx2, llbenchza-avx512
#     llbenchzo, llbenchzo-avx2
#     llbench-b-za, llbench-b-zo
#     llindex-za, llindex-zo
#     llspec-za, llspec-zo
#     perfavxza, perfavxzo
#     classify
#     clipindex, clipsrch
#     lltestza-avx2, lltestza-avx512
#
# Configure:
#   cmake -S . -B build -G "Visual Studio 17 2022" -A x64
# Build:
#   cmake --build build --config Release
#

cmake_minimum_required(VERSION 3.20)

project(z-slmapp
        VERSION 1.0.0
        LANGUAGES C CXX)

# ---- Sanity checks ----------------------------------------------------------

if (NOT MSVC)
    message(FATAL_ERROR "z-slmapp requires the MSVC toolchain (cl.exe).")
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR "z-slmapp is x64 only.")
endif()

# ---- Language standards -----------------------------------------------------

set(CMAKE_C_STANDARD          11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Razzle uses /O2 /Ob1; CMake's default Release adds /Ob2. Replace it so we
# don't emit warning D9025 ("overriding '/Ob2' with '/Ob1'") on every TU.
foreach (_lang C CXX)
    foreach (_cfg RELEASE RELWITHDEBINFO MINSIZEREL)
        if (DEFINED CMAKE_${_lang}_FLAGS_${_cfg})
            string(REGEX REPLACE "/Ob2( |$)" "/Ob1 "
                CMAKE_${_lang}_FLAGS_${_cfg}
                "${CMAKE_${_lang}_FLAGS_${_cfg}}")
        endif()
    endforeach()
endforeach()

# ---- Output layout ---------------------------------------------------------

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Honor the project's default Release optimization (Razzle uses /O2 /Ob1).
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Helper: common flags / definitions -------------------------------------
#
# In the Razzle build, every sources(.inc) file pins:
#   USE_MSVCRT=1, USE_UNICRT=1, USE_STL=1 (current STL), USE_NATIVE_EH=1,
#   NO_WCHAR_T=1, GUARD=0, INIT_ALL_OPTIN=0, /fp:fast, /d2jumptablerdata,
#   /O2 /Ob1 (default release optimization).
#
# Translated to MSVC switches:
#   /MD          : dynamic CRT (USE_MSVCRT)
#   /EHsc        : C++ native EH (USE_NATIVE_EH)
#   /Zc:wchar_t- : disable native wchar_t (NO_WCHAR_T)
#   /fp:fast
#   /d2jumptablerdata
#   /O2 /Ob1     : already the default Release flag for MSVC in CMake
#
# /guard:cf is OFF by default in MSVC unless explicitly enabled, matching GUARD=0.

function(zslm_apply_common_flags target)
    # MSVC dynamic CRT, multi-processor compile, native EH, no wchar_t.
    target_compile_options(${target} PRIVATE
        /MP
        /EHsc
        /Zc:wchar_t-
        /fp:fast
        /d2jumptablerdata)

    # Use the dynamically-linked CRT (USE_MSVCRT / USE_UNICRT).
    set_property(TARGET ${target} PROPERTY MSVC_RUNTIME_LIBRARY
                 "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    # Match Razzle: O2 + Ob1 in Release. The default Release flags are patched
    # at top-level to swap /Ob2 -> /Ob1; nothing else needed here.

    target_compile_definitions(${target} PRIVATE
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        _CRT_NONSTDC_NO_DEPRECATE
        NOMINMAX)
endfunction()

# ---- Helper: AVX2 / AVX512 flag application --------------------------------
#
# MSVC's /arch:AVX512 implies AVX2, AVX, etc. but does NOT auto-define
# __AVX512VNNI__ or __AVX512VL__ . Razzle za-ggml\sources.inc adds those
# manually; we mirror that for the AVX512 ggml variants.

function(zslm_apply_avx2 target)
    target_compile_options(${target} PRIVATE /arch:AVX2)
endfunction()

function(zslm_apply_avx512 target)
    target_compile_options(${target} PRIVATE /arch:AVX512)
    # MSVC does not predefine these even with /arch:AVX512.
    # Required by za-ggml's sources to enable VNNI/VL fast paths.
    target_compile_definitions(${target} PRIVATE
        __AVX512VNNI__
        __AVX512VL__)
endfunction()

# Enable OpenMP for the ggml libraries (Razzle uses /openmp -> vcomp.lib).
find_package(OpenMP REQUIRED COMPONENTS C CXX)

# ---- Subdirectories --------------------------------------------------------

add_subdirectory(za-ggml)
add_subdirectory(zo-ggml)
add_subdirectory(common)
add_subdirectory(batchslm)
add_subdirectory(minslm-za)
add_subdirectory(llbench)
add_subdirectory(llbench-b)
add_subdirectory(llindex)
add_subdirectory(llspec)
add_subdirectory(perfavx)
add_subdirectory(classify)
add_subdirectory(clip)
add_subdirectory(lltest)

message(STATUS "")
message(STATUS "z-slmapp build configured.")
message(STATUS "  Compiler         : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP found     : ${OpenMP_CXX_FOUND}")
message(STATUS "")
