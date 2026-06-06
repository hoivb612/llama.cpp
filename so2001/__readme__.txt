 - Top-level: CMakeLists.txt, CMakePresets.json, README.md, plus the original __readme__.txt, config.inc,custom_prompts.txt
 - 13 component dirs, each with its own CMakeLists.txt and the copied .c/.cpp/.h/.def files

Build produced 22 EXEs, 5 DLLs, 6 static libs — exactly the active target set from the Razzle dirs file.

Razzle → MSVC flag mapping (all in helpers zslm_apply_common_flags / _avx2 / _avx512):

┌───────────────────────────┬──────────────────────────────────────────────────────────────┐
│ Razzle                              │ MSVC                                                                                │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ USE_MSVCRT/USE_UNICRT               │ /MD                                                                                  │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ USE_NATIVE_EH                       │ /EHsc                                                                                │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ NO_WCHAR_T                          │ /Zc:wchar_t-                                                                         │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ /fp:fast                            │ same                                                                                 │
│ /d2jumptablerdata                   │                                                                                      │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ MSC_OPTIMIZATION=/O2 /Ob1           │ patched CMAKE_*_FLAGS_RELEASE to swap /Ob2→/Ob1                                      │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ /openmp + vcomp.lib                 │ OpenMP::OpenMP_C/CXX                                                                 │
├───────────────────────────┼──────────────────────────────────────────────────────────────┤
│ /arch:AVX2                          │ zslm_apply_avx2()                                                                    │
├───────────────────────────┼───────────────────────── ─────────────────────────────────────┤
│ /arch:AVX512                        │ zslm_apply_avx512() (also defines __AVX512VNNI__, __AVX512VL__ since MSVC doesn't    │
│                                     │ predefine them)                                                                      │
├───────────────────────────┼───────────────────────────────────────────────────────────────┤
│ __GEN_AVX512__                      │ per AVX512 ggml lib + perfavx                                                        │
└───────────────────────────┴────────────────────────── ────────────────────────────────────┘

Dependencies outside z-slmapp: none. Source files reference only standard headers, Windows SDK (<windows.h>,<powrprof.h>, <immintrin.h>), and local sibling files. Libs referenced from Razzle (kernel32.lib, powrprof.lib,vcomp.lib) all come from the Windows SDK / MSVC toolchain.

To build:

 cd d:\llama.cpp\so2001\z-slmapp
 cmake --preset msvc-x64-ninja        # or msvc-x64 for VS solution
 cmake --build build\ninja-release

Build was verified end-to-end against MSVC 19.50 (VS 2022 v18 / 14.50.35717) — all 118 compile + link steps succeededwith zero CMake/linker errors.

Summary — three ways to get RelWithDebInfo:

┌──────────────────┬─────────────────────────────────────────────────────────────────────┐
│ Method                  │ Command                                                                                       │
├──────────────────┼─────────────────────────────────────────────────────────────────────┤
│ Ninja preset            │ cmake --preset msvc-x64-ninja-relwithdebinfo then cmake --build build\ninja-relwithdebinfo    │
│ (new)                   │                                                                                               │
├──────────────────┼─────────────────────────────────────────────────────────────────────┤
│ VS solution             │ cmake --preset msvc-x64 then cmake --build build\msvc-x64 --config RelWithDebInfo             │
│ preset                  │                                                                                               │
├──────────────────┼─────────────────────────────────────────────────────────────────────┤
│ Ad-hoc (no              │ cmake -S . -B build\rwdi -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo then cmake --build        │
│ preset)                 │ build\rwdi                                                                                    │
└──────────────────┴─────────────────────────────────────────────────────────────────────┘

Output lands in build\ninja-relwithdebinfo\bin\ with matching .pdb files alongside each .exe/.dll. Flags become /O2 /Ob1 /Zi /DEBUG (Razzle-equivalent optimization plus full debug info).

