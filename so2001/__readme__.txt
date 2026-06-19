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

========================================================

C:\llama.cpp\b612.dc_061426\so2001>"c:\Program Files\Microsoft Visual Studio\2022\Community\vc\Auxiliary\Build\vcvars64.bat"
**********************************************************************
** Visual Studio 2022 Developer Command Prompt v17.14.8
** Copyright (c) 2025 Microsoft Corporation
**********************************************************************
[vcvarsall.bat] Environment initialized for: 'x64'

C:\llama.cpp\b612.dc_061426\so2001>md build

C:\llama.cpp\b612.dc_061426\so2001>cd build

C:\llama.cpp\b612.dc_061426\so2001\build>cmake ..
-- Building for: Visual Studio 17 2022
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.26200.
-- The C compiler identification is MSVC 19.44.35211.0
-- The CXX compiler identification is MSVC 19.44.35211.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenMP_C: -openmp (found version "2.0")
-- Found OpenMP_CXX: -openmp (found version "2.0")
-- Found OpenMP: TRUE (found version "2.0") found components: C CXX
--
-- z-slmapp build configured.
--   Compiler         : MSVC 19.44.35211.0
--   CMAKE_BUILD_TYPE :
--   OpenMP found     : TRUE
--
-- Configuring done (5.0s)
-- Generating done (0.2s)
-- Build files have been written to: C:/llama.cpp/b612.dc_061426/so2001/build

C:\llama.cpp\b612.dc_061426\so2001\build>cmake --build . --config RelWithDebInfo
MSBuild version 17.14.14+a129329f1 for .NET Framework

  1>Checking Build System
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/za-ggml/CMakeLists.txt
  ggml-alloc.c
  ggml-backend.c
  ggml-repack.c
  ggml-quants.c
  ggml-q4.c
  ggml.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\za-ggml\za-ggml-avx512.vcxproj]
  (compiling source file '../../za-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\za-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  za-ggml-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-ggml-avx512.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/common/CMakeLists.txt
  common.cpp
  grammar-parser.cpp
  json-schema-to-grammar.cpp
  llama.cpp
  sampling.cpp
  unicode.cpp
  unicode-data.cpp
     Creating library C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-common.lib and object C
  :/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-common.exp
  za-common-shared.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-common.dll
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/batchslm/CMakeLists.txt
  batchslm.cpp
  batchslmza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\batchslmza.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/common/CMakeLists.txt
  common.cpp
  grammar-parser.cpp
  json-schema-to-grammar.cpp
  llama.cpp
  sampling.cpp
  unicode.cpp
  unicode-data.cpp
  zo-common.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-common.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/zo-ggml/CMakeLists.txt
  ggml-alloc.c
  ggml-backend.c
  ggml-quants.c
  ggml.c
  ggml-q4.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx512.vcxproj]
  (compiling source file '../../zo-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx512.vcxproj]
  (compiling source file '../../zo-ggml/ggml-quants.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  zo-ggml-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-ggml-avx512.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/batchslm/CMakeLists.txt
  batchslm.cpp
  batchslmzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\batchslmzo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/classify/CMakeLists.txt
  classify.c
  classify.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\classify.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/common/CMakeLists.txt
  common.cpp
  grammar-parser.cpp
  json-schema-to-grammar.cpp
  llama.cpp
  sampling.cpp
  unicode.cpp
  unicode-data.cpp
  za-common.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-common.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/clip/CMakeLists.txt
  clip.cpp
  common-clip.cpp
  clipIndex.cpp
  clipindex.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\clipindex.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/clip/CMakeLists.txt
  clip.cpp
  common-clip.cpp
  clipSrch.cpp
  clipsrch.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\clipsrch.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench-b/CMakeLists.txt
  llbench-b.cpp
  llbench-b-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbench-b-za.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench-b/CMakeLists.txt
  llbench-b.cpp
  llbench-b-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbench-b-zo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench/CMakeLists.txt
  llbench.cpp
  llbenchza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/za-ggml/CMakeLists.txt
  ggml-alloc.c
  ggml-backend.c
  ggml-repack.c
  ggml-quants.c
  ggml-q4.c
  ggml.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\za-ggml\za-ggml-avx2.vcxproj]
  (compiling source file '../../za-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\za-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  za-ggml-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-ggml-avx2.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench/CMakeLists.txt
  llbench.cpp
  llbenchza-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza-avx2.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench/CMakeLists.txt
  llbench.cpp
  llbenchza-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza-avx512.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench/CMakeLists.txt
  llbench.cpp
  llbenchzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchzo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/zo-ggml/CMakeLists.txt
  ggml-alloc.c
  ggml-backend.c
  ggml-quants.c
  ggml.c
  ggml-q4.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx2.vcxproj]
  (compiling source file '../../zo-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx2.vcxproj]
  (compiling source file '../../zo-ggml/ggml-quants.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  zo-ggml-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-ggml-avx2.lib
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llbench/CMakeLists.txt
  llbench.cpp
  llbenchzo-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchzo-avx2.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llindex/CMakeLists.txt
  llindex.cpp
  llindex-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llindex-za.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llindex/CMakeLists.txt
  llindex.cpp
  llindex-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llindex-zo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llspec/CMakeLists.txt
  llspec.cpp
  llspec-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llspec-za.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/llspec/CMakeLists.txt
  llspec.cpp
  llspec-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llspec-zo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/lltest/CMakeLists.txt
  lltest.c
C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20): warning C4477: 'printf' : format string '%zd' requires an
argument of type 'unsigned __int64', but variadic argument 2 has type 'uint32_t' [C:\llama.cpp\b612.dc_061426\so2001\bu
ild\lltest\lltestza-avx2.vcxproj]
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20):
      consider using '%d' in the format string
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20):
      consider using '%I32d' in the format string

C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20): warning C4477: 'printf' : format string '%zd' requires an
argument of type 'unsigned __int64', but variadic argument 2 has type 'uint32_t' [C:\llama.cpp\b612.dc_061426\so2001\bu
ild\lltest\lltestza-avx2.vcxproj]
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20):
      consider using '%d' in the format string
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20):
      consider using '%I32d' in the format string

  lltestza-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\lltestza-avx2.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/lltest/CMakeLists.txt
  lltest.c
C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20): warning C4477: 'printf' : format string '%zd' requires an
argument of type 'unsigned __int64', but variadic argument 2 has type 'uint32_t' [C:\llama.cpp\b612.dc_061426\so2001\bu
ild\lltest\lltestza-avx512.vcxproj]
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20):
      consider using '%d' in the format string
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3013,20):
      consider using '%I32d' in the format string

C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20): warning C4477: 'printf' : format string '%zd' requires an
argument of type 'unsigned __int64', but variadic argument 2 has type 'uint32_t' [C:\llama.cpp\b612.dc_061426\so2001\bu
ild\lltest\lltestza-avx512.vcxproj]
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20):
      consider using '%d' in the format string
      C:\llama.cpp\b612.dc_061426\so2001\lltest\lltest.c(3023,20):
      consider using '%I32d' in the format string

  lltestza-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\lltestza-avx512.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/minslm-za/CMakeLists.txt
  minslm.cpp
  slminfer.cpp
  minslmza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\minslmza.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/minslm-za/CMakeLists.txt
  minslm.cpp
  slminfer.cpp
  minslmzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\minslmzo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/perfavx/CMakeLists.txt
  perfavx.c
  perfavxza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\perfavxza.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/perfavx/CMakeLists.txt
  perfavx.c
  perfavxzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\perfavxzo.exe
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/za-ggml/CMakeLists.txt
  dllmain.cpp
     Creating library C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-ggml-avx2.lib and objec
  t C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-ggml-avx2.exp
  za-ggml-avx2-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-ggml-avx2.dll
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/za-ggml/CMakeLists.txt
  dllmain.cpp
     Creating library C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-ggml-avx512.lib and obj
  ect C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/za-ggml-avx512.exp
za-ggml-avx512.exp : warning LNK4070: /OUT:za-ggml-avx2.dll directive in .EXP differs from output filename 'C:\llama.cp
p\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-ggml-avx512.dll'; ignoring directive [C:\llama.cpp\b612.dc_061426\s
o2001\build\za-ggml\za-ggml-avx512-dll.vcxproj]
  za-ggml-avx512-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-ggml-avx512.dll
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/zo-ggml/CMakeLists.txt
  dllmain.cpp
  ggml-alloc.c
  ggml-backend.c
  ggml-quants.c
  ggml.c
  ggml-q4.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx2-dll.vcxproj]
  (compiling source file '../../zo-ggml/ggml-quants.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx2-dll.vcxproj]
  (compiling source file '../../zo-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/zo-ggml/CMakeLists.txt
  dllmain.cpp
  ggml-alloc.c
  ggml-backend.c
  ggml-quants.c
  ggml.c
  ggml-q4.c
C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx512-dll.vcxproj]
  (compiling source file '../../zo-ggml/ggml-q4.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\ucrt\assert.h(21,9): warning C4005: 'static_assert': macro
redefinition [C:\llama.cpp\b612.dc_061426\so2001\build\zo-ggml\zo-ggml-avx512-dll.vcxproj]
  (compiling source file '../../zo-ggml/ggml-quants.c')
      C:\llama.cpp\b612.dc_061426\so2001\zo-ggml\ggml-common.h(64,9):
      see previous definition of 'static_assert'

  za-ggml-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-ggml-avx512.lib
  za-common-shared.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-common.dll
  batchslmza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\batchslmza.exe
  zo-common.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-common.lib
  ggml.c
  zo-ggml-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-ggml-avx512.lib
  batchslmzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\batchslmzo.exe
  classify.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\classify.exe
  za-common.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-common.lib
  clipindex.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\clipindex.exe
  clipsrch.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\clipsrch.exe
  llbench-b-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbench-b-za.exe
  llbench-b-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbench-b-zo.exe
  llbenchza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza.exe
  za-ggml-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\za-ggml-avx2.lib
  llbenchza-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza-avx2.exe
  llbenchza-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchza-avx512.exe
  llbenchzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchzo.exe
  ggml.c
  zo-ggml-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\lib\RelWithDebInfo\zo-ggml-avx2.lib
  llbenchzo-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llbenchzo-avx2.exe
  llindex-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llindex-za.exe
  llindex-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llindex-zo.exe
  llspec-za.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llspec-za.exe
  llspec-zo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\llspec-zo.exe
  lltestza-avx2.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\lltestza-avx2.exe
  lltestza-avx512.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\lltestza-avx512.exe
  minslmza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\minslmza.exe
  minslmzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\minslmzo.exe
  perfavxza.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\perfavxza.exe
  perfavxzo.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\perfavxzo.exe
  za-ggml-avx2-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-ggml-avx2.dll
  za-ggml-avx512-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\za-ggml-avx512.dll
  ggml.c
  LINK : C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\zo-ggml-avx2.dll not found or not built by the las
  t incremental link; performing full link
     Creating library C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/zo-ggml-avx2.lib and objec
  t C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/zo-ggml-avx2.exp
  zo-ggml-avx2-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\zo-ggml-avx2.dll
  ggml.c
  LINK : C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\zo-ggml-avx512.dll not found or not built by the l
  ast incremental link; performing full link
     Creating library C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/zo-ggml-avx512.lib and obj
  ect C:/llama.cpp/b612.dc_061426/so2001/build/lib/dll-import/RelWithDebInfo/zo-ggml-avx512.exp
zo-ggml-avx512.exp : warning LNK4070: /OUT:zo-ggml-avx2.dll directive in .EXP differs from output filename 'C:\llama.cp
p\b612.dc_061426\so2001\build\bin\RelWithDebInfo\zo-ggml-avx512.dll'; ignoring directive [C:\llama.cpp\b612.dc_061426\s
o2001\build\zo-ggml\zo-ggml-avx512-dll.vcxproj]
  zo-ggml-avx512-dll.vcxproj -> C:\llama.cpp\b612.dc_061426\so2001\build\bin\RelWithDebInfo\zo-ggml-avx512.dll
  Building Custom Rule C:/llama.cpp/b612.dc_061426/so2001/CMakeLists.txt

C:\llama.cpp\b612.dc_061426\so2001\build
