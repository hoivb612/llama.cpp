# z-slmapp (CMake / MSVC build)

Standalone MSVC + CMake build of the GameCore `z-slmapp` tree, derived from
`d:\xbc\src\gamecore\so2001\z-slmapp` (Razzle / NMake `sources` files).

The active set of targets from the original `dirs` file is built. No source
files have any dependency above `z-slmapp` other than the Windows SDK
(`kernel32.lib`, `powrprof.lib`) and the MSVC OpenMP runtime (`vcomp*.dll`,
delivered by `/openmp`).

## Requirements

- Visual Studio 2022 (17.6+) — MSVC v143 toolset
- Windows 10/11 SDK
- CMake 3.21+
- A host CPU with at least AVX2 (AVX512 needed for the `-avx512` variants)

## Configure & build

From a "x64 Native Tools Command Prompt for VS 2022" (or any shell with
`cl.exe` and `cmake` on PATH):

```
cd d:\llama.cpp\so2001\z-slmapp
cmake --preset msvc-x64
cmake --build build\msvc-x64 --config Release -j
```

Or with Ninja:

```
cmake --preset msvc-x64-ninja
cmake --build build\ninja-release -j
```

All executables, DLLs, and import libs end up under `build\<preset>\bin\Release`
(VS) or `build\<preset>\bin` (Ninja). Static libs land in `build\<preset>\lib`.

## Targets

| Target                | Type | Notes                                                  |
| --------------------- | ---- | ------------------------------------------------------ |
| `za-ggml-avx2`        | LIB  | Zen ggml, AVX2 codegen                                 |
| `za-ggml-avx512`      | LIB  | Zen ggml, AVX512 codegen, `__GEN_AVX512__`             |
| `za-ggml-avx2-dll`    | DLL  | `za-ggml-avx2.dll` exporting `za-ggmlapi.def`          |
| `za-ggml-avx512-dll`  | DLL  | `za-ggml-avx512.dll`                                   |
| `zo-ggml-avx2`        | LIB  | Original ggml, AVX2                                    |
| `zo-ggml-avx512`      | LIB  | Original ggml, AVX512                                  |
| `zo-ggml-avx2-dll`    | DLL  | `zo-ggml-avx2.dll`                                     |
| `zo-ggml-avx512-dll`  | DLL  | `zo-ggml-avx512.dll`                                   |
| `za-common`           | LIB  | llama+sampling+unicode (za side, static)               |
| `za-common-shared`    | DLL  | `za-common.dll` exporting `za-common.def`              |
| `zo-common`           | LIB  | llama+sampling+unicode (zo side, static)               |
| `batchslmza` / `batchslmzo`        | EXE | batch driver                              |
| `minslmza`   / `minslmzo`          | EXE | minimal SLM driver                        |
| `llbenchza`, `llbenchza-avx2`, `llbenchza-avx512` | EXE | benchmarks    |
| `llbenchzo`, `llbenchzo-avx2`      | EXE |                                           |
| `llbench-b-za` / `llbench-b-zo`    | EXE | batch benchmark                           |
| `llindex-za`  / `llindex-zo`       | EXE | index runner                              |
| `llspec-za`   / `llspec-zo`        | EXE | speculative decoder                       |
| `perfavxza`   / `perfavxzo`        | EXE | AVX microbench (loads ggml DLLs at runtime) |
| `classify`                         | EXE | small classifier helper                   |
| `clipindex`   / `clipsrch`         | EXE | CLIP image embedding tools                |
| `lltestza-avx2` / `lltestza-avx512`| EXE | low-level tests                           |

## Mapping Razzle flags to MSVC

The Razzle `sources(.inc)` files use macros that translate to MSVC flags as
follows (all driven from helpers in the root `CMakeLists.txt`):

| Razzle directive            | MSVC equivalent                                |
| --------------------------- | ---------------------------------------------- |
| `USE_MSVCRT=1 / USE_UNICRT=1` | `/MD`  (`MSVC_RUNTIME_LIBRARY = MultiThreadedDLL`) |
| `USE_NATIVE_EH=1`           | `/EHsc`                                        |
| `USE_STL=1, STL_VER_CURRENT`| `CMAKE_CXX_STANDARD = 17`                      |
| `NO_WCHAR_T=1`              | `/Zc:wchar_t-`                                 |
| `GUARD=0`                   | (no `/guard:cf` flag — default off in MSVC)    |
| `/fp:fast`                  | `/fp:fast`                                     |
| `/d2jumptablerdata`         | `/d2jumptablerdata`                            |
| `MSC_OPTIMIZATION=/O2 /Ob1` | Release default `/O2` + explicit `/Ob1`        |
| `/arch:AVX2`                | `zslm_apply_avx2(target)` -> `/arch:AVX2`      |
| `/arch:AVX512`              | `zslm_apply_avx512(target)` -> `/arch:AVX512` + `-D__AVX512VNNI__ -D__AVX512VL__` |
| `/openmp` + `vcomp.lib`     | `find_package(OpenMP)` + `OpenMP::OpenMP_C/CXX` |
| `$(C_DEFINES) -D__GEN_AVX512__` | Set per AVX512 target via `target_compile_definitions` |

`__AVX512VNNI__` and `__AVX512VL__` are forced because MSVC does not
predefine them with `/arch:AVX512`; the original `za-ggml\sources.inc` adds
them so the VNNI fast paths compile.

## Running

Two GGUF models live on `\\hydrango\public\reuse\GameCore\so2001\slmapp\`:

- `Starling-lm-7b-alpha.Q2_K.gguf` — small, fast
- `Starling-lm-7b-alpha.Q8_0.gguf` — larger, higher fidelity

Short example (needs `cpf.txt`):

```
batchslmza.exe -m Starling-lm-7b-alpha.Q2_K.gguf --color -c 2048 -t 16 -i --temp 0.1 -s 42 -cpf cpf.txt --top-p 0.95
```

Long internal test (uses `custom_prompts.txt`):

```
batchslmza.exe -m Starling-lm-7b-alpha.Q2_K.gguf --color -c 2048 -t 16 -i --temp 0.1 -s 42 -cpf --top-p 0.95
```
