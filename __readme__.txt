Offspring for hoivb612
https://github.com/hoivb612/llama.cpp

===========================================

Just make sure you never edit master. Keep it in the state of the original upstream fork (github.com/ggerganov/llama.cpp.git). Then, you can always:

1.	“Sync fork” to merge the master branch of the original upstream fork (github.com/ggerganov/llama.cpp.git) into the master branch of your downstream GitHub fork (github.com/HoiV/llama.cpp.git).
2.	“git fetch” to merge the master branch of your downstream GitHub fork (github.com/HoiV/llama.cpp.git) to the master branch of your local clone.
3.	While on your topic branch, “git merge origin/master” to merge the master branch of your local clone to your topic branch.
4.	#3 might have conflicts that you need to resolve.

===========================================

MSYS2 ucrt support: 
pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain

===========================================

For ARM64: 
cmake --preset arm64-windows-llvm-release -D GGML_LLAMAFILE=OFF -D GGML_OPENMP=OFF -B build.arm
cd build.arm
cmake --build . --config RelWithDebInfo --target llama-bench xbapp

bin\llama-bench.exe -m c:\llama.cpp\models\Llama-3.2-3B-Instruct-Q4_0_4_8.gguf -t 8 -p 128 -n 64
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 3B Q4_0_4_8              |   2.08 GiB |     3.61 B | CPU        |       8 |         pp128 |        306.69 ± 9.23 |
| llama 3B Q4_0_4_8              |   2.08 GiB |     3.61 B | CPU        |       8 |          tg64 |         45.39 ± 0.74 |

bin\llama-bench.exe -m c:\llama.cpp\models\Llama-3.2-3B-Instruct-Q2_K-Second.gguf -t 8 -p 128 -n 64
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| llama 3B Q2_K - Medium         |   1.56 GiB |     3.61 B | CPU        |       8 |         pp128 |         71.69 ± 0.43 |
| llama 3B Q2_K - Medium         |   1.56 GiB |     3.61 B | CPU        |       8 |          tg64 |         46.66 ± 0.47 |

bin\llama-bench.exe -m c:\llama.cpp\models\Phi-3.5-mini-instruct-Q4_0_4_8.gguf -t 8 -p 128 -n 64
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| phi3 3B Q4_0_4_8               |   2.03 GiB |     3.82 B | CPU        |       8 |         pp128 |        233.87 ± 6.45 |
| phi3 3B Q4_0_4_8               |   2.03 GiB |     3.82 B | CPU        |       8 |          tg64 |         40.70 ± 0.47 |

bin\llama-bench.exe -m c:\llama.cpp\models\Phi-3.5-mini-instruct-Q2_K.gguf -t 8 -p 128 -n 64
| model                          |       size |     params | backend    | threads |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | ------------: | -------------------: |
| phi3 3B Q2_K - Medium          |   1.32 GiB |     3.82 B | CPU        |       8 |         pp128 |         50.47 ± 5.81 |
| phi3 3B Q2_K - Medium          |   1.32 GiB |     3.82 B | CPU        |       8 |          tg64 |         34.63 ± 0.20 |

===============================================

D:\llama.cpp\b612.dc_052026\build>cmake --build . --config RelWithDebInfo --target minslminfer
CMake is re-running because D:/llama.cpp/b612.dc_052026/build/ggml/src/CMakeFiles/generate.stamp dependency file is missing.
-- Selecting Windows SDK version 10.0.26100.0 to target Windows 10.0.26200.
CMAKE_BUILD_TYPE=
-- Warning: ccache not found - consider installing it for faster compilation or disable this warning with GGML_CCACHE=OFF
-- CMAKE_SYSTEM_PROCESSOR: AMD64
-- CMAKE_GENERATOR_PLATFORM:
-- GGML_SYSTEM_ARCH: x86
-- Including CPU backend
-- set Compiler def for 'GGML_USE_CPU'
-- x86 detected
-- GGML_B612: Skipping x86 quants.c - already in b612
-- MSVC - enable AVX512
-- MSVC - enable /fp:fast
-- Add flag to generate B612 intrinsics: __GEN_AVX512__ / GGML_B612
-- Add flag to collect Xbox perf data GGML_XBOX_PERF
-- Adding CPU backend variant ggml-cpu: /arch:AVX512;/fp:fast;/d2jumptablerdata;/Zc:forScope;/O2;/Ob1 GGML_AVX512
-- ggml version: 0.12.0
-- ggml commit:  ff93581cf
-- Could NOT find OpenSSL, try to set the path to OpenSSL root folder in the system variable OPENSSL_ROOT_DIR (missing: OPENSSL_CRYPTO_LIBRARY OPENSSL_INCLUDE_DIR)
CMake Warning at vendor/cpp-httplib/CMakeLists.txt:152 (message):
  OpenSSL not found, HTTPS support disabled


-- Generating embedded license file for target: llama-common
-- Configuring done (5.0s)
-- Generating done (1.3s)
-- Build files have been written to: D:/llama.cpp/b612.dc_052026/build
MSBuild version 18.7.8+1ac568fee for .NET Framework

  Building Custom Rule D:/llama.cpp/b612.dc_052026/ggml/src/CMakeLists.txt
  ggml-opt.cpp
  ggml-quants.c
  ggml-threading.cpp
  ggml.c
  ggml-backend.cpp
  gguf.cpp
  ggml.cpp
  ggml-backend-meta.cpp
  ggml-alloc.c
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
     Creating library D:/llama.cpp/b612.dc_052026/build/ggml/src/RelWithDebInfo/ggml-base.lib and object D:/llama.cpp/b612.dc_052026/b
  uild/ggml/src/RelWithDebInfo/ggml-base.exp
  Generating code
  Previous IPDB not found, fall back to full compilation.
  All 3906 functions were compiled because no usable IPDB/IOBJ from previous compilation was found.
  Finished generating code
  ggml-base.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\ggml-base.dll
  Building Custom Rule D:/llama.cpp/b612.dc_052026/ggml/src/CMakeLists.txt
  quants-b612.c
  hbm.cpp
  amx.cpp
  mmq.cpp
  ggml-cpu.cpp
  repack.cpp
  traits.cpp
  ops-b612.cpp
  GGML-IMPL.h: using AVX512 F16C intrinsics
  ggml-cpu-b612.c
  vec-b612.cpp
  GGML-IMPL.h: using AVX512 F16C intrinsics
  repack.cpp
  ggml-cpu-repack.c
  GGML-IMPL.h: using AVX512 F16C intrinsics
  Building AVX512F vec_dot_q4_0_q8_0 version
  Building AVX512F vec_dot_q8_0_q8_0 version
  Building AVX512F vec_dot_q2_K_q8_K version
  Building AVX512F vec_dot_q3_K_q8_K version
  Building AVX512F vec_dot_q4_K_q8_K version
  Building AVX512F vec_dot_q6_K_q8_K
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  GGML-IMPL.h: using AVX512 F16C intrinsics
  Building AVX512F ggml_bf16_to_fp32_row_cpu
  Building AVX512F ggml_fp32_to_bf16_row_cpu
  Building AVX512F ggml_fp16_to_fp32_row_cpu
  Building AVX512F ggml_fp32_to_fp16_row_cpu
  Building AVX512F dequantize_row_q2_K_cpu
  Building AVX512F dequantize_row_q3_K_cpu
  Building AVX512F dequantize_row_q4_0_cpu
  Building AVX2/AVX512F dequantize_row_q4_K_cpu
  Building AVX512F dequantize_row_q6_K_cpu
  Building AVX512F dequantize_row_q8_0_cpu
  Building AVX512F dequantize_row_q8_K_cpu
  Building AVX512F ggml_vec_dot_f32 version
  Building AVX512F ggml_vec_dot_f16 version
  Building AVX512F ggml_vec_dot_bf16_f32 version
  Building AVX512F ggml_vec_dot_f16_f32 version
  Building AVX512F ggml_vec_silu_f32
  Building AVX512F ggml_vec_soft_max_f32
  Building AVX512F ggml_vec_sum_f32 version
  Building AVX512F ggml_vec_sumsq_f32 version
  Building AVX512F ggml_vec_add_f32 version
  Building AVX512F ggml_vec_add1_f32 version
  Building AVX512F ggml_vec_acc_f32 version
  Building AVX512F ggml_vec_acc1_f32 version
  Building AVX512F ggml_vec_sub_f32 version
  Building AVX512F ggml_vec_set_f32 version
  Building AVX512F ggml_vec_cpy_f32 version
  Building AVX512F ggml_vec_neg_f32 version
  Building AVX512F ggml_vec_mul_f32 version
  Building AVX512F ggml_vec_div_f32 version
  Building AVX512F ggml_vec_normsq_f32 version
  Building AVX512F ggml_vec_sqrt_f32 version
  Building AVX512F ggml_vec_abs_f32 version
  Building AVX512F ggml_vec_mad_f32 version
  Building AVX512F ggml_vec_mad_f16 version
  Building AVX512F ggml_vec_scale_f32 version
  Building AVX512F ggml_vec_scale_f16 version
D:\llama.cpp\b612.dc_052026\ggml\src\ggml-cpu\ggml-cpu.cpp(122,6): warning C4273: 'ggml_graph_dump_dot_b612': inconsistent dll linkage
 [D:\llama.cpp\b612.dc_052026\build\ggml\src\ggml-cpu.vcxproj]
      D:\llama.cpp\b612.dc_052026\ggml\include\ggml.h(2794,19):
      see previous definition of 'ggml_graph_dump_dot_b612'

     Creating library D:/llama.cpp/b612.dc_052026/build/ggml/src/RelWithDebInfo/ggml-cpu.lib and object D:/llama.cpp/b612.dc_052026/bu
  ild/ggml/src/RelWithDebInfo/ggml-cpu.exp
  Generating code
  Previous IPDB not found, fall back to full compilation.
  All 1243 functions were compiled because no usable IPDB/IOBJ from previous compilation was found.
  Finished generating code
  ggml-cpu.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\ggml-cpu.dll
  Building Custom Rule D:/llama.cpp/b612.dc_052026/ggml/src/CMakeLists.txt
  ggml-backend-reg.cpp
  ggml-backend-dl.cpp
  GGML-IMPL.h: using AVX512 F16C intrinsics
     Creating library D:/llama.cpp/b612.dc_052026/build/ggml/src/RelWithDebInfo/ggml.lib and object D:/llama.cpp/b612.dc_052026/build/
  ggml/src/RelWithDebInfo/ggml.exp
  Generating code
  Previous IPDB not found, fall back to full compilation.
  All 762 functions were compiled because no usable IPDB/IOBJ from previous compilation was found.
  Finished generating code
  ggml.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\ggml.dll
  llama.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\llama.dll
  cpp-httplib.vcxproj -> D:\llama.cpp\b612.dc_052026\build\vendor\cpp-httplib\RelWithDebInfo\cpp-httplib.lib
  llama-common-base.vcxproj -> D:\llama.cpp\b612.dc_052026\build\common\RelWithDebInfo\llama-common-base.lib
  license.cpp
  Auto build dll exports
     Creating library D:/llama.cpp/b612.dc_052026/build/common/RelWithDebInfo/llama-common.lib and object D:/llama.cpp/b612.dc_052026/
  build/common/RelWithDebInfo/llama-common.exp
  llama-common.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\llama-common.dll
  slm-infer.vcxproj -> D:\llama.cpp\b612.dc_052026\build\examples\llm-infer\slm-infer.dir\RelWithDebInfo\slm-infer.lib
  llm-infer-static.vcxproj -> D:\llama.cpp\b612.dc_052026\build\examples\llm-infer\RelWithDebInfo\llm-infer-static.lib
  minslminfer.vcxproj -> D:\llama.cpp\b612.dc_052026\build\bin\RelWithDebInfo\minslminfer.exe
