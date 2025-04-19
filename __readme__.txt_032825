git merge origin/hv/b612 // -> hv/b612_dc_merged

origin/hv/b612 came from 03282025
commit 3714c3ee1a62ed64ac328ec7d699410ad1219150 (HEAD -> master, origin/master, origin/HEAD)
Author: Sigbjørn Skjæret <sigbjorn.skjaeret@scala.com>
Date:   Fri Mar 28 22:13:02 2025 +0100

    llama : fix incorrect Qwen2Moe ffn_moe_out graph callback (#12631)

C:\llama.cpp\llama.cpp.b612.dc>git merge origin/hv/b612
Auto-merging CMakeLists.txt
Auto-merging common/CMakeLists.txt
Auto-merging common/arg.cpp
CONFLICT (content): Merge conflict in common/arg.cpp ----
Auto-merging common/common.cpp
Auto-merging common/common.h
CONFLICT (content): Merge conflict in common/common.h ----
Auto-merging common/log.cpp
Auto-merging convert_hf_to_gguf.py
CONFLICT (content): Merge conflict in convert_hf_to_gguf.py ----
Auto-merging convert_hf_to_gguf_update.py
CONFLICT (content): Merge conflict in convert_hf_to_gguf_update.py ----
Auto-merging examples/CMakeLists.txt
Auto-merging examples/batched-bench/batched-bench.cpp
Auto-merging examples/llama-bench/llama-bench.cpp
CONFLICT (content): Merge conflict in examples/llama-bench/llama-bench.cpp ----
Auto-merging examples/retrieval/1liners.txt
CONFLICT (add/add): Merge conflict in examples/retrieval/1liners.txt ----
Auto-merging examples/retrieval/retrieval.cpp
CONFLICT (content): Merge conflict in examples/retrieval/retrieval.cpp ----
Auto-merging ggml/CMakeLists.txt
CONFLICT (content): Merge conflict in ggml/CMakeLists.txt ----
Auto-merging ggml/include/ggml-cpu.h
CONFLICT (content): Merge conflict in ggml/include/ggml-cpu.h ----
Auto-merging ggml/include/ggml.h
Auto-merging ggml/src/CMakeLists.txt
CONFLICT (content): Merge conflict in ggml/src/CMakeLists.txt ----
Auto-merging ggml/src/ggml-cpu/CMakeLists.txt
CONFLICT (content): Merge conflict in ggml/src/ggml-cpu/CMakeLists.txt ----
Auto-merging ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp
CONFLICT (content): Merge conflict in ggml/src/ggml-cpu/ggml-cpu-aarch64.cpp ----
Auto-merging ggml/src/ggml-cpu/ggml-cpu.c
CONFLICT (content): Merge conflict in ggml/src/ggml-cpu/ggml-cpu.c ----
Auto-merging ggml/src/ggml-cpu/ggml-cpu.cpp
Auto-merging ggml/src/ggml-cuda/CMakeLists.txt
Auto-merging ggml/src/ggml-impl.h
Auto-merging ggml/src/ggml-quants.c
CONFLICT (content): Merge conflict in ggml/src/ggml-quants.c ----
Auto-merging ggml/src/ggml.c
Auto-merging include/llama.h
CONFLICT (content): Merge conflict in include/llama.h ----
Auto-merging src/CMakeLists.txt
Auto-merging src/llama-vocab.cpp
CONFLICT (content): Merge conflict in src/llama-vocab.cpp ----
Auto-merging src/llama.cpp
CONFLICT (content): Merge conflict in src/llama.cpp ----
Auto-merging tests/test-tokenizer-0.cpp
Automatic merge failed; fix conflicts and then commit the result.

====================================

Offspring for hoivb612
https://github.com/hoivb612/llama.cpp

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

