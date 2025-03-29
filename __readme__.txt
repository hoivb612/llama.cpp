===== Branch hv/b612_dc_120624 =====

commit ba816402f1d6c160ffa6fe883d89aa66aa3c7705
Merge: c1cdf909 c5ede384
Author: Hoi Vo <hoiv@microsoft.com>
Date:   Fri Dec 6 17:10:48 2024 -0800

    Merge remote-tracking branch 'origin/master' into hv/b612

commit c5ede3849fc021174862f9c0bf8273808d8f0d39 (master)
Author: Georgi Gerganov <ggerganov@gmail.com>
Date:   Fri Dec 6 21:33:15 2024 +0200

    convert : add custom attention mapping

=================================================================

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

