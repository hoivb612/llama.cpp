<overview>
The user has been building tools in `D:\llama.cpp\b612\examples\llm-infer\` and analyzing/optimizing quantized matrix multiplication implementations across multiple codebases (`b612` custom, `b612.dc_080625`, and `b612.dc_041126`). Work spans: (1) updating APIs from an older codebase, (2) creating multi-turn conversation drivers with auto chat template detection, (3) building a teacher/student model comparison tool, (4) analyzing and optimizing Q4_K/Q2_K/Q3_K/Q6_K/Q4_0/Q8_0 repack vec_dot implementations for batch processing, (5) fixing Clang/ClangCL build compatibility issues for AVX-512 VNNI/BF16 intrinsics, (6) fixing compilation/linker errors across codebases, and (7) diagnosing a repack buffer corruption issue with Gemma-4 models. The approach has been incremental: build, test, fix based on user feedback, with recent focus on cross-codebase compatibility and in-place repack safety.
</overview>

<history>
1. User asked to update all APIs in `examples/llm-infer` to build in the b612 codebase (copied from older `b612.dc_080625`)
   - Identified 4 categories of breaking changes: KV cache API rename, memory API rename, removed tensor op perf, removed tensor repack mode
   - Updated all occurrences across multiple files
   - Added `add_subdirectory(llm-infer)` to `examples/CMakeLists.txt`
   - Successfully built all 10 targets

2. User asked to create `minslm-multi-og.cpp` based on `minslm-multi.cpp` using SYSTEM/PROMPT script format from `cpf_gem4mm.cpp`
   - Created new file with OG script parser, /context stats, /rewind support, interactive stdin fallback
   - Added CMake target, built successfully
   - Later renamed from `minslminfer-multi-og.cpp` to `minslm-multi-og.cpp`

3. User reported corrupted output — model echoing prompt without chatting
   - Root cause: no chat template wrapping when template file not provided
   - Added `llm_get_chat_template()` wrapping `llama_model_chat_template()`
   - Added `detect_chat_format_from_jinja()` — scans jinja source for known token patterns
   - Implemented 3-level strategy: API → jinja pattern scan → ChatML fallback
   - Gemma-4 matched correctly, model responded properly

4. User reported `/REWIND` (uppercase) not recognized
   - Added `to_lower()` and `is_meta_command()` helpers for case-insensitive matching

5. User asked to add `stop_char` support from `minslm.cpp`
   - Added `params.stop_char = '}'` as default, output truncation in non-streaming path

6. User asked to build `teacher.cpp` — a teacher/student model comparison tool
   - Created `teacher.cpp` with config JSON, 3-pass execution, JSON answer parsing
   - Fixed multiple issues: "VALID" label, results trimming on rewind, teacher reasoning prompt
   - Added original system prompt to reasoning context and results JSON

7. User asked to analyze `Gemma-4_E2B_Q3_K_M_results.json` and compose improved system prompt
   - Analyzed 8 mismatches across 3 failure patterns
   - Created `mara_system_prompt_v2.txt` with 6 explicit decision rules

8. User asked to compare Q4_K repack implementations across two codebases
   - Analyzed `ggml-cpu-repack.c` (b612 custom) vs `arch/x86/repack.cpp` (upstream)
   - b612: Vec-dot approach, 1 row × 1 col, AVX-512 VNNI dpbusd, in-place repack
   - Upstream: True GEMM, 16×16 tiles, maddubs (not VNNI), 8-interleaved blocks
   - Concluded: b612 wins at batch=1 (token gen), upstream wins at batch≥4 (prefill)

9. User asked to build batch-optimized vec_dot for Q4_K (`xx_vec_dot_q4_k_q8_k_x8`)
   - Designed 4-column tiling: load weights once, dpbusd against 4 activation columns
   - Implemented full function with Phase 1 (4-col batch) and Phase 2 (1-col remainder)
   - No repack format change needed — same `block_q4_K_repack`

10. User asked for same analysis and implementation for Q2_K
    - Same verdict as Q4_K: b612 wins batch=1, upstream wins batch≥4
    - Implemented `xx_vec_dot_q2_k_q8_k_x8` batch-tiled version

11. User reported Clang 20.1.8 build errors with AVX-512 intrinsics
    - Fixed `_mm512_dpbf16_ps` in `vec-b612.cpp` — per-function Clang target attributes
    - Fixed `_mm512_cvtne2ps_pbh` in `ggml-cpu-b612.c` — per-function attribute
    - Fixed `_mm512_dpbusd_epi32` in `quants-b612.c` — file-level pragma
    - Proactively fixed `ggml-cpu-repack.c` with same pragma approach
    - Consolidated ALL four b612 source files to use file-level `#pragma clang attribute push/pop`

12. User asked to analyze Q3_K and Q6_K vec_dot routines and implement batch tiling
    - Q3_K: 16 scales in 6-bit packed, hmask high bit for bias, 8 dpbusd/super-block
    - Q6_K: 16 scales 8-bit direct, 6-bit unpack via ql+qh, 8 dpbusd/super-block
    - Implemented batch-tiled versions for both, build verified

13. User asked to analyze Q4_0 and implement batch tiling
    - Key difference: scale `d` is a VECTOR of 8 fp16 (not scalar like K-quants)
    - Clang workaround: `_mm_loadu_si128 + _mm256_cvtph_ps` instead of `_mm_loadu_ph`
    - Implemented batch-tiled `xx_vec_dot_q4_0_q8_0_x8`, build verified

14. User asked to analyze Q8_0 and implement batch tiling
    - Q8_0: signed×signed via abs+mask workaround, 272 bytes/block (largest), no bias needed
    - Unique advantage: abs/mask ops are fully shareable across columns
    - Implemented batch-tiled `xx_vec_dot_q8_0_q8_0_x8`, build verified

15. User asked to fix compilation errors in `D:\llama.cpp\b612.dc_041126\examples\llm-infer\`
    - Fixed 8 KV cache API calls: `llama_kv_self_clear` → `llama_memory_clear`, `llama_kv_self_seq_rm` → `llama_memory_seq_rm`
    - Fixed ggml-base: removed duplicate designated initializers for `GGML_TYPE_Q8_0_x8` / `GGML_TYPE_Q8_0_Q8_0_x8`
    - Fixed ggml-base: added missing `dequantize_row_nvfp4` declaration in `GGML_B612` branch of `ggml-quants.h`
    - Fixed ggml-cpu linker errors: added missing symbols to `ggml-cpu-b612.c` (ops.h declarations, e8m0 table, rvv_vlen stub, threadpool chunk functions, e8m0 init)
    - Fixed duplicate BITNET case block in `llama-model.cpp`
    - Fixed `use_ref` field missing from `ggml_compute_params` initializer causing NULL `generation` crash

16. User asked to fix compilation errors in `examples\xbapp`
    - Fixed 4 KV cache API calls in `slminfer.cpp`
    - Stubbed 3 `llama_set_tensor_repack_mode()` calls and 1 `llama_print_tensor_op_perf()` in `xbapp.cpp`
    - Build verified

17. User reported access violation in `ggml_wait_for_done_xbox()` because `params->generation` is NULL
    - Root cause: `ggml_compute_params` struct in dc_041126 has a `use_ref` field (line 29) that dc_080625 doesn't have
    - The initializer in `ggml-cpu-b612.c` was missing `use_ref`, causing positional misalignment: `barrier` got `barrier_tb` pointer (→ `use_ref` as bool=true), `generation` got `generation_tb` (→ `barrier`), and `generation` was left uninitialized (NULL)
    - Fixed by adding `/*.use_ref =*/ false` to the initializer
    - Build verified (MSVC ICE in repack.cpp is pre-existing, unrelated)

18. User asked to fix compilation errors in `examples\chatbot`, `retrieval`, and `llama-bench`
    - **chatbot**: Fixed 3 KV cache API calls in `hist.cpp`; fixed console enum names `console::user_input` → `DISPLAY_TYPE_USER_INPUT` and `console::reset` → `DISPLAY_TYPE_RESET` in `main.cpp`
    - **llama-bench**: Removed duplicate `#ifdef GGML_B612` block that redefined `tensor_repack_mode`
    - **retrieval**: Fixed `common_init_result` → `auto` + `->model()`/`->context()` (2 functions); replaced undefined `batch_process()` with existing `batch_decode()` (3 calls); added missing `t_embeddings_stop` variable
    - All three targets built successfully

19. User asked about in-place repack buffer corruption with Gemma-4 models in dc_041126
    - The `ggml_repack_tensor()` function at line 4520 only allocates a new buffer when `GGML_TENSOR_FLAG_DUP` is set; otherwise it repacks in-place on the mmap'd buffer
    - User confirmed: commenting out line 4520 (forcing all tensors to use malloc'd buffers) fixes Gemma-4 output
    - Investigated upstream approach: upstream always uses separate src/dst buffers via `set_tensor()` callback
    - Investigated `GGML_TENSOR_FLAG_DUP` — only set for `TOKEN_EMBD` with `TENSOR_DUPLICATED` flag
    - Analysis was ongoing when user clarified the problem is only in dc_041126 code (dc_080625 can't process Gemma-4 anyway)
    - **NOT YET RESOLVED** — diagnosis still in progress
</history>

<work_done>
Files created:
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`: Multi-turn driver with OG script format, auto chat template detection
- `D:\llama.cpp\b612\examples\llm-infer\teacher\teacher.cpp`: Teacher/student comparison tool
- `D:\llama.cpp\b612\examples\llm-infer\teacher\mara_system_prompt_v2.txt`: Improved system prompt

Files modified in b612 main codebase:
- `D:\llama.cpp\b612\examples\CMakeLists.txt`: Added `add_subdirectory(llm-infer)`
- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`: Added targets
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`: Updated deprecated API calls; added `llm_get_chat_template()`
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`: Added `llm_get_chat_template()` declaration

Files modified in b612.dc_080625 codebase:
- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`: Added `_cp` declarations for Q2_K, Q3_K, Q4_K, Q4_0, Q6_K, Q8_0 batch-tiled variants
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`: Added batch-tiled implementations for ALL 6 quant formats; renamed originals to `_dc`; added Clang pragma
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\vec-b612.cpp`: File-level Clang pragma for avx512 features
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`: File-level Clang pragma
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\quants-b612.c`: File-level Clang pragma

Files modified in b612.dc_041126 codebase:
- `D:\llama.cpp\b612.dc_041126\examples\llm-infer\dll\llm-infer.cpp`: Updated 8 KV cache API calls
- `D:\llama.cpp\b612.dc_041126\examples\xbapp\slminfer.cpp`: Updated 4 KV cache API calls
- `D:\llama.cpp\b612.dc_041126\examples\xbapp\xbapp.cpp`: Stubbed removed repack/perf APIs
- `D:\llama.cpp\b612.dc_041126\examples\chatbot\hist.cpp`: Updated 3 KV cache API calls
- `D:\llama.cpp\b612.dc_041126\examples\chatbot\main.cpp`: Fixed console enum names
- `D:\llama.cpp\b612.dc_041126\examples\retrieval\retrieval.cpp`: Fixed init API, batch_process→batch_decode, missing t_embeddings_stop
- `D:\llama.cpp\b612.dc_041126\tools\llama-bench\llama-bench.cpp`: Removed duplicate #ifdef GGML_B612 block
- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml.c`: Removed duplicate designated initializers at ~lines 1223-1254
- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-quants.h`: Added missing `dequantize_row_nvfp4` in GGML_B612 else branch
- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\ops.h`: Added 4 declarations: reshape, view, permute, transpose
- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`: Added `ggml_table_f32_e8m0_half` + init, `ggml_cpu_get_rvv_vlen()` stub, `ggml_threadpool_chunk_set/add`, fixed `use_ref` in compute_params initializer
- `D:\llama.cpp\b612.dc_041126\src\llama-model.cpp`: Removed duplicate BITNET case block

Work completed:
- [x] Update llm-infer APIs for b612 codebase
- [x] Create minslm-multi-og.cpp with OG script format
- [x] Fix corrupted output — auto-detect chat template
- [x] Make meta commands case-insensitive
- [x] Add stop_char support
- [x] Create teacher.cpp
- [x] Compose improved system prompt v2
- [x] Implement batch-tiled Q4_K vec_dot
- [x] Implement batch-tiled Q2_K vec_dot
- [x] Fix Clang avx512 errors across all 4 b612 source files
- [x] Implement batch-tiled Q3_K vec_dot
- [x] Implement batch-tiled Q6_K vec_dot
- [x] Implement batch-tiled Q4_0 vec_dot
- [x] Implement batch-tiled Q8_0 vec_dot
- [x] Fix llm-infer API calls in b612.dc_041126
- [x] Fix ggml-base build errors in b612.dc_041126
- [x] Fix ggml-cpu linker errors in b612.dc_041126
- [x] Fix xbapp API calls in b612.dc_041126
- [x] Fix use_ref crash in b612.dc_041126
- [x] Fix chatbot, retrieval, llama-bench in b612.dc_041126
- [ ] **Diagnose/fix Gemma-4 repack buffer corruption in b612.dc_041126** ← IN PROGRESS
</work_done>

<technical_details>
### API Migration (b612.dc_080625/041126 → b612)
- `llama_kv_self_clear(ctx)` → `llama_memory_clear(llama_get_memory(ctx), true)`
- `llama_kv_self_seq_rm(ctx, seq, p0, p1)` → `llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, p1)`
- `llama_kv_self_seq_add(ctx, seq, p0, p1, delta)` → `llama_memory_seq_add(llama_get_memory(ctx), seq, p0, p1, delta)`
- `llama_memory_t` obtained via `llama_get_memory(ctx)` (typedef for `llama_memory_i *`)
- `llama_print_tensor_op_perf()` and `llama_set_tensor_repack_mode()` completely removed in b612 but still exist in b612.dc_041126
- `common_init_from_params()` now returns `common_init_result_ptr` (unique_ptr); access via `->model()`, `->context()` instead of `.model.get()`, `.context.get()`
- Console API changed: `console::user_input` → `DISPLAY_TYPE_USER_INPUT`, `console::reset` → `DISPLAY_TYPE_RESET`

### Chat Template Detection (3-level strategy in minslm-multi-og.cpp)
1. `llama_chat_apply_template()` with model's GGUF template
2. `detect_chat_format_from_jinja()` — pattern scan for known special tokens (Phi-3/4, Llama-3, Gemma-4, ChatML, Mistral, Command-R)
3. ChatML fallback via `tmpl=nullptr`

### Repack Batch-Tiling Pattern (applied to Q2_K, Q3_K, Q4_K, Q4_0, Q6_K, Q8_0 in b612.dc_080625)
**Naming convention:**
- `xx_vec_dot_<quant>_x8`: Production function (batch-tiled version)
- `xx_vec_dot_<quant>_x8_dc`: Original/baseline version (renamed, dead code preserved)
- `xx_vec_dot_<quant>_x8_cp`: Linker export alias (MSVC only, `#ifndef __clang__`)

**Design pattern:**
- Phase 1: groups of 4 columns, load weight data ONCE per super-block, dpbusd against 4 activation columns
- Phase 2: remainder 1-at-a-time for `ncols % 4`
- No repack format change — same block structs, drop-in replacement
- Reduces weight memory traffic by ~4× for batch ≥ 4

**Per-quant characteristics:**
- Q4_K: 144 bytes, 4 dpbusd/block, scale+offset decode from 6-bit packed, ~12 zmm
- Q2_K: 84 bytes, 4 dpbusd/block, 256-bit mins path, ~17 zmm
- Q3_K: 110 bytes, 8 dpbusd/block (4 sumi + 4 bias via hmask), ~16 zmm
- Q6_K: 210 bytes, 8 dpbusd/block, 6-bit unpack via ql+qh, ~17 zmm
- Q4_0: 144 bytes, VECTOR scale d[8] (8×fp16 not scalar), 8 dpbusd/block, Clang `_mm_loadu_ph` workaround, ~15 zmm
- Q8_0: 272 bytes (LARGEST), signed×signed via abs+mask, NO bias, shared abs/mask ops, ~14 zmm

### Clang/ClangCL AVX-512 Compatibility
- MSVC doesn't enforce target feature requirements; Clang does via `always_inline` checks
- Solution: file-level `#pragma clang attribute push/pop` guarded by `#ifdef __clang__`
- All 4 b612 source files use: `avx512f,avx512vnni,avx512vl,avx512bf16` (repack.c uses no bf16)

### b612.dc_041126 Build Issues (ALL RESOLVED except repack corruption)
- **GGML_B612 conditional build**: When ON (default), `ggml-cpu.c` is EXCLUDED from compilation — replaced by `ggml-cpu-b612.c`. But newer code (repack.cpp, ggml-cpu.cpp) references symbols only in `ggml-cpu.c`
- **GGML_B612 IS defined for ggml-base** (confirmed via vcxproj PreprocessorDefinitions) despite only being added in ggml-cpu CMakeLists — this is because `add_compile_definitions` propagates to directory scope
- **Duplicate designated initializers**: `GGML_TYPE_Q8_0_x8` and `GGML_TYPE_Q8_0_Q8_0_x8` appeared twice in ggml.c type_traits array — MSVC C mode rejects this (C2099)
- **Missing nvfp4 declaration**: `dequantize_row_nvfp4` was only in `#if !defined(GGML_B612)` branch of ggml-quants.h
- **Missing ggml_cpu_get_rvv_vlen**: defined in ggml-cpu.c (excluded), called from ggml-cpu.cpp (included)
- **Missing ggml_threadpool_chunk_set/add**: defined in ggml-cpu.c, referenced in repack.cpp (top-level, NOT arch/x86/repack.cpp)
- **Missing ggml_table_f32_e8m0_half**: defined in ggml-cpu.c, referenced via simd-mappings.h
- **Missing ops.h declarations**: reshape/view/permute/transpose defined in ops-b612.cpp but not declared in ops.h
- **use_ref field mismatch**: dc_041126 added `bool use_ref` to `ggml_compute_params` struct between `threadpool` and `barrier`/`generation`, but the b612 initializer didn't include it, causing positional shift → `generation` was NULL → crash in `ggml_wait_for_done_xbox()`
- **Duplicate BITNET case block**: `llama-model.cpp` had two identical `case LLM_ARCH_BITNET_B158 / LLM_ARCH_BITNET_25` blocks in the same switch
- **Duplicate GGML_B612 block in llama-bench**: `tensor_repack_mode` variable defined twice
- **MSVC ICE**: `arch/x86/repack.cpp` triggers internal compiler error C1001 — pre-existing, unrelated to our changes

### Gemma-4 Repack Buffer Corruption (IN PROGRESS — dc_041126 only)
- **Symptom**: Garbage output when running Gemma-4 models with repack-xbox mode
- **Workaround confirmed**: Commenting out line 4520 (`if (tensor->flags & GGML_TENSOR_FLAG_DUP)`) and forcing ALL tensors to malloc makes it work
- **Key difference vs upstream**: b612 repacks IN-PLACE (same src/dst pointer); upstream always uses separate src buffer (original data from `set_tensor()` callback) and dst buffer (newly allocated)
- **DUP flag**: Only set on `TOKEN_EMBD` tensors with `TENSOR_DUPLICATED` flag (line 2201 in llama-model.cpp), intended for output/embed weight tying
- **Analysis status**: Was investigating whether Gemma-4 has weight sharing beyond TOKEN_EMBD that the DUP check doesn't catch. The user clarified this is a dc_041126-only problem. The `ggml_repack_tensor()` multi-threaded path (line 4492) already has the if commented out (line 4520). The single-thread path (line 4370) still has it active at line 4401.
- **Approach options being considered**:
  1. Detect data-pointer aliasing at repack time (scan all tensors for shared data pointers)
  2. Always malloc for in-place repack (user's working workaround, but doubles memory for all weights)
  3. Lazy/on-demand approach: only malloc when the same data pointer is seen more than once
  4. Investigate whether mmap COW semantics on Windows could cause issues

### Build Environment
- Windows, Visual Studio (MSVC) and Clang 20.1.8 (ClangCL via `-TClangCL`)
- b612 build dir: `D:\llama.cpp\b612\build.Vulkan`
- b612.dc_080625 build dir: `D:\llama.cpp\b612.dc_080625\build\`
- b612.dc_041126 build dir: `D:\llama.cpp\b612.dc_041126\build\`
- Build command: `cmake --build <dir> --config RelWithDebInfo --target <name>`

### Edit Gotcha
When using the edit tool to replace text at function boundaries, the next function's signature can be accidentally removed if `old_str` captures too far. Always verify subsequent function signatures after boundary edits.
</technical_details>

<important_files>
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`
  - Multi-turn driver with OG script format, chat template detection
  - Created in this session

- `D:\llama.cpp\b612\examples\llm-infer\teacher\teacher.cpp`
  - Teacher/student comparison tool with 3-pass execution
  - Created in this session

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`
  - Core llm-infer library — all API calls updated for b612
  - `llm_get_chat_template()` at ~line 142

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`
  - Public API header — model_params struct, stop_char at line 68

- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`
  - Header for repack — all `_cp` declarations for 6 quant formats
  - block_q4_0_repack (lines 14-17), block_q8_0_repack (lines 21-24)

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`
  - **PRIMARY FILE** — all b612 batch-tiled implementations
  - All 6 quant formats: Q2_K, Q3_K, Q4_K, Q4_0, Q6_K, Q8_0
  - Each has `_dc` (original) and batch-tiled (takes `_x8` name) versions
  - Clang pragma push near top, pop at end

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\vec-b612.cpp`
  - BF16 vec_dot functions, file-level Clang pragma

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`
  - Core b612 functions, file-level Clang pragma

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\quants-b612.c`
  - Vec_dot functions using dpbusd, file-level Clang pragma

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`
  - Added `ggml_table_f32_e8m0_half` (after line 98) + init (after line 6678)
  - Added `ggml_cpu_get_rvv_vlen()` (after line 6504)
  - Added `ggml_threadpool_chunk_set/add` (after threadpool struct ~line 632)
  - Fixed `use_ref` in compute_params initializer (line ~5955)

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`
  - **ACTIVE INVESTIGATION** — Gemma-4 repack corruption
  - Line 4401: DUP check in single-thread path (still active)
  - Line 4520: DUP check in multi-thread path (currently commented out by user)
  - Line 4492: `ggml_repack_tensor()` multi-threaded entry point
  - Line 4370: `ggml_repack_tensor_single_thread()` single-threaded entry point
  - The in-place repack (lines 4580+) shares src/dst pointer

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\ggml-cpu-impl.h`
  - `ggml_compute_params` struct at line 18 — has `use_ref` field at line 29 that dc_080625 doesn't have
  - `barrier`/`generation` at lines 33-34 under `#ifdef GGML_XBOX_PERF`

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\CMakeLists.txt`
  - Key: when `GGML_B612` ON, `ggml-cpu.c` excluded, b612 files used instead (lines 46-79)

- `D:\llama.cpp\b612.dc_041126\src\llama-model.cpp`
  - DUP flag logic at ~line 2201: only TOKEN_EMBD + TENSOR_DUPLICATED gets DUP flag
  - Gemma3 tensor loading at ~line 3478, Gemma3N at ~line 3512
  - Removed duplicate BITNET case block (was at ~line 5636)

- `D:\llama.cpp\b612.dc_041126\examples\chatbot\hist.cpp`
  - Fixed 3 KV cache API calls

- `D:\llama.cpp\b612.dc_041126\examples\chatbot\main.cpp`
  - Fixed console enum names

- `D:\llama.cpp\b612.dc_041126\examples\retrieval\retrieval.cpp`
  - Fixed init API, batch_process→batch_decode, missing t_embeddings_stop

- `D:\llama.cpp\b612.dc_041126\tools\llama-bench\llama-bench.cpp`
  - Removed duplicate #ifdef GGML_B612 block

- `D:\llama.cpp\b612.dc_041126\examples\xbapp\slminfer.cpp`
  - Fixed 4 KV cache API calls

- `D:\llama.cpp\b612.dc_041126\examples\xbapp\xbapp.cpp`
  - Stubbed removed repack/perf APIs

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\repack.cpp`
  - Top-level repack.cpp (NOT arch/x86) — references `ggml_threadpool_chunk_set/add` at lines 4349, 4376, 4382
  - This is the upstream repack approach with separate src/dst buffers via `set_tensor()` callback

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ops-b612.cpp`
  - Defines the 4 NOP forward functions (reshape, view, permute, transpose) at lines 7136-7174
</important_files>

<next_steps>
Remaining work for Gemma-4 repack buffer corruption (dc_041126):

1. **Root cause analysis** — Need to determine WHY in-place repack corrupts Gemma-4 but not other models. Two hypotheses:
   a. **Data aliasing**: Multiple tensor objects share the same `data` pointer beyond what DUP flag covers. Need to check if Gemma-4's model loading creates tensors that alias data but aren't flagged DUP.
   b. **Compute graph ordering**: A tensor is repacked in-place by one MUL_MAT node, then a later node reads the same data expecting original format (e.g., through a different tensor object or for a non-MUL_MAT op).

2. **Proposed approach** — Instead of always malloc'ing (doubles memory), detect aliasing at repack time:
   - Build a hash set of `data` pointers during the repack callgraph pass
   - If a data pointer appears more than once across different tensor objects, mark those tensors as needing separate buffers
   - This gives the safety of always-malloc with the efficiency of in-place for non-aliased tensors
   - Alternative: only malloc on first repack of each unique data pointer, then subsequent tensor objects sharing that pointer can be detected

3. **Immediate next actions**:
   - Examine dc_041126's `llama-model.cpp` Gemma3/Gemma3N tensor loading to find all `TENSOR_DUPLICATED` uses and data-sharing patterns
   - Add diagnostic printf in `ggml_repack_tensor` to log tensor name + data pointer, then run Gemma-4 to find aliased pointers
   - Implement the chosen fix (aliasing detection or selective malloc)
   - Build and test with `minslminfer.exe` against Gemma-4 model
</next_steps>