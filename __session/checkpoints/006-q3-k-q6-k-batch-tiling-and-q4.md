<overview>
The user has been building tools in `D:\llama.cpp\b612\examples\llm-infer\` and analyzing/optimizing quantized matrix multiplication implementations across two codebases (`b612` custom and upstream `b612.dc_080625`). Work spans: (1) updating APIs from an older codebase, (2) creating multi-turn conversation drivers with auto chat template detection, (3) building a teacher/student model comparison tool, (4) analyzing and optimizing Q4_K/Q2_K/Q3_K/Q6_K repack vec_dot implementations for batch processing, (5) fixing Clang/ClangCL build compatibility issues for AVX-512 VNNI/BF16 intrinsics, and (6) analyzing Q4_0 for the same batch-tiling optimization. The approach has been incremental: build, test, fix based on user feedback, with recent focus on SIMD optimization and cross-compiler compatibility.
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
   - First attempt: `llama_chat_apply_template()` returned -1 for Gemma-4 (unrecognized)
   - Added `detect_chat_format_from_jinja()` — scans jinja source for known token patterns
   - Implemented 3-level strategy: API → jinja pattern scan → ChatML fallback
   - Gemma-4 matched correctly, model responded properly

4. User reported `/REWIND` (uppercase) not recognized
   - Added `to_lower()` and `is_meta_command()` helpers for case-insensitive matching

5. User asked to add `stop_char` support from `minslm.cpp`
   - Added `params.stop_char = '}'` as default, output truncation in non-streaming path

6. User asked to build `teacher.cpp` — a teacher/student model comparison tool
   - Created plan: 3-pass sequential design (teacher → student → teacher reasoning)
   - User corrected: `/rewind` must execute in both passes (affects KV cache state)
   - Created `teacher.cpp` with config JSON, script parser, 3-pass loop
   - Fixed multiple issues: "VALID" label, results trimming on rewind, teacher reasoning prompt composition
   - Added original system prompt to reasoning context and results JSON

7. User asked to analyze `Gemma-4_E2B_Q3_K_M_results.json` and compose improved system prompt
   - Analyzed 8 mismatches across 3 failure patterns
   - Created `mara_system_prompt_v2.txt` with 6 explicit decision rules

8. User asked to compare Q4_K repack implementations across two codebases
   - Analyzed `ggml-cpu-repack.c` (b612 custom) vs `arch/x86/repack.cpp` (upstream)
   - b612: Vec-dot approach, 1 row × 1 col, AVX-512 VNNI dpbusd, in-place repack
   - Upstream: True GEMM, 16×16 tiles, maddubs (not VNNI), 8-interleaved blocks
   - Concluded: b612 wins at batch=1 (token gen), upstream wins at batch≥4 (prefill)

9. User asked to build `xx_vec_dot_q4_k_q8_k_x8_cp()` — batch-optimized vec_dot
   - Designed 4-column tiling: load weights once, dpbusd against 4 activation columns
   - Implemented full function with Phase 1 (4-col batch) and Phase 2 (1-col remainder)
   - Added declaration to header, implementation to .c file
   - No repack format change needed — same `block_q4_K_repack`

10. User asked to do the same analysis for Q2_K
    - Analyzed b612 `xx_vec_dot_q2_k_q8_k_x8` vs upstream `ggml_gemm_q2_K_8x8_q8_K`
    - Same verdict as Q4_K: b612 wins batch=1, upstream wins batch≥4, gap narrower for Q2_K

11. User asked to implement `xx_vec_dot_q2_k_q8_k_x8_cp()` for Q2_K
    - Followed same pattern as Q4_K: renamed original to `_dc`, new batch-tiled version
    - Phase 1: 4-column tiling with shared q2 weight loads and 256-bit mins path
    - Phase 2: remainder columns 1-at-a-time

12. User reported Clang 20.1.8 build errors with AVX-512 intrinsics
    - Fixed `_mm512_dpbf16_ps` in `vec-b612.cpp` — per-function Clang target attributes
    - Fixed `_mm512_cvtne2ps_pbh` in `ggml-cpu-b612.c` — per-function attribute
    - Fixed `_mm512_dpbusd_epi32` in `quants-b612.c` — file-level `#pragma clang attribute push/pop`
    - Proactively fixed `ggml-cpu-repack.c` with same pragma approach
    - Then consolidated ALL four b612 source files to use file-level pragma for consistency
    - Removed per-function attributes from `vec-b612.cpp` and `ggml-cpu-b612.c`, replaced with file-level pragma
    - User confirmed: Clang build succeeded

13. User asked to analyze Q3_K and Q6_K vec_dot routines (no upstream equivalents)
    - Analyzed `xx_vec_dot_q3_k_q8_k_x8` (159 lines): 16 scales in 6-bit packed, 2 dpbusd per iteration (sumi + bias via hmask), 8 dpbusd total per super-block, 96 bytes weight/block
    - Analyzed `xx_vec_dot_q6_k_q8_k_x8` (178 lines): 16 scales 8-bit direct (simplest decode), 6-bit unpack via ql+qh merge, 8 dpbusd total, 192 bytes weight/block (largest)
    - Concluded both are strong candidates for batch-tiling; Q6_K benefits most due to largest weight blocks

14. User asked to implement batch tiling for Q3_K and Q6_K
    - Renamed originals to `_dc`, created new batch-tiled versions taking `_x8` name
    - Q3_K: shared q3 low-bits extraction + hmask→blend across 4 columns, 8 bias + 8 sumi dpbusd per column per super-block
    - Q6_K: shared ql/qh load + 6-bit unpack, shared q2bits rotation, 16 dpbusd per super-block (4 cols × 4 dpbusd)
    - Added `_cp` linker exports and header declarations for both
    - Build succeeded, verified symbols in object file
    - Hit one issue: Q4_K_dc function signature was accidentally removed during edit, fixed

15. User asked to analyze Q4_0 for the same treatment
    - Analyzed `xx_vec_dot_q4_0_q8_0_x8` (133 lines): custom `block_q4_0_repack` (d[8] + qs[128] = 144B), unsigned + bias subtraction, 8 dpbusd per super-block, vector scale (8 distinct fp16 multipliers)
    - Analyzed upstream `ggml_gemm_q4_0_8x8_q8_0` (~745 lines): same 144B blocks but different interleaving, signextendlut for signed conversion, maddubs not VNNI, 16×16 output tile
    - Key Q4_0 difference: scale `d` is a vector of 8 fp16 (one per sub-block), not a single scalar — shared part is weight's d[8], per-column part is activation d[8]
    - Concluded: excellent candidate for batch tiling, simplest inner loop, lightest register budget

16. User asked to implement batch tiling for Q4_0
    - Started implementation — this is where compaction occurs
</history>

<work_done>
Files created:
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`: Multi-turn driver with OG script format, auto chat template detection (3-level), case-insensitive meta commands, stop_char support
- `D:\llama.cpp\b612\examples\llm-infer\teacher\teacher.cpp`: Teacher/student comparison tool — config JSON, 3-pass execution, JSON answer parsing, mismatch detection, teacher reasoning
- `D:\llama.cpp\b612\examples\llm-infer\teacher\mara_system_prompt_v2.txt`: Improved system prompt with 6 explicit decision rules

Files modified in b612 main codebase:
- `D:\llama.cpp\b612\examples\CMakeLists.txt`: Added `add_subdirectory(llm-infer)` after `add_subdirectory(xbapp)`
- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`: Added `minslminfer-multi-og` target and `teacher` target
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`: Updated deprecated API calls; added `llm_get_chat_template()` implementation
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`: Added `llm_get_chat_template()` declaration after line 77

Files modified in b612.dc_080625 codebase:
- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`: Added `_cp` declarations for Q2_K, Q3_K, Q4_K, Q6_K batch-tiled variants
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`: Added batch-tiled implementations for Q2_K, Q3_K, Q4_K, Q6_K; renamed originals to `_dc`; added Clang pragma push/pop for avx512vnni; added analysis comment blocks for Q4_0
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\vec-b612.cpp`: File-level Clang pragma push/pop for avx512f,avx512vnni,avx512vl,avx512bf16 (replaced per-function attrs)
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`: File-level Clang pragma push/pop for avx512f,avx512vnni,avx512vl,avx512bf16 (replaced per-function attr)
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\quants-b612.c`: File-level Clang pragma push/pop for avx512f,avx512vnni,avx512vl,avx512bf16

Work completed:
- [x] Update llm-infer APIs for current b612 codebase
- [x] Build all llm-infer targets successfully
- [x] Create minslm-multi-og.cpp with OG script format parser
- [x] Fix corrupted output — auto-detect chat template from model GGUF metadata
- [x] Add Gemma-4 pattern detection for `<|turn>/<turn|>` tokens
- [x] Make meta commands case-insensitive
- [x] Add stop_char='}' default and output truncation
- [x] Create teacher.cpp with 3-pass comparison logic
- [x] Fix rewind result trimming bug (keep all answers for comparison)
- [x] Include original system prompt in teacher reasoning and results JSON
- [x] Analyze mismatch results and create improved system prompt v2
- [x] Compare Q4_K repack implementations (b612 vs upstream)
- [x] Implement `xx_vec_dot_q4_k_q8_k_x8` batch-tiled Q4_K vec_dot (was `_cp`)
- [x] Compare Q2_K repack implementations (b612 vs upstream)
- [x] Implement `xx_vec_dot_q2_k_q8_k_x8` batch-tiled Q2_K vec_dot (was `_cp`)
- [x] Fix Clang avx512bf16/avx512vnni errors across all 4 b612 source files (file-level pragmas)
- [x] Analyze Q3_K and Q6_K vec_dot routines for batch-tiling potential
- [x] Implement `xx_vec_dot_q3_k_q8_k_x8` batch-tiled Q3_K vec_dot
- [x] Implement `xx_vec_dot_q6_k_q8_k_x8` batch-tiled Q6_K vec_dot
- [x] Build verified for all batch-tiled implementations (Q2_K, Q3_K, Q4_K, Q6_K)
- [x] Analyze Q4_0 vec_dot for batch-tiling potential
- [ ] **Implement `xx_vec_dot_q4_0_q8_0_x8` batch-tiled Q4_0 vec_dot** ← IN PROGRESS
</work_done>

<technical_details>
### API Migration (b612.dc_080625 → b612)
- `llama_kv_self_clear(ctx)` → `llama_memory_clear(llama_get_memory(ctx), true)`
- `llama_kv_self_seq_rm(ctx, seq, p0, p1)` → `llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, p1)`
- `llama_memory_t` obtained via `llama_get_memory(ctx)` (typedef for `llama_memory_i *`)
- `llama_print_tensor_op_perf()` and `llama_set_tensor_repack_mode()` completely removed

### Chat Template Detection (3-level strategy)
1. `llama_chat_apply_template()` with model's GGUF template — works for API-recognized templates
2. `detect_chat_format_from_jinja()` — scans jinja source for known special-token patterns:
   - `<|user|>` + `<|assistant|>` + `<|end|>` → Phi-3/4
   - `<|start_header_id|>` → Llama-3
   - `<|turn>` + `<turn|>` → Gemma-4
   - `<start_of_turn>` → Gemma 1/2
   - `<|im_start|>` → ChatML
   - `[INST]` → Mistral/Llama-2
   - `<|START_OF_TURN_TOKEN|>` → Command-R
3. ChatML fallback via `tmpl=nullptr`

### Gemma-4 Chat Template
```
<|turn>system\n{system_prompt}<turn|>\n<|turn>user\n{user_text}<turn|>\n<|turn>model\n
```

### teacher.cpp Design
- 3-pass sequential: teacher → student → teacher reasoning
- Uses same SYSTEM/PROMPT script format as minslm-multi-og
- `/rewind` MUST execute in both passes (affects KV cache state)
- `/context` is a no-op (skipped silently)
- Results NOT trimmed on rewind — all generated answers kept for comparison
- Answer comparison: extract first letter from `"answer"` JSON field, compare lowercase
- nlohmann/json vendored at `examples/llm-infer/include/nlohmann/json.hpp`

### Repack Batch-Tiling Pattern (applied to Q2_K, Q3_K, Q4_K, Q6_K)
**Naming convention:**
- `xx_vec_dot_<quant>_x8`: Production function name (called by runtime) — NOW the batch-tiled version
- `xx_vec_dot_<quant>_x8_dc`: Original/baseline version (renamed, dead code preserved)
- `xx_vec_dot_<quant>_x8_cp`: Linker export alias of the batch-tiled `_x8` function (MSVC only, `#ifndef __clang__`)

**Design pattern (all 4 quants follow this):**
- Phase 1: groups of 4 columns (`ncols & ~3ULL`), load weight data ONCE per super-block, dpbusd against 4 activation columns
- Phase 2: remainder 1-at-a-time for `ncols % 4`
- No repack format change — same block structs, drop-in replacement
- Reduces weight memory traffic by ~4× for batch ≥ 4

### Per-quant characteristics:

**Q4_K** (block_q4_K_repack = block_q4_K, 144 bytes):
- 8 scales + 8 mins decoded from 6-bit packed (12-byte `scales[]` via kmask1/2/4)
- 4 dpbusd per super-block, 128-bit mins path (`_mm_madd_epi16` with 8 bsums)
- Has both `d` and `dmin` (scale + offset quant)
- ~12 zmm registers used in batch version

**Q2_K** (block_q2_K_repack = block_q2_K, 84 bytes):
- 16 scales (4-bit packed), simple AND/shift decode
- 4 dpbusd per super-block, 256-bit mins path (`_mm256_madd_epi16` with 16 bsums)
- `REDUCE_Q2_CP` macro: 512→256, subtract 256-bit mins, 256→128, hadd→scalar
- ~17 zmm registers in batch version

**Q3_K** (block_q3_K_repack = block_q3_K, 110 bytes):
- 16 scales in 6-bit packed format (complex decode via kmask1/kmask2)
- NO mins/dmin — uses signed scales (scales - 32) + bias via hmask high bit
- High bits: `uint64 → __mmask64 → mask_blend(m4, zero512)` — unique branchy-mask approach
- 8 dpbusd per super-block (4 sumi + 4 bias)
- ~16 zmm in batch version

**Q6_K** (block_q6_K_repack = block_q6_K, 210 bytes — LARGEST):
- 16 scales, 8-bit direct (simplest decode: `_mm_loadu_si128 + cvtepi8_epi32`)
- NO mins/dmin — bias via constant 32 × q8 sums
- 6-bit unpack: ql (4-bit low) + qh (2-bit high) merged via OR, qh rotated via `ror_epi32`
- 8 dpbusd per super-block (4 sumi + 4 bias)
- ~17 zmm in batch version
- Benefits MOST from batch tiling due to largest weight blocks

**Q4_0** (block_q4_0_repack, custom struct, 144 bytes) — ANALYSIS COMPLETE, IMPLEMENTATION IN PROGRESS:
- `d[8]` (8 × fp16 scales) + `qs[128]` (nibble-packed, interleaved from 8 q4_0 blocks)
- Packs 8 original block_q4_0 (20 bytes each = 160) into 144 bytes
- **Unsigned** values (0-15), bias subtracted by constant 8
- Scale is a **VECTOR** of 8 fp16 values (one per sub-block), NOT a single scalar like K-quants
- Uses `_mm512_insertf32x8` to broadcast 8 scales → 16 lanes
- 8 dpbusd per super-block (4 sumi + 4 bias with constant offset=8)
- Clang workaround: `_mm_loadu_ph` not available, uses `_mm_loadu_si128 + _mm256_cvtph_ps` instead
- Simplest inner loop of any quant — no scale decode, no mins
- ~8 zmm → room for 4× accumulators easily
- For batch tiling: SHARED = weight d[8] fp16→fp32 conversion; PER-COLUMN = activation d[8] × weight d[8] element-wise

### Clang/ClangCL AVX-512 Intrinsic Compatibility
- **Problem**: MSVC doesn't enforce target feature requirements on intrinsics — any intrinsic works if the header is included. Clang enforces `always_inline` target matching.
- **Solution**: File-level `#pragma clang attribute push/pop` guarded by `#ifdef __clang__`
- **All 4 b612 source files** now use this pattern consistently:
  - `vec-b612.cpp`: `avx512f,avx512vnni,avx512vl,avx512bf16`
  - `ggml-cpu-b612.c`: `avx512f,avx512vnni,avx512vl,avx512bf16`
  - `quants-b612.c`: `avx512f,avx512vnni,avx512vl,avx512bf16`
  - `ggml-cpu-repack.c`: `avx512f,avx512vnni,avx512vl` (no bf16 intrinsics)
- Pragma placed after includes, pop at end of file; MSVC unaffected

### Build Environment
- Windows, Visual Studio (MSVC) and Clang 20.1.8 (ClangCL via `-TClangCL`)
- Build dir: `D:\llama.cpp\b612\build.Vulkan` (Vulkan backend) — for b612 main codebase
- Build dir: `D:\llama.cpp\b612.dc_080625\build\` — for dc_080625 codebase (ClangCL build)
- Build command (MSVC): `cmake --build build.Vulkan --config RelWithDebInfo --target <name>`
- Build command (Clang): `cmake .. -TClangCL && cmake --build . --config RelWithDebInfo --target minslminfer`
- ggml-cpu target builds both ggml-base.dll and ggml-cpu.dll

### stop_char Mechanism
- `model_params.stop_char` (char, default 0) in llm-infer.h line 68
- Already checked in `llm_infer_multiturn()` at line 695 and `llm_inference()` at line 469

### Edit Gotcha Discovered
- When using the edit tool to replace text at the boundary between two functions, the function signature of the NEXT function can get accidentally removed if the `old_str` capture extends too far or the `new_str` doesn't include it. This happened with Q4_K_dc — the `void\nxx_vec_dot_q4_k_q8_k_x8_dc (` header was lost. Always verify the subsequent function's signature after boundary edits.
</technical_details>

<important_files>
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`
  - Main multi-turn driver created in this session
  - OG script parser, chat template detection (3-level), main loop
  - Key helpers: `to_lower()`, `is_meta_command()`, `apply_model_chat_template()`

- `D:\llama.cpp\b612\examples\llm-infer\teacher\teacher.cpp`
  - Teacher/student comparison tool
  - Config JSON parsing, `run_model_pass()`, `parse_model_answer()`, `run_teacher_reasoning()`

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`
  - Core llm-infer library — all API calls updated
  - `llm_get_chat_template()` at ~line 142

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`
  - Public API header — `model_params` struct (lines 37-69), stop_char at line 68

- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`
  - Build config for all targets including `minslminfer-multi-og` and `teacher`

- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`
  - Header for repack functions — declarations for all `_cp` variants
  - `block_q4_0_repack` struct definition at lines 14-17
  - `block_q8_0_repack` struct definition at lines 21-24
  - K-quant repack typedefs at lines 28-32
  - All vec_dot declarations at lines 55-174

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`
  - **PRIMARY FILE** — all b612 repack implementations
  - `make_q4_0_repack_quant` at ~line 26
  - `make_q2_k_repack_quant` at ~line 122
  - `make_q3_k_repack_quant` at ~line 221
  - `make_q4_k_repack_quant` at ~line 380
  - `make_q6_k_repack_quant` at ~line 480
  - Q4_0 analysis comment block at ~line 865
  - `xx_vec_dot_q4_0_q8_0_x8` (CURRENT, not yet batch-tiled) at ~line 1015+
  - `xx_vec_dot_q2_k_q8_k_x8_dc` (original Q2_K) and batch-tiled `xx_vec_dot_q2_k_q8_k_x8` nearby
  - `xx_vec_dot_q3_k_q8_k_x8_dc` and batch-tiled `xx_vec_dot_q3_k_q8_k_x8`
  - `xx_vec_dot_q4_k_q8_k_x8_dc` and batch-tiled `xx_vec_dot_q4_k_q8_k_x8`
  - `xx_vec_dot_q6_k_q8_k_x8_dc` and batch-tiled `xx_vec_dot_q6_k_q8_k_x8`
  - `xx_vec_dot_q8_0_q8_0_x8` (not modified)
  - Clang pragma push near top, pop at end of file

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\vec-b612.cpp`
  - BF16 vec_dot and sumsq functions
  - File-level Clang pragma for avx512f,avx512vnni,avx512vl,avx512bf16
  - Pop at end of file (~line 2947)

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`
  - Contains `ggml_fp32_to_bf16_row_cpu` and many other core functions
  - File-level Clang pragma push at ~line 67, pop at end (~line 6774)

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\quants-b612.c`
  - Many vec_dot functions using dpbusd intrinsics
  - File-level Clang pragma push/pop for avx512f,avx512vnni,avx512vl,avx512bf16

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\arch\x86\repack.cpp`
  - Upstream repack GEMM implementations (analyzed, not modified)
  - `ggml_gemm_q4_0_8x8_q8_0()` at line 1165 (~745 lines)
  - `ggml_gemm_q4_K_8x8_q8_K()` at line 1913
  - `ggml_gemm_q2_K_8x8_q8_K()` at line 3367

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\repack.h`
  - Upstream repack struct definitions (C++ templates)
  - `block_q4_0x8 = block<4, 8>` at line 35
  - `block_q4_Kx8` at line 39, `block_q2_Kx8` at line 47
</important_files>

<next_steps>
Remaining work:
- **Implement `xx_vec_dot_q4_0_q8_0_x8` batch-tiled version** ← IMMEDIATE, user explicitly requested this

Implementation plan for Q4_0 batch tiling:
1. Rename current `xx_vec_dot_q4_0_q8_0_x8` to `xx_vec_dot_q4_0_q8_0_x8_dc`
2. Create new batch-tiled `xx_vec_dot_q4_0_q8_0_x8` with `_cp` linker export, following the established pattern
3. Key Q4_0 specifics to handle:
   - Scale is a VECTOR `d[8]` (8 × fp16), not a scalar — SHARED part is weight d[8] conversion to fp32, PER-COLUMN is element-wise multiply with activation d[8]
   - Clang workaround: use `_mm_loadu_si128 + _mm256_cvtph_ps` instead of `_mm_loadu_ph` (must appear in both MSVC and Clang paths, or use `#ifdef __clang__` conditional as in original)
   - Bias uses constant `offset = _mm512_set1_epi8(8)` (not a scale-dependent bias)
   - Inner loop: 4 iterations per super-block, each does `load 256-bit q4 → split nibbles → 512-bit + load 512-bit q8 → dpbusd sumi + dpbusd bias`
4. Add `xx_vec_dot_q4_0_q8_0_x8_cp` declaration to `ggml-cpu-repack.h`
5. Build and verify with `cmake --build . --config RelWithDebInfo --target ggml-cpu`

The analysis comment block for Q4_0 is already inserted in ggml-cpu-repack.c starting around line 865. The existing function starts around line 1015 (after the comment block). The function to edit ends around line 998 (from the original line numbers — may have shifted due to earlier insertions).
</next_steps>