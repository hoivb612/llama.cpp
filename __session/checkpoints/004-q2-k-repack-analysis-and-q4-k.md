<overview>
The user has been building tools in `D:\llama.cpp\b612\examples\llm-infer\` and analyzing quantized matrix multiplication implementations across two codebases (`b612` custom and upstream `b612.dc_080625`). Work spans: (1) updating APIs from an older codebase, (2) creating multi-turn conversation drivers with auto chat template detection, (3) building a teacher/student model comparison tool, and (4) analyzing and optimizing Q4_K/Q2_K repack vec_dot implementations for batch processing. The approach has been incremental: build, test, fix based on user feedback, with recent focus shifting to low-level SIMD optimization analysis and implementation.
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
   - **b612**: Vec-dot approach, 1 row × 1 col, AVX-512 VNNI dpbusd, in-place repack
   - **Upstream**: True GEMM, 16×16 tiles, maddubs (not VNNI), 8-interleaved blocks
   - Concluded: b612 wins at batch=1 (token gen), upstream wins at batch≥4 (prefill)

9. User asked to build `xx_vec_dot_q4_k_q8_k_x8_cp()` — batch-optimized vec_dot
   - Designed 4-column tiling: load weights once, dpbusd against 4 activation columns
   - Implemented full function with Phase 1 (4-col batch) and Phase 2 (1-col remainder)
   - Added declaration to header, implementation to .c file
   - No repack format change needed — same `block_q4_K_repack`
   - User confirmed: only vec_dot changed, repack unchanged (by design)

10. User asked to do the same analysis for Q2_K
    - Read `make_q2_k_repack_quant()` (lines 115-212): unpacks 2-bit→8-bit, interleaves 16 lanes of 4 bytes, repacks 4 sub-blocks into single 512-bit store
    - Read `xx_vec_dot_q2_k_q8_k_x8()` (lines 994-1099): similar structure to Q4_K but with 2-bit unpacking (AND with 3, shift right 2), 16 scales (not 8), 256-bit mins accumulator
    - Started reading `ggml_gemm_q2_K_8x8_q8_K()` (line 3367+): massive function, 16×16 tiles with `block_q2_Kx8`, enormous shuffle/permute overhead
    - **Analysis was in progress when compaction occurred** — read through line ~3850 of the upstream GEMM
</history>

<work_done>
Files created:
- `examples/llm-infer/minslm/minslm-multi-og.cpp`: Multi-turn driver with OG script format, auto chat template detection (3-level), case-insensitive meta commands, stop_char support
- `examples/llm-infer/teacher/teacher.cpp`: Teacher/student comparison tool — config JSON, 3-pass execution, JSON answer parsing, mismatch detection, teacher reasoning
- `examples/llm-infer/teacher/mara_system_prompt_v2.txt`: Improved system prompt with 6 explicit decision rules

Files modified:
- `D:\llama.cpp\b612\examples\CMakeLists.txt`: Added `add_subdirectory(llm-infer)` after `add_subdirectory(xbapp)`
- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`: Added `minslminfer-multi-og` target (line ~124-135) and `teacher` target (after line 135)
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`: Updated deprecated API calls; added `llm_get_chat_template()` implementation
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`: Added `llm_get_chat_template()` declaration after line 77
- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`: Added `xx_vec_dot_q4_k_q8_k_x8_cp()` declaration (lines 98-110)
- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`: Added `xx_vec_dot_q4_k_q8_k_x8_cp()` implementation (~280 lines, inserted after line 1582)

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
- [x] Implement `xx_vec_dot_q4_k_q8_k_x8_cp()` — batch-tiled Q4_K vec_dot
- [ ] **Compare Q2_K repack implementations (b612 vs upstream) — IN PROGRESS**
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

### Q4_K Repack Analysis
**b612 (`xx_vec_dot_q4_k_q8_k_x8`)**:
- `block_q4_K_repack` is `typedef block_q4_K` — same struct, in-place repack
- Repack rearranges nibbles: unpack 4→8 bit, interleave 4-byte lanes, repack as 64-value groups
- Vec-dot: 1 row × 1 col, AVX-512 VNNI `dpbusd`, 4 iterations per super-block
- Scale decode: 12-byte packed → 8 × 6-bit scales/mins via bitmask (kmask1/2/4)
- Mins: 128-bit path (`_mm_madd_epi16` with pre-folded 8 bsums)
- Reduction: 512→256→128→scalar via add+hadd

**Upstream (`ggml_gemm_q4_K_8x8_q8_K`)**:
- `block_q4_Kx8`: 8 blocks interleaved (d[8], dmin[8], scales[96], qs[1024])
- True GEMM: 16 rows × 16 cols per tile
- Uses `maddubs_epi16` (NOT dpbusd) — requires extra `madd_epi16` step
- Massive shuffle/blend/permute overhead, ~60+ registers
- Weight data reuse: each load serves 4-16 rows

**Verdict**: b612 wins batch=1 (simpler, VNNI), upstream wins batch≥4 (data reuse)

### xx_vec_dot_q4_k_q8_k_x8_cp Design
- 4-column tiling: load weight data ONCE per super-block, dpbusd against 4 activation columns
- Phase 1: groups of 4 columns (ncols & ~3), Phase 2: remainder 1-at-a-time
- Register budget: 4×acc + 4×sumi + 1×scale + 1×q4v + 1×m4 ≈ 12 zmm (of 32 available)
- No repack format change — same `block_q4_K_repack`
- Reduces weight memory traffic by ~4× for batch ≥ 4
- Scale extraction shared across 4 columns (amortized 4×)
- MSVC linker export pragma included

### Q2_K Repack Analysis (PARTIAL)
**b612 (`xx_vec_dot_q2_k_q8_k_x8`)**:
- `block_q2_K_repack` is `typedef block_q2_K` — same struct, in-place repack
- Repack: unpack 2-bit→8-bit (AND with 3, shift right 2 in loop), interleave 16 lanes of 4 bytes, repack 4 sub-blocks by OR+shift into single 512-bit store
- Vec-dot: q2bits loaded as single 512-bit, extracted 4 ways via AND+shift in 4 iterations
- **16 scales** (not 8 like Q4_K) — `scales[16]` packed as 4-bit pairs (scale|min per byte)
- Scale decode: `_mm_and_si128(mins_and_scales, m4)` for scales, `_mm_srli_epi16 + and` for mins — much simpler than Q4_K's 6-bit decode
- Scale vector: `_mm512_cvtepi8_epi32(scales8)` — 16 → 16 int32 (no replication needed)
- Mins: **256-bit** path (`_mm256_madd_epi16` with 16 bsums, `_mm256_fmadd_ps`)
- Reduction: 512→256, subtract mins at 256-bit level, then 256→128→scalar via add+hadd

**Upstream (`ggml_gemm_q2_K_8x8_q8_K`)** — read through ~line 3850 of ~4356:
- `block_q2_Kx8`: 8 blocks interleaved, `scales[128]` (8×16 bytes), `qs[512]` (8×64 bytes)
- Processes 16 rows × 16 cols per iteration (same tiling as Q4_K GEMM)
- Q2_K has 8 sub-blocks per super-block (QK_K/128=2 outer × 4 shift positions), each producing 16 values
- Massive unrolling: 32 `_mm512i` RHS vectors (8 sub-blocks × 2 column groups × 2 halves)
- Each further split into sp1 (perm 136) and sp2 (perm 221) shuffle patterns — **64 shuffled RHS vectors**
- Scale extraction: 4×16-byte loads per column group, shuffle+mask into per-sub-block scale vectors
- 8 scale pairs × 2 shuffle patterns = 16 scale vectors
- LHS: 32 loads per `rp` iteration (8 sub-blocks × 4 halves), each duplicated to 512-bit
- Bsums: 4×128-bit loads, expanded to 512-bit for mins accumulation
- Uses `_mm512_maddubs_epi16` (not dpbusd), same as Q4_K GEMM
- **Still reading**: activation multiply-accumulate and output store sections not yet analyzed

### Build Environment
- Windows, Visual Studio (MSVC), CMake
- Build dir: `D:\llama.cpp\b612\build.Vulkan` (Vulkan backend) — for b612 main codebase
- No build directory exists in `D:\llama.cpp\b612.dc_080625\` — cannot compile-test there
- Build command: `cmake --build build.Vulkan --config RelWithDebInfo --target <name>`

### stop_char Mechanism
- `model_params.stop_char` (char, default 0) in llm-infer.h line 68
- Already checked in `llm_infer_multiturn()` at line 695 and `llm_inference()` at line 469
</technical_details>

<important_files>
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`
  - Main multi-turn driver created in this session
  - OG script parser (~338-419), chat template detection (~484-692), main loop (~949-1083)
  - Key helpers: `to_lower()` (~311), `is_meta_command()` (~318), `apply_model_chat_template()` (~640)

- `D:\llama.cpp\b612\examples\llm-infer\teacher\teacher.cpp`
  - Teacher/student comparison tool
  - Config JSON parsing in main() (~770-820)
  - `run_model_pass()` (~475-615), `parse_model_answer()` (~390-430)
  - `run_teacher_reasoning()` (~630-725) with original system prompt context

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`
  - Core llm-infer library — all API calls updated
  - `llm_get_chat_template()` at ~line 142

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`
  - Public API header — `model_params` struct (lines 37-69), stop_char at line 68

- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`
  - Build config for all targets including `minslminfer-multi-og` and `teacher`

- `D:\llama.cpp\b612.dc_080625\ggml\include\ggml-cpu-repack.h`
  - Header for repack functions — added `xx_vec_dot_q4_k_q8_k_x8_cp()` declaration at lines 98-110
  - `block_q4_K_repack = typedef block_q4_K` (line 30), `block_q2_K_repack = typedef block_q2_K` (line 28)

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`
  - All b612 repack implementations
  - `make_q2_k_repack_quant()` at line 116 — Q2_K in-place repack
  - `make_q4_k_repack_quant()` at line 374 — Q4_K in-place repack
  - `xx_vec_dot_q2_k_q8_k_x8()` at line 994 — Q2_K vec_dot (analysis target)
  - `xx_vec_dot_q4_k_q8_k_x8()` at line 1263 — Q4_K vec_dot (original)
  - `xx_vec_dot_q4_k_q8_k_x8_cp()` at line ~1602 — NEW batch-tiled Q4_K vec_dot (added this session)

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\arch\x86\repack.cpp`
  - Upstream repack GEMM implementations
  - `ggml_gemm_q4_K_8x8_q8_K()` at line 1913 — Q4_K GEMM (analyzed)
  - `ggml_gemm_q2_K_8x8_q8_K()` at line 3367 — Q2_K GEMM (analysis in progress)
  - File has at least 4356 lines total

- `D:\llama.cpp\b612.dc_080625\ggml\src\ggml-cpu\repack.h`
  - Upstream block structs: `block_q4_Kx8` (line 39), `block_q2_Kx8` (line 47)

- `D:\llama.cpp\b612\examples\llm-infer\teacher\mara_system_prompt_v2.txt`
  - Improved system prompt with 6 decision rules
</important_files>

<next_steps>
Remaining work:
- **Complete Q2_K repack comparison analysis** — user wants the same compare/contrast as Q4_K

Analysis in progress — key observations so far:

**b612 `xx_vec_dot_q2_k_q8_k_x8`**:
- Vec-dot approach: 1 row × 1 col per inner iteration
- Uses AVX-512 VNNI `dpbusd` — loads single 512-bit q2 block, extracts 4 sub-blocks via AND+shift
- 16 scales (4-bit packed as scale|min per byte) — simpler decode than Q4_K's 6-bit
- 256-bit mins path (16 bsums vs 8 in Q4_K)
- Very compact: 4 dpbusd per super-block

**Upstream `ggml_gemm_q2_K_8x8_q8_K`** (partially read):
- True GEMM: 16×16 tiles, `block_q2_Kx8` (8 interleaved blocks)
- Q2_K has MORE sub-blocks than Q4_K (8 vs 4 per super-block) — more shift+mask operations
- Enormous code: 64 shuffled RHS vectors per sub-block iteration, 32 LHS loads per rp
- Uses `maddubs_epi16` (not VNNI dpbusd)
- Read through line ~3850 — still need to read the multiply-accumulate core and output store

Immediate next steps:
1. Continue reading upstream Q2_K GEMM from ~line 3850 to end (multiply-accumulate, scale application, output store)
2. Count instruction throughput per output element for both Q2_K implementations
3. Compare memory access patterns and register pressure
4. Provide final Q2_K comparative analysis answering "which is faster and why"
5. Potentially implement `xx_vec_dot_q2_k_q8_k_x8_cp()` batch-tiled version if user requests
</next_steps>