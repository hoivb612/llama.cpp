<overview>
The user is working on fixing Gemma-4 model repack buffer corruption in the `b612.dc_041126` codebase, where in-place tensor repacking corrupts shared weight data. The approach evolved from diagnosing the root cause (data pointer aliasing between MUL_MAT and non-MUL_MAT ops like GET_ROWS) to implementing a graph-scan aliasing detection system that automatically marks shared tensors for buffer duplication. Additional work included removing superseded loader code, enabling early repack for all b612 modes, adding a graph build timer diagnostic, and guarding against GGML repack mode access violations.
</overview>

<history>
1. Continued diagnosing Gemma-4 repack buffer corruption (carried over from prior checkpoint)
   - Investigated `GGML_TENSOR_FLAG_DUP` — found it's NEVER set in dc_041126's `llama-model.cpp` because `ggml_set_duplicated()` is never called
   - Traced Gemma-4 tensor loading: `output` and `tok_embd` share the same tensor when `TENSOR_DUPLICATED` is used (lines 4631-4637 in llama-model.cpp)
   - Identified root cause: `tok_embd` used for GET_ROWS (embedding lookup) AND as `output` for MUL_MAT (logits). In-place repack corrupts the shared data.
   - Proposed graph-scan approach: detect aliasing by scanning compute graph, user approved

2. Implemented graph-scan aliasing detection (first version — too broad)
   - Added `ggml_repack_scan_aliased_data_pointers()` in `ggml-cpu-b612.c`
   - Pass 1: collected ALL data pointers from non-MUL_MAT op sources (too broad — hundreds of pointers)
   - Pass 2: checked MUL_MAT src0 against that set
   - Restored `GGML_TENSOR_FLAG_DUP` guard in multi-threaded `ggml_repack_tensor()` (was commented out as user's workaround)
   - Built successfully with Clang, user confirmed Gemma-4 worked (1 tensor duplicated, 216 MB)

3. Tightened the graph scan to intersection approach
   - User pointed out the protected set was too broad (captures nearly every materialized tensor)
   - Rewrote to intersection: Pass 1 collects repackable MUL_MAT src[0] pointers (small set ~200), Pass 2 checks non-MUL_MAT sources against that set
   - Added optimization: matched candidates are removed from array for early termination
   - Used `repack_candidate_t` struct to track both data pointer and tensor object
   - Built and verified with Clang

4. Analyzed and removed old DUP loader code in `llama-model-loader.cpp`
   - Lines 1054-1065: extra tensor slot reservation for XBOX mode (no longer needed)
   - Lines 1275-1295: `ggml_dup_tensor()` + `ggml_set_duplicated()` for TOKEN_EMBD (superseded by graph scan)
   - First commented out with `#if 0`, user tested and confirmed working, then fully removed

5. Enabled early repack for ALL b612 modes (not just XBCG)
   - Removed `if (g_tensor_repack_mode != GGML_TENSOR_REPACK_MODE_XBCG)` guard
   - Now XBOX and XBOX_SINGLE_THREAD modes also repack during `build_graph()` instead of lazily during MUL_MAT
   - User noted tradeoff: early repack is single-threaded (slower repack) but eliminates first-inference penalty

6. Added graph build timer diagnostic
   - Uncommented existing `t_start_us` timer around `model.build_graph(gparams)` in `process_ubatch()` (llama-context.cpp line 1202-1206)
   - User observed 3 graph builds per inference (expected: 2 prompt ubatches + 1 token gen)
   - Graph build times ~0.3ms each, confirming repack NOT happening during timed inferences

7. User added GGML mode guard (their own edit, verified by me)
   - Added `g_tensor_repack_mode == GGML_TENSOR_REPACK_MODE_GGML` to early return in `ggml_cpu_repack_tensor_callgraph()`
   - Prevents access violation when running upstream GGML repack path

8. Analyzed TTFT timing and repack location
   - First inference TTFT (1791ms) ≈ second inference (1772ms) — no penalty!
   - Discovered repack happens during `llama_context::llama_context()` → `sched_reserve()` → `graph_reserve()` → `build_graph()` — well before any user-visible inference
   - Call stack: `llama_init_from_model` → constructor → `sched_reserve` → `graph_reserve` → `build_graph` → `llama_repack_tensor_callgraph`
   - 206 tensors repacked in 1.10 sec during initialization, 0 inline repacks during inference
</history>

<work_done>
Files modified in b612.dc_041126:
- `ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`: Added `ggml_repack_scan_aliased_data_pointers()` with intersection-based graph scan; modified `ggml_cpu_repack_tensor_callgraph()` to run aliasing scan + early repack for all b612 modes; user added GGML mode guard
- `ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`: Restored `GGML_TENSOR_FLAG_DUP` guard in multi-threaded `ggml_repack_tensor()` (was commented out as user's "always malloc" workaround)
- `src\llama-model-loader.cpp`: Removed two `#ifdef GGML_B612` blocks (lines 1054-1065 and 1275-1295) — old DUP tensor logic superseded by graph scan
- `src\llama-context.cpp`: Uncommented graph build timer around `model.build_graph()` in `process_ubatch()` (diagnostic)

Work completed:
- [x] Diagnose Gemma-4 repack buffer corruption root cause
- [x] Implement graph-scan aliasing detection (intersection approach)
- [x] Restore DUP guard in multi-threaded repack path
- [x] Remove superseded loader DUP code
- [x] Enable early repack for all b612 modes
- [x] Add GGML mode guard (user edit, verified)
- [x] Add graph build timer diagnostic
- [x] Analyze TTFT timing — confirmed repack happens at context init, not during inference

Current state: Everything builds and works. Gemma-4 runs correctly with repack-xbox mode. Only 1 tensor (tok_embd/output, 216 MB) is duplicated. TTFT is consistent between first and subsequent inferences.
</work_done>

<technical_details>
### Graph-Scan Aliasing Detection
- `ggml_repack_scan_aliased_data_pointers()` uses a two-pass intersection approach:
  - Pass 1: Collect data pointers from repackable MUL_MAT src[0] tensors (Q4_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K) — small set, ~200 entries max
  - Pass 2: Scan non-MUL_MAT ops, check if any src tensor's data pointer matches a candidate → set `GGML_TENSOR_FLAG_DUP` on the candidate
  - Matched candidates are removed from the array for O(1) early termination
  - Uses `repack_candidate_t` struct (data pointer + tensor pointer) on stack, max 512 entries

### Why Gemma-4 Corrupted
- `tok_embd` and `output` share the same `ggml_tensor *` (and thus same `data` pointer) via `TENSOR_DUPLICATED` in `create_tensor()`
- `tok_embd` is used in `build_inp_embd()` → GET_ROWS (expects original quant format)
- `output` is used in `build_lora_mm()` → MUL_MAT (triggers repack to _x8 format)
- In-place repack of MUL_MAT src0 corrupts the GET_ROWS data → garbage output

### Repack Timing Discovery
- Repack happens during `llama_init_from_model()` → `llama_context()` constructor → `sched_reserve()` → `graph_reserve()` → `build_graph()` → `llama_repack_tensor_callgraph()`
- This is BEFORE any user-facing inference, so TTFT timers don't capture repack cost
- All 206 tensors repacked in ~1.1 sec during init, 0 inline repacks during inference
- Graph build time during inference is ~0.3ms (just scan, no repack needed)

### Graph Builds Per Inference
- 3 builds per inference: prompt ubatch 1, prompt ubatch 2 (remainder), first generated token
- Graph reuse (`can_reuse`) handles subsequent tokens of same size
- `LLAMA_LOG_INFO` for graph build time goes to stderr

### Repack Modes and Guards
- `GGML_TENSOR_REPACK_MODE_NONE` (0): no repack
- `GGML_TENSOR_REPACK_MODE_GGML` (1): upstream repack (separate src/dst buffers) — must skip b612 scan to avoid AV
- `GGML_TENSOR_REPACK_MODE_XBOX` (2): b612 multi-threaded inline repack
- `GGML_TENSOR_REPACK_MODE_XBCG` (3): b612 callgraph early repack
- `GGML_TENSOR_REPACK_MODE_XBOX_SINGLE_THREAD` (4): b612 single-threaded inline repack
- Now all b612 modes (2,3,4) do early repack via callgraph; inline paths see already-repacked types and skip

### Old Loader DUP Mechanism (Removed)
- `llama-model-loader.cpp` had `GGML_B612` blocks that created a separate tensor object via `ggml_dup_tensor()` for TOKEN_EMBD when `TENSOR_DUPLICATED` + XBOX mode
- This allocated a separate buffer at load time, duplicating the weight data
- Superseded by graph-scan approach which detects aliasing automatically and defers duplication to repack time
- Removed both blocks (tensor slot reservation + dup_tensor creation)

### Build Environment
- Clang build: `D:\llama.cpp\b612.dc_041126\build.clang` (created this session)
- MSVC build: `D:\llama.cpp\b612.dc_041126\build` (pre-existing MSVC ICE in arch/x86/repack.cpp — unrelated)
- Clang config: `cmake .. -TClangCL -DGGML_AVX512=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VNNI=ON`
</technical_details>

<important_files>
- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ggml-cpu-b612.c`
  - PRIMARY: Contains `ggml_repack_scan_aliased_data_pointers()` (lines ~6734-6834) and modified `ggml_cpu_repack_tensor_callgraph()` (lines ~6836-6906)
  - Added `repack_candidate_t` typedef for the intersection scan
  - Changed callgraph function: removed XBCG-only guard, now runs aliasing scan + early repack for all b612 modes
  - User added GGML mode guard at line 6838-6839

- `D:\llama.cpp\b612.dc_041126\ggml\src\ggml-cpu\b612\ggml-cpu-repack.c`
  - Contains `ggml_repack_tensor()` (multi-threaded, line ~4492) and `ggml_repack_tensor_single_thread()` (line ~4370)
  - Restored `GGML_TENSOR_FLAG_DUP` guard at line ~4525 in multi-threaded path (was commented out)
  - Single-threaded path at line 4401 already had the DUP guard

- `D:\llama.cpp\b612.dc_041126\src\llama-model-loader.cpp`
  - Removed two `#ifdef GGML_B612` blocks: tensor slot reservation (~line 1054) and ggml_dup_tensor for TOKEN_EMBD (~line 1275)

- `D:\llama.cpp\b612.dc_041126\src\llama-context.cpp`
  - `process_ubatch()` at line 1171: uncommented graph build timer at lines 1202/1206
  - `graph_reserve()` at line ~2139: this is where the first `build_graph()` + repack happens during context init
  - `sched_reserve()` at line ~430: calls `graph_reserve()`

- `D:\llama.cpp\b612.dc_041126\src\llama-model.cpp`
  - `build_graph()` at line 9365: calls `llama_repack_tensor_callgraph(gf)`
  - Gemma-4 tensor loading at lines 4619-4710: shows TOKEN_EMBD/output weight tie pattern

- `D:\llama.cpp\b612.dc_041126\src\llama.cpp`
  - `llama_repack_tensor_callgraph()` at line 1301: dispatches to `ggml_cpu_repack_tensor_callgraph` via backend registry
</important_files>

<next_steps>
Completed work:
- Gemma-4 repack corruption is fully resolved in dc_041126
- All timing analysis is done — repack happens at context init, no first-inference penalty

Potential follow-up work (user hasn't requested yet):
- Port the graph-scan aliasing fix to `b612.dc_080625` codebase (user mentioned this earlier but deferred)
- Consider whether the graph build timer diagnostic should remain or be re-commented
- The early repack being single-threaded (vs multi-threaded XBOX inline) is a known tradeoff — user is aware

No immediate blockers.
</next_steps>