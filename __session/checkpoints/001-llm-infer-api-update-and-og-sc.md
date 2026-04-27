<overview>
The user has a `llm-infer` example directory in `D:\llama.cpp\b612` that was copied from an older codebase (`b612.dc_080625`) with outdated APIs. The work involved: (1) updating the APIs in llm-infer to compile with the current b612 codebase, (2) creating a new multi-turn driver (`minslm-multi-og.cpp`) that uses the SYSTEM/PROMPT script format from `cpf_gem4mm.cpp` so both programs can share the same prompt input files, and (3) fixing a bug where the new driver produces corrupted output because user messages lack chat template wrapping.
</overview>

<history>
1. User asked to update all APIs in `examples/llm-infer` to build successfully in b612
   - Explored both codebases (b612 and b612.dc_080625) to identify API differences
   - Found 4 categories of breaking changes:
     - `llama_kv_self_clear(ctx)` → `llama_memory_clear(llama_get_memory(ctx), true)` (5 occurrences)
     - `llama_kv_self_seq_rm(ctx, ...)` → `llama_memory_seq_rm(llama_get_memory(ctx), ...)` (3 occurrences)
     - `llama_print_tensor_op_perf()` → stubbed (removed from b612 API)
     - `llama_set_tensor_repack_mode()` + `GGML_TENSOR_REPACK_MODE_*` → removed (2 switch blocks deleted)
   - Added `add_subdirectory(llm-infer)` to `examples/CMakeLists.txt`
   - Successfully built all 10 targets (slm-infer, llm-infer, llm-infer-static, buildrag-dll, ragBuilder-dll, server-dll, minslminfer-dll, minslminfer, minslminfer-multi, testme)

2. User asked to create `minslminfer-multi-og.cpp` based on `minslm-multi.cpp` using the SYSTEM/PROMPT script format from `cpf_gem4mm.cpp`
   - Studied `cpf_gem4mm.cpp` script parser (SYSTEM/PROMPT blocks, T:/I:/A: prefixes, /context, /rewind N, quit() meta commands)
   - Studied `sample_script_textonly.txt` format
   - Created new file with OG script parser, /context stats display, /rewind support, interactive stdin fallback, optional template= CLI arg
   - Added CMakeLists.txt target `minslminfer-multi-og`
   - Built successfully

3. User asked to rename `minslminfer-multi-og.cpp` to `minslm-multi-og.cpp`
   - Renamed file and updated CMakeLists.txt source path
   - Rebuilt successfully (binary still named `minslminfer-multi-og.exe` per CMake target name)

4. User reported corrupted output — model echoes prompt or produces empty responses
   - Root cause identified: when no template file is provided, user text is sent raw to the model without chat template wrapping (no `<|user|>`, `<|assistant|>` tokens etc.)
   - In the current code path with no template file: `params.custom_template_prompt = script.system_prompt` (raw text like "You are a helpful AI assistant.") and `params.prompt = user_text` (raw text like "Where is Paris?")
   - The model sees concatenated raw text with no structure, so it doesn't know to generate a response
   - **This bug is NOT YET FIXED** — compaction occurred during diagnosis
</history>

<work_done>
Files created:
- `examples/llm-infer/minslm/minslm-multi-og.cpp`: New multi-turn driver using OG script format

Files modified:
- `examples/CMakeLists.txt`: Added `add_subdirectory(llm-infer)` after line with `add_subdirectory(xbapp)`
- `examples/llm-infer/CMakeLists.txt`: Added `minslminfer-multi-og` target (source: `minslm/minslm-multi-og.cpp`)
- `examples/llm-infer/dll/llm-infer.cpp`: Updated all deprecated/removed API calls

Work completed:
- [x] Update llm-infer APIs for current b612 codebase
- [x] Build all llm-infer targets successfully
- [x] Create minslm-multi-og.cpp with OG script format parser
- [x] Add CMake target and verify build
- [x] Rename file from minslminfer-multi-og.cpp to minslm-multi-og.cpp
- [ ] **FIX: Output is corrupted because user messages have no chat template wrapping**

Current state:
- All targets compile and link successfully
- The new `minslm-multi-og.cpp` runs but produces wrong output (echoes prompt, empty responses)
- The bug is that raw user text needs to be wrapped in model-specific chat template tokens before being passed to `llm_infer_multiturn()`
</work_done>

<technical_details>
### API Migration (b612.dc_080625 → b612)
- KV cache API renamed: `llama_kv_self_clear(ctx)` → `llama_memory_clear(llama_get_memory(ctx), true)` and `llama_kv_self_seq_rm(ctx, seq, p0, p1)` → `llama_memory_seq_rm(llama_get_memory(ctx), seq, p0, p1)`
- `llama_memory_t` is obtained via `llama_get_memory(ctx)` (typedef for `llama_memory_i *`)
- `llama_print_tensor_op_perf()` and `llama_set_tensor_repack_mode()` were completely removed from b612 — no replacement exists
- All other APIs (sampler, tokenizer, batch, context, model) remain unchanged

### How llm_multiturn_begin() works
- Takes `params.custom_template_prompt` and looks for `{message}` placeholder
- If `{message}` found: splits at last user tag to extract system prefix (before user tag)
- If NO `{message}`: treats entire `custom_template_prompt` as the system prefix
- Decodes system prefix into KV cache as the starting point
- For PFC mode: uses shared prefix from saved state instead

### How llm_infer_multiturn() works
- Takes `params.prompt` directly, tokenizes it, decodes into KV cache (appending after existing)
- Generates tokens until EOS or max_len
- KV cache accumulates across calls (multi-turn)

### The Bug (unfixed)
- When no template file is given, `minslm-multi-og.cpp` sets:
  - `params.custom_template_prompt = "You are a helpful AI assistant."` (raw system text)
  - `params.prompt = "Where is Paris?"` (raw user text)
- The model sees raw concatenated text without chat structure tokens
- It doesn't know the boundary between system/user/assistant roles
- Fix needed: wrap system text and user messages in a default chat template (e.g., ChatML: `<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n`)
- Challenge: different models use different templates (Phi-3 uses `<|system|>/<|user|>/<|end|>/<|assistant|>`, ChatML uses `<|im_start|>/<|im_end|>`, Gemma uses `<|turn>system/user/model`, Llama-3 uses `<|start_header_id|>` etc.)
- `parse_special = true` is already set, so special tokens in templates will be parsed correctly

### Build Environment
- Windows, Visual Studio (MSVC), CMake
- Build dir: `D:\llama.cpp\b612\build.Vulkan` (Vulkan backend)
- Build command: `cmake --build build.Vulkan --config RelWithDebInfo --target <name>`
- All llm-infer targets build under `examples/llm-infer/` in the CMake tree

### Script Format (OG / cpf_gem4mm)
```
SYSTEM
<system prompt text>

PROMPT
T: <text>
I: <image_path>  (ignored in text-only)
A: <audio_path>  (ignored in text-only)

PROMPT
/context | /rewind [N] | quit()
```
- ReadBlockLines() reads until next SYSTEM/PROMPT keyword or EOF
- ParsePromptBlock() extracts T:/I:/A: prefixed content and meta commands
</technical_details>

<important_files>
- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi-og.cpp`
  - The new file created for this task — multi-turn driver with OG script format
  - **Has the unfixed bug**: lines ~490-498 where prompt is formatted without template wrapping
  - Key sections: Script parser (~lines 170-250), /context display (~340-373), main loop (~460-550)
  
- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.cpp`
  - Core llm-infer library with all the API calls that were updated
  - `llm_multiturn_begin()` ~line 527: how system prefix is extracted and decoded
  - `llm_infer_multiturn()` ~line 597: how each turn's prompt is processed
  - `llm_print_tensor_op_perf_stats()` ~line 135: stubbed function

- `D:\llama.cpp\b612\examples\llm-infer\dll\llm-infer.h`
  - Public API header — model_params struct, function declarations
  - `custom_template_prompt`, `turn_template`, `stop_char` fields are key for multi-turn

- `D:\llama.cpp\b612\examples\llm-infer\CMakeLists.txt`
  - Build config for all llm-infer targets including the new `minslminfer-multi-og`

- `D:\llama.cpp\b612\examples\CMakeLists.txt`
  - Parent CMakeLists — has `add_subdirectory(llm-infer)` added at line 37

- `D:\llama.cpp\b612\examples\llm-infer\minslm\minslm-multi.cpp`
  - Original multi-turn driver (old CUSTOM_TEMPLATE_PROMPT format) — reference for how templates work
  - Lines 380-393: how turn template wraps user messages (the pattern we need to replicate)

- `D:\llama.cpp\b612_Onnx\examples\cpp\Gemma-4\cpf_gem4mm.cpp`
  - Reference for the OG script format parser (lines 130-224)
  - Shows how /context, /rewind, quit() are handled (lines 848-923)

- `D:\llama.cpp\b612_Onnx\examples\cpp\Gemma-4\prompts\sample_script_textonly.txt`
  - The test script file both programs should be able to use
  - Has SYSTEM block + 11 PROMPT blocks including /context, /rewind commands

- `D:\llama.cpp\b612\include\llama.h`
  - Current b612 API — `llama_memory_clear/seq_rm` at lines 716-729, no tensor repack APIs
</important_files>

<next_steps>
Remaining work:
- **Fix the corrupted output bug in minslm-multi-og.cpp** — user messages need chat template wrapping

Immediate next steps:
1. Add a default chat template when no `template=FILE` is provided. Options:
   - Use ChatML as default: system prefix = `<|im_start|>system\n{text}<|im_end|>\n`, turn template = `<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n`
   - Or ask the user which model/template they're using
2. Modify the code around lines 480-498 in minslm-multi-og.cpp where `params.custom_template_prompt` and `params.prompt` are set:
   - Wrap `script.system_prompt` in system template tokens for `custom_template_prompt`
   - Set `custom_turn_template` to a default user/assistant template with `{message}` placeholder
   - Apply the turn template to each user message before passing to `llm_infer_multiturn()`
3. Rebuild and test with `sample_script_textonly.txt`

Open questions:
- Which model is the user running? (affects which chat template tokens to use as default)
- Should the default template be ChatML, Phi-3, or something else?
- Should we require the template= argument and error out without it, or use a best-guess default?
</next_steps>