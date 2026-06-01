Original idea:

 .gguf file ──> llama.cpp loader ──> ggml_tensor weights + ggml_cgraph
                                               │
                                               ▼
                                   ggml-backend-sched
                                               │
                               ┌───────────────┼────────────────┐
                               ▼               ▼                ▼
                         ggml-cpu        ggml-ryzenai       ggml-vulkan
                                               │
                                               ▼
                               translate subgraph → ONNX ModelProto (in memory)
                                               │
                                               ▼
                               onnxruntime.dll + VitisAI EP
                                               │
                                               ▼
                                        compile → NPU .xclbin
                                               │
                                               ▼
                                        execute on XDNA NPU

What ONNX means here

ONNX is just the wire format VitisAI EP requires as input. It's the only API AMD exposes for "give me a graph to compile and run on the NPU." Same reason ggml-openvino builds ov::Model objects — OpenVINO Runtime won't accept anything else.

In our backend, ONNX is:

 - Constructed in memory as onnx::ModelProto (protobuf), per subgraph the scheduler hands us.
 - Never written to disk (except optionally for debugging / Vitis compile cache).
 - Never seen by the llama.cpp user.
 - The user has no idea ONNX is involved — just like ggml-openvino users don't know ov::Model exists.

What gets translated to ONNX, exactly

Not the whole model. The subgraph the scheduler decides to send to this backend per graph_compute() call:

 - Could be the entire transformer (best case — all ops supported → one big subgraph).
 - Could be individual MUL_MATs (worst case — only MUL_MAT supported → many tiny subgraphs).
 - Weights inside the subgraph are emitted as ONNX Initializer tensors (or QDQ-quantized Initializers for INT8 NPU path), sourced from the GGUF data already in tensor->data.

Caching

Because protobuf-build + Vitis compile is slow, you cache:

 - First run: build ONNX → VitisAI compiles → produces .xclbin → cached on disk under a hash of the subgraph.
 - Subsequent runs: cache hit, skip compile, load .xclbin directly. Same trick OpenVINO uses with its cache_dir.

So at steady state the ONNX construction happens once per model (during the first graph_compute call), not every token.

Comparison

┌───────────────────────────┬────────────────────────────┬───────────────────────────────┬─────────────────────────┐
│ Path                      │ What user gives            │ Internal IR                   │ NPU runtime             │
├───────────────────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────┤
│ AMD official LLM demo     │ ONNX model (pre-converted) │ —                             │ ORT + VitisAI EP        │
│ (OGA)                     │                            │                               │                         │
├───────────────────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────┤
│ ggml-ryzenai (what we'd   │ GGUF (unchanged llama.cpp  │ ONNX (built in-memory per     │ ORT + VitisAI EP        │
│ build)                    │ UX)                        │ subgraph)                     │                         │
├───────────────────────────┼────────────────────────────┼───────────────────────────────┼─────────────────────────┤
│ ggml-openvino (existing)  │ GGUF                       │ ov::Model (built in-memory)   │ OpenVINO Runtime + NPU  │
│                           │                            │                               │ plugin                  │
└───────────────────────────┴────────────────────────────┴───────────────────────────────┴─────────────────────────┘

Same pattern. The "convert back to ONNX" framing is accurate mechanically, but importantly: there's no offline conversion step the user runs. It's a runtime translation hidden inside the backend, exactly analogous to how the OpenVINO backend works today.

Why not bypass ONNX?

You'd want to if AMD exposed a lower API. They do — XRT + MLIR-AIE — but at that level you're writing AIE-tile kernels by hand (the months-long path). ONNX/VitisAI is the only "give me a graph, I'll figure it out" door AMD opens today.

===============================================================

What this drop actually is

~380 lines of glue code that hijacks ggml_compute_forward_mul_mat for one specific case:

 - Q4_0 weights × F32 activations → F32 output
 - ne3==ne2==1, ne0 ≥ 4096
 - Calls into ryzenai::qlinear_2<int16_t, int8_t, float, float> (~64 KB header)
 - qlinear_2 wraps XRT + pre-built .xclbin kernel binaries

The MLIR-AIE part is invisible to us — it was used by AMD to compile the .bin files. We just consume the binaries at runtime via XRT. That's a huge win for us.

What's actually in the drop

┌───────────────────────────────────┬────────────────────────────────────────────────────────────────────┬─────────┐
│ Layer                             │ Contents                                                           │ Size    │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────┤
│ ggml-ryzenai.{h,cpp}              │ The hijack/glue (Q4_0 unpack, transpose trick, singleton,          │ 14 KB   │
│                                   │ dispatch)                                                          │         │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────┤
│ ops/cpp/qlinear_2/qlinear_2.hpp   │ XRT wrapper class (initialize_weights_int4, execute)               │ 64 KB   │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────┤
│ dll/phx/qlinear_2/*.bin           │ Pre-compiled kernel binaries for Phoenix (XDNA 1)                  │ ~50     │
│                                   │                                                                    │ files   │
├───────────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────┤
│ dll/stx/qlinear_2/*.bin           │ Pre-compiled kernels for Strix Point (XDNA 2) — incl. mladf_*      │ ~70     │
│                                   │ variants                                                           │ files   │
└───────────────────────────────────┴────────────────────────────────────────────────────────────────────┴─────────┘

The stx kernels matter for you. They're named like mladf_4x4_a16fw4acc16f_<M>_<N>_<K>_<gs>.bin:

 - a16 = bfloat16 input
 - fw4 = uint4 weights
 - acc16f / acc32f = accumulator precision
 - shapes: 4096/11008/12288/32768 + group sizes 32/128

⚠️ These are Llama-2 shapes (dim 4096, FFN 11008/12288, vocab 32768). Other models with different dims will not have a matching kernel.

How the integration works in this drop

 // Inside ggml_compute_forward_mul_mat (old llama.cpp style):
 if (ggml_ryzenai_can_mul_mat(src0, src1, dst)) {
     ggml_ryzenai_mul_mat(src0, src1, dst, wdata, wsize); // → NPU
     return;
 }
 // ... otherwise fall through to existing CPU MUL_MAT

It's a CPU-path hijack, NOT a ggml-backend plugin. Predates ggml-backend.

Honest assessment — value vs current llama.cpp

What we can directly leverage (HIGH value)

 1. qlinear_2.hpp — the entire XRT+xclbin wrapper. No re-implementation needed.
 2. stx/*.bin kernels — if they run on Strix Halo (TBD), we skip MLIR-AIE entirely.
 3. unpack_row_q4_0 — Q4_0 block → int4 weights + fp16 scales + zeros=8.
 4. Transpose trick — B^T·A^T = C^T so qlinear_2 produces ggml-expected layout.
 5. Singleton pattern with per-weight executor cache — unordered_map<tensor_name, qlinear_2>. Weights uploaded to NPU once per tensor, then reused.
 6. Shape gate — ne0 ≥ 4096 is correct: NPU only wins on big mats.

What needs work (MEDIUM)

 - Modernize the integration: convert from CPU hijack → proper ggml-backend plugin (the right way today). Or, for a fast PoC, keep the hijack and just wire it into the current ggml_compute_forward_mul_mat path in ggml-cpu.c.
 - CMake: the drop's CMake is old llama.cpp era; need to re-do for current ggml_add_backend_library pattern.

What needs verification (UNKNOWN)

 1. Do stx/*.bin files run on Strix Halo (XDNA 2 with more tiles)? Strix Halo is a different SKU than Strix Point. Likely partially compatible but unverified. Need to test with xrt-smi on the 395.
 2. Does current Ryzen AI SDK still ship ryzenai*#COLON|**#COLON|*qlinear_2? Recent drops (1.3+) lean into VitisAI EP; qlinear_2 may be deprecated. We may need to either: - Pin the older transformers SDK that includes it, or
 - Re-link the same qlinear_2.hpp against current XRT (very plausible — it's mostly XRT + xclbin loading).
 3. mladf xclbin loader path — newer Strix kernels use mladf format; the qlinear_2 in this drop appears to support it (we'd need to verify by reading the .hpp).

Hard limitations to accept

 - MUL_MAT only, Q4_0 only, specific shapes only. For LLama-2-7B-Q4_0, this likely covers >90% of compute. For Gemma/Llama-3/Phi at different dims → not zero-cost; you'd need new xclbins.
 - No batching (ne2=ne3=1) — decode only, no parallel sequences. Prefill works because it walks tokens one-at-a-time through MUL_MAT.
 - F32 activations — wastes upload bandwidth (NPU consumes bf16, code converts on the fly).

Two paths forward, ranked by effort

Path A — "Get tokens out of NPU this week" (recommended start)

Minimal-touch retrofit into b612_052026:

 1. Copy ggml-ryzenai.{h,cpp} → ggml/src/ggml-cpu/ryzenai/
 2. Bring qlinear_2.hpp + the stx/*.bin files
 3. Resolve the XRT + Ryzen AI SDK dep (find which SDK release still has qlinear_2)
 4. Add one call site in ggml_compute_forward_mul_mat:
 if (ggml_ryzenai_can_mul_mat(src0, src1, dst)) {
     ggml_ryzenai_mul_mat(...);
     return;
 }
 5. Behind GGML_RYZENAI CMake flag.

Best-case result: Llama-2-7B-Q4_0 decode runs partially on NPU within days. Slow but interesting — and proves the hardware path works end-to-end on your 395.

Path B — "Proper backend, harder model coverage" (after Path A works)

 - Wrap qlinear_2 in a real ggml-ryzenai backend (backend_i / device_i / reg_i).
 - supports_op returns true only when ggml_ryzenai_can_mul_mat would.
 - Backend-sched offloads matching nodes; CPU handles the rest. Cleaner, allows future op additions.

Suggested first concrete steps

Before writing any code:

 1. On your Strix Halo 395 box: install AMD Ryzen AI Software (whichever version still ships ryzenai::qlinear_2 — likely 1.1 or 1.2) and confirm xrt-smi examine sees the NPU.
 2. Run AMD's own LLama-2 transformers demo from this repo to confirm the stx/*.bin files actually execute on Strix Halo, not just Strix Point. This is the make-or-break question. If they don't run, we need new xclbins → MLIR-AIE → multi-week effort.
 3. Once confirmed working, I can produce the surgical patch to b612_052026 (Path A) — it's genuinely small.

===========================================================

What qlinear_2 is

Name decoded: quantized linear, version 2. In ML terminology, "linear" = y = x · Wᵀ + b — a fully-connected layer, i.e. a matrix multiply with a quantized weight matrix. The "2" is just AMD's revision number (v1 was symmetric INT8; v2 added grouped INT4).

It's a C++ template class that wraps an XRT kernel invocation for one specific compute pattern:

 template <typename InT, typename WtT, typename AccT, typename OutT>
 class qlinear_2;

Instantiated in the drop as:

 qlinear_2<int16_t,    // InT  = bfloat16 activations (carried as int16)
           int8_t,     // WtT  = packed int4 weights
           float,      // AccT = float32 accumulator
           float>      // OutT = float32 output

Computes: Y[M,N] = X[M,K] · dequant(W_int4[K,N], scale[K/gs,N], zero[K/gs,N]) + bias[N]
where gs = group size (32 or 128 elements per scale/zero).

It's dequantize-on-the-fly inside the matmul — weights stay int4 in NPU memory; scales lift them back to bf16 just before the MAC.

Why it's interesting for an NPU

There are three layers to the answer.

1. The math layer — why grouped INT4 matmul is the right primitive for LLM decode

LLM decode is dominated by y = W · x where:

 - W is huge: tens to hundreds of MB per layer, used once per token.
 - x is tiny: a single vector, one row.

So decode is bandwidth-bound, not compute-bound. Whichever device gets to read W from the smallest, fastest memory wins. Compressing W to int4 instead of fp16 means:

 - 4× less bandwidth to feed the multiplier array.
 - 4× more weights fit in on-chip SRAM/scratch.
 - Tiny accuracy loss when grouped (one scale per 32–128 weights) — proven by Q4_0/Q4_K/AWQ/GPTQ.

Activation precision matters less because there's so little of it; keeping bf16 there preserves dynamic range for the accumulator. That's the a16fw4acc32f recipe.

2. The architecture layer — why this fits XDNA tiles specifically

XDNA NPU = a 2D mesh of AI Engine (AIE) tiles, each with:

 - A VLIW vector unit (the MAC array) — wants long sequential MAC streams.
 - A small local memory (a few hundred KB).
 - DMA from "memory tiles" that pull from DDR.

This architecture's worst enemy is random-access weight fetch from DDR and its best friend is streaming a packed blob through tiles. Grouped INT4 matmul matches both:

 - Weights are a contiguous packed buffer: int4 stream + scale stream + zero stream. Perfect DMA target.
 - The compute pattern is one giant outer loop of MAC + dequant. Perfect VLIW target — AMD compiles this loop once into an .xclbin per shape and that's the kernel binary.
 - Activations are tiny → fit in tile local memory.

So qlinear_2 is essentially the one shape of work an NPU is actually good at in 2024–2026 LLM inference.

3. The systems layer — why hide it behind a class

The qlinear_2 class encapsulates everything that's annoying about XRT:

 qlinear_2(a_dtype, w_dtype, c_dtype)        ← chooses xclbin
   .initialize_weights_int4(W, zeros, scales, bias, shape)
                                             ← uploads weights to NPU memory ONCE
   .execute(activations, shape, output)       ← per-token dispatch

Behind the scenes per .execute():

 1. Pack activation bf16 vector into an XRT BO (buffer object).
 2. Submit a kernel call (xrt::run) referencing the pre-uploaded weight BO and the kernel binary loaded from the matching .xclbin.
 3. Wait for completion via XRT fence.
 4. Memcpy output back to host pointer.

Critically: weights upload only once per tensor (kept in NPU-side BO for the lifetime of the singleton). Every subsequent token just streams a 4 KB-ish activation vector across PCIe/IF and gets back a ~16 KB output. That's why decode actually has a chance to be fast — the heavy weight traffic happens at model load, not per token.

Why this is the right wedge for us, specifically

If we tried to write a full ONNX/VitisAI backend, we'd be giving the NPU general-purpose responsibility — every op, every shape. We'd spend months wrestling with the compiler.

qlinear_2 instead gives us the one op the NPU is great at, with kernels already compiled, with weights already in the right layout. We don't compile anything. We don't write any AIE code. We just:

 1. Load a pre-built .xclbin for the shape.
 2. Upload weights once.
 3. Stream activations per token.

That's why this old drop is disproportionately valuable for fast time-to-working-demo. The hard parts (MLIR-AIE kernel authoring, weight layout, DMA scheduling) are baked into the .bin files. We're just plumbers.

Limits this approach inherits

 - One shape = one xclbin. If your model's MUL_MAT dimensions don't match an existing binary, you get nothing. Llama-2-7B-Q4_0 is covered; Llama-3/Gemma/Phi at different dims are not without new xclbins.
 - Q4_0 only (in this drop). Q4_K (the more accurate llama.cpp default) has a different per-block layout (mins instead of symmetric zeros, super-blocks). Would need either: dequantize Q4_K → re-quantize to the qlinear_2 format at load (cheap, one-time), or new xclbins that consume Q4_K natively (hard).
 - Single-batch only. Fine for chat decode; bad for serving multiple users.
 - MUL_MAT only. Attention softmax, RoPE, RMSNorm still run CPU. That's actually fine — those are cheap.

So: qlinear_2 is essentially "give the NPU a 4096×N int4 matrix once, then ask it Wx per token forever." For LLM decode on Strix Halo, that's exactly the deal the hardware wants to make.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Support Q8_0 with q_linear2

Surprisingly there's no plug-and-play "bf16 × int8" option in the shipped SDK. This makes Q8_0 a bigger lift than I implied earlier. Let me walk through what's actually available.

xclbin naming = <a_dtype><b_dtype><acc_dtype>.xclbin

What we have in xclbin/stx/:

┌────────────────────────────────┬────────────────┬────────────┬────────────┬─────────────────────────────┐
│ Filename                       │ a (activation) │ b (weight) │ c (output) │ Use                         │
├────────────────────────────────┼────────────────┼────────────┼────────────┼─────────────────────────────┤
│ gemm_4x4_a16fw4acc32f.xclbin   │ bf16           │ uint4      │ fp32       │ Q4_0 — current              │
├────────────────────────────────┼────────────────┼────────────┼────────────┼─────────────────────────────┤
│ gemm_4x4_a16w8acc64.xclbin     │ int16          │ int8       │ int64      │ Q8_0 candidate              │
├────────────────────────────────┼────────────────┼────────────┼────────────┼─────────────────────────────┤
│ gemm_4x4_a8w8acc32.xclbin      │ int8           │ int8       │ int32      │ Smaller activations         │
├────────────────────────────────┼────────────────┼────────────┼────────────┼─────────────────────────────┤
│ mladf_*                        │ bf16           │ uint4      │ varies     │ larger/fused matmul kernels │
└────────────────────────────────┴────────────────┴────────────┴────────────┴─────────────────────────────┘

Notice the gap: there is no a16f_w8 xclbin (bf16 activation × int8 weight). For Q4_0 we got a kernel that takes bf16 in and produces fp32 out — life was easy. For Q8_0 we have to either:

 - Path A: use a16_w8 (int16 activation × int8 weight → int64 output) — need to quantize F32 activations to int16 with a dynamic per-call scale, and dequantize int64 accumulator back to F32 using combined scales. Different math, different code.
 - Path B: use a8_w8 — even more aggressive (int8 activations). Activations are dynamic and small-dynamic-range so quality loss is real.
 - Path C: get a new a16f_w8 xclbin — would need AMD to ship one, or build it from AIE-NN. Not user-side work.

What full Q8_0 support actually requires

It's not just "change b_dtype to int8". The list:

 1. New template instantiation: qlinear_2<int16_t, int8_t, int64_t, float> (or similar — needs verification that int64 accumulator path is wired up for float output).
 2. New activation pipeline. Replace the F32→BF16 conversion segment with F32→int16 quantization:
 - Compute per-call activation scale (max-abs or similar)
 - Convert F32 → int16 saturating
 - Track the scale so we can dequantize the output
 3. New output pipeline. Replace the "just write fp32 to dst" path with:
 - int64 accumulator → F32 via dst = acc_int64
  4. activation_scale
  * weight_scale
 - Weight scale is per-32-element (from Q8_0's fp16 scale)
 - Activation scale is per-call
 - This is matrix-shaped fp32 output, with per-column weight scales applied
 5. New ensure_op_for_weight path for Q8_0. Q8_0 stores int8 weights directly (no nibble unpacking) + one fp16 scale per 32 weights. The "transposed unpack" becomes "transposed copy" — actually simpler than Q4_0.
 6. CMake / runtime check that gemm_4x4_a16w8acc64.xclbin is present. Different sentinel.
 7. Update can_mul_mat to accept GGML_TYPE_Q8_0 with appropriate constraints.
 8. Per-tensor xclbin caching. Currently we eagerly load one xclbin at startup. With Q8_0 mixed in (likely K/V projections in some configs), we may need to lazily load both, or pick the right one based on the model.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Sample build and session run:

C:\llama.cpp\b612_052026\build.ryzenai>git branch
  hv/b612_052026
* hv/b612_052026_ryzenai

===== REM - for HX 370
C:\llama.cpp\b612_052026\build.ryzenai>conda env list (this is RyzenAI-v-1.6)

# conda environments:
#
                       C:\ProgramData\anaconda3
                     * C:\ProgramData\anaconda3\envs\ryzenai-transformers
                       C:\ProgramData\miniconda3
                       c:\ProgramData\anaconda3\envs\ryzen-ai-1.2.0
                       c:\ProgramData\anaconda3\envs\ryzen-ai-1.6.0
base                   c:\llama.cpp\miniforge3


C:\llama.cpp\b612_052026\build.ryzenai>conda activate C:\ProgramData\anaconda3\envs\ryzenai-transformers

C:\llama.cpp\b612_052026\build.ryzenai>cmake .. -DGGML_RYZENAI=ON -DRyzenAI_DIR=C:/ProgramData/anaconda3/envs/ryzenai-transformers/Lib/cmake/ryzenai -DXRT_DIR=c:/llama.cpp/Ryzen/example/transformers/third_party/xrt-ipu/xrt/share/cmake/XRT

===== REM - for MAX 395
C:\Program Files\Microsoft Visual Studio\2022\Community>conda env list

# conda environments:
#
base                   c:\llama.cpp\miniforge3
ryzen-ai-1.6.0         c:\llama.cpp\miniforge3\envs\ryzen-ai-1.6.0
ryzen-ai-1.7.0         c:\llama.cpp\miniforge3\envs\ryzen-ai-1.7.0
ryzenai-transformers   c:\llama.cpp\miniforge3\envs\ryzenai-transformers <<< Fake one but works! >>>

C:\Program Files\Microsoft Visual Studio\2022\Community>conda activate ryzenai-transformers

REM - if NOT running with ryzenai-transformers Conda environment
cmake .. -DGGML_RYZENAI=ON -DRyzenAI_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Lib/cmake/ryzenai -DXRT_DIR=c:/llama.cpp/RyzenAI-SW-1.6_Q_liner2/example/transformers/third_party/xrt-ipu/xrt/share/cmake/XRT -Dspdlog_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Library/lib/cmake/spdlog -DEigen3_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Library/share/eigen3/cmake -Daie_controller_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Lib/cmake/aie_controller -Dxaiengine_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Lib/cmake/xaiengine

REM - if running with env ryzenai-transformers activated
C:\llama.cpp\b612_052026\build.test>cmake .. -DGGML_RYZENAI=ON -DGGML_RYZENAI=ON -DRyzenAI_DIR=C:/llama.cpp/miniforge3/envs/ryzenai-transformers/Lib/cmake/ryzenai -DXRT_DIR=c:/llama.cpp/RyzenAI-SW-1.6_Q_liner2/example/transformers/third_party/xrt-ipu/xrt/share/cmake/XRT

C:\llama.cpp\b612_052026\build.ryzenai>cmake --build . --config RelWithDebInfo --target minslm-cli  

 Directory of C:\llama.test\RyzenAI

05/25/2026  04:49 PM    <DIR>          .
04/24/2026  01:22 PM    <DIR>          ..
10/06/2024  06:08 PM    <DIR>          dll
05/25/2026  06:13 PM           837,632 ggml-base.dll
05/25/2026  05:06 PM         1,598,976 ggml-cpu.dll
05/25/2026  06:13 PM           195,072 ggml-ryzenai.dll
05/25/2026  05:06 PM           316,416 ggml.dll
05/25/2026  05:34 PM         3,116,544 llama.dll
10/10/2024  02:30 PM    <DIR>          llama_cache
10/18/2024  01:24 PM         3,906,048 llbench.exe
10/10/2024  02:37 PM         2,872,832 llbench.org
05/25/2026  05:34 PM           300,032 minslm-cli.exe
03/01/2025  11:58 AM    <DIR>          models
10/10/2024  02:27 PM    <DIR>          prompts
10/10/2024  03:57 PM               362 run_llb.cmd
10/10/2024  03:51 PM               341 run_tg.cmd
10/18/2024  01:23 PM         3,227,136 tg.exe
10/10/2024  02:37 PM         2,194,944 tg.org
10/06/2024  06:04 PM    <DIR>          xclbin
              12 File(s)     18,566,335 bytes
               7 Dir(s)  21,843,542,016 bytes free

REM - On MAX 395
C:\llama.cpp\b612_052026\build.ryzen>set PYTORCH_AIE_PATH=c:\llama.cpp\RyzenAI-SW-1.6_Q_liner2\example\transformers 
C:\llama.cpp\b612_052026\build.ryzen>set DEVICE=stx  

:\llama.cpp\b612_052026\build.ryzen>bin\RelWithDebInfo\minslm-cli c:\llama.cpp\models\Phi-3\Phi-3-mini-4k-instruct-Q4_0-MS.gguf 4 ..\examples\llm-infer\prompts\single_prompt.txt v2
[main]: loaded 1 prompt(s) from '..\examples\llm-infer\prompts\single_prompt.txt'
llama_model_loader: direct I/O is enabled, disabling mmap
llama_model_loader: loaded meta data with 25 key-value pairs and 291 tensors from c:\llama.cpp\models\Phi-3\Phi-3-mini-4k-instruct-Q4_0-MS.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                           llama.vocab_size u32              = 32064
llama_model_loader: - kv   3:                       llama.context_length u32              = 4096
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 3072
llama_model_loader: - kv   5:                          llama.block_count u32              = 32
llama_model_loader: - kv   6:                  llama.feed_forward_length u32              = 8192
llama_model_loader: - kv   7:                 llama.rope.dimension_count u32              = 96
llama_model_loader: - kv   8:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   9:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv  10:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  11:                       llama.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  12:                          general.file_type u32              = 2
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  14:                      tokenizer.ggml.tokens arr[str,32064]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  15:                      tokenizer.ggml.scores arr[f32,32064]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,32064]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 32000
llama_model_loader: - kv  19:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  20:            tokenizer.ggml.padding_token_id u32              = 32000
llama_model_loader: - kv  21:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  22:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  23:                    tokenizer.chat_template str              = {{ bos_token }}{% for message in mess...
llama_model_loader: - kv  24:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_0:  225 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q4_0
print_info: file size   = 2.03 GiB (4.55 BPW)
init_tokenizer: initializing tokenizer for type 1
load: control-looking token:  32020 '<|fim_prefix|>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token:  32021 '<|fim_middle|>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token:  32022 '<|fim_suffix|>' was not control-type; this is probably a bug in the model. its type will be overridden
load: control-looking token:  32007 '<|end|>' was not control-type; this is probably a bug in the model. its type will be overridden
load: 0 unused tokens
load: control token:      1 '<s>' is not marked as EOG
load: control token:  32020 '<|fim_prefix|>' is not marked as EOG
load: control token:  32021 '<|fim_middle|>' is not marked as EOG
load: control token:  32022 '<|fim_suffix|>' is not marked as EOG
load: control-looking token:  32000 '<|endoftext|>' was not control-type; this is probably a bug in the model. its type will be overridden
load: setting token '<|message|>' (32019) attribute to USER_DEFINED (16), old attributes: 16
load: setting token '<|start|>' (32018) attribute to USER_DEFINED (16), old attributes: 16
load: printing all EOG tokens:
load:   - 2 ('</s>')
load:   - 32000 ('<|endoftext|>')
load:   - 32007 ('<|end|>')
load: special tokens cache size = 67
load: token to piece cache size = 0.1691 MB
print_info: arch                  = llama
print_info: vocab_only            = 0
print_info: no_alloc              = 0
print_info: n_ctx_train           = 4096
print_info: n_embd                = 3072
print_info: n_embd_inp            = 3072
print_info: n_layer               = 32
print_info: n_head                = 32
print_info: n_head_kv             = 32
print_info: n_rot                 = 96
print_info: n_swa                 = 0
print_info: is_swa_any            = 0
print_info: n_embd_head_k         = 96
print_info: n_embd_head_v         = 96
print_info: n_gqa                 = 1
print_info: n_embd_k_gqa          = 3072
print_info: n_embd_v_gqa          = 3072
print_info: f_norm_eps            = 0.0e+00
print_info: f_norm_rms_eps        = 1.0e-05
print_info: f_clamp_kqv           = 0.0e+00
print_info: f_max_alibi_bias      = 0.0e+00
print_info: f_logit_scale         = 0.0e+00
print_info: f_attn_scale          = 0.0e+00
print_info: f_attn_value_scale    = 0.0000
print_info: n_ff                  = 8192
print_info: n_expert              = 0
print_info: n_expert_used         = 0
print_info: n_expert_groups       = 0
print_info: n_group_used          = 0
print_info: causal attn           = 1
print_info: pooling type          = -1
print_info: rope type             = 0
print_info: rope scaling          = linear
print_info: freq_base_train       = 10000.0
print_info: freq_scale_train      = 1
print_info: n_ctx_orig_yarn       = 4096
print_info: rope_yarn_log_mul     = 0.0000
print_info: rope_finetuned        = unknown
print_info: model type            = 7B
print_info: model params          = 3.82 B
print_info: general.name          = LLaMA v2
print_info: vocab type            = SPM
print_info: n_vocab               = 32064
print_info: n_merges              = 0
print_info: BOS token             = 1 '<s>'
print_info: EOS token             = 32000 '<|endoftext|>'
print_info: EOT token             = 32007 '<|end|>'
print_info: UNK token             = 0 '<unk>'
print_info: PAD token             = 32000 '<|endoftext|>'
print_info: LF token              = 13 '<0x0A>'
print_info: FIM PRE token         = 32020 '<|fim_prefix|>'
print_info: FIM SUF token         = 32022 '<|fim_suffix|>'
print_info: FIM MID token         = 32021 '<|fim_middle|>'
print_info: EOG token             = 2 '</s>'
print_info: EOG token             = 32000 '<|endoftext|>'
print_info: EOG token             = 32007 '<|end|>'
print_info: max token length      = 48
load_tensors: loading model tensors, this can take a while... (mmap = false, direct_io = true)
load_tensors: layer   0 assigned to device CPU, is_swa = 0
load_tensors: layer   1 assigned to device CPU, is_swa = 0
load_tensors: layer   2 assigned to device CPU, is_swa = 0
load_tensors: layer   3 assigned to device CPU, is_swa = 0
load_tensors: layer   4 assigned to device CPU, is_swa = 0
load_tensors: layer   5 assigned to device CPU, is_swa = 0
load_tensors: layer   6 assigned to device CPU, is_swa = 0
load_tensors: layer   7 assigned to device CPU, is_swa = 0
load_tensors: layer   8 assigned to device CPU, is_swa = 0
load_tensors: layer   9 assigned to device CPU, is_swa = 0
load_tensors: layer  10 assigned to device CPU, is_swa = 0
load_tensors: layer  11 assigned to device CPU, is_swa = 0
load_tensors: layer  12 assigned to device CPU, is_swa = 0
load_tensors: layer  13 assigned to device CPU, is_swa = 0
load_tensors: layer  14 assigned to device CPU, is_swa = 0
load_tensors: layer  15 assigned to device CPU, is_swa = 0
load_tensors: layer  16 assigned to device CPU, is_swa = 0
load_tensors: layer  17 assigned to device CPU, is_swa = 0
load_tensors: layer  18 assigned to device CPU, is_swa = 0
load_tensors: layer  19 assigned to device CPU, is_swa = 0
load_tensors: layer  20 assigned to device CPU, is_swa = 0
load_tensors: layer  21 assigned to device CPU, is_swa = 0
load_tensors: layer  22 assigned to device CPU, is_swa = 0
load_tensors: layer  23 assigned to device CPU, is_swa = 0
load_tensors: layer  24 assigned to device CPU, is_swa = 0
load_tensors: layer  25 assigned to device CPU, is_swa = 0
load_tensors: layer  26 assigned to device CPU, is_swa = 0
load_tensors: layer  27 assigned to device CPU, is_swa = 0
load_tensors: layer  28 assigned to device CPU, is_swa = 0
load_tensors: layer  29 assigned to device CPU, is_swa = 0
load_tensors: layer  30 assigned to device CPU, is_swa = 0
load_tensors: layer  31 assigned to device CPU, is_swa = 0
load_tensors: layer  32 assigned to device CPU, is_swa = 0
create_tensor: loading tensor token_embd.weight
create_tensor: loading tensor output_norm.weight
create_tensor: loading tensor output.weight
create_tensor: loading tensor blk.0.attn_norm.weight
create_tensor: loading tensor blk.0.attn_q.weight
create_tensor: loading tensor blk.0.attn_k.weight
create_tensor: loading tensor blk.0.attn_v.weight
create_tensor: loading tensor blk.0.attn_output.weight
create_tensor: loading tensor blk.0.ffn_norm.weight
create_tensor: loading tensor blk.0.ffn_gate.weight
create_tensor: loading tensor blk.0.ffn_down.weight
create_tensor: loading tensor blk.0.ffn_up.weight
create_tensor: loading tensor blk.1.attn_norm.weight
create_tensor: loading tensor blk.1.attn_q.weight
create_tensor: loading tensor blk.1.attn_k.weight
create_tensor: loading tensor blk.1.attn_v.weight
create_tensor: loading tensor blk.1.attn_output.weight
create_tensor: loading tensor blk.1.ffn_norm.weight
create_tensor: loading tensor blk.1.ffn_gate.weight
create_tensor: loading tensor blk.1.ffn_down.weight
create_tensor: loading tensor blk.1.ffn_up.weight
create_tensor: loading tensor blk.2.attn_norm.weight
create_tensor: loading tensor blk.2.attn_q.weight
create_tensor: loading tensor blk.2.attn_k.weight
create_tensor: loading tensor blk.2.attn_v.weight
create_tensor: loading tensor blk.2.attn_output.weight
create_tensor: loading tensor blk.2.ffn_norm.weight
create_tensor: loading tensor blk.2.ffn_gate.weight
create_tensor: loading tensor blk.2.ffn_down.weight
create_tensor: loading tensor blk.2.ffn_up.weight
create_tensor: loading tensor blk.3.attn_norm.weight
create_tensor: loading tensor blk.3.attn_q.weight
create_tensor: loading tensor blk.3.attn_k.weight
create_tensor: loading tensor blk.3.attn_v.weight
create_tensor: loading tensor blk.3.attn_output.weight
create_tensor: loading tensor blk.3.ffn_norm.weight
create_tensor: loading tensor blk.3.ffn_gate.weight
create_tensor: loading tensor blk.3.ffn_down.weight
create_tensor: loading tensor blk.3.ffn_up.weight
create_tensor: loading tensor blk.4.attn_norm.weight
create_tensor: loading tensor blk.4.attn_q.weight
create_tensor: loading tensor blk.4.attn_k.weight
create_tensor: loading tensor blk.4.attn_v.weight
create_tensor: loading tensor blk.4.attn_output.weight
create_tensor: loading tensor blk.4.ffn_norm.weight
create_tensor: loading tensor blk.4.ffn_gate.weight
create_tensor: loading tensor blk.4.ffn_down.weight
create_tensor: loading tensor blk.4.ffn_up.weight
create_tensor: loading tensor blk.5.attn_norm.weight
create_tensor: loading tensor blk.5.attn_q.weight
create_tensor: loading tensor blk.5.attn_k.weight
create_tensor: loading tensor blk.5.attn_v.weight
create_tensor: loading tensor blk.5.attn_output.weight
create_tensor: loading tensor blk.5.ffn_norm.weight
create_tensor: loading tensor blk.5.ffn_gate.weight
create_tensor: loading tensor blk.5.ffn_down.weight
create_tensor: loading tensor blk.5.ffn_up.weight
create_tensor: loading tensor blk.6.attn_norm.weight
create_tensor: loading tensor blk.6.attn_q.weight
create_tensor: loading tensor blk.6.attn_k.weight
create_tensor: loading tensor blk.6.attn_v.weight
create_tensor: loading tensor blk.6.attn_output.weight
create_tensor: loading tensor blk.6.ffn_norm.weight
create_tensor: loading tensor blk.6.ffn_gate.weight
create_tensor: loading tensor blk.6.ffn_down.weight
create_tensor: loading tensor blk.6.ffn_up.weight
create_tensor: loading tensor blk.7.attn_norm.weight
create_tensor: loading tensor blk.7.attn_q.weight
create_tensor: loading tensor blk.7.attn_k.weight
create_tensor: loading tensor blk.7.attn_v.weight
create_tensor: loading tensor blk.7.attn_output.weight
create_tensor: loading tensor blk.7.ffn_norm.weight
create_tensor: loading tensor blk.7.ffn_gate.weight
create_tensor: loading tensor blk.7.ffn_down.weight
create_tensor: loading tensor blk.7.ffn_up.weight
create_tensor: loading tensor blk.8.attn_norm.weight
create_tensor: loading tensor blk.8.attn_q.weight
create_tensor: loading tensor blk.8.attn_k.weight
create_tensor: loading tensor blk.8.attn_v.weight
create_tensor: loading tensor blk.8.attn_output.weight
create_tensor: loading tensor blk.8.ffn_norm.weight
create_tensor: loading tensor blk.8.ffn_gate.weight
create_tensor: loading tensor blk.8.ffn_down.weight
create_tensor: loading tensor blk.8.ffn_up.weight
create_tensor: loading tensor blk.9.attn_norm.weight
create_tensor: loading tensor blk.9.attn_q.weight
create_tensor: loading tensor blk.9.attn_k.weight
create_tensor: loading tensor blk.9.attn_v.weight
create_tensor: loading tensor blk.9.attn_output.weight
create_tensor: loading tensor blk.9.ffn_norm.weight
create_tensor: loading tensor blk.9.ffn_gate.weight
create_tensor: loading tensor blk.9.ffn_down.weight
create_tensor: loading tensor blk.9.ffn_up.weight
create_tensor: loading tensor blk.10.attn_norm.weight
create_tensor: loading tensor blk.10.attn_q.weight
create_tensor: loading tensor blk.10.attn_k.weight
create_tensor: loading tensor blk.10.attn_v.weight
create_tensor: loading tensor blk.10.attn_output.weight
create_tensor: loading tensor blk.10.ffn_norm.weight
create_tensor: loading tensor blk.10.ffn_gate.weight
create_tensor: loading tensor blk.10.ffn_down.weight
create_tensor: loading tensor blk.10.ffn_up.weight
create_tensor: loading tensor blk.11.attn_norm.weight
create_tensor: loading tensor blk.11.attn_q.weight
create_tensor: loading tensor blk.11.attn_k.weight
create_tensor: loading tensor blk.11.attn_v.weight
create_tensor: loading tensor blk.11.attn_output.weight
create_tensor: loading tensor blk.11.ffn_norm.weight
create_tensor: loading tensor blk.11.ffn_gate.weight
create_tensor: loading tensor blk.11.ffn_down.weight
create_tensor: loading tensor blk.11.ffn_up.weight
create_tensor: loading tensor blk.12.attn_norm.weight
create_tensor: loading tensor blk.12.attn_q.weight
create_tensor: loading tensor blk.12.attn_k.weight
create_tensor: loading tensor blk.12.attn_v.weight
create_tensor: loading tensor blk.12.attn_output.weight
create_tensor: loading tensor blk.12.ffn_norm.weight
create_tensor: loading tensor blk.12.ffn_gate.weight
create_tensor: loading tensor blk.12.ffn_down.weight
create_tensor: loading tensor blk.12.ffn_up.weight
create_tensor: loading tensor blk.13.attn_norm.weight
create_tensor: loading tensor blk.13.attn_q.weight
create_tensor: loading tensor blk.13.attn_k.weight
create_tensor: loading tensor blk.13.attn_v.weight
create_tensor: loading tensor blk.13.attn_output.weight
create_tensor: loading tensor blk.13.ffn_norm.weight
create_tensor: loading tensor blk.13.ffn_gate.weight
create_tensor: loading tensor blk.13.ffn_down.weight
create_tensor: loading tensor blk.13.ffn_up.weight
create_tensor: loading tensor blk.14.attn_norm.weight
create_tensor: loading tensor blk.14.attn_q.weight
create_tensor: loading tensor blk.14.attn_k.weight
create_tensor: loading tensor blk.14.attn_v.weight
create_tensor: loading tensor blk.14.attn_output.weight
create_tensor: loading tensor blk.14.ffn_norm.weight
create_tensor: loading tensor blk.14.ffn_gate.weight
create_tensor: loading tensor blk.14.ffn_down.weight
create_tensor: loading tensor blk.14.ffn_up.weight
create_tensor: loading tensor blk.15.attn_norm.weight
create_tensor: loading tensor blk.15.attn_q.weight
create_tensor: loading tensor blk.15.attn_k.weight
create_tensor: loading tensor blk.15.attn_v.weight
create_tensor: loading tensor blk.15.attn_output.weight
create_tensor: loading tensor blk.15.ffn_norm.weight
create_tensor: loading tensor blk.15.ffn_gate.weight
create_tensor: loading tensor blk.15.ffn_down.weight
create_tensor: loading tensor blk.15.ffn_up.weight
create_tensor: loading tensor blk.16.attn_norm.weight
create_tensor: loading tensor blk.16.attn_q.weight
create_tensor: loading tensor blk.16.attn_k.weight
create_tensor: loading tensor blk.16.attn_v.weight
create_tensor: loading tensor blk.16.attn_output.weight
create_tensor: loading tensor blk.16.ffn_norm.weight
create_tensor: loading tensor blk.16.ffn_gate.weight
create_tensor: loading tensor blk.16.ffn_down.weight
create_tensor: loading tensor blk.16.ffn_up.weight
create_tensor: loading tensor blk.17.attn_norm.weight
create_tensor: loading tensor blk.17.attn_q.weight
create_tensor: loading tensor blk.17.attn_k.weight
create_tensor: loading tensor blk.17.attn_v.weight
create_tensor: loading tensor blk.17.attn_output.weight
create_tensor: loading tensor blk.17.ffn_norm.weight
create_tensor: loading tensor blk.17.ffn_gate.weight
create_tensor: loading tensor blk.17.ffn_down.weight
create_tensor: loading tensor blk.17.ffn_up.weight
create_tensor: loading tensor blk.18.attn_norm.weight
create_tensor: loading tensor blk.18.attn_q.weight
create_tensor: loading tensor blk.18.attn_k.weight
create_tensor: loading tensor blk.18.attn_v.weight
create_tensor: loading tensor blk.18.attn_output.weight
create_tensor: loading tensor blk.18.ffn_norm.weight
create_tensor: loading tensor blk.18.ffn_gate.weight
create_tensor: loading tensor blk.18.ffn_down.weight
create_tensor: loading tensor blk.18.ffn_up.weight
create_tensor: loading tensor blk.19.attn_norm.weight
create_tensor: loading tensor blk.19.attn_q.weight
create_tensor: loading tensor blk.19.attn_k.weight
create_tensor: loading tensor blk.19.attn_v.weight
create_tensor: loading tensor blk.19.attn_output.weight
create_tensor: loading tensor blk.19.ffn_norm.weight
create_tensor: loading tensor blk.19.ffn_gate.weight
create_tensor: loading tensor blk.19.ffn_down.weight
create_tensor: loading tensor blk.19.ffn_up.weight
create_tensor: loading tensor blk.20.attn_norm.weight
create_tensor: loading tensor blk.20.attn_q.weight
create_tensor: loading tensor blk.20.attn_k.weight
create_tensor: loading tensor blk.20.attn_v.weight
create_tensor: loading tensor blk.20.attn_output.weight
create_tensor: loading tensor blk.20.ffn_norm.weight
create_tensor: loading tensor blk.20.ffn_gate.weight
create_tensor: loading tensor blk.20.ffn_down.weight
create_tensor: loading tensor blk.20.ffn_up.weight
create_tensor: loading tensor blk.21.attn_norm.weight
create_tensor: loading tensor blk.21.attn_q.weight
create_tensor: loading tensor blk.21.attn_k.weight
create_tensor: loading tensor blk.21.attn_v.weight
create_tensor: loading tensor blk.21.attn_output.weight
create_tensor: loading tensor blk.21.ffn_norm.weight
create_tensor: loading tensor blk.21.ffn_gate.weight
create_tensor: loading tensor blk.21.ffn_down.weight
create_tensor: loading tensor blk.21.ffn_up.weight
create_tensor: loading tensor blk.22.attn_norm.weight
create_tensor: loading tensor blk.22.attn_q.weight
create_tensor: loading tensor blk.22.attn_k.weight
create_tensor: loading tensor blk.22.attn_v.weight
create_tensor: loading tensor blk.22.attn_output.weight
create_tensor: loading tensor blk.22.ffn_norm.weight
create_tensor: loading tensor blk.22.ffn_gate.weight
create_tensor: loading tensor blk.22.ffn_down.weight
create_tensor: loading tensor blk.22.ffn_up.weight
create_tensor: loading tensor blk.23.attn_norm.weight
create_tensor: loading tensor blk.23.attn_q.weight
create_tensor: loading tensor blk.23.attn_k.weight
create_tensor: loading tensor blk.23.attn_v.weight
create_tensor: loading tensor blk.23.attn_output.weight
create_tensor: loading tensor blk.23.ffn_norm.weight
create_tensor: loading tensor blk.23.ffn_gate.weight
create_tensor: loading tensor blk.23.ffn_down.weight
create_tensor: loading tensor blk.23.ffn_up.weight
create_tensor: loading tensor blk.24.attn_norm.weight
create_tensor: loading tensor blk.24.attn_q.weight
create_tensor: loading tensor blk.24.attn_k.weight
create_tensor: loading tensor blk.24.attn_v.weight
create_tensor: loading tensor blk.24.attn_output.weight
create_tensor: loading tensor blk.24.ffn_norm.weight
create_tensor: loading tensor blk.24.ffn_gate.weight
create_tensor: loading tensor blk.24.ffn_down.weight
create_tensor: loading tensor blk.24.ffn_up.weight
create_tensor: loading tensor blk.25.attn_norm.weight
create_tensor: loading tensor blk.25.attn_q.weight
create_tensor: loading tensor blk.25.attn_k.weight
create_tensor: loading tensor blk.25.attn_v.weight
create_tensor: loading tensor blk.25.attn_output.weight
create_tensor: loading tensor blk.25.ffn_norm.weight
create_tensor: loading tensor blk.25.ffn_gate.weight
create_tensor: loading tensor blk.25.ffn_down.weight
create_tensor: loading tensor blk.25.ffn_up.weight
create_tensor: loading tensor blk.26.attn_norm.weight
create_tensor: loading tensor blk.26.attn_q.weight
create_tensor: loading tensor blk.26.attn_k.weight
create_tensor: loading tensor blk.26.attn_v.weight
create_tensor: loading tensor blk.26.attn_output.weight
create_tensor: loading tensor blk.26.ffn_norm.weight
create_tensor: loading tensor blk.26.ffn_gate.weight
create_tensor: loading tensor blk.26.ffn_down.weight
create_tensor: loading tensor blk.26.ffn_up.weight
create_tensor: loading tensor blk.27.attn_norm.weight
create_tensor: loading tensor blk.27.attn_q.weight
create_tensor: loading tensor blk.27.attn_k.weight
create_tensor: loading tensor blk.27.attn_v.weight
create_tensor: loading tensor blk.27.attn_output.weight
create_tensor: loading tensor blk.27.ffn_norm.weight
create_tensor: loading tensor blk.27.ffn_gate.weight
create_tensor: loading tensor blk.27.ffn_down.weight
create_tensor: loading tensor blk.27.ffn_up.weight
create_tensor: loading tensor blk.28.attn_norm.weight
create_tensor: loading tensor blk.28.attn_q.weight
create_tensor: loading tensor blk.28.attn_k.weight
create_tensor: loading tensor blk.28.attn_v.weight
create_tensor: loading tensor blk.28.attn_output.weight
create_tensor: loading tensor blk.28.ffn_norm.weight
create_tensor: loading tensor blk.28.ffn_gate.weight
create_tensor: loading tensor blk.28.ffn_down.weight
create_tensor: loading tensor blk.28.ffn_up.weight
create_tensor: loading tensor blk.29.attn_norm.weight
create_tensor: loading tensor blk.29.attn_q.weight
create_tensor: loading tensor blk.29.attn_k.weight
create_tensor: loading tensor blk.29.attn_v.weight
create_tensor: loading tensor blk.29.attn_output.weight
create_tensor: loading tensor blk.29.ffn_norm.weight
create_tensor: loading tensor blk.29.ffn_gate.weight
create_tensor: loading tensor blk.29.ffn_down.weight
create_tensor: loading tensor blk.29.ffn_up.weight
create_tensor: loading tensor blk.30.attn_norm.weight
create_tensor: loading tensor blk.30.attn_q.weight
create_tensor: loading tensor blk.30.attn_k.weight
create_tensor: loading tensor blk.30.attn_v.weight
create_tensor: loading tensor blk.30.attn_output.weight
create_tensor: loading tensor blk.30.ffn_norm.weight
create_tensor: loading tensor blk.30.ffn_gate.weight
create_tensor: loading tensor blk.30.ffn_down.weight
create_tensor: loading tensor blk.30.ffn_up.weight
create_tensor: loading tensor blk.31.attn_norm.weight
create_tensor: loading tensor blk.31.attn_q.weight
create_tensor: loading tensor blk.31.attn_k.weight
create_tensor: loading tensor blk.31.attn_v.weight
create_tensor: loading tensor blk.31.attn_output.weight
create_tensor: loading tensor blk.31.ffn_norm.weight
create_tensor: loading tensor blk.31.ffn_gate.weight
create_tensor: loading tensor blk.31.ffn_down.weight
create_tensor: loading tensor blk.31.ffn_up.weight
done_getting_tensors: tensor 'token_embd.weight' (q4_0) (and 290 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
load_tensors:          CPU model buffer size =  2074.66 MiB
load_all_data: no device found for buffer type CPU for async uploads
................................................................................................
load_tensors: releasing mmaps (not needed after tensor copy)
llama_context: constructing llama_context
llama_context: n_seq_max     = 1
llama_context: n_ctx         = 2048
llama_context: n_ctx_seq     = 2048
llama_context: n_batch       = 512
llama_context: n_ubatch      = 512
llama_context: causal_attn   = 1
llama_context: flash_attn    = auto
llama_context: kv_unified    = false
llama_context: freq_base     = 10000.0
llama_context: freq_scale    = 1
llama_context: n_rs_seq      = 0
llama_context: n_ctx_seq (2048) < n_ctx_train (4096) -- the full capacity of the model will not be utilized
ggml-ryzenai: NPU initialized (xclbin loaded)
set_abort_callback: call
llama_context:        CPU  output buffer size =     0.12 MiB
llama_kv_cache: layer   0: dev = CPU
llama_kv_cache: layer   1: dev = CPU
llama_kv_cache: layer   2: dev = CPU
llama_kv_cache: layer   3: dev = CPU
llama_kv_cache: layer   4: dev = CPU
llama_kv_cache: layer   5: dev = CPU
llama_kv_cache: layer   6: dev = CPU
llama_kv_cache: layer   7: dev = CPU
llama_kv_cache: layer   8: dev = CPU
llama_kv_cache: layer   9: dev = CPU
llama_kv_cache: layer  10: dev = CPU
llama_kv_cache: layer  11: dev = CPU
llama_kv_cache: layer  12: dev = CPU
llama_kv_cache: layer  13: dev = CPU
llama_kv_cache: layer  14: dev = CPU
llama_kv_cache: layer  15: dev = CPU
llama_kv_cache: layer  16: dev = CPU
llama_kv_cache: layer  17: dev = CPU
llama_kv_cache: layer  18: dev = CPU
llama_kv_cache: layer  19: dev = CPU
llama_kv_cache: layer  20: dev = CPU
llama_kv_cache: layer  21: dev = CPU
llama_kv_cache: layer  22: dev = CPU
llama_kv_cache: layer  23: dev = CPU
llama_kv_cache: layer  24: dev = CPU
llama_kv_cache: layer  25: dev = CPU
llama_kv_cache: layer  26: dev = CPU
llama_kv_cache: layer  27: dev = CPU
llama_kv_cache: layer  28: dev = CPU
llama_kv_cache: layer  29: dev = CPU
llama_kv_cache: layer  30: dev = CPU
llama_kv_cache: layer  31: dev = CPU
llama_kv_cache:        CPU KV buffer size =   768.00 MiB
llama_kv_cache: size =  768.00 MiB (  2048 cells,  32 layers,  1/1 seqs), K (f16):  384.00 MiB, V (f16):  384.00 MiB
llama_kv_cache: attn_rot_k = 0, n_embd_head_k_all = 96
llama_kv_cache: attn_rot_v = 0, n_embd_head_k_all = 96
llama_context: enumerating backends
llama_context: backend_ptrs.size() = 2
sched_reserve: reserving ...
sched_reserve: max_nodes = 2328
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
ggml-ryzenai: preloaded 64 new weight tensor(s) (64 eligible in cgraph)
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =    98.64 MiB
sched_reserve: graph nodes  = 999
sched_reserve: graph splits = 65 (with bs=512), 1 (with bs=1)
sched_reserve: reserve took 24592.81 ms, sched copies = 1

[main]: n_ctx = 2048, n_threads = 4, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |

> Running with custom prompt => [1/1]: [This is Izzy's bedroom]
  prefill: 14866.0ms (574 tokens, 38.6 t/s)
 {
    "answer": "a. Are you sure Mara, what makes you believe that is true?",
    "justification": "Mara has just given an answer but she seems unsure about it which could be due to her discomfort in discussing the topic."
}
  [all token ids (63): 426 13 1678 376 12011 1115 376 29874 29889 4683 366 1854 1085 29874 29892 825 3732 366 4658 393 338 1565 29973 613 13 1678 376 5143 2450 1115 376 29924 2518 756 925 2183 385 1234 541 1183 2444 9644 545 1048 372 607 1033 367 2861 304 902 766 510 3921 297 5353 292 278 11261 1213 29871 13 29913]
  [output hex (226 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 61 2e 20 41 72 65 20 79 6f 75 20 73 75 72  | "a. Are you sur|
    0020: 65 20 4d 61 72 61 2c 20 77 68 61 74 20 6d 61 6b  |e Mara, what mak|
    0030: 65 73 20 79 6f 75 20 62 65 6c 69 65 76 65 20 74  |es you believe t|
    0040: 68 61 74 20 69 73 20 74 72 75 65 3f 22 2c 0a 20  |hat is true?",. |
    0050: 20 20 20 22 6a 75 73 74 69 66 69 63 61 74 69 6f  |   "justificatio|
    0060: 6e 22 3a 20 22 4d 61 72 61 20 68 61 73 20 6a 75  |n": "Mara has ju|
    0070: 73 74 20 67 69 76 65 6e 20 61 6e 20 61 6e 73 77  |st given an answ|
    0080: 65 72 20 62 75 74 20 73 68 65 20 73 65 65 6d 73  |er but she seems|
    0090: 20 75 6e 73 75 72 65 20 61 62 6f 75 74 20 69 74  | unsure about it|
    00a0: 20 77 68 69 63 68 20 63 6f 75 6c 64 20 62 65 20  | which could be |
    00b0: 64 75 65 20 74 6f 20 68 65 72 20 64 69 73 63 6f  |due to her disco|
    00c0: 6d 66 6f 72 74 20 69 6e 20 64 69 73 63 75 73 73  |mfort in discuss|
    00d0: 69 6e 67 20 74 68 65 20 74 6f 70 69 63 2e 22 20  |ing the topic." |
    00e0: 0a 7d                                            |.}|
  decode: 4827.8ms (62 tokens, 12.84 t/s)
  decode (no-TTFT): 4827.7ms (63 tokens, 13.05 t/s) [TTFT 0.1ms]


===== Summary =====
  prompts:    1
  tokens gen: 62
  avg t/s:    12.84
  avg t/s (no-TTFT): 13.05
  avg TTFT:          0.1ms
  total time: 19.74s
llama_perf_context_print:        load time =   40319.03 ms
llama_perf_context_print: prompt eval time =   14865.94 ms /   574 tokens (   25.90 ms per token,    38.61 tokens per second)
llama_perf_context_print:        eval time =    4816.63 ms /    62 runs   (   77.69 ms per token,    12.87 tokens per second)
llama_perf_context_print:       total time =   45179.78 ms /   636 tokens
llama_perf_context_print:    graphs reused =         61
process_memory: peak working_set=4914.4 MiB, current working_set=4912.0 MiB, private=4987.1 MiB
~llama_context:        CPU compute buffer size is  98.6387 MiB, matches expectation of  98.6387 MiB
~llama_context:        CPU compute buffer size is   0.0000 MiB, matches expectation of   0.0000 MiB

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

---

# qlinear_2: xclbin and DPU .bin location

There are actually **two** separate binary asset families, and they get located by **different code paths**.

## The two asset families

### 1. `.xclbin` — the **hardware kernel** loaded onto the NPU
- One per (device, design, dtype-combo). Big static asset that programs the AIE tiles.
- Shared across all shapes of the same dtype combo.
- File examples: `mladf_gemm_4x4_a16fw4acc32f.xclbin`, `gemm_4x4_a8w8acc32.xclbin`.

### 2. `.bin` — per-shape **DPU instruction stream** that drives the kernel
- One per (shape M×K×N, group_size). Tells the loaded kernel how to walk THIS particular matmul.
- File examples we saw: `a16fw4acc32f_1_4096_4096_128.bin`, `mladf_2x4x4_a16fw4acc16f_1_4096_4096_128.bin`.

Same kernel + many instruction blobs = many supported shapes without recompiling the NPU program.

## Path resolution — the components involved

```
ggml-ryzenai.cpp
    └─> qlinear_2 ctor                              ← decides WHICH xclbin
            ├─ reads env: PYTORCH_AIE_PATH, DEVICE, MLADF
            ├─ composes XCLBIN_FNAME from dtype headers
            │     a16f + w4 + acc32f  →  a16fw4acc32f
            ├─ calls xrt_context::get_instance(XCLBIN_FNAME)   ← LOADS xclbin
            └─ calls setup_instr_registry()         ← decides WHICH .bin files
                    └─ instr_registry.add_instructions(list, DPU_DIR)
                          └─ loads .bin per supported shape into XRT BO

later, per execute():
qlinear_2::execute(a, shape, c)
    └─ builds instr_bo_key from (txn_prefix, M, K, N, grp)
    └─ instr_reg_.get_instr_bo(key + ".bin")        ← PICKS .bin for THIS call
    └─ xrt::run(kernel, a_bo, w_bo, c_bo, instr_bo)
```

## The two paths concretely

### xclbin selection — `qlinear_2.hpp:563–586`
```cpp
XCLBIN_FNAME =
    getenv("PYTORCH_AIE_PATH") + "\\xclbin\\" + getenv("DEVICE") +
    (mladf ? (M4x4 ? "\\mladf_gemm_4x4_" : "\\mladf_gemm_2x4x4_")
           : "\\gemm_4x4_") +
    xclbin_a_header[a_dtype]  +    // "a16f"  (bfloat16)
    xclbin_b_header[b_dtype]  +    // "w4"    (uint4)
    xclbin_acc_header[c_dtype]+    // "acc32f"(float32)
    ".xclbin";
```
Path template:
```
%PYTORCH_AIE_PATH%\xclbin\%DEVICE%\mladf_gemm_2x4x4_a16fw4acc32f.xclbin
```
- `PYTORCH_AIE_PATH` = root of the SDK install
- `DEVICE` = `"phx"` (XDNA 1 / Phoenix) or `"stx"` (XDNA 2 / Strix Point)
- `MLADF` = env var (`""`, `"4x4"`, or `"2x4x4"`) — selects the design variant

Then loaded by `xrt_context::get_instance(XCLBIN_FNAME)` (`xrt_context.hpp:25–33`):
```cpp
xclbin_ = xrt::xclbin(xclbin_fname);            // parse from disk
device_.register_xclbin(xclbin_);                // upload to NPU
context_ = xrt::hw_context(device_, uuid);       // get hw context
kernel_ = xrt::kernel(context_, KERNEL_NAME);    // bind kernel
```

`xrt_context` is a **singleton per xclbin path** — same xclbin reused across all `qlinear_2` instances using it. Critical for performance: one xclbin load per process.

### DPU `.bin` (instruction) selection
- **Directory** built once (`qlinear_2.hpp:302–304`):
  ```
  DPU_DIR = %PYTORCH_AIE_PATH%\dll\%DEVICE%\qlinear_2\
  ```
  This is where the `a16fw4acc32f_*.bin` / `mladf_*.bin` files live.

- **Which `.bin` to use per call** built dynamically (`get_instr_key`, `qlinear_2.hpp:495–504`):
  ```cpp
  return prefix + "_" + M + "_" + K + "_" + N + "_" + grp_size + ".bin";
  // e.g. "mladf_2x4x4_a16fw4acc16f_1_4096_4096_128.bin"
  ```

- `instr_registry` (separate helper) loads all supported `.bin` files at init (`setup_instr_registry`, line 508) and per-call `get_instr_bo(key)` returns the right preloaded XRT BO. The actual xrt run takes that BO as the instruction stream.

## Summary table

| Decision | Component | Inputs |
|---|---|---|
| **Which xclbin file** | `qlinear_2` ctor (lines 561–586) | env vars `PYTORCH_AIE_PATH`, `DEVICE`, `MLADF` + dtype combo |
| **Load xclbin into NPU** | `xrt_context::get_instance()` | xclbin path (singleton-cached) |
| **Where DPU .bin files live** | `qlinear_2::DPU_DIR` (lines 302–304) | env vars `PYTORCH_AIE_PATH`, `DEVICE` |
| **Which .bin per call** | `qlinear_2::execute` → `instr_reg_.get_instr_bo(key)` | M, K, N, group_size + dtype prefix |
| **Cache .bin → XRT BO** | `instr_registry` | shape list passed at `setup_instr_registry()` |

## What this means for the retrofit

A few practical implications to know before wiring it in:

1. **The SDK install is the source of truth, not our repo.** Path resolution depends on two env vars (`PYTORCH_AIE_PATH`, `DEVICE`, optionally `MLADF`). When we ship `ggml-ryzenai`, the user still needs the AMD SDK installed and these env vars set. We don't have to bundle the binaries — they're already on disk under the SDK root.

2. **`DEVICE` is the key compatibility knob for Strix Halo.** Today the code supports `"phx"` and `"stx"`. Strix Halo (Ryzen AI MAX+ 395) is a **different SKU** from Strix Point. We need to verify whether AMD's newer SDK adds a `"krk"` or `"stxh"` directory or treats Strix Halo as `"stx"`. This is the make-or-break check.

3. **`MLADF` env var selects the better kernels.** Without it set, you get the older `gemm_4x4_*` xclbin. With `MLADF=2x4x4` you get the larger Llama-2 kernels. We'll want to default to MLADF when available.

4. **Each `qlinear_2` instance ⇒ one xclbin load attempt + many `.bin` preloads.** Since `xrt_context` is a singleton per xclbin string, all `MUL_MAT` calls naturally share one xclbin load.

So in our retrofit:
- We do **not** copy `.xclbin`/`.bin` files into the llama.cpp tree.
- We **do** copy `ggml-ryzenai.{h,cpp}` + the relevant `qlinear_2` / `xrt_context` / `instr_registry` headers.
- We rely on the user having the Ryzen AI SDK installed and `PYTORCH_AIE_PATH` + `DEVICE` set.
- We add a `GGML_RYZENAI` CMake flag that requires `XRT_DIR` + `RyzenAI_DIR` to find headers/libs.

+++++++++++++++++++++++++++++++++++++++++++++++++++

---

# Design summary: ggml-ryzenai backend (initial scaffold + preload wiring)

This section summarizes the design landed in the first three commits on
`hv/b612_052026_ryzenai`:

| Commit | Title |
|---|---|
| `7ce773ea6` | `ggml-ryzenai: add NPU backend for AMD XDNA via qlinear_2` |
| `fa7eeb142` | `llama: wire ryzenai weight preload at context graph_reserve` |
| `4fe067e92` | `ggml-ryzenai: only log preload when work was done` |

## 1. Backend scaffold (`7ce773ea6`)

### Goal

Add a first-class `ggml-ryzenai` backend that dispatches Q4_0 MUL_MAT ops
to AMD XDNA NPUs (Strix Point / Strix Halo) through the `qlinear_2`
kernel from the AMD Ryzen AI SDK, while keeping `ggml-cpu` completely
unmodified. This replaces the in-tree hooks that lived in
`b612_llama.dc` and gives us a clean separation that follows the modern
backend pattern.

### Pattern choice: BLAS-style op-level extras backend

Reviewed `ggml-openvino`, `ggml-blas`, `ggml-cpu` as templates. Picked
**BLAS** as the model:

- Tensors live in CPU buffers (no host-to-device transfer per op).
- `supports_buft` returns `ggml_backend_buft_is_host(buft)` (CPU only).
- `device_get_buffer_type` returns `ggml_backend_cpu_buffer_type()`.
- `device_get_type` is `GGML_BACKEND_DEVICE_TYPE_ACCEL`.
- `caps.buffer_from_host_ptr = true`.
- `supports_op` only claims individual MUL_MAT ops when our gate passes.

This means we sit *on top of* the CPU buffer plumbing and selectively
claim individual nodes. No `n_gpu_layers`-style offload semantics.

### Files added

```
ggml/include/ggml-ryzenai.h                       # public API
ggml/src/ggml-ryzenai/CMakeLists.txt              # SDK detect + EMU fallback
ggml/src/ggml-ryzenai/ggml-ryzenai.cpp            # backend/device/reg interface
ggml/src/ggml-ryzenai/ggml-ryzenai-impl.h         # internal API
ggml/src/ggml-ryzenai/ggml-ryzenai-impl.cpp      # qlinear_2 wrapper
```

### Files modified (registration only)

- `ggml/CMakeLists.txt` — added `option(GGML_RYZENAI ... OFF)` and listed
  the public header.
- `ggml/src/CMakeLists.txt` — `ggml_add_backend(RYZENAI)`.
- `ggml/src/ggml-backend-reg.cpp` — include + `register_backend` +
  `load_best("ryzenai", ...)` (three insertions under
  `#ifdef GGML_USE_RYZENAI`).

No changes to any other backend, no changes to `ggml-cpu`.

### Gating heuristic (`can_mul_mat`)

Mirrors the proven b612_llama.dc heuristic:

- `src0->type == GGML_TYPE_Q4_0`
- `src1->type == GGML_TYPE_F32`
- `src1` contiguous
- `src0->ne[0] >= 4096`
- `src0->ne[2] == 1 && src0->ne[3] == 1`

Everything else falls through to whatever the scheduler picks next
(typically CPU).

### Per-weight `qlinear_2` cache

A process-wide singleton `RyzenAIContext` holds:

```cpp
std::unordered_map<std::string_view, op_t> map;
std::mutex mtx_;
```

Key: `std::string_view(src0->name)`. Lifetime assumption: tensor names
are stable for the lifetime of the loaded model (inherited from
b612_llama.dc; verified to hold in llama.cpp's model loader).

`ensure_op_for_weight_locked(ctx, src0)` is the single function that
creates an entry on first sight of a weight tensor. It is shared by
both the lazy mul_mat path and the preload path.

### Eager xclbin load (sentinel `op_t`)

`ggml_ryzenai_impl_init()` constructs and immediately destroys a
sentinel `op_t("bfloat16", "uint4", "float32")`. Constructing the
qlinear_2 instance is enough to trigger
`xrt_context::get_instance(XCLBIN_FNAME)`, which loads the .xclbin onto
the NPU. The singleton `xrt_context` inside the SDK persists, so the
sentinel can be discarded safely.

This means xclbin load happens at backend init time, not on first
matmul.

### Fail-fast on missing NPU

If the build links against the real SDK (`GGML_RYZENAI=ON` and SDK
found) but the NPU driver is absent or the device is missing at
runtime, the sentinel ctor throws. We let it propagate as a
`GGML_ABORT` so the user sees the failure at backend init time. No
silent CPU fallback. This is the user-confirmed policy; transparent
CPU fallback is left as future work.

### Emulation mode — never silent

When the SDK is not found at CMake time, the build still succeeds:

- `target_compile_definitions(ggml-ryzenai PRIVATE RYZENAI_EMULATION)`
- The impl `.cpp` uses `#ifdef RYZENAI_EMULATION` to swap in a scalar
  software GEMM (correctness baseline, intentionally slow).

Three visibility layers ensure no one ever runs EMU thinking it is the
real path:

1. **Configure-time:** `message(WARNING ...)` banner with a multi-line
   warning box from `ggml-ryzenai/CMakeLists.txt`.
2. **Runtime:** five-line `GGML_LOG_WARN` banner in
   `ggml_ryzenai_impl_init()`.
3. **Device description:** `device_get_description` returns
   `"AMD Ryzen AI NPU (emulated)"` so any device enumeration shows it.

Counterpart for the real path: a single
`GGML_LOG_INFO("NPU initialized (xclbin loaded)")` line.

### Memory layout of the qlinear_2 path

For each Q4_0 weight tensor we maintain (per-tensor, kept alive in the
singleton map):

- The qlinear_2 op handle (kernel state + NPU shared memory)
- BF16 weights (via `initialize_weights_int4` internally)

Per-op work (every mul_mat) is:

- Convert F32 activations → BF16 (`ryzenai::float_buffer_to_bfloat16`,
  AVX512+BF16 path detected once and cached in a static bool)
- Submit `op.execute(bf16_inputs, shape, dst)` to the NPU

### Q4_0 specifics inherited from b612_llama.dc

- Zero point is hard-wired to 8 (Q4_0 is symmetric int4).
- The transpose trick: ggml computes `A·Bᵀ = C`, but qlinear_2 expects
  `A·B = C`. Using `Bᵀ·Aᵀ = Cᵀ`, the weights and scales are transposed
  once at upload time (`transpose_inner` helper) so qlinear_2 directly
  produces ggml's expected `Cᵀ` layout.

## 2. Preload wiring (`fa7eeb142`)

### Problem

After scaffold landed, observed on Strix HX 370:

- xclbin load happens at init (good).
- Per-tensor `qlinear_2` upload (Q4_0 → unpacked FP32 → transpose → BF16
  → DMA to NPU) still happened **inside the first matmul** for each
  weight. Visible as a warm-up dip in NPU utilization at the start of
  the first prefill.

### Solution: call graph walk, parallel to repack-xbcg

Followed the **same pattern** as the b612 repack-xbcg path
(`llama_repack_tensor_callgraph`). Both are call-graph-driven
post-processing hooks that run once per context setup.

### Wiring (3 layers)

1. **Backend exposes a cgraph-only function** via proc address:
   ```cpp
   static void * ggml_backend_ryzenai_get_proc_address(...) {
       if (strcmp(name, "ggml_ryzenai_preload_weights_cgraph") == 0)
           return (void *) ggml_ryzenai_preload_weights_cgraph;
       ...
   }
   ```
   The 2-arg `ggml_backend_ryzenai_preload_weights(backend, cgraph)`
   public API is retained as a thin wrapper for direct backend users.
   The cgraph-only variant is what llama.cpp uses, because the proc
   address pattern in llama.cpp doesn't easily produce a backend handle.

2. **llama.cpp wrapper** (`src/llama.cpp`), parallel to
   `llama_repack_tensor_callgraph`:
   ```cpp
   void llama_ryzenai_preload_weights(struct ggml_cgraph * cgraph) {
       // find reg by name "RyzenAI", get proc, call. No-op if absent.
   }
   ```
   **No `#ifdef GGML_USE_RYZENAI` needed.** The wrapper looks up the
   backend by name via the registry; absence is a no-op. This keeps
   llama.cpp clean of yet another backend-specific guard.

3. **Call site** in `llama_context::graph_reserve`
   (`src/llama-context.cpp`), right after the existing repack hook:
   ```cpp
   auto * gf = model.build_graph(gparams);
   #if defined(GGML_XBOX_PERF) && defined(GGML_B612)
       llama_repack_tensor_callgraph(gf);
   #endif
       llama_ryzenai_preload_weights(gf);
   ```
   `graph_reserve` runs at context setup (during init), not on every
   decode, so preload fires at the right time.

## 3. Honest preload logging (`4fe067e92`)

### Observation

`graph_reserve` is called **multiple times** during context init — once
per worst-case ubatch shape (e.g. `n_tokens=1` for decode, `16` for
prompt-prefix, `512` for full prefill, plus a couple for FA / Gated
Delta Net resolution). On Phi-3-mini this fires ~6 times.

Our cache in `ensure_op_for_weight_locked` already no-ops on subsequent
calls, so the actual upload work happens exactly once. But the log line
claimed `preloaded N weight tensor(s)` on every invocation, which made
it look like we were redoing the work.

### Fix

`ensure_op_for_weight_locked` and `ggml_ryzenai_impl_preload_weight`
now return `bool`:

- `true`  → a new instance was created and weights were uploaded
- `false` → cache hit, or tensor failed `can_mul_mat`

The cgraph walker tracks both `eligible` (claimed by NPU) and
`uploaded` (actually new), and only emits the INFO line when
`uploaded > 0`:

```
ggml-ryzenai: preloaded 64 new weight tensor(s) (64 eligible in cgraph)
```

After the first `graph_reserve` call, subsequent ones stay silent — the
log honestly reflects that no work was redone.

## Open work (not in this scaffold)

- Memory bloat: each Q4_0 weight is currently materialized as a full
  FP32 unpack buffer (~8× model size) before BF16 conversion + DMA.
  Cause of the observed ~17 GB resident for a 2.1 GB Phi-3-mini model.
  Future optimization: go Q4_0 → BF16 directly and free intermediates
  after upload.
- Coverage: only tested with Phi-3-mini Q4_0 on Strix HX 370. Llama-3,
  Gemma, or different `ne0` dims may require additional xclbins.
- Q4_K not supported (no qlinear_2 kernel for it in this SDK drop).
- Transparent CPU fallback when NPU is unavailable (currently we abort
  fail-fast).

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

---

# Prefill vs Decode: same layers, different M

A common confusion when reasoning about NPU performance is the difference
between prefill and decode. They use the **same model, same layers, same
weight tensors**, in the **same sequential order**. The only thing that
changes is the batch dimension on the activations.

## Layers are always sequential. Tokens within a step are parallel.

Both prefill and decode walk the layers in the same order:
`layer 0 → 1 → 2 → ... → 31 → output head`. You cannot start layer N+1
until layer N's output is ready, because each layer feeds the next.

The difference is **how many tokens flow through each layer per pass**.

## The actual data flow

For a Phi-3-mini layer, the main matmul has shape:

```
Y = X @ Wᵀ
where:
   Wᵀ  has shape (K, N)   ← weight tensor (Q4_0, fixed across all calls)
   X   has shape (M, K)   ← activations (variable, depends on call)
   Y   has shape (M, N)   ← output
```

- `K`, `N` are model dimensions (fixed, e.g. 3072 and 8192)
- `M` is the number of tokens being processed in this single matmul call

### Prefill case

User submits a 578-token prompt. The model processes those tokens
**together** in batches:

- `M = 512` for the first chunk, `M = 66` for the remainder
- Layer 0: `X` has shape `(512, 3072)` → one MUL_MAT call → `Y` has shape `(512, 8192)`
- Layer 1: same shape in, same shape out
- ... 32 layers ...
- Total NPU calls for one prefill chunk: 2 calls/layer × 32 layers = **64 calls** at `M = 512`

The weight `Wᵀ` is read **once per layer** but is reused across all 512
tokens in `M`. That's the high arithmetic intensity case: each weight
byte produces 512 multiply-adds before being discarded.

### Decode case

After prefill, we generate output tokens one at a time:

- `M = 1` (just the one new token we're predicting)
- Layer 0: `X` has shape `(1, 3072)` → one MUL_MAT call → `Y` has shape `(1, 8192)`
- ... 32 layers ...
- Total NPU calls per output token: 2 × 32 = **64 calls** at `M = 1`

Same number of calls, same weight tensors, but each call now uses each
weight byte for only **1 multiply-add**. Arithmetic intensity is `512x`
lower than prefill.

## Why "shared weights" matters

The weight tensor `Wᵀ` for layer 7's `Wq` (for example) is **the same
object in memory** whether we're doing prefill or decode. That's why our
`preload_weights` call during `graph_reserve` covers both paths — we
upload the same 64 weight tensors once, then both `M = 512` prefill and
`M = 1` decode matmuls reuse them.

## Why NPU loves prefill and hates decode

The qlinear_2 measurement on Strix HX 370 showed per-call NPU overhead
(kernel launch + DMA descriptor setup + AIE program start + completion
signal) is **~2.26 ms regardless of M**:

| Case    | M   | NPU calls per output token | Real compute per call | Useful work per 2 ms overhead |
|---------|-----|---------------------------|-----------------------|-------------------------------|
| Prefill | 512 | 64 / 512 = 0.125          | huge                  | excellent — pipeline saturated |
| Decode  | 1   | 64                        | tiny                  | terrible — pipeline ~empty, overhead dwarfs compute |

For decode, you pay `64 × 2.26 ms = 145 ms` in overhead alone, before any
compute. For prefill of 512 tokens, the same 64 calls × 2.26 ms is
spread across 512 tokens, so **0.28 ms/token** in overhead.

## Visual analogy

Think of the NPU pipeline like a long conveyor belt at a factory:

- **Prefill** = 512 boxes arrive in a truck. Load them all onto the belt,
  belt runs full for a while, 512 boxes come off the other end. Fixed
  setup + teardown cost is amortized over 512 outputs.
- **Decode** = 1 box arrives. Spin up the belt, run it (mostly empty), 1
  box exits, spin it down. Repeat 64 times per output token.

The CPU is more like a hand assembly bench — no startup cost, but
limited throughput. For 1 box at a time, the bench beats the factory.

## Consequence for our gating heuristic

This is exactly why commit `92e247a15` introduced `GGML_RYZENAI_MIN_M`
(default = 2). At `M = 1`, the NPU is fundamentally the wrong tool for
the job on this generation of XDNA. Route decode to the CPU's AVX-512
Q4_0 GEMV path (which keeps weights in cache and has zero per-call
overhead) and reserve the NPU for batched prefill where the pipeline can
actually be filled.



---

# Results table: GGML_RYZENAI_MIN_M sweep

Phi-3-mini Q4_0, 574-token prompt, 8 threads, Strix HX 370.

| Metric            | MIN_M=1 (legacy, everything to NPU) | MIN_M=2 (default, decode to CPU) | Δ            |
|-------------------|-------------------------------------|----------------------------------|--------------|
| Prefill           | 53.3 t/s                            | 51.9 t/s                         | -2.6% (noise)|
| **Decode**        | **7.93 t/s**                        | **18.82 t/s**                    | **+137%**    |
| Total wall time   | 18.6 s                              | 14.3 s                           | -23%         |
| NPU calls         | 4096                                | 124                              | 33x fewer    |
| NPU execute total | 10.78 s                             | 5.81 s                           | -46%         |

Stats output with default (`MIN_M=2`):

```
[ggml-ryzenai stats]
  calls           = 124  (M=1: 0, M>1: 124)
  alloc bf16 buf  =    27.45 ms  ( 0.47%, avg 221.38 us/call)
  F32->BF16 conv  =    14.58 ms  ( 0.25%, avg 117.61 us/call)
  NPU execute()   =  5806.99 ms  (99.28%, avg 46830.56 us/call)
  total measured  =  5849.02 ms
  activations DMA =   208.52 MB
```

Notice how the average `NPU execute()` time per call jumped from
~2.6 ms (at M=1) to ~46.8 ms (at M=512). The wall-clock cost per call
went up by 18x, but the per-output-token compute went down by 33x
because each call now produces 512 tokens of output instead of 1.
This is the dataflow architecture finally being used as intended.

---

# ONNX RyzenAI (AMD official path) head-to-head

AMD ships a separate NPU path through ORT-genai with a custom op
(`onnx_custom_ops.dll`) and a precompiled model bundle
(`prefill.bin`, `dd_metastate_Llm_Token_MatMulNBits_2_0.*`).
That path is closed-source but we can measure it.

Comparison on the exact same prompt (557-token version of
`single_prompt.txt`, Phi-3-mini-4k, Strix HX 370):

| Metric                  | qlinear_2 (ours, MIN_M=2) | ONNX/AMD       | Ratio (AMD / ours) |
|-------------------------|---------------------------|----------------|--------------------|
| Prefill t/s             | 51.9                      | **641.6**      | **12.4x**          |
| Prefill total           | 11,053 ms                 | 868 ms         | -                  |
| Decode steady-state     | 18.8 t/s (MIN_M=2)        | 12.6 t/s       | 0.67x (we are faster) |
| Decode (legacy MIN_M=1) | 7.9 t/s                   | 12.6 t/s       | 1.59x (they win)   |
| Decode jitter (std-dev) | unmeasured                | ~1.5 ms (2%)   | extremely flat     |
| Working set peak        | ~4.7 GB (est)             | 6.5 GB         | 1.38x              |
| Private bytes peak      | -                         | 6.95 GB        | -                  |

## Where AMD's prefill win comes from

Our path makes 124 separate M>1 NPU dispatches per prefill
(~47 ms each), totalling 5.8 s of pure NPU compute time, plus
~5 s of host overhead (F32->BF16 staging, transitions, sync) =
~11 s total.

AMD's prefill is 868 ms total. Even being generous about how
much of that is NPU vs host, the structural difference is:
they bundle the per-layer matmuls into far fewer NPU dispatches
(likely 1 mega-dispatch per layer or per token group, batched
across Q/K/V/O/gate_up/down). The precompiled
`dd_metastate_Llm_Token_MatMulNBits_2_0.fconst` (2.04 GB)
contains all matmul weights pre-arranged for one fused tile graph.

This is a build-time fusion done by their RAI toolchain — not
something we can replicate at runtime in llama.cpp without
equivalent tooling. The only available knob on our side is to
batch dispatches more aggressively in `ggml-ryzenai-impl.cpp`.

## Where AMD's decode win comes from

Our MIN_M=2 default path keeps decode on CPU (18.8 t/s), so we
already beat AMD's 12.6 t/s on decode by 1.5x in that mode.

For the apples-to-apples comparison of NPU decode only
(MIN_M=1), AMD wins by 1.59x. The gap is per-token NPU dispatch
count: our path issues ~64 M=1 dispatches per token (avg 0.77 ms
each) for ~49 ms of pure NPU + ~77 ms of overhead = ~126 ms.
Their path appears to do ~32 dispatches (one per layer) with the
matmuls fused inside each.

## Memory tradeoff

AMD's 6.5 GB working set vs our 4.7 GB is the cost of the
duplicated weight blobs: `prefill.bin` (2.05 GB) and
`fconst` (2.04 GB) are two copies of the int4 weights laid out
differently for prefill vs decode kernels.  Private bytes
of 6.95 GB are committed up front during prefill and don't grow
during decode.

## Strategic takeaway

- **Don't try to match AMD's prefill** by hand-fusing operations.
  Their advantage is a multi-month NPU compiler investment we
  can't backfill in software. CPU prefill on 8 threads with
  `--repack-xbox` is 109.8 t/s — slower than NPU but acceptable.
- **Our decode story is already good** at MIN_M=2 (18.8 t/s,
  faster than AMD).  Keep CPU on decode by default; only ever
  send M>1 prefill work to NPU.
- **The natural niche for our NPU backend** is contended-CPU
  scenarios (gaming, multi-process) where the 8 cores AMD
  benchmarks against aren't actually available. There, even a
  partial NPU contribution is a net win.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

---

# Architecture: how MLADF, xclbin, XRT, and our backend fit together

This is the stack our `ggml-ryzenai` backend sits in. Boxes are
software layers (top = our code, bottom = silicon). Vertical arrows
are calls; horizontal arrows are data movement.

```
+--------------------------------------------------------------------------+
|  llama.cpp graph (ggml_cgraph)                                           |
|  ggml_tensor src0 (Q4_0 weight)  + src1 (F32 act)  -> dst (F32)          |
+----------------------------------|---------------------------------------+
                                   | ggml_backend_sched dispatches
                                   v
+--------------------------------------------------------------------------+
|  ggml-ryzenai backend  (ggml/src/ggml-ryzenai/)                          |
|                                                                          |
|  - can_mul_mat()        : eligibility check (Q4_0, dims, M>=MIN_M)       |
|  - preload_weight()     : unpack Q4_0 -> transposed int4 + fp32 scales   |
|                           (fused, single pass; scratch freed eagerly)    |
|  - mul_mat_npu() hot path:                                               |
|        1. allocate bf16 staging buf for activations                      |
|        2. F32 -> BF16 conversion (CPU AVX-512)                           |
|        3. qlinear_2.execute()             <----- NPU dispatch            |
|        4. [MLADF only] BF16 -> F32 output conversion                     |
|  - RyzenAIContext: per-tensor qlinear_2 instances, lifetime owned        |
|  - env: GGML_RYZENAI_MIN_M, GGML_RYZENAI_MLADF, GGML_RYZENAI_STATS       |
+----------------------------------|---------------------------------------+
                                   | C++ template call
                                   v
+--------------------------------------------------------------------------+
|  qlinear_2  (AMD RyzenAI SDK, header-only template)                      |
|  qlinear_2<InT=int16_t, WtT=int8_t, AccT, OutT>                          |
|                                                                          |
|  Constructor takes runtime dtype strings + reads env vars:               |
|     DEVICE = "stx"                                                       |
|     MLADF  = "" | "4x4" | "2x4x4"                                        |
|                                                                          |
|  These pick the xclbin filename from a string-keyed table:               |
|                                                                          |
|   +-------------------+------------------------------------------------+ |
|   |  MLADF env var    |  xclbin loaded                                 | |
|   +-------------------+------------------------------------------------+ |
|   |  (unset)          |  gemm_4x4_a16fw4acc32f.xclbin                  | |
|   |  4x4              |  mladf_gemm_4x4_a16fw4acc16f.xclbin            | |
|   |  2x4x4            |  mladf_gemm_2x4x4_a16fw4acc16f.xclbin          | |
|   +-------------------+------------------------------------------------+ |
|                                                                          |
|  weights_bo_  : std::vector<xrt::bo>   <-- NPU-side packed weights       |
|  initialize_weights_int4() : host scratch -> BO, then host can free      |
|  execute()    : stage activations -> NPU compute -> writeback            |
+----------------------------------|---------------------------------------+
                                   | XRT C++ API
                                   v
+--------------------------------------------------------------------------+
|  XRT  (Xilinx Runtime, xrt_coreutil.dll)                                 |
|                                                                          |
|  xrt::device   -> opens NPU device handle                                |
|  xrt::xclbin   -> loads kernel binary onto NPU                           |
|  xrt::kernel   -> handle for one kernel entry point                      |
|  xrt::bo       -> DMA-mapped buffer (host <-> NPU memory)                |
|  xrt::run      -> a single dispatch (start / wait / get_state)           |
+----------------------------------|---------------------------------------+
                                   | Driver ioctl / shared memory
                                   v
+--------------------------------------------------------------------------+
|  AMDXDNA driver (Windows: amdxdna.sys)                                   |
|  - Submits command packets to NPU hardware queue                         |
|  - Manages NPU memory partitions                                         |
|  - Routes interrupts back to userspace XRT                               |
+----------------------------------|---------------------------------------+
                                   |
                                   v
+--------------------------------------------------------------------------+
|  Strix HX 370 NPU silicon  (XDNA2 / AIE-ML architecture)                 |
|                                                                          |
|  +----------------------------------------------------------------+      |
|  | SHIM row     | DMA to/from system memory                       |      |
|  +----------------------------------------------------------------+      |
|  | MEM tile row | L2 scratchpad (SRAM)                            |      |
|  +----------------------------------------------------------------+      |
|  | AIE-ML grid  | 4x4 = 16 cores  (gemm_4x4_*, mladf_4x4_*)       |      |
|  |              | OR                                              |      |
|  |              | 2x(4x4) = 32 cores (mladf_2x4x4_*)              |      |
|  | per core: VLIW + vector unit, local SRAM, stream switch        |      |
|  +----------------------------------------------------------------+      |
+--------------------------------------------------------------------------+
```

## What MLADF actually changes

MLADF (ML-Accelerated Dataflow) is a different mapping of GEMM onto
the AIE-ML grid. Three things change vs the default kernel:

1. **Tile geometry**
   - default `gemm_4x4_*`: 16 AIE cores arranged 4x4
   - `mladf_4x4_*`        : 16 AIE cores, different tile shape and
                            dataflow tuned for LLM matmul aspect ratios
   - `mladf_2x4x4_*`      : **32 AIE cores** (two 4x4 groups working in
                            parallel on different blocks of one matmul)

2. **Output dtype**
   - default: bf16 -> fp32 inside NPU, fp32 written back to host
   - mladf  : bf16 -> bf16 written back (half the writeback bandwidth)
   - Our backend must do BF16 -> F32 on CPU before handing to ggml

3. **No host-side accumulation along K**
   - The kernel does full K accumulation in NPU SRAM, no chunking
     across multiple AIE dispatches
   - Simplifies driving the kernel but requires K to fit in the
     mapped tile shape

The 2x4x4 variant is the headline: **doubling core count** with the
same xclbin loader code. Strix HX has the silicon; we just have to
ask for it via the right xclbin.

## What we still do not control

Layers below qlinear_2's xclbin selection are sealed AMD/Xilinx
artifacts:

- The `.xclbin` itself is a packaged set of AIE micro-code + DMA
  descriptors + stream-switch routing tables. Building a new one
  requires the MLIR-AIE / IRON toolchain (open source, but a
  multi-week investment).
- The XRT runtime and amdxdna driver are AMD-shipped binaries.

What we DO control:

- Which xclbin we ask qlinear_2 to load (via DEVICE / MLADF env)
- Whether dispatches are synchronous or pipelined
  (potential future change to use xrt::run::start() async pattern)
- How we batch ggml ops into qlinear_2 calls
- Whether we feed bf16-out or fp32-out (MLADF requires bf16-out)
- The Q4_0 -> int4 + scales unpack format (already optimized)

## Usage: enabling MLADF in the ggml-ryzenai backend

The patch wires `GGML_RYZENAI_MLADF` to qlinear_2's MLADF/DEVICE env vars
and instantiates `qlinear_2<int16_t, int8_t, int16_t, int16_t>` (bf16 out)
instead of the default `<int16_t, int8_t, float, float>` (fp32 out). A
BF16 -> F32 conversion is added on the host before writeback to ggml.

```cmd
:: default path (unchanged from before)
minslm-cli -m model.gguf -p "hi" -n 64

:: MLADF, 16 cores, bf16 output
set GGML_RYZENAI_MLADF=4x4
minslm-cli -m model.gguf -p "hi" -n 64

:: MLADF, 32 cores (Strix only — needs 4-col AIE-ML grid x 2)
set GGML_RYZENAI_MLADF=2x4x4
minslm-cli -m model.gguf -p "hi" -n 64

:: turn on per-call stats to compare
set GGML_RYZENAI_STATS=1
```

Prereqs already required by the SDK and unchanged by this patch:

- `PYTORCH_AIE_PATH` must point at the RyzenAI SDK install root so
  qlinear_2 can locate xclbins under `xclbin/stx/`
- AMD XRT runtime + amdxdna driver installed; xrt_coreutil.dll on PATH
- Strix HX 370 (or any XDNA2 part with the 4-col AIE-ML grid for 2x4x4)

What the stats summary adds:

```
[ggml-ryzenai stats]
  BF16->F32 out conv  =   xx.xx ms  ( y.yy%, avg z.zz us/call)
```

This is the only host-side cost MLADF adds. If the line shows >5% of
total it's a hint to vectorize the conversion (AVX-512 `_mm512_slli_epi32`
on `cvtepu16_epi32` — left as future work).

If `GGML_RYZENAI_MLADF` is set to anything other than `4x4` or `2x4x4`
the backend logs a warning and silently falls back to the default path.
