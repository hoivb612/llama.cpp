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

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Sample build and session run:

C:\llama.cpp\b612_052026\build.ryzenai>git branch
  hv/b612_052026
* hv/b612_052026_ryzenai

C:\llama.cpp\b612_052026\build.ryzenai>cmake .. -DGGML_RYZENAI=ON -DRyzenAI_DIR=C:/ProgramData/anaconda3/envs/ryzenai-transformers/Lib/cmake/ryzenai -DXRT_DIR=c:/llama.cpp/Ryzen/example/transformers/third_party/xrt-ipu/xrt/share/cmake/XRT                                

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

C:\llama.test\RyzenAI>minslm-cli.exe models\Phi-3.bin 4 prompts\cpf_2_Phi-3.txt v2
> Running with custom prompt => [18/18]: [I don't care, go away!]
...
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
...
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
ggml-ryzenai: preloaded 64 new weight tensor(s) (64 eligible in cgraph)
sched_reserve: Flash Attention was auto, set to enabled
sched_reserve: resolving fused Gated Delta Net support:
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
sched_reserve: fused Gated Delta Net (autoregressive) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =   16, n_seqs =  1, n_outputs =   16
sched_reserve: fused Gated Delta Net (chunked) enabled
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
graph_reserve: reserving a graph for ubatch with n_tokens =    1, n_seqs =  1, n_outputs =    1
graph_reserve: reserving a graph for ubatch with n_tokens =  512, n_seqs =  1, n_outputs =  512
sched_reserve:        CPU compute buffer size =    98.64 MiB
sched_reserve: graph nodes  = 999
sched_reserve: graph splits = 65
sched_reserve: reserve took 29310.20 ms, sched copies = 1

[main]: n_ctx = 2048, n_threads = 4, gpu_layers = 999
[main]: flash-attn = auto (runtime-selected)
[main]: system_info: CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | AVX512 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
...
> Running with custom prompt => [18/18]: [I don't care, go away!]
  prefill: 14621.3ms (574 tokens, 39.3 t/s)
 {
    "answer": "e. Are you trying to avoid the question?",
    "justification": "Mara's response indicates she may not want discuss this topic, but it doesn’t directly force her about Izzy."
}
  [all token ids (53): 426 13 1678 376 12011 1115 376 29872 29889 4683 366 1811 304 4772 278 1139 29973 613 13 1678 376 5143 2450 1115 376 29924 2518 29915 29879 2933 14088 1183 1122 451 864 5353 445 11261 29892 541 372 1838 30010 29873 4153 4889 902 1048 15674 1537 1213 13 29913]
  [output hex (196 bytes):]
    0000: 20 7b 0a 20 20 20 20 22 61 6e 73 77 65 72 22 3a  | {.    "answer":|
    0010: 20 22 65 2e 20 41 72 65 20 79 6f 75 20 74 72 79  | "e. Are you try|
    0020: 69 6e 67 20 74 6f 20 61 76 6f 69 64 20 74 68 65  |ing to avoid the|
    0030: 20 71 75 65 73 74 69 6f 6e 3f 22 2c 0a 20 20 20  | question?",.   |
    0040: 20 22 6a 75 73 74 69 66 69 63 61 74 69 6f 6e 22  | "justification"|
    0050: 3a 20 22 4d 61 72 61 27 73 20 72 65 73 70 6f 6e  |: "Mara's respon|
    0060: 73 65 20 69 6e 64 69 63 61 74 65 73 20 73 68 65  |se indicates she|
    0070: 20 6d 61 79 20 6e 6f 74 20 77 61 6e 74 20 64 69  | may not want di|
    0080: 73 63 75 73 73 20 74 68 69 73 20 74 6f 70 69 63  |scuss this topic|
    0090: 2c 20 62 75 74 20 69 74 20 64 6f 65 73 6e e2 80  |, but it doesn..|
    00a0: 99 74 20 64 69 72 65 63 74 6c 79 20 66 6f 72 63  |.t directly forc|
    00b0: 65 20 68 65 72 20 61 62 6f 75 74 20 49 7a 7a 79  |e her about Izzy|
    00c0: 2e 22 0a 7d                                      |.".}|
  decode: 6348.6ms (52 tokens, 8.19 t/s)
  decode (no-TTFT): 6348.5ms (53 tokens, 8.35 t/s) [TTFT 0.1ms]

===== Summary =====
  prompts:    18
  tokens gen: 1068
  avg t/s:    8.29
  avg t/s (no-TTFT): 8.43
  avg TTFT:          0.1ms
  total time: 385.81s
llama_perf_context_print:        load time =   44746.95 ms
llama_perf_context_print: prompt eval time =  256322.22 ms / 10347 tokens (   24.77 ms per token,    40.37 tokens per second)
llama_perf_context_print:        eval time =  128586.53 ms /  1068 runs   (  120.40 ms per token,     8.31 tokens per second)
llama_perf_context_print:       total time =  415872.67 ms / 11415 tokens
llama_perf_context_print:    graphs reused =       1050
process_memory: peak working_set=4915.9 MiB, current working_set=4912.8 MiB, private=4987.8 MiB
~llama_context:        CPU compute buffer size is  98.6387 MiB, matches expectation of  98.6387 MiB
~llama_context:        CPU compute buffer size is   0.0000 MiB, matches expectation of   0.0000 MiB
