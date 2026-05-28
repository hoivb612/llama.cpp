# ggml-ryzenai — status summary

Branch: `hv/b612_052026_ryzenai` (the parallel branch `hv/b612_052026`
does NOT carry the RyzenAI backend; the two diverge only on this
backend plus a couple of wxemem-only commits on the main branch).

For the full design rationale, perf tables, and architecture diagram,
see `__readme_ryzenai__.md` at the repo root. This file is the
operator-facing pointer: what works today, what we tried, what's
still open.

## What this backend is

A `ggml-backend` implementation that offloads eligible Q4_0 weight x
F32 activation `mul_mat` ops to the AMD XDNA NPU on Phoenix /
Strix Point / Strix Halo. Built on AMD's RyzenAI SDK 1.6 via the
`qlinear_2` header-only template.

Strategic niche: **contended-CPU scenarios**. When all CPU cores are
free, AVX-512 + Q4_0 repack beats the NPU on this workload. The NPU
wins when the CPU is busy with other work (foreground apps, multiple
inference streams, background tasks). We are not chasing absolute
peak throughput on a quiet machine.

## Current state — what works

- Q4_0 x F32 `mul_mat` offload via `qlinear_2<int16_t, int8_t, float, float>`
- Fused Q4_0 unpack + transpose at weight preload time
- Weight preload hooked into `llama_context::graph_reserve` so all
  weight uploads happen before the first decode (no first-token jank)
- `GGML_RYZENAI_MIN_M=2` default — M=1 (decode) stays on CPU because
  NPU per-call overhead dominates at M=1
- `GGML_RYZENAI_STATS=1` per-call timing summary at process exit,
  split by M bucket (M=1 vs M>1)
- Emulation build path (`-DRYZENAI_EMULATION`) for development on
  non-NPU hosts; falls back to scalar GEMM

## Honest perf vs AMD's official ONNX/VitisAI path

Phi-3-mini-4k Q4_0, 557-token prompt, Strix HX 370:

|                 | Prefill (t/s) | Decode steady (t/s) | Peak WS (MB) |
|---|---|---|---|
| ours, MIN_M=1   | ~52  | 7.9  | ~4700 |
| ours, MIN_M=2   | ~52  | **18.8** | ~4700 |
| AMD run_llm     | **642** | 12.6 | 6500  |

- Prefill: AMD wins by ~12x. This is structural (offline graph
  fusion via VitisAI EP + custom `.fconst`). Not closeable in
  software.
- Decode steady-state: our MIN_M=2 default (decode on CPU) beats AMD
  by ~1.5x.
- Memory: AMD's stack carries a ~1.8 GB tax from shipping prefill +
  decode as separate packed binaries.

## MLADF investigation — outcome (this session)

Added a behind-an-env-var MLADF code path:

```cmd
set GGML_RYZENAI_MLADF=4x4     :: 16 AIE cores, bf16 output
set GGML_RYZENAI_MLADF=2x4x4   :: 32 AIE cores, bf16 output
```

Three commits (local on `hv/b612_052026_ryzenai`, not pushed):

1. `16a6d9e8a` — dual `op_t_f32`/`op_t_bf16` instantiations, env-var
   wiring (`GGML_RYZENAI_MLADF` -> qlinear_2's `MLADF` + `DEVICE=stx`),
   bf16->f32 output conversion on the host
2. `241152729` — call `initialize_weights_int4_mladf` directly (the
   non-mladf `initialize_weights_int4` only forwards
   `set_kernel_shapes_kn` to the mladf variant; the rest of the body
   uses the non-mladf QuantMatrix layout -> corrupts BOs)
3. `a50622f20` — pad activation K up to the kernel's K-bucket
   (Phi-3 K=3072/8192 vs MLADF Llama-2 buckets K=4096/11008;
   qlinear_2 refuses to do this for us at execute() time)

### Why MLADF doesn't yet help us — the blocker

MLADF kernels are dispatched via precompiled DPU instruction binaries
named `mladf_<grid>_a16fw4acc16f_<M>_<K>_<N>_<grp>.bin`. The SDK
(Ryzen AI 1.6) ships these for two group sizes:

| grp_size | shapes shipped (4x4 overlay)                                       |
|---|---|
| **128**  | (4096,4096), (4096,12288), (4096,22528), (4096,32768), (11008,4096), (256,2048) |
| **32**   | (4096,32768), (256,2048)   <-- lm_head + debug only                |

**Q4_0 has QK4_0 = 32**, so qlinear_2 builds keys ending in `_32.bin`.
For every Phi-3 layer except lm_head, no `_32.bin` exists, and we get:

```
ggml-ryzenai exception: Failed to get instruction buffer for key:
  mladf_4x4_a16fw4acc16f_128_4096_12288_32.bin
```

This is an architectural mismatch, not a bug. AMD ships their models
with custom grp=128 packing (the `.fconst` blob in their model dir)
specifically to hit the grp=128 binaries.

### What MLADF would require to actually run on a Q4_0 gguf

Three options were considered; all left on the shelf:

- **Hybrid lm_head-only MLADF** — flip MLADF on for just the
  (4096, 32768) shape; one large matmul per token, real but modest
  win, no quality hit. ~1 day of work.
- **Lossy regroup at preload** — collapse 4 Q4_0 grp=32 scales into
  1 grp=128 scale (max-abs + rescale int4 codes). Enables MLADF
  everywhere but adds quantization noise. Quality TBD.
- **Real regroup** — quantize the model at gguf-creation time with
  grp=128. Requires a new ggml quant type and re-shipping the model.

### What did get answered

- MLADF init wiring works end to end (xclbin load, weight BO format,
  kernel dispatch) on Strix HX 370.
- The K-padding workaround is correct (compiles, links, runs to the
  point of binary lookup).
- The block above the kernel (env var -> MLADF mode -> xclbin path)
  is verified.

So if a grp=128 model ever lands, the existing MLADF code in this
backend will just work — `GGML_RYZENAI_MLADF=4x4` flips it on.

## Files

| Path | Purpose |
|---|---|
| `ggml-ryzenai.cpp` | ggml-backend interface (registration, alloc, compute) |
| `ggml-ryzenai-impl.cpp` | NPU dispatch hot path, qlinear_2 wrappers, MLADF wiring |
| `ggml-ryzenai-impl.h` | Internal API between .cpp files |
| `CMakeLists.txt` | Build rules; sets `RYZENAI_EMULATION` when SDK is absent |
| `__summary__.md` | This file |

## Environment variables

| Var | Default | Effect |
|---|---|---|
| `GGML_RYZENAI_MIN_M` | `2` | Minimum M to claim mul_mat. `2` keeps decode on CPU. |
| `GGML_RYZENAI_STATS` | unset | When set, prints per-call timing summary at exit |
| `GGML_RYZENAI_MLADF` | unset | When `4x4` or `2x4x4`, enables MLADF path (blocked by grp-size mismatch on Q4_0 today; see above) |
| `PYTORCH_AIE_PATH` | (required by SDK) | Path to RyzenAI SDK install root (xclbins resolved under `xclbin/stx/`) |

## Open items

- Hybrid lm_head-only MLADF (cheap experiment, no quality hit)
- Lossy regroup MLADF (would tell us the upper-bound speed for
  general Q4_0 on this NPU)
- SwiGLU MLP fusion via `mladf_4x4_gemm_silu_mul_a16fw4.xclbin`
  (needs ggml graph pattern recognition; only useful once MLADF
  is actually doing work)
- AVX-512 vectorization of bf16->f32 output conversion (only matters
  if MLADF is active)
- Port instrumentation / numbers to RyzenAI SW 1.7 once available
