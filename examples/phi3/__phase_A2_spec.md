# A2 Spec — Phi-3 custom decode/prefill forward (Pattern C)

Target model: Phi-3-mini-4k-instruct Q4_K_M (resolved in A0).

**Acceptance bar (yes/no): 3 prompts × 256 greedy tokens must be byte-exact
token-identical to the baseline with `flash_attn=false`, seed=1234, temp=0,
min_p=0.** Top-2 logit margin at divergence is *diagnostic only* — not a
relaxation of the bar. If we cannot pass this bar we fix the bug; we do not
weaken the bar. [duck r2 #6]

Perf: parity (±5%) acceptable for decode; speedup belongs in A3.

Out of scope (A2) — **enforced via explicit init-time rejection, not silent
fallback**: SWA, GQA, MoE, all bias terms, LoRA, control vectors, all weight
scale tensors (`*_s`, `*_in_s`), KQV clamp, attention soft-cap, ALiBi,
flash-attention, YARN long-context, non-greedy sampler, non-generation
context (embeddings/non-causal/multi-seq). The oracle must run with
`cparams.flash_attn=false`. [duck r1 #4, r1 #6, r1 #9, r2 #4, r2 #5, r2 suggestion #2]
See §1.5 for the full rejection list.

---

## 0. Math (single source of truth)

A single forward step at position `pos` with input token `tok` does:

```
x = tok_embd[tok, :]                               # F32 [n_embd]    (dequant of one row)

for il = 0 .. n_layer-1:
    residual = x
    h = rms_norm(x, attn_norm[il], eps)            # F32 [n_embd]
    qkv = wqkv[il] · h                             # F32 [n_qkv]      n_qkv = 3*n_embd  (no GQA)
    Q, K, V = qkv[0:n_embd], qkv[n_embd:2n], qkv[2n:3n]    # F32 [n_head, head_dim] each
    Q = rope_neox(Q, pos, rope_factors)            # in place
    K = rope_neox(K, pos, rope_factors)            # in place
    KV[il].K[:, :, pos] = K                        # F16 store
    KV[il].V_T[pos, :, :] = V                      # F16 store        (V stored transposed)
    Q *= 1.0 / sqrt(head_dim)
    # Attention over KV positions [0 .. pos]
    for h in 0 .. n_head-1:
        scores[0..pos] = dot(Q[h, :], KV.K[:, h, 0..pos])      # F32 [pos+1]
        p[0..pos]      = softmax(scores)                       # F32 [pos+1]
        ctx[h, :]      = sum_{k=0..pos} p[k] * KV.V_T[k, h, :] # F32 [head_dim]
    attn = wo[il] · ctx                            # F32 [n_embd]
    x = residual + attn
    residual = x
    h = rms_norm(x, ffn_norm[il], eps)             # F32 [n_embd]
    up_gate = ffn_up[il] · h                       # F32 [2*n_ff]     gate | up fused on dim 0
    ff = silu(up_gate[0:n_ff]) * up_gate[n_ff:2n_ff]   # F32 [n_ff]
    out = ffn_down[il] · ff                        # F32 [n_embd]
    x = residual + out

h = rms_norm(x, output_norm, eps)                  # F32 [n_embd]
tok_next = argmax_v ( output[:, v] · h )           # greedy
```

Prefill is the same loop run for `n_tokens` positions in sequence, with all
KV writes happening per step. (Vectorized prefill — batched matmuls — belongs
in A3.)

---

## 1. Public API surface (added to `phi3_fused_graph.h`)

### 1a. Sub-structs (avoid one mega-struct) [duck r2 suggestion]

```cpp
// RoPE configuration, resolved once at init from cparams/hparams. [duck r1 #7]
struct Phi3RoPEConfig {
    int32_t       n_rot        = 0;
    float         freq_base    = 0.0f;
    float         freq_scale   = 0.0f;
    float         ext_factor   = 0.0f;
    float         attn_factor  = 0.0f;
    float         beta_fast    = 0.0f;
    float         beta_slow    = 0.0f;
    const float * rope_factors = nullptr;  // may be NULL → unit factors (1.0)
    int32_t       n_ctx_orig   = 0;
};

// Per-step scratch buffers. Allocated once at init, reused every step.
// All F32 except hq/ffq which hold quantized inputs to matmuls. [duck r1 #2]
struct Phi3ForwardScratch {
    // Quantization types resolved from weight traits.
    ggml_type            q_type_attn = GGML_TYPE_COUNT;
    ggml_type            q_type_ffn  = GGML_TYPE_COUNT;

    std::vector<float>   x_buf;       // [n_embd]
    std::vector<float>   h_buf;       // [n_embd]
    std::vector<uint8_t> hq_buf;      // [row_size(q_type_attn, n_embd)]
    std::vector<uint8_t> ffq_buf;     // [row_size(q_type_ffn, n_ff)]
    std::vector<float>   qkv_buf;     // [n_qkv]
    std::vector<float>   ctx_buf;     // [n_embd]
    std::vector<float>   scores_buf;  // [ctx_max]
    std::vector<float>   upgate_buf;  // [2*n_ff]
    std::vector<float>   ff_buf;      // [n_ff]

    // F32-debug mode: per-layer TRANSIENT F32 mirrors. We allocate space for
    // ONE layer's dequantized weights, populate before processing each layer,
    // and overwrite for the next. lm_head is streamed in vocab-row chunks
    // (no full F32 mirror). This keeps peak memory bounded to ~150 MiB at
    // mini-4k Q4_K_M (one layer worth: norm + wqkv + wo + ffn_norm + ffn_up +
    // ffn_down) rather than 24 GiB for the whole model. [duck r3 #1]
    bool                 f32_debug   = false;
    std::vector<float>   w_f32_attn_norm;  // [n_embd]               one layer
    std::vector<float>   w_f32_wqkv;       // [n_embd * n_qkv]       one layer
    std::vector<float>   w_f32_wo;         // [n_embd * n_embd]      one layer
    std::vector<float>   w_f32_ffn_norm;   // [n_embd]               one layer
    std::vector<float>   w_f32_ffn_up;     // [n_embd * 2 * n_ff]    one layer
    std::vector<float>   w_f32_ffn_down;   // [n_ff * n_embd]        one layer
    std::vector<float>   w_f32_vocab_chunk; // [chunk * n_embd]      lm-head chunk
    int32_t              f32_vocab_chunk = 256; // rows per streamed argmax pass
    // Layer index whose mirrors are currently populated (-1 = none).
    int32_t              f32_cached_layer = -1;
};

// Per-context state. Borrows weights & pool; owns KV, scratch, RoPE config.
struct Phi3FusedCtx {
    const Phi3Weights *  w           = nullptr;     // borrowed
    Phi3MatmulPool *     matmul_pool = nullptr;     // borrowed (spin-policy, NOT lm_head pool)
    float                eps         = 1e-5f;

    Phi3RoPEConfig       rope;
    Phi3KV               kv;
    Phi3ForwardScratch   scratch;
    int32_t              cur_pos     = 0;
};
```

bool phi3_fused_ctx_init(
    Phi3FusedCtx &       out,
    const Phi3Weights &  w,
    Phi3MatmulPool *     matmul_pool,            // may be null → serial matmul
    const struct llama_context * lctx,           // for cparams (RoPE, flash_attn, cvec)
    int                  ctx_max,
    bool                 f32_debug,
    std::string &        error);                 // on false return, error names the offending feature

void phi3_fused_ctx_free(Phi3FusedCtx & cx);

// Run a single decode step. Reads cx.cur_pos and KV, writes new K/V at cur_pos,
// advances cur_pos by 1, returns the greedy-argmax token id.
llama_token phi3_fused_decode_step(Phi3FusedCtx & cx, llama_token tok);

// Run prefill: feeds n tokens sequentially, discards intermediate token outputs,
// returns the greedy argmax of the LAST step. Equivalent to n calls to
// phi3_fused_decode_step but signals intent.
llama_token phi3_fused_prefill(Phi3FusedCtx & cx, const llama_token * toks, int n);

// Rewind for multi-turn (re-uses Phi3KV ops).
void phi3_fused_truncate(Phi3FusedCtx & cx, int32_t new_len);
void phi3_fused_drop_range(Phi3FusedCtx & cx, int32_t pos0, int32_t pos1);
void phi3_fused_keep_prefix(Phi3FusedCtx & cx, int32_t keep_len);
```

---

## 1.5 Init-time rejection list [duck r1 #4 #6, r2 #4 #5, r2 suggestion #2]

`phi3_fused_ctx_init` MUST verify every item below and return false with a
specific error message naming the offending feature. The runtime treats a
false return as fatal when `--phi3-fused-forward` was explicitly requested
(no silent fallback). This catches feature combinations the spec doesn't
cover instead of producing garbage tokens.

### Per-layer checks (loop `il = 0 .. n_layer-1`)
| Feature                          | Required state              | If present, oracle    |
|----------------------------------|------------------------------|-----------------------|
| MoE                              | `ffn_gate_inp`, `ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps` all NULL | `src/models/phi3.cpp:319,327-340` |
| QKV bias                         | `attn_qkv.bias` tensor absent | `build_qkv` adds if present |
| Attn output bias                 | resolved `attn_output.bias` absent | `phi3.cpp:292` |
| Attn-norm bias                   | `attn_norm.bias` absent      | `phi3.cpp:264-265` |
| FFN-norm bias                    | `ffn_norm.bias` absent       | `phi3.cpp:313` |
| **All scale tensors**            | `wqkv_s`, `wo_s`, `ffn_up_s`, `ffn_down_s` all absent | `src/llama-model.cpp:1305-1323`, `1420-1427`; applied by `build_lora_mm` `llama-graph.cpp:1001-1003` |
| Per-layer rope_long / rope_short | absent (else we'd need YARN logic) | `model.get_rope_factors()` |
| All weight tensors host-canonical | `w->data != nullptr` and `nb[1] == ggml_row_size(w->type, w->ne[0])` | duck r1 #1 |
| All weight types accepted        | `ggml_get_type_traits_cpu(w->type)->vec_dot_type != GGML_TYPE_COUNT` | duck r1 #2 |

### Global checks
| Feature                          | Required state              | Oracle |
|----------------------------------|------------------------------|--------|
| Output bias                      | `output.bias` absent         | `phi3.cpp:370-372` |
| Output scales                    | `output_s`, `output_in_s` absent | `phi3.cpp:368`, `llama-model.cpp:1305-1323` |
| `output_norm.bias`               | absent                       | final `build_norm(.., output_norm_b, ..)` |
| Token-embed dequantize           | a `dequantize_row_*` is available for `tok_embd->type` | direct |
| GQA                              | `hparams.n_head == hparams.n_head_kv` | implicit via n_embd_gqa |
| SWA                              | `hparams.swa_type == LLAMA_SWA_TYPE_NONE` | `phi3.cpp:165-176` |
| **KQV clamp**                    | `hparams.f_clamp_kqv == 0` | `llama-graph.cpp:1092-1094`, `1112-1133` |
| **Attention soft-cap**           | `hparams.attn_soft_cap == 0` (no tanh cap) | `llama-graph.cpp:2026-2032` |
| **ALiBi**                        | `hparams.f_max_alibi_bias == 0` and `hparams.use_alibi == false` | `llama-graph.cpp:2040`, `llama-model.cpp:1168-1169` |
| LoRA                             | no active LoRA adapters via accessor (see below) | `build_lora_mm` `llama-graph.cpp:977-1005` |
| Control vectors                  | no active cvec via accessor (see below) | `build_cvec` `phi3.cpp:344` |
| Flash attention                  | `cparams.flash_attn == false` | `llama-graph.cpp:1963-1981` |
| YARN long-context                | `cparams.n_ctx_seq <= hparams.n_ctx_orig_yarn` OR `rope_factors_long/short` absent | `llama-model.cpp:2026-2038` |
| **Generation context only**      | `cparams.embeddings == false`, `cparams.causal_attn == true`, `cparams.n_seq_max == 1` | `llama-cparams.h`, `llama-graph.cpp:213-215` |
| Sampler bypass safety            | runtime path must guarantee greedy (caller responsibility) | docs |

### Accessor plan for LoRA + cvec [duck r2 #3]
`lctx->loras` and `lctx->cvec` are not public. Two options:

- **(A)** Add tiny non-mutating accessors to `include/llama_b612.h`:
  - `bool llama_b612_has_active_lora(const llama_context * ctx);`
  - `bool llama_b612_has_active_cvec(const llama_context * ctx);`

  Implementations in `src/llama-context.cpp` read internal fields directly
  (`loras` field at `llama-context.h:273-274`; an entry is only inserted
  for nonzero scale per `llama-context.cpp:1228-1233`, so `!loras.empty()`
  is the right check). For cvec, `cvec.tensor_for(il)` returning non-null
  for any il indicates active (`llama-adapter.cpp:14-19`).

- **(B)** Include the internal `llama-context.h` / `llama-adapter.h` from
  our example. Faster but ties examples/ to internal headers.

→ **Pick (A).** Two-function delta in the public header is the right cost.
Implementation lands as part of A2.2 alongside `phi3_fused_validate_supported`.

Implementation: a single `phi3_fused_validate_supported(model, lctx, error)`
function returns false with the first failing check named. `phi3_fused_ctx_init`
calls it before any allocation.

---

### KV cache (already done in A1)
- `K[il]`  : F16 `[head_dim, n_head_kv, ctx_max]`
- `V_T[il]`: F16 `[ctx_max, n_head_kv, head_dim]`  (positions are innermost)

### Scratch (per-step, allocated once at init)
All F32 except where noted. Sizes are independent of ctx_max except `scores_buf`.

| buffer       | bytes (mini-4k, Q8_K case) | purpose                           |
|--------------|----------------------------|-----------------------------------|
| `x_buf`      | 4 · 3072  = 12,288         | residual stream                   |
| `h_buf`      | 4 · 3072  = 12,288         | post-norm                         |
| `hq_buf`     | row_size(q_type_attn, 3072) ≈ 3360 | quantized input to wqkv/wo |
| `ffq_buf`    | row_size(q_type_ffn, 8192) ≈ 8960 | quantized input to ffn_down |
| `qkv_buf`    | 4 · 9216  = 36,864         | output of wqkv                    |
| `ctx_buf`    | 4 · 3072  = 12,288         | attention output                  |
| `scores_buf` | 4 · 4096  = 16,384         | per-head attention scores         |
| `upgate_buf` | 4 · 16384 = 65,536         | output of ffn_up (gate∥up)        |
| `ff_buf`     | 4 · 8192  = 32,768         | post-SwiGLU                       |
| **total**    | **≈ 200 KiB**              | all stays L2-resident             |

In `f32_debug` mode add `k_f32`/`v_f32` at ctx_max × n_head_kv × head_dim each
(48 MiB at ctx_max=4096 for mini-4k); this allocation only happens when the
debug flag is set.

### Borrowed (not owned)
- `Phi3Weights * w` — set at init, never freed.
- `Phi3MatmulPool * matmul_pool` — optional; if null we run serial.

---

## 3. Per-op kernel mapping

| math op                 | implementation                                            |
|-------------------------|-----------------------------------------------------------|
| `tok_embd[tok, :]` deq  | inline switch on `w->tok_embd->type`; calls type-specific `dequantize_row_*` for ONE row of the token-embed matrix. **Layout: `tok_embd` is `{n_embd, n_vocab}`** (`src/models/phi3.cpp:182`); `ggml_get_rows` indexes the outer dim, so row `tok` lives at `base + tok * nb[1]` and is `n_embd` elements long (`ggml.c:3964`, `ops.cpp:4743-4749`). Assert + tiny test in A2.1. [duck non-blocking #1] |
| `rms_norm + scale*γ`    | inline; **double sum_sq** (matches `ggml_compute_forward_rms_norm_f32`) → multiply by γ in F32 → output F32 |
| quantize for matmul     | `q_traits = ggml_get_type_traits_cpu(q_type)` where `q_type = ggml_get_type_traits_cpu(w->type)->vec_dot_type` (NOT hardcoded Q8_K); call `q_traits->from_float(src_f32, dst, n)`. The output buffer size is `ggml_row_size(q_type, n)`. [duck #2] |
| matmul (K-quant·Q8_K)   | per-output-row loop calling `traits->vec_dot(K, &row_out, 0, w_row_ptr, 0, q_src, 0, 1)` where `traits = ggml_get_type_traits_cpu(w->type)`. Stride `w_row_ptr = w_base + row_i * ggml_row_size(w->type, K)`. |
| RoPE NeoX               | inline; pair `dim_i ↔ dim_{i + n_rot/2}` for `i ∈ [0, n_rot/2)` (`ops.cpp:5900-5903, 5761-5775`). Per-dim θ uses `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow`. **Null factors → divide by 1.0** per `ops.cpp:5677-5681`. [duck non-blocking #3] |
| Q×K dot                 | inline F32 dot of length head_dim against F16 K row (or F32-cached K row in f32_debug mode) |
| softmax                 | inline two-pass (max then exp/sum); plain F32 |
| V combine               | inline `sum_k p[k] * V_T[il][k, h, :]` — contiguous F16 read of head_dim values per k |
| SwiGLU                  | inline `silu(a) * b` over n_ff elements; serial (cheap). **Debug-only assert: no NaN/Inf in output before quantize** (`ops.cpp:3181-3185`). [duck non-blocking #2] |
| argmax (lm_head)        | re-use existing `phi3_fused_lmhead_argmax(..., lm_pool, h_buf, ...)` — note this uses the existing `Phi3LmHeadPool` (cv-park policy is correct for the single end-of-step argmax), distinct from the per-layer `Phi3MatmulPool` (spin policy) introduced in §4 |

### Why we can call ggml internals directly
`ggml/src/ggml-cpu/quants.h` is in the source tree (not just include/) and is
already pulled in by our `phi3_fused_ops.cpp`. The functions are plain C ABI.
We do NOT need to go through `ggml_compute_forward_*` (which expects a
`ggml_compute_params *` with workspace + thread index). We use the inner
vec_dot directly.

### Constants per arch
- `rope_type = LLAMA_ROPE_TYPE_NEOX` for Phi3 (verified `llama-model.cpp:2466`).
- `n_rot = head_dim` for Phi3-mini-4k (= 96). All dims are rotated.
- `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`,
  `beta_slow` come from cparams; we read them at `phi3_fused_ctx_init` time
  and store on `Phi3FusedCtx`. (Mini-4k: defaults — no YARN extension.)

---

## 4. Threading plan (Pattern C)

### What's parallel
Each of these is a `pool->run(...)`-style fork-join over `n_threads_gen` workers:
1. **wqkv matmul** — n_qkv output rows split across workers.
2. **wo matmul**   — n_embd output rows split.
3. **ffn_up matmul** — 2*n_ff rows split.
4. **ffn_down matmul** — n_embd rows split.
5. **lm_head argmax** — re-uses existing `phi3_fused_lmhead_argmax` (same cv-park `Phi3LmHeadPool`; only ONE invocation per token).

### What's serial (on driver)
- Token-embed dequant (one row, trivial).
- RMSNorm (one pass, 3072 elements — ~3 µs).
- Q-quantize of inputs (one row, ~5 µs).
- RoPE (Q+K, 3072 elements — ~2 µs).
- KV write (3072+3072 F16 stores — trivial).
- Per-head attention (32 heads × (dot + softmax + combine)).
- SwiGLU (8192 elements).
- Residual adds.

### Why this is fine for n=1 decode
Driver-serial work is ~80 µs / layer × 32 layers = 2.5 ms (out of ~40 ms per token).
The 4 matmuls per layer are the bottleneck (mostly memory-bound), and those go
to the pool.

### Pool design [duck #3 — REVISED]

We need **two pools** because they have opposing parking policies:

| pool                  | jobs per token | parking policy            | rationale |
|-----------------------|----------------|---------------------------|-----------|
| `Phi3LmHeadPool` (existing) | 1 (lm_head argmax) | cv-park | designed to sleep during the ~40 ms `llama_decode`; one wake/token is cheap |
| `Phi3MatmulPool` (NEW)      | 128 (4×32 per-layer matmuls) | spin policy with bounded park | cv-park × 128 ≈ 0.5 ms wasted/token; need a short-spin handoff |

Earlier spec (Option α) said "one pool, two job_kinds". Duck flagged this as
wrong: the cv-park policy that's correct for lm_head is wrong for 128
fork-joins/token. Either:

- **(α-revised)** Add a `spin_park_us` knob to `Phi3LmHeadPool` and set it
  high (e.g. 200 µs) only while inside fused-forward; reset to 0 before
  exiting. Workers spin for `spin_park_us` after each `done_count++` before
  parking. Pros: one pool. Cons: stateful policy that has to be carefully
  toggled across the lm_head call at end of token.
- **(β-revised)** Add a separate `Phi3MatmulPool` struct that **always** spins
  (workers run a tight `done_count`/`job_seq` loop with `_mm_pause` and a
  bounded backoff timer that parks after, say, 50 ms of idle). Pros: simple,
  independent lifetimes, no policy toggling. Cons: extra ~64 KiB of state,
  extra worker threads competing with the OS scheduler during prefill (when
  we may want them spinning too — see §1 KV-rewind requirement implies own
  prefill too).

**Choice: β-revised.** Dedicated `Phi3MatmulPool` with spin policy. The
toggle approach (α-revised) is brittle — easy to leave the knob in the
wrong state across an early-exit code path. Pool count is bounded by
`runtime.n_threads_gen`, same as the lm_head pool.

`Phi3MatmulPool` definition lives in `examples/phi3/phi3_fused_ops.{h,cpp}`
(adjacent to lm_head pool) but is a separate type. Workers run a
`while(!stop)` loop polling `job_seq.load(acquire)`; when changed, read the
shared job (carefully — single-publisher = driver), compute their row range,
write to per-worker slot, `done_count.fetch_add(1, release)`.

### Idle backoff policy [duck r2 NB#1, r3 #2]

Tight `_mm_pause` for the entire generation (potentially seconds) is too
hostile to the OS. Workers use **wall-clock-measured** phases after the
last `done_count++` (using `QueryPerformanceCounter` on Windows or
`std::chrono::steady_clock::now()` portably):

| phase | duration       | action                              |
|-------|----------------|-------------------------------------|
| 1     | first ~10 µs   | `_mm_pause` tight loop              |
| 2     | next ~1 ms     | continue light `_mm_pause` loop with wall-clock check; **do NOT use `std::this_thread::yield()`** — on Windows it can return immediately or sleep a full 15.6 ms quantum |
| 3     | after ~1 ms idle | cv-park with predicate (`done_count != target || job_seq changed`); driver `notify_all` on next kick |

Sub-µs handoff for the typical case (matmul work arrives every ~1 ms during
decode) while not pinning a CPU at 100 % during user think-time or sampling
gaps. Phase durations are tunable; A2.5 measures the right values.

### Matmul job shape
```cpp
struct Phi3MatmulJob {
    const struct ggml_type_traits_cpu * w_traits;
    const uint8_t * w_base;      // weight matrix base (already-quantized rows)
    size_t          w_row_bytes; // ggml_row_size(w->type, K)
    const void *    src_q;       // pre-quantized src (q_type matches w_traits->vec_dot_type)
    float *         dst;         // F32 output
    int             K;           // inner dim
    int             N_total;     // total output rows
};

// Per-worker (computed in worker from job_seq+wid+N_total):
//   lo = (wid * N_total) / n_workers
//   hi = ((wid+1) * N_total) / n_workers
```

### Prefill threading [duck answer #4]
Prefill also goes through `Phi3MatmulPool` (NOT serial). Owning the KV
implies owning the prefill — we cannot call `llama_decode` for the prompt
because it would populate llama's KV cache, not ours. Serial-prefill is
only a debug bring-up mode for A2.3.

---

## 5. Wiring into Phi3Runtime

Add to `phi3_runtime.h`:
```cpp
struct Phi3Runtime {
    ...
    bool             use_fused_forward = false;
    bool             use_fused_f32_debug = false;
    Phi3Weights      fused_weights;
    Phi3MatmulPool   fused_matmul_pool;   // NEW — spin-policy, distinct from lm_head pool
    Phi3FusedCtx     fused_ctx;
};
```

In `Phi3.cpp`:
- New CLI flags `--phi3-fused-forward` and `--phi3-fused-f32-debug`.
- In `phi3_runtime_init`, after the existing weight/pool init:
  ```cpp
  if (runtime.use_fused_forward) {
      ok &= phi3_weights_resolve(model, runtime.fused_weights, err);
      ok &= phi3_matmul_pool_init(runtime.fused_matmul_pool, runtime.n_threads_gen, err);
      ok &= phi3_fused_ctx_init(
                runtime.fused_ctx,
                runtime.fused_weights,
                &runtime.fused_matmul_pool,
                lctx,                                 // for cparams/lora/cvec checks
                runtime.n_ctx,
                runtime.use_fused_f32_debug,
                err);
  }
  ```
- In the gen loop, before `llama_decode`:
  ```cpp
  if (runtime.use_fused_forward) {
      llama_token tok = phi3_fused_decode_step(runtime.fused_ctx, last_tok);
      // skip llama_decode + sampler entirely
      // append to output_tokens, continue
  }
  ```
- Prefill: similar, calling `phi3_fused_prefill(ctx, prompt_tokens, n_prompt)`.
- In `phi3_runtime_free`: `phi3_fused_ctx_free`, then `phi3_matmul_pool_free`,
  then existing `phi3_lmhead_pool_free`.

Fused-forward is **mutually exclusive** with `--phi3-fused-decode` and
`--phi3-fused-lmhead` (they operate on the standard graph). Enforce at
flag parse: error if any combination is set. Internally, fused-forward
still uses the existing `Phi3FusedLmHead` weight/scratch resolution +
`Phi3LmHeadPool` for the per-token argmax — it just does NOT set
`cparams.fused_lmhead` (no llama-graph mutation). [duck r2 non-blocking #3]

### Pre-A2 validation flag (recommended)
Add `--phi3-fused-forward-dry` that runs both paths in parallel and asserts
greedy tokens match. Cheap insurance during A2 development; can be removed
later.

---

## 6. Test plan (in order)

1. **Unit (per-kernel)**: a `--phi3-kernel-self-test` flag that:
   - Builds random F32 inputs.
   - Compares our `rms_norm` against ggml's by exact byte match.
   - Compares our **token-embed dequant** for a few token ids against
     `ggml_get_rows` driven through a tiny ggml graph. [duck non-blocking #1, test gap]
   - Compares our **matmul** (1 row, 1 weight tensor of each type that
     appears in this model) against a standalone `ggml_mul_mat` evaluated
     via a tiny graph. Be careful to call `ggml_build_forward_expand`,
     allocate via a CPU backend, and populate inputs before compute. [duck test gap]
   - Compares our **NeoX RoPE** against `ggml_rope_ext`. Test both the
     factors=NULL case (mini-4k) and the factors=non-NULL case (use a
     synthetic factor table — we want the math right even if mini-4k
     doesn't exercise it).
   - Compares our **full single-layer forward** (norm→qkv→rope→attn→wo→res
     →norm→ffn→res) against a tiny ggml graph that does the same. This is
     the high-value test that catches layout/stride/transpose bugs early.
     [duck test gap]
2. **F32-everywhere mode (bring-up oracle)**: `--phi3-fused-forward --phi3-fused-f32-debug`
   runs our forward with F32 KV and F32 matmul reductions (no quantize).
   This is intentionally much slower; its job is to isolate
   layout/RoPE/attention bugs before introducing Q8_K/F16 KV noise.
   First-token match against baseline at this mode is the gate to drop the
   debug mode. [duck test plan suggestion]
3. **Integration single-step**: run prompt "Hello", emit 1 token via fused
   path (no f32_debug) and via baseline. Tokens must match.
4. **Integration multi-step**: run prompt "Why is the sky blue?", emit 256
   tokens via fused path and via baseline (seed=1234, greedy, **flash_attn=false**).
   **All 256 tokens MUST match (yes/no acceptance bar).** On any divergence:
   log top-2 logit margin at the diverging step for diagnostic purposes only;
   the bar is still "all 256 match". [duck r2 #6]
5. **3-prompt regression**: run 3 different prompts × 256 tokens. All match
   under the same yes/no policy as (4).
6. **Multi-turn rewind**: prefill A, decode 32, drop_range, prefill B from
   middle, decode 32. Compare against a clean state that produces the same
   final sequence.

If steps 1–5 pass, A2 is done. Step 6 exercises the rewind primitives we
already self-tested in A1.

---

## 7. Known pitfalls (informed by duck + experience)

| trap | mitigation |
|---|---|
| RMSNorm sum_sq must be `double` to match ggml | spec says `double` explicitly; assert in code comment |
| Q-quant block sizes must divide n_embd and n_ff | assert at init for chosen `q_type_attn` / `q_type_ffn` |
| RoPE long/short selection is **per-context-length, not per-position** | resolve once at `phi3_fused_ctx_init` time |
| `rope_factors_long/short` may both be NULL | guard with `if (rope_factors == nullptr) use 1.0 per dim` per `ops.cpp:5677-5681` |
| K stored F16, dequant on the fly during Q·K dot | inline F16→F32 conversion via `GGML_FP16_TO_FP32` |
| V_T stride: V_T[il][k, h, :] is `head_dim` contiguous F16 — fast read | the whole point of transposing V — verify with sizeof asserts |
| Weight stride: row i = base + i * `ggml_row_size(w->type, w->ne[0])` | already rejected at init if `nb[1] != row_size` (§1.5) |
| Token embed `tok_embd` is `{n_embd, n_vocab}` (`src/models/phi3.cpp:182`); `ggml_get_rows` indexes outer dim (`ggml.c:3964`) → row `tok` at `base + tok * nb[1]`, n_embd elements long | A2.1 self-test compares dequant against `ggml_get_rows` byte-for-byte |
| matmul fan-out per layer is large (128 fork-joins/token) | spin-policy `Phi3MatmulPool` with 3-phase backoff (see §4) |
| Prefill uses pooled matmuls too (own KV requirement) | n_threads is `runtime.n_threads_gen` for both decode and prefill in A2; separate `--threads-fused-prefill` is an A3 concern |
| `cur_pos` must equal `kv.current_len` at all times | assert in decode_step entry |
| greedy sampler bypass means logit_bias, repetition penalty, etc. are also bypassed | document; A2 is greedy-only |
| When using `Phi3LmHeadPool` for argmax, do NOT set `cparams.fused_lmhead` — that mutates the llama graph and forces embeddings | runtime uses the pool directly, leaves cparams alone (`src/llama-context.cpp:1119-1133`) [duck r2 non-blocking #3] |

---

## 8. Files touched

- `examples/phi3/phi3_fused_graph.h`  — add Phi3FusedCtx + Phi3RoPEConfig + Phi3ForwardScratch + public API
- `examples/phi3/phi3_fused_graph.cpp`— implement everything above
- `examples/phi3/phi3_fused_ops.h`    — add Phi3MatmulPool + Phi3MatmulJob (new spin-policy pool, distinct from existing Phi3LmHeadPool)
- `examples/phi3/phi3_fused_ops.cpp`  — implement Phi3MatmulPool init/run/free + worker loop with 3-phase backoff
- `examples/phi3/phi3_runtime.h`      — add `use_fused_forward`, `use_fused_f32_debug` bools, Phi3MatmulPool field, Phi3FusedCtx field
- `examples/phi3/phi3_runtime.cpp`    — init/free fused ctx + matmul pool
- `examples/phi3/Phi3.cpp`            — CLI flags, gen-loop branch, mutual exclusion check
- `include/llama_b612.h`              — add `llama_b612_has_active_lora` and `llama_b612_has_active_cvec` accessors [duck r2 #3]
- `src/llama-context.cpp`             — implement those two accessors
- (no other changes to src/, include/, ggml/)

---

## 9. Rollout plan

A2 sub-commits (each builds & passes):

1. **A2.1 — Pool + per-kernel self-tests (foundation)**:
   - Introduce `Phi3MatmulPool` (new spin-policy pool with 3-phase backoff,
     distinct from `Phi3LmHeadPool`).
   - Add `llama_b612_has_active_lora` and `llama_b612_has_active_cvec`
     accessors in `include/llama_b612.h` + `src/llama-context.cpp`.
   - Add `--phi3-kernel-self-test` covering: RMSNorm vs ggml, token-embed
     dequant vs `ggml_get_rows`, matmul (one row, every weight type that
     appears in the model) vs `ggml_mul_mat`, and **NeoX RoPE vs
     `ggml_rope_ext`** (both NULL-factors and synthetic-factors cases).
   - Existing fused-lmhead regression must still pass.

2. **A2.2 — Init + validation (no forward yet)**:
   - Implement `phi3_fused_validate_supported` (§1.5 rejection list, using
     the new accessors).
   - Implement `phi3_fused_ctx_init` / `phi3_fused_ctx_free`: scratch alloc,
     RoPE config resolution, q_type derivation from each weight's traits.
   - Self-test: init fails with a specific named message on a deliberately
     unsupported config (e.g. inject a fake bias pointer).
   - Add `--phi3-fused-forward` flag (parse + mutual exclusion check) but
     forward is still a stub that returns 0.

3. **A2.3 — Single-layer self-test**:
   - **Before any full-stack integration**, add `--phi3-kernel-self-test`
     case that exercises the full single-layer F32 reference forward
     (norm→qkv→rope→attn→wo→res→norm→ffn→res) against a tiny ggml graph
     building the same layer. This is the single highest-value test;
     catches layout/stride/transpose bugs before they get buried in
     32-layer noise.

4. **A2.4 — F32-everywhere decode integration**:
   - Implement `phi3_fused_decode_step` in F32-everywhere mode only
     (`--phi3-fused-f32-debug`).
   - Run single-step "Hello", verify token matches baseline.
   - Run 256-token "Why is the sky blue?", all tokens must match.

5. **A2.5 — Q-quant decode + prefill + regression**:
   - Drop F32-debug for the matmul path; use `traits->vec_dot_type` quant
     paths. Keep F16 KV. Re-run 256-token test.
   - Implement `phi3_fused_prefill` (pooled matmuls, not serial).
   - Run 3-prompt × 256-token regression. All must match (yes/no bar).
   - Multi-turn rewind smoke test.

Estimated total: ~1000 LOC, 5 commits.

---

## Resolved open questions (duck answers folded in)

1. **Single pool vs second pool?** → **Second pool.** `Phi3MatmulPool` spin
   policy is incompatible with `Phi3LmHeadPool` cv-park; toggling at runtime
   is brittle. See §4.
2. **Own prefill or bypass to `llama_decode`?** → **Own prefill.** Owning the
   KV makes bypass impossible (llama's KV is separate from ours). See §4
   "Prefill threading".
3. **Byte-identical logits achievable?** → **No, only token-identical.**
   Acceptance is greedy token match — yes/no, 3 prompts × 256 tokens, all
   must match. See §0 acceptance bar.
4. **Serial prefill regression on long prompts?** → **Yes, badly.** Prefill
   goes through `Phi3MatmulPool` from the start; serial-prefill is only the
   A2.4 bring-up debug mode.
