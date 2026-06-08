///////////////thinking/
I'm mapping out the Phi-3 architecture layer by layer, starting with the normalization step, then the fused query-key-value projection, and moving into the rotary position embeddings.

  The core bottleneck is clear: per-token decode takes 38ms with the model at ~2GB, which puts us squarely in memory-bound territory at around 50 GB/s. I'm identifying fusion opportunities that can reduce memory traffic without major architectural changes. The highest-impact wins are fusing RMSNorm with the following matmul to avoid redundant reads, combining SiLU and multiplication for the SwiGLU gate operation, and merging the Q-scale operation into the RoPE computation. I'm also considering residual additions that can be folded into the next layer's normalization to eliminate kernel boundaries.

  For the QKV split, since Phi-3 packs these into a single tensor, I can fuse the split with RoPE application in one pass rather than materializing three separate views. The lm_head argmax is another quick win—instead of computing all 32K logits and then finding the max, I can track the argmax during the dot product computation itself, saving a full pass over the vocabulary. For the heavier lifts, a custom GEMV kernel tuned specifically for single-token decoding with AVX-512 and on-the-fly Q4_K dequantization could significantly outperform the general-purpose matmul, and similarly, attention over the KV-cache can be optimized as a per-step GEMV with f16 accumulation directly on the cached pages.

  The constraint is that these fused kernels need to live in `examples/phi3/` rather than in core ggml, which limits how deeply I can hook into llama's compute graph. I'm weighing three approaches: building a custom Phi-3 forward pass using ggml directly with custom ops, wiring fusions through ggml's graph hooks, or pulling the post-norm embedding after the last layer and doing lm_head + argmax outside llama's control. The third angle looks promising since llama already exposes the embedding via `llama_get_embeddings`, so I can skip materializing the full logit buffer entirely.

  But I'm realizing the real blocker: if llama still computes lm_head logits in its graph, I don't actually save anything by fusing argmax on my side. I'd need to suppress lm_head from llama's graph entirely, which isn't possible from the examples layer. So the realistic path forward is to set up a proper A/B benchmark harness first—add a `--bench` flag to run fixed prompts repeatedly and measure median time-to-first-token, p95 generation step latency, and tokens-per-second—then identify the actual decode-time hotspots worth optimizing.

  The highest-leverage option that's actually feasible is building a custom Phi3 forward pass directly in examples/phi3 using ggml primitives, which would let me fuse operations like SwiGLU and bypass llama's graph entirely. But that's a massive undertaking. Let me present the user with a tiered plan they can choose from based on effort versus impact. So the concrete plan is to start with measurement and A/B testing—adding a benchmark mode to track per-stage timing breakdowns for attention versus feed-forward layers. Then move into a custom forward pass that fuses operations like RMSNorm directly into the QKV computation, combines the SwiGLU activation with the FFN output read, and skips materializing the full logits by computing argmax over the language model head rows instead. If those wins justify it, I'd explore a hand-rolled Q4_K GEMV kernel with AVX-512 and better cache locality.

  But I'm realizing the user's already signaled they want real fused kernel work, not more profiling—and from the examples/phi3 context, that means either hijacking llama's graph computation or building a parallel forward pass that directly uses ggml. I should present this honestly as a scope question and let them pick which direction to pursue.

◐ Now I'm mapping out the computational flow for Phi3's attention mechanism, working through the per-token costs across the 32 layers—starting with the normalization step, then the QKV projection that splits into separate query, key, and value matrices, followed by the rotary position embeddings applied to Q and K.

  The attention computation itself involves the scaled dot product between Q and K, softmax normalization, and multiplication with V, then a final output projection. After that comes the residual connection, then the feed-forward network with its own normalization, an up-projection using SwiGLU gating, and a down-projection back to the model dimension, capped with another residual add.

  Once all 32 layers complete, there's a final normalization and the language model head that projects to the full vocabulary of 32,064 tokens. Looking at the memory bandwidth requirements, each layer reads roughly 2GB of quantized weights per token, which at the observed latency of 38ms per token translates to around 53 GB/s—essentially hitting the memory bandwidth ceiling. The real optimization opportunities lie in either reducing the weight reads themselves or improving the dequantization-to-multiply fusion that's already partially handled by the underlying framework.

  The most practical approach would be to build a parallel forward pass implementation using the framework's primitives directly with custom operation fusions, sidestepping the standard decode path. I'm thinking through a phased plan to propose to the user so they can choose what scope makes sense for their needs.
  ////////////////thinking/

Phi3-mini per-token decode budget (~38 ms, gen_tps ≈ 25, AVX-512):

┌───────────────────────────────┬─────────────┬────────────┐
│ op (×32 layers)               │ weight read │ est. share │
├───────────────────────────────┼─────────────┼────────────┤
│ QKV mm (3 tensors)            │ ~16 MB      │ ~20%       │
├───────────────────────────────┼─────────────┼────────────┤
│ wo mm                         │ ~5.6 MB     │ ~7%        │
├───────────────────────────────┼─────────────┼────────────┤
│ ffn_up (2·n_ff)               │ ~30 MB      │ ~38%       │
├───────────────────────────────┼─────────────┼────────────┤
│ ffn_down                      │ ~15 MB      │ ~20%       │
├───────────────────────────────┼─────────────┼────────────┤
│ norms + RoPE + softmax + adds │ small       │ ~5%        │
├───────────────────────────────┼─────────────┼────────────┤
│ lm_head once (3072→32064)     │ ~60 MB      │ ~8%        │
└───────────────────────────────┴─────────────┴────────────┘

~2 GB read/token ÷ 38 ms ≈ 53 GB/s — we're already ~70–90% of typical DDR5 bandwidth. So speedups come from reading less and tighter dequant↔gemv fusion, not from kernel cleverness on FMAs.

What "fused kernel" can realistically buy us from examples/phi3/:

The catch: real matmul fusion lives in ggml-cpu (where dequant + gemv already share a pass). From examples/phi3/ we cannot replacethe matmul kernel without touching core. Three realistic angles:

 - A. Bypass llama_decode and build a custom Phi3 forward pass in examples/phi3/ using ggml directly + ggml_map_custom* for true opfusion (SwiGLU fuse, lm_head+argmax fuse, RMSNorm→matmul fuse). This is the only path that lets us land real fused kernels withoutmodifying core.
 - Highest upside, biggest lift. Expected ~10–25% gen_tps if done well.
 - Loses llama's KV-cache management — we'd manage it ourselves.
 - B. Add fused custom-op nodes via a thin core hook (gated under GGML_B612_PHI3_FUSED): RMSNorm+QKV-mm fusion, SwiGLU fusion,lm_head+argmax fusion injected into the existing llama graph.
 - Medium lift, medium upside (~5–15%). Touches a few lines in core, fusion implementations live in examples/phi3/.
 - C. Post-decode-only fused argmax: configure llama to emit embeddings (last hidden state) instead of logits, then run our ownAVX-512 Q4_K-aware lm_head+argmax in examples/phi3/. Skip the 32064-wide logits write.
 - Smallest lift, smallest upside (~3–8% on lm_head alone, ~1–2% overall). Good warm-up & validation.

My recommendation: C → B → A, in that order, gated by measured wins at each step. C also exercises the AVX-512 + Q4_K-aware GEMV path we'd reuse in B/A.

A — full custom forward pass. Yes, exactly what you said. We write build_phi3_fused(ggml_context*, ...) in examples/phi3/ thatconstructs a ggml_cgraph from scratch using ggml_* op constructors. We borrow only the compute flow (ggml_graph_compute_with_ctx onthe CPU backend) and weight pointers (resolved via llama_model_get_tensor(model, "blk.0.attn_norm.weight") etc., already loaded onceby llama). The graph itself is ours; we are free to:

 - replace any N consecutive standard ops with a single ggml_map_custom1/2/3 node whose compute lives inexamples/phi3/phi3_fused_ops.cpp;
 - drop ops entirely (e.g. don't even emit a lm_head node — emit a fused lm_head_argmax custom node that returns just the token id);
 - own and lay out our own KV-cache as plain ggml_tensors (no llama_kv_cache).
 Cost: we lose llama's KV-cache, batching, perf telemetry — we re-implement what we need.

B — in-graph fusion via thin core hook. llama still owns the graph; we mutate it at construction time. Concretely:

 - Where the hook lives: in src/models/phi3.cpp::graph<iswa>::graph() (the only Phi3 graph builder). Three #if defined(GGML_B612_PHI3_FUSED) branches replace standard nodes with ggml_map_custom* nodes whose compute lives inexamples/phi3/phi3_fused_ops.cpp (linked via a small registration C API like llama_set_b612_phi3_fused_ops(const struct llama_phi3_fused_ops
  *) so the kernels stay in examples/, only the dispatch lives in core).
 - When active: compile-time cmake -DGGML_B612_PHI3_FUSED=ON and runtime flag (Phi3.exe --phi3-fused-graph) and model is Phi3 withthe right shape. Any of those false → standard graph, zero impact.
 - What it replaces (3 fusion points):
  1. [attn_norm → wqkv mm] → one custom node (saves a hidden-state read).
 2. [swiglu_split → ffn_down_mm] → one custom node (saves the n_ff intermediate write/read, ~8 KB × layer).
 3. [final_norm → lm_head mm → argmax] → one custom node when greedy (saves the 32064×4B logits buffer write/read entirely).

So B's hook is a few lines in src/models/phi3.cpp + a small registration C API; the kernels remain in examples. A's "hook" is justexamples/phi3/ calling ggml_* directly — no core changes at all, but we re-implement the orchestration.

Q1 — next round when we suppress t_logits. Nothing changes for round N+1. The autoregressive loop only carries forward the sampled token id — that token id feeds tok_embd lookup at the next decode and runs all 32 layers afresh. The hidden state and logits fromround N are not consumed by round N+1; only the KV-cache (already written by attention during round N) crosses turn boundaries.

So the rule is: as long as something computes the next token id from the final hidden state before we suppress t_logits, the loop iscorrect. Our fused final_norm → lm_head → argmax custom node does exactly that and returns just the token id (a tiny i32 tensor). Wefeed that id straight back into the next state.batch.token — the existing wire-up in phi3_runtime.cpp line ~320 / ~352 already doesthis with gen_input_token = sampled_token.

One important nuance: t_logits doubles as the "training/eval" output for non-greedy sampling. We must only suppress it on greedy steps (which is our fused-path gate already). When temp > 0 or min_p > 0, the standard graph runs and we sample normally.

Q2 — Q4_K GEMV already exists. You're right, and I should have caught that. ggml_get_type_traits_cpu(GGML_TYPE_Q4_K)->vec_dot returns ggml_vec_dot_q4_K_q8_K (with AVX-512 specialization at ggml/src/ggml-cpu/arch/x86/quants.c). The xbox mul_mat path also uses thesetraits. We should reuse them, not roll our own.

That changes the Phase C kernel design — it becomes a thin orchestrator:

 phi3_fused_lmhead_argmax(h_f32[n_embd], W_q4k[n_vocab, n_embd]):
   quantize_row_q8_K(h_f32 -> h_q8k)           // type_traits.from_float once
   best_logit = -inf; best_id = 0
   for v in 0..n_vocab:
     s = 0
     ggml_vec_dot_q4_K_q8_K(n_embd, &s, 0,
                            W_q4k + v*row_size_q4k, 0,
                            h_q8k, 0, 1)
     if s > best_logit: best_logit = s; best_id = v
   return best_id

No new SIMD code — just a loop over rows + argmax tracker. Wins:

 - skip the n_vocab×4B logits write/read
 - skip the sampler chain (softmax/min_p over 32064 entries — small but real)
 - skip llama_token_to_piece on a logits buffer

Revised Phase C scope (smaller and cleaner):

 1. One-line core gate in src/models/phi3.cpp: when LLAMA_B612 + runtime flag + greedy step, set res->t_logits = nullptr so thelm_head matmul is dropped from the graph. (Pruning happens automatically in ggml_build_forward_expand — the unreferenced lm_head node won't run.)
 2. Read res->t_embd via llama_get_embeddings_ith(ctx, -1).
 3. New small file examples/phi3/phi3_fused_ops.cpp containing only the orchestrator above, calling existing type_traits_cpu for theactual SIMD.
 4. Pull model.output (lm_head) via llama_model_get_tensor for raw weights.

The KV-cache writes are explicitly forced into the forward graph by the attention builder itself:

 ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
 ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));

This happens inside build_attn for every layer, completely independently of t_logits or t_embd. The KV-cache cpy_k/cpy_v nodes areroots in the forward graph — they will execute regardless of what's downstream.

So the dependency picture for round N's graph is:

 inp_tokens
    ↓
 [layer 0..31]
    ├── KV cpy_k/cpy_v  → KV-cache  ← independent root, ALWAYS runs
    └── residual stream
         ↓
    final_norm  → t_embd  (expanded only when embeddings requested)
         ↓
    lm_head    → t_logits (expanded by default)
         ↓
    argmax/sample → next token

When we set res->t_logits = nullptr in Phi3's graph builder (greedy steps only), ggml's reachability pruning drops the lm_head matmul because nothing references it anymore. But:

 - KV writes: untouched — they're independent roots in gf.
 - All 32 layers: untouched — they feed both KV writes and the residual stream.
 - final_norm: stays if we also expand t_embd (which we will, so our fused argmax has its input).

So suppressing t_logits has zero effect on KV-cache correctness or content for round N+1. The KV-cache for round N's token is fullypopulated by the attention layers themselves; round N+1 will read those cache entries normally when computing its own attention.

The exact mechanism: we'll need to also call ggml_build_forward_expand(gf, res->t_embd) ourselves (or set cparams.embeddings = true)so that final_norm isn't pruned alongside lm_head. That's the one extra wire.

===========================================================================

The Phi3 layer (per step) at memory-bound 40 ms:

┌──────────────────────────────────────┬──────────────┬─────────────┬──────────────┐
│ stage                                │ inputs       │ bytes read  │ wall time    │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ attn_norm + qkv (combined wqkv)      │ residual f32 │ ~16 MB Q4_K │ ~280 µs      │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ RoPE + scale                         │ small        │ small       │ ~25 µs       │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ attn (QK^T, softmax, V, wo)          │ KV cache     │ varies      │ ~160 µs      │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ residual add                         │ f32          │ tiny        │ ~5 µs        │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ ffn_norm + ffn_up (combined gate+up) │ residual f32 │ ~17 MB Q4_K │ ~430 µs      │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ silu * up                            │ f32          │ ~64 KB      │ ~10 µs       │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ ffn_down                             │ f32          │ ~16 MB Q4_K │ ~330 µs      │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ residual add                         │ f32          │ tiny        │ ~5 µs        │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ total/layer                          │              │ ~49 MB      │ ~1.24 ms     │
├──────────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ × 32 layers                          │              │ ~1.56 GB    │ ~39.7 ms     │
└──────────────────────────────────────┴──────────────┴─────────────┴──────────────┘

Realistic Phase B targets, ranked:

 1. Fuse RMSNorm → q8_K-quantize into one op (2 fusion points/layer × 32 = 64 calls): eliminates intermediate f32 write + the nextmatmul's from_float. Estimated +1–2% TPS. ~1 day. Risk: medium (new ggml op).
 2. Custom per-layer fused gen kernel (phi3_decode_block): replaces one whole block with a single C++ function that reuses scratchbuffers, skips ggml graph overhead, and pipelines norm→matmul→RoPE→KV-write→… Estimated +5–10% TPS. ~2–3 days. Risk: high(correctness + KV cache layout coupling).
 3. Persistent worker pool for fused argmax (Phase C polish): +2–3% TPS. ~1 hour. Risk: low.

Phase B as originally written (option 1) is genuinely modest. Option 2 is what would give real gains but it slides into Phase Aterritory.

My recommendation: do (3) first as a free win, then go straight to Phase A (option 2-ish). That gets us most of the upside with theleast wasted effort.

Option 1 and Option 3 are completely independent code paths and fully additive:

┌───────────────────┬──────────────────────────────────────────────────┬──────────────────────────────────────┐
│                   │ Option 1 (norm→q8K fuse)                         │ Option 3 (persistent pool)           │
├───────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
│ Stage             │ Inside the decode graph (per layer)              │ After decode (sample step)           │
├───────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
│ Code location     │ src/models/phi3.cpp + new ggml op                │ examples/phi3/phi3_fused_ops.cpp     │
├───────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
│ What it saves     │ Intermediate f32 write + downstream from_float   │ std::thread spawn/join overhead      │
├───────────────────┼──────────────────────────────────────────────────┼──────────────────────────────────────┤
│ Threading         │ Runs inside ggml's persistent thread pool (free) │ Replaces our std::thread with a pool │
└───────────────────┴──────────────────────────────────────────────────┴──────────────────────────────────────┘

The insight worth keeping: Option 3 exists only because our Phase C fused argmax runs outside the ggml graph as our own post-decodeC++ code. Anything we put inside the graph as a ggml op gets parallelized by ggml's pre-allocated thread pool, so the spawn/joinproblem doesn't exist there. Option 1 doesn't need its own thread pool.

How Option 1 would be built

The current graph for the attn_norm + QKV stage is:

 attn_norm_out = RMSNorm(inpL, norm_weight)      // op 1: writes n_embd × f32
 qkv = mul_mat(wqkv, attn_norm_out)              // op 2: internally calls
                                                 //         from_float(f32 → q8_K)
                                                 //         then vec_dot per row

Two ops, an intermediate buffer of n_embd × 4 B, and from_float walks the input again.

Design choice: custom ggml op vs ggml_map_custom

Option A — register a real GGML_OP_NORM_RMS_Q8K: Edit ggml.h (enum), ggml.c (forward decl), ggml-cpu.c (kernel dispatch), per-archquants. Cleanest but invasive — changes ggml core.

Option B — ggml_map_custom3: A black-box custom op already supported by ggml. You hand it 3 input tensors + a function pointer + howmany tasks (threads) to use. ggml's scheduler allocates those tasks from its persistent pool when the op runs. No core changes. This is what "thin core hook" means in the plan. I'd go with B.

But there's a subtlety: ggml's mul_mat always calls from_float on its src1 tensor; it has no way to accept a pre-quantized q8_K input through the standard path. So fusing norm → q8K alone doesn't help — the next mul_mat redoes the work.

Resolution: fuse the whole norm + matmul into one ggml_map_custom3. The op takes (inpL, norm_weight, wqkv) and emits the QKV f32output directly. The downstream graph (RoPE, attention, …) sees the QKV tensor and is unchanged.

The kernel (pseudocode)

 // Runs on each of n_tasks worker threads, dispatched by ggml's pool.
 static void phi3_fused_norm_qkv(
         ggml_tensor * dst,                       // [n_embd_qkv, n_tokens] f32
         const ggml_tensor * inpL,                // [n_embd, n_tokens] f32
         const ggml_tensor * norm_w,              // [n_embd] f32
         const ggml_tensor * wqkv,                // [n_embd, n_embd_qkv] q4_K
         int ith, int nth, void * userdata) {

     const float eps = *(const float *)userdata;
     const int64_t n_embd     = inpL->ne[0];
     const int64_t n_embd_qkv = wqkv->ne[1];

     auto w_tr = ggml_get_type_traits_cpu(wqkv->type);
     auto q_tr = ggml_get_type_traits_cpu(w_tr->vec_dot_type);  // q8_K

     // Per-token; for 1-token gen, ne[1]=1 so this loop runs once.
     for (int64_t t = 0; t < inpL->ne[1]; ++t) {
         const float * x = (const float *)inpL->data + t * n_embd;
         float * y       = (float *)dst->data        + t * n_embd_qkv;

         // 1. RMSNorm — one pass over x to compute sum of squares.
         //    Each task handles part of the row only on thread 0,
         //    then broadcasts the scale via simple memory barrier.
         //    (Simpler for 1-token: do norm on ith==0 into a small scratch,
         //     ggml's thread sync happens at op boundary anyway.)
         float ss = 0.0f;
         for (int64_t i = 0; i < n_embd; ++i) ss += x[i] * x[i];
         float scale = 1.0f / sqrtf(ss / n_embd + eps);

         // 2. norm * gamma → q8_K scratch buffer (size n_embd / 256 blocks).
         //    Allocated once per call; reused across blocks.
         alignas(64) uint8_t qbuf[N_EMBD_MAX_Q8K_BYTES];
         for (int64_t b = 0; b < n_embd; b += 256) {
             float tmp[256];
             for (int j = 0; j < 256; ++j) {
                 tmp[j] = x[b + j] * scale * ((const float *)norm_w->data)[b + j];
             }
             q_tr->from_float(tmp, qbuf + (b/256)*sizeof(block_q8_K), 256);
         }

         // 3. Per-vocab-row vec_dot, split across nth tasks.
         const size_t row_bytes = ggml_row_size(wqkv->type, n_embd);
         const uint8_t * w_base = (const uint8_t *)wqkv->data;
         const int64_t rows_per = (n_embd_qkv + nth - 1) / nth;
         const int64_t r0 = ith * rows_per;
         const int64_t r1 = std::min(r0 + rows_per, n_embd_qkv);
         for (int64_t r = r0; r < r1; ++r) {
             float s = 0.0f;
             w_tr->vec_dot(n_embd, &s, 0,
                           w_base + r * row_bytes, 0,
                           qbuf, 0, 1);
             y[r] = s;
         }
     }
 }

Integration in src/models/phi3.cpp

 if (cparams.fused_decode_phi3) {
     // Single op replaces build_norm + build_qkv's wqkv matmul.
     ggml_tensor * qkv = ggml_map_custom3(
         ctx0, inpL, model.layers[il].attn_norm, model.layers[il].wqkv,
         phi3_fused_norm_qkv,
         /*n_tasks=*/GGML_N_TASKS_MAX,
         /*userdata=*/&hparams.f_norm_rms_eps);
     qkv = ggml_reshape_2d(ctx0, qkv, n_embd_qkv, n_tokens);
     // Continue with the views into Q/K/V exactly as before.
     Qcur = ggml_view_3d(...);
     Kcur = ggml_view_3d(...);
     Vcur = ggml_view_3d(...);
 } else {
     // Existing path
     attn_norm_output = build_norm(...);
     auto [Qcur,Kcur,Vcur] = build_qkv(...);
 }

You'd do the same for ffn_norm + ffn_up (combined gate+up, output 2*n_ff). That's 2 fusion sites × 32 layers = 64 fused-opinvocations per step.

What you actually win

 - Skip from_float on the matmul input — one fewer pass over n_embd f32 = 12 KB/site → ~10 µs × 64 = ~640 µs/step.
 - Skip the intermediate attn_norm_out f32 write+read — another 12 KB/site = ~5 µs × 64 = ~320 µs/step.
 - Total realistic gain: ~1 ms/step → ~+2.5% gen TPS (matches my earlier estimate).

The cost

 - One new C++ kernel function (~50 lines).
 - Two ggml_map_custom3 substitutions in phi3.cpp under a new cparams.fused_decode_phi3 flag.
 - Reuses everything else (existing vec_dot, ggml's thread pool, scheduler, graph reuse).

 =========================================================

 Option 3: phi3: persistent worker pool for fused lm_head argmax

 A/B (Phi-3-mini-4k Q4_K_M, seed=1234, n=256, prefill=32/gen=10 threads, greedy):

┌────────────────────────────────────────┬─────────────────┬────────────────────┬─────────────────┐
│ variant                                │ gen_tps         │ sample_step_avg_ms │ gen_step_avg_ms │
├────────────────────────────────────────┼─────────────────┼────────────────────┼─────────────────┤
│ baseline (no fused-lmhead)             │ 26.09           │ 0.004              │ 38.49           │
├────────────────────────────────────────┼─────────────────┼────────────────────┼─────────────────┤
│ fused-lmhead + legacy spawn (previous) │ 26.88           │ 2.42               │ 37.34           │
├────────────────────────────────────────┼─────────────────┼────────────────────┼─────────────────┤
│ fused-lmhead + cv-park pool            │ 27.32–27.69     │ 1.6–1.9            │ 36.3–36.8       │
└────────────────────────────────────────┴─────────────────┴────────────────────┴─────────────────┘

Net: +1.2–1.6 t/s (~+5–6% TPS) over baseline; pool also slightly reduces decode_avg because workers are parked between argmax callsand don't steal cycles from the ggml decode pool. Output is byte-identical to baseline.

=========================================================

Option 1 (in-graph fused RMSNorm + quantize-to-Q8_K): IMPLEMENTED, gated by --phi3-fused-decode, kept for educational reference.

Empirical result: roughly neutral (within noise) for greedy decode on Phi-3-mini-4k Q4_K_M, threads-prefill=32, threads-gen=10:

  4-iteration mean (n=256, seed=1234):
    BASE      : 25.03 gen_tps  40.12 ms/step
    --fused-decode (Option 1)        : 25.16 gen_tps  40.12 ms/step  (+0.5%, within ±5% noise)
    --fused-lmhead (Phase C+Option 3): 27.45 gen_tps  36.62 ms/step  (+9.7%)
    both flags (decode + lmhead)     : 28.27 gen_tps  35.63 ms/step  (+12.9%)
    (note: --phi3-fused-lmhead already includes Phase C skip-logits +
     Option 3 persistent worker pool — those are not separate flags)

Single-thread isolation (--threads-gen 1, n=64, 3 iters):
    BASE_1t                          :  9.29 gen_tps  109.34 ms/step
    --phi3-fused-decode  (alone)     :  9.35 gen_tps  108.60 ms/step  (+0.6%, within noise)
    --phi3-fused-lmhead  (alone)     :  9.63 gen_tps  105.45 ms/step  (+3.7%)
    both flags                       :  9.47 gen_tps  107.31 ms/step  (+1.9%, lmhead win partly offset by decode overhead)
(Lesson: an earlier single-iter run reported decode at -1.7%; that was noise.
 Always run >=3 iters before drawing perf conclusions.)

Output is byte-identical to baseline (requires double-precision sum_sq accumulation to match ggml_compute_forward_rms_norm_f32 which uses ggml_float == double at vec.h:15).

Why it doesn't win meaningfully:
 - Per-site fusion savings: ~5 us (eliminate f32 intermediate buffer + skip mul_mat internal from_float).
 - Per-site GGML_OP_CUSTOM dispatch overhead is comparable to those savings, so net per-site delta is within measurement noise.
 - 64 sites x ~0 us = a wash. Decode contributes a small win at high thread count (where from_float savings beat dispatch) and is neutral at low thread count.
 - When combined with --phi3-fused-lmhead, the two flags are additive at nth=10 (+12.9% total) but slightly compete at nth=1 (+1.9% total, vs +3.7% for lmhead alone).

Lesson: ggml_custom_4d is the wrong tool when the per-call work (here ~10-15 us of real work) is comparable to its dispatch overhead. Real wins require either (a) a native ggml-cpu op with full compute_params/threadpool/barrier access, or (b) Phase A custom decode block bypassing the graph entirely.

The rubber-duck pre-implementation critique predicted a regression (it identified loss-of-parallelism for the original token-only-split design); the actual cause turned out to be dispatch overhead, but the directional prediction was correct.

Real takeaways (revised):

 1. DEC at nth=1 is within noise of baseline (+0.6%) — my earlier −1.7% number was a single-run artifact. The ~30 µs/site dispatchoverhead I theorized either doesn't fully materialize or is balanced by the from_float savings even at single-thread. Lessonrestated: always run ≥3 iters before drawing perf conclusions.
 2. LMH win shrinks at nth=1: +3.7% vs +9.7% at nth=10 — expected. The persistent worker pool's parallel argmax over 32K vocabbenefits hugely from threads; at nth=1 the pool degenerates to a single-thread loop and the only remaining win is skipping thelm_head matmul.
 3. BOTH at nth=1 (+1.9%) is LESS than LMH alone (+3.7%) — a real and consistent finding across the 3 iters (each BOTH iter beats its BASE iter, but trails the LMH iter). At single-thread, DEC's custom-op dispatch slightly competes with LMH's gains. With morethreads (nth=10), DEC turns from "wash" to "small win" and they combine constructively to +12.9%.

So the corrected story is:

 - nth=10: BOTH = +12.9% (real win, lmhead does the heavy lifting, decode contributes a little)
 - nth=1: BOTH = +1.9% (lmhead win partly offset by decode overhead) — still better than baseline, but worse than lmhead alone

 =====================================================================

 Today's flow (one gen step)

 Driver: pick next token
    ↓
 llama_decode(ctx, batch[1 token])
    ↓ (inside llama)
    1. Build a ggml graph — a DAG of ~320 nodes describing the Phi3 forward pass:
         tok_embd → [RMSNorm → wqkv mul → split → RoPE → attn → wo → add
                     → RMSNorm → ffn_up mul → swiglu → ffn_down → add] × 32
                   → final_norm → lm_head
       Each box is a ggml op (ggml_rms_norm, ggml_mul_mat, ggml_rope_ext, …)
    2. Scheduler walks the DAG topologically, dispatching each op to the CPU
       threadpool. Threads sync at barriers between stages.
    3. Each op:
         - allocates/zeros a result tensor (intermediate f32 buffer)
         - dispatches its kernel
         - thread fan-out + fan-in
         - writes result, next op reads it
    4. Logits land in a designated output tensor.
    ↑
 llama_get_logits(ctx) → float* logits
    ↓
 Driver: argmax → token

The cost of this for n_tokens=1 isn't dominated by the math — it's dominated by 320 ops × per-op dispatch + 30+ barriers per token.Our Option 1 experiment hit this wall: a single ggml_custom_4d "fused op" cost ~5 µs of actual savings but ~30 µs of dispatch.

Phase A flow (one gen step)

 Driver: pick next token
    ↓
 phi3_custom_decode(custom_ctx, tok_id, pos) — ONE C++ function
    ↓ (inside our code, NO ggml graph, NO scheduler)
    embed = tok_embd_row[tok_id]                 // memcpy, ~3 µs
    for il = 0..31:
        // all of this stays in registers/L1, no intermediate ggml tensors
        norm1   = rmsnorm_inplace_with_gain(embed, attn_norm[il])
        norm1_q = quantize_to_Q8K(norm1)         // hot in L1
        qkv     = q4k_matmul(wqkv[il], norm1_q)  // calls the existing vec_dot kernel directly
        Q, K, V = split_qkv(qkv)
        rope_inplace(Q, K, pos, rope_factors)
        our_kv.K[il][pos] = K                    // append to OUR cache
        our_kv.V[il][pos] = V
        attn    = scaled_dot_attn(Q, our_kv.K[il, 0..pos], our_kv.V[il, 0..pos])
        attn    = q4k_matmul(wo[il], attn)
        embed  += attn                           // residual
        norm2   = rmsnorm_inplace_with_gain(embed, ffn_norm[il])
        gate_up = q4k_matmul(ffn_up[il], quantize_to_Q8K(norm2))
        ff      = swiglu_then_q4k_matmul(ffn_down[il], gate_up)
        embed  += ff
    final  = rmsnorm_with_gain(embed, output_norm)
    token  = lmhead_argmax(final, lm_head OR tok_embd)   // reuses Phase C pool
    ↓
 Driver: token

That's it. No graph. No scheduler. No llama_decode. No llama_get_logits. The whole thing is one C++ function that knows it's doingPhi3 and runs straight through, with the inner matmul/norm primitives being the same kernels ggml-cpu already implements — we justcall them inline instead of going through the op-dispatch machinery.

What we're actually trading

┌──────────────────────────┬────────────────────────────┬─────────────────────────────────────────────────────────────────────────┐
│                          │ Today (graph)              │ Phase A (direct)                                                        │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Per gen step: graph      │ ~320                       │ 0 (we just call functions)                                              │
│ nodes executed           │                            │                                                                         │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Per gen step: thread     │ ~30+                       │ 1 fork-join per matmul we parallelize (~7/layer × 32 = ~220) but with   │
│ barriers                 │                            │ no per-node dispatch overhead between them                              │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Intermediate f32 tensors │ every edge in the DAG      │ only what we explicitly choose to materialize                           │
│ materialized             │                            │                                                                         │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Cache locality control   │ scheduler's whim           │ ours                                                                    │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ KV cache                 │ llama-owned (paged,        │ ours (one contiguous block per layer)                                   │
│                          │ iswa-capable, multi-seq)   │                                                                         │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Math kernels             │ ggml's vec_dot Q4_K×Q8_K   │ same vec_dot routine, called inline                                     │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────────────────────────────────────────────┤
│ Correctness risk         │ zero (it's the oracle)     │ high — must reproduce byte-identical output                             │
└──────────────────────────┴────────────────────────────┴─────────────────────────────────────────────────────────────────────────┘

So the perf win comes from eliminating dispatch and barrier overhead, materializing fewer intermediate buffers, and keeping data hot in cache — not from inventing new math kernels. The matmuls themselves cost the same.

Where prefill fits in

Prefill = the very first call after llama_init_from_model, processing the entire prompt as one big batch (n_tokens = prompt_len, say15-256 tokens). Then every subsequent gen step is n_tokens=1.

The current win story is entirely about gen steps — prefill happens once and its overhead is already amortized over many matmul rows. So we only NEED Phase A to be fast at gen.

But: gen step 1 needs the KV cache populated with the prompt's K/V. Three ways to get there:

 1. Own prefill too — re-implement prefill in our custom code. Our cache is populated by our code from the start. Singlearchitecture, but means re-doing all the multi-token causal-mask correctness.
 2. Let llama prefill, then copy KV — call standard llama_decode on the prompt (one call). The KV ends up in llama's internal cache.Just before gen step 1, copy llama's K/V into ours. Then never touch llama again. This needs an internal (not public-API) function in llama-context.cpp to expose layer K/V tensors so we can memcpy them out.
 3. Share llama's KV — most aggressive, most fragile. Skip.

That's what the architectural question is. Option 2 = small surgical hook, prefill stays the oracle, we only own the perf-criticalcode. Option 1 = no llama leakage but we re-prove prefill correctness ourselves.

========================================================================

 we don't lose parallelism; we just choose where it lives. Let me walk through the three viable patterns.

What ggml does today (for context)

 ggml scheduler                                  CPU threadpool (N threads)
 ─────────────                                   ──────────────
 for each op in topological order:               while !done:
     pick threads, set ith for each              wait for op...
     fan out: launch all threads → ┐             enter op(ith, nth)
                                   ├─────────►   compute MY slice
                                   ▼             arrive at barrier
                                                 wait for siblings
     fan in: barrier                             return
     move to next op                             wait for op...

Inside each op (say ggml_compute_forward_mul_mat): every thread enters the same function with its own ith (0..N-1) and nth (N). Thefunction divides the rows of the output among threads, each does its slice, joins via an atomic-counter barrier. Then the schedulermoves to the next op.

Three things to notice:

 1. Threads stay "live" the whole time — they don't get destroyed and re-created per op. The pool is persistent.
 2. Per-op overhead = setup (1-3 µs) + barrier (1-2 µs) ≈ 3-5 µs/op of pure overhead.
 3. The math kernel inside doesn't know about the graph — it just sees (ith, nth, src0, src1, dst).

That last point is the key insight for Phase A: the inner math kernels are reusable. We can call ggml_compute_forward_mul_matdirectly from our own code.

Phase A — three parallelism patterns

Pattern A: "Per-block fork-join" (simplest, closest to today)

 main thread (driver)                            persistent worker pool (N threads)
 ────────────────────                            ──────────────────────────────────
 phi3_custom_decode(tok, pos):                   while !done:
     embed = tok_embd[tok]                         park on cv
     for il = 0..31:
        norm1 = rmsnorm(embed, gain)
        pool.run(N, |ith, nth| {                  ←────  wake all N
           q4k_matmul_slice(wqkv,                        each thread:
                             norm1, qkv,                   compute MY rows of qkv
                             ith, nth)                     atomic.fetch_add → barrier
        })                                                 signal done
        ... rope, attn, etc, each as pool.run ──── ►  park again

This is essentially our existing Phi3LmHeadPool pattern (the one we landed in Option 3), just extended to many call sites per token.Each pool.run(...) is the fork-join.

Cost: 3 µs cv wake + ~2 µs barrier = ~5 µs per pool.run.
Calls per token: ~5 matmuls × 32 layers = ~160 → **0.8 ms/token of pool overhead** (vs today's ~10 ms of total overhead). Big win.

Pattern B: "SPMD" (all threads run the whole forward together)

 driver thread is just one of N worker threads. All N enter the same function:

 phi3_custom_decode_spmd(tok, pos, ith, nth):
     embed = (ith == 0) ? tok_embd[tok] : nullptr
     barrier()
     for il = 0..31:
        if (ith == 0) norm1 = rmsnorm(embed, gain)    // tiny op, single thread
        barrier()
        q4k_matmul_slice(wqkv, norm1, qkv, ith, nth)  // all threads, divide rows
        barrier()
        ...

Every thread takes the same code path; serial ops are guarded by if (ith == 0); parallel ops divide work by (ith, nth). Only one fork at the start, one join at the end of the whole decode — barriers in between are just atomic-counter spins (no cv wake).

Cost: 0.5 µs per barrier × ~200 barriers = **0.1 ms/token of sync overhead**. Lowest possible.

But: every line of the forward function has to be written with ith/nth awareness. More restructuring. This is exactly how ggml-cpuwrites its op kernels — we'd be doing it for the whole graph rather than just one op.

Pattern C: "Hybrid" — large ops parallelized, small ops single-threaded

 phi3_custom_decode(tok, pos):
     embed = tok_embd[tok]                     // single thread, ~3 µs (tiny copy)
     for il = 0..31:
         norm1 = rmsnorm(embed, gain)          // single thread, ~30 µs (small reduce)
         pool.run_matmul(wqkv, norm1, qkv)     // FORK only for the big matmul
         rope_inplace(Q, K, ...)               // single thread, ~10 µs
         pool.run_matmul(wo, attn, ...)        // FORK only for big matmul
         ...

Only the matmuls (the only things that are actually big enough to benefit from N threads) get the fork-join treatment. RMSNorm, RoPE, residual add, and softmax all run single-threaded on the driver — they're each only ~5-30 µs, not worth the ~5 µs fork-join cost toparallelize.

This is what I'd actually implement first because it matches the existing Phi3LmHeadPool shape and is easiest to reason about. We can later promote to Pattern A or B if profiling says it helps.

Quick math: do thread counts matter?

For Phi-3-mini, the dominant cost per token is matmuls — call it ~7 ms at nth=10 today. If we have 10 threads on a matmul, eachthread does 1/10 of the rows. That parallelism MUST be preserved or we lose 10x. Patterns A, B, C all preserve it (each calls aparallel matmul slice).

What changes is only the overhead between ops (dispatch, intermediate buffer allocation, scheduler bookkeeping):

 - Today: ~10 ms/token total time, of which ~3 ms is overhead.
 - Phase A Pattern C: ~7 ms/token total time, of which ~0.5-1 ms is overhead. Expected gain: ~30-40%.
 - Patterns A, B: marginal further gains (saving the last ~0.5 ms of overhead).

Connecting to the kernels we already have

The inner functions like ggml_compute_forward_mul_mat, ggml_compute_forward_rms_norm_f32, ggml_compute_forward_rope are already parameterized by (ith, nth). They are pure C functions that take a ggml_compute_params* and a tensor pair. We can call them from ourpool tasks without reimplementing anything. The win comes from calling them sequentially without a graph in between, not fromrewriting the math.

For A3 (the aggressive fusion step) we may eventually inline the inner loops to fuse RMSNorm+quantize+matmul into one cache-hot pass. But that's a follow-up.

--------------------------------------------------------------------------------------------------------------------------------------

Recommended starting point: Pattern C with our existing Phi3LmHeadPool extended to take "matmul-style" tasks. Once we measure that,we can promote the per-step orchestration to Pattern B if the residual overhead is still too high.

================================================================

Files (planned)

 examples/phi3/
   phi3_fused_graph.h        Phi3Weights, Phi3KV, Phi3FusedCtx structs
   phi3_fused_graph.cpp      weight resolution, KV alloc, forward fn, rewind ops
   phi3_kernels.{h,cpp}      already exists — add matmul-task entrypoints
   phi3_runtime.{h,cpp}      route gen+prefill through custom path when flag set
   Phi3.cpp                  add --phi3-fused-forward CLI flag
 include/llama_b612.h        no public API change (Phi3 custom lives in examples/)

Sub-phase rollout

A0 — Weight resolution & oracle harness (1 commit)

 - Resolve all Phi3 tensors via llama_model_get_tensor_by_name.
 - Dtype check uses phi3_weight_accepts_q8K() (not type==Q4_K) — accepts Q4_K_M's mixed K-quants per duck #2.
 - Handle tied output → tok_embd fallback per duck #9.
 - Resolve rope_long/rope_short once based on n_ctx_seq > n_ctx_orig_yarn per duck #3 (not per-position, not per-layer —TENSOR_DUPLICATED per duck #8).
 - Assert non-SWA, non-MoE, single-seq.
 - No code path changes yet; just the harness + a --phi3-dump-weights debug flag.

A1 — Phi3KV alloc + rewind primitives (1 commit)

 - Allocate ~1.5 GiB KV (per duck #1 corrected math) — log exact size loudly at init.
 - Implement phi3_kv_truncate / drop_range / keep_prefix.
 - Unit test: allocate, write dummy values, truncate, verify.
 - No forward yet.

  A0 + A1 summary

  ┌──────┬───────────────────────┬──────┬─────────────────────────────────────────────────────────────────────────────┬─────────────┐
  │ step │ files                 │ LOC  │ verification                                                                │ commit      │
  ├──────┼───────────────────────┼──────┼─────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ A0   │ weights resolver +    │ ~340 │ --phi3-dump-weights confirms 12 K-quant + F32 tensors per layer × 32 +      │ 05456a51f   │
  │      │ dump                  │      │ globals                                                                     │             │
  ├──────┼───────────────────────┼──────┼─────────────────────────────────────────────────────────────────────────────┼─────────────┤
  │ A1   │ KV alloc + rewind ops │ ~340 │ --phi3-test-kv self-test PASS (init/write/truncate/drop_range/keep_prefix)  │ 2a52d4217   │
  └──────┴───────────────────────┴──────┴─────────────────────────────────────────────────────────────────────────────┴─────────────┘

  Both commits are pure-additive: no existing behavior changed, both new CLI flags are opt-in, normal generation still runs tocompletion alongside the diagnostics.

  The bigger sub-phases are next:

   - A2 (the actual functional path): phi3_custom_decode + phi3_custom_prefill calling ggml inner kernels with Pattern C threading.Single biggest piece of code. Byte-identical to baseline is the acceptance bar.
   - A3 (perf step, optional): aggressive fusion. +10% gen_tps is the acceptance bar.

  A2 is going to take real care — it's where bugs live (RoPE config, KV cell stride, attention scaling, residual order). Want me to:

A2 — Direct decode forward (1 commit, the big one)

 - phi3_custom_decode(ctx, tok_id, pos) -> token_id for n_tokens=1.
 - phi3_custom_prefill(ctx, tokens[], n) for the prompt.
 - Pattern C threading: matmuls go through pool, rest is serial.
 - Wire into Phi3Runtime: when --phi3-fused-forward set, replace llama_decode for both prefill and gen.
 - Threadpool ownership: dedicated pool sized to n_threads_gen for decode, n_threads_prefill for prefill (per duck #10 — don't assume llama_context's pool reusable).
 - Verification gate: 3 prompts × 256 tokens × seed 1234, byte-identical to baseline. No commit if not byte-identical.

A3 — Aggressive fusion (perf step) (1 commit, optional)

 - Fuse RMSNorm + quantize + matmul as a single cache-hot inlined function (the win Option 1 couldn't get because of ggml dispatchoverhead).
 - Fuse SwiGLU split + silu + mul into the path between gate_up matmul and ffn_down matmul.
 - Acceptance: gen_tps ≥ +10% over both-flags 28.27 (target ~31.0+). If not achieved, document and stop — A2 alone may already be the win.

Invariants (unchanged)

 - Non-repack only. Non-SWA only. Single-seq only. Greedy (temp=0, min_p=0) only.
 - Standard llama path remains the oracle.
 - Every commit byte-identical to baseline.

Things explicitly out of scope

 - Multi-sequence / batch > 1
 - GPU backend
 - Multi-turn seq_cp (we have keep_prefix instead)
 - Public LLAMA_API additions (everything lives in examples/phi3/)

=============================================================================

The standard Phi-3 graph supports a bunch of optional features beyond what mini-4k Q4_K_M uses. The spec says these are "out ofscope" but doesn't enforce that. If a user runs our fused path on a model file that has any of these, we silently produce wrongtokens (no crash, no error — just garbage). The duck wants me to add a concrete check at init that lists every unsupported feature.

The exhaustive rejection list (every check phi3_fused_ctx_init must perform, with the file:line in the standard graph that handleseach case):

┌──────────────┬──────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────┐
│ Feature      │ Check at init                                                                │ Standard graph handles at         │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ MoE          │ for il: layers[il].ffn_gate_inp == nullptr && ffn_up_exps == nullptr &&      │ phi3.cpp:319, 327-340             │
│              │ ffn_gate_exps == nullptr && ffn_down_exps == nullptr                         │                                   │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ GQA          │ hparams.n_head == hparams.n_head_kv                                          │ implicit via n_embd_gqa           │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ SWA          │ hparams.swa_type == LLAMA_SWA_TYPE_NONE                                      │ phi3.cpp:165-176 (already         │
│              │                                                                              │ auto-disabled but defensive)      │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ QKV bias     │ for il: wqkv_b == nullptr (resolve via attn_qkv.bias)                        │ build_qkv adds bias if present    │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Attn output  │ for il: wo_b == nullptr                                                      │ build_attn(.., wo_b, ..) arg      │
│ bias         │                                                                              │ phi3.cpp:292                      │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Attn output  │ for il: wo_s == nullptr                                                      │ build_attn(.., wo_s, ..)          │
│ scale        │                                                                              │                                   │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Attn-norm    │ for il: attn_norm_b == nullptr                                               │ build_norm(.., attn_norm_b, ..)   │
│ bias         │                                                                              │ phi3.cpp:264-265                  │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ FFN-norm     │ for il: ffn_norm_b == nullptr                                                │ build_norm(.., ffn_norm_b, ..)    │
│ bias         │                                                                              │ phi3.cpp:313                      │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Final output │ output_b == nullptr                                                          │ phi3.cpp:370-372                  │
│ bias         │                                                                              │                                   │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Output scale │ output_s == nullptr                                                          │ build_lora_mm(.., output_s)       │
│              │                                                                              │ phi3.cpp:368                      │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ LoRA         │ loras->empty() (need accessor or pass from runtime)                          │ build_lora_mm checks per-layer    │
│ adapters     │                                                                              │                                   │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Control      │ no active cvec (need accessor)                                               │ build_cvec(cur, il) phi3.cpp:344  │
│ vectors      │                                                                              │                                   │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Flash        │ cparams.flash_attn == false                                                  │ build_attn branches               │
│ attention    │                                                                              │ llama-graph.cpp:1963-1981         │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ YARN long    │ cparams.n_ctx_seq <= hparams.n_ctx_orig_yarn (or rope_long/short are absent) │ model.get_rope_factors()          │
│ context      │                                                                              │ llama-model.cpp:2026-2038         │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Embed type   │ traits(tok_embd->type) supports dequantize_row                               │ direct                            │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Weight types │ for every K-quant weight: traits->vec_dot_type != GGML_TYPE_COUNT            │ direct                            │
├──────────────┼──────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│ Tensor       │ for every weight: w->data != nullptr && w->buffer is host && nb[1] ==        │ this is duck issue #1             │
│ canonicality │ row_size                                                                     │                                   │
└──────────────┴──────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────┘

The behavior contract:

 bool phi3_fused_ctx_init(...) {
     // Run all checks. On first failure, set error to a specific message
     // ("MoE not supported in fused path: layer 3 has ffn_gate_inp") and
     // return false. Caller decides whether to abort or fall back.
 }

Runtime contract:

 if (runtime.use_fused_forward) {
     if (!phi3_fused_ctx_init(...)) {
         // User explicitly asked --phi3-fused-forward.
         // Abort with clear error rather than silently falling back.
         fprintf(stderr, "phi3 fused-forward unavailable: %s\n", err.c_str());
         return 1;
     }
 }

Why "rejection" beats "documented as out of scope":

Right now if I write the spec saying "MoE: out of scope, falls back", and a user grabs Phi-3.5-MoE.gguf (same LLM_ARCH_PHI3 archenum, different tensors), they pass --phi3-fused-forward, our path runs, we read wqkv (which doesn't exist for MoE),phi3_weights_resolve either crashes on a NULL deref or hands back a Phi3Weights with garbage in it, and we emit nonsense tokens withno warning. The explicit rejection turns that into a one-line error message before any decode happens.

Same logic for every row above: each is something a real Phi-3 family model could have that breaks our assumed math.

===================================================================================

CLI reference — Phi3.exe
=========================

Source of truth: examples/phi3/Phi3.cpp (print_usage at line 18, dispatch lines
105-322). One-line semantics + defaults + when to use. Group order mirrors the
order in print_usage.

Core (always honoured)
----------------------
  -m PATH                          GGUF model file. Required for everything
                                   except --phi3-kernel-test and the cached
                                   prefill load path (which still needs the
                                   model to rebuild weights).
  -p "TEXT"                        Single prompt. If absent, runs an internal
                                   chat loop. Some test/debug flags treat this
                                   as the prompt to validate against.
  -c N                             llama context size (default model max).
  -ngl N                           Layers to offload (default 99 = all GPU when
                                   a GPU backend is built; -ngl 0 forces CPU).
  -n N                             Tokens to predict in the chat/gen loop
                                   (default 256).
  -s SEED                          RNG seed (default 1234).
  --temp F                         Sampling temperature (default 0.0 = greedy).
                                   Must be 0 when --phi3-fused-lmhead is set.
  --min-p F                        min-p sampler threshold (default 0.05).
                                   Must be 0 when --phi3-fused-lmhead is set.
  --threads-prefill N              Prefill thread count (default = system).
  --threads-gen N                  Decode/gen thread count (default 0 = auto:
                                   inherits prefill count unless overridden).
  --threads-gen-auto               Probe a small grid (1/2/4/...) at runtime
                                   and lock in the fastest count for steady-
                                   state decode.

Runtime fast paths (chat loop, opt-in)
--------------------------------------
  --phi3-fused-lmhead              Replace upstream lm_head matmul with our
                                   AVX-512 Q4_K-aware fused lm_head+argmax.
                                   Greedy only (requires --temp 0 --min-p 0).
                                   Returns a single token id; skips writing
                                   the 32064-wide logits buffer.
  --phi3-fused-decode              Enable the full custom Phi-3 forward path
                                   for decode steps (Phase A). Pairs with the
                                   fused lmhead automatically.

Tensor repack (one of, mutually exclusive)
------------------------------------------
  --repack-ggml                    Run upstream ggml repack pass after model
                                   load (vanilla _x8 layout).
  --repack-xbox                    Internal XBOX repack layout.
  --repack-xbcg                    Internal XBCG repack layout.
                                   None of the above => no repack.

One-shot tests (run, print PASS/FAIL, exit; no chat loop)
---------------------------------------------------------
  --phi3-dump-weights              Resolve+print Phi3Weights schema. No
                                   compute. Good first run on a new GGUF.
  --phi3-test-kv                   Exercise the Phi3KV cache shapes against
                                   the resolved weights.
  --phi3-matmul-test               Compare our matmul shim against ggml on
                                   the model's actual weight shapes.
  --phi3-kernel-test               Pure kernel unit tests (RMSNorm, RoPE,
                                   SwiGLU, rmsnorm+quant_q8K, ...). Does NOT
                                   need a model file beyond what the build
                                   links; safe to run early.
  --phi3-qmatmul-test              Q4_K / Q6_K matmul shim parity vs upstream
                                   ggml_mul_mat on real weight tensors.
  --phi3-layer-test                Hand-rolled single-layer forward vs upstream
                                   oracle, on layer 0. Prints max-abs / cos-sim.
  --phi3-full-test                 Full hand network vs upstream last-token
                                   logits on a baked-in prompt.
  --phi3-validate-fused            Run the Phase A fused-forward validator
                                   (rejects unsupported variants, ACCEPTS on
                                   supported ones). Forces flash_attn=off so
                                   the validator can demonstrate the ACCEPT
                                   path on a normal model.

Fused-forward debug harnesses (F32 reference path)
--------------------------------------------------
  --phi3-fused-f32-debug           Run the F32-reference custom forward for a
                                   short gen and compare token-by-token to
                                   the llama baseline. Sampling caveat: warns
                                   if --temp > 0 (baseline is sampling so
                                   token-level match may diverge).
  --phi3-fused-f32-n-gen N         Tokens to generate in --phi3-fused-f32-
                                   debug (default 4).

Fused-forward debug harnesses (qquant production path)
------------------------------------------------------
  --phi3-fused-qquant-debug        Per-token spot-check: run our qquant decode
                                   alongside llama baseline and report top-1
                                   agreement + per-stage timing for the first
                                   N tokens. Pairs with several flags below.
  --phi3-fused-qquant-n-gen N      Tokens to spot-check (default 4).
  --phi3-fused-qquant-threads N    Override decode thread count for the qquant
                                   path only (default 0 = use --threads-gen).
                                   Lets you tune qquant pool independently
                                   from the regular llama decode pool.
  --phi3-fused-qquant-rmsnorm-fuse 0|1
                                   A/B switch for the fused RMSNorm + quantize
                                   path at the two *_norm sites. Default 0;
                                   1 enables. Bit-identical (covered by
                                   --phi3-kernel-test's "rmsnorm+quant_q8K").
                                   Measured net effect on Phi-3-mini was
                                   neutral-to-negative at 8 threads.
  --phi3-fused-qquant-attn-parallel 0|1
                                   Parallelise the per-head attention loop in
                                   qquant decode across the matmul pool.
                                   Default 0. Bit-identical; only matters
                                   when --threads-gen > 1.
  --phi3-fused-qquant-hybrid 0|1   Replace per-token qquant prefill with a
                                   single batched llama_decode and bridge the
                                   resulting K/V into Phi3KV (A5.2/A5.3).
                                   Steady-state gen path is unchanged.
                                   Default 0. No effect unless
                                   --phi3-fused-qquant-debug is also set.
  --phi3-fused-qquant-profile      Enable the per-op RAII profiler in qquant
                                   decode and print a bucketed µs/token + a
                                   matmul-vs-other split at end of gen. Resets
                                   counters between prefill and gen so the
                                   summary is steady-state. Off by default.
  --phi3-fused-qquant-regress [N]  Run-and-exit: 3-prompt agreement vs
                                   baseline. N = tokens per prompt (default
                                   256). Exit status 0 = PASS, 1 = FAIL.
                                   Replaces the chat loop.

Fused-forward A/B harness
-------------------------
  --phi3-fused-qquant-ab           Run-and-exit: llama-batched (oracle) vs
                                   qquant-per-token vs qquant-hybrid back-to-
                                   back and print a comparison table (timings
                                   + top-1 agreement). Replaces the regular
                                   --phi3-fused-qquant-debug spot-check.
  --phi3-fused-qquant-ab-ngen N    Tokens to generate inside the AB harness
                                   (default 32).
  --phi3-fused-qquant-ab-ngen-compare N
                                   Tokens to top-1 compare across the three
                                   modes (default 20, must be <= ab-ngen).

Cached prefill (A5.5/A5.6) — mutually exclusive
-----------------------------------------------
  --phi3-fused-qquant-save-kv PATH Save the post-prefill Phi3KV state to PATH
                                   after prefill completes. Requires
                                   --phi3-fused-qquant-hybrid 1 AND
                                   --phi3-fused-qquant-debug (the hybrid path
                                   is what populates llama's K/V before the
                                   bridge).
  --phi3-fused-qquant-load-kv PATH Skip prefill entirely; resume gen from a
                                   cached Phi3KV state. -p is optional and
                                   only used for an advisory hash check.
  --phi3-fused-qquant-load-kv-strict
                                   Promote the prompt-hash mismatch warning
                                   from --phi3-fused-qquant-load-kv into a
                                   fatal error.
  --phi3-fused-qquant-a56          Long-prompt cached-prefill sweep harness
                                   (A5.6). Run-and-exit; needs no prompt.

  --phi3-bench                     llama-bench-style throughput table.
                                   Run-and-exit. Prints a markdown table
                                   identical in shape to `llama-bench.exe`
                                   output, with one pp{N} row and one
                                   tg{N} row per backend selected. Uses
                                   random in-vocab tokens (content-agnostic
                                   throughput probe; matches llama-bench's
                                   test_prompt / test_gen helpers).
  --bench-pp N                     Prefill size (default 64). Backed by a
                                   single phi3_run_qquant_decode call with
                                   prompt=N random tokens, n_gen=1
                                   (qquant) or a single llama_decode batch
                                   of N tokens (upstream).
  --bench-tg N                     Decode size (default 64). N single-token
                                   decode steps after a 1-token warmup.
  --bench-reps N                   Measured repeats per test (default 3).
                                   One implicit warmup pass per test is
                                   always discarded.
  --bench-threads N                Thread count for both backends. When
                                   set, overrides the default resolution
                                   (--phi3-fused-qquant-threads →
                                   --threads-gen → 1).
  --bench-backend qquant|upstream|both
                                   Which backends to bench (default both).
                                   qquant honours --phi3-fused-qquant-
                                   rmsnorm-fuse / --phi3-fused-qquant-
                                   attn-parallel; upstream uses
                                   llama_decode with flash_attn DISABLED.
                                   Thread count comes from
                                   --phi3-fused-qquant-threads (or
                                   --threads-gen).

Quick recipes
-------------
  # Sanity (no model needed beyond build)
  Phi3.exe --phi3-kernel-test
  # Model schema + shapes
  Phi3.exe -m M.gguf --phi3-dump-weights
  Phi3.exe -m M.gguf --phi3-test-kv --phi3-qmatmul-test --phi3-layer-test
  # Hand-vs-oracle full-network smoke
  Phi3.exe -m M.gguf --phi3-full-test
  # Greedy chat with both runtime fast paths
  Phi3.exe -m M.gguf -p "Hi" -n 64 --temp 0 --min-p 0 \
           --phi3-fused-decode --phi3-fused-lmhead --threads-gen 8
  # Qquant gen spot-check + profile
  Phi3.exe -m M.gguf -p "Hi" --phi3-fused-qquant-debug \
           --phi3-fused-qquant-profile --phi3-fused-qquant-n-gen 16
  # Cached-prefill round trip
  Phi3.exe -m M.gguf -p "long prompt..." \
           --phi3-fused-qquant-debug --phi3-fused-qquant-hybrid 1 \
           --phi3-fused-qquant-save-kv kv.bin
  Phi3.exe -m M.gguf --phi3-fused-qquant-load-kv kv.bin -n 64
  # 3-prompt regression gate
  Phi3.exe -m M.gguf --phi3-fused-qquant-regress 256
  # llama-bench-style throughput (qquant + upstream side-by-side)
  Phi3.exe -m M.gguf --phi3-bench --bench-pp 64 --bench-tg 64 \
           --bench-reps 3 --threads-gen 8
  # qquant only at a larger context, with attn-parallel on
  Phi3.exe -m M.gguf --phi3-bench --bench-backend qquant \
           --bench-pp 128 --bench-tg 128 \
           --phi3-fused-qquant-attn-parallel 1 --threads-gen 8
