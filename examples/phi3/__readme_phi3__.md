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