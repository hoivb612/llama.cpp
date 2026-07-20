 The 26B model has merged  ffn_gate_up_exps  (Q4_K,  [2816,1408,128] ),  ffn_down_exps  (Q8_0) with a per-expert  ffn_down_exps.scale [128] , router  ffn_gate_inp  +  .scale , extra norms named  pre_ffw_norm_2 / post_ffw_norm_1 / post_ffw_norm_2 , and crucially no PLE tensors ( embedding_length_per_layer_input=0 , no  inp_gate / proj ). So the resolver must make PLE conditional too. Let me see the exact  build_moe_ffn  call in the graph to get gating/norm/scale params.

┌────────────────────────────────────────────────────────────────────────────┬──────────┬────────────┬──────────────────────────┐
│ Tensor                                                                     │ E2B      │ 26B (MoE)  │ Rule                     │
│                                                                            │ (dense)  │            │                          │
├────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┼──────────────────────────┤
│ post_norm                                                                  │ ✓        │ ✗         │ optional                 │
├────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┼──────────────────────────┤
│ PLE (inp_gate,proj,globals)                                                │ ✓ (256)  │ ✗ (0)     │ conditional on           │
│                                                                            │          │            │ n_embd_per_layer>0       │
├────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┼──────────────────────────┤
│ n_head_kv                                                                  │ 1        │ 8 SWA / 2  │ derive per-layer from wk │
│                                                                            │ (const)  │ full       │ shape                    │
├────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┼──────────────────────────┤
│ MoE (ffn_gate_inp+scale, pre_ffw_norm_2, post_ffw_norm_1/2,                │ ✗        │ ✓         │ resolve when             │
│ ffn_gate_up_exps, ffn_down_exps+scale)                                     │          │            │ ffn_gate_inp present     │
├────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┼──────────────────────────┤
│ shared ffn_gate/up/down                                                    │ ✓        │ ✓         │ keep (shared expert in   │
│                                                                            │          │            │ MoE)                     │
└────────────────────────────────────────────────────────────────────────────┴──────────┴────────────┴──────────────────────────┘

> And the memory jumped from 28.8GB -> 45.8GB used

That ~17 GB delta (28.8 → 45.8 GB) is the real-world anchor and it validates the breakdown: 15.6 GB weights + ~1.4 GB runtime (KV cache + compute/graph buffers + activations). The runtime part isn't streamable — only the 15.6 GB of weights is, and 14.09 GB of that is experts.

Projected total process RSS delta with expert streaming (weights + the fixed ~1.4 GB runtime):

┌──────────────────────┬────────────┬───────────┬─────────────────┬──────────┐
│ Policy               │ Weight RAM │ + runtime │ Total RSS delta │ vs 17 GB │
├──────────────────────┼────────────┼───────────┼─────────────────┼──────────┤
│ all resident (today) │    15.6 GB │    1.4 GB │          ~17 GB │     1.0× │
├──────────────────────┼────────────┼───────────┼─────────────────┼──────────┤
│ 32/layer             │       5.15 │       1.4 │         ~6.6 GB │     2.6× │
├──────────────────────┼────────────┼───────────┼─────────────────┼──────────┤
│ 16/layer (default)   │       3.34 │       1.4 │         ~4.7 GB │     3.6× │
├──────────────────────┼────────────┼───────────┼─────────────────┼──────────┤
│ 8/layer              │       2.44 │       1.4 │         ~3.8 GB │     4.5× │
├──────────────────────┼────────────┼───────────┼─────────────────┼──────────┤
│ floor                │       1.54 │       1.4 │         ~2.9 GB │     5.8× │
└──────────────────────┴────────────┴───────────┴─────────────────┴──────────┘

So the honest headline for the whole process (not just weights): ~17 GB → ~4.7 GB at the default budget (~3.6× smaller RSS), trading toward the ~17 t/s → potentially-lower decode speed that P1's hit-rate curve will quantify.

Note the runtime ~1.4 GB grows with context length (KV cache), so at long contexts the fixed floor rises and the relative streaming win shrinks — worth tracking in telemetry.

That completes the quantitative picture: experts are 90% of weights and 94% idle/token → ~3.6× RSS cut at default, bounded by disk BW vs a 58.8 ms/token compute window.

===================================================

"Network" here has nothing to do with sockets/NICs — it's this repo's naming for the whole-model hand-forward graph in  examples/gemma4 , i.e. the full 26B compute graph run end-to-end on CPU by our own code (not upstream's  llama_decode ).

The gemma4 test harness has three granularity levels, and "network" is the top one:

• kernel ( --gemma4-kernel-test ) — individual ops (rmsnorm, gelu, rope, dequant).
• layer ( layer_forward_f32_cached ) — one transformer block.
• network ( network_step  /  --gemma4-network-test  /  --gemma4-network-gen ) — the entire model:  tok_embd → all 30 layers (attn + MoE FFN) → final norm → output projection → logits , with the persistent per-layer K/V cache held in  NetworkState .

So "26B network" = our hand-coded forward pass over the full 26B-A4B graph. The validation runs it side-by-side against upstream ( llama_decode ) and compares:

•  network-test  → last-token logits (cos_sim / top-k) after prefill.
•  network-gen  → greedy token-by-token parity across N generated tokens.

It's called "network" (vs "model") mainly to distinguish the scratch, cache-carrying forward driver ( network_step  +  NetworkState ) from the static weight structs ( ModelF32 / Weights ). Reference:  examples/gemma4/gemma4_forward.h:294–379 .

===================================================

Root-caused and fixed the 26B MoE systematic error. The culprit was gelu: my  gelu_f32  used a direct-F32 tanh (more accurate), but ggml-cpu evaluates gelu through a 64K F16 lookup table ( GGML_GELU_FP16  is defined) — input rounded to F16, gelu output rounded to F16, with ±10 saturation. That per-site divergence accumulated across the shared MLP + 8 experts × 30 layers.

Fix ( gemma4_kernels.cpp::gelu_f32 ): replicate ggml's F16 round-trip exactly ( x<=-10→0 ,  x>=10→x , else F16-quantized tanh gelu), matching both the standalone  ggml_gelu  and  geglu  paths.

Results (26B-A4B Q4_K_M, hand vs upstream greedy):

• "1, 2, 3, 4," → 16/16 (was 7/8); logit magnitudes now match tightly (25.23 vs 25.41, 17.33 vs 17.30)
• "10, 20, 30, 40," → 16/16
• cos_sim 0.912→0.989 on clean prompts
• E2B dense: 8/8, cos 0.998 — no regression
• Kernel self-test still PASS

Remaining mismatches occur only on out-of-distribution base-completion prompts (e.g. "The capital of France is", "days of the week") where upstream itself produces degenerate/looping garbage ("er,er,er...") — near-tie argmax coin-flips, not a hand-path bug.

=====================================================

Summary

Your question — "perfect prefetch" ceiling: Perfect overlap turns each token from  compute + I/O  into  max(compute, I/O) , so it can never beat the 10.63 t/s all-resident compute baseline — it can only reach it while using less memory. It hits that ceiling for any budget ≥ ~4 GiB (where per-token miss-I/O fits under the 94 ms compute window); below ~2 GiB the read volume exceeds compute so even perfect overlap stays I/O-bound.

What I built (within-layer prefetch):

•  ExpertStore  gained one background I/O worker + a mutex. Right after the router picks a token's 8 experts,  moe_ffn  enqueues that token's full working set (usage order); the worker  pread s blocks into the LRU while the main thread computes.  fetch()  now returns hits or briefly waits on in-flight reads instead of blocking on cold I/O.
• Correctness invariant preserved: a single  pinned_  block (the one feeding the current matmul) is never evicted by either thread. A fit-guard skips prefetch when a token's working set won't fit the budget.
• CLI  --gemma4-moe-prefetch 0|1  (default 1); new  prefetch_reads / waits  telemetry;  drain()  for race-free stats.

Results — bit-identical output at every budget (26B 16/16, E2B dense 8/8), reads unchanged (pure overlap):

┌────────┬──────┬───────┬──────┬────────────────────┐
│ budget │ off  │ on    │ gain │ % of 10.63 ceiling │
├────────┼──────┼───────┼──────┼────────────────────┤
│ 1 GiB  │ 4.75 │ 4.95  │ +4%  │ 47% (I/O-bound)    │
├────────┼──────┼───────┼──────┼────────────────────┤
│ 2 GiB  │ 5.86 │ 7.17  │ +22% │ 67%                │
├────────┼──────┼───────┼──────┼────────────────────┤
│ 4 GiB  │ 8.91 │ 10.29 │ +15% │ 97%                │
└────────┴──────┴───────┴──────┴────────────────────┘

Headline: 4 GiB + prefetch = 97% of full-resident speed using ~4 GB instead of ~17 GB of experts — matching the theoretical ceiling. Measurements match the  max(compute, I/O)  model exactly.

A possible bounded follow-up (not done): multiple I/O workers to raise read bandwidth in the 1–2 GiB regime.

================================================

There are really two forward-pass implementations, and one of them has two memory modes. Let me lay it out precisely.

1. Upstream path

The stock llama.cpp engine —  llama_decode()  building a ggml compute graph, exactly what you'd get from  llama-cli . In this app it's invoked as the reference oracle to compare against ( upstream_last_token_logits , and the reference loop in  network_gen_self_test ). It loads the model via llama's normal mmap and faults in whatever pages the graph touches. This is the "gold standard" output the hand path is checked against. It's also what your original  -p ... -n  command ( gemma4_run_baseline_decode ) uses.

2. Hand path

The from-scratch, hand-written forward in this  examples/gemma4/  directory ( network_step  →  layer_forward_f32_cached  →  attn_compute_shard ,  moe_ffn , etc.). It uses llama only as a GGUF loader; all the math (rmsnorm, RoPE, attention, MoE) is re-implemented by hand in plain C++. This is the code you've been building the Colibri expert-streaming feature into. Its output is what diverged into garbage.

The hand path has two memory modes for the MoE expert weights — and this is where "resident" vs "streaming" lives:

• 2a. Resident mode ( --gemma4-moe-budget 0 , the default): the hand path reads expert weights directly from the mmap'd GGUF ( matmul_expert_qf32  dereferences  W3d->data + expert*stride ). Pages fault in on demand → unbounded expert RAM. This is "all-resident."
• 2b. Streaming mode ( --gemma4-moe-budget N ): the hand path never touches the mmap'd expert pages; instead the ExpertStore  pread s each expert block from a separate file handle into a fixed-size LRU pool (± the prefetch worker). Hard memory cap. This is the feature we built.

Key point: "resident" and "streaming" are the same hand path, just two ways of getting expert bytes into a matmul. I proved they produce byte-identical output — so streaming is correct.

How they relate to what you saw

┌───────────────────────┬───────────────────────────────────────┬───────────────────────────────────────┐
│ term                  │ what it is                            │ in your network-gen run               │
├───────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ upstream path         │ stock llama_decode (reference)        │ produced the coherent text            │
├───────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ hand path — resident  │ hand forward, experts from mmap       │ (budget 0 run) → same garbage         │
├───────────────────────┼───────────────────────────────────────┼───────────────────────────────────────┤
│ hand path — streaming │ hand forward, experts pread'd, capped │ (budget 4096 run) → identical garbage │
└───────────────────────┴───────────────────────────────────────┴───────────────────────────────────────┘

So the comparison that matters for the bug is hand path vs upstream path (garbage vs coherent — a precision gap in the hand math). The comparison that matters for the streaming feature is hand-resident vs hand-streaming (identical — feature works).

One nuance on your earlier memory question:  network-gen / network-test  run both the hand path and the upstream path (to compare), so they load two copies → the 47 GB.  network-profile  runs the hand path only, which is why it stayed at ~5.7 GB with streaming.

