In GGML, a weight tensor's shape prints as ne[0], ne[1] = [K, N] where ne[0] (K) is the contraction/input dim and ne[1] (N) is the output dim. ggml_mul_mat(W, x) requires x->ne[0] == W->ne[0], and produces [N, n_tokens].

Of your 6 tensors, only 4 are mul_mat weights. The two *_norm.weight (f32, [3072]) are not matmuls — they're the RMSNorm scale, applied via ggml_rms_norm then elementwise ggml_mul.

For Phi-3-mini (n_embd=3072, n_head=n_head_kv=32, head_dim=96, n_ff=8192), the standard GGML graph per layer:

┌───────────────┬─────────────┬───────┬────────────────────────────────────────────────────────────────────────────────────┐
│ Tensor        │ Shape [K,N] │ Quant │ mul_mat role                                                                       │
├───────────────┼─────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────┤
│ attn_qkv      │ [3072,      │ q5_K  │ qkv = mul_mat(Wqkv, attn_norm_out) → [9216, T], split Q/K/V (3072 each;            │
│               │ 9216]       │       │ 9216=3×3072, no GQA)                                                               │
├───────────────┼─────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────┤
│ attn_output   │ [3072,      │ q4_K  │ cur = mul_mat(Wo, attn_out) → [3072, T]                                            │
│               │ 3072]       │       │                                                                                    │
├───────────────┼─────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ffn_up        │ [3072,      │ q4_K  │ gate_up = mul_mat(Wup, ffn_norm_out) → [16384, T]; 16384=2·n_ff, SwiGLU splits     │
│               │ 16384]      │       │ gate∥up → [8192, T]                                                                │
├───────────────┼─────────────┼───────┼────────────────────────────────────────────────────────────────────────────────────┤
│ ffn_down      │ [8192,      │ q6_K  │ cur = mul_mat(Wdown, swiglu_out) → [3072, T]                                       │
│               │ 3072]       │       │                                                                                    │
└───────────────┴─────────────┴───────┴────────────────────────────────────────────────────────────────────────────────────┘

Notes on the GGML execution:

 - src0 = weight (quantized), src1 = activation (f32). Inside ggml_compute_forward_mul_mat, src1 is quantized on-the-fly to the weight's vec_dot_type (q8_K for all K-quants here) via from_float, then the quantized dot product runs. Output is always f32.
 - The two attention-core matmuls (QK^T and softmax·V) are NOT in this list — they multiply activations against the KV-cache, not these weight tensors.
 - So per layer you get 4 weight-matmuls (qkv, output, up, down) plus 2 cache-matmuls in attention. The element counts in your dump's first column are just ne[0]×ne[1] (e.g. ffn_up: 3072×16384 = 50331648).

This is exactly the plan src/models/phi3.cpp builds — your phi3_fused_graph.cpp collapses the norm→matmul junctions and the SwiGLU split to avoid the f32 intermediates, but the matmul roles are identical.

The src1 tensor in the graph is always f32. The key line is ggml-cpu.c:1651:

 const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

So what actually happens in ggml_compute_forward_mul_mat:

 1. The weight (src0) determines vec_dot_type — for all K-quants that's GGML_TYPE_Q8_K.
 2. Since src1->type (f32) ≠ vec_dot_type (q8_K), GGML quantizes src1 into a separate scratch work buffer params->wdata (sized in row_size), leaving the src1 tensor itself untouched and f32.
 3. vec_dot (e.g. ggml_vec_dot_q4_K_q8_K) then runs over src0's q4_K rows against the q8_K copy in wdata.

So my earlier "src1 is quantized" was sloppy shorthand. Accurately: src1 stays f32; a quantized q8_K copy of it is staged in the op's wdata scratch. If you ever feed a matmul a src1 that's already the right vec_dot_type, GGML skips the copy and reads src1->data directly — that's the ? : branch, and it's exactly the optimization your phi3_fused_graph exploits by pre-quantizing the norm output to q8_K.

src1 is the activation (hidden state) — computed live at inference time, not loaded from the GGUF. The GGUF only supplies the weights (src0). src1 depends on your actual input tokens, so it cannot be precomputed/stored.

Here's where the f32 in src1 comes from, traced through one Phi-3 layer:

 1. Origin — token embeddings. The graph starts with inpL = ggml_get_rows(token_embd.weight, input_tokens). token_embd is from the GGUF (quantized), but get_rows dequantizes the selected rows to f32. Result inpL = [n_embd, n_tokens] f32. This is the only point where GGUF data directly seeds the activation stream.
 2. attn_norm = ggml_rms_norm(inpL) then ggml_mul by attn_norm.weight → f32. → this is src1 for the attn_qkv matmul (src0 = q5_K weight).
 3. qkv output (f32) → split Q/K/V → RoPE → attention (QK^T, softmax, ·V, all on f32/cache) → attn_out f32 → src1 for attn_output matmul.
 4. residual add → ffn_norm (rms_norm × ffn_norm.weight) f32 → src1 for ffn_up matmul.
 5. SwiGLU (silu(gate)·up) f32 → src1 for ffn_down matmul.
 6. residual add → next layer's inpL.

So the pattern is: src0 (weights) = static, from GGUF, quantized. src1 (activations) = dynamic, computed by the preceding op, f32. Every matmul's src1 is the f32 output of the norm/attention/swiglu step just before it. The q8_K quantization of src1 happens inside each matmul into scratch wdata — the activation tensor that lives in the graph stays f32 the whole way through.

input_tokens is a 1-D tensor of token IDs of length n_tokens, where n_tokens is the batch size of that particular forward pass. It's not fed one-at-a-time per layer; the whole batch flows through all layers together.

Two cases:

 - Prefill (prompt processing): input_tokens = the entire prompt at once, e.g. 128 IDs. get_rows produces inpL = [n_embd, 128]. All 128 tokens pass through layer 0, then layer 1, … layer 34 together. This is why prefill is compute-bound — the matmuls have M = 128 columns of work (GEMM).
 - Decode (generation): input_tokens = just 1 ID (the token sampled from the previous step). inpL = [n_embd, 1]. One token through all layers. The matmuls are now GEMV (M = 1) — memory-bandwidth-bound, which is why tg is slower per-FLOP than pp.

So to your sub-question: it's a series of tokens during prefill, a single token during decode — never "one token per layer." Within a pass, the n_tokens dimension is the batch/sequence axis (ne[1] of the activation, the M in the [N, M] matmul output). Each token's hidden state is an independent column; they only interact inside the attention op (via QK^T over the KV cache), not in the qkv/output/ffn matmuls — there, each column is processed independently.

Where the IDs come from:

 - Prefill IDs = tokenizer output on your prompt text.
 - Decode IDs = the sampler's previous output (argmax/top-k/etc. over the final logits), appended one at a time and fed back.

token_embd.weight (from GGUF, quantized) is the lookup table; get_rows selects and dequantizes the rows for whichever IDs are in the current batch.

 A hidden state (activation) is the model's internal numerical representation of a token at some point in the network — a vector of n_embd floats (3072 for Phi-3-mini) that encodes "what this token means in this context so far."

  Concretely: for each token, the activation is a 1-D f32 vector of length n_embd. For a batch of n_tokens, you stack them into the [n_embd, n_tokens] matrix that flows through the graph. One column = one token's current state.

  Why "hidden": it's neither the input (token IDs) nor the output (logits/probabilities) — it's the intermediate state living between layers. It's not human-interpretable; it's just a point in a 3072-dimensional space the model learned to use.

  Why "activation": the term comes from neural nets — it's the output values of a layer's neurons after they "activate" (compute + nonlinearity), as opposed to the weights (the learned parameters). Weights are static; activations are recomputed for every input.

  Its life through Phi-3 (one token's column):

   1. Born as the embedding vector: get_rows(token_embd, id) → the raw 3072-vector for that token. At this point it encodes the token in isolation (no context).
   2. Refined by each of the 35 layers. Each layer reads the current hidden state and adds an update: - Attention sublayer mixes in information from other tokens (via the KV cache) → now the vector encodes context.
   - FFN sublayer transforms it further per-token.
   - h = h + attn(h) then h = h + ffn(h) — the residual structure means each layer nudges the state rather than replacing it. The hidden state is a running accumulator.
   3. Consumed at the end: after the last layer + final norm, the hidden state is multiplied by the output/token_embd matrix → logits → next-token probabilities.

  Key properties:

   - Same shape everywhere — n_embd wide at every layer boundary, so layers can stack uniformly and residuals can add.
   - Dynamic — depends entirely on the input tokens; this is exactly the f32 src1 that can't come from the GGUF, because it doesn't exist until you run the model on real input.
   - One per token — a 10-token prompt has 10 hidden states evolving in parallel through the layers; attention is the only place they exchange information.

  So in our earlier matmul discussion: every src1 was a hidden state — the current [n_embd, n_tokens] activation matrix being projected by a weight (qkv, output, ffn_up/down) to produce the next hidden state.

What happens at the end of prefill:

After the prompt's [n_embd, n_tokens] activation exits layer 34 + final norm, you have n_tokens final hidden states. Only the last one (position n_tokens-1) matters, because that position predicts the next token. So:

 - Take column n_tokens-1 → multiply by the output head → logits [n_vocab] → sample → token #n_tokens+1.
 - The other n_tokens-1 final hidden states are discarded. (llama.cpp optimizes this: via n_outputs/the logits flag it usually computes logits for only the last position, skipping the output matmul for the rest.)

So how does the single decode token "remember" the whole prompt? The KV cache. This is the piece that does the real work:

 - During prefill, every layer's attention computed K and V vectors for all n_tokens and stored them in the cache (per layer, per token, per head). That's 2 × n_layers × n_tokens × n_embd_kv of saved state.
 - The prompt's context lives there — not in any single hidden state.

Decode (sequential phase):

 1. Feed the 1 newly-sampled token → get_rows → its [n_embd, 1] hidden state.
 2. It flows through all 35 layers as a single column (GEMV matmuls).
 3. At each layer's attention, its query attends over the entire KV cache (all prompt tokens + everything generated so far). New K/V for this one token are appended to the cache.
 4. Final hidden state → logits → sample next token → repeat.

So the transition is: prefill fills the KV cache and emits one token from the last position; decode then runs one token at a time, each reading the whole cache and growing it by one entry. Nothing is "collapsed" — you just stop carrying the full sequence in the activation stream (that would be redundant, since the KV cache already holds each past token's contribution) and let attention re-gather context from the cache on every step.

1. The KV cache stores all n_tokens, not n_tokens − 1. Every prompt token — including the last one — gets its K/V written during prefill. So after prefill the cache holds n_tokens entries. (The last token's K/V must be there too, otherwise it couldn't attend to itself.) What's discarded is the last token's final hidden state after it's been turned into logits — not its K/V.

2. The token fed sequentially is the newly sampled token, not the n_tokens-th prompt token. The last prompt token's logits predict a brand-new token #(n_tokens+1). That sampled token is what enters decode first — you don't re-feed a prompt token. Each decode step:

 - feed the just-sampled token → flows through layers → its K/V appended to the cache (now n_tokens+1, then +2, …)
 - last-position logits → sample next → repeat until EOS or token budget.

So the corrected picture:

 Prefill writes K/V for all n_tokens into the cache and emits logits from the last position → sample token #(n_tokens+1). That sampled token is fed alone through the layers; its K/V is appended; sample again; repeat until EOS.

And yes — how llama.cpp's session/prompt-cache save-restore works: serialize the KV cache (+ token list) to disk, reload later to skip re-prefilling a known prompt. The one caveat is it's tied to the same model, cache settings, and (with some quant-KV configs) the same backend layout — but conceptually it's just persisting that 2 × n_layers × n_tokens × n_embd_kv tensor blob.
