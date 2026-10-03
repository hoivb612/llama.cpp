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

======================================================================

❯ So if all K/V are being generated in parallel through the all the layers, How are the logits being computed from the last position? And what is the last position given all tokens are generated in parallel?                                       

◐ The user is asking how logits are computed from the last position when all tokens are processed in parallel through the transformer layers, and what "last position" actually means in this context. The key insight is that "parallel" refers to data parallelism across the batch and sequence dimensions—all tokens flow through the same operations simultaneously as distinct columns, with "last position" referring to the column index corresponding to the final token in the sequence, where position is determined by the original sequence order.

  During inference, causal masking ensures each token only attends to positions up to and including itself, so the last position has context from all tokens and produces logits for predicting the next token. While every position technically produces a hidden state and could generate logits, during prefill inference only the last position's logits are needed. To optimize this, llama.cpp extracts just the last column of the hidden state matrix before applying the output projection, computing only a single column of logits rather than the full matrix.

● "Parallel" here means data-parallel over the sequence dimension — all n_tokens are processed by the same ops at the same time, but they remain distinct columns in the [n_embd, n_tokens] matrix. Parallelism doesn't erase order; column j always belongs to sequence position j. So "the last position" is simply column index n_tokens − 1 — the rightmost column, corresponding to the last token of the prompt.

  Two things make this well-defined and correct:

  1. Position is explicitly carried. Order isn't implicit in "when" a token is processed (they're simultaneous) — it's encoded two ways: (a) the column index in the activation matrix, and (b) RoPE, which rotates each token's Q/K by an amount tied to its position. So position 5 is mathematically distinct from position 9 even though both are computed in the same pass.

  2. Causal masking makes every column a valid predictor. Attention uses a causal mask: the token at position i may only attend to positions ≤ i. Therefore:

   - column 0's final hidden state has "seen" only token 0 → it predicts token 1,
   - column j has seen tokens 0..j → it predicts token j+1,
   - column n_tokens − 1 has seen the whole prompt → it predicts the next token.

  So every position actually produces a perfectly good next-token prediction (this is exactly how training works — loss is computed at all positions). During inference prefill you only need the last one, since the earlier predictions are tokens you already have.

  How the logits are computed: the final hidden state is H = [n_embd, n_tokens]. Logits would be output_head · H → [n_vocab, n_tokens], one column of vocab-scores per position. But computing all n_tokens columns is wasteful when you only want the last. So llama.cpp slices first: it does ggml_get_rows(H, [n_tokens-1]) to extract just the last column [n_embd, 1], then applies the output matmul → [n_vocab, 1]. (More generally it keeps only the positions flagged in n_outputs/output_ids; for plain generation that's just the last one.)

  So: all positions run in parallel and all write K/V to the cache, but only the last column is pulled out and pushed through the output head to make the single logits vector you sample from. The "lastness" is a deterministic index into the sequence axis, not something that depends on execution timing.

❯ If the last token (n-tokens-1 index) is used to generate the next token, how do all the other tokens generated in parallel participate in the next token generation since only the last one matters?                                                   

◐ So the key insight is that while only the last token's hidden state generates the next-token logits, all previous tokens still influence the output through the attention mechanism — their key and value representations live in the KV cache, and the final token's query attends over all of them to gather information. This means earlier tokens participate not through their own final hidden states, but through the attention computation that aggregates their cached representations.

● Through attention — specifically, the last token's hidden state is not computed in isolation; at every layer it reads the other tokens' K/V. So the other tokens participate by feeding the last token, not by producing their own logits.

  The crucial distinction: each token produces two different outputs at every layer:

   1. Its own hidden state (the activation that continues down its column). For logits, only the last column's final hidden state is used — the rest are discarded.
   2. Its K/V vectors, written to the KV cache. These are read by other tokens' attention.

  These are separate. Discarding token 5's final hidden state does not discard token 5's contribution — that contribution was already injected into later tokens through attention.

  How it flows into the last token: at each of the 35 layers, the last token's attention does:

   q_last = W_q · h_last                    # query from the last token only
   scores = q_last · [k_0, k_1, …, k_last]  # dot against EVERY token's key (causal: all ≤ last)
   weights = softmax(scores)                # how much to pull from each token
   attn_out = Σ weights[j] · v_j            # weighted blend of ALL tokens' values
   h_last  += attn_out                      # mixed back into the last token's state

  So the last token's hidden state is a weighted summary of all preceding tokens' values. Token 5 influences the result proportionally to how much attention weight the last token (and intermediate tokens, layer by layer) assigns to it.

  And it compounds across depth: at layer 0, the last token gathers from the raw prompt tokens. By layer 1, those tokens' hidden states have themselves already absorbed context from their neighbors (each via its own attention), so when the last token reads them again it's reading increasingly context-rich representations. After 35 layers of this, the last token's hidden state encodes information aggregated from the entire prompt through many indirect paths.

  Analogy: in a meeting, only the last person speaks the conclusion (the logits). But that conclusion reflects everything everyone said, because the last speaker listened to all of them (attention), and they had listened to each other. Silencing the others' final summaries doesn't erase what they contributed — it's already baked into what the last speaker says.

  So: parallel processing produces every token's K/V (the durable contribution) and every token's hidden state (mostly thrown away). The "only the last matters" applies only to the logit readout — the other tokens matter enormously, via attention, in shaping what that last hidden state actually contains.

❯ Let's imagine we have a 3-layer (0, 1 and 2) model and the input string is "I am". How does the flow look like in this     context? I have a hard time visualizing hidden state, the last token from layer 0, and how it flows into layer 1, and then what happens after layer 2. And how does parallelization work and how do the results contribute to the next token.         

◐ I'll create a concrete walkthrough showing how two tokens—"I" and "am"—flow through a 3-layer transformer model, visualizing how each layer processes them in parallel and where attention allows them to interact.

● Great example. Two tokens: "I" at position 0, "am" at position 1. Three layers. Let me make it fully concrete and clear up the biggest misconception first:

   Between layers, BOTH tokens' hidden states flow forward — not just the last one. The "only the last token matters" rule applies only once, at the very end, after layer 2, for reading out logits.

  Let me use a small hidden size of 4 numbers just to make it tangible (Phi-3 uses 3072). I'll write a hidden state as [· · · ·].

  ---------------------------------------------------------------------------------------------------------------------------

  The grid

  Think of it as a 2-column matrix flowing upward through layers. Each column is one token. Time/position goes left→right; layers go bottom→top.

                    pos 0 ("I")        pos 1 ("am")
                    ───────────        ────────────
    after L2  →     h2_I  [· · · ·]    h2_am [· · · ·]   ← only THIS column → logits
                       ▲                  ▲
                    ┌──┴──────────────────┴──┐
                    │       LAYER 2          │   (attention mixes columns, then FFN)
                    └──▲──────────────────▲──┘
    after L1  →     h1_I  [· · · ·]    h1_am [· · · ·]
                       ▲                  ▲
                    ┌──┴──────────────────┴──┐
                    │       LAYER 1          │
                    └──▲──────────────────▲──┘
    after L0  →     h0_I  [· · · ·]    h0_am [· · · ·]
                       ▲                  ▲
                    ┌──┴──────────────────┴──┐
                    │       LAYER 0          │
                    └──▲──────────────────▲──┘
    embeddings →    e_I   [· · · ·]    e_am  [· · · ·]
                       ▲                  ▲
                    get_rows(token_embd, "I")  and  ("am")

  Both columns exist at every level. They travel up together as a [4, 2] matrix.

  ---------------------------------------------------------------------------------------------------------------------------

  Step 0 — Embeddings (the birth of the hidden states)

  Look up each token ID in token_embd.weight (from the GGUF):

   - e_I  = embed("I")  = [0.2, -0.1, 0.5, 0.0]  ← context-free meaning of "I"
   - e_am = embed("am") = [0.1,  0.3, -0.2, 0.4] ← context-free meaning of "am"

  (Numbers invented.) These two vectors are the input hidden states to layer 0. At this point "am" knows nothing about "I".
 The 4 numbers are one row of token_embd.weight, selected by the token's integer ID and dequantized from the GGUF's quant format to f32.

Concretely:

 1. Tokenizer maps the string "I" to an integer ID, say 306.
 2. token_embd.weight is the embedding table stored in the GGUF. Its shape (ggml convention) is [n_embd, n_vocab] — for our toy model [4, n_vocab]; for Phi-3-mini [3072, 32064]. Think of it as one column (an n_embd-long vector) per vocabulary entry.
 3. get_rows(token_embd, 306) pulls out the slice at index 306 → the 4-number vector [0.2, -0.1, 0.5, 0.0]. That's a learned lookup: row 306 is the model's stored representation of token #306 (https://github.com/hoivb612/llama.cpp/issues/306).

 token_embd.weight  (stored in GGUF, quantized)
         n_vocab columns →
       ┌────┬────┬─────┬────┬────┐
  n_   │ .. │ .. │ 0.2 │ .. │ .. │   ← row 0 of hidden dim
  embd │ .. │ .. │-0.1 │ .. │ .. │   ← row 1
  rows │ .. │ .. │ 0.5 │ .. │ .. │   ← row 2
       │ .. │ .. │ 0.0 │ .. │ .. │   ← row 3
       └────┴────┴──┬──┴────┴────┘
                    │
               column 306 = "I"  → e_I = [0.2, -0.1, 0.5, 0.0]

One nuance — quantization. In a Q4_K_M file, token_embd.weight is stored quantized (4-5 bits/weight in block format), not as raw floats. So the bytes on disk aren't literally 0.2, -0.1, …. get_rows dequantizes the selected row back to f32 on the fly, producing those f32 values. So:

 The 4 numbers are derived from the GGUF — they're row "I" of token_embd.weight, read and dequantized to f32. They are learned parameters (fixed at training time, baked into the file), not computed at runtime.

This is the one and only point where GGUF weight data flows directly into the activation stream. Everything after that — the per-layer hidden states — is computed from these embeddings plus the layer weights, and is never stored in the file.

(Footnote: many models, Phi-3 included, tie the input embedding and the output head — the same token_embd.weight matrix is reused at the very end to turn the final hidden state into logits. So this one table does double duty: ID→vector at the bottom, vector→vocab-scores at the top.)

  ---------------------------------------------------------------------------------------------------------------------------

  Step 1 — Layer 0 (both columns processed in parallel)

  A layer = attention sublayer then FFN sublayer, each with a residual add.

 h_I_in  = e_I  = [0.2, -0.1, 0.5, 0.0]     ← from token_embd, dequantized
 h_am_in = e_am = [0.1,  0.3, -0.2, 0.4]

Each token projects Query/Key/Value from its hidden state using the layer's attn_qkv.weight (from GGUF):

 q_I  = Wq · e_I     k_I  = Wk · e_I     v_I  = Wv · e_I
 q_am = Wq · e_am    k_am = Wk · e_am    v_am = Wv · e_am

So e_I and e_am are consumed right here — they're the raw material the Q/K/V matmuls operate on. Then for "am":

 scores  = [ q_am·k_I , q_am·k_am ] / sqrt(d_head)
 w       = softmax(scores)
 attn_am = w[0]·v_I + w[1]·v_am
 h_am'   = e_am + attn_am          ← residual on top of the embedding
 h0_am   = h_am' + FFN(norm(h_am'))

e_am appears twice: once as the source of q_am/k_am/v_am, and once as the residual base (e_am + attn_am). Same for e_I. So the embeddings are very much part of layer 0 — I just folded them into the generic h symbols without spelling it out.

  (a) Attention — the ONLY place columns interact. Causal:

   - Position 0 ("I") may attend to {pos 0} only → it can just look at itself.
   - Position 1 ("am") may attend to {pos 0, pos 1} → it looks at "I" and itself.

  Compute scores for the attention:
  scores = [ q_am·k_I , q_am·k_am ] / sqrt(d_head)
  w      = softmax(scores)

 - q_am, k_I, k_am are produced by the learned Wq/Wk (from the GGUF) applied to the actual hidden states.
 - The two dot products are just real numbers; softmax turns them into a probability distribution that sums to 1.
 - Whatever those dot products happen to be for this specific model and these specific tokens determines the split. It could be [0.5, 0.5], [0.1, 0.9], etc. 
   Let's pick [0.4, 0.6] arbitrarily to show the mechanism (a weighted blend), not because the model produces that.
  
  For "am":

   q_am = Wq · h_am
   scores = [ q_am·k_I ,  q_am·k_am ] / sqrt(d_head)     # two numbers
   w      = softmax(scores) = [0.4, 0.6]                 # e.g. 40% weight on "I", 60% on itself as example ratio from data
   attn_am = 0.4·v_I + 0.6·v_am                          # a blend of both tokens' values
   h_am = h_am + attn_am                                 # residual: "am" now contains some "I" info

  For "I": it only sees itself, so attn_I = v_I, h_I = h_I + v_I.

  ➡ After this, h_am has absorbed information about "I". h_I has not changed who it sees (nothing before it). The K and V computed here for both tokens are written to the KV cache (layer-0 slot).

  (b) FFN — per-token, no mixing:

   h_I  = h_I  + FFN(norm(h_I))     # independent
   h_am = h_am + FFN(norm(h_am))    # independent

  Output of layer 0: h0_I = [· · · · ], h0_am = [· · · ·]. 
  Both become the input columns to layer 1. There is no "the last token from layer 0 flows into layer 1" — both flow.

  ---------------------------------------------------------------------------------------------------------------------------

  Step 2 — Layer 1, then Step 3 — Layer 2

  Identical structure, stacked:

   - Layer 1 attention: "am" again attends to {"I", "am"} — but now it's reading the layer-0-refined versions of those states, which themselves already carry mixed context. So "am" gets a richer summary. K/V written to the layer-1 cache slot.
   - Layer 2: same again. After it, h2_am is a deeply context-aware vector encoding "what comes after 'I am'".

  Each layer nudges the states via residual adds (h = h + update), so the hidden state is a running accumulator that gets progressively more contextual with depth.

  ---------------------------------------------------------------------------------------------------------------------------

  Step 4 — Readout (now, and only now, "last token" matters)

  After layer 2 we have two final hidden states: h2_I and h2_am.

   - h2_I would predict "what comes after 'I'" — we don't care, we already know it's "am". Discarded.
   - h2_am predicts "what comes after 'I am'" — this is what we want.

  So take only the last column:

   h_last = h2_am                       # [4, 1]
   logits = output_head · norm(h_last)  # [n_vocab, 1]
           → e.g. {"happy": 8.1, "tired": 6.3, "going": 5.9, ...}
   sample → "happy"

  llama.cpp slices h2_am out before the output matmul, so it never bothers computing h2_I's logits.

  ---------------------------------------------------------------------------------------------------------------------------

  How "I" still participates (your earlier question)

  Even though h2_I's logits are thrown away, the token "I" shaped the answer through attention: at every layer, "am" pulled in "I"'s K/V. By layer 2, h2_am is partly built from "I". And "I"'s K/V vectors sit in the KV cache, ready for the next step.

  ---------------------------------------------------------------------------------------------------------------------------

  What happens next (decode)

  "happy" was sampled. Now n_tokens = 1:

    feed "happy" (pos 2)  →  e_happy → L0 → L1 → L2 → h2_happy → logits → next token
                                 │       │      │
      at each layer, "happy" attends over the CACHE: {I, am, happy}
      and appends its own K/V → cache grows to 3 entries

  Notice "happy" runs alone (1 column, GEMV), but attends over all 3 cached positions. Repeat until EOS.

  ---------------------------------------------------------------------------------------------------------------------------

  Parallelization, precisely

  "Parallel" = the same op processes both columns at once. In prefill, the qkv/output/ffn matmuls have M = 2 (a GEMM over 2 columns); attention computes both queries' scores together. The hardware does both tokens' arithmetic simultaneously — but they stay separate columns, only ever exchanging information inside attention (and only causally: earlier→later). In decode, M = 1 (a single column, GEMV), which is why generation is memory-bound and slower per token than prefill.

  One-line summary: all tokens flow up through every layer as parallel columns, mixing only via causal attention; the prompt's information is preserved in each later token and in the KV cache; at the top you read out just the last column to predict the next token, then feed that token back in alone.

>>> Note: 
h2_I (the layer-2 output of the "I" column) is indeed not strictly needed, but "I" still has to be carried up through the layers for a different reason — its K/V. Let me separate the two roles of the "I" column.

"I" can't be skipped entirely

To compute h2_am, layer 2's attention for "am" must read the keys and values of "I" at layer 2. Those K/V are projected from layer 2's input for "I", which is h1_I (layer 1's output for "I"). And h1_I needs h0_I, which needs e_I. So the "I" column must be computed up through the layers — otherwise "am" would have nothing to attend to.

 e_I → h0_I → h1_I → (k_I, v_I at L2) ──┐
                                        ├──► attention of "am" at L2 needs these
 e_am→ h0_am→ h1_am→ (q_am at L2) ──────┘

So "I"'s hidden state is needed at every layer to supply that layer's K/V.

But h2_I specifically is redundant

Here's the subtlety you spotted. Within layer 2, the "I" column does two things:

 1. Project its K/V (k_I, v_I) from the layer-2 input h1_I. → needed (am attends to it).
 2. Finish its own update: h2_I = h_I' + FFN(norm(h_I')). → not needed — nothing reads h2_I. There's no layer 3, and the logits only use h2_am.

So the last layer's FFN (and the residual finishing) for non-final tokens is genuinely wasted work. The K/V that "am" needs were already produced from the layer's input, before "I"'s FFN runs.

Why the naive graph computes it anyway

In the standard GGML prefill graph, all positions are processed as one batched matrix (a GEMM with M = n_tokens columns). Every op — including the last layer's FFN — naturally runs over all columns at once. To skip just the "I" column you'd have to slice it out mid-graph, which adds a gather and complicates the fixed graph structure. For a single layer out of 32, pruning that FFN saves only (n_tokens−1)/n_tokens of one layer's FFN cost — a small fraction of total prefill — so it's usually not worth the complexity.

The optimization llama.cpp does do

The big, cheap win isn't pruning the FFN — it's skipping the output head. Turning a hidden state into logits is output_head · h, where output_head is [n_embd, n_vocab] ≈ 3072 × 32064 ≈ 98M params — huge. Doing that for every prompt position would be enormously wasteful. So llama.cpp slices the final hidden state down to just the requested positions (normally only the last) before the output matmul:

 h_final = [h2_I, h2_am]                  # [n_embd, 2]
 h_keep  = get_rows(h_final, [last])      # [n_embd, 1]  ← drop h2_I here
 logits  = output_head · norm(h_keep)     # [n_vocab, 1] computed once, not twice

That's where "only the last token matters" is actually enforced: the redundant h2_I is dropped at the boundary between the transformer stack and the output head, so the expensive vocab projection runs only for the position you care about.

Summary: "I" must flow up all layers because every layer's attention for "am" consumes "I"'s K/V. Only the final layer's FFN output for "I" is truly redundant, and the standard graph computes it anyway because it's batched and the savings are tiny — whereas the genuinely expensive redundancy (the output-head projection for non-final positions) is pruned via a get_rows slice before the logits.