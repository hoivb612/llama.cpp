  Critical: attn_q.weight = [1536, 2048] means head_dim = 2048/8 = 256 for layer 0, not the 512 advertised in hparams.
 
 - Per-layer head_dim: SWA layers = 256, full layers = 512 (4:1 pattern)
 - Per-layer n_ff: most = 6144, last 5 = 12288 ("double-wide MLP")
 - rope_freqs is a single global tensor (not per-layer)
 - n_head_kv = 1 for E2B (extreme GQA)

The key and value dimensions are 512 tokens for standard layers but drop to 256 for the sliding window attention layers, with the window size itself set to 512 tokens, and there's a per-layer pattern that determines which layers use sliding window attention.

The RoPE rotations also differ between layer types — the full attention layers use a frequency base of 1e6 while the sliding window layers use 1e4. Since I can't directly read the pattern array through the public API, I'll infer whether each layer uses sliding window attention by checking its actual head dimension against the known values: if it matches 256, it's a sliding window layer; if it matches 512, it's full attention. This approach avoids needing array metadata access and lets me verify the pattern through the layer counts.

I'm setting up validation rules for each layer—checking that attention tensors have the right dimensions, that the query and key-value heads align properly, that normalization tensors are float32 with the correct sizes, and that the feed-forward and projection layers match their expected shapes. For the sliding window attention flag, I'll try reading from the gguf metadata first, and if that's not available, I'll infer it from the head dimension or default to false. I'm realizing I might have the layer types backwards — the SWA layers could be the ones with the smaller head_dim of 256, while the full-attention layers use 512. The metadata naming is ambiguous, so I should inspect the actual GGUF fields directly to clarify which dimension corresponds to which layer type.

Key new findings exposed by the dump (these were not visible at the G1 hparam level):

 1. head_dim varies per layer: SWA = 256, full = 512 (5:1 pattern). Reading hparam key_length=512 alone would have been wrong for 80% of layers.
 2. n_ff varies per layer (E2B only): 6144 for layers 0-14, 12288 for layers 15-34 ("double-wide MLP"). E4B is flat 10240.
 3. rope.freq_base differs by attention type: 1e6 (full) vs 1e4 (SWA). Bug-trap for the forward.
 4. final_logit_softcap = 30.0 on both — needs the softcap
  * tanh(x/softcap) tail at lm_head.
 5. output is tied to tok_embd on both — no separate lm_head weights.
 6. rope_freqs is a single global (not per-layer despite upstream code structure).

The kernels needed for gemma4:

 1. rmsnorm_mul_f32 (with optional weight)
 2. rope_neox_f32 (with optional freq_factors)
 3. gelu_f32 (gemma4 uses GELU, not SiLU as phi3)
 4. dequant_row_to_f32 (read row from K-quant)
 5. qk_norm_per_head_f32 (apply [head_dim] norm to each of [n_head, head_dim] heads)

   - rmsnorm_mul_f32 needs to support w == nullptr (V in attention uses unweighted rms_norm)
   - New gelu_f32 (gemma4 uses GELU, not SiLU)
   - Otherwise mirrors phi3 patterns

┌────────────────────────────────────┬───────────────────────────────────────────────────────┐
│ Kernel                             │ Notes                                                 │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ rmsnorm + weight                   │ Standard                                              │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ rmsnorm (no weight, V path)        │ New: gemma4-specific, V uses unweighted rms_norm      │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ rmsnorm per-head                   │ Q/K norm pattern (broadcast [head_dim] across n_head) │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ gelu (tanh approx)                 │ gemma4 uses GELU (vs phi3 SiLU)                       │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ rope_neox SWA (head_dim=256)       │ freq_base=1e4                                         │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ rope_neox full (factors, hd=512)   │ freq_base=1e6 + freq_factors                          │
├────────────────────────────────────┼───────────────────────────────────────────────────────┤
│ dequant_row Q4_K                   │ Token embed reader                                    │
└────────────────────────────────────┴───────────────────────────────────────────────────────┘

 1. attn_ctx dim is n_head*head_dim not n_embd — for E2B SWA: 2048 vs
  1536. wo projects 2048→1536. Hand path already had this right; oracle crashed on ggml_reshape_2d.
 2. rope_dim per layer must be the layer's own head_dim, not the global hparam (which reflects full-attn 512). SWA layers crashed ggml_rope_ext with n_dims > ne[0].
 3. Added rope_freq_base_swa (1e4) alongside rope_freq_base (1e6); baked into LayerF32.rope_base per-layer so forward never brancheson is_swa for RoPE.

 Dumping weights:
 bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-dump-weights

cd D:\llama.cpp\b612_052026; python -c "import gguf; r = gguf.GGUFReader(r'D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf'); [print(f.name, '=', str(f.parts[f.data[0]])[:100] if f.data else '?') for f in r.fields.values() if 'shared' in f.name.lower() or 'sliding' in f.name.lower() or 'kv' in f.name.lower()]"

@Read gguf metadata for shared kv layers
$py = 'D:\llama.cpp\b612_Onnx\onnxruntime_EP\plugin-ep-vulkan\.venv-test\Scripts\python.exe'; & $py -c @"                       │ │
import gguf                                                                                                                     │ │
r = gguf.GGUFReader(r'D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf')                                                  │ │
for f in r.fields.values():                                                                                                     │ │
    n = f.name.lower()                                                                                                          │ │
    if 'shared' in n or 'sliding' in n or 'kv' in n or 'layer' in n:                                                            │ │
        try:                                                                                                                    │ │
            if len(f.data) == 1:                                                                                                │ │
                print(f.name, '=', f.parts[f.data[0]].tolist())                                                                 │ │
            else:                                                                                                               │ │
                print(f.name, '= (array, len=' + str(len(f.data)) + ')', [f.parts[i].tolist() for i in f.data[:5]], '...')      │ │
        except Exception as e:                                                                                                  │ │
            print(f.name, '=ERR', e)                                                                                            │ │
"@

◐ For Gemma-4, I'm understanding how KV cache reuse works in the later layers. When a layer index is at or beyond the starting point for KV reuse, it pulls cached key-value pairs from an earlier layer—either two layers back if it's a sliding window attention layer, or one layer back if it's full attention. This pattern applies to models like E2B with 35 layers when KV sharing is enabled, though the thought cuts off before explaining the full implications.

◐ Found it—E2B has `shared_kv_layers = 20`, which means the first 15 layers (35 - 20) maintain their own K/V cache while layers 15-34 reuse K/V from earlier layers according to the reuse logic in llama-model.cpp.

  For layers 15-34, they should compute Q normally but pull K and V from layer 13 (if SWA) or layer 14 (if FULL) instead of computing their own. My current implementation incorrectly computes fresh K/V for every layer, so I need to add a `kv_reuse_il` field to track which earlier layer to pull from, then store and reuse the K/V tensors during the forward pass. er il is FULL → reuse layer 14's K/V. So the rule is checking whether the *reusing* layer is SWA or FULL, not the source layer. That means FULL layers in 15-34 reuse layer 14 (which is SWA), which seems odd but might be intentional—perhaps FULL layers can reuse SWA K/V caches.

  Let me reconsider the actual layer pattern. The array shows `[True, True, True, True, False]` repeating, so it's "4 SWA + 1 FULL" not "5 SWA + 1 FULL". That changes where the FULL layers actually are, so I need to recount which layers are FULL and verify the reuse logic against the correct pattern.

@Print full sliding window pattern
│ │ $py = 'D:\llama.cpp\b612_Onnx\onnxruntime_EP\plugin-ep-vulkan\.venv-test\Scripts\python.exe'; & $py -c @"                       │ │
│ │ import gguf                                                                                                                     │ │
│ │ r = gguf.GGUFReader(r'D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf')                                                  │ │
│ │ for f in r.fields.values():                                                                                                     │ │
│ │     if 'sliding_window_pattern' in f.name:                                                                                      │ │
│ │         vals = [bool(f.parts[i].tolist()[0]) for i in f.data]                                                                   │ │
│ │         print('n=', len(vals))                                                                                                  │ │
│ │         for i,v in enumerate(vals):                                                                                             │ │
│ │             print(' layer', i, 'swa=' if True else '', v)                                                                       │ │
│ │ "@

 Gemma-4 E2B has shared_kv_layers = 20, meaning layers 0-14 own KV; layers 15-34 REUSE K/V from earlier layers (SWA→13, FULL→14). Trying to compute fresh K/V for every layer is wrong for layers 15-34.

@ Check which layers have attn_k/v tensors
│ │ $py = 'D:\llama.cpp\b612_Onnx\onnxruntime_EP\plugin-ep-vulkan\.venv-test\Scripts\python.exe'; & $py -c @"                       │ │
│ │ import gguf                                                                                                                     │ │
│ │ r = gguf.GGUFReader(r'D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf')                                                  │ │
│ │ names = sorted(t.name for t in r.tensors)                                                                                       │ │
│ │ for n in names:                                                                                                                 │ │
│ │     if '.attn_k.' in n or '.attn_v.' in n:                                                                                      │ │
│ │         if '.0.' in n or '.4.' in n or '.13.' in n or '.14.' in n or '.15.' in n or '.20.' in n or '.34.' in n:                 │ │
│ │             print(n)                                                                                                            │ │
│ │ "@               

@ Count tensors per layer
│ │ $py = 'D:\llama.cpp\b612_Onnx\onnxruntime_EP\plugin-ep-vulkan\.venv-test\Scripts\python.exe'; & $py -c @"                       │ │
│ │ import gguf                                                                                                                     │ │
│ │ r = gguf.GGUFReader(r'D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf')                                                  │ │
│ │ names = sorted(t.name for t in r.tensors)                                                                                       │ │
│ │ # count per-layer tensor variety                                                                                                │ │
│ │ import collections                                                                                                              │ │
│ │ prefix = collections.Counter()                                                                                                  │ │
│ │ for n in names:                                                                                                                 │ │
│ │     if n.startswith('blk.'):                                                                                                    │ │
│ │         parts = n.split('.')                                                                                                    │ │
│ │         il = int(parts[1])                                                                                                      │ │
│ │         prefix[il] += 1                                                                                                         │ │
│ │ for il in sorted(prefix):                                                                                                       │ │
│ │     print(il, prefix[il])                                                                                                       │ │
│ │ "@

◐ All the tensors are present in each layer, but the upstream graph doesn't actually use the key/value weights for shared-KV layers. My approach is to set the `kv_reuse_il` flag in the dequant layer based on the LayerWeights configuration, then modify the layer forward pass to skip computing K and V from their respective weights when this flag is set, using externally-provided values instead.

  The cleanest way to handle this is to add optional output parameters for the computed K and V tensors, and optional input parameters for externally-provided K and V. When a layer is reusing KV from an earlier layer, it reads from the external inputs; otherwise it computes and writes to the output buffers, which then become the external inputs for subsequent layers that need them.

@ Kernel self-test
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-kernel-test

@ Run network test with KV reuse
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-network-test "The capital of France is" -ngl 0 --threads-prefill 8

@ Run single-layer self-test on full-attn layer 4 
\bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-layer-test 4 --gemma4-layer-test-ntok

@ Run E2B network test with another prompt
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-network-test "1, 2, 3, 4," -ngl 0 --threads-prefill 8 

● Committed as ec6e8a0b3.

  Summary

  Root cause: Gemma-4 dense uses a shared-KV pattern. For E2B (n_layer=35), gemma4.attention.shared_kv_layers=20 means layers 0–14 ownKV; layers 15–34 reuse K/V from earlier layers (SWA→13, FULL→14). My hand-path was recomputing fresh K/V from wk/wv for every layer,which silently consumed unused weights and produced wrong attention.

  Fix:

   - gemma4_weights reads shared_kv_layers; sets n_layer_kv_from_start and per-layer kv_reuse_il (with is_swa-match sanity check).
   - layer_forward_f32 gains 4 optional pointer args for KV in/out; reused layers skip wk/wv projection.
   - network_forward_f32 manages per-owning-layer K/V storage; threads pointers based on kv_reuse_il.
   - dequant_layer skips wk/wv for reuse layers (memory saver).

  Results (E2B, "The capital of France is"):

   - Before: top-1=" the", cos_sim=0.94, upstream top-1 rank 10202
   - After: top-1=" Paris" ✓ MATCH, cos_sim=0.9995, top-10 overlap 9/10

  E4B same prompt: top-1 MATCH, top-10=10/10, cos_sim=0.9998. All prior self-tests still PASS.

  ==================================================
◐ I'm shifting focus to design the greedy decode loop with KV cache support. The key challenge is moving from stateless forward passes to persistent K/V buffers that accumulate across decode steps, where each layer maintains its own storage for all previously computed key-value pairs.

  During prefill, I compute K/V for all prompt tokens and store them; then in decode steps, I process one token at a time, appending its K/V to the persistent buffer and attending over all cached tokens with a causal mask. For SWA layers, I'll restrict attention to only the last 512 tokens by filtering which cached positions contribute to each query. Since layers 15-34 reuse K/V from layers 13-14, those layers just read from the shared buffer while the owning layers handle the appends.

Now I'm thinking through the storage layout and whether to pre-allocate buffers or grow them dynamically during prefill. For SWA layers, I need to enforce both causal masking and the sliding window constraint, so positions outside the window get skipped. During prefill with 6 tokens, each layer computes and stores its own K/V, while shared KV layers reuse storage from their paired layers. For decode steps, new tokens append to existing K/V storage, and attention masks based on the SWA window size to only attend to recent positions. Now I'm ready to implement this.

I'm setting up a `NetworkState` struct to hold the K/V caches per layer along with tracking how many positions have been processed and the maximum sequence length. The main `network_step` function will handle both prefill and decode by appending new K/V entries to the caches and applying the appropriate attention masks. I'm also planning a CLI command to run token generation with this network while comparing outputs against the upstream model in parallel, printing token-by-token matches.

Refactoring layer forward... Refining the API design...

I'm simplifying this by always passing pre-sized K/V cache buffers for the full sequence length (n_past + n_new), having the function write new entries at offset n_past, then using the entire buffer for attention. The reuse_kv flag controls whether the function writes the new K/V or assumes they're already populated. This unifies prefill and decode into a single code path where prefill has n_past=0 and decode has n_past equal to the prompt length. Writing the layer forward function... I'm deciding whether to keep the original layer_forward_f32 function unchanged for backward compatibility with existing tests, or refactor everything into a single cached version. The cleaner approach is to add a new layer_forward_f32_cached function that handles the KV cache logic while leaving the original function intact for the single-layer self-test, then eventually phase out the legacy network_forward_f32 once the new cached network_step is in place. Writing the wrapper function...

The wrapper handles three cases for K and V buffers: reusing externally-provided ones, writing to caller-supplied storage, or allocating local scratch space. Then it delegates to the cached implementation with prefill mode (n_past=0) and a very large SWA window to effectively disable the mask. For a 6-token prompt with n_swa=512, the sliding window never activates anyway, so this approach is sound.

Now I need to implement the cached function itself, add the NetworkState struct, and create the network_step entry point for G3.4b.

 I'll create a unified cached layer forward function, then refactor the existing layer forward to be a simple wrapper that allocates its own scratch space, and update the network forward to use the cached version with n_past initialized to zero.

Defining the cached layer forward function... Setting up the cache logic to handle reuse cases, local allocation, or external buffers, then calling the cached version with n_past set to 0 and SWA disabled to preserve the original behavior. I'm reconsidering the approach to keep layer_forward_f32 simpler. pos_all[n_past + t], and for sliding window attention, the distance between query and key positions must be less than n_swa. In the typical case where pos_all is just [0, 1, 2, ...], a new token at position n_past + t attends to all prior keys within the window.

Key design:

 1. Refactor layer_forward_f32 to delegate to a new layer_forward_f32_cached(n_past, n_swa, K_cache, V_cache, reuse_kv) — singlesource of truth.
 2. Add NetworkState { K_cache[per_layer], V_cache[per_layer], n_past, pos_all }.
 3. Add network_step(state, m, n_new, token_ids, ...) — handles both prefill and 1-token decode.
 4. Add network_generate_and_compare(m, prompt, n_gen) and CLI --gemma4-network-gen.

@ Run layer self-test to verify refactor
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-layer-test 4 -ngl 0 --threads-prefill 8

@ Run greedy decode self-test (8 tokens)
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-network-gen "The capital of France is" 8 -ngl 0 --threads-prefill 8
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-network-gen "1, 2, 3, 4," 16 -ngl 0 --threads-prefill 8

│ Commit G3.4b                                                                                                                        │
│ ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ cd D:\llama.cpp\b612_052026; $msg = @"                                                                                          │ │
│ │ gemma4 G3.4b: greedy decode with persistent KV cache                                                                            │ │
│ │                                                                                                                                 │ │
│ │ Refactor + extend the hand-coded F32 forward to support multi-step                                                              │ │
│ │ decode with a persistent per-layer K/V cache, mirroring the staged                                                              │ │
│ │ Phi-3 approach.                                                                                                                 │ │
│ │                                                                                                                                 │ │
│ │ * layer_forward_f32 split:                                                                                                      │ │
│ │   - layer_forward_f32_cached(L, n_new, n_past, n_swa, ..., K_cache,                                                             │ │
│ │     V_cache, reuse_kv): single source of truth. Writes new K/V at                                                               │ │
│ │     offset n_past*n_kv into the caller-supplied buffer. Applies SWA                                                             │ │
│ │     mask when L.is_swa (pos_t - pos_k >= n_swa => -INF). Pass                                                                   │ │
│ │     n_swa=INT32_MAX to disable.                                                                                                 │ │
│ │   - layer_forward_f32 reduced to a thin wrapper: allocates local                                                                │ │
│ │     K/V scratch (or uses external buffers via the 4 optional                                                                    │ │
│ │     pointer args) and calls the cached path with n_past=0,                                                                      │ │
│ │     n_swa=INT32_MAX. Preserves G3.3 / G3.4a behaviour byte-for-byte.                                                            │ │
│ │                                                                                                                                 │ │
│ │ * New NetworkState struct: per-layer K_cache / V_cache vectors                                                                  │ │
│ │   (only owning layers; shared-KV layers read from their source),                                                                │ │
│ │   pos_all vector, n_past counter, cap_seq.                                                                                      │ │
│ │                                                                                                                                 │ │
│ │ * network_state_reserve allocates K_cache / V_cache up front for                                                                │ │
│ │   cap_seq tokens so the inner loop never reallocates.                                                                           │ │
│ │                                                                                                                                 │ │
│ │ * network_step performs one forward pass over n_new new tokens at                                                               │ │
│ │   positions [n_past, n_past+n_new), produces logits for either the                                                              │ │
│ │   last new token or all new tokens, and advances s.n_past. Uniform                                                              │ │
│ │   for prefill (n_new = n_prompt) and decode (n_new = 1).                                                                        │ │
│ │                                                                                                                                 │ │
│ │ * network_gen_self_test runs greedy decode on both the hand path                                                                │ │
│ │   (network_step prefill + n_gen steps of size 1) and the upstream                                                               │ │
│ │   path (persistent llama_context + llama_decode + argmax), then                                                                 │ │
│ │   compares token sequences position-by-position. PASS iff all                                                                   │ │
│ │   n_gen tokens match.                                                                                                           │ │
│ │                                                                                                                                 │ │
│ │ * CLI: --gemma4-network-gen [PROMPT] [N] (default PROMPT=                                                                       │ │
│ │   "The capital of France is", N=32).                                                                                            │ │
│ │                                                                                                                                 │ │
│ │ * dequant_row + compute_per_layer_inputs lifted from anonymous to                                                               │ │
│ │   file-scope static so both network_forward_f32 and network_step                                                                │ │
│ │   can reuse them.                                                                                                               │ │
│ │                                                                                                                                 │ │
│ │ Validation (all PASS):                                                                                                          │ │
│ │   E2B "The capital of France is" n_gen=8  -> 8/8 match                                                                          │ │
│ │   E2B "1, 2, 3, 4,"              n_gen=16 -> 16/16 match                                                                        │ │
│ │   E4B "1, 2, 3, 4,"              n_gen=16 -> 16/16 match                                                                        │ │
│ │                                                                                                                                 │ │
│ │ All prior self-tests still PASS:                                                                                                │ │
│ │   --gemma4-kernel-test                                                                                                          │ │
│ │   --gemma4-layer-test 4                                                                                                         │ │
│ │   --gemma4-network-test "The capital of France is"                                                                              │ │
│ │     (E2B top-1=" Paris" MATCH, cos_sim=0.9995)                                                                                  │ │
│ │                                                                                                                                 │ │
│ │ Hand path is ~2 s/tok (E2B) / ~4.5 s/tok (E4B) - correctness scaffold                                                           │ │
│ │ only; G4 will bring in qquant + cached prefill from A5.x for speed.     

Summary:

 - Refactored layer_forward_f32 into layer_forward_f32_cached (new source of truth with n_past, n_swa, K/V cache writes at offset) +thin wrapper preserving G3.3/G3.4a behaviour byte-for-byte.
 - New API: NetworkState, network_state_reserve, network_step (uniform prefill/decode), network_gen_self_test.
 - CLI: --gemma4-network-gen [PROMPT] [N].

Validation:

┌───────┬────────────────────────────┬────┬─────────────┐
│ Model │ Prompt                     │ N  │ Result      │
├───────┼────────────────────────────┼────┼─────────────┤
│ E2B   │ "The capital of France is" │ 8  │ 8/8 match   │
├───────┼────────────────────────────┼────┼─────────────┤
│ E2B   │ "1, 2, 3, 4,"              │ 16 │ 16/16 match │
├───────┼────────────────────────────┼────┼─────────────┤
│ E4B   │ "1, 2, 3, 4,"              │ 16 │ 16/16 match │
└───────┴────────────────────────────┴────┴─────────────┘

All prior self-tests (kernel, layer, network) still PASS.

Hand path is ~2 s/tok (E2B), correctness-only scaffold. Ready for G3.5 multi-prompt parity gate, then G4 brings qquant + cachedprefill from A5.x for speed.

@ Profile prefill + 4 decode
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-network-profile "The capital of France is" 4 -ngl 0 --threads-prefill 8

◐ Looking at the performance profile, the decode step is taking about 1485 milliseconds per token, with the computational bottleneck spread across the feed-forward network layers—the down, gate, and up projections each consuming roughly a fifth of the total time, while the language model head contributes the remainder.

  The real insight here is that nearly all of the decode latency comes from matrix multiplications: the four FFN operations account for about 63% of the time, the output projection adds another 23%, and the Q/wo projections bring it to 97% combined. The attention mechanism itself is negligible at just 0.06% because the KV cache is doing its job effectively. When I look at the lm_head timing more carefully—accounting for the fact that the decode profile is summed across multiple steps—the per-step cost aligns reasonably well between prefill and decode phases, both around 350 milliseconds for a single token output.

  The bottleneck is clear: these matmuls are dequantizing Q4_K weights to F32 before doing the multiplication, which is expensive. The A5.x quantized matmul patterns could apply directly here—skipping the dequantization step and using specialized Q4_K × F32 operations instead. The lm_head is particularly interesting because it scans all 262,144 vocabulary rows with dequantization overhead, but could be restructured as a proper matrix multiply of [vocab, embedding] × [embedding, 1]. Prefill shows the same matmul dominance, just scaled linearly with the token count, so the per-token costs are comparable even though the absolute numbers are higher.

  For G4, I'm thinking a staged approach: first, a quantized matmul shim that replaces the dequant-then-matmul pattern by keeping the original quantized tensors and doing row-by-row dequantization inline during the multiply. Then, a second stage that leverages ggml's native mul_mat path for Q4_K to get the real speedup.

  The key insight is that for repeated decode, we're already caching the F32 weights, so we could skip dequantization entirely and just use the quantized weights with ggml's mul_mat directly. The lm_head optimization comes along for free since it uses the same Q4_K weights. Before moving forward, I need to verify that the profiling scope hasn't broken the layer self-test.

│ Commit G3.x consolidation + profiling                                                                                               │
│ ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ cd D:\llama.cpp\b612_052026; $msg = @"                                                                                          │ │
│ │ gemma4 G3.x consolidation + profiling instrumentation                                                                           │ │
│ │                                                                                                                                 │ │
│ │ Consolidation (zero behaviour change; all self-tests still PASS):                                                               │ │
│ │ * network_forward_f32 now delegates to network_step (reserve scratch                                                            │ │
│ │   NetworkState + single step). This removes ~140 lines of duplicate                                                             │ │
│ │   layer-loop / K-V management code and lifts the obsolete SWA guard                                                             │ │
│ │   (n_tokens > n_swa - SWA mask is now actually applied per layer).                                                              │ │
│ │ * dequant_row + compute_per_layer_inputs hoisted to file-scope static                                                           │ │
│ │   (no longer in anon namespace) so both network_forward_f32 and                                                                 │ │
│ │   network_step share the canonical implementation.                                                                              │ │
│ │ * CLI help text updated to list all gemma4-* self-test flags.                                                                   │ │
│ │                                                                                                                                 │ │
│ │ Profiling infrastructure (opt-in; zero-overhead when disabled):                                                                 │ │
│ │ * prof::Scope RAII helper accumulates wall-clock ns into named                                                                  │ │
│ │   counters guarded by a single bool gate.                                                                                       │ │
│ │ * layer_forward_f32_cached instrumented at all 17 internal stages                                                               │ │
│ │   (attn_norm, Q/K/V proj, q/k/v rms-norm, RoPE, attention, wo,                                                                  │ │
│ │   post_attn+res, ffn_norm, gate, up, gelu*up, ffn_down,                                                                         │ │
│ │   post_ffw+res, PLE merge, out_scale).                                                                                          │ │
│ │ * network_step instrumented at top level (embed, ple, layer_loop,                                                               │ │
│ │   out_norm, lm_head, softcap).                                                                                                  │ │
│ │ * New helpers: profile_set_enabled / profile_reset / profile_print.                                                             │ │
│ │ * network_profile driver: runs 1 prefill + N decode steps with                                                                  │ │
│ │   profiling on, prints two separate breakdowns.                                                                                 │ │
│ │ * CLI flag: --gemma4-network-profile [PROMPT] [N_DECODE]                                                                        │ │
│ │                                                                                                                                 │ │
│ │ Profile findings (E2B Q4_K_M decode, per token):                                                                                │ │
│ │   ffn_gate  : 307 ms  (21%)                                                                                                     │ │
│ │   ffn_up    : 306 ms  (21%)                                                                                                     │ │
│ │   ffn_down  : 327 ms  (22%)                                                                                                     │ │
│ │   lm_head   : 349 ms  (23%)                                                                                                     │ │
│ │   Q proj    :  79 ms  (5%)                                                                                                      │ │
│ │   wo        :  80 ms  (5%)                                                                                                      │ │
│ │   attention :   1 ms  (0.06%)  <- KV cache paying off                                                                           │ │
│ │   -----------------------                                                                                                       │ │
│ │   TOTAL     ~1485 ms / decode token                                                                                             │ │
│ │ * 97% of decode time is in matmul_f32 (dequant-then-F32-multiply).                                                              │ │
│ │ * Single biggest opportunity: replace matmul_f32 with a quant matmul                                                            │ │
│ │   path (Q4_K x F32 directly via ggml mul_mat) - similar to A5.x port.  

=====================================================================

Proposed G4 stages

G4.1 — qquant matmul shim (lowest risk, big win)

 - Add matmul_quant(W_q, x_f32, y_f32) using ggml's mul_mat graph (1-op graph: build context, ggml_mul_mat, ggml_graph_compute). Keep weights as the original quantized ggml_tensor
  * in LayerF32 (we already store these pointers in Weights).
 - Replace all 7 matmul_f32 calls in layer_forward_f32_cached (Q, K, V, wo, gate, up, ffn_down) + lm_head loop with matmul_quant.
 - Drop the F32 dequant of these tensors in dequant_layer (huge memory win too).
 - Validate parity via existing --gemma4-network-test (cos_sim should improve slightly since we avoid double-rounding) and--gemma4-network-gen (must still hit 64/64).
 - Re-run profile; expected decode time ~250–500 ms/tok.

G4.2 — multithread the matmul (if G4.1 isn't enough)

 - ggml's CPU backend already multithreads mul_mat via ggml_threadpool / ggml_set_n_threads. Make sure the threadpool is set up inthe test driver and we get parallel speedup. May land for free in G4.1.

G4.3 — batched prefill matmul (A4.1 pattern, low priority)

 - Profile shows prefill is ~same per-token cost as decode. With qquant matmul, prefill matmuls naturally amortize weight accessacross n_tokens (same kernel — the matmul is over [n_embd] × [n_embd, n_tokens]). Should "just work" via G4.1.

G4.4 — cached prefill (A5.5 pattern; conditional)

 - Only worth doing if the user wants repeated prompts with shared prefix (chat history). Not on the critical path for first-passspeed.

G4.5 — lm_head sparse / chunked (very conditional)

 - 262 144 × 1536 Q4_K × F32 = big. After G4.1 this may already be fast enough. If not, can compute only the top-K rows viaprior-step argmax candidate set (sampling already throws away most rows).

@ Layer test il=15 SWA double-wide MLP 
bin\Release\Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf --gemma4-layer-test 15 -ngl 0 --threads-prefill 8


Results on Gemma-4-E2B-it Q4_K_M, 8t UMA:

┌────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│ Test                                   │ Result                                                                                 │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ kernel self-test                       │ PASS (no change)                                                                       │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ layer self-test il=0/4 (F32 fallback)  │ PASS, max_abs 7.9e-4 / 1.0e-3 (matches baseline)                                       │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ network-test "The capital of France    │ top-1=" Paris" MATCH, cos_sim=0.9992 (vs F32 0.9995), prefill 134ms (was ~2s+)         │
│ is"                                    │                                                                                        │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ network-gen 8 tok                      │ 8/8 match byte-for-byte, hand 666ms (was ~14 s)                                        │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ network-gen 16 tok                     │ 16/16 match, hand 1265ms (was ~30 s)                                                   │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ E4B network-gen 16 tok                 │ 16/16 match                                                                            │
└────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

Profile shift — per decode token (E2B Q4_K_M, 8 threads):

┌───────────────┬───────────────────┬──────────────────┬──────────┐
│ Stage         │ F32 baseline (1t) │ qquant G4.1 (8t) │ Speedup  │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ ffn_gate      │ 307 ms            │ 39.9 ms          │ 7.7×     │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ ffn_up        │ 306 ms            │ 39.7 ms          │ 7.7×     │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ ffn_down      │ 327 ms            │ 49.0 ms          │ 6.7×     │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ lm_head       │ 349 ms            │ 30.0 ms          │ 11.6×    │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ Q proj        │ 79 ms             │ 10.9 ms          │ 7.2×     │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ wo            │ 80 ms             │ 11.0 ms          │ 7.3×     │
├───────────────┼───────────────────┼──────────────────┼──────────┤
│ TOTAL/tok     │ ~1485 ms          │ 63.9 ms          │ ~23×     │
└───────────────┴───────────────────┴──────────────────┴──────────┘

Changes (4 files, 172 ins / 42 del + new gemma4_matmul.{h,cpp}):

 - gemma4_matmul.h/.cpp — new MatmulCtx + matmul_qf32 shim (persistent 32 MiB arena,ggml_init+ggml_mul_mat+ggml_graph_compute_with_ctx per call, multithreaded via mm.n_threads).
 - gemma4_forward.h — added 9 ggml_tensor
  * shadow fields to LayerF32; added mutable MatmulCtx mm to ModelF32; new dequant_to_f32 param on dequant_layer.
 - gemma4_forward.cpp — dispatch_matmul() helper routes through matmul_qf32 (qquant path) or matmul_f32 (F32 fallback) based onoptional MatmulCtx* arg on layer_forward_f32_cached; dequant_model skips F32 dequant of large matmul weights (~5.6 GB saved on E2B)and initializes mm; lm_head row-loop replaced by a single matmul_qf32(m.mm, m.tok_embd_quant, ...) call; network_self_test /network_gen_self_test / network_profile now wire n_threads into mm.n_threads.

Notes:

 - F32 layer-self-test paths remain bit-equivalent (use F32 fallback when mm == nullptr).
 - compute_per_layer_inputs still uses F32 matmul_f32 (per_layer_model_proj is F32 in source); could be moved to qquant in a laterpass (ple bucket is ~14% of decode).
 - Memory: ~5.6 GB saved on E2B since FFN/wo/wq/etc. no longer dequant to F32.

commit 2711e28d3be39f803d6f4cc4edb7782f8409c749 (HEAD -> hv/b612_052026)
Date:   Fri Jun 5 11:46:26 2026 -0700

 gemma4 G4.1: qquant matmul shim (matmul_qf32) + lm_head + FFN/QKV in ggml_mul_mat

 * New gemma4_matmul.{h,cpp}: MatmulCtx (persistent ~32 MiB arena +
   n_threads) and matmul_qf32(W_tensor, x_f32, y_f32, n_in, n_out, n_cols)
   -- builds a 1-op cgraph (ggml_mul_mat) and runs it on the CPU backend
   with multi-threading. Supports any W type ggml_mul_mat accepts
   (Q4_K, Q5_K, F32, ...). Cross-context: W is from the model loader,
   x and y live in the MatmulCtx arena.

 * LayerF32 gains *_t ggml_tensor pointer shadows (wq_t, wk_t, wv_t,
   wo_t, ffn_gate_t, ffn_up_t, ffn_down_t, inp_gate_t, proj_t) -- always
   populated. The F32 std::vector buffers stay valid for the F32
   fallback path used by layer_self_test.

 * dequant_layer gains `dequant_to_f32` bool (default true). When false
   (used by dequant_model) the large matmul weights are NOT dequanted,
   saving ~5.6 GB on E2B Q4_K_M.

 * ModelF32 gains `mutable MatmulCtx mm` -- initialised in dequant_model.
   Test drivers (network_self_test, network_gen_self_test,
   network_profile) wire n_threads in after dequant_model.

 * layer_forward_f32_cached gains optional `MatmulCtx * mm = nullptr`.
   dispatch_matmul() routes each per-layer matmul through matmul_qf32
   (when mm != nullptr) or matmul_f32 (when mm == nullptr, F32 path
   for layer_self_test parity).

 * network_step lm_head loop replaced by a single matmul_qf32 on
   m.tok_embd_quant -- one ggml_mul_mat over [n_embd, n_vocab].

 Validation (Gemma-4-E2B-it Q4_K_M, 8t UMA):
   --gemma4-kernel-test                : PASS
   --gemma4-layer-test 0 / 4           : PASS (F32 fallback, max_abs ~1e-3)
   --gemma4-network-test "...France..." : top-1 " Paris" MATCH,
                                          cos_sim 0.9992 (was F32 0.9995)
                                          prefill 134 ms (was ~2 s)
   --gemma4-network-gen "...France..." 8 : 8/8 match
   --gemma4-network-gen "1,2,3,4," 16    : 16/16 match (E2B and E4B)

 Profile (per decode token, E2B Q4_K_M, 8t):
   TOTAL: 1485 ms -> 63.9 ms (~23x)
   matmul stages: 6-12x each
   attention loop unchanged (~3 ms, KV cache still paying off)

 Commit G4.2 threadpool                                                                                                              │
│ ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── │
│ ╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ cd D:\llama.cpp\b612_052026; git add examples/gemma4/gemma4_forward.cpp examples/gemma4/gemma4_forward.h                        │ │
│ │ examples/gemma4/gemma4_matmul.cpp examples/gemma4/gemma4_matmul.h 2>&1; $msg = @"                                               │ │
│ │ gemma4 G4.2: persistent ggml_threadpool in MatmulCtx (~21% decode speedup)                                                      │ │
│ │                                                                                                                                 │ │
│ │ Per-call ggml_graph_compute_with_ctx spun up + tore down a fresh worker                                                         │ │
│ │ team for every matmul. With 7+ matmuls per layer x 35 layers + lm_head                                                          │ │
│ │ that was ~250+ thread spawn/join events per decode token, each costing                                                          │ │
│ │ ~30-100 us on Windows. Persistent pool eliminates that overhead.                                                                │ │
│ │                                                                                                                                 │ │
│ │ Changes:                                                                                                                        │ │
│ │ * MatmulCtx gains std::unique_ptr<ggml_threadpool, GgmlThreadpoolDeleter>                                                       │ │
│ │   pool (RAII-cleaned) and a shared std::vector<uint8_t> work_buf for                                                            │ │
│ │   ggml_cplan.work_data (sized lazily to max work_size seen).                                                                    │ │
│ │ * matmul_ctx_init builds the threadpool with                                                                                    │ │
│ │   ggml_threadpool_params_default(n_threads) when n_threads > 1; falls                                                           │ │
│ │   back to the original single-thread compute_with_ctx path when 1.                                                              │ │
│ │ * matmul_qf32 now calls ggml_graph_plan(gf, n_threads, pool) +                                                                  │ │
│ │   ggml_graph_compute(gf, &cplan) on the multi-thread path. The arena                                                            │ │
│ │   is still ggml_init-reset per call (same as G4.1); only the worker                                                             │ │
│ │   threads are shared.                                                                                                           │ │
│ │ * dequant_model gained an n_threads parameter so the pool is sized at                                                           │ │
│ │   creation time (avoids leak + resize races). Test drivers                                                                      │ │
│ │   (network_self_test, network_gen_self_test, network_profile) thread                                                            │ │
│ │   their n_threads through to dequant_model and drop the post-init                                                               │ │
│ │   ``mf.mm.n_threads = n_threads;`` workaround.                                                                                  │ │
│ │                                                                                                                                 │ │
│ │ Validation (Gemma-4-E2B-it Q4_K_M, 8t, ngl=0):                                                                                  │ │
│ │   --gemma4-kernel-test                : PASS                                                                                    │ │
│ │   --gemma4-layer-test 4               : PASS  max_abs 1.0e-3 (F32 fallback)                                                     │ │
│ │   --gemma4-network-test ``...France'' : top-1 '' Paris'' MATCH,                                                                 │ │
│ │                                         cos_sim 0.9992 (unchanged)                                                              │ │
│ │   --gemma4-network-gen ``1,2,3,4,'' 16: 16/16 (E2B 1185 ms, E4B 2114 ms)                                                        │ │
│ │                                                                                                                                 │ │
│ │ Profile (per decode token, E2B Q4_K_M, 8t):                                                                                     │ │
│ │                      G4.1 (no pool) -> G4.2 (pool)                                                                              │ │
│ │   ffn_gate           39.9 ms        ->  25.9 ms (-35%)                                                                          │ │
│ │   ffn_up             39.7 ms        ->  25.8 ms (-35%)                                                                          │ │
│ │   ffn_down           49.0 ms        ->  41.4 ms (-16%)                                                                          │ │
│ │   lm_head            30.0 ms        ->  18.8 ms (-37%)                                                                          │ │
│ │   Q proj             10.9 ms        ->   7.7 ms (-29%)                                                                          │ │
│ │   wo                 11.0 ms        ->   9.7 ms (-12%)                                                                          │ │
│ │   -------------------------------------------------------                                                                       │ │
│ │   TOTAL              63.9 ms        ->  50.6 ms (-21%)                                                                          │ │
│ │   (vs F32 baseline   1485 ms        ->  50.6 ms ~29x)                                                                           │ │
│ │                                                                                                                                 │ │
│ │ Smaller matmuls (Q proj, wo, lm_head) gain the most: thread-spawn was a                                                         │ │
│ │ bigger fraction of their kernel time. ffn_down already amortised the                                                            │ │
│ │ spawn cost over more compute. Per-call savings: roughly 200-300 us per                                                          │ │
│ │ matmul x ~250 matmuls/token => ~10-15 ms/token saved -- matches the                                                             │ │
│ │ observed delta.                                                                                                                 │ │
│ │                                                                                                                                 │ │
│ │ Single-batch prefill is essentially unchanged (133 vs 134 ms) because                                                           │ │
│ │ there's only one matmul-per-stage to amortise across.                                                                           │ │
│ │                                                                       

Summary:

┌──────────────────────┬────────────────┬─────────────┬───────┐
│ Metric               │ G4.1 (no pool) │ G4.2 (pool) │ Delta │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ decode/tok (E2B, 8t) │ 63.9 ms        │ 50.6 ms     │ -21%  │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ ffn_gate / ffn_up    │ 39.9 / 39.7    │ 25.9 / 25.8 │ -35%  │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ lm_head              │ 30.0 ms        │ 18.8 ms     │ -37%  │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ Q proj               │ 10.9 ms        │ 7.7 ms      │ -29%  │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ 16-tok gen E2B       │ 1265 ms        │ 1185 ms     │ -6%   │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ 16-tok gen E4B       │ 2163 ms        │ 2114 ms     │ -2%   │
├──────────────────────┼────────────────┼─────────────┼───────┤
│ vs F32 baseline      │ 23×            │ ~29×        │       │
└──────────────────────┴────────────────┴─────────────┴───────┘

Implementation: MatmulCtx now owns a RAII unique_ptr<ggml_threadpool> + shared work_buf. matmul_qf32 uses ggml_graph_plan +ggml_graph_compute instead of ggml_graph_compute_with_ctx. dequant_model takes n_threads so the pool is sized once at creation.

Smaller matmuls gain most — Q/wo/lm_head shed ~30-37% each because thread-spawn was a large fraction of their kernel time. ffn_downonly shed 16% (already amortised).

Open follow-ups noted in plan: (a) per_layer_model_proj → MatmulCtx (~5-10 ms), (d) lm_head argmax fusion, (e) cache Q8_K activations across layer matmuls.

===================================================================================

CLI reference — Gemma4.exe
==========================

Source of truth: examples/gemma4/Gemma4.cpp (print_usage at line 33, dispatch
lines 72-127). The Gemma-4 driver is a pure run-and-exit test/inspection tool
— there is no chat loop. Pick exactly one of the --gemma4-* mode flags per
invocation (kernel-test runs without a model; everything else requires -m).

Core
----
  -m PATH                          GGUF model file (gemma-4-E2B / E4B). Required
                                   for everything except --gemma4-kernel-test.
  -p "TEXT"                        Prompt for whichever mode consumes one. Most
                                   modes that take a prompt also accept it as
                                   the positional [PROMPT] argument after the
                                   mode flag (see each mode below); -p is the
                                   fallback when none is given inline.
  -n N                             Generic token count (default 64). Currently
                                   only consumed by modes that don't accept an
                                   inline N (kept for forward compatibility).
  -c N                             llama context size (default 0 = model max).
  -ngl N                           Layers to offload (default 99). Use -ngl 0
                                   for the CPU-only path (recommended for the
                                   hand-rolled forward — keeps weight pointers
                                   in host memory).
  --threads-prefill N              Threads used for the upstream oracle prefill
                                   in --gemma4-network-test / -gen / -profile
                                   (default = system).
  --threads-gen N                  Threads used for the hand decode + sizes
                                   the persistent ggml_threadpool inside
                                   MatmulCtx at dequant_model() time
                                   (default = system). Higher counts help the
                                   large MLP matmuls.
  -h, --help                       Print usage and exit.

Modes (pick one)
----------------
  --gemma4-dump-weights            Resolve + pretty-print the tensor schema for
                                   the GGUF: per-layer head_dim, n_ff, SWA vs
                                   full attention, rope freq base, softcap,
                                   tied lm_head, per-layer model_proj sizes.
                                   Best first run on a new GGUF.

  --gemma4-kernel-test             Pure kernel unit tests (rmsnorm, gelu,
                                   rope_neox SWA/full, dequant_row,
                                   qk_norm_per_head, matmul shims). NO model
                                   required. Run after a clean build.

  --gemma4-layer-test [IL]         Hand-rolled single-layer forward vs the
                                   upstream ggml oracle for layer IL
                                   (default 0). Prints max-abs / cos-sim.
                                   IL=0 is a SWA layer; pick a "full attention"
                                   IL (4 in E2B) to exercise the freq-factors
                                   RoPE path.
  --gemma4-layer-test-ntok N       Token count fed into the layer test
                                   (default 8). Larger N stresses the per-row
                                   matmul path more thoroughly.

  --gemma4-network-test [PROMPT]   Hand-rolled full-network forward vs upstream
                                   last-token logits. Reports max-abs, cos-sim
                                   and top-1 agreement. PROMPT is positional
                                   (defaults to a baked-in sentence if absent).
                                   This is the byte-for-byte parity gate.

  --gemma4-network-gen [PROMPT] [N]
                                   Greedy decode comparison: run the hand
                                   network and the upstream baseline side by
                                   side for N tokens (default 16) and print
                                   match-count. Used for end-to-end gen
                                   regression — must report N/N for a clean
                                   pass.

  --gemma4-network-profile [PROMPT] [N_DECODE]
                                   Run the hand network with the per-op
                                   profiler enabled and print a stage
                                   breakdown after N_DECODE tokens
                                   (default 4). Stages: prefill bucket then
                                   per-token decode bucket — matmul stages
                                   broken out (q/k/v/wo/ffn_gate/up/down/
                                   lm_head/ple), plus rope/attn/rmsnorm/
                                   residual. Use this to choose the next
                                   optimisation target.

Quick recipes
-------------
  # 1. New GGUF inspection
  Gemma4.exe -m D:\llama.cpp\models\gemma-4\gemma-4-E2B-it-Q4_K_M.gguf \
             --gemma4-dump-weights

  # 2. Pure kernel sanity (no model needed)
  Gemma4.exe --gemma4-kernel-test

  # 3. Per-layer parity (covers SWA + full-attn paths)
  Gemma4.exe -m ...E2B...gguf --gemma4-layer-test 0  --gemma4-layer-test-ntok 8
  Gemma4.exe -m ...E2B...gguf --gemma4-layer-test 4  --gemma4-layer-test-ntok 8

  # 4. End-to-end last-token agreement (baked prompt or your own)
  Gemma4.exe -m ...E2B...gguf -ngl 0 \
             --gemma4-network-test "The capital of France is"

  # 5. End-to-end greedy decode 16 tokens, both E2B and E4B should be 16/16
  Gemma4.exe -m ...E2B...gguf -ngl 0 --threads-gen 8 \
             --gemma4-network-gen "The capital of France is" 16
  Gemma4.exe -m ...E4B...gguf -ngl 0 --threads-gen 8 \
             --gemma4-network-gen "The capital of France is" 16

  # 6. Decode-step profile (current G4.2 baseline ~50 ms/tok on E2B Q4_K_M, 8t)
  Gemma4.exe -m ...E2B...gguf -ngl 0 --threads-gen 8 \
             --gemma4-network-profile "The capital of France is" 4

Regression gate (must all PASS after any change to the hand path)
-----------------------------------------------------------------
  --gemma4-kernel-test
  --gemma4-layer-test 4 --gemma4-layer-test-ntok 8
  --gemma4-network-test "The capital of France is"
  --gemma4-network-gen  "The capital of France is" 16  (both E2B and E4B)

Notes
-----
 * There is no qquant on/off switch: the qquant matmul shim (G4.1) and the
   persistent threadpool (G4.2) are always on. Routing inside the hand path
   uses dispatch_matmul (gemma4_forward.cpp ~313) which sends quantised
   weights through matmul_qf32 and F32 fallback weights through matmul_f32.
 * --threads-gen sizes the MatmulCtx pool at construction; changing it
   between runs is the only way to retune the pool.
 * --gemma4-network-gen requires --threads-gen >= 1 (default is system threads).
 * Cached prefill / sampling / chat loop are not yet implemented for Gemma-4
   (Phi-3 has them; tracked under G4.3+ in the session plan).


