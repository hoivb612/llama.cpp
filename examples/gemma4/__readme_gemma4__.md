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

Improving perf:

The full prefill breakdown (14.2 s) is far more revealing:

┌────────────────────────────────────┬─────────┬───────┐
│ Bucket                             │ Time    │ %     │
├────────────────────────────────────┼─────────┼───────┤
│ ple (per-layer embeddings)         │ 4006 ms │ 28.2% │
├────────────────────────────────────┼─────────┼───────┤
│ attention (scalar QK^T/softmax/×V) │ 3755 ms │ 26.4% │
├────────────────────────────────────┼─────────┼───────┤
│ gelu*up (elementwise GeLU)         │ 2716 ms │ 19.1% │
├────────────────────────────────────┼─────────┼───────┤
│ ffn_gate+up+down (matmuls)         │ 2361 ms │ 16.6% │
├────────────────────────────────────┼─────────┼───────┤
│ Q/K/V/wo proj + RoPE + norms       │ ~650 ms │ 4.6%  │
└────────────────────────────────────┴─────────┴───────┘

So prefill isn't one bottleneck — it's three scalar hand-kernels eating 74%: PLE (28%), attention (26%), and gelu*up (19%). The actual matmuls are only 17%. The  gelu*up  at 2.7 s for a pure elementwise op is a screaming red flag — that should be SIMD and near-free.

Decode has the same scalar-kernel story (24 tokens, 34 ms/tok):

┌────────────────────────────────────────┬────────┬──────┐
│ Decode bucket                          │ Time   │ %    │
├────────────────────────────────────────┼────────┼──────┤
│ ple + PLE merge (per-layer embeddings) │ 259 ms │ ~32% │
├────────────────────────────────────────┼────────┼──────┤
│ ffn matmuls (gate+up+down)             │ 261 ms │ 32%  │
├────────────────────────────────────────┼────────┼──────┤
│ gelu*up (elementwise)                  │ 135 ms │ 24%  │
├────────────────────────────────────────┼────────┼──────┤
│ lm_head                                │ 42 ms  │ 5%   │
├────────────────────────────────────────┼────────┼──────┤
│ Q/wo/attention                         │ 84 ms  │ 10%  │
└────────────────────────────────────────┴────────┴──────┘

Verdict — the gap is scalar hand-kernels, not the matmuls

The ccx-affin work already optimized the matmuls (ffn/lm_head). What's left — and what separates gemma4 from minslm — is three scalar kernels that ggml does in SIMD:

1.  gelu*up  — 19% prefill / 24% decode, for a pure elementwise op. This should be near-free. Biggest bang-for-effort, helps both gaps.
2.  ple  +  PLE merge  (per-layer embedding dequant/merge) — 28% prefill / 32% decode. Gather+dequant+scale — SIMD-able.
3.  attention  (scalar QK^T/softmax/×V) — 26% prefill, negligible in decode. The unique prefill cost; hardest to fix.

This directly matches your standing preference to never leave SIMD hot paths in scalar mode. For the decode 1.86× gap, vectorizing  gelu*up  +  ple  (together ~56% of decode) is the lever. For the prefill 5× gap, add scalar  attention .

Confirmed the root cause.  gelu_f32  (gemma4_kernels.cpp:48) is scalar and does the worst-case per element: two F16 round-trips ( fp16_to_fp32(fp32_to_fp16(x)) ) plus a  std::tanh  — for every element. That's why a "free" elementwise op costs 24% of decode.

The key insight: it's written this way to bit-match ggml's gelu — but ggml itself doesn't compute tanh per element. ggml uses a 64K F16 lookup table ( ggml_table_gelu_f16 ): one F16 conversion + one table index, no tanh. So replacing this scalar tanh loop with the same LUT ggml uses is both bit-identical to the oracle (the oracle is that table) and ~5–10× faster. Clean, safe, high-value.

Summary of the pinpoint (both gaps)

┌───────────────────────┬───────────┬──────────┬─────────────────────────────────────┬─────────────────────────────────┐
│ Kernel                │ Prefill % │ Decode % │ Nature                              │ Fix                             │
├───────────────────────┼───────────┼──────────┼─────────────────────────────────────┼─────────────────────────────────┤
│ gelu*up               │ 19%       │ 24%      │ scalar tanh + 2× F16 roundtrip/elem │ F16 LUT (bit-exact, easy)       │
├───────────────────────┼───────────┼──────────┼─────────────────────────────────────┼─────────────────────────────────┤
│ ple / PLE merge       │ 28%       │ 32%      │ scalar dequant/gather/merge         │ SIMD / cache                    │
├───────────────────────┼───────────┼──────────┼─────────────────────────────────────┼─────────────────────────────────┤
│ attention             │ 26%       │ ~0%      │ scalar QK^T/softmax/×V              │ flash/SIMD (prefill-only, hard) │
├───────────────────────┼───────────┼──────────┼─────────────────────────────────────┼─────────────────────────────────┤
│ matmuls (ffn/lm_head) │ 17%       │ 32%      │ ggml SIMD (already ccx-tuned)       │ done                            │
└───────────────────────┴───────────┴──────────┴─────────────────────────────────────┴─────────────────────────────────┘

To reproduce on your UMA box (same profiler, realistic ~476-tok prefill):

bin\RelWithDebInfo\Gemma4.exe -m c:\llama.cpp\models\gemma-4-E2B-it-Q4_K_M.gguf --threads-prefill 8 --threads-gen 8 --gemma4-ccx-affin --gemma4-network-profile "<paste a ~400-word paragraph>" 16

(The bucket ratios should match; absolute times differ.)

Step 1 (gelu F16-LUT) done — clean, bit-exact win. Apples-to-apples (short prompt, ccx-affin):

┌─────────────────┬──────────┬────────────────────────────┐
│                 │ Before   │ After                      │
├─────────────────┼──────────┼────────────────────────────┤
│ decode gelu*up  │ 134.7 ms │ 22.8 ms (5.9×)             │
├─────────────────┼──────────┼────────────────────────────┤
│ decode/tok      │ 33.98 ms │ 29.24 ms (+16% t/s → 34.2) │
├─────────────────┼──────────┼────────────────────────────┤
│ prefill gelu*up │ 2716 ms  │ 386 ms (7×)                │
├─────────────────┼──────────┼────────────────────────────┤
│ prefill t/s     │ 33.5     │ 39.6 (+18%)                │
└─────────────────┴──────────┴────────────────────────────┘

 gelu*up  fell from 24%→5% of decode. Kernel self-test passes; table is bit-identical to the old scalar formula.

Now step 2:  ple  — the biggest remaining bucket (prefill 32% / decode 30%).

The  lf_ple_ns  scope wraps the whole PLE block, which includes two matmuls ( inp_gate  1536×256,  proj  256×1536) plus scalar ops — per layer × 35. Two tiny matvecs shouldn't cost 8.8 ms/tok. Let me see how  dispatch_matmul  handles these vs the fast ffn path:

The big top-level  ple  bucket (4006 ms) is outside the layer loop — separate from the per-layer PLE block I just read (that's "PLE merge"). It's likely the upfront per-layer-embedding dequant for all tokens. Compute_per_layer_inputs  does a scalar  matmul_f32  (line 1191) projecting  [n_embd=1536] → [M = n_epl×n_layer = 256×35 = 8960]  per token — that's the biggest matmul in the model after lm_head, run single-threaded scalar instead of the fast ggml path that ffn/lm_head use. The readme already flagged this: "compute_per_layer_inputs still uses F32 matmul_f32... could be moved to qquant (ple bucket ~14% of decode)." The quantized tensor  w.per_layer_model_proj  exists (gemma4_weights.h:144) but ModelF32 only keeps the F32 vector.

matmul_f32  is scalar, single-threaded, double-accumulate — worst case. For prefill that's 476×8960×1536 ≈ 6.5B double-MACs single-threaded. The fix: route through  matmul_qf32  with the quantized  per_layer_model_proj  tensor (same path as ffn) — faster and closer to upstream's quantized matmul.  m.mm  is  mutable  (accessible from const), and  compute_per_layer_inputs  is only ever called from  network_step  where  mm  is ready. I'll add the quantized tensor handle to ModelF32, populate it, and branch to  matmul_qf32. Now the core change — route the projection through  matmul_qf32.

Step 2 (ple → qf32) is a massive win:

┌───────────────┬─────────────────────┬─────────────────────┐
│               │ Before              │ After               │
├───────────────┼─────────────────────┼─────────────────────┤
│ prefill ple   │ 3875 ms             │ 31 ms (125×)        │
├───────────────┼─────────────────────┼─────────────────────┤
│ decode ple    │ 211 ms              │ 6.9 ms (30×)        │
├───────────────┼─────────────────────┼─────────────────────┤
│ prefill total │ 12.0 s (39.6 t/s)   │ 8.0 s (59.2 t/s)    │
├───────────────┼─────────────────────┼─────────────────────┤
│ decode/tok    │ 29.24 ms (34.2 t/s) │ 20.45 ms (48.9 t/s) │
└───────────────┴─────────────────────┴─────────────────────┘

16/16 greedy tokens still match upstream.

Combined progress (steps 1+2, Threadripper profiler, E2B, ccx-affin)

┌────────────┬─────────────────────┬─────────────────────┬──────────┐
│            │ Baseline            │ +gelu +ple          │ Gain     │
├────────────┼─────────────────────┼─────────────────────┼──────────┤
│ decode/tok │ 33.98 ms (29.4 t/s) │ 20.45 ms (48.9 t/s) │ +66% t/s │
├────────────┼─────────────────────┼─────────────────────┼──────────┤
│ prefill    │ 33.5 t/s            │ 59.2 t/s            │ +77% t/s │
└────────────┴─────────────────────┴─────────────────────┴──────────┘

Both were scalar-kernel wins routed to the fast path (gelu→ggml's F16 LUT, ple→qf32 matmul), bit-faithful to upstream.

Remaining: prefill is now dominated by  attention  (3642 ms, 48%) — that's step 3, the scalar O(n²) QK^T/softmax/×V. It's the biggest remaining prefill cost but also the hardest/riskiest change (decode is barely affected by it).

Now step 3: the scalar attention (48% of prefill). Let me study the current implementation (gemma4_forward.cpp).

The scalar attention has two inefficiencies: (1) the QK^T dot is scalar double-accumulate over head_dim=512 contiguous floats (ideal for SIMD), and (2) the V-accumulation has a cache-hostile loop order (inner loop over  k  strides  V  by  n_kv  per element). There's already a pattern for reusing ggml's SIMD  vec_dot  (used in  lmhead_argmax_qf32 ). For F32 Q·K I can fetch ggml's F32  vec_dot  the same way — SIMD, no hand intrinsics, consistent with the codebase. And the V-accumulation just needs a loop-order fix (k-outer, contiguous d-inner) to become cache-friendly + auto-vectorizable.  I'll use ggml's F32  vec_dot  for QK^T (SIMD, reused) and reorder the V-accumulation to be cache-friendly + auto-vectorizable.

Step 3 (attention) is a big prefill win:

┌───────────────────┬──────────────────┬──────────────────────────┐
│                   │ Before           │ After                    │
├───────────────────┼──────────────────┼──────────────────────────┤
│ prefill attention │ 3642 ms          │ 1314 ms (2.77×)          │
├───────────────────┼──────────────────┼──────────────────────────┤
│ prefill total     │ 8.0 s (59.2 t/s) │ 5.8 s (82.1 t/s)         │
├───────────────────┼──────────────────┼──────────────────────────┤
│ decode            │ 20.45 ms/tok     │ 20.75 ms/tok (unchanged) │
└───────────────────┴──────────────────┴──────────────────────────┘

Prefill is now matmul-bound (ffn ~41%, attention 25%) — like minslm.

Final summary:

Optimization results — E2B, Threadripper profiler, ccx-affin

┌────────────┬─────────────────────┬─────────────────────┬───────────────┐
│ Metric     │ Baseline            │ Final               │ Gain          │
├────────────┼─────────────────────┼─────────────────────┼───────────────┤
│ decode/tok │ 33.98 ms (29.4 t/s) │ 20.45 ms (48.9 t/s) │ +66% t/s      │
├────────────┼─────────────────────┼─────────────────────┼───────────────┤
│ prefill    │ 33.5 t/s            │ 82.1 t/s            │ +145% (2.45×) │
└────────────┴─────────────────────┴─────────────────────┴───────────────┘

What changed (two commits, all bit-faithful)

 b1b625368  — steps 1+2:

• gelu → ggml's 65536-entry F16 LUT (drops per-element  tanh ): decode 135→23 ms, prefill 2716→386 ms
• ple (per-layer-embedding projection) →  matmul_qf32  on the quantized weight (was scalar double-accumulate  matmul_f32 ): decode 211→7 ms, prefill 3875→31 ms

 ca52e79e1  — step 3:

• attention Q·K → ggml SIMD F32  vec_dot ; scores·V reordered to cache-friendly k-outer with a bit-exact double accumulator: prefill 3642→1314 ms

Outstanding results on UMA 395. Here's the full end-to-end journey (real  -cpf  workload, 555-tok prompt):

UMA 395 E2B — before vs after all three optimizations

┌───────────────────────────┬────────────────────┬───────────────┬───────────────────┬──────────────┐
│                           │ Prefill (no-affin) │ Prefill (ccx) │ Decode (no-affin) │ Decode (ccx) │
├───────────────────────────┼────────────────────┼───────────────┼───────────────────┼──────────────┤
│ Before (start of session) │ 34.8 t/s           │ 35.8 t/s      │ 14.0 t/s          │ 23.3 t/s     │
├───────────────────────────┼────────────────────┼───────────────┼───────────────────┼──────────────┤
│ After (gelu+ple+attn)     │ 98.2               │ 103.3         │ 28.3              │ 40.9         │
├───────────────────────────┼────────────────────┼───────────────┼───────────────────┼──────────────┤
│ gain                      │ +182%              │ +189%         │ +102%             │ +76%         │
└───────────────────────────┴────────────────────┴───────────────┴───────────────────┴──────────────┘

vs the target (minslm-cli ccx on UMA: decode 43.4, prefill 180)

• Decode: 40.9 vs 43.4 t/s → gemma4 is now ~94% of minslm (was 54%). Effectively at parity. 🎯
• Prefill: 103.3 vs 180 t/s → 57% of minslm (was 20%). Closed most of the gap; the remainder is minslm's more efficient batched ffn/attention matmuls.

The image confirms the ccx-spread working — the burst pattern shows work landing across both CCDs' cores rather than packing one CCD.

Bottom line: the decode gap vs minslm is essentially closed on the dense model — which was the key prerequisite for MoE streaming to match resident perf. Prefill improved 2.9× but still trails (batched-matmul efficiency).

==========================================================

Findings — the prefill gap is not repack, and not dispatch overhead

┌────────────────────────────────────────────────┬──────────┬───────┐
│ Component                                      │ Time     │ Share │
├────────────────────────────────────────────────┼──────────┼───────┤
│ matmul compute (ggml_graph_compute, 245 calls) │ 1470 ms  │ 53%   │
├────────────────────────────────────────────────┼──────────┼───────┤
│ matmul build/setup (init+graph+plan, per-call) │ 103 ms   │ 3.7%  │
├────────────────────────────────────────────────┼──────────┼───────┤
│ non-matmul (attention pool + scalar glue)      │ ~1193 ms │ 43%   │
└────────────────────────────────────────────────┴──────────┴───────┘

Key conclusions:

• Per-dispatch build overhead is only 6.6% of matmul time → a prefill-shape graph cache would barely help (~+4%). Ruled out.
• The matmul compute uses the identical ggml kernel minslm uses, yet gemma4's total is ~2× slower — so the gap lives in the structural difference: minslm runs one fused graph per prefill (all ops multithreaded, pool stays hot, ~1 barrier), while gemma4 issues 245 separate  ggml_graph_compute  calls and runs the glue (gelu*up = 190 ms, rope, rmsnorm, residuals, PLE scalar) single-threaded between them.

Options to close it

1. Parallelize the single-threaded scalar kernels (geluup, rope, rmsnorm, residuals) across the existing pool — low risk, self-contained, est. +8–15% prefill. geluup alone (190→~25 ms) is the biggest single item.
2. Fuse each layer's ops into one ggml sub-graph (norm→qkv→...→ffn as one graph, ~35 computes instead of 245) — medium change, keeps hand control of MoE, recovers barrier + quantization-reuse efficiency.
3. Full fused ggml graph for the dense path (matches minslm) — biggest win, biggest change/risk, and would diverge from the hand-coded forward that the MoE-streaming path relies on.

Excellent — big drops in every scalar bucket:

• gelu*up: 190 → 27.8 ms (6.8×)
• RoPE: 58 → 8.5 ms
• q/k/v rms-norm: 20 → 3.9 ms
• attn_norm/ffn_norm/post_attn/post_ffw all ~3-4 ms (were 9-12)

Prefill 85.3 → 94.4 t/s (+10.6%) on this box.

Strong result on the full 555-token prompts on Threadripper:

┌─────────┬───────────┬────────────────────────┐
│         │ Before    │ After                  │
├─────────┼───────────┼────────────────────────┤
│ prefill │ 78–86 t/s │ 96–105 t/s (+~22%)     │
├─────────┼───────────┼────────────────────────┤
│ decode  │ ~37 t/s   │ ~37–38 t/s (unchanged) │
└─────────┴───────────┴────────────────────────┘

The longer prompts benefit more than the 236-tok profile (+10.6%) because there's more parallelizable scalar work amortizing each pool dispatch. Decode is untouched — the 32K-element threshold keeps the single-token path fully serial.

Verification:

• Bit-identical: my binary reproduces the exact baseline 4/16 "solitary" divergence and cos_sim, proving the parallelized prefill matches the serial math element-for-element.
• kernel self-test: PASS.
• Only  gemma4_forward.cpp  changed (added  <functional> , a threshold-guarded  parallel_for  over the existing attn pool, and applied it to gelu*up, RoPE Q/K, all rmsnorm sites, residual adds, and the PLE scalar tail).

UMA 395 confirms the win:

┌────────────────────┬───────────┬──────────────────────┐
│ UMA 395, ccx-affin │ Before    │ After                │
├────────────────────┼───────────┼──────────────────────┤
│ prefill            │ 103.3 t/s │ 113.4 t/s (+9.8%)    │
├────────────────────┼───────────┼──────────────────────┤
│ decode             │ 40.9 t/s  │ 40.9 t/s (unchanged) │
└────────────────────┴───────────┴──────────────────────┘

Prefill is now 63% of minslm (was 57%); decode holds at ~93% parity. Solid, low-risk win banked and verified on both boxes.

The remaining prefill gap (113 vs 180) is now almost entirely the matmul compute itself — the scalar glue is largely defused. The next lever is option #2: batch each layer's ops into one (or a few) fused ggml graphs so all 245 separate  ggml_graph_compute  dispatches per prefill collapse into ~35, keeping the pool hot and letting ggml quantize/reuse activations across nodes. That's a medium-risk change but the clearest path to close more of the gap.

The next lever (if you want to keep pushing prefill toward minslm's 163) would be option #2 — batching per-layer ops into fewer ggml graphs to cut the 245 separate dispatches.

There's already a full fused single-layer ggml graph ( oracle_layer_forward_f32 ) mirroring the reference model — used as the correctness oracle. This is essentially the option-#2 implementation already written in gemma4_forward.cpp.

oracle_layer_forward_f32  is a complete fused single-layer ggml graph, but it uses F32 weights + single-thread + fresh 1GB arena (correctness oracle only). For the prototype I'll adapt it into a prefill path: quantized weight handles ( L.*_t ), the persistent  mm.pool , a reused arena, SWA-aware mask, and K/V persisted to the cache. 

Prototype #2 (fused per-layer prefill graph) works. Summary:

What I built ( --gemma4-prefill-fused 0|1 , default off):

• New  layer_forward_fused_prefill()  in  gemma4_forward.cpp : builds one ggml graph per layer using the quantized weight handles ( L.wq_t … L.proj_t ) as cross-context leaves, runs it multithreaded on the persistent  mm.pool , with an SWA-aware causal mask, a reused persistent arena, and persists roped-K / normed-V into the external  K_cache / V_cache  so decode (still the hand path) reads them.
• Wired into  network_step  only for the prefill case ( n_new>1 ,  n_past==0 , non-reuse layers, pool present); decode + reuse layers untouched.
• Global setter/getter ( set/get_prefill_fused ) + CLI flag, mirroring the existing  attn-parallel / matmul-cache  pattern.

Results (Threadripper dev box, E2B Q4_K_M, 8t):

┌───────────┬──────────────┬───────────┐
│           │ prefill      │ decode    │
├───────────┼──────────────┼───────────┤
│ fused OFF │ ~90–98 t/s   │ unchanged │
├───────────┼──────────────┼───────────┤
│ fused ON  │ ~106–112 t/s │ unchanged │
└───────────┴──────────────┴───────────┘

≈ +10% prefill, decode identical (uses hand path).

Correctness:

•  --gemma4-network-gen "1, 2, 3, 4," 24 : 24/24 match (fused on & off).
•  -cpf  18-prompt run: output JSON byte-coherent with the known-good answers.
• The only divergence ( "…France…" 16  → 15/16) is a tie-break at step 13 deep in the degenerate  <turn|>/<eos>  collapse tail; first real token " Paris." matches. This is expected ULP-level drift (fused F32 ggml reductions vs hand double accumulators) — the fused graph is numerically upstream-like.

The prototype collapses ~245 per-prefill dispatches into ~35 fused graphs. To test on UMA 395 (after rebuild):

Gemma4.exe -m <E2B.gguf> --threads-gen 8 --threads-prefill 8 --gemma4-ccx-affin \
  --gemma4-prefill-fused 1 -cpf ..\examples\llm-infer\prompts\custom_prompts_gemma4.txt

Compare prefill t/s against the same run without  --gemma4-prefill-fused 1 . Default is off, so the baseline path is unaffected.

Confirmed on UMA 395 — the fused prefill stacks cleanly on top of ccx-affin:

┌───────────────────────────┬────────────────────┬────────┐
│ config                    │ prefill (prompt 2) │ decode │
├───────────────────────────┼────────────────────┼────────┤
│ no ccx-affin, no fused    │ 96.0 t/s           │ 25.4   │
├───────────────────────────┼────────────────────┼────────┤
│ ccx-affin, no fused       │ 120.2 t/s          │ 41.6   │
├───────────────────────────┼────────────────────┼────────┤
│ ccx-affin + prefill-fused │ 132.7 t/s          │ 41.2   │
└───────────────────────────┴────────────────────┴────────┘

So fused adds ~+8–10% prefill on top of ccx-affin (120→133, 114→124), decode unchanged, output coherent. That moves prefill from ~63% to ~71% of minslm (128 vs 180 t/s). Same relative gain as the dev box — consistent win on both boxes.

The prototype is validated and pushed. Where it stands vs minslm: the remaining gap is now in the fused-graph compute itself (matmul kernel + ggml op overhead per layer), not dispatch count.

===================================================================================

How repack works here (XBCG):  ggml_cpu_repack_tensor_callgraph(graph)  walks a graph, and for each  MUL_MAT  node repacks  src0  (the weight) in place and flips  src0->type  to the repacked type — a one-time cost that persists. You enable it globally via  llama_set_tensor_repack_mode(XBCG) . Both APIs are reachable from gemma4 (it links llama + ggml).

Two hard prerequisites I found:

1.  use_mmap=false  when repack is on. Resident weights currently point into read-only mmap ( gemma4_loader.cpp  uses default params) — in-place repack would fault. Must load writable.
2. The callgraph pass only handles  GGML_OP_MUL_MAT , not  MUL_MAT_ID . So the fused resident MoE ( matmul_moe_id_qf32 ) won't repack via this path.

How each matmul shim maps to repack effort:

┌──────────────────────┬────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
│ Path                 │ Shim               │ Repack fit                                                                        │
├──────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Dense (attn q/k/v/o, │ matmul_qf32        │ Easy — references W directly, cache already rebuilds on type drift. Repack once,  │
│ shared MLP, lm_head) │                    │ persists. This is the bulk of minslm's gain, identical in resident & streaming.   │
├──────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Resident MoE experts │ matmul_expert_qf32 │ Medium — src0 is a view of the bank; must repack the whole bank tensor once       │
│                      │                    │ up-front, not per-view                                                            │
├──────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Fused resident MoE   │ matmul_moe_id_qf32 │ Hard — MUL_MAT_ID, not covered by the callgraph pass                              │
├──────────────────────┼────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
│ Streaming MoE        │ matmul_qblock_qf32 │ Tricky — blocks are transient; repack-per-fetch likely helps prefill but not      │
│ experts              │                    │ decode GEMV; would need repack-in-store to persist                                │
└──────────────────────┴────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘

My recommendation: phase it, starting with the dense weights — that's where the observed Gemma4 gain in minslm comes from (attn + MLP dominate the non-expert matmul time), it works identically in both resident and streaming modes, and it's the lowest-risk piece (the shim already expects in-place repack).

Phase 1 — dense-weight repack (attn/MLP/lm_head) — IMPLEMENTED

┌─────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────┐
│ Path                                    │ Repack effect                                                                       │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Dense attn/MLP/lm_head (prefill +       │ Phase 1 _x8 repack (already shipped)                                                │
│ decode)                                 │                                                                                     │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ --gemma4-prefill-fused 1 (dense layers) │ unchanged, still fused                                                              │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Resident MoE experts                    │ auto-forced to non-fused, gate/up banks repacked to _x8 (down bank skipped,         │
│ (--gemma4-moe-fused)                    │ 704%256≠0)                                                                          │
├─────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────┤
│ Streaming MoE experts                   │ separate phase (transient blocks) — not covered by this change                      │
└─────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────┘

-------------------------------------------------------------
CLI: --gemma4-repack-ggml | --gemma4-repack-xbox | --gemma4-repack-xbcg
Wiring: llama_set_tensor_repack_mode() after ggml_backend_load_all();
        loader gets use_mmap=false when any repack mode is on (in-place
        repack must not write read-only mmap pages).
Injection: ggml_cpu_repack_tensor_callgraph(gf) in BOTH matmul_qf32 paths
        (cached build + per-call). No-op unless a mode is set. Repacks
        src0 (=W, the model weight) in place once; the matmul_qf32 cache
        already rebuilds on w_type drift so it self-heals to the _x8 type.
CRITICAL FIX: tok_embd (and per_layer_tok_embd) are DUAL-USE — read
        row-by-row via dequant_row() for per-token input embeddings
        (original K-quant layout) AND used as the tied lm_head via
        matmul_qf32(). Repacking the lm_head flips the shared tensor to
        _x8 in place -> the next token's embedding lookup reads _x8 bytes
        as Q4_K -> access violation. Marked both GGML_TENSOR_FLAG_NO_REPACK
        in dequant_model so the callgraph pass skips them. Dense attn/MLP
        weights are matmul-only and stay repackable.

Results (dev box, Threadripper, AVX-512), 8 threads:
  E2B (dense, Q4_K_M):
    baseline  decode 26.46 t/s   prefill 18.6 t/s   network-gen 16/16
    xbcg      decode 29.22 t/s   prefill 22.7 t/s   network-gen 15/16
              (+10% decode, +22% prefill; the one flip is a benign
               near-tie "occupied"/"puzzled" from _x8 reduction-order
               drift — baseline is 16/16, same tradeoff as minslm repack)
  26B-A4B (MoE, Q4_K_M), -cpf, moe-budget 4096, prefetch on:
    baseline  gen ~8.1 t/s   prefill ~29.3 t/s   coherent JSON
    xbcg      gen ~6.7 t/s   prefill ~29   t/s   coherent JSON  (-17% gen)

Key finding: Phase 1 helps DENSE models (E2B/E4B — it repacks all their
matmuls) but REGRESSES the 26B MoE on this box, because the dense weights
(attn + shared MLP) are a small slice while the EXPERTS dominate ~85% of
FFN compute and Phase 1 does not touch them. On the AVX-512 dev box the
native Q4_K kernel already beats _x8, so repacking only the small dense
slice is a net loss. The 26B win requires Phase 2 (repack the experts).
Verdict for dense-only repack should be taken on UMA 395 (the target box)
where the _x8 gain is not masked by strong AVX-512.

Phase 2 — resident MoE expert repack (IMPLEMENTED)
--------------------------------------------------
Goal: extend repack to the routed experts, which dominate ~80-85% of the 26B
FFN compute (Phase 1 only touched the small dense slice). Only the resident
(all-experts-in-memory) path is covered here; streaming is Phase 3.

Why a special case is needed (and why "do nothing" is not enough): the expert
banks never flow through matmul_qf32, so the Phase 1 callgraph never touches
them — with no extra work they simply run unrepacked (safe, zero gain). To get
the _x8 speedup we must repack the gate/up banks ourselves. But the DEFAULT
resident path is the FUSED ggml_mul_mat_id (matmul_moe_id_qf32), and mul_mat_id
has no _x8 kernel. Repacking a bank flips its shared ->type to _x8 for every
reader, so once repacked the fused path can no longer consume it. Therefore
repack + resident MUST force the per-expert non-fused path (matmul_expert_qf32),
whose ggml_view_2d over each expert slab inherits the _x8 bank type.

Wiring:
  set_repack_active(mode != NONE) in Gemma4.cpp (after llama_set_tensor_repack_mode).
  dequant_model(): if repack active && any MoE layer && NOT streaming, repack
    each layer's gate_up (merged) or gate+up banks via repack_expert_bank(),
    then set_moe_fused(false) and print a warning.
  repack_expert_bank() (gemma4_matmul.cpp): builds a throwaway no_alloc graph
    with one MUL_MAT per expert slab (each slab a 2D src0 aliasing bank->data
    + e*nb[2]), runs ggml_cpu_repack_tensor_callgraph() to repack every slab's
    bytes in place, then flips bank->type once to the _x8 type. Per-slab is
    mandatory: single_thread uses ne[1] (not ne[1]*ne[2]) as nrows, so calling
    it on the whole 3D bank would repack only expert 0 yet retype the whole
    tensor. Repack is a pure in-place reorder so the bank's nb[] stay valid.

Not repacked: down_exps has ne[0]=n_ff_exp=704 (704%256=192!=0), which
single_thread rejects — left as Q4_K. Streaming (ExpertStore ready) is skipped
entirely (the resident bank is unused there; blocks come from pread).

Q4_0 nb[] fixup: after flipping bank->type we recompute nb[] for the new type.
The repack is in-place (row stride nb[1] and total bytes preserved), but some
_x8 linkage types change (blck_size, type_size): Q4_0 (blck 32, tsize 18) ->
Q4_0_x8 (blck 256, tsize 144). ggml_nbytes() reads ne[0]*nb[0]/blck_size, so a
stale nb[0]=18 with the new blck 256 undercounts the bank 8x and the per-expert
ggml_view_2d bounds assert fires (seen on a QAT smart-Q4_0 build). K-quant _x8
types keep the base (blck, tsize) so the recompute is a no-op for them; the
dense matmul_qf32 path is unaffected because it uses W directly (no view / no
ggml_nbytes bounds check).

A/B (dev box TR7995WX AVX-512, 26B-A4B Q4_K, hand resident, network-profile
"The meaning of life is", 6-tok prefill + 8 decode, threads 8/8):
  baseline (fused, no repack)  prefill 6.8 t/s   decode 9.49 t/s
  repack-xbcg (non-fused, _x8) prefill 12.2 t/s  decode 11.90 t/s
  => prefill +79%, decode +25%. Coherent JSON on -cpf (first 4 prompts spot-
     checked; down bank staying Q4_K keeps numerics sane). Note: unlike the
     dense-only Phase 1 (which regressed 26B on this AVX-512 box), Phase 2 wins
     here because it moves the experts — the bulk of the compute — to _x8, and
     the win over the fused baseline includes the cost of dropping fusion.
     Real verdict still on UMA 395.

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

  --gemma4-save-kv PATH [PROMPT] [N]
                                   G4.3 — Cached prefill (save side).
                                   Run prefill on PROMPT, sample
                                   first_gen_token = argmax of the
                                   post-softcap last-token logits, then
                                   serialise NetworkState + first_gen_token
                                   + model fingerprints to PATH (atomic
                                   write via "<PATH>.tmp" + rename).
                                   After saving, continue greedy gen for
                                   N total tokens (hand-only; no upstream
                                   comparison). Default N=32.

  --gemma4-load-kv PATH [PROMPT] [N]
                                   G4.3 — Cached prefill (load side).
                                   Skip prefill entirely; restore
                                   NetworkState from PATH, seed gen with
                                   the cached first_gen_token, and greedy
                                   decode N total tokens. PROMPT is
                                   optional and used only to compute an
                                   advisory prompt-hash mismatch warning;
                                   the cache is valid even if PROMPT
                                   differs from the one used at save
                                   time. Default N=32.

  --gemma4-load-kv-strict          Promote weight_hash mismatch on
                                   --gemma4-load-kv from warning to fatal.
                                   The topology_hash check (per-layer
                                   dims + RoPE config) is always strict.

  --gemma4-chat                    G4.4 — Hand-path chat loop. Without
                                   -p, runs interactively (one line per
                                   turn; blank line or EOF to quit).
                                   With -p, runs a single user turn and
                                   exits. KV grows across turns; each
                                   turn's user-side delta is hand-built
                                   in the Gemma format
                                     <start_of_turn>user\n{Q}<end_of_turn>\n<start_of_turn>model\n
                                   (with a leading <end_of_turn>\n on
                                   turn N>=2 to close any prior
                                   assistant turn that stopped at the
                                   n_predict cap without natural EOG).
                                   Assistant gen tokens are committed
                                   directly to KV without ever being
                                   re-tokenized through text, which is
                                   required for determinism on
                                   BPE/SentencePiece vocabs (round
                                   trips are not safe). Uses -n as the
                                   per-turn gen cap (-1 = no cap).

  --gemma4-chat-test               G4.4 — Scripted 2-turn determinism
                                   test. Runs the same conversation
                                   incrementally (turn-by-turn KV
                                   reuse) and via a fresh full prefill
                                   that re-decodes turn-1 gen tokens
                                   into a clean NetworkState, then
                                   PASSes iff turn-2 token sequences
                                   match. Greedy; no model output is
                                   sampled.

  --gemma4-chat-ctx N              NetworkState capacity for chat
                                   sessions (default 4096). The chat
                                   loop fails fast if a prefill would
                                   exceed this cap.

  --temp F                         Sampling temperature for --gemma4-chat
                                   (default 0.0 = greedy argmax with no
                                   sampler chain allocated).

  --min-p F                        min-p cutoff (default 0.05). Ignored
                                   when --temp 0.

  --seed N                         Sampler seed (default
                                   LLAMA_DEFAULT_SEED). Ignored when
                                   --temp 0.

Cache file format ("G4KV0001"): 64-byte LE header (magic, version,
n_layer, n_kv_owning, n_tokens, kv_type=F32, first_gen_token, n_vocab,
softcap, prompt_hash, topology_hash) followed by a payload prologue
(weight_hash), pos_all (i32 LE, contiguous 0..n_tokens-1), then one
slab per owning layer (kv_reuse_il == -1) carrying (il, n_head_kv,
head_dim) + raw F32 K + V data. Shared-KV layers are not serialised.
--gemma4-save-kv and --gemma4-load-kv are mutually exclusive in a
single invocation.

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

  # 7. G4.3 cached prefill round-trip (E2B; byte-identical gen text)
  Gemma4.exe -m ...E2B...gguf -ngl 0 --threads-gen 8 \
             --gemma4-save-kv %TEMP%\g4.kvc "The capital of France is" 16
  Gemma4.exe -m ...E2B...gguf -ngl 0 --threads-gen 8 \
             --gemma4-load-kv %TEMP%\g4.kvc "The capital of France is" 16

  # 8. G4.4 chat: scripted 2-turn determinism test (must PASS on both)
  Gemma4.exe -m ...E2B...gguf --gemma4-chat-test --threads-gen 8
  Gemma4.exe -m ...E4B...gguf --gemma4-chat-test --threads-gen 8

  # 9. G4.4 chat: one-shot single-turn greedy
  Gemma4.exe -m ...E2B...gguf --gemma4-chat -p "What is 2+2?" \
             -n 32 --threads-gen 8

  # 10. G4.4 chat: one-shot single-turn sampled (temp+min-p+seed)
  Gemma4.exe -m ...E2B...gguf --gemma4-chat -p "What is 2+2?" \
             -n 32 --temp 0.8 --min-p 0.05 --seed 42 --threads-gen 8

  # 11. G4.4 chat: interactive multi-turn (Ctrl-Z<Enter> or blank line to quit)
  Gemma4.exe -m ...E2B...gguf --gemma4-chat \
             -n 64 --gemma4-chat-ctx 4096 --threads-gen 8

  # 12. llama-bench-style throughput table (qquant + upstream side-by-side)
  Gemma4.exe -m ...E2B...gguf --gemma4-bench \
             --bench-pp 64 --bench-tg 64 --bench-reps 3 --threads-gen 8

  # 12a. qquant only (skip the upstream comparison rows)
  Gemma4.exe -m ...E2B...gguf --gemma4-bench --bench-backend qquant \
             --bench-pp 128 --bench-tg 128 --threads-gen 8

CLI flags for --gemma4-bench (run-and-exit)
-------------------------------------------
  --gemma4-bench                    Enable bench harness; prints a markdown
                                    table identical in shape to
                                    `llama-bench.exe` output, then exits.
  --bench-pp N                      Prefill (prompt processing) size in
                                    tokens. One pp{N} row per backend.
                                    Default 64.
  --bench-tg N                      Decode (token generation) size in
                                    tokens. One tg{N} row per backend.
                                    Default 64.
  --bench-reps N                    Measured repeats per test. Each test
                                    also runs one implicit warmup pass
                                    (always discarded). Default 3.
  --bench-threads N                 Thread count for both backends. When
                                    set, overrides the default resolution
                                    (--threads-gen → --threads-prefill → 4).
                                    Also sizes the qquant ModelF32::mm pool.
  --bench-backend qquant|upstream|both
                                    Which backends to bench. qquant drives
                                    network_step (the hand path);
                                    upstream drives llama_decode on the
                                    same model. Default both.
  --threads-gen N                   Thread count for both backends (also
                                    sizes the qquant ModelF32::mm pool).
                                    Falls back to --threads-prefill, then
                                    to 4 if neither is set.
  --seed N                          Seed for the bench's random-token
                                    generator (qquant and upstream get
                                    distinct seeds derived from this).
                                    Default 1234.

  Test names follow llama-bench: `pp{N}` measures a single batched prefill
  of N random in-vocab tokens (t/s = N / prefill_ms * 1000); `tg{N}`
  measures N single-token decode steps after a 1-token warmup
  (t/s = N / decode_ms * 1000). Random tokens are content-agnostic --
  this is a throughput probe, not a quality check. Use a small `--bench-reps`
  (2-3) for a quick sanity check; reps>=5 for stable stddev.

Regression gate (must all PASS after any change to the hand path)
-----------------------------------------------------------------
  --gemma4-kernel-test
  --gemma4-layer-test 4 --gemma4-layer-test-ntok 8
  --gemma4-network-test "The capital of France is"
  --gemma4-network-gen  "The capital of France is" 16  (both E2B and E4B)
  --gemma4-chat-test                                   (both E2B and E4B)

Notes
-----
 * There is no qquant on/off switch: the qquant matmul shim (G4.1) and the
   persistent threadpool (G4.2) are always on. Routing inside the hand path
   uses dispatch_matmul (gemma4_forward.cpp ~313) which sends quantised
   weights through matmul_qf32 and F32 fallback weights through matmul_f32.
 * --threads-gen sizes the MatmulCtx pool at construction; changing it
   between runs is the only way to retune the pool.
 * --gemma4-network-gen requires --threads-gen >= 1 (default is system threads).
 * --gemma4-chat ignores llama_chat_apply_template even when the model ships
   one: Gemma-4 GGUFs ship a complex Jinja template the built-in template
   matcher does not recognise, and re-tokenising the assistant's prior
   response through text is not safe on BPE/SentencePiece vocabs (see
   examples/gemma4/__bpe_chat_pitfall__.md for the full story). Per-turn
   user deltas are hand-built in the Gemma <start_of_turn>...<end_of_turn>
   format and tokenised atomically; assistant gen tokens are committed
   directly to KV.

========================================================================

 - Gemma4: --bench-threads > --threads-gen > --threads-prefill > 4
 - Phi3:   --bench-threads > --phi3-fused-qquant-threads > --threads-gen > 1

 Sub-flags (both binaries): --bench-pp N --bench-tg N --bench-reps N --bench-backend qquant|upstream|both. On Phi-3,--phi3-fused-qquant-rmsnorm-fuse / --phi3-fused-qquant-attn-parallel / --phi3-fused-qquant-threads are honored when picking theqquant configuration.



==========================================================

G7 (prototype) - fused per-layer prefill graph (--gemma4-prefill-fused 0|1)

Prefill option #2: collapse the ~7 scalar hand-kernels + separate matmul
dispatches per layer into ONE ggml graph per layer, built from the QUANTIZED
weight handles (L.wq_t .. L.proj_t) as cross-context leaves and run
multithreaded on the persistent mm.pool. This turns ~245 ggml_graph_compute
dispatches per prefill into ~35, and lets ggml multithread every op (rms_norm,
rope, gelu, mul, add, softmax) instead of the single-thread scalar path.

layer_forward_fused_prefill() (gemma4_forward.cpp) mirrors the correctness
oracle oracle_layer_forward_f32() exactly, but:
  * quantized weights instead of F32 dequant (no big-weight F32 copies);
  * multithreaded on mm.pool (ggml_graph_plan / ggml_graph_compute), not 1 thread;
  * persistent reused arena (no 1 GiB malloc/free per layer);
  * SWA-aware causal mask when L.is_swa (window = model n_swa);
  * persists roped-K / rms-normed-V into the external K_cache / V_cache so the
    subsequent hand DECODE path reads them (layout matches: [head_dim,
    n_head_kv, n_tokens] == cache row t*n_kv + h*head_dim + d).

Engaged only for prefill (n_new > 1, n_past == 0, non-reuse layers, pool set);
decode and KV-reuse layers stay on the hand path unchanged. Default OFF.

Numbers (Threadripper 7995WX dev box, E2B Q4_K_M, 8 threads, -cpf 555-tok):
  fused OFF : prefill ~90-98 t/s
  fused ON  : prefill ~106-112 t/s   (approx +10%)
  decode    : unchanged (hand path)

Correctness:
  --gemma4-network-gen "1, 2, 3, 4," 24 : 24/24 match (fused on and off)
  -cpf 18-prompt run : JSON answers byte-coherent with the known-good output
  The lone 15/16 on "...France..." is a tie-break at gen step 13 deep in the
  degenerate <turn|>/<eos> collapse tail (first real token " Paris." matches);
  expected ULP-level drift - fused uses F32 ggml reductions (upstream-like)
  vs the hand path's double accumulators.

==========================================================

G7 update - fused per-layer prefill is now the DEFAULT (--gemma4-prefill-fused 1)

After validating on both the Threadripper dev box and the UMA 395 (consistent
~+10% prefill, decode unchanged, output coherent), the fused per-layer prefill
graph is enabled by default. Pass --gemma4-prefill-fused 0 to fall back to the
hand scalar-kernel path for A/B comparison.

Multi-layer fusion (fusing >1 layer per ggml graph) was prototyped and
measured: prefill stayed within run-to-run noise (+-2 t/s) across chunk sizes
1/4/8/18/35, while arena memory grew linearly (~8 GB at 35 layers). The
per-graph ggml_init/build/free overhead is cheap and the matmul+op compute is
identical regardless of layers-per-graph, so multi-layer fusion was dropped -
the single-layer fused path already captures the entire available win.

G7 fix - fused prefill must NOT run on MoE layers. The fused per-layer prefill
graph hardcodes the DENSE FFN (+PLE); it does not emit the MoE shared-MLP +
router + experts path. When fused was made the default, 26B-A4B (all-MoE
layers) prefill silently ran shared-MLP-only and skipped the experts,
producing garbage ("로, 1, 2, 3, 4, 5" vs the correct " {100, 500, ...").
Fixed by adding `&& !L.is_moe_layer` to the network_step fused guard so MoE
layers always take the hand path. Verified: E2B dense 24/24 (fused on),
26B fused-on now byte-identical to fused-off. Dense models are unaffected.

P2b - multi-worker prefetch (--gemma4-moe-prefetch-threads N, default 2)

The P2 background prefetch used a SINGLE I/O worker, so in the tight-budget
(I/O-bound) streaming regime the consumer GEMV raced ahead and blocked on
in-flight reads (high `waits`). Profiling 26B-A4B showed the streaming path is
I/O-bound only BELOW ~4 GiB budget; at/above 4 GiB it already matches the
all-resident compute ceiling (streaming == resident == 11.3 t/s on UMA), and
the fused mul_mat_id resident path gives no extra decode headroom there. So the
lever that matters for the memory-saving regime is raising read bandwidth, not
compute batching.

ExpertStore::set_prefetch(on, n_workers) now spawns N worker threads on the
same queue. pread / ReadFile-with-OVERLAPPED-offset are thread-safe for
concurrent positioned reads on one handle, and the existing mtx_ + claimed_
logic already serializes all pool mutations and prevents duplicate reads, so
worker_loop needed ZERO logic changes - each key is still popped once and read
with the lock released, now concurrently. Data path is provably unchanged:
fetches / bytes_read / prefetch_reads are byte-identical across worker counts,
and hand output is bit-identical (workers only affect I/O timing).

Results (26B-A4B Q4_K, budget 1024 MiB, 24 decode tok, ccx-affin):
  UMA 395:      1w 6.06  2w 8.31 (+37%)  4w 8.41 (+39%, sweet spot)  8w 8.10
  TR 7995WX:    1w 4.98  2w 6.52 (+31%, sweet spot)  4w 6.51  8w 6.26
Prefill also gains (~+30% on UMA, 4.3 -> 5.6 t/s) since prefill streams too.
Past the sweet spot, N I/O workers contend with the N gen-compute threads
(16-core UMA: 8 gen + 8 io saturates), so throughput dips even as `waits`
keeps falling. Neutral in the compute-bound regime (4 GiB: 10.11 -> 10.16).

P2c - effective streaming bandwidth instrumentation (io_bw)

ExpertStoreStats now tracks read_ns: wall-ns summed across every read_at()
call (I/O busy thread-time). log_stats reports io_read (total thread-ms in
I/O) and io_bw = bytes_read / read_ns in GiB/s/stream -- the effective
PER-STREAM read bandwidth. Aggregate wall-clock bandwidth ~= io_bw times the
number of overlapping workers.

This quantifies the streaming I/O path directly. Dev box (TR7995WX, warm
page cache), 26B-A4B, budget 1024, decode:
  1w io_bw 3.12 GiB/s (agg 3.1)   2w 2.46 (agg 4.9)   4w 1.35 (agg 5.4)
Per-stream BW falls as workers rise because they contend for the same
memory-copy bandwidth; aggregate saturates ~5 GiB/s, which is why decode t/s
plateaus past 2 workers.

CAVEAT - warm page cache masks storage speed. On big-RAM boxes the whole
model fits the OS page cache after warm-up, so read_at() is a kernel memcpy,
not a device read: io_bw reports RAM-copy speed (~3 GiB/s/stream here), NOT
disk/NVMe speed. To measure actual storage (and see NVMe gains) the reads
must miss the cache: either drop the OS standby/page cache between runs
(cold run) or use an uncached read path (FILE_FLAG_NO_BUFFERING with
sector-aligned rounding). The io_bw metric is what makes those comparisons
visible once the cache is bypassed.
