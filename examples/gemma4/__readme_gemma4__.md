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


