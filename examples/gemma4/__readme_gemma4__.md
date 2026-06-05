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
