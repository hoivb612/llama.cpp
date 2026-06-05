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
   