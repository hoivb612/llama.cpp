// flash_attn_128.hlsl - Mid-D variant for head dimensions in (64, 128].
// Used by ViT (Qwen3-VL D=72) and many small-LLM Q-heads (D=80, 96, 128).
//
// The default flash_attn.hlsl uses GROUP_SIZE = TILE_KV = 256, which leaves
// (256 - D) threads idle in Pass-3 V accumulation when D < 256 — only 28%
// utilization at D=72. Halving GROUP_SIZE doubles occupancy on small-wave
// GPUs (Intel Arc, wave=16) and lifts Pass-3 utilization to 56% at D=72,
// 100% at D=128.
//
// TILE_KV is also halved to 128 so the per-group work per tile drops in
// proportion to the thread count.
#define TILE_KV    128
#define GROUP_SIZE 128
#include "flash_attn.hlsl"
