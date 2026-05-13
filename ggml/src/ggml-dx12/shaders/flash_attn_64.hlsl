// flash_attn_64.hlsl - Decode-friendly FA variant for small head dimensions (D <= 64).
// Uses GROUP_SIZE = TILE_KV = 64 so Pass-3 V accumulation has 100% thread utilization
// when D == 64 (SmolVLM2/SmolLM2). Smaller workgroups also let more workgroups run
// concurrently on small-wave GPUs (Intel Arc, wave=16).
#define TILE_KV    64
#define GROUP_SIZE 64
#include "flash_attn.hlsl"
