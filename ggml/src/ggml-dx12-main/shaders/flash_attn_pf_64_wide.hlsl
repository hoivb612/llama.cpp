// Wide-tile D=64 prefill FA, for small-wave iGPUs (Intel UHD runs 256 threads
// as many narrow waves). Measured on Intel UHD at pp6144, with the host-side
// fa_tile_br kept in step with FA_PF_BR:
//
//   BR/BC/threads   LDS     t/s
//    8/64/256       21.8 KB  373
//   16/64/256                476
//   32/32/256       22.3 KB  522   <- this config
//   64/16/256       26.9 KB  481
//   16/16/128       10.6 KB  406
//
// NVIDIA prefers the opposite trade (see flash_attn_pf_64.hlsl): its barriers
// are cheap and it wants the extra workgroups for occupancy.
//
// Two theories for this curve were tested and rejected. Barrier cost: forcing
// the wave=32 blob on Intel does create successfully, so the width really
// changes, yet pp6144 moved by 0.3 t/s. LDS-driven occupancy: the 10.6 KB
// config is the slowest of the set, not the fastest.
//
// FA_PF_BR must match fa_tile_br in ggml-dx12.cpp; a mismatch under-dispatches
// query groups and silently drops rows.
#define HEAD_DIM 64
#define FA_PF_BR 32
#define FA_PF_BC 32
#include "flash_attn_pf.hlsli"
