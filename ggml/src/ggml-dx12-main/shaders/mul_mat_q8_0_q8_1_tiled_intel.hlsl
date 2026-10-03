// Intel UHD path: avoid cross-word Q8_0 load reconstruction.
#define Q8_TILED_BYTE_LOADS 1
#include "mul_mat_q8_0_q8_1_tiled.hlsl"
