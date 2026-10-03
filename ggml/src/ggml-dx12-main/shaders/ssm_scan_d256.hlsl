// ssm_scan_d256.hlsl - Mamba2 selective scan, d_state=256 variant
// (used by Falcon-H1 and similar models).
// See ssm_scan_body.hlsli for full input/output / op_params documentation.
#define D_STATE 256
#include "ssm_scan_body.hlsli"
