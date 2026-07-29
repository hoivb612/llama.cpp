// ssm_conv_silu.hlsl - SSM_CONV fused with the trailing SiLU activation.
// Same I/O as ssm_conv.hlsl; the only difference is `sum = sum * sigmoid(sum)`
// before the store.  See ssm_conv.hlsl for the input/output layout notes.
#define APPLY_SILU 1
#include "ssm_conv_impl.hlsli"
