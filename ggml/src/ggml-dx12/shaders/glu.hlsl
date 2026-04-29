// glu.hlsl - Gated Linear Unit: supports SWIGLU, REGLU, GEGLU variants
// op0 = glu_op type (0=REGLU, 1=GEGLU, 2=SWIGLU, 3=SWIGLU_OAI, 4=GEGLU_ERF, 5=GEGLU_QUICK)
// op1 = swapped flag
// op2 = alpha (for SWIGLU_OAI)
// op3 = limit (for SWIGLU_OAI)

#include "ggml_common.hlsli"

float glu_activation(float x, uint glu_op, float alpha, float limit) {
    switch (glu_op) {
        case 0: return max(x, 0.0f);  // REGLU
        case 2: return x / (1.0f + exp(-x));  // SWIGLU
        case 3: { float cx = min(x, limit); return cx / (1.0f + exp(alpha * (-cx))); }  // SWIGLU_OAI
        case 5: return x / (1.0f + exp(-1.702f * x));  // GEGLU_QUICK
        case 1: {  // GEGLU
            float x3 = x * x * x;
            return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x3)));
        }
        case 4: {  // GEGLU_ERF
            float a = abs(x * 0.7071067811865f);
            float p = 1.0f / (1.0f + 0.3275911f * a);
            float e = p * (0.254829592f + p * (-0.284496736f + p * (1.421413741f + p * (-1.453152027f + p * 1.061405429f))));
            float erf_val = sign(x) * (1.0f - e * exp(-a * a));
            return 0.5f * x * (1.0f + erf_val);
        }
        default: return x;
    }
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    uint total = ne0 * ne1 * ne2 * ne3;
    if (idx >= total) return;

    // Inline index decomposition (avoids flat_to_4d function call overhead)
    uint i0 = idx % ne0;
    uint rem = idx / ne0;
    uint i1 = rem % ne1;
    rem = rem / ne1;
    uint i2 = rem % ne2;
    uint i3 = rem / ne2;

    uint glu_op = op0;
    uint swapped = op1;

    float gate_val, up_val;
    bool split_mode = (ne10 > 0);

    if (split_mode) {
        uint off0 = offset_4d(i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off1 = offset_4d(i0, i1, i2, i3, nb10, nb11, nb12, nb13, src1_offset);
        gate_val = asfloat(src0.Load(off0));
        up_val   = asfloat(src1.Load(off1));
    } else {
        uint nc = ne0;
        uint gate_i0 = swapped ? (i0 + nc) : i0;
        uint up_i0   = swapped ? i0 : (i0 + nc);
        uint off_gate = offset_4d(gate_i0, i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        uint off_up   = offset_4d(up_i0,   i1, i2, i3, nb00, nb01, nb02, nb03, src0_offset);
        gate_val = asfloat(src0.Load(off_gate));
        up_val   = asfloat(src0.Load(off_up));
    }

    float activated = glu_activation(gate_val, glu_op, asfloat(op2), asfloat(op3));
    float result = (glu_op == 3) ? activated * (up_val + 1.0f) : activated * up_val;

    uint off_d = offset_4d(i0, i1, i2, i3, nb0, nb1, nb2, nb3, dst_offset);
    dst.Store(off_d, asuint(result));
}
