// phi3_bench: llama-bench-style throughput numbers for the Phi-3
// hand-built qquant decode path and, for comparison, an upstream
// llama_decode forward.
//
// Mirrors examples/gemma4/gemma4_bench.{h,cpp} so the two binaries
// produce visually identical tables.
//
// CLI flags (wired in Phi3.cpp):
//   --phi3-bench                  enable
//   --bench-pp N                  prefill size  (default 64)
//   --bench-tg N                  gen   size    (default 64)
//   --bench-reps N                measured reps (default 3)
//   --bench-backend qquant|upstream|both   default both
//
// The qquant rows go through phi3_run_qquant_decode (the same per-token
// driver used by --phi3-fused-qquant-debug). The upstream rows drive a
// fresh llama_context with llama_decode directly, matching what
// `llama-bench.exe` does. fuse_rmsnorm_quant / attn_parallel default to
// OFF here; flip them on with the existing
// --phi3-fused-qquant-rmsnorm-fuse / --phi3-fused-qquant-attn-parallel
// flags (they're read by run_bench).
#pragma once

#include "llama.h"
#include <string>

namespace phi3_bench {

struct Params {
    int  pp_n             = 64;
    int  tg_n             = 64;
    int  reps             = 3;
    int  n_threads        = 1;
    bool include_qquant   = true;
    bool include_upstream = true;
    bool fuse_rmsnorm_quant = false;  // mirrors --phi3-fused-qquant-rmsnorm-fuse
    bool attn_parallel    = false;    // mirrors --phi3-fused-qquant-attn-parallel
    unsigned seed         = 1234;
};

bool run_bench(const llama_model * model, const Params & p, std::string & error);

}  // namespace phi3_bench
