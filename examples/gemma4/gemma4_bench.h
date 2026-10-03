// gemma4_bench: llama-bench-style throughput numbers for the Gemma-4
// hand-built qquant decode path and, for comparison, the upstream
// llama_decode forward.
//
// Mirrors tools/llama-bench/llama-bench.cpp's pp/tg protocol:
//
//   pp{N}  -- prefill of N tokens via a single batched forward call.
//             t/s = N / prefill_elapsed_seconds.
//   tg{N}  -- N single-token decode steps after a 1-token warmup.
//             t/s = N / decode_elapsed_seconds.
//
// The bench feeds random token IDs (parse_special=false content is
// irrelevant -- llama-bench does the same). Each test is run with a
// warmup pass plus `reps` measured passes; output is mean +- stddev.
//
// CLI flags (wired in Gemma4.cpp):
//   --gemma4-bench                 enable
//   --bench-pp N                   prefill size  (default 64)
//   --bench-tg N                   gen   size    (default 64)
//   --bench-reps N                 measured reps (default 3; warmup is implicit)
//   --bench-backend qquant|upstream|both   default both
#pragma once

#include "llama.h"
#include <string>

namespace gemma4 {

struct Weights;

struct BenchParams {
    int  pp_n             = 64;
    int  tg_n             = 64;
    int  reps             = 3;
    int  n_threads        = 1;
    bool include_qquant   = true;
    bool include_upstream = true;
    unsigned seed         = 1234;
};

// Runs the bench and prints a llama-bench-formatted table to stdout.
// Returns false (and sets error) only on a hard failure (e.g. dequant
// failed); individual rep failures are reported in-line as "FAIL" rows
// but do not abort the run.
bool run_bench(const llama_model * model, const Weights & w,
               const BenchParams & p, std::string & error);

}  // namespace gemma4
