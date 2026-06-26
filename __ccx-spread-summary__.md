# CCX-Spread Decode Affinity — Summary

## Problem
LLM **decode** re-streams the entire model weights (~2.2 GB for Phi-3 Q4_K_M)
per token — far larger than any cache — so it is **memory-bandwidth-bound**, not
compute-bound. On chiplet CPUs each CCX (core complex) has its own GMI/fabric
link to memory. Aggregate read bandwidth therefore scales with the **number of
distinct CCXs engaged**, not with L3 locality.

The Windows scheduler does not deterministically spread worker threads one per
CCX, and the legacy affinity calls are group-0 capped:
`SetProcessAffinityMask` / `SetThreadAffinityMask` only reach one processor
group (64 LPs). Only **`SetThreadGroupAffinity`** can cross processor groups.

## Solution
`ggml_b612_pin_ccx_spread()` in `ggml/src/ggml-cpu/ggml-cpu.c`, called at the
top of `ggml_graph_compute_thread`. It pins worker `ith` to a distinct CCX
(interleaving processor groups) so consecutive workers engage different fabric
links.

- **Opt-in:** set env `GGML_B612_CCX_SPREAD=1`.
- **No-op** when unset, on non-Windows, or pre-Win7 SDK. Zero impact otherwise
  (one branch-and-return per token).
- Builds the CCX map from `GetLogicalProcessorInformationEx(RelationCache)`
  L3 entries; uses primary SMT thread of each physical core.
- Prints once: `ggml: B612 CCX-spread affinity enabled (N CCXs)`.

## Results

### 96-core box (192 LP, 3 groups, 12 CCXs, single NUMA node)
Memory bandwidth (bwtest, 8 threads):

| placement     | GB/s |
|---------------|------|
| single-CCX    | ~30  |
| floating      | ~40  |
| **spread**    | ~145 |

Model decode (Phi-3 Q4_K_M, -t 8, same binary/session):

| mode        | us/op  | tps   |
|-------------|--------|-------|
| floating    | 317.2  | 16.36 |
| **spread**  | 161.9  | 23.71 |

→ **1.96× faster decode matmul, +45% tps**, deterministic.

### UMA APU (32 LP, 1 group, 2 CCXs, single NUMA node)
Memory bandwidth (bwtest, 8 threads):

| placement     | GB/s |
|---------------|------|
| single-CCX    | ~60  |
| floating      | ~65  |
| **spread**    | ~110 |

→ Engaging both CCXs ≈ **1.85×** the bandwidth of one. The existing UMA
"straddle" heuristic is simply the **2-CCX special case** of CCX-spread.
Model A/B confirmed: `GGML_B612_CCX_SPREAD=1` ≈ straddle default ≫ floating.

## Key takeaways
- Decode bandwidth scales with **# of CCXs / GMI links engaged**, not L3 locality.
- Prefill is compute-bound → CCX-insensitive (unchanged by this hook).
- **One unified mechanism** (`GGML_B612_CCX_SPREAD=1`) now covers both the
  2-CCX UMA APU and the 12-CCX 96-core box — no per-machine affinity juggling.
- Floating placement is unreliable even with only 2 CCXs; a deterministic pin
  still wins (~1.6× on UMA).

## Topology reference (how to read it)
- Dump with `topo.exe`; measure with `bwtest.exe [nthreads] [buf_GiB] [iters]`.
  Both are Windows-only utilities built alongside `examples/wxemem`.
- CCX = the set of LPs sharing one **L3** cache. Single NUMA node means memory
  is hardware-interleaved, so `/NODE` and `VirtualAllocExNuma` give no locality.
- Consecutive SMT pairing observed on both boxes: physical core N = LPs {2N, 2N+1};
  the hook pins the primary thread (even LP) of each core.

## Possible future cleanup
Retire the bespoke straddle heuristic in `common/b612-cpu.h` in favor of the
unified hook (optional; not yet done).
