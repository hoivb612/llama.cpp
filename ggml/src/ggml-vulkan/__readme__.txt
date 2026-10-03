Vulkan has the building blocks:

┌────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ DX12 Feature                                   │ Vulkan Equivalent                                                      │
├────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Reserved Resources (tiled)                     │ Sparse Buffers (VK_BUFFER_CREATE_SPARSE_BINDING_BIT)                   │
├────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ UpdateTileMappings (commit/decommit)           │ vkQueueBindSparse — bind/unbind 64KB pages                             │
├────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ OpenExistingHeapFromAddress (zero-copy mmap)   │ VK_EXT_external_memory_host — already used in the Vulkan backend       │
├────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ ID3D12Heap (explicit heap)                     │ vkAllocateMemory with explicit memory type                             │
├────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Memory budget query                            │ VK_EXT_memory_budget — already queried                                 │
└────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

What makes it harder than DX12:

 1. Sparse binding requires a dedicated queue — vkQueueBindSparse needs a queue with sparseBinding capability. You'd need to 
synchronize this with the compute queue (semaphores), unlike DX12 where UpdateTileMappings is just a CPU call on the command queue.
 2. Not all GPUs support sparse buffers — you'd need to check sparseBinding and sparseResidencyBuffer device features. AMD RDNA 
supports it, Intel iGPUs often don't. The current Vulkan backend doesn't even query these features.
 3. No equivalent to OEHA — VK_EXT_external_memory_host imports host pointers but creates a regular buffer, not a sparse one. You 
can't combine sparse binding with host pointer import. So zero-copy reload isn't possible — you'd have to memcpy from mmap → staging
 → vkCmdCopyBuffer.
 4. UMA path is different — on UMA with Vulkan, HOST_VISIBLE | DEVICE_LOCAL memory is already unified. Sparse binding adds overhead 
for something the OS pager could handle transparently.

My recommendation: It's doable and would be valuable on dGPU (discrete GPU with limited VRAM), where you want to fit a 12GB model
into 8GB VRAM. On UMA it's less compelling since the OS pager already works. The biggest engineering cost is the sparse binding
queue synchronization — DX12 made this much simpler with its tile mapping API.

● The 8060S UMA passes every check:

   - ✅ sparseBinding + sparseResidencyBuffer — the two required features
   - ✅ 64KB page granularity — identical to DX12 tiles
   - ✅ Sparse queue on all 3 compute-capable families (0, 1, 2)
   - ✅ Bind + unbind both work — confirmed commit/decommit cycle
   - ✅ 11 GB DEVICE_LOCAL heap with HOST_VISIBLE types (Type 2, 6, 10, 14) — true UMA
   - ✅ All three nice-to-have extensions present

  The Vulkan sparse path on this hardware would be a direct 1:1 port of your DX12 windowing. Same page size, same bind/unbind
  semantics, and on UMA you can even allocate from DEVICE_LOCAL | HOST_VISIBLE memory types for zero-copy behavior without needing
  OEHA.

Vulkan Sparse Buffer Layer Windowing — Committed & Pushed (5fd361f01 on hv/b612_mr)

What was implemented:

 1. Feature detection — sparse_binding + sparse_residency_buffer queried at device init
 2. Sparse buffer struct — is_sparse, backing memory pool, page tracking (64KB pages), commit/decommit methods
 3. ggml_vk_create_sparse_buffer() — creates buffer with SPARSE_BINDING|SPARSE_RESIDENCY, allocates backing pool (budget + 128 MiB),
 prefers DEVICE_LOCAL|HOST_VISIBLE on UMA
 4. commit_range/decommit_range — page allocation with refcounting, batched vkQueueBindSparse
 5. vk_tensor_decommit callback — registered via the backend API
 6. Auto-commit in set_tensor — UMA direct-map path (no staging); dGPU falls through to existing staging
 7. Budget routing — alloc_buffer checks weight_budget_hint and routes to sparse when appropriate

Gemma-4 results:
llama_memory_breakdown_print: | memory breakdown [MiB]                  | total    free    self   model   context   compute         
unaccounted |                                                                                                                       
llama_memory_breakdown_print: |   - Vulkan0 (Radeon(TM) 8060S Graphics) | 16486 = 13454 + (1959 =  1407 +      36 +     515) +        1072 |                                                                                                                            
llama_memory_breakdown_print: |   - Host                                |                  1787 =  1756 +       0 +      31                |                                                                                                                            
--- Budget 1024 on Gemma-4:                                                                                                         
llama_perf_context_print:        load time =    2627.45 ms                                                                          
llama_perf_context_print: prompt eval time =    7200.00 ms /  9999 tokens (    0.72 ms per token,  1388.75 tokens per second)       
llama_perf_context_print:        eval time =   25252.63 ms /   932 runs   (   27.10 ms per token,    36.91 tokens per second)       
llama_perf_context_print:       total time =   36266.08 ms / 10931 tokens                                                           
llama_perf_context_print:    graphs reused =        914                                                                             
layer_window: 968 passes, 4836 loads (155448.0 MiB), 4836 evictions (155442.3 MiB)                                                  
process_memory: peak working_set=7115.4 MiB, current working_set=7111.2 MiB, private=4258.6 MiB                                     
~llama_context:    Vulkan0 compute buffer size is 515.5000 MiB, matches expectation of 515.5000 MiB                                 
~llama_context: Vulkan_Host compute buffer size is  31.5235 MiB, matches expectation of  31.5235 MiB                                
--- No Budget:                                                                                                                      
llama_perf_context_print:        load time =    2801.73 ms                                                                          
llama_perf_context_print: prompt eval time =    5797.43 ms /  9999 tokens (    0.58 ms per token,  1724.73 tokens per second)       
llama_perf_context_print:        eval time =   10994.68 ms /   932 runs   (   11.80 ms per token,    84.77 tokens per second)       
llama_perf_context_print:       total time =   21049.67 ms / 10931 tokens                                                           
llama_perf_context_print:    graphs reused =        914                                                                             
process_memory: peak working_set=6502.5 MiB, current working_set=4151.7 MiB, private=4241.3 MiB                                     
~llama_context:    Vulkan0 compute buffer size is 515.5000 MiB, matches expectation of 515.5000 MiB                                 
~llama_context: Vulkan_Host compute buffer size is  31.5235 MiB, matches expectation of  31.5235 MiB   

Key observations from your results:

┌─────────┬─────────────┬───────────┐
│ Metric  │ Budget 1024 │ No Budget │
├─────────┼─────────────┼───────────┤
│ Prefill │ 1389 t/s    │ 1725 t/s  │
├─────────┼─────────────┼───────────┤
│ Decode  │ 36.9 t/s    │ 84.8 t/s  │
├─────────┼─────────────┼───────────┤
│ Peak WS │ 7115 MiB    │ 6503 MiB  │
└─────────┴─────────────┴───────────┘

 - 4836 loads + 4836 evictions — sparse commit/decommit cycling correctly
 - The unaccounted: 1072 MiB is the sparse backing pool (budget 1024 + 128 MiB headroom ≈ 1152 MiB, minus some unused pages)
 - Decode is ~2.3× slower with windowing, which matches the DX12 pattern — the cost of constant page rebinding + data reload
 - All 18 prompts produced correct output (no garbling!)

❯ For Phi-3:                                                                                                                          
  llama_memory_breakdown_print: | memory breakdown [MiB]                  | total    free    self   model   context   compute         
  unaccounted |                                                                                                                       
  llama_memory_breakdown_print: |   - Vulkan0 (Radeon(TM) 8060S Graphics) | 16486 = 12493 + (3075 =  2228 +     768 +      78) +           917 |                                                                                                                            
  llama_memory_breakdown_print: |   - Host                                |                    68 =    52 +       0 +      16                  |                                                                                                                            
  --- Budget 1024:                                                                                                                    
  layer_window: budget = 1024 MiB, initial resident = 15 of 32 layers (1011.7 MiB)                                                    
  layer_window: non-layer (always resident) = 77.1 MiB                                                                                
  layer_window: total initial memory = 1088.8 MiB (non-layer + 15 layers)                                                             
  layer_window: 17 layers deferred (will stream on demand during compute)                                                             
  ...                                                                                                                                 
  layer_window: released 2151.8 MiB of mmap pages after initial load                                                                  
  load_tensors: keeping mmaps alive (host_ptr=0, layer_window=1)                                                                      
  ...                                                                                                                                 
  llama_perf_context_print:        load time =    2002.61 ms                                                                          
  llama_perf_context_print: prompt eval time =   11470.24 ms / 10023 tokens (    1.14 ms per token,   873.83 tokens per second)       
  llama_perf_context_print:        eval time =   89239.16 ms /  1070 runs   (   83.40 ms per token,    11.99 tokens per second)       
  llama_perf_context_print:       total time =  102186.96 ms / 11093 tokens                                                           
  llama_perf_context_print:    graphs reused =       1052                                                                             
  layer_window: 1106 passes, 19543 loads (1314544.1 MiB), 19543 evictions (1314550.3 MiB)                                             
  process_memory: peak working_set=5576.1 MiB, current working_set=5576.1 MiB, private=3315.0 MiB                                     
  ~llama_context:    Vulkan0 compute buffer size is  78.6367 MiB, matches expectation of  78.6367 MiB                                 
  ~llama_context: Vulkan_Host compute buffer size is  16.0137 MiB, matches expectation of  16.0137 MiB                                
  --- No Budget:                                                                                                                      
  llama_perf_context_print:        load time =   58078.78 ms                                                                          
  llama_perf_context_print: prompt eval time =    8585.12 ms / 10023 tokens (    0.86 ms per token,  1167.49 tokens per second)       
  llama_perf_context_print:        eval time =   18090.32 ms /  1070 runs   (   16.91 ms per token,    59.15 tokens per second)       
  llama_perf_context_print:       total time =   85259.39 ms / 11093 tokens                                                           
  llama_perf_context_print:    graphs reused =       1052                                                                             
  process_memory: peak working_set=4696.9 MiB, current working_set=3290.7 MiB, private=3306.8 MiB                                     
  ~llama_context:    Vulkan0 compute buffer size is  78.6367 MiB, matches expectation of  78.6367 MiB                                 
  ~llama_context: Vulkan_Host compute buffer size is  16.0137 MiB, matches expectation of  16.0137 MiB     

● Phi-3 results look solid. Interesting comparison:

  ┌────────────┬─────────────┬───────────┐
  │ Metric     │ Budget 1024 │ No Budget │
  ├────────────┼─────────────┼───────────┤
  │ Prefill    │ 874 t/s     │ 1167 t/s  │
  ├────────────┼─────────────┼───────────┤
  │ Decode     │ 12.0 t/s    │ 59.2 t/s  │
  ├────────────┼─────────────┼───────────┤
  │ Load time  │ 2.0s        │ 58.1s     │
  ├────────────┼─────────────┼───────────┤
  │ Total time │ 102s        │ 85s       │
  ├────────────┼─────────────┼───────────┤
  │ Peak WS    │ 5576 MiB    │ 4697 MiB  │
  └────────────┴─────────────┴───────────┘

  Notable observations:

   - 19,543 loads/evictions — Phi-3 has 32 layers with only 15 fitting in budget, so 17 layers stream per pass. That's a LOT of page 
  rebinding (1.3 TB total data moved!)
   - Decode 5× slower with windowing — worse ratio than Gemma-4 (2.3×). Makes sense: Phi-3's layers are ~67 MiB each vs Gemma-4's ~32 
  MiB, so each evict/reload cycle moves more data
   - Load time 29× faster with windowing (2s vs 58s) — only 15 layers loaded upfront instead of 32
   - Total wall time is only 20% slower (102s vs 85s) despite the massive streaming — the fast load time offsets much of the decode 
  penalty

  The windowing implementation is working correctly across both models. The performance characteristics match expectations — larger
  models with more layers-per-budget-miss pay more per token but gain on initial load.

The core issue: on UMA, sparse windowing doesn't save memory.

On the 8060S, GPU VRAM is system RAM. So with budget mode you're paying for:

 - 917 MiB sparse backing pool (the "unaccounted")  
 - mmap file pages faulted into RAM during streaming

That's more than just loading the full 2228 MiB model normally, because you have the backing pool AND the mmap pages simultaneously
resident. The numbers confirm this:

┌────────────┬──────────────┬───────────┐
│            │ Budget 1024  │ No Budget │
├────────────┼──────────────┼───────────┤
│ Private    │ 3315 MiB     │ 3307 MiB  │
├────────────┼──────────────┼───────────┤
│ Peak WS    │ 5576 MiB     │ 4697 MiB  │
├────────────┼──────────────┼───────────┤
│ Current WS │ 5576 MiB     │ 3291 MiB  │
└────────────┴──────────────┴───────────┘

Layer windowing's real value is on dGPU where:

 - VRAM is separate and limited (e.g., 4GB card)
 - System RAM is plentiful for mmap streaming
 - You trade CPU→GPU bandwidth to fit a model that wouldn't fit in VRAM at all

On UMA where it's all one pool, you get all the overhead (page rebinding, streaming) with no memory savings. The feature only helps
UMA if the model is so large it wouldn't fit in the GPU-addressable heap at all — but both Gemma-4 (1.4 GB) and Phi-3 (2.2 GB) fit
easily in the 8060S's 16 GB.

Dedicated GPU memory: 5.2 GB in BOTH cases. Zero VRAM savings.

And the budget mode made things worse:

┌────────────────┬───────────────┬──────────────────────┐
│                │ No Budget     │ Budget 1024          │
├────────────────┼───────────────┼──────────────────────┤
│ Dedicated VRAM │ 5.2 GB        │ 5.2 GB (same!)       │
├────────────────┼───────────────┼──────────────────────┤
│ System RAM     │ 27.6 GB (44%) │ 29.9 GB (47%)        │
├────────────────┼───────────────┼──────────────────────┤
│ Private        │ 3255 MiB      │ 5613 MiB (+2.3 GB)   │
├────────────────┼───────────────┼──────────────────────┤
│ Decode         │ 115.7 t/s     │ 5.6 t/s (20× slower) │
├────────────────┼───────────────┼──────────────────────┤
│ GPU util       │ 100%          │ 23%                  │
└────────────────┴───────────────┴──────────────────────┘

Root cause: The AMD Vulkan driver appears to reserve the full sparse buffer virtual size in VRAM regardless of which pages are
committed. Unlike DX12's CreateReservedResource — which explicitly separates virtual address space from committed memory — Vulkan's
sparse binding seems to just be a logical mapping layer over fully-allocated VRAM on AMD.

Budget mode added ~2.3 GB of system RAM for mmap streaming while saving zero VRAM — all cost, no benefit.

Bottom line: DX12 reserved resources remain the only path to actual VRAM reduction on AMD. The Vulkan sparse implementation is
functionally correct (output is right, bind/unbind works) but doesn't achieve the memory goal. It might behave differently on
NVIDIA/Intel drivers, but for AMD RDNA 3, DX12 is the answer.