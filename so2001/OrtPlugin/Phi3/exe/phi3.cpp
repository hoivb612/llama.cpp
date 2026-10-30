// phi3.cpp -- Phi-3-mini int4 driver. Shared scaffolding lives in harness_common.h.

// Razzle builds at /W4 /WX. Silence benign warnings from the ported driver and
// the vendored ONNX Runtime / tokenizers headers so the verbatim sources build
// unmodified: C4005 (WIN32_LEAN_AND_MEAN redefined vs the build command line),
// C4100 (unreferenced params in tokenizers_cpp.h), C4189 (unused local), and
// C4505 (unreferenced static helpers in harness_common.h).
#pragma warning(disable: 4005 4100 4189 4505)

#include "harness_common.h"

// OGA-style per-prompt timing summary (matches Phi/src/cpf_v2.cpp output):
//   - TTFT <prefill_ms>ms - <prefill_tps> tps (<prompt_tokens>)
//   - Generation time <gen_ms>ms - <gen_tps> tps (<gen_tokens>)
// TTFT is the prefill Run (full prompt -> first token); generation covers the
// decode Run()s only, so the two throughputs are directly comparable to the OGA
// driver's prompt-processing and token-generation rates.
static void PrintOgaStyleTiming(const TimingStats& ts) {
    const double ttft_ms = ts.t_prefill_us / 1000.0;
    const double ttft_s = ts.t_prefill_us / 1000000.0;
    const double prefill_tps = ttft_s > 0.0 ? ts.prompt_tokens / ttft_s : 0.0;

    const int64_t gen_us = ts.t_decode_first_us + ts.t_decode_rest_us;
    const int64_t gen_tokens = (ts.t_decode_first_us > 0 ? 1 : 0) + ts.n_decode_rest;
    const double gen_ms = gen_us / 1000.0;
    const double gen_s = gen_us / 1000000.0;
    const double gen_tps = gen_s > 0.0 ? gen_tokens / gen_s : 0.0;

    printf("  - TTFT   %9.2fms - %5.2f tps (%lld)\n",
           ttft_ms, prefill_tps, static_cast<long long>(ts.prompt_tokens));
    printf("  - Generation time %9.2fms - %5.2f tps (%lld)\n",
           gen_ms, gen_tps, static_cast<long long>(gen_tokens));
}

// Runs generation for a SINGLE prompt against an already-loaded session +
// tokenizer (both owned by main so the model is loaded only once, even when a
// batch of custom prompts is run). In custom_mode the per-prompt banner and
// OGA-style TTFT/generation timing are printed here so batch logs line up with
// the OGA reference driver.
std::string generate_text(
    const Args& args,
    GenerationConfig& cfg,
    Ort::Session& infer_session,
    TokenizerWrap& tokenizer,
    const std::string& prompt_text,
    int64_t t_session_init_us,
    int64_t t_tokenizer_us,
    bool custom_mode = false,
    size_t prompt_index = 0,
    size_t prompt_total = 1,
    const std::string& display_label = "") {

    TimingStats ts;
    const int64_t t_gen_start = timer_us();
    ts.t_session_init_us = t_session_init_us;
    ts.t_tokenizer_us = t_tokenizer_us;

    const int64_t T = cfg.total_sequence;
    const int64_t W = cfg.window;
    const int64_t C = cfg.context;
    if (W <= 0 || T <= 0 || C <= 0) throw std::invalid_argument("bad dims");
    if (W > T) throw std::invalid_argument("window > total_sequence");
    if (C < W) throw std::invalid_argument("context must be >= window");

    // Per-prompt banner matching the OGA reference driver's format
    // ("[idx/total][raw message line]") for side-by-side comparison. The banner
    // shows only the user message (display_label), not the full chat template.
    if (custom_mode) {
        printf("\n> Running with custom prompt => [%zu/%zu][%s]\n",
               prompt_index + 1, prompt_total, display_label.c_str());
    }

    // Prepare text input_ids via tokenizer
    std::vector<int64_t> prompt_ids = tokenizer.encode_ids(prompt_text);
    ts.prompt_tokens = static_cast<int64_t>(prompt_ids.size());
    if (!custom_mode) {
        printf("Encoding prompt: [%s]", prompt_text.c_str());
    }

    int64_t t_io_start = timer_us();
    Ort::AllocatorWithDefaultOptions alloc;
    auto io = discover_model_io(infer_session, alloc);
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Identify canonical inputs/outputs by substring
    auto idx_input_ids       = io.idx_input_matching("input_ids");
    auto idx_attention_mask  = io.idx_input_matching("attention_mask");
    auto idx_position_ids    = io.idx_input_matching("position_ids");
    auto idx_tree_attention  = io.idx_input_matching("tree_attention");
    if (!idx_input_ids || !idx_attention_mask) {
        throw std::runtime_error("Model inputs must include input_ids, attention_mask");
    }
    // 'position_ids' is optional: the cpu-int4-rtn acc-level-4 genai Phi-3 export
    // computes rotary positions internally and declares NO position_ids input,
    // whereas the DirectML int4-awq-block-128 drop does. Only build/bind it when
    // the model actually declares it.
    const bool has_position_ids = idx_position_ids.has_value();
    if (!has_position_ids && !custom_mode) {
        std::cerr << "[info] Model has no 'position_ids' input; skipping position binding.\n";
    }
    // 'tree_attention' is optional: some Phi-3 exports (e.g. the DirectML
    // int4-awq-block-128 drop) omit it. Only build and bind the upper-triangle
    // mask when the model actually declares such an input.
    const bool has_tree_attention = idx_tree_attention.has_value();
    if (has_tree_attention && (!args.save_kv.empty() || !args.load_kv.empty())) {
        throw std::runtime_error(
            "--save-kv/--load-kv are only supported on the non-DML grow-and-swap path "
            "(this model uses the fixed-window tree_attention path)");
    }
    if (!has_tree_attention && !custom_mode) {
        std::cerr << "[info] Model has no 'tree_attention' input; skipping tree-mask binding.\n";
    }
    // Outputs: logits and KV "present"
    auto idx_logits = io.idx_output_matching("logits");
    if (!idx_logits) {
        throw std::runtime_error("Model outputs must include logits");
    }
    // KV cache inputs (all that contain "past_key_values")
    std::vector<size_t> kv_in_indices;
    for (size_t i = 0; i < io.input_names.size(); ++i) {
        if (io.input_names[i].find("past_key_values") != std::string::npos) {
            kv_in_indices.push_back(i);
        }
    }
    // Map from output present name -> index
    std::unordered_map<std::string, size_t> present_out_index;
    for (size_t i = 0; i < io.output_names.size(); ++i) {
        if (io.output_names[i].find("present") != std::string::npos) {
            present_out_index[io.output_names[i]] = i;
        }
    }

    // =====================================================================
    // Dynamic prefill+decode path (standard HF-style contract).
    // Used when the model does NOT declare a 'tree_attention' input
    // (e.g. the DirectML int4-awq-block-128 Phi-3 drop). Models that DO
    // declare tree_attention fall through to the fixed-window path below.
    // Shape contract:
    //   input_ids       [1, seq_len]                seq_len = prompt_size (prefill) or 1 (decode)
    //   position_ids    [1, seq_len]                must match input_ids' dim 1
    //   attention_mask  [1, total_len]              total_len = past + new, grows each step
    //   past_key_values.N.{key,value}
    //                   [1, n_heads, past_len, hd]  past_len = 0 (prefill) then grows by seq_len
    // =====================================================================
    if (!has_tree_attention) {
        const int64_t prompt_size_dyn = static_cast<int64_t>(prompt_ids.size());
        if (prompt_size_dyn <= 0) {
            throw std::runtime_error("Empty prompt: tokenizer produced 0 ids");
        }
        const int64_t max_new_dyn = std::max<int64_t>(1, cfg.max_gen_tokens);

        // Per-iteration backing buffers (mutated each step; Ort::Value wrappers
        // are recreated each iteration AFTER the mutation, so pointers are current).
        std::vector<int64_t> ids_buf;
        std::vector<int64_t> pos_buf;
        std::vector<int64_t> mask_buf;
        mask_buf.reserve(static_cast<size_t>(prompt_size_dyn + max_new_dyn));

        // Detect the KV cache element type from the model so the empty past seed
        // (and the DML static buffers below) use the right dtype. The DirectML
        // AWQ Phi-3 drop is fp16; the cpu-int4-rtn acc-level-4 genai export is fp32.
        ONNXTensorElementDataType kv_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        if (!kv_in_indices.empty()) {
            try {
                auto kv_type_info = infer_session.GetInputTypeInfo(kv_in_indices.front());
                kv_elem_type = kv_type_info.GetTensorTypeAndShapeInfo().GetElementType();
            } catch (...) {}
        }
        const bool kv_is_fp16 = (kv_elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
        if (!kv_is_fp16 && (!args.save_kv.empty() || !args.load_kv.empty())) {
            throw std::runtime_error(
                "--save-kv/--load-kv currently support only an fp16 KV cache; this "
                "model's KV cache is not fp16.");
        }

        // Initial past KV: empty [1, n_heads, 0, head_dim] tensors of the model's
        // KV dtype. Backing buffers carry one dummy element so .data() is non-null
        // (zero-length CreateTensor wants a valid pointer even with byte_count=0).
        std::vector<std::vector<uint16_t>> empty_kv_seed(
            kv_in_indices.size(), std::vector<uint16_t>(1, 0));
        std::vector<Ort::Value> kv_pasts;
        kv_pasts.reserve(kv_in_indices.size());
        for (size_t k = 0; k < kv_in_indices.size(); ++k) {
            std::array<int64_t, 4> sh{1, cfg.n_heads, 0, cfg.head_dim};
            kv_pasts.emplace_back(Ort::Value::CreateTensor(
                mem, empty_kv_seed[k].data(), /*byte_count=*/0,
                sh.data(), sh.size(), kv_elem_type));
        }

        auto make_i64_dyn = [&](std::vector<int64_t>& b, std::initializer_list<int64_t> shape) {
            return Ort::Value::CreateTensor<int64_t>(mem, b.data(), b.size(),
                                                     std::data(shape), shape.size());
        };

        // ===== One-time IoBinding setup =====
        // Without IoBinding, ORT auto-stages every GPU-produced output back to CPU
        // and every CPU-resident input back to GPU between Run() calls, causing
        // ~14 MB/token of Default->CPU and ~13 MB/token of CPU->Default staging
        // on this model (32 layers x 2 KV growing per step). With IoBinding we
        // pre-declare where each I/O lives so the EP keeps KV cache on its own
        // device across iterations -- only the logits tensor (which we read on
        // CPU for argmax) is materialized on CPU.
        Ort::MemoryInfo cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto out_mem_infos = infer_session.GetMemoryInfoForOutputs();
        Ort::IoBinding binding(infer_session);

        // ===== KV-cache strategy depends on the EP's GroupQueryAttention kernel =====
        // The DML (and CUDA) GQA kernels do an *in-place* KV update: they do NOT read
        // past_key/past_value as separate tensors. They append the new K/V into the
        // *present* buffer at offset `seqlens_k` and require that buffer to already
        // hold the past -- i.e. past_key_values.N.* and present.N.* must ALIAS the
        // same, persistent, pre-allocated max-length device allocation. The CPU GQA
        // kernel instead tolerates the dynamic grow-and-swap of separate past/present
        // buffers used by the default path below (which is why --ep cpu and the Vulkan
        // plugin, whose GQA falls back to CPU, both work with grow-and-swap).
        // So for DML we pre-allocate one device buffer per KV tensor and bind it to
        // BOTH past_key_values.N.* and present.N.* every step (onnxruntime-genai style).
        auto is_dml_ep = [](const std::string& e) {
            return e == "DML" || e == "dml" || e == "DmlExecutionProvider" ||
                   e == "DirectML" || e == "directml";
        };
        const bool is_dml = is_dml_ep(args.ep);
        if (is_dml && (!args.save_kv.empty() || !args.load_kv.empty())) {
            throw std::runtime_error(
                "--save-kv/--load-kv are not supported with the DML EP (static in-place KV path); "
                "use --ep cpu or --ep VulkanEP");
        }

        // Derive KV head count / head dim from the model (fall back to cfg defaults).
        int64_t kv_heads = cfg.n_heads, kv_head_dim = cfg.head_dim;
        if (!kv_in_indices.empty()) {
            try {
                auto sh = infer_session.GetInputTypeInfo(kv_in_indices.front())
                              .GetTensorTypeAndShapeInfo().GetShape();
                if (sh.size() == 4) {
                    if (sh[1] > 0) kv_heads = sh[1];
                    if (sh[3] > 0) kv_head_dim = sh[3];
                }
            } catch (...) {}
        }

        // present.N.* output name corresponding to each past_key_values.N.* input.
        std::vector<std::string> present_names(kv_in_indices.size());
        for (size_t k = 0; k < kv_in_indices.size(); ++k) {
            std::string pn = io.input_names[kv_in_indices[k]];
            if (auto p = pn.find("past_key_values"); p != std::string::npos)
                pn.replace(p, std::string("past_key_values").size(), "present");
            present_names[k] = pn;
        }

        // DML: pre-allocate the shared past==present device buffers, once.
        // Capacity must cover the largest total sequence length we will reach.
        const int64_t kv_capacity = prompt_size_dyn + max_new_dyn;
        std::vector<Ort::Value> kv_shared;
        std::optional<Ort::Allocator> dml_kv_allocator;
        if (is_dml && !kv_in_indices.empty()) {
            const size_t any_present_out = present_out_index.begin()->second;
            dml_kv_allocator.emplace(infer_session, out_mem_infos[any_present_out]);
            const std::array<int64_t, 4> ksh{1, kv_heads, kv_capacity, kv_head_dim};
            kv_shared.reserve(kv_in_indices.size());
            for (size_t k = 0; k < kv_in_indices.size(); ++k) {
                kv_shared.emplace_back(Ort::Value::CreateTensor(
                    *dml_kv_allocator, ksh.data(), ksh.size(),
                    kv_elem_type));
            }
            printf("[DML] static KV cache: %zu buffers x [1,%lld,%lld,%lld] elem_type=%d "
                   "(past==present in-place)\n",
                   kv_shared.size(), static_cast<long long>(kv_heads),
                   static_cast<long long>(kv_capacity),
                   static_cast<long long>(kv_head_dim), static_cast<int>(kv_elem_type));
        }

        // IO discovery + IoBinding setup is done; everything from here is per-step.
        ts.t_io_setup_us = timer_us() - t_io_start;

        std::vector<int64_t> generated;
        generated.reserve(static_cast<size_t>(max_new_dyn));
        int64_t cur_position = 0;   // absolute position of the next NEW token
        int64_t next_token = -1;
        bool is_prefill = true;

        if (!custom_mode) {
            printf("=== Begin inference loop (dynamic) ... prompt=%lld max_new=%lld\n",
                   static_cast<long long>(prompt_size_dyn),
                   static_cast<long long>(max_new_dyn));
        }

        int64_t n_valid = 0;   // count of valid (1) mask positions == growing total length

        // ===== Prompt-cache LOAD: skip prefill, resume decoding from cached KV =====
        // Storage for the loaded fp16 KV must outlive the first Run (kv_pasts[k]
        // aliases loaded_kv[k].data until the first decode step swaps in the
        // freshly produced present.N.*).
        std::vector<promptcache::LoadedTensor> loaded_kv;
        int64_t steps_to_run = max_new_dyn;
        if (!args.load_kv.empty()) {
            int64_t cached_len = 0, first_token = 0;
            promptcache::Load(args.load_kv, present_names, kv_heads, kv_head_dim,
                              first_token, cached_len, loaded_kv);
            for (size_t k = 0; k < kv_in_indices.size(); ++k) {
                kv_pasts[k] = Ort::Value::CreateTensor(
                    mem, loaded_kv[k].data.data(),
                    loaded_kv[k].data.size() * sizeof(uint16_t),
                    loaded_kv[k].dims.data(), loaded_kv[k].dims.size(),
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            }
            // Resume state: the first sampled token (from the cached prefill) is
            // already known; decode it at absolute position cached_len with the
            // loaded past covering [0, cached_len).
            is_prefill = false;
            cur_position = cached_len;
            n_valid = cached_len;
            next_token = first_token;
            generated.push_back(first_token);
            // The full run produced `first_token` during prefill (1 of max_new_dyn);
            // here it is pre-seeded, so run one fewer step to match the token count.
            steps_to_run = std::max<int64_t>(0, max_new_dyn - 1);
            if (first_token == cfg.eos_token_id) steps_to_run = 0;
            printf("[prompt-cache] Loaded KV from '%s': cached_len=%lld, first_token=%lld "
                   "(prefill skipped, --prompt ignored)\n",
                   args.load_kv.c_str(), static_cast<long long>(cached_len),
                   static_cast<long long>(first_token));
        }

        for (int64_t step = 0; step < steps_to_run; ++step) {
            // --- Build per-iteration buffers ---
            if (is_prefill) {
                ids_buf = prompt_ids;
                pos_buf.resize(static_cast<size_t>(prompt_size_dyn));
                for (int64_t i = 0; i < prompt_size_dyn; ++i) pos_buf[i] = i;
                cur_position = prompt_size_dyn;
                n_valid = prompt_size_dyn;
            } else {
                ids_buf.assign(1, next_token);
                pos_buf.assign(1, cur_position);
                cur_position += 1;
                n_valid += 1;
            }
            const int64_t seq_len = static_cast<int64_t>(ids_buf.size());

            // attention_mask layout:
            //   non-DML : grows to exactly n_valid ones (HF dynamic contract; the CPU
            //             and Vulkan-plugin GQA both run on the CPU GQA kernel, which
            //             tolerates a growing mask / changing total_sequence_length).
            //   DML     : FIXED length = kv_capacity, with 1s in [0, n_valid) and 0s
            //             after. total_sequence_length is a *required-constant CPU
            //             input* (input 6) that DML BAKES into the fused GQA graph and
            //             uses to size the present-KV output tensors. In this model
            //             total_sequence_length = attention_mask.shape[1]; a growing
            //             mask therefore forces DML to recompile the whole fused graph
            //             every decode step (~197 ms/step fixed cost). Pinning the mask
            //             length keeps total_sequence_length constant so the graph is
            //             compiled ONCE. The true valid length is still carried by
            //             seqlens_k = ReduceSum(mask)-1 = n_valid-1, which DML maps to
            //             MHA PastSequenceLengths for causal masking. The masked KV tail
            //             [n_valid, kv_capacity) is never attended (bounded by seqlens_k)
            //             and is zero on the D3D12 default heap, so no NaN leaks in.
            int64_t total_len;
            if (is_dml) {
                mask_buf.assign(static_cast<size_t>(kv_capacity), 0);
                std::fill_n(mask_buf.begin(), static_cast<size_t>(n_valid), int64_t{1});
                total_len = kv_capacity;
            } else {
                mask_buf.assign(static_cast<size_t>(n_valid), 1);
                total_len = n_valid;
            }

            // --- Wrap buffers as Ort::Values (must outlive the Run below) ---
            Ort::Value ids_val  = make_i64_dyn(ids_buf,  {1, seq_len});
            Ort::Value mask_val = make_i64_dyn(mask_buf, {1, total_len});
            std::optional<Ort::Value> pos_val;
            if (has_position_ids) pos_val.emplace(make_i64_dyn(pos_buf, {1, seq_len}));

            // --- Bind inputs + outputs ---
            binding.ClearBoundInputs();
            binding.ClearBoundOutputs();
            binding.BindInput(io.input_names[*idx_input_ids].c_str(),      ids_val);
            if (has_position_ids)
                binding.BindInput(io.input_names[*idx_position_ids].c_str(), *pos_val);
            binding.BindInput(io.input_names[*idx_attention_mask].c_str(), mask_val);
            for (size_t k = 0; k < kv_in_indices.size(); ++k) {
                const Ort::Value& past_val = is_dml ? kv_shared[k] : kv_pasts[k];
                binding.BindInput(io.input_names[kv_in_indices[k]].c_str(), past_val);
            }
            // logits -> CPU (argmax'd on host).
            binding.BindOutput(io.output_names[*idx_logits].c_str(), cpu_mem_info);
            // present.N.*:
            //   DML  -> bind to the SAME persistent device buffer as its past input so
            //           the GQA kernel updates the KV cache in place (correctness +
            //           keeps KV on-device across steps, killing the per-step bounce).
            //   other-> the EP's native output memory, fed back as past next iteration.
            //   SAVE step (prefill + --save-kv) -> bind present to CPU so the KV can be
            //           read back for serialization; decoding then continues from the
            //           CPU-resident past (exactly the path a later --load-kv run takes,
            //           so save-run and load-run token streams are identical by design).
            const bool save_this_step = is_prefill && !args.save_kv.empty();
            for (size_t k = 0; k < kv_in_indices.size(); ++k) {
                if (is_dml) {
                    binding.BindOutput(present_names[k].c_str(), kv_shared[k]);
                } else if (save_this_step) {
                    binding.BindOutput(present_names[k].c_str(), cpu_mem_info);
                } else {
                    const size_t oi = present_out_index.at(present_names[k]);
                    binding.BindOutput(present_names[k].c_str(), out_mem_infos[oi]);
                }
            }

            const int64_t t_run_start = timer_us();
            infer_session.Run(Ort::RunOptions{nullptr}, binding);
            const int64_t t_run_us = timer_us() - t_run_start;
            if (is_prefill) {
                ts.t_prefill_us = t_run_us;
            } else if (ts.t_decode_first_us == 0) {
                ts.t_decode_first_us = t_run_us;
            } else {
                ts.t_decode_rest_us += t_run_us;
                ++ts.n_decode_rest;
            }
            auto output = binding.GetOutputValues();

            // --- Argmax last step of logits (CPU-resident by binding) ---
            Ort::Value& logits_val = output[*idx_logits];
            auto linfo = logits_val.GetTensorTypeAndShapeInfo();
            auto lshape = linfo.GetShape();
            if (lshape.size() != 3) {
                throw std::runtime_error("Expected logits of rank 3 [B, steps, vocab]");
            }
            const int64_t steps = lshape[1];
            const int64_t vocab = lshape[2];
            switch (linfo.GetElementType()) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    next_token = argmax_last_step_f16(
                        logits_val.GetTensorData<uint16_t>(), steps, vocab);
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    next_token = argmax_last_step_logits_f32(
                        logits_val.GetTensorData<float>(), steps, vocab);
                    break;
                default:
                    throw std::runtime_error("Unsupported logits dtype for argmax");
            }
            if (next_token < 0 || next_token >= vocab) {
                std::ostringstream oss;
                oss << "Argmax produced invalid id " << next_token
                    << " for vocab=" << vocab;
                throw std::runtime_error(oss.str());
            }

            generated.push_back(next_token);

            // --- Prompt-cache SAVE: write KV + first token after prefill ---
            // `output[...]` present tensors are CPU-resident this step (save_this_step
            // bound them to CPU above), so they are directly serializable. Do this
            // BEFORE the std::move handoff below invalidates the present handles.
            if (save_this_step) {
                std::vector<Ort::Value*> present_vals(kv_in_indices.size());
                for (size_t k = 0; k < kv_in_indices.size(); ++k) {
                    present_vals[k] = &output[present_out_index.at(present_names[k])];
                }
                promptcache::Save(args.save_kv, present_names, present_vals,
                                  /*first_token=*/next_token, /*cached_len=*/prompt_size_dyn,
                                  kv_heads, kv_head_dim);
                printf("[prompt-cache] Saved KV to '%s': cached_len=%lld, first_token=%lld\n",
                       args.save_kv.c_str(), static_cast<long long>(prompt_size_dyn),
                       static_cast<long long>(next_token));
            }
#if DBG
            std::string s = tokenizer.decode_ids({next_token}, true);
            printf("%s", s.c_str());
            fflush(stdout);
#endif
            if (next_token == cfg.eos_token_id) break;

            // Custom-prompt (OGA-comparison) mode mirrors the reference driver's
            // stop rule: end the response as soon as a '}' is emitted (the chat
            // template asks for a single JSON object), keeping outputs and token
            // counts comparable. The single-token decode here is outside the Run
            // timing window, so it does not perturb the measured decode rate.
            if (custom_mode) {
                const std::string piece = tokenizer.decode_ids({next_token}, true);
                if (piece.find('}') != std::string::npos) break;
            }

            // --- KV cache handoff for next iteration ---
            // DML shares one persistent buffer for past==present (updated in place by
            // the GQA kernel), so there is nothing to move. The CPU/plugin path swaps
            // the freshly produced present.N.* device tensors in as next step's past.
            if (!is_dml) {
                for (size_t kin = 0; kin < kv_in_indices.size(); ++kin) {
                    auto it = present_out_index.find(present_names[kin]);
                    if (it == present_out_index.end()) {
                        throw std::runtime_error(
                            "Missing present output for input " + io.input_names[kv_in_indices[kin]]);
                    }
                    kv_pasts[kin] = std::move(output[it->second]);
                }
            }
            is_prefill = false;
        }

        ts.n_generated = static_cast<int64_t>(generated.size());
        ts.t_total_us = timer_us() - t_gen_start;
        std::string reply = tokenizer.decode_ids(generated, true);
        if (custom_mode) {
            PrintOgaStyleTiming(ts);
            printf("%s\n", reply.c_str());
        } else {
            printf("\nGenerated %zu tokens\n", generated.size());
            ts.Print(args.ep.empty() ? std::string("cpu") : args.ep);
        }
        return reply;
    }
    // -------- fixed-window path follows (used when tree_attention IS present) --------

    // Build initial inputs based on Python logic
    const int64_t prompt_size = static_cast<int64_t>(prompt_ids.size());
    if (prompt_size + cfg.max_gen_tokens > T) {
        throw std::runtime_error("ERROR: Longer total sequence is needed!");
    }

    // Build the first window input = left pad with BOS to 'window' if prompt shorter
    std::vector<int64_t> total_input; total_input.reserve(std::max<int64_t>(prompt_size, W) + cfg.max_gen_tokens);
    if (prompt_size < W) {
        int64_t pad = W - prompt_size;
        total_input.insert(total_input.end(), pad, cfg.bos_token_id);
        total_input.insert(total_input.end(), prompt_ids.begin(), prompt_ids.end());
    } else {
        // take only the first window for initial slice; later will slide over prompt
#if 1
        // provision only W tokens since that is the window size
        total_input.insert(total_input.end(), prompt_ids.begin(), prompt_ids.begin() + W);
#else
        // keep the entire prompt so we can slide across it
        total_input.assign(prompt_ids.begin(), prompt_ids.end());
#endif
    }

    // ---------- Persistent backing buffers ----------
    // 2.a Core inputs
    std::vector<int64_t> ids_buf(W);
    std::vector<int64_t> mask_buf(T);
    std::vector<int64_t> pos_buf(T);

    // 2.b Tree attention (upper triangle = -65504 in FP16) — only when model declares the input.
    std::vector<uint16_t> tree_buf;
    if (has_tree_attention) {
        tree_buf.assign(static_cast<size_t>(T * T), 0);
        const uint16_t neg65504 = 0xFBFFu; // IEEE754 half for -65504
        for (int64_t r = 0; r < T; ++r) {
            for (int64_t c = r + 1; c < T; ++c) {
                tree_buf[static_cast<size_t>(r * T + c)] = neg65504;
            }
        }
    }

    // 2.c KV caches (one persistent buffer per KV input)
    const int64_t kv_S = C - W;
    const int64_t kv_elems = cfg.n_heads * kv_S * cfg.head_dim;
    std::vector<std::vector<uint16_t>> kv_bufs(kv_in_indices.size(),
        std::vector<uint16_t>((kv_elems), 0));

    // ---------- Wrap persistent buffers in tensors once ----------
    auto make_i64 = [&](std::vector<int64_t>& b, std::initializer_list<int64_t>shape) {
        return Ort::Value::CreateTensor<int64_t>(mem, b.data(), b.size(),
                                                 std::data(shape), shape.size());
    };
    auto make_f16 = [&](std::vector<uint16_t>& b, std::initializer_list<int64_t> shape) {
       return Ort::Value::CreateTensor(mem, b.data(), b.size() * sizeof(uint16_t),
                                       std::data(shape), shape.size(),
                                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
    };

    Ort::Value ids_tensor  = make_i64(ids_buf,  {1, W});
    Ort::Value mask_tensor = make_i64(mask_buf, {1, T});
    Ort::Value pos_tensor  = make_i64(pos_buf,  {1, T});
    Ort::Value tree_tensor{nullptr};
    if (has_tree_attention) {
        tree_tensor = make_f16(tree_buf, {1, 1, T, T});
    }

    std::vector<Ort::Value> kv_tensors; kv_tensors.reserve(kv_in_indices.size());
    for (auto& b : kv_bufs) {
        kv_tensors.emplace_back(make_f16(b, {1, cfg.n_heads, kv_S, cfg.head_dim}));
    }

    // ---------- Initialize first call inputs buffers ----------
    // Fill ids_buf with the initial window slice (total_input[0:W])
    std::copy_n(total_input.begin(), W, ids_buf.begin()); // ids_buf <- initial window

    // Attention: zeros then ones for last W positions
    std::fill(mask_buf.begin(), mask_buf.end() - W, 0);
    std::fill(mask_buf.end() - W, mask_buf.end(), 1);

    // Position: zeros then arange(W) at tail
    std::fill(pos_buf.begin(), pos_buf.end() - W, 0);
    for (int64_t i = 0; i < W; ++i) pos_buf[T - W + i] = i;

    // ---- Assemble inputs ONCE; DO NOT reassign these Ort::Value later ----
    std::vector<Ort::Value> inputs(io.input_names.size());
    inputs[*idx_input_ids]      = std::move(ids_tensor);
    inputs[*idx_attention_mask] = std::move(mask_tensor);
    inputs[*idx_position_ids]   = std::move(pos_tensor);
    if (has_tree_attention) {
        inputs[*idx_tree_attention] = std::move(tree_tensor);
    }
    for (size_t k=0;k<kv_in_indices.size();++k)
        inputs[kv_in_indices[k]] = std::move(kv_tensors[k]);

    // ---------- Runtime state ----------
    const int64_t max_gen_total = cfg.max_gen_tokens + prompt_size;
    int64_t next_index = W;     // next consumed position
    int64_t old_j = 0;          // start index of previous window

    printf("=== Begin inference loop... %lld - %lld\n", next_index, max_gen_total);
    // Inference loop
    while (next_index < max_gen_total) {
        // Sanity check
        if (!inputs[*idx_input_ids].IsTensor())
            throw std::runtime_error("input_ids Ort::Value is not a tensor");
        if (ids_buf.empty())
            throw std::runtime_error("ids_buf is empty");

        auto output = infer_session.Run(Ort::RunOptions{nullptr},
            io.input_name_ptrs.data(), inputs.data(), inputs.size(),
            io.output_name_ptrs.data(), io.output_name_ptrs.size());

        // output order matches io.output_names
        // Get logits and produce next token
        Ort::Value& logits_val = output[*idx_logits];
        auto logits_info = logits_val.GetTensorTypeAndShapeInfo();
        auto logits_shape = logits_info.GetShape(); // [1, steps, vocal]
        if (logits_shape.size() != 3) {
            throw std::runtime_error("Expected logits of rank 3 [B, steps, vocab]");
        }
        auto logits_type = logits_info.GetElementType();
        // std::cout << "logits element type=" << logits_type
        //           << " shape=[" << logits_shape[0] << "," << logits_shape[1] << "," << logits_shape[2] << "]\n";
        int64_t B = logits_shape[0];
        int64_t steps = logits_shape[1];   // typically == window
        int64_t vocab = logits_shape[2];

        int64_t new_token = -1;
        switch (logits_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
                const uint16_t* p = logits_val.GetTensorData<uint16_t>();
                new_token = argmax_last_step_f16(p, steps, vocab);
                break;
            }
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                const float* p = logits_val.GetTensorData<float>();
                new_token = argmax_last_step_logits_f32(p, steps, vocab);
                break;
            }
            default:
                throw std::runtime_error("Unsupported logits dtype for argmax");
        }

#if DBG
        std::vector<int64_t> new_ids = {new_token};
        std::string new_reply = tokenizer.decode_ids(new_ids, true);
        printf("%s", new_reply.c_str());
#endif // DBG

        if (new_token < 0 || new_token >= vocab) {
          std::cout << "Argmax produced invalid id " << new_token
                    << " for vocab size " << vocab
                    << " (dtype=" << logits_type << ")";
          throw std::runtime_error("Internal: Argmax output issue...");
        }

        // Prepare inputs for next iteration
        // We need to advance over the prompt first; only after we pass prompt_size we append generated tokens.
        old_j = next_index;

        if (next_index < prompt_size) {
            // Still consuming prompt tokens (slide by W - W
            // or to prompt_size)
            if (prompt_size - next_index >= W) {
                next_index += W;
            } else {
                next_index = prompt_size;
            }
            // j = next_index - window
        } else {
            next_index += 1;
            // Append the newly generated token
            total_input.push_back(new_token);
            // Early stop if EOS
            if (new_token == cfg.eos_token_id) {
                break;
            }
        }
        int64_t j = next_index - W;  // inclusive slice start for the sliding window

        // 1) Update input_ids window (persistent ids_buf)
        //    Copy total_input[j:next_index] into ids_buf
        std::copy_n(total_input.begin() + j, W, ids_buf.begin());

        // Optional sanity: ensure ids in range 0..vocab-1 (during generation phase)
        // (During early prompt consumption, total_input contains BOS + prompt ids)
        for (int64_t t : ids_buf) {
            if (t < -vocab || t >= vocab) { // allow negatives only if your model expects them (most don't)
                std::ostringstream oss;
                oss << "ids_buf contains out-of-range id: " << t << " for vocab=" << vocab;
                throw std::runtime_error(oss.str());
            }
        }

        // 2) Update attention_mask
        std::fill(mask_buf.begin(), mask_buf.end() - next_index, 0);
        std::fill(mask_buf.end() - next_index, mask_buf.end(), 1);

        // 3) Update position_ids
        std::fill(pos_buf.begin(), pos_buf.end() - next_index, 0);
        for (int64_t i = 0; i < next_index; ++i) pos_buf[T - next_index + i] = i;

        // 4) KV present -> past slice copied into persistent kv_bufs
        //    Slice: [ :, :, start:end, : ] where start = delta, end = (context - window) + delta
        const int64_t delta = next_index - old_j;
        int64_t start = delta;
        int64_t end   = (C - W) + delta; // exclusive
        for (size_t kin = 0; kin < kv_in_indices.size(); ++kin) {
            // Map this past_key_values input to its "present" output
            std::string present_name = io.input_names[kv_in_indices[kin]];
            if (auto p = present_name.find("past_key_values"); p != std::string::npos) {
                present_name.replace(p, std::string("past_key_values").size(), "present");
            }
            auto it = present_out_index.find(present_name);
            if (it == present_out_index.end()) {
                // Fallback: use first "present" if names don't match exactly
                if (present_out_index.empty())
                throw std::runtime_error("No present outputs to feed KV caches");
                it = present_out_index.begin();
            }

            Ort::Value& present_val = output[it->second];
            auto pinfo = present_val.GetTensorTypeAndShapeInfo();
            auto p_shape = pinfo.GetShape(); // [1, H, S_total, D]
            if (pinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                throw std::runtime_error("present KV not FP16");
            const int64_t pB = p_shape[0];
            const int64_t pH = p_shape[1];
            const int64_t pS = p_shape[2];
            const int64_t pD = p_shape[3];
            if (pB != 1 || pH != cfg.n_heads || pD != cfg.head_dim) {
                throw std::runtime_error("present KV shape mismatch");
            }

            // Clamp slice to available S
            if (end > pS) end = pS;
            if (start < 0) start = 0;
            int64_t slice_S = std::max<int64_t>(0, end - start);

            const uint16_t* src_all = present_val.GetTensorData<uint16_t>();
            uint16_t* dst = kv_bufs[kin].data();
            // Copy per head
            for (int64_t h=0; h<pH; ++h) {
                const uint16_t* src_head = src_all + (h * pS + start) * pD;
                uint16_t* dst_head = dst + h * (kv_S * pD);
                // copy rows
                for (int64_t s=0; s<slice_S && s < kv_S; ++s) {
                    std::memcpy(dst_head + s*pD, src_head + s*pD, size_t(pD)*sizeof(uint16_t));
                }
                // zero remainder if any
                if (slice_S < kv_S) {
                    std::memset(dst_head + slice_S*pD, 0, size_t((kv_S - slice_S) * pD) * sizeof(uint16_t));
                }
            }
            // kv_tensors[k] already wraps dst; no need to rebuild the Ort::Value
        }
    }

    printf("\nGenerated %zu tokens\n", total_input.size());
    std::string reply = tokenizer.decode_ids(total_input, true);
    if (custom_mode) {
        printf("%s\n", reply.c_str());
    }
    return reply;
}

int main(int argc, char *argv[]) {
    try {
        Args args = parse_args(argc, argv);

        // Phi-3 chat-formatted default prompt, applied only when the user gave
        // neither --prompt nor --prompt-file (the shared Args default is empty
        // because the chat template is model-specific).
        if (args.prompt.empty()) {
            args.prompt =
                "<|system|>\n"
                "You are a helpful assistant who knows everything about literature!<|end|>\n"
                "<|user|>\n"
                "Tell me what you know about the Little Prince<|end|>\n"
                "<|assistant|>\n";
        }

        // Forward --verbose to the Vulkan plugin EP via env var. Must be set
        // BEFORE the EP DLL is loaded so the plugin sees it at session init
        // (and definitely before the EP unloads -- the summary print is
        // gated on this in ~VulkanDataTransfer()). Set unconditionally
        // (cheap) but only "1" when the user actually asked for it.
        if (args.verbose) {
            _putenv_s("VULKAN_EP_DT_STATS", "1");
        }

        timer_init();

        GenerationConfig cfg;
        cfg.max_gen_tokens = std::max<int64_t>(64, args.max_new_tokens);
        cfg.total_sequence = std::max<int64_t>(256, cfg.max_gen_tokens + 256);
        cfg.window = 32;
        cfg.context = 2048;
        cfg.n_heads = 32;
        cfg.head_dim = 96;
        cfg.bos_token_id = 1;  // to be set when tokenizer is initialized
        cfg.eos_token_id = 2;  // to be set when tokenizer is initialized

        // ===== One-time model + tokenizer load (shared across all prompts) =====
        // Done here (not inside generate_text) so a custom-prompt batch loads the
        // model ONCE, matching the OGA reference driver and giving a fair,
        // load-amortized throughput comparison.
        Onnx rt(args.n_threads, args.ep, args.ep_dll, args.device_id);

        // Resolve the model filename. The .onnx filename is NOT uniform across
        // published flavors, so prefer genai_config.json's decoder.filename (what
        // ORT GenAI uses) and fall back to model.onnx. --model_name always wins.
        std::string model_file;
        const char* model_src;
        if (args.model_name_set) {
            model_file = args.model_name;
            model_src = "--model_name";
        } else {
            const std::string detected = ResolveModelFilenameFromGenaiConfig(args.model_dir);
            if (!detected.empty()) {
                model_file = detected;
                model_src = "genai_config.json";
            } else {
                model_file = "model.onnx";
                model_src = "default";
            }
        }
        const std::string model_path = args.model_dir + "/" + model_file;
        const int64_t t_session_init_start = timer_us();
        printf("Model file: '%s' (resolved from %s)\n", model_file.c_str(), model_src);
        printf("Loading model '%s' with %d threads\n", model_path.c_str(), args.n_threads);
        Ort::Session infer_session = rt.load(model_path);
        const int64_t t_session_init_us = timer_us() - t_session_init_start;
        printf("ORT session initialized...: %8.2fs\n",
               t_session_init_us / (1000.0 * 1000.0));

        const std::string tokenizer_json = args.model_dir + "/tokenizer.json";
        printf("loading tokenizer.json from '%s'\n", tokenizer_json.c_str());
        const int64_t t_tok_start = timer_us();
        TokenizerWrap tokenizer(tokenizer_json);
        const int64_t t_tokenizer_us = timer_us() - t_tok_start;
        cfg.bos_token_id = tokenizer.TokenToId("<s>");
        cfg.eos_token_id = tokenizer.TokenToId("<|endoftext|>");
        printf("Token to ID: '<s>'->[%lld]\n", cfg.bos_token_id);
        printf("Token to ID: '<|endoftext|>'->[%lld]\n", cfg.eos_token_id);

        // ===== Build the prompt list =====
        // --cpf <file> expands a CUSTOM_TEMPLATE_PROMPT/CUSTOM_PROMPT batch into
        // one prompt per message (overrides --prompt / the default). Otherwise a
        // single prompt is run from --prompt / --prompt-file / the default.
        std::vector<std::string> prompts;
        std::vector<std::string> labels;  // banner labels (raw message lines)
        bool custom_mode = false;
        if (!args.cpf.empty()) {
            CustomPromptFile cpf = LoadCustomPromptFile(args.cpf);
            if (!cpf.ok) {
                throw std::runtime_error(
                    "Custom-prompt file '" + args.cpf +
                    "' produced no prompts (need CUSTOM_PROMPT entries).");
            }
            prompts = BuildCustomPrompts(cpf);
            labels = cpf.messages;
            custom_mode = true;
            if (args.cpf_limit > 0 &&
                static_cast<size_t>(args.cpf_limit) < prompts.size()) {
                prompts.resize(static_cast<size_t>(args.cpf_limit));
                labels.resize(static_cast<size_t>(args.cpf_limit));
            }
            printf("Custom prompts: %zu message(s) from '%s'\n",
                   prompts.size(), args.cpf.c_str());
        } else {
            prompts.push_back(args.prompt);
            labels.push_back(args.prompt);
        }

        const int64_t t_batch_start = timer_us();
        std::string reply;
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            reply = generate_text(args, cfg, infer_session, tokenizer,
                                  prompts[pi], t_session_init_us, t_tokenizer_us,
                                  custom_mode, pi, prompts.size(), labels[pi]);
        }

        if (custom_mode) {
            printf("\n\nTotal elapsed time:     %9.2fs\n\n",
                   (timer_us() - t_batch_start) / (1000.0 * 1000.0));
        } else {
            std::cout << "Generated: \n   [" << reply << "]\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
