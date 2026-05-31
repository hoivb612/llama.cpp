#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include "llm-infer.h"

#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "log.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cctype>

#include "json.hpp" // For JSON handling

// For SLM 
llama_model *llm_model;
llama_context *llm_ctx;
std::vector<llama_token> llm_session_tokens;
std::vector<llama_token> llm_tokens_shared;
common_speculative_ptr llm_spec;

// For embeddings
llama_model * embed_model;
llama_context * embed_ctx;


void default_log_callback(
    ggml_log_level level, 
    const char * text, 
    void * user_data) {
    GGML_UNUSED(text);

    ggml_log_level llindex_log_level = GGML_LOG_LEVEL_NONE;
    if (user_data != nullptr) {
        llindex_log_level = *(ggml_log_level *)user_data;
    }

    // Treat configured level as a threshold:
    // ERROR -> only errors, WARN -> warnings+errors, INFO -> info+warn+error.
    bool print = false;
    switch (llindex_log_level) {
        case GGML_LOG_LEVEL_ERROR:
            print = (level == GGML_LOG_LEVEL_ERROR);
            break;
        case GGML_LOG_LEVEL_WARN:
            print = (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR);
            break;
        case GGML_LOG_LEVEL_INFO:
            print = (level == GGML_LOG_LEVEL_INFO || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR);
            break;
        default:
            print = false;
            break;
    }

    if (print) {
        fputs(text, stdout);
    }
}

static std::string slm_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static bool slm_parse_flash_attn_mode(const std::string & mode, llama_flash_attn_type & out) {
    const std::string v = slm_lower(mode);
    if (v == "on" || v == "1" || v == "true" || v == "yes" || v == "enabled") {
        out = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return true;
    }
    if (v == "off" || v == "0" || v == "false" || v == "no" || v == "disabled") {
        out = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return true;
    }
    if (v == "auto" || v.empty()) {
        out = LLAMA_FLASH_ATTN_TYPE_AUTO;
        return true;
    }
    return false;
}

std::string pfx_file_path(
    std::string pfx) {

    static std::hash<std::string> hasher;
    static std::string dir = "./ggml_cache";
    
    // create the cache dir if it does not exist yet
    if (!CreateDirectoryA(dir.c_str(), NULL)) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "%s: Failed to create directory: %s - use current dir for prefix cache\n",
                __func__, dir.c_str());
            dir = ".";
        }
    }

    // default generated file name
    std::string full_file_path = dir + "/" + std::to_string(hasher(pfx));

    return full_file_path;
}

std::vector<llama_token> slm_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

std::string slm_token_to_piece(const struct llama_context * ctx, llama_token token, bool special = false) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
    const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    } else {
        piece.resize(n_chars);
    }

    return piece;
}

static ggml_log_level g_llm_log_level = GGML_LOG_LEVEL_NONE;

static ggml_log_level llm_log_level_from_verbose(int verbose) {
    // Keep llama internal logs quiet by default; avoid INFO spam in perf runs.
    if (verbose == 2) return GGML_LOG_LEVEL_INFO;
    (verbose >= 1 ? GGML_LOG_LEVEL_WARN : GGML_LOG_LEVEL_NONE);
}

void llm_log_callback(ggml_log_level level, const char * text, void * user_data) {
    default_log_callback(level, text, user_data);
}

LLM_INFER_API
void llm_disable_log() {
    g_llm_log_level = GGML_LOG_LEVEL_NONE;
    llama_log_set(llm_log_callback, &g_llm_log_level);

}

LLM_INFER_API
void llm_enable_log() {
    g_llm_log_level = GGML_LOG_LEVEL_WARN;
    llama_log_set(llm_log_callback, &g_llm_log_level);
}

LLM_INFER_API 
const char * llm_system_info() {
   return(llama_print_system_info());
}

LLM_INFER_API
void llm_print_tensor_op_perf_stats() {
    // Query all registered backends for a perf print function
    typedef void (*perf_fn_t)();
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        auto fn = (perf_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_cpu_print_tensor_op_perf");
        if (fn) { fn(); return; }
    }
}

LLM_INFER_API
const char * llm_get_chat_template() {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif
    if (!llm_model) return nullptr;
    return llama_model_chat_template(llm_model, nullptr);
}

LLM_INFER_API
bool llm_initialize(
    model_params & params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    // init Llama backend
    llama_backend_init();

    // tensor repack mode: 0=none, 1=ggml, 2=xbox, 3=xbcg, 4=xbox-st, 5=mulmat-xbox
    llama_set_tensor_repack_mode((ggml_tensor_repack_mode_t) params.tensor_repack_mode);
    if (params.verbose >= 1) {
        printf("%s: tensor_repack_mode = %d\n", __func__, params.tensor_repack_mode);
    }

    // Control llama.cpp log verbosity using stable storage.
    g_llm_log_level = llm_log_level_from_verbose(params.verbose);
    llama_log_set(default_log_callback, &g_llm_log_level);

    // initialize the model
    llama_model_params model_params = llama_model_default_params();

#ifdef GGML_USE_CUDA
    #pragma message("++++++++ Support both CUDA and CPU for inference")
    if ((ggml_backend_cuda_get_device_count() != 0) && (params.force_cpu_mode == 0)) {
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode\n", __func__);
        }
    } else {
        // either there is no GPU or CPU forcing function
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    }
#elif defined(GGML_USE_VULKAN)
    #pragma message("++++++++ Support both Vulkan and CPU for inference")
    if (params.force_cpu_mode != 0) {
        // CPU forcing mode
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    } else {
        // GPU offload is default on a Vulkan build
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode\n", __func__);
        }
    }
#elif defined(GGML_USE_DX12)
    #pragma message("++++++++ Support both DX12 and CPU for inference")
    if (params.force_cpu_mode != 0) {
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    } else {
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode (DX12)\n", __func__);
        }
    }
#else
    #pragma message("++++++++ Support CPU-only for inference")
    // CPU is the default (there is no CUDA nor Vulkan devices)
    if (params.verbose == 2) {
        printf("%s: Running in full CPU mode\n", __func__);
    }
    model_params.n_gpu_layers = 0;
#endif // GGML_USE_CUDA

    // GPU adapter and split mode selection
    model_params.main_gpu = params.main_gpu;
    if (params.split_mode >= 0) {
        model_params.split_mode = (enum llama_split_mode)params.split_mode;
    }

    llm_model = llama_model_load_from_file(params.model_name.c_str(), model_params);
    if (llm_model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return false;
    }

    // initialize the context
    llama_context_params llm_ctx_params = llama_context_default_params();

    // Grab values from defaults or command line args 
    llm_ctx_params.n_ctx = params.n_ctx;
    llm_ctx_params.n_batch = params.n_batch;
    llm_ctx_params.n_threads = params.n_threads;
    llm_ctx_params.n_threads_batch = params.n_threads;
    llm_ctx_params.no_perf = false;
    if (!slm_parse_flash_attn_mode(params.flash_attn_mode, llm_ctx_params.flash_attn_type)) {
        printf("%s: warning: invalid flash-attn mode '%s'; using auto\n", __func__, params.flash_attn_mode.c_str());
        llm_ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    }

    llm_ctx = llama_init_from_model(llm_model, llm_ctx_params);

    if (llm_ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return false;
    }

    llm_spec.reset();
    params.total_mtp_draft_accepted = 0;
    params.total_mtp_draft_proposed = 0;
    params.total_mtp_draft_rounds = 0;
    const std::string spec_type = slm_lower(params.spec_type);
    const bool wants_mtp = (spec_type == "mtp" || spec_type == "draft-mtp");

    bool mtp_attached = false;
    if (wants_mtp && !params.mtp_head_path.empty()) {
        llama_model_params mp = model_params; // inherit GPU/split/mmap config
        const int rc = llama_model_load_mtp_from_file(llm_model, params.mtp_head_path.c_str(), mp);
        if (rc != 0 || !llama_model_has_mtp_assistant(llm_model)) {
            printf("%s: warning: failed to load MTP head '%s' (rc=%d); continuing without MTP\n",
                   __func__, params.mtp_head_path.c_str(), rc);
        } else {
            mtp_attached = true;
            printf("%s: loaded MTP assistant head '%s' (n_embd_backbone=%u)\n",
                   __func__, params.mtp_head_path.c_str(),
                   (unsigned) llama_model_mtp_n_embd_backbone(llm_model));
        }
    } else if (wants_mtp && params.mtp_head_path.empty()) {
        printf("%s: warning: --spec-type mtp requested but no --mtp-head provided; continuing without MTP\n", __func__);
    }

    if (wants_mtp) {
        if (!mtp_attached) {
            printf("%s: warning: MTP speculative decoding disabled because assistant MTP head is unavailable\n", __func__);
        } else {
            const int block_size = std::max(2, std::min(32, params.draft_block_size));
            common_params_speculative spec_params;
            spec_params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            spec_params.draft.ctx_tgt = llm_ctx;
            spec_params.draft.ctx_dft = nullptr; // assistant is attached to target model
            spec_params.draft.n_max = block_size - 1;
            llm_spec.reset(common_speculative_init(spec_params, /* n_seq */ 1));
            if (!llm_spec) {
                printf("%s: warning: failed to initialize MTP speculative path; continuing without MTP\n", __func__);
            } else {
                printf("%s: MTP speculative decoding active, draft_block_size=%d (drafts %d tokens/round)\n",
                       __func__, block_size, block_size - 1);
            }
        }
    } else if (spec_type != "none") {
        printf("%s: warning: unsupported spec_type '%s'; expected none|mtp|draft-mtp. Continuing without speculative decoding\n",
               __func__, params.spec_type.c_str());
    }

    printf("\n%s: n_len = %d, n_ctx = %d\n", __func__, params.n_len, llama_n_ctx(llm_ctx));
    printf("%s: n_threads = %d, n_threads_batch = %d\n\n", __func__, llm_ctx_params.n_threads, llm_ctx_params.n_threads_batch);
    printf("%s: flash-attn = %s%s\n\n",
           __func__,
           llama_flash_attn_type_name(llm_ctx_params.flash_attn_type),
           llm_ctx_params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? " (runtime-selected)" : "");

    llm_tokens_shared = {};
    if (params.pfc_mode) {
        // start from a known point
        llama_memory_clear(llama_get_memory(llm_ctx), true);

        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            llm_tokens_shared = slm_tokenize(llm_ctx, params.pfx_shared, params.add_special, params.parse_special);
            // build the cache file directory
            params.pfx_file = pfx_file_path(params.pfx_shared);
            // load the cache and create one if it does not exist
            llm_session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = params.first_prompt ? 0xffffffff : 0;
            if (!llama_state_load_file(llm_ctx, 
                                       params.pfx_file.c_str(),
                                       llm_session_tokens.data(),
                                       llm_session_tokens.capacity(),
                                       &n_token_count_out)) {
                printf("%s: State file does not exist or load failed: '%s'\n", __func__, params.pfx_file.c_str());
                llm_session_tokens.resize(0);
                params.save_llm_state = true;
                // the load failed so start from scratch to initialize all internal 
                // state with the first full prompt 
                llm_tokens_shared.clear();
                params.pfx_shared = "";

            } else {
                printf("%s: Loading saved state successfully from '%s'...\n", __func__, params.pfx_file.c_str());
                llm_session_tokens.resize(n_token_count_out);
                // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, llm_session_tokens).c_str());

                // sanity check
                // assert(llm_tokens_shared.size() <= llm_session_tokens.size());
                for (size_t i = 0; i < llm_tokens_shared.size(); i++) {
                    // assert(llm_tokens_shared[i] == llm_session_tokens[i]);
                    if (llm_tokens_shared[i] != llm_session_tokens[i]) {
                        printf("Mismatched pfx tokens!!!!\n");
                        return false;
                    }
                }
                //printf("%s: token_shared=%zd\n", __func__, llm_tokens_shared.size());
            }

        } else {
            // no shared prompt detected
            llm_tokens_shared.clear();
        }

    } else {
        // No pfc mode
        llm_tokens_shared.clear();
    }

    return true;
}

LLM_INFER_API
bool llm_inference(
    model_params& params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    std::vector<llama_token> embd_inp;
    int n_past = 0;
    int n_kv_pfx = 0;

    if (params.pfc_mode) {
        // remove any "future" tokens that we might have inherited from the previous session
        if (llm_tokens_shared.size() != 0) {
            //
            // current version of llama does not support seq_id being negative
            // llama_kv_self_seq_rm(llm_ctx, -1, llm_tokens_shared.size(), -1);
            //
            llama_memory_seq_rm(llama_get_memory(llm_ctx), 0, llm_tokens_shared.size(), -1);
            embd_inp.insert(embd_inp.end(), llm_tokens_shared.begin(), llm_tokens_shared.end());
        }
        n_past = llm_tokens_shared.size();
        n_kv_pfx = llm_tokens_shared.size();

    } else {
        // start from a known point
        llama_memory_clear(llama_get_memory(llm_ctx), true);
        embd_inp.clear();
        n_past = 0;
        n_kv_pfx = 0;
    }

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_input = slm_tokenize(llm_ctx, params.prompt, params.add_special, params.parse_special);

    // append the variant part of the prompt (pfc mode) or the full prompt (for non-pfc mode)
    embd_inp.insert(embd_inp.end(), tokens_input.begin(), tokens_input.end());

    const int n_ctx = llama_n_ctx(llm_ctx);
    const int n_kv_req = tokens_input.size() + (params.n_len - tokens_input.size() - n_kv_pfx);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req(%d-%d) > n_ctx(%d), the required KV cache size is not big enough\n",
            __func__,
            n_kv_pfx,
            n_kv_req,
            n_ctx);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return false;
    }

    //
    // calculate how much has been processed through the saved state file (pfc mode)
    // 
    int prompt_index = 0;
    if (params.pfc_mode) {
        int n_tokens_processed = 0;
        for (; prompt_index < embd_inp.size(); prompt_index++) {
            // not fully matched with shared tokens
            if (embd_inp[prompt_index] != llm_tokens_shared[prompt_index]) {
                break;
            }

            n_tokens_processed++;

            // embd_inp is fully matched with shared prefix cache
            if (n_tokens_processed >= (int)llm_tokens_shared.size()) {
                ++prompt_index;
                break;
            }
        }
    }

    // build token list of new tokens for inference
    std::vector<llama_token> embd;
    for (int i = prompt_index; i < embd_inp.size(); i++) {
        embd.push_back(embd_inp[i]);
    }

    // initialize the sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // sample the most likely token (greedy sampling algo)
    const int   top_k = 40;
    const float top_p = 0.9f;
    const float min_keep = 1.0f;
    const float temp = 0.1f;
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, min_keep));  
    llama_sampler_chain_add(smpl, llama_sampler_init_temp (temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));
    // llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int64_t t1_start = ggml_time_us();
    GGML_UNUSED(t1_start );

    // decode the remaining prompt not covered by the shared portion
    if (llm_spec) {
        llama_set_embeddings(llm_ctx, common_speculative_need_embd(llm_spec.get()));
    }
    for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }

        if (llama_decode(llm_ctx, llama_batch_get_one(&embd[i], n_eval))) {
            printf("%s : failed to eval\n", __func__);
            llama_sampler_free(smpl);
            return false;
        }

        // update n_past to reflect what has been decoded
        n_past += n_eval;
    }

    // Keep prompt/decode timing split accurate on async backends.
    llama_synchronize(llm_ctx);
    int64_t t2_start = ggml_time_us();
    float t_prompt_eval_ms = (t2_start - t1_start) / 1000.0f;
    if (params.verbose == 2) {
        printf("Prompt TTFT = %.2fms (size = %zu) (%.2ft/s) (%.2fms)\n", 
            t_prompt_eval_ms, 
            embd.size(), 
            (embd.size() * 1000.0f) / t_prompt_eval_ms, 
            t_prompt_eval_ms / embd.size());
    }

    // 
    // Save the state file here because we need llama to process once
    // to initialize all the different internal states - doing this in 
    // llm_initialize() does not work because at that time, llama has not 
    // done any token processing yet.
    // 

    if (params.pfc_mode && params.save_llm_state) {
        llm_session_tokens.insert(llm_session_tokens.end(), embd_inp.begin(), embd_inp.end());

        llama_state_save_file(llm_ctx, params.pfx_file.c_str(), llm_session_tokens.data(), llm_session_tokens.size());
        params.save_llm_state = false;

        // llm_token_shared must be updated for next inference 
        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            llm_tokens_shared = slm_tokenize(llm_ctx, params.pfx_shared, params.add_special, params.parse_special);
        }
    }

    // compute max generated tokens for this call (same limit style as minslm-cli)
    int max_gen = std::min(params.n_len - n_past, 128);
    if (max_gen < 0) {
        max_gen = 0;
    }

    std::string slm_output;
    int n_tokens_generated = 0;

    // sample the last token just received
    const llama_vocab * vocab = llama_model_get_vocab(llm_model);

    int mtp_draft_accepted = 0;
    int mtp_draft_proposed = 0;
    int mtp_draft_rounds = 0;

    if (llm_spec) {

        auto gparams = llama_sampler_chain_default_params();
        gparams.no_perf = false;
        llama_sampler * smpl_mtp = llama_sampler_chain_init(gparams);
        llama_sampler_chain_add(smpl_mtp, llama_sampler_init_greedy());

        auto emit_token = [&](llama_token id) -> bool {
            if (llama_vocab_is_eog(vocab, id)) {
                if (params.streaming_reply) {
                    printf("\n");
                }
                return false;
            }

            const std::string token_str = slm_token_to_piece(llm_ctx, id);
            if (params.streaming_reply) {
                printf("%s", token_str.c_str());
            } else {
                slm_output += token_str;
            }

            if (params.stop_char && token_str.find(params.stop_char) != std::string::npos) {
                if (params.streaming_reply) {
                    printf("\n");
                }
                return false;
            }
            return true;
        };

        common_speculative_begin(llm_spec.get(), /*seq_id=*/0, embd_inp);
        llama_token id_last = llama_sampler_sample(smpl_mtp, llm_ctx, -1);
        if (emit_token(id_last)) {
            n_tokens_generated += 1;
            params.total_llm_tokens_generated += 1;

            const int K_max = std::max(1, std::max(2, std::min(32, params.draft_block_size)) - 1);
            llama_batch vbatch = llama_batch_init(/*n_tokens=*/K_max + 1, /*embd=*/0, /*n_seq_max=*/1);
            std::vector<llama_token> drafts;
            drafts.reserve(K_max);

            auto set_row = [&](int i, llama_token tok, llama_pos pos) {
                vbatch.token[i] = tok;
                vbatch.pos[i] = pos;
                vbatch.n_seq_id[i] = 1;
                vbatch.seq_id[i][0] = 0;
                vbatch.logits[i] = 1;
            };

            while (n_tokens_generated < max_gen) {
                drafts.clear();
                auto & dp = common_speculative_get_draft_params(llm_spec.get(), 0);
                dp.drafting = true;
                dp.n_max = K_max;
                dp.n_past = n_past;
                dp.id_last = id_last;
                dp.prompt = &embd_inp;
                dp.result = &drafts;
                common_speculative_draft(llm_spec.get());

                const int K = (int) drafts.size();
                mtp_draft_rounds += 1;
                mtp_draft_proposed += K;
                set_row(0, id_last, n_past);
                for (int i = 0; i < K; ++i) {
                    set_row(i + 1, drafts[i], n_past + 1 + i);
                }
                vbatch.n_tokens = K + 1;

                if (llama_decode(llm_ctx, vbatch) != 0) {
                    printf("%s : failed MTP verify decode\n", __func__);
                    llama_batch_free(vbatch);
                    llama_sampler_free(smpl_mtp);
                    llama_sampler_free(smpl);
                    return false;
                }

                if (!common_speculative_process(llm_spec.get(), vbatch)) {
                    printf("%s : failed MTP speculative process\n", __func__);
                    llama_batch_free(vbatch);
                    llama_sampler_free(smpl_mtp);
                    llama_sampler_free(smpl);
                    return false;
                }

                int n_accepted = 0;
                llama_token new_id_last = id_last;
                for (int i = 0; i <= K; ++i) {
                    llama_token sampled = llama_sampler_sample(smpl_mtp, llm_ctx, i);
                    new_id_last = sampled;
                    if (i < K && sampled == drafts[i]) {
                        n_accepted += 1;
                        continue;
                    }
                    break;
                }

                common_speculative_accept(llm_spec.get(), /*seq=*/0, (uint16_t) n_accepted);
                mtp_draft_accepted += n_accepted;

                if (n_accepted < K) {
                    llama_memory_seq_rm(llama_get_memory(llm_ctx), /*seq=*/0, n_past + 1 + n_accepted, -1);
                }

                bool keep_going = true;
                for (int i = 0; i < n_accepted && keep_going; ++i) {
                    if (n_tokens_generated >= max_gen) {
                        keep_going = false;
                        break;
                    }
                    keep_going = emit_token(drafts[i]);
                    n_tokens_generated += 1;
                    params.total_llm_tokens_generated += 1;
                }
                if (keep_going && n_tokens_generated < max_gen) {
                    keep_going = emit_token(new_id_last);
                    n_tokens_generated += 1;
                    params.total_llm_tokens_generated += 1;
                }
                if (!keep_going) {
                    break;
                }

                n_past += n_accepted + 1;
                id_last = new_id_last;
            }

            llama_batch_free(vbatch);
        }

        params.total_mtp_draft_accepted += mtp_draft_accepted;
        params.total_mtp_draft_proposed += mtp_draft_proposed;
        params.total_mtp_draft_rounds += mtp_draft_rounds;

        llama_sampler_free(smpl_mtp);
    } else {
        while (n_tokens_generated < max_gen) {
            // sample the last token just received
            llama_token new_token_id = llama_sampler_sample(smpl, llm_ctx, -1);

            // check wither it is the end of text generation
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                if (params.streaming_reply) {
                    printf("\n");
                }
                break;
            }

            const std::string token_str = slm_token_to_piece(llm_ctx, new_token_id);
            if (params.streaming_reply) {
                printf("%s", token_str.c_str());
            } else {
                slm_output += token_str;
            }

            // Stop on caller-specified character
            if (params.stop_char && token_str.find(params.stop_char) != std::string::npos) {
                if (params.streaming_reply) printf("\n");
                break;
            }

            // save this new token for next evaluation
            embd.clear();
            embd.push_back(new_token_id);

            n_tokens_generated += 1;
            params.total_llm_tokens_generated += 1;

            // bump current generated token index
            n_past += 1;

            // decode the output for the new generated token
            if (llama_decode(llm_ctx, llama_batch_get_one(&embd[0], 1))) {
                printf("%s : failed to eval, return code %d\n", __func__, 1);
                llama_sampler_free(smpl);
                return false;
            }
        }
    }

    // Keep generation timing accurate on async backends.
    llama_synchronize(llm_ctx);
    int64_t t_us = (ggml_time_us() - t2_start);

    if (params.verbose == 2) {
        printf("> token generation time = %.2fms (%d) (%.2ft/s) (%.2fms)\n", 
            t_us / 1000.0f,
            n_tokens_generated, 
            n_tokens_generated / (t_us / 1000000.0f),
            (n_tokens_generated > 0 ? (t_us / (n_tokens_generated * 1000.0f)) : 0.0f));
        if (llm_spec && mtp_draft_proposed > 0) {
            const double mtp_accept_rate = 100.0 * (double) mtp_draft_accepted / (double) mtp_draft_proposed;
            printf("> MTP drafts: %d accepted / %d proposed (%.1f%%), rounds=%d, block_size=%d\n",
                mtp_draft_accepted,
                mtp_draft_proposed,
                mtp_accept_rate,
                mtp_draft_rounds,
                params.draft_block_size);
        }
    }

    // accumulate the token generation time
    params.total_tokens_gen_time += t_us;

    params.reply = slm_output;

    llama_sampler_free(smpl);

    return true;
}

// Multi-turn global state
static int llm_mt_n_past = 0;
static std::vector<int> llm_mt_turn_starts;  // n_past at the START of each turn's prompt

LLM_INFER_API
void llm_multiturn_begin(const model_params& params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    llm_mt_turn_starts.clear();

    if (!llm_tokens_shared.empty()) {
        // PFC mode: shared prefix is already decoded in KV cache
        // Remove any tokens past the shared prefix
        llama_memory_seq_rm(llama_get_memory(llm_ctx), 0, (int)llm_tokens_shared.size(), -1);
        llm_mt_n_past = (int)llm_tokens_shared.size();
        if (params.verbose >= 1) {
            printf("%s: multi-turn started with %d prefix tokens in KV cache (PFC)\n",
                    __func__, llm_mt_n_past);
        }
    } else {
        // Fresh start — clear everything
        llama_memory_clear(llama_get_memory(llm_ctx), true);
        llm_mt_n_past = 0;

        // Pre-decode the system prefixfrom the template so it's protected from REWIND.
        // If {message} exists in the template, split at the last user tag to avoid
        // overlap with the turn template. If no {message}, the entire template IS
        // the system prefix (turn template is provided separately).
        std::string tmpl = params.custom_template_prompt;
        std::string sys_prefix;
        size_t msg_pos = tmpl.find("{message}");

        if (msg_pos != std::string::npos && msg_pos > 0) {
            // Template contains {message} — split at the last user tag
            std::string before_msg = tmpl.substr(0, msg_pos);

            // Find the last user tag in the text before {message}
            size_t split = std::string::npos;
            size_t p;
            if ((p = before_msg.rfind("<|user|>"))            != std::string::npos) split = p;
            if ((p = before_msg.rfind("<|im_start|>user"))    != std::string::npos && (split == std::string::npos || p > split)) split = p;
            if ((p = before_msg.rfind("[INST]"))              != std::string::npos && (split == std::string::npos || p > split)) split = p;
            if ((p = before_msg.rfind("<start_of_turn>user")) != std::string::npos && (split == std::string::npos || p > split)) split = p;

            sys_prefix = (split != std::string::npos)
                         ? tmpl.substr(0, split)
                         : before_msg;
        } else if (!tmpl.empty()) {
            // No {message} in template — the entire template is the system prefix
            // (turn template is provided via CUSTOM_TURN_TEMPLATE section)
            sys_prefix = tmpl;
        }

        if (!sys_prefix.empty()) {
            std::vector<llama_token> sys_tokens = slm_tokenize(llm_ctx, sys_prefix,
                                                                params.add_special, params.parse_special);
            if (!sys_tokens.empty()) {
                for (int i = 0; i < (int)sys_tokens.size(); i += params.n_batch) {
                    int n_eval = std::min((int)sys_tokens.size() - i, params.n_batch);
                    llama_decode(llm_ctx, llama_batch_get_one(&sys_tokens[i], n_eval));
                }
                llm_mt_n_past = (int)sys_tokens.size();
            }
        }

        if (params.verbose >= 1) {
            printf("%s: multi-turn started with %d system prefix tokens in KV cache\n",
                    __func__, llm_mt_n_past);
        }
    }
}

LLM_INFER_API
bool llm_infer_multiturn(model_params& params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    // Record turn boundary (position before this turn's prompt)
    llm_mt_turn_starts.push_back(llm_mt_n_past);

    // Tokenize the prompt for this turn
    std::vector<llama_token> tokens_input = slm_tokenize(llm_ctx, params.prompt,
                                                          params.add_special, params.parse_special);

    // Check context size — leave room for generation
    const int n_ctx = llama_n_ctx(llm_ctx);
    int projected = llm_mt_n_past + (int)tokens_input.size() + 128;
    if (projected > n_ctx) {
        printf("%s: error: projected tokens (%d) > n_ctx (%d), context full\n",
               __func__, projected, n_ctx);
        printf("%s:        use llm_multiturn_rewind() to free space or increase n_ctx\n", __func__);
        llm_mt_turn_starts.pop_back();
        return false;
    }

    int64_t t1_start = ggml_time_us();

    // Decode prompt tokens — positions auto-assigned from KV cache state
    for (int i = 0; i < (int)tokens_input.size(); i += params.n_batch) {
        int n_eval = std::min((int)tokens_input.size() - i, params.n_batch);

        if (llama_decode(llm_ctx, llama_batch_get_one(&tokens_input[i], n_eval))) {
            printf("%s: failed to decode prompt\n", __func__);
            llm_mt_turn_starts.pop_back();
            return false;
        }

        llm_mt_n_past += n_eval;
    }

    int64_t t2_start = ggml_time_us();
    float t_prompt_eval_ms = (t2_start - t1_start) / 1000.0f;
    if (params.verbose == 2) {
        printf("Prompt TTFT = %.2fms (size = %zu) (%.2ft/s) (%.2fms)\n",
            t_prompt_eval_ms,
            tokens_input.size(),
            (tokens_input.size() * 1000.0f) / t_prompt_eval_ms,
            t_prompt_eval_ms / tokens_input.size());
    }

    // Handle PFC state save on first inference (same as llm_inference)
    if (params.pfc_mode && params.save_llm_state) {
        std::vector<llama_token> save_tokens;
        save_tokens.insert(save_tokens.end(), llm_tokens_shared.begin(), llm_tokens_shared.end());
        save_tokens.insert(save_tokens.end(), tokens_input.begin(), tokens_input.end());

        llama_state_save_file(llm_ctx, params.pfx_file.c_str(), save_tokens.data(), save_tokens.size());
        params.save_llm_state = false;

        // Update llm_tokens_shared for consistency
        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            params.pfx_shared = template_prompt.substr(0, pos);
            llm_tokens_shared = slm_tokenize(llm_ctx, params.pfx_shared, params.add_special, params.parse_special);
        }
    }

    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    const int   top_k = 40;
    const float top_p = 0.9f;
    const float min_keep = 1.0f;
    const float temp = 0.1f;
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, min_keep));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.seed));

    // Compute max generation length
    // In multi-turn, use n_ctx as the ceiling (not n_len which was designed for single-turn)
    int max_len = std::min(n_ctx - 1, (llm_mt_n_past + 128));

    std::string slm_output;
    int n_tokens_generated = 0;
    const llama_vocab * vocab = llama_model_get_vocab(llm_model);
    std::vector<llama_token> embd;

    while (llm_mt_n_past <= max_len) {

        llama_token new_token_id = llama_sampler_sample(smpl, llm_ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            if (params.streaming_reply) {
                printf("\n");
            }
            break;
        }

        const std::string token_str = slm_token_to_piece(llm_ctx, new_token_id);
        if (params.streaming_reply) {
            printf("%s", token_str.c_str());
        } else {
            slm_output += token_str;
        }

        // Stop on caller-specified character
        if (params.stop_char && token_str.find(params.stop_char) != std::string::npos) {
            if (params.streaming_reply) printf("\n");
            break;
        }

        embd.clear();
        embd.push_back(new_token_id);

        n_tokens_generated += 1;
        params.total_llm_tokens_generated += 1;
        llm_mt_n_past += 1;

        if (llama_decode(llm_ctx, llama_batch_get_one(&embd[0], 1))) {
            printf("%s: failed to decode generated token\n", __func__);
            llama_sampler_free(smpl);
            return false;
        }
    }

    int64_t t_us = (ggml_time_us() - t2_start);

    if (params.verbose == 2) {
        printf("> token generation time = %.2fms (%d) (%.2ft/s) (%.2fms)\n",
            t_us / 1000.0f,
            n_tokens_generated,
            n_tokens_generated / (t_us / 1000000.0f),
            (t_us / (n_tokens_generated * 1000.0f)));
    }

    // Accumulate timing
    params.total_tokens_gen_time += t_us;

    params.reply = slm_output;

    if (params.verbose >= 1) {
        printf("%s: turn %d complete, n_past = %d / %d\n",
               __func__, (int)llm_mt_turn_starts.size(), llm_mt_n_past, n_ctx);
    }

    llama_sampler_free(smpl);

    return true;
}

LLM_INFER_API
void llm_multiturn_rewind(int n_turns) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    if (n_turns <= 0 || llm_mt_turn_starts.empty()) return;

    // Clamp to available turns
    n_turns = std::min(n_turns, (int)llm_mt_turn_starts.size());

    // Get the position to rewind to
    int rewind_pos = llm_mt_turn_starts[llm_mt_turn_starts.size() - n_turns];

    printf("%s: rewinding %d turn(s), n_past %d -> %d\n",
           __func__, n_turns, llm_mt_n_past, rewind_pos);

    // Remove tokens from KV cache from rewind_pos onward
    llama_memory_seq_rm(llama_get_memory(llm_ctx), 0, rewind_pos, -1);

    // Update state
    llm_mt_n_past = rewind_pos;
    llm_mt_turn_starts.resize(llm_mt_turn_starts.size() - n_turns);
}

LLM_INFER_API
void llm_multiturn_clear() {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    printf("%s: clearing all %d turns, n_past %d -> 0\n",
           __func__, (int)llm_mt_turn_starts.size(), llm_mt_n_past);

    llama_memory_clear(llama_get_memory(llm_ctx), true);
    llm_mt_n_past = 0;
    llm_mt_turn_starts.clear();
}

LLM_INFER_API
int llm_multiturn_turn_count() {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif
    return (int)llm_mt_turn_starts.size();
}

LLM_INFER_API
int llm_multiturn_token_count() {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif
    return llm_mt_n_past;
}

LLM_INFER_API 
void llm_terminate(const model_params& params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    g_llm_log_level = GGML_LOG_LEVEL_INFO;
    llama_log_set(default_log_callback, &g_llm_log_level);
    llama_perf_context_print(llm_ctx);

    llm_spec.reset();
    llama_free(llm_ctx);
    llama_model_free(llm_model);
    llm_ctx = nullptr;
    llm_model = nullptr;

    llama_backend_free();
}

static void batch_decode(
    llama_context * ctx, 
    llama_batch & batch, 
    float * output, 
    int /* n_seq */, 
    int n_embd) {

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_memory_clear(llama_get_memory(ctx), true);

    // run model
    // fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        common_embd_normalize(embd, out, n_embd, 2);
    }
}

LLM_INFER_API
bool embed_initialize(
    model_params & params) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    llama_backend_init();

    g_llm_log_level = llm_log_level_from_verbose(params.verbose);
    llama_log_set(default_log_callback, &g_llm_log_level);

    // initialize the model
    llama_model_params model_params = llama_model_default_params();

#ifdef GGML_USE_CUDA
    #pragma message("++++++++ Support both CUDA and CPU for inference")
    if ((ggml_backend_cuda_get_device_count() != 0) && (params.force_cpu_mode == 0)) {
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode\n", __func__);
        }
    } else {
        // either there is no GPU or CPU forcing function
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    }
#elif defined(GGML_USE_VULKAN)
    #pragma message("++++++++ Support both Vulkan and CPU for inference")
    if (params.force_cpu_mode != 0) {
        // CPU forcing mode
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    } else {
        // GPU offload is default on a Vulkan build
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode\n", __func__);
        }
    }
#elif defined(GGML_USE_DX12)
    #pragma message("++++++++ Support both DX12 and CPU for inference")
    if (params.force_cpu_mode != 0) {
        model_params.n_gpu_layers = 0;
        if (params.verbose == 2) {
            printf("%s: Running in full CPU mode\n", __func__);
        }
    } else {
        model_params.n_gpu_layers = 999;
        if (params.verbose == 2) {
            printf("%s: Running in full GPU-offload mode (DX12)\n", __func__);
        }
    }
#else
    #pragma message("++++++++ Support CPU-only for inference")
    // CPU is the default (there is no CUDA nor Vulkan devices)
    if (params.verbose == 2) {
        printf("%s: Running in full CPU mode\n", __func__);
    }
    model_params.n_gpu_layers = 0;
#endif // GGML_USE_CUDA

    // GPU adapter and split mode selection
    model_params.main_gpu = params.main_gpu;
    if (params.split_mode >= 0) {
        model_params.split_mode = (enum llama_split_mode)params.split_mode;
    }

    embed_model = llama_model_load_from_file(params.model_name.c_str(), model_params);
    if (embed_model == NULL) {
       printf("%s: error: unable to load model\n" , __func__);
       return false;
    }

   // initialize the context
   llama_context_params embed_ctx_params = llama_context_default_params();

   // Grab default params or values from command line args
   embed_ctx_params.n_ctx = params.n_ctx;
   embed_ctx_params.n_batch = params.n_batch;
   embed_ctx_params.n_threads = params.n_threads;
   embed_ctx_params.n_threads_batch = params.n_threads;

   // For BERT models batch size must be equal to ubatch size
   embed_ctx_params.n_ubatch = params.n_batch;
   embed_ctx_params.embeddings = true;

   embed_ctx = llama_init_from_model(embed_model, embed_ctx_params);

   if (embed_ctx == NULL) {
       printf("%s: error: failed to create the llama_context\n" , __func__);
       return false;
   }

   // Save dimension from model hparams
   params.n_dim = llama_model_n_embd(embed_model);

   const int n_ctx_train = llama_model_n_ctx_train(embed_model);
    const int n_ctx = llama_n_ctx(embed_ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(embed_ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        fprintf(stderr, "%s: error: pooling type NONE not supported\n", __func__);
        return false;
    }

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
        return false;
    }

    return true;
}

LLM_INFER_API
bool embed_encode_batch(
    const model_params & params, 
    std::vector<chunk> & chunks) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    const size_t n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // Allocate output
    int n_embd = llama_model_n_embd(embed_model);
    const llama_vocab * vocab = llama_model_get_vocab(llm_model);

    // Tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = slm_tokenize(embed_ctx, chunk.textdata, params.add_special, params.parse_special);
        if (inp.size() > params.n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) params.n_batch);
            return false;
        }

        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (inp.empty() || inp.back() != llama_vocab_eos(vocab))) {
            inp.push_back(llama_vocab_eos(vocab));
        }

        chunk.tokens = inp;
    }

    // Break into batches
    int p = 0; // number of prompts already processed
    int s = 0; // number of prompts in current batch

    std::vector<float> embeddings(n_chunks * n_embd, 0);
    float * emb = embeddings.data();

    for (int k = 0; k < n_chunks; k++) {
        // Clamp to n_batch tokens
        auto & rag_tokens = chunks[k].tokens;
        const uint64_t n_toks = rag_tokens.size();

        // Encode if at capacity
        if (batch.n_tokens + n_toks > params.n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(embed_ctx, batch, out, s, n_embd);
            common_batch_clear(batch);
            p += s;
            s = 0;
        }

        // Add to batch
        size_t n_tokens = rag_tokens.size();
        for (size_t i = 0; i < n_tokens; i++) {
            common_batch_add(batch, rag_tokens[i], (llama_pos) i, { s }, true);
        }

        s += 1;
    }

    // Final batch
    float * out = emb + p * n_embd;
    batch_decode(embed_ctx, batch, out, s, n_embd);

    // Save embeddings to chunks in memory
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embeddings = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
    }

    llama_batch_free(batch);

    return true;
}

LLM_INFER_API
bool embed_encode_batch_single(
    const model_params & params, 
    std::vector<chunk> & chunks) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    const size_t n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // Allocate output
    int n_embd = llama_model_n_embd(embed_model);
    const llama_vocab * vocab = llama_model_get_vocab(llm_model);
    std::vector<float> chunk_embeddings(n_embd, 0);

    // Tokenize the prompts and trim
    for (auto & chunk : chunks) {
        chunk.tokens = slm_tokenize(embed_ctx, chunk.textdata, params.add_special, params.parse_special);
        size_t n_tokens = chunk.tokens.size();
        if (n_tokens > params.n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) n_tokens, (long long int) params.n_batch);
            return false;
        }

        // add eos if not present
        if (llama_vocab_eos(vocab) >= 0 && (chunk.tokens.empty() || chunk.tokens.back() != llama_vocab_eos(vocab))) {
            chunk.tokens.push_back(llama_vocab_eos(vocab));
        }


        // Add to batch
        for (size_t i = 0; i < n_tokens; i++) {
            common_batch_add(batch, chunk.tokens[i], (llama_pos) i, { 0 }, true);
        }

        batch_decode(embed_ctx, batch, chunk_embeddings.data(), 1, n_embd);
        chunk.embeddings = chunk_embeddings;

        common_batch_clear(batch);
    }

    llama_batch_free(batch);

    return true;
}

LLM_INFER_API
bool embed_encode_single(
    const model_params & params,
    const std::string& query,
    std::vector<float> & embeddings) {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    const int n_embd = llama_model_n_embd(embed_model);
    const llama_vocab * vocab = llama_model_get_vocab(llm_model);
    std::vector<float> query_embeddings(n_embd, 0);
    
    std::vector<int32_t> query_tokens = slm_tokenize(embed_ctx, query, params.add_special, params.parse_special);
    size_t n_tokens = query_tokens.size();
    if (n_tokens > params.n_batch) {
        fprintf(stderr, "%s: error: query string size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                __func__, (long long int) n_tokens, (long long int) params.n_batch);
        return false;
    }

    // add eos if not present
    if ((llama_vocab_eos(vocab) >= 0) && 
        (query_tokens.empty() || (query_tokens.back() != llama_vocab_eos(vocab)))) {
        query_tokens.push_back(llama_vocab_eos(vocab));
    }

    
    struct llama_batch embed_query_batch = llama_batch_init(params.n_batch, 0, 1);
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(embed_query_batch, query_tokens[i], (llama_pos) i, { 0 }, true);
    }

    // query_embeddings contains the result from sentencepiece (all-MiniLM-L6-v2)
    batch_decode(embed_ctx, embed_query_batch, query_embeddings.data(), 1, n_embd);
    embeddings = query_embeddings;

    common_batch_clear(embed_query_batch);
    llama_batch_free(embed_query_batch);

    return true;
}

LLM_INFER_API
void embed_terminate() {
#ifndef __clang__
#pragma comment(linker, "/EXPORT:" __FUNCTION__"=" __FUNCDNAME__)
#endif

    llama_free(embed_ctx);
    llama_model_free(embed_model);

    // This step is done via the llm_terminate() side
    // llama_backend_free();
}
