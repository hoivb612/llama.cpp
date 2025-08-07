#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "xbapp.h"

llama_context *ctx;
llama_context_params ctx_params;
llama_model *model;
llama_model_params model_params;
static int total_tokens_generated = 0;
bool save_slm_state = false;
std::vector<llama_token> session_tokens;
static int64_t t_token_generation_ms = 0;
std::vector<llama_token> tokens_shared;

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
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

std::string pfx_file_path(std::string pfx) {
#ifdef _WIN32

    static std::hash<std::string> hasher;
    static std::string dir = "./llama_cache";
    
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

#else

    static std::hash<std::string> hasher;
    static const char* dir = "./llama_cache";

    // create the cache dir if it does not exist yet
    struct stat info;
    if (stat(dir, &info) != 0) {
        mkdir(dir, 0777);
    }

    // default generated file name
    std::string pfx_path(dir);
    std::string full_file_path = pfx_path + "/" + std::to_string(hasher(pfx));

#endif // _WIN32

    return full_file_path;
}

int slm_init(xbapp_params& xbparams) {
    // init LLM
    llama_backend_init();

    // initialize the model
    model_params = llama_model_default_params();
    model_params.n_gpu_layers = xbparams.n_ngl;

    model = llama_model_load_from_file(xbparams.model_path.c_str(), model_params);
    if (model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    ctx_params = llama_context_default_params();

    ctx_params.n_ctx = xbparams.n_ctx;
    if ((xbparams.n_len + xbparams.n_seqlen - 1) > xbparams.n_ctx) {
        printf("%s: context size (%d) < prompt (%d) + seq generated length (%d)\n",
            __func__, xbparams.n_ctx, xbparams.n_len, xbparams.n_seqlen);
        return 1;
    }
    // n_batch is the max number of tokens processed in a single call to llama_decode()
    ctx_params.n_batch = xbparams.n_batch;
    ctx_params.n_threads = xbparams.n_threads;
    ctx_params.n_threads_batch = xbparams.n_threads;
    // enable perf counters
    ctx_params.no_perf = false;

    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("\n%s: n_len = %d, n_ctx = %d\n", __func__, xbparams.n_len, llama_n_ctx(ctx));
    printf("%s: n_threads = %d, n_threads_batch = %d\n\n", __func__, ctx_params.n_threads, ctx_params.n_threads_batch);

    if (xbparams.pfc_mode) {
        // start from a known point
        llama_kv_self_clear(ctx);

        std::string template_prompt = xbparams.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            xbparams.pfx_shared = ::trim(template_prompt.substr(0, pos));
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = slm_tokenize(ctx, xbparams.pfx_shared, false, true);

#if 1 // use llama_state_load_file()
            // build the cache file directory
            xbparams.pfx_file = pfx_file_path(xbparams.pfx_shared);
            // load the cache and create one if it does not exist
            session_tokens.resize(xbparams.n_ctx);
            size_t n_token_count_out = xbparams.first_prompt ? 0xffffffff : 0;
            if (!llama_state_load_file(ctx, 
                                       xbparams.pfx_file.c_str(),
                                       session_tokens.data(),
                                       session_tokens.capacity(),
                                       &n_token_count_out)) {
                printf("%s: Load state file failed: %s\n", __func__, xbparams.pfx_file.c_str());
                session_tokens.resize(0);
                save_slm_state = true;
                tokens_shared.clear();
                xbparams.pfx_shared = "";

                // for now this plug-in should not create cache files - comment this out for cache generation
                // return 1;

            } else {
                printf("%s: Loading saved state from '%s' (size %zd)...\n", __func__, xbparams.pfx_file.c_str(), tokens_shared.size());
                session_tokens.resize(n_token_count_out);
                // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, session_tokens).c_str());

                // sanity check
                GGML_ASSERT(tokens_shared.size() <= session_tokens.size());
                for (size_t i = 0; i < tokens_shared.size(); i++) {
                    if (tokens_shared[i] != session_tokens[i]) {
                        printf("Mismatched pfx tokens [%zd]-%2X %2X %2X-%2X %2X %2X!!!!\n", i, 
                                tokens_shared[i-1], tokens_shared[i], tokens_shared[i+1],
                                session_tokens[i-1], session_tokens[i], session_tokens[i+1]);
                        return 1;
                    }
                }

                //printf("%s: token_shared=%zd - %s\n", __func__, tokens_shared.size(), LOG_TOKENS_TOSTR_PRETTY(ctx, tokens_shared).c_str());

                // remove any "future" tokens that we might have inherited from the previous session
                llama_kv_self_seq_rm(ctx, -1, tokens_shared.size(), -1);
            }

#else // use llama_set_state_data()

            printf("%s: Loading saved state...\n", __func__);
            FILE *slm_state_fp = fopen("./slm_state.bin", "rb");
            if (slm_state_fp != NULL) {
                auto state_size = llama_state_get_size(ctx);
                //if (state_size != state_size2) {
                //    cerr << "state size differs\n";
                //}
                auto state_mem = new uint8_t[state_size];
                fread(state_mem, 1, state_size, slm_state_fp);
                llama_state_set_data(ctx, state_mem);  // could also read directly from memory mapped file
                fclose(slm_state_fp);
                printf("%s: saved state loaded successfully\n", __func__);
            } else {
                printf("%s: slm state file not available - set flag to save it later...\n", __func__);
                save_slm_state = true;
            }

#endif

        } else {
            // no shared prompt detected
            tokens_shared.clear();
        }

    } else {
        // No pfc mode
        tokens_shared.clear();
    }

    return 0;
}

int slm_inference(xbapp_params& xbparams) {
    std::vector<llama_token> embd_inp;
    int n_past = 0;
    int n_kv_pfx = 0;

    if (xbparams.pfc_mode) {
        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_self_seq_rm(ctx, -1, tokens_shared.size(), -1);
        embd_inp.insert(embd_inp.end(), tokens_shared.begin(), tokens_shared.end());
        n_past = tokens_shared.size();
        n_kv_pfx = tokens_shared.size();

        // re-apply the template since it was destroyed in pfc mode
        xbparams.prompt.append("\"\n<|end|>\n<|Assistant|>\nYou:");
    } else {
        // start from a known point for each new user prompt
        llama_kv_self_clear(ctx);
        n_past = 0;
        n_kv_pfx = 0;
    }

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_input = slm_tokenize(ctx, xbparams.prompt, false, true);

    // append the variant part of the prompt or the full prompt for non pfc mode
    embd_inp.insert(embd_inp.end(), tokens_input.begin(), tokens_input.end());

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_input.size() + (xbparams.n_len - tokens_input.size() - n_kv_pfx);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req(%d-%d) > n_ctx(%d), the required KV cache size is not big enough\n",
            __func__, n_kv_pfx, n_kv_req, n_ctx);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // printf("%s: before eval n_past=%d, eval: %s\n", __func__, n_past, LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // calculate how much has been processed through the saved state file
    int prompt_index = 0;
    if (xbparams.pfc_mode) {
        int n_tokens_processed = 0;
        for (; prompt_index < embd_inp.size(); prompt_index++) {
            // not fully matched with shared tokens
            if (embd_inp[prompt_index] != tokens_shared[prompt_index]) {
                break;
            }

            n_tokens_processed++;

            // embd_inp is fully matched with shared prefix cache
            if (n_tokens_processed >= (int)tokens_shared.size()) {
                ++prompt_index;
                break;
            }
        }

        // printf("%s: pfc mode - tokens processed =%d - prompt_index=%d - n_past=%d\n", __func__, n_tokens_processed, prompt_index, n_past);
    }

    // build token list for inference
    std::vector<llama_token> embd;
    for (int i = prompt_index; i < embd_inp.size(); i++) {
        embd.push_back(embd_inp[i]);
    }
    // printf("%s: decode: %s\n", __func__, LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

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
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(xbparams.seed));
    // llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // printf("%s: start decoding @n_past = %d - inference size = %zd\n", __func__, n_past, embd.size());
    int64_t t_start_decoding = ggml_time_us();

    // decode the remaining prompt not covered by the shared portion
    for (int i = 0; i < (int)embd.size(); i += xbparams.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > xbparams.n_batch) {
            n_eval = xbparams.n_batch;
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
            printf("%s : failed to eval\n", __func__);
            return 1;
        }

        n_past += n_eval;
        // printf("%s: decoded %d tokens\n", __func__, n_eval);
    }

    int64_t t_start_generation = ggml_time_us();
    float t_prompt_eval_ms = (t_start_generation - t_start_decoding) / 1000.0f;
    printf("Prompt TTFT = %.2fms (size = %zu) (%.2ft/s) (%.2fms)\n", 
        t_prompt_eval_ms, 
        embd.size(), 
        (embd.size() * 1000.0f) / t_prompt_eval_ms, 
        t_prompt_eval_ms / embd.size());

    if (xbparams.pfc_mode && save_slm_state) {
        session_tokens.insert(session_tokens.end(), embd_inp.begin(), embd_inp.end());
    }

    if (save_slm_state) {
#if 1 // use llama_state_load_file()
        llama_state_save_file(ctx, xbparams.pfx_file.c_str(), session_tokens.data(), session_tokens.size());
        save_slm_state = false;

        // update token_shared
        std::string template_prompt = xbparams.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            xbparams.pfx_shared = ::trim(template_prompt.substr(0, pos));
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = slm_tokenize(ctx, xbparams.pfx_shared, false, true);
        }

#else // use llama_set_state_data()
        // save state (rng, logits, embedding and kv_cache) to file
        printf("%s: saving SLM state...\n", __func__);
        FILE *slm_state_fp = fopen("./slm_state.bin", "wb");
        auto state_size = llama_state_get_size(ctx);
        auto state_mem = new uint8_t[state_size];
        llama_state_get_data(ctx, state_mem);
        fwrite(state_mem, 1, state_size, slm_state_fp);
        fclose(slm_state_fp);
        save_slm_state = false;
        printf("%s: DONE saving SLM state...\n", __func__);
#endif
    }

    // compute max_len output
    int max_len = std::min(xbparams.n_len, (n_past + xbparams.n_seqlen));

    std::string slm_output;
    bool valid_reply = false;
    int n_tokens_generated = 0;

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    while (n_past <= max_len) {

        // sample the last token just received
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

        {
            // is it an end of generation - are we done?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            const std::string token_str = slm_token_to_piece(ctx, new_token_id);

            if (token_str.find('{') != std::string::npos) {
                // accepted answers have '{' characters
                valid_reply = true;
            }

            if (valid_reply) {
#if 0
            // enable the following printf for streaming replies
            printf("%s", token_str.c_str());
#else
            // batched output
            slm_output += token_str;
#endif
            }

            if (token_str.find('}') != std::string::npos) {
                // force end of output since we have a valid JSON reply
                break;
            }

            n_tokens_generated += 1;
            total_tokens_generated += 1;
        }

        // bump current generated token index
        n_past += 1;

        // decode the output for the new generated token
        if (llama_decode(ctx, llama_batch_get_one(&new_token_id, 1))) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    // we have reached max_len of output, hit eog char or "}"
    if (!valid_reply) {
        // reply not correctly formatted or unhelpful
        printf("%s: ***** invalid formatted reply from model *****\n", __func__);
    }

    printf("%s\n", slm_output.c_str());
    slm_output.clear();
    valid_reply = false;
    fflush(stdout);

    int64_t t_end_generation = ggml_time_us();
    double t_ms = (t_end_generation - t_start_generation) / 1000.0f;
    printf("> token generation time = %.2fms (%d) (%.2ft/s) (%.2fms)\n", 
        t_ms,
        n_tokens_generated, 
        (t_ms == 0) ? 0.0 : n_tokens_generated / (t_ms / 1000.0f),
        (t_ms / n_tokens_generated));

    t_token_generation_ms += t_ms;
    return 0;
}

void slm_terminate() {
    printf("\n");

    printf("%s: generated %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, 
            total_tokens_generated, (t_token_generation_ms / 1000.0f), 
            (t_token_generation_ms == 0) ? 0.0: total_tokens_generated / (t_token_generation_ms / 1000.0f));

    llama_perf_context_print(ctx);

    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();
}

