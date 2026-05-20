// minslm_cli - Cross-platform LLM inference benchmark
// Self-contained: no dependency on llm-infer.h/cpp, Windows headers, or hnswlib.
// Builds on Linux, WSL2, macOS, and Windows.
// Links directly against libllama (same as llama-cli).
//
// Usage: minslm_cli MODEL_PATH N_THREADS CUSTOM_PROMPT_FILE [v1|v2] [cpu] [stream]
//                   [-d N] [-sm none|layer|row] [--weight-budget MB] [-fa on|off|auto]

#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_GAMING_XBOX)
// On the Xbox GDKX games partition there is no usable stderr/stdout (the
// process has no console). All log output must be routed through the title's
// debug channel via OutputDebugStringA, which the developer kit surfaces in
// PIX, the dev-kit "Debug Spew" view, and a host-side debugger.
//
// We install this callback unconditionally on Xbox, replacing both the
// default stderr-writing callback and the silence callbacks the verbose
// gating logic uses on PC.
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>

static void minslm_ods_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (!text) return;
    const char * lvl = "    ";
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: lvl = "ERR "; break;
        case GGML_LOG_LEVEL_WARN:  lvl = "WRN "; break;
        case GGML_LOG_LEVEL_INFO:  lvl = "INF "; break;
        case GGML_LOG_LEVEL_DEBUG: lvl = "DBG "; break;
        case GGML_LOG_LEVEL_CONT:  lvl = "    "; break;
        default: break;
    }
    // Mirror to stderr first so the existing console-style output (the
    // build's normal log destination) keeps working over telnet / kit
    // console. fputs is what ggml's default callback uses, so format stays
    // identical for that channel.
    fputs(text, stdout);
    fflush(stdout);
    OutputDebugStringA(text);
}
#endif // _GAMING_XBOX

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t timer_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::string trim(const std::string & str) {
    if (str.empty()) return str;
    size_t start = 0, end = str.length();
    while (start < end && isspace((unsigned char)str[start])) start++;
    while (end > start && isspace((unsigned char)str[end - 1])) end--;
    return str.substr(start, end - start);
}

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool parse_flash_attn_mode(const char * arg, llama_flash_attn_type & out) {
    if (!arg) {
        return false;
    }

    const std::string v = to_lower(trim(arg));
    if (v == "on" || v == "1" || v == "true" || v == "yes" || v == "enabled") {
        out = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return true;
    }
    if (v == "off" || v == "0" || v == "false" || v == "no" || v == "disabled") {
        out = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return true;
    }
    if (v == "auto") {
        out = LLAMA_FLASH_ATTN_TYPE_AUTO;
        return true;
    }

    return false;
}

static std::vector<llama_token> tokenize(const llama_context * ctx, const std::string & text,
                                         bool add_special, bool parse_special) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_tokens = (int)text.length() + 2 * (int)add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), (int)text.length(), result.data(), (int)result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        llama_tokenize(vocab, text.data(), (int)text.length(), result.data(), (int)result.size(), add_special, parse_special);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

static std::string token_to_piece(const llama_context * ctx, llama_token token) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string piece;
    piece.resize(piece.capacity());
    const int n = llama_token_to_piece(vocab, token, &piece[0], (int)piece.size(), 0, false);
    if (n < 0) {
        piece.resize(-n);
        llama_token_to_piece(vocab, token, &piece[0], (int)piece.size(), 0, false);
    } else {
        piece.resize(n);
    }
    return piece;
}

// ---------------------------------------------------------------------------
// Custom prompt file parser (same format as minslm.cpp)
// ---------------------------------------------------------------------------

struct prompt_file_t {
    std::string custom_template_prompt;
    std::vector<std::string> prompts;
};

static bool parse_prompt_file(const char * path, prompt_file_t & pf) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open prompt file '%s'\n", path);
        return false;
    }

    std::string line;
    bool in_template = false;
    bool in_prompts = false;

    while (std::getline(f, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") { in_template = true; in_prompts = false; continue; }
        if (line == "CUSTOM_PROMPT")          { in_prompts = true; in_template = false; continue; }
        if (line == "END_SECTION")            { in_template = false; in_prompts = false; continue; }

        if (in_template) {
            pf.custom_template_prompt += line + '\n';
        } else if (in_prompts) {
            pf.prompts.push_back(line);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct cli_params {
    std::string model_path;
    std::string prompt_file = "custom_prompts.txt";
    int32_t n_threads    = 4;
    int32_t n_ctx        = 2048;
    int32_t n_batch      = 512;
    int32_t n_len        = 1532;
    int32_t main_gpu     = 0;
    int32_t split_mode   = -1;  // -1=default, 0=none, 1=layer, 2=row
    int32_t weight_budget_mb = 0;  // 0=unlimited (all layers resident)
    int     verbose      = 0;
    bool    streaming    = false;
    bool    force_cpu    = false;
    bool    no_mmap      = false;
    bool    direct_io    = true;
    // Flash Attention default. On the GDKX/Xbox build the non-FA path
    // (separate QK matmul + softmax + attn*V) currently outperforms the FA
    // shader for prompt processing on RDNA2, so we default to OFF there.
    // PC builds default to AUTO and let the runtime decide based on the
    // backend's supports_op (see src/llama-context.cpp's auto_fa block).
    // Either default can be overridden via `-fa on|off|auto`.
#if defined(_GAMING_XBOX)
    enum llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
#else
    enum llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
#endif
};

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    cli_params p;

    if (argc < 2 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [N_THREADS] [PROMPT_FILE] [v1|v2|stream|cpu|-d N|-sm none|layer|row|--weight-budget MB|-fa on|off|auto]\n", argv[0]);
        return 1;
    }

    p.model_path = argv[1];

    if (argc >= 3) {
        int n = atoi(argv[2]);
        if (n <= 0) {
            n = (int)std::thread::hardware_concurrency();
            n = (n <= 4) ? n : (n / 2);
            if (n <= 0) n = 4;
        }
        p.n_threads = n;
    }

    if (argc >= 4) {
        p.prompt_file = argv[3];
    }

    // Parse optional flags (argv[4] onwards)
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "v1"))          { p.verbose = 1; }
        else if (!strcmp(argv[i], "v2"))     { p.verbose = 2; }
        else if (!strcmp(argv[i], "verbose")){ p.verbose = 1; }
        else if (!strcmp(argv[i], "stream")) { p.streaming = true; }
        else if (!strcmp(argv[i], "cpu"))    { p.force_cpu = true; }
        else if (!strcmp(argv[i], "--no-mmap") || !strcmp(argv[i], "-nm")) { p.no_mmap = true; }
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) { p.main_gpu = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-sm") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "none"))       p.split_mode = 0;
            else if (!strcmp(argv[i], "layer")) p.split_mode = 1;
            else if (!strcmp(argv[i], "row"))   p.split_mode = 2;
        }
        else if (!strcmp(argv[i], "--weight-budget") && i + 1 < argc) {
            p.weight_budget_mb = atoi(argv[++i]);
        }
        else if ((!strcmp(argv[i], "-fa") || !strcmp(argv[i], "--flash-attn")) && i + 1 < argc) {
            i++;
            if      (!strcmp(argv[i], "on")   || !strcmp(argv[i], "1") || !strcmp(argv[i], "true"))  p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            else if (!strcmp(argv[i], "off")  || !strcmp(argv[i], "0") || !strcmp(argv[i], "false")) p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            else if (!strcmp(argv[i], "auto"))                                                       p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
            else {
                fprintf(stderr, "Error: -fa expects on|off|auto, got '%s'\n", argv[i]);
                return 1;
            }
        }
        else {
            fprintf(stderr, "Warning: unknown argument '%s' ignored\n", argv[i]);
        }
    }

    // --- Parse prompt file ---
    prompt_file_t pf;
    if (!parse_prompt_file(p.prompt_file.c_str(), pf)) return 1;
    printf("[%s]: loaded %zu prompt(s) from '%s'\n", __func__, pf.prompts.size(), p.prompt_file.c_str());

#if defined(_GAMING_XBOX)
    // Route all llama / ggml log output to OutputDebugStringA. Installed
    // once and never overridden -- the verbose-gated silence path below is
    // skipped on Xbox so this callback stays active for the whole run.
    llama_log_set(minslm_ods_log_callback, nullptr);
#endif    

    // --- Initialize llama backend ---
    llama_backend_init();
    // ggml_backend_load_all();

#if !defined(_GAMING_XBOX)
    // Quiet mode unless verbose
    if (p.verbose < 2) {
        llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    }
#endif

    // --- Load model ---
    // Set weight budget env var if specified (picked up by load_all_data)
    if (p.weight_budget_mb > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", p.weight_budget_mb);
#ifdef _WIN32
        _putenv_s("GGML_WEIGHT_BUDGET_MB", buf);
#else
        setenv("GGML_WEIGHT_BUDGET_MB", buf, 1);
#endif
        printf("[%s]: weight budget = %d MiB (layer windowing enabled)\n", __func__, p.weight_budget_mb);
    }

    llama_model_params model_params = llama_model_default_params();
    if (p.force_cpu) {
        model_params.n_gpu_layers = 0;
    } else {
        model_params.n_gpu_layers = 999;
    }
    model_params.main_gpu = p.main_gpu;
    if (p.split_mode >= 0) {
        model_params.split_mode = (enum llama_split_mode)p.split_mode;
    }
    if (p.no_mmap) {
        model_params.use_mmap = false;
    }
    if (p.direct_io) {
        model_params.use_direct_io = true;
    }

    llama_model * model = llama_model_load_from_file(p.model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: failed to load model '%s'\n", p.model_path.c_str());
        return 1;
    }

    // --- Create context ---
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = p.n_ctx;
    ctx_params.n_batch  = p.n_batch;
    ctx_params.n_threads       = p.n_threads;
    ctx_params.n_threads_batch = p.n_threads;
    ctx_params.no_perf  = false;
    ctx_params.flash_attn_type = p.flash_attn_type;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    printf("\n[%s]: n_ctx = %d, n_threads = %d, gpu_layers = %d\n",
           __func__, llama_n_ctx(ctx), p.n_threads, model_params.n_gpu_layers);
    printf("[%s]: flash-attn = %s%s\n",
           __func__,
           llama_flash_attn_type_name(p.flash_attn_type),
           p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? " (runtime-selected)" : "");
    printf("[%s]: system_info: %s\n", __func__, llama_print_system_info());

    // NOTE: upstream removed llama_memory_breakdown_print() from the public
    // libllama API (PR #22171, "fit-params: refactor"). The breakdown printer
    // now lives in libcommon as common_memory_breakdown_print(). minslm-cli
    // is intentionally self-contained (links only against `llama`, not
    // `common`), so we skip the printout here rather than pull in libcommon.
    // If you want the breakdown back, add `common` to minslm-cli's
    // target_link_libraries and call common_memory_breakdown_print(ctx).
    printf("\n");

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // --- Run prompts ---
    int64_t t_total_start = timer_us();
    int total_tokens_generated = 0;
    int total_tokens_emitted = 0;
    int64_t total_tg_us = 0;
    int64_t total_tg_core_us = 0;
    int64_t total_ttft_us = 0;

    for (size_t pi = 0; pi < pf.prompts.size(); pi++) {
        std::string user_msg = trim(pf.prompts[pi]);
        user_msg.erase(std::remove(user_msg.begin(), user_msg.end(), '\"'), user_msg.end());

        // Build full prompt: substitute {message} in template (same as minslminfer)
        std::string full_prompt = trim(pf.custom_template_prompt);
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            full_prompt.replace(pos, 9, user_msg);
        }

        printf("> Running with custom prompt => [%zu/%zu]: [%s]\n",
               pi + 1, pf.prompts.size(), user_msg.c_str());

        // Clear KV cache for each prompt (no prefix-cache in this version)
        llama_memory_clear(llama_get_memory(ctx), true);

        // Tokenize (add_special=false, parse_special=true — same as minslminfer)
        std::vector<llama_token> tokens = tokenize(ctx, full_prompt, false, true);

        // --- Prompt eval (prefill) ---
        int64_t t1 = timer_us();
        for (int i = 0; i < (int)tokens.size(); i += p.n_batch) {
            int n_eval = std::min((int)tokens.size() - i, p.n_batch);
            if (llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval))) {
                fprintf(stderr, "Error: llama_decode failed during prefill\n");
                goto cleanup;
            }
        }
        // Keep prefill/decode accounting explicit when backends run asynchronously.
        llama_synchronize(ctx);
        int64_t t2 = timer_us();

        if (p.verbose >= 2) {
            float pp_ms = (t2 - t1) / 1000.0f;
            printf("  prefill: %.1fms (%zu tokens, %.1f t/s)\n",
                   pp_ms, tokens.size(), tokens.size() * 1000.0f / pp_ms);
        }

        // --- Token generation (decode) ---
        {
            int n_past = (int)tokens.size();
            int max_gen = std::min(p.n_len - n_past, 128);
            std::string output;
            int n_gen = 0;
            int n_emit = 0;
            size_t empty_pieces = 0;
            std::vector<llama_token> empty_ids; empty_ids.reserve(16);
            std::vector<llama_token> all_ids;   all_ids.reserve((size_t)max_gen);

            auto sparams = llama_sampler_chain_default_params();
            sparams.no_perf = false;
            llama_sampler * smpl = llama_sampler_chain_init(sparams);
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(128, 1.3f, 0.1f, 0.1f));
            // DRY: penalizes repeated n-gram sequences
            const char * dry_breakers[] = { "\n", ":", "\"", "*" };
            llama_sampler_chain_add(smpl, llama_sampler_init_dry(vocab, 2048, 0.8f, 1.75f, 2, 128,
                                          dry_breakers, 4));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1.0f));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.3f));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

            int64_t t3 = timer_us();
            int64_t t_first_tok = 0;

            while (n_gen < max_gen) {
                llama_token id = llama_sampler_sample(smpl, ctx, -1);

                if (llama_vocab_is_eog(vocab, id)) {
                    if (p.streaming) printf("\n");
                    break;
                }

                if (t_first_tok == 0) {
                    t_first_tok = timer_us();
                }
                n_emit++;

                std::string piece = token_to_piece(ctx, id);
                all_ids.push_back(id);
                if (piece.empty()) {
                    ++empty_pieces;
                    if (empty_ids.size() < 16) empty_ids.push_back(id);
                }
                if (p.streaming) {
                    printf("%s", piece.c_str());
                    fflush(stdout);
                } else {
                    output += piece;
                }

                // Stop on closing brace (end of JSON/structured output)
                if (piece.find('}') != std::string::npos) {
                    if (p.streaming) printf("\n");
                    break;
                }

                if (llama_decode(ctx, llama_batch_get_one(&id, 1))) {
                    fprintf(stderr, "Error: llama_decode failed during generation\n");
                    llama_sampler_free(smpl);
                    goto cleanup;
                }

                n_gen++;
                n_past++;
            }

            // Drain outstanding decode work before stopping the decode timer.
            llama_synchronize(ctx);
            int64_t t4 = timer_us();
            int64_t tg_us = t4 - t3;
            int64_t ttft_us = t_first_tok > 0 ? (t_first_tok - t3) : 0;
            int64_t tg_core_us = t_first_tok > 0 ? (t4 - t_first_tok) : 0;
            total_tokens_generated += n_gen;
            total_tokens_emitted += n_emit;
            total_tg_us += tg_us;
            total_tg_core_us += tg_core_us;
            total_ttft_us += ttft_us;

            llama_sampler_free(smpl);

            if (!p.streaming) {
                if (!output.empty()) {
                    // Use fwrite to handle embedded NUL bytes from byte-fallback tokens.
                    // Also escape unprintable control chars so log viewers don't truncate.
                    std::string esc;
                    esc.reserve(output.size() + 16);
                    size_t nuls = 0;
                    size_t ctrl = 0;
                    for (unsigned char c : output) {
                        if (c == 0) {
                            esc += "\\x00";
                            ++nuls;
                        } else if (c < 0x20 && c != '\n' && c != '\t' && c != '\r') {
                            char buf[5];
                            snprintf(buf, sizeof(buf), "\\x%02X", c);
                            esc += buf;
                            ++ctrl;
                        } else {
                            esc += (char)c;
                        }
                    }
                    fwrite(esc.data(), 1, esc.size(), stdout);
                    putchar('\n');
                    fflush(stdout);
                    if (p.verbose >= 2 && (nuls || ctrl || empty_pieces)) {
                        printf("  [output diag: bytes=%zu nuls=%zu other_ctrl=%zu empty_pieces=%zu/%d]\n",
                               output.size(), nuls, ctrl, empty_pieces, n_gen);
                        if (empty_pieces) {
                            printf("  [empty piece token ids (first %zu):",
                                   empty_ids.size());
                            for (auto t : empty_ids) printf(" %d", t);
                            printf("]\n");
                        }
                    }
                } else if (p.verbose >= 2 && !all_ids.empty()) {
                    printf("  [output diag: empty output, %zu tokens all produced empty pieces]\n",
                           all_ids.size());
                }
                // Print full token-id sequence for off-line decoding/inspection
                if (p.verbose >= 2 && !all_ids.empty()) {
                    printf("  [all token ids (%zu):", all_ids.size());
                    for (auto t : all_ids) printf(" %d", t);
                    printf("]\n");
                }
                // Hex dump of the raw output bytes (16 bytes/line)
                if (p.verbose >= 2 && !output.empty()) {
                    printf("  [output hex (%zu bytes):]\n", output.size());
                    for (size_t off = 0; off < output.size(); off += 16) {
                        printf("    %04zx:", off);
                        size_t end = std::min(off + 16, output.size());
                        for (size_t i = off; i < end; ++i) {
                            printf(" %02x", (unsigned char)output[i]);
                        }
                        // pad alignment for partial lines
                        for (size_t i = end - off; i < 16; ++i) printf("   ");
                        printf("  |");
                        for (size_t i = off; i < end; ++i) {
                            unsigned char c = (unsigned char)output[i];
                            putchar((c >= 0x20 && c < 0x7f) ? (char)c : '.');
                        }
                        printf("|\n");
                    }
                }
            }

            if (p.verbose >= 2 && n_gen > 0) {
                printf("  decode: %.1fms (%d tokens, %.2f t/s)\n",
                       tg_us / 1000.0f, n_gen, n_gen / (tg_us / 1000000.0f));
                if (n_emit > 0 && tg_core_us > 0) {
                    printf("  decode (no-TTFT): %.1fms (%d tokens, %.2f t/s) [TTFT %.1fms]\n",
                           tg_core_us / 1000.0f, n_emit, n_emit / (tg_core_us / 1000000.0f), ttft_us / 1000.0f);
                }
            }
        }

        if (p.verbose >= 1) printf("\n");
    }

    {
        int64_t t_total = timer_us() - t_total_start;
        printf("\n===== Summary =====\n");
        printf("  prompts:    %zu\n", pf.prompts.size());
        printf("  tokens gen: %d\n", total_tokens_generated);
        if (total_tokens_generated > 0) {
            printf("  avg t/s:    %.2f\n", total_tokens_generated / (total_tg_us / 1000000.0));
        }
        if (total_tokens_emitted > 0 && total_tg_core_us > 0) {
            printf("  avg t/s (no-TTFT): %.2f\n", total_tokens_emitted / (total_tg_core_us / 1000000.0));
            printf("  avg TTFT:          %.1fms\n", (double) total_ttft_us / (double) pf.prompts.size() / 1000.0);
        }
        printf("  total time: %.2fs\n", t_total / 1000000.0);
    }

    // Re-enable llama logging for perf report
#if !defined(_GAMING_XBOX)
    llama_log_set(nullptr, nullptr);
#endif
    llama_perf_context_print(ctx);

    // Print DX12 per-op perf stats if GGML_DX12_PERF was set
    {
        typedef void (*perf_fn_t)();
        for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            auto fn = (perf_fn_t)ggml_backend_reg_get_proc_address(reg, "ggml_cpu_print_tensor_op_perf");
            if (fn) { fn(); break; }
        }
    }

cleanup:
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
