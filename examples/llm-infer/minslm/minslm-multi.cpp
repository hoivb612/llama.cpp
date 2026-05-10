//
// minslm-multi.cpp — Multi-turn conversation driver for llm-infer
//
// Unlike minslm.cpp which resets the KV cache per prompt (single-turn),
// this version accumulates the KV cache across turns and supports
// REWIND N commands to drop the last N turns.
//
// Usage: minslm-multi MODEL_PATH #_threads CUSTOM_PROMPT_file [pfc] [stream] [verbose|v1|v2] ...
//
// Prompt file format:
//   CUSTOM_TEMPLATE_PROMPT       — full template with {message} placeholder (first turn)
//   CUSTOM_TURN_TEMPLATE         — user-turn-only template for subsequent turns (optional, auto-derived)
//   CUSTOM_PROMPT                — list of prompts + REWIND N directives
//

#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "llm-infer.h"
#include "b612-cpu.h"

#include "log.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

std::vector<std::string> custom_prompts;
std::vector<std::string>::iterator custom_prompts_it;
std::string custom_turn_template = "";

namespace console {
    enum display_t {
        reset = 0,
        prompt,
        stats,
        error
    };

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_t display);
}

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_BOLD          "\x1b[1m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"

namespace console {

    static bool      color_display    = false;
    static display_t current_display  = reset;
    static FILE*     out              = stdout;
    static void*     hConsole;

    void init(bool use_color) {
        color_display = use_color;

        DWORD dwMode = 0;
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
            if (hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(hConsole, &dwMode))) {
                hConsole = nullptr;
            }
        }
        if (hConsole) {
            if (color_display && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
                !SetConsoleMode(hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                color_display = false;
            }
            SetConsoleOutputCP(CP_UTF8);
        }
    }

    void cleanup() {
        set_display(reset);
    }

    void set_display(display_t display) {
        if (color_display && current_display != display) {
            fflush(stdout);
            switch(display) {
                case reset:
                    fprintf(out, ANSI_COLOR_RESET);
                    break;
                case stats:
                    fprintf(out, ANSI_COLOR_YELLOW);
                    break;
                case prompt:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                    break;
                case error:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_RED);
            }
            current_display = display;
            fflush(out);
        }
    }
} // namespace console

static int64_t timer_freq = 0, timer_start = 0;
void timer_init(void) {
    if (!timer_freq) {
        LARGE_INTEGER t;
        QueryPerformanceFrequency(&t);
        timer_freq = t.QuadPart;
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

int64_t timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}

static std::string trim(const std::string & str) {
    if (str.empty()) return str;
    size_t start = 0;
    size_t end = str.length();
    while ((start < end) && isspace(str[start])) start += 1;
    while ((end > start) && isspace(str[end - 1])) end -= 1;
    return str.substr(start, end - start);
}

bool processCustomPromptsFromFile(model_params& params) {
    std::ifstream cpfile(params.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, params.custom_p_file.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool turnTemplateMode = false;
    bool userPromptMode = false;
    custom_prompts.clear();
    std::string custom_template_prompt = "";
    custom_turn_template = "";

    while (std::getline(cpfile, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") {
            templatePromptMode = true;
            continue;
        } else if (line == "CUSTOM_TURN_TEMPLATE") {
            turnTemplateMode = true;
            continue;
        } else if (line == "CUSTOM_PROMPT") {
            userPromptMode = true;
            continue;
        } else if (line == "END_SECTION") {
            templatePromptMode = false;
            turnTemplateMode = false;
            userPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (turnTemplateMode) {
            custom_turn_template += line + '\n';
        } else if (userPromptMode) {
            custom_prompts.push_back(line);
        }
    }

    params.custom_template_prompt = custom_template_prompt;
    params.turn_template = custom_turn_template;

    cpfile.close();
    return true;
}

// Auto-derive the turn template from the full template by finding the user tag
void derive_turn_template(model_params& params) {
    if (!custom_turn_template.empty()) return;

    std::string tmpl = ::trim(params.custom_template_prompt);
    size_t msg_pos = tmpl.find("{message}");
    if (msg_pos == std::string::npos) return;

    std::string prefix = tmpl.substr(0, msg_pos);
    std::string suffix = tmpl.substr(msg_pos + std::string("{message}").length());

    // Find the last user tag in the prefix
    size_t user_tag = std::string::npos;
    size_t p;
    if ((p = prefix.rfind("<|user|>"))            != std::string::npos) user_tag = p;
    if ((p = prefix.rfind("<|im_start|>user"))    != std::string::npos && (user_tag == std::string::npos || p > user_tag)) user_tag = p;
    if ((p = prefix.rfind("[INST]"))              != std::string::npos && (user_tag == std::string::npos || p > user_tag)) user_tag = p;
    if ((p = prefix.rfind("<start_of_turn>user")) != std::string::npos && (user_tag == std::string::npos || p > user_tag)) user_tag = p;

    if (user_tag != std::string::npos) {
        custom_turn_template = prefix.substr(user_tag) + "{message}" + suffix;
        params.turn_template = custom_turn_template;
        printf("[%s]: auto-derived turn template: [%s]\n", __func__,
               ::trim(custom_turn_template).c_str());
    }
}

void print_system_info(int32_t n_threads, int32_t n_batch) {
    std::ostringstream os;

    os << "system_info: n_threads = " << n_threads;
    if (n_threads != -1) {
    os << " (n_batch = " << n_batch << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__)
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " LPs | " << llm_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llm_system_info();
#endif

    printf("\n%s: %s\n\n", __func__, os.str().c_str());
}

int main(int argc, char** argv) {
    model_params params = {0};

    timer_init();
    int64_t t0 = timer_us();

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s arg_MODEL_PATH arg_#_threads arg_CUSTOM_PROMPT_file [pfc] [paffin] [stream] [verbose] [repack-xbox | repack-ggml] [--weight-budget MB]\n", argv[0]);
        return 1;
    }

    if (argc >= 2) {
        params.model_name = argv[1];
    }

    params.n_threads = 4;
    if (argc >= 3) {
        int32_t n_threads = std::stoi(argv[2]);
        if (n_threads <= 0) {
            n_threads = std::thread::hardware_concurrency();
            if (n_threads > 0) {
                n_threads = (n_threads <= 4) ? n_threads : (n_threads / 2);
            } else {
                n_threads = 4;
            }
        }
        params.n_threads = n_threads;
    }

    if (argc >= 4) {
        params.custom_p_file = argv[3];
    }

    while (argc >= 5) {
        if (!strcmp(argv[4], "verbose")) {
            params.verbose = 1;

        } else if (!strcmp(argv[4], "v1")) {
            params.verbose = 1;

        } else if (!strcmp(argv[4], "v2")) {
            params.verbose = 2;

        } else if (!strcmp(argv[4], "paffin")) {
            params.process_affinity = true;

        } else if (!strcmp(argv[4], "pfc")) {
            params.pfc_mode = true;

        } else if (!strcmp(argv[4], "repack-ggml")) {
            params.tensor_repack_mode = 1;

        } else if (!strcmp(argv[4], "repack-xbox")) {
            params.tensor_repack_mode = 2;

        } else if (!strcmp(argv[4], "repack-xbcg")) {
            params.tensor_repack_mode = 3;

        } else if (!strcmp(argv[4], "repack-xbox-st")) {
            params.tensor_repack_mode = 4;

        } else if (!strcmp(argv[4], "mulmat-xbox")) {
            params.tensor_repack_mode = 5;

        } else if (!strcmp(argv[4], "stream")) {
            params.streaming_reply = true;

        } else if (!strcmp(argv[4], "add-special")) {
            params.add_special = true;

        } else if (!strcmp(argv[4], "parse-special")) {
            params.parse_special = true;

        } else if (!strcmp(argv[4], "cpu")) {
            params.force_cpu_mode = true;

        } else if (!strcmp(argv[4], "-d") && argc >= 6) {
            params.main_gpu = atoi(argv[5]);
            argv += 1; argc -= 1;

        } else if (!strcmp(argv[4], "-sm") && argc >= 6) {
            if (!strcmp(argv[5], "none"))       params.split_mode = 0;
            else if (!strcmp(argv[5], "layer")) params.split_mode = 1;
            else if (!strcmp(argv[5], "row"))   params.split_mode = 2;
            argv += 1; argc -= 1;

        } else if (!strcmp(argv[4], "--weight-budget") && argc >= 6) {
            params.weight_budget_mb = atoi(argv[5]);
            argv += 1; argc -= 1;
        }

        argv += 1;
        argc -= 1;
    }

    console::init(true);
    printf("[%s]: processing cpf input file [%s]\n", __func__, params.custom_p_file.c_str());
    processCustomPromptsFromFile(params);
    custom_prompts_it = custom_prompts.begin();

    llm_disable_log();

    print_system_info(params.n_threads, params.n_batch);

    if (!params.parse_special) {
        params.parse_special = true;
    }

    if (params.process_affinity) {
        int64_t cpu_affinity_mask = ggml_b612::xb_set_optimal_process_affinity(params.n_threads, params.verbose);
        printf("[%s]: Setting process affinity mask 0x%016llX\n", __func__, cpu_affinity_mask);
    }

    // Set weight budget env var if specified (picked up by load_all_data)
    if (params.weight_budget_mb > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", params.weight_budget_mb);
        _putenv_s("GGML_WEIGHT_BUDGET_MB", buf);
        printf("[%s]: weight budget = %d MiB (layer windowing enabled)\n", __func__, params.weight_budget_mb);
    }

    // Initialize the model
    if (!llm_initialize(params)) {
        printf("%s: Error during llm_initialize()\n", __func__);
        return 1;
    }

    // Auto-derive turn template from the full template if not explicitly provided
    derive_turn_template(params);

    // Start multi-turn session
    llm_multiturn_begin(params);
    printf("[%s]: multi-turn mode — KV cache will accumulate across turns\n", __func__);

    // Stop generation when '}' is encountered (for JSON responses)
    params.stop_char = '}';

    int prompt_index = 1;

    while (custom_prompts_it != custom_prompts.end())
    {
        std::string raw_line = ::trim(*custom_prompts_it);

        // Handle REWIND N command
        if (raw_line.substr(0, 6) == "REWIND") {
            int n_rewind = 1;
            if (raw_line.length() > 7) {
                n_rewind = std::stoi(raw_line.substr(7));
            }
            console::set_display(console::stats);
            printf("> REWIND %d (turns: %d, tokens: %d)\n",
                   n_rewind, llm_multiturn_turn_count(), llm_multiturn_token_count());
            llm_multiturn_rewind(n_rewind);
            printf("  after rewind: turns=%d, tokens=%d\n",
                   llm_multiturn_turn_count(), llm_multiturn_token_count());
            console::set_display(console::reset);
            custom_prompts_it++;
            continue;
        }

        // Format the user prompt
        std::string custom_prompt = raw_line;
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        // System prefix is pre-decoded in llm_multiturn_begin(), so ALL turns
        // (including the first) use the turn template — only the user portion.
        std::string tmpl;
        if (!custom_turn_template.empty()) {
            tmpl = ::trim(custom_turn_template);
        } else {
            // Fallback: use full template (system will be re-encoded — not ideal)
            tmpl = ::trim(params.custom_template_prompt);
        }

        size_t pos = tmpl.find("{message}");
        if (pos != std::string::npos) {
            tmpl.replace(pos, std::string("{message}").length(), custom_prompt);
        }

        params.prompt = tmpl;

        console::set_display(console::prompt);
        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        console::set_display(console::reset);

        if (!params.parse_special) {
            params.parse_special = true;
        }

        // Run multi-turn inference (KV cache accumulates)
        if (!llm_infer_multiturn(params)) {
            printf("Failed token generation for query: %s\n", params.prompt.c_str());
            params.reply = "";
        }

        if (params.streaming_reply) {
            printf("\n<<\n");
        } else {
            for (auto c: params.reply) {
                printf("%c", c);
                if (c == '}') {
                    break;
                }
            }
            printf("\n\n");
        }

        params.first_prompt = false;

        custom_prompts_it++;
    }

    t0 = timer_us() - t0;
    printf("\n\n total elapsed time %7.2fsec\n", (double)t0 / (1000. * 1000.));

    llm_enable_log();
    llm_terminate(params);

    console::set_display(console::stats);

    if (params.verbose == 2) {
        llm_print_tensor_op_perf_stats();
    }

    console::cleanup();

    return 0;
}
