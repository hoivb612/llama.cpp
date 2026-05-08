#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "llm-infer.h"
#include "b612-cpu.h"
#include "llama.h"

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

int current_custom_prompt_index = 0;
std::vector<std::string> custom_prompts;
std::vector<std::string>::iterator custom_prompts_it;
std::string custom_prompts_output;
bool switch_prompt = false; // set true every time switch to a new prompt

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

    //
    // Console state
    //

    static bool      color_display    = false;
    static display_t current_display  = reset;
    static FILE*     out              = stdout;
    static void*     hConsole;

    //
    // Init and cleanup
    //

    void init(bool use_color) {
        color_display = use_color;

        // Windows-specific console initialization
        DWORD dwMode = 0;
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
            if (hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(hConsole, &dwMode))) {
                hConsole = nullptr;
            }
        }
        if (hConsole) {
            // Check conditions combined to reduce nesting
            if (color_display && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
                !SetConsoleMode(hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                color_display = false;
            }
            // Set console output codepage to UTF8
            SetConsoleOutputCP(CP_UTF8);
        }
    }

    void cleanup() {
        // Reset console display
        set_display(reset);
    }

    //
    // Display and IO
    //

    // Keep track of current display and only emit ANSI code if it changes
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

        // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
        // and the uptime is high enough.
        // We subtract the program start time to reduce the likelihood of that happening.
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

int64_t timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    if (str.empty()) {
        return str; // Return early if the string is empty
    }

    size_t start = 0;
    size_t end = str.length();

    // trim leading white space
    while ((start < end) && isspace(str[start])) {
        start += 1;
    }

    // trim trailing white space
    while ((end > start) && isspace(str[end - 1])) {
        end -= 1;
    }

    assert(end >= start);
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
    bool userPromptMode = false;
    custom_prompts.clear();
    std::string custom_template_prompt = "";

    // process CUSTOM_TEMPLATE_PROMPT
    while (std::getline(cpfile, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") {
            templatePromptMode = true;
            continue;
        } else if (line == "CUSTOM_PROMPT") {
            userPromptMode = true;
            continue;
        } else if (line == "END_SECTION") {
            templatePromptMode = false;
            userPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (userPromptMode) {
            custom_prompts.push_back(line);
        }
    }

    params.custom_template_prompt = custom_template_prompt;

    cpfile.close();

    return true;
}

void print_system_info(int32_t n_threads, int32_t n_batch) {
    std::ostringstream os;

    os << "system_info: n_threads = " << n_threads;
    if (n_threads != -1) {
    os << " (n_batch = " << n_batch << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " LPs | " << llm_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llm_system_info();
#endif

    printf("\n%s: %s\n\n", __func__, os.str().c_str());
}

// ─── Chat Template Auto-Detection ────────────────────────────────────────────
// Same logic as minslm-multi-og.cpp: auto-detect the model's chat template from
// GGUF metadata and construct system prefix + turn template.

static std::string custom_turn_template;

static std::string apply_chat_template(
    const char * tmpl,
    const std::vector<llama_chat_message> & msgs,
    bool add_generation_prompt)
{
    int32_t len = llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), add_generation_prompt, nullptr, 0);
    if (len < 0) return "";

    std::vector<char> buf(len + 2, 0);
    llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), add_generation_prompt, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), len);
}

static bool try_build_chat_templates(
    const char * tmpl,
    const char * label,
    const std::string & system_prompt,
    std::string & out_sys_prefix,
    std::string & out_turn_tmpl)
{
    std::vector<llama_chat_message> sys_msgs = {{ "system", system_prompt.c_str() }};
    std::string sys_prefix = apply_chat_template(tmpl, sys_msgs, false);
    if (sys_prefix.empty()) {
        printf("[chat_template]: %s — system prefix format failed\n", label);
        return false;
    }

    std::vector<llama_chat_message> full_msgs = {
        { "system", system_prompt.c_str() },
        { "user",   "{message}" }
    };
    std::string full_text = apply_chat_template(tmpl, full_msgs, true);
    if (full_text.empty()) {
        printf("[chat_template]: %s — full turn format failed\n", label);
        return false;
    }

    std::string turn_tmpl;
    if (full_text.size() > sys_prefix.size() &&
        full_text.substr(0, sys_prefix.size()) == sys_prefix) {
        turn_tmpl = full_text.substr(sys_prefix.size());
    } else {
        std::vector<llama_chat_message> user_msgs = {{ "user", "{message}" }};
        turn_tmpl = apply_chat_template(tmpl, user_msgs, true);
    }

    if (turn_tmpl.find("{message}") == std::string::npos) {
        printf("[chat_template]: %s — turn template missing {{message}} placeholder\n", label);
        return false;
    }

    out_sys_prefix = std::move(sys_prefix);
    out_turn_tmpl  = std::move(turn_tmpl);
    return true;
}

static bool detect_chat_format_from_jinja(
    const char * jinja,
    const std::string & system_prompt,
    std::string & out_sys_prefix,
    std::string & out_turn_tmpl,
    const char ** out_label)
{
    std::string t(jinja);

    // Phi-3 / Phi-3.5 / Phi-4
    if (t.find("<|user|>") != std::string::npos &&
        t.find("<|assistant|>") != std::string::npos &&
        t.find("<|end|>") != std::string::npos) {
        *out_label = "Phi-3/4";
        out_sys_prefix = "<|system|>\n" + system_prompt + "<|end|>\n";
        out_turn_tmpl  = "<|user|>\n{message}<|end|>\n<|assistant|>\n";
        return true;
    }

    // Llama-3 / Llama-3.1
    if (t.find("<|start_header_id|>") != std::string::npos &&
        t.find("<|end_header_id|>")   != std::string::npos) {
        *out_label = "Llama-3";
        out_sys_prefix = "<|start_header_id|>system<|end_header_id|>\n\n"
                         + system_prompt + "<|eot_id|>";
        out_turn_tmpl  = "<|start_header_id|>user<|end_header_id|>\n\n"
                         "{message}<|eot_id|>"
                         "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return true;
    }

    // Gemma-4
    if (t.find("<|turn>") != std::string::npos &&
        t.find("<turn|>") != std::string::npos) {
        *out_label = "Gemma-4";
        out_sys_prefix = "<|turn>system\n" + system_prompt + "<turn|>\n";
        out_turn_tmpl  = "<|turn>user\n{message}<turn|>\n<|turn>model\n";
        return true;
    }

    // Gemma / Gemma-2
    if (t.find("<start_of_turn>") != std::string::npos &&
        t.find("<end_of_turn>")   != std::string::npos) {
        *out_label = "Gemma";
        out_sys_prefix = "<start_of_turn>user\n" + system_prompt + "\n";
        out_turn_tmpl  = "{message}<end_of_turn>\n<start_of_turn>model\n";
        return true;
    }

    // ChatML
    if (t.find("<|im_start|>") != std::string::npos &&
        t.find("<|im_end|>")   != std::string::npos) {
        *out_label = "ChatML";
        out_sys_prefix = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        out_turn_tmpl  = "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n";
        return true;
    }

    // Mistral / Llama-2
    if (t.find("[INST]") != std::string::npos &&
        t.find("[/INST]") != std::string::npos) {
        *out_label = "Mistral/Llama-2";
        out_sys_prefix = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n";
        out_turn_tmpl  = "{message} [/INST] ";
        return true;
    }

    // Command-R
    if (t.find("<|START_OF_TURN_TOKEN|>") != std::string::npos) {
        *out_label = "Command-R";
        out_sys_prefix = "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>"
                         + system_prompt + "<|END_OF_TURN_TOKEN|>";
        out_turn_tmpl  = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
                         "{message}<|END_OF_TURN_TOKEN|>"
                         "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        return true;
    }

    return false;
}

static bool apply_model_chat_template(
    const std::string & system_prompt,
    model_params & params)
{
    std::string sys_prefix, turn_tmpl;

    const char * model_tmpl = llm_get_chat_template();
    if (model_tmpl) {
        printf("[chat_template]: model template found (%zu chars)\n", strlen(model_tmpl));
        if (try_build_chat_templates(model_tmpl, "model template",
                                     system_prompt, sys_prefix, turn_tmpl)) {
            printf("[chat_template]: OK — using model's built-in chat template\n");
            goto apply;
        }
        printf("[chat_template]: model template not recognized by llama_chat_apply_template()\n");

        const char * family = nullptr;
        if (detect_chat_format_from_jinja(model_tmpl, system_prompt,
                                          sys_prefix, turn_tmpl, &family)) {
            printf("[chat_template]: OK — detected %s format from jinja template\n", family);
            goto apply;
        }
        printf("[chat_template]: no known token patterns found in jinja template\n");
    } else {
        printf("[chat_template]: model has no built-in chat template in GGUF metadata\n");
    }

    printf("[chat_template]: falling back to ChatML default\n");
    if (try_build_chat_templates(nullptr, "ChatML fallback",
                                 system_prompt, sys_prefix, turn_tmpl)) {
        printf("[chat_template]: OK — using ChatML default template\n");
        goto apply;
    }

    printf("[chat_template]: ERROR — all template strategies failed\n");
    return false;

apply:
    params.custom_template_prompt = sys_prefix;
    custom_turn_template = turn_tmpl;
    params.turn_template = custom_turn_template;

    printf("[chat_template]: system prefix (%zu chars):\n  [%s]\n",
           sys_prefix.size(), sys_prefix.c_str());
    printf("[chat_template]: turn template (%zu chars):\n  [%s]\n",
           turn_tmpl.size(), turn_tmpl.c_str());

    return true;
}

int main(int argc, char** argv) {
    model_params params = {0};

    timer_init();
    int64_t t0 = timer_us(); 

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s arg_MODEL_PATH arg_#_threads arg_CUSTOM_PROMPT_file [pfc] [omp] [paffin] [stream] [verbose] [repack-xbox | repack-ggml] [--weight-budget MB]\n", argv[0]);
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

    bool use_auto_chat_template = false;
    std::string system_prompt_override;

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
            // only meaningful for GGML_CUDA or GGML_VULKAN builds
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

        } else if (!strcmp(argv[4], "chat-auto")) {
            use_auto_chat_template = true;
        }

        argv += 1;
        argc -= 1;
    }

    console::init(true);
    printf("[%s]: processing cpf input file [%s]\n", __func__, params.custom_p_file.c_str());
    processCustomPromptsFromFile(params);
    custom_prompts_it = custom_prompts.begin();

    // turn off llama info until there is a need for it to show up
    llm_disable_log();

    print_system_info(params.n_threads, params.n_batch);
    
    // the default for add_special for shared custom prompt is 'false'
    if (!params.parse_special) {
        // the default config for parse_special for shared custom pprompt is 'true'
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
#ifdef _WIN32
        _putenv_s("GGML_WEIGHT_BUDGET_MB", buf);
        _putenv_s("GGML_MODEL_LAYERS_STAT", "1");
#else
        setenv("GGML_WEIGHT_BUDGET_MB", buf, 1);
        setenv("GGML_MODEL_LAYERS_STAT", "1", 1);
#endif
        printf("[%s]: weight budget = %d MiB (layer windowing enabled)\n", __func__, params.weight_budget_mb);
    }

    // initialize the model
    if (!llm_initialize(params)) {
        printf("%s: Error during slm_init()\n", __func__);
        return 1;
    }

    // Auto-detect chat template from model metadata when "chat-auto" is specified,
    // or when the prompt file doesn't provide a CUSTOM_TEMPLATE_PROMPT section.
    if (use_auto_chat_template || params.custom_template_prompt.empty()) {
        std::string sys_prompt = "You are a helpful assistant.";
        if (apply_model_chat_template(sys_prompt, params)) {
            printf("[%s]: using auto-detected chat template from model\n", __func__);
            // Auto-detected templates include special tokens that need parsing,
            // and models like Gemma require BOS token
            params.add_special = true;
        } else {
            printf("[%s]: WARNING: chat template auto-detection failed, using raw prompt\n", __func__);
        }
    }

    int prompt_index = 1;
    while (custom_prompts_it != custom_prompts.end())
    {
        // Create custom user prompt
        std::string custom_prompt = ::trim(*custom_prompts_it);
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        std::string full_prompt;
        if (!params.turn_template.empty()) {
            // Auto-detected template: system prefix + turn template with {message}
            std::string turn = params.turn_template;
            size_t pos = turn.find("{message}");
            if (pos != std::string::npos) {
                turn.replace(pos, 9, custom_prompt);
            }
            full_prompt = params.custom_template_prompt + turn;
        } else {
            // Legacy: single template with {message} placeholder
            full_prompt = ::trim(params.custom_template_prompt);
            size_t pos = full_prompt.find("{message}");
            if (pos != std::string::npos) {
                full_prompt.replace(pos, 9, custom_prompt);
            }
        }

        if (params.pfc_mode && !params.pfx_shared.empty()) {
            // In PFC mode, strip the shared prefix
            size_t pos = full_prompt.find(custom_prompt);
            params.prompt = (pos != std::string::npos) ? full_prompt.substr(pos) : full_prompt;
        }
        else {
            params.prompt = full_prompt;
        }

        console::set_display(console::prompt);
        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        // reset for the output
        console::set_display(console::reset);

        // the default for add_special for each prompt is 'false'
        // unless it is overridden by the command line

        if (!params.parse_special) {
            // for minslm the config for parse_special is 'true' if not specified explicitly
            params.parse_special = true;
        }

        // running an inference based on a prompt
        if (!llm_inference(params)) {
            printf("Failed token generation for query: %s\n", params.prompt.c_str());
            params.reply = "";
        }

        if (params.streaming_reply) {
            // in streaming mode the reply should already be done post llm_inference()
            printf("\n<<\n");
        } else {
            // flush entire reply from SLM
            bool end_of_reply = false;
            for (auto c: params.reply) {
                if (!end_of_reply) {
                    printf("%c", c);
                }
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
