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

int main(int argc, char** argv) {
    model_params params = {0};

    timer_init();
    int64_t t0 = timer_us(); 

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s arg_MODEL_PATH arg_#_threads arg_CUSTOM_PROMPT_file [pfc] [omp] [paffin] [stream] [verbose] [repack-xbox | repack-ggml]\n", argv[0]);
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

        } else if (!strcmp(argv[4], "repack-xbox-st")) {
            params.tensor_repack_mode = 3;

        } else if (!strcmp(argv[4], "mulmat-xbox")) {
            params.tensor_repack_mode = 4;

        } else if (!strcmp(argv[4], "stream")) {
            params.streaming_reply = true;

        } else if (!strcmp(argv[4], "add-special")) {
            params.add_special = true;

        } else if (!strcmp(argv[4], "parse-special")) {
            params.parse_special = true;

        } else if (!strcmp(argv[4], "gpu")) {
            params.gpu_offload = true;
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

    // initialize the model
    if (!llm_initialize(params)) {
        printf("%s: Error during slm_init()\n", __func__);
        return 1;
    }

    int prompt_index = 1;
    while (custom_prompts_it != custom_prompts.end())
    {
        // Create custom user prompt
        std::string custom_prompt = ::trim(*custom_prompts_it);
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        std::string full_prompt = ::trim(params.custom_template_prompt);
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            full_prompt.replace(pos, std::string("{message}").length(), custom_prompt);
        }
        else {
            pos = 0;
        }

        if (params.pfc_mode && !params.pfx_shared.empty()) {
            params.prompt = full_prompt.substr(pos);
        }
        else {
            // non pfc mode 
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
