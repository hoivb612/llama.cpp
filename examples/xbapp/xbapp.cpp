#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...
#pragma warning (disable:4715) //  not all control paths return a value

#include "xbapp.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

int current_custom_prompt_index = 0;
std::vector<std::string> custom_prompts;
std::vector<std::string>::iterator custom_prompts_it;
std::string custom_prompts_output;
bool switch_prompt = false; // set true every time switch to a new prompt

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
}

bool processCustomPromptsFromFile(xbapp_params& xbparams) {
    std::ifstream cpfile(xbparams.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, xbparams.custom_p_file.c_str());
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

    xbparams.custom_template_prompt = custom_template_prompt;

    cpfile.close();

    return true;
}

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_seqlen] [-ngl n_gpu_layers] [-t n_threads] [-cpf cpf_prompt] [-pfc] [-omp] [prompt...]\n", argv[0]);
    printf("\n");
}

int64_t t0;
int main(int argc, char** argv) {
    xbapp_params xbparams = {0};

    ggml_time_init();
    t0 = ggml_time_us();

    // parse command line args
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    xbparams.model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_seqlen = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_threads = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-cpf") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.custom_p_file = argv[++i];
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-omp") == 0) {
                xbparams.openmp = true;
            } else if (strcmp(argv[i], "-pfc") == 0) {
                xbparams.pfc_mode = true;
            } else {
                // single prompt starts here
                break;
            }
        }

        if (xbparams.model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            // walk the single prompt until end
            xbparams.prompt = argv[i++];
            for (; i < argc; i++) {
                xbparams.prompt += " ";
                xbparams.prompt += argv[i];
            }
        }
    }

    if (xbparams.n_threads <= 0) {
        int32_t n_threads = std::thread::hardware_concurrency();
        if (n_threads > 0) {
            n_threads = (n_threads <= 4) ? n_threads : (n_threads / 2);
        } else {
            n_threads = 4;
        }
        xbparams.n_threads = n_threads;

#ifdef GGML_USE_OPENMP
        xbparams.n_threads = MIN(n_threads, omp_get_max_threads());
        if (xbparams.openmp) {
            printf("%s: OpenMP selected\n", __func__);
            // ggml_select_omp();
        }
#else
        xbparams.n_threads = n_threads;
#endif

        // ggml_set_process_affinity(xbparams.n_threads);

        printf("%s: Number of hw threads asked: %d - actual number: %d\n", __func__, n_threads, xbparams.n_threads);
    }

    console::init(true);
    printf("[%s]: processing cpf input file [%s]\n", __func__, xbparams.custom_p_file.c_str());
    processCustomPromptsFromFile(xbparams);
    custom_prompts_it = custom_prompts.begin();

    // initialize the model
    if (slm_init(xbparams) != 0) {
        printf("%s: Error during slm_init()\n", __func__);
        return 1;
    }

    int prompt_index = 1;
    while (custom_prompts_it != custom_prompts.end())
    {
        // Create custom user prompt
        std::string& custom_prompt = *custom_prompts_it;
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        std::string full_prompt = xbparams.custom_template_prompt;
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            full_prompt.replace(pos, std::string("{message}").length(), custom_prompt);
        }

        if (xbparams.pfc_mode && !xbparams.pfx_shared.empty()) {
            xbparams.prompt = ::trim(custom_prompt);

        } else {
            // non pfc mode 
            xbparams.prompt = ::trim(full_prompt);
        }
    
        console::set_display(console::prompt);
        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        // reset for the output
        console::set_display(console::reset);

        // running an inference based on a prompt
        slm_inference(xbparams);

        xbparams.first_prompt = false;
        
        custom_prompts_it++;
    }

    slm_terminate();
    
    console::set_display(console::stats);
    t0 = ggml_time_us() - t0;
    printf("\n\n total elapsed time %7.2fsec\n", (double)t0 / (1000. * 1000.));
#ifdef GGML_TENSOR_OP_PERF
    print_tensor_op_perf_data();
#endif // GGML_TENSOR_OP_PERF

    console::cleanup();

    return 0;
}