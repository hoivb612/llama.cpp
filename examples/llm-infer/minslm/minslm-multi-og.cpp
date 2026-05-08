//
// minslminfer-multi-og.cpp — Multi-turn conversation driver for llm-infer
//
// Uses the OG (SYSTEM/PROMPT block) script format from cpf_gem4mm.cpp
// so that both programs can share the same prompt input files.
//
// Script format (see prompts/sample_script_textonly.txt):
//   SYSTEM
//   <system prompt text>
//
//   PROMPT
//   T: <text>
//
//   PROMPT
//   /context | /rewind [N] | quit()
//
// Usage:
//   minslminfer-multi-og MODEL_PATH #_threads SCRIPT_file [template_file] [options...]
//
// Options: verbose|v1|v2  pfc  paffin  stream  cpu  add-special  parse-special
//
// The optional template=FILE uses CUSTOM_TEMPLATE_PROMPT / CUSTOM_TURN_TEMPLATE
// sections (same as minslm-multi) to supply model-specific chat formatting.
// If omitted, the model's built-in chat template (from GGUF metadata) is used
// automatically to wrap system and user messages.
//

/* 
This is a multi-turn conversation driver that uses the SYSTEM/PROMPT block format from cpf_gem4mm.cpp, allowing both programs to share the same script files
like sample_script_textonly.txt.

Key features:

 - Script parser — Reads SYSTEM / PROMPT blocks with T: text prefix
 - Meta commands — /context (stats table), /rewind N, quit()
 - Interactive fallback — Falls back to stdin when scripted prompts are exhausted
 - Optional template file — template=FILE CLI arg for model-specific chat formatting
 - Multi-turn KV cache — Uses llm_multiturn_begin/llm_infer_multiturn/llm_multiturn_rewind

Usage:

 minslminfer-multi-og MODEL #threads SCRIPT [template=FILE] [stream] [verbose] ...

Template file format — two sections delimited by END_SECTION:

 CUSTOM_TEMPLATE_PROMPT
 <|turn>system
 You are a helpful assistant.
 <turn|>
 END_SECTION
 CUSTOM_TURN_TEMPLATE
 <|turn>user
 {message}<turn|>
 <|turn>model
 END_SECTION

 - CUSTOM_TEMPLATE_PROMPT — the system prefix (decoded once into KV cache at start)
 - CUSTOM_TURN_TEMPLATE — the per-user-turn wrapper; must contain {message} placeholder

When to use:

 - If you omit template=, the program auto-detects the model's chat format from GGUF metadata (Gemma-4, Phi-3/4, Llama-3, ChatML, etc.)
 - Use template= only when auto-detection fails or you want to override the format manually

- Fix template: Gemma-4 uses <|turn>role/<turn|> tokens (not <start_of_turn>/<end_of_turn> like older Gemma), and its 16K-char jinja template isn't recognized 
by llama_chat_apply_template()
- Fix: 3-level template detection — API → jinja pattern scan → ChatML fallback — with Gemma-4 pattern added to the scanner

 1. Get the model's raw template string

  After llm_initialize(), llm_get_chat_template() — a thin wrapper I added around llama_model_chat_template(model, nullptr). 
  This reads the chat_template field from the GGUF metadata. For the Gemma-4 model, it returned a 16,317-character jinja2 
  template string.

  2. Try the official API first (failed)

  Pass the template string to llama_chat_apply_template(), which pattern-matches against a hardcoded list of 
  ~20 known template formats. Gemma-4 template isn't in that list, so it returns -1.

  3. Scan the jinja source for known tokens

  The function detect_chat_format_from_jinja() checks the following in order:

  ┌──────────────────────────────────────────┬────────────────────────────┐
  │ Pattern searched                         │ Model family               │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ <|user|> + <|assistant|> + <|end|>       │ Phi-3/4                    │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ <|start_header_id|>                      │ Llama-3                    │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ <|turn> + <turn|>                        │ Gemma-4                    │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ <start_of_turn>                          │ Gemma 1/2                  │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ <|im_start|>                             │ ChatML                     │
  ├──────────────────────────────────────────┼────────────────────────────┤
  │ [INST]                                   │ Mistral/Llama-2            │
  └──────────────────────────────────────────┴────────────────────────────┘

  The 16K jinja string contains the literals <|turn> and <turn|>, so Gemma-4 matches.

  3. Extract the template format

  From the reference implementation:

  b612_Onnx/examples/cpp/Gemma-4/gemma4_tokenizer.cpp, lines 93–105:

   // System turn
   text += "<|turn>system\n";
   text += system_prompt;
   text += "<turn|>\n";
   
   // User turn
   text += "<|turn>user\n";
   text += user_text;
   text += "<turn|>\n";
   
   // Generation prompt
   text += "<|turn>model\n";

  From this construct the full prompt:

   - System prefix: <|turn>system\n{sys_text}<turn|>\n
   - Turn template: <|turn>user\n{message}<turn|>\n<|turn>model\n

  The system prefix is decoded once into the KV cache by llm_multiturn_begin(). 
  Each user turn substitutes {message} with the actual text and feeds it to llm_infer_multiturn().

========================================

  Jinja is a Python templating language (from the Flask ecosystem) that uses {{ variable }}, {% if %}, {% for %} blocks to generate text
  from data. HuggingFace adopted it as the standard way to define chat templates in model configs.

  Every GGUF model can embed a chat_template metadata field — a Jinja string that describes how to format a conversation. For example, a
  ChatML model's template looks like:

   {% for message in messages %}
   <|im_start|>{{ message.role }}
   {{ message.content }}<|im_end|>
   {% endfor %}

  Given messages = [{role: "user", content: "Hello"}], the Jinja engine renders:

   <|im_start|>user
   Hello<|im_end|>

  Gemma-4 model has a 16,317-character Jinja template — complex with conditionals for tools, thinking, multimodal content, etc.
  llama.cpp's llama_chat_apply_template() doesn't run a Jinja engine; it pattern-matches against ~20 hardcoded template names. Your
  template wasn't recognized → returned -1.

  Instead of parsing the Jinja, detect_chat_format_from_jinja() searches the raw Jinja source text for known special-token literals:

   // The Jinja source contains these literal strings
   if (t.find("<|turn>") != std::string::npos &&
       t.find("<turn|>") != std::string::npos) {
       // → Gemma-4 format

  Scan it for token markers to identify the model family, then hardcode the correct format. It's a
  pragmatic shortcut that avoids needing a full Jinja parser in C++.

  There are two levels to extract the Jinja template from the model

  1. At runtime in our code

  We call the llama.cpp API:

   // llm-infer.cpp — our wrapper
   const char * llm_get_chat_template() {
       return llama_model_chat_template(llm_model, nullptr);
   }

  llama_model_chat_template() reads the tokenizer.chat_template key from the GGUF metadata that was loaded when the model file was
  opened. It returns the raw Jinja string (or nullptr if absent).

  2. What's inside the GGUF file

  GGUF is a binary format with a key-value metadata section. When a model is converted from HuggingFace (via convert_hf_to_gguf.py), the
  converter copies tokenizer_config.json's chat_template field into the GGUF metadata.

  You can inspect it offline with Python:

   python -c "
   from gguf.gguf_reader import GGUFReader
   r = GGUFReader('your_model.gguf')
   for k in r.fields:
       if 'chat_template' in k:
           val = bytes(r.fields[k].parts[-1]).decode('utf-8')
           print(f'{k}: {val[:200]}...')
   "

  The chain

   HuggingFace tokenizer_config.json
     → "chat_template": "{% for message in messages %}..."
       → convert_hf_to_gguf.py bakes it into GGUF metadata
         → llama_model_chat_template() reads it at load time
           → our code gets the 16K-char Jinja string

*/

#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "llm-infer.h"
#include "b612-cpu.h"
#include "llama.h"

#include "log.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
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

// ─── Console Helpers ────────────────────────────────────────────────────────

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

// ─── Timer ──────────────────────────────────────────────────────────────────

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

// ─── String Helpers ─────────────────────────────────────────────────────────

static std::string trim(const std::string & str) {
    if (str.empty()) return str;
    size_t start = 0;
    size_t end = str.length();
    while ((start < end) && isspace(str[start])) start += 1;
    while ((end > start) && isspace(str[end - 1])) end -= 1;
    return str.substr(start, end - start);
}

static std::string to_lower(const std::string & str) {
    std::string s = str;
    for (auto & c : s) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    return s;
}

// Case-insensitive check for meta commands (/context, /rewind, quit())
static bool is_meta_command(const std::string & t) {
    std::string lc = to_lower(t);
    return lc == "/context" || lc.substr(0, 7) == "/rewind" || lc == "quit()";
}

// ─── Parsed Prompt / Script Types (OG format from cpf_gem4mm) ───────────────

struct ParsedPrompt {
    std::string text;          // T: content (or raw text)
    bool is_meta = false;      // /context, /rewind, quit()
    std::string raw;           // display string
};

struct Script {
    std::string system_prompt;
    std::queue<ParsedPrompt> prompts;
};

// ─── OG Script Parser ───────────────────────────────────────────────────────

static std::vector<std::string> ReadBlockLines(std::ifstream& file) {
    std::vector<std::string> lines;
    std::string line;
    std::streampos pos;
    while (true) {
        pos = file.tellg();
        if (!std::getline(file, line)) break;
        std::string t = trim(line);
        if (t == "SYSTEM" || t == "PROMPT") {
            file.seekg(pos);
            break;
        }
        lines.push_back(line);
    }
    while (!lines.empty() && trim(lines.front()).empty()) lines.erase(lines.begin());
    while (!lines.empty() && trim(lines.back()).empty()) lines.pop_back();
    return lines;
}

static ParsedPrompt ParsePromptBlock(const std::vector<std::string>& lines) {
    ParsedPrompt p;
    std::string raw_parts;

    for (auto& line : lines) {
        std::string t = trim(line);
        if (t.empty()) continue;

        if (t.size() >= 3 && t[1] == ':' && t[2] == ' ') {
            char prefix = static_cast<char>(toupper(static_cast<unsigned char>(t[0])));
            std::string value = trim(t.substr(2));
            switch (prefix) {
                case 'T': p.text = value; break;
                // I: and A: are image/audio paths — ignored for text-only inference
                default:
                    if (p.text.empty()) p.text = t;
                    break;
            }
        }
        else if (is_meta_command(t)) {
            p.is_meta = true;
            p.text = t;
        }
        else {
            if (p.text.empty()) p.text = t;
        }

        if (!raw_parts.empty()) raw_parts += " | ";
        raw_parts += t;
    }

    p.raw = raw_parts;
    return p;
}

Script ParseScriptFile(const std::string& filepath) {
    Script script;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        printf("[%s]: failed to open script file [%s]\n", __func__, filepath.c_str());
        return script;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::string t = trim(line);
        if (t == "SYSTEM") {
            auto block = ReadBlockLines(file);
            for (auto& l : block) {
                if (!script.system_prompt.empty()) script.system_prompt += "\n";
                script.system_prompt += l;
            }
        } else if (t == "PROMPT") {
            auto block = ReadBlockLines(file);
            if (!block.empty()) {
                auto prompt = ParsePromptBlock(block);
                if (!prompt.raw.empty())
                    script.prompts.push(std::move(prompt));
            }
        }
    }
    return script;
}

// Returns N for "/rewind N" (default 1). Returns 0 if not a rewind command.
static int ParseRewindCommand(const std::string& input) {
    std::string t = to_lower(trim(input));
    if (t.size() < 7 || t.substr(0, 7) != "/rewind") return 0;
    std::string rest = trim(t.substr(7));
    if (rest.empty()) return 1;
    try {
        int n = std::stoi(rest);
        return (n > 0) ? n : 0;
    } catch (...) {
        return 0;
    }
}

// ─── Template File Parser ───────────────────────────────────────────────────
// Optional template file with CUSTOM_TEMPLATE_PROMPT / CUSTOM_TURN_TEMPLATE
// sections (same format as the old minslm-multi custom prompt files).

static std::string custom_turn_template;

bool loadTemplateFile(const std::string& filepath, model_params& params) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        printf("[%s]: failed to open template file [%s]\n", __func__, filepath.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool turnTemplateMode = false;
    std::string custom_template_prompt;
    custom_turn_template = "";

    while (std::getline(file, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") {
            templatePromptMode = true;
            continue;
        } else if (line == "CUSTOM_TURN_TEMPLATE") {
            turnTemplateMode = true;
            continue;
        } else if (line == "END_SECTION") {
            templatePromptMode = false;
            turnTemplateMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (turnTemplateMode) {
            custom_turn_template += line + '\n';
        }
    }

    params.custom_template_prompt = custom_template_prompt;
    params.turn_template = custom_turn_template;
    return true;
}

// ─── Chat Template Helper ───────────────────────────────────────────────────
// Uses the llama.h llama_chat_apply_template() API to format messages using
// the model's built-in chat template (auto-detected from GGUF metadata).
// Falls back to ChatML (tmpl=nullptr) if the model's template is unrecognized.

static std::string apply_chat_template(
    const char * tmpl,
    const std::vector<llama_chat_message> & msgs,
    bool add_generation_prompt)
{
    int32_t len = llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), add_generation_prompt, nullptr, 0);
    if (len < 0) return "";  // template not recognized

    std::vector<char> buf(len + 2, 0);
    llama_chat_apply_template(
        tmpl, msgs.data(), msgs.size(), add_generation_prompt, buf.data(), (int32_t)buf.size());
    return std::string(buf.data(), len);
}

// Try to build system prefix + turn template from a given template string.
// Returns true only if both pieces are non-empty and turn template has {message}.
static bool try_build_chat_templates(
    const char * tmpl,
    const char * label,
    const std::string & system_prompt,
    std::string & out_sys_prefix,
    std::string & out_turn_tmpl)
{
    // Format system prefix: [{system, sys_text}] without generation prompt
    std::vector<llama_chat_message> sys_msgs = {{ "system", system_prompt.c_str() }};
    std::string sys_prefix = apply_chat_template(tmpl, sys_msgs, false);
    if (sys_prefix.empty()) {
        printf("[chat_template]: %s — system prefix format failed\n", label);
        return false;
    }

    // Format full first turn: [{system, sys_text}, {user, "{message}"}] with assistant prompt
    std::vector<llama_chat_message> full_msgs = {
        { "system", system_prompt.c_str() },
        { "user",   "{message}" }
    };
    std::string full_text = apply_chat_template(tmpl, full_msgs, true);
    if (full_text.empty()) {
        printf("[chat_template]: %s — full turn format failed\n", label);
        return false;
    }

    // Derive turn template = full text minus system prefix
    std::string turn_tmpl;
    if (full_text.size() > sys_prefix.size() &&
        full_text.substr(0, sys_prefix.size()) == sys_prefix) {
        turn_tmpl = full_text.substr(sys_prefix.size());
    } else {
        // Prefix mismatch — try user-only as the turn template
        std::vector<llama_chat_message> user_msgs = {{ "user", "{message}" }};
        turn_tmpl = apply_chat_template(tmpl, user_msgs, true);
    }

    // Validate: turn template must contain the {message} placeholder
    if (turn_tmpl.find("{message}") == std::string::npos) {
        printf("[chat_template]: %s — turn template missing {{message}} placeholder\n", label);
        return false;
    }

    out_sys_prefix = std::move(sys_prefix);
    out_turn_tmpl  = std::move(turn_tmpl);
    return true;
}

// Detect model family by scanning the jinja template string for known
// special-token patterns, then construct system prefix and turn template
// directly.  Covers the major model families whose complex jinja templates
// are not recognized by llama_chat_apply_template().
static bool detect_chat_format_from_jinja(
    const char * jinja,
    const std::string & system_prompt,
    std::string & out_sys_prefix,
    std::string & out_turn_tmpl,
    const char ** out_label)
{
    std::string t(jinja);

    // Phi-3 / Phi-3.5 / Phi-4  — <|system|> <|user|> <|assistant|> <|end|>
    if (t.find("<|user|>") != std::string::npos &&
        t.find("<|assistant|>") != std::string::npos &&
        t.find("<|end|>") != std::string::npos) {
        *out_label = "Phi-3/4";
        out_sys_prefix = "<|system|>\n" + system_prompt + "<|end|>\n";
        out_turn_tmpl  = "<|user|>\n{message}<|end|>\n<|assistant|>\n";
        return true;
    }

    // Llama-3 / Llama-3.1 — <|start_header_id|> <|end_header_id|> <|eot_id|>
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

    // Gemma-4 — <|turn>role <turn|>  (new format, different from older Gemma)
    if (t.find("<|turn>") != std::string::npos &&
        t.find("<turn|>") != std::string::npos) {
        *out_label = "Gemma-4";
        out_sys_prefix = "<|turn>system\n" + system_prompt + "<turn|>\n";
        out_turn_tmpl  = "<|turn>user\n{message}<turn|>\n<|turn>model\n";
        return true;
    }

    // Gemma / Gemma-2 — <start_of_turn> <end_of_turn>
    if (t.find("<start_of_turn>") != std::string::npos &&
        t.find("<end_of_turn>")   != std::string::npos) {
        *out_label = "Gemma";
        // Gemma has no system role; embed system text in the first user turn
        out_sys_prefix = "<start_of_turn>user\n" + system_prompt + "\n";
        out_turn_tmpl  = "{message}<end_of_turn>\n<start_of_turn>model\n";
        return true;
    }

    // ChatML — <|im_start|> <|im_end|>  (Qwen, Yi, Deepseek-v2, etc.)
    if (t.find("<|im_start|>") != std::string::npos &&
        t.find("<|im_end|>")   != std::string::npos) {
        *out_label = "ChatML";
        out_sys_prefix = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        out_turn_tmpl  = "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n";
        return true;
    }

    // Mistral / Llama-2 — [INST] [/INST]
    if (t.find("[INST]") != std::string::npos &&
        t.find("[/INST]") != std::string::npos) {
        *out_label = "Mistral/Llama-2";
        out_sys_prefix = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n";
        out_turn_tmpl  = "{message} [/INST] ";
        return true;
    }

    // Command-R — <|START_OF_TURN_TOKEN|>
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

// Build system prefix and turn template from the model's built-in chat template.
// Strategy:
//   1. Try llama_chat_apply_template() with the model's template
//   2. Scan the jinja source for known token patterns and construct directly
//   3. Fall back to ChatML default (tmpl=nullptr)
static bool apply_model_chat_template(
    const std::string & system_prompt,
    model_params & params)
{
    std::string sys_prefix, turn_tmpl;

    // ── Attempt 1: llama_chat_apply_template() with model's template ──
    const char * model_tmpl = llm_get_chat_template();
    if (model_tmpl) {
        printf("[chat_template]: model template found (%zu chars)\n", strlen(model_tmpl));
        if (try_build_chat_templates(model_tmpl, "model template",
                                     system_prompt, sys_prefix, turn_tmpl)) {
            printf("[chat_template]: OK — using model's built-in chat template\n");
            goto apply;
        }
        printf("[chat_template]: model template not recognized by llama_chat_apply_template()\n");

        // ── Attempt 2: pattern-match jinja source for known token families ──
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

    // ── Attempt 3: ChatML default (tmpl=nullptr → "chatml") ──
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

    // Always print confirmation so user can verify formatting
    printf("[chat_template]: system prefix (%zu chars):\n  [%s]\n",
           sys_prefix.size(), sys_prefix.c_str());
    printf("[chat_template]: turn template (%zu chars):\n  [%s]\n",
           turn_tmpl.size(), turn_tmpl.c_str());

    return true;
}

// ─── Turn Stats ─────────────────────────────────────────────────────────────

struct TurnStats {
    double ttft_ms;
    double tokens_per_sec;
    int generated_tokens;
    int prompt_tokens;
    std::string summary;
};

static std::vector<TurnStats> turn_stats;

// ─── /context display ───────────────────────────────────────────────────────

static void print_context_info() {
    auto default_prec = std::cout.precision();
    std::cout << "\n=== Context Info ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Total turns:       " << llm_multiturn_turn_count() << std::endl;
    std::cout << "  Total KV tokens:   " << llm_multiturn_token_count() << std::endl;
    std::cout << "  User exchanges:    " << turn_stats.size() << std::endl;

    if (!turn_stats.empty()) {
        std::cout << "\n  Turn  Prompt  Generated     TTFT       Speed" << std::endl;
        for (size_t i = 0; i < turn_stats.size(); i++) {
            auto& ts = turn_stats[i];
            std::cout << "  " << std::setw(4) << (i + 1)
                      << "  " << std::setw(6) << ts.prompt_tokens
                      << "  " << std::setw(9) << ts.generated_tokens
                      << "  " << std::setw(9) << ts.ttft_ms << "ms"
                      << "  " << std::setw(7) << ts.tokens_per_sec << " tok/s";
            if (!ts.summary.empty())
                std::cout << "  " << ts.summary;
            std::cout << std::endl;
        }
        const auto& last = turn_stats.back();
        std::cout << "\n  Last turn — TTFT: " << last.ttft_ms << "ms, "
                  << last.tokens_per_sec << " tok/s ("
                  << last.generated_tokens << " tokens)" << std::endl;
    }
    std::cout << std::setprecision(static_cast<int>(default_prec));
    std::cout << "====================" << std::endl;
}

// ─── System Info ────────────────────────────────────────────────────────────

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

// ─── Interactive stdin prompt reader ────────────────────────────────────────

ParsedPrompt ReadPromptFromStdin() {
    std::cout << "\nPrompt (T: text | /context | /rewind [N] | quit()):\n";
    std::vector<std::string> lines;
    std::string line;

    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) {
            ParsedPrompt p;
            p.is_meta = true;
            p.text = "quit()";
            p.raw = "quit() [EOF]";
            return p;
        }
        std::string t = trim(line);
        if (t.empty()) {
            if (lines.empty()) continue;
            break;
        }
        lines.push_back(line);

        // Single-line meta commands
        if (is_meta_command(t)) break;

        // If first line has no T: prefix, treat as single-line text
        if (lines.size() == 1 && !(t.size() >= 3 && t[1] == ':' && t[2] == ' ' &&
            toupper(static_cast<unsigned char>(t[0])) == 'T'))
        {
            break;
        }
    }

    if (lines.empty()) {
        ParsedPrompt p;
        return p;
    }

    // Single line without prefix → pure text
    if (lines.size() == 1) {
        std::string t = trim(lines[0]);
        bool has_prefix = (t.size() >= 3 && t[1] == ':' && t[2] == ' ');
        bool is_meta = is_meta_command(t);
        if (!has_prefix && !is_meta) {
            ParsedPrompt p;
            p.text = t;
            p.raw = t;
            return p;
        }
    }

    return ParsePromptBlock(lines);
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    model_params params = {0};

    timer_init();
    int64_t t0 = timer_us();

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH #_threads SCRIPT_file [template=FILE] [options...]\n", argv[0]);
        printf("\n");
        printf("  Script format (SYSTEM/PROMPT blocks — compatible with cpf_gem4mm):\n");
        printf("    SYSTEM\n");
        printf("    <system prompt text>\n");
        printf("    PROMPT\n");
        printf("    T: <user message>\n");
        printf("    PROMPT\n");
        printf("    /context | /rewind [N] | quit()\n");
        printf("\n");
        printf("  Options: verbose|v1|v2  pfc  paffin  stream  cpu\n");
        printf("           add-special  parse-special  template=<file>\n");
        printf("           -d N (GPU device index)  -sm none|layer|row\n");
        printf("           --weight-budget MB (layer windowing budget)\n");
        printf("           repack-ggml  repack-xbox  repack-xbcg  repack-xbox-st  mulmat-xbox\n");
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

    // Parse script file (arg 3)
    Script script;
    std::string script_file;
    if (argc >= 4) {
        script_file = argv[3];
    }

    // Parse remaining options (arg 4+)
    std::string template_file;
    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "verbose" || arg == "v1") {
            params.verbose = 1;
        } else if (arg == "v2") {
            params.verbose = 2;
        } else if (arg == "paffin") {
            params.process_affinity = true;
        } else if (arg == "pfc") {
            params.pfc_mode = true;
        } else if (arg == "stream") {
            params.streaming_reply = true;
        } else if (arg == "add-special") {
            params.add_special = true;
        } else if (arg == "parse-special") {
            params.parse_special = true;
        } else if (arg == "cpu") {
            params.force_cpu_mode = true;
        } else if (arg == "-d" && i + 1 < argc) {
            params.main_gpu = std::stoi(argv[++i]);
        } else if (arg == "-sm" && i + 1 < argc) {
            std::string sm = argv[++i];
            if (sm == "none")       params.split_mode = 0;
            else if (sm == "layer") params.split_mode = 1;
            else if (sm == "row")   params.split_mode = 2;
        } else if (arg == "repack-ggml") {
            params.tensor_repack_mode = 1;
        } else if (arg == "repack-xbox") {
            params.tensor_repack_mode = 2;
        } else if (arg == "repack-xbcg") {
            params.tensor_repack_mode = 3;
        } else if (arg == "repack-xbox-st") {
            params.tensor_repack_mode = 4;
        } else if (arg == "mulmat-xbox") {
            params.tensor_repack_mode = 5;
        } else if (arg.substr(0, 9) == "template=") {
            template_file = arg.substr(9);
        } else if (arg == "--weight-budget" && i + 1 < argc) {
            params.weight_budget_mb = std::stoi(argv[++i]);
        }
    }

    console::init(true);

    // Load script
    if (!script_file.empty()) {
        printf("[%s]: loading script file [%s]\n", __func__, script_file.c_str());
        script = ParseScriptFile(script_file);
        printf("[%s]: loaded %zd prompt(s), system: \"%s\"\n",
               __func__, script.prompts.size(),
               script.system_prompt.substr(0, 60).c_str());
    }

    // Load optional template file
    if (!template_file.empty()) {
        printf("[%s]: loading template file [%s]\n", __func__, template_file.c_str());
        loadTemplateFile(template_file, params);
    }

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
#ifdef _WIN32
        _putenv_s("GGML_WEIGHT_BUDGET_MB", buf);
        _putenv_s("GGML_MODEL_LAYERS_STAT", "1");
#else
        setenv("GGML_WEIGHT_BUDGET_MB", buf, 1);
        setenv("GGML_MODEL_LAYERS_STAT", "1", 1);
#endif
        printf("[%s]: weight budget = %d MiB (layer windowing enabled)\n", __func__, params.weight_budget_mb);
    }

    // Initialize the model
    if (!llm_initialize(params)) {
        printf("%s: Error during llm_initialize()\n", __func__);
        return 1;
    }

    // If no template file given, use the model's built-in chat template to
    // format the system prefix and turn template automatically.
    if (template_file.empty() && !script.system_prompt.empty()) {
        if (!apply_model_chat_template(script.system_prompt, params)) {
            // Model has no built-in template — fall back to raw system prompt
            printf("[%s]: WARNING: no chat template available, sending raw text "
                   "(output may be incorrect)\n", __func__);
            params.custom_template_prompt = script.system_prompt;
        }
    }

    // Start multi-turn session
    llm_multiturn_begin(params);
    printf("[%s]: multi-turn mode — KV cache will accumulate across turns\n", __func__);

    // Stop generation when '}' is encountered (for JSON responses)
    params.stop_char = '}';

    int prompt_index = 1;

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-turn conversation loop
    // ═══════════════════════════════════════════════════════════════════════════

    printf("\n--- Conversation started ---\n");

    while (true) {
        ParsedPrompt prompt;

        // Get next prompt (scripted or interactive)
        if (!script.prompts.empty()) {
            prompt = script.prompts.front();
            script.prompts.pop();
            std::cout << "\nScript> " << prompt.raw << std::endl;
        } else {
            prompt = ReadPromptFromStdin();
        }

        // Skip empty prompts
        if (prompt.raw.empty() && prompt.text.empty())
            continue;

        // ── Handle quit() ──
        if (to_lower(prompt.text) == "quit()") {
            std::cout << "Exiting." << std::endl;
            break;
        }

        // ── Handle /context ──
        if (to_lower(prompt.text) == "/context") {
            print_context_info();
            continue;
        }

        // ── Handle /rewind ──
        int rewind_count = ParseRewindCommand(prompt.text);
        if (rewind_count > 0) {
            int total_turns = llm_multiturn_turn_count();
            if (total_turns == 0) {
                std::cout << "No conversation history to rewind." << std::endl;
                continue;
            }
            if (rewind_count > total_turns) {
                printf("Only %d exchange(s) in history. Rewinding all.\n", total_turns);
                rewind_count = total_turns;
            }

            console::set_display(console::stats);
            printf("> /rewind %d (turns: %d, tokens: %d)\n",
                   rewind_count, llm_multiturn_turn_count(), llm_multiturn_token_count());
            llm_multiturn_rewind(rewind_count);
            printf("  after rewind: turns=%d, tokens=%d\n",
                   llm_multiturn_turn_count(), llm_multiturn_token_count());

            // Trim turn stats to match
            if ((int)turn_stats.size() >= rewind_count) {
                turn_stats.erase(turn_stats.end() - rewind_count, turn_stats.end());
            } else {
                turn_stats.clear();
            }

            console::set_display(console::reset);
            continue;
        }

        // ── Handle other meta commands ──
        if (prompt.is_meta) {
            std::cout << "Unknown meta command: " << prompt.text << std::endl;
            continue;
        }

        // ══════════════════════════════════════════════════════════════════════
        // Process a user turn
        // ══════════════════════════════════════════════════════════════════════

        std::string user_text = prompt.text;
        user_text.erase(
            std::remove(user_text.begin(), user_text.end(), '\"'),
            user_text.end());

        // Wrap user text in turn template if available
        if (!custom_turn_template.empty()) {
            std::string tmpl = ::trim(custom_turn_template);
            size_t pos = tmpl.find("{message}");
            if (pos != std::string::npos) {
                tmpl.replace(pos, std::string("{message}").length(), user_text);
            }
            params.prompt = tmpl;
        } else {
            // No turn template — send user text directly
            params.prompt = user_text;
        }

        console::set_display(console::prompt);
        printf("> Turn %d: [%s]\n", prompt_index, user_text.c_str());
        console::set_display(console::reset);

        if (!params.parse_special) {
            params.parse_special = true;
        }

        int64_t t_turn_start = timer_us();

        // Run multi-turn inference (KV cache accumulates)
        if (!llm_infer_multiturn(params)) {
            printf("Failed token generation for query: %s\n", params.prompt.c_str());
            params.reply = "";
        }

        int64_t t_turn_end = timer_us();

        if (params.streaming_reply) {
            printf("\n<<\n");
        } else {
            if (params.stop_char) {
                for (auto c : params.reply) {
                    printf("%c", c);
                    if (c == params.stop_char) {
                        break;
                    }
                }
                printf("\n\n");
            } else {
                printf("%s\n\n", params.reply.c_str());
            }
        }

        // Record turn stats
        TurnStats ts;
        double elapsed_ms = (t_turn_end - t_turn_start) / 1000.0;
        ts.ttft_ms = elapsed_ms;  // approximate (includes full generation)
        ts.generated_tokens = params.total_llm_tokens_generated;
        ts.prompt_tokens = 0;     // not directly available per-turn
        ts.tokens_per_sec = (elapsed_ms > 0 && ts.generated_tokens > 0)
            ? (ts.generated_tokens * 1000.0 / elapsed_ms) : 0;
        ts.summary = user_text.substr(0, std::min(user_text.size(), size_t(60)));
        turn_stats.push_back(std::move(ts));

        params.first_prompt = false;
        prompt_index++;
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
