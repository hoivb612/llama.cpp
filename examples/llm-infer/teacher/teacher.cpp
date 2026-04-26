//
// teacher.cpp — Teacher/Student model comparison tool for llm-infer
//
// Runs the same SYSTEM/PROMPT script through two models (teacher and student),
// compares their JSON answers, and when they disagree, asks the teacher model
// to explain the reasoning that leads to the correct answer.
//
// All configuration is specified via a JSON file:
//
//   {
//     "teacher_model": "path/to/teacher.gguf",
//     "student_model": "path/to/student.gguf",
//     "script_file":   "path/to/script.txt",
//     "output_file":   "results.json",
//     "n_threads":     4,
//     "n_ctx":         2048,
//     "verbose":       0,
//     "stop_char":     "}"
//   }
//
// Usage:
//   teacher CONFIG.json [verbose]
//
// Three-pass execution:
//   Pass 1 — Load teacher model, run all prompts, save JSON answers
//   Pass 2 — Load student model, run same prompts, save JSON answers
//   Pass 3 — Re-load teacher, for each mismatch ask for reasoning
//
// Output: a JSON file with mismatches and teacher reasoning.
//

#pragma warning (disable:4267)

#include "llm-infer.h"
#include "b612-cpu.h"
#include "llama.h"
#include "json.hpp"

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
#include <algorithm>

#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using json = nlohmann::json;

// ─── Timer ──────────────────────────────────────────────────────────────────

static int64_t timer_freq = 0, timer_start = 0;
static void timer_init(void) {
    if (!timer_freq) {
        LARGE_INTEGER t;
        QueryPerformanceFrequency(&t);
        timer_freq = t.QuadPart;
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

static int64_t timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
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

static bool is_meta_command(const std::string & t) {
    std::string lc = to_lower(t);
    return lc == "/context" || lc.substr(0, 7) == "/rewind" || lc == "quit()";
}

// ─── Parsed Prompt / Script Types (OG format) ──────────────────────────────

struct ParsedPrompt {
    std::string text;
    bool is_meta = false;
    std::string raw;
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

static Script ParseScriptFile(const std::string& filepath) {
    Script script;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        printf("[teacher]: failed to open script file [%s]\n", filepath.c_str());
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

// ─── Chat Template Helpers ──────────────────────────────────────────────────

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

    if (t.find("<|user|>") != std::string::npos &&
        t.find("<|assistant|>") != std::string::npos &&
        t.find("<|end|>") != std::string::npos) {
        *out_label = "Phi-3/4";
        out_sys_prefix = "<|system|>\n" + system_prompt + "<|end|>\n";
        out_turn_tmpl  = "<|user|>\n{message}<|end|>\n<|assistant|>\n";
        return true;
    }

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

    if (t.find("<|turn>") != std::string::npos &&
        t.find("<turn|>") != std::string::npos) {
        *out_label = "Gemma-4";
        out_sys_prefix = "<|turn>system\n" + system_prompt + "<turn|>\n";
        out_turn_tmpl  = "<|turn>user\n{message}<turn|>\n<|turn>model\n";
        return true;
    }

    if (t.find("<start_of_turn>") != std::string::npos &&
        t.find("<end_of_turn>")   != std::string::npos) {
        *out_label = "Gemma";
        out_sys_prefix = "<start_of_turn>user\n" + system_prompt + "\n";
        out_turn_tmpl  = "{message}<end_of_turn>\n<start_of_turn>model\n";
        return true;
    }

    if (t.find("<|im_start|>") != std::string::npos &&
        t.find("<|im_end|>")   != std::string::npos) {
        *out_label = "ChatML";
        out_sys_prefix = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
        out_turn_tmpl  = "<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n";
        return true;
    }

    if (t.find("[INST]") != std::string::npos &&
        t.find("[/INST]") != std::string::npos) {
        *out_label = "Mistral/Llama-2";
        out_sys_prefix = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n";
        out_turn_tmpl  = "{message} [/INST] ";
        return true;
    }

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

// ─── JSON Answer Extraction ─────────────────────────────────────────────────

struct AnswerResult {
    std::string raw_reply;     // full model output
    std::string answer_text;   // "answer" field value (e.g., "b. You might be confused...")
    char answer_letter = 0;    // extracted letter (e.g., 'b')
    std::string justification; // "justification" field value
    bool valid = false;        // true if JSON was parseable
};

// Extract the answer letter from a string like "b. You might be confused..."
static char extract_answer_letter(const std::string & answer) {
    for (char c : answer) {
        if (isspace(c)) continue;
        if (isalpha(c)) return static_cast<char>(tolower(static_cast<unsigned char>(c)));
        break;
    }
    return 0;
}

// Truncate reply at stop_char and parse as JSON
static AnswerResult parse_model_answer(const std::string & reply, char stop_char) {
    AnswerResult result;
    result.raw_reply = reply;

    // Truncate at stop_char (inclusive)
    std::string truncated;
    if (stop_char) {
        for (char c : reply) {
            truncated += c;
            if (c == stop_char) break;
        }
    } else {
        truncated = reply;
    }

    // Find the JSON object boundaries
    size_t brace_start = truncated.find('{');
    size_t brace_end   = truncated.rfind('}');
    if (brace_start == std::string::npos || brace_end == std::string::npos ||
        brace_end <= brace_start) {
        result.valid = false;
        return result;
    }

    std::string json_str = truncated.substr(brace_start, brace_end - brace_start + 1);

    try {
        json j = json::parse(json_str);

        if (j.contains("answer") && j["answer"].is_string()) {
            result.answer_text = j["answer"].get<std::string>();
            result.answer_letter = extract_answer_letter(result.answer_text);
            result.valid = true;
        }
        if (j.contains("justification") && j["justification"].is_string()) {
            result.justification = j["justification"].get<std::string>();
        }
    } catch (const json::parse_error &) {
        result.valid = false;
    }

    return result;
}

// ─── Per-Prompt Result ──────────────────────────────────────────────────────

struct PromptResult {
    int index;                  // 1-based prompt index (output prompts only)
    std::string input_prompt;   // user text
    AnswerResult answer;        // parsed answer
};

// ─── Run a Model Pass ───────────────────────────────────────────────────────
// Loads model, runs the script (honoring /rewind), collects results.
// Returns the vector of per-prompt results (meta commands excluded).

static std::vector<PromptResult> run_model_pass(
    const std::string & model_path,
    const Script & original_script,
    const std::string & pass_label,
    model_params & params)
{
    std::vector<PromptResult> results;

    // Copy the prompt queue (each pass needs its own copy)
    std::queue<ParsedPrompt> prompts = original_script.prompts;

    // Configure model
    params.model_name = model_path;
    params.first_prompt = true;

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  %s — loading model: %s\n", pass_label.c_str(), model_path.c_str());
    printf("════════════════════════════════════════════════════════════════\n\n");

    llm_disable_log();

    if (!llm_initialize(params)) {
        printf("[%s]: FAILED to initialize model [%s]\n", pass_label.c_str(), model_path.c_str());
        return results;
    }

    // Apply chat template from model
    if (params.custom_template_prompt.empty()) {
        if (!apply_model_chat_template(original_script.system_prompt, params)) {
            printf("[%s]: WARNING — chat template detection failed\n", pass_label.c_str());
            params.custom_template_prompt = original_script.system_prompt;
        }
    }

    // Begin multi-turn session
    llm_multiturn_begin(params);
    printf("[%s]: multi-turn session started\n", pass_label.c_str());

    int prompt_index = 0;

    while (!prompts.empty()) {
        ParsedPrompt prompt = prompts.front();
        prompts.pop();

        if (prompt.raw.empty() && prompt.text.empty())
            continue;

        // ── quit() ──
        if (to_lower(prompt.text) == "quit()") {
            printf("[%s]: quit() — ending pass\n", pass_label.c_str());
            break;
        }

        // ── /context — skip silently ──
        if (to_lower(prompt.text) == "/context") {
            continue;
        }

        // ── /rewind — execute to keep KV cache in sync ──
        int rewind_count = ParseRewindCommand(prompt.text);
        if (rewind_count > 0) {
            int total_turns = llm_multiturn_turn_count();
            if (rewind_count > total_turns) rewind_count = total_turns;
            if (rewind_count > 0) {
                llm_multiturn_rewind(rewind_count);
                printf("[%s]: /rewind %d — turns=%d, tokens=%d\n",
                       pass_label.c_str(), rewind_count,
                       llm_multiturn_turn_count(), llm_multiturn_token_count());
            }
            continue;
        }

        // ── Other meta commands — skip ──
        if (prompt.is_meta) continue;

        // ── Process user prompt ──
        prompt_index++;
        std::string user_text = prompt.text;
        user_text.erase(
            std::remove(user_text.begin(), user_text.end(), '\"'),
            user_text.end());

        // Wrap in turn template
        if (!custom_turn_template.empty()) {
            std::string tmpl = trim(custom_turn_template);
            size_t pos = tmpl.find("{message}");
            if (pos != std::string::npos) {
                tmpl.replace(pos, std::string("{message}").length(), user_text);
            }
            params.prompt = tmpl;
        } else {
            params.prompt = user_text;
        }

        if (!params.parse_special) params.parse_special = true;

        printf("[%s]: prompt %d: %s\n", pass_label.c_str(), prompt_index, user_text.c_str());

        if (!llm_infer_multiturn(params)) {
            printf("[%s]: FAILED token generation for: %s\n", pass_label.c_str(), user_text.c_str());
            params.reply = "";
        }

        // Parse the answer
        AnswerResult ans = parse_model_answer(params.reply, params.stop_char);

        // Print truncated reply
        if (params.stop_char) {
            for (auto c : params.reply) {
                printf("%c", c);
                if (c == params.stop_char) break;
            }
            printf("\n");
        } else {
            printf("%s\n", params.reply.c_str());
        }

        PromptResult pr;
        pr.index = prompt_index;
        pr.input_prompt = user_text;
        pr.answer = ans;
        results.push_back(std::move(pr));

        params.first_prompt = false;
    }

    // Cleanup
    llm_enable_log();
    llm_terminate(params);

    // Reset template state for next pass
    params.custom_template_prompt = "";
    custom_turn_template = "";
    params.turn_template = "";

    printf("[%s]: pass complete — %zu prompt results collected\n",
           pass_label.c_str(), results.size());

    return results;
}

// ─── Teacher Reasoning Pass ─────────────────────────────────────────────────
// Re-loads teacher model and for each mismatch, sends a meta-prompt asking
// for reasoning. Uses a fresh system prompt optimized for analysis.

struct MismatchEntry {
    int prompt_index;
    std::string input_prompt;
    std::string teacher_answer;
    std::string student_answer;
    std::string teacher_justification;
    std::string student_justification;
    std::string teacher_reasoning;
};

static std::vector<MismatchEntry> run_teacher_reasoning(
    const std::string & teacher_model,
    const std::vector<MismatchEntry> & mismatches,
    const std::string & original_system_prompt,
    model_params & params)
{
    std::vector<MismatchEntry> results = mismatches;

    if (results.empty()) {
        printf("[reasoning]: no mismatches to analyze\n");
        return results;
    }

    params.model_name = teacher_model;
    params.first_prompt = true;
    params.custom_template_prompt = "";
    custom_turn_template = "";
    params.turn_template = "";

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  Pass 3 — Teacher reasoning analysis (%zu mismatches)\n", results.size());
    printf("════════════════════════════════════════════════════════════════\n\n");

    llm_disable_log();

    if (!llm_initialize(params)) {
        printf("[reasoning]: FAILED to initialize teacher model\n");
        return results;
    }

    // Include the original system prompt so the teacher has full context
    std::string reasoning_system =
        "You are an expert teacher analyzing student mistakes.\n\n"
        "The student was given the following task instructions:\n"
        "---\n" + original_system_prompt + "\n---\n\n"
        "For each case, explain the reasoning or heuristic that would have "
        "led the student to the correct answer. Be concise and specific.";

    if (!apply_model_chat_template(reasoning_system, params)) {
        printf("[reasoning]: WARNING — chat template detection failed\n");
        params.custom_template_prompt = reasoning_system;
    }

    llm_multiturn_begin(params);

    // Disable stop_char for reasoning (we want free-form text, not JSON)
    char saved_stop_char = params.stop_char;
    params.stop_char = 0;

    for (size_t i = 0; i < results.size(); i++) {
        auto & entry = results[i];

        // Build the analysis prompt
        std::string meta_prompt =
            "The student was given this prompt: \"" + entry.input_prompt + "\"\n"
            "The student answered: \"" + entry.student_answer + "\"\n"
            "The correct answer is: \"" + entry.teacher_answer + "\"\n"
            "Explain the reasoning or heuristic that would have led the student to the correct answer.";

        // Wrap in turn template
        if (!custom_turn_template.empty()) {
            std::string tmpl = trim(custom_turn_template);
            size_t pos = tmpl.find("{message}");
            if (pos != std::string::npos) {
                tmpl.replace(pos, std::string("{message}").length(), meta_prompt);
            }
            params.prompt = tmpl;
        } else {
            params.prompt = meta_prompt;
        }

        if (!params.parse_special) params.parse_special = true;

        printf("[reasoning]: analyzing mismatch %zu/%zu (prompt %d)...\n",
               i + 1, results.size(), entry.prompt_index);

        if (!llm_infer_multiturn(params)) {
            printf("[reasoning]: FAILED generation for mismatch %zu\n", i + 1);
            entry.teacher_reasoning = "(generation failed)";
        } else {
            entry.teacher_reasoning = trim(params.reply);
            printf("  Reasoning: %s\n\n", entry.teacher_reasoning.c_str());
        }

        params.first_prompt = false;

        // Rewind after each reasoning turn to keep context clean
        llm_multiturn_rewind(1);
    }

    params.stop_char = saved_stop_char;

    llm_enable_log();
    llm_terminate(params);

    printf("[reasoning]: analysis complete\n");
    return results;
}

// ─── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    timer_init();
    int64_t t0 = timer_us();

    if (argc < 2 || argv[1][0] == '-') {
        printf("usage: %s CONFIG.json [verbose]\n", argv[0]);
        printf("\n");
        printf("  CONFIG.json fields:\n");
        printf("    teacher_model  — path to teacher model (.gguf)\n");
        printf("    student_model  — path to student model (.gguf)\n");
        printf("    script_file    — path to SYSTEM/PROMPT script (.txt)\n");
        printf("    output_file    — path for result JSON output\n");
        printf("    n_threads      — number of threads (default: 4)\n");
        printf("    n_ctx          — context size (default: 2048)\n");
        printf("    verbose        — 0, 1, or 2 (default: 0)\n");
        printf("    stop_char      — stop character (default: \"}\")\n");
        return 1;
    }

    // ── Load config JSON ──
    std::string config_path = argv[1];
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        printf("[teacher]: failed to open config file [%s]\n", config_path.c_str());
        return 1;
    }

    json config;
    try {
        config = json::parse(config_file);
    } catch (const json::parse_error & e) {
        printf("[teacher]: JSON parse error in [%s]: %s\n", config_path.c_str(), e.what());
        return 1;
    }
    config_file.close();

    // Required fields
    if (!config.contains("teacher_model") || !config.contains("student_model") ||
        !config.contains("script_file")   || !config.contains("output_file")) {
        printf("[teacher]: config must contain: teacher_model, student_model, script_file, output_file\n");
        return 1;
    }

    std::string teacher_model = config["teacher_model"].get<std::string>();
    std::string student_model = config["student_model"].get<std::string>();
    std::string script_file   = config["script_file"].get<std::string>();
    std::string output_file   = config["output_file"].get<std::string>();

    // Optional fields
    int n_threads = config.value("n_threads", 4);
    int n_ctx     = config.value("n_ctx", 2048);
    int verbose   = config.value("verbose", 0);
    std::string stop_char_str = config.value("stop_char", std::string("}"));
    char stop_char = stop_char_str.empty() ? '}' : stop_char_str[0];

    // CLI verbose override
    if (argc >= 3 && std::string(argv[2]) == "verbose") {
        verbose = 1;
    }

    printf("[teacher]: config loaded from [%s]\n", config_path.c_str());
    printf("[teacher]:   teacher_model = %s\n", teacher_model.c_str());
    printf("[teacher]:   student_model = %s\n", student_model.c_str());
    printf("[teacher]:   script_file   = %s\n", script_file.c_str());
    printf("[teacher]:   output_file   = %s\n", output_file.c_str());
    printf("[teacher]:   n_threads=%d  n_ctx=%d  verbose=%d  stop_char='%c'\n",
           n_threads, n_ctx, verbose, stop_char);

    // ── Load script ──
    Script script = ParseScriptFile(script_file);
    if (script.prompts.empty()) {
        printf("[teacher]: no prompts found in script file\n");
        return 1;
    }
    printf("[teacher]: loaded %zu prompt(s), system: \"%s\"\n",
           script.prompts.size(),
           script.system_prompt.substr(0, 60).c_str());

    // ── Setup model params ──
    model_params params = {0};
    params.n_threads = n_threads;
    params.n_ctx     = n_ctx;
    params.verbose   = verbose;
    params.stop_char = stop_char;
    params.parse_special = true;

    // ══════════════════════════════════════════════════════════════════════════
    // Pass 1 — Teacher
    // ══════════════════════════════════════════════════════════════════════════

    std::vector<PromptResult> teacher_results = run_model_pass(
        teacher_model, script, "Pass 1 (teacher)", params);

    // ══════════════════════════════════════════════════════════════════════════
    // Pass 2 — Student
    // ══════════════════════════════════════════════════════════════════════════

    std::vector<PromptResult> student_results = run_model_pass(
        student_model, script, "Pass 2 (student)", params);

    // ══════════════════════════════════════════════════════════════════════════
    // Compare answers
    // ══════════════════════════════════════════════════════════════════════════

    printf("\n════════════════════════════════════════════════════════════════\n");
    printf("  Comparing teacher vs student answers\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    size_t compare_count = std::min(teacher_results.size(), student_results.size());
    if (teacher_results.size() != student_results.size()) {
        printf("[teacher]: WARNING — result count mismatch: teacher=%zu, student=%zu\n",
               teacher_results.size(), student_results.size());
        printf("[teacher]: comparing first %zu results\n", compare_count);
    }

    std::vector<MismatchEntry> mismatches;
    int match_count = 0;

    for (size_t i = 0; i < compare_count; i++) {
        auto & t = teacher_results[i];
        auto & s = student_results[i];

        bool t_valid = t.answer.valid;
        bool s_valid = s.answer.valid;

        printf("  Prompt %d: \"%s\"\n", t.index, t.input_prompt.c_str());
        printf("    Teacher: %s [%c] %s\n",
               t_valid ? "VALID" : "INVALID",
               t.answer.answer_letter ? t.answer.answer_letter : '?',
               t.answer.answer_text.c_str());
        printf("    Student: %s [%c] %s\n",
               s_valid ? "VALID" : "INVALID",
               s.answer.answer_letter ? s.answer.answer_letter : '?',
               s.answer.answer_text.c_str());

        bool letters_match = (t.answer.answer_letter != 0 &&
                              s.answer.answer_letter != 0 &&
                              t.answer.answer_letter == s.answer.answer_letter);

        if (letters_match) {
            printf("    → MATCH (%c)\n\n", t.answer.answer_letter);
            match_count++;
        } else {
            printf("    → MISMATCH\n\n");

            MismatchEntry entry;
            entry.prompt_index         = t.index;
            entry.input_prompt         = t.input_prompt;
            entry.teacher_answer       = t.answer.answer_text;
            entry.student_answer       = s.answer.answer_text;
            entry.teacher_justification = t.answer.justification;
            entry.student_justification = s.answer.justification;
            mismatches.push_back(std::move(entry));
        }
    }

    printf("  Summary: %d match, %zu mismatch out of %zu compared\n\n",
           match_count, mismatches.size(), compare_count);

    // ══════════════════════════════════════════════════════════════════════════
    // Pass 3 — Teacher reasoning for mismatches
    // ══════════════════════════════════════════════════════════════════════════

    if (!mismatches.empty()) {
        mismatches = run_teacher_reasoning(teacher_model, mismatches,
                                             script.system_prompt, params);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Write result JSON
    // ══════════════════════════════════════════════════════════════════════════

    json output;
    output["config"] = {
        {"teacher_model", teacher_model},
        {"student_model", student_model},
        {"script_file",   script_file},
        {"n_threads",     n_threads},
        {"n_ctx",         n_ctx}
    };
    output["system_prompt"]  = script.system_prompt;
    output["total_prompts"] = (int)compare_count;
    output["matches"]       = match_count;
    output["mismatches"]    = (int)mismatches.size();

    json results_array = json::array();
    for (auto & m : mismatches) {
        json entry;
        entry["prompt_index"]          = m.prompt_index;
        entry["input_prompt"]          = m.input_prompt;
        entry["teacher_answer"]        = m.teacher_answer;
        entry["student_answer"]        = m.student_answer;
        entry["teacher_justification"] = m.teacher_justification;
        entry["student_justification"] = m.student_justification;
        entry["teacher_reasoning"]     = m.teacher_reasoning;
        results_array.push_back(std::move(entry));
    }
    output["results"] = results_array;

    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        printf("[teacher]: ERROR — failed to open output file [%s]\n", output_file.c_str());
        // Print to stdout as fallback
        printf("%s\n", output.dump(2).c_str());
    } else {
        out_file << output.dump(2) << std::endl;
        out_file.close();
        printf("[teacher]: results written to [%s]\n", output_file.c_str());
    }

    t0 = timer_us() - t0;
    printf("\n total elapsed time %7.2fsec\n", (double)t0 / (1000. * 1000.));

    return 0;
}
