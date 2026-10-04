// laya-cli — native Laya decision-model harness for examples/llm-infer
//
// Laya is a ModernBERT *scorer*, not a generator. llama builds the graph so
// that a decision model emits one scalar per question type (choice, score,
// noul) for every token, i.e. n_embd_out == 3. The systemone prompt places one
// [MASK] token in front of each option; the scalar read at that [MASK]
// position, in the column matching the question type, is that option's score.
// A softmax over those scores (with a temperature stored in the GGUF) gives the
// answer distribution. One forward pass answers a whole question.
//
// Prompt layout (mirrors conversion/bert.py::_systemone_template):
//
//   [CLS] {type} question: {instructions} [SEP]
//   ( [MASK] {option} )*
//   [SEP] {state} [SEP]
//
// This tool re-implements the laya path of tools/server/server-decision.cpp so
// the same custom_prompts-style scenario file can be run without a server.
// It evaluates two designs side-by-side against gold labels:
//
//   choice : one direct N-way question
//   noul   : a battery of independent yes/no questions folded back into an
//            answer by a policy table
//
// Usage:
//   laya-cli -m Laya-Q8_0.gguf -f prompts/custom_prompts_laya.txt [options]

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// must match server_decision_question_type: the value is the column index that
// is read out of the n_embd_out==3 embedding row
enum laya_qtype {
    LAYA_QTYPE_CHOICE = 0,
    LAYA_QTYPE_SCORE  = 1,
    LAYA_QTYPE_NOUL   = 2,
};

static const char * laya_qtype_name(laya_qtype t) {
    switch (t) {
        case LAYA_QTYPE_CHOICE: return "choice";
        case LAYA_QTYPE_SCORE:  return "score";
        case LAYA_QTYPE_NOUL:   return "noul";
    }
    return "?";
}

struct laya_option {
    std::string key;
    std::string description; // may be empty
};

struct laya_question {
    laya_qtype               type = LAYA_QTYPE_CHOICE;
    std::string              id;
    std::string              instructions;
    std::vector<laya_option> options;
};

struct laya_answer {
    std::vector<double> probs;      // one per option, sums to 1
    size_t              best = 0;   // argmax
    double              confidence = 0.0;
    double              noul = 0.0; // probability of "true", noul questions only
    int                 n_prompt_tokens = 0;
};

//
// string helpers
//

static std::string laya_trim(const std::string & s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) {
        return std::string();
    }
    const size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::string laya_replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return s;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

//
// the model context
//

struct laya_context {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    llama_token token_cls  = LLAMA_TOKEN_NULL;
    llama_token token_sep  = LLAMA_TOKEN_NULL;
    llama_token token_mask = LLAMA_TOKEN_NULL;

    std::string text_marker;   // decoded [MASK], stripped from user content
    size_t      max_head_tokens   = 0;
    size_t      max_option_tokens = 48; // same constant as server-decision.h
    int32_t     n_embd_out = 0;
    int32_t     n_ubatch   = 0;

    std::map<std::string, float> temperatures;

    bool verbose = false;
};

static std::string laya_meta_str(const llama_model * model, const std::string & key) {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
    if (n < 0) {
        return std::string();
    }
    return std::string(buf, n);
}

// mirrors server_decision_context::get_temperature
static float laya_get_temperature(const laya_context & lc, const laya_question & q) {
    const size_t n = q.options.size();
    const std::string type_name = laya_qtype_name(q.type);

    const std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";

    for (const std::string & name : { type_name + "." + bucket, type_name }) {
        const auto it = lc.temperatures.find(name);
        if (it != lc.temperatures.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

// mirrors decision_confidence_choice
static double laya_confidence_choice(const std::vector<double> & probs) {
    if (probs.size() < 2) {
        return 1.0;
    }
    const double uniform = 1.0 / probs.size();
    const double p_max   = *std::max_element(probs.begin(), probs.end());
    return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

//
// prompt rendering
//

// mirrors the non-Julia option branch of _systemone_template
static std::string laya_render_option(const laya_question & q, const laya_option & opt) {
    switch (q.type) {
        case LAYA_QTYPE_CHOICE:
            return opt.description.empty() ? opt.key : opt.key + ": " + opt.description;
        case LAYA_QTYPE_SCORE:
            return "level " + opt.key + ": " + opt.description;
        case LAYA_QTYPE_NOUL:
            if (!opt.description.empty()) {
                return opt.key + ": " + opt.description;
            }
            return opt.key + ": " + (opt.key == "true" ? "yes, the statement holds"
                                                       : "no, the statement does not hold");
    }
    return opt.key;
}

// Build the token stream and record the position of every option marker.
// Equivalent to rendering the template to text then tokenizing it with special
// tokens enabled: BPE merges never cross a special token, so tokenizing the
// pieces separately gives the same result and avoids re-parsing markers.
//
// Truncation mirrors server_decision_context::fill_task_laya.
static bool laya_build_prompt(const laya_context & lc,
                              const laya_question & q,
                              const std::string   & state,
                              llama_tokens        & out_tokens,
                              std::vector<int32_t> & out_markers,
                              std::string         & err) {
    const size_t n_options = q.options.size();
    if (n_options == 0) {
        err = "question '" + q.id + "' has no options";
        return false;
    }

    // the input must not contain the marker of the options
    const std::string instructions = laya_replace_all(q.instructions, lc.text_marker, " ");
    const std::string state_clean  = laya_replace_all(state,          lc.text_marker, " ");

    const std::string head_text = std::string(laya_qtype_name(q.type)) + " question: " + instructions;

    llama_tokens head_toks  = common_tokenize(lc.vocab, head_text,  false, false);
    llama_tokens state_toks = common_tokenize(lc.vocab, state_clean, false, false);

    // marker + text of each option
    std::vector<llama_tokens> options;
    options.reserve(n_options);
    for (const auto & opt : q.options) {
        llama_tokens t;
        t.push_back(lc.token_mask);
        const llama_tokens body = common_tokenize(lc.vocab, " " + laya_render_option(q, opt), false, false);
        t.insert(t.end(), body.begin(), body.end());
        options.push_back(std::move(t));
    }

    size_t n_options_tokens = 0;
    auto set_max = [&](size_t n_max) {
        n_options_tokens = 0;
        for (auto & opt : options) {
            opt.resize(std::min(opt.size(), n_max));
            n_options_tokens += opt.size();
        }
    };

    const size_t max_head = lc.max_head_tokens;

    set_max(lc.max_option_tokens + 1);
    if (n_options_tokens + 16 > max_head) {
        // too many or too long options, shrink them evenly
        set_max(std::max((size_t) 4, (max_head - std::min(max_head, (size_t) 16)) / n_options));
    }
    const size_t n_question_max = std::max((size_t) 8, max_head - std::min(max_head, n_options_tokens));

    if (head_toks.size() > n_question_max) {
        head_toks.resize(n_question_max);
    }

    out_tokens.clear();
    out_markers.clear();

    out_tokens.push_back(lc.token_cls);
    out_tokens.insert(out_tokens.end(), head_toks.begin(), head_toks.end());
    out_tokens.push_back(lc.token_sep);
    for (const auto & opt : options) {
        out_markers.push_back((int32_t) out_tokens.size());
        out_tokens.insert(out_tokens.end(), opt.begin(), opt.end());
    }
    out_tokens.push_back(lc.token_sep);
    out_tokens.insert(out_tokens.end(), state_toks.begin(), state_toks.end());
    out_tokens.push_back(lc.token_sep);

    if ((int32_t) out_tokens.size() > lc.n_ubatch) {
        err = "prompt is " + std::to_string(out_tokens.size()) +
              " tokens but the ubatch is " + std::to_string(lc.n_ubatch) +
              "; the whole prompt must fit in one batch (raise -c)";
        return false;
    }
    return true;
}

//
// evaluation
//

static bool laya_evaluate(laya_context       & lc,
                          const laya_question & q,
                          const std::string   & state,
                          laya_answer         & out,
                          std::string         & err) {
    llama_tokens         tokens;
    std::vector<int32_t> markers;
    if (!laya_build_prompt(lc, q, state, tokens, markers, err)) {
        return false;
    }

    if (lc.verbose) {
        std::string dump;
        for (size_t i = 0; i < tokens.size(); i++) {
            dump += common_token_to_piece(lc.ctx, tokens[i], true);
        }
        fprintf(stderr, "\n--- prompt for '%s' (%zu tokens, %zu options) ---\n%s\n---\n",
                q.id.c_str(), tokens.size(), markers.size(), dump.c_str());
    }

    // the model is an encoder: no state carries over between questions
    llama_memory_t mem = llama_get_memory(lc.ctx);
    if (mem != nullptr) {
        llama_memory_clear(mem, true);
    }

    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) tokens.size();
    for (size_t i = 0; i < tokens.size(); i++) {
        // every token is an output so a marker can be indexed by its batch index
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 1;
    }

    const int32_t rc = llama_decode(lc.ctx, batch);
    llama_batch_free(batch);

    if (rc != 0) {
        err = "llama_decode failed with " + std::to_string(rc);
        return false;
    }

    const int32_t column = (int32_t) q.type;
    if (column >= lc.n_embd_out) {
        err = "question type " + std::string(laya_qtype_name(q.type)) +
              " needs column " + std::to_string(column) +
              " but the model outputs only " + std::to_string(lc.n_embd_out);
        return false;
    }

    std::vector<float> scores;
    scores.reserve(markers.size());
    for (const int32_t marker : markers) {
        const float * embd = llama_get_embeddings_ith(lc.ctx, marker);
        if (embd == nullptr) {
            err = "failed to get embeddings at marker " + std::to_string(marker);
            return false;
        }
        scores.push_back(embd[column]);
    }

    // softmax with the fitted temperature (laya has a single variant)
    const float  temperature = laya_get_temperature(lc, q);
    const size_t n           = scores.size();
    const float  score_max   = *std::max_element(scores.begin(), scores.end());

    out.probs.assign(n, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        out.probs[i] = std::exp((double) (scores[i] - score_max) / temperature);
        sum += out.probs[i];
    }
    for (size_t i = 0; i < n; i++) {
        out.probs[i] /= sum;
    }

    out.best            = std::max_element(out.probs.begin(), out.probs.end()) - out.probs.begin();
    out.confidence      = laya_confidence_choice(out.probs);
    out.n_prompt_tokens = (int32_t) tokens.size();

    out.noul = 0.0;
    if (q.type == LAYA_QTYPE_NOUL) {
        for (size_t i = 0; i < n; i++) {
            if (q.options[i].key == "true") {
                out.noul = out.probs[i];
            }
        }
    }
    return true;
}

//
// scenario file
//

struct laya_policy_rule {
    std::string id;         // noul question id, or "*" for the fallback
    char        op = '>';   // '>' or '<'
    double      threshold = 0.0;
    std::string key;        // choice key to emit
};

struct laya_scenario {
    std::string                   state_template;
    laya_question                 choice;
    std::vector<laya_question>    nouls;
    std::vector<laya_policy_rule> policy;
    std::vector<std::string>      prompts;
    std::vector<std::string>      gold;
};

static bool laya_load_scenario(const std::string & path, laya_scenario & sc, std::string & err) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open '" + path + "'";
        return false;
    }

    sc.choice.type = LAYA_QTYPE_CHOICE;
    sc.choice.id   = "choice";

    std::string section;
    std::string line;
    std::vector<std::string> state_lines;

    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string trimmed = laya_trim(line);

        if (!trimmed.empty() && trimmed[0] == '#') {
            continue;
        }
        if (trimmed == "END_SECTION") {
            section.clear();
            continue;
        }
        if (section.empty()) {
            if (trimmed == "LAYA_STATE" || trimmed == "LAYA_CHOICE" || trimmed == "LAYA_NOUL" ||
                trimmed == "LAYA_POLICY" || trimmed == "LAYA_PROMPTS") {
                section = trimmed;
            }
            continue;
        }

        if (section == "LAYA_STATE") {
            state_lines.push_back(line);
            continue;
        }
        if (trimmed.empty()) {
            continue;
        }

        if (section == "LAYA_CHOICE") {
            const size_t colon = trimmed.find(':');
            if (colon == std::string::npos) {
                err = "LAYA_CHOICE: expected '<key>: <text>' but got: " + trimmed;
                return false;
            }
            const std::string key  = laya_trim(trimmed.substr(0, colon));
            const std::string text = laya_trim(trimmed.substr(colon + 1));
            if (key == "instructions") {
                sc.choice.instructions = text;
            } else {
                sc.choice.options.push_back({ key, text });
            }
        } else if (section == "LAYA_NOUL") {
            const size_t colon = trimmed.find(':');
            if (colon == std::string::npos) {
                err = "LAYA_NOUL: expected '<id>: <statement>' but got: " + trimmed;
                return false;
            }
            laya_question q;
            q.type         = LAYA_QTYPE_NOUL;
            q.id           = laya_trim(trimmed.substr(0, colon));
            q.instructions = laya_trim(trimmed.substr(colon + 1));
            // laya is not noul_true_first, so the order is false then true
            q.options.push_back({ "false", "" });
            q.options.push_back({ "true",  "" });
            sc.nouls.push_back(std::move(q));
        } else if (section == "LAYA_POLICY") {
            const size_t arrow = trimmed.find("->");
            if (arrow == std::string::npos) {
                err = "LAYA_POLICY: expected '... -> <key>' but got: " + trimmed;
                return false;
            }
            laya_policy_rule rule;
            rule.key = laya_trim(trimmed.substr(arrow + 2));

            std::istringstream cond(laya_trim(trimmed.substr(0, arrow)));
            std::string tok;
            cond >> tok;
            rule.id = tok;
            if (rule.id != "*") {
                std::string op;
                if (!(cond >> op) || !(cond >> rule.threshold) || (op != ">" && op != "<")) {
                    err = "LAYA_POLICY: expected '<id> > <threshold> -> <key>' but got: " + trimmed;
                    return false;
                }
                rule.op = op[0];
            }
            sc.policy.push_back(std::move(rule));
        } else if (section == "LAYA_PROMPTS") {
            // gold labels are optional: "<gold> | <utterance>" or just "<utterance>"
            const size_t bar = trimmed.find('|');
            if (bar == std::string::npos) {
                sc.gold.push_back("");
                sc.prompts.push_back(trimmed);
            } else {
                sc.gold.push_back(laya_trim(trimmed.substr(0, bar)));
                sc.prompts.push_back(laya_trim(trimmed.substr(bar + 1)));
            }
        }
    }

    // keep the blank lines inside the state but drop the leading/trailing ones
    while (!state_lines.empty() && laya_trim(state_lines.front()).empty()) {
        state_lines.erase(state_lines.begin());
    }
    while (!state_lines.empty() && laya_trim(state_lines.back()).empty()) {
        state_lines.pop_back();
    }
    for (size_t i = 0; i < state_lines.size(); i++) {
        sc.state_template += state_lines[i];
        if (i + 1 < state_lines.size()) {
            sc.state_template += "\n";
        }
    }

    if (sc.state_template.empty()) {
        err = "LAYA_STATE is missing or empty";
        return false;
    }
    if (sc.prompts.empty()) {
        err = "LAYA_PROMPTS is missing or empty";
        return false;
    }
    return true;
}

// Resolve the noul battery into an option key.
//
// A rule may target the literal "@choice" instead of an option key, in which
// case the answer is taken from the LAYA_CHOICE question. That lets a scenario
// use noul questions purely as *guards* for the cases a choice list cannot
// express -- a catch-all "none of the above" option, or a refusal -- and leave
// the discriminations the choice is actually good at to the choice itself.
// Laya scores every option positively, so a catch-all option can never win on
// its own; it has to be detected by a positively-phrased guard question.
static std::string laya_apply_policy(const laya_scenario & sc,
                                     const std::map<std::string, double> & nouls,
                                     const std::string & choice_pick) {
    for (const auto & rule : sc.policy) {
        std::string key;
        bool        hit = false;
        if (rule.id == "*") {
            hit = true;
        } else {
            const auto it = nouls.find(rule.id);
            if (it == nouls.end()) {
                continue;
            }
            hit = (rule.op == '>' ? it->second > rule.threshold : it->second < rule.threshold);
        }
        if (hit) {
            return rule.key == "@choice" ? choice_pick : rule.key;
        }
    }
    return std::string();
}

//
// main
//

static void laya_usage(const char * prog) {
    printf("usage: %s -m MODEL.gguf -f SCENARIO.txt [options]\n\n", prog);
    printf("  -m,  --model PATH     path to a laya decision GGUF (required)\n");
    printf("  -f,  --file PATH      scenario file (required)\n");
    printf("  -c,  --ctx-size N     context / batch size      (default 2048)\n");
    printf("  -t,  --threads N      threads                   (default 8)\n");
    printf("  -ngl,--n-gpu-layers N layers offloaded to a GPU (default 0)\n");
    printf("       --mode MODE      choice | noul | both      (default both)\n");
    printf("       --verbose        dump every rendered prompt\n");
    printf("  -h,  --help           this message\n");
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string file_path;
    int32_t n_ctx    = 2048;
    int32_t n_thread = 8;
    int32_t n_gpu    = 0;
    std::string mode = "both";
    bool verbose     = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            model_path = next("--model");
        } else if (a == "-f" || a == "--file") {
            file_path = next("--file");
        } else if (a == "-c" || a == "--ctx-size") {
            n_ctx = std::atoi(next("--ctx-size").c_str());
        } else if (a == "-t" || a == "--threads") {
            n_thread = std::atoi(next("--threads").c_str());
        } else if (a == "-ngl" || a == "--n-gpu-layers") {
            n_gpu = std::atoi(next("--n-gpu-layers").c_str());
        } else if (a == "--mode") {
            mode = next("--mode");
        } else if (a == "--verbose") {
            verbose = true;
        } else if (a == "-h" || a == "--help") {
            laya_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            laya_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty() || file_path.empty()) {
        laya_usage(argv[0]);
        return 1;
    }
    if (mode != "choice" && mode != "noul" && mode != "both") {
        fprintf(stderr, "error: --mode must be choice, noul or both\n");
        return 1;
    }

    laya_scenario sc;
    std::string err;
    if (!laya_load_scenario(file_path, sc, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (mode != "noul" && sc.choice.options.empty()) {
        fprintf(stderr, "error: --mode %s needs a LAYA_CHOICE section with options\n", mode.c_str());
        return 1;
    }
    if (mode != "choice" && sc.nouls.empty()) {
        fprintf(stderr, "error: --mode %s needs a LAYA_NOUL section\n", mode.c_str());
        return 1;
    }

    llama_backend_init();

    laya_context lc;
    lc.verbose = verbose;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;

    lc.model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (lc.model == nullptr) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    const std::string arch   = laya_meta_str(lc.model, "general.architecture");
    const std::string prefix = arch + ".decision.";
    const std::string dtype  = laya_meta_str(lc.model, prefix + "type");
    if (dtype != "laya") {
        fprintf(stderr, "error: '%s' is not a laya decision model (%s.decision.type = '%s')\n",
                model_path.c_str(), arch.c_str(), dtype.c_str());
        llama_model_free(lc.model);
        llama_backend_free();
        return 1;
    }

    lc.max_head_tokens = std::strtoul(laya_meta_str(lc.model, prefix + "max_head_tokens").c_str(), nullptr, 10);
    if (lc.max_head_tokens == 0) {
        fprintf(stderr, "error: model has no valid %smax_head_tokens\n", prefix.c_str());
        llama_model_free(lc.model);
        llama_backend_free();
        return 1;
    }
    for (const char * t : { "choice", "score", "noul" }) {
        for (const char * b : { "", ".2", ".3_5", ".6_10", ".11" }) {
            const std::string name = std::string(t) + b;
            const std::string val  = laya_meta_str(lc.model, prefix + "temperature." + name);
            if (!val.empty()) {
                lc.temperatures[name] = std::strtof(val.c_str(), nullptr);
            }
        }
    }

    lc.vocab      = llama_model_get_vocab(lc.model);
    lc.token_sep  = llama_vocab_sep(lc.vocab);
    lc.token_mask = llama_vocab_mask(lc.vocab);
    lc.token_cls  = llama_vocab_bos(lc.vocab); // [CLS] is the BOS of a BERT vocab
    lc.n_embd_out = llama_model_n_embd_out(lc.model);

    if (lc.token_sep == LLAMA_TOKEN_NULL || lc.token_mask == LLAMA_TOKEN_NULL ||
        lc.token_cls == LLAMA_TOKEN_NULL) {
        fprintf(stderr, "error: model is missing a cls, sep or mask token\n");
        llama_model_free(lc.model);
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_ctx;
    cparams.n_batch         = n_ctx; // the whole prompt must fit in one batch
    cparams.n_ubatch        = n_ctx;
    cparams.n_threads       = n_thread;
    cparams.n_threads_batch = n_thread;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE; // one row per token

    lc.ctx = llama_init_from_model(lc.model, cparams);
    if (lc.ctx == nullptr) {
        fprintf(stderr, "error: failed to create the context\n");
        llama_model_free(lc.model);
        llama_backend_free();
        return 1;
    }
    lc.n_ubatch    = (int32_t) llama_n_ubatch(lc.ctx);
    lc.text_marker = common_token_to_piece(lc.ctx, lc.token_mask, true);

    printf("\n");
    printf("model          : %s\n", model_path.c_str());
    printf("scenario       : %s\n", file_path.c_str());
    printf("arch           : %s (decision.type = %s)\n", arch.c_str(), dtype.c_str());
    printf("n_embd_out     : %d (choice, score, noul)\n", lc.n_embd_out);
    printf("max_head_tokens: %zu\n", lc.max_head_tokens);
    printf("marker / sep   : '%s' (%d) / %d\n", lc.text_marker.c_str(), lc.token_mask, lc.token_sep);
    printf("prompts        : %zu\n", sc.prompts.size());
    printf("\n");

    const bool do_noul_requested = (mode == "noul" || mode == "both");

    const bool uses_choice_fallback =
        std::any_of(sc.policy.begin(), sc.policy.end(),
                    [](const laya_policy_rule & r) { return r.key == "@choice"; });

    const bool do_choice = (mode == "choice" || mode == "both") || (do_noul_requested && uses_choice_fallback);
    const bool do_noul   = do_noul_requested;

    size_t n_ok_choice = 0;
    size_t n_ok_policy = 0;
    size_t n_labeled   = 0;
    bool   failed      = false;

    std::map<std::string, size_t> hist_choice;
    std::map<std::string, size_t> hist_policy;
    double sum_conf = 0.0;

    for (size_t i = 0; i < sc.prompts.size() && !failed; i++) {
        const std::string state = laya_replace_all(sc.state_template, "{message}", sc.prompts[i]);
        const bool        has_gold = !sc.gold[i].empty();
        n_labeled += has_gold ? 1 : 0;

        if (has_gold) {
            printf("[%2zu] gold=%s  \"%s\"\n", i + 1, sc.gold[i].c_str(), sc.prompts[i].c_str());
        } else {
            printf("[%2zu] \"%s\"\n", i + 1, sc.prompts[i].c_str());
        }

        std::string choice_pick;

        if (do_choice) {
            laya_answer ans;
            if (!laya_evaluate(lc, sc.choice, state, ans, err)) {
                fprintf(stderr, "error: choice question failed: %s\n", err.c_str());
                failed = true;
                break;
            }
            const std::string picked = sc.choice.options[ans.best].key;
            choice_pick = picked;
            const bool ok = has_gold && (picked == sc.gold[i]);
            n_ok_choice += ok ? 1 : 0;
            hist_choice[picked]++;
            sum_conf += ans.confidence;

            printf("     choice : %s %s conf=%.3f  [", picked.c_str(),
                   has_gold ? (ok ? "OK   " : "MISS ") : "", ans.confidence);
            for (size_t k = 0; k < ans.probs.size(); k++) {
                printf("%s%s=%.3f", k ? " " : "", sc.choice.options[k].key.c_str(), ans.probs[k]);
            }
            printf("]  (%d tok)\n", ans.n_prompt_tokens);
        }

        if (do_noul) {
            std::map<std::string, double> nouls;
            printf("     noul   : ");
            for (const auto & q : sc.nouls) {
                laya_answer ans;
                if (!laya_evaluate(lc, q, state, ans, err)) {
                    fprintf(stderr, "\nerror: noul question '%s' failed: %s\n", q.id.c_str(), err.c_str());
                    failed = true;
                    break;
                }
                nouls[q.id] = ans.noul;
                printf("%s=%.3f  ", q.id.c_str(), ans.noul);
            }
            if (failed) {
                break;
            }
            printf("\n");

            if (!sc.policy.empty()) {
                const std::string picked = laya_apply_policy(sc, nouls, choice_pick);
                const bool ok = has_gold && (picked == sc.gold[i]);
                n_ok_policy += ok ? 1 : 0;
                hist_policy[picked]++;
                printf("     policy : %s %s\n", picked.empty() ? "(none)" : picked.c_str(),
                       has_gold ? (ok ? "OK" : "MISS") : "");
            }
        }
        printf("\n");
    }

    if (!failed) {
        const size_t n = sc.prompts.size();
        printf("================ summary ================\n");
        if (n_labeled > 0) {
            if (do_choice) {
                printf("  direct choice   : %2zu / %2zu  (%.1f%%)\n",
                       n_ok_choice, n_labeled, 100.0 * (double) n_ok_choice / (double) n_labeled);
            }
            if (do_noul && !sc.policy.empty()) {
                printf("  noul + policy   : %2zu / %2zu  (%.1f%%)\n",
                       n_ok_policy, n_labeled, 100.0 * (double) n_ok_policy / (double) n_labeled);
            }
            if (n_labeled < n) {
                printf("  (%zu of %zu prompts carry a gold label)\n", n_labeled, n);
            }
        } else {
            printf("  no gold labels -- reporting distribution only (%zu prompts)\n", n);
        }
        if (do_choice) {
            printf("  mean confidence : %.3f\n", sum_conf / (double) n);
            printf("  choice spread   :");
            for (const auto & o : sc.choice.options) {
                const size_t c = hist_choice.count(o.key) ? hist_choice.at(o.key) : 0;
                printf(" %s=%zu", o.key.c_str(), c);
            }
            printf("\n");
        }
        if (do_noul && !sc.policy.empty()) {
            printf("  policy spread   :");
            for (const auto & kv : hist_policy) {
                printf(" %s=%zu", kv.first.empty() ? "(none)" : kv.first.c_str(), kv.second);
            }
            printf("\n");
        }
        printf("=========================================\n\n");
    }

    llama_free(lc.ctx);
    llama_model_free(lc.model);
    llama_backend_free();

    return failed ? 1 : 0;
}
