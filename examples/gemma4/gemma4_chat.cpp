// G4.4 -- Hand-path chat loop implementation.
//
// Mirrors the design of examples/phi3/phi3_runtime.cpp::phi3_runtime_run_*
// but drives our NetworkState + network_step instead of llama_decode.
//
// Sampler: when temp > 0, builds a [min_p -> temp -> dist] chain once
// per session, and per gen step populates a llama_token_data_array
// from the F32 logits returned by network_step. Calls
// llama_sampler_apply followed by llama_sampler_accept (the latter is
// needed for stateful samplers per the llama.h docs). When temp <= 0,
// runs a plain argmax with no per-step allocation.
//
// Multi-turn KV invariant: each turn's user-side delta is a hand-built
// Gemma-format string tokenized atomically. We never re-tokenize the
// assistant's prior gen tokens through text (BPE/SentencePiece is not
// round-trip stable). Gen tokens are committed directly to kv_tokens
// and replayed through network_step verbatim. On turn N>=2 the delta
// starts with "<end_of_turn>\n" so a prior turn that exited at the
// n_predict cap (no natural EOG) still closes cleanly in KV.

#include "gemma4_chat.h"
#include "gemma4_forward.h"

#include "llama.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace gemma4 {

namespace {

// ----------------------------------------------------------------------
// Sampler context: either greedy (chain==nullptr) or chain-based.
struct SamplerCtx {
    llama_sampler * chain = nullptr;          // null = greedy
    std::vector<llama_token_data> scratch;    // [n_vocab], reused per step
};

bool sampler_init(SamplerCtx & sc, int n_vocab, const ChatParams & p, std::string & error) {
    sc.scratch.resize((size_t) n_vocab);
    if (p.temp <= 0.0f) {
        sc.chain = nullptr;
        return true;
    }
    sc.chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sc.chain) { error = "sampler_init: llama_sampler_chain_init failed"; return false; }
    llama_sampler_chain_add(sc.chain, llama_sampler_init_min_p(p.min_p, 1));
    llama_sampler_chain_add(sc.chain, llama_sampler_init_temp(p.temp));
    llama_sampler_chain_add(sc.chain, llama_sampler_init_dist(p.seed));
    return true;
}

void sampler_free(SamplerCtx & sc) {
    if (sc.chain) { llama_sampler_free(sc.chain); sc.chain = nullptr; }
    sc.scratch.clear();
}

llama_token sample_token(SamplerCtx & sc, const float * logits, int n_vocab) {
    if (!sc.chain) {
        // Greedy.
        int best = 0;
        float best_v = logits[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        }
        return (llama_token) best;
    }
    for (int v = 0; v < n_vocab; ++v) {
        sc.scratch[v].id    = (llama_token) v;
        sc.scratch[v].logit = logits[v];
        sc.scratch[v].p     = 0.0f;
    }
    llama_token_data_array cur_p {
        sc.scratch.data(), (size_t) n_vocab, -1, false
    };
    llama_sampler_apply(sc.chain, &cur_p);
    if (cur_p.selected < 0 || cur_p.selected >= (int64_t) cur_p.size) {
        // Should not happen with the standard chain; fall back to argmax.
        int best = 0; float best_v = logits[0];
        for (int v = 1; v < n_vocab; ++v) if (logits[v] > best_v) { best_v = logits[v]; best = v; }
        return (llama_token) best;
    }
    const llama_token tok = cur_p.data[cur_p.selected].id;
    llama_sampler_accept(sc.chain, tok);
    return tok;
}

// Tokenize a UTF-8 byte buffer with the given vocab.
// Returns true on success.
bool tokenize_to(const llama_vocab * vocab, const char * text, int text_len,
                 bool add_special, bool parse_special,
                 std::vector<int32_t> & out) {
    out.resize((size_t) std::max(8, text_len + 16));
    int n = llama_tokenize(vocab, text, text_len, out.data(), (int) out.size(),
                           add_special, parse_special);
    if (n < 0) {
        out.resize((size_t) (-n + 16));
        n = llama_tokenize(vocab, text, text_len, out.data(), (int) out.size(),
                           add_special, parse_special);
    }
    if (n < 0) return false;
    out.resize((size_t) n);
    return true;
}

// Build the user-side delta string for a given turn.
//
// This bypasses apply_template entirely after the first call so that
// we never re-tokenize the assistant's prior response (round-tripping
// gen tokens -> text -> tokens is NOT safe in BPE/SentencePiece, and
// observed drift on E4B confirmed this).
//
// Format (Gemma):
//   first_turn == true:
//     "<start_of_turn>user\n{Q}<end_of_turn>\n<start_of_turn>model\n"
//   first_turn == false:
//     "<end_of_turn>\n<start_of_turn>user\n{Q}<end_of_turn>\n<start_of_turn>model\n"
//
// The leading "<end_of_turn>\n" on turn >=2 closes the prior assistant
// response that the model may not have naturally emitted (e.g. when
// gen stopped at the n_predict cap). It is tokenized atomically with
// the rest of the delta so the boundary tokens are deterministic.
std::string build_gemma_user_delta(const std::string & q, bool first_turn) {
    std::string s;
    if (!first_turn) s += "<end_of_turn>\n";
    s += "<start_of_turn>user\n";
    s += q;
    s += "<end_of_turn>\n<start_of_turn>model\n";
    return s;
}

// Convert one token to its display piece.
std::string token_piece(const llama_vocab * vocab, llama_token tok) {
    char buf[256] = {0};
    int n = llama_token_to_piece(vocab, tok, buf, (int) sizeof(buf) - 1, 0, true);
    if (n <= 0) return std::string();
    return std::string(buf, buf + n);
}

// Resolve a usable chat template. The Gemma-4 GGUF ships a complex
// Jinja template (with macros for tool calling) that the built-in
// llama_chat_apply_template parser does not recognize, so it returns
// -1 on every call. We probe the model's template with a one-message
// dry run; on failure we substitute a minimal hint string that the
// matcher will detect as the GEMMA template (it just needs to contain
// "<start_of_turn>"; the actual format function in llama-chat.cpp
// handles role mapping including "assistant" -> "model"). If the
// fallback also fails the model is incompatible and we return nullptr.
const char * resolve_chat_template(const llama_model * model) {
    static const char kGemmaHint[] = "<start_of_turn>";
    const char * raw = llama_model_chat_template(model, nullptr);
    if (raw) {
        char tmp[1024];
        llama_chat_message m{"user", "x"};
        const int n = llama_chat_apply_template(raw, &m, 1, true, tmp, (int) sizeof(tmp));
        if (n > 0) return raw;
    }
    // Try the gemma hint fallback.
    char tmp[1024];
    llama_chat_message m{"user", "x"};
    const int n = llama_chat_apply_template(kGemmaHint, &m, 1, true, tmp, (int) sizeof(tmp));
    if (n > 0) {
        std::fprintf(stderr,
            "[gemma4 chat: model template is unrecognized Jinja; "
            "using built-in 'gemma' chat template]\n");
        return kGemmaHint;
    }
    return nullptr;
}

// ----------------------------------------------------------------------
// Run one chat turn.
//
// Strategy: hand-build the user-side delta as a Gemma-format string
// (see build_gemma_user_delta) and tokenize it as one atomic chunk.
// This avoids any re-tokenization of the assistant's prior response
// (which is unsafe in BPE/SentencePiece -- text round-trips do not
// always give back the same token sequence). Decoded gen tokens are
// appended directly to kv_tokens.
//
// Inputs:
//   first_turn  -- drives the delta format (with/without leading
//                  "<end_of_turn>\n") AND add_special on tokenize.
// Outputs (counts/timing):
//   prefill_ms, gen_ms, prompt_tokens, gen_count, response_out.
bool run_one_turn(const ModelF32 & mf, const llama_vocab * vocab, int n_vocab,
                  NetworkState & st, SamplerCtx & sc,
                  const ChatParams & params,
                  std::vector<int32_t> & kv_tokens,
                  const std::string & user_text,
                  bool first_turn,
                  double & prefill_ms, double & gen_ms,
                  int & prompt_tokens, int & gen_count,
                  std::string & response_out,
                  std::string & error)
{
    response_out.clear();
    prefill_ms = 0.0; gen_ms = 0.0; prompt_tokens = 0; gen_count = 0;

    const std::string delta_str = build_gemma_user_delta(user_text, first_turn);
    std::vector<int32_t> delta;
    if (!tokenize_to(vocab, delta_str.data(), (int) delta_str.size(),
                     /*add_special=*/first_turn,
                     /*parse_special=*/true,
                     delta)) {
        error = "run_one_turn: tokenize delta failed";
        return false;
    }
    if (delta.empty()) { error = "run_one_turn: empty delta"; return false; }
    prompt_tokens = (int) delta.size();

    if (st.n_past + (int) delta.size() + 1 > st.cap_seq) {
        std::ostringstream oss;
        oss << "run_one_turn: prefill would exceed chat_ctx (n_past=" << st.n_past
            << " + n_prompt=" << delta.size() << " + 1 > cap=" << st.cap_seq
            << "). Raise --gemma4-chat-ctx.";
        error = oss.str();
        return false;
    }

    std::vector<float> logits((size_t) n_vocab, 0.0f);
    const auto t_pre0 = std::chrono::steady_clock::now();
    if (!network_step(st, mf, (int) delta.size(), delta.data(),
                      /*last_token_only=*/true, logits.data(), error)) {
        return false;
    }
    prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pre0).count();
    kv_tokens.insert(kv_tokens.end(), delta.begin(), delta.end());

    llama_token tok = sample_token(sc, logits.data(), n_vocab);
    const auto t_gen0 = std::chrono::steady_clock::now();
    while (true) {
        if (llama_vocab_is_eog(vocab, tok)) break;
        response_out += token_piece(vocab, tok);
        if (params.stream) {
            std::printf("%s", token_piece(vocab, tok).c_str());
            std::fflush(stdout);
        }
        ++gen_count;

        if (params.n_predict >= 0 && gen_count >= params.n_predict) {
            // Commit captured token into KV so kv_tokens stays in sync.
            if (st.n_past + 1 <= st.cap_seq) {
                int32_t feed = tok;
                if (!network_step(st, mf, 1, &feed,
                                  /*last_token_only=*/true, logits.data(), error)) {
                    return false;
                }
                kv_tokens.push_back(tok);
            }
            break;
        }

        if (st.n_past + 1 + 1 > st.cap_seq) {
            std::fprintf(stderr, "\n[gemma4 chat: context full at n_past=%d cap=%d; stopping]\n",
                         st.n_past, st.cap_seq);
            if (st.n_past + 1 <= st.cap_seq) {
                int32_t feed = tok;
                if (!network_step(st, mf, 1, &feed,
                                  /*last_token_only=*/true, logits.data(), error)) {
                    return false;
                }
                kv_tokens.push_back(tok);
            }
            break;
        }

        int32_t feed = tok;
        if (!network_step(st, mf, 1, &feed,
                          /*last_token_only=*/true, logits.data(), error)) {
            return false;
        }
        kv_tokens.push_back(tok);
        tok = sample_token(sc, logits.data(), n_vocab);
    }
    gen_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_gen0).count();
    if (params.stream) { std::printf("\n"); std::fflush(stdout); }
    return true;
}

}  // anonymous namespace

// ======================================================================
// Public API
// ======================================================================

bool run_chat_loop(const llama_model * model, const Weights & w,
                   const std::string & single_prompt,
                   const ChatParams & params,
                   std::string & error) {
    if (params.chat_ctx <= 0) { error = "run_chat_loop: chat_ctx<=0"; return false; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Probe template just for the informational message; the chat loop
    // itself uses a hand-built Gemma format and never calls
    // llama_chat_apply_template, so an unrecognized model template is
    // not fatal here.
    (void) resolve_chat_template(model);

    const int n_threads = params.n_threads > 0 ? params.n_threads : 1;
    ModelF32 mf;
    if (!dequant_model(model, w, mf, error, n_threads)) return false;

    NetworkState st;
    if (!network_state_reserve(st, mf, params.chat_ctx, error)) return false;

    SamplerCtx sc;
    if (!sampler_init(sc, n_vocab, params, error)) return false;

    std::vector<int32_t> kv_tokens;

    auto cleanup = [&]() { sampler_free(sc); };

    int turn_index = 0;
    auto run_one = [&](const std::string & user_text) -> bool {
        std::string terr, response;
        double prefill_ms = 0, gen_ms = 0;
        int    n_prompt = 0, n_gen = 0;
        const bool first = (turn_index == 0);
        const bool ok = run_one_turn(mf, vocab, n_vocab, st, sc, params,
                                     kv_tokens, user_text, first,
                                     prefill_ms, gen_ms, n_prompt, n_gen,
                                     response, terr);
        if (!ok) { error = terr; return false; }
        const double gen_tps = (gen_ms > 0.0 && n_gen > 0) ? (1000.0 * n_gen) / gen_ms : 0.0;
        const double pre_tps = (prefill_ms > 0.0 && n_prompt > 0) ? (1000.0 * n_prompt) / prefill_ms : 0.0;
        std::fprintf(stderr,
            "[gemma4 chat turn %d: prompt_tok=%d gen_tok=%d  "
            "prefill %.1f ms (%.1f t/s)  gen %.1f ms (%.1f t/s)  "
            "n_past=%d / cap=%d]\n",
            turn_index, n_prompt, n_gen,
            prefill_ms, pre_tps, gen_ms, gen_tps,
            st.n_past, st.cap_seq);
        ++turn_index;
        return true;
    };

    if (!single_prompt.empty()) {
        const bool ok = run_one(single_prompt);
        cleanup();
        return ok;
    }

    // Interactive mode.
    std::fprintf(stderr,
        "[gemma4 chat: interactive mode (empty line or EOF to quit). "
        "temp=%.2f min_p=%.2f seed=%u n_predict=%d chat_ctx=%d threads=%d]\n",
        params.temp, params.min_p, params.seed,
        params.n_predict, params.chat_ctx, n_threads);
    while (true) {
        std::fprintf(stderr, "\n> ");
        std::fflush(stderr);
        std::string line;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) break;
        if (!run_one(line)) {
            std::fprintf(stderr, "[gemma4 chat: turn failed: %s]\n", error.c_str());
            cleanup();
            return false;
        }
    }
    cleanup();
    error.clear();
    return true;
}

// ----------------------------------------------------------------------
// Scripted multi-turn determinism test.
//
// Stage A: incremental chat (greedy).
//   Turn 1: user "Q1", greedy decode N1 tokens. Capture turn-1 tokens.
//   Turn 2: user "Q2", greedy decode N2 tokens. Capture turn-2 tokens.
//
// Stage B: fresh full prefill of the equivalent conversation.
//   Build the SAME token stream from scratch by tokenizing
//     turn-1 delta + gen_1 tokens + turn-2 delta
//   into a fresh NetworkState, then greedy decode N2 tokens.
//
// PASS iff turn-2 token sequences match exactly.
//
// Note: the delta strings are built with build_gemma_user_delta so
// turn-2 includes a leading "<end_of_turn>\n" closer. This matches
// what the incremental path does at runtime. Verifying determinism
// here gives us confidence the hand-built deltas plus direct gen-token
// commit produce identical KV evolution as a fresh full re-prefill.
// ----------------------------------------------------------------------
bool chat_self_test(const llama_model * model, const Weights & w,
                    int n_threads, std::string & error) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_threads <= 0) n_threads = 1;

    ModelF32 mf;
    if (!dequant_model(model, w, mf, error, n_threads)) return false;

    const std::string q1 = "What is the capital of France?";
    const std::string q2 = "And the capital of Germany?";
    const int N1 = 12;
    const int N2 = 12;
    const int cap = 512;

    auto greedy_argmax = [&](const float * lg) -> int {
        int best = 0; float bv = lg[0];
        for (int v = 1; v < n_vocab; ++v) if (lg[v] > bv) { bv = lg[v]; best = v; }
        return best;
    };

    // Helper: greedy decode for `max_gen` tokens, advancing st/kv.
    // Captures decoded tokens into out_tokens. Returns false on
    // network_step failure.
    auto greedy_decode = [&](NetworkState & st,
                             std::vector<int32_t> & kv_tokens,
                             const std::vector<int32_t> & delta,
                             int max_gen,
                             std::vector<int32_t> & out_tokens) -> bool {
        if (st.n_past + (int) delta.size() + 1 > st.cap_seq) {
            error = "greedy_decode: prefill exceeds cap"; return false;
        }
        std::vector<float> logits((size_t) n_vocab, 0.0f);
        if (!network_step(st, mf, (int) delta.size(), delta.data(), true, logits.data(), error)) {
            return false;
        }
        kv_tokens.insert(kv_tokens.end(), delta.begin(), delta.end());
        int tok = greedy_argmax(logits.data());
        for (int g = 0; g < max_gen; ++g) {
            if (llama_vocab_is_eog(vocab, tok)) break;
            out_tokens.push_back(tok);
            if (st.n_past + 1 > st.cap_seq) break;
            int32_t feed = tok;
            if (!network_step(st, mf, 1, &feed, true, logits.data(), error)) return false;
            kv_tokens.push_back(tok);
            if (g + 1 >= max_gen) break;
            tok = greedy_argmax(logits.data());
        }
        return true;
    };

    // ============= Stage A: incremental =============
    NetworkState stA;
    if (!network_state_reserve(stA, mf, cap, error)) return false;
    std::vector<int32_t> kvA;
    std::vector<int32_t> T1_inc, T2_inc;

    // Turn 1.
    {
        const std::string s = build_gemma_user_delta(q1, /*first_turn=*/true);
        std::vector<int32_t> delta;
        if (!tokenize_to(vocab, s.data(), (int) s.size(), true, true, delta)) {
            error = "stageA turn1 tokenize failed"; return false;
        }
        if (!greedy_decode(stA, kvA, delta, N1, T1_inc)) return false;
    }
    // Turn 2.
    {
        const std::string s = build_gemma_user_delta(q2, /*first_turn=*/false);
        std::vector<int32_t> delta;
        if (!tokenize_to(vocab, s.data(), (int) s.size(), false, true, delta)) {
            error = "stageA turn2 tokenize failed"; return false;
        }
        if (!greedy_decode(stA, kvA, delta, N2, T2_inc)) return false;
    }

    auto piece_join = [&](const std::vector<int32_t> & v) {
        std::string s;
        for (int t : v) s += token_piece(vocab, t);
        return s;
    };
    std::fprintf(stderr,
        "gemma4 chat_self_test stageA(incremental): T1_inc=%zu toks  T2_inc=%zu toks\n"
        "  turn1: \"%s\"\n  turn2: \"%s\"\n",
        T1_inc.size(), T2_inc.size(),
        piece_join(T1_inc).c_str(), piece_join(T2_inc).c_str());

    // ============= Stage B: fresh full prefill =============
    // Reconstruct the equivalent KV by tokenizing turn-1 delta,
    // injecting turn-1 gen tokens, then tokenizing turn-2 delta.
    // No re-tokenization of assistant text -- gen tokens are inserted
    // directly, matching what the incremental path does.
    NetworkState stB;
    if (!network_state_reserve(stB, mf, cap, error)) return false;
    std::vector<int32_t> kvB;

    {
        const std::string s = build_gemma_user_delta(q1, true);
        std::vector<int32_t> delta;
        if (!tokenize_to(vocab, s.data(), (int) s.size(), true, true, delta)) {
            error = "stageB turn1 tokenize failed"; return false;
        }
        std::vector<float> logits((size_t) n_vocab, 0.0f);
        if (!network_step(stB, mf, (int) delta.size(), delta.data(), true, logits.data(), error)) return false;
        kvB.insert(kvB.end(), delta.begin(), delta.end());
        // Inject turn-1 gen tokens.
        if (!T1_inc.empty()) {
            if (!network_step(stB, mf, (int) T1_inc.size(), T1_inc.data(), true, logits.data(), error)) return false;
            kvB.insert(kvB.end(), T1_inc.begin(), T1_inc.end());
        }
    }
    // Verify kvA == kvB so far (sanity check on the harness itself).
    if (kvA.size() < kvB.size() ||
        !std::equal(kvB.begin(), kvB.end(), kvA.begin())) {
        std::ostringstream oss;
        oss << "stageB: post-turn1 KV mismatch vs stageA "
            << "(stageA=" << kvA.size() << " stageB=" << kvB.size() << ")";
        error = oss.str(); return false;
    }

    std::vector<int32_t> T2_fresh;
    {
        const std::string s = build_gemma_user_delta(q2, false);
        std::vector<int32_t> delta;
        if (!tokenize_to(vocab, s.data(), (int) s.size(), false, true, delta)) {
            error = "stageB turn2 tokenize failed"; return false;
        }
        if (!greedy_decode(stB, kvB, delta, N2, T2_fresh)) return false;
    }

    std::fprintf(stderr,
        "gemma4 chat_self_test stageB(fresh full prefill): T2_fresh=%zu toks\n"
        "  turn2: \"%s\"\n",
        T2_fresh.size(), piece_join(T2_fresh).c_str());

    const size_t n_cmp = std::min(T2_inc.size(), T2_fresh.size());
    int matched = 0;
    int first_mismatch = -1;
    for (size_t i = 0; i < n_cmp; ++i) {
        if (T2_inc[i] == T2_fresh[i]) ++matched;
        else if (first_mismatch < 0) first_mismatch = (int) i;
    }
    if (T2_inc.size() != T2_fresh.size() || first_mismatch >= 0) {
        std::ostringstream oss;
        oss << "chat_self_test: token-sequence mismatch (inc=" << T2_inc.size()
            << " fresh=" << T2_fresh.size()
            << " matched=" << matched
            << " first_mismatch_at=" << first_mismatch << ")";
        error = oss.str();
        return false;
    }
    std::fprintf(stderr,
        "gemma4 chat_self_test: PASS  (T2_inc == T2_fresh, %d/%d tokens)\n",
        matched, (int) T2_inc.size());
    return true;
}

}  // namespace gemma4
