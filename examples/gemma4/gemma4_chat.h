// G4.4 -- Hand-path chat loop for Gemma-4.
//
// Drives the custom F32 forward (NetworkState + network_step from
// gemma4_forward) through a multi-turn chat. Supports greedy
// (temp <= 0) and sampled (temp > 0, min-p) decoding via the same
// llama_sampler chain that examples/phi3 uses.
//
// Key invariants (see internal notes in gemma4_chat.cpp):
//   * KV-history sync: every token in formatted[0..prev_len) must be in
//     NetworkState. The chat-template delta tokenization naturally
//     restores any model-closing markers (e.g. <end_of_turn>) on the
//     next turn's prefill, so we don't have to feed the EOG ourselves.
//   * Message-content lifetime: llama_chat_message::content is a raw
//     pointer; we strdup/free it like examples/phi3 does.
//   * Atomic turn commit: the user message is staged in a temporary
//     vector and only committed on a successful prefill+gen turn.
//   * Capacity check is split into prefill (hard) and per-decode-step
//     (graceful early stop), so n_predict = -1 (until EOG / context
//     full) works correctly.
//
// CLI surface (Gemma4.cpp):
//   --gemma4-chat                   single-turn from -p, or interactive
//                                   stdin if -p is empty
//   --gemma4-chat-test              scripted multi-turn determinism
//                                   test (see chat_self_test below)
//   --gemma4-chat-ctx N             NetworkState capacity (default 4096)
//   --temp F                        0 = greedy (default), >0 sampled
//   --min-p F                       min-p cutoff (default 0.05;
//                                   ignored when temp <= 0)
//   --seed N                        sampler seed (default LLAMA_DEFAULT_SEED)
//   -n N                            max gen tokens per turn (default 256;
//                                   -1 = until EOG or context full)
//
// Out of scope for G4.4:
//   * KV rewind / message editing.
//   * Loading a cached KV before chat (orthogonal; future combine of
//     G4.3 + G4.4).
//   * top-k / top-p / repetition penalty (matches Phi-3 surface).

#pragma once

#include "llama.h"
#include "gemma4_forward.h"
#include "gemma4_weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gemma4 {

struct ChatParams {
    float    temp      = 0.0f;                // 0.0 = greedy
    float    min_p     = 0.05f;               // ignored when temp <= 0
    uint32_t seed      = LLAMA_DEFAULT_SEED;
    int      n_predict = 256;                 // max gen tokens per turn (-1 = until EOG)
    int      chat_ctx  = 4096;                // NetworkState capacity (full chat history)
    int      n_threads = 1;
    bool     stream    = true;                // print pieces as they generate
};

// Run a chat session on the hand path.
//
// If `single_prompt` is non-empty, runs one turn with that user message
// and returns. Otherwise, reads user messages from stdin until EOF or
// an empty line. Always prints the generated text and a short profile
// line per turn to stderr.
//
// Returns true on clean exit (single turn completed OK, or interactive
// reached EOF). Returns false with `error` set on any per-turn failure
// that aborts the whole session (e.g. chat-template unavailable,
// dequant failed, prefill of the first turn exceeds capacity).
bool run_chat_loop(const llama_model * model, const Weights & w,
                   const std::string & single_prompt,
                   const ChatParams & params,
                   std::string & error);

// One item of a custom-prompt-file (cpf) script.
struct ScriptItem {
    enum Kind { USER, CONTEXT, REWIND, QUIT } kind = USER;
    std::string text;   // USER: the user message
    int         n = 0;  // REWIND: number of turns to drop
};

// Run a scripted multi-turn chat on the hand path (same custom-prompt-file
// format as minslm-cli / cpf_gem4mm: SYSTEM block + PROMPT/T: blocks with
// /context, /rewind N and quit() meta commands). `system_prompt` (may be
// empty) is folded into the first user turn, since Gemma has no system role.
// Each user turn's answer streams to stdout; a per-turn profile line goes to
// stderr. Lets us run real prompts end-to-end and judge whether the hand
// path's numerical drift actually degrades answers.
//
// Returns true when the script completes (or hits quit()); false with
// `error` set on any fatal per-turn failure.
bool run_chat_script(const llama_model * model, const Weights & w,
                     const std::vector<ScriptItem> & items,
                     const std::string & system_prompt,
                     const ChatParams & params,
                     std::string & error);

// Run a template-substitution prompt file (CUSTOM_TEMPLATE_PROMPT +
// CUSTOM_PROMPT list, same format as minslm-cli / minslm-multi). `tmpl`
// contains the full chat markup and a {message} placeholder; each entry in
// `prompts` is substituted for {message} and run INDEPENDENTLY (fresh KV),
// tokenized raw (add_special=false, parse_special=true). Each generation
// streams to stdout with a per-prompt profile line on stderr.
bool run_template_prompts(const llama_model * model, const Weights & w,
                          const std::string & tmpl,
                          const std::vector<std::string> & prompts,
                          const ChatParams & params,
                          std::string & error);

// Scripted multi-turn determinism test for the chat loop.
//
// Runs a fixed 2-turn greedy dialogue twice:
//   (a) incremental via run_chat_loop machinery (delta prefill +
//       per-turn decode), capturing turn-2's generated token sequence;
//   (b) fresh prefill of the fully formatted 2-turn conversation
//       (full prompt up to "<start_of_turn>model\n" of turn 2),
//       then greedy decode the same number of tokens.
// PASS iff the two token sequences are identical.
//
// This validates the rubber-duck blocker: that KV state advanced via
// delta tokenization across turn boundaries matches a clean fresh
// prefill of the equivalent fully-formatted conversation.
bool chat_self_test(const llama_model * model, const Weights & w,
                    int n_threads, std::string & error);

}  // namespace gemma4
