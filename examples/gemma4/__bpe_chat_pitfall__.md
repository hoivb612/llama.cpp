BPE / SentencePiece round-trip drift in chat KV caches
======================================================

This document records a real bug we hit in G4.4 (Gemma-4 hand-path chat
loop), with the concrete token IDs that surfaced it. The lesson is
general — it applies to any incremental decode + KV-cache pipeline
that builds the next prompt by formatting the conversation history
through `apply_template` (or any text-templating step) and then
re-tokenising the full string.

Short version: text is NOT a stable serialisation of a token stream.
Tokenise(Detokenise(ids)) is in general NOT equal to ids.

The setup
---------

Gemma-4 chat is multi-turn. After turn 1, the NetworkState KV cache
holds the tokens the model has actually seen:

  [..., 9079, 236761, <EOG>]
        |       |       |
        |       |       +-- gen-stop, not written to KV
        |       +---------- "."
        +------------------ " Paris"

For turn 2, the natural-looking implementation is:

  1. Append the assistant's decoded TEXT (" Paris.") to a `messages`
     vector alongside the original user question.
  2. Append the new user question.
  3. Call `llama_chat_apply_template(messages, add_assistant=true)` to
     get the full formatted conversation as a string.
  4. Tokenise that string.
  5. Sanity check: the first `kv_tokens.size()` ids must equal
     `kv_tokens` (they describe the same prefix of the conversation).
  6. Prefill the tail (delta) into NetworkState.

Step 5 fails. Loudly. We see "re-tokenization drift at 39" on Gemma-4
E4B. The KV cache and the re-tokenisation disagree about what token
sits at position 39 of the conversation.

The smoking gun
---------------

Run the live oracle (the `--gemma4-tokenize-probe` flag was added to
make this reproducible at any time):

  Gemma4.exe -m ...E4B...gguf --gemma4-tokenize-probe 2 \
    "The capital of France is Paris."             \
    "The capital of France is Paris.<end_of_turn>"

Output (token ids in `[]`; piece text in quotes):

  "The capital of France is Paris."
    [  2]      2 -> "<bos>"
    [  1]    818 -> "The"
    [  2]   5279 -> " capital"
    [  3]    529 -> " of"
    [  4]   7001 -> " France"
    [  5]    563 -> " is"
    [  6]   9079 -> " Paris"
    [  7] 236761 -> "."

  "The capital of France is Paris.<end_of_turn>"
    [  2]      2 -> "<bos>"
    [  1]    818 -> "The"
    [  2]   5279 -> " capital"
    [  3]    529 -> " of"
    [  4]   7001 -> " France"
    [  5]    563 -> " is"
    [  6]   9079 -> " Paris"
    [  7]  21603 -> ".<"        <-- !!!  was [236761 "."] before
    [  8]    643 -> "end"
    [  9] 236779 -> "_"
    [ 10]   1340 -> "of"
    [ 11] 236779 -> "_"
    [ 12]    887 -> "turn"
    [ 13] 236813 -> ">"

Two ways to describe the same byte sequence " Paris." produce different
token sequences depending on what comes AFTER them, because BPE /
SentencePiece tokenisation is GREEDY ACROSS BYTE BOUNDARIES. The merge
table has an entry for ".<" (id 21603) that fires when "." is followed
by "<". So:

  decoded incrementally : [..., 9079 (" Paris"),  236761 (".") , 236820 ("<"), 643 ("end"), ...]
                                                  ^^^ pos 39    ^^^ pos 40

  re-tokenised wrapped  : [..., 9079 (" Paris"),  21603  (".<"),                643 ("end"), ...]
                                                  ^^^ pos 39  (different id!)

Same number of characters, ONE FEWER token. At position 39 the KV cache
has 236761 ("."), the re-tokenised expected stream has 21603 (".<").
The prefix check correctly fires.

Why this is a bug, not just an aesthetic issue
----------------------------------------------

You might think "so what, it's only one token". But:

* If you prefill the re-tokenised delta starting from
  `position = kv_tokens.size()`, you have NOT actually closed turn 1 with
  the bytes "<end_of_turn>": you've appended a delta starting from
  whatever comes AFTER ".<", missing those bytes entirely. The model's
  view of the conversation is corrupt — it will respond as if turn 1
  never ended.
* If instead you naively replace KV from position 39 onwards, you're
  STILL inconsistent: position 38 in KV is 9079 (" Paris") with its
  attention having been computed against the prefix [..., 9079, ...],
  but the re-tokenised stream wants position 38 to be 9079 followed by
  21603 (".<") — which produces different attention weights at every
  subsequent layer. Any partial overwrite leaves the KV in a state that
  no actual forward pass would ever produce, and the next gen is
  garbage.
* Worse: this drift is CONTENT-DEPENDENT. With our second test prompt
  ("Berlin.x"), `.` followed by `x` does NOT have a merge:

    "Berlin."   ->  [ 89946 "Berlin",  236761 "."]
    "Berlin.x"  ->  [ 89946 "Berlin",  236761 ".",  236781 "x"]

  Same `.` (236761) in both cases. So the drift may not show up on the
  developer's first prompt and only blow up months later when a user's
  conversation ends with one of the unlucky merge characters.

The general law
---------------

For ANY BPE / SentencePiece tokeniser:

  tokenise(detokenise(ids)) is NOT guaranteed to equal ids.

A round trip is only safe when every token in `ids` is also the
greedy-best tokenisation of its piece text INSIDE THE LARGER CONTEXT it
sits in. For Gemma-4's vocab the merge table contains rules like
".<" → 21603 that will fire as soon as the assistant's terminal "."
is followed by the template's "<end_of_turn>" prefix. The same pattern
exists in every modern BPE vocab; the specific merge pairs vary by
model.

This means: ANY chat implementation that

  1. decodes assistant gen tokens incrementally into a KV cache,
  2. captures the assistant's response as TEXT, and
  3. rebuilds the next turn's prompt by re-templating + re-tokenising
     that text in the larger conversation context

is vulnerable to silent KV corruption at every turn boundary.

Three fixes (ordered worst -> best)
-----------------------------------

WORST: ignore the drift, hope for the best. The model's "next" turn
will sometimes work, sometimes hallucinate, often start mid-sentence.

WORKABLE but expensive: throw the KV cache away every turn, re-prefill
the entire (re-templated, re-tokenised) conversation from scratch.
Correct, but defeats the purpose of having a KV cache; cost grows
quadratically with conversation length.

BEST (what we shipped in G4.4): never re-tokenise the assistant's gen
tokens at all. Instead:

  1. Track `kv_tokens` as an explicit list of token IDs alongside the
     NetworkState KV cache. Each gen token decoded by the model is
     pushed directly onto this list as the model emits it (no text
     round trip).
  2. For each new user turn, hand-build ONLY the user-side delta as a
     short string using the model-family's chat-template literals
     (Gemma: "<start_of_turn>user\n{Q}<end_of_turn>\n<start_of_turn>model\n";
     on turn N>=2 prepend "<end_of_turn>\n" to close the prior assistant
     response that may have stopped at the n_predict cap without
     emitting a natural EOG).
  3. Tokenise the delta string ATOMICALLY with `parse_special=true`.
     This guarantees boundary deterministically because the leading
     boundary character (e.g. the "<" of "<end_of_turn>") sits at the
     very start of the tokenised chunk and has no preceding context
     to merge into.
  4. Prefill the delta into NetworkState. Append delta tokens to
     `kv_tokens`.

The hand-built delta + token-stream tracking design completely sidesteps
the BPE round-trip problem: the assistant's gen tokens never get
re-tokenised, and the template wrapping tokens always tokenise into the
same ids because they always appear in the same lexical neighbourhood
(start of a fresh delta string).

The implementation lives in `examples/gemma4/gemma4_chat.cpp`:

* `build_gemma_user_delta(user_text, first_turn)` -- the delta builder.
* `run_one_turn(...)` -- prefill + sample + append loop. Notice that
  there is no `apply_template` call anywhere in the per-turn path;
  the only model-state mutation comes from `network_step` calls
  driven by either the delta tokens or single gen tokens.
* `chat_self_test(...)` -- determinism harness. Stage A runs two turns
  incrementally; Stage B replays the same conversation by tokenising
  turn-1's delta + injecting turn-1's gen tokens (NOT re-tokenised
  text) + turn-2's delta, then greedy-decodes turn 2. PASS iff turn-2
  gen token sequences match exactly.

Reproduce the failure mode (now and forever)
--------------------------------------------

The `--gemma4-tokenize-probe N S1 ... SN` flag is kept in the tree
specifically as a forensic tool: it does NOT depend on any of our
chat-loop machinery, just on the GGUF's vocab + `llama_tokenize` with
`parse_special=true`. Use it on any new model to discover which "."-like
characters merge into which template-prefix characters before you trust
an `apply_template`-based chat loop:

  Gemma4.exe -m <PATH>.gguf --gemma4-tokenize-probe 4 \
    "answer."                                          \
    "answer.<end_of_turn>"                             \
    "answer\n"                                         \
    "answer\n<end_of_turn>"

If any of the suffix-extended forms produce a DIFFERENT token id for the
suffix character than the bare form does, you have the same kind of
landmine waiting in your chat path. The mitigation is the same: hand-
build the per-turn delta and keep gen tokens out of any re-tokenisation
pipeline.

Other model families
--------------------

The exact merge ids change per vocab, but the failure shape is identical
on Phi-3 / Llama-3 / Qwen / etc. -- the only thing that varies is which
characters happen to have BPE merges with the template boundary
characters. Phi-3's chat path (examples/phi3/phi3_runtime.cpp) avoids
the issue differently: it owns its own KV via llama_context and uses
llama_decode for both prefill and per-token gen, so the per-token gen
tokens are written into KV directly by the upstream kernel and never
need to be re-templated.

Any new model-family chat loop in this tree should either follow Phi-3's
"let the upstream kernel own KV" pattern OR follow Gemma-4's "hand-
built delta + token-stream tracking" pattern. Do NOT mix the two by
incrementally decoding into a custom KV and then re-templating gen
text back into the next prompt -- that path is the bug.
