# Decision-model harness (`laya/`)

Tooling for the **systemone** decision-model family — Laya, Lev, Julia-1, Kev-4B,
Nimble. These are *not* generative SLMs. They are encoder-based **scorers**: you
give them a state, a question and a list of options, and they return a
probability distribution over the options. There is no sampling loop and no
text output.

This directory holds the harness, the analysis tools, and the findings from
adapting our existing choice/prompt system to that model class.

---

## Contents

| file | what it is |
|---|---|
| `laya_cli.cpp` | Native harness. Loads a decision GGUF directly and scores a custom-prompt file. Oracle-validated **for Laya only**. Built as the `laya-cli` target. |
| `decision_server_cli.py` | Model-agnostic driver against `llama-server`'s `/v1/systemone` endpoint. The only way to test the non-Laya families (lev / kev / nimble / julia). |
| `convert_reasoning_to_laya.py` | Converts a `prompts/reasoning/DN.txt` scenario into the decision-model prompt format. |
| `analyze_probes.py` | Probe separability analyser. **The main design tool** — see below. |
| `d8_ensemble.py` | End-to-end scorer for the D8 ensemble (generative extraction + code lookup + scorer guard override). |
| `__readme__.md` | Raw working transcript of the original investigation. Unedited; kept for provenance, not intended as documentation. |

Prompt files live in `../prompts/`:
- `custom_prompts_laya.txt` — the Mara reference scenario (18 utterances).
- `../prompts/reasoning/` — source scenarios, generative format.
- `../prompts/reasoning_laya/` — the same scenarios converted to scorer format.

---

## The central finding: scorers and SLMs fail at opposite things

The intuitive assumption is that a decision model handles the mechanical
classes and a generative model handles the ones needing understanding. The D8
scenario measures the exact opposite.

| subtask | nature | decision model | SLM (Gemma-4-E4B) |
|---|---|---|---|
| `z` deflections | **semantic** | 6/10 at 0 FP | 0/10 |
| `a/b/c/d` digit classes | **symbolic** | no usable signal | 26/26 |
| Mara, 18 utterances | **semantic** | 18/18 | — |

D8 is constructed to force the distinction. Rows 16–30 (`c`) and rows 31–40
(`z`) **reuse the same phone numbers**: row 26 *"Is Josie's number 960564?"* and
row 31 *"Why don't we ring Nicky B? His number is 960564"* are identical on
digits and opposite on label. Nothing but meaning separates them — and that is
the class the scorer gets and the SLM misses completely.

The converse was tested directly, and the digit failure is **representational,
not a wording problem**. See `../prompts/reasoning_laya/laya_D8_spelled.txt`:
eight probes asking about digit content in spelled-out form ("ends in
eighty-four") produced a best margin of **−0.107**; every probe peaked on class
`a` regardless of what it asked about; cross-probe correlation averaged
**r = 0.806**. The probes were not reading digits — they were all reading one
shared utterance-level factor. The encoder does not represent digit identity in
any direction the scoring head can read.

**Rule to carry forward:**

> scorers → holistic utterance-level gist (intent, stance, deflection, tone).
> SLMs → extraction, copying, sequential symbol manipulation.
> **symbols → generative, gist → scorer, decision → code.**

Routed the other way round, the ensemble scores *worse* than either model alone.

---

## Secondary findings

**Detection and calibration are separate capabilities.** E4B was asked directly
whether an utterance proposes a different person (`proposes_other_person`). It
caught 8/10 deflections — good detection — at **12/30 false positives**, a net
loss. The scorer gets 6/10 at **zero** false positives, because its thresholds
sit under a *measured* FP floor. The SLM detects well and thresholds badly; the
scorer's contribution is the calibration, not the semantics.

**Field order in a generative prompt is computation order.** The original
`reasoning/D8.txt` asks for `{answer, justification}` — answer first. In an
autoregressive model the verdict is emitted before any reasoning token exists,
so the justification is causally downstream of the decision and cannot inform
it. The result is fluent post-hoc confabulation: one row answered
`d. Josie (585584)` for Sophie's `849456` and justified it with *"correctly
identified the phone number for 'Josie'"*. Measured ladder:

| variant | score | digit classes |
|---|---|---|
| `D8.txt` as written (answer first) | 27.8% | 10/26 |
| justification first, nothing else | 33.3% | 12/26 — **16/40 unparseable** |
| named intermediate fields (`D8_extract.txt`) | 61.1% | 22/26 |
| same run, re-scored from the fields in code | **72.2%** | **26/26** |

Note that simply reordering is *not* enough — freedom to ramble destabilised the
output format. Named intermediate fields with a fixed schema are what worked.

**Prompt decomposition beats prompt wording.** The guard battery did not make
the scorer smarter; it changed the question being asked. Likewise the jump from
61.1% → 72.2% costs nothing — it is the *same run*, re-scored by ignoring the
model's own `answer` and mapping its extracted `player_number` through the
phonebook in code.

---

## `analyze_probes.py` — measure the floor before tuning

The tool that matters most going forward. For every (probe, class) pair it
reports

```
margin = min(score over target rows) − max(score over all other rows)
```

A **positive** margin means some threshold cleanly separates that class — a
usable guard exists. A **negative** margin means **no threshold exists at any
wording**, which is the signal to move that subtask out of the scorer entirely
rather than keep rewriting prompts. It also prints a cross-probe correlation
matrix; a battery with high mean off-diagonal `r` is measuring one factor
several times over, not accumulating independent evidence.

Run it against a `--mode both` output file:

```powershell
python examples\llm-infer\laya\analyze_probes.py --run build\laya_d8_spelled.txt
```

---

## Reproducing the D8 ensemble (88.9%)

```powershell
build\bin\RelWithDebInfo\minslm-cli.exe `
    D:\llama.cpp\models\gemma-4\gemma-4-E4B-it-Q4_K_M.gguf 8 `
    examples\llm-infer\prompts\reasoning\D8_extract.txt > build\d8_extract_run.txt

build-laya\bin\laya-cli.exe -m <Laya-Q8_0.gguf> `
    -f examples\llm-infer\prompts\reasoning_laya\laya_D8_guarded.txt `
    --mode both > build\laya_d8_guards.txt

python examples\llm-infer\laya\d8_ensemble.py `
    --extract build\d8_extract_run.txt --guards build\laya_d8_guards.txt
```

Expected:

```
E4B alone (its own `answer`)       22/36  ( 61.1%)   a=10/10 b=3/3 c=8/12  d=1/1 z=0/10
Laya alone (guards + @choice)      24/36  ( 66.7%)   a=8/10  b=3/3 c=7/12  d=0/1 z=6/10
E4B digits -> code lookup          26/36  ( 72.2%)   a=10/10 b=3/3 c=12/12 d=1/1 z=0/10
ENSEMBLE (code + guard override)   32/36  ( 88.9%)   a=10/10 b=3/3 c=12/12 d=1/1 z=6/10
guard false positives : 0
```

For comparison, the best **single** model on D8 was Kev-4B at 58.3%.

The four residual misses are all `z` rows (33, 38, 39, 40) whose digits extract
perfectly — *"might know something"*, *"let's try her"*, *"might have
information"*, *"let's see if she knows"*. The digit problem is fully solved;
only guard recall remains.

---

## Gotchas

- **`laya-cli` space-pads row indices** — `[ 1]`, `[11]`. A parser regex of
  `\[(\d+)\]` silently drops rows 1–9 and still prints a plausible score. Both
  Python tools now hard-assert full 40/40 coverage from each stage. *Partial-parse
  bugs that still produce believable output are the main hazard in this harness.*
- The custom-prompt parser (`minslm/minslm_cli.cpp`) only reads inside
  `CUSTOM_TEMPLATE_PROMPT` / `CUSTOM_PROMPT` / `END_SECTION` blocks and silently
  drops everything else, so `#` header comments in prompt files are safe.
- `laya-cli` prints **all** rows, preserving file numbering; unlabeled rows print
  without `gold=`.
- The Laya-tuned noul thresholds **do not transfer** across the family. Nimble is
  the one exception — its noul scores are crisply bimodal (≈0.99 / ≈0.01) so
  Laya's thresholds happen to land in the gap. Every other model needs its
  thresholds re-fitted.
- Option sets in these scenarios are frequently **not mutually exclusive** (in
  D8, `a` "585584" and `b` "Home" are both subsets of `c` "a number from the
  list"). Softmax then forces mutually-true options to compete with no stated
  priority. If a scenario scores badly, check this before blaming the model.

---

## Verified results

Only numbers re-verified in-session are listed. Others are in `__readme__.md`.

| model | Mara (18) | D8 (36) |
|---|---|---|
| Laya | 18/18 — direct | 24/36 (66.7%) with guard battery |
| Nimble-9B | 14/18 direct → 16/18 with noul+policy | 11/36 direct → 20/36 (55.6%) |
| Kev-4B | 16/18 (88.9%) untuned | 21/36 (58.3%) — best single model |
| Gemma-4-E4B (generative) | — | 22/36 own answer → 26/36 code-mapped |
| **ensemble** | — | **32/36 (88.9%)** |

Nimble's D8 collapse is scenario-specific rather than positional decay — its
Mara predictions were well distributed while its D8 predictions collapsed onto
a single option.

---

## Open questions

- Can the guard battery reach 10/10 on `z` with 2–3 more OR'd probes against the
  four missed wordings? That would put D8 near 100%. Measure each candidate's FP
  floor with `analyze_probes.py` *before* adding it to the battery.
- Does the ensemble pattern transfer to the other 12 reasoning scenarios? Only
  D8 currently has gold labels.
- Would the extraction schema work on a smaller generative model (E2B, Phi-3)?
  The entire ladder above is E4B-only.
- A fitted decision-tree policy over a larger probe bank (rather than a
  hand-thresholded OR battery) was designed but never run. The spelled-numeral
  result makes it unattractive for digit classes, but it remains the right tool
  for `z` recall and for per-model threshold fitting.
