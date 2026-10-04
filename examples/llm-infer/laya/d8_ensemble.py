#!/usr/bin/env python3
"""
D8 ensemble scorer: generative extraction + code lookup + scorer-based guards.

The point of this file is the routing, not the code. D8 asks one question whose
answer depends on three unrelated capabilities, and no single model is good at
all three:

    subtask                  mechanism                     measured
    -----------------------  ----------------------------  -----------------
    read the digits out      Gemma-4-E4B generative        36/40 (every
                             (prompts/reasoning/           labeled utterance)
                              D8_extract.txt)
    look the digits up       this file, PHONEBOOK dict     exact
    spot a deflection ('z')  Laya guard battery            6/10 at 0 FP
                             (prompts/reasoning_laya/
                              laya_D8_guarded.txt)

Each stage is fed the thing it is actually good at. E4B extracts digits
perfectly and then maps them wrong -- it will write same_as_josie:"no" and
answer "Josie" in the same object -- so we throw its `answer` away. Laya cannot
see a digit difference at all, but its guards fire on deflections with a
measured false-positive floor, so they can only ever add recall.

Usage:
    python d8_ensemble.py --extract <e4b_run.txt> --guards <laya_run.txt>

Both inputs are raw stdout captures:
    minslm-cli.exe MODEL 8 prompts/reasoning/D8_extract.txt            > e4b_run.txt
    laya-cli.exe -m Laya-Q8_0.gguf -f .../laya_D8_guarded.txt --mode both > laya_run.txt
"""

import argparse
import collections
import io
import re
import sys

# --- the phonebook, exactly as it appears in the scenario state --------------
PHONEBOOK = {
    "01738857761": "Home",
    "07930406588": "Dad",
    "849456":      "Sophie T",
    "585584":      "Josie A",
    "960564":      "Nicky B",
    "014194960903": "Tanya R",
    "960194":      "Barry T",
}

# Gold labels in laya-label space, 1-indexed. Rows 12-15 are deliberately
# unlabeled: they name two numbers at once ("585584, or maybe 585594") and have
# no defensible single answer.
GOLD = ([None]
        + ["a"] * 10                       # 1-10   Josie's number, correct
        + ["d"]                            # 11     585594, not on the list
        + [None] * 4                       # 12-15  ambiguous, excluded
        + ["c", "c", "c", "c", "b", "c", "c", "c", "b", "c", "c", "b", "c", "c", "c"]
        + ["z"] * 10)                      # 31-40  deflections

N_PROMPTS = 40


def parse_extraction(path):
    """Pull the JSON fields out of a minslm-cli run of D8_extract.txt."""
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"> Running with custom prompt => \[(\d+)/%d\]: \[" % N_PROMPTS, txt)
    out = {}
    for i in range(1, len(blocks), 2):
        idx, body = int(blocks[i]), blocks[i + 1]

        def field(key):
            m = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, body)
            return m.group(1).strip() if m else ""

        letter = re.search(r'"answer"\s*:\s*"\s*([a-zA-Z])', body)
        out[idx] = {
            "utterance":     body.split("]\n")[0],
            "player_number": field("player_number"),
            "number_owner":  field("number_owner"),
            # D8.txt letters are shuffled relative to the laya file
            "model_answer":  {"a": "b", "b": "c", "c": "d",
                              "d": "a", "e": "e", "z": "z"}.get(
                                  letter.group(1).lower()) if letter else None,
        }
    return out


def parse_guards(path):
    """Pull the per-utterance policy verdict out of a laya-cli run."""
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    out, idx = {}, None
    for line in txt.splitlines():
        # laya-cli space-pads the index to width 2: "[ 1]", "[11]"
        m = re.match(r"\s*\[\s*(\d+)\s*\]", line)
        if m:
            idx = int(m.group(1))
            continue
        m = re.search(r"policy\s*:\s*([a-z])\b", line)
        if m and idx is not None:
            out[idx] = m.group(1)
            idx = None
    return out


def classify(digits):
    """Map an extracted number to an option. This is the whole 'hard' part."""
    d = re.sub(r"\D", "", digits or "")
    if not d:
        return "e"
    who = PHONEBOOK.get(d)
    if who == "Josie A":
        return "a"
    if who == "Home":
        return "b"
    return "c" if who else "d"


def score(name, pred):
    ok = tot = 0
    per, totals = collections.Counter(), collections.Counter()
    for i in range(1, N_PROMPTS + 1):
        gold = GOLD[i]
        if gold is None:
            continue
        tot += 1
        totals[gold] += 1
        if pred.get(i) == gold:
            ok += 1
            per[gold] += 1
    detail = "  ".join("%s=%d/%d" % (c, per[c], totals[c]) for c in sorted(totals))
    print("  %-34s %2d/%2d  (%5.1f%%)   %s" % (name, ok, tot, 100.0 * ok / tot, detail))
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True, help="minslm-cli run of D8_extract.txt")
    ap.add_argument("--guards",  required=True, help="laya-cli run of laya_D8_guarded.txt")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ext = parse_extraction(args.extract)
    grd = parse_guards(args.guards)
    if not ext:
        sys.exit("no extraction blocks parsed from %s" % args.extract)
    if not grd:
        sys.exit("no policy verdicts parsed from %s" % args.guards)
    # Silent partial-parse bugs are the main hazard here: a regex that misses
    # rows degrades a stage without failing, and the ensemble still prints a
    # plausible number. Demand full coverage from both inputs.
    for label, got in (("extraction", ext), ("guards", grd)):
        missing = [i for i in range(1, N_PROMPTS + 1) if i not in got]
        if missing:
            sys.exit("%s parse incomplete: %d/%d rows, missing %s"
                     % (label, len(got), N_PROMPTS, missing))

    model_only = {i: ext.get(i, {}).get("model_answer") for i in range(1, N_PROMPTS + 1)}
    guards_only = dict(grd)
    code_only = {i: classify(ext.get(i, {}).get("player_number")) for i in range(1, N_PROMPTS + 1)}
    # the ensemble: guards may only ever override with 'z'
    ensemble = {i: ("z" if grd.get(i) == "z" else code_only[i]) for i in range(1, N_PROMPTS + 1)}

    print("\n================= D8 ensemble =================")
    print("  36 of 40 utterances carry a gold label\n")
    score("E4B alone (its own `answer`)", model_only)
    score("Laya alone (guards + @choice)", guards_only)
    score("E4B digits -> code lookup",     code_only)
    ok, tot = score("ENSEMBLE (code + guard override)", ensemble)

    fired = [i for i in range(1, N_PROMPTS + 1) if grd.get(i) == "z"]
    bad = [i for i in fired if GOLD[i] not in (None, "z")]
    print("\n  guard fired on rows   : %s" % (", ".join(map(str, fired)) or "none"))
    print("  guard false positives : %d  (a non-zero value here would mean the" % len(bad))
    print("                             guards are breaking correct digit rows)")
    print("  final                 : %d / %d  (%.1f%%)" % (ok, tot, 100.0 * ok / tot))
    print("===============================================\n")

    if args.verbose:
        for i in range(1, N_PROMPTS + 1):
            e = ext.get(i, {})
            flag = "--" if GOLD[i] is None else ("OK  " if ensemble[i] == GOLD[i] else "MISS")
            print("%2d gold=%-4s ens=%-4s %s num=%-14s %s"
                  % (i, GOLD[i] or "-", ensemble[i], flag,
                     e.get("player_number", "")[:14], e.get("utterance", "")[:46]))


if __name__ == "__main__":
    main()
