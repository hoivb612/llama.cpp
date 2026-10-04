#!/usr/bin/env python3
"""Run a LAYA_* scenario file against a llama-server /v1/systemone endpoint.

This is the model-agnostic sibling of laya-cli. laya-cli implements the laya
prompt template directly, which makes it fast and dependency-free but ties it
to one decision-model family. The lev and kev families differ in ways that are
not small -- lev renders a choice in two option orders and reads a noul off a
9-point rating scale, kev flattens state and instructions through its own text
renderer -- and tools/server/server-decision.cpp already implements all of
them. The server normalises every family down to the same answer shape
(a choice key plus probabilities, a noul as a single 0-1 probability), so this
driver needs no per-model branches.

Usage:
  python decision_server_cli.py -f <scenario.txt> [--url http://127.0.0.1:8099]
                                [--mode choice|noul|both] [--verbose]
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


class Scenario:
    def __init__(self):
        self.state = ""
        self.choice_instructions = ""
        self.options = []        # list of (key, description), order preserved
        self.nouls = []          # list of (id, statement), order preserved
        self.policy = []         # list of (id, op, threshold, key)
        self.prompts = []        # list of (gold_or_None, utterance)


def parse_scenario(path):
    sc = Scenario()
    section = None
    state_lines = []

    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()

            if stripped.startswith("#") or (not stripped and section is None):
                continue

            if stripped in ("LAYA_STATE", "LAYA_CHOICE", "LAYA_NOUL",
                            "LAYA_POLICY", "LAYA_PROMPTS"):
                section = stripped
                continue
            if stripped == "END_SECTION":
                section = None
                continue
            if section is None:
                continue

            if section == "LAYA_STATE":
                state_lines.append(line)
            elif section == "LAYA_CHOICE":
                if not stripped:
                    continue
                key, _, val = stripped.partition(":")
                key, val = key.strip(), val.strip()
                if key == "instructions":
                    sc.choice_instructions = val
                else:
                    sc.options.append((key, val))
            elif section == "LAYA_NOUL":
                if not stripped:
                    continue
                key, _, val = stripped.partition(":")
                sc.nouls.append((key.strip(), val.strip()))
            elif section == "LAYA_POLICY":
                if not stripped:
                    continue
                lhs, _, key = stripped.partition("->")
                key = key.strip()
                lhs = lhs.strip()
                if lhs == "*":
                    sc.policy.append(("*", None, None, key))
                else:
                    parts = lhs.split()
                    if len(parts) != 3 or parts[1] not in (">", "<"):
                        raise ValueError("bad policy rule: %r" % stripped)
                    sc.policy.append((parts[0], parts[1], float(parts[2]), key))
            elif section == "LAYA_PROMPTS":
                if not stripped:
                    continue
                if "|" in stripped:
                    gold, _, utt = stripped.partition("|")
                    sc.prompts.append((gold.strip(), utt.strip()))
                else:
                    sc.prompts.append((None, stripped))

    sc.state = "\n".join(state_lines).strip()
    if "{message}" not in sc.state:
        raise ValueError("LAYA_STATE has no {message} slot")
    return sc


def apply_policy(sc, nouls, choice_pick):
    """Ordered, first-match-wins. A rule may target '@choice'."""
    for ident, op, threshold, key in sc.policy:
        if ident == "*":
            hit = True
        elif ident not in nouls:
            continue
        else:
            hit = nouls[ident] > threshold if op == ">" else nouls[ident] < threshold
        if hit:
            return choice_pick if key == "@choice" else key
    return None


def post(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--mode", default="both", choices=["choice", "noul", "both"])
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sc = parse_scenario(args.file)

    uses_choice_fallback = any(key == "@choice" for _, _, _, key in sc.policy)
    want_noul = args.mode in ("noul", "both") and sc.nouls
    want_choice = args.mode in ("choice", "both") or (want_noul and uses_choice_fallback)

    if want_choice and not sc.options:
        sys.exit("error: scenario has no LAYA_CHOICE options")

    endpoint = args.url.rstrip("/") + "/v1/systemone"

    n_labeled = 0
    n_ok_choice = 0
    n_ok_policy = 0
    hist_choice = {}
    hist_policy = {}
    sum_conf = 0.0
    n_conf = 0
    model_name = None

    for idx, (gold, utt) in enumerate(sc.prompts, start=1):
        state = sc.state.replace("{message}", utt)

        questions = {}
        if want_choice:
            questions["__choice"] = {
                "type": "choice",
                "instructions": sc.choice_instructions,
                "criteria": {k: v for k, v in sc.options},
            }
        if want_noul:
            for nid, statement in sc.nouls:
                questions[nid] = {"type": "noul", "instructions": statement}

        try:
            resp = post(endpoint, {"state": state, "questions": questions}, args.timeout)
        except urllib.error.URLError as exc:
            sys.exit("error: request failed on prompt %d: %s" % (idx, exc))

        model_name = model_name or resp.get("model")
        answers = resp.get("answers", {})

        if gold is not None:
            n_labeled += 1

        print('[%d] %s"%s"' % (idx, ("gold=%s  " % gold) if gold else "", utt))

        picked_choice = None
        if want_choice:
            ca = answers.get("__choice", {})
            picked_choice = ca.get("choice")
            probs = ca.get("probabilities", {})
            conf = ca.get("confidence", 0.0)
            sum_conf += conf
            n_conf += 1
            hist_choice[picked_choice] = hist_choice.get(picked_choice, 0) + 1
            verdict = ""
            if gold is not None:
                ok = picked_choice == gold
                n_ok_choice += ok
                verdict = " OK " if ok else " MISS"
            spread = " ".join("%s=%.3f" % (k, probs.get(k, 0.0)) for k, _ in sc.options)
            print("     choice : %s%s  conf=%.3f  [%s]"
                  % (picked_choice, verdict, conf, spread))

        if want_noul:
            vals = {}
            for nid, _ in sc.nouls:
                vals[nid] = answers.get(nid, {}).get("noul", 0.0)
            print("     noul   : " + "  ".join("%s=%.3f" % (k, v) for k, v in vals.items()))

            picked = apply_policy(sc, vals, picked_choice)
            hist_policy[picked] = hist_policy.get(picked, 0) + 1
            verdict = ""
            if gold is not None:
                ok = picked == gold
                n_ok_policy += ok
                verdict = " OK " if ok else " MISS"
            print("     policy : %s%s" % (picked if picked else "(none)", verdict))

        if args.verbose:
            print("     tokens : %s" % resp.get("usage", {}).get("input_tokens"))

    def pct(n):
        return 100.0 * n / n_labeled if n_labeled else 0.0

    print("\n================ summary ================")
    print("  model           : %s" % model_name)
    if n_labeled:
        if want_choice:
            print("  direct choice   : %2d / %2d  (%.1f%%)" % (n_ok_choice, n_labeled, pct(n_ok_choice)))
        if want_noul:
            print("  noul + policy   : %2d / %2d  (%.1f%%)" % (n_ok_policy, n_labeled, pct(n_ok_policy)))
        if n_labeled != len(sc.prompts):
            print("  (%d of %d prompts carry a gold label)" % (n_labeled, len(sc.prompts)))
    else:
        print("  (no gold labels -- reporting distribution only)")
    if n_conf:
        print("  mean confidence : %.3f" % (sum_conf / n_conf))
    if hist_choice:
        print("  choice spread   : " + " ".join(
            "%s=%d" % (k, hist_choice.get(k, 0)) for k, _ in sc.options))
    if hist_policy:
        print("  policy spread   : " + " ".join(
            "%s=%d" % (k, v) for k, v in sorted(hist_policy.items(), key=lambda kv: str(kv[0]))))
    print("=========================================")


if __name__ == "__main__":
    main()
