#!/usr/bin/env python3
"""
Probe separability analyser for laya-cli --mode noul runs.

Given a scenario whose utterances carry gold labels, this answers one question
per probe: is there a threshold that separates one gold class from all others?

That is the only thing that matters when designing a guard. A probe can look
responsive -- wide score range, sensible-looking ordering -- and still be
useless, because what a guard needs is a gap between the class you want and
the highest-scoring row you do not want. This reports that gap directly as
`margin` (min of target class minus max of the rest). A positive margin means
a usable guard exists; a negative margin means no threshold can work, however
the probe is reworded.

Usage:
    python analyze_probes.py --run <laya_run.txt> [--classes a,b,c,d,z]
"""

import argparse
import collections
import io
import re
import sys

GOLD = ([None]
        + ["a"] * 10
        + ["d"]
        + [None] * 4
        + ["c", "c", "c", "c", "b", "c", "c", "c", "b", "c", "c", "b", "c", "c", "c"]
        + ["z"] * 10)
N = 40


def parse(path):
    """Return {row_index: {probe_name: score}}."""
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    rows, idx = {}, None
    for line in txt.splitlines():
        m = re.match(r"\s*\[\s*(\d+)\s*\]", line)
        if m:
            idx = int(m.group(1))
            continue
        if idx is not None and "noul" in line and ":" in line:
            pairs = re.findall(r"(\w+)=([0-9.]+)", line)
            if pairs:
                rows[idx] = {k: float(v) for k, v in pairs}
                idx = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--classes", default="a,b,c,d,z")
    args = ap.parse_args()

    rows = parse(args.run)
    missing = [i for i in range(1, N + 1) if i not in rows]
    if missing:
        sys.exit("parse incomplete: %d/%d rows, missing %s" % (len(rows), N, missing))

    probes = list(rows[1].keys())
    classes = [c for c in args.classes.split(",") if c]

    labeled = [i for i in range(1, N + 1) if GOLD[i] is not None]
    by_class = collections.defaultdict(list)
    for i in labeled:
        by_class[GOLD[i]].append(i)

    print("\n  per-class mean score (%d labeled utterances)\n" % len(labeled))
    print("  %-10s %s" % ("probe", "  ".join("%-14s" % ("%s (n=%d)" % (c, len(by_class[c])))
                                             for c in classes)))
    print("  " + "-" * (10 + 16 * len(classes)))
    for p in probes:
        cells = []
        for c in classes:
            vals = [rows[i][p] for i in by_class[c]]
            cells.append("%-14s" % ("%.3f" % (sum(vals) / len(vals))))
        print("  %-10s %s" % (p, "  ".join(cells)))

    print("\n  separability: best single-threshold guard per (probe, target class)")
    print("  margin = min(target) - max(everything else). Positive = usable.\n")
    print("  %-10s %-7s %8s %8s %9s   %s" %
          ("probe", "target", "min(tgt)", "max(rest)", "margin", "verdict"))
    print("  " + "-" * 72)

    best = None
    for p in probes:
        for c in classes:
            tgt = [rows[i][p] for i in by_class[c]]
            rest = [rows[i][p] for i in labeled if GOLD[i] != c]
            if not tgt or not rest:
                continue
            margin = min(tgt) - max(rest)
            if best is None or margin > best[0]:
                best = (margin, p, c)
            if margin > 0:
                print("  %-10s %-7s %8.3f %8.3f %+9.3f   SEPARABLE" %
                      (p, c, min(tgt), max(rest), margin))

    if best and best[0] <= 0:
        print("  (none)")
        print("\n  no probe separates any class at any threshold.")
        print("  closest was %s on class '%s', margin %+.3f" % (best[1], best[2], best[0]))

    # partial-recall view: a guard can still be useful with zero false positives
    # even when it cannot capture the whole class.
    print("\n  partial-recall view: rows catchable at the zero-false-positive floor\n")
    print("  %-10s %-7s %10s %8s   %s" % ("probe", "target", "fp_floor", "caught", "of class"))
    print("  " + "-" * 60)
    any_row = False
    for p in probes:
        for c in classes:
            rest = [rows[i][p] for i in labeled if GOLD[i] != c]
            floor = max(rest)
            caught = sum(1 for i in by_class[c] if rows[i][p] > floor)
            if caught:
                any_row = True
                print("  %-10s %-7s %10.3f %8d   %d" % (p, c, floor, caught, len(by_class[c])))
    if not any_row:
        print("  (nothing: every class's top scorer is outscored by some other class)")
    print()

    # Are the probes measuring different things at all? If a battery of probes
    # asking about different digit content produces near-identical rankings,
    # then probe score is dominated by some utterance-level property and is
    # invariant to what the probe actually asks -- which means no rewording
    # will ever help.
    def pearson(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return num / (dx * dy) if dx and dy else float("nan")

    series = {p: [rows[i][p] for i in range(1, N + 1)] for p in probes}
    print("  cross-probe correlation (are these 8 questions measuring 8 things?)\n")
    print("  %-10s %s" % ("", "  ".join("%-7s" % p[:7] for p in probes)))
    offdiag = []
    for a in probes:
        cells = []
        for b in probes:
            r = pearson(series[a], series[b])
            cells.append("%-7.2f" % r)
            if a != b:
                offdiag.append(r)
        print("  %-10s %s" % (a, "  ".join(cells)))
    if offdiag:
        mean_r = sum(offdiag) / len(offdiag)
        print("\n  mean off-diagonal r = %.3f  (min %.3f, max %.3f)"
              % (mean_r, min(offdiag), max(offdiag)))
        if mean_r > 0.75:
            print("  => the probes are largely collinear: they share one dominant")
            print("     utterance-level factor rather than measuring their own content.")
            print("     Corroborate with the per-class table above: if probes peak on a")
            print("     class they do not target, rewording cannot help.")
    print()


if __name__ == "__main__":
    main()
