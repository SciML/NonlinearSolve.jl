"""Summarize bounded solver accuracy and actual return-code fallback orders."""
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

NATIVE = ("bounded_trust_region", "reflective", "bounded_lm", "dogbox", "gn_linesearch", "gn_trustregion", "gn_hybrid")

def load_rows(paths):
    rows = {}
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                rows[row["case"], row["algorithm"]] = row
    return rows

def evaluate_order(case, order, rows):
    elapsed, attempted = 0.0, []
    for name in order:
        row = rows[case, name]
        elapsed += float(row["time_ns"])
        attempted.append(row)
        if row["retcode"] == "Exception":
            return False, elapsed
        if row["retcode"] in ("Success", "Terminated", "ExactSolutionLeft", "ExactSolutionRight", "FloatingPointLimit"):
            return row["passed"] == "true", elapsed
    key = "residual" if attempted[0]["kind"] == "root" else "cost"
    best = min(attempted, key=lambda row: float(row[key]))
    return best["passed"] == "true", elapsed

def summarize(rows):
    cases = sorted({case for case, _ in rows if all((case, name) in rows for name in NATIVE)})
    families = Counter(rows[case, NATIVE[0]]["family"] for case in cases)
    weight = {case: 1 / (len(families) * families[rows[case, NATIVE[0]]["family"]]) for case in cases}
    fastest = {case: min((float(rows[case, name]["time_ns"]) for name in NATIVE if rows[case, name]["passed"] == "true"), default=math.inf) for case in cases}
    def score(order):
        outcomes = {case: evaluate_order(case, order, rows) for case in cases}
        successes = [case for case in cases if outcomes[case][0]]
        coverage = sum(weight[case] for case in successes)
        logs = sum(weight[case] * math.log(outcomes[case][1] / fastest[case]) for case in successes if math.isfinite(fastest[case]))
        speed = math.exp(logs / coverage) if coverage else math.inf
        profile = {str(tau): sum(weight[case] for case in successes if outcomes[case][1] <= tau * fastest[case]) for tau in (1, 2, 5, 10, 100)}
        return dict(order=order, passed=len(successes), cases=len(cases), family_weighted_coverage=coverage, geometric_time_ratio=speed, profile=profile)
    individual = [score((name,)) for name in NATIVE]
    algorithms = sorted({name for _, name in rows})
    for name in algorithms:
        if name not in NATIVE and all((case, name) in rows for case in cases):
            individual.append(score((name,)))
    candidates = []
    for length in (2, 3, 4):
        ranked = [score(order) for order in itertools.permutations(NATIVE, length)]
        ranked.sort(key=lambda row: (-round(row["family_weighted_coverage"], 12), row["geometric_time_ratio"]))
        candidates.extend(ranked[:10])
    return dict(cases=len(cases), families=dict(families), individual=individual, candidates=candidates)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: analyze_bounded.py OUTPUT.json INPUT.csv [REPLACEMENT.csv ...]")
    result = summarize(load_rows(sys.argv[2:]))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n")
    print("Complete cases:", result["cases"], "families:", result["families"])
    for row in result["individual"] + result["candidates"][::10]:
        print(" -> ".join(row["order"]), f'{row["passed"]}/{row["cases"]}', f'weighted={row["family_weighted_coverage"]:.3f}', f'ratio={row["geometric_time_ratio"]:.2f}')
