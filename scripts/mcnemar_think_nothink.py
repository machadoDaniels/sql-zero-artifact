#!/usr/bin/env python3
"""McNemar exact paired test — volume-matched think x no-think, iters 1-3 (3B, BIRD env).

Each iteration's two arms (think `bird{1,2,3}b20k`, no-think `bird{1,2,3}nt20k`) trained
on the SAME 20k self-play generation and were scored at step 78 on the SAME dev examples
under the unified eval protocol (`eval_spider.py`, greedy@1, full schema DDL+3 rows), so
correctness is paired. Per-example 0/1 comes straight from the `match` field already stored
in the result JSONs — the same field the reported EX aggregates are computed from, so no
re-execution and no second metric implementation. Pairs by (db_id, question); reuses
`mcnemar`/`holm` from scripts/mcnemar_paired.py. Holm is applied across the two benchmarks
WITHIN each iteration (same family as job 28558, iter1).

CPU only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcnemar_paired import mcnemar, holm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# control = no-think arm, treatment = think arm; n10 = think-only correct.
# expected EX (no-think, think) as reported in docs/RESULTADOS.csv — asserted, not trusted.
ITERATIONS = {
    "iter1": {
        "BIRD dev": ("results/bird1nt20k_s78_bird.json", "results/g1b20k_s78_bird.json",
                     0.3911, 0.4048),
        "Spider dev": ("results/bird1nt20k_s78_spider.json", "results/g1b20k_s78_spider.json",
                       0.7321, 0.7292),
    },
    "iter2": {
        "BIRD dev": ("results/bird2nt20k_s78_bird.json", "results/bird2b20k_s78_bird.json",
                     0.3911, 0.4087),
        "Spider dev": ("results/bird2nt20k_s78_spider.json", "results/bird2b20k_s78_spider.json",
                       0.7311, 0.7215),
    },
    "iter3": {
        "BIRD dev": ("results/bird3nt20k_s78_bird.json", "results/bird3b20k_s78_bird.json",
                     0.4061, 0.4100),
        "Spider dev": ("results/bird3nt20k_s78_spider.json", "results/bird3b20k_s78_spider.json",
                       0.7331, 0.7157),
    },
}


def load(path):
    """Return (per-example {(db_id, question): 0/1}, ordered key list, aggregate EX)."""
    items = json.load(open(os.path.join(ROOT, path)))["results"]
    keys = [(it["db_id"], it["question"]) for it in items]
    per = {k: int(bool(it["match"])) for k, it in zip(keys, items)}
    assert len(per) == len(keys), f"{path}: duplicate (db_id, question) keys"
    return per, keys, sum(per.values()) / len(keys)


def run_iteration(it, benchmarks):
    rows, pvals = [], []
    for name, (nt_path, th_path, nt_exp, th_exp) in benchmarks.items():
        nt, nt_keys, nt_ex = load(nt_path)
        th, th_keys, th_ex = load(th_path)
        aligned = nt_keys == th_keys
        print(f"[{it} {name}] positional alignment: {aligned}; "
              f"key overlap {len(set(nt) & set(th))}/{len(nt_keys)} vs {len(th_keys)}; "
              f"EX no-think {nt_ex:.5f} (expected {nt_exp}) "
              f"think {th_ex:.5f} (expected {th_exp}); "
              f"files {nt_path} | {th_path}", file=sys.stderr)
        # stop hard if the per-example aggregates do not reproduce the reported EX:
        # that means the wrong JSON was picked.
        assert abs(nt_ex - nt_exp) < 5e-4, f"{it} {name}: no-think EX mismatch {nt_ex} vs {nt_exp}"
        assert abs(th_ex - th_exp) < 5e-4, f"{it} {name}: think EX mismatch {th_ex} vs {th_exp}"
        r = mcnemar(nt, th)  # joins on the shared (db_id, question) keys
        both_right = sum(1 for k in set(nt) & set(th) if nt[k] == 1 and th[k] == 1)
        both_wrong = sum(1 for k in set(nt) & set(th) if nt[k] == 0 and th[k] == 0)
        r["both_correct"] = both_right
        r["both_wrong"] = both_wrong
        r["aligned_positionally"] = aligned
        r["overlap"] = len(set(nt) & set(th))
        rows.append((name, r))
        pvals.append(r["p_exact_two_sided"])
    return rows, holm(pvals)


def main():
    print("\nMcNemar exact — volume-matched think vs no-think, BIRD-env 3B, step 78, 20k pairs")
    print("(Holm across the two benchmarks within each iteration)\n")
    hdr = ["iter", "benchmark", "n", "EX no-think", "EX think", "both correct",
           "think only", "no-think only", "both wrong", "discordant", "disc. ratio",
           "p exact (2-sided)", "p Holm"]
    table = ["| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    dumps = []
    for it, benchmarks in ITERATIONS.items():
        rows, adj = run_iteration(it, benchmarks)
        for (name, r), pa in zip(rows, adj):
            table.append(
                f"| {it} | {name} | {r['n']} | {r['acc_control']} | {r['acc_selfplay']} | "
                f"{r['both_correct']} | {r['n10_selfplay_only']} | {r['n01_control_only']} | "
                f"{r['both_wrong']} | {r['discordant']} | {r['disc_ratio']} | "
                f"{r['p_exact_two_sided']:.4g} | {pa:.4g} |")
            dumps.append(f"{it} {name}: {json.dumps(r)}")
    print("\n".join(table))
    print()
    print("\n".join(dumps))


if __name__ == "__main__":
    main()
