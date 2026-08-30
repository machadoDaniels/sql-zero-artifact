#!/usr/bin/env python3
"""McNemar exact paired significance test on official per-example EX.

The gold control (goldbird3b) and the self-play checkpoints were scored on the
SAME BIRD dev examples, so correctness is paired. This re-executes each model's
predictions with the mini_dev (BIRD official) execution comparison to get
per-example 0/1, pairs by (db_id, question), and runs McNemar's exact binomial
test on the discordant pairs. Reuses the official comparison (calculate_ex =
set-equality of execution results) — no reimplementation of the metric.

CPU only. Run on a node with the BIRD dev databases and mini_dev checked out.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIRD_EVAL = os.path.join(ROOT, "official_eval/mini_dev/evaluation")

DB_ROOT = os.path.join(ROOT, "bird/dev_databases")
TIMEOUT = 30.0

# headline BIRD-dev comparisons: matched gold control vs each self-play iteration
MODELS = {
    "goldbird3b": "results/goldbird3b_s32_bird.json",
    "g1b": "results/hf_g1b-s78_bird.json",
    "bird2": "results/hf_bird2-s78_bird.json",
    "bird3": "results/hf_bird3-s78_bird.json",
}
CONTROL = "goldbird3b"
CACHE = "results/mcnemar_perex_cache.json"


def _score_one(args):
    """Return 0/1 official EX for one (pred, gold, db_id) via mini_dev exec."""
    from func_timeout import func_timeout, FunctionTimedOut  # noqa: E402
    sys.path.insert(0, BIRD_EVAL)
    from evaluation_utils import execute_sql  # noqa: E402
    from evaluation_ex import calculate_ex  # noqa: E402
    pred_sql, gold_sql, db_id = args
    db_path = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    try:
        return func_timeout(TIMEOUT, execute_sql,
                            args=(pred_sql, gold_sql, db_path, "SQLite", calculate_ex))
    except (FunctionTimedOut, Exception):
        return 0


def score_model(path):
    """Per-example {(db_id, question): 0/1} for a prediction JSON."""
    items = json.load(open(os.path.join(ROOT, path)))["results"]
    keys = [(it["db_id"], it["question"]) for it in items]
    payload = [(it["pred_sql"], it["gold_sql"], it["db_id"]) for it in items]
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(_score_one, payload))
    return {k: r for k, r in zip(keys, res)}


def mcnemar(a, b):
    """a, b: dicts key->0/1 over the same keys. Returns stats for `b beats a`."""
    keys = sorted(set(a) & set(b))
    n01 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)  # control right, self-play wrong
    n10 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)  # self-play right, control wrong
    disc = n01 + n10
    # exact two-sided McNemar = binomial test on discordant pairs vs p=0.5
    p = binomtest(n10, disc, 0.5, alternative="two-sided").pvalue if disc else 1.0
    return {
        "n": len(keys),
        "acc_control": round(sum(a[k] for k in keys) / len(keys), 4),
        "acc_selfplay": round(sum(b[k] for k in keys) / len(keys), 4),
        "n01_control_only": n01,
        "n10_selfplay_only": n10,
        "discordant": disc,
        "disc_ratio": round(disc / len(keys), 4),
        "p_exact_two_sided": p,
    }


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, preserving input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = round(running, 5)
    return adj


def main():
    if os.path.exists(os.path.join(ROOT, CACHE)) and "--fresh" not in sys.argv:
        cache = json.load(open(os.path.join(ROOT, CACHE)))
        scored = {m: {tuple(k.split("␟")): v for k, v in d.items()} for m, d in cache.items()}
    else:
        scored = {}
        for m, path in MODELS.items():
            print(f"scoring {m} ...", file=sys.stderr)
            scored[m] = score_model(path)
        dump = {m: {"␟".join(k): v for k, v in d.items()} for m, d in scored.items()}
        json.dump(dump, open(os.path.join(ROOT, CACHE), "w"))

    rows, pvals = [], []
    for m in ("g1b", "bird2", "bird3"):
        r = mcnemar(scored[CONTROL], scored[m])
        rows.append((m, r))
        pvals.append(r["p_exact_two_sided"])
    adj = holm(pvals)

    print(f"\nMcNemar exact — matched BIRD-gold control ({CONTROL}) vs self-play (BIRD dev)\n")
    hdr = ["comparison", "n", "acc_gold", "acc_sp", "n01(gold+)", "n10(sp+)", "disc", "p_exact", "p_holm"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join("---" for _ in hdr) + "|")
    for (m, r), pa in zip(rows, adj):
        print(f"| gold vs {m} | {r['n']} | {r['acc_control']} | {r['acc_selfplay']} | "
              f"{r['n01_control_only']} | {r['n10_selfplay_only']} | {r['discordant']} | "
              f"{r['p_exact_two_sided']:.4g} | {pa:.4g} |")


if __name__ == "__main__":
    main()
