#!/usr/bin/env python3
"""Descriptive coverage: self-play generated pairs vs human BIRD gold.

The paper invokes "broader coverage" as the mechanism by which zero-data self-play
matches/beats the gold control. This measures it (or bounds it) over the SAME 64 BIRD
training databases: template diversity, structural-complexity histograms, and template
overlap between the two corpora. Descriptive only — no overclaiming.

CPU only. Reuses the codebase's own SQL masking (search.sql_utils.sql_template) so the
"template" notion matches the challenger's dedup.
"""
import json
import os
import re
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from search.sql_utils import sql_template, complexity_level  # noqa: E402

KEEP = os.path.join(ROOT, "bird/train_dbs_keep.txt")
CORPORA = {
    "self-play iter1 (20k)": "data/t2sql_solver_bird1_20k.parquet",
    "self-play iter2 (20k)": "data/t2sql_solver_bird2_20k.parquet",
    "self-play iter3 (20k)": "data/t2sql_solver_bird3_20k.parquet",
    "human gold (8390)": "data/t2sql_solver_gold_bird.parquet",
}

AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", re.I)
SETOP = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.I)


def load(path, keep):
    df = pd.read_parquet(os.path.join(ROOT, path))
    rows = []
    for _, r in df.iterrows():
        gt = json.loads(r["reward_model"]["ground_truth"])
        if gt["db_id"] in keep:
            rows.append((gt["db_id"], gt["gold_sql"]))
    return rows


def features(sql):
    up = sql.upper()
    # subquery = a SELECT nested inside parentheses
    n_select = len(re.findall(r"\bSELECT\b", up))
    return {
        "joins": len(re.findall(r"\bJOIN\b", up)),
        "group_by": int("GROUP BY" in up),
        "order_by": int("ORDER BY" in up),
        "having": int("HAVING" in up),
        "subquery": int(n_select > 1),
        "set_op": int(bool(SETOP.search(sql))),
        "aggs": len(AGG.findall(sql)),
        "tables": len(re.findall(r"\bFROM\b", up)) + len(re.findall(r"\bJOIN\b", up)),
        "tokens": len(sql.split()),
        "level": complexity_level(sql),
    }


def summarize(rows):
    templates = [sql_template(s) for _, s in rows]
    tc = Counter(templates)
    n = len(rows)
    feats = [features(s) for _, s in rows]
    top10 = sum(c for _, c in tc.most_common(10))
    return {
        "n_pairs": n,
        "unique_templates": len(tc),
        "templates_per_100": round(100 * len(tc) / n, 2),
        "top10_template_share": round(top10 / n, 3),
        "mean_joins": round(sum(f["joins"] for f in feats) / n, 3),
        "mean_aggs": round(sum(f["aggs"] for f in feats) / n, 3),
        "mean_tables": round(sum(f["tables"] for f in feats) / n, 3),
        "mean_tokens": round(sum(f["tokens"] for f in feats) / n, 1),
        "pct_group_by": round(100 * sum(f["group_by"] for f in feats) / n, 1),
        "pct_order_by": round(100 * sum(f["order_by"] for f in feats) / n, 1),
        "pct_having": round(100 * sum(f["having"] for f in feats) / n, 1),
        "pct_subquery": round(100 * sum(f["subquery"] for f in feats) / n, 1),
        "pct_set_op": round(100 * sum(f["set_op"] for f in feats) / n, 1),
        "pct_level3": round(100 * sum(f["level"] == 3 for f in feats) / n, 1),
        "_templates_by_db": _templates_by_db(rows),
        "_feats": feats,
    }


def _templates_by_db(rows):
    d = {}
    for db, s in rows:
        d.setdefault(db, set()).add(sql_template(s))
    return d


def overlap(sp_by_db, gold_by_db):
    """Per-DB template coverage, averaged over shared DBs."""
    dbs = sorted(set(sp_by_db) & set(gold_by_db))
    gold_cov, sp_cov = [], []
    for db in dbs:
        sp, go = sp_by_db[db], gold_by_db[db]
        if go:
            gold_cov.append(len(sp & go) / len(go))   # gold templates hit by self-play
        if sp:
            sp_cov.append(len(sp & go) / len(sp))      # self-play templates that gold also has
    return {
        "shared_dbs": len(dbs),
        "gold_templates_covered_by_selfplay": round(sum(gold_cov) / len(gold_cov), 3),
        "selfplay_templates_in_gold": round(sum(sp_cov) / len(sp_cov), 3),
    }


def main():
    keep = {l.strip() for l in open(KEEP) if l.strip()}
    summ = {name: summarize(load(path, keep)) for name, path in CORPORA.items()}

    cols = ["n_pairs", "unique_templates", "templates_per_100", "top10_template_share",
            "mean_joins", "mean_aggs", "mean_tables", "mean_tokens",
            "pct_group_by", "pct_order_by", "pct_having", "pct_subquery", "pct_set_op", "pct_level3"]
    print("\n## Coverage: self-play vs human gold (64 shared BIRD train DBs)\n")
    print("| metric | " + " | ".join(summ) + " |")
    print("|" + "|".join("---" for _ in range(len(summ) + 1)) + "|")
    for c in cols:
        print(f"| {c} | " + " | ".join(str(summ[m][c]) for m in summ) + " |")

    print("\n## Template overlap (self-play iter1 vs gold, per-DB)\n")
    ov = overlap(summ["self-play iter1 (20k)"]["_templates_by_db"],
                 summ["human gold (8390)"]["_templates_by_db"])
    for k, v in ov.items():
        print(f"- {k}: {v}")

    # figure
    try:
        make_figure(summ)
        print("\nfigure -> ara/evidence/figures/fig_coverage.pdf", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}", file=sys.stderr)


def make_figure(summ):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42  # AAAI forbids Type 3 fonts
    import matplotlib.pyplot as plt
    # Okabe-Ito
    OI = {"selfplay": "#0072B2", "gold": "#D55E00"}
    struct = ["pct_group_by", "pct_order_by", "pct_having", "pct_subquery", "pct_set_op", "pct_level3"]
    labels = ["GROUP BY", "ORDER BY", "HAVING", "subquery", "set op", "level 3"]
    sp = summ["self-play iter1 (20k)"]
    go = summ["human gold (8390)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    x = range(len(struct))
    ax1.bar([i - 0.2 for i in x], [sp[k] for k in struct], 0.4, label="self-play iter1", color=OI["selfplay"])
    ax1.bar([i + 0.2 for i in x], [go[k] for k in struct], 0.4, label="human gold", color=OI["gold"])
    ax1.set_xticks(list(x)); ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("% of pairs"); ax1.set_title("Structural complexity"); ax1.legend(frameon=False)
    div = ["unique_templates", "templates_per_100"]
    ax2.bar([i - 0.2 for i in range(len(div))], [sp[k] for k in div], 0.4, color=OI["selfplay"])
    ax2.bar([i + 0.2 for i in range(len(div))], [go[k] for k in div], 0.4, color=OI["gold"])
    ax2.set_xticks(range(len(div))); ax2.set_xticklabels(["unique\ntemplates", "templates\nper 100"])
    ax2.set_title("Template diversity")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "ara/evidence/figures/fig_coverage.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
