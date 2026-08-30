"""Re-measure BIRD-train schema sizes with the project's OWN serializer.

Read-only. Reuses search.sql_exec_server.get_schema (the exact function
t2sql_data.schema_provider calls) so the numbers are the ones that mattered.
"""
import csv
import os
import sys

sys.path.insert(0, "/workspace/projects/dr-zero-t2sql")
from search.sql_exec_server import ExecConfig, get_schema  # noqa: E402

ROOT = "/workspace/projects/dr-zero-t2sql"
DB_ROOT = os.path.join(ROOT, "bird/train_databases")
KEEP = os.path.join(ROOT, "bird/train_dbs_keep.txt")
OUT = os.path.join(ROOT, "bird/census_recheck_2026-07-28.csv")

EXCLUDED = ["bike_share_1", "codebase_comments", "donor", "movie_platform", "works_cycles"]

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

with open(KEEP) as f:
    keep = [ln.strip() for ln in f if ln.strip()]
print(f"keep list: {len(keep)} entries", flush=True)

cfg = ExecConfig(db_root=DB_ROOT, timeout=10.0, max_rows=50, num_workers=8)

rows = []
for db_id in keep + EXCLUDED:
    status = "kept" if db_id in keep else "excluded"
    p = os.path.join(DB_ROOT, db_id, f"{db_id}.sqlite")
    size = os.path.getsize(p) if os.path.exists(p) else -1
    rec = {"db_id": db_id, "class": status, "sqlite_bytes": size}
    if size < 0:
        rec.update(present=0, n_tables=-1, chars_sr2=-1, tok_sr2=-1,
                   chars_sr0=-1, tok_sr0=-1, chars_sr3=-1, tok_sr3=-1, note="db-file-absent")
        rows.append(rec)
        print(rec, flush=True)
        continue
    rec["present"] = 1
    rec["note"] = ""
    for sr in (0, 2, 3):
        r = get_schema(cfg, db_id, sample_rows=sr)
        if not r["exec_ok"]:
            rec["note"] = f"schema-fail:{r['error']}"
            rec[f"chars_sr{sr}"] = -1
            rec[f"tok_sr{sr}"] = -1
            continue
        s = r["schema"]
        rec[f"chars_sr{sr}"] = len(s)
        rec[f"tok_sr{sr}"] = len(tok.encode(s))
        if sr == 0:
            rec["n_tables"] = s.count("CREATE TABLE") + s.count("create table")
    rows.append(rec)
    print(rec, flush=True)

cols = ["db_id", "class", "present", "n_tables", "sqlite_bytes",
        "chars_sr0", "tok_sr0", "chars_sr2", "tok_sr2", "chars_sr3", "tok_sr3", "note"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in cols})

kept = [r for r in rows if r["class"] == "kept" and r.get("tok_sr2", -1) > 0]
print(f"\nWROTE {OUT}  ({len(rows)} rows, {len(kept)} kept measured)")
print(f"kept tok_sr2  max={max(r['tok_sr2'] for r in kept)} "
      f"min={min(r['tok_sr2'] for r in kept)}")
top = sorted(kept, key=lambda r: -r["tok_sr2"])[:8]
for r in top:
    print(f"  TOP kept: {r['db_id']:32s} tables={r['n_tables']:3d} tok_sr2={r['tok_sr2']}")
print("CENSUS-RECHECK-COMPLETO")
