#!/bin/bash
#SBATCH --job-name=mcnemar-chained
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -uo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
# Mesma comparacao pareada do paper, com os solvers encadeados no lugar dos cold-start.
# iter1 e compartilhada entre as duas receitas, entao g1b permanece.
python3 - <<'PY'
import sys, json, os
sys.argv=['mcnemar']
sys.path.insert(0,'scripts')
import mcnemar_paired as M
M.MODELS={
  "goldbird3b": "results/goldbird3b_s32_bird.json",
  "g1b":        "results/g1b20k_s78_bird.json",
  "bird2c":     "results/bird2c20k_s78_bird.json",
  "bird3c":     "results/bird3c20k_s27_bird.json",
}
M.CACHE="results/mcnemar_perex_cache_chained.json"
scored={m:M.score_model(p) for m,p in M.MODELS.items()}
rows=[];pv=[]
for m in ("g1b","bird2c","bird3c"):
    r=M.mcnemar(scored["goldbird3b"],scored[m]); rows.append((m,r)); pv.append(r["p_exact_two_sided"])
adj=M.holm(pv)
print("\nMcNemar exact — gold control vs CHAINED self-play (BIRD dev)\n")
hdr=["comparison","n","acc_gold","acc_sp","n01(gold+)","n10(sp+)","disc","p_exact","p_holm"]
print("| "+" | ".join(hdr)+" |"); print("|"+"|".join("---" for _ in hdr)+"|")
for (m,r),pa in zip(rows,adj):
    print(f"| gold vs {m} | {r['n']} | {r['acc_control']} | {r['acc_selfplay']} | {r['n01_control_only']} | {r['n10_selfplay_only']} | {r['discordant']} | {r['p_exact_two_sided']:.4g} | {pa:.4g} |")
PY
