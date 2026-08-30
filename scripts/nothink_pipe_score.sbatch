#!/bin/bash
#SBATCH --job-name=nt-pipe-score
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -uo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
TAG="$1"

# ---- Spider test: convert greedy.json -> pred txt ----
python3 - <<PY
import json,re
p=json.load(open('omnisql_pipeline/results/${TAG}_spidertest_greedy.json'))
def one(r):
    s=(r.get('pred_sqls') or [''])
    s=str(s[0] if s else '')
    s=re.sub(r'/\*.*?\*/',' ',s,flags=re.S)
    s='\n'.join(l for l in s.split('\n') if not l.strip().startswith('--'))
    s=re.sub(r'--[^\n]*',' ',s)
    return ' '.join(s.split()).rstrip(';') or 'SELECT 1'
open('official_eval/work/pred_${TAG}_spidertest.txt','w').write('\n'.join(one(r) for r in p)+'\n')
print('ST preds:',len(p))
PY

cd official_eval/test-suite-sql-eval
echo "=== SCORE Spider test ($TAG) ==="
singularity exec --bind /workspace/anon:/workspace \
  --pwd /workspace/projects/dr-zero-t2sql/official_eval/test-suite-sql-eval \
  --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/workspace/python-packages/eval-extra \
  --env NLTK_DATA=/workspace/nltk_data \
  /workspace/anon/images/verl0.5-sglang048.sif \
  python3 evaluation.py --gold ../work/gold_spidertest.txt --pred ../work/pred_${TAG}_spidertest.txt \
    --db ../../spider/test_database --table ../../spider/test_tables.json \
    --etype exec --plug_value --keep_distinct --progress_bar_for_each_datapoint 2>&1 | tail -15
echo "SCORE_ST_DONE_$TAG"

# ---- Spider-Syn: convert greedy.json -> pred txt ----
cd /workspace/anon/projects/dr-zero-t2sql
python3 - <<PY
import json,re
p=json.load(open('official_eval/work/${TAG}_syn_pred.json'))
def one(r):
    s=(r.get('pred_sqls') or [''])
    s=str(s[0] if s else '')
    s=re.sub(r'/\*.*?\*/',' ',s,flags=re.S)
    s='\n'.join(l for l in s.split('\n') if not l.strip().startswith('--'))
    s=re.sub(r'--[^\n]*',' ',s)
    return ' '.join(s.split()).rstrip(';') or 'SELECT 1'
open('official_eval/work/pred_${TAG}_syn.txt','w').write('\n'.join(one(r) for r in p)+'\n')
print('SYN preds:',len(p))
PY

cd official_eval/test-suite-sql-eval
echo "=== SCORE Spider-Syn ($TAG) ==="
singularity exec --bind /workspace/anon:/workspace \
  --pwd /workspace/projects/dr-zero-t2sql/official_eval/test-suite-sql-eval \
  --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/workspace/python-packages/eval-extra \
  --env NLTK_DATA=/workspace/nltk_data \
  /workspace/anon/images/verl0.5-sglang048.sif \
  python3 evaluation.py --gold ../work/gold_syn.txt --pred ../work/pred_${TAG}_syn.txt \
    --db ../../spider/database --table ../../spider/tables.json \
    --etype exec --plug_value --keep_distinct --progress_bar_for_each_datapoint 2>&1 | tail -15
echo "SCORE_SYN_DONE_$TAG"
