#!/bin/bash
set -euo pipefail
python3 - <<PY
import sys
sys.path.insert(0, ".")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, importlib.util
spec = importlib.util.spec_from_file_location("p", "verl/prompts.py"); pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)
from search.sql_exec_server import ExecConfig, get_schema
CH = "./checkpoints/dr-zero-t2sql/t2sql_challenger_bird1_3b_grpo_batch_group1-5_qwen2.5-coder-3b-instruct/hf_model"
tok = AutoTokenizer.from_pretrained(CH)
model = AutoModelForCausalLM.from_pretrained(CH, torch_dtype=torch.bfloat16, device_map="cuda")
cfg = ExecConfig(db_root="./bird/train_databases", timeout=15.0, max_rows=50, num_workers=1)
for db, level in [("airline", 3), ("books", 2), ("movielens", 1)]:
    schema = get_schema(cfg, db, sample_rows=2)["schema"]
    prompt = pr.T2SQL_CHALLENGER_PREFIX.format(level=level, level_description=pr.T2SQL_LEVEL_DESCRIPTIONS[level], schema=schema)
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(ids, max_new_tokens=400, temperature=1.0, do_sample=True, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print(); print("="*70); print("DB=", db, " LEVEL=", level); print(text[:900])
PY
