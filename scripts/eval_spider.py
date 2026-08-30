# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Zero-shot / checkpoint evaluation on Spider dev (execution accuracy).
#
# This is "passo 0" of the POC (DECISOES_ARQUITETURA.md §6): the baseline that
# any self-play iteration must beat, using the SAME solver prompt and schema
# serialization as training so gains are attributable to the RL, not the prompt.
#
# The model is served by any OpenAI-compatible endpoint (vLLM / sglang):
#   vllm serve Qwen/Qwen2.5-Coder-3B-Instruct --port 9000
#   python scripts/eval_spider.py --endpoint http://localhost:9000/v1 \
#       --model Qwen/Qwen2.5-Coder-3B-Instruct
#
# Execution and result-set comparison run in-process against the local Spider
# SQLite files (no exec server needed).

import argparse
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from search.sql_exec_server import ExecConfig, compare_results, get_schema  # noqa: E402
from search.sql_utils import complexity_level  # noqa: E402


def _load_module(path, name):
    """Load a module by file path, avoiding heavy package __init__ imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# verl/prompts.py is dependency-free, but `import verl.prompts` would pull the
# whole training stack via verl/__init__.py.
_prompts = _load_module(os.path.join(REPO_ROOT, "verl", "prompts.py"), "prompts")
T2SQL_SOLVER_PREFIX = _prompts.T2SQL_SOLVER_PREFIX
SQL_PATTERN = _prompts.SQL_PATTERN
extract_sql = _prompts.extract_sql

# Template NATIVO do OmniSQL (model card seeklhy/OmniSQL-7B) — para medir a
# penalidade de formato do protocolo unificado sobre checkpoints externos.
# db_details usa a NOSSA serializacao DDL+rows (aproximacao declarada).
OMNISQL_TEMPLATE = """Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{schema}
This schema describes the database\'s structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def extract_sql_any_fence(text):
    """extract_sql + fallback para bloco ``` generico (formato de saida do OmniSQL)."""
    sql = extract_sql(text)
    if sql:
        return sql
    blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if not blocks:
        return None
    body = blocks[-1].strip()
    lines = [l for l in body.splitlines() if not l.strip().startswith("--")]
    return "\n".join(lines).strip() or None


def generate(endpoint, model, prompt, max_tokens, timeout):
    """One greedy chat completion against an OpenAI-compatible endpoint."""
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def evaluate_item(item, schema_cache, config, args):
    """Accepts Spider (gold in 'query') and BIRD (gold in 'SQL' + 'evidence') items."""
    db_id = item["db_id"]
    gold_sql = item.get("query") or item["SQL"]
    question = item["question"]
    if item.get("evidence"):  # BIRD external-knowledge hint (standard practice)
        question = f"{question}\nHint: {item['evidence']}"

    if db_id not in schema_cache:
        schema_cache[db_id] = get_schema(config, db_id, sample_rows=args.sample_rows)["schema"]
    if args.prompt_style == "omnisql":
        prompt = OMNISQL_TEMPLATE.format(schema=schema_cache[db_id], question=question)
    else:
        prompt = T2SQL_SOLVER_PREFIX.format(schema=schema_cache[db_id], question=question)

    try:
        response = generate(args.endpoint, args.model, prompt, args.max_tokens, args.request_timeout)
    except Exception as e:  # noqa: BLE001 - record and move on
        return {"db_id": db_id, "question": question, "gold_sql": gold_sql,
                "pred_sql": None, "match": False, "exec_ok": False,
                "level": complexity_level(gold_sql), "error": f"generation: {e}"}

    pred_sql = extract_sql_any_fence(response) if args.prompt_style == "omnisql" else extract_sql(response)
    result = {
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "pred_sql": pred_sql,
        "level": complexity_level(gold_sql),
        "match": False,
        "exec_ok": False,
        "error": None,
    }
    if pred_sql is None:
        result["error"] = "no <sql> tag in response"
        return result

    comparison = compare_results(config, db_id, pred_sql, gold_sql)
    result["match"] = comparison["match"]
    result["exec_ok"] = comparison["pred"]["exec_ok"]
    if not comparison["pred"]["exec_ok"]:
        result["error"] = comparison["pred"]["error"]
    return result


def summarize(results):
    n = len(results)
    matches = sum(r["match"] for r in results)
    summary = {
        "n": n,
        "execution_accuracy": matches / n if n else 0.0,
        "exec_ok_rate": sum(r["exec_ok"] for r in results) / n if n else 0.0,
        "missing_sql_tag_rate": sum(r["pred_sql"] is None for r in results) / n if n else 0.0,
        "by_level": {},
    }
    by_level = Counter(r["level"] for r in results)
    for level in sorted(by_level):
        level_results = [r for r in results if r["level"] == level]
        summary["by_level"][level] = {
            "n": len(level_results),
            "execution_accuracy": sum(r["match"] for r in level_results) / len(level_results),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on Spider dev (execution accuracy).")
    parser.add_argument("--dev_json", default="./spider/dev.json")
    parser.add_argument("--db_root", default="./spider/database")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://host:9000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="./results/eval_spider.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--sample_rows", type=int, default=3)
    parser.add_argument("--exec_timeout", type=float, default=10.0)
    parser.add_argument("--request_timeout", type=float, default=180.0)
    parser.add_argument("--prompt_style", choices=["ours", "omnisql"], default="ours",
                        help="Formato do prompt: unificado (ours) ou nativo do OmniSQL.")
    args = parser.parse_args()

    with open(args.dev_json) as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    config = ExecConfig(db_root=args.db_root, timeout=args.exec_timeout, max_rows=50, num_workers=args.concurrency)
    schema_cache = {}
    # Warm the schema cache serially to avoid duplicate work across threads.
    for db_id in {item["db_id"] for item in items}:
        schema_cache[db_id] = get_schema(config, db_id, sample_rows=args.sample_rows)["schema"]

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda it: evaluate_item(it, schema_cache, config, args), items))

    summary = summarize(results)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "results": results}, f, indent=2)

    print(f"\n=== Spider dev — {args.model} ===")
    print(f"examples:            {summary['n']}")
    print(f"execution accuracy:  {summary['execution_accuracy']:.4f}")
    print(f"exec-ok rate:        {summary['exec_ok_rate']:.4f}")
    print(f"missing <sql> rate:  {summary['missing_sql_tag_rate']:.4f}")
    for level, stats in summary["by_level"].items():
        print(f"  level {level}: EX {stats['execution_accuracy']:.4f}  (n={stats['n']})")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
