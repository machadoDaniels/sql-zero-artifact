# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Data processing CLI for the Text-to-SQL adaptation of Dr. Zero.
#
# Subcommands (see DECISOES_ARQUITETURA.md §6):
#   challenger-prompts  -> parquet of challenger prompts (RL training of the proposer)
#   test-set            -> parquet of Spider dev in solver format (validation / eval)
#   gen-solver-data     -> generate (question, gold SQL) pairs with a served proposer
#                          model, apply the validity filters, and write the solver
#                          training parquet. Used both for iter0 (base model as
#                          proposer) and iter1+ (trained challenger checkpoint).
#
# Hard rule: challenger prompts and solver data only ever use the Spider TRAIN
# databases; the dev databases stay unseen until evaluation (§8).

import argparse
import json
import logging
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from search.sql_exec_server import ExecConfig, execute_sql, get_schema  # noqa: E402
from search.sql_utils import (  # noqa: E402
    complexity_level,
    is_deterministic,
    is_single_select,
    sql_template,
)
# verl/prompts.py is dependency-free, but `import verl.prompts` would pull the
# whole training stack via verl/__init__.py; load it by file path instead.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "t2sql_prompts", os.path.join(REPO_ROOT, "verl", "prompts.py")
)
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)

QUESTION_PATTERN = _prompts.QUESTION_PATTERN
SQL_PATTERN = _prompts.SQL_PATTERN
T2SQL_CHALLENGER_PREFIX = _prompts.T2SQL_CHALLENGER_PREFIX
T2SQL_LEVEL_DESCRIPTIONS = _prompts.T2SQL_LEVEL_DESCRIPTIONS
T2SQL_SOLVER_PREFIX = _prompts.T2SQL_SOLVER_PREFIX

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LEVELS = sorted(T2SQL_LEVEL_DESCRIPTIONS)


# --------------------------------------------------------------------------
# Spider helpers
# --------------------------------------------------------------------------

def load_db_splits(spider_dir):
    """Return (train_db_ids, dev_db_ids); they are disjoint in Spider."""
    def db_ids(*files):
        ids = set()
        for name in files:
            path = os.path.join(spider_dir, name)
            if os.path.exists(path):
                with open(path) as f:
                    ids.update(item["db_id"] for item in json.load(f))
        return ids

    train = db_ids("train_spider.json", "train_others.json")
    dev = db_ids("dev.json")
    train -= dev  # safety: never let a dev database into the training pool
    if not train:
        raise SystemExit(f"no train databases found under {spider_dir}")
    return sorted(train), sorted(dev)


def schema_provider(spider_dir, sample_rows, db_root=None):
    """Cached db_id -> serialized schema (DDL + example rows)."""
    config = ExecConfig(
        db_root=db_root or os.path.join(spider_dir, "database"), timeout=10.0, max_rows=50, num_workers=8
    )
    cache = {}

    def get(db_id):
        if db_id not in cache:
            result = get_schema(config, db_id, sample_rows=sample_rows)
            if not result["exec_ok"]:
                raise RuntimeError(f"schema failed for {db_id}: {result['error']}")
            cache[db_id] = result["schema"]
        return cache[db_id]

    return get, config


# --------------------------------------------------------------------------
# Parquet row builders (schema mirrors the original Dr. Zero parquets)
# --------------------------------------------------------------------------

def challenger_row(index, db_id, level, schema):
    prompt = T2SQL_CHALLENGER_PREFIX.format(
        level=level, level_description=T2SQL_LEVEL_DESCRIPTIONS[level], schema=schema
    )
    return {
        "data_source": f"t2sql_zero_{level}",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "t2sql",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": index,
            "split": "train",
            "db_id": db_id,
            "level": level,
            "need_tools_kwargs": False,
        },
    }


def solver_row(index, split, db_id, question, gold_sql, schema):
    level = complexity_level(gold_sql)
    prompt = T2SQL_SOLVER_PREFIX.format(schema=schema, question=question)
    return {
        "data_source": f"t2sql_zero_{level}" if split == "train" else "t2sql_spider_dev",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "t2sql",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({"db_id": db_id, "gold_sql": gold_sql}),
        },
        "extra_info": {
            "index": index,
            "split": split,
            "db_id": db_id,
            "level": level,
            "need_tools_kwargs": False,
        },
    }


def save_parquet(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logger.info(f"saved {len(df)} rows to {out_path}")
    return df


# --------------------------------------------------------------------------
# Subcommand: challenger-prompts
# --------------------------------------------------------------------------

def cmd_challenger_prompts(args):
    if args.db_root:  # ambiente arbitrario (ex.: BIRD train): bancos vem de --db_list
        with open(args.db_list) as f:
            train_dbs = [line.strip() for line in f if line.strip()]
        get_schema_for, _ = schema_provider(args.spider_dir, args.sample_rows, db_root=args.db_root)
    else:  # padrao: Spider train (splits derivados dos jsons)
        train_dbs, _ = load_db_splits(args.spider_dir)
        get_schema_for, _ = schema_provider(args.spider_dir, args.sample_rows)
    rng = random.Random(args.seed)

    rows = []
    for index in range(args.num_prompts):
        db_id = rng.choice(train_dbs)
        level = rng.choice(LEVELS)
        rows.append(challenger_row(index, db_id, level, get_schema_for(db_id)))

    df = save_parquet(rows, args.out)
    logger.info(f"train databases: {len(train_dbs)} | levels: {df['data_source'].value_counts().to_dict()}")


# --------------------------------------------------------------------------
# Subcommand: test-set
# --------------------------------------------------------------------------

def cmd_test_set(args):
    dev_json = args.dev_json or os.path.join(args.spider_dir, "dev.json")
    with open(dev_json) as f:
        items = json.load(f)
    get_schema_for, _ = schema_provider(args.spider_dir, args.sample_rows, db_root=args.db_root)

    rows = []
    for i, item in enumerate(items):
        gold = item.get("query") or item["SQL"]  # Spider | BIRD
        question = item["question"]
        if item.get("evidence"):  # BIRD external-knowledge hint (pratica padrao)
            question = f"{question}\nHint: {item['evidence']}"
        rows.append(solver_row(i, "dev", item["db_id"], question, gold,
                               get_schema_for(item["db_id"])))
    save_parquet(rows, args.out)


# --------------------------------------------------------------------------
# Subcommand: gold-train (supervised control — "price of annotation")
# --------------------------------------------------------------------------

def cmd_gold_train(args):
    """Solver training set from the HUMAN-annotated Spider train pairs.

    Same format/protocol as the self-play solver data, but the (question, gold)
    pairs are the human labels from train_spider.json (+ train_others.json).
    Trains the exact same GRPO solver -> measures what human annotation buys."""
    items = []
    for name in ("train_spider.json", "train_others.json"):
        path = os.path.join(args.spider_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                items.extend(json.load(f))
    get_schema_for, _ = schema_provider(args.spider_dir, args.sample_rows)

    rows = [
        solver_row(i, "train", item["db_id"], item["question"], item["query"],
                   get_schema_for(item["db_id"]))
        for i, item in enumerate(items)
    ]
    df = save_parquet(rows, args.out)
    logger.info(f"gold pairs: {len(df)} | levels: {df['data_source'].value_counts().to_dict()}")


# --------------------------------------------------------------------------
# Subcommand: gold-train-bird (supervised control on BIRD, same protocol)
# --------------------------------------------------------------------------

def cmd_gold_train_bird(args):
    """Solver training set from the HUMAN-annotated BIRD train pairs.

    Mirrors cmd_gold_train but reads BIRD's train.json (db_id/question/evidence/SQL)
    and restricts to the vetted train-db subset (bird/train_dbs_keep.txt) used by
    the rest of the BIRD self-play experiments in this project."""
    with open(args.train_json) as f:
        items = json.load(f)
    if args.db_list:
        with open(args.db_list) as f:
            keep = {line.strip() for line in f if line.strip()}
        items = [item for item in items if item["db_id"] in keep]
    get_schema_for, _ = schema_provider(args.spider_dir, args.sample_rows, db_root=args.db_root)

    rows = []
    for i, item in enumerate(items):
        question = item["question"]
        if item.get("evidence"):
            question = f"{question}\nHint: {item['evidence']}"
        rows.append(solver_row(i, "train", item["db_id"], question, item["SQL"],
                                get_schema_for(item["db_id"])))
    df = save_parquet(rows, args.out)
    logger.info(f"gold pairs: {len(df)} | levels: {df['data_source'].value_counts().to_dict()}")


# --------------------------------------------------------------------------
# Subcommand: gen-solver-data
# --------------------------------------------------------------------------

def _generate_pairs(endpoint, model, prompt, n, max_tokens, timeout):
    """One proposer request with n samples; returns [(sql, question), ...]."""
    resp = requests.post(
        f"{endpoint}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": 1.0,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    pairs = []
    for choice in resp.json()["choices"]:
        text = choice["message"]["content"]
        sqls = re.findall(SQL_PATTERN, text, re.DOTALL)
        questions = re.findall(QUESTION_PATTERN, text, re.DOTALL)
        if sqls and questions:
            pairs.append((sqls[-1].strip(), questions[-1].strip()))
    return pairs


def valid_gold(sql, question, db_id, exec_config):
    """Validity filters from DECISOES_ARQUITETURA.md §2: static + execution."""
    if not sql or not question:
        return False
    if sql.lower() in question.lower() or "select " in question.lower():
        return False
    if not is_single_select(sql) or not is_deterministic(sql):
        return False
    result = execute_sql(exec_config, db_id, sql)
    return result["exec_ok"] and result["n_rows"] > 0


def cmd_gen_solver_data(args):
    if args.db_root:
        with open(args.db_list) as f:
            train_dbs = [line.strip() for line in f if line.strip()]
        get_schema_for, exec_config = schema_provider(args.spider_dir, args.sample_rows, db_root=args.db_root)
    else:
        train_dbs, _ = load_db_splits(args.spider_dir)
        get_schema_for, exec_config = schema_provider(args.spider_dir, args.sample_rows)
    rng = random.Random(args.seed)

    # One challenger prompt per (db, level) draw.
    draws = [(rng.choice(train_dbs), rng.choice(LEVELS)) for _ in range(args.num_prompts)]

    def run(draw):
        db_id, level = draw
        prompt = T2SQL_CHALLENGER_PREFIX.format(
            level=level, level_description=T2SQL_LEVEL_DESCRIPTIONS[level],
            schema=get_schema_for(db_id),
        )
        try:
            return db_id, level, _generate_pairs(
                args.endpoint, args.model, prompt,
                args.samples_per_prompt, args.max_tokens, args.request_timeout,
            )
        except Exception as e:  # noqa: BLE001 - skip failed generations
            logger.warning(f"generation failed for {db_id}: {e}")
            return db_id, level, []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run, draws))

    rows, seen_templates = [], set()
    stats = {"generated": 0, "invalid": 0, "duplicate": 0, "level_mismatch": 0}
    for db_id, level, pairs in results:
        for sql, question in pairs:
            stats["generated"] += 1
            if not valid_gold(sql, question, db_id, exec_config):
                stats["invalid"] += 1
                continue
            template_key = (db_id, sql_template(sql))
            if template_key in seen_templates:
                stats["duplicate"] += 1
                continue
            seen_templates.add(template_key)
            if complexity_level(sql) != level:
                stats["level_mismatch"] += 1  # logged, not rejected (§2.1)
            rows.append(
                solver_row(len(rows), "train", db_id, question, sql, get_schema_for(db_id))
            )
            if args.max_pairs and len(rows) >= args.max_pairs:
                break
        if args.max_pairs and len(rows) >= args.max_pairs:
            break

    logger.info(
        f"generated={stats['generated']} invalid={stats['invalid']} "
        f"duplicate={stats['duplicate']} level_mismatch={stats['level_mismatch']} "
        f"kept={len(rows)}"
    )
    if not rows:
        raise SystemExit("no valid pairs survived the filters")
    df = save_parquet(rows, args.out)
    logger.info(f"levels: {df['data_source'].value_counts().to_dict()}")


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T2SQL data processing (Dr. Zero adaptation).")
    parser.add_argument("--spider_dir", default="./spider")
    parser.add_argument("--sample_rows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("challenger-prompts", help="Challenger RL training prompts.")
    p.add_argument("--out", default="./data/t2sql_zero.parquet")
    p.add_argument("--num_prompts", type=int, default=20000)
    p.add_argument("--db_root", default=None, help="Env alternativo: raiz dos .sqlite (ex.: ./bird/train_databases).")
    p.add_argument("--db_list", default=None, help="Arquivo com um db_id por linha (obrigatorio com --db_root).")
    p.set_defaults(fn=cmd_challenger_prompts)

    p = sub.add_parser("test-set", help="Spider/BIRD dev in solver format.")
    p.add_argument("--out", default="./data/t2sql_test.parquet")
    p.add_argument("--dev_json", default=None, help="Override do json (ex.: BIRD dev.json).")
    p.add_argument("--db_root", default=None, help="Override da raiz dos bancos.")
    p.set_defaults(fn=cmd_test_set)

    p = sub.add_parser("gold-train", help="Human-annotated Spider train pairs (supervised control).")
    p.add_argument("--out", default="./data/t2sql_solver_gold.parquet")
    p.set_defaults(fn=cmd_gold_train)

    p = sub.add_parser("gold-train-bird", help="Human-annotated BIRD train pairs (supervised control).")
    p.add_argument("--out", default="./data/t2sql_solver_gold_bird.parquet")
    p.add_argument("--train_json", required=True, help="BIRD train.json (db_id/question/evidence/SQL).")
    p.add_argument("--db_root", required=True, help="Raiz dos bancos BIRD train (ex.: ./bird/train_databases).")
    p.add_argument("--db_list", default=None, help="db_id por linha p/ restringir (ex.: bird/train_dbs_keep.txt).")
    p.set_defaults(fn=cmd_gold_train_bird)

    p = sub.add_parser("gen-solver-data", help="Generate + filter solver training pairs.")
    p.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL of the proposer.")
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="./data/t2sql_solver_train.parquet")
    p.add_argument("--num_prompts", type=int, default=4000)
    p.add_argument("--samples_per_prompt", type=int, default=5)
    p.add_argument("--max_pairs", type=int, default=None)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--request_timeout", type=float, default=300.0)
    p.add_argument("--db_root", default=None, help="Env alternativo (ex.: ./bird/train_databases).")
    p.add_argument("--db_list", default=None, help="db_id por linha (obrigatorio com --db_root).")
    p.set_defaults(fn=cmd_gen_solver_data)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
