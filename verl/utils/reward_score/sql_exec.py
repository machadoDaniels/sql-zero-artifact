# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Execution-accuracy reward for the Text-to-SQL solver (Dr. Zero -> T2SQL).
#
# Three-tier reward (Arctic-Text2SQL-R1):
#   1.0  -> predicted SQL executes and its result-set matches the gold result-set
#   0.1  -> predicted SQL executes without error but the result does not match
#   0.0  -> no <sql> tag, SQL fails to execute, or comparison unavailable
#
# Never negative (destabilizes GRPO). Result-set comparison (multiset,
# order-agnostic unless the gold has ORDER BY) is delegated to the local
# SQLite execution server (search/sql_exec_server.py) via POST /compare.

import json
import logging
import os
import re
import time

import requests

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SQL_PATTERN = r"<sql>(.*?)</sql>"

SQL_EXEC_URL = os.getenv("SQL_EXEC_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT = float(os.getenv("SQL_EXEC_REQUEST_TIMEOUT", "60"))
MAX_RETRIES = 3

REWARD_MATCH = 1.0
REWARD_EXEC_ONLY = 0.1
REWARD_FAIL = 0.0


def extract_sql(solution_str):
    """Extract the final SQL from the last <sql>...</sql> block."""
    matches = re.findall(SQL_PATTERN, solution_str, re.DOTALL)
    if not matches:
        return None
    sql = matches[-1].strip()
    return sql if sql else None


def parse_ground_truth(ground_truth):
    """ground_truth is {"db_id": ..., "gold_sql": ...} as dict or JSON string."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    return ground_truth["db_id"], ground_truth["gold_sql"]


def compare_batch(items, base_url=None, timeout=None):
    """POST a batch of {db_id, pred_sql, gold_sql} to the exec server /compare.

    Returns the list of comparison dicts. Retries transient failures, then
    raises: a dead executor must fail loudly, not silently zero all rewards.
    """
    base_url = base_url or SQL_EXEC_URL
    timeout = timeout or REQUEST_TIMEOUT
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{base_url}/compare", json={"items": items}, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()["result"]
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_err = e
            logger.warning(f"/compare attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            time.sleep(2**attempt)
    raise RuntimeError(
        f"SQL execution server unreachable at {base_url} after {MAX_RETRIES} attempts: {last_err}"
    )


def score_from_comparison(comparison):
    """Map one /compare result dict to the three-tier reward."""
    if comparison["match"]:
        return REWARD_MATCH
    if comparison["pred"]["exec_ok"]:
        return REWARD_EXEC_ONLY
    return REWARD_FAIL


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """Single-sample scorer wired into default_compute_score (data_source t2sql*)."""
    sql = extract_sql(solution_str)
    if sql is None:
        return REWARD_FAIL

    db_id, gold_sql = parse_ground_truth(ground_truth)
    comparison = compare_batch(
        [{"db_id": db_id, "pred_sql": sql, "gold_sql": gold_sql}]
    )[0]

    reward = score_from_comparison(comparison)

    if os.getenv("SQL_EXEC_DEBUG"):
        print(
            f"[sql_exec] db={db_id} reward={reward} "
            f"pred_ok={comparison['pred']['exec_ok']} match={comparison['match']}\n"
            f"  pred: {sql[:200]}\n  gold: {gold_sql[:200]}"
        )
    return reward
