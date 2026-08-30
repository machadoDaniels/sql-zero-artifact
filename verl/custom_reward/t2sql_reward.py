# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Challenger (proposer) reward for the Text-to-SQL adaptation of Dr. Zero.
#
# The challenger emits, single-shot: <think> ... </think> <sql> gold </sql>
# <question> NL question </question>. Its reward is
#
#     final = 0.5 * format + difficulty
#
# format (0..1): integrity gate (sql + question present, no SQL leaked into
#   the question) x components (think tag, gold validity). Gold validity is
#   checked by EXECUTION: single SELECT, deterministic, runs without error,
#   non-empty result (DECISOES_ARQUITETURA.md §2/§3).
#
# difficulty (0..1): the Dr. Zero hard-yet-solvable score. The CURRENT solver
#   answers the question n times (single-shot generations against an
#   OpenAI-compatible sglang server); with k = #exec-matches against the gold,
#   difficulty = (n-k)/(n-1) if 0 < k < n else 0 (peak at k=1).
#
# This module is deliberately lightweight (no torch / verl imports) so the
# logic can be unit-tested locally; shared code is loaded by file path.

import importlib.util
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(relative_path, name):
    path = os.path.normpath(os.path.join(_THIS_DIR, relative_path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_prompts = _load_module("../prompts.py", "t2sql_prompts")
_sql_utils = _load_module("../../search/sql_utils.py", "t2sql_sql_utils")

SQL_PATTERN = _prompts.SQL_PATTERN
QUESTION_PATTERN = _prompts.QUESTION_PATTERN
THINK_PATTERN = _prompts.THINK_PATTERN
T2SQL_SOLVER_PREFIX = _prompts.T2SQL_SOLVER_PREFIX

complexity_level = _sql_utils.complexity_level
is_deterministic = _sql_utils.is_deterministic
is_single_select = _sql_utils.is_single_select
sql_template = _sql_utils.sql_template

MAX_RETRIES = 3
# Exec-server request timeout (s). Configurable so that, under CPU contention on
# shared nodes, the /compare batch of a 7B run does not exceed the client deadline
# and kill training (default preserves prior behavior). See docs armadilha exec-service.
EXEC_TIMEOUT_S = float(os.environ.get("EXEC_TIMEOUT_S", "120"))
_schema_cache = {}


# --------------------------------------------------------------------------
# Parsing and scoring helpers
# --------------------------------------------------------------------------

def parse_challenger_response(text):
    """Extract (think_ok, gold_sql, question) from a challenger generation."""
    think_ok = bool(re.match(THINK_PATTERN, text.strip(), re.DOTALL))
    sqls = re.findall(SQL_PATTERN, text, re.DOTALL)
    questions = re.findall(QUESTION_PATTERN, text, re.DOTALL)
    sql = sqls[-1].strip() if sqls else None
    question = questions[-1].strip() if questions else None
    return think_ok, sql or None, question or None


def question_integrity(sql, question):
    """The pair is usable and the question does not leak the SQL."""
    if not sql or not question:
        return False
    if sql.lower() in question.lower() or "select " in question.lower():
        return False
    return True


def compute_difficulty_score(binary_list):
    """Dr. Zero hard-yet-solvable score: peak at exactly one solver success."""
    assert len(binary_list) > 1
    n = len(binary_list)
    num_ones = sum(binary_list)
    if num_ones == 0 or num_ones == n:
        return 0.0
    return (n - num_ones) / (n - 1)


# --------------------------------------------------------------------------
# HTTP helpers (SQL exec server + solver sglang server)
# --------------------------------------------------------------------------

def _post_json(url, payload, timeout):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - retry any transport error
            last_err = e
            logger.warning(f"POST {url} attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            time.sleep(2**attempt)
    raise RuntimeError(f"POST {url} failed after {MAX_RETRIES} attempts: {last_err}")


def execute_batch(exec_url, items, timeout=EXEC_TIMEOUT_S):
    return _post_json(f"{exec_url}/execute", {"queries": items}, timeout)["result"]


def compare_batch(exec_url, items, timeout=EXEC_TIMEOUT_S):
    return _post_json(f"{exec_url}/compare", {"items": items}, timeout)["result"]


def get_schema(exec_url, db_id, timeout=60.0):
    if db_id not in _schema_cache:
        result = _post_json(f"{exec_url}/schema", {"db_ids": [db_id]}, timeout)["result"][0]
        _schema_cache[db_id] = result["schema"]
    return _schema_cache[db_id]


def solve_question(base_url, model_name, prompt, n, max_tokens, timeout=300.0):
    """n single-shot solver generations; returns the extracted SQLs (or None)."""
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": 1.0,
            "max_tokens": max_tokens,
        }
        choices = _post_json(f"{base_url}/v1/chat/completions", payload, timeout)["choices"]
    except RuntimeError as e:
        logger.error(f"solver generation failed, difficulty set to 0: {e}")
        return [None] * n

    sqls = []
    for choice in choices:
        # format-agnostic (<sql> preferido, fallback ```sql) — justo para
        # oponentes congelados com formato proprio, ex. OmniSQL no G3.
        sqls.append(_prompts.extract_sql(choice["message"]["content"]))
    return sqls


# --------------------------------------------------------------------------
# Solver reward, batched (reward_model.reward_manager=batch)
# --------------------------------------------------------------------------

def solver_think_ok(solution):
    """Reasoning present and well-formed: a non-empty <think>...</think> that
    precedes the final <sql> (anti-hacking against empty/placeholder think)."""
    m = re.search(THINK_PATTERN, solution.strip(), re.DOTALL)
    if not m or len(m.group(1).strip()) < 10:
        return False
    sqls = list(re.finditer(SQL_PATTERN, solution, re.DOTALL))
    return bool(sqls) and m.end() <= sqls[-1].start()


def compute_t2sql_solver_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    """Execution reward (1.0 / 0.1 / 0.0), one /compare batch per step.

    Same semantics as verl/utils/reward_score/sql_exec.py, but batched: the
    per-sample dispatch would issue thousands of serial HTTP calls per step.

    Optional reasoning shaping (think-vs-no-think ablation): with
    SOLVER_THINK_BONUS=b>0, reward = (1-b)*exec + b*think_ok, mirroring the
    challenger's think component (Think2SQL / R1-Zero style). Default b=0 keeps
    the pure execution reward byte-identical to iter0/iter1.
    """
    exec_url = kwargs.get("sql_exec_url") or os.getenv("SQL_EXEC_URL", "http://127.0.0.1:8000")
    think_bonus = float(kwargs.get("think_bonus") or os.getenv("SOLVER_THINK_BONUS", "0.0"))

    exec_scores = [0.0] * len(solution_strs)
    items, owners = [], []
    for i, (solution, ground_truth) in enumerate(zip(solution_strs, ground_truths)):
        sql = _prompts.extract_sql(solution)
        if not sql:
            continue
        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)
        items.append(
            {"db_id": ground_truth["db_id"], "pred_sql": sql, "gold_sql": ground_truth["gold_sql"]}
        )
        owners.append(i)

    if items:
        for i, result in zip(owners, compare_batch(exec_url, items)):
            if result["match"]:
                exec_scores[i] = 1.0
            elif result["pred"]["exec_ok"]:
                exec_scores[i] = 0.1

    if think_bonus > 0:
        think = [float(solver_think_ok(s)) for s in solution_strs]
        scores = [(1 - think_bonus) * e + think_bonus * t for e, t in zip(exec_scores, think)]
    else:
        think = None
        scores = exec_scores

    msg = (
        f"🎯 T2SQL solver | avg {np.mean(scores):.3f} | "
        f"match {sum(e == 1.0 for e in exec_scores)}/{len(exec_scores)} | "
        f"exec-only {sum(e == 0.1 for e in exec_scores)}/{len(exec_scores)}"
    )
    if think is not None:
        msg += f" | think_ok {sum(think) / len(think):.2f}"
    print(msg)
    return scores


# --------------------------------------------------------------------------
# Challenger reward entry point (custom_reward_function.name in the config)
# --------------------------------------------------------------------------

def compute_t2sql_challenger_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs):
    model_name = kwargs["model_name"]
    base_url = kwargs["base_url"]
    n_rollouts = kwargs["reward_rollout_n"]
    exec_url = kwargs.get("sql_exec_url") or os.getenv("SQL_EXEC_URL", "http://127.0.0.1:8000")
    solver_max_tokens = kwargs.get("solver_max_tokens", 1024)
    concurrency = kwargs.get("gen_concurrency", 32)

    levels = [int(str(ds).split("_")[-1]) for ds in data_sources]
    db_ids = [info["db_id"] for info in extra_infos]

    # 1. Parse challenger outputs.
    parsed = [parse_challenger_response(s) for s in solution_strs]

    # 2. Gold validity: static checks + execution (non-empty, no error).
    gold_ok = []
    exec_idx, exec_items = [], []
    for i, (_, sql, _) in enumerate(parsed):
        if sql and is_single_select(sql) and is_deterministic(sql):
            exec_idx.append(i)
            exec_items.append({"db_id": db_ids[i], "sql": sql})
        gold_ok.append(False)
    if exec_items:
        for i, result in zip(exec_idx, execute_batch(exec_url, exec_items)):
            gold_ok[i] = result["exec_ok"] and result["n_rows"] > 0

    # 3. Format score: integrity gate x (constant, think tag, gold validity).
    format_scores = []
    for i, (think_ok, sql, question) in enumerate(parsed):
        if question_integrity(sql, question):
            format_scores.append((1.0 + float(think_ok) + float(gold_ok[i])) / 3)
        else:
            format_scores.append(0.0)

    # 4. Difficulty via n solver rollouts, only for valid pairs.
    eligible = [i for i in range(len(parsed)) if format_scores[i] > 0 and gold_ok[i]]
    difficulty = [0.0] * len(parsed)
    if eligible:
        prompts = []
        for i in eligible:
            schema = get_schema(exec_url, db_ids[i])
            prompts.append(T2SQL_SOLVER_PREFIX.format(schema=schema, question=parsed[i][2]))

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            all_solver_sqls = list(
                pool.map(
                    lambda p: solve_question(base_url, model_name, p, n_rollouts, solver_max_tokens),
                    prompts,
                )
            )

        # One flat /compare batch for all (question, sample) pairs.
        compare_items, owners = [], []
        for i, solver_sqls in zip(eligible, all_solver_sqls):
            for sql in solver_sqls:
                if sql is not None:
                    compare_items.append(
                        {"db_id": db_ids[i], "pred_sql": sql, "gold_sql": parsed[i][1]}
                    )
                    owners.append(i)
        matches = {i: [] for i in eligible}
        # Samples without an extracted SQL count as failures.
        for i, solver_sqls in zip(eligible, all_solver_sqls):
            matches[i].extend(0 for sql in solver_sqls if sql is None)
        if compare_items:
            for owner, result in zip(owners, compare_batch(exec_url, compare_items)):
                matches[owner].append(int(result["match"]))
        for i in eligible:
            difficulty[i] = compute_difficulty_score(matches[i])

    # 4b. Repetition penalty (R-Zero): cluster the batch's valid golds by masked
    # SQL template; penalize proportional to cluster size to fight the mode
    # collapse that wrecked gen2 (66% duplicates). rep_i = lambda * |cluster_i|/B.
    valid_pairs = [i for i in range(len(parsed)) if gold_ok[i]]
    rep_lambda = float(kwargs.get("rep_lambda") or os.getenv("CHALLENGER_REP_LAMBDA", "0.0"))
    rep_penalty = [0.0] * len(parsed)
    if rep_lambda > 0 and valid_pairs:
        from collections import Counter

        templates = {i: sql_template(parsed[i][1]) for i in valid_pairs}
        cluster_size = Counter(templates.values())
        batch = len(parsed)
        for i in valid_pairs:
            rep_penalty[i] = rep_lambda * cluster_size[templates[i]] / batch

    final_scores = [
        max(0.0, 0.5 * f + d - r)
        for f, d, r in zip(format_scores, difficulty, rep_penalty)
    ]

    # 5. Diagnostics (level mismatch is logged, not penalized — §2.1).
    mismatch = [i for i in valid_pairs if complexity_level(parsed[i][1]) != levels[i]]
    example = valid_pairs[0] if valid_pairs else None
    # k-histogram over eligible questions: where the mass sits (k=0 too hard/
    # ambiguous, k=n too easy, 0<k<n = the learnable band that pays reward).
    k_hist = {k: 0 for k in range(n_rollouts + 1)}
    for i in eligible:
        k_hist[sum(matches[i])] += 1
    eligible_diff = [difficulty[i] for i in eligible]

    # Componentes separados no wandb (o critic/score/mean só mostra o total).
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "challenger/format": float(np.mean(format_scores)),
                    "challenger/difficulty": float(np.mean(difficulty)),
                    "challenger/difficulty_eligible": float(np.mean(eligible_diff)) if eligible_diff else 0.0,
                    "challenger/valid_gold_frac": len(valid_pairs) / max(1, len(parsed)),
                    "challenger/unique_templates": (
                        len({sql_template(parsed[i][1]) for i in valid_pairs}) / max(1, len(valid_pairs))
                    ),
                    "challenger/rep_penalty": float(np.mean(rep_penalty)),
                    **{f"challenger/k_hist/{k}": v for k, v in k_hist.items()},
                },
                commit=False,
            )
    except Exception:  # noqa: BLE001 - logging must never break the reward
        pass
    print(
        f"🚀 T2SQL challenger | format: avg {np.mean(format_scores):.3f} | "
        f"difficulty: avg {np.mean(difficulty):.3f} | final: avg {np.mean(final_scores):.3f}\n"
        f"🎚️ k-hist (n={n_rollouts}): {k_hist} | difficulty entre elegíveis: "
        f"avg {np.mean(eligible_diff) if eligible_diff else 0.0:.3f}\n"
        f"📊 valid gold: {len(valid_pairs)}/{len(parsed)} | "
        f"level mismatch: {len(mismatch)}/{max(1, len(valid_pairs))}"
        + (
            f"\n🧑‍🏫 example gold: {parsed[example][1]}\n🙋 example question: {parsed[example][2]}"
            if example is not None
            else ""
        )
    )
    return final_scores
