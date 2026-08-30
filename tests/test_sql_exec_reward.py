# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Integration test for the solver execution reward:
#   real SQLite exec server (search/sql_exec_server.py) on a temp database
#   -> verl/utils/reward_score/sql_exec.py 3-tier reward over HTTP.
#
# Run: .venv/bin/python -m pytest tests/test_sql_exec_reward.py -v

import os
import sqlite3
import sys
import threading
import time

import pytest
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from search import sql_exec_server  # noqa: E402

PORT = 8765


@pytest.fixture(scope="module")
def exec_server(tmp_path_factory):
    """Start the real FastAPI exec server over a tiny Spider-style database."""
    db_root = tmp_path_factory.mktemp("db_root")
    db_dir = db_root / "concert"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "concert.sqlite")
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        INSERT INTO singer VALUES
            (1, 'Joe', 'Netherlands', 52),
            (2, 'Timbaland', 'USA', 32),
            (3, 'Ana', 'Brazil', 29);
        """
    )
    conn.commit()
    conn.close()

    sql_exec_server.config = sql_exec_server.ExecConfig(
        db_root=str(db_root), timeout=5.0, max_rows=50, num_workers=4
    )
    from concurrent.futures import ThreadPoolExecutor

    sql_exec_server.executor = ThreadPoolExecutor(max_workers=4)

    server = uvicorn.Server(
        uvicorn.Config(sql_exec_server.app, host="127.0.0.1", port=PORT, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started, "exec server failed to start"

    os.environ["SQL_EXEC_URL"] = f"http://127.0.0.1:{PORT}"
    yield
    server.should_exit = True


@pytest.fixture()
def compute_score(exec_server):
    # Load sql_exec.py standalone: importing the verl package would pull in
    # torch and other training-only deps that are not installed locally.
    import importlib.util

    path = os.path.join(REPO_ROOT, "verl", "utils", "reward_score", "sql_exec.py")
    spec = importlib.util.spec_from_file_location("sql_exec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


GOLD = {"db_id": "concert", "gold_sql": "SELECT name FROM singer WHERE age > 30"}


def solution(sql):
    return f"<think> some reasoning </think>\n<sql> {sql} </sql>"


def test_match_gives_full_reward(compute_score):
    assert compute_score(solution("SELECT name FROM singer WHERE age > 30"), GOLD) == 1.0


def test_equivalent_sql_different_order_matches(compute_score):
    # Same result-set, different row order; gold has no ORDER BY -> order-agnostic.
    sql = "SELECT name FROM singer WHERE age > 30 ORDER BY name DESC"
    assert compute_score(solution(sql), GOLD) == 1.0


def test_wrong_result_gives_partial_reward(compute_score):
    assert compute_score(solution("SELECT name FROM singer WHERE age > 50"), GOLD) == 0.1


def test_invalid_sql_gives_zero(compute_score):
    assert compute_score(solution("SELECT nope FROM nowhere"), GOLD) == 0.0


def test_missing_sql_tag_gives_zero(compute_score):
    assert compute_score("<think> no sql here </think>", GOLD) == 0.0


def test_empty_sql_tag_gives_zero(compute_score):
    assert compute_score(solution(""), GOLD) == 0.0


def test_ground_truth_as_json_string(compute_score):
    import json

    assert compute_score(solution(GOLD["gold_sql"]), json.dumps(GOLD)) == 1.0


def test_uses_last_sql_block(compute_score):
    text = solution("SELECT 1") + "\n" + solution(GOLD["gold_sql"])
    assert compute_score(text, GOLD) == 1.0


def test_gold_with_order_by_enforces_order(compute_score):
    gold = {"db_id": "concert", "gold_sql": "SELECT name FROM singer ORDER BY age ASC"}
    wrong_order = "SELECT name FROM singer ORDER BY age DESC"
    assert compute_score(solution(wrong_order), gold) == 0.1
    assert compute_score(solution(gold["gold_sql"]), gold) == 1.0
