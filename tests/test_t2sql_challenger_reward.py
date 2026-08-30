# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Tests for the T2SQL challenger reward: real SQLite exec server + a stub
# solver endpoint with controllable per-question accuracy.
#
# Run: .venv/bin/python -m pytest tests/test_t2sql_challenger_reward.py -v

import importlib.util
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import uvicorn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from search import sql_exec_server  # noqa: E402

EXEC_PORT = 8767
SOLVER_PORT = 8768

GOLD_SQL = "SELECT name FROM singer WHERE age > 30"
WRONG_SQL = "SELECT name FROM singer WHERE age > 100"


class StubSolver(BaseHTTPRequestHandler):
    """Returns k correct answers out of n, keyed by a marker in the question."""

    K_BY_MARKER = {"[k1]": 1, "[k5]": 5, "[k0]": 0, "[k2]": 2}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        n = body["n"]
        k = next((v for m, v in self.K_BY_MARKER.items() if m in prompt), 0)
        choices = [
            {"message": {"content": f"<sql>{GOLD_SQL if i < k else WRONG_SQL}</sql>"}}
            for i in range(n)
        ]
        data = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    # Toy Spider-style database.
    db_root = tmp_path_factory.mktemp("db_root")
    db_dir = db_root / "concert"
    db_dir.mkdir()
    conn = sqlite3.connect(db_dir / "concert.sqlite")
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Joe', 'NL', 52), (2, 'Tim', 'US', 32), (3, 'Ana', 'BR', 29);
        """
    )
    conn.commit()
    conn.close()

    # Real exec server.
    sql_exec_server.config = sql_exec_server.ExecConfig(
        db_root=str(db_root), timeout=5.0, max_rows=50, num_workers=4
    )
    sql_exec_server.executor = ThreadPoolExecutor(max_workers=4)
    server = uvicorn.Server(
        uvicorn.Config(sql_exec_server.app, host="127.0.0.1", port=EXEC_PORT, log_level="error")
    )
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started

    # Stub solver endpoint.
    solver = ThreadingHTTPServer(("127.0.0.1", SOLVER_PORT), StubSolver)
    threading.Thread(target=solver.serve_forever, daemon=True).start()

    yield
    server.should_exit = True
    solver.shutdown()


@pytest.fixture(scope="module")
def reward(servers):
    path = os.path.join(REPO_ROOT, "verl", "custom_reward", "t2sql_reward.py")
    spec = importlib.util.spec_from_file_location("t2sql_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def call(solution_strs, levels=None):
        n = len(solution_strs)
        levels = levels or [1] * n
        return module.compute_t2sql_challenger_score_batch(
            data_sources=[f"t2sql_zero_{lv}" for lv in levels],
            solution_strs=solution_strs,
            ground_truths=[None] * n,
            extra_infos=[{"db_id": "concert"}] * n,
            model_name="stub",
            base_url=f"http://127.0.0.1:{SOLVER_PORT}",
            reward_rollout_n=5,
            sql_exec_url=f"http://127.0.0.1:{EXEC_PORT}",
        )

    return call


def pair(sql, question, think=True):
    prefix = "<think> planning the pair </think>\n" if think else ""
    return f"{prefix}<sql> {sql} </sql>\n<question> {question} </question>"


def test_hard_yet_solvable_gets_max_reward(reward):
    # k=1 of 5 -> difficulty (5-1)/4 = 1.0; format 1.0 -> final 1.5.
    scores = reward([pair(GOLD_SQL, "Which artists are older than thirty? [k1]")])
    assert scores == [pytest.approx(1.5)]


def test_too_easy_and_too_hard_get_format_only(reward):
    scores = reward(
        [
            pair(GOLD_SQL, "Which artists are older than thirty? [k5]"),  # trivial
            pair(GOLD_SQL, "Which artists are older than thirty? [k0]"),  # unsolvable
        ]
    )
    assert scores == [pytest.approx(0.5), pytest.approx(0.5)]


def test_intermediate_difficulty(reward):
    # k=2 of 5 -> difficulty (5-2)/4 = 0.75 -> final 0.5 + 0.75 = 1.25.
    scores = reward([pair(GOLD_SQL, "Which artists are older than thirty? [k2]")])
    assert scores == [pytest.approx(1.25)]


def test_invalid_gold_sql(reward):
    # Executes with error -> gold invalid -> format (1+1+0)/3, no difficulty.
    scores = reward([pair("SELECT nope FROM nowhere", "Broken question? [k1]")])
    assert scores == [pytest.approx(0.5 * 2 / 3)]


def test_empty_result_gold_rejected(reward):
    scores = reward([pair("SELECT name FROM singer WHERE age > 999", "Nobody? [k1]")])
    assert scores == [pytest.approx(0.5 * 2 / 3)]


def test_nondeterministic_gold_rejected(reward):
    scores = reward([pair("SELECT name FROM singer LIMIT 2", "Some singers? [k1]")])
    assert scores == [pytest.approx(0.5 * 2 / 3)]


def test_non_select_gold_rejected(reward):
    scores = reward([pair("DROP TABLE singer", "Delete everything? [k1]")])
    assert scores == [pytest.approx(0.5 * 2 / 3)]


def test_sql_leaked_into_question_gets_zero(reward):
    scores = reward([pair(GOLD_SQL, f"Run {GOLD_SQL} please [k1]")])
    assert scores == [0.0]


def test_missing_question_gets_zero(reward):
    scores = reward(["<think> hmm </think><sql> SELECT 1 </sql>"])
    assert scores == [0.0]


def test_missing_think_lowers_format(reward):
    # format (1+0+1)/3 = 2/3 -> 0.5*2/3 + difficulty 1.0 (k=1).
    scores = reward([pair(GOLD_SQL, "Which artists are older than thirty? [k1]", think=False)])
    assert scores == [pytest.approx(0.5 * 2 / 3 + 1.0)]


def test_solver_batch_reward(servers):
    import importlib.util

    path = os.path.join(REPO_ROOT, "verl", "custom_reward", "t2sql_reward.py")
    spec = importlib.util.spec_from_file_location("t2sql_reward_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    gt = json.dumps({"db_id": "concert", "gold_sql": GOLD_SQL})
    solutions = [
        f"<think> ok </think><sql>{GOLD_SQL}</sql>",          # match -> 1.0
        f"<think> hmm </think><sql>{WRONG_SQL}</sql>",        # executes, wrong -> 0.1
        "<think> bad </think><sql>SELECT nope FROM x</sql>",  # exec error -> 0.0
        "<think> no tag </think>",                            # missing sql -> 0.0
    ]
    scores = module.compute_t2sql_solver_score_batch(
        data_sources=["t2sql_zero_1"] * 4,
        solution_strs=solutions,
        ground_truths=[gt] * 4,
        extra_infos=[None] * 4,
        sql_exec_url=f"http://127.0.0.1:{EXEC_PORT}",
    )
    assert scores == [1.0, 0.1, 0.0, 0.0]


def test_batch_mixes_all_cases(reward):
    scores = reward(
        [
            pair(GOLD_SQL, "Good hard question [k1]"),
            pair("SELECT nope FROM nowhere", "Broken [k1]"),
            "<think> no tags at all </think>",
        ]
    )
    assert scores[0] == pytest.approx(1.5)
    assert scores[1] == pytest.approx(0.5 * 2 / 3)
    assert scores[2] == 0.0


def test_solver_think_bonus(servers):
    import importlib.util, os
    path = os.path.join(REPO_ROOT, "verl", "custom_reward", "t2sql_reward.py")
    spec = importlib.util.spec_from_file_location("t2sql_reward_think", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    gt = json.dumps({"db_id": "concert", "gold_sql": GOLD_SQL})
    think = "<think> reasoning about the query here </think>"
    sols = [
        f"{think}<sql>{GOLD_SQL}</sql>",          # match + think -> (1-b)*1 + b*1 = 1.0
        f"<sql>{GOLD_SQL}</sql>",                 # match, no think -> 0.9
        f"{think}<sql>{WRONG_SQL}</sql>",         # exec-only + think -> 0.9*0.1 + 0.1 = 0.19
        "<think></think><sql>SELECT 1</sql>",     # empty think -> no bonus
    ]
    scores = m.compute_t2sql_solver_score_batch(
        data_sources=["t2sql_zero_1"] * 4, solution_strs=sols,
        ground_truths=[gt] * 4, extra_infos=[None] * 4,
        sql_exec_url=f"http://127.0.0.1:{EXEC_PORT}", think_bonus=0.1,
    )
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.9)
    assert scores[2] == pytest.approx(0.19)
    # empty <think> gives no bonus; SELECT 1 executes-but-wrong -> 0.9*0.1
    assert scores[3] == pytest.approx(0.09)


def test_solver_think_ok_helper(servers):
    import importlib.util, os
    path = os.path.join(REPO_ROOT, "verl", "custom_reward", "t2sql_reward.py")
    spec = importlib.util.spec_from_file_location("t2sql_reward_h", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.solver_think_ok("<think> real reasoning content </think><sql>SELECT 1</sql>")
    assert not m.solver_think_ok("<think></think><sql>SELECT 1</sql>")          # empty
    assert not m.solver_think_ok("<sql>SELECT 1</sql>")                          # missing
    assert not m.solver_think_ok("<sql>SELECT 1</sql><think> after the sql </think>")  # wrong order


def test_repetition_penalty(reward):
    import importlib.util, os
    path = os.path.join(REPO_ROOT, "verl", "custom_reward", "t2sql_reward.py")
    spec = importlib.util.spec_from_file_location("t2sql_reward_rep", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    # Batch of 4 valid golds: 3 share template A (GOLD_SQL), 1 is template B.
    # (The stub only "solves" GOLD_SQL, so template A gets difficulty 1.0 and
    # template B gets 0 — irrelevant here; we assert the repetition penalty.)
    A = lambda q: pair(GOLD_SQL, q + " [k1]")
    tmplA = [A("Older than thirty?"), A("Above thirty?"), A("More than thirty years?")]
    tmplB = pair("SELECT country FROM singer WHERE age > 30", "Which countries? [k1]")
    scores = m.compute_t2sql_challenger_score_batch(
        data_sources=["t2sql_zero_1"] * 4, solution_strs=tmplA + [tmplB],
        ground_truths=[None] * 4, extra_infos=[{"db_id": "concert"}] * 4,
        model_name="stub", base_url=f"http://127.0.0.1:{SOLVER_PORT}",
        reward_rollout_n=5, sql_exec_url=f"http://127.0.0.1:{EXEC_PORT}",
        rep_lambda=0.4,
    )
    # cluster A size=3, B size=1; batch=4; lambda=0.4 -> penA=0.4*3/4=0.3, penB=0.1
    # A: 0.5*1 + difficulty(1.0) - 0.3 = 1.2 ; B: 0.5*1 + 0 - 0.1 = 0.4
    for i in range(3):
        assert scores[i] == pytest.approx(1.2)
    assert scores[3] == pytest.approx(0.4)


def test_no_repetition_penalty_by_default(reward):
    # Without rep_lambda, a duplicated-template batch keeps full scores.
    A = lambda q: pair(GOLD_SQL, q + " [k1]")
    scores = reward([A("Older than thirty?"), A("Above thirty?")])
    assert scores == [pytest.approx(1.5), pytest.approx(1.5)]
