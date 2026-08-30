# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# End-to-end tests for the t2sql_data.py CLI (challenger-prompts, test-set,
# gen-solver-data) over a toy Spider layout and a stub proposer endpoint.
#
# Run: .venv/bin/python -m pytest tests/test_t2sql_data.py -v

import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8769


def make_db(path, script):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(script)
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def spider_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("spider")
    make_db(
        root / "database" / "concert" / "concert.sqlite",
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Joe', 52), (2, 'Tim', 32), (3, 'Ana', 29);
        """,
    )
    make_db(
        root / "database" / "pets" / "pets.sqlite",
        """
        CREATE TABLE pet (id INTEGER PRIMARY KEY, kind TEXT);
        INSERT INTO pet VALUES (1, 'dog'), (2, 'cat');
        """,
    )
    # concert -> train, pets -> dev
    (root / "train_spider.json").write_text(json.dumps(
        [{"db_id": "concert", "question": "q", "query": "SELECT 1"}]
    ))
    (root / "dev.json").write_text(json.dumps([
        {"db_id": "pets", "question": "How many pets are there?",
         "query": "SELECT count(*) FROM pet"},
        {"db_id": "pets", "question": "List the kinds of pets.",
         "query": "SELECT kind FROM pet"},
    ]))
    return root


class StubProposer(BaseHTTPRequestHandler):
    """Returns a fixed batch: valid pair, invalid SQL, empty result, duplicate."""

    RESPONSES = [
        "<think> ok </think><sql>SELECT name FROM singer WHERE age > 30</sql>"
        "<question>Which artists are older than thirty?</question>",
        "<think> bad </think><sql>SELECT nope FROM nowhere</sql>"
        "<question>Broken?</question>",
        "<think> empty </think><sql>SELECT name FROM singer WHERE age > 999</sql>"
        "<question>Nobody?</question>",
        # Duplicate template of the first pair (different literal).
        "<think> dup </think><sql>SELECT name FROM singer WHERE age > 50</sql>"
        "<question>Which artists are older than fifty?</question>",
        "no tags at all",
    ]

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = body["n"]
        choices = [{"message": {"content": self.RESPONSES[i % len(self.RESPONSES)]}}
                   for i in range(n)]
        data = json.dumps({"choices": choices}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def proposer_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), StubProposer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{PORT}/v1"
    server.shutdown()


def run_cli(spider_dir, *cli_args):
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "t2sql_data.py"),
         "--spider_dir", str(spider_dir), *cli_args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def test_challenger_prompts(spider_dir, tmp_path):
    out = tmp_path / "zero.parquet"
    run_cli(spider_dir, "challenger-prompts", "--out", str(out), "--num_prompts", "12")
    df = pd.read_parquet(out)

    assert len(df) == 12
    # Hard rule: only train databases (concert), never dev (pets).
    assert {info["db_id"] for info in df["extra_info"]} == {"concert"}
    # data_source carries the sampled level (HRPO-lite grouping key).
    assert set(df["data_source"]) <= {"t2sql_zero_1", "t2sql_zero_2", "t2sql_zero_3"}
    # Prompt embeds the schema and the level description.
    content = df["prompt"][0][0]["content"]
    assert "CREATE TABLE singer" in content and "complexity level" in content


def test_test_set(spider_dir, tmp_path):
    out = tmp_path / "test.parquet"
    run_cli(spider_dir, "test-set", "--out", str(out))
    df = pd.read_parquet(out)

    assert len(df) == 2
    assert set(df["data_source"]) == {"t2sql_spider_dev"}
    gt = json.loads(df["reward_model"][0]["ground_truth"])
    assert gt == {"db_id": "pets", "gold_sql": "SELECT count(*) FROM pet"}
    # COUNT(*) -> level 2; plain SELECT -> level 1.
    assert [info["level"] for info in df["extra_info"]] == [2, 1]
    assert "How many pets are there?" in df["prompt"][0][0]["content"]


def test_gen_solver_data(spider_dir, proposer_endpoint, tmp_path):
    out = tmp_path / "solver.parquet"
    proc = run_cli(
        spider_dir, "gen-solver-data",
        "--endpoint", proposer_endpoint, "--model", "stub",
        "--out", str(out), "--num_prompts", "2", "--samples_per_prompt", "5",
    )
    df = pd.read_parquet(out)

    # Per prompt, the stub yields: 1 valid, 1 invalid, 1 empty-result,
    # 1 duplicate-template, 1 untagged. Across 2 prompts the second valid
    # is also a duplicate template -> exactly 1 row survives.
    assert len(df) == 1
    gt = json.loads(df["reward_model"][0]["ground_truth"])
    assert gt["db_id"] == "concert"
    assert gt["gold_sql"] == "SELECT name FROM singer WHERE age > 30"
    assert df["data_source"][0] == "t2sql_zero_1"
    assert "Which artists are older than thirty?" in df["prompt"][0][0]["content"]
    # Filter accounting is reported.
    assert "invalid=" in proc.stderr or "invalid=" in proc.stdout
