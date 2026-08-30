# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# End-to-end test of scripts/eval_spider.py: toy Spider layout + stub
# OpenAI-compatible endpoint -> checks the reported execution accuracy.
#
# Run: .venv/bin/python -m pytest tests/test_eval_spider.py -v

import json
import os
import sqlite3
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8766


class StubEndpoint(BaseHTTPRequestHandler):
    """Answers by question keyword: correct SQL, wrong SQL, or no tag at all."""

    ANSWERS = {
        "how many singers": "<think> count </think><sql>SELECT COUNT(*) FROM singer</sql>",
        "oldest singer": "<think> wrong </think><sql>SELECT name FROM singer ORDER BY age ASC LIMIT 1</sql>",
        "countries": "no sql tag in this response",
    }

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"].lower()
        content = next((a for k, a in self.ANSWERS.items() if k in prompt), "<sql>SELECT 1</sql>")
        payload = {"choices": [{"message": {"content": content}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture(scope="module")
def spider_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("spider")
    db_dir = root / "database" / "concert"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "concert.sqlite")
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
        INSERT INTO singer VALUES (1, 'Joe', 'NL', 52), (2, 'Tim', 'US', 32), (3, 'Ana', 'BR', 29);
        """
    )
    conn.commit()
    conn.close()

    dev = [
        # Stub answers correctly (same result-set).
        {"db_id": "concert", "question": "How many singers are there?",
         "query": "SELECT count(*) FROM singer"},
        # Stub answers ASC instead of DESC -> executes but wrong result.
        {"db_id": "concert", "question": "What is the name of the oldest singer?",
         "query": "SELECT name FROM singer ORDER BY age DESC LIMIT 1"},
        # Stub returns no <sql> tag.
        {"db_id": "concert", "question": "List all countries of singers.",
         "query": "SELECT DISTINCT country FROM singer"},
    ]
    (root / "dev.json").write_text(json.dumps(dev))
    return root


@pytest.fixture(scope="module")
def stub_endpoint():
    server = HTTPServer(("127.0.0.1", PORT), StubEndpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{PORT}/v1"
    server.shutdown()


def test_eval_spider_end_to_end(spider_dir, stub_endpoint, tmp_path):
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [
            sys.executable, os.path.join(REPO_ROOT, "scripts", "eval_spider.py"),
            "--dev_json", str(spider_dir / "dev.json"),
            "--db_root", str(spider_dir / "database"),
            "--endpoint", stub_endpoint,
            "--model", "stub-model",
            "--out", str(out),
            "--concurrency", "2",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    report = json.loads(out.read_text())
    summary = report["summary"]

    # 1 of 3 correct; 2 of 3 executed; 1 of 3 without <sql> tag.
    assert summary["n"] == 3
    assert summary["execution_accuracy"] == pytest.approx(1 / 3)
    assert summary["exec_ok_rate"] == pytest.approx(2 / 3)
    assert summary["missing_sql_tag_rate"] == pytest.approx(1 / 3)

    # Gold levels under the rubric: COUNT(*) -> 2 (aggregation); the other two -> 1.
    by_level = {str(k): v for k, v in summary["by_level"].items()}
    assert by_level["1"]["n"] == 2 and by_level["1"]["execution_accuracy"] == 0.0
    assert by_level["2"]["n"] == 1 and by_level["2"]["execution_accuracy"] == 1.0

    # Console output includes the headline metric.
    assert "execution accuracy" in proc.stdout
