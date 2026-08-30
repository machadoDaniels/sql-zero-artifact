# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SQLite execution server for the Dr. Zero Text-to-SQL adaptation.
#
# Mirrors the interface of `search/retrieval_server.py` (FastAPI on :8000) but,
# instead of retrieving Wikipedia passages, it executes SQL against the Spider
# SQLite databases. It is used in two places in the self-play pipeline:
#   * reward time      -> run the solver SQL and the gold SQL, compare result-sets
#   * proposer (SQL-first -> NL) -> execute the proposed gold SQL to guarantee it
#                                   is valid and returns a well-defined result.
#
# Database layout expected (standard Spider release):
#   <db_root>/<db_id>/<db_id>.sqlite

import math
import os
import re
import argparse
import sqlite3
import threading
from typing import List, Dict, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# ----------------------------------------------------------------------------
# Execution core
# ----------------------------------------------------------------------------

# Match a top-level ORDER BY so that comparison is order-sensitive only when the
# gold query itself asks for an ordering.
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)

# Default round precision for float cells to avoid spurious mismatches.
_FLOAT_NDIGITS = 4


class ExecConfig:
    def __init__(self, db_root: str, timeout: float, max_rows: int, num_workers: int):
        self.db_root = db_root
        self.timeout = timeout
        self.max_rows = max_rows
        self.num_workers = num_workers


def _db_path(db_root: str, db_id: str) -> str:
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        # inf/nan quebram o encoder JSON (dados reais do BIRD produzem overflow,
        # ex. POWER com expoente grande); sentinela string é determinística e
        # comparável dos dois lados do /compare.
        if not math.isfinite(value):
            return f"<{value!r}>"
        return round(value, _FLOAT_NDIGITS)
    if isinstance(value, bytes):
        # Blobs are not comparable in a meaningful way for EX; hash them.
        return ("<blob>", len(value))
    return value


def _normalize_row(row: Tuple) -> Tuple:
    return tuple(_normalize_cell(c) for c in row)


def execute_sql(
    config: ExecConfig,
    db_id: str,
    sql: str,
    timeout: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a single SQL statement against `db_id` in read-only mode.

    Returns a dict with keys:
      exec_ok (bool), error (str|None), columns (list[str]),
      rows (list[list]), n_rows (int), truncated (bool)
    """
    timeout = config.timeout if timeout is None else timeout
    max_rows = config.max_rows if max_rows is None else max_rows

    path = _db_path(config.db_root, db_id)
    if not os.path.exists(path):
        return {
            "exec_ok": False,
            "error": f"database not found: {db_id}",
            "columns": [],
            "rows": [],
            "n_rows": 0,
            "truncated": False,
        }

    conn = None
    timer = None
    try:
        # Read-only + immutable: no locking, safe for concurrent rollouts.
        conn = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")

        # Hard timeout: interrupt the connection from a watchdog timer.
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()

        cur = conn.cursor()
        cur.execute(sql)

        columns = [d[0] for d in cur.description] if cur.description else []
        # Fetch one extra row to detect truncation.
        fetched = cur.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows
        rows = [list(_normalize_row(tuple(r))) for r in fetched[:max_rows]]

        return {
            "exec_ok": True,
            "error": None,
            "columns": columns,
            "rows": rows,
            "n_rows": len(rows),
            "truncated": truncated,
        }
    except Exception as e:  # noqa: BLE001 - report any execution error to caller
        return {
            "exec_ok": False,
            "error": f"{type(e).__name__}: {e}",
            "columns": [],
            "rows": [],
            "n_rows": 0,
            "truncated": False,
        }
    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            conn.close()


def _result_multiset(rows: List[List], order_matters: bool):
    tuples = [tuple(r) for r in rows]
    if order_matters:
        return tuples
    # Order-agnostic multiset comparison.
    from collections import Counter

    return Counter(tuples)


def compare_results(
    config: ExecConfig,
    db_id: str,
    pred_sql: str,
    gold_sql: str,
    timeout: Optional[float] = None,
    order_matters: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execution-accuracy comparison between a predicted and a gold SQL.

    Order-sensitivity defaults to whether the gold query contains ORDER BY.
    """
    pred = execute_sql(config, db_id, pred_sql, timeout=timeout)
    gold = execute_sql(config, db_id, gold_sql, timeout=timeout)

    if order_matters is None:
        order_matters = bool(_ORDER_BY_RE.search(gold_sql or ""))

    match = False
    if pred["exec_ok"] and gold["exec_ok"]:
        # Truncated results cannot be safely compared as equal.
        if not pred["truncated"] and not gold["truncated"]:
            match = _result_multiset(pred["rows"], order_matters) == _result_multiset(
                gold["rows"], order_matters
            )

    return {
        "match": match,
        "order_matters": order_matters,
        "pred": pred,
        "gold": gold,
    }


def _format_sample_rows(columns: List[str], rows: List[Tuple], max_cell: int = 40) -> str:
    """Render a few sample rows as an OmniSQL/Arctic-style SQL comment block."""
    def _cell(v: Any) -> str:
        if v is None:
            return "NULL"
        s = str(v).replace("\n", " ")
        return s if len(s) <= max_cell else s[:max_cell] + "…"

    header = " | ".join(columns)
    lines = [" | ".join(_cell(c) for c in row) for row in rows]
    body = "\n".join([header] + lines)
    return f"/*\n{len(rows)} example rows:\n{body}\n*/"


def get_schema(
    config: ExecConfig, db_id: str, sample_rows: int = 3
) -> Dict[str, Any]:
    """Return the CREATE statements for a database, each followed by a few sample
    rows (OmniSQL/Arctic prompt format). Set sample_rows=0 for DDL only."""
    path = _db_path(config.db_root, db_id)
    if not os.path.exists(path):
        return {"exec_ok": False, "error": f"database not found: {db_id}", "schema": ""}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
        )
        tables = cur.fetchall()

        blocks = []
        for name, create_sql in tables:
            block = create_sql.strip()
            if sample_rows > 0:
                try:
                    rcur = conn.cursor()
                    # Quote the identifier to tolerate reserved words / odd names.
                    rcur.execute(f'SELECT * FROM "{name}" LIMIT {sample_rows}')
                    cols = [d[0] for d in rcur.description] if rcur.description else []
                    rows = [tuple(_normalize_row(tuple(r))) for r in rcur.fetchall()]
                    if rows:
                        block += "\n" + _format_sample_rows(cols, rows)
                except Exception:  # noqa: BLE001 - sampling is best-effort
                    pass
            blocks.append(block)

        return {"exec_ok": True, "error": None, "schema": "\n\n".join(blocks)}
    except Exception as e:  # noqa: BLE001
        return {"exec_ok": False, "error": f"{type(e).__name__}: {e}", "schema": ""}
    finally:
        if conn is not None:
            conn.close()


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------

class ExecItem(BaseModel):
    db_id: str
    sql: str


class ExecRequest(BaseModel):
    queries: List[ExecItem]
    timeout: Optional[float] = None
    max_rows: Optional[int] = None


class CompareItem(BaseModel):
    db_id: str
    pred_sql: str
    gold_sql: str


class CompareRequest(BaseModel):
    items: List[CompareItem]
    timeout: Optional[float] = None
    order_matters: Optional[bool] = None


class SchemaRequest(BaseModel):
    db_ids: List[str]


app = FastAPI()


@app.get("/health")
def health_endpoint():
    return {"status": "ok", "db_root": config.db_root}


@app.post("/execute")
def execute_endpoint(request: ExecRequest):
    """Batch-execute SQL. Input: {"queries": [{"db_id", "sql"}], "timeout"?, "max_rows"?}"""
    def _run(item: ExecItem):
        return execute_sql(
            config, item.db_id, item.sql,
            timeout=request.timeout, max_rows=request.max_rows,
        )

    results = list(executor.map(_run, request.queries))
    return {"result": results}


@app.post("/compare")
def compare_endpoint(request: CompareRequest):
    """Batch execution-accuracy comparison of predicted vs gold SQL."""
    def _run(item: CompareItem):
        return compare_results(
            config, item.db_id, item.pred_sql, item.gold_sql,
            timeout=request.timeout, order_matters=request.order_matters,
        )

    results = list(executor.map(_run, request.items))
    return {"result": results}


@app.post("/schema")
def schema_endpoint(request: SchemaRequest):
    """Return CREATE statements for the requested databases."""
    results = [get_schema(config, db_id) for db_id in request.db_ids]
    return {"result": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the local SQLite execution server.")
    parser.add_argument("--db_root", type=str, default="./spider/database",
                        help="Root dir with <db_id>/<db_id>.sqlite (Spider layout).")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Per-query hard timeout in seconds.")
    parser.add_argument("--max_rows", type=int, default=50,
                        help="Max rows returned/compared per query.")
    parser.add_argument("--num_workers", type=int, default=64,
                        help="Thread pool size for batch execution.")
    args = parser.parse_args()

    config = ExecConfig(
        db_root=args.db_root,
        timeout=args.timeout,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
    )
    executor = ThreadPoolExecutor(max_workers=args.num_workers)

    uvicorn.run(app, host="0.0.0.0", port=args.port, access_log=False)
