# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Run: .venv/bin/python -m pytest tests/test_sql_utils.py -v

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from search.sql_utils import (  # noqa: E402
    complexity_level,
    is_deterministic,
    is_single_select,
    sql_template,
)


class TestComplexityLevel:
    def test_simple_select(self):
        assert complexity_level("SELECT name FROM singer WHERE age > 30") == 1

    def test_order_by_limit_still_simple(self):
        assert complexity_level("SELECT name FROM singer ORDER BY age LIMIT 3") == 1

    def test_one_join_is_moderate(self):
        sql = "SELECT s.name FROM singer s JOIN concert c ON s.id = c.singer_id"
        assert complexity_level(sql) == 2

    def test_aggregation_is_moderate(self):
        assert complexity_level("SELECT country, COUNT(*) FROM singer GROUP BY country") == 2
        assert complexity_level("SELECT AVG(age) FROM singer") == 2

    def test_two_joins_is_complex(self):
        sql = (
            "SELECT s.name FROM singer s JOIN concert c ON s.id = c.singer_id "
            "JOIN stadium t ON c.stadium_id = t.id"
        )
        assert complexity_level(sql) == 3

    def test_subquery_is_complex(self):
        sql = "SELECT name FROM singer WHERE age > (SELECT AVG(age) FROM singer)"
        assert complexity_level(sql) == 3

    def test_having_is_complex(self):
        sql = "SELECT country FROM singer GROUP BY country HAVING COUNT(*) > 2"
        assert complexity_level(sql) == 3

    def test_set_operation_is_complex(self):
        sql = "SELECT name FROM singer UNION SELECT name FROM band"
        assert complexity_level(sql) == 3

    def test_column_named_max_is_not_aggregation(self):
        # `max_speed` is a column, not the MAX( function.
        assert complexity_level("SELECT max_speed FROM cars") == 1


class TestIsDeterministic:
    def test_plain_select_ok(self):
        assert is_deterministic("SELECT name FROM singer WHERE age > 30")

    def test_limit_with_order_by_ok(self):
        assert is_deterministic("SELECT name FROM singer ORDER BY age LIMIT 3")

    def test_limit_without_order_by_rejected(self):
        assert not is_deterministic("SELECT name FROM singer LIMIT 3")

    def test_random_rejected(self):
        assert not is_deterministic("SELECT name FROM singer ORDER BY random() LIMIT 1")

    def test_now_functions_rejected(self):
        assert not is_deterministic("SELECT * FROM logs WHERE ts > datetime('now')")
        assert not is_deterministic("SELECT CURRENT_TIMESTAMP")

    def test_date_on_column_ok(self):
        # date() over a column is deterministic; only 'now' is not.
        assert is_deterministic("SELECT date(birthday) FROM singer")


class TestIsSingleSelect:
    def test_select_ok(self):
        assert is_single_select("SELECT 1")

    def test_with_cte_ok(self):
        assert is_single_select("WITH t AS (SELECT 1 AS x) SELECT x FROM t")

    def test_trailing_semicolon_ok(self):
        assert is_single_select("SELECT 1;")

    def test_multiple_statements_rejected(self):
        assert not is_single_select("SELECT 1; SELECT 2")

    def test_non_select_rejected(self):
        assert not is_single_select("DROP TABLE singer")
        assert not is_single_select("UPDATE singer SET age = 0")


class TestSqlTemplate:
    def test_masks_numbers_and_strings(self):
        a = sql_template("SELECT name FROM singer WHERE age > 30 AND country = 'USA'")
        b = sql_template("SELECT name FROM singer WHERE age > 50 AND country = 'Brazil'")
        assert a == b
        assert "[NUM]" in a and "[STR]" in a

    def test_different_structure_differs(self):
        a = sql_template("SELECT name FROM singer WHERE age > 30")
        b = sql_template("SELECT name FROM singer WHERE country = 'USA'")
        assert a != b

    def test_normalizes_case_and_whitespace(self):
        a = sql_template("select  name\nfrom SINGER")
        b = sql_template("SELECT name FROM singer")
        assert a == b

    def test_escaped_quote_inside_string(self):
        a = sql_template("SELECT * FROM t WHERE name = 'O''Brien'")
        assert "[STR]" in a and "brien" not in a
