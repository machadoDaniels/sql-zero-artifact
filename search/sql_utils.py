# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Lightweight SQL analysis utilities for the Dr. Zero -> Text-to-SQL adaptation.
# Shared by the baseline eval, the gen_data filters and the challenger reward.
#
# Regex-based on purpose: no parser dependency, robust enough for the
# OmniSQL-style filtering this project needs (see DECISOES_ARQUITETURA.md §2.1).

import re

# Structural features -----------------------------------------------------

_JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bgroup\s+by\b", re.IGNORECASE)
_HAVING_RE = re.compile(r"\bhaving\b", re.IGNORECASE)
_AGG_FUNC_RE = re.compile(r"\b(count|avg|sum|max|min)\s*\(", re.IGNORECASE)
_SELECT_RE = re.compile(r"\bselect\b", re.IGNORECASE)
_SET_OP_RE = re.compile(r"\b(union|intersect|except)\b", re.IGNORECASE)

_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)
_NON_DETERMINISTIC_RE = re.compile(
    r"\brandom\s*\(|\bcurrent_(time|date|timestamp)\b|\b(date|time|datetime|julianday|strftime)\s*\(\s*['\"]now['\"]",
    re.IGNORECASE,
)

# Literals: single-quoted strings first, then standalone numbers.
_STRING_LITERAL_RE = re.compile(r"'(?:[^']|'')*'")
_NUMBER_LITERAL_RE = re.compile(r"\b\d+(\.\d+)?\b")


def complexity_level(sql: str) -> int:
    """Map a SQL query to the 3-level complexity rubric.

    3 (complex):  subquery, HAVING, set operation, or >= 2 JOINs.
    2 (moderate): exactly 1 JOIN, or any aggregation (GROUP BY / agg function).
    1 (simple):   single table, no join/aggregation/subquery.
    """
    n_joins = len(_JOIN_RE.findall(sql))
    has_subquery = len(_SELECT_RE.findall(sql)) > 1
    has_aggregation = bool(_GROUP_BY_RE.search(sql) or _AGG_FUNC_RE.search(sql))

    if has_subquery or _HAVING_RE.search(sql) or _SET_OP_RE.search(sql) or n_joins >= 2:
        return 3
    if n_joins == 1 or has_aggregation:
        return 2
    return 1


def is_deterministic(sql: str) -> bool:
    """Reject SQL whose result-set is not well-defined (Absolute Zero-style filter).

    Non-deterministic: LIMIT without ORDER BY (undefined row subset), random(),
    and current date/time functions.
    """
    if _LIMIT_RE.search(sql) and not _ORDER_BY_RE.search(sql):
        return False
    if _NON_DETERMINISTIC_RE.search(sql):
        return False
    return True


def is_single_select(sql: str) -> bool:
    """Accept exactly one SELECT statement (optionally starting with a CTE)."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return False
    return bool(re.match(r"^\s*(with\b.*?)?\s*select\b", stripped, re.IGNORECASE | re.DOTALL))


def sql_template(sql: str) -> str:
    """Mask literals and normalize whitespace/case, for template-level dedup
    (OmniSQL): `WHERE age > 30` and `WHERE age > 50` share one template."""
    template = " ".join(sql.lower().split())
    template = _STRING_LITERAL_RE.sub("[STR]", template)
    template = _NUMBER_LITERAL_RE.sub("[NUM]", template)
    return template
