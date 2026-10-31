"""Utility functions for formatting, result inspection, and evaluation helpers.
"""

from __future__ import annotations

import math
from typing import Any


def format_results_preview(columns: list[str], rows: list[list[Any]], max_rows: int = 5) -> str:
    """Formats SQL result rows as a compact string table for prompt previews."""
    if not rows:
        return "No rows returned (empty result set)."

    header = " | ".join(str(c) for c in columns)
    separator = "-+-".join("-" * len(str(c)) for c in columns)
    lines = [header, separator]

    for row in rows[:max_rows]:
        line = " | ".join(str(val) if val is not None else "NULL" for val in row)
        lines.append(line)

    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} additional rows truncated for brevity)")

    return "\n".join(lines)


def normalize_value(val: Any) -> Any:
    """Normalizes a single value for robust comparison (handling floats, casing, etc.)."""
    if val is None:
        return None
    if isinstance(val, (int, bool)):
        return val
    if isinstance(val, float):
        if math.isnan(val):
            return "NAN"
        return round(val, 2)
    # Try converting numeric strings to float
    s = str(val).strip()
    try:
        f = float(s)
        return round(f, 2)
    except ValueError:
        return s.lower()


def compare_result_sets(
    pred_columns: list[str],
    pred_rows: list[list[Any]],
    gold_columns: list[str],
    gold_rows: list[list[Any]],
    ordered: bool = False,
) -> bool:
    """Compares generated query results against gold query results.

    Normalizes floats, casing, and row order (multiset equality unless ordered=True).
    """
    if len(pred_rows) != len(gold_rows):
        return False

    if not pred_rows and not gold_rows:
        return True

    # Normalize each row into a tuple of normalized values
    norm_pred = [tuple(normalize_value(v) for v in row) for row in pred_rows]
    norm_gold = [tuple(normalize_value(v) for v in row) for row in gold_rows]

    if ordered:
        return norm_pred == norm_gold

    # Order-insensitive multiset comparison using sorted tuples
    try:
        return sorted(norm_pred, key=str) == sorted(norm_gold, key=str)
    except Exception:
        # Fallback to counts comparison
        from collections import Counter
        return Counter(str(r) for r in norm_pred) == Counter(str(r) for r in norm_gold)

