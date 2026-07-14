"""Shared helpers across all 19 Phase 12 section modules — number formatting,
cosine similarity, and the "excluded metric" convention (print WHY a number
isn't available rather than faking it, per the brief's hard rules)."""
from __future__ import annotations

import html as _html
import math


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def score2(x) -> str:
    """Scores rounded to 2 decimals, per the brief's hard rules."""
    return "—" if x is None else f"{round(x, 2):g}"


def esc(text) -> str:
    return _html.escape(str(text), quote=True)


def excluded(reason: str) -> dict:
    """The brief's 'print the exclusion reason' convention for any metric
    that needs a paid API / data this pipeline doesn't have."""
    return {"available": False, "reason": reason}


def collapse_identical(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """'Repeated identical rows collapse into counts' (brief hard rule).
    Groups rows whose key_fields match, adding a 'count' field."""
    groups: dict[tuple, dict] = {}
    order: list[tuple] = []
    for r in rows:
        key = tuple(r.get(f) for f in key_fields)
        if key not in groups:
            groups[key] = dict(r)
            groups[key]["count"] = 1
            order.append(key)
        else:
            groups[key]["count"] += 1
    return [groups[k] for k in order]
