"""Shared helpers across all 19 Phase 12 section modules — number formatting,
cosine similarity, and the "excluded metric" convention (print WHY a number
isn't available rather than faking it, per the brief's hard rules)."""
from __future__ import annotations

import html as _html
import math
import re as _re

_SLUG_RE = _re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Same scheme as src/explain.py's _slug (kept in lockstep) -- used here
    only to key ref_index entries so a recommendation's 'entity:<slug>'
    finding_id can resolve back to that entity's page, not just its raw id."""
    return _SLUG_RE.sub("-", str(text).lower()).strip("-")[:40] or "fix"


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


def build_ref_index(chunks: list[dict], entities: list[dict] | None = None) -> dict[str, dict]:
    """Maps chunk_id/entity_id -> {url, snippet} so the HTML renderer can turn
    every bare id a section emits into a link + snippet instead of printing
    the opaque id itself (the brief's hard rule: bare chunk/entity ids in the
    rendered report are a build failure)."""
    idx: dict[str, dict] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid is None:
            continue
        content = (c.get("content") or "").strip().replace("\n", " ")
        idx[cid] = {"url": c.get("url"), "snippet": content[:160]}
    for e in (entities or []):
        eid = e.get("id")
        if eid is None:
            continue
        url = next((m.get("url") for m in e.get("mentions", []) if m.get("url")), None)
        ref = {"url": url, "snippet": e.get("name", "")}
        idx[eid] = ref
        name = e.get("name")
        if name:
            idx.setdefault(_slug(name), ref)
    return idx


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
