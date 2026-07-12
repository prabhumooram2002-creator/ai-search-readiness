"""Layer 1 steps 11-14 — Trust / Citation / Source Authority / Temporal
Freshness (CLAUDE.md; kept — heuristic, logged).

Explicit 0-1 scoring functions. Every formula is logged and every score is
labeled "heuristic" — never "AI trust level". Every Page gets all four; chunks
inherit their page's signals.

Honest limits (also logged per page):
- citation counts only INTERNAL inbound links until step 10b (Common Crawl
  external graph) is built;
- freshness defaults to 0.5 when no lastmod/date evidence exists.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .core.logging import get_logger

logger = get_logger(__name__)

FORMULAS = {
    "trust": "heuristic: 0.35*has_schema + 0.25*has_byline + 0.15*has_sameas "
             "+ 0.25*(1 - contradiction_rate)",
    "citation": "heuristic: 1 - exp(-internal_inbound/3)  [external inbound "
                "pending step 10b]",
    "source_authority": "heuristic: 0.6*citation + 0.4*trust  [referring-domain "
                        "proxy pending step 10b]",
    "temporal_freshness": "heuristic: exp(-age_days/365) from lastmod/dates; "
                          "0.5 when no date evidence",
}

_BYLINE = re.compile(r"\bby\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)+|\bauthor\b", re.I)
_DATE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def score_pages(
    pages: list[dict],
    contradictions_by_url: dict[str, float] | None = None,
    now: datetime | None = None,
) -> dict[str, dict]:
    """pages: [{"url", "markdown"/"content", "schema_jsonld"?: list,
                "internal_inbound"?: int, "lastmod"?: iso str}]
    Returns {url: {trust, citation, source_authority, temporal_freshness,
                   notes[]}} — all heuristic, formulas in FORMULAS."""
    now = now or datetime.now(timezone.utc)
    out: dict[str, dict] = {}
    for p in pages:
        url = p["url"]
        text = p.get("markdown") or p.get("content") or ""
        notes: list[str] = []

        # 11 Trust
        has_schema = 1.0 if p.get("schema_jsonld") else 0.0
        has_byline = 1.0 if _BYLINE.search(text[:4000]) else 0.0
        has_sameas = 1.0 if "sameas" in str(p.get("schema_jsonld", "")).lower() else 0.0
        contra = float((contradictions_by_url or {}).get(url, 0.0))
        trust = 0.35 * has_schema + 0.25 * has_byline + 0.15 * has_sameas \
            + 0.25 * (1.0 - min(1.0, contra))

        # 12 Citation (internal only until 10b)
        inbound = int(p.get("internal_inbound", 0))
        citation = 1.0 - math.exp(-inbound / 3.0)
        notes.append("citation counts internal inbound only (step 10b pending)")

        # 13 Source authority
        authority = 0.6 * citation + 0.4 * trust
        notes.append("authority lacks referring-domain proxy (step 10b pending)")

        # 14 Temporal freshness
        date_src = p.get("lastmod")
        if not date_src:
            m = _DATE.search(text)
            date_src = m.group(0) if m else None
        if date_src:
            try:
                dt = datetime.fromisoformat(str(date_src)[:10]).replace(
                    tzinfo=timezone.utc)
                age_days = max(0.0, (now - dt).days)
                freshness = math.exp(-age_days / 365.0)
            except ValueError:
                freshness = 0.5
                notes.append("unparseable date evidence -> freshness 0.5")
        else:
            freshness = 0.5
            notes.append("no date evidence -> freshness 0.5 (neutral)")

        out[url] = {
            "trust": round(trust, 4),
            "citation": round(citation, 4),
            "source_authority": round(authority, 4),
            "temporal_freshness": round(freshness, 4),
            "notes": notes,
        }
    logger.info(f"Signals (heuristic) scored for {len(out)} pages; "
                f"formulas: {list(FORMULAS)}")
    return out
