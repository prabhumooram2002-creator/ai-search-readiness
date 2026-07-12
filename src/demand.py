"""PHASE 4 — Query demand ingestion (v3 brief). No paid APIs — file imports.

- Parse a Google Keyword Planner CSV export (user downloads it free): keyword +
  avg monthly searches -> volume bucket -> query weight.
- Conversationalize: GKP keywords are terse Google-style; AI prompts are long
  and constraint-loaded. For each seed, the local LLM (json, temp 0.7)
  generates 2-3 conversational question variants tagged {seed, variant_type,
  volume_bucket, weight}.
- Optional People-Also-Ask paste file merged in.
- Volume weight orders reporting: dead ends on high-volume queries surface first.
"""
from __future__ import annotations

import csv
import io
import json as _json
import math
import re
from pathlib import Path
from typing import Optional

from . import providers
from .core.logging import get_logger

logger = get_logger(__name__)

# avg-monthly-searches -> bucket label (GKP-style ranges)
_BUCKETS = [(0, "0"), (1, "1-10"), (11, "10-100"), (101, "100-1K"),
            (1001, "1K-10K"), (10001, "10K-100K"), (100001, "100K+")]

CONVERSATIONALIZE_PROMPT = (
    'A user types this into Google: "{keyword}"\n'
    "Rewrite it as 2-3 natural, conversational questions someone would actually "
    "ask an AI assistant (longer, with context/constraints).\n"
    'Return ONLY JSON: {{"questions": ["...", "..."]}}')


def volume_bucket(searches: int) -> str:
    label = "0"
    for lo, name in _BUCKETS:
        if searches >= lo:
            label = name
    return label


def volume_weight(searches: int) -> float:
    """Log-scaled 0-1 weight; 100K searches ~= 1.0."""
    return round(min(1.0, math.log10(max(searches, 0) + 1) / 5.0), 4)


def _to_int(val: str) -> int:
    m = re.findall(r"\d[\d,]*", str(val))
    if not m:
        return 0
    # GKP ranges like "1K – 10K": take the low end conservatively
    return int(m[0].replace(",", ""))


def parse_gkp_csv(path: str | Path) -> list[dict]:
    """Parse a Google Keyword Planner export (CSV or TSV, UTF-8/UTF-16).

    GKP files carry 2-3 preamble lines before the real header row that holds
    'Keyword' and an 'Avg. monthly searches' column. Returns
    [{"keyword", "searches", "volume_bucket", "weight"}].
    """
    raw = Path(path).read_bytes()
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16-le")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16-be")
    else:
        text = raw.decode("utf-8-sig", errors="replace")

    lines = text.splitlines()
    # find the header row (contains 'keyword'); detect the delimiter from IT,
    # not from a preamble line that may have neither tab nor comma.
    header_idx = next((i for i, l in enumerate(lines)
                       if "keyword" in l.lower()), 0)
    header_line = lines[header_idx] if lines else ""
    delim = "\t" if "\t" in header_line else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])),
                            delimiter=delim)
    kw_col = vol_col = None
    for col in (reader.fieldnames or []):
        cl = col.lower().strip()
        if kw_col is None and cl == "keyword":
            kw_col = col
        if vol_col is None and "avg" in cl and "search" in cl:
            vol_col = col
    if kw_col is None:
        kw_col = (reader.fieldnames or ["Keyword"])[0]

    out, seen = [], set()
    for row in reader:
        kw = (row.get(kw_col) or "").strip()
        if not kw or kw.lower() in seen:
            continue
        seen.add(kw.lower())
        searches = _to_int(row.get(vol_col, 0)) if vol_col else 0
        out.append({"keyword": kw, "searches": searches,
                    "volume_bucket": volume_bucket(searches),
                    "weight": volume_weight(searches)})
    logger.info(f"[demand] parsed {len(out)} keywords from GKP export "
                f"(kw_col={kw_col!r}, vol_col={vol_col!r})")
    return out


def _json_obj(raw: str) -> dict:
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    try:
        return _json.loads(t.strip())
    except _json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}") + 1
        if 0 <= s < e:
            try:
                return _json.loads(t[s:e])
            except _json.JSONDecodeError:
                pass
    return {}


def conversationalize(seeds: list[dict], max_variants: int = 3,
                      trace=None) -> list[dict]:
    """Generate conversational question variants per seed keyword (local LLM,
    temp 0.7). Returns query records with seed/variant_type/bucket/weight."""
    step_cm = trace.start_step("conversationalize",
                               n_seeds=len(seeds)) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        queries: list[dict] = []
        for seed in seeds:
            kw = seed["keyword"]
            raw = providers.llm_complete(
                CONVERSATIONALIZE_PROMPT.format(keyword=kw), json=True,
                temperature=0.7,
                system="You rewrite keywords as conversational questions. "
                       "Output ONLY valid JSON.")
            variants = [str(q).strip() for q in _json_obj(raw).get("questions", [])
                        if str(q).strip()][:max_variants]
            for v in variants:
                queries.append({
                    "query": v, "seed": kw, "variant_type": "conversational",
                    "volume_bucket": seed["volume_bucket"],
                    "weight": seed["weight"]})
        logger.info(f"[demand] conversationalized {len(seeds)} seeds -> "
                    f"{len(queries)} query variants")
        if st is not None:
            st.outputs["n_queries"] = len(queries)
            st.scores["n_queries"] = float(len(queries))
        return queries
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)


def parse_paa(path: str | Path) -> list[dict]:
    """Parse a People-Also-Ask paste file (one question per line)."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    out = []
    for line in text.splitlines():
        q = line.strip().lstrip("-*0123456789. ").strip()
        if len(q) >= 8 and "?" in q or (len(q) >= 12 and q[0].isupper()):
            out.append({"query": q, "seed": None, "variant_type": "paa",
                        "volume_bucket": "unknown", "weight": 0.5})
    logger.info(f"[demand] parsed {len(out)} PAA questions")
    return out


def build_query_set(gkp_path: Optional[str] = None,
                    paa_path: Optional[str] = None,
                    max_variants: int = 3, trace=None) -> list[dict]:
    """Full ingestion: GKP -> conversationalized variants (+ optional PAA),
    ordered by weight (high-volume first)."""
    queries: list[dict] = []
    if gkp_path:
        queries += conversationalize(parse_gkp_csv(gkp_path),
                                     max_variants=max_variants, trace=trace)
    if paa_path:
        queries += parse_paa(paa_path)
    queries.sort(key=lambda q: -q["weight"])   # high-volume gaps surface first
    return queries
