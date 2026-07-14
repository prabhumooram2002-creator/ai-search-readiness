"""Section 6 — Keyword Intelligence [IMPORT-BASED — no paid APIs]

GKP CSV (Phase 4, reused via src.demand.parse_gkp_csv) + NEW: Google Search
Console CSV import (free for owned/verified sites). Everything else is
derived with zero external calls; anything needing a paid provider prints
its exclusion reason instead of a fake number.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .core_utils import cosine, excluded

EXCLUDED_METRICS = {
    "difficulty": excluded("keyword difficulty needs a paid rank-data provider"),
    "cpc": excluded("CPC needs a paid ads-data provider"),
    "competitor_keyword_gaps": excluded("needs a paid rank-data provider or a "
                                       "competitor KG (see Section 15)"),
}


def parse_gsc_csv(path: str | Path) -> list[dict]:
    """Google Search Console 'Queries' export: Query/Clicks/Impressions/
    CTR/Position columns (export naming varies slightly by locale — matched
    case-insensitively). Row also carries the ranking URL only if the
    export includes a Page column (the combined Queries+Pages report)."""
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    q_col = col("query", "queries", "top queries")
    clicks_col = col("clicks")
    impr_col = col("impressions")
    ctr_col = col("ctr")
    pos_col = col("position", "avg. position", "average position")
    page_col = col("page", "top pages", "landing page")
    if not q_col:
        return []

    def _num(v, pct=False):
        if not v:
            return 0.0
        s = str(v).replace(",", "").replace("%", "").strip()
        try:
            n = float(s)
            return n / 100 if pct and "%" in str(v) else n
        except ValueError:
            return 0.0

    out = []
    for row in reader:
        q = (row.get(q_col) or "").strip()
        if not q:
            continue
        out.append({
            "query": q,
            "clicks": _num(row.get(clicks_col)),
            "impressions": _num(row.get(impr_col)),
            "ctr": _num(row.get(ctr_col), pct=True),
            "avg_position": _num(row.get(pos_col)),
            "ranking_url": (row.get(page_col) or "").strip() or None,
        })
    return out


def keyword_topic_mapping(keywords: list[dict], topic_centroids: dict,
                          embed_fn, threshold: float = 0.3) -> list[dict]:
    """Embed each keyword, match to its best topic by cosine."""
    if not keywords or not topic_centroids:
        return []
    vecs = embed_fn([k["query"][:512] for k in keywords])
    out = []
    for k, v in zip(keywords, vecs):
        best_topic, best_sim = None, 0.0
        for tid, centroid in topic_centroids.items():
            sim = cosine(v, centroid)
            if sim > best_sim:
                best_topic, best_sim = tid, sim
        out.append({**k, "best_topic": best_topic if best_sim >= threshold else None,
                   "topic_similarity": round(best_sim, 4)})
    return out


def keyword_cannibalization(gsc_rows: list[dict]) -> list[dict]:
    """Two+ distinct ranking URLs competing for the same query (GSC-reported)."""
    by_query: dict[str, set] = {}
    for r in gsc_rows:
        if r.get("ranking_url"):
            by_query.setdefault(r["query"], set()).add(r["ranking_url"])
    return [{"query": q, "competing_urls": sorted(urls)}
           for q, urls in by_query.items() if len(urls) > 1]


def missing_and_long_tail(gkp_keywords: list[dict], best_retrieval_by_keyword: dict,
                         retrieval_floor: float = 0.25) -> dict:
    """missing_keywords: imported queries whose best retrieval score is
    below floor. long_tail_opportunities: conversational variants with no
    covering chunk (same floor, from the same retrieval data)."""
    missing, opportunities = [], []
    for kw in gkp_keywords:
        score = best_retrieval_by_keyword.get(kw.get("keyword") or kw.get("query"), 0.0)
        if score < retrieval_floor:
            entry = {**kw, "best_retrieval_score": round(score, 4)}
            (opportunities if kw.get("variant_type") == "conversational" else missing).append(entry)
    return {"missing_keywords": missing, "long_tail_opportunities": opportunities}
