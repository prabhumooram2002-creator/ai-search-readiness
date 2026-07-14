"""Section 7 — Backlink Intelligence [BACKLOG 10b, degraded until built]

Common Crawl enrichment (--enrich) is backlog — ExternalDomain/ExternalPage
stay empty without it. Internal PageRank is fully local and always
included regardless.
"""
from __future__ import annotations

from .core_utils import excluded
from .graph_stats import orphan_pages, page_pagerank

DEGRADED_NOTE = ("external backlink data requires Common Crawl enrichment "
                 "(--enrich, BACKLOG 10b) — not run this session; showing "
                 "internal PageRank only")

EXCLUDED_WITHOUT_PROVIDER = {
    "authority_scores": excluded("needs a paid backlink/authority provider"),
    "toxic_links": excluded("needs a paid backlink/authority provider"),
    "link_velocity": excluded("needs historical crawl snapshots from a paid provider"),
    "lost_new_links": excluded("needs historical crawl snapshots from a paid provider"),
}


def backlink_report(kg, l0_pages: list[dict]) -> dict:
    pr = page_pagerank(l0_pages)
    top_pages = sorted(pr.items(), key=lambda kv: -kv[1])[:20]

    res = kg._exec("MATCH (d:ExternalDomain) RETURN d.host LIMIT 1")
    enriched = res.has_next()

    external = []
    if enriched:
        res2 = kg._exec(
            "MATCH (p:Page)-[r:ExternalLinksTo]->(ep:ExternalPage) "
            "RETURN p.url, ep.url, r.anchor_text, r.topical_relevance")
        while res2.has_next():
            src, dst, anchor, rel = res2.get_next()
            external.append({"from": src, "to": dst, "anchor_text": anchor,
                            "topical_relevance": round(rel, 4) if rel is not None else None})

    return {
        "enriched": enriched,
        "degraded_note": None if enriched else DEGRADED_NOTE,
        "internal_pagerank_top_pages": [{"url": u, "pagerank": round(p, 5)} for u, p in top_pages],
        "orphan_pages": orphan_pages(l0_pages),
        "external_links": external,
        "excluded_metrics": EXCLUDED_WITHOUT_PROVIDER,
    }
