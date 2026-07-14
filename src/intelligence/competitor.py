"""Section 15 — Competitor Knowledge Graph [BACKLOG 6 — build here]

Runs Layers 0-2 on competitor domains with a SEPARATE Kuzu DB + Chroma
collection per domain (never merged with the site's own KG), then compares.

Actually crawling a competitor is expensive (a full Layers 0-2 run per
domain) and needs a real competitor URL the user provides (CLI/config) --
this module provides the orchestration + comparison logic; running it
against a live competitor is a separate, explicit invocation (see
run_audit.py's `competitor` subcommand), not part of the default audit.
"""
from __future__ import annotations

from pathlib import Path

from .graph_stats import page_pagerank


def competitor_kg_paths(domain: str, base_dir: Path) -> dict:
    """Per-domain, never-merged store paths — mirrors the site's own
    data/kg.kuzu / data/chromadb layout, namespaced under data/competitors/."""
    slug = domain.replace("https://", "").replace("http://", "").rstrip("/").replace("/", "_")
    root = base_dir / "data" / "competitors" / slug
    return {"slug": slug, "root": root,
           "kg_path": str(root / "kg.kuzu"), "chroma_path": str(root / "chromadb")}


def run_competitor_index(domain: str, base_dir: Path, max_pages: int = 60) -> dict:
    """Runs the same build_index() pipeline (Layers 0-2) against a
    competitor domain. Import is deferred to avoid a hard dependency for
    callers that only need the comparison functions below (e.g. tests) to
    import cleanly.

    KNOWN GAP (documented, not silently papered over): build_index() already
    accepts a kg_path override, so the KG is properly namespaced per
    domain -- but its internal VectorStore() call has no matching
    collection_name override, so a competitor run currently reuses the
    SAME Chroma collection as the site's own index and would overwrite it.
    Wiring this needs a small, low-risk addition to build_index() (thread
    a collection_name through to VectorStore(collection_name=...)) before
    this function is safe to call against a real competitor domain. Not
    done in this pass since no real competitor URL was provided this
    session -- see BUILD_LOG's Phase 12 section 15 note.
    """
    from run_audit import build_index  # local import: avoid a circular/heavy import at module load

    paths = competitor_kg_paths(domain, base_dir)
    paths["root"].mkdir(parents=True, exist_ok=True)
    ctx = build_index(domain, max_pages, fresh=True, claims_cap=25,
                      kg_path=paths["kg_path"])
    return {"domain": domain, "paths": paths, "ctx": ctx}


def comparison_table(site_stats: dict, competitor_stats: list[dict]) -> list[dict]:
    """site_stats / each of competitor_stats: {"domain","pages","chunks",
    "entities","relations","claims","evidenced_claim_share","n_faq_pages",
    "n_trust_pages","n_topics","invisibility_score"} + l0_pages for PageRank
    concentration."""
    rows = [{"domain": "this site", **site_stats}]
    for c in competitor_stats:
        rows.append({"domain": c["domain"], **c})
    for row, stats in zip(rows, [site_stats] + competitor_stats):
        pr = page_pagerank(stats.get("l0_pages", []))
        top_share = sum(sorted(pr.values(), reverse=True)[:3]) if pr else None
        row["pagerank_concentration_top3"] = round(top_share, 4) if top_share is not None else None
        row.pop("l0_pages", None)
    return rows


def per_query_competitor_diff(our_trace, competitor_trace) -> dict:
    """Their winning chunk vs ours, for the same query."""
    return {
        "query": our_trace.query,
        "our_winning_chunks": [r["chunk_id"] for r in our_trace.reranked[:1]],
        "our_confidence": our_trace.confidence,
        "their_winning_chunks": [r["chunk_id"] for r in competitor_trace.reranked[:1]],
        "their_confidence": competitor_trace.confidence,
        "we_win": (our_trace.confidence or 0) >= (competitor_trace.confidence or 0),
    }
