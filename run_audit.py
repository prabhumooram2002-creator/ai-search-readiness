#!/usr/bin/env python
"""run_audit.py — plain synchronous, fully-traced audit pipeline (CLAUDE.md).

Layers 1->3, fully local, ZERO API keys:

  index:  crawl -> chunk -> BGE-M3 embed -> Chroma store -> GLiNER entities ->
          GLiREL relations -> qwen claims -> NLI evidence -> HDBSCAN topics ->
          heuristic signals -> Kùzu knowledge graph
  query:  Layer 2 simulator (intent -> expansion -> hybrid retrieve -> rerank
          -> graph traversal -> evidence validation -> contradiction ->
          confidence -> synthesis -> citations), one Trace per query
  report: Layer 3 explainability (pure introspection over the Trace)

Queries are never hardcoded — only --queries (file) or --query (repeatable).

Usage:
    python run_audit.py --url https://example.com --queries queries.txt
    python run_audit.py --url https://example.com --query "..." --max-pages 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.core.logging import get_logger
from src.trace import Trace
from src import providers

logger = get_logger("run_audit")


def load_queries(path: str | None, inline: list[str] | None) -> list[str]:
    queries: list[str] = list(inline or [])
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Queries file not found: {path}")
        suffix = p.suffix.lower()
        if suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                queries += [q if isinstance(q, str) else q.get("query", "") for q in data]
            elif isinstance(data, dict) and "queries" in data:
                queries += list(data["queries"])
        elif suffix == ".csv":
            import csv
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        queries += [row[0].strip()]
        else:
            queries += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    queries = [q for q in (q.strip() for q in queries) if q]
    if not queries:
        raise ValueError("No queries provided. Pass --queries <file> or one or "
                         "more --query. There are no hardcoded brand defaults.")
    return queries


def _unload_extraction_models() -> None:
    """Free GLiNER/GLiREL/spaCy before Layer 2 — the full local model set does
    not fit in RAM at once on this class of machine (see BUILD_LOG OOM)."""
    from src import ner, relations
    ner._gliner_model = None
    ner._spacy_nlp = None
    relations._glirel_model = None
    gc.collect()


def build_index(url: str, max_pages: int | None, fresh: bool,
                claims_cap: int, kg_path: str | None = None) -> dict:
    """Layer 1: crawl -> ... -> knowledge graph. Returns site-level context."""
    from src.crawl import crawl_site_sync
    from src.chunk import chunk_pages
    from src.embed import embed_chunks_sync
    from src.vector import VectorStore
    from src.ner import extract_entities, crosscheck_with_spacy
    from src.relations import extract_relations
    from src.claims import extract_claims
    from src.evidence import extract_evidence
    from src.topics import cluster_topics
    from src.signals import score_pages
    from src.kg import KnowledgeGraph

    if max_pages is not None:
        from src.core import config as _cfg
        _cfg.CFG.setdefault("crawl", {})["max_pages"] = max_pages

    site_trace = Trace(query="__site_index__")

    logger.info(f"[index] crawling {url}")
    crawl_results = crawl_site_sync(url)
    ok = [r for r in crawl_results if getattr(r, "success", False)]
    if not ok:
        raise RuntimeError(f"No pages could be crawled from {url}")
    logger.info(f"[index] {len(ok)}/{len(crawl_results)} pages crawled")

    chunk_objs = chunk_pages(crawl_results)
    chunks = [{"content": c.content, "chunk_id": c.chunk_id, "url": c.url}
              for c in chunk_objs]
    logger.info(f"[index] {len(chunks)} chunks")

    # step 9 — embeddings: Chroma + (below) Kùzu Chunk nodes
    vs = VectorStore()
    if fresh:
        vs.reset()
    embeddings = embed_chunks_sync([c["content"] for c in chunks], use_cache=True)
    vs.add_chunks(chunk_objs, embeddings)
    chroma_parity = vs.count() == len(chunks)
    logger.info(f"[index] vectors: chroma={vs.count()} chunks={len(chunks)} "
                f"parity={'OK' if chroma_parity else 'MISMATCH'}")

    # steps 3-4 — GLiNER entities + GLiREL relations (local zero-shot)
    entities = extract_entities(chunks, trace=site_trace)
    xcheck = crosscheck_with_spacy(chunks[:20], entities)
    relationships = extract_relations(chunks, entities, trace=site_trace)

    # steps 5-6 — claims (local LLM; capped for CPU runtime) + NLI evidence
    claim_chunks = chunks[:claims_cap]
    if len(chunks) > claims_cap:
        logger.warning(f"[index] claims capped to first {claims_cap}/{len(chunks)} "
                       f"chunks (CPU runtime); raise --claims-cap to widen")
    claims = extract_claims(claim_chunks, trace=site_trace)
    evidence = extract_evidence(claim_chunks, claims, trace=site_trace)

    # step 7 — topics
    topics = cluster_topics(chunks, trace=site_trace)

    # steps 11-14 — heuristic signals (formulas logged, labeled heuristic)
    inbound: dict[str, int] = {}
    for r in ok:
        for dest in getattr(r, "internal_links", []) or []:
            inbound[dest] = inbound.get(dest, 0) + 1
    pages = [{"url": r.url, "title": r.title, "content": r.content,
              "internal_inbound": inbound.get(r.url, 0),
              "internal_links": [{"anchor_text": "", "dest_url": d}
                                 for d in (getattr(r, "internal_links", []) or [])]}
             for r in ok]
    page_signals = score_pages(pages)

    # step 8 (+9) — knowledge graph with embeddings on Chunk nodes
    kg = KnowledgeGraph(kg_path)
    kg_stats = kg.build(chunks, entities, relationships, claims, evidence,
                        topics, pages=pages, embeddings=embeddings,
                        trace=site_trace)

    _unload_extraction_models()

    return {
        "site_report": {
            "url": url, "pages_crawled": len(ok), "pages_total": len(crawl_results),
            "chunks": len(chunks), "vector_count": vs.count(),
            "chroma_kuzu_parity": chroma_parity,
            "embed_dim": len(embeddings[0]) if embeddings else None,
            "n_entities": len(entities), "n_relationships": len(relationships),
            "n_claims": len(claims), "n_topics": len(topics),
            "ner_crosscheck_mean_disagreement": xcheck["mean_disagreement"],
            "kg": kg_stats,
        },
        "chunks": chunks, "embeddings": embeddings, "kg": kg,
        "page_signals": page_signals, "site_trace": site_trace,
    }


def write_reports(ctx: dict, traces: list[Trace], explanations: list[dict],
                  out_dir: Path) -> Path:
    from src.explain import render_markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "site": ctx["site_report"],
        "provider": providers.provider_info(),
        "site_trace": ctx["site_trace"].to_dict(),
        "queries": [t.to_dict() for t in traces],
        "explanations": explanations,
    }
    json_path = out_dir / "audit_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")

    sr = ctx["site_report"]
    lines = [f"# AI-Search Readiness Audit — {sr['url']}", "",
             f"- Pages: {sr['pages_crawled']}/{sr['pages_total']}  "
             f"Chunks: {sr['chunks']}  Embed dim: {sr['embed_dim']}",
             f"- Entities: {sr['n_entities']}  Relations: {sr['n_relationships']}  "
             f"Claims: {sr['n_claims']}  Topics: {sr['n_topics']}",
             f"- KG: {sr['kg']['nodes']} | orphan_chunks={sr['kg']['orphan_chunks']} "
             f"claims_without_support={sr['kg']['claims_without_support']}",
             f"- Provider: {providers.provider_info()['llm_provider_effective']}"
             f" / {providers.provider_info()['embed_provider_effective']} (zero keys required)",
             ""]
    for rows in explanations:
        lines.append(render_markdown(rows))
        lines.append("")
    md_path = out_dir / "audit_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def run(url: str, queries: list[str], out_dir: Path, max_pages: int | None,
        fresh: bool, claims_cap: int = 25) -> dict:
    from src.simulator import run_query
    from src.explain import explain_query

    ctx = build_index(url, max_pages, fresh, claims_cap)
    traces, explanations = [], []
    for q in queries:
        t = run_query(q, ctx["chunks"], ctx["embeddings"], kg=ctx["kg"],
                      page_signals=ctx["page_signals"])
        traces.append(t)
        explanations.append(explain_query(t, kg=ctx["kg"]))
    json_path = write_reports(ctx, traces, explanations, out_dir)
    logger.info(f"[report] wrote {json_path}")
    return {"site_report": ctx["site_report"], "traces": traces,
            "explanations": explanations, "report_path": str(json_path)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Traced, local-first AI-search readiness audit (Layers 1-3).")
    ap.add_argument("--url", required=True, help="Root URL to audit")
    ap.add_argument("--queries", help="Path to a queries file (.txt/.csv/.json)")
    ap.add_argument("--query", action="append", help="Inline query (repeatable)")
    ap.add_argument("--max-pages", type=int, default=None, help="Cap pages crawled")
    ap.add_argument("--claims-cap", type=int, default=25,
                    help="Max chunks to run LLM claim extraction on (CPU cost)")
    ap.add_argument("--out", default="data/reports", help="Report output directory")
    ap.add_argument("--no-fresh", action="store_true",
                    help="Do not reset the vector store before indexing")
    args = ap.parse_args()

    queries = load_queries(args.queries, args.query)
    logger.info(f"Auditing {args.url} with {len(queries)} queries "
                f"(LLM={providers.LLM_PROVIDER}:{providers.LLM_MODEL}, "
                f"EMBED={providers.EMBED_PROVIDER}:{providers.EMBED_MODEL})")
    result = run(args.url, queries, Path(args.out), args.max_pages,
                 fresh=not args.no_fresh, claims_cap=args.claims_cap)
    print(f"\nReport written to {result['report_path']}")


if __name__ == "__main__":
    main()
