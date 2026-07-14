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


def load_query_weights(path: str | None, gkp_path: str | None) -> dict[str, float]:
    """PHASE 10 S2 prioritization needs real query-volume weight (CLAUDE.md:
    'Volume bucket becomes a query weight in reporting'). Separate from
    load_queries() (which stays list[str] — untouched, zero risk to its
    existing callers/tests) so this is purely additive.

    - A Phase-4 import-queries JSON file (``{query, weight, ...}`` records)
      carries its own weights straight through.
    - A fresh --gkp CSV (not yet conversationalized) is matched to the
      literal --queries text by substring, best-effort.
    - Anything else defaults to uniform weight 1.0 — honest, not zero;
      CLAUDE.md never says un-weighted queries should be ignored, just that
      real demand data should surface gaps first when it exists.
    """
    weights: dict[str, float] = {}
    if path:
        p = Path(path)
        if p.suffix.lower() == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for q in data:
                        if isinstance(q, dict) and q.get("query"):
                            weights[q["query"]] = float(q.get("weight", 1.0))
            except (json.JSONDecodeError, OSError):
                pass
    if gkp_path:
        from src.demand import parse_gkp_csv
        for kw in parse_gkp_csv(gkp_path):
            weights.setdefault(kw["keyword"], kw["weight"])
            for q in list(weights):
                if kw["keyword"].lower() in q.lower():
                    weights[q] = max(weights.get(q, 0.0), kw["weight"])
    return weights


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

    from src.layer0 import run_layer0, fetch_sitemap

    site_trace = Trace(query="__site_index__")

    # LAYER 0: sitemap-FIRST seeding — sitemap URLs join the crawl queue
    sitemap = fetch_sitemap(url)
    logger.info(f"[index] crawling {url} "
                f"(sitemap seeded: {len(sitemap)} URLs)")
    crawl_results = crawl_site_sync(url, seed_urls=[s["url"] for s in sitemap])
    ok = [r for r in crawl_results if getattr(r, "success", False)]
    if not ok:
        raise RuntimeError(f"No pages could be crawled from {url}")
    logger.info(f"[index] {len(ok)}/{len(crawl_results)} pages crawled")

    # LAYER 0 enrichment: JSON-LD/headings/links, 4-bot re-fetch ->
    # access_gaps, robots cross-check, PDFs
    l0 = run_layer0(url, crawl_results, trace=site_trace, sitemap=sitemap)

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

    # steps 11-14 — heuristic signals, now fed real Layer-0 enrichment
    # (JSON-LD presence, anchored internal links)
    inbound: dict[str, int] = {}
    for p in l0["pages"]:
        for link in p["internal_links"]:
            inbound[link["dest_url"]] = inbound.get(link["dest_url"], 0) + 1
    pages = [{"url": p["url"], "title": p["title"], "content": p["content"],
              "schema_jsonld": p["schema_jsonld"],
              "internal_inbound": inbound.get(p["url"], 0),
              "internal_links": p["internal_links"]}
             for p in l0["pages"]]
    page_signals = score_pages(pages)

    # step 8 (+9) — knowledge graph with embeddings on Chunk nodes
    kg = KnowledgeGraph(kg_path)
    kg_stats = kg.build(chunks, entities, relationships, claims, evidence,
                        topics, pages=pages, embeddings=embeddings,
                        trace=site_trace)

    # Phase 2 — capture baseline state so the next run can be incremental
    from src.incremental import IncrementalState, baseline_capture
    _state = IncrementalState()
    baseline_capture(_state, crawl_results, chunk_objs, l0_pages=l0["pages"])
    _state.close()

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
            "sitemap": l0["sitemap_stats"], "n_pdfs": len(l0["pdfs"]),
            "robots_conflicts": sum(len(p["robots_conflict"]) for p in l0["pages"]),
        },
        "chunks": chunks, "embeddings": embeddings, "kg": kg,
        "page_signals": page_signals, "site_trace": site_trace,
        "l0_pages": l0["pages"],
    }


def build_index_incremental(url: str, claims_cap: int) -> dict:
    """Phase 2: incremental index update (4-level skip chain), then rebuild
    the query context from stored state. Falls back to full when no baseline."""
    import json as _json
    from src.incremental import IncrementalState, incremental_update
    from src.embed import embed_chunks_sync
    from src.vector import VectorStore
    from src.ner import extract_entities
    from src.relations import extract_relations
    from src.claims import extract_claims
    from src.evidence import extract_evidence
    from src.topics import cluster_topics
    from src.signals import score_pages
    from src.kg import KnowledgeGraph
    from src.layer0 import invisibility_score  # noqa: F401 (used by run())

    state = IncrementalState()
    if not state.has_baseline():
        logger.warning("[incremental] no baseline — running a FULL build instead")
        state.close()
        return build_index(url, None, fresh=False, claims_cap=claims_cap)

    site_trace = Trace(query="__site_index__")
    kg = KnowledgeGraph()
    vs = VectorStore()

    def process(new_chunk_objs):
        cd = [{"chunk_id": c.chunk_id, "content": c.content, "url": c.url}
              for c in new_chunk_objs]
        embs = embed_chunks_sync([c.content for c in new_chunk_objs], use_cache=True)
        vs.add_chunks(new_chunk_objs, embs)
        ents = extract_entities(cd, trace=site_trace)
        rels = extract_relations(cd, ents, trace=site_trace)
        cls = extract_claims(cd[:claims_cap], trace=site_trace)
        ev = extract_evidence(cd[:claims_cap], cls, trace=site_trace)
        kg.build(cd, ents, rels, cls, ev, embeddings=embs, trace=site_trace)

    stats = incremental_update(url, state, kg, vs, process, trace=site_trace)

    # rebuild query context from state
    chunks = state.active_chunks()
    embeddings = embed_chunks_sync([c["content"] for c in chunks], use_cache=True)
    topics = cluster_topics(chunks, trace=site_trace) if stats["rerun_topics"] else []

    pages, inbound = [], {}
    for row in state.all_pages():
        links = _json.loads(row["internal_links_json"] or "[]")
        for l in links:
            inbound[l.get("dest_url", "")] = inbound.get(l.get("dest_url", ""), 0) + 1
        pages.append({
            "url": row["url"], "title": row["title"],
            "content": " ".join(c["content"] for c in chunks
                                if c["url"] == row["url"])[:8000],
            "internal_links": links,
            "access_gaps": _json.loads(row["access_gaps_json"] or "[]"),
        })
    for p in pages:
        p["internal_inbound"] = inbound.get(p["url"], 0)
    page_signals = score_pages(pages)
    _unload_extraction_models()

    return {
        "site_report": {
            "url": url, "mode": "incremental", **stats,
            "chunks": len(chunks), "vector_count": vs.count(),
            "chroma_kuzu_parity": vs.count() == len(chunks),
            "embed_dim": len(embeddings[0]) if embeddings else None,
            "n_topics": len(topics), "kg": kg.verify(),
        },
        "chunks": chunks, "embeddings": embeddings, "kg": kg,
        "page_signals": page_signals, "site_trace": site_trace,
        "l0_pages": pages,
    }


def write_reports(ctx: dict, traces: list[Trace], explanations: list[dict],
                  out_dir: Path) -> Path:
    from src.explain import render_markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "invisibility": ctx.get("invisibility"),   # headline metric, measured
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
    inv = ctx.get("invisibility") or {}
    lines = [f"# AI-Search Readiness Audit — {sr['url']}", ""]
    if inv:
        lines += [f"## {inv['headline']}",
                  f"(AI Invisibility Score: {inv['site_invisibility']:.0%} — "
                  f"{inv['label']}; weighting: {inv['weighting']})", ""]
        for wp in inv["worst_pages"][:3]:
            if wp["mean_invisible_ratio"] > 0:
                lines.append(f"- {wp['url']}: {wp['mean_invisible_ratio']:.0%} invisible"
                             f" — e.g. missing: "
                             f"{(wp['worst_missing_blocks'] or ['—'])[0][:90]!r}")
        lines.append("")
    lines += [
             f"- Mode: {sr.get('mode', 'full')}  "
             f"Pages: {sr.get('pages_crawled', sr.get('pages_total', '?'))}  "
             f"Chunks: {sr['chunks']}  Embed dim: {sr['embed_dim']}",
             f"- Entities: {sr.get('n_entities', '-')}  "
             f"Relations: {sr.get('n_relationships', '-')}  "
             f"Claims: {sr.get('n_claims', '-')}  Topics: {sr.get('n_topics', 0)}",
             f"- KG: {sr['kg']['nodes']} | orphan_chunks={sr['kg']['orphan_chunks']} "
             f"claims_without_support={sr['kg']['claims_without_support']}",
             f"- Provider: {providers.provider_info()['llm_provider_effective']}"
             f" / {providers.provider_info()['embed_provider_effective']} (zero keys required)",
             ""]
    for rows in explanations:
        lines.append(render_markdown(rows))
        lines.append("")
    # PHASE 10: this trace dump is the debug appendix now — engineers need it,
    # clients never see it. The client-facing document is report.html
    # (build_decision_report, called from run() when --fixes is set).
    md_path = out_dir / "audit_report_debug.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def build_decision_report(ctx: dict, traces: list[Trace], explanations: list[dict],
                          struct: list[dict], fx: dict | None,
                          query_weights: dict[str, float], out_dir: Path) -> Path:
    """PHASE 10: assemble the five-section client report.html. Only called
    when --fixes was passed — 10a's impact chain has nothing to chain
    without Phase 8 fix artifacts to run through the what-if overlay."""
    from src.impact import attach_predicted_impact
    from src.report.decision_report import (
        build_verdict, build_action_plan, build_battle_cards,
        build_technical_section, build_trend_section, render_report_html,
        enforce_renderer_rules)
    from src.snapshots import list_snapshots, load_snapshot, diff_snapshots

    # NOTE: deliberately NOT passing kg.kuzu (or chromadb) here as a
    # persist_paths integrity check. That check hashes the file directly
    # (whatif._hash_paths) to prove the overlay never wrote to it — a real
    # safety net for the standalone `run_audit.py whatif` CLI command, where
    # the DB is at rest. Here we're calling in-process while ctx["kg"]'s own
    # KuzuDB connection is still open on that exact file, so a second raw
    # read hits Kuzu's own file lock and fails every time with
    # "[Errno 13] Permission denied" (100% reproducible, not the transient
    # antivirus-scan case load_draft()'s retry handles) — found running the
    # actual penny-test (BUILD_LOG PENNY TEST run #1). We already trust this
    # code path not to persist (it's our own whatif() call, not a CLI
    # invocation of unknown code), so the check adds no real safety here.
    persist_paths: list[str] = []
    manifest = (fx or {}).get("manifest", [])

    # chunk_id -> {url, heading_path, text} — every S2/S3 row resolves through
    # this; heading_path is empty for chunks from the default chunker (only
    # chunking2.py's formal chunker populates it — see BUILD_LOG Phase 10 note).
    chunk_lookup = {c["chunk_id"]: {"url": c.get("url", ""),
                                    "heading_path": c.get("heading_path", []),
                                    "text": c.get("content", "")}
                    for c in ctx["chunks"]}

    query_recommendations = []
    for t, rows in zip(traces, explanations):
        recs = rows["recommendations"]
        if manifest:
            attach_predicted_impact(recs, manifest, t.query, ctx["chunks"],
                                    ctx["embeddings"], persist_paths=persist_paths)
        query_recommendations.append((t.query, query_weights.get(t.query, 1.0), recs))

    inv = ctx.get("invisibility") or {}
    pages_no_schema = [{"url": p["url"]} for p in ctx["l0_pages"] if not p.get("schema_jsonld")]
    action_plan = build_action_plan(
        query_recommendations=query_recommendations, structure_flags=struct,
        invisibility_worst_pages=inv.get("worst_pages", []),
        schema_missing_pages=pages_no_schema[:10],
        visited_but_invisible=[], chunk_lookup=chunk_lookup)

    struct_scores = [s["structure_score"] for s in struct]
    mean_structure = round(sum(struct_scores) / len(struct_scores), 4) if struct_scores else 0.0
    auth_scores = [v["source_authority"] for v in ctx["page_signals"].values()]
    mean_authority = round(sum(auth_scores) / len(auth_scores), 4) if auth_scores else 0.0
    # "top 5 actions" for the predicted-lift sentence means the 5 highest-
    # impact ones. Sourced from query_recommendations (every recommendation
    # with an impact attached), NOT from action_plan[:max_rows] — a page's
    # JS-invisibility fix can legitimately outrank a single dead-end
    # sub-query in the printed plan and push real coverage-delta rows past
    # the row cap entirely, but the lift sentence still needs the true best
    # deltas wherever they landed, or a run that DID generate fixes ends up
    # falsely claiming it didn't.
    coverage_deltas = sorted(
        (rec["predicted_impact"]["delta"] for _, _, recs in query_recommendations
         for rec in recs if rec.get("predicted_impact", {}).get("kind") == "coverage"),
        reverse=True)
    top_deltas = coverage_deltas[:5]
    verdict = build_verdict(inv, [{"weighted_coverage": next(
        (s.outputs.get("cluster_coverage", {}).get("weighted_coverage")
         for s in t.steps if s.name == "retriever"), None)} for t in traces],
        mean_structure, mean_authority, top_deltas)

    query_records = [{"query": t.query, "weight": query_weights.get(t.query, 1.0),
                      "answer": t.answer, "confidence": t.confidence,
                      "n_citations": len(t.citations),
                      "retrieval_dead_ends": t.retrieval_dead_ends} for t in traces]
    battle_cards = build_battle_cards(query_records, action_plan, top_n=10)

    inbound: dict[str, int] = {}
    for p in ctx["l0_pages"]:
        for link in p.get("internal_links", []):
            inbound[link["dest_url"]] = inbound.get(link["dest_url"], 0) + 1
    orphan_pages = [p["url"] for p in ctx["l0_pages"] if inbound.get(p["url"], 0) == 0]
    robots_conflicts = sum(len(p.get("robots_conflict", [])) for p in ctx["l0_pages"])
    # Missing page-to-page links: pages sharing a Topic with no direct
    # InternalLink edge between them — a lightweight version of the spec's
    # "pairs of pages covering the same fan-out sub-intent" check, since that
    # needs a per-page sub-intent map this pipeline doesn't compute; topic
    # co-membership is the closest signal already in the graph.
    missing_links: list[dict] = []
    try:
        res = ctx["kg"]._exec(
            "MATCH (a:Page)-[:HasChunk]->(:Chunk)-[:BelongsToTopic]->(t:Topic)"
            "<-[:BelongsToTopic]-(:Chunk)<-[:HasChunk]-(b:Page) "
            "WHERE a.url < b.url "
            "AND NOT EXISTS { MATCH (a)-[:InternalLink]->(b) } "
            "AND NOT EXISTS { MATCH (b)-[:InternalLink]->(a) } "
            "RETURN DISTINCT a.url, b.url, t.label LIMIT 15")
        while res.has_next():
            r = res.get_next()
            missing_links.append({"a": r[0], "b": r[1], "topic": r[2]})
    except Exception as exc:
        logger.warning(f"[decision_report] missing-links query skipped: {exc}")
    technical = build_technical_section(
        invisibility=inv, robots_conflicts=robots_conflicts, orphan_pages=orphan_pages,
        missing_links=missing_links,
        llms_txt_present=any(m["type"] == "llms_txt" for m in manifest),
        serverlogs=None)

    trend = None
    runs = list_snapshots()
    if len(runs) >= 2:
        from src.report.decision_report import resolve_chunk_structure_urls
        raw_diff = diff_snapshots(load_snapshot(runs[-2]), load_snapshot(runs[-1]))
        trend = build_trend_section(resolve_chunk_structure_urls(raw_diff, chunk_lookup))

    report_ctx = {"url": ctx["site_report"]["url"], "verdict": verdict,
                 "action_plan": action_plan, "battle_cards": battle_cards,
                 "technical": technical, "trend": trend}
    html = render_report_html(report_ctx)
    violations = enforce_renderer_rules(html)
    if violations:
        # 10c is a hard rule, not a suggestion — a report that leaks jargon or
        # bare chunk ids to a client is a build failure, not a warning.
        raise RuntimeError(f"Phase 10 renderer rule violation(s): {violations}")

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"[decision_report] wrote {report_path} "
               f"(grade={verdict['grade']['grade']}, {len(action_plan)} action rows, "
               f"{len(battle_cards)} battle cards)")
    return report_path


def run(url: str, queries: list[str], out_dir: Path, max_pages: int | None,
        fresh: bool, claims_cap: int = 25, mode: str = "full",
        emit_fixes: bool = False, query_weights: dict[str, float] | None = None,
        emit_intelligence: bool = False) -> dict:
    from src.simulator import run_query
    from src.explain import explain_query

    from src.layer0 import invisibility_score

    if mode == "incremental":
        ctx = build_index_incremental(url, claims_cap)
    else:
        ctx = build_index(url, max_pages, fresh, claims_cap)

    # Phase 12 P2 — query hygiene: dedupe + park off-brand/irrelevant queries
    # against this run's own topic centroids (Section 13 lists them, never
    # silently drops them).
    from src.query_hygiene import clean_query_set
    hygiene = clean_query_set(queries, kg=ctx["kg"], embed_fn=providers.embed_texts)
    queries = hygiene["relevant"]
    skipped_queries = hygiene["skipped"]

    traces, explanations = [], []
    for q in queries:
        t = run_query(q, ctx["chunks"], ctx["embeddings"], kg=ctx["kg"],
                      page_signals=ctx["page_signals"])
        traces.append(t)
        explanations.append(explain_query(t, kg=ctx["kg"]))
    # AI Invisibility Score (measured) — weighted by PageRank and by the pages
    # that actually won retrievals for this query set
    winner_urls = {c["page_url"] for t in traces for c in t.citations if c["page_url"]}
    ctx["invisibility"] = invisibility_score(ctx["l0_pages"], winner_urls)

    # Phase 3b structural scores + Phase 5 snapshot persistence
    from src.structure import score_chunks
    from src.snapshots import build_snapshot, save_snapshot
    pages_with_schema = {p["url"] for p in ctx["l0_pages"]
                         if p.get("schema_jsonld")}
    struct = score_chunks(
        [{"id": c["chunk_id"], "text": c["content"], "page_url": c["url"],
          "heading_path": c.get("heading_path", [])} for c in ctx["chunks"]],
        pages_with_schema=pages_with_schema)
    snap = build_snapshot(url, ctx["invisibility"], struct,
                          ctx["page_signals"], traces)
    run_id = save_snapshot(snap)
    ctx["snapshot_run_id"] = run_id

    # Phase 8 — emit ready-to-use fix artifacts (draft, human-review) if asked
    fx = None
    if emit_fixes:
        from src.fixes import generate_fixes
        pages_no_schema = [p for p in ctx["l0_pages"] if not p.get("schema_jsonld")]
        dead_ends = [{"sub_query": de.get("sub_query", de) if isinstance(de, dict) else de}
                     for t in traces for de in t.retrieval_dead_ends][:5]
        flagged = [{"chunk_id": s["chunk_id"],
                    "text": next((c["content"] for c in ctx["chunks"]
                                  if c["chunk_id"] == s["chunk_id"]), ""),
                    "flags": s["flags"]} for s in struct if s["flags"]][:5]
        site = {"name": url, "description": "",
                "pages": [{"url": p["url"], "title": p.get("title", ""),
                           "pagerank": 0} for p in ctx["l0_pages"]],
                "topics": []}
        fx = generate_fixes(run_id, pages_without_schema=pages_no_schema[:5],
                            site=site, dead_ends=dead_ends, structure_flagged=flagged,
                            trace=ctx["site_trace"])
        ctx["fixes_dir"] = fx["run_dir"]
        logger.info(f"[fixes] {len(fx['manifest'])} artifacts -> {fx['run_dir']}")

    json_path = write_reports(ctx, traces, explanations, out_dir)
    logger.info(f"[report] wrote {json_path}")

    # Phase 10 — the client-facing decision report. Needs Phase 8's fix
    # artifacts to chain through 10a's impact overlay, so it only renders
    # when --fixes was passed (same as the fixes_dir gate above).
    report_html_path = None
    if emit_fixes:
        report_html_path = build_decision_report(
            ctx, traces, explanations, struct, fx, query_weights or {}, out_dir)

    # Phase 12 — the Website Intelligence Report (19 sections). Independent
    # of --fixes (unlike Phase 10's report, it doesn't need fix artifacts to
    # chain through), gated behind its own --intelligence flag since it adds
    # real runtime (cross-page claim evidence re-scores every claim against
    # its site-wide top-k chunks).
    intelligence_html_path = None
    if emit_intelligence:
        from src.intelligence.report import build_intelligence_report
        from src.intelligence.html_renderer import render_intelligence_html
        payload = build_intelligence_report(
            ctx, traces, explanations, struct, fx, query_weights or {}, skipped_queries)
        (out_dir / "intelligence.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        intelligence_html_path = out_dir / "intelligence.html"
        intelligence_html_path.write_text(render_intelligence_html(payload), encoding="utf-8")
        logger.info(f"[intelligence] wrote {intelligence_html_path}")

    return {"site_report": ctx["site_report"], "traces": traces,
            "explanations": explanations, "report_path": str(json_path),
            "decision_report_path": str(report_html_path) if report_html_path else None,
            "intelligence_report_path": str(intelligence_html_path) if intelligence_html_path else None}


def cmd_import_queries(argv: list[str]) -> None:
    """`run_audit.py import-queries --gkp <csv> [--paa <file>] --out <json>`."""
    from src.demand import build_query_set
    ap = argparse.ArgumentParser(prog="run_audit.py import-queries",
                                 description="Ingest real query demand (Phase 4).")
    ap.add_argument("--gkp", help="Google Keyword Planner CSV/TSV export")
    ap.add_argument("--paa", help="People-Also-Ask paste file (one per line)")
    ap.add_argument("--max-variants", type=int, default=3)
    ap.add_argument("--out", default="data/queries_imported.json")
    args = ap.parse_args(argv)
    if not args.gkp and not args.paa:
        ap.error("provide --gkp and/or --paa")
    queries = build_query_set(args.gkp, args.paa, args.max_variants)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(queries, indent=2), encoding="utf-8")
    seeds = len({q["seed"] for q in queries if q.get("seed")})
    print(f"Imported {len(queries)} queries from {seeds} seeds "
          f"(ordered by volume weight) -> {args.out}")


def cmd_diff(argv: list[str]) -> int:
    """`run_audit.py diff [--from N --to M]` — regressions since a prior run.
    Non-zero exit when a regression exceeds threshold (cron/CI usable)."""
    from src.snapshots import (list_snapshots, load_snapshot, diff_snapshots,
                               render_diff)
    ap = argparse.ArgumentParser(prog="run_audit.py diff")
    ap.add_argument("--from", dest="frm", type=int)
    ap.add_argument("--to", dest="to", type=int)
    args = ap.parse_args(argv)
    runs = list_snapshots()
    if len(runs) < 2 and not (args.frm and args.to):
        print("Need at least two snapshots to diff.")
        return 2
    frm = args.frm or runs[-2]
    to = args.to or runs[-1]
    d = diff_snapshots(load_snapshot(frm), load_snapshot(to))
    print(render_diff(d))
    return 1 if d["ci_status"] == "fail" else 0


def cmd_calibrate(argv: list[str]) -> int:
    """`run_audit.py calibrate --run N [--template]` — compare the simulator's
    winning pages against real ChatGPT/Perplexity/Gemini citations (manual
    panel). Observed numbers are reported separately from simulated."""
    from src.calibration import (generate_templates, load_run, compute_agreement)
    from src.snapshots import list_snapshots, load_snapshot
    ap = argparse.ArgumentParser(prog="run_audit.py calibrate")
    ap.add_argument("--run", type=int, help="snapshot run id (default: latest)")
    ap.add_argument("--template", action="store_true",
                    help="write blank paste-templates for the top-20 queries")
    args = ap.parse_args(argv)
    runs = list_snapshots()
    if not runs:
        print("No snapshots yet — run an audit first.")
        return 2
    run_id = args.run or runs[-1]
    snap = load_snapshot(run_id)
    if args.template:
        qs = [{"query": q, "weight": v.get("weight", 0)}
              for q, v in snap["queries"].items()]
        root = generate_templates(run_id, qs)
        print(f"Wrote calibration templates -> {root}\n"
              f"Fill each engine's answer + cited URLs, then run "
              f"`run_audit.py calibrate --run {run_id}`.")
        return 0
    calibration = load_run(run_id)
    if not calibration:
        print(f"No filled calibration files for run {run_id}. "
              f"Run with --template first.")
        return 2
    wins = {q: v.get("winning_pages", []) for q, v in snap["queries"].items()}
    ag = compute_agreement(calibration, wins)
    print(f"OBSERVED citation agreement (run {run_id}) — reported separately "
          f"from simulated confidence:")
    print(f"  overall: {ag['overall_agreement']}  per-engine: "
          f"{ag['per_engine_agreement']}")
    print(f"  {ag['note']}")
    return 0


def cmd_whatif(argv: list[str]) -> int:
    """`run_audit.py whatif --draft <md|txt|docx> --query "..." [...]` — predict
    a draft's coverage impact against a non-persisting overlay index."""
    from src.whatif import whatif
    from src.incremental import IncrementalState
    from src.embed import embed_chunks_sync
    ap = argparse.ArgumentParser(prog="run_audit.py whatif")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--query", action="append", required=True,
                    help="target query (repeatable)")
    args = ap.parse_args(argv)

    state = IncrementalState()
    base_chunks = state.active_chunks()
    state.close()
    if not base_chunks:
        print("No baseline index — run an audit first."); return 2
    base_embs = embed_chunks_sync([c["content"] for c in base_chunks], use_cache=True)
    persist = [str(BASE_DIR / "data" / "chromadb"),
               str(BASE_DIR / "data" / "kg.kuzu")]
    res = whatif(args.draft, args.query, base_chunks, base_embs, persist_paths=persist)
    print(f"What-if for {args.draft} ({res['n_draft_chunks']} draft chunks, "
          f"mean structure {res['mean_structure_score']}):")
    for pq in res["per_query"]:
        print(f"  {pq['query'][:50]!r}: coverage {pq['coverage_before']} -> "
              f"{pq['coverage_after']} (delta {pq['coverage_delta']:+}, "
              f"+{pq['newly_covered']} sub-intents)")
    print(f"  persistent stores unchanged: {res['stores_unchanged']}")
    if res["structure_flags"]:
        print(f"  draft structure flags: "
              f"{sorted({f['flag'] for f in res['structure_flags']})}")
    return 0


def cmd_import_logs(argv: list[str]) -> int:
    """`run_audit.py import-logs <access.log ...>` — AI-bot crawl-demand map."""
    from src.serverlogs import import_logs
    from src.incremental import IncrementalState
    ap = argparse.ArgumentParser(prog="run_audit.py import-logs")
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args(argv)
    state = IncrementalState()
    urls = {c["url"] for c in state.active_chunks() if c.get("url")}
    state.close()
    pages = [{"url": u} for u in urls]
    r = import_logs(args.logs, pages)
    print(r["headline"])
    print(f"  parsed {r['lines_total']} lines ({r['lines_malformed']} malformed), "
          f"{r['bot_hits']} AI-bot hits")
    print(f"  per bot: {r['per_bot']}")
    if r["never_visited"]:
        print(f"  never visited by AI bots ({len(r['never_visited'])}): "
              f"{r['never_visited'][:5]}{' ...' if len(r['never_visited'])>5 else ''}")
    if r["visited_but_invisible"]:
        print(f"  HIGH PRIORITY (visited AND invisible): {r['visited_but_invisible']}")
    return 0


from src.core.config import BASE_DIR  # noqa: E402 (used by cmd_whatif)


def main() -> None:
    # Lightweight subcommand dispatch (keeps `run_audit.py --url ...` working;
    # Phase 4+ add import-queries / diff / whatif / calibrate / import-logs).
    if len(sys.argv) > 1 and sys.argv[1] == "import-queries":
        cmd_import_queries(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        raise SystemExit(cmd_diff(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        raise SystemExit(cmd_calibrate(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "whatif":
        raise SystemExit(cmd_whatif(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "import-logs":
        raise SystemExit(cmd_import_logs(sys.argv[2:]))

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
    ap.add_argument("--incremental", action="store_true",
                    help="Incremental update (default once a baseline exists)")
    ap.add_argument("--full", action="store_true",
                    help="Force a full rebuild even when a baseline exists")
    ap.add_argument("--fixes", action="store_true",
                    help="Emit ready-to-use fix artifacts (draft) to fixes/<run>/, "
                         "and (Phase 10) the client report.html chaining them "
                         "through the predicted-impact overlay")
    ap.add_argument("--gkp", help="Google Keyword Planner CSV — real query-volume "
                                  "weight for Phase 10's action-plan prioritization "
                                  "(optional; uniform weight 1.0 without it)")
    ap.add_argument("--intelligence", action="store_true",
                    help="Emit intelligence.html/.json (Phase 12) — the 19-section "
                         "website knowledge-model report. Independent of --fixes.")
    args = ap.parse_args()

    # Phase 2 mode resolution: incremental is the default once a baseline
    # exists; --full always forces a rebuild.
    from src.incremental import IncrementalState
    _s = IncrementalState()
    has_baseline = _s.has_baseline()
    _s.close()
    if args.full:
        mode = "full"
    elif args.incremental or has_baseline:
        mode = "incremental"
    else:
        mode = "full"

    queries = load_queries(args.queries, args.query)
    query_weights = load_query_weights(args.queries, args.gkp)
    logger.info(f"Auditing {args.url} with {len(queries)} queries mode={mode} "
                f"(LLM={providers.LLM_PROVIDER}:{providers.LLM_MODEL}, "
                f"EMBED={providers.EMBED_PROVIDER}:{providers.EMBED_MODEL})"
                + (f" gkp={args.gkp}" if args.gkp else ""))
    result = run(args.url, queries, Path(args.out), args.max_pages,
                 fresh=not args.no_fresh, claims_cap=args.claims_cap, mode=mode,
                 emit_fixes=args.fixes, query_weights=query_weights,
                 emit_intelligence=args.intelligence)
    print(f"\nReport written to {result['report_path']}")
    if result.get("decision_report_path"):
        print(f"Decision report (Phase 10): {result['decision_report_path']}")
    if result.get("intelligence_report_path"):
        print(f"Intelligence report (Phase 12): {result['intelligence_report_path']}")


if __name__ == "__main__":
    main()
